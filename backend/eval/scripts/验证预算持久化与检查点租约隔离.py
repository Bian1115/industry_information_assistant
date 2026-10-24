"""隔离schema验证真实预算账本、检查点事务和SIGKILL章节续跑；零外部模型请求。"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "eval/tests")]
from dotenv import load_dotenv
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from service.research_events import ResearchRunLease, ResearchRunLeaseLost, acquire_research_run_lease
from service.deep_research_v2.checkpointer import LeaseFencedPostgresSaver, build_pool, checkpoint_database_url
from service.deep_research_v2.budget_ledger import PostgresRuntimeBudget, current_runtime_budget, read_runtime_budget
from service.deep_research_v2.graph import agents_by_phase, build_graph, graph_config, pack_state
from service.deep_research_v2.stage_runtime import StageBudgetExceeded, consumed_tokens, make_execute_stage
from service.deep_research_v2.state import create_initial_state
from models.research import ResearchRun
from models.research_budget import ResearchBudget, ResearchBudgetAttempt, ResearchBudgetCall
from test_单主链路图 import FakeAgent, make_agents
from test_Writer章节检查点 import Scenario, initial_state


def save(out, name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def schema_options(schema):
    if not re.fullmatch(r"budget_fence_test_[0-9a-f]{32}", schema):
        raise ValueError("仅允许本脚本创建的隔离schema")
    return "-c search_path=" + schema


async def connect(schema=None):
    return await psycopg.AsyncConnection.connect(checkpoint_database_url(), autocommit=True,
        row_factory=dict_row, connect_timeout=5, **({"options": schema_options(schema)} if schema else {}))


async def open_pool(schema):
    pool = build_pool(make_conninfo(checkpoint_database_url(), options=schema_options(schema), connect_timeout=5))
    await pool.open()
    await pool.wait()
    return pool


async def collect(graph, value, run_id):
    return [event async for event in graph.astream(value, graph_config(run_id), stream_mode="custom", durability="sync")]


async def metered_call(tokens=100, reserved=120):
    store, attempt = current_runtime_budget()
    call = await store.reserve(attempt, "固定假模型", reserved)
    await store.complete(attempt, call, {"total_tokens": tokens})


def graph_for(pool, lease, agents):
    return build_graph(agents, LeaseFencedPostgresSaver(pool, lease=lease), run_id=lease.run_id,
        user_id=None, session_id=lease.session_id, execute_stage=make_execute_stage(
            run_id=lease.run_id, session_id=lease.session_id, agents_of=agents_by_phase(agents),
            budget_store=PostgresRuntimeBudget(pool, lease)))


def checkpoint(marker):
    value = empty_checkpoint()
    value["channel_values"] = {"research": {"marker": marker}}
    value["channel_versions"] = {"research": str(uuid.uuid4())}
    return value


async def write(saver, run_id, marker):
    value = checkpoint(marker)
    return await saver.aput({"configurable": {"thread_id": run_id, "checkpoint_ns": ""}}, value,
                           {"source": "input", "step": 0, "parents": {}}, value["channel_versions"])


async def worker(args):
    out = Path(args.output_dir)
    lease_data = json.loads((out / ("故障进程租约.json" if args.worker == "crash" else "续跑进程租约.json")).read_text())
    lease_data["lease_until"] = datetime.fromisoformat(lease_data["lease_until"])
    lease = ResearchRunLease(**lease_data)
    pool = await open_pool(args.schema)
    try:
        scenario = Scenario()
        writer = scenario.writer()
        original = writer.call_llm

        async def call_llm(**kwargs):
            store, attempt = current_runtime_budget()
            call = await store.reserve(attempt, "固定假模型", 120)
            if args.worker == "crash" and "标题: 章节3\n" in kwargs["user_prompt"]:
                scenario.calls["sec_3"] += 1
                save(out, "第三章崩溃屏障.json", {"calls": dict(scenario.calls), "pid": os.getpid(), "pending_call": call})
                await asyncio.Event().wait()
            answer = await original(**kwargs)
            await store.complete(attempt, call, {"total_tokens": 100})
            writer.llm_call_count += 1
            writer.llm_total_tokens += 100
            return answer

        writer.call_llm = call_llm
        graph = graph_for(pool, lease, make_agents(writer=writer))
        with patch("service.deep_research_v2.stage_runtime.is_research_cancelled", return_value=False):
            events = await collect(graph, pack_state(initial_state()) if args.worker == "crash" else None, lease.run_id)
        state = (await graph.aget_state(graph_config(lease.run_id))).values["research"]
        budget = await read_runtime_budget(pool, lease.run_id)
        save(out, "章节续跑与预算结果.json", {"calls": dict(scenario.calls), "status": state["status"],
            "restored_sections": [e["content"]["section_id"] for e in events
                                  if e["type"] == "section_content" and e["content"].get("restored")],
            "known_tokens": budget["known_tokens"], "charged_tokens": budget["charged_tokens"],
            "unknown_usage_calls": budget["unknown_usage_calls"], "budget": budget,
            "checkpoint_charged_tokens": consumed_tokens(state)})
    finally:
        await pool.close()


class Validation:
    def __init__(self, out, schema, pool, factory, engine):
        self.out, self.schema, self.pool, self.factory = out, schema, pool, factory
        self.engine = engine
        self.results = []

    async def lease(self, run_id=None, owner="执行器甲", seconds=60):
        with patch("core.database.SessionLocal", self.factory):
            return await asyncio.to_thread(acquire_research_run_lease,
                run_id=run_id or "隔离验证-" + uuid.uuid4().hex, session_id="隔离验证会话",
                user_id=None, owner_id=owner, lease_seconds=seconds)

    async def expire(self, run_id):
        async with self.pool.connection() as conn:
            await conn.execute("UPDATE research_runs SET lease_until = clock_timestamp() AT TIME ZONE 'UTC' - interval '1 second' WHERE run_id = %s", (run_id,))

    async def case(self, name, func):
        started = time.monotonic()
        try:
            details = await func()
            row = {"场景": name, "通过": True, "详情": details}
        except Exception as exc:
            row = {"场景": name, "通过": False, "错误类型": type(exc).__name__, "错误": str(exc), "堆栈": traceback.format_exc()}
        row["耗时秒"] = time.monotonic() - started
        self.results.append(row)
        save(self.out, "逐场景结果.json", self.results)
        print(json.dumps({k: v for k, v in row.items() if k != "堆栈"}, ensure_ascii=False), flush=True)

    async def migration(self):
        def migrate():
            path = BACKEND / "alembic/versions/20261014_0014_研究运行预算账本.py"
            spec = importlib.util.spec_from_file_location("isolated_budget_migration", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            tables = {"research_budgets", "research_budget_attempts", "research_budget_calls"}
            with self.engine.begin() as connection, patch.object(module.op, "get_bind", return_value=connection):
                module.upgrade()
                assert tables <= set(inspect(connection).get_table_names())
                module.downgrade()
                assert not tables & set(inspect(connection).get_table_names())
                module.upgrade()
                assert tables <= set(inspect(connection).get_table_names())
            return {"升级建表": 3, "降级删除": 3, "再次升级建表": 3, "范围": "空的自建隔离schema"}
        return await asyncio.to_thread(migrate)

    async def concurrent_reservation(self):
        lease = await self.lease()
        store = PostgresRuntimeBudget(self.pool, lease)
        with patch.dict(os.environ, {"RESEARCH_MAX_LLM_TOKENS": "150"}):
            attempt, _ = await store.begin({"_node": "research"}, "researching", "DeepScout")
        answers = await asyncio.gather(
            store.reserve(attempt, "并行请求甲", 80), store.reserve(attempt, "并行请求乙", 80),
            return_exceptions=True,
        )
        assert sum(isinstance(v, str) for v in answers) == 1
        assert sum(isinstance(v, StageBudgetExceeded) for v in answers) == 1
        await store.finish(attempt, 0.1, "failed", 0, 0)
        budget = await read_runtime_budget(self.pool, lease.run_id)
        assert budget["charged_tokens"] == 80 and budget["unknown_usage_calls"] == 1
        return {"并行预留": [80, 80], "上限": 150, "允许数": 1, "拒绝数": 1, "扣账token": 80}

    async def single_connection_reservation(self):
        lease = await self.lease()
        async with await connect(self.schema) as connection:
            store = PostgresRuntimeBudget(connection, lease)
            with patch.dict(os.environ, {"RESEARCH_MAX_LLM_TOKENS": "150"}):
                attempt, _ = await store.begin({"_node": "research"}, "researching", "DeepScout")
            answers = await asyncio.gather(
                store.reserve(attempt, "单连接请求甲", 80), store.reserve(attempt, "单连接请求乙", 80),
                return_exceptions=True,
            )
            assert sum(isinstance(v, str) for v in answers) == 1
            assert sum(isinstance(v, StageBudgetExceeded) for v in answers) == 1
            saver = LeaseFencedPostgresSaver(connection, lease=lease)
            config = await write(saver, lease.run_id, "单连接有效结果")
            reading = None
            try:
                async with saver.lock:
                    reading = asyncio.create_task(saver.aget_tuple(config))
                    await asyncio.sleep(0.01)
                    # 等待saver实例锁的读操作不能先占住连接锁，否则形成反向锁序。
                    projected = await asyncio.wait_for(read_runtime_budget(connection, lease.run_id), 1)
                    assert projected["charged_tokens"] == 80
                await asyncio.wait_for(reading, 1)
            finally:
                if reading is not None and not reading.done():
                    reading.cancel()
                    await asyncio.gather(reading, return_exceptions=True)
            await store.finish(attempt, 0.1, "completed", 0, 0)
        budget = await read_runtime_budget(self.pool, lease.run_id)
        assert budget["charged_tokens"] == 80
        return {"单连接并发请求": 2, "提交预留数": 1, "扣账token": 80, "检查点写入": "成功", "连接与saver锁顺序": "无反向锁等待"}

    async def failed_budget(self):
        class Scout(FakeAgent):
            async def process(agent, state):
                await metered_call(agent.tokens, agent.tokens)
                agent.llm_total_tokens += agent.tokens
                agent.llm_call_count += 1
                if agent.fail:
                    raise RuntimeError("注入搜索节点失败")
                return state
        lease = await self.lease()
        first = Scout("DeepScout")
        first.fail, first.tokens = True, 100
        graph = graph_for(self.pool, lease, make_agents(scout=first))
        try:
            await collect(graph, pack_state(create_initial_state("固定问题", lease.session_id, [])), lease.run_id)
            raise AssertionError("未触发故障")
        except RuntimeError as exc:
            assert str(exc) == "注入搜索节点失败"
        before = await read_runtime_budget(self.pool, lease.run_id)
        assert before["known_tokens"] == 100
        second = Scout("DeepScout")
        second.fail, second.tokens = False, 40
        resumed = graph_for(self.pool, lease, make_agents(scout=second))
        await collect(resumed, None, lease.run_id)
        state = (await resumed.aget_state(graph_config(lease.run_id))).values["research"]
        final = await read_runtime_budget(self.pool, lease.run_id)
        assert final["known_tokens"] == consumed_tokens(state) == 140
        assert final["node_attempts"]["research"] == 2
        save(self.out, "失败节点预算恢复.json", {"故障后": before, "最终": final})
        return {"故障后token": 100, "最终token": 140, "搜索物理尝试": 2}

    async def reservation_and_legacy(self):
        lease = await self.lease()
        store = PostgresRuntimeBudget(self.pool, lease)
        work = {"_node": "research", "node_attempts": {"research": 2}, "runtime_budget": {
            "elapsed_seconds": 10, "phases": {"DeepScout": {"runs": [{"llm_tokens": 30, "llm_calls": 1}]}}}}
        with patch.dict(os.environ, {"RESEARCH_MAX_LLM_TOKENS": "150"}):
            attempt, _ = await store.begin(work, "researching", "DeepScout")
        call = await store.reserve(attempt, "固定假模型", 100)
        await store.complete(attempt, call, None)
        await store.finish(attempt, 0.1, "failed", 0, 1)
        next_id, budget = await PostgresRuntimeBudget(self.pool, lease).begin(work, "researching", "DeepScout")
        assert budget["known_tokens"] == 30 and budget["charged_tokens"] == 130
        try:
            await store.reserve(next_id, "固定假模型", 21)
            raise AssertionError("未知请求预留未阻止超预算")
        except StageBudgetExceeded:
            pass
        await store.complete(attempt, call, {"total_tokens": 40})
        await store.complete(attempt, call, {"total_tokens": 40})
        budget = await read_runtime_budget(self.pool, lease.run_id)
        assert budget["known_tokens"] == budget["charged_tokens"] == 70
        assert budget["token_limit"] == 150
        try:
            await store.complete(attempt, call, {"total_tokens": 41})
            raise AssertionError("不一致usage被覆盖")
        except RuntimeError as exc:
            assert "usage不一致" in str(exc)
        await store.finish(next_id, 0.1, "completed", 0, 0)
        save(self.out, "预留幂等与历史导入.json", budget)
        return {"历史导入token": 30, "未知时扣账token": 130, "真实usage对账后token": 70, "重复结算": "无重复", "冻结上限": 150}

    async def stale_writes(self):
        lease_a = await self.lease()
        saver_a = LeaseFencedPostgresSaver(self.pool, lease=lease_a)
        config = await write(saver_a, lease_a.run_id, "甲的有效结果")
        await self.expire(lease_a.run_id)
        lease_b = await self.lease(lease_a.run_id, "执行器乙")
        saver_b = LeaseFencedPostgresSaver(self.pool, lease=lease_b)
        await write(saver_b, lease_b.run_id, "乙的有效结果")
        counts = []
        for operation in (lambda: write(saver_a, lease_a.run_id, "甲的迟到结果"),
                          lambda: saver_a.aput_writes(config, [("__return__", {"late": True})], "迟到章节任务")):
            try:
                await operation()
                raise AssertionError("旧执行器提交成功")
            except ResearchRunLeaseLost:
                counts.append("已拒绝")
        async with self.pool.connection() as conn:
            row = await conn.execute("SELECT count(*) AS n FROM checkpoints WHERE thread_id = %s", (lease_a.run_id,))
            assert (await row.fetchone())["n"] == 2
            row = await conn.execute("SELECT count(*) AS n FROM checkpoint_writes WHERE thread_id = %s", (lease_a.run_id,))
            assert (await row.fetchone())["n"] == 0
        return {"旧执行器完整检查点": counts[0], "旧执行器pending writes": counts[1], "有效检查点数量": 2}

    async def expiry_transaction(self):
        entered, release = asyncio.Event(), asyncio.Event()
        lease_a = await self.lease(seconds=1)
        from contextlib import asynccontextmanager
        class BlockedSaver(LeaseFencedPostgresSaver):
            @asynccontextmanager
            async def _cursor(saver, *, pipeline=False):
                async with super()._cursor(pipeline=pipeline) as cur:
                    yield cur
                    if pipeline:
                        entered.set()
                        await release.wait()
        saving = asyncio.create_task(write(BlockedSaver(self.pool, lease=lease_a), lease_a.run_id, "到期前写入的blob"))
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.sleep(1.1)
        takeover = asyncio.create_task(self.lease(lease_a.run_id, "执行器乙"))
        await asyncio.sleep(0.05)
        assert not takeover.done(), "租约接管绕过了尚未结束的检查点事务"
        release.set()
        try:
            await saving
            raise AssertionError("到期事务没有回滚")
        except ResearchRunLeaseLost:
            pass
        lease_b = await asyncio.wait_for(takeover, 5)
        async with self.pool.connection() as conn:
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                row = await conn.execute(sql.SQL("SELECT count(*) AS n FROM {} WHERE thread_id = %s").format(sql.Identifier(table)), (lease_a.run_id,))
                assert (await row.fetchone())["n"] == 0
        await write(LeaseFencedPostgresSaver(self.pool, lease=lease_b), lease_b.run_id, "乙接管后的结果")
        return {"接管等待行锁": True, "到期事务三表回滚": True, "新执行器提交": "成功"}

    async def crash_resume(self):
        lease_a = await self.lease()
        save(self.out, "故障进程租约.json", lease_a.__dict__)
        proc = None
        async def launch(role, log):
            return await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).resolve()),
                "--worker", role, "--schema", self.schema, "--output-dir", str(self.out),
                stdout=log, stderr=asyncio.subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        try:
            with (self.out / "故障进程日志.txt").open("w") as log:
                proc = await launch("crash", log)
                async def ready():
                    reader = AsyncPostgresSaver(self.pool)
                    while True:
                        if proc.returncode is not None:
                            raise RuntimeError("故障进程提前退出")
                        if (self.out / "第三章崩溃屏障.json").exists():
                            item = await reader.aget_tuple(graph_config(lease_a.run_id))
                            artifacts = [v for _, c, v in item.pending_writes if c == "__return__" and isinstance(v, dict)]
                            if {v.get("section_id") for v in artifacts} == {"sec_1", "sec_2"}:
                                return
                        await asyncio.sleep(0.05)
                await asyncio.wait_for(ready(), 40)
                before = await read_runtime_budget(self.pool, lease_a.run_id)
                assert before["known_tokens"] == 200 and before["charged_tokens"] == 320
                save(self.out, "SIGKILL前独立预算.json", before)
                proc.kill()
                killed_code = await proc.wait()
            await self.expire(lease_a.run_id)
            lease_b = await self.lease(lease_a.run_id, "执行器乙")
            save(self.out, "续跑进程租约.json", lease_b.__dict__)
            with (self.out / "续跑进程日志.txt").open("w") as log:
                proc = await launch("resume", log)
                code = await asyncio.wait_for(proc.wait(), 40)
                if code:
                    raise RuntimeError("续跑进程失败，请查看隔离验证日志")
            final = json.loads((self.out / "章节续跑与预算结果.json").read_text())
            assert final["status"] == "completed"
            assert final["calls"] == {"sec_3": 1, "synthesis": 1}
            assert final["restored_sections"] == ["sec_1", "sec_2"]
            assert final["known_tokens"] == 400 and final["charged_tokens"] == final["checkpoint_charged_tokens"] == 520
            assert final["unknown_usage_calls"] == 1
            return {"SIGKILL退出码": killed_code, "已保存章节重复生成": 0, "累计已知token": 400,
                    "未知请求预留": 120, "最终扣账token": 520, "写作物理尝试": final["budget"]["node_attempts"]["write"]}
        finally:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()


async def controller(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "逐场景结果.json").exists():
        raise FileExistsError("请使用新的中文结果目录，不覆盖历史实验")
    schema = "budget_fence_test_" + uuid.uuid4().hex
    created, pool, engine = False, None, None
    result = {"评估日期": "2026-10-14", "时区": "Asia/Singapore", "隔离schema": schema,
              "真实模型或搜索请求": 0, "业务schema修改": False, "隔离schema已清理": False}
    try:
        async with await connect() as admin:
            await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            created = True
        async with await connect(schema) as conn:
            await conn.execute("CREATE TABLE users (id uuid PRIMARY KEY)")
            for model in (ResearchRun,):
                await conn.execute(str(CreateTable(model.__table__).compile(dialect=postgresql.dialect())))
            await AsyncPostgresSaver(conn).setup()
        info = conninfo_to_dict(checkpoint_database_url())
        engine = create_engine(URL.create("postgresql+psycopg", username=info.get("user"), password=info.get("password"),
            host=info.get("host"), port=int(info.get("port", 5432)), database=info["dbname"]),
            connect_args={"options": schema_options(schema), "connect_timeout": 5})
        pool = await open_pool(schema)
        validation = Validation(out, schema, pool, sessionmaker(bind=engine), engine)
        with patch("service.deep_research_v2.stage_runtime.is_research_cancelled", return_value=False), patch.dict(
            os.environ, {"RESEARCH_MAX_LLM_TOKENS": "0", "RESEARCH_TOTAL_BUDGET_SECONDS": "2400"}
        ):
            await validation.case("实际迁移升级、降级与重新升级", validation.migration)
            await validation.case("并发预留的硬预算约束", validation.concurrent_reservation)
            await validation.case("单连接入口的事务隔离", validation.single_connection_reservation)
            await validation.case("失败节点预算跨图恢复", validation.failed_budget)
            await validation.case("未知预留、幂等结算与历史一次导入", validation.reservation_and_legacy)
            await validation.case("旧执行器检查点及pending writes拒绝", validation.stale_writes)
            await validation.case("事务内自然到期回滚与接管行锁", validation.expiry_transaction)
            await validation.case("SIGKILL章节复用与累计预算跨进程恢复", validation.crash_resume)
        result.update(场景数=len(validation.results), 通过数=sum(r["通过"] for r in validation.results),
                      状态="通过" if all(r["通过"] for r in validation.results) else "失败")
    except Exception as exc:
        result.update(状态="失败", 错误类型=type(exc).__name__, 错误=str(exc), 堆栈=traceback.format_exc())
    finally:
        if pool is not None:
            await pool.close()
        if engine is not None:
            engine.dispose()
        if created:
            async with await connect() as admin:
                await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
                result["隔离schema已清理"] = True
        save(out, "PostgreSQL故障验证汇总.json", result)
        sources = [Path(__file__).resolve(), ROOT / "backend/app/service/deep_research_v2/checkpointer.py",
                   ROOT / "backend/app/service/deep_research_v2/budget_ledger.py",
                   ROOT / "backend/app/service/deep_research_v2/stage_runtime.py",
                   ROOT / "backend/app/service/deep_research_v2/agents/base.py"]
        save(out, "验证源码哈希.json", [{"路径": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources])
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result.get("状态") == "通过"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker", choices=["crash", "resume"])
    parser.add_argument("--schema")
    args = parser.parse_args()
    load_dotenv(BACKEND / ".env")
    if args.worker:
        asyncio.run(worker(args))
    else:
        sys.exit(0 if asyncio.run(controller(args)) else 1)
