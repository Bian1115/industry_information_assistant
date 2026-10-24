"""四组真实研究配对实验；只在新建隔离 schema 内运行，不修改生产恢复实现。"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import types
import uuid

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND)]
CASES = [(8, "部分搜索"), (25, "第三章已持久化"), (26, "检查点提交前"),
         (27, "检查点提交后"), (28, "终态交付提交前")]
GROUPS = {"A": "无故障关闭检查点", "B": "无故障开启检查点",
          "C": "故障后从头重跑", "D": "同位置检查点续跑"}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def append(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps({"at": now(), "monotonic": time.monotonic(), **value},
                           ensure_ascii=False, default=str) + "\n")
        f.flush(); os.fsync(f.fileno())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def configure(schema):
    if not re.fullmatch(r"checkpoint_pair_[0-9a-f]{32}", schema):
        raise ValueError("只允许本运行器自建的隔离schema")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import core.database as db
    original = db.engine
    db.engine = create_engine(original.url, pool_pre_ping=True, hide_parameters=True,
        connect_args={"options": "-c search_path=" + schema, "connect_timeout": 8})
    db.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db.engine)
    original.dispose()
    import models  # noqa: F401 -- 注册所有表，禁止遗漏表回退到业务schema
    return db


async def open_pool(schema):
    from psycopg.conninfo import make_conninfo
    from service.deep_research_v2.checkpointer import build_pool, checkpoint_database_url
    pool = build_pool(make_conninfo(checkpoint_database_url(),
        options="-c search_path=" + schema, connect_timeout=8))
    await pool.open(); await pool.wait()
    return pool


def crash(folder, detail):
    save(folder / "故障触发.json", {"at": now(), "monotonic": time.monotonic(),
         "pid": os.getpid(), "signal": "SIGKILL", **detail})
    os.kill(os.getpid(), signal.SIGKILL)


def install_hooks(module, folder, identity, phase):
    fault = phase == "共同故障前缀"
    case = identity["fault_case"]
    original_factory = module.create_agents

    def agents(config):
        result = original_factory(config)
        for role, agent in result.items():
            if os.getenv("CHECKPOINT_PAIR_NON_THINKING") == "1":
                original_create = agent.client.chat.completions.create
                async def create_request(*arguments, _original=original_create, _role=role, **kwargs):
                    kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "enable_thinking": False}
                    append(folder / "供应商请求实参.jsonl", {"role":_role,
                        "model":kwargs.get("model"), "max_tokens":kwargs.get("max_tokens"),
                        "temperature":kwargs.get("temperature"), "extra_body":kwargs["extra_body"]})
                    return await _original(*arguments, **kwargs)
                agent.client.chat.completions.create = create_request
            original = agent.process
            async def process(state, original=original, role=role):
                append(folder / "执行轨迹.jsonl", {"kind": "node_start", "role": role,
                    "node": state.get("_node"), "attempt": state.get("_attempt"),
                    "iteration": state.get("iteration")})
                value = await original(state)
                append(folder / "执行轨迹.jsonl", {"kind": "node_return", "role": role,
                    "node": state.get("_node")})
                return value
            agent.process = process
        scout = result["scout"]
        original_search = scout._execute_search_once
        async def search(query, count=10, *, state=None):
            append(folder / "执行轨迹.jsonl", {"kind": "search_start", "query": query,
                "iteration": (state or {}).get("iteration")})
            value = await original_search(query, count, state=state)
            append(folder / "执行轨迹.jsonl", {"kind": "search_return", "query": query,
                "result_count": len(value) if value else 0})
            if fault and case == "部分搜索" and value:
                crash(folder, {"case": case, "boundary": "首次成功搜索返回、research未提交"})
            return value
        scout._execute_search_once = search
        writer = result["writer"]
        original_section = writer._write_section
        async def section(state, target):
            append(folder / "执行轨迹.jsonl", {"kind": "section_generate_start",
                "section_id": target["id"], "iteration": state.get("iteration")})
            value = await original_section(state, target)
            append(folder / "执行轨迹.jsonl", {"kind": "section_generate_return",
                "section_id": target["id"], "iteration": state.get("iteration")})
            return value
        writer._write_section = section
        return result
    module.create_agents = agents
    original_execution = module.for_execution
    committed_sections = set()

    def execution(saver, lease):
        bound = original_execution(saver, lease)
        original_put, original_writes = bound.aput, bound.aput_writes
        async def put(self, config, checkpoint, metadata, versions):
            work = (checkpoint.get("channel_values") or {}).get("research") or {}
            attempts = work.get("node_attempts") or {}
            target = bool("research" in versions and work.get("final_report")
                          and attempts.get("write") and not attempts.get("review"))
            if target and fault and case == "检查点提交前":
                save(folder / "故障边界业务状态.json", work)
                crash(folder, {"case": case, "boundary": "Writer成功pending writes之后、后继aput之前"})
            start = time.monotonic()
            value = await original_put(config, checkpoint, metadata, versions)
            append(folder / "执行轨迹.jsonl", {"kind": "checkpoint_committed",
                "seconds": time.monotonic() - start, "checkpoint_id": checkpoint["id"]})
            if target and fault and case == "检查点提交后":
                save(folder / "故障边界业务状态.json", work)
                crash(folder, {"case": case, "boundary": "Writer后继aput确认提交、review开始前"})
            return value
        async def writes(self, config, values, task_id, task_path=""):
            start = time.monotonic()
            value = await original_writes(config, values, task_id, task_path)
            artifacts = [v for k, v in values if k == "__return__" and isinstance(v, dict)
                         and v.get("version") == 1 and v.get("section_id")]
            append(folder / "执行轨迹.jsonl", {"kind": "pending_writes_committed",
                "seconds": time.monotonic() - start, "task_id": task_id,
                "channels": [k for k, _ in values], "section_artifacts": artifacts})
            for artifact in artifacts:
                committed_sections.add(artifact["section_id"])
            if fault and case == "第三章已持久化" and len(committed_sections) >= 3:
                crash(folder, {"case": case, "boundary": "第三个唯一章节task返回值已持久化",
                    "committed_section_ids": sorted(committed_sections)})
            return value
        bound.aput = types.MethodType(put, bound)
        bound.aput_writes = types.MethodType(writes, bound)
        return bound
    module.for_execution = execution
    original_build = module.build_graph

    class MonitoredGraph:
        def __init__(self, graph, disabled):
            self.graph, self.disabled, self.last = graph, disabled, None
            self.live_id = "live-terminal-" + str(uuid.uuid4())
        def __getattr__(self, key):
            return getattr(self.graph, key)
        async def astream(self, value, **kwargs):
            kwargs["stream_mode"] = ["custom", "values"]
            if self.disabled:
                kwargs.pop("durability", None)
            async for mode, event in self.graph.astream(value, **kwargs):
                if mode == "values":
                    self.last = event
                    save(folder / "最近内存业务状态.json", event)
                else:
                    yield event
        async def aget_state(self, config):
            if not self.disabled:
                return await self.graph.aget_state(config)
            return types.SimpleNamespace(values=self.last or {}, next=(), tasks=(),
                config={"configurable": {"thread_id": identity["run_id"], "checkpoint_id": self.live_id}})

    def build(agents, saver, **kwargs):
        disabled = identity["group"] == "A"
        return MonitoredGraph(original_build(agents, False if disabled else saver, **kwargs), disabled)
    module.build_graph = build
    original_commit_factory = module.make_commit_result
    def commit_factory(**kwargs):
        commit = original_commit_factory(**kwargs)
        def invoke(snapshot):
            save(folder / "交付前图状态.json", {"values": snapshot.values,
                "config": snapshot.config, "next": list(snapshot.next)})
            if fault and case == "终态交付提交前":
                crash(folder, {"case": case, "boundary": "终态检查点已提交、业务交付事务前"})
            result = commit(snapshot)
            append(folder / "执行轨迹.jsonl", {"kind": "delivery_committed"})
            return result
        return invoke
    module.make_commit_result = commit_factory


async def worker(args):
    folder = args.folder
    identity = read(folder / "运行身份.json")
    configure(args.schema)
    os.environ["EVAL_TRACE_ENABLED"] = "true"
    os.environ["EVAL_TRACE_PAYLOAD_CHARS"] = "10000000"
    os.environ["EVAL_TRACE_LOCAL_FILE"] = str(folder / "完整模型工具埋点.jsonl")
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from service.deep_research_v2 import service as module
    from service.research_events import acquire_research_run_lease, ResearchRunLeaseConflict
    from eval.scripts.run_deepresearch_bench import decode_service_event
    install_hooks(module, folder, identity, args.phase)
    pool = await open_pool(args.schema)
    status = {"phase": args.phase, "started_at": now(), "started_monotonic": time.monotonic(),
              "completed": False, "errors": [], "lease_wait_seconds": 0}
    save(folder / "执行状态.json", status)
    try:
        lease_started = time.monotonic()
        while True:
            try:
                lease = await asyncio.to_thread(acquire_research_run_lease,
                    run_id=identity["run_id"], session_id=identity["session_id"], user_id=identity["user_id"])
                break
            except ResearchRunLeaseConflict:
                if time.monotonic() - lease_started > 130:
                    raise
                await asyncio.sleep(1)
        status["lease_wait_seconds"] = time.monotonic() - lease_started
        save(folder / "执行状态.json", status)
        service = module.DeepResearchV2Service(AsyncPostgresSaver(pool))
        async for chunk in service.research(query=identity["query"], session_id=identity["session_id"],
            user_id=identity["user_id"], run_id=identity["run_id"], turn_id=identity["turn_id"],
            resume=args.phase == "检查点续跑", lease=lease,
            research_context={"original_query": identity["query"], "resolved_query": identity["query"],
                "is_follow_up": False, "conversation_context": [], "memory_context": []}):
            event = decode_service_event(chunk)
            if event is None:
                continue
            append(folder / "研究事件.jsonl", {"event": event})
            if event["type"] == "research_complete":
                payload = event.get("data") or event
                status["completed"] = bool(payload.get("final_report"))
                (folder / "研究报告.md").write_text(payload.get("final_report") or "")
            if event["type"] in {"error", "research_paused", "research_cancelled"}:
                status["errors"].append(event)
    except BaseException as exc:
        status.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        status.update(ended_at=now(), duration_seconds=time.monotonic()-status["started_monotonic"])
        save(folder / "执行状态.json", status)
        await pool.close()


async def export_run(folder, schema):
    identity = read(folder / "运行身份.json")
    from service.deep_research_v2.service import DeepResearchV2Service
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from sqlalchemy import text
    from core.database import engine
    pool = await open_pool(schema)
    try:
        if identity["group"] == "A":
            live = folder / "最近内存业务状态.json"
            state = {"research": read(live).get("research", {})} if live.exists() else {"research": {}}
            state["persistence_disabled"] = True
        else:
            state = await DeepResearchV2Service(AsyncPostgresSaver(pool)).read_state(identity["run_id"])
        save(folder / "数据库与图状态.json", state)
        with engine.connect() as connection:
            records = {}
            for name, query in {
                "工作历史": "SELECT * FROM agent_messages WHERE run_id=:run ORDER BY created_at,id",
                "事件账本": "SELECT * FROM agent_events WHERE run_id=:run ORDER BY seq",
                "正式报告记录": "SELECT * FROM chat_messages WHERE run_id=:run AND role='assistant'",
                "运行记录": "SELECT * FROM research_runs WHERE run_id=:run",
                "预算调用": "SELECT c.* FROM research_budget_calls c JOIN research_budget_attempts a ON a.attempt_id=c.attempt_id WHERE a.run_id=:run ORDER BY a.started_at,c.call_id",
                "预算尝试": "SELECT * FROM research_budget_attempts WHERE run_id=:run ORDER BY started_at",
            }.items():
                rows = [dict(r) for r in connection.execute(text(query), {"run": identity["run_id"]}).mappings()]
                save(folder / (name + ".json"), rows); records[name] = rows
            counts = {}
            for table in ["checkpoints", "checkpoint_writes", "checkpoint_blobs"]:
                row = connection.execute(text(f"SELECT count(*) AS rows, coalesce(sum(pg_column_size(t)),0) AS bytes FROM {table} t WHERE thread_id=:run"), {"run": identity["run_id"]}).mappings().one()
                counts[table] = dict(row)
            save(folder / "检查点存储统计.json", counts)
        report = (state.get("research") or {}).get("final_report") or ""
        messages = records["正式报告记录"]
        complete = [r for r in records["事件账本"] if r["event_type"] == "research_complete"]
        consistent = bool(report and len(messages) == len(complete) == 1
            and digest(report.strip()) == digest(messages[0]["content"].strip())
            == digest(complete[0]["payload"]["final_report"].strip()))
        save(folder / "交付一致性.json", {"consistent": consistent, "report_sha256": digest(report.strip()),
            "report_records": len(messages), "completion_events": len(complete)})
    finally:
        await pool.close()


def create_identity(folder, owner, query, source_id, case, group, inherited=None):
    from models.chat import ChatSession, ChatMessage
    from core.database import SessionLocal
    if inherited:
        identity = {**inherited, "group": group}
    else:
        identity = {"source_id": source_id, "fault_case": case, "group": group,
            "query": query, "user_id": str(owner), "session_id": str(uuid.uuid4()),
            "run_id": str(uuid.uuid4()), "turn_id": str(uuid.uuid4())}
        with SessionLocal() as db:
            db.add(ChatSession(id=uuid.UUID(identity["session_id"]), user_id=owner,
                title=f"四组恢复评测 原题{source_id} {group}", session_type="deepsearch"))
            db.flush()
            db.add(ChatMessage(session_id=uuid.UUID(identity["session_id"]),
                turn_id=uuid.UUID(identity["turn_id"]), role="user", content=query,
                execution_mode="research", run_id=identity["run_id"], research_status="running"))
            db.commit()
    save(folder / "运行身份.json", identity)
    return identity


def run_child(script, folder, phase, schema, timeout):
    command = [sys.executable, "-u", str(script), "--worker", "--folder", str(folder),
               "--phase", phase, "--schema", schema]
    started = time.monotonic()
    with (folder / "进程日志.txt").open("w") as log:
        process = subprocess.Popen(command, cwd=script.parents[2], stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
        save(folder / "进程启动.json", {"pid": process.pid, "started_at": now(), "phase": phase})
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
            save(folder / "外部超时.json", {"limit_seconds": timeout, "at": now()})
    result = {"exit_code": process.returncode, "wall_seconds_including_process_startup": time.monotonic()-started, "ended_at": now()}
    save(folder / "进程计时.json", result)
    return result


def controller(args):
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env", override=False)
    if args.main_model:
        for role in ["architect", "data_analyst", "wizard", "writer", "critic", "summary"]:
            os.environ["LLM_MODEL_" + role.upper()] = args.main_model
        os.environ["OPENAI_MODEL"] = args.main_model
        os.environ["LLM_FALLBACK_MODEL"] = args.main_model
    if args.non_thinking:
        os.environ["CHECKPOINT_PAIR_NON_THINKING"] = "1"
    import psycopg
    from psycopg import sql
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    schema = "checkpoint_pair_" + uuid.uuid4().hex
    manifest = {"date": "2026-10-15", "timezone": "Asia/Singapore", "status": "准备中",
        "started_at": now(), "schema": schema, "groups": GROUPS,
        "fault_cases": CASES, "shared_fault_prefix": True, "real_models": True,
        "section_tasks_enabled": True, "host_awake_guard": os.getenv("CHECKPOINT_PAIR_SLEEP_GUARD", ""),
        "thinking_mode":"disabled" if args.non_thinking else "provider_default",
        "completed_phases": [], "errors": []}
    if args.main_model:
        manifest["evaluation_model_override"] = {"main_and_summary": args.main_model,
            "reason": "原配置qwen3-max返回404 model_not_found；按供应商实际可用目录统一替换，所有组一致，未修改.env或生产配置。"}
    save(out / "运行清单.json", manifest)
    source = args.dataset.resolve()
    tasks = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    assert [t["source"]["id"] for t in tasks] == [sid for sid, _ in CASES]
    save(out / "冻结五题故障注入输入.json", [{"source_id": t["source"]["id"],
        "fault_case": case, "query": t["turns"][0]["user_query"]} for t, (_, case) in zip(tasks, CASES)])
    shutil.copyfile(source, out / "原始开发集五题.jsonl")
    snapshot = out / "冻结源码" / "backend"
    shutil.copytree(BACKEND / "app", snapshot / "app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(BACKEND / "eval", snapshot / "eval", ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc"))
    script = snapshot / "eval/scripts" / Path(__file__).name
    save(out / "源码哈希.json", {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in snapshot.rglob("*") if p.is_file()})
    created = False
    try:
        with psycopg.connect(host=os.getenv("POSTGRES_HOST", "localhost"), port=os.getenv("POSTGRES_PORT", "5432"),
            user=os.getenv("POSTGRES_USER", "postgres"), password=os.getenv("POSTGRES_PASSWORD", "postgres123"),
            dbname=os.getenv("POSTGRES_DB", "industry_assistant"), connect_timeout=8, autocommit=True) as admin:
            admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        created = True
        db = configure(schema)
        db.Base.metadata.create_all(db.engine)
        async def setup():
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            pool = await open_pool(schema)
            try:
                await AsyncPostgresSaver(pool).setup()
            finally:
                await pool.close()
        asyncio.run(setup())
        from models.user import User
        owner = uuid.uuid4()
        with db.SessionLocal() as session:
            session.add(User(id=owner, username="eval_pair_" + owner.hex[:20],
                email=owner.hex+"@eval.invalid", hashed_password="!disabled-evaluation", is_active=False))
            session.commit()
        from config.llm_config import get_config
        from service.context_policy import selected_policy
        config = get_config()
        policy = selected_policy().snapshot()
        save(out / "冻结上下文配置.json", policy)
        os.environ["CONTEXT_ABLATION_CONFIG"] = str(out / "冻结上下文配置.json")
        manifest.update(status="运行中", dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            models={r:getattr(config.agents,r).to_dict() for r in ["architect","scout","data_analyst","writer","critic"]},
            max_iterations=config.research.max_iterations, context_policy_hash=policy["config_hash"])
        save(out / "运行清单.json", manifest)
        manifest_lock = threading.Lock()
        def one_case(task, sid, case):
            query = task["turns"][0]["user_query"]
            base = out / f"题目{sid}-{case}"; base.mkdir()
            for group in (["A", "B"] if sid % 2 == 0 else ["B", "A"]):
                folder = base / (group + "组-" + GROUPS[group]); folder.mkdir()
                create_identity(folder, owner, query, sid, case, group)
                print(json.dumps({"event":"start","source_id":sid,"group":group},ensure_ascii=False),flush=True)
                run_child(script, folder, "正常运行", schema, args.timeout)
                asyncio.run(export_run(folder, schema))
                with manifest_lock:
                    manifest["completed_phases"].append({"source_id":sid,"group":group})
                    save(out / "运行清单.json", manifest)
                print(json.dumps({"event":"end","source_id":sid,"group":group},ensure_ascii=False),flush=True)
            prefix = base / "共同故障前缀"; prefix.mkdir()
            identity = create_identity(prefix, owner, query, sid, case, "D")
            print(json.dumps({"event":"start","source_id":sid,"group":"共同故障前缀"},ensure_ascii=False),flush=True)
            run_child(script, prefix, "共同故障前缀", schema, args.timeout)
            asyncio.run(export_run(prefix, schema))
            if not (prefix / "故障触发.json").exists():
                with manifest_lock:
                    manifest["errors"].append({"source_id":sid,"error":"目标故障未触发"})
            else:
                for group in ["D", "C"]:
                    folder = base / (group + "组-" + GROUPS[group]); folder.mkdir()
                    create_identity(folder, owner, query, sid, case, group, identity if group=="D" else None)
                    print(json.dumps({"event":"start","source_id":sid,"group":group},ensure_ascii=False),flush=True)
                    run_child(script, folder, "检查点续跑" if group=="D" else "从头重跑", schema, args.timeout)
                    asyncio.run(export_run(folder, schema))
                    with manifest_lock:
                        manifest["completed_phases"].append({"source_id":sid,"group":group})
                        save(out / "运行清单.json", manifest)
                    print(json.dumps({"event":"end","source_id":sid,"group":group},ensure_ascii=False),flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(one_case, task, sid, case) for task,(sid,case) in zip(tasks,CASES)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        manifest.update(status="运行结束", ended_at=now())
    except BaseException as exc:
        manifest.update(status="中断", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        manifest["schema_preserved_for_analysis"] = created
        save(out / "运行清单.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, choices=[1,2], default=2)
    parser.add_argument("--main-model")
    parser.add_argument("--non-thinking", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--schema")
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--phase")
    args = parser.parse_args()
    if args.worker:
        asyncio.run(worker(args))
    else:
        controller(args)


if __name__ == "__main__":
    main()
