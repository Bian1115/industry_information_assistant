"""固定假 Agent + 隔离 PostgreSQL schema，验证章节入库后的 SIGKILL 恢复。

只终止本脚本创建的子进程；不发模型/搜索请求，不操作业务 schema。
用法：backend/.venv/bin/python backend/eval/scripts/验证Writer章节跨进程恢复.py --output-dir 中文结果目录
"""

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "eval/tests")]

from dotenv import load_dotenv
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from service.deep_research_v2.checkpointer import checkpoint_database_url
from service.deep_research_v2.graph import graph_config, pack_state
from test_Writer章节检查点 import Scenario, collect, graph_for, initial_state


def save(out, name, data):
    (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


async def connect(schema=None):
    kwargs = {"autocommit": True, "row_factory": dict_row, "connect_timeout": 5}
    if schema:
        if not re.fullmatch(r"writer_checkpoint_test_[0-9a-f]{32}", schema):
            raise ValueError("仅允许隔离验证 schema")
        kwargs["options"] = f"-c search_path={schema}"
    return await psycopg.AsyncConnection.connect(checkpoint_database_url(), **kwargs)


async def worker(args):
    out = Path(args.output_dir)
    async with await connect(args.schema) as conn:
        saver = AsyncPostgresSaver(conn)
        scenario = Scenario()
        writer = scenario.writer()
        if args.worker == "crash":
            original = writer.call_llm

            async def call_llm(**kwargs):
                if "标题: 章节3\n" in kwargs["user_prompt"]:
                    scenario.calls["sec_3"] += 1
                    save(out, "第三章调用屏障.json", {"calls": dict(scenario.calls), "pid": os.getpid()})
                    await asyncio.Event().wait()
                return await original(**kwargs)

            writer.call_llm = call_llm
        graph = graph_for(saver, writer, args.run_id)
        events = await collect(graph, pack_state(initial_state()) if args.worker == "crash" else None,
                               args.run_id)
        result = (await graph.aget_state(graph_config(args.run_id))).values["research"]
        save(out, "续跑进程结果.json", {
            "calls": dict(scenario.calls), "verification_calls": len(scenario.verifications),
            "status": result["status"], "quality_verdict": result["quality_verdict"],
            "draft_sections": result["draft_sections"], "section_claims": result["section_claims"],
            "references": result["references"],
            "restored_sections": [e["content"]["section_id"] for e in events
                                  if e["type"] == "section_content" and e["content"].get("restored")],
        })
        (out / "续跑完成报告.txt").write_text(result["final_report"])


async def controller(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ["跨进程故障恢复结果.json", "第三章调用屏障.json"]):
        raise FileExistsError("请使用新的中文结果目录，禁止覆盖历史故障恢复验证")
    schema = "writer_checkpoint_test_" + uuid.uuid4().hex
    run_id = "writer-section-crash-" + uuid.uuid4().hex
    proc = None
    created = False
    result = {"evaluation_date": "2026-10-07", "scope": "真实PostgreSQL、隔离schema、两个独立进程、假Agent",
              "status": "failed", "run_id": run_id, "schema": schema,
              "model_or_search_requests": 0, "schema_cleaned": False}
    try:
        async with await connect() as admin:
            await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            created = True
        async with await connect(schema) as conn:
            await AsyncPostgresSaver(conn).setup()

        async def launch(role, handle):
            return await asyncio.create_subprocess_exec(
                sys.executable, str(Path(__file__).resolve()), "--worker", role,
                "--schema", schema, "--run-id", run_id, "--output-dir", str(out),
                stdout=handle, stderr=asyncio.subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED="1"))

        with (out / "故障进程日志.txt").open("w") as log:
            proc = await launch("crash", log)

            async def wait_for_saved_sections():
                async with await connect(schema) as conn:
                    saver = AsyncPostgresSaver(conn)
                    while True:
                        if proc.returncode is not None:
                            raise RuntimeError("故障进程在到达注入屏障前退出，请查看故障进程日志")
                        if (out / "第三章调用屏障.json").exists():
                            stored = await saver.aget_tuple(graph_config(run_id))
                            artifacts = [value for _, channel, value in stored.pending_writes
                                         if channel == "__return__" and isinstance(value, dict)
                                         and value.get("version") == 1]
                            if {a["section_id"] for a in artifacts} == {"sec_1", "sec_2"}:
                                return artifacts
                        await asyncio.sleep(0.1)

            artifacts = await asyncio.wait_for(wait_for_saved_sections(), timeout=40)
            save(out, "被终止前已提交章节.json", artifacts)
            proc.kill()
            result["killed_process_exit_code"] = await proc.wait()
        with (out / "续跑进程日志.txt").open("w") as log:
            proc = await launch("resume", log)
            code = await asyncio.wait_for(proc.wait(), timeout=40)
            if code:
                raise RuntimeError("续跑进程失败，请查看续跑进程日志")
        resumed = json.loads((out / "续跑进程结果.json").read_text())
        before = json.loads((out / "第三章调用屏障.json").read_text())
        assert before["calls"] == {"sec_1": 1, "sec_2": 1, "sec_3": 1}, before
        assert resumed["calls"] == {"sec_3": 1, "synthesis": 1}, resumed["calls"]
        assert resumed["verification_calls"] == 1
        assert resumed["status"] == "completed"
        assert resumed["restored_sections"] == ["sec_1", "sec_2"]
        for artifact in artifacts:
            key = artifact["section_id"]
            assert resumed["draft_sections"][key] == artifact["content"]
            assert resumed["section_claims"][key] == artifact["claims"]
        result.update(status="passed", committed_sections=[a["section_id"] for a in artifacts],
                      calls_before_kill=before["calls"], calls_after_restart=resumed["calls"],
                      verification_calls=resumed["verification_calls"],
                      restored_sections=resumed["restored_sections"],
                      completed_draft_count=len(resumed["draft_sections"]),
                      repeated_completed_chapter_calls=0)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)[:500])
    finally:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        if created:
            try:
                async with await connect() as admin:
                    await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
                result["schema_cleaned"] = True
            except Exception as exc:
                result.update(status="failed", cleanup_error_type=type(exc).__name__)
        save(out, "跨进程故障恢复结果.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return result["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker", choices=["crash", "resume"])
    parser.add_argument("--schema")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    load_dotenv(BACKEND / ".env")
    if args.worker:
        asyncio.run(worker(args))
    else:
        sys.exit(0 if asyncio.run(controller(args)) else 1)
