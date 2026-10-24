"""真实服务同首稿Critic消融，只生成和归档，不调用质量裁判。"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import dataclasses
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from unittest.mock import patch
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


async def consume(service, identity, folder, *, resume=False):
    from eval.scripts.run_deepresearch_bench import decode_service_event
    events, completed, error = [], None, None
    started = time.monotonic()
    with (folder / "SSE事件.jsonl").open("a") as stream:
        async for chunk in service.research(query=identity["query"], session_id=identity["session_id"],
                user_id=identity["user_id"], run_id=identity["run_id"], turn_id=identity["turn_uuid"],
                resume=resume, critic_enabled=identity["critic_enabled"]):
            event = decode_service_event(chunk)
            if event is None:
                continue
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            events.append(event)
            kind = event["type"]
            data = event.get("data") or event
            if kind == "research_complete":
                completed = data
            elif kind == "error":
                error = data
            if kind in {"phase", "research_complete", "error", "research_paused"}:
                save(folder.parent / "题目进度.json", {"updated_at": now(), "id": identity["id"],
                     "group": folder.name, "stage": kind, "phase": data.get("phase"),
                     "iteration": data.get("iteration"), "content": data.get("content")})
    return {"active_seconds": time.monotonic() - started, "completed": completed,
            "error": error, "event_count": len(events)}


async def export(service, identity, folder):
    from eval.scripts.采集通用研究评测集 import export_database
    from eval.runtime import tracer
    from service.deep_research_v2.graph import graph_config
    await asyncio.to_thread(tracer._queue.join)
    counts = await asyncio.to_thread(export_database, {"research_run_id": identity["run_id"],
                "session_id": identity["session_id"]}, folder)
    states = []
    async for item in service.checkpointer.alist(graph_config(identity["run_id"])):
        states.append({"config": item.config, "checkpoint": item.checkpoint, "metadata": item.metadata,
                       "parent_config": item.parent_config, "pending_writes": item.pending_writes})
    save(folder / "完整检查点历史.json", states)
    snapshot = await service.checkpointer.aget_tuple(graph_config(identity["run_id"]))
    state = deepcopy(snapshot.checkpoint["channel_values"]["research"]) if snapshot else {}
    save(folder / "研究状态.json", state)
    def evidence():
        from core.database import SessionLocal
        from models.session_evidence import SessionEvidenceDocument, SessionEvidenceChunk
        with SessionLocal() as db:
            docs = db.query(SessionEvidenceDocument).filter_by(session_id=uuid.UUID(identity["session_id"])).all()
            chunks = db.query(SessionEvidenceChunk).filter(SessionEvidenceChunk.document_key.in_([d.id for d in docs])).all()
            return ([deepcopy(d.payload) for d in docs],
                    [{c.name: getattr(row, c.name) for c in row.__table__.columns} for row in chunks])
    docs, chunks = await asyncio.to_thread(evidence)
    save(folder / "完整来源正文.json", docs or state.get("raw_sources") or [])
    save(folder / "证据索引快照.json", chunks)
    calls = read(folder / "LLM调用埋点.json")
    usage = {"calls": len(calls), "input_tokens": sum(r.get("prompt_tokens") or 0 for r in calls),
             "output_tokens": sum(r.get("completion_tokens") or 0 for r in calls),
             "total_tokens": sum(r.get("total_tokens") or 0 for r in calls),
             "calls_missing_usage": sum(r.get("total_tokens") is None for r in calls)}
    return state, calls, {"archive_counts": counts, "usage": usage}


def clone_stores(a, b, state):
    """复制数据库中的历史与原文，B不做检索，向量索引标为未建立。"""
    from core.database import SessionLocal
    from models.agent_history import AgentContext, AgentMessage
    from models.session_evidence import SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk
    from service.agent_history import HistoryScope, research_history_scope_id, canonical_json
    from service.session_evidence_chunks import stable_id
    scope_id = research_history_scope_id(HistoryScope(store=None, run_id=b["run_id"],
                                       session_id=b["session_id"], user_id=b["user_id"]))
    counts = {"contexts": 0, "messages": 0, "documents": 0, "chunks": 0}
    with SessionLocal.begin() as db:
        contexts = db.query(AgentContext).filter_by(run_id=a["run_id"]).all()
        for old in contexts:
            ident = uuid.uuid5(uuid.NAMESPACE_URL, canonical_json([scope_id, old.agent_name, old.context_key]))
            values = {c.name: deepcopy(getattr(old, c.name)) for c in old.__table__.columns}
            values.update(id=ident, run_id=b["run_id"], history_scope_id=scope_id)
            db.add(AgentContext(**values)); db.flush(); counts["contexts"] += 1
            for msg in db.query(AgentMessage).filter_by(context_id=old.id).order_by(AgentMessage.seq).all():
                values = {c.name: deepcopy(getattr(msg, c.name)) for c in msg.__table__.columns}
                values.update(id=uuid.uuid4(), context_id=ident, run_id=b["run_id"])
                db.add(AgentMessage(**values)); counts["messages"] += 1
        space = db.get(SessionEvidenceSpace, uuid.UUID(a["session_id"]))
        if space:
            values = {c.name: deepcopy(getattr(space, c.name)) for c in space.__table__.columns}
            values.update(session_id=uuid.UUID(b["session_id"]), user_id=uuid.UUID(b["user_id"]))
            db.add(SessionEvidenceSpace(**values)); db.flush()
            for doc in db.query(SessionEvidenceDocument).filter_by(session_id=uuid.UUID(a["session_id"])).all():
                key = stable_id(b["session_id"], doc.topic_id, doc.document_id)
                db.add(SessionEvidenceDocument(id=key, session_id=uuid.UUID(b["session_id"]),
                       topic_id=doc.topic_id, document_id=doc.document_id, payload=deepcopy(doc.payload)))
                db.flush(); counts["documents"] += 1
                for chunk in db.query(SessionEvidenceChunk).filter_by(document_key=doc.id).all():
                    locator = deepcopy(chunk.locator)
                    locator.update(session_id=b["session_id"], user_id=b["user_id"])
                    db.add(SessionEvidenceChunk(id=stable_id(b["session_id"], chunk.id), document_key=key,
                           unit_type=chunk.unit_type, content=chunk.content, locator=locator, index_status="pending"))
                    counts["chunks"] += 1
            state["evidence_space"].update(session_id=b["session_id"], user_id=b["user_id"])
    return counts


async def worker(args):
    from eval.scripts.run_deepresearch_bench import benchmark_service
    from eval.scripts.采集真实追问开发集 import create_identity, create_session, save_user
    from service.deep_research_v2 import service as service_module
    from service.deep_research_v2.graph import build_graph, create_agents, graph_config, pack_state
    from service.deep_research_v2.stage_runtime import make_execute_stage
    from service.research_events import acquire_research_run_lease, release_research_run_lease
    task = next(t for t in read(args.output / "固定五题.json") if t["id"] == args.only_id)
    directory = args.output / f"题目{task['id']}"
    directory.mkdir(parents=True, exist_ok=True)
    folders = {name: directory / name for name in ["公共首稿", "开启审核", "关闭审核"]}
    for folder in folders.values(): folder.mkdir(exist_ok=True)
    os.environ["EVAL_TRACE_LOCAL_FILE"] = str(directory / "完整过程埋点.jsonl")
    os.environ["WEB_READER_CACHE_DIR"] = str(directory / "网页缓存")
    owner = await asyncio.to_thread(create_identity, directory.name)
    identities = {}
    for name, enabled in [("开启审核", True), ("关闭审核", False)]:
        sid = await asyncio.to_thread(create_session, owner, f"Critic消融 {name} 原题{task['id']}")
        item = {"id": task["id"], "query": task["prompt"], "user_id": owner,
                "session_id": sid, "run_id": str(uuid.uuid4()), "turn_uuid": str(uuid.uuid4()),
                "critic_enabled": enabled}
        await asyncio.to_thread(save_user, owner, sid, item["turn_uuid"], item["run_id"], item["query"])
        identities[name] = item
        save(folders[name] / "运行身份.json", item)
    result = {"id": task["id"], "started_at": now(), "status": "running", "groups": {},
              "quality_evaluator": "Codex主代理直接审读，不调用外部LLM裁判"}
    save(directory / "配对运行结果.json", result)
    try:
        async with benchmark_service(2) as service:
            a = identities["开启审核"]
            # 在首稿已提交、审核未启动的位置暂停一次真实服务执行。
            def prefix_graph(*pos, **kw):
                return build_graph(*pos, **kw, interrupt_before=["review"])
            with patch.object(service_module, "build_graph", prefix_graph):
                prefix = await asyncio.wait_for(consume(service, a, folders["公共首稿"]), args.timeout)
            state, prefix_calls, prefix_export = await export(service, a, folders["公共首稿"])
            if prefix["error"] or not state.get("final_report") or (state.get("node_attempts") or {}).get("review", 0):
                raise RuntimeError("公共首稿未正常到达审核前冻结点：" + str(prefix["error"]))
            assert not state.get("critic_feedback") and state["quality_verdict"] == "not_reviewed"
            (folders["公共首稿"] / "研究报告.md").write_text(state["final_report"])
            prefix_hash = digest(state["final_report"])
            save(folders["公共首稿"] / "公共成本.json", {**prefix, **prefix_export, "report_sha256": prefix_hash})
            b = identities["关闭审核"]
            bstate = deepcopy(state)
            bstate.update(critic_enabled=False, session_id=b["session_id"], ablation_parent_run_id=a["run_id"])
            lease = await asyncio.to_thread(acquire_research_run_lease, run_id=b["run_id"],
                         session_id=b["session_id"], user_id=owner)
            await asyncio.to_thread(release_research_run_lease, lease, "paused")
            copies = await asyncio.to_thread(clone_stores, a, b, bstate)
            # 新thread用真实write后继路由提交冻结状态；只调度finish。
            bgraph = build_graph(create_agents(service.config), service.checkpointer, run_id=b["run_id"], user_id=owner,
                         session_id=b["session_id"], execute_stage=make_execute_stage(
                         run_id=b["run_id"], session_id=b["session_id"]),
                         history_store=None)
            await bgraph.aupdate_state(graph_config(b["run_id"]), pack_state(bstate), as_node="write")
            bresult = await consume(service, b, folders["关闭审核"], resume=True)
            final_b, b_calls, b_export = await export(service, b, folders["关闭审核"])
            assert bresult["completed"] and not bresult["error"], bresult
            assert not b_calls and final_b["completion_reason"] == "critic_disabled"
            assert digest(final_b["final_report"]) == prefix_hash
            (folders["关闭审核"] / "研究报告.md").write_text(final_b["final_report"])
            result["groups"]["关闭审核"] = {"success": True, "branch_seconds": bresult["active_seconds"],
                    "logical_total_seconds": prefix["active_seconds"] + bresult["active_seconds"],
                    "new_usage": b_export["usage"], "shared_prefix_usage": prefix_export["usage"],
                    "completion_reason": final_b["completion_reason"], "quality_verdict": "not_reviewed",
                    "internal_quality_score": None, "report_sha256": prefix_hash, "clone_counts": copies}
            save(directory / "配对运行结果.json", result)
            afinal = await asyncio.wait_for(consume(service, a, folders["开启审核"], resume=True), args.timeout)
            final_a, all_a_calls, a_export = await export(service, a, folders["开启审核"])
            old_ids = {r["id"] for r in prefix_calls}
            branch_calls = [r for r in all_a_calls if r["id"] not in old_ids]
            save(folders["开启审核"] / "审核返工新增调用.json", branch_calls)
            success = bool(afinal["completed"] and final_a.get("final_report") and not afinal["error"])
            if success: (folders["开启审核"] / "研究报告.md").write_text(final_a["final_report"])
            result["groups"]["开启审核"] = {"success": success, "error": afinal["error"],
                 "branch_seconds": afinal["active_seconds"],
                 "logical_total_seconds": prefix["active_seconds"] + afinal["active_seconds"],
                 "full_usage": a_export["usage"], "new_calls": len(branch_calls),
                 "new_tokens": sum(r.get("total_tokens") or 0 for r in branch_calls),
                 "completion_reason": final_a.get("completion_reason"), "quality_verdict": final_a.get("quality_verdict"),
                 "node_attempts": final_a.get("node_attempts"), "iteration": final_a.get("iteration"),
                 "report_sha256": digest(final_a.get("final_report", "")) if success else None,
                 "search_budget": final_a.get("search_budget")}
            result.update(status="completed" if success else "partial", prefix_report_sha256=prefix_hash,
                          prefix_seconds=prefix["active_seconds"], prefix_usage=prefix_export["usage"])
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result["finished_at"] = now()
        save(directory / "配对运行结果.json", result)
        print(json.dumps({"id": task["id"], "status": result["status"], "groups": list(result["groups"]),
                          "error": result.get("error")}, ensure_ascii=False), flush=True)


def prepare(args):
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = read(BACKEND / "eval/data/Critic通用研究五题消融设计更正-20261009/通用研究五题冻结输入.json")
    assert [t["id"] for t in tasks] == [35, 31, 46, 16, 49]
    save(args.output / "固定五题.json", tasks)
    # 配置仅作用于本实验的独立进程，凭据不写入产物。
    database = "critic_eval_" + datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S")
    import psycopg
    from psycopg import sql
    with psycopg.connect(host=os.getenv("POSTGRES_HOST", "localhost"), port=os.getenv("POSTGRES_PORT", "5432"),
            user=os.getenv("POSTGRES_USER", "postgres"), password=os.getenv("POSTGRES_PASSWORD", "postgres123"),
            dbname="postgres", autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    os.environ["POSTGRES_DB"] = database
    os.environ.pop("RESEARCH_CHECKPOINT_DATABASE_URL", None)
    os.environ.update(EVAL_TRACE_ENABLED="true", EVAL_TRACE_PAYLOAD_CHARS="10000000", EVAL_TRACE_QUEUE_SIZE="10000")
    from core.database import Base, engine
    import models
    import eval.models
    Base.metadata.create_all(engine)
    from service.deep_research_v2.checkpointer import prepare_checkpoint_schema
    asyncio.run(prepare_checkpoint_schema())
    from config.llm_config import get_config
    from service.context_policy import selected_policy
    from service.session_evidence_rag import SessionEvidenceSettings
    from service.deep_research_v2.citation_verification import resolve_citation_verification_options
    from service.deep_research_v2.runtime_budget import total_budget_seconds, max_llm_tokens
    config = get_config()
    save(args.output / "运行配置.json", {"created_at": now(), "database": database,
         "agents": dataclasses.asdict(config.agents), "research": dataclasses.asdict(config.research),
         "max_iterations": 2, "context_policy": selected_policy().snapshot(),
         "session_evidence": SessionEvidenceSettings.from_env().model_dump(),
         "citation_verification": resolve_citation_verification_options(None).model_dump(),
         "total_seconds": total_budget_seconds(), "token_limit": max_llm_tokens(),
         "external_timeout_seconds_per_service_part": args.timeout, "concurrency": args.concurrency,
         "quality_evaluation": "Codex主代理直接评估；禁止运行任何外部LLM裁判",
         "design": "每题同首稿分叉；关闭组先正常交付，开启组继续生产审核返工；只跑通用单轮五题",
         "generation_only": True})
    from eval.scripts.运行假设开关五题对照 import freeze
    runtime = freeze(args.output)
    save(args.output / "运行清单.json", {"started_at": now(), "status": "running", "source_ids": [t["id"] for t in tasks],
         "database": database, "runtime_source": str(runtime), "planned_reports": 10,
         "assessment": "Codex直接审读正文和证据，无外部LLM评分"})
    return tasks, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-id", type=int)
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args(); args.output = args.output.resolve()
    if args.only_id:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env")
        asyncio.run(worker(args)); return
    tasks, runtime = prepare(args)
    script = runtime / "backend/eval/scripts/运行Critic通用研究五题消融.py"
    pending = list(tasks); running = []; finished = []
    guard = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-m", "-s", "-w", str(os.getpid())])
    try:
        while pending or running:
            while pending and len(running) < args.concurrency:
                task = pending.pop(0); folder = args.output / f"题目{task['id']}"; folder.mkdir(exist_ok=True)
                log = (folder / "运行日志.log").open("w")
                proc = subprocess.Popen([sys.executable, "-u", str(script), "--output", str(args.output),
                        "--only-id", str(task["id"]), "--timeout", str(args.timeout)], cwd=runtime,
                        env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                running.append({"task": task, "proc": proc, "log": log, "started": time.monotonic()})
                print(json.dumps({"event": "题目启动", "id": task["id"], "pid": proc.pid}, ensure_ascii=False), flush=True)
            remaining = []
            for item in running:
                proc = item["proc"]
                if proc.poll() is None:
                    if time.monotonic() - item["started"] > args.timeout * 2 + 300:
                        proc.terminate()
                    remaining.append(item); continue
                item["log"].close()
                path = args.output / f"题目{item['task']['id']}" / "配对运行结果.json"
                outcome = read(path) if path.exists() else {"id": item["task"]["id"], "status": "failed", "error": "进程无结果"}
                outcome["exit_code"] = proc.returncode; finished.append(outcome)
                print(json.dumps({"event": "题目结束", "id": item["task"]["id"], "status": outcome["status"]}, ensure_ascii=False), flush=True)
            running = remaining
            save(args.output / "运行状态.json", {"updated_at": now(), "phase": "生成中" if pending or running else "生成结束待Codex评估",
                "running": [{"id": x["task"]["id"], "pid": x["proc"].pid} for x in running],
                "pending_ids": [t["id"] for t in pending], "finished": finished,
                "reports_completed": sum(g.get("success", False) for r in finished for g in r.get("groups", {}).values())})
            if pending or running: time.sleep(5)
        save(args.output / "生成结果汇总.json", finished)
    finally:
        guard.terminate(); guard.wait(timeout=5)


if __name__ == "__main__":
    main()
