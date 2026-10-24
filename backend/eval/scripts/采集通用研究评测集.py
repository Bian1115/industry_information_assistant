"""随机中文原题运行与完整轨迹采集；不执行后续裁判评分。"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


def serial(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serial), encoding="utf-8")


def prepare(output, development_manifest, seed):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "抽样清单.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        tasks = [json.loads(line) for line in (output / "中文原题评测集.jsonl").read_text().splitlines()]
        source = (output / "官方题库快照.jsonl").read_bytes()
        assert hashlib.sha256(source).hexdigest() == manifest["source_file_sha256"]
        originals = {str(row["id"]): row for row in map(json.loads, source.splitlines())}
        assert all(task == originals[str(task["id"])] for task in tasks), "评测集原文已变化"
        assert [task["id"] for task in tasks] == manifest["evaluation_ids"]
        assert len(tasks) == 5 and all(t["language"] == "zh" for t in tasks)
        assert not set(manifest["evaluation_ids"]) & set(manifest["development_ids"])
        return tasks, manifest
    split = json.loads(development_manifest.read_text())
    with urllib.request.urlopen(split["source_url"], timeout=30) as response:
        source = response.read()
    assert hashlib.sha256(source).hexdigest() == split["source_file_sha256"], "题库版本与开发集不一致"
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    excluded = set(map(str, split["assigned"]["development"]))
    candidates = sorted([row for row in rows if row["language"] == "zh" and str(row["id"]) not in excluded], key=lambda row: int(row["id"]))
    tasks = random.Random(seed).sample(candidates, 5)
    manifest = {"created_at": now(), "source_url": split["source_url"],
                "source_revision": split["source_revision"], "source_file_sha256": split["source_file_sha256"],
                "sampling": "random.Random(seed).sample(sorted(chinese_non_development), 5)",
                "seed": seed, "candidate_count": len(candidates), "candidate_ids": [t["id"] for t in candidates],
                "development_ids": split["assigned"]["development"], "evaluation_ids": [t["id"] for t in tasks],
                "prompt_rewritten": False, "evaluation_scope": "通用研究能力；中文单轮原题",
                "quality_scoring": "尚未执行；后续讨论"}
    (output / "官方题库快照.jsonl").write_bytes(source)
    (output / "中文原题评测集.jsonl").write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8")
    save(manifest_path, manifest)
    return tasks, manifest


def freeze_code(output):
    repo = BACKEND.parent
    save(output / "代码版本.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                                  "working_tree_has_changes": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)),
                                  "runtime_source_archive": "运行源码快照.zip"})
    hashes = {}
    paths = list((BACKEND / "app").rglob("*.py")) + [p for p in (BACKEND / "app" / "prompts").rglob("*") if p.is_file()]
    paths += list((BACKEND / "eval" / "runtime").glob("*.py")) + [Path(__file__), BACKEND / "eval/scripts/run_deepresearch_bench.py"]
    with zipfile.ZipFile(output / "运行源码快照.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if "__pycache__" in path.parts:
                continue
            relative = str(path.relative_to(repo))
            content = path.read_bytes()
            archive.writestr(relative, content)
            hashes[relative] = hashlib.sha256(content).hexdigest()
    save(output / "源码与提示词哈希.json", hashes)


def export_database(meta, output):
    from core.database import SessionLocal
    from models.agent_history import AgentContext, AgentMessage
    from models.research import AgentEvent, ResearchRun
    from eval.models import AgentRun
    with SessionLocal() as db:
        context_ids = db.query(AgentMessage.context_id).filter(AgentMessage.run_id == meta["research_run_id"])
        contexts = db.query(AgentContext).filter(AgentContext.id.in_(context_ids)).all()
        messages = db.query(AgentMessage).join(AgentContext).filter(AgentMessage.run_id == meta["research_run_id"]).order_by(AgentContext.agent_name, AgentContext.context_key, AgentMessage.seq).all()
        events = db.query(AgentEvent).filter(AgentEvent.run_id == meta["research_run_id"]).order_by(AgentEvent.seq).all()
        generations = db.query(AgentRun).filter(AgentRun.session_id == meta["session_id"]).order_by(AgentRun.created_at).all()
        runs = db.query(ResearchRun).filter(ResearchRun.run_id == meta["research_run_id"]).all()
        groups = {"智能体上下文": contexts, "智能体原始历史": messages, "持久化事件账本": events,
                  "LLM调用埋点": generations, "研究运行状态": runs}
        for name, items in groups.items():
            save(output / f"{name}.json", [{column.name: getattr(row, column.name) for column in row.__table__.columns} for row in items])
        return {name: len(items) for name, items in groups.items()}


async def execute(args, tasks, manifest):
    # 环境覆盖仅作用于此次采集进程；新运行冻结该配置，保留摘要功能。
    for key, value in {"RESEARCH_ALLOWED_DOMAINS": "", "AGENT_SUMMARY_NLI_ENABLED": "0",
        "AGENT_SUMMARY_CHECKS_ENABLED": "0", "RESEARCH_GATE_CITATION_SUPPORT_ENABLED": "false",
        "RESEARCH_GATE_CITATION_REPAIR_ENABLED": "false", "EVAL_TRACE_ENABLED": "true",
        "EVAL_TRACE_PAYLOAD_CHARS": "10000000", "EVAL_TRACE_QUEUE_SIZE": "10000",
        "EVAL_TRACE_LOCAL_FILE": str(args.output / "完整过程埋点.jsonl")}.items():
        os.environ[key] = value
    from eval.scripts.run_deepresearch_bench import benchmark_service, run_one
    from eval.runtime import tracer, local_trace
    from eval.runtime.langfuse_tracing import flush_langfuse
    from config.llm_config import get_config
    from service.context_policy import selected_policy
    from service.deep_research_v2.graph import graph_config
    config = get_config()
    policy = selected_policy()
    assert not policy.semantic_verification and policy.nli is None
    config_path = args.output / "运行配置.json"
    if not config_path.exists():
        save(config_path, {"started_at": now(), "agents": dataclasses.asdict(config.agents),
        "research": dataclasses.asdict(config.research), "context_policy": policy.snapshot(),
        "max_iterations": config.research.max_iterations, "timeout_seconds": args.timeout, "concurrency": 1,
        "citation_verification": "关闭报告引用核验、摘要NLI与语义检查；保留原文引用和摘要",
        "allowed_domains": [], "local_trace_full_payload": True, "langfuse_enabled": os.getenv("LANGFUSE_ENABLED"),
            "sampling_manifest": "抽样清单.json"})
        freeze_code(args.output)
    records = []
    batch_stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S")
    async with benchmark_service(config.research.max_iterations) as service:
        for index, task in enumerate(tasks, 1):
            output = args.output / f"题目{task['id']}"
            if (output / "运行结果.json").exists():
                raise RuntimeError("此目录已有运行结果。首次失败记录应保留，新的尝试请使用新目录。")
            print(f"[{index}/{len(tasks)}] 开始 #{task['id']}：{task['prompt'][:65]}", flush=True)
            record = await run_one(service, task, "通用中文初测", batch_stamp, args.timeout, False, artifact_dir=output)
            meta = record["_meta"]
            # 先保存结果；后续副本导出失败不会抹除已完成的报告。
            records.append(record)
            with (args.output / "报告与运行结果.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            await asyncio.to_thread(tracer._queue.join)
            counts = await asyncio.to_thread(export_database, meta, output)
            checkpoints = []
            async for snapshot in service.checkpointer.alist(graph_config(meta["research_run_id"])):
                checkpoints.append({"config": snapshot.config, "checkpoint": snapshot.checkpoint,
                                    "metadata": snapshot.metadata, "parent_config": snapshot.parent_config,
                                    "pending_writes": snapshot.pending_writes})
            save(output / "完整检查点历史.json", checkpoints)
            counts["检查点"] = len(checkpoints)
            save(output / "归档统计.json", counts)
            print(f"[{index}/{len(tasks)}] #{task['id']} {'完成' if record['article'] else '失败'}，{meta['duration_s']}秒，{meta['article_chars']}字符；归档 {counts}", flush=True)
            save(args.output / "采集进度.json", {"updated_at": now(), "completed_attempts": len(records),
                "successes": sum(bool(r["article"]) for r in records), "task_ids": manifest["evaluation_ids"],
                "local_trace_write_errors": local_trace.error_count(), "database_trace_dropped": tracer._dropped})
    flush_langfuse()
    save(args.output / ("采集完成.json" if len(tasks) == 5 else f"题目{tasks[0]['id']}采集结束.json"), {"finished_at": now(), "attempts": len(records),
        "successes": sum(bool(r["article"]) for r in records), "local_trace_write_errors": local_trace.error_count(),
        "database_trace_dropped": tracer._dropped, "quality_scoring": "尚未执行"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--only-id", type=int, help="仅运行已固定评测集中的一题，用于独立进程超时保护")
    args = parser.parse_args()
    args.output = args.output.resolve()
    tasks, manifest = prepare(args.output, args.development_manifest, args.seed)
    if args.only_id is not None:
        tasks = [task for task in tasks if task["id"] == args.only_id]
        if not tasks:
            parser.error("指定题号不在已固定的评测集中")
    print(f"中文候选 {manifest['candidate_count']} 题，抽中 {manifest['evaluation_ids']}，开发集 {manifest['development_ids']}", flush=True)
    if not args.prepare_only:
        asyncio.run(execute(args, tasks, manifest))


if __name__ == "__main__":
    main()
