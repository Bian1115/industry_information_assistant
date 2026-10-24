"""为已固定的评测集提供进程级时限；保留首次尝试，补齐中断归档。"""
import argparse
import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

script = Path(__file__).with_name("采集通用研究评测集.py")
spec = importlib.util.spec_from_file_location("collector", script)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


async def recover(output, reason):
    identity = json.loads((output / "运行身份.json").read_text())
    task = identity["task"]
    meta = {k: identity[k] for k in ("session_id", "research_run_id")}
    with (output / "原始SSE轨迹.jsonl").open() as source:
        received = [json.loads(line) for line in source if line.endswith("\n")]
    meta.update(topic=task["topic"], language=task["language"],
                duration_s=round(time.time() - received[0]["received_at"], 1) if received else None,
                error=reason, article_chars=0, references_count=0,
                quality_score=None, quality_verdict="not_reviewed", finished_at=collector.now(),
                externally_stopped=True)
    record = {"id": task["id"], "prompt": task["prompt"], "article": "", "_meta": meta}
    collector.save(output / "运行结果.json", record)
    counts = collector.export_database(meta, output)
    from service.deep_research_v2.checkpointer import checkpointer_lifespan
    from service.deep_research_v2.graph import graph_config
    from types import SimpleNamespace
    async with checkpointer_lifespan(SimpleNamespace(state=SimpleNamespace())) as saver:
        checkpoints = []
        async for item in saver.alist(graph_config(meta["research_run_id"])):
            checkpoints.append({"config": item.config, "checkpoint": item.checkpoint,
                                "metadata": item.metadata, "parent_config": item.parent_config,
                                "pending_writes": item.pending_writes})
        collector.save(output / "完整检查点历史.json", checkpoints)
    counts["检查点"] = len(checkpoints)
    collector.save(output / "归档统计.json", counts)
    print(f"#{task['id']} 中断归档完成：{counts}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    args.output = args.output.resolve()
    tasks, manifest = collector.prepare(args.output, args.development_manifest, 20260930)
    collector.save(args.output / "进程保护配置.json", {"configured_at": collector.now(),
        "task_timeout_seconds": args.timeout, "shutdown_grace_seconds": 10,
        "reason": "首次运行内部超时未及时收尾；后续用独立进程约束相同单题时限",
        "research_models_and_budgets_changed": False})
    for task in tasks:
        folder = args.output / f"题目{task['id']}"
        if (folder / "运行结果.json").exists():
            continue
        if (folder / "运行身份.json").exists():
            asyncio.run(recover(folder, "ExternalTimeout: 首次采集超过1800秒单题时限，外部停止后保全已产生轨迹；没有最终报告"))
            continue
        command = [sys.executable, "-u", str(script), "--output", str(args.output),
                   "--development-manifest", str(args.development_manifest),
                   "--only-id", str(task["id"]), "--timeout", str(args.timeout)]
        print(f"开始独立进程 #{task['id']}，时限{args.timeout}s", flush=True)
        with (args.output / f"题目{task['id']}运行日志.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.time() + args.timeout
                while process.poll() is None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, args.timeout)
                    try:
                        process.wait(timeout=min(1, remaining))
                    except subprocess.TimeoutExpired:
                        pass
                code = process.returncode
                reason = f"CollectorExit: 采集进程返回{code}，未完成结果归档"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                reason = f"ExternalTimeout: 超过{args.timeout}秒单题时限，由独立进程保护停止；没有最终报告"
        if not (folder / "运行结果.json").exists():
            if not (folder / "运行身份.json").exists():
                raise RuntimeError(f"题目{task['id']}尚未开始实际研究，查看该题运行日志")
            asyncio.run(recover(folder, reason))
        print(f"#{task['id']} 已保存首次尝试", flush=True)
    records = [json.loads((args.output / f"题目{task['id']}" / "运行结果.json").read_text()) for task in tasks]
    (args.output / "报告与运行结果.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    collector.save(args.output / "采集完成.json", {"finished_at": collector.now(), "attempts": len(records),
        "successes": sum(bool(record["article"]) for record in records), "task_ids": manifest["evaluation_ids"],
        "quality_scoring": "尚未执行", "external_process_timeout_used": True})
    print("5题首次尝试全部归档", flush=True)


if __name__ == "__main__":
    main()
