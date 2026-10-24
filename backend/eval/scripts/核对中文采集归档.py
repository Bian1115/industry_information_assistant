"""研究进程结束后核对完整轨迹并生成可追溯的指标和压缩档案。"""
import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-user-stopped", action="store_true", help="仅归档有明确用户停止记录的中止批次")
    args = parser.parse_args()
    root = args.output.resolve()
    stopped = args.allow_user_stopped and (root / "用户停止记录.json").exists() and (root / "采集中止.json").exists()
    assert read(root / "采集进程结束.json")["exit_code"] == 0 or stopped, "采集进程未正常结束，且无明确用户停止记录"
    local = [json.loads(line) for line in (root / "完整过程埋点.jsonl").open()]
    tasks = []
    problems = []
    config = read(root / "运行配置.json")
    for task_id in [35, 31, 46, 16, 49]:
        folder = root / f"题目{task_id}"
        if not (folder / "运行身份.json").exists():
            assert stopped and task_id in read(root / "采集中止.json")["not_started_task_ids"], f"题目{task_id}缺少身份，不能确认未开始"
            tasks.append({"id": task_id, "outcome": "not_started", "report_generated": False,
                          "report_chars": 0, "tokens": 0, "recorded_llm_calls": 0,
                          "sse_events": 0, "histories": 0, "checkpoints": 0, "summary_related": {}})
            continue
        result = read(folder / "运行结果.json")
        stats = read(folder / "归档统计.json")
        meta = result["_meta"]
        histories = read(folder / "智能体原始历史.json")
        calls = read(folder / "LLM调用埋点.json")
        events = [json.loads(line) for line in (folder / "研究事件.jsonl").open()]
        ledger = read(folder / "持久化事件账本.json")
        checkpoints = read(folder / "完整检查点历史.json")
        contexts = read(folder / "智能体上下文.json")
        checkpoint_state = checkpoints[0]["checkpoint"]["channel_values"].get("research", {}) if checkpoints else {}
        stages = {row["call_id"]: row["payload"]["stage"] for row in histories if row["kind"] == "summary_stage"}
        summary = collections.Counter()
        for row in histories:
            if row["kind"] == "llm_usage" and row["call_id"] in stages:
                summary["successful_calls"] += 1
                summary["tokens"] += row["payload"]["usage"]["total_tokens"]
        generations = [row for row in local if row.get("session_id") == meta["session_id"] and row.get("event") == "start" and row.get("as_type") == "generation"]
        task = {
            "id": task_id, "research_run_id": meta["research_run_id"], "session_id": meta["session_id"],
            "report_generated": bool(result["article"]), "report_chars": len(result["article"]),
            "stage_report_chars": len((folder / "阶段报告.md").read_text()) if (folder / "阶段报告.md").exists() else 0,
            "internal_quality_verdict": checkpoint_state.get("quality_verdict"),
            "internal_quality_score_status": checkpoint_state.get("quality_score_status"),
            "duration_s": meta.get("duration_s"), "error": meta.get("error"),
            "outcome": "user_stopped" if meta.get("stop_reason") == "user_requested" else "report_generated" if result["article"] else "system_failed",
            "tokens": sum(row.get("total_tokens") or 0 for row in calls), "recorded_llm_calls": len(calls),
            "local_generation_records": len(generations), "sse_events": len(events), "database_events": len(ledger),
            "histories": len(histories), "checkpoints": len(checkpoints), "archive_counts": stats,
            "summary_related": dict(summary), "summary_stage_counts": dict(collections.Counter(stages.values())),
            "summary_errors": [row["payload"] for row in histories if row["kind"] in {"summary_error", "input_history_error"}],
            "saved_summaries": [{"agent": row["agent_name"], "version": row["summary_version"], "covered_seq": row["covered_seq"],
                                "validation": (row.get("summary_metadata") or {}).get("validation", {})} for row in contexts],
            "model_counts": dict(collections.Counter(row["model"] for row in calls)),
            "node_history_counts": dict(collections.Counter(row["node"] for row in histories)),
            "report_citation_verification": checkpoint_state.get("citation_verification"),
            "checkpoint_context_policy_hash": checkpoint_state.get("context_policy", {}).get("config_hash"),
        }
        if len(events) != len(ledger):
            problems.append(f"题目{task_id}: SSE与数据库事件数量不同")
        if len(generations) != len(calls):
            problems.append(f"题目{task_id}: 本地generation与数据库调用数量不同")
        if result["prompt"] != read(folder / "运行身份.json")["task"]["prompt"]:
            problems.append(f"题目{task_id}: 原题内容不一致")
        if task["checkpoint_context_policy_hash"] != config["context_policy"]["config_hash"]:
            problems.append(f"题目{task_id}: 检查点配置与冻结配置不同")
        if (task["report_citation_verification"] or {}).get("status") != "not_executed":
            problems.append(f"题目{task_id}: 引用核验状态与预期不同")
        tasks.append(task)
    summary_tokens = sum(task["summary_related"].get("tokens", 0) for task in tasks)
    total_tokens = sum(task["tokens"] for task in tasks)
    metrics = {"tasks": tasks, "totals": {"reports": sum(task["report_generated"] for task in tasks),
               "stage_reports": sum(bool(task.get("stage_report_chars")) for task in tasks),
               "system_failed": sum(task["outcome"] == "system_failed" for task in tasks),
               "user_stopped": sum(task["outcome"] == "user_stopped" for task in tasks),
               "not_started": sum(task["outcome"] == "not_started" for task in tasks),
               "tokens": total_tokens, "recorded_llm_calls": sum(task["recorded_llm_calls"] for task in tasks),
               "sse_events": sum(task["sse_events"] for task in tasks), "histories": sum(task["histories"] for task in tasks),
               "checkpoints": sum(task["checkpoints"] for task in tasks), "summary_tokens": summary_tokens,
               "summary_token_percent": round(summary_tokens / total_tokens * 100, 2) if total_tokens else None},
               "archive_checks": {"problems": problems, "matched": not problems},
               "quality_scoring": "外部模型评分与人工核验尚未执行",
               "configuration": {"citation_verification": config["citation_verification"], "context_policy": config["context_policy"]}}
    save(root / "轨迹归档核对.json", metrics)
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != "文件校验清单.json":
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(root / "文件校验清单.json", hashes)
    archive = root.parent / f"{root.name}完整归档.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                output.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
    print(json.dumps({"totals": metrics["totals"], "archive_checks": metrics["archive_checks"], "archive": str(archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
