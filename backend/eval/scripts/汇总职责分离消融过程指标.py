"""汇总真实运行的token与引用指标；不调用模型，不生成质量分数。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
from eval.scripts.记录模型调用用量 import summarize


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(); root = args.directory.resolve()
    tasks = read(root / "固定五题.json")
    actual, pairs = [], []
    for task in tasks:
        directory = root / f"题目{task['id']}"
        rows = []
        malformed = 0
        path = directory / "全部模型API用量.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
        actual.extend(rows)
        run = read(directory / "配对运行结果.json", {})
        off_rows = [row for row in rows if row["group"] == "关闭审核"]
        maintenance_verified = all(row["kind"] == "embedding" and row.get("stage_agent") is None for row in off_rows)
        item = {"id": task["id"], "query": task["prompt"], "status": run.get("status", "pending"),
            "failure": {"error_type": run.get("error_type"), "error": run.get("error")} if run.get("status") == "failed" else None,
            "actual_usage": summarize(rows), "partial_jsonl_lines": malformed, "groups": {}}
        for group in ("关闭审核", "开启审核"):
            folder = directory / group
            selected = [row for row in rows if row["group"] in {"公共首稿", group}]
            audit = read(folder / "引用核验完整结果.json", {})
            report = folder / "研究报告.md"
            matches = hashlib.sha256(report.read_bytes()).hexdigest() == audit["report_sha256"] if report.exists() and audit.get("report_sha256") else None
            item["groups"][group] = {"logical_full_usage": summarize(selected),
                "incremental_usage": summarize([row for row in rows if row["group"] == group]),
                "citation_metrics": read(folder / "引用核验指标.json"),
                "citation_diagnostics": {"error_type": audit.get("error_type"), "reason": audit.get("reason"),
                    "report_sha256_matches_actual_text": matches},
                "completed": bool((run.get("groups") or {}).get(group, {}).get("success")),
                "generation_result": (run.get("groups") or {}).get(group, {})}
            # 关闭分支的新请求只可能是隔离克隆后的索引维护；与真实无Critic业务链分开。
            core = [row for row in rows if row["group"] == "公共首稿" or
                    ((group == "开启审核" or not maintenance_verified) and row["group"] == group)]
            item["groups"][group]["production_logical_usage_without_fork_maintenance"] = summarize(core)
        item["fork_maintenance_usage"] = summarize(off_rows)
        item["fork_maintenance_verified"] = maintenance_verified
        item["off_branch_language_model_calls"] = sum(row["kind"] == "language_model" for row in off_rows)
        public = directory / "公共首稿/研究报告.md"
        off = directory / "关闭审核/研究报告.md"
        item["identical_public_and_off_report"] = public.read_bytes() == off.read_bytes() if public.exists() and off.exists() else None
        item["fork_maintenance_note"] = "关闭分支克隆后索引维护计入实际实验总量，单列而不归于无Critic的业务生成成本"
        pairs.append(item)
    data = {"updated_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(), "pairs": pairs,
        "actual_experiment_usage": summarize(actual),
        "counts": dict(Counter(row["status"] for row in pairs)),
        "reports_completed": sum(group["completed"] for row in pairs for group in row["groups"].values()),
        "quality_evaluation": "尚未并入；质量分数须由Codex主代理全文审读后填写"}
    save(root / "运行与引用核验指标汇总.json", data)
    print(json.dumps({"counts": data["counts"], "reports_completed": data["reports_completed"],
        "model_calls": len(actual), "known_total_tokens": data["actual_experiment_usage"]["total_tokens_known"],
        "missing_total_usage_calls": data["actual_experiment_usage"]["total_tokens_missing_calls"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
