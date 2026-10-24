"""汇总主代理已落盘的十篇匿名审读评分；本脚本不生成或修改评分。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

DIMENSIONS = ("depth", "comprehensiveness", "readability", "requirements")
SUPPLEMENTARY = ("logical_coherence", "viewpoint_balance")


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def citation_summary(pairs, group):
    audits = [p["groups"][group]["citation_metrics"] or {} for p in pairs]
    hashes_match = [p["groups"][group]["citation_diagnostics"]["report_sha256_matches_actual_text"] for p in pairs]
    complete = [a for a, matches in zip(audits, hashes_match) if a.get("extracted_claims_checked") is True and matches is True]
    measurable = [a for a in complete if type(a.get("claims_count")) is int and a["claims_count"] > 0]
    total = sum(a["claims_count"] for a in measurable)
    keys = ("supported_count", "missing_citation_count", "prediction_as_fact_count", "review_count")
    assert all(type(a.get(key)) is int for a in measurable for key in keys), "完整核验仍缺计数，不能填零"
    counts = {key: sum(a[key] for a in measurable) for key in keys}
    statuses = Counter()
    for audit in measurable:
        statuses.update(audit.get("claim_status_counts") or {})
    return {"reports": len(audits), "complete_reports": len(complete), "nonempty_measurable_reports": len(measurable),
        "report_hash_match_counts": dict(Counter(str(value) for value in hashes_match)),
        "execution_status_counts": dict(Counter(a.get("execution_status", "unavailable") for a in audits)),
        "measured_claims_count": total, **counts, "claim_status_counts": dict(statuses),
        "micro_supported_rate": counts["supported_count"] / total if total else None,
        "micro_missing_citation_rate": counts["missing_citation_count"] / total if total else None,
        "micro_prediction_as_fact_rate": counts["prediction_as_fact_count"] / total if total else None,
        "scope": "只聚合完整核验且N>0的报告；不完整报告另列，不纳入比率分母"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="明确只分析已交付配对，保留所有未评分题及失败；不把条件均值称总体均值")
    args = parser.parse_args(); root = args.directory.resolve()
    frozen = read(root / "Codex匿名逐篇质量评估.json")
    grades = frozen["reports"]
    tasks = read(root / "固定五题.json")
    runtime = read(root / "运行与引用核验指标汇总.json")
    run_by_id = {row["id"]: row for row in runtime["pairs"]}
    scored_tasks = [task for task in tasks if all(g["completed"] for g in run_by_id[task["id"]]["groups"].values())] if args.allow_incomplete else tasks
    assert scored_tasks, "没有完整交付的配对可评估"
    expected = {(task["id"], label) for task in scored_tasks for label in ("报告甲", "报告乙")}
    observed = [(row["task_id"], row["blind_label"]) for row in grades]
    assert len(observed) == len(set(observed)) and set(observed) == expected, "须先完成所有纳入配对报告的审读"
    for row in grades:
        path = root / "匿名审读材料" / f"题目{row['task_id']}" / (row["blind_label"] + ".md")
        assert row["full_text_read"] is True
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["report_sha256"]
        for dimension in DIMENSIONS:
            assert type(row["scores"][dimension]) in (int, float) and 0 <= row["scores"][dimension] <= 10
            assert row["dimension_reasons"][dimension]
        for dimension in SUPPLEMENTARY:
            assert type(row["supplementary_scores"][dimension]) in (int, float)
            assert 0 <= row["supplementary_scores"][dimension] <= 10
        assert row["source_checks"] and row["requirement_coverage"]
    # 所有匿名评分通过完整性检查后，才打开组别映射。
    pairs = []
    for task in scored_tasks:
        mapping = read(root / f"题目{task['id']}" / "盲评标签映射.json")
        groups = {}
        for row in [r for r in grades if r["task_id"] == task["id"]]:
            identity = mapping[row["blind_label"]]
            assert identity["report_sha256"] == row["report_sha256"]
            group = identity["group"]
            groups[group] = {**row, "group": group, "overall": mean(row["scores"].values()),
                "report_chars": len((root / f"题目{task['id']}" / group / "研究报告.md").read_text()),
                "citation_metrics": run_by_id[task["id"]]["groups"][group]["citation_metrics"],
                "citation_diagnostics": run_by_id[task["id"]]["groups"][group]["citation_diagnostics"],
                "token_usage": run_by_id[task["id"]]["groups"][group]["production_logical_usage_without_fork_maintenance"]}
        off, on = groups["关闭审核"], groups["开启审核"]
        pairs.append({"id": task["id"], "query": task["prompt"], "groups": groups,
            "overall_gain": on["overall"] - off["overall"],
            "dimension_gains": {k: on["scores"][k] - off["scores"][k] for k in DIMENSIONS},
            "supplementary_gains": {k: on["supplementary_scores"][k] - off["supplementary_scores"][k] for k in SUPPLEMENTARY},
            "critic_incremental_usage": run_by_id[task["id"]]["groups"]["开启审核"]["incremental_usage"],
            "fork_maintenance_usage": run_by_id[task["id"]]["fork_maintenance_usage"]})
    averages = {group: {"overall": mean(p["groups"][group]["overall"] for p in pairs),
        "dimensions": {k: mean(p["groups"][group]["scores"][k] for p in pairs) for k in DIMENSIONS},
        "supplementary": {k: mean(p["groups"][group]["supplementary_scores"][k] for p in pairs) for k in SUPPLEMENTARY}}
        for group in ("关闭审核", "开启审核")}
    outcome = Counter("win" if p["overall_gain"] > 0 else "loss" if p["overall_gain"] < 0 else "tie" for p in pairs)
    result = {"evaluated_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "evaluator": "当前Codex主代理全文审读；单一AI评审，非人类金标", "pairs": pairs,
        "averages": averages, "mean_gain": mean(p["overall_gain"] for p in pairs),
        "win_tie_loss": {k: outcome[k] for k in ("win", "tie", "loss")},
        "citation_summary_by_group": {group: citation_summary(pairs, group) for group in ("关闭审核", "开启审核")},
        "actual_experiment_usage": runtime["actual_experiment_usage"],
        "planned_pairs": len(tasks), "scored_pairs": len(pairs),
        "unscored_tasks": [{"id":t["id"], "status":run_by_id[t["id"]]["status"], "failure":run_by_id[t["id"]].get("failure")} for t in tasks if t not in scored_tasks],
        "aggregation_scope": "已完整交付且完成主代理审读的配对条件均值，未对失败题填零或推算效果",
        "frozen_anonymous_grades_sha256": hashlib.sha256((root / "Codex匿名逐篇质量评估.json").read_bytes()).hexdigest(),
        "limits": ["五个历史固定题，一次生成和单次AI评分，未检验显著性或泛化", "开启组可使用更多token，并非等预算比较", "引用支持率不是现实事实准确率"]}
    save(root / "Codex质量增益汇总.json", result)
    print(json.dumps({k: result[k] for k in ("averages", "mean_gain", "win_tie_loss")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
