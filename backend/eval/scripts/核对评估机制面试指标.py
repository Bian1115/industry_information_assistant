"""只读取既有评分和标签，独立复算面试指标；不调用模型、不覆盖历史运行。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "backend/eval/data"
DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新的中文结果目录")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = {
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return json.loads(raw)

    def weighted(task, result):
        scores = {r["criterion_id"]: Decimal(str(r["score"])) for r in result["scores"]}
        expected = {c["criterion_id"] for c in task["criteria"]}
        assert len(result["scores"]) == len(scores) and set(scores) == expected
        assert all(s.is_finite() and 0 <= s <= 10 for s in scores.values())
        dimensions = {}
        for dim in DIMS:
            criteria = [c for c in task["criteria"] if c["dimension"] == dim]
            assert criteria and all(c["weight"] > 0 for c in criteria)
            denominator = sum(Decimal(str(c["weight"])) for c in criteria)
            dimensions[dim] = sum(scores[c["criterion_id"]] * Decimal(str(c["weight"]))
                                  for c in criteria) / denominator
        assert sum(Decimal(str(task["dimension_weight"][d])) for d in DIMS) == 1
        dimensions["total"] = sum(dimensions[d] * Decimal(str(task["dimension_weight"][d])) for d in DIMS)
        return {k: float(v) for k, v in dimensions.items()}

    def agreement(predicted, reference):
        n = len(predicted)
        errors = [a-b for a, b in zip(predicted, reference)]
        am, bm = statistics.mean(predicted), statistics.mean(reference)
        covariance = sum((a-am)*(b-bm) for a, b in zip(predicted, reference))/n
        denominator = statistics.pvariance(predicted)+statistics.pvariance(reference)+(am-bm)**2
        return {"n": n, "mae": statistics.mean(map(abs, errors)),
                "rmse": (statistics.mean(e*e for e in errors))**.5,
                "bias": statistics.mean(errors), "ccc": 2*covariance/denominator if denominator else None,
                "within_0_5_count": sum(abs(e) <= .5+1e-10 for e in errors),
                "within_1_count": sum(abs(e) <= 1+1e-10 for e in errors)}

    bench = DATA / "DeepResearchBench参考报告30题双裁判一致性-20261007"
    tasks = read(bench / "冻结输入.json")
    cfg = read(bench / "评估配置.json")
    historical = read(bench / "一致性汇总.json")
    read(bench / "双裁判综合状态.json")
    assert hashlib.sha256((bench / "冻结输入.json").read_bytes()).hexdigest() == cfg["input_sha256"]
    old_rows = {r["task_id"]: r for r in historical["tasks"]}
    rows, counts, quote_counts = [], Counter(), Counter()
    for task in tasks:
        own_path = bench / "助手独立评分" / f"题{task['id']}.json"
        own = read(own_path)
        assert hashlib.sha256(own_path.read_bytes()).hexdigest() == cfg["assistant_file_sha256"][own_path.name]
        assert hashlib.sha256(task["article"].encode()).hexdigest() == task["report_sha256"] == own["report_sha256"]
        row = {"task_id": task["id"], "assistant": weighted(task, own), "repeats": {}, "scores": {}}
        for model in ["qwen", "claude"]:
            repeats = []
            for repeat in [1, 2]:
                path = bench / "裁判原始记录" / f"题{task['id']}-细则-{model}-{repeat}.json"
                if not path.exists():
                    continue
                raw = read(path)
                if raw.get("result") is None:
                    continue
                assert hashlib.sha256(raw["prompt"].encode()).hexdigest() == raw["prompt_sha256"]
                assert task["article"] in raw["prompt"]
                repeats.append(weighted(task, raw["result"]))
                counts[model] += 1
                for criterion in raw["result"]["scores"]:
                    quote = criterion["evidence_quote"]
                    quote_counts["empty" if not quote else "exact_substring" if quote in task["article"] else "not_exact_substring"] += 1
            row["repeats"][model] = repeats
            if len(repeats) == 2:
                row["scores"][model] = {d: statistics.mean(r[d] for r in repeats) for d in DIMS+["total"]}
        if len(row["scores"]) == 2:
            row["scores"]["ensemble"] = {d: statistics.mean(row["scores"][m][d] for m in ["qwen", "claude"]) for d in DIMS+["total"]}
        for model, values in row["scores"].items():
            assert all(abs(value-old_rows[task["id"]]["scores"][model][dim]) < 1e-10 for dim, value in values.items())
        rows.append(row)
    complete = [r for r in rows if "ensemble" in r["scores"]]
    reference = [r["assistant"]["total"] for r in complete]
    comparisons = {m: agreement([r["scores"][m]["total"] for r in complete], reference)
                   for m in ["qwen", "claude", "ensemble"]}
    dimensions = {d: agreement([r["scores"]["ensemble"][d] for r in complete], [r["assistant"][d] for r in complete]) for d in DIMS}
    repeat_stability = {}
    for m in ["qwen", "claude"]:
        eligible = [r for r in rows if len(r["repeats"][m]) == 2]
        repeat_stability[m] = agreement([r["repeats"][m][0]["total"] for r in eligible], [r["repeats"][m][1]["total"] for r in eligible])
    inter_judge = agreement([r["scores"]["qwen"]["total"] for r in complete], [r["scores"]["claude"]["total"] for r in complete])

    citation = DATA / "通用研究五题引用事实独立复核与模块一致性-20261007"
    claims = read(citation / "逐条独立复核及模块对齐.json")
    citation_old = read(citation / "独立复核与模块一致性指标.json")

    def citation_metrics(items):
        cited = [r for r in items if r["references"]]
        eligible = [r for r in cited if r["joint_support"] != "unknown"]
        matrix = Counter()
        for r in eligible:
            gold, predicted = r["joint_support"] == "full", r["module_status"] == "supported"
            matrix["TP" if gold and predicted else "FN" if gold else "FP" if predicted else "TN"] += 1
        tp, tn, fp, fn = [matrix[k] for k in ["TP", "TN", "FP", "FN"]]
        return {"all": len(items), "cited": len(cited), "unknown_cited": len(cited)-len(eligible),
                "eligible": len(eligible), "labels": dict(Counter(r["joint_support"] for r in items)),
                "confusion_full_support_positive": dict(matrix), "precision": tp/(tp+fp),
                "recall": tp/(tp+fn), "f1": 2*tp/(2*tp+fp+fn), "agreement": (tp+tn)/len(eligible),
                "problem_precision": tn/(tn+fn), "problem_recall": tn/(tn+fp)}

    citation_total = citation_metrics(claims)
    assert citation_total["labels"] == citation_old["total"]["labels"]
    by_topic = {str(t): citation_metrics([r for r in claims if r["topic"] == t]) for t in sorted({r["topic"] for r in claims})}
    examples = [r for r in claims if r["id"] in {"31-body-131", "49-body-009", "49-body-012", "46-body-028", "31-body-115"}]

    critic_dir = DATA / "Critic职责分离通用研究五题消融-20261017-231745"
    critic = read(critic_dir / "Codex质量增益汇总.json")
    usage = read(critic_dir / "运行与引用核验指标汇总.json")
    critic_pairs = []
    for pair in critic["pairs"]:
        scores = {g: statistics.mean(v["scores"].values()) for g, v in pair["groups"].items()}
        critic_pairs.append({"id": pair["id"], "scores": scores, "delta": scores["开启审核"]-scores["关闭审核"]})
    web = read(DATA / "网页读取链路公开基准评估-20261018-原生Markdown复评/核心指标与分层汇总.json")
    source_files = ["backend/eval/scripts/run_deepresearch_bench.py", "backend/eval/runtime/tracer.py", "backend/eval/runtime/local_trace.py",
        "backend/eval/scripts/评估五篇正文与Scout.py", "backend/eval/scripts/DeepResearchBench参考报告双裁判一致性.py",
        "backend/eval/scripts/evaluate_race_fact_claude.py", "backend/eval/verification/scoring.py",
        "backend/app/service/deep_research_v2/agents/critic.py", "backend/app/service/deep_research_v2/citation_verification.py",
        "backend/app/service/deep_research_v2/report_claim_extraction.py", "backend/app/service/deep_research_v2/report_citation_inputs.py",
        "backend/app/service/nli_verifier.py"]
    for name in source_files+[str(Path(__file__).resolve().relative_to(ROOT))]:
        raw = (ROOT/name).read_bytes()
        sources[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    report = {"date": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "scope": "已有评分及标签离线复算；未重新生成报告、调用裁判、重新标注或核查现实真值",
        "bench": {"tasks": len(tasks), "criteria": sum(len(t["criteria"]) for t in tasks), "valid_calls": dict(counts),
            "complete_dual_tasks": len(complete), "missing_complete_ids": [r["task_id"] for r in rows if "ensemble" not in r["scores"]],
            "same_26_task_comparison": comparisons, "dimensions": dimensions, "repeat_stability": repeat_stability,
            "inter_judge": inter_judge, "literal_quote_audit": dict(quote_counts), "tasks_recomputed": rows,
            "interpretation": "同一26题MAE对照为描述性复算；独立GPT参照不是人类真值，不证明跨任务泛化。"},
        "citation": {"total": citation_total, "per_topic": by_topic, "failure_examples": examples,
            "historical_locator": citation_old["input_quality"]["reference_locator_valid"],
            "unknown_causes": citation_old["unknown_causes"], "historical_code_date": "2026-10-06回放、2026-10-07独立复核；非当前版本重跑"},
        "critic": {"pairs_recomputed": critic_pairs, "mean_delta": statistics.mean(r["delta"] for r in critic_pairs),
            "historical_planned_pairs": critic["planned_pairs"], "historical_scored_pairs": critic["scored_pairs"],
            "historical_limits": critic["limits"], "usage_summary_keys": list(usage)},
        "web": {"total_pages": web["total_pages"], "datasets": web["datasets"], "scope": "引用历史冻结HTML评估汇总；本次未重新抽取或逐页重算"},
        "sources": sources}
    (args.output / "指标复算与来源指纹.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"bench": comparisons, "citation": citation_total, "critic": critic_pairs,
                      "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
