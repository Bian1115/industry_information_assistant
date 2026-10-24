"""汇总冻结的逐页确定性指标，输出分层统计和待人工核对的失败候选。"""
from collections import Counter, defaultdict
from pathlib import Path
import importlib.util
import json
import statistics

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "backend/eval/data/网页读取链路公开基准评估-20261018-首轮"
MODES = ["unlimited_pre_gate", "production_pre_gate", "production_cleaned", "accepted_output"]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    a, b = int(x), min(int(x) + 1, len(values) - 1)
    return values[a] + (values[b] - values[a]) * (x - a)


def table_aggregate(tables):
    result = {"pages": len(tables), "reference_tables": sum(t["reference_tables"] for t in tables),
              "predicted_tables_on_reference_pages": sum(t["predicted_tables"] for t in tables),
              "matched_tables": sum(t["matched_tables"] for t in tables),
              "pages_all_tables_missing": sum(t["predicted_tables"] == 0 for t in tables),
              "table_recall_macro": mean(t["table_recall"] for t in tables),
              "teds_like": mean(t.get("teds_like") for t in tables),
              "structure_teds_like": mean(t.get("structure_teds_like") for t in tables),
              "teds_eligible_pages": sum(t.get("teds_like") is not None for t in tables),
              "teds_skipped_pages": sum(bool(t.get("teds_skipped")) for t in tables),
              "teds_skip_reasons": dict(Counter(s["reason"] for t in tables for s in t.get("teds_skipped", [])))}
    for category in ["cells", "table_text", "table_text_in_whole_output", "horizontal_relations", "vertical_relations"]:
        count_key = {"cells": "cells_reference_n", "horizontal_relations": "horizontal_reference_n", "vertical_relations": "vertical_reference_n"}.get(category)
        valid = [t for t in tables if not count_key or t.get(count_key, 0) > 0]
        result[category] = {key: mean(t[category][key] for t in valid) for key in ["precision", "recall", "f1"]}
        result[category]["applicable_pages"] = len(valid)
    return result


def aggregate(rows):
    if not rows:
        return None
    scored_rows = [r for r in rows if r["reference_chars"] > 0]
    result = {"n": len(rows), "n_scored": len(scored_rows), "excluded_empty_reference": [r["id"] for r in rows if r["reference_chars"] == 0],
              "accepted": sum(r["production_result"]["fetch_status"] == "success" for r in rows),
              "failure_categories": dict(Counter(r["production_result"].get("failure_category") or "success" for r in rows)),
              "quality_status": dict(Counter(r["production_result"]["body_quality"].get("status", "unknown") for r in rows)),
              "quality_reasons": dict(Counter(x for r in rows for x in r["production_result"]["body_quality"].get("reasons", []))),
              "empty_extractions": sum(not r["chars"]["unlimited_pre_gate"] for r in rows),
              "truncated": sum(r["truncated"] for r in rows),
              "browser_requested_unavailable": sum(r["browser_fallback_requested_but_not_executed"] > 0 for r in rows),
              "extraction_seconds_p50": quantile([r["extract_and_gate_seconds"] for r in rows], .5),
              "extraction_seconds_p95": quantile([r["extract_and_gate_seconds"] for r in rows], .95), "modes": {}}
    for mode in MODES:
        scores = [r["scores"][mode] for r in scored_rows]
        text = {key: mean(s["text"].get(key) for s in scores) for key in ["precision", "recall", "f1", "edit_similarity", "reference_external_token_ratio"]}
        text["reference_external_ratio_nonempty_pages"] = sum(s["text"]["reference_external_token_ratio"] is not None for s in scores)
        text["ordered_5gram_recall"] = mean((s["text"]["ordered_5gram"] or {}).get("recall") for s in scores)
        entry = {"text": text}
        wcxb = [s for s in scores if "official_wcxb_words" in s]
        if wcxb:
            entry["official_wcxb_words"] = {key: mean(s["official_wcxb_words"][key] for s in wcxb) for key in ["precision", "recall", "f1"]}
            entry["snippets"] = {}
            for key in ["with", "without"]:
                eligible = [s["snippets"][key] for s in wcxb if s["snippets"][key]["n"]]
                total, found = sum(s["n"] for s in eligible), sum(s["found"] for s in eligible)
                entry["snippets"][key] = {"snippets": total, "found": found, "micro_rate": found / total if total else None,
                                           "page_macro_rate": mean(s["rate"] for s in eligible), "eligible_pages": len(eligible),
                                           "pages_with_hit": sum(s["found"] > 0 for s in eligible)}
        tables = [s["table"] for s in scores if s.get("table", {}).get("eligible")]
        if tables:
            entry["tables"] = table_aggregate(tables)
            grids = [t["grid_tables"] for t in tables if t.get("grid_tables", {}).get("eligible")]
            if grids:
                entry["grid_tables"] = table_aggregate(grids)
        entry["table_metric_errors"] = [r["id"] for r in rows if r["scores"][mode].get("table", {}).get("error")]
        result["modes"][mode] = entry
    return result


def group(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[str(key(row))].append(row)
    return {k: aggregate(v) for k, v in sorted(out.items())}


def main():
    rows = json.loads((RUN / "逐页指标汇总.json").read_text())
    assert len(rows) == 745 and len({r["id"] for r in rows}) == 745
    wm = [r for r in rows if r["dataset"] == "WebMainBench545"]
    wc = [r for r in rows if r["dataset"] == "WCXB分层200"]
    overall = {"date": "2026-10-18", "paid_api_calls": 0, "total_pages": len(rows),
               "datasets": {"WebMainBench545": aggregate(wm), "WCXB分层200": aggregate(wc)},
               "WebMainBench语言": group(wm, lambda r: r["meta"].get("language") or "unknown"),
               "WebMainBench难度": group(wm, lambda r: r["meta"].get("level") or "unknown"),
               "WCXB页面类型": group(wc, lambda r: r["meta"]["page_type"]),
               "WebMainBench标注与参考差异": {"metadata_table_pages": sum(bool(r["meta"].get("table")) for r in wm),
                                                       "reference_structured_table_pages": sum(r["reference_tables_detected"] > 0 for r in wm),
                                                       "null_style_pages": sum(r["meta"].get("style") is None for r in wm)},
               "snippet_reference_integrity": {"with_snippets_absent_from_reference": sum(len(r["reference_snippet_integrity"]["with"]["misses"]) for r in wc),
                                                 "without_snippets_present_in_reference": sum(r["reference_snippet_integrity"]["without"]["found"] for r in wc)}}
    allocation = json.loads((RUN / "分层抽样配置.json").read_text())["allocation"]
    total_pop = sum(v["population"] for v in allocation.values())
    weighted = {}
    for mode in MODES:
        weighted[mode] = {key: sum(allocation[t]["population"] / total_pop * a["modes"][mode]["text"][key]
                                   for t, a in overall["WCXB页面类型"].items()) for key in ["precision", "recall", "f1"]}
        weighted[mode]["official_wcxb_f1"] = sum(allocation[t]["population"] / total_pop * a["modes"][mode]["official_wcxb_words"]["f1"]
                                                  for t, a in overall["WCXB页面类型"].items())
    overall["WCXB开发集类型权重校正"] = weighted
    dump(RUN / "核心指标与分层汇总.json", overall)

    candidates = []
    groups = {
        "短正文被拒": [r for r in rows if r["production_result"].get("failure_category") == "content_too_short" and r["scores"]["production_pre_gate"]["text"]["f1"] >= .85],
        "完全空提取": [r for r in rows if r["chars"]["unlimited_pre_gate"] == 0],
        "严重遗漏": sorted([r for r in rows if r["chars"]["production_pre_gate"]], key=lambda r: r["scores"]["production_pre_gate"]["text"]["recall"]),
        "已标注噪声残留": sorted([r for r in wc if r["scores"]["production_pre_gate"]["snippets"]["without"]["found"]],
                              key=lambda r: -r["scores"]["production_pre_gate"]["snippets"]["without"]["found"]),
        "表格消失但文字仍在": sorted([r for r in wm if r["scores"]["production_pre_gate"]["table"].get("grid_tables", {}).get("eligible") and r["scores"]["production_pre_gate"]["table"].get("predicted_tables") == 0],
                                    key=lambda r: -r["scores"]["production_pre_gate"]["table"]["table_text_in_whole_output"]["recall"]),
        "表格行列损坏": sorted([r for r in wm if r["scores"]["production_pre_gate"]["table"].get("grid_tables", {}).get("eligible") and r["scores"]["production_pre_gate"]["table"].get("predicted_tables", 0) > 0],
                                key=lambda r: r["scores"]["production_pre_gate"]["table"]["grid_tables"]["horizontal_relations"]["f1"]),
        "预算导致遗漏": sorted([r for r in rows if r["truncated"]], key=lambda r: -(r["scores"]["unlimited_pre_gate"]["text"]["recall"] - r["scores"]["production_pre_gate"]["text"]["recall"]))}
    seen = set()
    for reason, samples in groups.items():
        selected = 0
        for r in samples:
            if r["id"] in seen:
                continue
            seen.add(r["id"]); selected += 1
            folder = RUN / "逐页结果" / r["id"]
            candidates.append({"id": r["id"], "automatic_category": reason, "url": r["url"], "meta": r["meta"],
                               "production_status": r["production_result"]["fetch_status"], "failure_category": r["production_result"].get("failure_category"),
                               "chars": r["chars"], "reference_chars": r["reference_chars"], "scores": r["scores"]["production_pre_gate"],
                               "full_recall": r["scores"]["unlimited_pre_gate"]["text"]["recall"],
                               "reference_preview": (folder / "参考正文.txt").read_text()[:900],
                               "prediction_preview": (folder / "生产门禁前正文.txt").read_text()[:900],
                               "manual_review": "pending"})
            if selected == 2:
                break
    dump(RUN / "失败样例候选.json", candidates)
    print(json.dumps({"datasets": {k: {"n": v["n"], "n_scored": v["n_scored"], "accepted": v["accepted"], "pre_gate": v["modes"]["production_pre_gate"]["text"],
                                        "accepted_output": v["modes"]["accepted_output"]["text"], "table": v["modes"]["production_pre_gate"].get("tables")}
                                   for k, v in overall["datasets"].items()}, "failure_candidates": len(candidates)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    main()
