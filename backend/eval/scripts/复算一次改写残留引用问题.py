"""读取2026-10-06冻结结果，复算面试指标；不调用模型或修改历史标签。"""

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "backend/eval/data/正文引用核验消融-20261006-170312"
OUTPUT = ROOT / "backend/eval/data/一次改写残留引用问题复算-20261030"


def main():
    inputs = {}

    def read(path):
        content = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    aggregate = read(SOURCE / "加权指标汇总.json")
    summaries = {row["id"]: row for row in read(SOURCE / "指标汇总.json")}
    config = read(SOURCE / "评估配置.json")
    fact_counts, relation_counts, unbound_counts, runtime_counts = (
        Counter(), Counter(), Counter(), Counter()
    )
    per_topic, non_full_facts, non_full_relations, rejected_batches = [], [], [], []
    for topic in [16, 31, 35, 46, 49]:
        folder = SOURCE / f"第{topic}题" / "一次改写组"
        rows = read(folder / "最终事实与判定.json")
        state = read(folder / "处理后状态.json")
        facts, relations, unbound = Counter(), Counter(), Counter()
        for row in rows:
            judgment = row["judgment"]
            label = judgment["joint_support"]
            facts[label] += 1
            refs = []
            if "refs" in row:
                for ref in row["refs"]:
                    original = next(
                        (value for value in judgment.get("ref_support", [])
                         if value["id"] == ref["id"]), {}
                    )
                    override = bool(judgment.get("reviewer_override")) and len(row["refs"]) == 1
                    refs.append({
                        "source_id": ref["id"],
                        "label": label if override else original.get("label", "unknown"),
                        "reason": judgment.get("reason") if override else original.get("reason"),
                        "single_source_override": override,
                    })
            else:
                refs = [{"source_id": ref["id"], "label": ref["label"],
                         "reason": ref.get("reason"), "single_source_override": False}
                        for ref in judgment.get("ref_support", [])]
            relations.update(ref["label"] for ref in refs)
            if not refs:
                unbound[label] += 1
            record = {
                "topic": topic, "id": row["id"], "claim": row["claim"],
                "label": label, "reason": judgment.get("reason"),
                "has_bound_citation": bool(refs),
                "origin": "改写单元重新抽取" if "-改写-" in row["id"] else "原稿保留",
                "source_file": str((folder / "最终事实与判定.json").relative_to(ROOT)),
            }
            if label != "full":
                non_full_facts.append({**record, "references": refs})
            non_full_relations.extend(
                {**record, "relation": ref} for ref in refs if ref["label"] != "full"
            )
        runtime = Counter(row["status"] for row in state["citation_verification"]["claims"])
        failed = sum(n for status, n in runtime.items() if status not in {"supported", "unverified"})
        expected = summaries[topic]["metrics"]["rewrite_once"]
        assert dict(facts) == expected["support_counts"]
        assert len(rows) == expected["fact_count"]
        assert relations["full"] == expected["citation_support"]["numerator"]
        assert sum(relations.values()) == expected["citation_support"]["denominator"]
        assert failed == expected["remaining_runtime_failures"]
        errors = state["citation_repair"].get("errors", [])
        rejected_batches.extend({"topic": topic, **error} for error in errors)
        per_topic.append({
            "topic": topic, "facts": dict(facts), "citation_relations": dict(relations),
            "unbound_facts": dict(unbound), "runtime_statuses": dict(runtime),
            "runtime_failures": failed, "rejected_batch_count": len(errors),
            "rejected_unit_count": sum(len(error["unit_ids"]) for error in errors),
        })
        fact_counts.update(facts)
        relation_counts.update(relations)
        unbound_counts.update(unbound)
        runtime_counts.update(runtime)

    assert fact_counts == Counter(full=194, partial=36, unsupported=117, unknown=8, contradicted=2)
    assert relation_counts == Counter(full=195, partial=39, unsupported=17, unknown=8, contradicted=2)
    assert unbound_counts == Counter(unsupported=100)
    assert len(non_full_facts) == 163 and len(non_full_relations) == 66
    assert runtime_counts == Counter(supported=170, unknown=58, missing_evidence=92, contradiction=1)
    assert len(rejected_batches) == 6
    rejected_units = sum(len(error["unit_ids"]) for error in rejected_batches)
    assert rejected_units == 47

    def change(metric):
        before, after = (aggregate[arm][metric] for arm in ["audit", "rewrite_once"])
        b, a = (x["numerator"] / x["denominator"] for x in [before, after])
        return {"before": before, "after": after, "percentage_point_change": (a - b) * 100,
                "relative_percent_change": (a / b - 1) * 100}

    summary = {
        "analysis_date": "2026-10-30", "timezone": "Asia/Singapore",
        "original_evaluation_date": "2026-10-06",
        "scope": "既有冻结消融结果复算；未重新标注或调用模型，沿用单源reviewer_override规则",
        "reference_labels": "历史模型复评标签，非人工金标；未改动事实复用历史标签，改动单元独立抽取复评",
        "config": config,
        "metric_changes": {name: change(name) for name in ["supported_fact_precision", "citation_support"]},
        "historical_aggregate_metrics": aggregate,
        "rewritten_fact_counts": dict(fact_counts),
        "rewritten_relation_counts": dict(relation_counts),
        "unbound_fact_counts": dict(unbound_counts),
        "runtime_status_counts": dict(runtime_counts),
        "rejected_batches": len(rejected_batches), "rejected_units": rejected_units,
        "per_topic": per_topic,
        "boundaries": {
            "measured": "冻结标签复算、指标分母、残留计数、改写批次错误记录",
            "indirect_inference": "证据缺失、语义外推和改写提交失败提示对应优化方向；未做因果消融",
            "not_evaluated": "人工金标、新优化收益、最新线上表现、跨任务泛化及全量现实真值",
        },
    }
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
               for path, digest in inputs.items())
    OUTPUT.mkdir(exist_ok=False)
    artifacts = {
        "指标与残留问题复算.json": summary,
        "未完整支持事实逐条明细.json": non_full_facts,
        "未完整支持引用逐条明细.json": non_full_relations,
        "改写批次失败记录.json": rejected_batches,
        "来源与完整性核对.json": {
            "all_checks_passed": True, "model_calls": 0, "web_calls": 0,
            "historical_labels_unchanged": True, "source_sha256": inputs,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    for name, value in artifacts.items():
        with (OUTPUT / name).open("x") as file:
            file.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "fact_counts": dict(fact_counts),
                      "relation_counts": dict(relation_counts), "checks": "passed",
                      "changes": summary["metric_changes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
