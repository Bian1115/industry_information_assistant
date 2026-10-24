"""复算既有引用关系标签及未知原因，不调用模型、不改动历史判定。"""

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "backend/eval/data/通用研究五题引用事实独立复核与模块一致性-20261007"
OUTPUT = ROOT / "backend/eval/data/通用研究五题未完整支持引用归因-20261016"
INPUT_NAMES = [
    "逐条独立复核及模块对齐.json",
    "冻结论断与原始证据.json",
    "独立复核与模块一致性指标.json",
    "核验分歧主因分类补充-20261007.json",
    "原始轨迹实际在线核验状态.json",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((SOURCE / name).read_text())


def main():
    fingerprints = {str(SOURCE / n): digest(SOURCE / n) for n in INPUT_NAMES}
    reviews = load(INPUT_NAMES[0])
    frozen = {row["id"]: row for row in load(INPUT_NAMES[1])}
    metrics = load(INPUT_NAMES[2])
    unknown_map = {}
    for cause, group in metrics["unknown_causes"].items():
        for claim_id in group["ids"]:
            assert claim_id not in unknown_map
            unknown_map[claim_id] = cause

    labels = Counter()
    per_topic = {}
    unknown_causes = Counter()
    details = []
    relation_keys = set()
    cited_claim_count = 0
    for row in reviews:
        assert row["claim"] == frozen[row["id"]]["claim"]
        assert len(row["references"]) == len(row["ref_support"])
        cited_claim_count += bool(row["references"])
        topic = str(row["topic"])
        topic_labels = per_topic.setdefault(topic, Counter())
        for ref in row["ref_support"]:
            relation_key = (row["id"], ref["key"])
            assert relation_key not in relation_keys
            relation_keys.add(relation_key)
            label = ref["label"]
            labels[label] += 1
            topic_labels[label] += 1
            if label == "full":
                continue
            original = next(
                x for x in frozen[row["id"]]["refs"] if x["id"] == ref["source_id"]
            )
            cause = None
            if label == "unknown":
                cause = unknown_map.get(row["id"])
                if cause is None:
                    # 原2026-10-07关系标签细化：联合partial，单个X-MOL来源unknown。
                    assert row["id"] == "16-body-034"
                    assert ref["source_id"] == "source_88fbfa43"
                    cause = "导航、广告、目录或残缺窗口"
                unknown_causes[cause] += 1
            details.append({
                "id": row["id"],
                "topic": row["topic"],
                "section": row["section"],
                "paragraph_id": row["paragraph_id"],
                "claim": row["claim"],
                "reference_key": ref["key"],
                "source_id": ref["source_id"],
                "relation_label": label,
                "joint_support": row["joint_support"],
                "relation_reason": ref["reason"],
                "joint_reason": row["reason"],
                "unknown_cause": cause,
                "source_title": original.get("title"),
                "source_url": original.get("url"),
                "document_id": original.get("document_id"),
                "evidence_windows": [
                    {
                        "start": w.get("start"), "end": w.get("end"),
                        "sentence_ids": w.get("sentence_ids"),
                        "quote_sha256": hashlib.sha256(w["quote"].encode()).hexdigest(),
                        "quote_chars": len(w["quote"]),
                    }
                    for w in original.get("windows", [])
                ],
                "module_status": row["module_status"],
                "original_module_path": row["original_module_path"],
                "provenance": str(SOURCE / INPUT_NAMES[0]),
                "evidence_provenance": str(SOURCE / INPUT_NAMES[1]),
            })

    expected = {"full": 144, "partial": 76, "unknown": 64, "unsupported": 7, "contradicted": 5}
    assert dict(labels) == expected == metrics["reference_relation_support"]
    assert len(reviews) == len(frozen) == 511
    assert cited_claim_count == 293
    assert sum(labels.values()) == len(relation_keys) == 296
    assert len(details) == 152
    assert unknown_causes == Counter({
        "导航、广告、目录或残缺窗口": 29,
        "中文编码或PDF字体损坏": 15,
        "未绑定来源": 11,
        "二进制DOC误作文本": 5,
        "验证页或JS未得到正文": 3,
        "摘要不足以确认全文缺项": 1,
    })
    assert all(digest(Path(p)) == value for p, value in fingerprints.items())

    summary = {
        "analysis_date": "2026-10-16",
        "timezone": "Asia/Singapore",
        "scope": "2026-10-03五份历史报告、2026-10-07既有标签的补充归因；未重新标注或调用模型",
        "review_origin": "历史Codex助手复核，非人工金标",
        "total_claims": 511,
        "cited_claims": cited_claim_count,
        "citation_relations": 296,
        "citation_support": {"numerator": 144, "denominator": 296, "value": 144 / 296},
        "relation_labels": dict(labels),
        "not_fully_supported_relations": 152,
        "non_full_decomposition": [
            {"label": label, "count": labels[label], "share_of_non_full": labels[label] / 152,
             "share_of_all_relations": labels[label] / 296}
            for label in ["partial", "unknown", "unsupported", "contradicted"]
        ],
        "partial_plus_unknown": {"numerator": 140, "denominator": 152, "value": 140 / 152},
        "unknown_relation_causes": dict(unknown_causes),
        "unknown_unit_note": "历史unknown原因按63条联合unknown论断计；本轮按引用关系计64组，多出16-body-034的单个X-MOL来源。",
        "per_topic": {
            topic: {"labels": dict(counts), "relations": sum(counts.values()),
                    "non_full": sum(counts.values()) - counts["full"],
                    "support_rate": counts["full"] / sum(counts.values())}
            for topic, counts in sorted(per_topic.items(), key=lambda item: int(item[0]))
        },
        "sensitivity_excluding_unknown": {"numerator": 144, "denominator": 232, "value": 144 / 232},
        "boundaries": {
            "measured": "冻结标签的关系计数、逐题计数、未知原因映射及来源指纹",
            "indirect_inference": "写作扩大结论、取证不全和来源绑定失败是改进方向；未做因果消融",
            "not_evaluated": "76组partial的互斥细分频次、最新生产版本、修复收益、全部论断现实真值",
        },
    }
    checks = {
        "all_passed": True,
        "claims_aligned": 511,
        "unique_relations": 296,
        "non_full_detail_rows": 152,
        "unknown_relations_categorized": 64,
        "historical_labels_unchanged": True,
        "historical_input_hashes_unchanged": True,
        "model_calls": 0,
        "web_calls": 0,
    }
    OUTPUT.mkdir(exist_ok=False)
    for name, data in [
        ("归因汇总.json", summary),
        ("未完整支持引用逐条明细.json", details),
        ("来源文件校验.json", {"inputs": fingerprints, "script_sha256": digest(Path(__file__))}),
        ("完整性核对.json", checks),
    ]:
        (OUTPUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "labels": dict(labels), "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
