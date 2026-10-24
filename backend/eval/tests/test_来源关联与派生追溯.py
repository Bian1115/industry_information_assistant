"""单份原文、按需证据视图和派生输入追溯的离线回归。"""

import copy
import hashlib
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from service.deep_research_v2.evidence import (
    normalize_evidence_state, fact_views, build_evidence_view,
    hypothesis_evidence_views, data_point_views,
)
from service.deep_research_v2.quality_pipeline import (
    bind_verified_evidence,
    evaluate_report_gate, evidence_fingerprint,
)
from service.deep_research_v2.deterministic_analysis import run_deterministic_analysis
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.agents.writer import LeadWriter


URL = "https://example.gov.cn/report"
NEWS_URL = "https://news.example.org/report"


def state_fixture():
    return {
        "raw_sources": [
            {"id": "s1", "url": URL, "title": "官方报告", "content": "期初100，期末121。", "source_type": "official"},
            {"id": "s2", "url": NEWS_URL, "title": "新闻", "content": "期末121。", "source_type": "news"},
        ],
        "facts": [{"id": "f1", "source_id": "s1", "content": "期初100。", "related_sections": ["sec_1"]}],
        "hypotheses": [{"id": "h1", "content": "收入增长", "status": "partially_supported"}],
        "hypothesis_evidence": [{"id": "r1", "fact_id": "f1", "hypothesis_id": "h1", "evidence_type": "supports", "evidence_quote": "期初100"}],
        "data_points": [
            {"id": "dp1", "source_id": "s1", "entity": "甲", "name": "期初", "value": 100, "year": 2022},
            {"id": "dp2", "source_id": "s2", "entity": "甲", "name": "期末", "value": 121, "year": 2024},
        ],
        "research_contract": {"required_entities": ["甲"], "required_calculations": [
            {"id": "cagr", "type": "cagr", "inputs": ["期初", "期末"], "output": "增长率"},
        ]},
        "allowed_domains": ["example.gov.cn", "news.example.org"],
        "final_report": f"期初100（[官方报告]({URL})）。",
    }


def legacy_fixture():
    state = state_fixture()
    source = state["raw_sources"][0]
    source.update(snapshot_id="old_snapshot", source_url=URL, fetched_at="2026-09-09T12:00:00")
    fact = state["facts"][0]
    fact.pop("source_id")
    fact.update(source_url=URL, source_name="官方报告", source_type="official", verified=True,
                metadata={"source_id": "old_snapshot", "evidence_excerpt": "重复摘录", "snapshot_fetched_at": "时间"})
    relation = state["hypothesis_evidence"][0]
    relation.update(snapshot_id="old_snapshot", snapshot={"content_sha256": "旧哈希"}, source_url=URL)
    state["hypotheses"][0]["evidence_for"] = [copy.deepcopy(relation)]
    state["evidence_ledger"] = [{"claim_id": "f1", "excerpt": "旧账本", "source_url": URL}]
    return state


class SourceReferenceTests(unittest.TestCase):
    def test_legacy_migration_is_idempotent_and_preserves_only_canonical_provenance(self):
        state = legacy_fixture()
        normalize_evidence_state(state)
        first = copy.deepcopy(state)
        normalize_evidence_state(state)
        self.assertEqual(state, first)
        self.assertEqual(state["facts"][0]["source_id"], "s1")
        for key in ("source_url", "source_name", "source_type", "verified", "snapshot_id"):
            self.assertNotIn(key, state["facts"][0])
        self.assertEqual(state["facts"][0]["metadata"], {})
        self.assertEqual(state["raw_sources"][0]["fetched_at"], "2026-09-09T12:00:00")
        self.assertEqual(state["raw_sources"][0]["content_sha256"], hashlib.sha256("期初100，期末121。".encode()).hexdigest())
        for collection in (state["raw_sources"], state["hypothesis_evidence"], state["hypotheses"][0]["evidence_for"]):
            for record in collection:
                self.assertNotIn("snapshot_id", record)
                self.assertNotIn("snapshot", record)
                self.assertNotIn("source_url", record)
        self.assertNotIn("evidence_ledger", state)

    def test_views_are_read_only_and_reflect_changed_source_and_relationship(self):
        state = state_fixture()
        before = copy.deepcopy(state)
        self.assertEqual(build_evidence_view(state)[0]["hypothesis_evidence_ids"], ["r1"])
        self.assertEqual(hypothesis_evidence_views(state)[0]["source_url"], URL)
        payload = json.loads(LeadWriter.__new__(LeadWriter)._format_hypothesis_ledger(state))
        self.assertEqual(payload[0]["evidence"][0]["fact_id"], "f1")
        self.assertNotIn("source_id", payload[0]["evidence"][0])
        self.assertEqual(state, before)
        state["raw_sources"][0]["title"] = "更新的标题"
        self.assertEqual(fact_views(state)[0]["source_name"], "更新的标题")
        self.assertNotIn("source_name", state["facts"][0])

    def test_wrong_explicit_reference_does_not_fall_back_to_available_url(self):
        for source_id in ("missing", "s2"):
            with self.subTest(source_id=source_id):
                state = state_fixture()
                state["facts"][0].update(source_id=source_id, source_url=URL, reader_verified=True)
                bind_verified_evidence(state)
                self.assertEqual(state["facts"], [])

    def test_ledger_excerpt_alone_cannot_authorize_evidence(self):
        state = state_fixture()
        state["raw_sources"] = []
        state["facts"][0].update(reader_verified=True, source_url=URL)
        state["evidence_ledger"] = [{"source_url": URL, "reader_verified": True, "excerpt": "期初100"}]
        bind_verified_evidence(state)
        self.assertEqual(state["facts"], [])
        self.assertEqual(state["data_points"], [])
        self.assertNotIn("evidence_ledger", state)

    def test_report_gate_only_enforces_source_scope(self):
        state = state_fixture()
        self.assertTrue(evaluate_report_gate(state)["passed"])
        # 同一来源范围内的其它链接不再被发布门禁拦截：
        # 这条链路已不做引用核验，门禁只管非空与来源范围。
        state["final_report"] = f"另见[新闻]({NEWS_URL})。"
        self.assertTrue(evaluate_report_gate(state)["passed"])
        state["final_report"] = "另见[范围外](https://outside.example/x)。"
        self.assertFalse(evaluate_report_gate(state)["passed"])

    def test_fingerprint_reads_whole_content_and_relation_and_ignores_legacy_ledger(self):
        state = state_fixture()
        state["raw_sources"][0]["content"] = "前文" * 2000 + "甲"
        initial = evidence_fingerprint(state)
        state["raw_sources"][0]["content"] = "前文" * 2000 + "乙"
        self.assertNotEqual(initial, evidence_fingerprint(state))
        changed = evidence_fingerprint(state)
        state["hypothesis_evidence"][0]["evidence_type"] = "refutes"
        self.assertNotEqual(changed, evidence_fingerprint(state))
        current = evidence_fingerprint(state)
        state["raw_sources"].reverse()
        state["evidence_ledger"] = [{"excerpt": "不应使用"}]
        self.assertEqual(current, evidence_fingerprint(state))

    def test_graph_checkpoint_update_keeps_business_data_and_drops_runtime_fields(self):
        from service.deep_research_v2.graph import pack_state

        state = legacy_fixture()
        state["messages"] = [{"type": "旧进度", "content": "旧检查点临时队列"}]
        state["_run_id"] = "run-1"
        state["_emit_event"] = lambda event: None
        before = copy.deepcopy({k: v for k, v in state.items() if not k.startswith("_")})

        saved = pack_state(state)["research"]

        # 运行期字段不进检查点；业务数据原样保留，输入也不被改写。
        self.assertEqual([k for k in saved if k.startswith("_")], [])
        self.assertEqual(saved["facts"], before["facts"])
        self.assertNotIn("messages", saved)
        self.assertEqual(state["messages"], before["messages"])
        self.assertEqual(state["_run_id"], "run-1")

        # 旧账本字段在读取侧被迁移掉，来源关联仍指向原始记录。
        normalize_evidence_state(saved)
        self.assertEqual(saved["facts"][0]["source_id"], "s1")
        self.assertNotIn("evidence_ledger", saved)
        self.assertTrue(evaluate_report_gate(saved)["passed"])


class DerivedProvenanceTests(unittest.TestCase):
    def test_cagr_separates_calculation_status_from_minimum_input_source_tier(self):
        for kind, expected in (("official", "primary"), ("news", "secondary")):
            with self.subTest(kind=kind):
                state = state_fixture()
                state["raw_sources"][1]["source_type"] = kind
                point = run_deterministic_analysis(state)["derived_points"][0]
                self.assertAlmostEqual(point["value"], 10.0)
                self.assertEqual(point["origin"], "derived")
                self.assertEqual(point["input_point_ids"], ["dp1", "dp2"])
                self.assertEqual(point["calculation_status"], "passed")
                self.assertTrue(point["formula"])
                self.assertTrue(point["reader_verified"])
                self.assertEqual(point["source_tier"], expected)

    def test_missing_body_does_not_become_primary_because_arithmetic_passed(self):
        state = state_fixture()
        state["raw_sources"][1]["content"] = ""
        point = run_deterministic_analysis(state)["derived_points"][0]
        self.assertEqual(point["calculation_status"], "passed")
        self.assertFalse(point["reader_verified"])
        self.assertEqual(point["source_tier"], "low")
        bind_verified_evidence(state)
        self.assertEqual([p["id"] for p in state["data_points"]], ["dp1"])

    def test_news_calculation_preserves_derived_source_tier(self):
        state = state_fixture()
        run_deterministic_analysis(state)
        growth = next(p for p in data_point_views(state) if p["name"] == "增长率")
        self.assertEqual(growth["source_tier"], "secondary")
        state["raw_sources"][1]["source_type"] = "official"
        growth = next(p for p in data_point_views(state) if p["name"] == "增长率")
        self.assertEqual(growth["source_tier"], "primary")

    def test_legacy_marker_cannot_self_authorize_and_missing_input_or_cycle_is_rejected(self):
        for inputs in ([], ["missing"], ["derived"]):
            with self.subTest(inputs=inputs):
                state = state_fixture()
                state["data_points"].append({"id": "derived", "source": "deterministic_calculation",
                    "reader_verified": True, "source_tier": "primary", "input_point_ids": inputs,
                    "calculation_status": "passed", "formula": "x*2"})
                point = data_point_views(state)[-1]
                self.assertFalse(point["reader_verified"])
                self.assertEqual(point["source_tier"], "low")

    def test_nested_derivation_tracks_weakest_input_and_duplicate_ids_are_rejected(self):
        state = state_fixture()
        first = run_deterministic_analysis(state)["derived_points"][0]
        state["data_points"].append({"id": "nested", "origin": "derived", "input_point_ids": [first["id"]],
                                     "calculation_status": "passed", "formula": "x*2"})
        self.assertEqual(data_point_views(state)[-1]["source_tier"], "secondary")
        state["data_points"].append(copy.deepcopy(state["data_points"][1]))
        nested = next(p for p in data_point_views(state) if p["id"] == "nested")
        self.assertFalse(nested["reader_verified"])

    def test_weighted_score_and_ranking_capture_all_inputs_and_do_not_duplicate_on_rerun(self):
        state = state_fixture()
        state["data_points"][1].update(entity="乙", name="期初")
        state["research_contract"].update(required_entities=["甲", "乙"], required_calculations=[
            {"id": "score", "type": "weighted_score", "weights": {"期初": 1.0}, "output": "得分"},
            {"id": "rank", "type": "ranking", "inputs": ["得分"], "output": "排名"},
        ])
        first = run_deterministic_analysis(state)["derived_points"]
        scores = [p for p in first if p["name"] == "得分"]
        ranks = [p for p in first if p["name"] == "排名"]
        self.assertEqual([p["input_point_ids"] for p in scores], [["dp1"], ["dp2"]])
        for rank in ranks:
            self.assertEqual(set(rank["input_point_ids"]), {p["id"] for p in scores})
            self.assertEqual(rank["source_tier"], "secondary")
        second = run_deterministic_analysis(state)["derived_points"]
        self.assertEqual(first, second)
        self.assertEqual(len(state["data_points"]), 6)


# 本地知识库检索已随"仅指定网站"模式退出主链路：Scout 不再产生 local:// 结果，
# 对应的本地分块落库用例一并删除，而不是改成永远跳过的空壳测试。


if __name__ == "__main__":
    unittest.main()
