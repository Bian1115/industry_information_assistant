"""假设证据三分桶、页面快照引用与结论映射机制测试。"""

import hashlib
import json
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
sys.path.insert(0, str(APP_DIR))

from service.deep_research_v2.agents.scout import DeepScout  # noqa: E402
from service.deep_research_v2.agents.writer import LeadWriter  # noqa: E402
from service.deep_research_v2.evidence import hypothesis_evidence_views
from service.deep_research_v2.state import create_initial_state  # noqa: E402


SOURCE_URL = "https://official.example.cn/report"
SOURCE_CONTENT = "低代码平台在标准化应用中缩短交付周期，但复杂系统维护成本仍有争议。"


def make_scout():
    scout = DeepScout.__new__(DeepScout)
    scout.name = "DeepScout"
    scout.logger = logging.getLogger("test.hypothesis.scout")
    return scout


def make_writer():
    writer = LeadWriter.__new__(LeadWriter)
    writer.name = "LeadWriter"
    writer.logger = logging.getLogger("test.hypothesis.writer")
    return writer


def make_state():
    return {
        "hypotheses": [{
            "id": "h_1",
            "content": "低代码平台能显著缩短企业应用交付周期",
            "status": "unverified",
            "evidence_for": [],
            "evidence_against": [],
            "evidence_inconclusive": [],
        }],
        "facts": [{
            "id": f"fact_{kind}",
            "content": f"{kind}: {SOURCE_CONTENT}",
            "source_url": SOURCE_URL,
        } for kind in ("supports", "refutes", "inconclusive")],
        "hypothesis_evidence": [],
        "raw_sources": [{
            "id": "source_001",
            "snapshot_id": "source_001",
            "source_url": SOURCE_URL,
            "url": SOURCE_URL,
            "title": "独立行业调研",
            "content": SOURCE_CONTENT,
            "content_sha256": hashlib.sha256(
                SOURCE_CONTENT.encode("utf-8")
            ).hexdigest(),
            "fetched_at": "2026-08-30T12:00:00+08:00",
        }],
    }


class HypothesisEvidencePersistenceTests(unittest.TestCase):
    def test_initial_state_has_traceability_fields(self):
        state = create_initial_state("测试问题", "session-1", ["example.gov.cn"])

        self.assertEqual(state["hypothesis_evidence"], [])
        self.assertEqual(state["conclusion_evidence_map"], [])

    def test_inconclusive_is_persisted_with_fact_reference(self):
        scout = make_scout()
        state = make_state()

        scout._update_hypothesis_status(state, [{
            "hypothesis_id": "h_1",
            "fact_id": "fact_inconclusive",
            "evidence_type": "inconclusive",
            "evidence_summary": "厂商声称效率提升，但未公开样本和对照组",
            "evidence_quote": "报告没有披露样本量和对照组",
            "source_url": SOURCE_URL,
        }])

        evidence = state["hypothesis_evidence"][0]
        self.assertTrue(evidence["id"].startswith("hyp_ev_"))
        self.assertEqual(evidence["evidence_type"], "inconclusive")
        self.assertEqual(evidence["fact_id"], "fact_inconclusive")
        self.assertNotIn("source_url", evidence)
        self.assertNotIn("snapshot_id", evidence)
        self.assertNotIn("snapshot", evidence)
        view = hypothesis_evidence_views(state)[0]
        self.assertEqual(view["source_url"], SOURCE_URL)
        self.assertEqual(view["source_id"], "source_001")
        self.assertEqual(
            state["hypotheses"][0]["evidence_inconclusive"][0]["id"],
            evidence["id"],
        )
        self.assertEqual(state["hypotheses"][0]["status"], "uncertain")

    def test_support_and_refute_produce_mixed_status_and_deduplicate(self):
        scout = make_scout()
        state = make_state()
        items = [{
            "hypothesis_id": "h_1",
            "fact_id": "fact_supports",
            "evidence_type": "supports",
            "evidence_quote": SOURCE_CONTENT,
            "evidence_summary": "标准化内部应用的交付周期缩短",
            "source_url": SOURCE_URL,
        }, {
            "hypothesis_id": "h_1",
            "fact_id": "fact_refutes",
            "evidence_type": "refutes",
            "evidence_quote": SOURCE_CONTENT,
            "evidence_summary": "复杂系统的后期维护成本上升",
            "source_url": SOURCE_URL,
        }]

        scout._update_hypothesis_status(state, items)
        scout._update_hypothesis_status(state, items)

        self.assertEqual(len(state["hypothesis_evidence"]), 2)
        self.assertEqual(len(state["hypotheses"][0]["evidence_for"]), 1)
        self.assertEqual(len(state["hypotheses"][0]["evidence_against"]), 1)
        self.assertEqual(state["hypotheses"][0]["status"], "mixed")

    def test_unverified_page_is_not_promoted_to_auditable_evidence(self):
        scout = make_scout()
        state = make_state()

        scout._update_hypothesis_status(state, [{
            "hypothesis_id": "h_1",
            "fact_id": "fact_supports",
            "evidence_type": "supports",
            "evidence_quote": SOURCE_CONTENT,
            "evidence_summary": "只有搜索摘要，没有全文快照",
            "source_url": "https://unread.example.cn/article",
        }])

        self.assertEqual(state["hypothesis_evidence"], [])
        self.assertEqual(state["hypotheses"][0]["status"], "unverified")


class PageSnapshotPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_reader_content_is_saved_as_stable_snapshot(self):
        scout = make_scout()
        scout.web_reader = SimpleNamespace(read_many=AsyncMock(return_value=[
            SimpleNamespace(
                url=SOURCE_URL,
                final_url=SOURCE_URL,
                fetch_status="success",
                extractor="test_reader",
                error="",
                failure_category="",
                http_status=200,
                recovery_action="none",
                attempts=[],
                from_cache=False,
                fetched_at="2026-08-30T12:00:00+08:00",
                success=True,
                content=SOURCE_CONTENT,
                title="独立行业调研",
                published_at="2026-08-01",
                content_type="text/html",
            )
        ]))
        scout._update_web_reader_stats = lambda state: None
        state = {
            "raw_sources": [], "web_reader_attempts": [],
            "allowed_domains": ["official.example.cn"],
        }
        results = [{"url": SOURCE_URL, "title": "搜索结果"}]

        await scout._enrich_results_with_full_text(
            state, results, section_id="sec_1", limit=1
        )

        snapshot = state["raw_sources"][0]
        self.assertTrue(snapshot["id"])
        self.assertNotIn("snapshot_id", snapshot)
        self.assertNotIn("source_url", snapshot)
        self.assertEqual(snapshot["content"], SOURCE_CONTENT)
        self.assertEqual(
            snapshot["content_sha256"],
            hashlib.sha256(SOURCE_CONTENT.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(snapshot["fetched_at"], "2026-08-30T12:00:00+08:00")

        # 同一来源再次返回新正文时，事实提炼仍必须使用已保存的正文。
        scout.web_reader.read_many.return_value[0].content = "另一版正文"
        repeated_results = [{"url": SOURCE_URL, "title": "再次检索"}]
        await scout._enrich_results_with_full_text(state, repeated_results, section_id="sec_2", limit=1)
        self.assertEqual(len(state["raw_sources"]), 1)
        self.assertEqual(state["raw_sources"][0]["id"], snapshot["id"])
        self.assertEqual(repeated_results[0]["full_content"], SOURCE_CONTENT)
        self.assertEqual(state["raw_sources"][0]["related_sections"], ["sec_1", "sec_2"])


class ConclusionEvidenceMappingTests(unittest.TestCase):
    def test_writer_filters_fabricated_ids_and_outputs_three_way_mapping(self):
        scout = make_scout()
        writer = make_writer()
        state = make_state()
        scout._update_hypothesis_status(state, [{
            "hypothesis_id": "h_1",
            "fact_id": "fact_supports",
            "evidence_type": "supports",
            "evidence_quote": SOURCE_CONTENT,
            "evidence_summary": "标准化应用缩短交付周期",
            "source_url": SOURCE_URL,
        }, {
            "hypothesis_id": "h_1",
            "fact_id": "fact_refutes",
            "evidence_type": "refutes",
            "evidence_quote": SOURCE_CONTENT,
            "evidence_summary": "复杂系统维护成本上升",
            "source_url": SOURCE_URL,
        }, {
            "hypothesis_id": "h_1",
            "fact_id": "fact_inconclusive",
            "evidence_type": "inconclusive",
            "evidence_quote": SOURCE_CONTENT,
            "evidence_summary": "总体成本证据仍然不足",
            "source_url": SOURCE_URL,
        }])
        ids = {
            item["evidence_type"]: item["id"]
            for item in state["hypothesis_evidence"]
        }
        conclusions = [{
            "id": "c_1",
            "text": "低代码适合标准化内部应用，但复杂系统长期成本仍不确定。",
            "hypothesis_ids": ["h_1", "h_fake"],
            "supporting_evidence_ids": [ids["supports"], "hyp_ev_fake"],
            "contradicting_evidence_ids": [ids["refutes"]],
            "inconclusive_evidence_ids": [ids["inconclusive"]],
            "confidence": "medium",
            "boundary_conditions": ["仅适用于标准化场景"],
        }]

        mapping = writer._normalize_conclusion_evidence_map(conclusions, state)

        self.assertEqual(mapping[0]["hypothesis_ids"], ["h_1"])
        self.assertEqual(
            mapping[0]["supporting_evidence_ids"], [ids["supports"]]
        )
        self.assertEqual(
            mapping[0]["contradicting_evidence_ids"], [ids["refutes"]]
        )
        self.assertEqual(
            mapping[0]["inconclusive_evidence_ids"], [ids["inconclusive"]]
        )
        self.assertEqual(mapping[0]["traceability_status"], "mapped")

    def test_writer_view_contains_evidence_and_source_ids(self):
        scout = make_scout()
        writer = make_writer()
        state = make_state()
        scout._update_hypothesis_status(state, [{
            "hypothesis_id": "h_1",
            "fact_id": "fact_supports",
            "evidence_type": "supports",
            "evidence_quote": SOURCE_CONTENT,
            "evidence_summary": "标准化应用缩短交付周期",
            "source_url": SOURCE_URL,
        }])

        payload = json.loads(writer._format_hypothesis_ledger(state))

        self.assertEqual(payload[0]["hypothesis_id"], "h_1")
        self.assertTrue(payload[0]["evidence"][0]["evidence_id"])
        self.assertEqual(payload[0]["evidence"][0]["fact_id"], "fact_supports")
        self.assertNotIn("source_id", payload[0]["evidence"][0])


if __name__ == "__main__":
    unittest.main()
