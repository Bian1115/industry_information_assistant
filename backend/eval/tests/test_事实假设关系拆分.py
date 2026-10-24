"""事实与假设关系拆分：多对多关联、来源绑定、去重和检索链路回归。"""

import copy
import json
import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.evidence import build_evidence_view, hypothesis_evidence_views
from service.deep_research_v2.quality_pipeline import bind_verified_evidence
from service.deep_research_v2.state import create_initial_state, ResearchPhase


URL = "https://example.gov.cn/report"
OTHER_URL = "https://example.gov.cn/second-report"
CONTENT = "示例企业本期收入增长20%，运营成本增长30%。"


def make_state():
    state = create_initial_state("示例企业经营情况", "关系拆分测试", ["example.gov.cn"])
    state["hypotheses"] = [{
        "id": f"h_{i}", "content": f"示例假设{i}", "status": "unverified",
        "evidence_for": [], "evidence_against": [], "evidence_inconclusive": [],
    } for i in range(1, 5)]
    state["raw_sources"] = [{
        "id": f"source_{i}", "snapshot_id": f"source_{i}",
        "source_url": url, "content": CONTENT, "title": "示例原始报告",
        "fetched_at": "2026-09-08T10:00:00+08:00", "related_sections": ["sec_1"],
    } for i, url in enumerate((URL, OTHER_URL), 1)]
    state["outline"] = [{
        "id": "sec_1", "title": "经营情况", "status": "pending",
        "description": "检索经营数据", "search_queries": ["示例企业经营报告"],
    }]
    return state


def make_scout():
    scout = DeepScout.__new__(DeepScout)
    scout.name = "DeepScout"
    scout.logger = logging.getLogger("test.事实假设关系拆分")
    scout.add_message = lambda *args, **kwargs: None
    return scout


def results():
    return [{
        "url": url, "title": "示例报告", "summary": CONTENT,
        "reader_verified": True, "full_content": CONTENT,
    } for url in (URL, OTHER_URL)]


def relation(hypothesis_id="h_1", polarity="supports", fact_id="f_1"):
    return {
        "fact_id": fact_id, "hypothesis_id": hypothesis_id,
        "evidence_type": polarity, "evidence_summary": "对该假设的判断依据",
        "evidence_quote": CONTENT,
    }


def analysis():
    return {
        "extracted_facts": [{
            "fact_id": "f_1", "content": CONTENT, "source_url": URL,
            "source_name": "示例报告", "source_type": "official",
            "credibility_score": 0.9,
            "data_points": [{"entity": "示例企业", "name": "收入增速", "value": 20, "unit": "%"}],
        }],
        "hypothesis_evidence": [relation(), relation("h_2", "refutes")],
    }


def ingest(state, value=None, section="sec_1", scout=None):
    return (scout or make_scout())._ingest_search_analysis(
        state, value if value is not None else analysis(), results(), related_sections=[section]
    )


class FactRelationTests(unittest.TestCase):
    def test_one_fact_can_support_and_refute_different_hypotheses(self):
        state = make_state()
        stats = ingest(state)
        self.assertEqual(stats["facts_added"], 1)
        self.assertEqual(stats["relations_added"], 2)
        fact = state["facts"][0]
        self.assertNotEqual(fact["id"], "f_1")
        self.assertNotIn("related_hypothesis", fact)
        self.assertNotIn("hypothesis_support", fact)
        self.assertEqual({r["fact_id"] for r in state["hypothesis_evidence"]}, {fact["id"]})
        self.assertEqual({(r["hypothesis_id"], r["evidence_type"]) for r in state["hypothesis_evidence"]},
                         {("h_1", "supports"), ("h_2", "refutes")})

    def test_fact_legacy_labels_do_not_override_relations(self):
        state = make_state()
        value = analysis()
        value["extracted_facts"][0].update(related_hypothesis="h_1", hypothesis_support="refutes")
        ingest(state, value)
        self.assertEqual(state["hypotheses"][0]["evidence_for"][0]["evidence_type"], "supports")
        self.assertNotIn("hypothesis_support", state["facts"][0])

    def test_legacy_fact_labels_alone_do_not_generate_a_second_judgment(self):
        state = make_state()
        value = analysis()
        value["hypothesis_evidence"] = []
        value["extracted_facts"][0].update(related_hypothesis="h_1", hypothesis_support="supports")
        ingest(state, value)
        self.assertEqual(len(state["facts"]), 1)
        self.assertEqual(state["hypothesis_evidence"], [])

    def test_duplicate_fact_keeps_id_and_new_relationship_after_checkpoint_restore(self):
        state = make_state()
        value = analysis()
        value["hypothesis_evidence"] = [relation()]
        ingest(state, value)
        original_id = state["facts"][0]["id"]
        state = json.loads(json.dumps(state))
        value["hypothesis_evidence"] = [relation("h_2", "refutes")]
        stats = ingest(state, value, section="sec_2")
        self.assertEqual(stats["facts_reused"], 1)
        self.assertEqual(len(state["facts"]), 1)
        self.assertEqual(state["facts"][0]["related_sections"], ["sec_1", "sec_2"])
        self.assertEqual({r["fact_id"] for r in state["hypothesis_evidence"]}, {original_id})

    def test_same_temporary_id_in_later_batch_maps_to_its_own_fact(self):
        state = make_state()
        ingest(state)
        value = analysis()
        value["extracted_facts"][0]["content"] = "示例企业现金储备增加。"
        value["hypothesis_evidence"] = [relation("h_3")]
        ingest(state, value)
        self.assertEqual(len(state["facts"]), 2)
        self.assertEqual(state["hypothesis_evidence"][-1]["fact_id"], state["facts"][-1]["id"])

    def test_same_content_from_different_sources_keeps_correct_source(self):
        state = make_state()
        ingest(state)
        value = analysis()
        value["extracted_facts"][0]["source_url"] = OTHER_URL
        ingest(state, value)
        self.assertEqual(len(state["facts"]), 2)
        self.assertEqual(len(state["hypothesis_evidence"]), 4)
        self.assertEqual(hypothesis_evidence_views(state)[-1]["source_url"], OTHER_URL)
        self.assertEqual(hypothesis_evidence_views(state)[-1]["source_id"], "source_2")

    def test_ambiguous_batch_fact_ids_do_not_create_relationships(self):
        state = make_state()
        value = analysis()
        other = copy.deepcopy(value["extracted_facts"][0])
        other["content"] = "另一条事实。"
        value["extracted_facts"].append(other)
        ingest(state, value)
        self.assertEqual(len(state["facts"]), 2)
        self.assertEqual(state["hypothesis_evidence"], [])

    def test_missing_unknown_or_rejected_fact_references_are_not_bound(self):
        for case in ("missing_id", "unknown_id", "rejected_source", "unknown_hypothesis"):
            with self.subTest(case=case):
                state = make_state()
                value = analysis()
                value["hypothesis_evidence"] = [relation()]
                if case == "missing_id":
                    value["hypothesis_evidence"][0].pop("fact_id")
                elif case == "unknown_id":
                    value["hypothesis_evidence"][0]["fact_id"] = "f_unknown"
                elif case == "rejected_source":
                    value["extracted_facts"][0]["source_url"] = "https://unread.example.cn/report"
                else:
                    value["hypothesis_evidence"][0]["hypothesis_id"] = "h_unknown"
                ingest(state, value)
                self.assertEqual(state["hypothesis_evidence"], [])

    def test_missing_quote_is_not_filled_with_fact_summary(self):
        state = make_state()
        value = analysis()
        for item in value["hypothesis_evidence"]:
            item.pop("evidence_quote")
        ingest(state, value)
        self.assertEqual(len(state["facts"]), 1)
        self.assertEqual(state["hypothesis_evidence"], [])

    def test_relation_cannot_override_fact_source(self):
        state = make_state()
        value = analysis()
        for item in value["hypothesis_evidence"]:
            item["source_url"] = OTHER_URL
        ingest(state, value)
        self.assertEqual(state["hypothesis_evidence"], [])

    def test_rephrasing_same_relationship_does_not_inflate_evidence_count(self):
        state = make_state()
        ingest(state)
        value = analysis()
        for item in value["hypothesis_evidence"]:
            item["evidence_summary"] = "同一判断的另一种表述"
        ingest(state, value)
        self.assertEqual(len(state["hypothesis_evidence"]), 2)
        self.assertEqual(state["hypotheses"][0]["status"], "partially_supported")

    def test_conflicting_polarities_for_same_pair_are_rejected_in_one_batch(self):
        state = make_state()
        value = analysis()
        value["hypothesis_evidence"] = [relation(), relation(polarity="refutes")]
        ingest(state, value)
        self.assertEqual(state["hypothesis_evidence"], [])

    def test_later_conflicting_polarity_does_not_overwrite_existing_relation(self):
        state = make_state()
        ingest(state)
        value = analysis()
        value["hypothesis_evidence"] = [relation(polarity="refutes")]
        ingest(state, value)
        self.assertEqual(state["hypotheses"][0]["evidence_against"], [])
        self.assertEqual(state["hypotheses"][0]["evidence_for"][0]["evidence_type"], "supports")

    def test_fact_view_references_all_relations_without_persisting_ledger(self):
        state = make_state()
        ingest(state)
        state["facts"][0].update(related_hypothesis="h_legacy", hypothesis_support="refutes")
        bind_verified_evidence(state)
        self.assertNotIn("evidence_ledger", state)
        self.assertEqual(set(build_evidence_view(state)[0]["hypothesis_evidence_ids"]),
                         {r["id"] for r in state["hypothesis_evidence"]})
        self.assertNotIn("hypothesis_id", build_evidence_view(state)[0])
        self.assertNotIn("hypothesis_support", build_evidence_view(state)[0])
        self.assertNotIn("related_hypothesis", state["facts"][0])
        self.assertEqual(state["data_points"][0]["source"], URL)

    def test_writer_receives_fact_links_and_preserves_conclusion_mapping(self):
        state = make_state()
        ingest(state)
        writer = LeadWriter.__new__(LeadWriter)
        payload = json.loads(writer._format_hypothesis_ledger(state))
        self.assertEqual(payload[0]["evidence"][0]["fact_id"], state["facts"][0]["id"])
        ev = state["hypothesis_evidence"][0]
        mapped = writer._normalize_conclusion_evidence_map([{
            "text": "示例结论", "hypothesis_ids": ["h_1"],
            "supporting_evidence_ids": [ev["id"], "fake"],
        }], state)
        self.assertEqual(mapped[0]["supporting_evidence_ids"], [ev["id"]])
        self.assertEqual(mapped[0]["traceability_status"], "mapped")

    def test_legacy_ledger_without_fact_id_remains_readable(self):
        state = make_state()
        state["hypothesis_evidence"] = [{
            "id": "legacy_ev", "hypothesis_id": "h_1", "evidence_type": "supports",
            "summary": "历史证据", "source_url": URL, "snapshot_id": "source_1",
        }]
        payload = json.loads(LeadWriter.__new__(LeadWriter)._format_hypothesis_ledger(state))
        self.assertEqual(payload[0]["evidence"][0]["evidence_id"], "legacy_ev")
        self.assertIsNone(payload[0]["evidence"][0]["fact_id"])


class RetrievalRouteTests(unittest.IsolatedAsyncioTestCase):
    async def run_route(self, route):
        from service.web_document import index_document
        state = make_state()
        scout = make_scout()
        # 实际正文读取已经执行质量门禁和程序编号；路由桩也必须提供同样的材料。
        numbered_results = results()
        for result in numbered_results:
            result.update(body_quality={"status": "passed"}, **index_document(CONTENT, result["url"]))
        for source in state["raw_sources"]:
            source.update(body_quality={"status": "passed"}, **index_document(CONTENT, source["source_url"]))
        scout._execute_search = AsyncMock(return_value=numbered_results)
        scout._enrich_results_with_full_text = AsyncMock()
        scout._emit_search_results_event = lambda state: None
        numbered_analysis = analysis()
        numbered_analysis["extracted_facts"][0].update(
            document_id=numbered_results[0]["document_id"], sentence_ids=["s000001"])
        scout.call_llm = AsyncMock(return_value=json.dumps(numbered_analysis, ensure_ascii=False))
        scout.parse_json_response = json.loads
        if route == "initial":
            await scout._research_section(state, state["outline"][0])
        elif route == "deep":
            await scout._execute_deep_search(
                state, "sec_1", ["原始报告"], "source_tracing", state["hypotheses"], max_depth=1
            )
        else:
            state["phase"] = ResearchPhase.RE_RESEARCHING.value
            state["pending_research_tasks"] = [{
                "query": "补充报告", "target_section": "sec_1", "issue_id": "issue_1",
            }]
            await scout.process(state)
            self.assertEqual(state["facts"][0]["critic_issue_id"], "issue_1")
        self.assertEqual(len(state["facts"]), 1)
        self.assertEqual(len(state["hypothesis_evidence"]), 2)
        self.assertEqual(scout.call_llm.await_count, 1)
        prompt = scout.call_llm.await_args.kwargs["user_prompt"]
        self.assertIn('"fact_id": "f_1"', prompt)
        self.assertIn("[h_4]", prompt)
        self.assertNotIn('"related_hypothesis"', prompt)
        self.assertNotIn('"hypothesis_support"', prompt)
        bind_verified_evidence(state)
        self.assertNotIn("evidence_ledger", state)
        self.assertEqual(len(state["data_points"]), 1)
        self.assertEqual(state["data_points"][0]["source"], URL)

    async def test_initial_search_uses_separate_relationships(self):
        await self.run_route("initial")

    async def test_deep_search_uses_separate_relationships(self):
        await self.run_route("deep")

    async def test_supplementary_search_uses_separate_relationships(self):
        await self.run_route("supplementary")


if __name__ == "__main__":
    unittest.main()
