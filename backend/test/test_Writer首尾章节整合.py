"""首尾只消费章节正文；章节依据回读原文的离线回归。"""
import copy
import json
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.citation_verification import _claims_for_stage, verify_citations
from service.deep_research_v2.state import create_initial_state
from service.deep_research_v2.writer_claims import (
    bind_synthesis_claims, build_synthesis_claim_index, collect_report_claims,
    format_synthesis_claim_index,
)
from service.web_document import index_document

URL = "https://official.example.cn/report"
ORIGINAL = "来源专用原句：2025年收入100亿元，仅覆盖国内业务。"
PARAGRAPH = f"国内业务2025年收入100亿元。[年度报告]({URL})"
SUMMARY = f"2025年国内收入为100亿元。[年度报告]({URL})"


def fixture():
    state = create_initial_state("研究国内业务收入及风险", "offline-session", ["official.example.cn"])
    state.update(
        original_query="请新增安全边界，保留其他章节，正文不超过1200字。",
        revision_instructions=["区分明确要求与建议", "证据不足标明待核实"],
        active_constraints={"max_body_chars": 1200},
        outline=[{"id": "sec_1", "title": "收入", "status": "drafted"}],
        draft_sections={"sec_1": PARAGRAPH},
        research_contract={"language": "中文", "output_format": "工程说明",
                           "active_constraints": {"max_body_chars": 2400},
                           "explicit_requirements": [{"original_requirement": "保留其他章节"}],
                           "unused_material": "旧契约素材不得注入"},
    )
    source = {"id": "source_1", "url": URL, "title": "年度报告", "content": ORIGINAL,
              "body_quality": {"status": "passed"}, **index_document(ORIGINAL, URL)}
    state["raw_sources"] = [source]
    evidence = {"document_id": source["document_id"], "sentence_ids": ["s000001"]}
    state["facts"] = [{"id": "fact_1", "source_id": "source_1", "content": "国内收入100亿元",
                       **evidence}]
    state["section_claims"] = {"sec_1": [{"claim": "国内收入100亿元", "report_quote": PARAGRAPH,
                                           "evidence": [evidence]}]}
    state["hypotheses"] = [{"id": "h_1", "content": "假设专用文本不得注入"}]
    state["conclusion_evidence_map"] = [{"hypothesis_ids": ["h_1"]}]
    return state


def synthesis_claim(**changes):
    return {"claim": "2025年国内收入为100亿元", "report_quote": SUMMARY,
            "source_quotes": [{"section_id": "sec_1", "quote": PARAGRAPH}], **changes}


def writer_fixture():
    writer = LeadWriter.__new__(LeadWriter)
    writer.name = "LeadWriter"
    writer.logger = logging.getLogger("test.writer.synthesis")
    writer.call_llm = AsyncMock(return_value=json.dumps({
        "executive_summary": SUMMARY, "conclusions": [{"text": "其他地区情况仍不确定。"}],
        "outlook": "后续需要补充证据。", "claims": [synthesis_claim(source_claim_ids=["C1"])],
    }, ensure_ascii=False))
    writer.parse_json_response = json.loads
    return writer


class SynthesisInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_input_preserves_current_requirements_without_sources_or_ledger(self):
        state, writer = fixture(), writer_fixture()
        before = copy.deepcopy(state)
        with patch("service.deep_research_v2.writer_claims.format_claim_evidence", side_effect=AssertionError("不能全量取原文")), \
                patch.object(writer, "_format_hypothesis_ledger", side_effect=AssertionError("不能输入账本")):
            await writer._synthesize_report(state)
        prompt = writer.call_llm.call_args.kwargs["user_prompt"]
        for text in (state["query"], state["original_query"], PARAGRAPH, "sec_1", "1200",
                     "区分明确要求与建议", "证据不足标明待核实", "工程说明", '"source_claim_ids"'):
            self.assertIn(text, prompt)
        for text in (ORIGINAL, "假设专用文本", "旧契约素材", "2400", "可逐字引用的原文片段", '"sentence_ids":'):
            self.assertNotIn(text, prompt)
        self.assertEqual(state["draft_sections"], before["draft_sections"])
        self.assertEqual(state["raw_sources"], before["raw_sources"])
        self.assertEqual(state["hypotheses"], before["hypotheses"])
        self.assertEqual(state["conclusion_evidence_map"], [])
        self.assertEqual(state["synthesis_claims"][0]["binding_errors"], [])
        self.assertEqual(state["synthesis_claims"][0]["evidence"], state["section_claims"]["sec_1"][0]["evidence"])
        self.assertIn(PARAGRAPH, state["final_report"])
        self.assertEqual(writer.call_llm.await_count, 1)

    async def test_input_size_does_not_grow_with_unused_evidence(self):
        first, second = fixture(), fixture()
        second["raw_sources"].extend([{"id": f"unused_{i}", "content": "大量无关来源原文" * 5000} for i in range(30)])
        second["facts"].extend([{"id": f"unused_fact_{i}", "content": "无关提取事实" * 1000} for i in range(30)])
        a, b = writer_fixture(), writer_fixture()
        await a._synthesize_report(first)
        await b._synthesize_report(second)
        self.assertEqual(a.call_llm.call_args.kwargs["user_prompt"], b.call_llm.call_args.kwargs["user_prompt"])


class SectionBindingTests(unittest.TestCase):
    def bind(self, state=None, value=None):
        return bind_synthesis_claims([value if value is not None else synthesis_claim()],
                                     state if state is not None else fixture(), [SUMMARY])[0]

    def test_model_supplied_sentence_ids_are_ignored_and_bindings_are_copied(self):
        state = fixture()
        result = self.bind(state, synthesis_claim(evidence=[{"document_id": "invented", "sentence_ids": ["bad"]}]))
        self.assertEqual(result["binding_errors"], [])
        self.assertEqual(result["source_claim_refs"], [{"section_id": "sec_1", "claim_index": 0}])
        self.assertEqual(result["evidence"], state["section_claims"]["sec_1"][0]["evidence"])
        result["evidence"][0]["sentence_ids"].append("mutation")
        self.assertEqual(state["section_claims"]["sec_1"][0]["evidence"][0]["sentence_ids"], ["s000001"])

    def test_multi_section_basis_keeps_all_evidence_and_deduplicates(self):
        state = fixture()
        extra = {"claim": "其他事实", "report_quote": "另一章原段落。",
                 "evidence": [{"document_id": "doc_2", "sentence_ids": ["s000002"]}]}
        state["outline"].append({"id": "sec_2"})
        state["draft_sections"]["sec_2"] = extra["report_quote"]
        state["section_claims"]["sec_2"] = [extra]
        anchors = [{"section_id": "sec_1", "quote": PARAGRAPH}] * 2 + [{"section_id": "sec_2", "quote": extra["report_quote"]}]
        result = self.bind(state, synthesis_claim(source_quotes=anchors))
        self.assertEqual(result["binding_errors"], [])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(len(result["source_claim_refs"]), 2)

    def test_fabricated_unknown_removed_or_partial_basis_is_not_accepted(self):
        invalids = [None, [], [None], [{"section_id": "unknown", "quote": PARAGRAPH}],
                    [{"section_id": "sec_1", "quote": PARAGRAPH.replace("100", "200")}],
                    [{"section_id": "sec_1", "quote": "100"}]]
        for anchors in invalids:
            with self.subTest(anchors=anchors):
                result = self.bind(value=synthesis_claim(source_quotes=anchors))
                self.assertTrue(result["binding_errors"])
                self.assertEqual(result["evidence"], [])
        state = fixture()
        state["outline"] = []
        self.assertEqual(self.bind(state)["evidence"], [])

    def test_missing_claim_or_evidence_is_reported_without_guessing(self):
        for claims in ([], [{"claim": "未出现在段落的论断", "evidence": []}],
                       [{"claim": "国内收入100亿元", "report_quote": PARAGRAPH, "evidence": []}]):
            state = fixture()
            state["section_claims"]["sec_1"] = claims
            result = self.bind(state)
            self.assertTrue(result["binding_errors"])
            self.assertEqual(result["evidence"], [])

    def test_legacy_claim_requires_literal_body_location(self):
        state = fixture()
        original = state["section_claims"]["sec_1"][0]
        original.pop("report_quote")
        original["claim"] = "国内业务2025年收入100亿元"
        self.assertEqual(self.bind(state)["binding_errors"], [])
        original["claim"] = "国内业务实现百亿收入"
        self.assertTrue(self.bind(state)["binding_errors"])

    def test_source_location_and_existing_errors_remain_auditable(self):
        state = fixture()
        state["section_claims"]["sec_1"][0]["binding_errors"] = ["existing_error"]
        result = self.bind(state, synthesis_claim(report_quote="未生成的段落"))
        self.assertIn("synthesis_report_quote_not_in_output", result["binding_errors"])
        self.assertIn("existing_error", result["binding_errors"])


class ClaimIndexTests(unittest.TestCase):
    def test_index_only_contains_declarations_locatable_in_current_chapters(self):
        state = fixture()
        state["section_claims"]["sec_1"].append({"claim": "未出现在正文的声明", "evidence": []})
        state["section_claims"]["removed"] = copy.deepcopy(state["section_claims"]["sec_1"])
        state["synthesis_claims"] = [{"claim": "旧摘要声明", "evidence": []}]
        index = build_synthesis_claim_index(state)
        self.assertEqual(list(index), ["C1"])
        self.assertEqual(index["C1"]["section_id"], "sec_1")
        self.assertEqual(index["C1"]["claim_index"], 0)

    def test_shared_paragraph_does_not_attach_other_claims_evidence(self):
        state = fixture()
        other = {"claim": "另一论断", "report_quote": PARAGRAPH,
                 "evidence": [{"document_id": "other_doc", "sentence_ids": ["s000002"]}]}
        state["section_claims"]["sec_1"].append(other)
        result = bind_synthesis_claims([synthesis_claim(source_claim_ids=["C1"])], state, [SUMMARY])[0]
        self.assertEqual(result["evidence"], state["section_claims"]["sec_1"][0]["evidence"])
        self.assertEqual(result["source_claim_refs"], [{"section_id": "sec_1", "claim_index": 0}])

    def test_explicit_bad_ids_never_fall_back_to_valid_source_quotes(self):
        for ids in (None, [], "C1", [None], ["unknown"]):
            with self.subTest(ids=ids):
                result = bind_synthesis_claims([synthesis_claim(source_claim_ids=ids)], fixture(), [SUMMARY])[0]
                self.assertEqual(result["evidence"], [])
                self.assertTrue(result["binding_errors"])

    def test_multi_claim_ids_preserve_distinct_evidence_and_deduplicate(self):
        state = fixture()
        original = copy.deepcopy(state["section_claims"]["sec_1"][0])
        original["evidence"] = [{"document_id": "other_doc", "sentence_ids": ["s000002"]}]
        state["section_claims"]["sec_1"].append(original)
        result = bind_synthesis_claims([synthesis_claim(source_claim_ids=["C1", "C2", "C1"])], state, [SUMMARY])[0]
        self.assertEqual(result["source_claim_ids"], ["C1", "C2"])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(result["binding_errors"], [])

    def test_current_call_index_is_a_snapshot_and_does_not_mutate_declarations(self):
        state = fixture()
        index = build_synthesis_claim_index(state)
        state["section_claims"]["sec_1"][0]["evidence"][0]["sentence_ids"] = ["changed"]
        result = bind_synthesis_claims([synthesis_claim(source_claim_ids=["C1"])], state, [SUMMARY], claim_index=index)[0]
        self.assertEqual(result["evidence"][0]["sentence_ids"], ["s000001"])
        result["evidence"][0]["sentence_ids"].append("mutation")
        self.assertEqual(index["C1"]["declaration"]["evidence"][0]["sentence_ids"], ["s000001"])

    def test_index_carries_existing_binding_failures_and_missing_evidence(self):
        state = fixture()
        original = state["section_claims"]["sec_1"][0]
        original["binding_errors"] = ["existing_error"]
        original["evidence"] = []
        result = bind_synthesis_claims([synthesis_claim(source_claim_ids=["C1"])], state, [SUMMARY])[0]
        self.assertIn("existing_error", result["binding_errors"])
        self.assertIn("synthesis_source_claim_without_evidence", result["binding_errors"])

    def test_model_index_only_exposes_claim_text_section_and_local_id(self):
        packet = json.loads(format_synthesis_claim_index(build_synthesis_claim_index(fixture())))
        self.assertEqual(set(packet[0]), {"id", "section_id", "claim"})
        self.assertNotIn(ORIGINAL, json.dumps(packet, ensure_ascii=False))
        self.assertNotIn("sentence_ids", json.dumps(packet))


class VerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_verifier_reads_original_from_inherited_ids(self):
        state = fixture()
        state["synthesis_claims"] = bind_synthesis_claims([synthesis_claim(source_claim_ids=["C1"])], state, [SUMMARY])
        state["claims"] = collect_report_claims(state)
        resolved = _claims_for_stage(state, "report_claims")[-1]
        self.assertEqual(resolved["binding_errors"], [])
        self.assertEqual(resolved["references"][0]["quote"], ORIGINAL)
        self.assertNotIn("quote", state["synthesis_claims"][0]["evidence"][0])

    async def test_invalid_section_binding_cannot_pass_even_if_nli_says_supported(self):
        state = fixture()
        state["synthesis_claims"] = bind_synthesis_claims(
            [synthesis_claim(report_quote="伪造定位", source_claim_ids=["C1"])], state, [SUMMARY])
        state["claims"] = [{**state["synthesis_claims"][0], "section_id": "__synthesis__"}]
        state["citation_verification_config"] = {"report_claims": True}
        nli = AsyncMock(return_value={"execution_status": "completed", "claims": [
            {"status": "supported", "needs_review": False, "evidence": [{}]}]})
        with patch("service.nli_verifier.verify_claims", nli):
            result = await verify_citations(_claims_for_stage(state, "report_claims"), enabled=True)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["claims"][0]["status"], "invalid_citation")


if __name__ == "__main__":
    unittest.main()
