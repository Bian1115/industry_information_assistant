"""首尾通过本次章节论断编号继承证据；模型与网络均使用离线替身。"""

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from backend.test.test_Writer首尾章节整合 import (
    ORIGINAL, PARAGRAPH, SUMMARY, fixture, writer_fixture,
)
from service.deep_research_v2.citation_verification import _claims_for_stage, verify_citations
from service.deep_research_v2.writer_claims import (
    bind_synthesis_claims, build_synthesis_claim_index, collect_report_claims,
    format_synthesis_claim_index,
)


def numbered_claim(**changes):
    return {"claim": "2025年国内收入为100亿元", "report_quote": SUMMARY,
            "source_claim_ids": ["C1"], **changes}


class ClaimIndexTests(unittest.TestCase):
    def test_index_is_compact_and_does_not_mutate_or_expose_originals(self):
        state = fixture()
        before = copy.deepcopy(state)
        index = build_synthesis_claim_index(state)
        packet = json.loads(format_synthesis_claim_index(index))
        self.assertEqual(packet, [{"id": "C1", "section_id": "sec_1", "claim": "国内收入100亿元"}])
        text = json.dumps(packet, ensure_ascii=False)
        for field in ("document_id", "sentence_ids", "report_quote", ORIGINAL):
            self.assertNotIn(field, text)
        self.assertEqual(state, before)
        index["C1"]["declaration"]["evidence"][0]["sentence_ids"].append("changed")
        self.assertEqual(state, before)

    def test_removed_empty_stale_and_unlocatable_claims_are_not_indexed(self):
        state = fixture()
        state["outline"].extend([{"id": "empty"}, {"id": "stale"}])
        state["draft_sections"].update(empty="", stale="新的正文")
        state["section_claims"].update(
            empty=copy.deepcopy(state["section_claims"]["sec_1"]),
            stale=copy.deepcopy(state["section_claims"]["sec_1"]),
            removed=copy.deepcopy(state["section_claims"]["sec_1"]),
        )
        state["section_claims"]["sec_1"].extend([None, {}, {"claim": "不存在的事实", "evidence": []}])
        index = build_synthesis_claim_index(state)
        self.assertEqual(list(index), ["C1"])
        self.assertEqual(index["C1"]["claim_index"], 0)

    def test_legacy_claim_without_location_requires_literal_match(self):
        state = fixture()
        original = state["section_claims"]["sec_1"][0]
        original.pop("report_quote")
        original["claim"] = "国内业务2025年收入100亿元"
        self.assertEqual(list(build_synthesis_claim_index(state)), ["C1"])
        original["claim"] = "国内业务实现百亿收入"
        self.assertEqual(build_synthesis_claim_index(state), {})


class NumberedBindingTests(unittest.TestCase):
    def bind(self, state=None, value=None, **kwargs):
        return bind_synthesis_claims(
            [numbered_claim() if value is None else value],
            fixture() if state is None else state, [SUMMARY], **kwargs,
        )[0]

    def test_selected_claim_does_not_inherit_other_claims_in_same_paragraph(self):
        state = fixture()
        state["section_claims"]["sec_1"].append({
            "claim": "其他地区收入情况", "report_quote": PARAGRAPH,
            "evidence": [{"document_id": "other_document", "sentence_ids": ["s000009"]}],
        })
        result = self.bind(state, numbered_claim(evidence=[{"document_id": "invented", "sentence_ids": ["bad"]}]))
        self.assertEqual(result["source_claim_ids"], ["C1"])
        self.assertEqual(result["source_claim_refs"], [{"section_id": "sec_1", "claim_index": 0}])
        self.assertEqual(result["evidence"], state["section_claims"]["sec_1"][0]["evidence"])
        self.assertEqual(result["binding_errors"], [])
        result["evidence"][0]["sentence_ids"].append("mutated")
        self.assertEqual(state["section_claims"]["sec_1"][0]["evidence"][0]["sentence_ids"], ["s000001"])

    def test_multiple_claims_keep_all_evidence_and_deduplicate_repeats(self):
        state = fixture()
        second = {"claim": "另一章的事实", "report_quote": "另一章的事实。",
                  "evidence": [{"document_id": "doc_2", "sentence_ids": ["s000002"]}]}
        state["outline"].append({"id": "sec_2"})
        state["draft_sections"]["sec_2"] = second["report_quote"]
        state["section_claims"]["sec_2"] = [second]
        result = self.bind(state, numbered_claim(source_claim_ids=["C1", "C2", "C1"]))
        self.assertEqual(result["source_claim_ids"], ["C1", "C2"])
        self.assertEqual(len(result["source_claim_refs"]), 2)
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(result["binding_errors"], [])
        second["evidence"] = copy.deepcopy(state["section_claims"]["sec_1"][0]["evidence"])
        self.assertEqual(len(self.bind(state, numbered_claim(source_claim_ids=["C1", "C2"]))["evidence"]), 1)

    def test_malformed_or_unknown_ids_are_audited_without_guessing(self):
        for ids in (None, [], "C1", [None], [{}], [1], ["unknown"]):
            with self.subTest(ids=ids):
                result = self.bind(value=numbered_claim(source_claim_ids=ids))
                self.assertTrue(result["binding_errors"])
                self.assertEqual(result["evidence"], [])
        partial = self.bind(value=numbered_claim(source_claim_ids=["C1", "unknown"]))
        self.assertIn("unknown_synthesis_source_claim_id", partial["binding_errors"])
        self.assertEqual(partial["source_claim_ids"], ["C1"])

    def test_empty_ids_do_not_fall_back_to_legacy_paragraph_or_model_evidence(self):
        result = self.bind(value=numbered_claim(
            source_claim_ids=[], source_quotes=[{"section_id": "sec_1", "quote": PARAGRAPH}],
            evidence=[{"document_id": "invented", "sentence_ids": ["bad"]}],
        ))
        self.assertEqual(result["evidence"], [])
        self.assertIn("missing_synthesis_source_claim_ids", result["binding_errors"])

    def test_missing_evidence_existing_errors_and_invalid_output_location_are_preserved(self):
        state = fixture()
        original = state["section_claims"]["sec_1"][0]
        original.update(evidence=[], binding_errors=["existing_error"])
        result = self.bind(state, numbered_claim(report_quote="不在本次输出"))
        for error in ("synthesis_source_claim_without_evidence", "existing_error", "synthesis_report_quote_not_in_output"):
            self.assertIn(error, result["binding_errors"])
        self.assertEqual(result["evidence"], [])

    def test_binding_uses_same_snapshot_as_prompt(self):
        state = fixture()
        snapshot = build_synthesis_claim_index(state)
        original_evidence = copy.deepcopy(snapshot["C1"]["declaration"]["evidence"])
        state["section_claims"]["sec_1"][0]["evidence"] = []
        result = self.bind(state, claim_index=snapshot)
        self.assertEqual(result["evidence"], original_evidence)
        self.assertEqual(result["binding_errors"], [])


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_writer_sends_index_and_inherits_ids_without_reading_all_sources(self):
        state, writer = fixture(), writer_fixture()
        writer.call_llm = AsyncMock(return_value=json.dumps({
            "executive_summary": SUMMARY, "conclusions": [], "outlook": "后续需要补充证据。",
            "claims": [numbered_claim()],
        }, ensure_ascii=False))
        with patch("service.deep_research_v2.writer_claims.format_claim_evidence", side_effect=AssertionError("不能全量读取原文")):
            await writer._synthesize_report(state)
        prompt = writer.call_llm.call_args.kwargs["user_prompt"]
        self.assertIn(format_synthesis_claim_index(build_synthesis_claim_index(state)), prompt)
        self.assertIn('"source_claim_ids"', prompt)
        self.assertNotIn('"source_quotes"', prompt)
        self.assertNotIn(ORIGINAL, prompt)
        self.assertNotIn('"sentence_ids":', prompt)
        self.assertEqual(state["synthesis_claims"][0]["source_claim_ids"], ["C1"])
        self.assertEqual(state["synthesis_claims"][0]["binding_errors"], [])
        self.assertEqual(writer.call_llm.await_count, 1)

    async def test_verifier_reads_inherited_original_sentences(self):
        state = fixture()
        state["synthesis_claims"] = bind_synthesis_claims([numbered_claim()], state, [SUMMARY])
        state["claims"] = collect_report_claims(state)
        resolved = _claims_for_stage(state, "report_claims")[-1]
        self.assertEqual(resolved["references"][0]["quote"], ORIGINAL)
        self.assertEqual(resolved["binding_errors"], [])

    async def test_inherited_evidence_does_not_bypass_support_verification(self):
        state = fixture()
        state["synthesis_claims"] = bind_synthesis_claims([numbered_claim()], state, [SUMMARY])
        state["claims"] = [{**state["synthesis_claims"][0], "section_id": "__synthesis__"}]
        state["citation_verification_config"] = {"report_claims": True}
        nli = AsyncMock(return_value={"execution_status": "completed", "claims": [
            {"status": "unsupported", "needs_review": True, "evidence": [{}]}]})
        with patch("service.nli_verifier.verify_claims", nli):
            result = await verify_citations(_claims_for_stage(state, "report_claims"), enabled=True)
        self.assertEqual(result["status"], "rejected")
        self.assertNotEqual(result["claims"][0]["status"], "supported")

    async def test_invalid_source_id_cannot_pass_even_if_nli_says_supported(self):
        state = fixture()
        state["synthesis_claims"] = bind_synthesis_claims(
            [numbered_claim(source_claim_ids=["C1", "unknown"])], state, [SUMMARY])
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
