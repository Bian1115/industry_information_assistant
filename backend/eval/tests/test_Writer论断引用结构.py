"""Writer 引用声明的离线契约与生命周期回归；不调用模型或网络。"""

import json
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

APP_DIR = Path(__file__).resolve().parents[2] / "app"
sys.path.insert(0, str(APP_DIR))

from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.delivery import build_complete_event
from service.deep_research_v2.graph import pack_state
from service.deep_research_v2.state import create_initial_state, reopen_sections_for_rewrite
from service.web_document import index_document
from service.deep_research_v2.citation_verification import _claims_for_stage, verify_stage_citations
from service.deep_research_v2.writer_claims import (
    collect_report_claims, format_claim_evidence, normalize_claims,
)


URL = "https://official.example.cn/report"
ORIGINAL = "2025年，公司实现营业收入100亿元，同比增长12%。"
DOCUMENT = index_document(ORIGINAL, URL)["document_id"]


def claim(text="公司2025年营业收入为100亿元", document_id=DOCUMENT, sentence_ids=None):
    return {"claim": text, "evidence": [{"document_id": document_id,
        "sentence_ids": sentence_ids if sentence_ids is not None else ["s000001"]}]}


def make_state():
    state = create_initial_state("分析公司表现", "test-session", ["official.example.cn"])
    state["outline"] = [
        {"id": "sec_1", "title": "收入", "status": "pending"},
        {"id": "sec_2", "title": "风险", "status": "pending"},
    ]
    state["raw_sources"] = [{
        "id": "source_001", "url": URL, "title": "年度报告", "content": ORIGINAL,
        "body_quality": {"status": "passed"}, **index_document(ORIGINAL, URL),
    }]
    state["facts"] = [{
        "id": "fact_001", "source_id": "source_001",
        "content": "这是提炼后的事实摘要，不能当作来源原句",
        "document_id": DOCUMENT, "sentence_ids": ["s000001"],
        "related_sections": ["sec_1"],
    }]
    return state


def make_writer(*responses):
    writer = LeadWriter.__new__(LeadWriter)
    writer.name = "LeadWriter"
    writer.logger = logging.getLogger("test.writer.claims")
    writer.call_llm = AsyncMock(side_effect=[json.dumps(r, ensure_ascii=False) for r in responses])
    writer.parse_json_response = json.loads
    return writer


def synthesis(claims=None):
    return {
        "executive_summary": "公司2025年营业收入为100亿元。",
        "conclusions": [{"id": "c_1", "text": "公司营业收入同比增长12%。"}],
        "outlook": "后续情况尚不确定。", "references": [],
        "claims": claims if claims is not None else [claim(), claim("公司营业收入同比增长12%")],
    }


class ClaimContractTests(unittest.TestCase):
    def test_original_source_and_fact_id_are_supplied_without_summary_as_quote(self):
        state = make_state()
        state["facts"].append({**state["facts"][0], "id": "fact_002"})
        packet = json.loads(format_claim_evidence(state))
        self.assertEqual(len(packet["facts"]), 2)
        self.assertEqual([f["evidence_id"] for f in packet["facts"]], ["fact_001", "fact_002"])
        self.assertEqual(len(packet["documents"]), 1)
        self.assertEqual(packet["documents"][0]["original_sentences"], [{"id": "s000001", "text": ORIGINAL}])
        self.assertNotIn("original_excerpt", json.dumps(packet))
        selected = json.loads(format_claim_evidence(state, ["fact_002"]))
        self.assertEqual([f["evidence_id"] for f in selected["facts"]], ["fact_002"])

    def test_missing_original_is_not_replaced_with_model_fact(self):
        state = make_state()
        state["raw_sources"] = []
        self.assertEqual(json.loads(format_claim_evidence(state)), {"facts": [], "documents": []})

    def test_unsupported_claims_and_questionable_evidence_are_preserved_for_verifier(self):
        suspicious = claim(document_id="invented_id", sentence_ids=["invented_sentence"])
        values = normalize_claims([{"claim": "没有来源的论断", "evidence": []}, suspicious])
        self.assertEqual(values[0]["evidence"], [])
        self.assertEqual(values[1], suspicious)
        self.assertNotIn("verified", values[1])

    def test_legacy_and_malformed_values_do_not_break_existing_reports(self):
        self.assertEqual(normalize_claims(None), [])
        self.assertEqual(normalize_claims({"claim": "x"}), [])
        self.assertEqual(normalize_claims([None, {}, {"claim": " "}]), [])
        self.assertEqual(normalize_claims([{"claim": "x", "evidence": "wrong"}]),
                         [{"claim": "x", "evidence": []}])

    def test_report_claims_are_copied_and_ignore_removed_drafts(self):
        state = make_state()
        state["draft_sections"] = {"sec_1": "正文"}
        state["section_claims"] = {"sec_1": [claim()], "sec_2": [claim("旧草稿")], "gone": [claim()]}
        result = collect_report_claims(state)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["section_id"], "sec_1")
        result[0]["evidence"][0]["sentence_ids"].append("修改副本")
        self.assertEqual(state["section_claims"]["sec_1"][0]["evidence"][0]["sentence_ids"], ["s000001"])

    def test_new_outputs_discard_quote_and_keep_invalid_ids_for_verification(self):
        value = claim()
        value["evidence"][0].update(quote="模型重复输出的原句", evidence_id="fact_001")
        self.assertEqual(normalize_claims([value]), [claim()])
        invalid = claim(sentence_ids="s000001")
        self.assertEqual(normalize_claims([invalid]), [invalid])
        legacy = {"claim": "旧输出", "evidence": [{"evidence_id": "fact_001", "quote": ORIGINAL}]}
        self.assertEqual(normalize_claims([legacy])[0]["evidence"],
                         [{"document_id": "", "sentence_ids": []}])

    def test_reopen_clears_affected_claims_and_stale_synthesis(self):
        state = make_state()
        state["draft_sections"] = {"sec_1": "旧正文", "sec_2": "保留正文"}
        state["section_claims"] = {"sec_1": [claim()], "sec_2": [claim("保留") ]}
        state["synthesis_claims"] = [claim("旧结论")]
        state["claims"] = collect_report_claims(state)
        reopen_sections_for_rewrite(state, ["sec_1"])
        self.assertNotIn("sec_1", state["section_claims"])
        self.assertEqual(state["section_claims"]["sec_2"][0]["claim"], "保留")
        self.assertEqual(state["synthesis_claims"], [])
        self.assertEqual(state["claims"], [])


class WriterLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_section_synthesis_events_and_checkpoint_preserve_claims(self):
        state = make_state()
        emitted = []
        state["_emit_event"] = emitted.append
        writer = make_writer({
            "content": f"收入为100亿元。[年度报告]({URL})", "key_points": [],
            "citations": [{"source": "年度报告", "url": URL}], "claims": [claim()],
        }, synthesis())
        await writer._write_section(state, state["outline"][0])
        await writer._synthesize_report(state)
        self.assertEqual(state["section_claims"]["sec_1"], [claim()])
        self.assertEqual(len(state["claims"]), 3)
        self.assertEqual(state["claims"][-1]["section_id"], "__synthesis__")
        for call in writer.call_llm.call_args_list:
            self.assertEqual(call.kwargs["max_tokens"], writer.max_tokens)
        section_prompt, synthesis_prompt = [call.kwargs["user_prompt"] for call in writer.call_llm.call_args_list]
        self.assertIn(ORIGINAL, section_prompt)
        self.assertIn("fact_001", section_prompt)
        self.assertIn('"sentence_ids"', section_prompt)
        self.assertNotIn(ORIGINAL, synthesis_prompt)
        self.assertNotIn("fact_001", synthesis_prompt)
        self.assertIn('"source_claim_ids"', synthesis_prompt)
        for kind in ("section_content", "report_draft"):
            event = next(event for event in emitted if event["type"] == kind)
            self.assertTrue(event["content"]["claims"])
        saved = json.loads(json.dumps(pack_state(state), ensure_ascii=False))
        self.assertEqual(saved["research"]["claims"], state["claims"])
        complete = build_complete_event(state, run_id="run", checkpoint_id="checkpoint")
        self.assertEqual(complete["claims"], state["claims"])
        self.assertEqual(complete["citation_verification"]["status"], "not_executed")

    async def test_revision_replaces_target_claims_and_preserves_other_sections(self):
        state = make_state()
        state["draft_sections"] = {"sec_1": "旧正文" * 150, "sec_2": "保留正文" * 100}
        state["section_claims"] = {"sec_1": [claim("过期论断")], "sec_2": [claim("保留论断")]}
        state["synthesis_claims"] = [claim("旧摘要")]
        state["critic_feedback"] = [{"issue_id": "issue_1", "resolved_section_ids": ["sec_1"], "target": {"chapter_id": "sec_1"}, "description": "修订本章", "evidence": "核对正文", "modification": "按已有材料修订本章", "requires_new_search": False}]
        writer = make_writer({
            "revised_content": "修订后的完整章节正文。" * 100,
            "claims": [claim("修订论断")], "addressed_issues": ["issue_1"],
            "unable_to_address": [], "citations": [],
        }, synthesis([claim("新摘要")]))
        await writer._revise_report(state)
        self.assertEqual([item["claim"] for item in state["claims"]], ["修订论断", "保留论断", "新摘要"])

    async def test_rejected_revision_does_not_replace_claims(self):
        state = make_state()
        state["draft_sections"] = {"sec_1": "旧正文" * 150}
        state["section_claims"] = {"sec_1": [claim("原论断")]}
        state["critic_feedback"] = [{"issue_id": "i", "resolved_section_ids": ["sec_1"], "target": {"chapter_id": "sec_1"}, "description": "修订本章", "evidence": "核对正文", "modification": "按已有材料修订本章", "requires_new_search": False}]
        writer = make_writer({"revised_content": "太短", "claims": [claim("错误替换")]})
        await writer._revise_report(state)
        self.assertEqual(state["section_claims"]["sec_1"][0]["claim"], "原论断")

    async def test_failed_synthesis_drops_old_claims_but_keeps_section_claims(self):
        state = make_state()
        state["draft_sections"] = {"sec_1": "正文"}
        state["section_claims"] = {"sec_1": [claim()]}
        state["synthesis_claims"] = [claim("旧摘要")]
        await make_writer({})._synthesize_report(state)
        self.assertEqual(state["synthesis_claims"], [])
        self.assertEqual(len(state["claims"]), 1)

    async def test_state_endpoint_exposes_claims_after_reload(self):
        from router.research_router import get_run_state
        state = make_state()
        state["claims"] = [{**claim(), "section_id": "sec_1"}]
        service = SimpleNamespace(read_state=AsyncMock(return_value={"research": state}))
        ident = UUID("00000000-0000-0000-0000-000000000001")
        with patch("router.research_router._owned_run"), patch(
            "router.research_router.get_research_service", return_value=service,
        ):
            result = await get_run_state(ident, SimpleNamespace(), SimpleNamespace(id=ident), object())
        self.assertEqual(result["state"]["draft"]["claims"], state["claims"])


class NumberedCitationTests(unittest.IsolatedAsyncioTestCase):
    def report(self, citation=None):
        state = make_state()
        state["claims"] = [{**(citation or claim()), "section_id": "sec_1"}]
        return state

    def test_program_reads_original_without_writer_quote_or_fact_id(self):
        state = self.report()
        result = _claims_for_stage(state, "report_claims")[0]
        self.assertEqual(result["binding_errors"], [])
        self.assertEqual(result["references"][0]["quote"], ORIGINAL)
        self.assertEqual(result["references"][0]["sentence_ids"], ["s000001"])
        self.assertNotIn("quote", state["claims"][0]["evidence"][0])

    def test_unknown_missing_duplicate_and_malformed_ids_cannot_resolve(self):
        invalids = [claim(document_id="unknown"), claim(document_id=""),
                    claim(sentence_ids=[]), claim(sentence_ids=["s999999"]),
                    claim(sentence_ids=["s000001", "s000001"]), claim(sentence_ids="s000001")]
        for citation in invalids:
            with self.subTest(citation=citation):
                result = _claims_for_stage(self.report(citation), "report_claims")[0]
                self.assertTrue(result["binding_errors"])
                self.assertEqual(result["references"], [])

    def test_modified_or_ambiguous_document_cannot_supply_original(self):
        for mutation in ("content", "index", "duplicate", "quality"):
            state = self.report()
            source = state["raw_sources"][0]
            if mutation == "content":
                source["content"] += "修改。"
            elif mutation == "index":
                source["sentence_index"][0]["end"] -= 1
            elif mutation == "duplicate":
                state["raw_sources"].append({**source, "id": "source_other"})
            else:
                source["body_quality"]["status"] = "rejected"
            with self.subTest(mutation=mutation):
                result = _claims_for_stage(state, "report_claims")[0]
                self.assertTrue(result["binding_errors"])
                self.assertEqual(result["references"], [])

    def test_selected_sentences_do_not_bridge_unselected_original(self):
        body = "项目于2023年投运。\n仅覆盖城区。\n服务人口6000人。"
        doc = index_document(body, URL)
        state = self.report(claim(document_id=doc["document_id"], sentence_ids=["s000003", "s000001"]))
        state["raw_sources"][0].update(content=body, **doc)
        references = _claims_for_stage(state, "report_claims")[0]["references"]
        self.assertEqual([r["quote"] for r in references], ["项目于2023年投运。", "服务人口6000人。"])
        state["claims"][0]["evidence"][0]["sentence_ids"] = ["s000001", "s000002"]
        references = _claims_for_stage(state, "report_claims")[0]["references"]
        self.assertEqual(references[0]["quote"], "项目于2023年投运。\n仅覆盖城区。")

    def test_multiple_documents_resolve_their_own_sentence_indices(self):
        state = self.report()
        other_body = "2025年员工人数为200人。"
        doc = index_document(other_body, URL + "2")
        state["raw_sources"].append({"id": "source_002", "url": URL + "2", "content": other_body,
                                    "body_quality": {"status": "passed"}, **doc})
        state["claims"][0]["evidence"].append({"document_id": doc["document_id"], "sentence_ids": ["s000001"]})
        references = _claims_for_stage(state, "report_claims")[0]["references"]
        self.assertEqual([r["quote"] for r in references], [ORIGINAL, other_body])

    def test_existing_legacy_checkpoint_can_still_be_verified(self):
        state = self.report({"claim": "公司2025年营业收入为100亿元",
                             "evidence": [{"evidence_id": "fact_001", "quote": ORIGINAL}]})
        result = _claims_for_stage(state, "report_claims")[0]
        self.assertEqual(result["binding_errors"], [])
        self.assertEqual(result["references"][0]["quote"], ORIGINAL)

    async def test_nli_receives_program_resolved_original_but_saved_audit_has_only_ids(self):
        state = self.report()
        state["citation_verification_config"] = {"report_claims": True}
        async def nli(claims, settings):
            self.assertEqual(claims[0]["references"][0]["quote"], ORIGINAL)
            return {"execution_status": "completed", "claims": [{"status": "supported",
                "evidence": [{"quote": ORIGINAL, "status": "scored", "scores": {"entailment": .99}}], "needs_review": False}]}
        extraction = {"status": "completed", "units_total": 1, "claims": [{
            "claim_id": "claim_1", "claim": "公司2025年营业收入为100亿元", "section_id": "sec_1",
            "assertion_kind": "fact", "report_quote": ORIGINAL, "citation_urls": [URL], "binding_errors": []}]}
        async def select(messages, settings):
            payload = json.loads(messages[1]["content"])
            return json.dumps({"claim_id": payload["claim_id"], "evidence": [
                {"source_id": block["source_id"], "sentence_ids": [s["sentence_id"] for s in block["sentences"]]}
                for block in payload["messages"]]}), {}
        with patch("service.deep_research_v2.report_claim_extraction.extract_report_claims", AsyncMock(return_value=extraction)), \
                patch("service.deep_research_v2.report_sentence_evidence.call_selector", side_effect=select), \
                patch("service.nli_verifier.verify_claims", AsyncMock(side_effect=nli)) as verifier:
            result = await verify_stage_citations(state, "report_claims")
        verifier.assert_awaited_once()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["claims"][0]["evidence"][0]["document_id"], DOCUMENT)
        self.assertNotIn("quote", result["claims"][0]["evidence"][0])
        self.assertNotIn("quote", state["claims"][0]["evidence"][0])

    async def test_disabled_verification_does_not_read_original_or_call_nli(self):
        state = self.report()
        with patch("service.deep_research_v2.citation_verification._claims_for_stage") as resolver, \
                patch("service.nli_verifier.verify_claims", AsyncMock()) as verifier:
            result = await verify_stage_citations(state, "report_claims")
        resolver.assert_not_called()
        verifier.assert_not_awaited()
        self.assertEqual(result["status"], "not_executed")


if __name__ == "__main__":
    unittest.main()
