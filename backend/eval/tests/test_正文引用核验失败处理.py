"""正文失败处理的语义边界、事务提交及失败不报错回归。"""
import copy
import json
import logging
import re
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from service.deep_research_v2.citation_verification import CitationVerificationOptions, verify_citations
from service.deep_research_v2.citation_repair import digest, locate_claims, parse_edits, repair_report_citations
from service.deep_research_v2.writer_claims import collect_report_claims


def assemble(state):
    p = state.get("report_synthesis") or {}
    return "\n\n".join([*state["draft_sections"].values(), p.get("executive_summary", ""),
                        *p.get("conclusions", []), p.get("outlook", "")]).strip()


async def verify(state, stage):
    rows = [{"claim": c["claim"], "status": "unknown" if "失败" in c["claim"] else "supported",
             "execution_status": "completed"} for c in state["claims"]]
    result = {"status": "rejected" if any(r["status"] == "unknown" for r in rows) else "passed",
              "execution_status": "completed", "claims": rows,
              "report_sha256": digest(state["final_report"])}
    state["citation_verification"] = result
    return result


async def fixture(action):
    ref = {"document_id": "doc", "sentence_ids": ["s000001"]}
    good = {"claim": "有证据事实。", "report_quote": "有证据事实。", "evidence": [ref]}
    bad = {"claim": "失败事实。", "report_quote": "失败事实。", "evidence": [ref]}
    state = {"citation_verification_config": {"report_claims": True, "report_failure_action": action},
             "outline": [{"id": "sec_1"}], "draft_sections": {"sec_1": "有证据事实。\n\n失败事实。"},
             "section_claims": {"sec_1": [good, bad]}, "synthesis_claims": [], "raw_sources": [],
             "report_synthesis": {"executive_summary": "", "conclusions": [], "outlook": ""}}
    state["claims"] = collect_report_claims(state)
    state["final_report"] = assemble(state)
    await verify(state, "report_claims")
    return state


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_rewriter_can_select_original_sentence_subset_but_cannot_add_ids(self):
        original = {"document_id": "doc", "sentence_ids": ["s000001", "s000002"]}
        packet = [{"unit_id": "u", "text": "原文。", "claims": [{"evidence": [original]}]}]
        response = {"edits": [{"unit_id": "u", "content": "改写事实。", "claims": [{"claim": "改写事实。",
            "report_quote": "改写事实。", "evidence": [{"document_id": "doc", "sentence_ids": ["s000002"]}]}]}]}
        self.assertEqual(len(parse_edits(response, packet)), 1)
        response["edits"][0]["claims"][0]["evidence"][0]["sentence_ids"] = ["s000003"]
        with self.assertRaises(ValueError):
            parse_edits(response, packet)

    def test_mutation_requires_enabled_verification(self):
        for action in ("delete", "rewrite_once"):
            with self.assertRaises(ValidationError):
                CitationVerificationOptions(report_failure_action=action)
        with self.assertRaises(ValidationError):
            CitationVerificationOptions(report_claims=True, report_failure_action="retry_forever")

    async def test_audit_never_edits_or_calls_rewriter(self):
        state = await fixture("audit")
        before = copy.deepcopy(state)
        call = AsyncMock()
        await repair_report_citations(state, call=call, assemble=assemble, verify=verify)
        self.assertEqual(state["final_report"], before["final_report"])
        call.assert_not_awaited()

    async def test_delete_updates_actual_body_and_claims(self):
        state = await fixture("delete")
        call = AsyncMock()
        await repair_report_citations(state, call=call, assemble=assemble, verify=verify)
        self.assertNotIn("失败事实", state["final_report"])
        self.assertEqual(len(state["claims"]), 1)
        self.assertEqual(state["citation_repair"]["before_claims"][1]["claim"], "失败事实。")
        self.assertEqual(state["citation_verification"]["report_sha256"], digest(state["final_report"]))
        call.assert_not_awaited()

    async def test_rewrite_once_rejected_content_is_recorded_and_delivered(self):
        state = await fixture("rewrite_once")
        call = AsyncMock(return_value=json.dumps({"edits": [{"unit_id": "sec_1:8", "content": "仍失败事实。",
            "claims": [{"claim": "仍失败事实。", "report_quote": "仍失败事实。",
                        "evidence": state["claims"][1]["evidence"]}]}]}, ensure_ascii=False))
        units, bindings, _ = locate_claims(state)
        uid = bindings[1]
        response = json.loads(call.return_value)
        response["edits"][0]["unit_id"] = uid
        call.return_value = json.dumps(response)
        result = await repair_report_citations(state, call=call, assemble=assemble, verify=verify)
        self.assertEqual(call.await_count, 1)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("仍失败事实", state["final_report"])
        self.assertTrue(state["citation_repair"]["remaining_failed_claims"])
        self.assertEqual(state["citation_repair"]["status"], "applied")

    async def test_incomplete_initial_audit_never_deletes(self):
        state = await fixture("delete")
        state["citation_verification"]["execution_status"] = "timeout"
        before = state["final_report"]
        await repair_report_citations(state, call=AsyncMock(), assemble=assemble, verify=verify)
        self.assertEqual(state["final_report"], before)
        self.assertEqual(state["citation_repair"]["status"], "skipped")

    async def test_stale_report_audit_never_deletes(self):
        state = await fixture("delete")
        state["final_report"] += "\n新内容"
        before = state["final_report"]
        await repair_report_citations(state, call=AsyncMock(), assemble=assemble, verify=verify)
        self.assertEqual(state["final_report"], before)

    async def test_unlocatable_claims_preserve_component(self):
        state = await fixture("delete")
        state["claims"][1]["report_quote"] = "改写的定位"
        await repair_report_citations(state, call=AsyncMock(), assemble=assemble, verify=verify)
        self.assertIn("失败事实", state["final_report"])
        self.assertEqual(state["citation_repair"]["status"], "unresolved")

    async def test_malformed_rewrite_is_failure_record_not_exception(self):
        state = await fixture("rewrite_once")
        before = state["final_report"]
        await repair_report_citations(state, call=AsyncMock(return_value="bad JSON"), assemble=assemble, verify=verify)
        self.assertEqual(state["final_report"], before)
        self.assertEqual(state["citation_repair"]["status"], "error")

    async def test_recheck_timeout_retains_original_report(self):
        state = await fixture("rewrite_once")
        _, bindings, _ = locate_claims(state)
        response = {"edits": [{"unit_id": bindings[1], "content": "修改事实。", "claims": [
            {"claim": "修改事实。", "report_quote": "修改事实。", "evidence": state["claims"][1]["evidence"]}]}]}
        async def timeout(proposal, stage):
            return {"status": "error", "execution_status": "timeout"}
        before = state["final_report"]
        await repair_report_citations(state, call=AsyncMock(return_value=response), assemble=assemble, verify=timeout)
        self.assertEqual(state["final_report"], before)
        self.assertEqual(state["citation_repair"]["status"], "skipped")

    async def test_local_rewrite_preserves_paragraph_separator_and_other_text(self):
        state = await fixture("rewrite_once")
        state["draft_sections"]["sec_1"] = "失败事实。\n\n有证据事实。"
        state["final_report"] = assemble(state)
        await verify(state, "report_claims")
        _, bindings, _ = locate_claims(state)
        response = {"edits": [{"unit_id": bindings[1], "content": "仍失败事实。", "claims": [
            {"claim": "仍失败事实。", "report_quote": "仍失败事实。", "evidence": state["claims"][1]["evidence"]}]}]}
        await repair_report_citations(state, call=AsyncMock(return_value=response), assemble=assemble, verify=verify)
        self.assertEqual(state["draft_sections"]["sec_1"], "仍失败事实。\n\n有证据事实。")

    async def test_declared_claims_are_batched_without_omission(self):
        from service.context_policy import NLISettings
        claims = [{"claim": str(i), "references": [{"source_id": "s", "quote": "原文"}]} for i in range(5)]
        async def engine(batch, settings):
            return {"execution_status": "completed", "claims": [
                {"claim": c["claim"], "status": "supported", "evidence": []} for c in batch]}
        with patch("service.nli_verifier.verify_claims", side_effect=engine) as mocked:
            result = await verify_citations(claims, enabled=True, settings=NLISettings(max_pairs=2))
        self.assertEqual(mocked.await_count, 3)
        self.assertEqual([r["claim"] for r in result["claims"]], [str(i) for i in range(5)])
        self.assertTrue(result["declared_claims_checked"])

    async def test_synthesis_mutation_updates_summary_and_conclusions(self):
        state = await fixture("delete")
        state["report_synthesis"]["executive_summary"] = "失败摘要。"
        state["synthesis_claims"] = [{"claim": "失败摘要。", "report_quote": "失败摘要。", "evidence": []}]
        state["claims"] = collect_report_claims(state)
        state["final_report"] = assemble(state)
        await verify(state, "report_claims")
        await repair_report_citations(state, call=AsyncMock(), assemble=assemble, verify=verify)
        self.assertEqual(state["report_synthesis"]["executive_summary"], "")
        self.assertFalse(state["synthesis_claims"])

    async def test_writer_entry_keeps_rejected_rewrite_without_raising(self):
        from service.deep_research_v2.agents.writer import LeadWriter
        from service.nli_verifier import aggregate_claims
        state = await fixture("rewrite_once")
        state["hypothesis_enabled"] = False
        from service.web_document import index_document
        source = {"id": "source_1", "url": "https://example.gov.cn/doc", "content": "有证据事实。", "body_quality": {"status": "passed"}, **index_document("有证据事实。", "https://example.gov.cn/doc")}
        state["raw_sources"] = [source]
        link = "[资料](https://example.gov.cn/doc)"
        state["draft_sections"]["sec_1"] = "有证据事实。" + link + "\n\n失败事实。" + link
        state["final_report"] = assemble(state)
        for claim in state["claims"]:
            claim["evidence"] = [{"document_id": source["document_id"], "sentence_ids": ["s000001"]}]
        _, bindings, _ = locate_claims(state)
        response = {"edits": [{"unit_id": bindings[1], "content": "仍失败事实。" + link, "claims": [
            {"claim": "仍失败事实。", "report_quote": "仍失败事实。", "evidence": state["claims"][1]["evidence"]}]}]}
        writer = LeadWriter.__new__(LeadWriter)
        writer.name, writer.logger, writer.max_tokens = "LeadWriter", logging.getLogger("test.repair.writer"), 60000
        writer.call_llm = AsyncMock(return_value=json.dumps(response))
        async def engine(claims, settings):
            predictions = [{"status": "scored", "scores": {"entailment": .99 if c["claim"] == "有证据事实。" else .005,
                "neutral": .005 if c["claim"] == "有证据事实。" else .99, "contradiction": .005}} for c in claims for ref in c["references"]]
            return aggregate_claims(claims, predictions, settings)
        async def extract(messages, settings):
            payload = json.loads(messages[1]["content"])
            if "messages" in payload:
                return json.dumps({"claim_id": payload["claim_id"], "evidence": [
                    {"source_id": block["source_id"], "sentence_ids": [s["sentence_id"] for s in block["sentences"]]}
                    for block in payload["messages"]]}), {}
            units = payload["units"]
            return json.dumps({"units": [{"unit_id": u["unit_id"], "status": "claims", "reason": "事实",
                "claims": [{"claim": re.sub(r"\[[^\]]+\]\([^)]+\)", "", u["text"]),
                    "report_quote": u["text"], "assertion_kind": "fact"}]} for u in units]}), {}
        with patch("service.deep_research_v2.report_claim_extraction.call_extractor", side_effect=extract), \
                patch("service.deep_research_v2.report_sentence_evidence.call_selector", side_effect=extract), \
                patch("service.nli_verifier.verify_claims", side_effect=engine):
            result = await writer._verify_and_repair_citations(state)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(writer.call_llm.await_count, 1)
        self.assertIn("仍失败事实。", state["final_report"])
        self.assertTrue(state["citation_repair"]["remaining_failed_claims"])
