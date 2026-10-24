"""最终报告独立提取、引用覆盖、预测限定与修复衔接；模型调用使用桩。"""
import copy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from service.context_policy import NLISettings
from service.web_document import index_document
from service.nli_verifier import aggregate_claims, QwenNLI, NLIResponseError
from service.deep_research_v2.report_claim_extraction import (
    report_units, parse_extraction, extract_report_claims, citation_urls, reference_catalog, call_extractor,
)
from service.deep_research_v2.report_citation_inputs import report_claim_inputs
from service.deep_research_v2.citation_verification import verify_stage_citations
from service.deep_research_v2.citation_repair import repair_report_citations
from service.deep_research_v2.agents.critic import CriticMaster, handoff_legacy_citation_issues

URL = "https://example.gov.cn/report"


def state_for(text=None, body="项目于2023年投运。", **options):
    text = text if text is not None else f"项目于2023年投运。[资料]({URL})"
    return {"final_report": text, "draft_sections": {"sec_1": text}, "outline": [{"id": "sec_1"}],
        "claims": [], "section_claims": {}, "report_synthesis": {}, "allowed_domains": [],
        "citation_verification_config": {"report_claims": True, **options},
        "raw_sources": [{"id": "s", "url": URL, "title": "资料", "content": body,
                         "body_quality": {"status": "passed"}, **index_document(body, URL)}]}


async def extractor_stub(messages, settings):
    """模拟独立模型：只读报告单元，完全不读Writer声明。"""
    payload = json.loads(messages[1]["content"])
    if "messages" in payload:
        return json.dumps({"claim_id": payload["claim_id"], "evidence": [
            {"source_id": block["source_id"], "sentence_ids": [s["sentence_id"] for s in block["sentences"]]}
            for block in payload["messages"]]}), {"mock": True}
    units = payload["units"]
    rows = []
    for unit in units:
        heading = unit["text"].startswith("#")
        rows.append({"unit_id": unit["unit_id"], "status": "no_external_claims" if heading else "claims",
            "reason": "标题" if heading else "外部事实", "claims": [] if heading else [{
                "claim": unit["text"].split("[资料]")[0], "report_quote": unit["text"],
                "assertion_kind": "forecast" if "预计" in unit["text"] else "fact"}]})
    return json.dumps({"units": rows}, ensure_ascii=False), {"mock": True}


async def supported(claims, settings):
    return aggregate_claims(claims, [{"status": "scored", "scores": {
        "entailment": .99, "neutral": .005, "contradiction": .005}}
        for claim in claims for _ in claim["references"]], settings)


class ParsingTests(unittest.TestCase):
    def test_full_report_units_include_summary_table_conclusion_and_tail(self):
        report = "# 报告\n\n## 摘要\n摘要事实。\n\n| 年份 | 数值 |\n|---|---|\n| 2023 | 10 |\n\n## 结论\n末尾事实。\n\n## 参考文献\n1. [资料](https://example.com)"
        units = report_units(report)
        self.assertTrue(any(u["text"] == "摘要事实。" for u in units))
        table = next(u for u in units if "2023" in u["text"])
        self.assertEqual(table["table_header"], "| 年份 | 数值 |")
        self.assertTrue(any(u["text"] == "末尾事实。" for u in units))
        self.assertFalse(any("https://example.com" in u["text"] for u in units))
        for unit in units:
            self.assertEqual(report[unit["start"]:unit["end"]], unit["text"])

    def test_every_unit_requires_explicit_coverage_and_unique_ids(self):
        state = state_for("第一句。\n\n第二句。")
        units = report_units(state["final_report"])
        for rows in ([], [{"unit_id": units[0]["unit_id"], "status": "no_external_claims", "reason": "无", "claims": []}] * 2):
            with self.assertRaises(ValueError):
                parse_extraction(json.dumps({"units": rows}), units, state)

    def test_sentence_without_link_does_not_borrow_citation_from_other_sentence(self):
        state = state_for(f"有引用事实。[资料]({URL}) 无引用事实。")
        units = report_units(state["final_report"])
        rows = [{"unit_id": units[0]["unit_id"], "status": "claims", "reason": "事实", "claims": [
            {"claim": "无引用事实。", "report_quote": "无引用事实。", "assertion_kind": "fact"}]}]
        claims, _ = parse_extraction(json.dumps({"units": rows}), units, state)
        self.assertEqual(claims[0]["binding_errors"], ["missing_inline_citation"])

    def test_links_immediately_after_sentence_are_attached(self):
        state = state_for()
        units = report_units(state["final_report"])
        rows = [{"unit_id": units[0]["unit_id"], "status": "claims", "reason": "事实", "claims": [
            {"claim": "项目于2023年投运。", "report_quote": "项目于2023年投运。", "assertion_kind": "fact"}]}]
        claims, _ = parse_extraction(json.dumps({"units": rows}), units, state)
        self.assertEqual(claims[0]["citation_urls"], [URL])
        self.assertEqual(state["final_report"][claims[0]["report_start"]:claims[0]["report_end"]], claims[0]["report_quote"])

    def test_numbered_and_reference_style_citations(self):
        catalog = reference_catalog(f"事实[1]。\n\n## 参考文献\n1. [资料]({URL})\n[x]: {URL}")
        self.assertEqual(citation_urls("事实[1]。", catalog), ([URL], []))
        self.assertEqual(citation_urls("事实[资料][x]。", catalog), ([URL], []))
        self.assertEqual(citation_urls("事实[2]。", catalog), ([], ["unresolved_report_citation"]))

    def test_invented_report_quote_rejected(self):
        state = state_for()
        units = report_units(state["final_report"])
        rows = [{"unit_id": units[0]["unit_id"], "status": "claims", "reason": "事实", "claims": [
            {"claim": "编造", "report_quote": "报告中没有的句子", "assertion_kind": "fact"}]}]
        with self.assertRaises(ValueError):
            parse_extraction(json.dumps({"units": rows}), units, state)


class VerificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.extract_patch = patch("service.deep_research_v2.report_claim_extraction.call_extractor", side_effect=extractor_stub)
        self.extract = self.extract_patch.start()
        self.addCleanup(self.extract_patch.stop)
        selection = patch("service.deep_research_v2.report_sentence_evidence.call_selector", side_effect=extractor_stub)
        selection.start()
        self.addCleanup(selection.stop)
        self.engine_patch = patch("service.nli_verifier.verify_claims", side_effect=supported)
        self.engine = self.engine_patch.start()
        self.addCleanup(self.engine_patch.stop)

    async def test_writer_omitted_all_claims_still_checks_final_report(self):
        state = state_for()
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["scope"], "final_report_extracted_claims")
        self.assertTrue(result["extracted_claims_checked"])
        self.assertFalse(result["all_claims_checked"])
        self.assertFalse(result["declared_claims_checked"])
        self.assertEqual(state["claims"], [])
        self.assertEqual(len(state["report_verification_claims"]), 1)

    async def test_missing_inline_reference_fails_even_if_writer_supplies_evidence(self):
        state = state_for("项目于2023年投运。")
        state["claims"] = [{"claim": "Writer额外声明", "evidence": [{"document_id": "fake"}]}]
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["claims"][0]["status"], "missing_citation")
        self.assertEqual(result["missing_citation_count"], 1)
        self.assertEqual(self.engine.call_args.args[0][0]["references"], [])

    async def test_unknown_source_not_replaced_with_other_known_source(self):
        state = state_for("项目于2023年投运。[资料](https://wrong.example/report)")
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "rejected")
        self.assertIn("unknown_or_ambiguous_cited_source", result["claims"][0]["binding_errors"])

    async def test_one_supported_source_cannot_hide_another_unsupported_citation(self):
        other = "https://example.gov.cn/other"
        state = state_for(f"项目于2023年投运。[资料]({URL})[其他]({other})")
        body = "其他地区正在建设新项目。"
        state["raw_sources"].append({"id": "other", "url": other, "content": body,
            "body_quality": {"status": "passed"}, **index_document(body, other)})
        async def partial(claims, settings):
            return aggregate_claims(claims, [{"status": "scored", "scores": {
                "entailment": .99 if r["source_id"] == "s" else .005,
                "neutral": .005 if r["source_id"] == "s" else .99, "contradiction": .005}}
                for c in claims for r in c["references"]], settings)
        self.engine.side_effect = partial
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["claims"][0]["status"], "unsupported_citation")

    async def test_rag_search_is_limited_to_report_cited_documents(self):
        state = state_for()
        extraction = await extract_report_claims(state, NLISettings())
        rag = AsyncMock()
        rag.retrieve.return_value = ([], {})
        rag.references.return_value = [{"document_id": "other", "source_id": "other", "quote": "其他来源"}]
        inputs = await report_claim_inputs(state, extraction["claims"], rag)
        self.assertEqual(rag.retrieve.call_args.kwargs["document_ids"], {state["raw_sources"][0]["document_id"]})
        self.assertEqual(inputs[0]["references"], [])
        self.assertIn("cited_source_evidence_unavailable", inputs[0]["binding_errors"])

    async def test_prediction_as_fact_cannot_pass_even_with_high_support_score(self):
        state = state_for(f"项目已于2030年投运。[资料]({URL})", body="项目预计于2030年投运。")
        async def temporal(claims, settings):
            return aggregate_claims(claims, [{"status": "scored", "prediction_as_fact": True,
                "scores": {"entailment": .99, "neutral": .005, "contradiction": .005}}
                for c in claims for _ in c["references"]], settings)
        self.engine.side_effect = temporal
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["claims"][0]["status"], "prediction_as_fact")
        self.assertEqual(result["prediction_as_fact_count"], 1)

    async def test_legitimate_forecast_preserves_qualifier(self):
        state = state_for(f"项目预计于2030年投运。[资料]({URL})", body="项目预计于2030年投运。")
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["claims"][0]["assertion_kind"], "forecast")
        self.assertIn("预计", self.engine.call_args.args[0][0]["claim"])

    async def test_changed_report_is_reextracted_ignoring_stale_writer_claims(self):
        state = state_for()
        before = await verify_stage_citations(state, "report_claims")
        state["final_report"] = "另一条没有引用的事实。"
        after = await verify_stage_citations(state, "report_claims")
        self.assertEqual(self.extract.await_count, 2)
        self.assertNotEqual(before["report_sha256"], after["report_sha256"])
        self.assertEqual(after["status"], "rejected")

    async def test_extractor_failure_clears_old_results_and_never_calls_nli(self):
        state = state_for()
        state["citation_verification"] = {"status": "passed"}
        self.extract.side_effect = ValueError("invalid JSON")
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "error")
        self.assertEqual(state["report_verification_claims"], [])
        self.engine.assert_not_awaited()

    async def test_uncertain_unit_is_incomplete(self):
        async def uncertain(messages, settings):
            units = json.loads(messages[1]["content"])["units"]
            return json.dumps({"units": [{"unit_id": u["unit_id"], "status": "uncertain",
                "reason": "歧义", "claims": []} for u in units]}), {}
        self.extract.side_effect = uncertain
        result = await verify_stage_citations(state_for(), "report_claims")
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["extracted_claims_checked"])

    async def test_explicitly_nonfactual_report_is_distinct_from_missing_extraction(self):
        state = state_for("# 请补充研究范围")
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["claims_count"], 0)
        self.assertTrue(result["extracted_claims_checked"])
        self.engine.assert_not_awaited()

    async def test_oversized_unit_fails_without_truncation_or_api(self):
        with self.assertRaisesRegex(ValueError, "未截断"):
            await extract_report_claims(state_for("长文" * 1000), NLISettings(max_length=16))
        self.extract.assert_not_awaited()

    async def test_disabled_does_not_extract_or_call_nli(self):
        state = state_for(report_claims=False)
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "not_executed")
        self.extract.assert_not_awaited()
        self.engine.assert_not_awaited()

    async def test_batches_cover_tail_without_truncation(self):
        state = state_for("\n\n".join(f"第{i}条事实。" for i in range(19)))
        result = await extract_report_claims(state, NLISettings())
        self.assertEqual(self.extract.await_count, 3)
        self.assertEqual(result["units_total"], 19)
        self.assertEqual(len(result["claims"]), 19)
        self.assertEqual(result["claims"][-1]["claim"], "第18条事实。")

    async def test_independent_claims_can_be_deleted_when_writer_claims_are_empty(self):
        state = state_for("没有引用的事实。\n\n" + f"有引用事实。[资料]({URL})", report_failure_action="delete")
        state["report_synthesis"] = {"executive_summary": "", "outlook": "", "conclusions": []}
        await verify_stage_citations(state, "report_claims")
        original = state["final_report"]
        def assemble(proposal):
            return proposal["draft_sections"]["sec_1"]
        result = await repair_report_citations(state, call=AsyncMock(), assemble=assemble)
        self.assertEqual(state["citation_repair"]["status"], "applied")
        self.assertNotEqual(state["final_report"], original)
        self.assertNotIn("没有引用的事实", state["final_report"])
        self.assertEqual(result["status"], "passed")


class BoundaryTests(unittest.TestCase):
    def test_temporal_flag_is_strict_and_overrides_entailment(self):
        row = {"id": 0, "reason": "原文为预测", "prediction_as_fact": True,
               "scores": {"entailment": .99, "neutral": .005, "contradiction": .005}}
        parsed = QwenNLI.parse_response(json.dumps({"items": [row]}), 1)
        self.assertTrue(parsed[0]["prediction_as_fact"])
        row["prediction_as_fact"] = "true"
        with self.assertRaises(NLIResponseError):
            QwenNLI.parse_response(json.dumps({"items": [row]}), 1)

    def test_critic_owns_report_level_checks_only(self):
        prompt = CriticMaster.REVIEW_PROMPT
        self.assertIn("观点平衡", prompt)
        self.assertIn("整体时效", prompt)
        self.assertNotIn('"issue_type": "missing_source/', prompt)
        self.assertIn("不在此重复裁定", prompt)

    def test_legacy_citation_issue_is_handed_off_without_claiming_it_was_resolved(self):
        state = {"critic_feedback": [{"id": "old", "issue_type": "missing_source", "resolved": False},
                                    {"id": "logic", "issue_type": "logic_error", "resolved": False}]}
        handoff_legacy_citation_issues(state)
        self.assertEqual([i["id"] for i in state["critic_feedback"]], ["logic"])
        self.assertEqual(state["citation_issue_handoffs"][0]["owner"], "citation_verification")
        self.assertFalse(state["citation_issue_handoffs"][0]["resolved"])


class ExtractorRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_request_uses_frozen_settings_and_reports_usage(self):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"units":[]}'))],
            usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5}), id="request"))
        budget = Mock()
        budget.reserve.return_value = {}
        messages = [{"role": "system", "content": "提取规则"}, {"role": "user", "content": "最终报告"}]
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}), \
                patch("service.llm_client.create_async_openai_client", return_value=client), \
                patch("service.verification_budget.current_budget", return_value=budget):
            content, metadata = await call_extractor(messages, NLISettings(output_tokens=8192))
        args = client.chat.completions.create.call_args.kwargs
        self.assertEqual(args["messages"], messages)
        self.assertEqual(args["max_tokens"], 8192)
        self.assertEqual(args["temperature"], 0)
        self.assertEqual(args["extra_body"], {"enable_thinking": False})
        self.assertEqual(metadata["usage"]["prompt_tokens"], 10)
        self.assertEqual(content, '{"units":[]}')
        budget.complete.assert_called_once()

    async def test_truncated_output_is_rejected_even_when_json_is_parseable(self):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"units":[]}'))],
            usage=None, id="truncated"))
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}), \
                patch("service.llm_client.create_async_openai_client", return_value=client), \
                patch("service.verification_budget.current_budget", return_value=None), \
                self.assertRaisesRegex(ValueError, "未完整结束"):
            await call_extractor([{"role": "user", "content": "报告"}], NLISettings())


if __name__ == "__main__":
    unittest.main()
