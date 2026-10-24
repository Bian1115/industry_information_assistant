"""引用核验显式阶段开关、绑定与生命周期的离线回归。"""

import copy
import json
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError
from service.context_policy import NLISettings
from service.nli_verifier import aggregate_claims
from service.web_document import index_document
from service.deep_research_v2.citation_verification import (
    CitationVerificationOptions, verify_citations, verify_stage_citations,
)
from service.deep_research_v2.state import create_initial_state, ResearchPhase
from service.deep_research_v2.graph import pack_state
from service.deep_research_v2.delivery import build_complete_event
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.agents.critic import CriticMaster

URL = "https://example.gov.cn/report"
BODY = "项目于2023年投运。\n服务人口为6000人。"


def make_state(**options):
    state = create_initial_state("分析项目", "test-session", [])
    state["citation_verification_config"] = CitationVerificationOptions(**options).model_dump()
    state["raw_sources"] = [{"id": "source_1", "url": URL, "title": "项目报告",
                             "content": BODY, "body_quality": {"status": "passed"},
                             **index_document(BODY, URL)}]
    state["facts"] = [{"id": "fact_1", "source_id": "source_1", "content": "项目在2023年投运",
                       "document_id": state["raw_sources"][0]["document_id"],
                       "sentence_ids": ["s000001"], "related_sections": ["sec_1"]}]
    state["claims"] = [{"claim": "项目在2023年投运", "section_id": "sec_1",
                        "evidence": [{"evidence_id": "fact_1", "quote": "项目于2023年投运。"}]}]
    state["outline"] = [{"id": "sec_1", "title": "项目进展", "status": "pending"}]
    state["draft_sections"] = {"sec_1": f"项目于2023年投运。[项目报告]({URL})"}
    state["section_claims"] = {"sec_1": copy.deepcopy(state["claims"])}
    state["final_report"] = state["draft_sections"]["sec_1"]
    return state


async def supported_engine(claims, settings):
    return aggregate_claims(claims, [
        {"status": "scored", "scores": {"entailment": .99, "neutral": .005, "contradiction": .005}}
        for item in claims for ref in item["references"]
    ], settings)


async def independent_extractor(messages, settings):
    """报告核验改由独立模型读取最终正文；这里只模拟该模型的返回。"""
    payload = json.loads(messages[1]["content"])
    if "messages" in payload:
        return json.dumps({"claim_id": payload["claim_id"], "evidence": [
            {"source_id": block["source_id"], "sentence_ids": [s["sentence_id"] for s in block["sentences"]]}
            for block in payload["messages"]]}), {"mock": True}
    units = payload["units"]
    return json.dumps({"units": [{"unit_id": u["unit_id"],
        "status": "no_external_claims" if u["text"].startswith("#") or u["text"] == "概述" else "claims",
        "reason": "测试报告单元", "claims": [] if u["text"].startswith("#") or u["text"] == "概述" else [{
            "claim": u["text"].split("[")[0], "report_quote": u["text"], "assertion_kind": "fact"}]}
        for u in units]}, ensure_ascii=False), {"mock": True}


class IndependentExtractionMixin:
    def setUp(self):
        super().setUp()
        extraction = patch("service.deep_research_v2.report_claim_extraction.call_extractor", side_effect=independent_extractor)
        extraction.start()
        self.addCleanup(extraction.stop)
        selection = patch("service.deep_research_v2.report_sentence_evidence.call_selector", side_effect=independent_extractor)
        selection.start()
        self.addCleanup(selection.stop)


class ConfigurationTests(unittest.TestCase):
    def test_default_is_explicitly_off_and_strict(self):
        self.assertEqual(CitationVerificationOptions().model_dump(), {"scout_facts": False, "report_claims": False, "report_failure_action": "audit"})
        for invalid in ({"scout_facts": "false"}, {"report_claims": 1}, {"enabled": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                CitationVerificationOptions.model_validate(invalid)

    def test_api_accepts_per_stage_switch_and_rejects_string_booleans(self):
        from router.research_router import ResearchRequest
        request = {"query": "项目", "session_id": "00000000-0000-0000-0000-000000000001",
                   "turn_id": "00000000-0000-0000-0000-000000000002"}
        self.assertIsNone(ResearchRequest(**request).citation_verification)
        value = ResearchRequest(**request, citation_verification={"report_claims": True})
        self.assertFalse(value.citation_verification.scout_facts)
        self.assertTrue(value.citation_verification.report_claims)
        with self.assertRaises(ValidationError):
            ResearchRequest(**request, citation_verification={"report_claims": "true"})


class VerificationTests(IndependentExtractionMixin, unittest.IsolatedAsyncioTestCase):
    async def test_disabled_generic_call_never_reaches_engine(self):
        with patch("service.nli_verifier.verify_claims", new_callable=AsyncMock) as engine:
            result = await verify_citations([{"not": "even a valid claim"}])
        engine.assert_not_awaited()
        self.assertEqual(result["status"], "not_executed")
        self.assertNotIn("supported_count", result)

    async def test_disabled_stage_does_not_even_resolve_evidence_or_settings(self):
        state = make_state()
        state["citation_verification_settings"] = {"garbage": True}
        with patch("service.deep_research_v2.citation_verification._claims_for_stage", side_effect=AssertionError):
            for stage in ("scout_facts", "report_claims"):
                self.assertEqual((await verify_stage_citations(state, stage))["status"], "not_executed")

    async def test_stage_switches_are_independent(self):
        for active, inactive in (("report_claims", "scout_facts"), ("scout_facts", "report_claims")):
            state = make_state(**{active: True})
            with patch("service.nli_verifier.verify_claims", side_effect=supported_engine) as engine:
                result = await verify_stage_citations(state, active)
                skipped = await verify_stage_citations(state, inactive)
            self.assertEqual(engine.await_count, 1)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(skipped["status"], "not_executed")
            self.assertTrue(result["extracted_claims_checked"] if active == "report_claims" else result["declared_claims_checked"])
            self.assertFalse(result["all_claims_checked"])

    async def test_scout_reads_original_sentence_and_preserves_facts(self):
        state = make_state(scout_facts=True)
        previous = copy.deepcopy(state["facts"])
        with patch("service.nli_verifier.verify_claims", side_effect=supported_engine) as engine:
            result = await verify_stage_citations(state, "scout_facts")
        self.assertEqual(engine.call_args.args[0][0]["references"][0]["quote"], "项目于2023年投运。")
        self.assertEqual(state["facts"], previous)
        self.assertNotIn("quote", result["claims"][0]["evidence"][0])
        self.assertEqual(result["claims"][0]["evidence"][0]["fact_id"], "fact_1")

    async def test_contradiction_is_recorded_without_deleting_report(self):
        state = make_state(report_claims=True)
        original = copy.deepcopy(state)
        async def contradiction(claims, settings):
            return aggregate_claims(claims, [{"status": "scored", "scores": {
                "entailment": .005, "neutral": .005, "contradiction": .99}}], settings)
        with patch("service.nli_verifier.verify_claims", side_effect=contradiction):
            result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["claims"][0]["status"], "contradiction")
        self.assertEqual(state["final_report"], original["final_report"])
        self.assertEqual(state["claims"], original["claims"])

    async def test_invalid_writer_declarations_do_not_control_report_verification(self):
        for citation in ({"evidence_id": "invented", "quote": "项目于2023年投运。"},
                         {"evidence_id": "fact_1", "quote": "服务人口为6000人。"},
                         {"evidence_id": "fact_1", "quote": "改写的句子"}):
            state = make_state(report_claims=True)
            state["claims"][0]["evidence"] = [citation]
            # 最终报告实际引用有效，不采用Writer虚构或错误绑定。
            with patch("service.nli_verifier.verify_claims", side_effect=supported_engine) as engine:
                result = await verify_stage_citations(state, "report_claims")
            self.assertTrue(engine.call_args.args[0][0]["references"])
            self.assertEqual(result["claims"][0]["status"], "supported")
            self.assertEqual(result["status"], "passed")

    async def test_writer_extra_bad_reference_does_not_override_actual_report_links(self):
        state = make_state(report_claims=True)
        state["claims"][0]["evidence"].append({"evidence_id": "invented", "quote": "未知引句"})
        with patch("service.nli_verifier.verify_claims", side_effect=supported_engine):
            result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["review_claim_indexes"], [])
        self.assertEqual(result["supported_count"], 1)

    async def test_missing_writer_claims_does_not_skip_final_report(self):
        state = make_state(report_claims=True)
        state["claims"] = []
        with patch("service.nli_verifier.verify_claims", side_effect=supported_engine) as engine:
            result = await verify_stage_citations(state, "report_claims")
        engine.assert_awaited_once()
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["declared_claims_checked"])

    async def test_errors_never_become_passed(self):
        for exc, status in ((FileNotFoundError(), "model_missing"), (TimeoutError(), "timeout"), (ValueError(), "error")):
            with patch("service.nli_verifier.verify_claims", side_effect=exc):
                result = await verify_stage_citations(make_state(report_claims=True), "report_claims")
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["execution_status"], status)
            self.assertFalse(result["declared_claims_checked"])

    async def test_incomplete_engine_result_is_not_passed(self):
        with patch("service.nli_verifier.verify_claims", return_value={"status": "passed", "execution_status": "incomplete", "claims": []}):
            result = await verify_stage_citations(make_state(report_claims=True), "report_claims")
        self.assertEqual(result["status"], "incomplete")

    async def test_checkpoint_and_delivery_preserve_stage_results(self):
        state = make_state(report_claims=True)
        with patch("service.nli_verifier.verify_claims", side_effect=supported_engine):
            result = await verify_stage_citations(state, "report_claims")
        saved = json.loads(json.dumps(pack_state(state)))
        self.assertEqual(saved["research"]["citation_verification"], result)
        event = build_complete_event(saved["research"], run_id="run", checkpoint_id="check")
        self.assertEqual(event["citation_verification"], result)
        self.assertFalse(event["citation_verification_config"]["scout_facts"])

    async def test_revision_rechecks_and_changes_fingerprint(self):
        state = make_state(report_claims=True)
        with patch("service.nli_verifier.verify_claims", side_effect=supported_engine) as engine:
            before = await verify_stage_citations(state, "report_claims")
            state["claims"][0]["claim"] = "新的论断"
            state["final_report"] = "新的报告"
            after = await verify_stage_citations(state, "report_claims")
        self.assertEqual(engine.await_count, 2)
        self.assertNotEqual(before["input_sha256"], after["input_sha256"])
        self.assertNotEqual(before["report_sha256"], after["report_sha256"])


class LifecycleTests(IndependentExtractionMixin, unittest.IsolatedAsyncioTestCase):
    async def test_api_passes_explicit_options_to_service_but_preserves_omitted_defaults(self):
        from router.research_router import ResearchRequest, stream_research
        async def empty_stream():
            if False:
                yield ""
        base = {"query": "项目", "session_id": "00000000-0000-0000-0000-000000000001",
                "turn_id": "00000000-0000-0000-0000-000000000002"}
        for supplied in (None, {"scout_facts": False, "report_claims": True}):
            request = ResearchRequest(**base, citation_verification=supplied)
            service = SimpleNamespace(resolve_allowed_domains=Mock(return_value=[]), research=Mock(return_value=empty_stream()))
            conversations = SimpleNamespace(require_session=Mock(return_value=object()),
                save_user_message=Mock(return_value=SimpleNamespace(content="项目")), get_turn_message=Mock(return_value=None))
            with patch("router.research_router.get_research_service", return_value=service), \
                 patch("router.research_router.get_conversation_service", return_value=conversations), \
                 patch("router.research_router.get_research_context_service", return_value=SimpleNamespace(build=Mock(return_value={}))), \
                 patch("router.research_router.acquire_research_run_lease", return_value=object()):
                await stream_research(request, object(), SimpleNamespace(id="user"), object())
            kwargs = service.research.call_args.kwargs
            if supplied is None:
                self.assertNotIn("citation_verification", kwargs)
            else:
                self.assertEqual(kwargs["citation_verification"], {**supplied, "report_failure_action": "audit"})

    async def test_supplementary_scout_does_not_run_removed_fact_stage_and_invalidates_report_audit(self):
        state = make_state(scout_facts=True)
        state["phase"] = ResearchPhase.RE_RESEARCHING.value
        state["pending_research_tasks"] = [{"query": "项目年份", "target_section": "sec_1"}]
        state["citation_verification"] = {"status": "passed"}
        scout = DeepScout.__new__(DeepScout)
        scout.name, scout.logger = "DeepScout", logging.getLogger("test.citation.scout")
        scout._bind_source_scope = lambda value: []
        scout.add_message = Mock()
        scout._execute_search = AsyncMock(return_value=[])
        scout._emit_search_results_event = Mock()
        with patch("service.nli_verifier.verify_claims", side_effect=supported_engine) as engine:
            await scout.process(state)
        self.assertEqual(engine.await_count, 0)
        self.assertEqual(state["scout_citation_verification"]["status"], "not_executed")
        self.assertEqual(state["citation_verification"]["status"], "not_executed")

    async def test_synthesis_failure_cannot_keep_old_passed_audit(self):
        state = make_state(report_claims=True)
        state["citation_verification"] = {"status": "passed"}
        writer = LeadWriter.__new__(LeadWriter)
        writer.name, writer.logger = "LeadWriter", logging.getLogger("test.citation.writer")
        writer.call_llm = AsyncMock(side_effect=RuntimeError("模拟写作故障"))
        writer.parse_json_response = json.loads
        with self.assertRaises(RuntimeError):
            await writer._synthesize_report(state)
        self.assertEqual(state["citation_verification"]["status"], "not_executed")

    async def test_scout_process_does_not_restore_removed_fact_extraction(self):
        for enabled in (False, True):
            state = make_state(scout_facts=enabled)
            state["phase"] = ResearchPhase.RESEARCHING.value
            scout = DeepScout.__new__(DeepScout)
            scout.name, scout.logger = "DeepScout", logging.getLogger("test.citation.scout")
            scout._bind_source_scope = lambda value: []
            scout.add_message = Mock()
            scout._research_section = AsyncMock()
            scout._emit_search_results_event = Mock()
            with patch("service.nli_verifier.verify_claims", side_effect=supported_engine) as engine:
                await scout.process(state)
            self.assertEqual(engine.await_count, 0)
            self.assertEqual(state["scout_citation_verification"]["status"], "not_executed")

    async def test_writer_synthesis_checks_new_report_and_emits_result(self):
        state = make_state(report_claims=True)
        state["citation_verification"] = {"status": "passed", "input_sha256": "old"}
        writer = LeadWriter.__new__(LeadWriter)
        writer.name, writer.logger = "LeadWriter", logging.getLogger("test.citation.writer")
        writer.call_llm = AsyncMock(return_value=json.dumps({"executive_summary": "概述", "conclusions": [], "outlook": "", "claims": []}))
        writer.parse_json_response = json.loads
        emitted = []
        state["_emit_event"] = emitted.append
        with patch("service.nli_verifier.verify_claims", side_effect=supported_engine) as engine:
            await writer._synthesize_report(state)
        self.assertEqual(engine.await_count, 1)
        result = state["citation_verification"]
        self.assertEqual(result["status"], "passed")
        self.assertNotEqual(result["input_sha256"], "old")
        event = next(e for e in emitted if e["type"] == "report_draft")
        self.assertEqual(event["content"]["citation_verification"], result)

    async def test_critic_keeps_actual_independent_verification(self):
        state = make_state()
        state["phase"] = ResearchPhase.REVIEWING.value
        audit = {"status": "rejected", "all_claims_checked": False}
        state["citation_verification"] = copy.deepcopy(audit)
        critic = CriticMaster.__new__(CriticMaster)
        critic.name, critic.logger = "CriticMaster", logging.getLogger("test.citation.critic")
        critic.add_message = Mock()
        critic._review_content = AsyncMock(return_value={"issues": [], "overall_assessment": {"quality_score": 8, "verdict": "pass"}})
        with patch("service.deep_research_v2.agents.critic.deterministic_critic_issues", return_value=[]):
            await critic.process(state)
        self.assertEqual(state["citation_verification"], audit)
        self.assertEqual(state["quality_score"], 8)

    async def test_new_run_freezes_options_and_nli_parameters(self):
        from service.deep_research_v2.service import DeepResearchV2Service
        from service.context_policy import selected_policy
        service = object.__new__(DeepResearchV2Service)
        service.config = SimpleNamespace(research=SimpleNamespace(max_iterations=2))
        service.resolve_allowed_domains = lambda domains: []
        custom = NLISettings(max_length=80000, output_tokens=6000)
        policy = selected_policy().model_copy(update={"nli": custom})
        kwargs = dict(query="项目", session_id="test", research_context=None, allowed_domains=None)
        with patch("service.context_policy.selected_policy", return_value=policy):
            result = await service._graph_input(None, {}, resume=False, citation_verification={"report_claims": True}, **kwargs)
        research = result["research"]
        self.assertEqual(research["citation_verification_settings"]["model_id"], "qwen3-max")
        self.assertEqual(research["citation_verification_settings"]["max_length"], custom.max_length)
        self.assertEqual(research["citation_verification_settings"]["output_tokens"], custom.output_tokens)
        self.assertEqual(research["citation_verification_config"], {"scout_facts": False, "report_claims": True, "report_failure_action": "audit"})
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values=result, next=("writer",))))
        self.assertIsNone(await service._graph_input(graph, {}, resume=True, **kwargs))
        with self.assertRaisesRegex(ValueError, "引用核验开关"):
            await service._graph_input(graph, {}, resume=True, citation_verification={}, **kwargs)

    async def test_legacy_resume_defaults_off(self):
        from service.deep_research_v2.service import DeepResearchV2Service
        service = object.__new__(DeepResearchV2Service)
        state = make_state()
        state.pop("citation_verification_config")
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={"research": state}, next=("scout",))))
        self.assertIsNone(await service._graph_input(graph, {}, resume=True, query="项目", session_id="test", research_context=None, allowed_domains=None))


if __name__ == "__main__":
    unittest.main()
