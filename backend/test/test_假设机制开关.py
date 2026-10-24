"""假设机制配置、关闭路径与续跑冻结回归；仅使用本地模拟响应。"""

import copy
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from pydantic import ValidationError
from service.deep_research_v2.agents.architect import ChiefArchitect
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.delivery import build_complete_event
from service.deep_research_v2.followup_context import inherit_artifacts
from service.deep_research_v2.graph import pack_state
from service.deep_research_v2.hypothesis_policy import hypotheses_enabled, resolve_hypothesis_enabled
from service.deep_research_v2.state import create_initial_state
from service.web_document import format_numbered_body, index_document


URL = "https://example.gov.cn/report"
BODY = "项目于2023年投运。\n服务人口为6000人。"
HYPOTHESIS = {"id": "h1", "content": "这是仅供机制使用的待验证假设", "status": "unverified"}


def fixture(enabled=True):
    state = create_initial_state("分析项目", "test-session", ["example.gov.cn"],
                                 hypothesis_enabled=enabled)
    state["outline"] = [{"id": "sec_1", "title": "项目", "description": "进展", "status": "pending"}]
    state["hypotheses"] = [copy.deepcopy(HYPOTHESIS)]
    source = {"id": "source_1", "url": URL, "title": "官方报告", "content": BODY,
              "body_quality": {"status": "passed"}, **index_document(BODY, URL)}
    state["raw_sources"] = [source]
    result = {**copy.deepcopy(source), "reader_verified": True, "full_content": BODY}
    format_numbered_body(result, "项目")
    return state, [result]


def scout_fixture():
    scout = DeepScout.__new__(DeepScout)
    scout.name, scout.logger = "DeepScout", logging.getLogger("test.hypothesis.scout")
    scout.add_message = Mock()
    scout.call_llm = AsyncMock(return_value='{"extracted_facts": []}')
    scout.parse_json_response = json.loads
    return scout


def writer_fixture():
    writer = LeadWriter.__new__(LeadWriter)
    writer.name, writer.logger = "LeadWriter", logging.getLogger("test.hypothesis.writer")
    writer.add_message = Mock()
    writer.parse_json_response = json.loads
    return writer


def extracted_analysis(state):
    return {"extracted_facts": [{"fact_id": "f1", "content": "2023年投运",
            "source_url": URL, "source_name": "官方报告", "source_type": "official",
            "document_id": state["raw_sources"][0]["document_id"], "sentence_ids": ["s000001"],
            "credibility_score": .9, "data_points": []}],
            "hypothesis_evidence": [{"fact_id": "f1", "hypothesis_id": "h1",
                "evidence_type": "supports", "evidence_summary": "支持假设",
                "evidence_quote": "项目于2023年投运。"}]}


class ConfigurationTests(unittest.TestCase):
    def test_default_enabled_and_environment_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(resolve_hypothesis_enabled())
        with patch.dict(os.environ, {"RESEARCH_HYPOTHESIS_ENABLED": "false"}):
            self.assertFalse(create_initial_state("项目", "test", [])["hypothesis_enabled"])
            self.assertTrue(create_initial_state("项目", "test", [], hypothesis_enabled=True)["hypothesis_enabled"])
        with patch.dict(os.environ, {"RESEARCH_HYPOTHESIS_ENABLED": "true"}):
            self.assertFalse(resolve_hypothesis_enabled(False))

    def test_invalid_values_rejected(self):
        for value in ("false", 0, 1, {}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_hypothesis_enabled(value)
        with patch.dict(os.environ, {"RESEARCH_HYPOTHESIS_ENABLED": "typo"}), self.assertRaises(ValueError):
            resolve_hypothesis_enabled()

    def test_legacy_state_is_enabled_independent_of_environment(self):
        with patch.dict(os.environ, {"RESEARCH_HYPOTHESIS_ENABLED": "false"}):
            self.assertTrue(hypotheses_enabled({}))

    def test_api_accepts_boolean_and_rejects_coercion(self):
        from router.research_router import ResearchRequest
        base = {"query": "项目", "session_id": "00000000-0000-0000-0000-000000000001",
                "turn_id": "00000000-0000-0000-0000-000000000002"}
        self.assertIsNone(ResearchRequest(**base).hypothesis_enabled)
        self.assertFalse(ResearchRequest(**base, hypothesis_enabled=False).hypothesis_enabled)
        for invalid in ("false", 1, 0):
            with self.subTest(value=invalid), self.assertRaises(ValidationError):
                ResearchRequest(**base, hypothesis_enabled=invalid)

    def test_followup_does_not_reintroduce_parent_hypotheses_when_disabled(self):
        parent, _ = fixture(True)
        parent["hypothesis_evidence"] = [{"id": "ev1", "fact_id": "f1"}]
        parent["facts"] = [{"id": "f1", "source_id": "source_1"}]
        parent["conclusion_evidence_map"] = [{"hypothesis_ids": ["h1"]}]
        parent["report_hash"] = "parent-report-hash"
        state = create_initial_state("追问", "test", ["example.gov.cn"], hypothesis_enabled=False)
        inherit_artifacts(state, parent)
        self.assertEqual(len(state["facts"]), 1)
        self.assertEqual(state["hypotheses"], [])
        self.assertEqual(state["hypothesis_evidence"], [])
        self.assertEqual(state["conclusion_evidence_map"], [])
        self.assertFalse(state["hypothesis_enabled"])
        self.assertEqual(parent["hypotheses"], [HYPOTHESIS])

    def test_checkpoint_and_complete_event_preserve_switch(self):
        state = create_initial_state("项目", "test", [], hypothesis_enabled=False)
        saved = json.loads(json.dumps(pack_state(state)))
        self.assertFalse(saved["research"]["hypothesis_enabled"])
        event = build_complete_event(saved["research"], run_id="run", checkpoint_id="check")
        self.assertFalse(event["hypothesis_enabled"])


class PlanningTests(unittest.IsolatedAsyncioTestCase):
    async def plan(self, enabled, extra=None):
        output = {"outline": [{"id": f"sec_{i}", "title": f"章节{i}", "search_queries": ["项目"]}
                               for i in range(1, 5)],
                  "research_contract": {"critical_questions": ["项目表现如何？"]}}
        if extra is not None:
            output["hypotheses"] = extra
        agent = ChiefArchitect.__new__(ChiefArchitect)
        agent.name, agent.logger = "ChiefArchitect", logging.getLogger("test.hypothesis.plan")
        agent.add_message = Mock()
        agent.call_llm = AsyncMock(return_value=json.dumps(output, ensure_ascii=False))
        agent.parse_json_response = json.loads
        state = create_initial_state("分析项目", "test", [], hypothesis_enabled=enabled)
        await agent._initial_planning(state)
        return agent, state

    async def test_disabled_plan_does_not_require_or_request_hypotheses(self):
        agent, state = await self.plan(False)
        self.assertEqual(agent.call_llm.await_count, 1)
        self.assertEqual(len(state["outline"]), 4)
        self.assertEqual(state["hypotheses"], [])
        self.assertNotIn("hypotheses", agent.call_llm.call_args.kwargs["user_prompt"])
        self.assertFalse(state["errors"])

    async def test_disabled_plan_ignores_model_hypotheses(self):
        agent, state = await self.plan(False, [HYPOTHESIS])
        self.assertEqual(agent.call_llm.await_count, 1)
        self.assertEqual(state["hypotheses"], [])

    async def test_enabled_plan_retains_existing_behavior(self):
        agent, state = await self.plan(True, [HYPOTHESIS])
        self.assertEqual(state["hypotheses"][0]["id"], "h1")
        self.assertIn('"hypotheses":', agent.call_llm.call_args.kwargs["user_prompt"])


class ScoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_three_analysis_prompts_obey_switch(self):
        for enabled in (True, False):
            for route in ("initial", "deep", "supplementary"):
                with self.subTest(enabled=enabled, route=route):
                    state, results = fixture(enabled)
                    scout = scout_fixture()
                    if route == "initial":
                        await scout._analyze_search_results("项目", state["outline"][0], results,
                            state["hypotheses"], hypothesis_enabled=enabled)
                    elif route == "deep":
                        await scout._analyze_deep_search_results("项目", "投运年份", results,
                            "source_tracing", state["hypotheses"], hypothesis_enabled=enabled)
                    else:
                        await scout._analyze_supplementary_results("项目", "投运年份", results,
                            state["hypotheses"], hypothesis_enabled=enabled)
                    prompt = scout.call_llm.call_args.kwargs["user_prompt"]
                    self.assertEqual('"hypothesis_evidence":' in prompt, enabled)
                    self.assertEqual(HYPOTHESIS["content"] in prompt, enabled)
                    self.assertIn('"extracted_facts":', prompt)
                    self.assertIn('"sentence_ids":', prompt)

    async def test_disabled_ingestion_keeps_fact_but_skips_relation_processing(self):
        state, results = fixture(False)
        scout = scout_fixture()
        scout._update_hypothesis_status = Mock()
        outcome = scout._ingest_search_analysis(state, extracted_analysis(state), results,
                                               related_sections=["sec_1"])
        self.assertEqual(outcome["facts_added"], 1)
        self.assertEqual(outcome["relations_added"], 0)
        self.assertEqual(state["facts"][0]["sentence_ids"], ["s000001"])
        scout._update_hypothesis_status.assert_not_called()

    async def test_enabled_ingestion_still_updates_relation_ledger(self):
        state, results = fixture(True)
        outcome = scout_fixture()._ingest_search_analysis(state, extracted_analysis(state), results,
                                                        related_sections=["sec_1"])
        self.assertEqual(outcome["facts_added"], 1)
        self.assertEqual(outcome["relations_added"], 1)

    async def test_direct_relation_update_is_noop_when_disabled(self):
        state, _ = fixture(False)
        before = copy.deepcopy(state)
        scout_fixture()._update_hypothesis_status(state, [{"hypothesis_id": "h1", "fact_id": "f1"}])
        self.assertEqual(state, before)


class WriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_writer_omits_hypothesis_input_and_mapping_but_keeps_claims(self):
        state, _ = fixture(False)
        state["facts"] = [{"id": "f1", "source_id": "source_1", "content": "2023年投运",
                           "document_id": state["raw_sources"][0]["document_id"],
                           "sentence_ids": ["s000001"], "related_sections": ["sec_1"]}]
        claim = {"claim": "项目于2023年投运", "evidence": [{
            "document_id": state["raw_sources"][0]["document_id"], "sentence_ids": ["s000001"]}]}
        state["hypothesis_evidence"] = [{"id": "ev1", "hypothesis_id": "h1", "fact_id": "f1", "evidence_type": "supports"}]
        writer = writer_fixture()
        writer.call_llm = AsyncMock(side_effect=[json.dumps({
            "content": f"2023年投运[来源]({URL})", "key_points": [], "claims": [claim]}),
            json.dumps({"executive_summary": "摘要", "conclusions": [{"text": "项目已投运", "hypothesis_ids": ["h1"]}],
                        "outlook": "仍需跟踪", "claims": [claim]})])
        await writer._write_section(state, state["outline"][0])
        await writer._synthesize_report(state)
        for call in writer.call_llm.call_args_list:
            prompt = call.kwargs["user_prompt"]
            self.assertNotIn("假设与", prompt)
            self.assertNotIn(HYPOTHESIS["content"], prompt)
            self.assertNotIn('"hypothesis_ids":', prompt)
        self.assertIn("s000001", writer.call_llm.call_args_list[0].kwargs["user_prompt"])
        self.assertNotIn("s000001", writer.call_llm.call_args_list[1].kwargs["user_prompt"])
        self.assertEqual(state["conclusion_evidence_map"], [])
        self.assertEqual(len(state["claims"]), 2)
        self.assertEqual(state["references"][0]["url"], URL)
        self.assertEqual(state["citation_verification"]["status"], "not_executed")

    async def test_enabled_writer_keeps_section_ledger_but_synthesizes_from_chapters(self):
        state, _ = fixture(True)
        writer = writer_fixture()
        self.assertIn(HYPOTHESIS["content"], writer._format_hypothesis_ledger(state))
        self.assertIn("{hypothesis_ledger}", writer._hypothesis_prompt(writer.SECTION_WRITING_PROMPT, state))
        self.assertNotIn("{hypothesis_ledger}", writer.SYNTHESIS_PROMPT)
        self.assertNotIn('"hypothesis_ids":', writer.SYNTHESIS_PROMPT)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_run_and_resume_freeze_configuration(self):
        from service.deep_research_v2.service import DeepResearchV2Service
        service = object.__new__(DeepResearchV2Service)
        service.config = SimpleNamespace(research=SimpleNamespace(max_iterations=2))
        service.resolve_allowed_domains = lambda domains: []
        kwargs = {"query": "项目", "session_id": "test", "research_context": None, "allowed_domains": None}
        result = await service._graph_input(None, {}, resume=False, hypothesis_enabled=False, citation_verification={}, **kwargs)
        self.assertFalse(result["research"]["hypothesis_enabled"])
        self.assertEqual(result["research"]["citation_verification_config"], {
            "scout_facts": False, "report_claims": False, "report_failure_action": "audit",
        })
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values=result, next=("scout",))))
        with patch.dict(os.environ, {"RESEARCH_HYPOTHESIS_ENABLED": "true"}):
            self.assertIsNone(await service._graph_input(graph, {}, resume=True, **kwargs))
            self.assertIsNone(await service._graph_input(graph, {}, resume=True, hypothesis_enabled=False, **kwargs))
            with self.assertRaisesRegex(ValueError, "假设机制开关"):
                await service._graph_input(graph, {}, resume=True, hypothesis_enabled=True, **kwargs)

    async def test_legacy_resume_keeps_enabled(self):
        from service.deep_research_v2.service import DeepResearchV2Service
        service = object.__new__(DeepResearchV2Service)
        state = create_initial_state("项目", "test", [])
        state.pop("hypothesis_enabled")
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={"research": state}, next=("scout",))))
        kwargs = {"query": "项目", "session_id": "test", "research_context": None, "allowed_domains": None}
        with patch.dict(os.environ, {"RESEARCH_HYPOTHESIS_ENABLED": "false"}):
            self.assertIsNone(await service._graph_input(graph, {}, resume=True, **kwargs))
            with self.assertRaisesRegex(ValueError, "假设机制开关"):
                await service._graph_input(graph, {}, resume=True, hypothesis_enabled=False, **kwargs)

    async def test_api_passes_false_and_omits_unspecified_override(self):
        from router.research_router import ResearchRequest, stream_research
        async def empty_stream():
            if False:
                yield ""
        base = {"query": "项目", "session_id": "00000000-0000-0000-0000-000000000001",
                "turn_id": "00000000-0000-0000-0000-000000000002"}
        for supplied in (None, False, True):
            request = ResearchRequest(**base, hypothesis_enabled=supplied)
            service = SimpleNamespace(resolve_allowed_domains=Mock(return_value=[]), research=Mock(return_value=empty_stream()))
            conversations = SimpleNamespace(require_session=Mock(return_value=object()),
                save_user_message=Mock(return_value=SimpleNamespace(content="项目")), get_turn_message=Mock(return_value=None))
            with patch("router.research_router.get_research_service", return_value=service), \
                 patch("router.research_router.get_conversation_service", return_value=conversations), \
                 patch("router.research_router.get_research_context_service", return_value=SimpleNamespace(build=Mock(return_value={}))), \
                 patch("router.research_router.acquire_research_run_lease", return_value=object()):
                await stream_research(request, object(), SimpleNamespace(id="user"), object())
            if supplied is None:
                self.assertNotIn("hypothesis_enabled", service.research.call_args.kwargs)
            else:
                self.assertIs(service.research.call_args.kwargs["hypothesis_enabled"], supplied)


if __name__ == "__main__":
    unittest.main()
