"""同 session 跨 run 的授权成果继承、规划消解与冻结续跑；不访问数据库和模型。"""
import copy
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "app"), str(Path(__file__).resolve().parent)]
from test_智能体独立历史 import CharCounter, MemoryStore
from service.agent_history import HistoryScope, bind_agent_history
from service.context_policy import ContextPolicy
from service.web_document import index_document
from service.deep_research_v2.followup_context import (
    SessionInheritanceSettings, parent_artifacts, planning_context, inherit_artifacts, digest,
)
from service.deep_research_v2.agents.architect import ChiefArchitect
from service.deep_research_v2.service import DeepResearchV2Service, LegacyRunNotResumableError
from service.deep_research_v2.state import create_initial_state, GRAPH_VERSION
from service.deep_research_v2.graph import graph_config
from service.session_task_service import validate_decision, report_hash, SessionTaskClarification

URL = "https://example.gov.cn/report"
BODY = "项目于2023年投运。\n服务人口为6000人。"


def parent_snapshot():
    value = create_initial_state("项目运营分析", "session", ["example.gov.cn"])
    value.update(status="completed", final_report="上一轮报告：项目已经投运。", iteration=9,
                 node_attempts={"research": 8}, router_attempts=[{"old": True}])
    value["outline"] = [{"id": f"sec_{i}", "title": f"章节{i}", "description": "项目运营", "status": "written"}
                        for i in range(4)]
    value["draft_sections"] = {"sec_0": "原章节内容：项目服务6000人。"}
    value["research_contract"] = {"contract_revision": 1, "query": "项目运营分析",
                                 "active_constraints": {"max_body_chars": 3000}, "time_range": "2023—2024"}
    value["active_constraints"] = {"max_body_chars": 3000}
    source = {"id": "source_0", "url": URL, "content": BODY, "body_quality": {"status": "passed"},
              **index_document(BODY, URL)}
    value["raw_sources"] = [source]
    value["facts"] = [{"id": "fact_0", "content": "项目服务6000人", "source_id": "source_0", "source_url": URL,
                       "document_id": source["document_id"], "sentence_ids": ["s000002"], "related_sections": ["sec_0"]}]
    return SimpleNamespace(values={"research": value}, next=(), config={"configurable": {"checkpoint_id": "parent-cp"}})


def service():
    value = DeepResearchV2Service.__new__(DeepResearchV2Service)
    value.config = SimpleNamespace(research=SimpleNamespace(max_iterations=2))
    value.resolve_allowed_domains = lambda domains=None: list(domains or [])
    return value


def plan_response():
    return {"outline": [{"id": f"sec_{i}", "title": f"新章节{i}"} for i in range(4)], "hypotheses": [],
            "research_contract": {}}


class ContextTests(unittest.TestCase):
    def test_parent_requires_same_session_completed_report_and_finished_graph(self):
        for modify in (lambda s: s.values["research"].update(session_id="other"),
                       lambda s: s.values["research"].update(status="running"),
                       lambda s: setattr(s, "next", ("write",)),
                       lambda s: s.values["research"].update(final_report="")):
            snapshot = parent_snapshot(); modify(snapshot)
            with self.assertRaises(ValueError):
                parent_artifacts(snapshot, "session")

    def test_indexed_source_hash_validates_and_tampering_is_rejected(self):
        parent = parent_artifacts(parent_snapshot(), "session")
        state = create_initial_state("继续", "session", ["example.gov.cn"])
        state["parent_run_id"] = "parent"
        inherit_artifacts(state, parent)
        self.assertEqual(len(state["facts"]), 1)
        self.assertEqual(state["facts"][0]["sentence_ids"], ["s000002"])
        state["draft_sections"]["sec_0"] = "修改后"
        self.assertNotEqual(state["draft_sections"], parent["draft_sections"])
        self.assertEqual(state["iteration"], 0)
        self.assertNotIn("node_attempts", state)
        parent["raw_sources"][0]["content"] += "篡改"
        with self.assertRaisesRegex(ValueError, "编号或哈希"):
            inherit_artifacts(create_initial_state("继续", "session", ["example.gov.cn"]), parent)

    def test_source_permission_narrowing_filters_parent_evidence(self):
        parent = parent_artifacts(parent_snapshot(), "session")
        state = create_initial_state("继续", "session", ["another.gov.cn"])
        inherit_artifacts(state, parent)
        self.assertEqual(state["raw_sources"], [])
        self.assertEqual(state["facts"], [])

    def test_planning_context_is_bounded_and_explicitly_marks_excerpts(self):
        parent = parent_artifacts(parent_snapshot(), "session")
        parent["draft_sections"] = {f"sec_{i}": "完整报告很长" * 1000 for i in range(30)}
        parent["query"] = "历史问题" * 5000
        with patch("service.qwen_token_counter.get_qwen_token_counter", return_value=CharCounter()):
            context = planning_context(parent, "parent", 512)
        self.assertLessEqual(CharCounter().count_text(json.dumps(context, ensure_ascii=False)), 512)
        self.assertGreater(context["omitted_items"], 0)
        self.assertIn("节选", context["note"])
        self.assertEqual(len(parent["draft_sections"]), 30)

    def test_configuration_is_strict(self):
        with patch.dict(os.environ, {"SESSION_RESEARCH_INHERITANCE_ENABLED": "bad"}), self.assertRaises(ValueError):
            SessionInheritanceSettings.from_env()
        with patch.dict(os.environ, {"SESSION_RESEARCH_INHERITANCE_ENABLED": "0"}):
            self.assertFalse(SessionInheritanceSettings.from_env().enabled)


class InputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"SESSION_RESEARCH_INHERITANCE_ENABLED": "1",
            "CONTEXT_ABLATION_CONFIG": "", "AGENT_COMPRESSION_CONFIG": "", "AGENT_SUMMARY_STRATEGY": "incremental-v3"})
        env.start(); self.addCleanup(env.stop)
        counter = patch("service.qwen_token_counter.get_qwen_token_counter", return_value=CharCounter())
        counter.start(); self.addCleanup(counter.stop)

    async def initial(self, snapshot=None, domains=None, overrides=None, parent_id="parent", query=None):
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=snapshot or parent_snapshot()))
        query = query or ("取消字数限制" if parent_id else "项目运营分析")
        candidates = [{"artifact_ref": "report:parent", "run_id": parent_id,
                       "report_hash": report_hash((snapshot or parent_snapshot()).values["research"]["final_report"])}] if parent_id else []
        task = validate_decision({"topic_relation": "continue" if parent_id else "new",
            "depends_on_history": bool(parent_id), "target_artifact_ref": "report:parent" if parent_id else None,
            "resolved_query": "项目运营分析，不限制字数" if parent_id else query,
            "operations": [{"kind": "edit" if parent_id else "research", "source_sentence_ids": ["S1"]}],
            "requirement_changes": [{"op": "remove", "field": "max_body_chars", "value": None,
                                     "source_sentence_ids": ["S1"]}] if parent_id else [],
            "revision_instructions": [query]}, query, candidates, turn_id="child")
        context = {"session_task": task, "conversation_context": [{"role": "user", "content": "项目运营分析"}] if parent_id else []}
        with patch("service.session_task_service.authorize_artifact") as authorize:
            output = await service()._graph_input(graph, graph_config("child"), resume=False,
                query=query, session_id="session", research_context=context,
                allowed_domains=domains, user_id="user", run_id="child", agent_compression=overrides)
        return output["research"], graph, authorize

    async def test_session_task_installs_parent_context_and_committed_contract(self):
        state, graph, authorize = await self.initial()
        authorize.assert_called_once()
        graph.aget_state.assert_awaited_once_with(graph_config("parent"))
        self.assertEqual(state["parent_run_id"], "parent")
        self.assertNotIn("router_config", state)
        self.assertNotIn("followup_route", state)
        self.assertEqual(state["research_contract"]["contract_update"]["status"], "committed")
        self.assertNotIn("pending_contract_update", state["research_contract"])
        self.assertEqual(state["active_constraints"], {})
        self.assertEqual(state["session_parent"]["draft_sections"]["sec_0"], "原章节内容：项目服务6000人。")
        self.assertIn("session_planning_context", state)
        self.assertEqual(state["iteration"], 0)
        self.assertEqual(state["facts"], [])  # 重规划在Architect完成后才安装适用证据。
        self.assertEqual(state["allowed_domains"], ["example.gov.cn"])
        self.assertEqual(state["conversation_context"][0]["content"], "项目运营分析")

    async def test_first_turn_without_parent_does_not_invent_parent_context(self):
        state, graph, _ = await self.initial(parent_id=None)
        graph.aget_state.assert_not_awaited()
        self.assertNotIn("session_parent", state)
        self.assertNotIn("parent_run_id", state)

    async def test_inheritance_can_be_disabled_and_domain_override_remains_current(self):
        with patch.dict(os.environ, {"SESSION_RESEARCH_INHERITANCE_ENABLED": "0"}), self.assertRaises(SessionTaskClarification):
            await self.initial()
        state, _, _ = await self.initial(domains=["another.gov.cn"])
        self.assertEqual(state["allowed_domains"], ["another.gov.cn"])

    async def test_api_overrides_are_frozen_into_new_run_and_cannot_change_on_resume(self):
        state, _, _ = await self.initial(overrides={"LeadWriter": {"enabled": True, "semantic_summary": False}})
        frozen = ContextPolicy.from_snapshot(state["context_policy"])
        self.assertTrue(frozen.agent_compression["LeadWriter"].enabled)
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={"research": state}, next=("plan",))))
        kwargs = dict(resume=True, query="取消字数限制", session_id="session", research_context=None, allowed_domains=None)
        with patch.dict(os.environ, {"AGENT_COMPRESSION_CONFIG": '{"LeadWriter":{"enabled":false}}'}):
            self.assertIsNone(await service()._graph_input(graph, graph_config("child"), **kwargs))
        with self.assertRaisesRegex(ValueError, "沿用已保存"):
            await service()._graph_input(graph, graph_config("child"), **kwargs, agent_compression={"LeadWriter": {"enabled": False}})
        self.assertIsNone(await service()._graph_input(graph, graph_config("child"), **kwargs,
            agent_compression={"LeadWriter": {"enabled": True, "semantic_summary": False}}))


    async def test_architect_uses_committed_contract_then_reuses_evidence_and_clears_old_section_binding(self):
        state, _, _ = await self.initial()
        value = ChiefArchitect.__new__(ChiefArchitect)
        value.name, value.logger = "ChiefArchitect", logging.getLogger("session-test")
        value.add_message = Mock(); value.call_llm = AsyncMock(return_value=json.dumps(plan_response(), ensure_ascii=False))
        parent = copy.deepcopy(state["session_parent"])
        out = await value.process(state)
        self.assertEqual(out["active_constraints"], {})
        self.assertEqual(out["research_contract"]["contract_revision"], 2)
        self.assertEqual(len(out["facts"]), 1)
        self.assertEqual(out["facts"][0]["related_sections"], [])
        self.assertEqual(out["facts"][0]["sentence_ids"], ["s000002"])
        self.assertIn("原章节内容", value.call_llm.call_args.kwargs["user_prompt"])
        self.assertNotIn("followup_contract_update", value.call_llm.call_args.kwargs["user_prompt"])
        self.assertNotIn("quote逐字", value.call_llm.call_args.kwargs["user_prompt"])
        self.assertNotIn("session_parent", out)
        self.assertEqual(parent["research_contract"]["active_constraints"], {"max_body_chars": 3000})

    async def test_topic_change_does_not_reuse_parent_facts_or_constraints(self):
        state, _, _ = await self.initial(parent_id=None, query="换主题为新能源研究")
        value = ChiefArchitect.__new__(ChiefArchitect)
        value.name, value.logger = "ChiefArchitect", logging.getLogger("session-topic-test")
        value.add_message = Mock(); value.call_llm = AsyncMock(return_value=json.dumps(plan_response(), ensure_ascii=False))
        out = await value.process(state)
        self.assertEqual(out["facts"], [])
        self.assertEqual(out["active_constraints"], {})
        self.assertNotIn("time_range", out["research_contract"])
        self.assertNotIn("session_planning_context", out)


    async def test_followup_without_session_task_cannot_use_architect_legacy_entry(self):
        for context in ({"conversation_context": [{"role": "user", "content": "项目运营分析"}]},
                        {"memory_context": [{"content": "历史要求"}]}, {"is_follow_up": True}):
            with self.subTest(context=context):
                graph = SimpleNamespace(aget_state=AsyncMock())
                with patch("service.deep_research_v2.followup_context.latest_parent_run") as locate:
                    with self.assertRaisesRegex(SessionTaskClarification, "会话层"):
                        await service()._graph_input(graph, graph_config("child"), resume=False,
                            query="取消字数限制", session_id="session", research_context=context,
                            allowed_domains=None, user_id="user", run_id="child")
                locate.assert_not_called()
                graph.aget_state.assert_not_awaited()

    async def test_pending_architect_contract_checkpoint_cannot_resume(self):
        state, _, _ = await self.initial()
        state["research_contract"]["pending_contract_update"] = {"base_contract": {}}
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={"research": state}, next=("plan",))))
        with self.assertRaisesRegex(LegacyRunNotResumableError, "旧追问契约更新"):
            await service()._graph_input(graph, graph_config("child"), resume=True,
                query="取消字数限制", session_id="session", research_context=None, allowed_domains=None)


    async def test_same_session_new_run_cannot_read_parent_agent_raw_history(self):
        store = MemoryStore()
        parent = HistoryScope(store, "parent", "session", "user")
        child = HistoryScope(store, "child", "session", "user")
        row = await store.append(parent, "LeadWriter", "llm_output", {"content": "父运行历史"}, "old")
        with self.assertRaises(PermissionError):
            await store.read_source(child, "LeadWriter", row["source_id"])
        self.assertEqual((await store.load(child, "LeadWriter"))["messages"], [])

    async def test_deleted_router_environment_no_longer_changes_new_run(self):
        with patch.dict(os.environ, {"FOLLOWUP_ROUTER_ENABLED": "1"}):
            state, _, _ = await self.initial()
        self.assertNotIn("router_config", state)
        self.assertNotIn("followup_route", state)
        self.assertIn("session_parent", state)

    async def test_legacy_plain_graph_can_resume_but_removed_route_cannot(self):
        state, _, _ = await self.initial()
        kwargs = dict(resume=True, query="取消字数限制", session_id="session", research_context=None, allowed_domains=None)
        for version in ("single-main-v1", "single-main-v2-followup"):
            legacy = dict(state, graph_version=version)
            if version == "single-main-v2-followup":
                legacy["router_config"] = {"enabled": False}
            graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={"research": legacy}, next=("research",))))
            self.assertIsNone(await service()._graph_input(graph, graph_config("child"), **kwargs))
        for changes, next_nodes in (({"router_config": {"enabled": True}}, ("plan",)),
                                    ({}, ("followup_route",)), ({}, ("user_revision",))):
            legacy = dict(state, graph_version="single-main-v2-followup", **changes)
            graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={"research": legacy}, next=next_nodes)))
            with self.assertRaises(LegacyRunNotResumableError):
                await service()._graph_input(graph, graph_config("child"), **kwargs)


if __name__ == "__main__":
    unittest.main()
