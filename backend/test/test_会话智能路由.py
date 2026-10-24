"""会话任务、真实图入口与用户编辑的机制验证；模型和业务数据库使用替身。"""
import copy
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "test"), str(BACKEND / "eval/tests")]
from service.session_task_service import validate_decision, fallback_task, commit_requirements, report_hash, SessionTaskClarification
from service.research_context_service import ResearchContext, ResearchContextService
from service.deep_research_v2.session_task_execution import prepare_task
from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.state import create_initial_state, ResearchPhase
from service.deep_research_v2.graph import build_graph, graph_config, route_entry, pack_state
from service.deep_research_v2.service import DeepResearchV2Service
from langgraph.checkpoint.memory import InMemorySaver
from test_会话成果继承 import parent_snapshot, service
from test_智能体独立历史 import CharCounter
from test_单主链路图 import make_agents, FakeAgent, writes_report


def parent():
    snap = parent_snapshot()
    state = snap.values["research"]
    state["draft_sections"] = {x["id"]: "项目已投运，服务6000人。" * 60 for x in state["outline"]}
    state["final_report"] = "## 项目报告\n" + "项目已投运，服务6000人。" * 240
    return snap


def candidate(snap=None, ref="report:one", run="parent"):
    state = (snap or parent()).values["research"]
    return {"artifact_ref": ref, "run_id": run, "query": state["query"], "report_hash": report_hash(state["final_report"]),
            "sections": [{"id": s["id"], "title": s["title"]} for s in state["outline"]]}


def decision(query="压缩到1500字", kinds=("edit",), relation="continue", depends=True, ref="report:one", **extra):
    value = {"topic_relation": relation, "depends_on_history": depends, "target_artifact_ref": ref,
        "resolved_query": "编辑项目运营报告，" + query, "operations": [{"kind": x, "quote": query} for x in kinds],
        "revision_instructions": [query], **extra}
    return value


def task(query="压缩到1500字", **kwargs):
    return validate_decision(decision(query, **kwargs), query, [candidate()])


class DecisionTests(unittest.TestCase):
    def test_pure_edit_and_combined_evidence_choose_complete_paths(self):
        self.assertEqual(task()["execution_plan"], "edit")
        value = task("补证据并压缩", kinds=("evidence", "edit"), search_tasks=[
            {"query": "项目2024年运营", "target_section": "sec_0", "evidence_goal": "近期证据"}])
        self.assertEqual(value["execution_plan"], "augment")
        self.assertEqual(len(value["operations"]), 2)
        self.assertEqual(task("增加章节并缩短", kinds=("structure", "edit"))["execution_plan"], "replan")

    def test_same_topic_complete_question_remains_original(self):
        value = task("分析项目2024年运营", kinds=("research",), depends=False)
        self.assertEqual(value["resolved_query"], "分析项目2024年运营")
        self.assertEqual(value["topic_relation"], "continue")

    def test_new_topic_cannot_inherit_an_old_target_or_rewrite(self):
        value = task("分析光伏行业", relation="new", depends=False, ref=None, kinds=("research",))
        self.assertIsNone(value["selected_artifact"])
        self.assertEqual(value["resolved_query"], "分析光伏行业")
        with self.assertRaises(ValueError):
            task("分析光伏行业", relation="new", depends=False)

    def test_return_to_earlier_artifact_not_latest(self):
        old, recent = candidate(ref="old", run="old-run"), candidate(ref="latest", run="new-run")
        value = validate_decision(decision("回到最初报告", relation="return_to_previous", ref="old"),
                                  "回到最初报告", [old, recent])
        self.assertEqual(value["selected_artifact"]["run_id"], "old-run")

    def test_ambiguous_reference_and_missing_report_require_clarification(self):
        for kwargs in ({"relation": "uncertain"}, {"ref": None}, {"unresolved_references": ["第二个"]}):
            self.assertEqual(task("第二个再详细", **kwargs)["status"], "clarify")

    def test_unknown_artifact_and_invalid_requirement_rejected(self):
        with self.assertRaises(ValueError): task(ref="other-owner")
        for change in ({"op": "set", "field": "max_body_chars", "value": True, "quote": "压缩到1500字"},
                       {"op": "set", "field": "unknown", "value": "x", "quote": "压缩到1500字"}):
            with self.assertRaises(ValueError): task(requirement_changes=[change])

    def test_legacy_quote_mismatch_is_trace_only(self):
        value = decision(); value["operations"][0]["quote"] = "模型改写的引用"
        result = validate_decision(value, "压缩到1500字", [candidate()])
        self.assertEqual(result["execution_plan"], "edit")
        self.assertEqual(result["source_trace"]["entries"][0]["legacy_quote"], "模型改写的引用")

    def test_evidence_requires_explicit_search_tasks(self):
        with self.assertRaisesRegex(ValueError, "搜索任务"):
            task("补更多证据", kinds=("evidence",))

    def test_relative_shortening_does_not_invent_number(self):
        value = task("字数少一些")
        contract = commit_requirements({"active_constraints": {}, "contract_revision": 2}, value, "字数少一些")
        self.assertEqual(contract["active_constraints"], {})

    def test_requirement_set_and_removal_use_actual_baseline(self):
        base = {"contract_revision": 3, "time_range": "2023—2024", "active_constraints": {"max_body_chars": 3000}}
        value = task(requirement_changes=[{"op": "set", "field": "max_body_chars", "value": 1500, "quote": "压缩到1500字"}])
        updated = commit_requirements(base, value, "压缩到1500字")
        self.assertEqual(updated["active_constraints"], {"max_body_chars": 1500})
        self.assertEqual(updated["time_range"], "2023—2024")
        self.assertEqual(base["active_constraints"], {"max_body_chars": 3000})
        value = task("取消字数限制", requirement_changes=[{"op": "remove", "field": "max_body_chars", "quote": "取消字数限制"}])
        self.assertEqual(commit_requirements(base, value, "取消字数限制")["active_constraints"], {})

    def test_failed_classifier_does_not_guess_implicit_target(self):
        self.assertEqual(fallback_task("这份报告缩短", "模型不可用")["status"], "clarify")
        self.assertEqual(fallback_task("分析光伏行业2025年竞争格局", "模型不可用")["status"], "fallback")


class ResolverTests(unittest.TestCase):
    def resolve(self, value, query):
        context = ResearchContext(query, query, artifact_candidates=[candidate()])
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value, ensure_ascii=False)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))))
        resolver = ResearchContextService.__new__(ResearchContextService); resolver.token_counter = CharCounter()
        with patch("service.research_context_service.create_openai_client", return_value=client):
            resolver._resolve_query(context)
        return context, client.chat.completions.create

    def test_one_call_returns_topic_rewrite_and_task(self):
        context, call = self.resolve(decision(), "压缩到1500字")
        self.assertEqual(context.session_task["execution_plan"], "edit")
        self.assertTrue(context.is_follow_up)
        self.assertEqual(call.call_count, 1)
        self.assertIn("elapsed_ms", context.task_metrics)

    def test_invalid_output_is_recorded_and_requests_clarification(self):
        context, _ = self.resolve({"wrong": True}, "这份报告缩短")
        self.assertEqual(context.session_task["status"], "clarify")
        self.assertEqual(context.resolved_query, context.original_query)

    def test_builder_new_topic_clears_old_context_and_counts(self):
        resolver = ResearchContextService.__new__(ResearchContextService)
        resolver.token_counter = CharCounter(); resolver.history_token_limit = 3000; resolver.memory_token_limit = 2000
        resolver._load_history = Mock(return_value=SimpleNamespace(messages=[{"role": "user", "content": "旧主题"}],
            cache_hit=False, last_message_id="old", compacted_tokens=3, original_tokens=3,
            summarized_messages=0, pruned_tool_results=0, semantic_context_version=0))
        resolver._load_memories = Mock(return_value=[])
        def resolve(context):
            context.session_task = task("分析光伏", kinds=("research",), relation="new", depends=False, ref=None)
        resolver._resolve_query = resolve
        with patch("service.session_task_service.load_artifact_candidates", return_value=[candidate()]):
            result = resolver.build(user_id="user", session_id="session", current_turn_id="turn", current_query="分析光伏")
        self.assertEqual(result["conversation_context"], [])
        self.assertEqual(result["token_counts"]["conversation"], 0)
        self.assertEqual(result["history_metrics"]["router_messages"], 1)
        self.assertEqual(result["history_metrics"]["execution_messages"], 0)
        self.assertTrue(result["history_metrics"]["filtered_new_topic"])

    def test_task_judgment_participates_in_shared_request_budget(self):
        from service.verification_budget import task_budget
        config = {"max_seconds": 60, "max_requests": 1, "max_tokens_by_model": {"test": 100000}}
        with patch.dict(os.environ, {"SESSION_TASK_MODEL": "test"}), task_budget(config) as budget:
            self.resolve(decision(), "压缩到1500字")
            self.assertEqual(len(budget.requests), 1)
            self.assertEqual(budget.requests[0]["component"], "session_task")
            with self.assertRaisesRegex(ValueError, "预算"):
                budget.reserve("test", 1, 1)

    def test_exhausted_judgment_budget_cannot_fall_back(self):
        from service.verification_budget import task_budget
        from service.session_task_service import SessionTaskBudgetExceeded
        config = {"max_seconds": 60, "max_requests": 1, "max_tokens_by_model": {"test": 100000}}
        with patch.dict(os.environ, {"SESSION_TASK_MODEL": "test"}), task_budget(config) as budget:
            budget.reserve("test", 1, 1)
            with self.assertRaises(SessionTaskBudgetExceeded): self.resolve(decision(), "压缩到1500字")


class InputTests(unittest.IsolatedAsyncioTestCase):
    async def initial(self, value, snap=None, domains=None):
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=snap or parent()))
        with patch("service.session_task_service.authorize_artifact") as authorize, patch(
            "service.qwen_token_counter.get_qwen_token_counter", return_value=CharCounter()
        ), patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": "", "AGENT_COMPRESSION_CONFIG": ""}):
            output = await service()._graph_input(graph, {}, resume=False, query=value["operations"][0]["quote"],
                session_id="session", user_id="user", run_id="child", allowed_domains=domains,
                research_context={"session_task": value, "conversation_context": [{"role": "user", "content": "旧主题"}]})
        return output["research"], graph, authorize

    async def test_edit_commits_requirements_without_old_runtime_state(self):
        state, _, authorize = await self.initial(task(requirement_changes=[
            {"op": "set", "field": "max_body_chars", "value": 1500, "quote": "压缩到1500字"}]))
        self.assertEqual(state["execution_plan"], "edit")
        self.assertEqual(state["active_constraints"], {"max_body_chars": 1500})
        self.assertNotIn("pending_contract_update", state["research_contract"])
        self.assertEqual(state["writing_mode"], "user_edit")
        self.assertEqual(state["iteration"], 0)
        self.assertNotIn("node_attempts", state)
        self.assertEqual(state["quality_verdict"], "not_reviewed")
        authorize.assert_called_once()

    async def test_new_topic_does_not_load_parent_or_old_context(self):
        value = task("分析光伏", relation="new", depends=False, ref=None, kinds=("research",))
        state, graph, authorize = await self.initial(value)
        self.assertEqual(state["conversation_context"], [])
        self.assertEqual(state["allowed_domains"], [])
        self.assertEqual(state["active_constraints"], {})
        self.assertNotIn("session_parent", state)
        graph.aget_state.assert_not_awaited(); authorize.assert_not_called()

    async def test_evidence_has_targeted_tasks_and_preserved_materials(self):
        value = task("补证据", kinds=("evidence",), search_tasks=[
            {"query": "项目2024年运营", "target_section": "sec_0", "evidence_goal": "近期数据"}])
        state, _, _ = await self.initial(value)
        self.assertEqual(state["execution_plan"], "augment")
        self.assertEqual(state["pending_research_tasks"][0]["target_section"], "sec_0")
        self.assertEqual(state["pending_research_tasks"][0]["origin"], "user_followup")
        self.assertTrue(state["raw_sources"])

    async def test_bad_target_section_cannot_search(self):
        with self.assertRaisesRegex(ValueError, "章节ID"):
            task("补证据", kinds=("evidence",), search_tasks=[
                {"query": "项目", "target_section": "不存在的章节", "evidence_goal": "证据"}])

    async def test_incomplete_old_section_materials_fall_back_to_plan(self):
        snap = parent(); snap.values["research"]["draft_sections"] = {}
        value = task(); value["selected_artifact"] = candidate(snap)
        state, _, _ = await self.initial(value, snap)
        self.assertEqual(state["execution_plan"], "replan")
        self.assertIn("session_parent", state)
        self.assertIn("route_fallback_reason", state)

    async def test_scope_change_forces_planning_even_if_classified_as_edit(self):
        value = task("仅研究2025年", requirement_changes=[
            {"op": "set", "field": "time_range", "value": "2025年", "quote": "仅研究2025年"}])
        state, _, _ = await self.initial(value)
        self.assertEqual(state["execution_plan"], "replan")

    async def test_report_checkpoint_hash_mismatch_is_rejected(self):
        value = task(); value["selected_artifact"]["report_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "不一致"): await self.initial(value)

    async def test_unauthorized_artifact_rejected_before_checkpoint_read(self):
        with patch("service.session_task_service.authorize_artifact", side_effect=PermissionError("无权限")), patch(
            "service.qwen_token_counter.get_qwen_token_counter", return_value=CharCounter()):
            graph = SimpleNamespace(aget_state=AsyncMock())
            with self.assertRaises(PermissionError):
                await service()._graph_input(graph, {}, resume=False, query="压缩到1500字", session_id="session",
                    user_id="user", run_id="child", allowed_domains=None, research_context={"session_task": task()})
            graph.aget_state.assert_not_awaited()

    async def test_resume_verifies_task_hash(self):
        state, _, _ = await self.initial(task())
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={"research": state}, next=("edit",))))
        self.assertIsNone(await service()._graph_input(graph, {}, resume=True, query="忽略这次输入", session_id="session",
            research_context=None, allowed_domains=None))
        state["session_task"]["execution_plan"] = "augment"
        with self.assertRaisesRegex(ValueError, "哈希"):
            await service()._graph_input(graph, {}, resume=True, query="", session_id="session", research_context=None, allowed_domains=None)


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def run_plan(self, plan):
        agents = make_agents(writer=FakeAgent("LeadWriter", writes_report))
        phases = []
        async def execute(action, state, phase):
            phases.append(phase)
            return await action(state)
        saver = InMemorySaver()
        graph = build_graph(agents, saver, run_id="route-" + plan, user_id=None, session_id="session", execute_stage=execute)
        state = create_initial_state("任务", "session", [])
        state["execution_plan"] = plan
        await graph.ainvoke(pack_state(state), graph_config("route-" + plan))
        final = await graph.aget_state(graph_config("route-" + plan))
        self.assertEqual(final.values["research"]["status"], "completed")
        return agents, phases

    async def test_edit_skips_plan_and_search_but_reviews(self):
        agents, phases = await self.run_plan("edit")
        self.assertEqual(phases, ["writing", "reviewing"])
        self.assertFalse(agents["architect"].calls); self.assertFalse(agents["scout"].calls)

    async def test_augment_skips_architect_and_includes_analyze_write_review(self):
        agents, phases = await self.run_plan("augment")
        self.assertEqual(phases, ["re_researching", "analyzing", "writing", "reviewing"])
        self.assertFalse(agents["architect"].calls)

    async def test_new_research_uses_full_path(self):
        _, phases = await self.run_plan("replan")
        self.assertEqual(phases, ["init", "researching", "analyzing", "writing", "reviewing"])

    def test_unknown_execution_plan_is_rejected(self):
        with self.assertRaises(ValueError): route_entry({"research": {"execution_plan": "arbitrary-node"}})

    async def test_edit_failure_resumes_edit_without_replanning(self):
        calls = {"n": 0}
        def write(state):
            calls["n"] += 1
            if calls["n"] == 1: raise RuntimeError("注入编辑中断")
            writes_report(state)
        agents = make_agents(writer=FakeAgent("LeadWriter", write))
        async def execute(action, state, phase): return await action(state)
        graph = build_graph(agents, InMemorySaver(), run_id="edit-resume", user_id=None,
                            session_id="session", execute_stage=execute)
        config = graph_config("edit-resume")
        state = create_initial_state("缩短", "session", []); state["execution_plan"] = "edit"
        with self.assertRaises(RuntimeError): await graph.ainvoke(pack_state(state), config)
        self.assertEqual((await graph.aget_state(config)).next, ("edit",))
        await graph.ainvoke(None, config)
        self.assertEqual(calls["n"], 2)
        self.assertFalse(agents["architect"].calls)
        self.assertFalse(agents["scout"].calls)


class ClarificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_preparation_failure_does_not_start_research_without_history(self):
        from fastapi import HTTPException
        from router.research_router import stream_research, ResearchRequest
        from service.semantic_context_service import SemanticContextError
        import uuid
        for failure in (SemanticContextError("摘要失败"), ValueError("摘要状态格式损坏")):
            with self.subTest(failure=type(failure).__name__):
                request = ResearchRequest(query="把上一份缩短", session_id=uuid.uuid4(), turn_id=uuid.uuid4())
                conversations = Mock()
                conversations.save_user_message.return_value = SimpleNamespace(content=request.query)
                conversations.get_turn_message.return_value = None
                builder = Mock(); builder.build.side_effect = failure
                runtime = Mock(); runtime.resolve_allowed_domains.return_value = []
                with patch.dict(os.environ, {"CONTEXT_COMPRESSION_ENABLED": "1", "SEMANTIC_CONTEXT_ENABLED": "true", "CONTEXT_ABLATION_CONFIG": ""}), patch(
                    "router.research_router.get_research_service", return_value=runtime), patch(
                    "router.research_router.get_conversation_service", return_value=conversations), patch(
                    "router.research_router.get_research_context_service", return_value=builder), patch(
                    "router.research_router.acquire_research_run_lease") as lease:
                    with self.assertRaises(HTTPException) as raised:
                        await stream_research(request, Mock(), SimpleNamespace(id=uuid.uuid4()), Mock())
                self.assertEqual(raised.exception.status_code, 503)
                lease.assert_not_called()
                runtime.research.assert_not_called()

    async def test_http_clarification_saved_without_starting_run(self):
        from router.research_router import stream_research, ResearchRequest
        import uuid
        request = ResearchRequest(query="第二个再详细", session_id=uuid.uuid4(), turn_id=uuid.uuid4())
        user = SimpleNamespace(id=uuid.uuid4())
        conversations = Mock()
        conversations.save_user_message.return_value = SimpleNamespace(content=request.query)
        conversations.get_turn_message.return_value = None
        conversations.save_assistant_message.return_value = SimpleNamespace(research_status="clarification", content="请明确报告。", turn_id="turn")
        trace_metrics = {"task_trace": {"raw_response": '{"wrong":true}', "validation_error": "任务结构无效"}}
        builder = Mock(); builder.build.return_value = {"session_task": task(request.query, relation="uncertain"),
                                                       "task_metrics": trace_metrics}
        runtime = Mock(); runtime.resolve_allowed_domains.return_value = []
        with patch("router.research_router.get_research_service", return_value=runtime), patch(
            "router.research_router.get_conversation_service", return_value=conversations), patch(
            "router.research_router.get_research_context_service", return_value=builder), patch(
            "router.research_router.acquire_research_run_lease") as lease:
            response = await stream_research(request, Mock(), user, Mock())
        events = [x async for x in response.body_iterator]
        self.assertEqual(json.loads(events[0][6:])["type"], "session_clarification")
        lease.assert_not_called(); runtime.research.assert_not_called()
        self.assertEqual(conversations.save_assistant_message.call_args.kwargs["research_status"], "clarification")
        self.assertEqual(conversations.save_assistant_message.call_args.kwargs["references_data"]["session_task_metrics"],
                         trace_metrics)

    async def test_replay_clarification_never_reports_research_complete(self):
        from router.research_router import _replay_research_message
        message = SimpleNamespace(research_status="clarification", content="请明确报告。", turn_id="turn")
        chunks = [x async for x in _replay_research_message(message)]
        event = json.loads(chunks[0][6:])
        self.assertEqual(event["type"], "session_clarification")
        self.assertNotIn("research_complete", chunks[0])


class WriterTests(unittest.IsolatedAsyncioTestCase):
    def state(self):
        from service.deep_research_v2.followup_context import parent_artifacts
        state = create_initial_state("压缩到500字", "session", [])
        value = task("压缩到500字", requirement_changes=[
            {"op": "set", "field": "max_body_chars", "value": 500, "quote": "压缩到500字"}])
        prepare_task(state, value, parent_artifacts(parent(), "session"))
        state["phase"] = "writing"
        return state

    def writer(self, result):
        value = LeadWriter.__new__(LeadWriter); value.name = "LeadWriter"
        value.logger = logging.getLogger("session-edit"); value.add_message = Mock()
        value.call_llm = AsyncMock(return_value=json.dumps(result, ensure_ascii=False))
        return value

    def result(self):
        return {"sections": [{"id": f"sec_{i}", "title": f"章节{i}", "content": "项目已投运，服务6000人。", "claims": []} for i in range(4)],
                "executive_summary": "", "conclusions": [], "outlook": ""}

    async def test_large_compression_updates_sections_and_report_without_critic_feedback(self):
        state = self.state(); before = copy.deepcopy(state["draft_sections"])
        writer = self.writer(self.result())
        with patch("service.deep_research_v2.citation_verification.verify_stage_citations", new=AsyncMock(return_value={"status": "not_executed"})):
            output = await writer.process(state)
        self.assertEqual(output["phase"], "reviewing")
        self.assertEqual(output["critic_feedback"], [])
        self.assertLess(output["user_edit_result"]["after_body_chars"], 500)
        self.assertTrue(all(len(v) < len(before[k]) * .55 for k, v in output["draft_sections"].items()))
        self.assertEqual(writer.call_llm.await_count, 1)

    @patch.dict(os.environ, {"REPORT_BODY_LENGTH_HARD_CHECK_ENABLED": "true"})
    async def test_invalid_section_ids_or_overlong_result_do_not_commit(self):
        for kind in ("missing", "long"):
            state = self.state(); before = copy.deepcopy(state)
            result = self.result()
            if kind == "missing": result["sections"].pop()
            else: result["sections"][0]["content"] = "长" * 1000
            writer = self.writer(result)
            with self.subTest(kind=kind), self.assertRaises(ValueError): await writer.process(state)
            self.assertEqual(state, before)
            self.assertEqual(writer.call_llm.await_count, 2)

    async def test_critic_rework_after_compression_keeps_short_report_limit(self):
        state = self.state(); state["phase"] = "revising"
        state["critic_feedback"] = [{"id": "issue1", "description": "保留服务人口", "resolved": False}]
        writer = self.writer(self.result())
        with patch("service.deep_research_v2.citation_verification.verify_stage_citations", new=AsyncMock(return_value={"status": "not_executed"})):
            result = await writer.process(state)
        self.assertLess(result["user_edit_result"]["after_body_chars"], 500)
        self.assertIn("保留服务人口", writer.call_llm.call_args.kwargs["user_prompt"])


@unittest.skipUnless(os.getenv("TEST_AGENT_HISTORY_POSTGRES") == "1", "显式启用隔离PostgreSQL成果授权测试")
class ArtifactPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import uuid
        from datetime import datetime, timezone, timedelta
        from sqlalchemy import text
        from models.chat import ChatSession, ChatMessage
        from test_智能体独立历史 import PostgresHistoryTests
        PostgresHistoryTests.setUpClass.__func__(cls)
        ChatSession.__table__.create(cls.engine)
        ChatMessage.__table__.create(cls.engine)
        cls.owner = uuid.uuid4(); cls.other = uuid.uuid4(); cls.session_id = uuid.uuid4()
        with cls.sessions.begin() as db:
            db.execute(text("INSERT INTO users(id) VALUES(:a),(:b)"), {"a": cls.owner, "b": cls.other})
            db.add(ChatSession(id=cls.session_id, user_id=cls.owner)); db.flush()
            for i in range(32):
                turn = uuid.uuid4(); when = datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
                db.add(ChatMessage(session_id=cls.session_id, turn_id=turn, role="user", content=f"课题{i}", created_at=when))
                db.add(ChatMessage(session_id=cls.session_id, turn_id=turn, role="assistant", content=f"## 报告{i}\n材料",
                    execution_mode="research", research_status="completed", run_id=f"artifact{i}", created_at=when,
                    references_data={"outline_sections": [{"id": "sec_0", "title": f"报告{i}"}]}))
            db.add(ChatMessage(session_id=cls.session_id, role="assistant", content="未交付",
                execution_mode="research", research_status="failed", run_id="failed-artifact"))

    @classmethod
    def tearDownClass(cls):
        from test_智能体独立历史 import PostgresHistoryTests
        PostgresHistoryTests.tearDownClass.__func__(cls)

    def test_candidates_keep_earliest_recent_and_exclude_current_failed(self):
        from service.session_task_service import load_artifact_candidates
        with patch("core.database.SessionLocal", self.sessions):
            rows = load_artifact_candidates(str(self.owner), str(self.session_id), "artifact31")
        runs = {x["run_id"] for x in rows}
        self.assertIn("artifact0", runs); self.assertIn("artifact30", runs)
        self.assertNotIn("artifact31", runs); self.assertNotIn("failed-artifact", runs)
        self.assertEqual(rows[0]["query"], "课题0")
        self.assertLessEqual(len(rows), 28)

    def test_authorized_report_and_hash_are_checked_in_database(self):
        from service.session_task_service import authorize_artifact
        with patch("core.database.SessionLocal", self.sessions):
            authorize_artifact(str(self.owner), str(self.session_id), "artifact0", report_hash("## 报告0\n材料"))
            with self.assertRaises(PermissionError): authorize_artifact(str(self.owner), str(self.session_id), "artifact0", "wrong")

    def test_wrong_owner_or_session_cannot_select_report(self):
        import uuid
        from service.session_task_service import authorize_artifact, load_artifact_candidates
        with patch("core.database.SessionLocal", self.sessions):
            self.assertEqual(load_artifact_candidates(str(self.other), str(self.session_id), "new"), [])
            for owner, session in ((self.other, self.session_id), (self.owner, uuid.uuid4())):
                with self.assertRaises(PermissionError): authorize_artifact(str(owner), str(session), "artifact0", report_hash("## 报告0\n材料"))

    def test_unfinished_report_is_not_authorized(self):
        from service.session_task_service import authorize_artifact
        with patch("core.database.SessionLocal", self.sessions), self.assertRaises(PermissionError):
            authorize_artifact(str(self.owner), str(self.session_id), "failed-artifact", report_hash("未交付"))


if __name__ == "__main__":
    unittest.main()
