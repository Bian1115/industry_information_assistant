"""入口错位回归：历史问题不能替代当前动作，重判与预算必须完整保留。"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "test"), str(BACKEND / "eval/tests")]
from service.research_context_service import ResearchContext, ResearchContextService
from service.session_task_service import SessionTaskBudgetExceeded
from service.verification_budget import task_budget
from test_会话智能路由 import candidate, decision
from test_智能体独立历史 import CharCounter

OLD = "检查连接和会话概念是否混用。"
CURRENT = "我调整一下工程选择：改用TypeScript，删除Python步骤；未知接口保留待核实。"


def response(value):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value, ensure_ascii=False)))],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def value(query=CURRENT, kinds=("edit",)):
    return decision(query, kinds=kinds, resolved_query=query)


def context(query=CURRENT):
    item = candidate(); item["query"] = OLD
    return ResearchContext(query, query, conversation_context=[
        {"role": "user", "content": OLD}, {"role": "assistant", "content": "旧稿：连接和会话说明"}],
        artifact_candidates=[item])


def resolve(ctx, values):
    call = Mock(side_effect=[response(v) for v in values])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call)))
    service = ResearchContextService.__new__(ResearchContextService); service.token_counter = CharCounter()
    with patch("service.research_context_service.create_openai_client", return_value=client):
        service._resolve_query(ctx)
    return call


class CurrentQueryTests(unittest.TestCase):
    def test_current_request_is_the_final_message_after_old_material(self):
        ctx = context(); call = resolve(ctx, [value()])
        messages = call.call_args.kwargs["messages"]
        self.assertIn(OLD, messages[1]["content"])
        self.assertIn(CURRENT, messages[-1]["content"])
        self.assertNotIn(OLD, messages[-1]["content"])
        self.assertIn("revision_instructions", messages[-1]["content"])
        self.assertEqual(ctx.session_task["execution_plan"], "edit")
        self.assertEqual(call.call_count, 1)

    def test_stale_query_regenerates_operations_and_instructions_not_just_query(self):
        ctx = context(); stale = value(OLD, kinds=("research",))
        corrected = value(); corrected["revision_instructions"] = [CURRENT]
        call = resolve(ctx, [stale, corrected])
        self.assertEqual(call.call_count, 2)
        self.assertEqual(ctx.session_task["resolved_query"], CURRENT)
        self.assertEqual(ctx.session_task["execution_plan"], "edit")
        self.assertEqual(ctx.session_task["revision_instructions"], [CURRENT])
        attempts = ctx.task_trace["query_alignment_attempts"]
        self.assertEqual(attempts[0]["parsed_response"], stale)
        self.assertIsNotNone(attempts[0]["historical_query_source"])
        self.assertIsNone(attempts[1]["historical_query_source"])
        self.assertEqual(ctx.task_metrics["input_tokens"], 200)
        self.assertEqual(ctx.task_metrics["output_tokens"], 40)

    def test_second_stale_output_clarifies_and_preserves_history_dependency(self):
        ctx = context(); call = resolve(ctx, [value(OLD), value(OLD)])
        self.assertEqual(call.call_count, 2)
        self.assertEqual(ctx.session_task["status"], "clarify")
        self.assertTrue(ctx.session_task["depends_on_history"])
        self.assertEqual(ctx.resolved_query, CURRENT)
        self.assertIn("validation_error", ctx.task_trace)

    def test_invalid_retry_cannot_turn_a_known_stale_task_into_new_research(self):
        ctx = context(); call = resolve(ctx, [value(OLD), {"wrong": True}])
        self.assertEqual(call.call_count, 2)
        self.assertEqual(ctx.session_task["status"], "clarify")
        self.assertTrue(ctx.session_task["depends_on_history"])
        self.assertIsNone(ctx.session_task.get("selected_artifact"))

    def test_identical_query_repeated_by_user_is_allowed(self):
        ctx = context(OLD); call = resolve(ctx, [value(OLD)])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(ctx.session_task["status"], "ready")

    def test_continue_can_expand_an_object_without_copying_old_action(self):
        ctx = context("继续")
        result = value("继续检查现有工程说明中的连接和会话边界。")
        call = resolve(ctx, [result])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(ctx.session_task["resolved_query"], result["resolved_query"])

    def test_stale_artifact_query_is_detected_when_raw_history_is_summarized(self):
        ctx = context(); ctx.conversation_context = [{"role": "system", "content": "旧对话摘要"}]
        call = resolve(ctx, [value(" \n" + OLD + "\n"), value()])
        self.assertEqual(call.call_count, 2)
        self.assertTrue(ctx.task_trace["query_alignment_attempts"][0]["historical_query_source"].startswith("artifact_candidates"))

    def test_alignment_retry_shares_budget_and_cannot_silently_fall_back(self):
        ctx = context()
        config = {"max_seconds": 60, "max_requests": 1, "max_tokens_by_model": {"test": 100000}}
        with patch.dict(os.environ, {"SESSION_TASK_MODEL": "test"}), task_budget(config) as budget:
            with self.assertRaises(SessionTaskBudgetExceeded):
                resolve(ctx, [value(OLD), value()])
            self.assertEqual(len(budget.requests), 1)
            self.assertEqual(budget.requests[0]["component"], "session_task")

    def test_both_requests_are_charged_to_shared_budget(self):
        ctx = context()
        config = {"max_seconds": 60, "max_requests": 2, "max_tokens_by_model": {"test": 100000}}
        with patch.dict(os.environ, {"SESSION_TASK_MODEL": "test"}), task_budget(config) as budget:
            resolve(ctx, [value(OLD), value()])
            self.assertEqual([r["component"] for r in budget.requests], ["session_task", "session_task_query_alignment"])
            self.assertTrue(all(r["execution_status"] == "completed" for r in budget.requests))


if __name__ == "__main__":
    unittest.main()
