"""异常判定与证据入库、整篇预算与超限缩写的离线机制验证。"""

import asyncio
import copy
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "app"), str(Path(__file__).resolve().parent)]

import httpx
from openai import APITimeoutError, RateLimitError
from config.llm_config import ResearchConfig
from service.agent_context import ContextBudgetExceeded, is_evidence_batch_input
from service.agent_history import AgentHistoryError
from service.deep_research_v2.report_constraints import body_character_count
from service.deep_research_v2.state import ResearchPhase
from service.web_document import source_usable
from service.web_reader_service import WebReadResult
from test_正文质量与编号证据 import scout
from test_Writer证据去重与程序参考文献 import state_fixture, writer_fixture, URL

OBSERVATIONS = []


def page():
    return WebReadResult(url=URL, final_url=URL, content="仅供预览\n项目2023年投运。", fetch_status="success")


class ReviewClassificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_errors_keep_fetch_success_and_uncertain_quality(self):
        request = httpx.Request("POST", "https://model.example/v1")
        errors = [
            (ContextBudgetExceeded(100, 200), "input_budget_exceeded"),
            (APITimeoutError(request=request), "model_timeout"),
            (RateLimitError("限流", response=httpx.Response(429, request=request), body=None), "model_rate_limited"),
            (httpx.ConnectError("连接失败", request=request), "model_connection_error"),
            (RuntimeError("意外内部错误"), "review_internal_error"),
        ]
        for error, category in errors:
            with self.subTest(category=category):
                agent, result = scout(), page()
                original = result.content
                agent.call_llm = AsyncMock(side_effect=error)
                await agent._check_source_body(result, {})
                self.assertTrue(result.success)
                self.assertEqual(result.content, original)
                self.assertIsNone(result.failure_category)
                self.assertEqual(result.body_quality["status"], "uncertain")
                self.assertFalse(source_usable({"content": result.content, "body_quality": result.body_quality}))
                review = result.body_quality["model_review"]
                self.assertEqual(review["execution_status"], "failed")
                self.assertEqual(review["error_category"], category)
                self.assertGreaterEqual(review["duration_ms"], 0)
                self.assertFalse(is_evidence_batch_input())
                OBSERVATIONS.append({"error": category, "fetch_status": result.fetch_status,
                                     "body_status": result.body_quality["status"]})

    async def test_empty_passed_selection_is_invalid_response_not_page_rejection(self):
        agent, result = scout(), page()
        agent.call_llm = AsyncMock(return_value='{"status":"passed","body_block_ids":[]}')
        await agent._check_source_body(result, {})
        self.assertEqual(result.body_quality["model_review"]["error_category"], "invalid_response")
        self.assertTrue(result.success)

    async def test_valid_rejection_remains_page_rejection(self):
        agent, result = scout(), page()
        agent.call_llm = AsyncMock(return_value='{"status":"rejected","reason":"目标正文缺失","body_block_ids":[]}')
        await agent._check_source_body(result, {})
        self.assertFalse(result.success)
        self.assertEqual(result.failure_category, "body_quality_rejected")
        self.assertEqual(result.body_quality["model_review"]["execution_status"], "succeeded")

    async def test_cancellation_and_history_failures_propagate(self):
        for error in (asyncio.CancelledError(), AgentHistoryError("模拟归档失败")):
            agent, result = scout(), page()
            agent.call_llm = AsyncMock(side_effect=error)
            with self.assertRaises(type(error)):
                await agent._check_source_body(result, {})
            self.assertFalse(is_evidence_batch_input())

    async def test_failed_review_is_not_admitted_and_is_counted_separately(self):
        agent, result = scout(), page()
        agent.call_llm = AsyncMock(side_effect=TimeoutError("复核超时"))
        agent.web_reader = object()
        agent._ensure_tool_runtime = lambda: None
        agent.tool_definitions = {"web_read": object()}
        agent.tool_runtime = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(value=[result.to_dict()])))
        state = {"allowed_domains": ["official.example.cn"], "raw_sources": []}
        items = [{"url": URL}]
        await agent._enrich_results_with_full_text(state, items, "sec_1")
        self.assertEqual(state["raw_sources"], [])
        self.assertFalse(items[0]["reader_verified"])
        self.assertNotIn("full_content", items[0])
        stats = state["web_reader_stats"]
        self.assertEqual(stats["failure_categories"]["body_quality_rejected"], 0)
        self.assertEqual(stats["body_reviews_failed"], 1)
        self.assertEqual(stats["urls_body_admitted"], 0)
        self.assertEqual(stats["urls_body_pending"], 1)
        self.assertEqual(stats["body_review_error_categories"], {"model_timeout": 1})

    async def test_retry_of_same_page_can_be_admitted_after_valid_review(self):
        agent, result = scout(), page()
        agent.call_llm = AsyncMock(side_effect=[TimeoutError("复核超时"),
            '{"status":"partial","body_block_ids":["b000002"]}'])
        await agent._check_source_body(result, {})
        await agent._check_source_body(result, {})
        self.assertTrue(result.success)
        self.assertEqual(result.content, "项目2023年投运。")
        self.assertEqual(result.body_quality["model_review"]["execution_status"], "succeeded")
        self.assertNotIn("quality_review_failed", result.body_quality["reasons"])


def writing_state(limit=2400, count=4):
    state = state_fixture()
    state["outline"] = [{"id": f"sec_{i}", "title": f"工程说明第{i}章", "status": "pending"} for i in range(1, count + 1)]
    state["active_constraints"] = {} if limit is None else {"max_body_chars": limit}
    state["phase"] = ResearchPhase.WRITING.value
    return state


def writer():
    value = writer_fixture()
    value.parse_json_response = json.loads
    value.add_message = lambda *args, **kwargs: None
    return value


def shortened(state, size=300):
    return {"sections": [{"id": s["id"], "content": "保留已核实的项目事实。" + "文" * size + f"[官方来源]({URL})",
                          "claims": []} for s in state["outline"]],
            "executive_summary": "项目已经投运。", "conclusions": ["按原始证据解释。"],
            "outlook": "继续核对适用边界。", "claims": []}


class ReportBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_defaults_are_300_to_500_in_config_and_writer(self):
        self.assertEqual((ResearchConfig().min_section_chars, ResearchConfig().max_section_chars), (300, 500))
        self.assertEqual((writer().min_section_chars, writer().max_section_chars), (300, 500))

    def test_total_budget_allocates_all_parts_and_can_override_default_minimum(self):
        value = writer()
        for limit in (2400, 1200, 500):
            state = writing_state(limit)
            plan = value._writing_budget(state)
            allocated = sum(x["max_chars"] for x in plan["sections"].values())
            allocated += sum(plan[k] for k in ("summary_chars", "conclusion_chars", "outlook_chars", "structure_chars", "safety_chars"))
            self.assertLessEqual(allocated, limit)
            self.assertTrue(all(1 <= x["min_chars"] <= x["max_chars"] <= 500 for x in plan["sections"].values()))
            if limit == 500:
                self.assertTrue(all(x["max_chars"] < 300 for x in plan["sections"].values()))
            OBSERVATIONS.append({"limit": limit, "allocated_total_chars": allocated, "plan": plan})
        self.assertEqual(value.target_report_chars, 7500)

    async def test_first_writing_prompts_use_task_budget_and_oversized_report_is_repaired(self):
        state, value = writing_state(), writer()
        section = json.dumps({"content": "正文" * 400 + f"[来源]({URL})", "claims": []})
        synthesis = json.dumps({"executive_summary": "摘要", "conclusions": [], "outlook": "展望", "claims": []})
        value.call_llm = AsyncMock(side_effect=[section] * 4 + [synthesis, json.dumps(shortened(state), ensure_ascii=False)])
        await value._write_report(state)
        plan = state["writing_budget"]
        maximum = plan["sections"]["sec_1"]["max_chars"]
        for call in value.call_llm.call_args_list[:4]:
            self.assertIn(f"300-{maximum} 字", call.kwargs["user_prompt"])
        self.assertIn(f"合计最多 {plan['conclusion_chars']} 字", value.call_llm.call_args_list[4].kwargs["user_prompt"])
        self.assertLessEqual(body_character_count(state["final_report"]), 2400)
        self.assertEqual(state["body_length_repair"]["attempts"], 1)
        self.assertGreater(state["body_length_repair"]["before_body_chars"], 2400)
        self.assertEqual(len(state["draft_sections"]), 4)
        self.assertIn(URL, state["final_report"])
        self.assertEqual(state["phase"], ResearchPhase.REVIEWING.value)
        OBSERVATIONS.append({"first_writing_repair": state["body_length_repair"]})

    async def test_under_limit_synthesis_does_not_make_repair_call(self):
        state, value = writing_state(), writer()
        state["draft_sections"] = {s["id"]: f"项目已投运[来源]({URL})。" for s in state["outline"]}
        value.call_llm = AsyncMock(return_value='{"executive_summary":"摘要","conclusions":[],"outlook":"展望","claims":[]}')
        await value._synthesize_report(state)
        value.call_llm.assert_awaited_once()
        self.assertEqual(state["body_length_repair"]["attempts"], 0)
        self.assertEqual(state["body_length_repair"]["status"], "not_needed")

    async def test_two_failed_repairs_do_not_publish_or_replace_original_sections(self):
        state, value = writing_state(800), writer()
        state["draft_sections"] = {s["id"]: "原文" * 1000 for s in state["outline"]}
        original = copy.deepcopy(state["draft_sections"])
        state["final_report"] = value._assemble_report(state, "摘要", [], "")
        value.call_llm = AsyncMock(return_value=json.dumps(shortened(state, 900), ensure_ascii=False))
        with self.assertRaisesRegex(ValueError, "报告缩写未满足正文上限"):
            await value._fit_report_to_budget(state, "摘要", [], "")
        self.assertEqual(value.call_llm.await_count, 2)
        self.assertEqual(state["final_report"], "")
        self.assertEqual(state["draft_sections"], original)
        self.assertEqual(state["body_length_repair"]["status"], "failed")

    async def test_missing_sections_are_rejected_before_valid_repair_is_committed(self):
        state, value = writing_state(), writer()
        state["draft_sections"] = {s["id"]: "正文" * 1000 for s in state["outline"]}
        state["final_report"] = value._assemble_report(state, "摘要", [], "")
        broken = shortened(state)
        broken["sections"].pop()
        value.call_llm = AsyncMock(side_effect=[json.dumps(broken), json.dumps(shortened(state))])
        await value._fit_report_to_budget(state, "摘要", [], "")
        self.assertEqual(state["body_length_repair"]["attempts"], 2)
        self.assertEqual(set(state["draft_sections"]), {s["id"] for s in state["outline"]})

    async def test_revision_can_shrink_old_long_section_under_user_budget(self):
        state, value = writing_state(800, 1), writer()
        state["draft_sections"] = {"sec_1": "原文" * 1500}
        state["critic_feedback"] = [{"issue_id": "issue1", "resolved_section_ids": ["sec_1"], "target": {"chapter_id": "sec_1"}, "description": "修订本章", "evidence": "核对正文", "modification": "按已有材料修订本章", "requires_new_search": False}]
        value.call_llm = AsyncMock(side_effect=[json.dumps({"revised_content": "正文" * 160 + f"[来源]({URL})",
            "claims": [], "addressed_issues": ["issue1"], "unable_to_address": []}),
            '{"executive_summary":"摘要","conclusions":[],"outlook":"展望","claims":[]}'])
        await value._revise_report(state)
        self.assertEqual(state["critic_feedback"][0]["writer_status"], "not_reported")
        self.assertLessEqual(body_character_count(state["final_report"]), 800)


if __name__ == "__main__":
    unittest.main()
