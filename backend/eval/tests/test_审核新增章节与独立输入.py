"""验证三输入实际发送边界、新增章节和无问题低分返工；外部模型使用替身。"""

from pathlib import Path
import json
import sys
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "test"), str(Path(__file__).parent)]
from test_智能体独立历史 import MemoryStore, CharCounter, make_agent
from service.agent_history import HistoryScope, bind_agent_history
from service.agent_context import isolated_agent_input, is_isolated_agent_input, ContextBudgetExceeded
from service.deep_research_v2.review_contract import bind_issue_targets, materialize_new_sections
from test_逐问题返工与复审 import work, issue, CONTENT, response
from test_Writer论断引用结构 import make_writer, synthesis
from test_补搜任务执行 import make_scout


class IsolatedReviewInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_request_has_only_current_system_and_three_input_prompt(self):
        store = MemoryStore()
        scope = HistoryScope(store, "独立审核", "会话", None, node="review", contract={"secret": "不应发送的契约"})
        agent = make_agent("CriticMaster")
        await store.append(scope, "CriticMaster", "llm_input", {"content": "不应发送的历史"}, "old")
        with patch("service.agent_context.get_qwen_token_counter", return_value=CharCounter()), \
             patch("service.deep_research_v2.agents.base.record_llm_call"), \
             patch("service.deep_research_v2.agents.base.flush_pending"), \
             patch("service.session_evidence_rag.fit_evidence_prompt", side_effect=AssertionError("不应裁剪完整报告")), \
             bind_agent_history(scope), isolated_agent_input():
            await agent.call_llm("审核系统规则", "用户问题、大纲和完整报告")
        sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(sent, [{"role": "system", "content": "审核系统规则"},
                                {"role": "user", "content": "用户问题、大纲和完整报告"}])
        rows = (await store.load(scope, "CriticMaster"))["messages"]
        self.assertIn("llm_output", [row["kind"] for row in rows])
        self.assertIn("isolated", json.dumps(rows))
        self.assertFalse(is_isolated_agent_input())

    async def test_review_cannot_append_history_via_model_request(self):
        store = MemoryStore()
        scope = HistoryScope(store, "审核回读", "会话", None, node="review")
        agent = make_agent("CriticMaster", '{"_history_read":{"source_id":"anything"}}')
        with patch("service.agent_context.get_qwen_token_counter", return_value=CharCounter()), \
             patch("service.deep_research_v2.agents.base.record_llm_call"), \
             patch("service.deep_research_v2.agents.base.flush_pending"), \
             bind_agent_history(scope), isolated_agent_input(), self.assertRaisesRegex(ValueError, "独立审核"):
            await agent.call_llm("sys", "只有当前三个输入")

    async def test_over_budget_report_fails_instead_of_truncating(self):
        store = MemoryStore()
        scope = HistoryScope(store, "审核预算", "会话", None, node="review")
        agent = make_agent("CriticMaster")
        with patch("service.agent_context.get_qwen_token_counter", return_value=CharCounter()), \
             patch("service.agent_context.scoped_input_budget", return_value=20), \
             bind_agent_history(scope), isolated_agent_input(), self.assertRaises(ContextBudgetExceeded):
            await agent.call_llm("sys", "完整报告" * 1000)
        agent.client.chat.completions.create.assert_not_awaited()


class NewChapterWritingTests(unittest.IsolatedAsyncioTestCase):
    async def scenario(self, search):
        state = work()
        row = issue(requires_new_search=search, search_query="公司监管政策风险" if search else "")
        row["target"] = {"target_type": "new_section", "chapter_id": None, "chapter_title": None,
            "specific_location": None, "insert_after_chapter_id": "1", "proposed_chapter_title": "监管风险"}
        bind_issue_targets(state, [row])
        state["critic_feedback"] = [row]
        materialize_new_sections(state)
        new_id = row["resolved_section_ids"][0]
        self.assertEqual([s["title"] for s in state["outline"]], ["收入", "监管风险", "风险"])
        writer = make_writer({"content": CONTENT, "claims": [], "issue_responses": [response()]}, synthesis([]))
        if search:
            state["pending_research_tasks"] = [{"issue_id": row["issue_id"], "target_section": new_id,
                                                 "query": row["search_query"], "acceptance": row["modification"]}]
            scout = make_scout()
            await scout._supplementary_research(state)
            self.assertEqual(scout._enrich_results_with_full_text.call_args.kwargs["section_id"], new_id)
            await writer._write_report(state)
        else:
            await writer._revise_report(state)
        self.assertIn(new_id, state["draft_sections"])
        self.assertIn("## 2 监管风险", state["final_report"])
        self.assertIn("## 3 风险", state["final_report"])
        self.assertIn(row["modification"], writer.call_llm.await_args_list[0].kwargs["user_prompt"])

    async def test_new_chapter_without_search_is_written(self):
        await self.scenario(False)

    async def test_new_chapter_with_search_receives_evidence_and_is_written(self):
        await self.scenario(True)

    async def test_summary_only_issue_still_synthesizes_report(self):
        state = work()
        state["critic_feedback"][0]["resolved_section_ids"] = ["__synthesis__"]
        writer = make_writer(synthesis([]))
        await writer._revise_report(state)
        self.assertEqual(writer.call_llm.await_count, 1)
        self.assertIn("按问题返工清单", writer.call_llm.await_args.kwargs["user_prompt"])

    async def test_low_score_empty_issues_rewrites_using_summary_and_original(self):
        state = work()
        state.update(critic_feedback=[], quality_score=50, quality_verdict="major_issues", quality_review_summary="需解释收入与风险之间的关系")
        old = state["draft_sections"]["sec_1"]
        writer = make_writer({"content": CONTENT, "claims": []}, {"content": CONTENT, "claims": []}, synthesis([]))
        await writer._revise_report(state)
        self.assertEqual(writer.call_llm.await_count, 3)
        prompt = writer.call_llm.await_args_list[0].kwargs["user_prompt"]
        self.assertIn(state["quality_review_summary"], prompt)
        self.assertIn(old, prompt)
        self.assertEqual(state["phase"], "reviewing")


if __name__ == "__main__":
    unittest.main()
