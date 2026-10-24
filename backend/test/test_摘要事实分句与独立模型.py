"""事实分句、任务正文保留与独立模型路由；所有模型调用使用桩。"""
import json
import os
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import httpx
from openai import APIConnectionError

from test_智能体独立历史 import MemoryStore, CharCounter, HistoryScope, make_agent, bind_agent_history
from test_摘要合并与引用分离 import clause, draft, citation
from service.agent_context import parse_summary, summarize_history, summary_reference_quotes
from service.summary_citations import summary_merge_view, parse_draft
from service.context_policy import ContextPolicy, VerificationSettings


class FactualSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = MemoryStore()
        self.scope = HistoryScope(self.store, "run", "session", None)
        self.row = await self.store.append(self.scope, "DeepScout", "llm_output",
                                           {"content": "研究A于2023年纳入800人。"}, "call")
        self.agent = make_agent()
        env = patch.dict(os.environ, {"AGENT_SUMMARY_STRATEGY": "split-citations-v4",
                                     "AGENT_SUMMARY_NLI_ENABLED": "0", "AGENT_SUMMARY_CHECKS_ENABLED": "0",
                                     "LLM_MODEL_SUMMARY": "qwen3-max"})
        env.start()
        self.addCleanup(env.stop)

    async def test_only_facts_are_cited_and_context_survives_incremental_merge(self):
        fact = clause(self.row["payload"]["content"], self.row["source_id"])
        items = [clause("目标：研究相关证据。", section="SESSION INTENT"),
                 clause("要求：报告不超过3000字。", section="SESSION INTENT"), fact,
                 clause("建议：比较研究差异。"), clause("待办：继续检索。")]
        self.agent._call_llm_impl = AsyncMock(side_effect=[
            draft(*items), citation(fact, [(self.row["source_id"], fact["claim"])])])
        with bind_agent_history(self.scope):
            result = await summarize_history(self.agent, self.scope, [self.row], CharCounter())
        parts, claims = parse_summary(result, {self.row["source_id"]: fact["claim"]})
        self.assertEqual([c["claim"] for c in claims], [fact["claim"]])
        self.assertIn("3000字", parts["SESSION INTENT"])
        merged = summary_merge_view(result, summary_reference_quotes(result))
        self.assertEqual(parse_draft(merged, {self.row["source_id"]}), items)
        self.assertEqual(self.agent._call_llm_impl.await_count, 2)
        for call in self.agent._call_llm_impl.call_args_list:
            self.assertEqual(call.kwargs["model_override"], "qwen3-max")
        # 后续增量摘要仍收到任务正文，而不是只剩 CLAIMS 中的事实。
        self.agent._call_llm_impl.reset_mock(side_effect=True)
        self.agent._call_llm_impl.side_effect = [draft(*items), citation(fact, [(self.row["source_id"], fact["claim"])])]
        with bind_agent_history(self.scope):
            await summarize_history(self.agent, self.scope, [self.row], CharCounter(),
                                    previous_summary=result, previous_sources=summary_reference_quotes(result))
        sent = json.loads(self.agent._call_llm_impl.call_args_list[0].args[1])
        self.assertIn("要求：报告不超过3000字。", sent["previous_summary"])
        self.assertIn("待办：继续检索。", sent["previous_summary"])

    async def test_pure_task_summary_has_empty_claims(self):
        self.agent._call_llm_impl = AsyncMock(return_value=draft(
            clause("要求：报告不超过3000字。", section="SESSION INTENT")))
        with bind_agent_history(self.scope):
            result = await summarize_history(self.agent, self.scope, [self.row], CharCounter())
        self.assertEqual(parse_summary(result, {})[1], [])
        self.agent._call_llm_impl.assert_awaited_once()

    def test_unmarked_fact_cannot_be_omitted_from_claims(self):
        text = draft(clause("目标：研究证据。", section="SESSION INTENT"),
                     clause("样本量是800人。")) + "\n\n## CLAIMS\n[]"
        with self.assertRaisesRegex(ValueError, "未列入"):
            parse_summary(text, {})

    async def test_contract_check_still_sees_requirements_outside_claims(self):
        policy = ContextPolicy(group="事实分句测试", summary_strategy="chunk-merge-v1", semantic_verification=True,
                               verification=VerificationSettings(nli_support=False, coverage=False))
        self.scope = replace(self.scope, context_policy=policy.snapshot(), contract={"word_limit": 3000})
        self.agent._call_llm_impl = AsyncMock(side_effect=[
            draft(clause("要求：报告不超过3000字。", section="SESSION INTENT")),
            json.dumps({"items": [{"status": "preserved", "quote": "报告不超过3000字。"}]}, ensure_ascii=False),
        ])
        with bind_agent_history(self.scope):
            result = await summarize_history(self.agent, self.scope, [self.row], CharCounter())
        sent = json.loads(self.agent._call_llm_impl.call_args_list[1].args[1])
        self.assertIn("要求：报告不超过3000字。", sent["body"])
        self.assertEqual(parse_summary(result, {})[1], [])

    async def test_legacy_incremental_summary_also_uses_independent_model(self):
        result = draft(clause("要求：报告不超过3000字。", section="SESSION INTENT")) + "\n\n## CLAIMS\n[]"
        self.agent._call_llm_impl = AsyncMock(return_value=result)
        with patch.dict(os.environ, {"AGENT_SUMMARY_STRATEGY": "incremental-v3"}), bind_agent_history(self.scope):
            await summarize_history(self.agent, self.scope, [self.row], CharCounter())
        self.assertEqual(self.agent._call_llm_impl.call_args.kwargs["model_override"], "qwen3-max")


class SummaryModelRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_override_reaches_provider_and_does_not_change_scout(self):
        agent = make_agent()
        agent.model = "qwen-plus"
        create = agent.client.chat.completions.create
        with patch("service.deep_research_v2.agents.base.record_llm_call"):
            await agent._call_llm_impl("系统", "正文", False, model_override="qwen3-max")
            self.assertEqual(create.call_args.kwargs["model"], "qwen3-max")
            await agent._call_llm_impl("系统", "检索", False)
        self.assertEqual(create.call_args.kwargs["model"], "qwen-plus")
        self.assertEqual(agent.model, "qwen-plus")

    async def test_summary_retry_keeps_explicit_model(self):
        agent = make_agent()
        create = agent.client.chat.completions.create
        create.side_effect = [APIConnectionError(request=httpx.Request("POST", "https://example.invalid")), create.return_value]
        with patch("service.deep_research_v2.agents.base.default_agent_max_attempts", return_value=2), \
             patch("service.deep_research_v2.agents.base.default_fallback_model", return_value="other-model"), \
             patch("service.deep_research_v2.agents.base.asyncio.sleep", new=AsyncMock()), \
             patch("service.deep_research_v2.agents.base.record_llm_call"):
            await agent._call_llm_impl("系统", "正文", False, model_override="qwen3-max")
        self.assertEqual([c.kwargs["model"] for c in create.call_args_list], ["qwen3-max", "qwen3-max"])


if __name__ == "__main__":
    unittest.main()
