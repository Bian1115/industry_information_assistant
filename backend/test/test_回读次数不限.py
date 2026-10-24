"""回读次数不限的机制回归；仅使用内存资料与模拟模型。"""

import json
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import uuid
from unittest.mock import AsyncMock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "app"), str(Path(__file__).resolve().parent)]

from service.agent_context import ContextBudgetExceeded
from service.agent_history import HistoryScope, bind_agent_history
from service.context_policy import ContextPolicy, policy_for_scope
from test_智能体独立历史 import CharCounter, MemoryStore, make_agent
from test_逐智能体压缩配置 import policy


class ReadPolicyTests(unittest.TestCase):
    def test_default_unlimited_roundtrips_and_keeps_material_methods_enabled(self):
        root = policy().with_agent_compression({"LeadWriter": {"enabled": True}})
        restored = ContextPolicy.from_snapshot(root.snapshot())
        scope = HistoryScope(None, "r", "s", None, context_policy=restored.snapshot())
        view = policy_for_scope(scope, "LeadWriter")
        self.assertIsNone(view.history_read_max_calls)
        self.assertEqual(view.history_read_max_chars, 8000)
        self.assertTrue(view.input_history.reference_materials)
        self.assertTrue(view.artifact_offload)
        self.assertEqual(restored.fingerprint, root.fingerprint)

    def test_explicit_old_limit_is_preserved(self):
        root = policy(history_read_max_calls=2)
        restored = ContextPolicy.from_snapshot(root.snapshot())
        self.assertEqual(restored.history_read_max_calls, 2)
        self.assertEqual(restored.fingerprint, root.fingerprint)

    def test_disabled_agent_still_disables_reads(self):
        root = policy().with_agent_compression({"LeadWriter": {"enabled": False}})
        scope = HistoryScope(None, "r", "s", None, context_policy=root.snapshot())
        self.assertEqual(policy_for_scope(scope, "LeadWriter").history_read_max_calls, 0)


class UnlimitedReadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.root = policy(semantic_summary=False).with_agent_compression({"LeadWriter": {
            "enabled": True, "reference_materials": False,
            "artifact_offload": False, "semantic_summary": False}})
        self.scope = HistoryScope(self.store, "r", "s", None, node="write", context_policy=self.root.snapshot())
        self.agent = make_agent("LeadWriter")
        for target in ("service.agent_context.get_qwen_token_counter",
                       "service.qwen_token_counter.get_qwen_token_counter"):
            counter = patch(target, return_value=CharCounter())
            counter.start()
            self.addCleanup(counter.stop)

    async def source(self, content):
        return await self.store.append(self.scope, "LeadWriter", "tool_output", {"content": content}, str(uuid.uuid4()))

    def request(self, source, *, start=0, chars=40):
        return json.dumps({"_history_read": {"source_id": source["source_id"], "start": start, "chars": chars}})

    async def test_five_reads_can_complete_answer(self):
        sources = [await self.source("evidence " + str(i) * 80) for i in range(5)]
        self.agent._call_llm_impl = AsyncMock(side_effect=[
            *(self.request(source) for source in sources), '{"done":true}'])
        with bind_agent_history(self.scope):
            result = await self.agent.call_llm("writer", "写报告")
        self.assertEqual(result, '{"done":true}')
        self.assertEqual(self.agent._call_llm_impl.await_count, 6)
        rows = (await self.store.load(self.scope, "LeadWriter"))["messages"]
        self.assertEqual(sum(row["kind"] == "history_read" for row in rows), 5)
        sent = self.agent._call_llm_impl.call_args.kwargs["prepared_messages"]
        self.assertIn("回读次数不设上限", sent[0]["content"])
        self.assertEqual(rows[-1]["kind"], "llm_output")

    async def test_explicit_two_read_limit_still_rejects_third_request(self):
        root = self.root.model_copy(update={"history_read_max_calls": 2})
        self.scope = replace(self.scope, context_policy=root.snapshot())
        source = await self.source("evidence " * 80)
        self.agent._call_llm_impl = AsyncMock(side_effect=[self.request(source, start=i * 40) for i in range(3)])
        with bind_agent_history(self.scope), self.assertRaisesRegex(ValueError, "历史回读次数超限"):
            await self.agent.call_llm("writer", "写报告")
        rows = (await self.store.load(self.scope, "LeadWriter"))["messages"]
        self.assertEqual(sum(row["kind"] == "history_read" for row in rows), 2)
        self.assertEqual(self.agent._call_llm_impl.await_count, 3)

    async def test_unlimited_reads_still_stop_at_input_budget(self):
        # 原文仅供回读，不预先加入历史；验证真实准备流程在第三次请求前拦截。
        source = await self.store.append(self.scope, "LeadWriter", "input_history_source",
                                         {"content": "A" * 24000}, "source")
        self.agent._call_llm_impl = AsyncMock(side_effect=[
            self.request(source, start=0, chars=8000),
            self.request(source, start=8000, chars=8000)])
        with bind_agent_history(self.scope):
            with self.assertRaises(ContextBudgetExceeded):
                await self.agent.call_llm("writer", "任务")
        self.assertEqual(self.agent._call_llm_impl.await_count, 2)
        rows = (await self.store.load(self.scope, "LeadWriter"))["messages"]
        self.assertEqual(rows[-1]["kind"], "llm_error")
        self.assertEqual(rows[-1]["payload"]["error_type"], "ContextBudgetExceeded")

    async def test_unlimited_reads_keep_single_fragment_length_limit(self):
        source = await self.source("evidence")
        self.agent._call_llm_impl = AsyncMock(side_effect=[self.request(source, chars=8001), '{"done":true}'])
        with bind_agent_history(self.scope):
            self.assertEqual(await self.agent.call_llm("writer", "任务"), '{"done":true}')
        self.assertEqual(self.agent._call_llm_impl.await_count, 2)
        rows = (await self.store.load(self.scope, "LeadWriter"))["messages"]
        self.assertFalse(any(row["kind"] == "history_read" for row in rows))
        self.assertTrue(any(row["kind"] == "history_read_error" for row in rows))


if __name__ == "__main__":
    unittest.main()
