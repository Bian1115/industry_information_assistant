"""非法回读ID应留痕并继续模型任务；不发起数据库读取。"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "app"), str(Path(__file__).resolve().parent)]

from service.agent_history import HistoryScope, bind_agent_history
from test_智能体独立历史 import CharCounter, MemoryStore, make_agent
from test_逐智能体压缩配置 import policy


class InvalidHistoryReadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = MemoryStore()
        root = policy(semantic_summary=False).with_agent_compression({"LeadWriter": {
            "enabled": True, "reference_materials": False,
            "artifact_offload": False, "semantic_summary": False}})
        self.scope = HistoryScope(self.store, "r", "s", None, node="write", context_policy=root.snapshot())
        self.agent = make_agent("LeadWriter")
        for target in ("service.agent_context.get_qwen_token_counter",
                       "service.qwen_token_counter.get_qwen_token_counter"):
            counter = patch(target, return_value=CharCounter())
            counter.start()
            self.addCleanup(counter.stop)

    async def test_placeholder_is_recorded_then_report_completes(self):
        self.agent._call_llm_impl = AsyncMock(side_effect=[
            '{"_history_read":{"source_id":"历史提供的编号","start":0,"chars":4000}}',
            '{"content":"完成正文"}'])
        with bind_agent_history(self.scope), patch.object(self.store, "read_source", new=AsyncMock()) as read:
            result = await self.agent.call_llm("writer", "写报告")
        self.assertEqual(result, '{"content":"完成正文"}')
        read.assert_not_awaited()
        rows = (await self.store.load(self.scope, "LeadWriter"))["messages"]
        errors = [r for r in rows if r["kind"] == "history_read_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["payload"]["source_id"], "历史提供的编号")
        self.assertTrue(errors[0]["payload"]["recoverable"])
        self.assertFalse(any(r["kind"] == "llm_error" for r in rows))
        self.assertEqual(rows[-1]["kind"], "llm_output")
        sent = self.agent._call_llm_impl.call_args.kwargs["prepared_messages"]
        self.assertIn("没有返回原文", sent[-2]["content"])
        self.assertEqual(sent[-1]["content"], "写报告")

    async def test_non_string_id_can_continue_to_a_valid_scoped_read(self):
        source = await self.store.append(self.scope, "LeadWriter", "tool_output", {"content": "真实原文"}, "source")
        self.agent._call_llm_impl = AsyncMock(side_effect=[
            json.dumps({"_history_read": {"source_id": None}}),
            json.dumps({"_history_read": {"source_id": source["source_id"], "start": 0, "chars": 4}}),
            '{"content":"完成正文"}'])
        with bind_agent_history(self.scope), patch.object(self.store, "read_source", wraps=self.store.read_source) as read:
            result = await self.agent.call_llm("writer", "写报告")
        self.assertEqual(result, '{"content":"完成正文"}')
        self.assertEqual(read.await_count, 1)
        rows = (await self.store.load(self.scope, "LeadWriter"))["messages"]
        self.assertEqual(sum(r["kind"] == "history_read_error" for r in rows), 1)
        self.assertEqual(sum(r["kind"] == "history_read" for r in rows), 1)

    async def test_valid_but_out_of_scope_id_still_cannot_read(self):
        self.agent._call_llm_impl = AsyncMock(return_value=
            '{"_history_read":{"source_id":"11111111-1111-4111-8111-111111111111","start":0,"chars":4}}')
        with bind_agent_history(self.scope), self.assertRaises(PermissionError):
            await self.agent.call_llm("writer", "写报告")

    async def test_out_of_range_read_is_recorded_then_answer_completes(self):
        source = await self.store.append(self.scope, "LeadWriter", "llm_output", {"content": "短原文"}, "source")
        self.agent._call_llm_impl = AsyncMock(side_effect=[
            json.dumps({"_history_read": {"source_id": source["source_id"], "start": 9000, "chars": 4000}}),
            '{"content":"完成正文"}'])
        with bind_agent_history(self.scope):
            result = await self.agent.call_llm("writer", "写报告")
        self.assertEqual(result, '{"content":"完成正文"}')
        rows = (await self.store.load(self.scope, "LeadWriter"))["messages"]
        errors = [r for r in rows if r["kind"] == "history_read_error"]
        self.assertEqual(errors[0]["payload"]["error_code"], "empty_history_read_range")
        self.assertEqual(errors[0]["payload"]["total_chars"], 3)
        self.assertFalse(any(r["kind"] == "history_read" for r in rows))

    async def test_invalid_length_is_recorded_without_reading_source(self):
        self.agent._call_llm_impl = AsyncMock(side_effect=[
            '{"_history_read":{"source_id":"11111111-1111-4111-8111-111111111111","start":0,"chars":8001}}',
            '{"content":"完成正文"}'])
        with bind_agent_history(self.scope), patch.object(self.store, "read_source", new=AsyncMock()) as read:
            result = await self.agent.call_llm("writer", "写报告")
        self.assertEqual(result, '{"content":"完成正文"}')
        read.assert_not_awaited()
        rows = (await self.store.load(self.scope, "LeadWriter"))["messages"]
        self.assertTrue(any(r["kind"] == "history_read_error" and
                            r["payload"]["error_code"] == "invalid_history_read_range" for r in rows))


if __name__ == "__main__":
    unittest.main()
