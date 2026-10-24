"""独立页面检查的输入隔离回归；本地 Qwen tokenizer、内存历史、模拟模型。"""

import json
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "test")]

from service.agent_context import (
    ContextBudgetExceeded, is_evidence_batch_input, prepare_agent_messages,
)
from service.agent_history import HistoryScope, bind_agent_history
from service.deep_research_v2.agents.scout import DeepScout
from service.qwen_token_counter import QwenTokenCounter
from service.web_reader_service import WebReadResult
from test_智能体独立历史 import MemoryStore
from test_逐智能体压缩配置 import policy

OBSERVATIONS = []
URL = "https://example.gov.cn/report"
BODY = "仅供预览\n项目于2023年投运。\n服务人口为6000人。"
ANSWER = json.dumps({"status": "partial", "reason": "保留展示的正文",
                     "body_block_ids": ["b000002", "b000003"]}, ensure_ascii=False)


class IsolatedBodyCheckTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = QwenTokenCounter(model_name=str(BACKEND / ".cache/qwen-tokenizer"))
        cls.counter.count_text("仅加载本地 tokenizer")

    def setUp(self):
        self.store = MemoryStore()
        root = policy(semantic_summary=False).with_agent_compression({"DeepScout": {
            "enabled": True, "reference_materials": False, "artifact_offload": False,
            "semantic_summary": False, "semantic_verification": False}})
        self.scope = HistoryScope(self.store, "独立检查测试", "测试会话", None,
                                  contract={"query": "研究项目运营，仅使用官方来源"},
                                  context_policy=root.snapshot())
        self.agent = DeepScout.__new__(DeepScout)
        self.agent.name = "DeepScout"
        self.agent.logger = logging.getLogger("test.isolated-body-review")
        self.agent.parse_json_response = json.loads
        self.agent._call_llm_impl = AsyncMock(return_value=ANSWER)
        for target in ("service.agent_context.get_qwen_token_counter",
                       "service.qwen_token_counter.get_qwen_token_counter"):
            item = patch(target, return_value=self.counter)
            item.start()
            self.addCleanup(item.stop)

    def page(self):
        return WebReadResult(url=URL, final_url=URL, title="项目运营报告",
                             content=BODY, fetch_status="success")

    async def seed(self, text):
        return await self.store.append(self.scope, "DeepScout", "tool_output",
                                       {"content": text}, "已完成的旧搜索")

    async def test_oversized_history_is_excluded_and_originals_are_archived(self):
        old_text = "旧研究标记不能进入正文检查。" + "history " * 50000
        source = await self.seed(old_text)
        before_tokens = self.counter.count_text(old_text)
        page = self.page()
        with bind_agent_history(self.scope):
            with self.assertRaises(ContextBudgetExceeded):
                await prepare_agent_messages(self.agent, self.scope, "检查", "本页", "普通输入对照", 1500)
            await self.agent._check_source_body(page, {})
        sent = self.agent._call_llm_impl.call_args.kwargs["prepared_messages"]
        actual = self.counter.count_messages(sent)
        self.assertEqual(len(sent), 2)
        self.assertLessEqual(actual, 4000)
        self.assertNotIn("旧研究标记", json.dumps(sent, ensure_ascii=False))
        self.assertIn("仅使用官方来源", sent[0]["content"])
        self.assertIn(URL, sent[1]["content"])
        self.assertIn("项目于2023年投运", sent[1]["content"])
        self.assertTrue(page.success)
        rows = (await self.store.load(self.scope, "DeepScout"))["messages"]
        self.assertEqual(rows[0], source)
        self.assertEqual(rows[0]["payload"]["content"], old_text)
        self.assertEqual(rows[-1]["kind"], "llm_output")
        views = [row["payload"] for row in rows if row["kind"] == "context_view"]
        self.assertTrue(views[0]["history_replay"])  # 前面刻意构造的普通输入超限对照。
        view = views[-1]  # 实际的独立正文质量检查。
        self.assertFalse(view["history_replay"])
        self.assertFalse(view["summary_attempted"])
        self.assertEqual(view["input_mode"], "evidence_batch")
        OBSERVATIONS.append({"case": self._testMethodName, "tokenizer": "本地Qwen",
                             "old_history_text_tokens": before_tokens, "review_input_tokens": actual,
                             "input_budget": 4000, "sent_message_count": len(sent),
                             "history_replay": False, "archived_original_unchanged": True})

    async def test_mode_restores_and_next_ordinary_scout_call_replays_history(self):
        await self.seed("普通Scout仍需继承的研究结论")
        modes = []

        async def answer(*args, **kwargs):
            modes.append(is_evidence_batch_input())
            return ANSWER if len(modes) == 1 else '{"ok":true}'

        self.agent._call_llm_impl.side_effect = answer
        with bind_agent_history(self.scope):
            await self.agent._check_source_body(self.page(), {})
            self.assertFalse(is_evidence_batch_input())
            await self.agent.call_llm("继续研究", "当前研究问题")
        self.assertEqual(modes, [True, False])
        sent = self.agent._call_llm_impl.call_args.kwargs["prepared_messages"]
        self.assertEqual(len(sent), 3)
        self.assertIn("普通Scout仍需继承的研究结论", sent[1]["content"])

    async def test_provider_error_restores_mode_and_keeps_error_archive(self):
        self.agent._call_llm_impl.side_effect = RuntimeError("模拟模型请求失败")
        page = self.page()
        with bind_agent_history(self.scope):
            await self.agent._check_source_body(page, {})
            self.assertFalse(is_evidence_batch_input())
        rows = (await self.store.load(self.scope, "DeepScout"))["messages"]
        self.assertEqual(rows[-1]["kind"], "llm_error")
        self.assertEqual(rows[-1]["payload"]["error_type"], "RuntimeError")
        self.assertTrue(page.success)
        self.assertIsNone(page.failure_category)
        self.assertEqual(page.body_quality["status"], "uncertain")
        self.assertEqual(page.body_quality["model_review"]["execution_status"], "failed")
        self.assertIn("quality_review_failed", page.body_quality["reasons"])

    async def test_current_contract_still_obeys_hard_input_budget(self):
        from dataclasses import replace
        scope = replace(self.scope, contract={"query": "不可删除的当前用户要求" * 3000})
        page = self.page()
        with bind_agent_history(scope):
            await self.agent._check_source_body(page, {})
            self.assertFalse(is_evidence_batch_input())
        self.agent._call_llm_impl.assert_not_awaited()
        rows = (await self.store.load(scope, "DeepScout"))["messages"]
        self.assertEqual(rows[-1]["payload"]["error_type"], "ContextBudgetExceeded")
        self.assertEqual(page.body_quality["model_review"]["error_category"], "input_budget_exceeded")

    async def test_independent_review_cannot_append_history_reads(self):
        source = await self.seed("无关旧网页")
        self.store.read_source = AsyncMock()
        self.agent._call_llm_impl.return_value = json.dumps({"_history_read": {
            "source_id": source["source_id"], "start": 0, "chars": 10}})
        page = self.page()
        with bind_agent_history(self.scope):
            await self.agent._check_source_body(page, {})
            self.assertFalse(is_evidence_batch_input())
        self.store.read_source.assert_not_awaited()
        self.agent._call_llm_impl.assert_awaited_once()
        rows = (await self.store.load(self.scope, "DeepScout"))["messages"]
        self.assertEqual(rows[-1]["payload"]["error_type"], "ValueError")
        self.assertIn("不允许追加历史回读", rows[-1]["payload"]["message"])


if __name__ == "__main__":
    unittest.main()
