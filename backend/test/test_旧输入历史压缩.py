"""旧输入原文引用替换、单条摘要移除及整体摘要保留的离线机制测试。"""
import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from test_智能体独立历史 import MemoryStore, CharCounter, make_agent, summary_text
from service.agent_history import HistoryScope, bind_agent_history, canonical_json
from service.agent_context import prepare_agent_messages, ContextBudgetExceeded
from service.context_policy import ContextPolicy, InputHistorySettings, LegacyInputHistorySettings, InputBudget, AGENTS
from service.input_history_view import reference_materials


def policy(**overrides):
    data = dict(group="旧输入压缩测试", summary_strategy="chunk-merge-v1", semantic_verification=False,
                # 来源匹配用例只验证引用；整体摘要用例显式开启，避免混淆两阶段。
                semantic_summary=False,
                input_history=InputHistorySettings(min_reference_tokens=128),
                # 本组验证阈值触发后的来源匹配；阈值边界另有专门回归。
                soft_token_cap=400,
                # 固定机制测试窗口，不依赖部署默认窗口的调整。
                stage_budgets={name: InputBudget(context_window=32768, max_input=30000, safety_tokens=1024)
                               for name in AGENTS},
                output_limits={name: 6000 for name in AGENTS})
    data.update(overrides)
    return ContextPolicy(**data)


class InputViewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = MemoryStore()
        self.scope = HistoryScope(self.store, "run", "session", None, context_policy=policy().snapshot(),
                                  contract={"length": "2000字"})
        self.agent = make_agent()
        self.counter = CharCounter()
        self.count_patch = patch("service.agent_context.get_qwen_token_counter", return_value=self.counter)
        self.count_patch.start()
        self.addCleanup(self.count_patch.stop)

    async def tool(self, content):
        return await self.store.append(self.scope, self.agent.name, "tool_output",
                                       {"value": {"content": content}}, "tool")

    async def old_input(self, text, completed=True):
        row = await self.store.append(self.scope, self.agent.name, "llm_input", {"content": text}, "old")
        if completed:
            await self.store.append(self.scope, self.agent.name, "llm_output", {"content": "完成"}, "old")
        return row

    async def prepare(self, scope=None):
        scope = scope or self.scope
        with bind_agent_history(scope):
            return await prepare_agent_messages(self.agent, scope, "系统提示", "最新要求：2000字", "current", 6000)

    def input_view(self, messages, sid):
        history = json.loads(messages[1]["content"].split("\n", 1)[1])
        return next(r["text"] for r in history if r.get("source_id") == sid)

    async def test_reference_changes_only_material_keeps_original_and_current_instruction(self):
        body = "论文正文，包括样本量1000人和日期2020年。" * 200
        tool = await self.tool(body)
        text = "旧要求：3000字。\n已读取正文: " + body[:4500] + "\n必须保留研究局限。"
        row = await self.old_input(text)
        messages = await self.prepare()
        view = self.input_view(messages, row["source_id"])
        self.assertIn(tool["source_id"], view)
        self.assertNotIn(body[:1000], view)
        self.assertIn("旧要求：3000字。", view)
        self.assertTrue(view.endswith("必须保留研究局限。"))
        self.assertEqual(messages[-1]["content"], "最新要求：2000字")
        self.assertIn("2000字", messages[0]["content"])
        original = await self.store.read_source(self.scope, self.agent.name, row["source_id"])
        self.assertEqual(original, row)
        self.assertEqual((await self.store.read_source(self.scope, self.agent.name, tool["source_id"])), tool)
        saved = await self.store.load(self.scope, self.agent.name)
        self.assertEqual(saved["covered_seq"], 0)
        self.assertFalse(any(r["kind"] == "input_history_view" for r in saved["messages"]))

    async def test_unlabelled_task_text_is_not_replaced(self):
        body = "这是用户要求而不是正文。" * 100
        await self.tool(body)
        row = await self.old_input(body)
        messages = await self.prepare()
        self.assertEqual(self.input_view(messages, row["source_id"]), body)

    async def test_no_saved_source_keeps_labelled_body(self):
        text = "已读取正文: " + "未找到来源。" * 100
        row = await self.old_input(text)
        self.assertEqual(self.input_view(await self.prepare(), row["source_id"]), text)

    async def test_long_unmatched_text_keeps_references_without_model_summary(self):
        body = "工具保存的完整网页正文。" * 200
        tool = await self.tool(body)
        background = "需要保留的长背景。" * 600
        row = await self.old_input(background + "\n已读取正文: " + body)
        scope = self.scope
        with patch("service.agent_context.summarize_history", AsyncMock()) as summarizer:
            messages = await self.prepare(scope)
        summarizer.assert_not_awaited()
        view = self.input_view(messages, row["source_id"])
        self.assertIn(background, view)
        self.assertIn(tool["source_id"], view)
        self.assertNotIn(body, view)
        self.assertEqual(await self.store.read_source(scope, self.agent.name, row["source_id"]), row)
        saved = await self.store.load(scope, self.agent.name)
        self.assertFalse(any(r["kind"] == "input_history_view" for r in saved["messages"]))

    async def test_long_history_still_commits_only_global_summary(self):
        row = await self.old_input("超长背景。" * 22000)
        scope = replace(self.scope, context_policy=policy(semantic_summary=True).snapshot())
        with patch("service.agent_context.summarize_history", AsyncMock(
                return_value=summary_text(row["source_id"], "超长背景。"))) as summarizer:
            messages = await self.prepare(scope)
        self.assertEqual(summarizer.await_count, 1)
        self.assertEqual([r["kind"] for r in summarizer.call_args.args[2]], ["llm_input", "llm_output"])
        self.assertIn('"kind": "history_summary"', messages[1]["content"])
        saved = await self.store.load(scope, self.agent.name)
        self.assertEqual(saved["summary_version"], 1)
        self.assertEqual(saved["covered_seq"], 2)
        self.assertFalse(any(r["kind"] == "input_history_view" for r in saved["messages"]))

    async def test_legacy_summary_settings_and_cache_cannot_restore_removed_branch(self):
        row = await self.old_input("必须保留的原始背景。" * 700)
        cache = await self.store.append(self.scope, self.agent.name, "input_history_view", {
            "mode": "summary", "source_id": row["source_id"], "text": "旧缓存摘要",
            "cache_key": "旧缓存", "source_hash": row["content_hash"],
        }, "old-cache")
        scope = replace(self.scope, context_policy=policy(input_history=LegacyInputHistorySettings(
            summarize_long_inputs=True, summary_trigger_tokens=1)).snapshot())
        with patch("service.agent_context.summarize_history", AsyncMock()) as summarizer:
            messages = await self.prepare(scope)
        summarizer.assert_not_awaited()
        self.assertEqual(self.input_view(messages, row["source_id"]), row["payload"]["content"])
        saved = await self.store.load(scope, self.agent.name)
        self.assertEqual(sum(r["kind"] == "input_history_view" for r in saved["messages"]), 1)
        self.assertEqual(await self.store.read_source(scope, self.agent.name, cache["source_id"]), cache)
        self.assertFalse(any(r["kind"] == "input_history_source" for r in saved["messages"]))

    def test_legacy_input_settings_snapshot_keeps_original_fields_and_hash(self):
        snapshot = policy(input_history=LegacyInputHistorySettings()).snapshot()
        restored = ContextPolicy.from_snapshot(copy.deepcopy(snapshot))
        self.assertEqual(restored.snapshot(), snapshot)
        self.assertEqual(restored.input_history.version, "input-view-v1")
        self.assertTrue(restored.input_history.summarize_long_inputs)

    async def test_future_tool_cannot_become_past_input_source(self):
        body = "后续工具才返回的资料。" * 100
        row = await self.old_input("已读取正文: " + body)
        await self.tool(body)
        self.assertEqual(self.input_view(await self.prepare(), row["source_id"]), row["payload"]["content"])

    async def test_reference_offsets_match_actual_readable_json(self):
        body = '正文包含"引号"与换行\n以及中文。' * 100
        tool = await self.tool(body)
        view, refs = reference_materials("## 文档内容\n" + body + "\n## 任务\n提取", [tool], self.counter, 128)
        self.assertEqual(len(refs), 1)
        ref = refs[0]
        self.assertEqual(canonical_json(tool["payload"])[ref["start"]:ref["end"]], canonical_json(body)[1:-1])
        self.assertTrue(view.endswith("\n## 任务\n提取"))

    async def test_offload_and_read_switches_control_reference(self):
        body = "工具正文。" * 200
        await self.tool(body)
        row = await self.old_input("已读取正文: " + body)
        for changes in [dict(artifact_offload=False), dict(history_read_max_calls=0),
                        dict(input_history=None), dict(input_history=InputHistorySettings(reference_materials=False))]:
            with self.subTest(changes=changes):
                scope = replace(self.scope, context_policy=policy(**changes).snapshot())
                self.assertEqual(self.input_view(await self.prepare(scope), row["source_id"]), row["payload"]["content"])

    async def test_unfinished_and_interrupted_inputs_are_not_referenced(self):
        row = await self.old_input("长输入" * 3000, completed=False)
        mock = AsyncMock()
        with patch("service.agent_context.summarize_history", mock):
            self.assertEqual(self.input_view(await self.prepare(), row["source_id"]), row["payload"]["content"])
            resumed = replace(self.scope, invocation_id="new-invocation")
            self.assertIn("上次执行中断", self.input_view(await self.prepare(resumed), row["source_id"]))
        mock.assert_not_awaited()

    async def test_total_budget_still_rejects_oversize_input_when_summary_disabled(self):
        await self.old_input("长背景" * 40000)
        scope = replace(self.scope, context_policy=policy(semantic_summary=False).snapshot())
        with self.assertRaises(ContextBudgetExceeded):
            await self.prepare(scope)

    async def test_source_hash_tampering_rejected_before_reference(self):
        body = "网页正文。" * 200
        tool = await self.tool(body)
        await self.old_input("已读取正文: " + body)
        self.store.data[self.store.key(self.scope, self.agent.name)]["messages"][0]["payload"]["value"]["content"] += "改写"
        with self.assertRaisesRegex(ValueError, "哈希"):
            await self.prepare()

    async def test_missing_read_permission_never_adopts_reference(self):
        body = "网页正文。" * 200
        await self.tool(body)
        await self.old_input("已读取正文: " + body)
        with patch.object(self.store, "read_source", AsyncMock(side_effect=PermissionError("不属于此上下文"))):
            with self.assertRaises(PermissionError):
                await self.prepare()

    async def test_total_history_still_uses_global_summary_when_individual_inputs_short(self):
        for i in range(36):
            await self.store.append(self.scope, self.agent.name, "llm_input", {"content": "短背景" * 1000}, f"c{i}")
            await self.store.append(self.scope, self.agent.name, "llm_output", {"content": "完成"}, f"c{i}")
        saved = await self.store.load(self.scope, self.agent.name)
        mock = AsyncMock(return_value=summary_text(saved["messages"][0]["source_id"], "短背景"))
        scope = replace(self.scope, context_policy=policy(semantic_summary=True).snapshot())
        with patch("service.agent_context.summarize_history", mock):
            messages = await self.prepare(scope)
        self.assertEqual(mock.await_count, 1)
        self.assertEqual(len(mock.call_args.args[2]), 72)
        self.assertIn('"kind": "history_summary"', messages[1]["content"])
        self.assertNotIn("validated_history_summary", messages[1]["content"])

    def test_old_snapshot_hash_stays_valid_without_new_settings(self):
        original = ContextPolicy(group="旧快照", semantic_verification=False).snapshot()
        self.assertNotIn("input_history", original["config"])
        self.assertEqual(ContextPolicy.from_snapshot(copy.deepcopy(original)).snapshot(), original)


if __name__ == "__main__":
    unittest.main()
