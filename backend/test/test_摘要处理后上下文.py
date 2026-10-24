"""摘要实际模型输入、已覆盖工具来源回查及原始游标的离线回归。"""
import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from test_智能体独立历史 import CharCounter, make_agent, summary_text
from test_摘要提交与Claimify开关 import Store, PROSE
from service.agent_history import HistoryScope, bind_agent_history, content_hash
from service.agent_context import prepare_agent_messages, source_text, previous_summary_sources, _summary_messages
from service.context_policy import ContextPolicy, InputHistorySettings, SummaryLifecycleSettings
from service.summary_history_view import VIEW_KIND, prepare_summary_records
from service.summary_lifecycle import pack_prose_summary
from service.summary_input_context import locate_context_quote
from service.verification_checks import source_type


class ProcessedSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store, self.counter, self.agent = Store(), CharCounter(), make_agent()
        self.policy = ContextPolicy(group="处理后上下文回归", semantic_verification=False,
            input_history=InputHistorySettings(), summary_strategy="claimify-v1",
            summary_lifecycle=SummaryLifecycleSettings())
        self.scope = HistoryScope(self.store, "run", "session", None, context_policy=self.policy.snapshot())
        self.agent._call_llm_impl = AsyncMock(return_value=PROSE)
        count = patch("service.agent_context.get_qwen_token_counter", return_value=self.counter)
        count.start()
        self.addCleanup(count.stop)

    async def append(self, kind, payload, call_id):
        return await self.store.append(self.scope, self.agent.name, kind, payload, call_id)

    async def prepare(self, budget=800, user_prompt="最新任务"):
        with bind_agent_history(self.scope), patch("service.agent_context.scoped_input_budget",
                side_effect=lambda *a, **kw: 30000 if kw.get("summary") else budget):
            return await prepare_agent_messages(self.agent, self.scope, "系统要求", user_prompt, "current", 6000)

    async def test_all_completed_tools_and_copied_bodies_avoid_summary_and_remain_readable(self):
        tools, bodies = [], []
        for i in range(5):
            body = f"第{i}篇网页的原文。" * 900
            bodies.append(body)
            # 末条仅有终态记录也应外置，不能依赖 tool_input 或排除最新轮次。
            if i < 4:
                await self.append("tool_input", {"url": f"https://example.org/{i}"}, f"tool{i}")
            tools.append(await self.append("tool_output", {"value": {"content": body}}, f"tool{i}"))
        old_input = await self.append("llm_input", {"content": "保留任务约束。\n" +
            "\n".join("已读取正文: " + body for body in bodies)}, "old")
        facts = await self.append("llm_output", {"content": "已提取事实和证据，接下来写作。"}, "old")
        current = "首次待分析正文：" + "当前材料。" * 200
        messages = await self.prepare(budget=4000, user_prompt=current)
        self.agent._call_llm_impl.assert_not_awaited()
        self.assertEqual(messages[-1]["content"], current)
        self.assertIn(facts["payload"]["content"], messages[1]["content"])
        for body in bodies:
            self.assertNotIn(body, messages[1]["content"])
        saved = await self.store.load(self.scope, self.agent.name)
        self.assertEqual(set(saved["offloaded_source_ids"]), {r["source_id"] for r in tools})
        event = next(r["payload"] for r in saved["messages"] if r["kind"] == "context_view")
        self.assertGreaterEqual(event["tokens_before_compression"], event["summary_threshold"])
        self.assertLess(event["tokens_after_early_offload"], event["summary_threshold"])
        self.assertFalse(event["summary_attempted"])
        self.assertIn("completed_tool_rounds", event["stages"])
        self.assertEqual(event["input_history"]["referenced_source_ids"], [old_input["source_id"]])
        for row in [*tools, old_input, facts]:
            self.assertEqual(await self.store.read_source(self.scope, self.agent.name, row["source_id"]), row)
        # 下次预算变大，已外置的记录仍以引用回放。
        next_messages = await self.prepare(budget=20000)
        for body in bodies:
            self.assertNotIn(body, next_messages[1]["content"])
        self.assertEqual((await self.store.load(self.scope, self.agent.name))["summary_version"], 0)

    async def test_initial_threshold_triggers_all_offload_even_when_references_lower_count(self):
        body = "历史网页正文。" * 1500
        tool = await self.append("tool_output", {"value": {"content": body}}, "tool")
        await self.append("llm_input", {"content": "已读取正文: " + body}, "old")
        await self.append("llm_output", {"content": "完成"}, "old")
        await self.prepare(budget=4000)
        saved = await self.store.load(self.scope, self.agent.name)
        event = next(r["payload"] for r in saved["messages"] if r["kind"] == "context_view")
        self.assertGreaterEqual(event["tokens_before_compression"], event["soft_threshold"])
        self.assertLess(event["tokens_after_input_history"], event["soft_threshold"])
        self.assertIn(tool["source_id"], saved["offloaded_source_ids"])
        self.agent._call_llm_impl.assert_not_awaited()

    async def test_remaining_background_summarizes_after_all_tools_are_offloaded(self):
        originals, body, arguments = await self.large_history()
        extra_bodies = [f"额外来源{i}的长原文。" * 600 for i in range(4)]
        for i, extra in enumerate(extra_bodies):
            await self.append("tool_output", {"value": {"content": extra}}, f"extra{i}")
        await self.prepare()
        self.agent._call_llm_impl.assert_awaited_once()
        request = str(json.loads(self.agent._call_llm_impl.call_args.args[1]))
        for original_body in [body, arguments, *extra_bodies]:
            self.assertNotIn(original_body, request)
        self.assertIn("需要保留的任务背景。", request)
        saved = await self.store.load(self.scope, self.agent.name)
        self.assertEqual(len(saved["offloaded_source_ids"]), 6)
        self.assertEqual(saved["summary_version"], 1)

    async def test_inflight_tool_input_is_not_offloaded(self):
        await self.append("tool_output", {"value": {"content": "已完成工具正文。" * 1500}}, "done")
        pending = await self.append("tool_input", {"arguments": {"content": "在途工具输入。" * 1800}}, "pending")
        self.scope = replace(self.scope, context_policy=self.policy.model_copy(update={
            "semantic_summary": False}).snapshot())
        messages = await self.prepare(budget=6000)
        self.assertIn(pending["payload"]["arguments"]["content"], messages[1]["content"])
        saved = await self.store.load(self.scope, self.agent.name)
        self.assertNotIn(pending["source_id"], saved["offloaded_source_ids"])
        self.agent._call_llm_impl.assert_not_awaited()

    async def covered_copy(self):
        body = "已经读取的网页正文。" * 200
        tool = await self.append("tool_output", {"value": {"content": body}}, "tool")
        old = pack_prose_summary(PROSE)
        await self.store.save_summary(self.scope, self.agent.name, expected_version=0,
            covered_seq=tool["seq"], summary=old,
            metadata={"summary_hash": content_hash(old), "reference_quotes": {}})
        row = await self.append("llm_input", {"content": "已读取正文: " + body + "\n保留限制。"}, "old")
        await self.append("llm_output", {"content": "完成"}, "old")
        return tool, row, body

    async def test_covered_tool_source_replaces_copy_without_replaying_old_source(self):
        # 已覆盖来源回查也必须在整体引用化阈值触发后执行。
        self.scope = replace(self.scope, context_policy=self.policy.model_copy(update={
            "soft_token_cap": 500}).snapshot())
        tool, row, body = await self.covered_copy()
        with patch.object(self.store, "load_tool_sources", wraps=self.store.load_tool_sources) as loader:
            messages = await self.prepare(budget=20000)
        loader.assert_awaited_once_with(self.scope, self.agent.name, through_seq=tool["seq"])
        history = json.loads(messages[1]["content"].split("\n", 1)[1])
        self.assertFalse(any(r.get("source_id") == tool["source_id"] for r in history))
        view = next(r["text"] for r in history if r.get("source_id") == row["source_id"])
        self.assertIn(tool["source_id"], view)
        self.assertNotIn(body, view)
        self.assertTrue(view.endswith("保留限制。"))
        self.assertEqual(await self.store.read_source(self.scope, self.agent.name, row["source_id"]), row)
        self.agent._call_llm_impl.assert_not_awaited()

    async def test_archived_tool_lookup_respects_agent_subtask_and_sequence(self):
        tool, row, body = await self.covered_copy()
        self.assertEqual(await self.store.load_tool_sources(self.scope, self.agent.name, through_seq=0), [])
        for scope, name in [(replace(self.scope, context_key="other"), self.agent.name),
                            (self.scope, "LeadWriter"), (replace(self.scope, session_id="other"), self.agent.name)]:
            self.assertEqual(await self.store.load_tool_sources(scope, name, through_seq=row["seq"]), [])
        future = await self.append("tool_output", {"value": {"content": body}}, "future")
        sources = await self.store.load_tool_sources(self.scope, self.agent.name, through_seq=tool["seq"])
        self.assertEqual([r["source_id"] for r in sources], [tool["source_id"]])
        self.assertNotIn(future["source_id"], str(sources))

    async def large_history(self):
        body = "研究纳入800人。" + "仅在原始正文尾部存在的资料。" * 350
        arguments = "搜索结果参数正文。" * 400
        records = [await self.append("tool_input", {"arguments": {"content": arguments}}, "tool1"),
                   await self.append("tool_output", {"tool_name": "v2_web_fulltext_read", "value": {"content": body}}, "tool1")]
        records += [await self.append("tool_input", {"arguments": {"url": "example"}}, "tool2"),
                    await self.append("tool_output", {"value": {"content": "短资料"}}, "tool2")]
        records += [await self.append("llm_input", {"content": "需要保留的任务背景。" * 800 + "\n已读取正文: " + body}, "old"),
                    await self.append("llm_output", {"content": "完成"}, "old")]
        return records, body, arguments

    async def test_actual_summary_request_uses_processed_views_preserves_raw_and_cursor(self):
        originals, body, arguments = await self.large_history()
        await self.prepare()
        self.assertEqual(self.agent._call_llm_impl.await_count, 1)
        request = json.loads(self.agent._call_llm_impl.call_args.args[1])
        fragments = request["new_records"]
        text = str(fragments)
        self.assertNotIn(body, text)
        self.assertNotIn(arguments, text)
        self.assertIn("资料已保存", text)
        self.assertIn("原文已保存", text)
        self.assertIn("需要保留的任务背景。", text)
        saved = await self.store.load(self.scope, self.agent.name)
        self.assertEqual(saved["covered_seq"], originals[-1]["seq"])
        snapshots = [r for r in saved["messages"] if r["kind"] == VIEW_KIND]
        self.assertEqual(len(snapshots), 3)
        self.assertTrue(all(r["seq"] > saved["covered_seq"] for r in snapshots))
        sent = {f["source_id"]: f["text"] for f in fragments}
        for snapshot in snapshots:
            self.assertEqual(sent[snapshot["source_id"]], source_text(snapshot))
            self.assertEqual(snapshot["content_hash"], content_hash(snapshot["payload"]))
        for original in originals:
            self.assertEqual(await self.store.read_source(self.scope, self.agent.name, original["source_id"]), original)
        metadata = saved["summary_metadata"]["summary_input"]
        self.assertGreater(metadata["saved_tokens"], 0)
        self.assertEqual(len(metadata["source_mapping"]), 3)
        event = next(r["payload"] for r in saved["messages"] if r["kind"] == "summary_input_prepared")
        self.assertEqual(event, metadata)
        messages = await self.prepare()
        self.assertEqual(self.agent._call_llm_impl.await_count, 1)
        self.assertNotIn(VIEW_KIND, messages[1]["content"])

    async def test_claimify_context_contains_only_actual_views_and_runs_after_commit(self):
        self.scope = replace(self.scope, context_policy=self.policy.model_copy(update={
            "summary_lifecycle": SummaryLifecycleSettings(claimify_enabled=True)}).snapshot())
        originals, body, _ = await self.large_history()
        claim = "研究纳入800人。"
        async def audit(*args, **kwargs):
            self.assertEqual((await self.store.load(self.scope, self.agent.name))["covered_seq"], originals[-1]["seq"])
            context = kwargs["input_context"]
            row = next(r for r in context["rows"].values() if r["kind"] == VIEW_KIND
                       and r["payload"]["original_source_id"] == originals[1]["source_id"])
            self.assertEqual(source_type(row), "research_source")
            text = source_text(row)
            start = text.index(claim)
            ref = {"source_id": row["source_id"], "quote": claim, "start": start, "end": start + len(claim)}
            located = locate_context_quote(ref, context)
            self.assertIn(claim, located["context"])
            self.assertNotIn(body, located["context"])
            with self.assertRaisesRegex(ValueError, "实际提供"):
                locate_context_quote({**ref, "start": len(text), "end": len(text) + 10}, context)
            return summary_text(row["source_id"], claim)
        with patch("service.summary_citations.summarize_with_citations", AsyncMock(side_effect=audit)) as auditor:
            await self.prepare()
        auditor.assert_awaited_once()
        metadata = (await self.store.load(self.scope, self.agent.name))["summary_metadata"]
        self.assertEqual(metadata["validation"]["claimify"], "completed")

    async def test_old_derived_quote_recheck_uses_original_cursor_not_snapshot_sequence(self):
        original = await self.append("llm_output", {"content": "研究纳入800人。" + "背景" * 1000}, "old")
        views = {original["source_id"]: "研究纳入800人。"}
        processed, _ = await prepare_summary_records(self.agent, self.scope, [original], views, self.counter, "summary")
        snapshot = processed[0]
        saved = {"summary": summary_text(snapshot["source_id"], "研究纳入800人。"),
                 "summary_metadata": {}, "covered_seq": original["seq"]}
        self.assertGreater(snapshot["seq"], saved["covered_seq"])
        self.assertEqual(await previous_summary_sources(self.scope, self.agent.name, saved),
                         {snapshot["source_id"]: ["研究纳入800人。"]})
        with self.assertRaisesRegex(ValueError, "已覆盖"):
            await previous_summary_sources(self.scope, self.agent.name, {**saved, "covered_seq": 0})

    async def test_tampered_original_cannot_become_processed_snapshot(self):
        original = await self.append("llm_output", {"content": "原文"}, "old")
        original["payload"]["content"] = "篡改"
        with self.assertRaisesRegex(ValueError, "哈希"):
            await prepare_summary_records(self.agent, self.scope, [original],
                {original["source_id"]: "视图"}, self.counter, "summary")
        self.assertFalse(any(r["kind"] == VIEW_KIND for r in (await self.store.load(self.scope, self.agent.name))["messages"]))

    async def test_legacy_summary_request_preserves_original_message_role(self):
        original = await self.append("llm_output", {"content": "助手陈述" + "背景" * 1000}, "old")
        processed, _ = await prepare_summary_records(self.agent, self.scope, [original],
            {original["source_id"]: "助手陈述"}, self.counter, "summary")
        item = json.loads(_summary_messages(processed)[1]["content"])["new_records"][0]
        self.assertEqual(item, {"source_id": processed[0]["source_id"], "kind": "llm_output", "text": "助手陈述"})
        self.assertEqual(source_type(processed[0]), "assistant_history")


if __name__ == "__main__":
    unittest.main()
