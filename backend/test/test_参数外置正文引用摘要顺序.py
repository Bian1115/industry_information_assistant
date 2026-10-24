"""三阶段历史处理顺序及最近5条参数保护；仅使用本地桩。"""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "app"), str(Path(__file__).resolve().parent)]
from test_智能体独立历史 import MemoryStore, CharCounter, make_agent
from service.agent_history import HistoryScope, bind_agent_history, canonical_json
from service.agent_context import prepare_agent_messages, source_text, REPLAY_KINDS
from service.context_policy import AGENTS, ContextPolicy, InputBudget, InputHistorySettings
from service.input_history_view import prepare_input_views

EMPTY_SUMMARY = ("## SESSION INTENT\nNone\n\n## SUMMARY\nNone\n\n## ARTIFACTS\nNone"
                 "\n\n## NEXT STEPS\nNone\n\n## CLAIMS\n[]")


class OrderedCompressionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store, self.counter = MemoryStore(), CharCounter()
        self.agent = make_agent("LeadWriter")
        self.policy = ContextPolicy(
            group="参数正文摘要顺序回归", summary_strategy="incremental-v3",
            semantic_summary=False, semantic_verification=False, soft_token_cap=10000,
            input_history=InputHistorySettings(),
            stage_budgets={name: InputBudget(context_window=30000, max_input=20000) for name in AGENTS},
            output_limits={name: 3000 for name in AGENTS},
        ).with_agent_compression({"LeadWriter": {"enabled": True,
            "reference_materials": True, "artifact_offload": True, "semantic_summary": False}})
        self.scope = HistoryScope(self.store, "run", "session", None,
            contract={"query": "当前契约"}, context_policy=self.policy.snapshot())
        counter = patch("service.agent_context.get_qwen_token_counter", return_value=self.counter)
        counter.start()
        self.addCleanup(counter.stop)

    async def append(self, kind, payload, call):
        return await self.store.append(self.scope, "LeadWriter", kind, payload, call)

    async def seed(self, background="保留限制"):
        self.parameters, self.outputs, self.arguments = [], [], []
        body = "已经处理的工具正文。" * 300
        for i in range(4):
            arguments = {"content": f"工具{i}的参数。" * 500, "query": f"检索{i}"}
            self.parameters.append(await self.append("tool_input",
                {"tool_name": "capture", "arguments": arguments, "request_id": str(i)}, f"tool{i}"))
            self.outputs.append(await self.append("tool_output",
                {"tool_name": "capture", "value": {"content": body}}, f"tool{i}"))
            self.arguments.append(arguments)
        self.old_input = await self.append("llm_input", {"content": canonical_json(
            {"source_evidence": body, "revision_instructions": [background]})}, "old")
        await self.append("llm_output", {"content": "任务完成"}, "old")
        self.body = body

    async def prepare(self, cap=500, *, summary=False, policy=None):
        root = (policy or self.policy).model_copy(update={"soft_token_cap": cap})
        root = root.with_agent_compression({"LeadWriter": {"semantic_summary": summary}})
        scope = replace(self.scope, context_policy=root.snapshot())
        with bind_agent_history(scope):
            messages = await prepare_agent_messages(self.agent, scope, "系统", "最新输入", "current", 3000)
        saved = await self.store.load(scope, "LeadWriter")
        event = [r["payload"] for r in saved["messages"] if r["kind"] == "context_view"][-1]
        return messages, event, saved

    def history(self, messages):
        return {r["source_id"]: r["text"] for r in json.loads(messages[1]["content"].split("\n", 1)[1])
                if "source_id" in r}

    async def test_parameters_run_before_bodies_and_protect_exactly_five_messages(self):
        await self.seed()
        async def inspect_first_stage(agent, scope, rows, completed, views, *args, **kwargs):
            self.assertNotIn(self.arguments[0]["content"], views[self.parameters[0]["source_id"]])
            self.assertIn(self.body, views[self.outputs[0]["source_id"]])
            self.assertIn(self.body, views[self.old_input["source_id"]])
            return await prepare_input_views(agent, scope, rows, completed, views, *args, **kwargs)
        with patch("service.input_history_view.prepare_input_views", side_effect=inspect_first_stage):
            messages, event, saved = await self.prepare()
        history = self.history(messages)
        self.assertEqual(event["tool_parameter_source_ids"], [r["source_id"] for r in self.parameters[:3]])
        originals = [r for r in saved["messages"] if r["kind"] in REPLAY_KINDS]
        self.assertEqual(event["tool_parameter_protected_source_ids"], [r["source_id"] for r in originals[-5:]])
        self.assertEqual(event["tool_parameter_keep_messages"], 5)
        for row in self.parameters[:3]:
            view = json.loads(history[row["source_id"]])
            self.assertEqual(view["tool_name"], "capture")
            self.assertEqual(view["request_id"], row["payload"]["request_id"])
            self.assertIn(row["source_id"], view["arguments"])
        self.assertEqual(history[self.parameters[-1]["source_id"]], source_text(self.parameters[-1]))
        # 最新工具结果仍参与正文引用；5条保护不能连带排除它。
        self.assertNotIn(self.body, history[self.outputs[-1]["source_id"]])
        self.assertLess(event["stages"].index("older_tool_parameters"), event["stages"].index("input_material_references"))
        self.assertLess(event["tokens_after_tool_parameters"], event["tokens_before_compression"])
        self.assertLess(event["tokens_after_body_references"], event["tokens_after_tool_parameters"])
        self.assertEqual(messages[-1]["content"], "最新输入")
        for row in self.parameters + self.outputs:
            self.assertEqual(await self.store.read_source(self.scope, "LeadWriter", row["source_id"]), row)

    async def test_below_context_threshold_keeps_even_long_parameters(self):
        await self.seed()
        _, baseline, _ = await self.prepare(10000)
        cap = baseline["tokens_before_compression"] + 1
        messages, event, _ = await self.prepare(cap)
        self.assertFalse(event["offload_triggered"])
        self.assertEqual(event["tool_parameter_source_ids"], [])
        for row in self.parameters:
            self.assertEqual(self.history(messages)[row["source_id"]], source_text(row))

    async def test_parameter_size_boundary_is_strictly_above_tool_limit(self):
        arguments = {"content": "参数" * 1700}
        row = await self.append("tool_input", {"tool_name": "capture", "arguments": arguments}, "old")
        await self.append("tool_output", {"value": "完成"}, "old")
        for i in range(4):
            await self.append("llm_output", {"content": "完成"}, f"tail{i}")
        size = self.counter.count_text(canonical_json(arguments))
        for limit, expected in [(size, False), (size - 1, True)]:
            with self.subTest(limit=limit):
                policy = self.policy.model_copy(update={"tool_content_tokens": limit})
                _, event, _ = await self.prepare(policy=policy)
                self.assertEqual(row["source_id"] in event["tool_parameter_source_ids"], expected)

    async def test_old_inflight_parameters_remain_raw_and_errors_stay_visible(self):
        arguments = {"content": "未结束参数" * 1000}
        pending = await self.append("tool_input", {"arguments": arguments}, "pending")
        done = await self.append("tool_input", {"arguments": arguments}, "failed")
        error = await self.append("tool_error", {"tool_name": "capture", "message": "定位失败详情" * 1000}, "failed")
        for i in range(5):
            await self.append("llm_output", {"content": "完成"}, f"tail{i}")
        messages, event, _ = await self.prepare()
        history = self.history(messages)
        self.assertEqual(history[pending["source_id"]], source_text(pending))
        self.assertEqual(history[error["source_id"]], source_text(error))
        self.assertEqual(event["tool_parameter_source_ids"], [done["source_id"]])

    async def test_body_references_still_run_after_parameters_drop_below_threshold(self):
        # 大参数位于最近5条之外；单独外置参数即可低于整体阈值。
        row = await self.append("tool_input", {"arguments": {"content": "很长参数" * 6000}}, "old")
        body = "可引用正文。" * 150
        tool = await self.append("tool_output", {"value": {"content": body}}, "old")
        await self.append("llm_input", {"content": canonical_json({"source_evidence": body})}, "llm")
        await self.append("llm_output", {"content": "完成"}, "llm")
        for i in range(2):
            await self.append("llm_output", {"content": "完成"}, f"tail{i}")
        summarizer = AsyncMock(return_value=EMPTY_SUMMARY)
        with patch("service.agent_context.summarize_history", summarizer):
            messages, event, _ = await self.prepare(2500, summary=True)
        self.assertGreaterEqual(event["tokens_before_compression"], event["soft_threshold"])
        self.assertLess(event["tokens_after_tool_parameters"], event["soft_threshold"])
        self.assertNotIn(body, self.history(messages)[tool["source_id"]])
        self.assertIn(row["source_id"], event["tool_parameter_source_ids"])
        summarizer.assert_not_awaited()

    async def test_summary_uses_same_threshold_and_only_processed_history(self):
        await self.seed(background="保留背景。" * 300)
        summarizer = AsyncMock(return_value=EMPTY_SUMMARY)
        with patch("service.agent_context.summarize_history", summarizer):
            _, event, saved = await self.prepare(500, summary=True)
        self.assertGreaterEqual(event["tokens_after_body_references"], event["soft_threshold"])
        self.assertLess(event["tokens_after_body_references"], event["configured_summary_threshold"])
        self.assertEqual(event["summary_threshold"], event["soft_threshold"])
        summarizer.assert_awaited_once()
        processed = "\n".join(source_text(row) for row in summarizer.call_args.args[2])
        self.assertNotIn(self.arguments[0]["content"], processed)
        self.assertIn(self.arguments[-1]["content"], processed)
        self.assertNotIn(self.body, processed)
        self.assertIn("保留背景。", processed)
        self.assertEqual(event["stages"][-1], "history_summary")
        self.assertEqual(saved["summary_version"], 1)

    async def test_saved_offload_never_bypasses_recent_parameter_protection(self):
        arguments = {"content": "近期参数" * 1000}
        row = await self.append("tool_input", {"arguments": arguments}, "done")
        await self.append("tool_output", {"value": {"content": "输出正文" * 1000}}, "done")
        await self.store.mark_offloaded(self.scope, "LeadWriter", [row["source_id"]])
        messages, event, _ = await self.prepare()
        self.assertEqual(self.history(messages)[row["source_id"]], source_text(row))
        self.assertNotIn(row["source_id"], event["tool_parameter_source_ids"])

    async def test_offload_switch_does_not_remove_parameters(self):
        await self.seed()
        policy = self.policy.with_agent_compression({"LeadWriter": {"artifact_offload": False}})
        messages, event, _ = await self.prepare(policy=policy)
        self.assertEqual(event["tool_parameter_source_ids"], [])
        self.assertTrue(event["input_history"]["referenced_source_ids"])
        for row in self.parameters:
            self.assertEqual(self.history(messages)[row["source_id"]], source_text(row))


if __name__ == "__main__":
    unittest.main()
