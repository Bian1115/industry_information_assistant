"""整体阈值、全量历史处理及原文保护的离线回归，不调用模型或数据库。"""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "app"), str(Path(__file__).resolve().parent)]
from test_智能体独立历史 import MemoryStore, CharCounter, make_agent
from service.agent_history import HistoryScope, bind_agent_history
from service.agent_context import prepare_agent_messages
from service.context_policy import AGENTS, ContextPolicy, InputBudget, InputHistorySettings


class ReferenceThresholdTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store, self.counter = MemoryStore(), CharCounter()
        self.policy = ContextPolicy(
            group="正文引用化阈值回归", summary_strategy="incremental-v3",
            semantic_summary=False, semantic_verification=False, soft_token_cap=10000,
            input_history=InputHistorySettings(min_reference_tokens=10000),
            stage_budgets={name: InputBudget(context_window=30000, max_input=20000) for name in AGENTS},
            output_limits={name: 3000 for name in AGENTS},
        ).with_agent_compression({"LeadWriter": {"enabled": True,
            "reference_materials": True, "artifact_offload": True, "semantic_summary": False}})
        self.scope = HistoryScope(self.store, "run", "session", None,
            contract={"query": "任务约束必须保留"}, context_policy=self.policy.snapshot())
        count = patch("service.agent_context.get_qwen_token_counter", return_value=self.counter)
        count.start()
        self.addCleanup(count.stop)

    async def seed(self, count=2):
        self.originals, self.bodies, self.inputs, self.outputs, self.tools = [], [], [], [], []
        for number in range(count):
            body = f"来源{number}的正文。" * 100
            tool = await self.store.append(self.scope, "LeadWriter", "tool_output",
                {"value": {"content": body}}, f"tool{number}")
            original = json.dumps({"source_evidence": body, "revision_instructions": [f"历史指令{number}"]}, ensure_ascii=False)
            item = await self.store.append(self.scope, "LeadWriter", "llm_input", {"content": original}, f"done{number}")
            output = await self.store.append(self.scope, "LeadWriter", "llm_output",
                {"content": json.dumps({"section_content": body, "addressed_issues": [f"issue{number}"]}, ensure_ascii=False)},
                f"done{number}")
            self.bodies.append(body)
            self.tools.append(tool)
            self.inputs.append(item)
            self.outputs.append(output)
            self.originals.extend([tool, item, output])

    async def prepare(self, cap, prompt="当前输入完整保留", policy=None):
        root = (policy or self.policy).model_copy(update={"soft_token_cap": cap})
        scope = replace(self.scope, context_policy=root.snapshot())
        with bind_agent_history(scope):
            messages = await prepare_agent_messages(make_agent("LeadWriter"), scope,
                "系统约束完整保留", prompt, "current", 3000)
        saved = await self.store.load(scope, "LeadWriter")
        event = [row["payload"] for row in saved["messages"] if row["kind"] == "context_view"][-1]
        return messages, event

    def history(self, messages):
        return {row["source_id"]: row["text"] for row in json.loads(messages[1]["content"].split("\n", 1)[1])}

    async def test_below_threshold_keeps_input_output_and_tools_as_originals(self):
        await self.seed()
        messages, event = await self.prepare(10000)
        history = self.history(messages)
        self.assertLess(event["tokens_before_compression"], event["soft_threshold"])
        self.assertFalse(event["offload_triggered"])
        self.assertEqual(event["stages"], [])
        for row in self.inputs + self.outputs:
            self.assertEqual(history[row["source_id"]], row["payload"]["content"])
        for body in self.bodies:
            self.assertIn(body, messages[1]["content"])
        self.assertEqual(event["tokens"], event["tokens_before_compression"])

    async def test_threshold_boundary_references_at_and_above_but_not_below(self):
        await self.seed()
        _, baseline = await self.prepare(10000)
        count = baseline["tokens_before_compression"]
        for cap, triggered in [(count + 1, False), (count, True), (count - 1, True)]:
            with self.subTest(cap=cap):
                _, event = await self.prepare(cap)
                self.assertEqual(event["offload_triggered"], triggered)
                self.assertEqual(bool(event["input_history"]["referenced_source_ids"]), triggered)

    async def test_trigger_traverses_all_completed_history_including_latest_and_small_fields(self):
        await self.seed(count=4)
        messages, event = await self.prepare(500)
        history = self.history(messages)
        self.assertEqual(set(event["input_history"]["referenced_source_ids"]), {row["source_id"] for row in self.inputs})
        self.assertEqual(set(event["shortened_source_ids"]), {row["source_id"] for row in self.tools})
        self.assertIn("agent_output_artifact_references", event["stages"])
        for number, body in enumerate(self.bodies):
            self.assertLess(self.counter.count_text(body), self.policy.tool_content_tokens)
            for row in [self.inputs[number], self.outputs[number]]:
                self.assertNotIn(body, history[row["source_id"]])
                self.assertIn("资料已保存", history[row["source_id"]])
            self.assertIn(f"历史指令{number}", history[self.inputs[number]["source_id"]])
            self.assertIn(f"issue{number}", history[self.outputs[number]["source_id"]])
        for row in self.originals:
            self.assertEqual(await self.store.read_source(self.scope, "LeadWriter", row["source_id"]), row)

    async def test_current_and_unfinished_inputs_are_not_referenced(self):
        await self.seed()
        current = json.dumps({"source_evidence": "本轮正文" * 200}, ensure_ascii=False)
        await self.store.append(self.scope, "LeadWriter", "llm_input", {"content": current}, "current")
        pending = json.dumps({"source_evidence": "未完成正文" * 300}, ensure_ascii=False)
        item = await self.store.append(self.scope, "LeadWriter", "llm_input", {"content": pending}, "pending")
        tool = await self.store.append(self.scope, "LeadWriter", "tool_input",
            {"content": "在途工具正文" * 200}, "pending_tool")
        messages, event = await self.prepare(500, prompt=current)
        self.assertEqual(messages[-1]["content"], current)
        self.assertIn("任务约束必须保留", messages[0]["content"])
        self.assertIn("系统约束完整保留", messages[0]["content"])
        self.assertEqual(self.history(messages)[item["source_id"]], pending)
        self.assertIn(tool["payload"]["content"], messages[1]["content"])
        self.assertNotIn(item["source_id"], event["input_history"]["referenced_source_ids"])
        self.assertNotIn(tool["source_id"], event["shortened_source_ids"])

    async def test_reference_and_output_offload_switches_remain_independent(self):
        await self.seed()
        root = self.policy.with_agent_compression({"LeadWriter": {"artifact_offload": False}})
        messages, event = await self.prepare(500, policy=root)
        self.assertTrue(event["input_history"]["referenced_source_ids"])
        self.assertNotIn("agent_output_artifact_references", event["stages"])
        for row in self.outputs:
            self.assertEqual(self.history(messages)[row["source_id"]], row["payload"]["content"])

    async def test_later_small_context_restores_originals_even_when_previously_offloaded(self):
        await self.seed(count=1)
        _, first = await self.prepare(1500, prompt="较大的当前任务" * 2000)
        self.assertTrue(first["offload_triggered"])
        self.assertTrue(first["shortened_source_ids"])
        messages, second = await self.prepare(1500)
        self.assertFalse(second["offload_triggered"])
        self.assertEqual(second["shortened_source_ids"], [])
        for body in self.bodies:
            self.assertIn(body, messages[1]["content"])

    async def test_short_body_is_kept_when_reference_would_increase_input(self):
        await self.seed()
        text = json.dumps({"source_evidence": "短正文"}, ensure_ascii=False)
        row = await self.store.append(self.scope, "LeadWriter", "llm_input", {"content": text}, "tiny")
        await self.store.append(self.scope, "LeadWriter", "llm_output", {"content": "完成"}, "tiny")
        messages, event = await self.prepare(500)
        self.assertTrue(event["offload_triggered"])
        self.assertEqual(self.history(messages)[row["source_id"]], text)


if __name__ == "__main__":
    unittest.main()
