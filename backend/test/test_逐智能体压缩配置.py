"""逐 Agent 方法选择、冻结兼容和实际发送视图；使用模拟模型与内存历史。"""
import json
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "app"), str(Path(__file__).resolve().parent)]
from test_智能体独立历史 import MemoryStore, CharCounter, make_agent, summary_text
from service.agent_history import HistoryScope, bind_agent_history
from service.agent_context import prepare_agent_messages, ContextBudgetExceeded, evidence_batch_input
from service.context_policy import (AGENTS, ContextPolicy, InputBudget, InputHistorySettings,
                                   policy_for_scope, selected_policy)
from service.input_history_view import reference_agent_materials


def policy(**kwargs):
    base = dict(group="逐智能体测试", summary_strategy="incremental-v3", semantic_verification=False,
                input_history=InputHistorySettings(), soft_token_cap=2000,
                stage_budgets={name: InputBudget(max_input=4000) for name in AGENTS})
    base.update(kwargs)
    return ContextPolicy(**base)


class SettingsTests(unittest.TestCase):
    def test_global_switch_preserves_replay_and_frozen_snapshot(self):
        root = policy(compression_enabled=False).with_agent_compression({"LeadWriter": {"enabled": True}})
        scope = HistoryScope(None, "r", "s", None, context_policy=root.snapshot())
        with patch.dict(os.environ, {"CONTEXT_COMPRESSION_ENABLED": "1"}):
            view = policy_for_scope(scope, "LeadWriter")
            self.assertEqual(view.replay_agents, ["LeadWriter"])
            self.assertFalse(view.artifact_offload)
            self.assertFalse(view.semantic_summary)
            self.assertFalse(view.semantic_verification)
            self.assertIsNone(view.input_history)
            self.assertFalse(ContextPolicy.from_snapshot(root.snapshot()).compression_enabled)
        old = policy()
        self.assertNotIn("compression_enabled", old.snapshot()["config"])
        self.assertEqual(ContextPolicy.from_snapshot(old.snapshot()).fingerprint, old.fingerprint)

    def test_partial_agent_overrides_do_not_change_other_agents(self):
        root = policy().with_agent_compression({"LeadWriter": {"enabled": True, "semantic_summary": False}})
        root2 = root.with_agent_compression({"ChiefArchitect": {"enabled": True, "artifact_offload": False}})
        self.assertEqual(root.agent_compression["LeadWriter"], root2.agent_compression["LeadWriter"])
        self.assertFalse(root.agent_compression["ChiefArchitect"].enabled)
        self.assertTrue(root2.agent_compression["ChiefArchitect"].enabled)
        self.assertTrue(root.agent_compression["DeepScout"].enabled)

    def test_old_snapshot_hash_and_behavior_remain_identical(self):
        root = policy()
        self.assertNotIn("agent_compression", root.snapshot()["config"])
        loaded = ContextPolicy.from_snapshot(root.snapshot())
        self.assertEqual(loaded.fingerprint, root.fingerprint)
        self.assertIsNone(loaded.agent_compression)

    def test_each_method_can_be_selected_independently(self):
        root = policy(semantic_summary=False).with_agent_compression({
            "LeadWriter": {"enabled": True, "reference_materials": True, "artifact_offload": False, "semantic_summary": False},
            "ChiefArchitect": {"enabled": True, "reference_materials": False, "artifact_offload": False, "semantic_summary": True},
            "DeepScout": {"enabled": False}})
        scope = HistoryScope(None, "r", "s", None, context_policy=root.snapshot())
        writer, architect = policy_for_scope(scope, "LeadWriter"), policy_for_scope(scope, "ChiefArchitect")
        self.assertTrue(writer.input_history.reference_materials)
        self.assertFalse(writer.artifact_offload)
        self.assertFalse(writer.semantic_summary)
        self.assertTrue(architect.semantic_summary)
        self.assertIsNone(architect.input_history)
        self.assertEqual(writer.fingerprint, root.fingerprint)
        self.assertEqual(writer.snapshot(), root.snapshot())
        self.assertEqual(policy_for_scope(scope, "DeepScout").replay_agents, [])
        self.assertEqual(policy_for_scope(scope, "DeepScout").history_read_max_calls, 0)

    def test_settings_are_strict_and_unknown_agents_rejected(self):
        for overrides in ({"Unknown": {"enabled": True}}, {"LeadWriter": {"enabled": "true"}},
                          {"LeadWriter": {"unknown": True}},
                          {"LeadWriter": {"enabled": True, "semantic_summary": False, "semantic_verification": True}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                policy().with_agent_compression(overrides)
        with self.assertRaisesRegex(ValueError, "有界历史回读"):
            policy(history_read_max_calls=0).with_agent_compression({"LeadWriter": {"enabled": True}})

    def test_environment_overrides_are_frozen_and_files_still_take_priority(self):
        with patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": "", "AGENT_SUMMARY_STRATEGY": "incremental-v3",
             "AGENT_COMPRESSION_CONFIG": '{"LeadWriter":{"enabled":true,"semantic_summary":false}}'}):
            root = selected_policy()
        scope = HistoryScope(None, "r", "s", None, context_policy=root.snapshot())
        with patch.dict(os.environ, {"AGENT_COMPRESSION_CONFIG": '{"LeadWriter":{"enabled":false}}'}):
            self.assertEqual(policy_for_scope(scope, "LeadWriter").replay_agents, ["LeadWriter"])
        preset = Path(__file__).resolve().parents[1] / "eval/configs/上下文压缩消融/02-新框架无压缩基线.json"
        with patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": str(preset), "AGENT_COMPRESSION_CONFIG": "invalid-json"}):
            self.assertIsNone(selected_policy().agent_compression)

    def test_api_accepts_partial_typed_overrides_and_rejects_unknown_agents(self):
        from router.research_router import ResearchRequest
        params = {"query": "研究", "session_id": "00000000-0000-0000-0000-000000000001",
                  "turn_id": "00000000-0000-0000-0000-000000000002"}
        value = ResearchRequest(**params, agent_compression={"LeadWriter": {"enabled": True}})
        self.assertEqual(value.agent_compression["LeadWriter"].model_dump(exclude_unset=True), {"enabled": True})
        for config in ({"NotAgent": {"enabled": True}}, {"LeadWriter": {"enabled": 1}}):
            with self.assertRaises(ValueError):
                ResearchRequest(**params, agent_compression=config)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store, self.counter = MemoryStore(), CharCounter()
        p = patch("service.agent_context.get_qwen_token_counter", return_value=self.counter)
        p.start(); self.addCleanup(p.stop)

    def scope(self, root):
        return HistoryScope(self.store, "r", "s", None, contract={"query": "当前约束"}, context_policy=root.snapshot())

    async def append(self, scope, agent, text, kind="llm_input"):
        row = await self.store.append(scope, agent, kind, {"content": text}, "old")
        if kind == "llm_input":
            await self.store.append(scope, agent, "llm_output", {"content": '{"done":true}'}, "old")
        return row

    async def prepare(self, scope, name, prompt="当前任务"):
        with bind_agent_history(scope):
            return await prepare_agent_messages(make_agent(name), scope, "系统", prompt, "new", 3000)

    async def test_writer_reference_only_hides_old_evidence_and_preserves_current_evidence(self):
        root = policy().with_agent_compression({"LeadWriter": {"enabled": True,
            "reference_materials": True, "artifact_offload": False, "semantic_summary": False}})
        scope = self.scope(root)
        old = json.dumps({"revision_instructions": ["旧修订决定"], "source_evidence": "旧证据" * 12000}, ensure_ascii=False)
        row = await self.append(scope, "LeadWriter", old)
        messages = await self.prepare(scope, "LeadWriter", "当前证据只提供一次：当期收入20亿元")
        self.assertIn("旧修订决定", messages[1]["content"])
        self.assertNotIn("旧证据" * 5, messages[1]["content"])
        self.assertIn(row["source_id"], messages[1]["content"])
        self.assertIn("当期收入20亿元", messages[-1]["content"])
        self.assertLessEqual(self.counter.count_messages(messages), 4000)
        self.assertEqual((await self.store.read_source(scope, "LeadWriter", row["source_id"]))["payload"]["content"], old)

    async def test_architect_summary_is_independent_of_writer_summary_switch(self):
        root = policy(semantic_summary=False).with_agent_compression({
            "ChiefArchitect": {"enabled": True, "reference_materials": False, "artifact_offload": False, "semantic_summary": True},
            "LeadWriter": {"enabled": True, "reference_materials": False, "artifact_offload": False, "semantic_summary": False}})
        scope = self.scope(root)
        text = "已有研究结论。" + "长背景" * 10000
        row = await self.append(scope, "ChiefArchitect", text)
        await self.append(scope, "LeadWriter", text)
        summarizer = AsyncMock(return_value=summary_text(row["source_id"], "已有研究结论。"))
        with patch("service.agent_context.summarize_history", summarizer):
            messages = await self.prepare(scope, "ChiefArchitect")
            self.assertLessEqual(self.counter.count_messages(messages), 4000)
            summarizer.assert_awaited_once()
            with self.assertRaises(ContextBudgetExceeded):
                await self.prepare(scope, "LeadWriter")
            self.assertEqual(summarizer.await_count, 1)
        self.assertEqual((await self.store.load(scope, "ChiefArchitect"))["summary_version"], 1)
        self.assertEqual((await self.store.load(scope, "LeadWriter"))["summary_version"], 0)

    async def test_output_offload_keeps_revision_decisions_without_old_draft_body(self):
        root = policy().with_agent_compression({"LeadWriter": {"enabled": True,
            "reference_materials": False, "artifact_offload": True, "semantic_summary": False}})
        scope = self.scope(root)
        row = await self.append(scope, "LeadWriter", json.dumps({"revised_content": "旧稿正文" * 12000,
            "addressed_issues": ["issue_1"], "unable_to_address": ["仍缺统计口径"]}, ensure_ascii=False), kind="llm_output")
        messages = await self.prepare(scope, "LeadWriter")
        self.assertIn("issue_1", messages[1]["content"])
        self.assertIn("仍缺统计口径", messages[1]["content"])
        self.assertIn(row["source_id"], messages[1]["content"])
        self.assertNotIn("旧稿正文" * 5, messages[1]["content"])

    async def test_disabled_agent_does_not_load_own_or_other_agent_history(self):
        root = policy().with_agent_compression({"ChiefArchitect": {"enabled": False}, "LeadWriter": {"enabled": True}})
        scope = self.scope(root)
        await self.append(scope, "LeadWriter", "Writer专属历史")
        await self.append(scope, "ChiefArchitect", "Architect旧历史")
        messages = await self.prepare(scope, "ChiefArchitect")
        self.assertEqual(len(messages), 2)
        writer = await self.prepare(scope, "LeadWriter")
        self.assertIn("Writer专属历史", writer[1]["content"])
        self.assertNotIn("Architect旧历史", writer[1]["content"])

    async def test_scout_extraction_stays_independent_even_when_compression_is_enabled(self):
        root = policy().with_agent_compression({"DeepScout": {"enabled": True}})
        scope = self.scope(root)
        await self.append(scope, "DeepScout", "Scout历史" * 10000)
        with evidence_batch_input():
            messages = await self.prepare(scope, "DeepScout", "当前编号原句")
        self.assertEqual(len(messages), 2)

    async def test_global_off_replays_original_history_and_stops_at_budget_without_summary(self):
        root = policy(compression_enabled=False).with_agent_compression({"LeadWriter": {"enabled": True}})
        scope = self.scope(root)
        await self.append(scope, "LeadWriter", "原始证据及旧稿" * 4000)
        await self.store.save_summary(scope, "LeadWriter", expected_version=0, covered_seq=2,
                                      summary="旧压缩摘要", metadata={})
        summarizer = AsyncMock()
        with patch("service.agent_context.summarize_history", summarizer):
            with self.assertRaises(ContextBudgetExceeded):
                await self.prepare(scope, "LeadWriter")
        summarizer.assert_not_awaited()
        saved = await self.store.load(scope, "LeadWriter")
        view = [r for r in saved['messages'] if r['kind'] == 'context_view'][-1]['payload']
        self.assertTrue(view['history_replay'])
        self.assertFalse(view['compression_enabled'])
        self.assertEqual(view['stages'], [])
        self.assertEqual(view['tokens'], view['tokens_before_compression'])
        self.assertEqual(view['shortened_source_ids'], [])
        self.assertEqual(saved['summary'], "旧压缩摘要")
        self.assertEqual(saved['covered_seq'], 2)

    def test_reference_spans_point_to_exact_original_material_not_nested_or_task_fields(self):
        original = json.dumps({"query": "必须保留的任务", "nested": {"source_evidence": "嵌套字段" * 1000},
                               "source_evidence": "原句正文" * 1000}, ensure_ascii=False)
        row = {"source_id": "source", "content_hash": "hash"}
        view, refs = reference_agent_materials(row, original, self.counter, 128)
        self.assertEqual(len(refs), 1)
        self.assertEqual(json.loads(original[refs[0]["start"]:refs[0]["end"]]), "原句正文" * 1000)
        self.assertIn("必须保留的任务", view)
        self.assertIn("嵌套字段", view)
        json.loads(view)

    async def test_writer_revision_template_externalizes_old_body_but_keeps_feedback(self):
        from service.deep_research_v2.agents.writer import LeadWriter
        root = policy().with_agent_compression({"LeadWriter": {"enabled": True,
            "reference_materials": True, "artifact_offload": False, "semantic_summary": False}})
        scope = self.scope(root)
        original = LeadWriter.SECTION_REVISION_PROMPT.format(query="当前研究", section_id="sec_1",
            section_title="项目", original_content="旧章节全文" * 12000, feedback="修改要求：保留审阅意见",
            new_info="补充事实", claim_evidence="[]", claim_instructions="引用要求", min_section_chars=600,
            max_section_chars=1200)
        row = await self.append(scope, "LeadWriter", original)
        messages = await self.prepare(scope, "LeadWriter", "本轮原章节正文和证据")
        self.assertIn("修改要求：保留审阅意见", messages[1]["content"])
        self.assertNotIn("旧章节全文" * 5, messages[1]["content"])
        self.assertIn(row["source_id"], messages[1]["content"])
        self.assertIn("本轮原章节正文和证据", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
