"""实际配置与发送路径回归；仅本地桩，不执行质量评测或付费调用。"""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from service.context_policy import (
    ContextPolicy, InputBudget, PRESET_DIR, legacy_policy, read_policy,
    selected_policy, policy_for_scope, storage_context_key, UnsupportedContextPolicy,
)
from service.agent_history import HistoryScope, bind_agent_history
from service.agent_context import prepare_agent_messages, ContextBudgetExceeded
from test_智能体独立历史 import MemoryStore, CharCounter, make_agent, summary_text


def runnable_policy(**overrides):
    # 只用于单组件机制测试，不篡改方案9组文件的功能声明。
    data = ContextPolicy(group="开关机制测试", semantic_summary=False, semantic_verification=False).model_dump()
    data.update(overrides)
    return ContextPolicy.model_validate(data)


class NamespacedStore(MemoryStore):
    def key(self, scope, agent):
        from service.agent_history import research_history_scope_id
        return research_history_scope_id(scope), agent, storage_context_key(scope)


class ConfigurationTests(unittest.TestCase):
    def test_writer_60000_default_historical_presets_and_explicit_override(self):
        from config.llm_config import AgentsConfig
        from service.llm_client import default_max_output_tokens
        with patch.dict(os.environ, {"LLM_MAX_OUTPUT_TOKENS": "3000"}, clear=True):
            self.assertEqual(AgentsConfig().writer.max_tokens, 60000)
            self.assertEqual(default_max_output_tokens("LeadWriter"), 60000)
            self.assertEqual(default_max_output_tokens("ChiefArchitect"), 3000)
            self.assertEqual(legacy_policy().output_limits["LeadWriter"], 60000)
            for path in PRESET_DIR.glob("*.json"):
                source = json.loads(path.read_text())
                expected = 60000 if source.get("baseline") == "capture-current-runtime" else 6000
                self.assertEqual(read_policy(path).output_limits["LeadWriter"], expected, str(path))
        with patch.dict(os.environ, {"LLM_MAX_OUTPUT_TOKENS_LEADWRITER": "4000"}):
            self.assertEqual(default_max_output_tokens("LeadWriter"), 4000)
        with patch.dict(os.environ, {"LLM_MAX_OUTPUT_TOKENS_LEADWRITER": "60000"}):
            self.assertEqual(default_max_output_tokens("LeadWriter"), 60000)

    def test_nine_groups_and_exact_matrix(self):
        paths = sorted(PRESET_DIR.glob("*.json"))
        self.assertEqual(len(paths), 9)
        expected = [
            (False, False, False, False, False, False),
            (True, True, True, True, True, True),
            (False, False, False, True, True, True),
            (True, True, True, False, True, False),
            (True, True, True, True, False, True),
            (True, True, True, True, True, False),
            (True, False, True, True, True, True),
            (True, True, False, True, True, True),
        ]
        keys = ["artifact_offload", "early_offload", "recent_offload", "semantic_summary", "pinned_contract", "semantic_verification"]
        for path, row in zip(paths[1:], expected):
            policy = read_policy(path)
            self.assertEqual(tuple(getattr(policy, k) for k in keys), row)
            self.assertEqual(policy.early_tool_rounds, 3)
            self.assertEqual(policy.tool_content_tokens, 800)
            self.assertEqual(policy.thresholds(20000), (17000, 19000))

    def test_strict_config_rejects_typos_strings_and_bad_thresholds(self):
        for invalid in [dict(artifact_offlod=False), dict(artifact_offload="false"),
                        dict(early_tool_rounds=0), dict(soft_ratio=0.97),
                        dict(semantic_summary=False, semantic_verification=True),
                        dict(stage_budgets={})]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ContextPolicy.model_validate({"group": "invalid", **invalid})

    def test_complete_presets_require_explicit_verification_dependencies(self):
        for name in ["03-完整方案.json", "07-去新增校验.json"]:
            read_policy(PRESET_DIR / name).require_supported()
        with self.assertRaises(UnsupportedContextPolicy):
            ContextPolicy(group="缺NLI配置", summary_strategy="chunk-merge-v1").require_supported()

    def test_snapshot_is_immutable_to_environment_changes_and_detects_tampering(self):
        policy = runnable_policy()
        scope = HistoryScope(None, "r", "s", None, context_policy=policy.snapshot())
        with patch.dict(os.environ, {"AGENT_CONTEXT_WINDOW": "9000", "AGENT_HISTORY_SUMMARY_ENABLED": "1"}):
            loaded = policy_for_scope(scope)
        self.assertEqual(loaded.fingerprint, policy.fingerprint)
        self.assertFalse(loaded.semantic_summary)
        snapshot = policy.snapshot()
        snapshot["config"]["artifact_offload"] = False
        with self.assertRaisesRegex(ValueError, "哈希"):
            ContextPolicy.from_snapshot(snapshot)

    def test_file_selection_ignores_conflicting_legacy_environment(self):
        path = PRESET_DIR / "02-新框架无压缩基线.json"
        with patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": str(path), "AGENT_HISTORY_SUMMARY_ENABLED": "1"}):
            self.assertFalse(selected_policy().semantic_summary)
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory) / "配置快照.json"
            frozen.write_text(json.dumps(read_policy(path).snapshot()))
            self.assertEqual(read_policy(frozen).fingerprint, read_policy(path).fingerprint)

    def test_baseline_captures_effective_existing_settings(self):
        with patch.dict(os.environ, {"AGENT_CONTEXT_WINDOW": "50000", "AGENT_CONTEXT_MAX_INPUT": "40000", "AGENT_HISTORY_SUMMARY_ENABLED": "0", "LLM_MAX_OUTPUT_TOKENS_LEADWRITER": "6000"}):
            policy = legacy_policy()
        self.assertFalse(policy.semantic_summary)
        self.assertFalse(policy.semantic_verification)
        self.assertEqual(policy.stage_budgets["DeepScout"].context_window, 50000)

    def test_runtime_default_keeps_references_and_global_summary_without_input_summary(self):
        # 已删除的环境开关即使残留为1，也不能恢复单条模型摘要。
        with patch.dict(os.environ, {"AGENT_INPUT_HISTORY_SUMMARY_ENABLED": "1"}, clear=True):
            policy = legacy_policy()
        self.assertTrue(policy.semantic_summary)
        self.assertTrue(policy.input_history.reference_materials)
        self.assertEqual(policy.input_history.model_dump(), {
            "version": "input-view-v2", "reference_materials": True, "min_reference_tokens": 128,
        })


class SwitchRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = NamespacedStore()
        self.scope = HistoryScope(self.store, "run", "session", None, contract={"篇幅": "3000字"})
        for name in ["service.agent_context.get_qwen_token_counter", "service.qwen_token_counter.get_qwen_token_counter"]:
            patcher = patch(name, return_value=CharCounter())
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("record_llm_call", "flush_pending"):
            patcher = patch("service.deep_research_v2.agents.base." + name)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def seed_tools(self, scope, count=4, chars=5000):
        sources = []
        for i in range(count):
            call = str(uuid.uuid4())
            await self.store.append(scope, "DeepScout", "tool_input", {"query": str(i)}, call)
            row = await self.store.append(scope, "DeepScout", "tool_output", {"body": str(i) * chars}, call)
            sources.append(row["source_id"])
        return sources

    async def prepare(self, policy, *, budget=6000, tools=4):
        scope = replace(self.scope, context_policy=policy.snapshot())
        ids = await self.seed_tools(scope, tools)
        with bind_agent_history(scope), patch("service.agent_context.scoped_input_budget", return_value=budget):
            messages = await prepare_agent_messages(make_agent(), scope, "system", "当前问题", "current", 6000)
        return scope, ids, messages

    async def test_offload_master_disables_both_stages_and_rejects_overflow(self):
        p = runnable_policy(artifact_offload=False, early_offload=True, recent_offload=True)
        with self.assertRaises(ContextBudgetExceeded):
            await self.prepare(p, budget=3000)
        rows = (await self.store.load(replace(self.scope, context_policy=p.snapshot()), "DeepScout"))["messages"]
        view = next(x["payload"] for x in rows if x["kind"] == "context_view")
        self.assertEqual(view["stages"], [])
        self.assertEqual(view["shortened_source_ids"], [])

    async def test_all_completed_rounds_and_recent_offload_have_separate_switches(self):
        p = runnable_policy(early_tool_rounds=2, soft_ratio=0.5, recent_offload=False)
        scope, ids, messages = await self.prepare(p)
        saved = await self.store.load(scope, "DeepScout")
        self.assertEqual(set(saved["offloaded_source_ids"]), set(ids))
        self.assertNotIn("3" * 5000, str(messages))
        q = runnable_policy(early_offload=False, recent_offload=True)
        scope, ids, _ = await self.prepare(q, budget=4500)
        self.assertIn(ids[-1], (await self.store.load(scope, "DeepScout"))["offloaded_source_ids"])

    async def test_both_subswitches_disabled_never_reuse_saved_offload(self):
        p = runnable_policy(early_offload=False, recent_offload=False)
        scope = replace(self.scope, context_policy=p.snapshot())
        ids = await self.seed_tools(scope, count=1)
        await self.store.mark_offloaded(scope, "DeepScout", ids)
        with bind_agent_history(scope):
            messages = await prepare_agent_messages(make_agent(), scope, "s", "q", "current", 6000)
        self.assertIn("0" * 5000, str(messages))

    async def test_summary_off_ignores_saved_summary_and_covered_cursor(self):
        p = runnable_policy(artifact_offload=False)
        scope = replace(self.scope, context_policy=p.snapshot())
        call = str(uuid.uuid4())
        row = await self.store.append(scope, "DeepScout", "llm_output", {"content": "必须保留的旧原文"}, call)
        await self.store.save_summary(scope, "DeepScout", expected_version=0, covered_seq=row["seq"], summary="不得读取的旧摘要", metadata={})
        agent = make_agent()
        agent._call_llm_impl = AsyncMock()
        with bind_agent_history(scope):
            messages = await prepare_agent_messages(agent, scope, "s", "q", "current", 6000)
        self.assertIn("必须保留的旧原文", str(messages))
        self.assertNotIn("不得读取的旧摘要", str(messages))
        agent._call_llm_impl.assert_not_awaited()

    async def test_contract_flag_only_controls_added_injection(self):
        for enabled in (True, False):
            scope = replace(self.scope, context_policy=runnable_policy(pinned_contract=enabled).snapshot())
            with bind_agent_history(scope):
                messages = await prepare_agent_messages(make_agent("LeadWriter"), scope, "s", "原有问题", "current", 3000)
            self.assertEqual("3000字" in str(messages), enabled)
            self.assertEqual(messages[-1]["content"], "原有问题")

    async def test_config_isolates_history_even_for_same_run_agent_subtask(self):
        first = replace(self.scope, context_policy=runnable_policy().snapshot())
        second = replace(self.scope, context_policy=runnable_policy(artifact_offload=False).snapshot())
        row = await self.store.append(first, "DeepScout", "llm_output", {"content": "别组结果"}, str(uuid.uuid4()))
        self.assertNotEqual(storage_context_key(first), storage_context_key(second))
        self.assertEqual((await self.store.load(second, "DeepScout"))["messages"], [])
        with self.assertRaises(PermissionError):
            await self.store.read_source(second, "DeepScout", row["source_id"])

    async def test_no_compression_rejects_before_model_and_logs_failure(self):
        p = read_policy(PRESET_DIR / "02-新框架无压缩基线.json")
        scope = replace(self.scope, context_policy=p.snapshot())
        agent = make_agent()
        with bind_agent_history(scope), self.assertRaises(ContextBudgetExceeded):
            await agent.call_llm("s", "当前不可删除输入" * 20000)
        agent.client.chat.completions.create.assert_not_awaited()
        rows = (await self.store.load(scope, agent.name))["messages"]
        self.assertEqual(rows[-1]["kind"], "llm_error")

    async def test_frozen_output_cap_and_read_limit_apply_to_actual_requests(self):
        data = runnable_policy(history_read_max_calls=0).model_dump()
        data["output_limits"]["DeepScout"] = 777
        p = ContextPolicy.model_validate(data)
        scope = replace(self.scope, context_policy=p.snapshot())
        agent = make_agent(response='{"_history_read":{"source_id":"anything"}}')
        with bind_agent_history(scope), patch.dict(os.environ, {"LLM_MAX_OUTPUT_TOKENS_DEEPSCOUT": "9999"}), self.assertRaisesRegex(ValueError, "回读次数"):
            await agent.call_llm("s", "q")
        self.assertEqual(agent.client.chat.completions.create.call_args.kwargs["max_tokens"], 777)
        self.assertEqual(agent.client.chat.completions.create.await_count, 1)

    async def test_writer_actual_request_uses_60000_and_historical_snapshot_remains_frozen(self):
        from config.llm_config import AgentsConfig
        with patch.dict(os.environ, {"LLM_MAX_OUTPUT_TOKENS": "3000"}, clear=True):
            configured_tokens = AgentsConfig().writer.max_tokens
            policy = legacy_policy()
            scope = replace(self.scope, context_policy=policy.snapshot())
            agent = make_agent("LeadWriter")
            with bind_agent_history(scope):
                await agent.call_llm("系统", "撰写章节和引用声明", max_tokens=configured_tokens)
            request = agent.client.chat.completions.create.call_args.kwargs
            self.assertEqual(request["max_tokens"], 60000)
            self.assertNotIn("enable_thinking", request)
            self.assertNotIn("extra_body", request)

            old_limits = {**policy.output_limits, "LeadWriter": 3000}
            historical = policy.model_copy(update={"output_limits": old_limits}).snapshot()
            scope = replace(self.scope, context_policy=historical)
            with bind_agent_history(scope):
                await agent.call_llm("系统", "继续历史运行", max_tokens=configured_tokens)
            self.assertEqual(agent.client.chat.completions.create.call_args.kwargs["max_tokens"], 3000)
            self.assertEqual(scope.context_policy, historical)

    async def test_unsupported_policy_cannot_reach_provider(self):
        unsupported = read_policy(PRESET_DIR / "03-完整方案.json").model_copy(update={"nli": None})
        scope = replace(self.scope, context_policy=unsupported.snapshot())
        agent = make_agent()
        with bind_agent_history(scope), self.assertRaises(UnsupportedContextPolicy):
            await agent.call_llm("s", "q")
        agent.client.chat.completions.create.assert_not_awaited()

    async def test_new_run_freezes_file_and_resume_does_not_reselect_environment(self):
        from service.deep_research_v2.service import DeepResearchV2Service
        from service.deep_research_v2.state import GRAPH_VERSION
        service = object.__new__(DeepResearchV2Service)
        service.config = SimpleNamespace(research=SimpleNamespace(max_iterations=2))
        service.resolve_allowed_domains = lambda domains: ["who.int"]
        kwargs = dict(query="任务", session_id="session", research_context=None, allowed_domains=None)
        path = str(PRESET_DIR / "02-新框架无压缩基线.json")
        with patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": path}):
            state = await service._graph_input(None, {}, resume=False, **kwargs)
        captured = state["research"]["context_policy"]
        self.assertEqual(captured, read_policy(path).snapshot())
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(
            values=state, next=("scout",))))
        with patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": str(PRESET_DIR / "03-完整方案.json")}):
            self.assertIsNone(await service._graph_input(graph, {}, resume=True, **kwargs))
        self.assertEqual(state["research"]["context_policy"], captured)


import test_智能体独立历史 as history_tests


@unittest.skipUnless(os.getenv("TEST_AGENT_HISTORY_POSTGRES") == "1", "显式启用隔离PostgreSQL集成测试")
class PostgresPolicyIsolationTests(unittest.IsolatedAsyncioTestCase):
    # 复用已有随机schema夹具；不修改用户研究表，不继承重复的测试方法。
    setUpClass = classmethod(history_tests.PostgresHistoryTests.setUpClass.__func__)
    tearDownClass = classmethod(history_tests.PostgresHistoryTests.tearDownClass.__func__)
    asyncSetUp = history_tests.PostgresHistoryTests.asyncSetUp

    async def test_real_postgres_partitions_messages_summary_and_read_permissions(self):
        first = replace(self.scope, context_policy=runnable_policy().snapshot())
        second = replace(self.scope, context_policy=runnable_policy(artifact_offload=False).snapshot())
        row = await self.store.append(first, "DeepScout", "llm_output", {"content": "第一组原文"}, str(uuid.uuid4()))
        await self.store.save_summary(first, "DeepScout", expected_version=0, covered_seq=row["seq"], summary="第一组摘要", metadata={})
        await self.store.mark_offloaded(first, "DeepScout", [row["source_id"]])
        isolated = await self.store.load(second, "DeepScout")
        self.assertEqual(isolated["summary"], "")
        self.assertEqual(isolated["messages"], [])
        self.assertEqual(isolated["offloaded_source_ids"], [])
        with self.assertRaises(PermissionError):
            await self.store.read_source(second, "DeepScout", row["source_id"])


if __name__ == "__main__":
    unittest.main()
