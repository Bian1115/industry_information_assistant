"""正文先提交、审计独立计时及异常类型的确定性回归，不调用真实模型。"""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import unittest
import uuid
from unittest.mock import AsyncMock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "test")]
from service.agent_context import ContextBudgetExceeded, parse_summary, prepare_agent_messages
from service.agent_history import AgentHistoryError, HistoryScope, bind_agent_history, content_hash
from service.context_policy import ContextPolicy, SummaryLifecycleSettings, legacy_policy
from service.summary_lifecycle import pack_prose_summary
from test_智能体独立历史 import MemoryStore, CharCounter, make_agent, summary_text

PROSE = "## SESSION INTENT\n继续研究。\n\n## SUMMARY\n研究纳入800人。\n\n尚需检索局限。\n\n## ARTIFACTS\nNone"


class Store(MemoryStore):
    async def save_summary(self, *args, **kwargs):
        kwargs["metadata"] = deepcopy(kwargs["metadata"])
        return await super().save_summary(*args, **kwargs)

    async def update_summary_metadata(self, scope, agent, *, expected_version, expected_summary_hash, metadata):
        target = self.data[self.key(scope, agent)]
        if target["summary_version"] != expected_version or content_hash(target["summary"]) != expected_summary_hash:
            return False
        target["summary_metadata"] = deepcopy(metadata)
        return True


class SettingsTests(unittest.TestCase):
    def test_atomic_switch_requires_claimify_and_old_nested_snapshot_keeps_hash(self):
        old = ContextPolicy(group='旧快照', semantic_verification=False,
            summary_lifecycle=SummaryLifecycleSettings(claimify_enabled=True))
        snapshot = old.snapshot()
        self.assertNotIn('atomic_citations', old.config_payload()['summary_lifecycle'])
        self.assertEqual(ContextPolicy.from_snapshot(snapshot).snapshot(), snapshot)
        with self.assertRaises(ValueError):
            ContextPolicy(group='错误开关', semantic_verification=False,
                summary_lifecycle=SummaryLifecycleSettings(atomic_citations=True))
        with patch.dict(os.environ, {'AGENT_SUMMARY_CLAIMIFY_ENABLED': '1',
                                    'AGENT_SUMMARY_ATOMIC_CITATIONS_ENABLED': '1'}):
            self.assertTrue(legacy_policy().summary_lifecycle.atomic_citations)

    def test_new_runtime_defaults_off_and_enabled_switch_is_frozen(self):
        with patch.dict(os.environ, {"AGENT_SUMMARY_STRATEGY": "claimify-v1", "AGENT_SUMMARY_CLAIMIFY_ENABLED": "0",
                                    "AGENT_SUMMARY_NLI_ENABLED": "1", "AGENT_SUMMARY_CHECKS_ENABLED": "1"}):
            policy = legacy_policy()
        self.assertFalse(policy.summary_lifecycle.claimify_enabled)
        self.assertFalse(policy.semantic_verification)
        self.assertIsNone(policy.nli)
        with patch.dict(os.environ, {"AGENT_SUMMARY_CLAIMIFY_ENABLED": "1"}):
            restored = ContextPolicy.from_snapshot(policy.snapshot())
        self.assertEqual(restored.snapshot(), policy.snapshot())

    def test_historical_evaluation_snapshot_hash_is_unchanged(self):
        path = BACKEND / "eval/data/通用研究中文5题-20260930/运行配置.json"
        if not path.exists():
            self.skipTest("本地首次评测档案不存在")
        snapshot = json.loads(path.read_text())["context_policy"]
        policy = ContextPolicy.from_snapshot(snapshot)
        self.assertIsNone(policy.summary_lifecycle)
        self.assertEqual(policy.snapshot(), snapshot)

    def test_unverified_format_is_explicit_and_cannot_claim_passed(self):
        summary = pack_prose_summary(PROSE)
        self.assertEqual(parse_summary(summary, {})[1], [])
        with self.assertRaisesRegex(ValueError, "状态标记"):
            parse_summary(summary.replace('"unverified"', '"passed"'), {})


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_atomic_audit_is_selected_and_keeps_commit(self):
        await self.setup_case(enabled=True)
        from dataclasses import replace
        policy = ContextPolicy.from_snapshot(self.scope.context_policy)
        updated = policy.model_copy(update={'summary_lifecycle': policy.summary_lifecycle.model_copy(update={'atomic_citations': True})})
        self.scope = replace(self.scope, context_policy=updated.snapshot())
        claim = {'claim': '研究纳入800人。', 'origin': {'parent_claim_id': 'a1'}, 'references': []}
        atomic = AsyncMock(return_value={'status': 'passed', 'mapped_claims': [claim], 'stage_totals': []})
        old = AsyncMock(side_effect=AssertionError('原子开关开启后不走旧抽取顺序'))
        with patch('service.summary_atomic_citations.audit_prose_atomic', atomic), \
             patch('service.summary_citations.summarize_with_citations', old):
            await self.prepare()
        atomic.assert_awaited_once(); old.assert_not_awaited()
        saved = await self.saved()
        self.assertEqual(saved['covered_seq'], self.row['seq'])
        self.assertEqual(saved['summary'], pack_prose_summary(PROSE))
        self.assertEqual(saved['summary_metadata']['validation']['nli']['citation_claims'], [claim])

    async def setup_case(self, enabled=False, generation_timeout=120.0, claimify_timeout=120.0):
        self.store = Store()
        policy = ContextPolicy(
            group="先保存摘要机制测试", summary_strategy="claimify-v1", semantic_verification=False,
            artifact_offload=False, input_history=None,
            summary_lifecycle=SummaryLifecycleSettings(claimify_enabled=enabled,
                generation_timeout_seconds=generation_timeout, claimify_timeout_seconds=claimify_timeout),
        )
        self.scope = HistoryScope(self.store, "test", "session", None, node="research", context_policy=policy.snapshot())
        self.agent = make_agent()
        self.agent._call_llm_impl = AsyncMock(return_value=PROSE)
        self.row = await self.store.append(self.scope, "DeepScout", "llm_output",
                                           {"content": "研究纳入800人。" * 1500}, str(uuid.uuid4()))
        for name, value in [("service.agent_context.get_qwen_token_counter", CharCounter()),
                            ("service.agent_context.scoped_input_budget", 2000)]:
            patcher = patch(name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def prepare(self):
        with bind_agent_history(self.scope):
            return await prepare_agent_messages(self.agent, self.scope, "研究系统", "当前要求", str(uuid.uuid4()), 6000)

    async def saved(self):
        return await self.store.load(self.scope, "DeepScout")

    async def test_off_commits_cursor_preserves_raw_and_skips_all_claimify(self):
        await self.setup_case()
        auditor = AsyncMock(side_effect=AssertionError("关闭时不得执行Claimify"))
        with patch("service.summary_citations.summarize_with_citations", auditor):
            messages = await self.prepare()
            calls = self.agent._call_llm_impl.await_count
            await self.prepare()
        auditor.assert_not_awaited()
        self.assertEqual(self.agent._call_llm_impl.await_count, calls)
        saved = await self.saved()
        self.assertEqual(saved["summary_version"], 1)
        self.assertEqual(saved["covered_seq"], self.row["seq"])
        self.assertEqual(saved["summary_metadata"]["validation"]["claimify"], "disabled")
        self.assertEqual(saved["summary_metadata"]["validation"]["semantic_support"], "not_run")
        self.assertIsNone(saved["summary_metadata"]["validation"]["nli"]["claimify_duration_seconds"])
        skipped = next(r["payload"] for r in saved["messages"] if r["kind"] == "summary_claimify_skipped")
        self.assertEqual(skipped["status"], "disabled")
        self.assertIsNone(skipped["duration_seconds"])
        self.assertEqual(await self.store.read_source(self.scope, "DeepScout", self.row["source_id"]), self.row)
        self.assertIn(self.row["source_id"], str(messages))
        self.assertNotIn("validated_history_summary", str(messages))

    async def test_audit_timeout_keeps_already_committed_body_and_cursor(self):
        await self.setup_case(enabled=True, claimify_timeout=0.02)
        async def audit(*args, **kwargs):
            saved = await self.saved()
            self.assertEqual(saved["covered_seq"], self.row["seq"])
            self.assertIn("研究纳入800人", saved["summary"])
            await asyncio.Event().wait()
        with patch("service.summary_citations.summarize_with_citations", AsyncMock(side_effect=audit)):
            await self.prepare()
        saved = await self.saved()
        self.assertEqual(saved["summary_version"], 1)
        self.assertEqual(saved["covered_seq"], self.row["seq"])
        validation = saved["summary_metadata"]["validation"]
        self.assertEqual(validation["claimify"], "error")
        self.assertEqual(validation["nli"]["error_type"], "TimeoutError")
        self.assertEqual(validation["semantic_support"], "error")
        error = next(r["payload"] for r in saved["messages"] if r["kind"] == "summary_claimify_error")
        self.assertEqual(error["error_type"], "TimeoutError")
        self.assertEqual(error["message"], "")
        metrics = validation["nli"]
        phases = {r["payload"]["phase"]: r["payload"] for r in saved["messages"] if r["kind"] == "summary_phase_end"}
        self.assertEqual(metrics["generation_duration_seconds"], phases["generation"]["duration_seconds"])
        self.assertEqual(metrics["claimify_duration_seconds"], phases["claimify"]["duration_seconds"])
        self.assertGreaterEqual(metrics["claimify_duration_seconds"], 0.02)
        self.assertEqual(phases["generation"]["summary_attempt_id"], phases["claimify"]["summary_attempt_id"])

    async def test_enabled_real_claimify_citations_run_only_after_commit(self):
        from test_Claimify摘要论断抽取 import answer, responses
        await self.setup_case(enabled=True)
        claim = "研究纳入800人。"
        outputs = iter([PROSE, answer("no_verifiable_claims"), *responses(claim),
                        answer("no_verifiable_claims"), json.dumps({
                            "section": "SUMMARY", "claim": claim, "references": [{
                                "source_id": self.row["source_id"], "quote": claim, "start": 0, "end": len(claim),
                            }],
                        }, ensure_ascii=False)])
        call_number = 0
        async def llm(*args, **kwargs):
            nonlocal call_number
            if call_number:
                self.assertEqual((await self.saved())["covered_seq"], self.row["seq"])
            call_number += 1
            return next(outputs)
        self.agent._call_llm_impl = AsyncMock(side_effect=llm)
        with patch("service.agent_context.scoped_input_budget", side_effect=lambda *a, **kw: 30000 if kw.get("summary") else 2000):
            await self.prepare()
        saved = await self.saved()
        self.assertEqual(saved["summary"], pack_prose_summary(PROSE))
        validation = saved["summary_metadata"]["validation"]
        self.assertEqual(validation["claimify"], "completed")
        self.assertEqual(validation["semantic_support"], "not_run")
        self.assertEqual(validation["nli"]["citation_claims"][0]["claim"], claim)
        self.assertEqual(call_number, 7)
        stages = [r["payload"] for r in saved["messages"] if r["kind"] == "summary_stage"]
        self.assertEqual(sum(s["phase"] == "generation" for s in stages), 1)
        self.assertTrue(all(s["phase"] == "claimify" for s in stages[1:]))
        self.assertEqual(len({s["summary_attempt_id"] for s in stages}), 1)


    async def test_generation_timeout_does_not_advance_cursor_and_records_type(self):
        await self.setup_case(generation_timeout=0.01)
        async def never(*args, **kwargs):
            await asyncio.Event().wait()
        with patch("service.summary_hierarchy.hierarchical_summary", AsyncMock(side_effect=never)):
            with self.assertRaises(ContextBudgetExceeded):
                await self.prepare()
        saved = await self.saved()
        self.assertEqual(saved["covered_seq"], 0)
        self.assertEqual(saved["summary_version"], 0)
        error = next(r["payload"] for r in saved["messages"] if r["kind"] == "summary_error")
        self.assertEqual(error["error_type"], "TimeoutError")
        self.assertEqual(error["phase"], "generation")

    async def test_generation_and_audit_receive_independent_deadlines(self):
        await self.setup_case(enabled=True, generation_timeout=0.075, claimify_timeout=0.075)
        async def generate(*args, **kwargs):
            await asyncio.sleep(0.04)
            from service.claimify import prose_messages
            return await args[3](prose_messages({'new_records': [{'source_id': self.row['source_id'],
                                                                'text': '研究纳入800人。'}]}), 'summary_merge')
        async def audit(*args, **kwargs):
            await asyncio.sleep(0.04)
            kwargs["verification_out"].update(status="not_run")
            return summary_text(self.row["source_id"], "研究纳入800人。")
        with patch("service.summary_hierarchy.hierarchical_summary", AsyncMock(side_effect=generate)), \
             patch("service.summary_citations.summarize_with_citations", AsyncMock(side_effect=audit)):
            await self.prepare()
        validation = (await self.saved())["summary_metadata"]["validation"]
        self.assertEqual(validation["claimify"], "completed")
        nli = validation["nli"]
        self.assertGreater(nli["generation_duration_seconds"] + nli["claimify_duration_seconds"], 0.075)

    async def test_audit_cancellation_preserves_commit_and_propagates(self):
        await self.setup_case(enabled=True)
        with patch("service.summary_citations.summarize_with_citations", AsyncMock(side_effect=asyncio.CancelledError())):
            with self.assertRaises(asyncio.CancelledError):
                await self.prepare()
        saved = await self.saved()
        self.assertEqual(saved["covered_seq"], self.row["seq"])
        self.assertEqual(saved["summary_metadata"]["validation"]["claimify"], "cancelled")

    async def test_persistence_failure_propagates_and_audit_does_not_run(self):
        await self.setup_case(enabled=True)
        self.store.save_summary = AsyncMock(side_effect=AgentHistoryError("提交失败"))
        auditor = AsyncMock()
        with patch("service.summary_citations.summarize_with_citations", auditor):
            with self.assertRaises(AgentHistoryError):
                await self.prepare()
        auditor.assert_not_awaited()
        self.assertEqual((await self.saved())["covered_seq"], 0)

    async def test_audit_annotation_does_not_overwrite_newer_summary(self):
        await self.setup_case(enabled=True)
        async def audit(*args, **kwargs):
            await self.store.save_summary(self.scope, "DeepScout", expected_version=1,
                covered_seq=self.row["seq"] + 1, summary=pack_prose_summary(PROSE.replace("800", "900")),
                metadata={"newer": True})
            raise TimeoutError()
        with patch("service.summary_citations.summarize_with_citations", AsyncMock(side_effect=audit)):
            await self.prepare()
        saved = await self.saved()
        self.assertEqual(saved["summary_version"], 2)
        self.assertEqual(saved["summary_metadata"], {"newer": True})
        self.assertIn("900", saved["summary"])

    async def test_next_summary_carries_raw_provenance_with_claimify_off(self):
        await self.setup_case()
        await self.prepare()
        new = await self.store.append(self.scope, self.agent.name, "llm_output",
                                      {"content": "新增结果。" * 2200}, "new-source")
        await self.prepare()
        saved = await self.saved()
        self.assertEqual(saved["summary_version"], 2)
        self.assertEqual(saved["covered_seq"], new["seq"])
        self.assertEqual(set(saved["summary_metadata"]["source_ids"]), {self.row["source_id"], new["source_id"]})
        self.assertEqual(saved["summary_metadata"]["source_hashes"][self.row["source_id"]], self.row["content_hash"])
        self.assertEqual(await self.store.read_source(self.scope, self.agent.name, self.row["source_id"]), self.row)


if __name__ == "__main__":
    unittest.main()
