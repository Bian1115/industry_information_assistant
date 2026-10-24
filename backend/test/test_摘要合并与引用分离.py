"""两阶段摘要机制测试，模型输出使用桩，不请求真实 API。"""

import asyncio
import json
import os
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from test_智能体独立历史 import (
    MemoryStore, CharCounter, HistoryScope, make_agent, summary_text,
    bind_agent_history, content_hash,
)
from service.agent_context import (
    summarize_history, prepare_agent_messages, parse_summary,
    summary_reference_quotes, ContextBudgetExceeded,
)
from service.summary_citations import (
    parse_draft, render_sections, summary_merge_view,
    citation_batches, citation_messages, parse_citation_result,
)
from service.context_policy import legacy_policy, ContextPolicy


def clause(claim, *sources, section="SUMMARY"):
    return {"section": section, "claim": claim, "source_ids": list(sources)}


def draft(*items):
    return render_sections(items, markers=True)


def citation(item, quotes):
    return json.dumps({"section": item["section"], "claim": item["claim"],
                       "references": [{"source_id": sid, "quote": quote} for sid, quote in quotes]}, ensure_ascii=False)


class CitationSeparationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.scope = HistoryScope(self.store, "run", "session", None, node="research")
        for p in [patch.dict(os.environ, {"AGENT_SUMMARY_STRATEGY": "split-citations-v4", "AGENT_SUMMARY_NLI_ENABLED": "0"}),
                  patch("service.agent_context.get_qwen_token_counter", return_value=CharCounter())]:
            p.start()
            self.addCleanup(p.stop)

    async def record(self, text):
        return await self.store.append(self.scope, "DeepScout", "llm_output", {"content": text}, str(uuid.uuid4()))

    async def seed_old(self):
        row = await self.record("原始研究显示旧结论。" + "旧全文不应重复发送" * 1000)
        summary = summary_text(row["source_id"], "旧结论。")
        summary = summary.replace('"quote": "旧结论。"', '"quote": "原始研究显示旧结论。"')
        await self.store.save_summary(self.scope, "DeepScout", expected_version=0, covered_seq=row["seq"], summary=summary,
                                      metadata={"summary_hash": content_hash(summary), "reference_quotes": summary_reference_quotes(summary)})
        return row, summary

    async def generate(self, responses, records, previous="", budget=5000):
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=responses)
        with patch("service.agent_context.input_budget", return_value=budget), bind_agent_history(self.scope):
            result = await summarize_history(agent, self.scope, records, CharCounter(),
                                             previous_summary=previous, previous_sources=summary_reference_quotes(previous))
        return result, agent

    async def test_merge_excludes_quote_ledger_and_final_claim_uses_actual_quote(self):
        old, previous = await self.seed_old()
        new = await self.record("新研究报告新结论。")
        items = [clause("旧结论。", old["source_id"]), clause("新结论。", new["source_id"])]
        final, agent = await self.generate([
            draft(*items), citation(items[0], [(old["source_id"], "原始研究显示旧结论。")]),
            citation(items[1], [(new["source_id"], "新研究报告新结论。")]),
        ], [new], previous)
        sent = [json.loads(c.args[1]) for c in agent._call_llm_impl.call_args_list]
        self.assertNotIn("CLAIMS", sent[0]["previous_summary"])
        self.assertNotIn("原始研究显示旧结论。", sent[0]["previous_summary"])
        self.assertIn(old["source_id"], sent[0]["previous_summary"])
        self.assertNotIn("旧全文不应重复发送", str(sent))
        self.assertIn("原始研究显示旧结论。", str(sent[1]))
        _, claims = parse_summary(final, {old["source_id"]: "原始研究显示旧结论。", new["source_id"]: "新研究报告新结论。"})
        self.assertEqual([c["claim"] for c in claims], ["旧结论。", "新结论。"])
        self.assertTrue(all(c["section"] == "SUMMARY" for c in claims))
        rows = (await self.store.load(self.scope, "DeepScout"))["messages"]
        stages = [r["payload"]["stage"] for r in rows if r["kind"] == "summary_stage"]
        self.assertEqual(stages, ["summary_merge", "citation_completion", "citation_completion"])

    async def test_all_chunks_merge_before_any_citation_and_no_raw_replay(self):
        records = [await self.record("新增甲。" + "甲背景" * 1400), await self.record("新增乙。" + "乙背景" * 1400)]
        a, b = [clause(text, r["source_id"]) for text, r in zip(("新增甲。", "新增乙。"), records)]
        result, agent = await self.generate([
            draft(a), draft(a, b), citation(a, [(records[0]["source_id"], "新增甲。")]),
            citation(b, [(records[1]["source_id"], "新增乙。")]),
        ], records, budget=2100)
        sent = [json.loads(c.args[1]) for c in agent._call_llm_impl.call_args_list]
        self.assertEqual(len(sent), 4)
        self.assertEqual(sent[1]["previous_summary"], draft(a))
        self.assertNotIn("甲背景", str(sent[1]))
        self.assertNotIn("CLAIMS", str(sent[:2]))
        self.assertIn("final_claim", sent[2])
        for c in agent._call_llm_impl.call_args_list:
            self.assertLessEqual(CharCounter().count_messages(c.kwargs["prepared_messages"]), 2100)
        self.assertEqual(len(parse_summary(result, {r["source_id"]: r["payload"]["content"] for r in records})[1]), 2)

    async def test_success_atomically_saves_and_no_new_history_means_no_new_summary(self):
        old, previous = await self.seed_old()
        row = await self.record("新结论。" + "新增完整原文" * 1000)
        item = clause("新结论。", row["source_id"])
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=[draft(item), citation(item, [(row["source_id"], "新结论。")])])
        with patch("service.agent_context.input_budget", side_effect=lambda out: 4000 if out == 3000 else 1400), bind_agent_history(self.scope):
            await prepare_agent_messages(agent, self.scope, "sys", "继续", "current", 6000)
            await prepare_agent_messages(agent, self.scope, "sys", "继续", "next", 6000)
        saved = await self.store.load(self.scope, "DeepScout")
        self.assertEqual(saved["summary_version"], 2)
        self.assertEqual(saved["covered_seq"], row["seq"])
        from service.summary_citations import PROMPT_VERSION
        self.assertEqual(saved["summary_metadata"]["prompt_version"], PROMPT_VERSION)
        self.assertEqual(saved["summary_metadata"]["validation"]["semantic_support"], "not_run")
        self.assertEqual(agent._call_llm_impl.await_count, 2)
        self.assertEqual(saved["messages"][0]["payload"]["content"], old["payload"]["content"])

    async def test_bad_citation_missing_reference_and_rewrite_do_not_advance_cursor(self):
        for failure in ("invented_quote", "missing", "rewrite", "wrong_section"):
            with self.subTest(failure=failure):
                self.store = MemoryStore()
                self.scope = HistoryScope(self.store, "run", "session", None)
                _, previous = await self.seed_old()
                row = await self.record("新结论。" + "背景" * 3000)
                item = clause("新结论。", row["source_id"])
                response = json.loads(citation(item, [(row["source_id"], "新结论。")]))
                if failure == "invented_quote":
                    response["references"][0]["quote"] = "不存在的原句"
                elif failure == "missing":
                    response["references"] = []
                elif failure == "rewrite":
                    response["claim"] = "改写后的分句"
                else:
                    response["section"] = "NEXT STEPS"
                agent = make_agent()
                agent._call_llm_impl = AsyncMock(side_effect=[draft(item), json.dumps(response, ensure_ascii=False)])
                with patch("service.agent_context.input_budget", side_effect=lambda out: 4000 if out == 3000 else 1400), bind_agent_history(self.scope), self.assertRaises(ContextBudgetExceeded):
                    await prepare_agent_messages(agent, self.scope, "sys", "继续", "current", 6000)
                saved = await self.store.load(self.scope, "DeepScout")
                self.assertEqual((saved["summary"], saved["summary_version"], saved["covered_seq"]), (previous, 1, 1))

    async def test_failure_after_one_completed_citation_does_not_save_partial_summary(self):
        _, previous = await self.seed_old()
        row = await self.record("甲结论。乙结论。" + "背景" * 3000)
        a, b = clause("甲结论。", row["source_id"]), clause("乙结论。", row["source_id"])
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=[draft(a, b), citation(a, [(row["source_id"], "甲结论。")]), "无效输出", "修复仍无效"])
        with patch("service.agent_context.input_budget", side_effect=lambda out: 4000 if out == 3000 else 1400), bind_agent_history(self.scope), self.assertRaises(ContextBudgetExceeded):
            await prepare_agent_messages(agent, self.scope, "sys", "继续", "current", 6000)
        saved = await self.store.load(self.scope, "DeepScout")
        self.assertEqual((saved["summary"], saved["summary_version"], saved["covered_seq"]), (previous, 1, 1))

    async def test_unknown_source_is_rejected_before_citation(self):
        row = await self.record("结论。")
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(return_value=draft(clause("结论。", "invented")))
        with bind_agent_history(self.scope), self.assertRaisesRegex(ValueError, "未知来源"):
            await summarize_history(agent, self.scope, [row], CharCounter())
        self.assertEqual(agent._call_llm_impl.await_count, 2)  # 一次局部修复失败后停止，不进入引用补齐。

    async def test_new_snapshot_hash_is_checked_before_provider_call(self):
        row = await self.record("原文")
        row["payload"]["content"] = "篡改"
        agent = make_agent()
        agent._call_llm_impl = AsyncMock()
        with bind_agent_history(self.scope), self.assertRaisesRegex(ValueError, "哈希"):
            await summarize_history(agent, self.scope, [row], CharCounter())
        agent._call_llm_impl.assert_not_awaited()

    def test_long_evidence_is_batched_without_dropping_tail(self):
        text = "长原文" * 1000 + "末尾关键证据。"
        item = clause("关键证据。", "s1")
        batches = list(citation_batches(item, {"s1": text}, CharCounter(), 400))
        self.assertGreater(len(batches), 1)
        self.assertIn("末尾关键证据。", str(batches[-1]))
        for batch in batches:
            self.assertLessEqual(CharCounter().count_messages(citation_messages(item, batch)), 400)
            for excerpt in batch:
                self.assertEqual(excerpt["text"], text[excerpt["start"]:excerpt["start"] + len(excerpt["text"])])

    def test_cannot_stitch_discontinuous_quotes_or_switch_sources(self):
        item = clause("综合结论。", "s1")
        excerpts = [{"source_id": "s1", "text": "第一句。"}, {"source_id": "s1", "text": "第二句。"}]
        for refs in [[("s1", "第一句。第二句。")], [("s2", "第一句。")]]:
            with self.assertRaises(ValueError):
                parse_citation_result(citation(item, refs), item, excerpts)

    def test_fixed_citation_input_over_budget_fails(self):
        with self.assertRaisesRegex(ValueError, "固定输入"):
            list(citation_batches(clause("要求" * 100, "s1"), {"s1": "原文"}, CharCounter(), 20))

    async def test_call_cap_stops_before_citation(self):
        row = await self.record("结论。")
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(return_value=draft(clause("结论。", row["source_id"])))
        with patch("service.summary_citations.MAX_CALLS", 1), bind_agent_history(self.scope), self.assertRaisesRegex(ValueError, "调用数量"):
            await summarize_history(agent, self.scope, [row], CharCounter())
        agent._call_llm_impl.assert_awaited_once()

    async def test_timeout_cancels_stage(self):
        row = await self.record("结论。")
        async def pending(*args, **kwargs):
            await asyncio.Event().wait()
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=pending)
        with patch("service.summary_citations.TIMEOUT_SECONDS", .01), bind_agent_history(self.scope), self.assertRaises(TimeoutError):
            await summarize_history(agent, self.scope, [row], CharCounter())
        self.assertEqual((await self.store.load(self.scope, "DeepScout"))["summary_version"], 0)

    async def test_unverified_suggestion_stays_explicit_and_has_no_invented_citation(self):
        row = await self.record("待规划")
        item = clause("建议：进一步核查。")
        result, agent = await self.generate([draft(item)], [row])
        _, claims = parse_summary(result, {})
        self.assertEqual(claims, [])
        self.assertIn(item["claim"], result)
        agent._call_llm_impl.assert_awaited_once()

    async def test_no_increment_does_not_call_model(self):
        _, previous = await self.seed_old()
        result, agent = await self.generate([], [], previous)
        self.assertEqual(result, previous)
        agent._call_llm_impl.assert_not_awaited()

    async def test_old_evidence_backread_happens_only_after_merge_and_only_when_needed(self):
        old = await self.record("旧结论。原文中还有新增限定条件。" + "背景" * 100)
        previous = summary_text(old["source_id"], "旧结论。")
        new = await self.record("请检查旧结论的限定条件。")
        item = clause("旧结论存在限定条件。", old["source_id"])
        self.store.read_source = AsyncMock(wraps=self.store.read_source)
        final, agent = await self.generate([
            draft(item), citation(item, []),
            citation(item, [(old["source_id"], "原文中还有新增限定条件。")]),
        ], [new], previous)
        self.store.read_source.assert_awaited_once_with(self.scope, "DeepScout", old["source_id"])
        sent = [json.loads(c.args[1]) for c in agent._call_llm_impl.call_args_list]
        self.assertNotIn("原文中还有新增限定条件。", str(sent[:2]))
        self.assertIn("原文中还有新增限定条件。", str(sent[2]))
        self.assertIn("原文中还有新增限定条件。", final)

    async def test_combined_final_claims_budget_is_enforced(self):
        records = [await self.record("甲结论。" + "甲" * 800), await self.record("乙结论。" + "乙" * 800)]
        a, b = [clause(text, r["source_id"]) for text, r in zip(("甲结论。", "乙结论。"), records)]
        responses = [draft(a, b), citation(a, [(records[0]["source_id"], records[0]["payload"]["content"])]),
                     citation(b, [(records[1]["source_id"], records[1]["payload"]["content"])])]
        with patch.dict(os.environ, {"AGENT_SUMMARY_OUTPUT_TOKENS": "400"}), self.assertRaisesRegex(ValueError, "合计超过"):
            await self.generate(responses, records)

    def test_final_section_mapping_handles_identical_claims_in_different_sections(self):
        a = clause("相同句。", "s1", section="SESSION INTENT")
        b = clause("相同句。", "s2", section="SUMMARY")
        final = render_sections([a, b]) + "\n\n## CLAIMS\n" + json.dumps([
            json.loads(citation(a, [("s1", "相同句。")])) , json.loads(citation(b, [("s2", "相同句。")]))
        ], ensure_ascii=False)
        converted = summary_merge_view(final, {"s1": "相同句。", "s2": "相同句。"})
        self.assertEqual(parse_draft(converted, {"s1", "s2"}), [a, b])

    def test_policy_new_default_and_old_snapshot_remain_distinct(self):
        policy = legacy_policy()
        self.assertEqual(policy.summary_strategy, "split-citations-v4")
        old = policy.model_copy(update={"summary_strategy": "incremental-v3"})
        restored = ContextPolicy.from_snapshot(old.snapshot())
        self.assertEqual(restored.summary_strategy, "incremental-v3")
        self.assertNotEqual(policy.fingerprint, old.fingerprint)


if __name__ == "__main__":
    unittest.main()
