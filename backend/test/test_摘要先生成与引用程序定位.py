"""v2摘要顺序、可选遗漏检查与精确定位；模型用桩，不运行线上任务。"""
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from test_智能体独立历史 import MemoryStore, CharCounter, HistoryScope, make_agent, bind_agent_history
from test_摘要合并与引用分离 import clause, draft, citation
from service.context_policy import ContextPolicy, NLISettings, VerificationSettings, PRESET_DIR
from service.summary_citations import summarize_with_citations, parse_citation_result
from service.verification_checks import resolve_quote_offsets, extract_ledger


class PositionTests(unittest.TestCase):
    def test_unique_quote_uses_program_position_even_with_wrong_model_offset(self):
        ref = {"quote": 'count": 10', "start": 26, "end": 35}
        text = '{"arguments": {"count": 10, "query": "search"}}'
        self.assertEqual(resolve_quote_offsets(ref, [{"start": 100, "text": text}]),
                         {"start": 116, "end": 126})

    def test_duplicates_need_valid_hint_and_overlaps_are_not_unique(self):
        for ref in [{"quote": "结论。"}, {"quote": "结论。", "start": 1}]:
            with self.assertRaisesRegex(ValueError, "不唯一"):
                resolve_quote_offsets(ref, [{"text": "结论。结论。"}])
        self.assertEqual(resolve_quote_offsets({"quote": "结论。", "start": 3},
                                              [{"text": "结论。结论。"}]), {"start": 3, "end": 6})
        with self.assertRaisesRegex(ValueError, "不唯一"):
            resolve_quote_offsets({"quote": "aa"}, [{"text": "aaa"}])

    def test_overlapping_excerpt_copies_do_not_create_false_ambiguity(self):
        self.assertEqual(resolve_quote_offsets({"quote": "原句"}, [
            {"start": 10, "text": "甲原句"}, {"start": 11, "text": "原句"}
        ]), {"start": 11, "end": 13})

    def test_missing_quote_and_cross_source_match_are_rejected(self):
        item = clause("结论。", "s1")
        excerpts = [{"source_id": "s1", "text": "材料甲。"}, {"source_id": "s2", "text": "材料乙。"}]
        for sid, quote in [("s1", "材料乙。"), ("s2", "材料乙。"), ("s1", "编造原句。")]:
            with self.assertRaises(ValueError):
                parse_citation_result(citation(item, [(sid, quote)]), item, excerpts, program_positions=True)

    def test_v1_keeps_strict_offsets(self):
        item = clause("结论。", "s1")
        result = json.loads(citation(item, [("s1", "原句。")]))
        result["references"][0]["start"] = 9
        text = json.dumps(result, ensure_ascii=False)
        excerpts = [{"source_id": "s1", "text": "甲原句。"}]
        with self.assertRaises(ValueError):
            parse_citation_result(text, item, excerpts)
        refs = parse_citation_result(text, item, excerpts, program_positions=True)
        self.assertEqual((refs[0]["start"], refs[0]["end"]), (1, 4))

    def test_presets_disable_coverage_but_keep_field_and_nli_checks(self):
        for path in PRESET_DIR.glob("*.json"):
            data = json.loads(path.read_text())
            if "verification" in data:
                settings = data["verification"]
                self.assertEqual(settings["rules_version"], "verification-v2")
                self.assertFalse(settings["coverage"])
                self.assertTrue(settings["key_fields"])
                self.assertTrue(settings["nli_support"])

    def test_old_frozen_rules_keep_order_configuration_and_hash(self):
        settings = VerificationSettings(rules_version="verification-v1", coverage=True)
        policy = ContextPolicy(group="旧冻结规则", nli=NLISettings(), verification=settings)
        snapshot = policy.snapshot()
        restored = ContextPolicy.from_snapshot(snapshot)
        self.assertEqual(restored.snapshot(), snapshot)
        self.assertTrue(restored.verification.coverage)
        self.assertEqual(restored.verification.rules_version, "verification-v1")


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def setup_case(self, **settings):
        self.store = MemoryStore()
        self.policy = ContextPolicy(group="摘要顺序验证", summary_strategy="chunk-merge-v1",
                                    nli=NLISettings(), verification=VerificationSettings(**settings))
        self.scope = HistoryScope(self.store, "run", "session", None, context_policy=self.policy.snapshot())
        self.quote = "研究A纳入800人。"
        self.row = await self.store.append(self.scope, "DeepScout", "tool_output",
            {"tool_name": "v2_web_fulltext_read", "value": {"text": self.quote}}, "call")
        self.item = clause(self.quote, self.row["source_id"])
        self.agent = make_agent()

    async def stages(self):
        rows = (await self.store.load(self.scope, "DeepScout"))["messages"]
        return [r["payload"]["stage"] for r in rows if r["kind"] == "summary_stage"]

    async def test_default_generates_then_cites_no_ledger_even_with_fields_and_old_facts(self):
        await self.setup_case()
        self.agent._call_llm_impl = AsyncMock(side_effect=[draft(self.item), citation(self.item, [(self.row["source_id"], self.quote)])])
        nli = {"status": "passed", "execution_status": "completed", "omission_check": "not_run",
               "claims": [{"confidence": .95, "needs_review": False}]}
        output = {}
        with bind_agent_history(self.scope), patch("service.nli_verifier.verify_claims", new=AsyncMock(return_value=nli)) as verify:
            result = await summarize_with_citations(self.agent, self.scope, [self.row], CharCounter(),
                previous_facts=[{"fact_id": "stale"}], verification_out=output)
        self.assertEqual(await self.stages(), ["summary_merge", "citation_completion"])
        self.assertEqual(output["fact_ledger"], [])
        self.assertEqual(output["omission_check"], "not_run")
        self.assertEqual(output["claims"][0]["confidence"], .95)
        self.assertIn('"start":', result)
        self.assertEqual(verify.call_args.args[0][0]["references"][0]["quote"], self.quote)

    async def test_optional_coverage_extracts_only_after_citation_and_never_reads_draft(self):
        await self.setup_case(nli_support=False, coverage=True)
        extracted = {"quote": self.quote, "topic": "样本量", "entity": "研究A",
                     "statement": self.quote, "reason": "任务必需", "kind": "fact", "fields": ["800"]}
        self.agent._call_llm_impl = AsyncMock(side_effect=[draft(self.item), citation(self.item, [(self.row["source_id"], self.quote)]),
            json.dumps([extracted], ensure_ascii=False), json.dumps({"status": "preserved", "claim_indexes": [0]})])
        output = {}
        with bind_agent_history(self.scope):
            await summarize_with_citations(self.agent, self.scope, [self.row], CharCounter(), verification_out=output)
        self.assertEqual(await self.stages(), ["summary_merge", "citation_completion", "fact_extraction", "coverage_check"])
        extraction = json.loads(self.agent._call_llm_impl.call_args_list[2].args[1])
        self.assertEqual(set(extraction), {"record"})
        self.assertEqual(output["omission_check"], "completed")
        self.assertEqual(len(output["fact_ledger"]), 1)

    async def test_field_check_still_rejects_wrong_number_without_fact_extraction(self):
        await self.setup_case(nli_support=False, repair_attempts=0)
        bad = clause("研究A纳入80人。", self.row["source_id"])
        self.agent._call_llm_impl = AsyncMock(side_effect=[draft(bad), citation(bad, [(self.row["source_id"], self.quote)])])
        with bind_agent_history(self.scope), self.assertRaisesRegex(ValueError, "核验拒绝"):
            await summarize_with_citations(self.agent, self.scope, [self.row], CharCounter())
        self.assertEqual(await self.stages(), ["summary_merge", "citation_completion"])
        rows = (await self.store.load(self.scope, "DeepScout"))["messages"]
        report = next(r["payload"] for r in rows if r["kind"] == "summary_verification")
        self.assertEqual(report["output_fields"]["status"], "rejected")

    async def test_optional_extractor_resolves_offsets_instead_of_trusting_model_count(self):
        fragment = {"source_id": "s", "start": 0, "text": '{"arguments": {"count": 10}}',
                    "source_type": "tool_record", "source_text_hash": "h"}
        item = {"quote": 'count": 10', "start": 26, "topic": "数量", "entity": "10",
                "statement": "搜索10条", "reason": "示例", "kind": "progress", "fields": ["count"]}
        call = AsyncMock(return_value=json.dumps([item]))
        facts = await extract_ledger([fragment], [], call, VerificationSettings(coverage=True))
        self.assertEqual(facts[0]["start"], 16)
        with self.assertRaisesRegex(ValueError, "定位"):
            await extract_ledger([fragment], [], call, VerificationSettings(rules_version="verification-v1", coverage=True))


if __name__ == "__main__":
    unittest.main()
