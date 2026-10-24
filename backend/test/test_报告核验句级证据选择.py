"""报告复用摘要选句协议，NLI与Critic仅消费选中原句；所有模型调用使用桩。"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from service.context_policy import NLISettings
from service.deep_research_v2.report_sentence_evidence import select_report_sentences, narrow_reference
from service.deep_research_v2.citation_verification import verify_stage_citations
from service.deep_research_v2.critic_evidence import build_critic_evidence
from test_最终报告独立引用核验 import state_for, extractor_stub, supported


class Counter:
    def count_text(self, text):
        return len(text)

    def count_messages(self, messages):
        payload = json.loads(messages[1]["content"])
        return 100 + sum(len(s["text"]) for b in payload["messages"] for s in b["sentences"])


def reference(text, ident="block_1"):
    return {"document_id": "doc_1", "source_id": "source_1", "source_url": "https://example.org/report",
            "evidence_unit_ids": [ident], "sentence_ids": ["s000001"], "quote": text}


def selector(predicate=lambda sentence: True):
    async def call(messages, settings):
        data = json.loads(messages[1]["content"])
        return json.dumps({"claim_id": data["claim_id"], "evidence": [
            {"source_id": block["source_id"], "sentence_ids": [s["sentence_id"] for s in block["sentences"]
             if predicate(s)]} for block in data["messages"]]}, ensure_ascii=False), {"mock": True}
    return call


class ReportSentenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_selected_sentences_and_conditions_reach_nli_input(self):
        refs = await select_report_sentences({"claim_id": "c", "claim": "研究结果"}, [
            reference("相关结果。无关插句。仅适用于成年人。")], NLISettings(),
            call=selector(lambda s: "无关" not in s["text"]), counter=Counter())
        self.assertEqual(refs[0]["quote"], "相关结果。\n仅适用于成年人。")
        self.assertEqual(refs[0]["sentence_selection"]["sentence_ids"], ["s0001", "s0003"])
        self.assertNotIn("无关插句", refs[0]["quote"])

    async def test_multiple_blocks_and_contradicting_sentences_are_preserved(self):
        value = {"claim_id": "c", "claim": "结果成立"}
        refs = await select_report_sentences(value, [reference("初步结果支持结论。"),
            reference("后续研究撤回该结论。", "block_2")], NLISettings(), call=selector(), counter=Counter())
        self.assertEqual(len(refs), 2)
        self.assertIn("撤回", refs[1]["quote"])
        self.assertEqual(len(value["evidence_selection"]["selected_ids"]), 2)

    async def test_empty_selection_does_not_fall_back_to_full_block(self):
        value = {"claim_id": "c", "claim": "未找到相关证据"}
        refs = await select_report_sentences(value, [reference("无关原文。")], NLISettings(),
            call=selector(lambda s: False), counter=Counter())
        self.assertEqual(refs, [])
        self.assertEqual(value["evidence_selection"]["phase"], "completed")

    async def test_invalid_or_fabricated_ids_and_quote_are_rejected(self):
        async def bad(messages, settings):
            data = json.loads(messages[1]["content"])
            return json.dumps({"claim_id": data["claim_id"], "evidence": [{
                "source_id": data["messages"][0]["source_id"], "sentence_ids": ["invented"], "quote": "伪造"}]}), {}
        value = {"claim_id": "c", "claim": "测试"}
        with self.assertRaises(ValueError):
            await select_report_sentences(value, [reference("原句。")], NLISettings(), call=bad, counter=Counter())
        self.assertEqual(value["evidence_selection"]["phase"], "error")

    async def test_batches_cover_tail_without_dropping_sentences(self):
        text = "".join(f"第{i}条证据包含必须保留的信息。" for i in range(50))
        call = AsyncMock(side_effect=selector())
        value = {"claim_id": "c", "claim": "测试"}
        refs = await select_report_sentences(value, [reference(text)], NLISettings(max_length=250),
                                             call=call, counter=Counter())
        self.assertGreater(call.await_count, 1)
        self.assertIn("第49条证据", refs[0]["quote"])
        self.assertEqual(len(refs[0]["sentence_selection"]["sentence_ids"]), 50)

    async def test_oversized_sentence_fails_before_call_without_truncation(self):
        call = AsyncMock()
        with self.assertRaisesRegex(ValueError, "未截断"):
            await select_report_sentences({"claim_id": "c", "claim": "测试"}, [reference("长句" * 100 + "。")],
                NLISettings(max_length=100), call=call, counter=Counter())
        call.assert_not_awaited()

    async def test_table_selection_retains_header_without_unselected_rows(self):
        table = "| 年份 | 指标 |\n| --- | --- |\n" + "\n".join(f"| {i} | 数值为{i} |" for i in range(60))
        refs = await select_report_sentences({"claim_id": "c", "claim": "第59行"}, [reference(table)],
            NLISettings(), call=selector(lambda s: "| 59 |" in s["text"]), counter=Counter())
        self.assertEqual(len(refs), 1)
        self.assertIn("| 年份 | 指标 |", refs[0]["quote"])
        self.assertIn("| 59 |", refs[0]["quote"])
        self.assertNotIn("| 0 |", refs[0]["quote"])

    async def test_critic_rag_rereads_selected_sentences_only(self):
        block = reference("需要保留。无关长篇。必要条件。")
        refs = await select_report_sentences({"claim_id": "c", "claim": "结论"}, [block], NLISettings(),
            call=selector(lambda s: "无关" not in s["text"]), counter=Counter())
        citation = {key: refs[0][key] for key in ("document_id", "evidence_unit_ids", "sentence_selection")}
        state = {"final_report": "结论", "claims": [{"claim": "结论", "evidence": [citation]}]}
        rag = SimpleNamespace(resolve_units=AsyncMock(return_value=[block]))
        with patch("service.deep_research_v2.critic_evidence.current_session_evidence", return_value=rag):
            packet = await build_critic_evidence(state)
        self.assertEqual(packet["excerpts"][0]["quote"], refs[0]["quote"])
        self.assertNotIn("无关长篇", packet["excerpts"][0]["quote"])
        rag.resolve_units.return_value = [{**block, "quote": "证据块已被替换。"}]
        with patch("service.deep_research_v2.critic_evidence.current_session_evidence", return_value=rag):
            packet = await build_critic_evidence(state)
        self.assertEqual(packet["excerpts"], [])
        self.assertTrue(packet["claims"][0]["unavailable_references"])

    async def test_block_hash_or_spans_cannot_be_tampered(self):
        with self.assertRaises(ValueError):
            narrow_reference(reference("原句。"), {"block_sha256": "wrong", "spans": [{"start": 0, "end": 2}]})
        selection = {"block_sha256": hashlib.sha256("原句。".encode()).hexdigest(),
                     "spans": [{"start": -1, "end": 2}]}
        with self.assertRaises(ValueError):
            narrow_reference(reference("原句。"), selection)


class ReportPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        extraction = patch("service.deep_research_v2.report_claim_extraction.call_extractor", side_effect=extractor_stub)
        extraction.start(); self.addCleanup(extraction.stop)
        selected = patch("service.deep_research_v2.report_sentence_evidence.call_selector", side_effect=selector(
            lambda s: "投运" in s["text"]))
        self.selected = selected.start(); self.addCleanup(selected.stop)
        nli = patch("service.nli_verifier.verify_claims", side_effect=supported)
        self.nli = nli.start(); self.addCleanup(nli.stop)

    async def test_report_nli_and_critic_share_same_selected_originals(self):
        state = state_for(body="无关背景。项目于2023年投运。无关尾句。")
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "passed")
        refs = self.nli.call_args.args[0][0]["references"]
        self.assertEqual([ref["quote"] for ref in refs], ["项目于2023年投运。"])
        self.assertIn("sentence_selection", result["claims"][0]["evidence"][0])
        packet = await build_critic_evidence(state)
        self.assertEqual([r["quote"] for r in packet["excerpts"]], ["项目于2023年投运。"])
        self.assertEqual(packet["claim_scope"], "final_report_extracted_claims")

    async def test_bad_selection_stops_before_nli_and_never_uses_candidates(self):
        self.selected.return_value = ('{"claim_id":"wrong","evidence":[]}', {})
        self.selected.side_effect = None
        state = state_for()
        result = await verify_stage_citations(state, "report_claims")
        self.assertEqual(result["status"], "error")
        self.nli.assert_not_awaited()
        self.assertEqual(state["report_verification_claims"][0]["evidence"], [])

    async def test_no_relevant_sentence_is_recorded_without_full_block_fallback(self):
        self.selected.side_effect = selector(lambda s: False)
        result = await verify_stage_citations(state_for(), "report_claims")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(self.nli.call_args.args[0][0]["references"], [])
        self.assertIn("cited_source_sentences_not_selected", result["claims"][0]["binding_errors"])


if __name__ == "__main__":
    unittest.main()
