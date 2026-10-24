"""Critic全量现有论断、原文回读和版本隔离；离线测试，不调用模型。"""

from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from service.deep_research_v2.agents.critic import CriticMaster
from service.deep_research_v2.critic_evidence import build_critic_evidence
from service.web_document import index_document


URL = "https://example.org/report"
ORIGINAL = "公司2025年收入为100亿元。"
REPORT = ORIGINAL + f"[年报]({URL})"
SOURCE = {"id": "source_1", "url": URL, "title": "年报", "content": ORIGINAL,
          "body_quality": {"status": "passed"},
          **index_document(ORIGINAL, URL)}


def claim(ident="claim_1"):
    return {"claim_id": ident, "claim": ORIGINAL, "report_quote": REPORT, "section_id": "sec_1",
            "evidence": [{"document_id": SOURCE["document_id"], "sentence_ids": ["s000001"]}]}


def state():
    return {"query": "分析公司收入", "outline": [{"id": "sec_1", "title": "收入"}],
            "final_report": REPORT, "raw_sources": [deepcopy(SOURCE)], "data_points": [],
            "claims": [claim()], "report_verification_claims": [], "research_contract": {}}


class CriticEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rag_patch = patch("service.deep_research_v2.critic_evidence.current_session_evidence", return_value=None)
        self.current_rag = self.rag_patch.start()
        self.addCleanup(self.rag_patch.stop)

    async def test_reads_saved_original_instead_of_model_quote(self):
        value = state()
        value["claims"][0]["evidence"][0]["quote"] = "模型编造的原文"
        packet = await build_critic_evidence(value)
        self.assertEqual(packet["excerpts"][0]["quote"], ORIGINAL)
        self.assertEqual(packet["excerpts"][0]["source_url"], URL)
        self.assertNotIn("模型编造的原文", json.dumps(packet, ensure_ascii=False))
        self.assertEqual(packet["claim_scope"], "writer_declared_claims_only")

    async def test_current_independent_extraction_takes_precedence(self):
        value = state()
        value["report_claim_extraction"] = {"report_sha256": hashlib.sha256(REPORT.encode()).hexdigest(), "status": "completed"}
        value["report_verification_claims"] = [claim("independent_1")]
        value["claims"][0]["claim"] = "不应混入的旧Writer声明"
        value["citation_verification"] = {"report_sha256": value["report_claim_extraction"]["report_sha256"],
            "status": "passed", "claims": [{"claim_id": "independent_1", "claim": ORIGINAL, "status": "supported"}]}
        packet = await build_critic_evidence(value)
        self.assertEqual(packet["claim_scope"], "final_report_extracted_claims")
        self.assertEqual(packet["claims"][0]["claim_id"], "independent_1")
        self.assertEqual(packet["claims"][0]["verification_status"], "supported")
        self.assertNotIn("旧Writer声明", json.dumps(packet, ensure_ascii=False))
        self.assertFalse(packet["all_claims_extracted"])

    async def test_current_empty_extraction_does_not_restore_old_claims(self):
        value = state()
        value["report_claim_extraction"] = {"report_sha256": hashlib.sha256(REPORT.encode()).hexdigest(), "status": "completed"}
        packet = await build_critic_evidence(value)
        self.assertEqual(packet["claims"], [])
        self.assertEqual(packet["claim_scope"], "final_report_extracted_claims")

    async def test_stale_extraction_and_verification_are_not_used(self):
        value = state()
        value["report_claim_extraction"] = {"report_sha256": "old", "status": "completed"}
        value["report_verification_claims"] = [{"claim": "过期论断"}]
        value["citation_verification"] = {"report_sha256": "old", "status": "passed", "claims": [
            {"claim_id": "claim_1", "claim": ORIGINAL, "status": "supported"}]}
        packet = await build_critic_evidence(value)
        self.assertEqual(packet["extraction_status"], "stale")
        self.assertEqual(packet["verification_status"], "not_current")
        self.assertEqual(packet["claims"][0]["verification_status"], "not_executed")
        self.assertNotIn("过期论断", json.dumps(packet, ensure_ascii=False))

    async def test_all_claims_and_shared_full_rag_table_are_retained(self):
        value = state()
        value["claims"] = [{**claim(f"claim_{i}"), "evidence": [
            {"document_id": SOURCE["document_id"], "evidence_unit_ids": ["table_1"]}]} for i in range(141)]
        table = "| 年份 | 收入 |\n| --- | --- |\n| 2025 | 100亿元 |\n" + "原表注释。" * 1000
        rag = SimpleNamespace(resolve_units=AsyncMock(return_value=[{
            "document_id": SOURCE["document_id"], "evidence_unit_ids": ["table_1"],
            "unit_type": "table", "source_url": URL, "quote": table}]))
        self.current_rag.return_value = rag
        packet = await build_critic_evidence(value)
        self.assertEqual(packet["claims_count"], 141)
        self.assertEqual(packet["claims"][-1]["claim_id"], "claim_140")
        self.assertEqual(packet["excerpts_count"], 1)
        self.assertEqual(packet["excerpts"][0]["quote"], table)
        self.assertTrue(all(row["evidence_ids"] == ["excerpt_1"] for row in packet["claims"]))
        self.assertEqual(rag.resolve_units.await_count, 1)

    async def test_missing_and_invalid_citations_keep_the_claim(self):
        value = state()
        value["claims"] = [{**claim("no_evidence"), "evidence": []}, {**claim("bad_evidence"),
            "evidence": [{"document_id": "unknown", "sentence_ids": ["s000001"]}]}]
        packet = await build_critic_evidence(value)
        self.assertEqual(packet["claims_count"], 2)
        self.assertEqual(packet["excerpts"], [])
        self.assertTrue(packet["claims"][1]["unavailable_references"])

    async def test_missing_unit_store_is_explicit_and_does_not_trust_quote(self):
        value = state()
        value["claims"][0]["evidence"] = [{"document_id": SOURCE["document_id"],
            "evidence_unit_ids": ["unit_1"], "quote": "伪造引用"}]
        packet = await build_critic_evidence(value)
        self.assertEqual(packet["excerpts"], [])
        self.assertEqual(packet["claims"][0]["unavailable_references"][0]["errors"], ["evidence_unit_store_unavailable"])

    async def test_permission_error_is_not_hidden(self):
        value = state()
        self.current_rag.return_value = SimpleNamespace(sentence_references=AsyncMock(side_effect=PermissionError("not owned")))
        with self.assertRaises(PermissionError):
            await build_critic_evidence(value)

    async def test_prompt_contains_claims_originals_and_old_issue(self):
        value = state()
        value["critic_feedback"] = [{"id": "旧问题1", "description": "需解释统计范围", "resolved": False}]
        critic = CriticMaster.__new__(CriticMaster)
        critic.logger = logging.getLogger("test.critic.evidence")
        critic.call_llm = AsyncMock(return_value=json.dumps({"issues": [], "overall_assessment": {
            "quality_score": 8, "verdict": "pass"}}))
        critic.parse_json_response = json.loads
        await critic._review_content(value)
        prompt = critic.call_llm.await_args.kwargs["user_prompt"]
        for token in ("claim_1", "excerpt_1", ORIGINAL, "旧问题1", "验收条件"):
            self.assertIn(token, prompt)
        self.assertNotIn("### 引用目录", prompt)


if __name__ == "__main__":
    unittest.main()
