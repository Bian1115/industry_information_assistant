"""研究状态只留来源引用；原文片段由RAG直接提供，不恢复或比对全文。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import uuid

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "app"), str(Path(__file__).parent)]
import test_会话临时双层RAG as fixtures
import test_取消写作前事实提炼 as source_fixtures
from service.deep_research_v2.evidence import normalize_evidence_state
from service.deep_research_v2.followup_context import inherit_artifacts
from service.deep_research_v2.quality_pipeline import evidence_fingerprint
from service.deep_research_v2.citation_verification import verify_stage_citations
from service.session_evidence_rag import SessionEvidenceRAG, bind_session_evidence
from service.session_evidence_store import EvidenceExpiredError
from service.web_document import source_available


class DirectChunkTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.TemporaryRAGTests.asyncSetUp
    asyncTearDown = fixtures.TemporaryRAGTests.asyncTearDown
    create_engine = fixtures.TemporaryRAGTests.create_engine
    rows = fixtures.TemporaryRAGTests.rows

    def assert_source_refs(self, state):
        for source in state.get("raw_sources") or []:
            self.assertTrue(source["evidence_document_ref"])
            for key in ("content", "full_content", "sentence_index"):
                self.assertNotIn(key, source)

    async def test_vector_failure_leaves_durable_source_but_no_full_text_in_state(self):
        new = fixtures.source("新材料：2026年销量下降。", "https://example.com/new")
        self.vectors.fail = True
        await self.rag.ingest(self.state, sources=[*self.state["raw_sources"], new])
        self.assert_source_refs(self.state)
        self.assertTrue(self.state["evidence_rag_metrics"][-1]["index_status"].startswith("lexical_only:"))
        saved = self.store.documents(self.uid, self.sid, self.rag.topic(self.state), document_ids=[new["document_id"]])
        self.assertEqual(saved[0]["content"], new["content"])

    async def test_chunk_citations_never_fetch_or_compare_source_document(self):
        table = self.rows({"table"})[0]
        self.state["claims"] = [{"claim": "2025年销量为100万台", "evidence": [{
            "document_id": table["metadata"]["document_id"], "evidence_unit_ids": [table["id"]]}]}]
        self.state["citation_verification_config"] = {"report_claims": True}
        extraction = {"status": "completed", "units_total": 1, "claims": [{
            "claim_id": "claim_1", "claim": "2025年销量为100万台", "section_id": "sec_1",
            "assertion_kind": "fact", "report_quote": "2025年销量为100万台",
            "citation_urls": [table["metadata"]["source_url"]], "binding_errors": []}]}
        verifier = AsyncMock(return_value={"status": "incomplete", "claims": []})
        with patch.object(self.store, "documents", side_effect=AssertionError("禁止引用全文回读")), \
                patch("service.deep_research_v2.report_claim_extraction.extract_report_claims", AsyncMock(return_value=extraction)), \
                patch("service.deep_research_v2.citation_verification.verify_citations", verifier), \
                bind_session_evidence(self.rag):
            await verify_stage_citations(self.state, "report_claims")
        refs = verifier.call_args.args[0][0]["references"]
        table_ref = next(ref for ref in refs if table["id"] in ref["evidence_unit_ids"])
        self.assertEqual(table_ref["quote"], table["content"])
        self.assertIn("销量（万台）", table_ref["quote"])
        self.assert_source_refs(self.state)

    async def test_quote_uses_chunk_text_even_when_offsets_are_not_readable(self):
        row = deepcopy(self.rows({"sentence"})[0])
        row["metadata"]["source_spans"] = [{"start": 999999, "end": 1000000}]
        with patch.object(self.store, "documents", side_effect=AssertionError("不得重建引句")):
            refs = await self.rag.references(self.state, [row])
        self.assertEqual(refs[0]["quote"], row["content"])

    async def test_normalization_and_followup_preserve_reference_hash_without_full_text(self):
        before_hash = self.state["raw_sources"][0]["content_sha256"]
        before_fingerprint = evidence_fingerprint(self.state)
        normalize_evidence_state(self.state)
        self.assertEqual(self.state["raw_sources"][0]["content_sha256"], before_hash)
        self.assertEqual(evidence_fingerprint(self.state), before_fingerprint)
        child = {"allowed_domains": [], "hypothesis_enabled": False}
        inherit_artifacts(child, {**self.state, "report_hash": "test-report"}, replan=True)
        self.assert_source_refs(child)
        self.assertTrue(source_available(child["raw_sources"][0]))

    async def test_restart_fetches_old_topic_chunks_without_hydrating_state(self):
        reloaded = json.loads(json.dumps(self.state))
        second = SessionEvidenceRAG(str(self.uid), str(self.sid), settings=self.rag.settings,
            store=self.store, api=fixtures.API(), vectors=self.vectors, counter=self.counter)
        try:
            with patch.object(self.store, "documents", side_effect=AssertionError("禁止恢复全文")):
                await second.prepare_stage(reloaded, "writing", "restart")
                prompt = await second.evidence_prompt(reloaded, "2025年苹果销量")
            self.assertIn("苹果销量增长", prompt)
            self.assert_source_refs(reloaded)
            self.assertFalse(hasattr(second, "hydrate"))
        finally:
            await second.close()

    async def test_deleted_reference_is_rejected_without_fabricating_text(self):
        self.state["raw_sources"][0]["document_id"] = "doc_missing"
        with self.assertRaises(EvidenceExpiredError):
            await self.rag.prepare_stage(self.state, "writing", "restart")
        self.assert_source_refs(self.state)

    async def test_scout_to_writer_keeps_full_text_out_of_runtime_state(self):
        work, scout, writer = source_fixtures.state(), source_fixtures.scout(), source_fixtures.writer()
        work["session_id"] = str(self.sid)
        await self.rag.attach(work, run_id=str(uuid.uuid4()))
        writer.call_llm = AsyncMock(return_value=json.dumps({"content": "公司2025年营业收入为100亿元。", "claims": []}))
        with tempfile.TemporaryDirectory() as cache, \
                patch("service.session_evidence_lifecycle.evidence_cache_root", return_value=Path(cache)), \
                patch.object(self.store, "documents", side_effect=AssertionError("不得恢复来源全文")), \
                bind_session_evidence(self.rag):
            await scout.process(work)
            self.assert_source_refs(work)
            await writer._write_section(work, work["outline"][0])
            self.assert_source_refs(work)
        self.assertIn("营业收入为100亿元", writer.call_llm.call_args.kwargs["user_prompt"])
        self.assertTrue(work["draft_sections"]["sec_1"])


if __name__ == "__main__":
    unittest.main()
