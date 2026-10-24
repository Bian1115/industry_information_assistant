"""正文直接入库与 Writer 消费原文回归；不请求真实模型或业务数据库。"""
import importlib
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import uuid

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "eval/tests")]

from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.agents.writer import LeadWriter, InsufficientEvidenceError
from service.deep_research_v2.agents.data_analyst import DataAnalyst
from service.deep_research_v2.state import create_initial_state
from service.deep_research_v2.quality_pipeline import evidence_fingerprint
from service.deep_research_v2.citation_verification import verify_stage_citations
from service.deep_research_v2.writer_claims import collect_report_claims, format_source_evidence
from service.session_evidence_rag import bind_session_evidence
from service.web_document import assess_body, index_document
from service.web_reader_service import WebReadResult

URL = "https://official.example.cn/report"
BODY = "公司2025年营业收入为100亿元。\n\n统计范围仅包含中国市场。\n\n本报告保留原始统计口径。"
OBSERVATIONS = []


def source():
    text, quality = assess_body(BODY, "年度报告")
    return {"id": "source_1", "url": URL, "title": "年度报告", "content": text,
            "related_sections": ["sec_1"], "body_quality": quality, **index_document(text, URL)}


def state():
    work = create_initial_state("分析公司收入与统计范围", "test", ["official.example.cn"])
    work["phase"] = "researching"
    work["outline"] = [{"id": "sec_1", "title": "营业收入", "description": "统计口径",
                        "status": "pending", "search_queries": ["公司营业收入"]}]
    return work


def scout():
    agent = DeepScout.__new__(DeepScout)
    agent.name, agent.logger = "DeepScout", logging.getLogger("test.scout.sources")
    agent.web_reader = SimpleNamespace(
        read_many=AsyncMock(return_value=[WebReadResult(url=URL, final_url=URL, title="年度报告",
            content=BODY, fetch_status="success", extractor="fixture", fetched_at=1)]),
        cache_dir=Path("unused"), _memory_cache={},
    )
    agent._execute_search = AsyncMock(return_value=[{"url": URL, "title": "年度报告"}])
    agent.call_llm = AsyncMock(side_effect=AssertionError("Scout不得提炼事实"))
    for name in ("_analyze_search_results", "_analyze_supplementary_results", "_analyze_deep_search_results"):
        setattr(agent, name, AsyncMock(side_effect=AssertionError("写作前事实提炼被调用")))
    return agent


def writer():
    agent = LeadWriter.__new__(LeadWriter)
    agent.name, agent.logger = "LeadWriter", logging.getLogger("test.writer.sources")
    agent.parse_json_response = json.loads
    return agent


class SourceCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_search_saves_original_sources_without_facts_or_llm(self):
        work, agent = state(), scout()
        previous_verification = dict(work["scout_citation_verification"])
        await agent.process(work)
        self.assertEqual(work["facts"], [])
        self.assertEqual(work["data_points"], [])
        self.assertEqual(work["hypothesis_evidence"], [])
        self.assertEqual(work["raw_sources"][0]["content"], BODY)
        self.assertEqual(work["raw_sources"][0]["related_sections"], ["sec_1"])
        agent.call_llm.assert_not_awaited()
        agent._analyze_search_results.assert_not_awaited()
        self.assertEqual(work["scout_citation_verification"], previous_verification)
        OBSERVATIONS.append({"case": self._testMethodName, "sources": len(work["raw_sources"]),
                             "facts": len(work["facts"]), "scout_llm_calls": agent.call_llm.await_count})

    async def test_supplementary_search_preserves_all_target_sections_on_source(self):
        work, agent = state(), scout()
        work.update(phase="re_researching", pending_research_tasks=[{
            "query": "收入原文", "target_section": "sec_1/sec_2", "issue_id": "issue_1"}])
        await agent.process(work)
        self.assertEqual(work["raw_sources"][0]["related_sections"], ["sec_1", "sec_2"])
        self.assertTrue(work["raw_sources"][0]["is_supplementary"])
        self.assertEqual(work["raw_sources"][0]["critic_issue_id"], "issue_1")
        self.assertEqual(work["facts"], [])
        self.assertEqual(work["pending_research_tasks"], [])
        agent._analyze_supplementary_results.assert_not_awaited()
        agent.call_llm.assert_not_awaited()

    async def test_deep_search_only_reads_and_saves_original_sources(self):
        work, agent = state(), scout()
        await agent._execute_deep_search(work, "sec_1", ["原始报告"], "source_tracing", [])
        self.assertEqual(work["raw_sources"][0]["search_type"], "source_tracing")
        self.assertEqual(work["facts"], [])
        agent._analyze_deep_search_results.assert_not_awaited()
        agent.call_llm.assert_not_awaited()

    async def test_data_analysis_does_not_recreate_semantic_extraction(self):
        work = state()
        work.update(phase="analyzing", raw_sources=[source()], facts=[{"content": "历史摘要"}])
        agent = DataAnalyst.__new__(DataAnalyst)
        agent.name, agent.logger, agent.generate_charts = "DataAnalyst", logging.getLogger("test.analyst.sources"), False
        agent._extract_data = AsyncMock(side_effect=AssertionError("不得提炼数据"))
        agent._build_knowledge_graph = AsyncMock(side_effect=AssertionError("不得提炼实体"))
        agent.call_llm = AsyncMock(side_effect=AssertionError("不得调用模型"))
        await agent.process(work)
        agent._extract_data.assert_not_awaited()
        agent._build_knowledge_graph.assert_not_awaited()
        agent.call_llm.assert_not_awaited()


class WriterSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_writer_accepts_original_sources_when_fact_list_is_empty(self):
        work, agent = state(), writer()
        work["raw_sources"] = [source()]
        agent._write_checkpointed_section = AsyncMock()
        async def synthesize(work): work["final_report"] = "已有原文的报告"
        agent._synthesize_report = AsyncMock(side_effect=synthesize)
        await agent._write_report(work)
        agent._write_checkpointed_section.assert_awaited_once()
        self.assertEqual(work["facts"], [])

    async def test_writer_rejects_fact_summary_without_original_source(self):
        work, agent = state(), writer()
        work["facts"] = [{"id": "old", "content": "只有旧摘要，没有原文"}]
        with self.assertRaises(InsufficientEvidenceError):
            await agent._write_report(work)

    async def test_nonpersistent_writer_receives_numbered_original_text_without_facts(self):
        work, agent = state(), writer()
        work["raw_sources"] = [source()]
        agent.call_llm = AsyncMock(return_value=json.dumps({"content": f"收入[报告]({URL})", "claims": []}))
        await agent._write_section(work, work["outline"][0])
        prompt = agent.call_llm.call_args.kwargs["user_prompt"]
        self.assertIn("公司2025年营业收入为100亿元", prompt)
        self.assertIn('"original_sentences"', prompt)
        self.assertIn('"facts":[]', prompt)
        self.assertNotIn("### 相关事实", prompt)
        self.assertNotIn("### 已有洞察", prompt)

    async def test_empty_retrieval_stops_section_before_calling_writer_model(self):
        work, agent = state(), writer()
        agent.call_llm = AsyncMock()
        with self.assertRaises(InsufficientEvidenceError):
            await agent._write_section(work, work["outline"][0])
        agent.call_llm.assert_not_awaited()

    def test_source_only_evidence_fingerprint_tracks_content_and_sections(self):
        work = state()
        work["raw_sources"] = [source()]
        first = evidence_fingerprint(work)
        work["raw_sources"][0]["related_sections"].append("sec_2")
        self.assertNotEqual(first, evidence_fingerprint(work))
        second = evidence_fingerprint(work)
        work["raw_sources"][0]["content"] += "新增原文。"
        self.assertNotEqual(second, evidence_fingerprint(work))

    def test_local_source_view_respects_section_scope(self):
        work = state()
        work["raw_sources"] = [source()]
        packet = json.loads(format_source_evidence(work, "收入", section_id="sec_2"))
        self.assertEqual(packet["documents"], [])

    def test_local_source_view_does_not_fabricate_unregistered_document_ids(self):
        work = state()
        work["raw_sources"] = [{"id": "unindexed", "url": URL, "content": BODY}]
        packet = json.loads(format_source_evidence(work, "收入"))
        self.assertEqual(packet["documents"], [])
        self.assertNotIn("document_id", work["raw_sources"][0])


class PersistentSourceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = importlib.import_module("test_会话临时双层RAG")
        self.case = self.fixture.TemporaryRAGTests("test_chunk_metadata_and_adjacent_ids")
        await self.case.asyncSetUp()

    async def asyncTearDown(self):
        await self.case.asyncTearDown()

    async def test_scout_ingest_writer_and_postwrite_citation_retrieval_without_facts(self):
        work, agent, rag = state(), scout(), self.case.rag
        work["session_id"] = str(self.case.sid)
        await rag.attach(work, run_id=str(uuid.uuid4()))
        with tempfile.TemporaryDirectory() as cache, \
                patch("service.session_evidence_lifecycle.evidence_cache_root", return_value=Path(cache)), \
                bind_session_evidence(rag):
            await agent.process(work)
            rows = rag.store.chunks(self.case.uid, self.case.sid, rag.topic(work), kinds={"paragraph"})
            self.assertTrue(rows)
            unit = rows[0]
            writing = writer()
            writing.call_llm = AsyncMock(return_value=json.dumps({
                "content": f"公司收入为100亿元。[报告]({URL})", "claims": [{
                    "claim": "公司2025年营业收入为100亿元", "report_quote": f"公司收入为100亿元。[报告]({URL})",
                    "evidence": [{"document_id": unit["metadata"]["document_id"], "evidence_unit_ids": [unit["id"]]}]}]}))
            await writing._write_section(work, work["outline"][0])
            work["final_report"] = work["draft_sections"]["sec_1"]
            work["claims"] = collect_report_claims(work)
            await verify_stage_citations(work, "report_claims")
        self.assertEqual(work["facts"], [])
        self.assertIn('"evidence_units"', writing.call_llm.call_args.kwargs["user_prompt"])
        self.assertEqual(work["citation_retrieval_status"]["status"], "completed")
        self.assertTrue(work["retrieved_citation_bindings"][0]["evidence"])
        OBSERVATIONS.append({"case": self._testMethodName, "stored_chunks": len(rows), "facts": 0,
                             "writer_calls": writing.call_llm.await_count,
                             "citation_retrieval": work["citation_retrieval_status"]})

    async def test_existing_source_adds_section_metadata_without_reembedding(self):
        case = self.case
        src = source()
        case.state["raw_sources"].append(src)
        await case.rag.ingest(case.state)
        embeds = len(case.api.embedded)
        before_revision = case.state["evidence_space"]["revision"]
        src = next(ref for ref in case.state["raw_sources"] if ref["document_id"] == src["document_id"])
        src["related_sections"].append("sec_2")
        await case.rag.ingest(case.state)
        stored = case.store.documents(case.uid, case.sid, case.rag.topic(case.state), document_ids=[src["document_id"]])[0]
        self.assertEqual(stored["related_sections"], ["sec_1", "sec_2"])
        chunks = [row for row in case.rows() if row["metadata"]["document_id"] == src["document_id"]]
        self.assertTrue(all(row["metadata"]["related_sections"] == ["sec_1", "sec_2"] for row in chunks))
        self.assertEqual(embeds, len(case.api.embedded))
        self.assertGreater(case.state["evidence_space"]["revision"], before_revision)


if __name__ == "__main__":
    unittest.main()
