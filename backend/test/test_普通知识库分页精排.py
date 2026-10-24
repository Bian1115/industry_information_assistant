"""普通知识库共用会话RAG参数：分页、第三页证据、精排预算和降级。"""
import asyncio
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from service import retrieval_service as retrieval
from service.session_evidence_rag import SessionEvidenceSettings


class Counter:
    def count_text(self, text):
        return len(text)


class RankingAPI:
    def __init__(self):
        self.calls = []
        self.closed = False

    async def rerank(self, query, documents, top_n, *, instruct):
        self.calls.append(list(documents))
        indexes = sorted(range(len(documents)), key=lambda i: -int(documents[i][:4]))[:top_n]
        return [{"index": i, "relevance_score": .9} for i in indexes]

    async def close(self):
        self.closed = True


def chunks(count, start=0, length=0):
    return [{"id": f"片段-{i:04d}", "doc_id": "已授权文档", "filename": "测试材料",
             "chunk_index": i, "content": f"{i:04d}" + "原" * length, "score": .9}
            for i in range(start, start + count)]


class PagingTests(unittest.IsolatedAsyncioTestCase):
    def coarse(self, dense, lexical, settings=None, *, audit=None, embedding_error=None, lexical_error=None):
        milvus = Mock()
        milvus.search.side_effect = lambda **kw: dense[kw["offset"]:kw["offset"] + kw["top_k"]]
        def search(index, question, *, top_k, offset):
            if lexical_error:
                raise lexical_error
            return lexical[offset:offset + top_k]
        with patch.object(retrieval, "generate_embedding", return_value=[[.1]], side_effect=embedding_error) as embed, \
             patch.object(retrieval, "get_milvus_service", return_value=milvus), \
             patch("service.es_service.search", side_effect=search) as lexical_search:
            result = retrieval.hybrid_retrieve("私有知识库", "研究", allow_benchmark=False,
                settings=settings, audit=audit)
        return result, milvus, lexical_search, embed

    async def test_third_page_evidence_survives_coarse_and_reranking(self):
        rows = chunks(384)
        audit, api = {}, RankingAPI()
        result, milvus, lexical, embed = self.coarse(rows, rows, SessionEvidenceSettings(), audit=audit)
        self.assertEqual(len(result), 384)
        self.assertEqual([c.kwargs["offset"] for c in milvus.search.call_args_list], [0, 128, 256])
        self.assertEqual([c.kwargs["offset"] for c in lexical.call_args_list], [0, 128, 256])
        self.assertTrue(all(c.kwargs["top_k"] == 128 for c in milvus.search.call_args_list))
        embed.assert_called_once_with(["研究"])
        selected = await retrieval.rerank_knowledge_candidates("研究", result,
            settings=SessionEvidenceSettings(), counter=Counter(), api=api, audit=audit)
        self.assertEqual([r["chunk_index"] for r in selected], list(range(383, 375, -1)))
        self.assertEqual(audit["rerank_scored_candidate_count"], 384)
        self.assertIn("0383", api.calls[-1])
        self.assertEqual(len(audit["coarse_pages"]), 3)

    async def test_two_disjoint_routes_are_capped_after_fusion(self):
        result, *_ = self.coarse(chunks(384), chunks(384, start=384), SessionEvidenceSettings())
        self.assertEqual(len(result), 384)
        self.assertEqual(len({r["chunk_id"] for r in result}), 384)
        self.assertTrue(any(r["chunk_index"] >= 384 for r in result))

    async def test_environment_parameters_are_shared_instead_of_old_ten(self):
        with patch.dict(os.environ, {"SESSION_EVIDENCE_COARSE_PAGE_SIZE": "16",
                                    "SESSION_EVIDENCE_COARSE_PAGES": "2",
                                    "SESSION_EVIDENCE_RRF_K": "12"}):
            audit = {}
            result, milvus, _, _ = self.coarse(chunks(80), chunks(80), audit=audit)
        self.assertEqual(len(result), 32)
        self.assertEqual([c.kwargs["offset"] for c in milvus.search.call_args_list], [0, 16])
        self.assertAlmostEqual(result[0]["rrf_score"], 2 / 13)
        self.assertEqual(audit["coarse_candidate_limit"], 32)

    async def test_partial_last_page_stops_without_padding(self):
        result, milvus, lexical, _ = self.coarse(chunks(140), chunks(140), SessionEvidenceSettings())
        self.assertEqual(len(result), 140)
        self.assertEqual(milvus.search.call_count, 2)
        self.assertEqual(lexical.call_count, 2)

    async def test_embedding_failure_still_returns_keyword_pages(self):
        result, milvus, lexical, _ = self.coarse([], chunks(270), SessionEvidenceSettings(),
            embedding_error=RuntimeError("模拟向量故障"))
        self.assertEqual(len(result), 270)
        milvus.search.assert_not_called()
        self.assertEqual(lexical.call_count, 3)

    async def test_keyword_failure_still_returns_all_dense_pages(self):
        result, milvus, lexical, _ = self.coarse(chunks(384), [], SessionEvidenceSettings(),
            lexical_error=RuntimeError("模拟关键词故障"))
        self.assertEqual(len(result), 384)
        self.assertEqual(milvus.search.call_count, 3)
        self.assertEqual(lexical.call_count, 1)

    async def test_long_candidates_all_score_before_final_common_batch(self):
        docs = [{"content": f"{i:04d}" + "原" * 1020} for i in range(384)]
        api, audit = RankingAPI(), {}
        result = await retrieval.rerank_knowledge_candidates("研究", docs,
            settings=SessionEvidenceSettings(), counter=Counter(), api=api, audit=audit)
        self.assertEqual(int(result[0]["content"][:4]), 383)
        self.assertEqual(audit["rerank_scored_candidate_count"], 384)
        self.assertGreater(audit["rerank_rounds"], 1)
        self.assertEqual(audit["rerank_calls"][-1]["phase"], "final")
        self.assertTrue(all(call["input_tokens"] <= 110000 for call in audit["rerank_calls"]))

    async def test_failed_rerank_keeps_coarse_order_and_context_budget(self):
        api = Mock(rerank=AsyncMock(side_effect=RuntimeError("模拟精排故障")))
        audit = {}
        docs = [{"content": f"{i:04d}" + "原" * 30} for i in range(20)]
        settings = SessionEvidenceSettings(context_tokens=128, context_top_k=3)
        result = await retrieval.rerank_knowledge_candidates("研究", docs,
            settings=settings, counter=Counter(), api=api, audit=audit)
        self.assertEqual(audit["rerank_status"], "failed")
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 128)
        self.assertEqual(result[0]["content"], docs[0]["content"])
        self.assertLessEqual(len(result), 3)

    async def test_oversized_query_fails_explicitly_without_api_call(self):
        api, audit = RankingAPI(), {}
        result = await retrieval.rerank_knowledge_candidates("问" * 4001, [{"content": "0000"}],
            settings=SessionEvidenceSettings(), counter=Counter(), api=api, audit=audit)
        self.assertEqual(api.calls, [])
        self.assertEqual(audit["rerank_scored_candidate_count"], 0)
        self.assertIn("rerank_error_reason", audit)
        self.assertEqual(result[0]["content"], "0000")

    async def test_invalid_rerank_indexes_cannot_bind_wrong_text(self):
        api = Mock(rerank=AsyncMock(return_value=[{"index": 900}]))
        result = await retrieval.rerank_knowledge_candidates("研究", [{"content": "0000"}],
            settings=SessionEvidenceSettings(), counter=Counter(), api=api)
        self.assertEqual(result[0]["content"], "0000")

    async def test_chat_worker_reuses_shared_reranker_and_closes_client(self):
        from service.chat_service import ChatService
        service = ChatService.__new__(ChatService)
        service.token_counter = Counter()
        api = RankingAPI()
        with patch("service.summary_temporary_retrieval.RetrievalAPI", return_value=api), \
             patch.object(retrieval, "knowledge_retrieval_settings", return_value=SessionEvidenceSettings()):
            result = await asyncio.to_thread(service.rerank_documents, "研究",
                [{"content": f"{i:04d}"} for i in range(384)])
        self.assertEqual(result[0]["content"], "0383")
        self.assertTrue(api.closed)


class IdentityAndBackendTests(unittest.TestCase):
    def test_same_prefix_different_chunks_are_kept_and_route_duplicates_count_once(self):
        a = {"chunk_id": "甲", "content_with_weight": "相" * 100 + "甲"}
        b = {"chunk_id": "乙", "content_with_weight": "相" * 100 + "乙"}
        result = retrieval.reciprocal_rank_fusion([[a, a, b], [a]], top_k=10)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["rrf_score"], 2 / 61)

    def test_elasticsearch_passes_offset_and_stable_chunk_id(self):
        from service import es_service
        client = Mock()
        client.search.return_value = {"hits": {"hits": [{"_id": "片段甲", "_score": 1,
            "_source": {"doc_id": "文档", "content": "正文", "chunk_index": 3}}]}}
        with patch.object(es_service, "get_es_client", return_value=client):
            result = es_service.search("私有库", "问题", top_k=128, offset=256)
        self.assertEqual(client.search.call_args.kwargs["from_"], 256)
        self.assertEqual(client.search.call_args.kwargs["size"], 128)
        self.assertEqual(result[0]["id"], "片段甲")

    def test_milvus_passes_offset_without_rebuilding_existing_index(self):
        from service import milvus_service
        service = milvus_service.MilvusService.__new__(milvus_service.MilvusService)
        collection = Mock()
        collection.search.return_value = [[]]
        with patch.object(milvus_service.utility, "has_collection", return_value=True), \
             patch.object(milvus_service, "Collection", return_value=collection):
            service.search("私有库", [.1], top_k=128, offset=256)
        self.assertEqual(collection.search.call_args.kwargs["param"]["offset"], 256)
        self.assertEqual(collection.search.call_args.kwargs["limit"], 128)
        collection.create_index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
