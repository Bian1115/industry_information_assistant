"""粗排大候选集的精排组批：输入有界、全部候选评分、跨批统一比较。"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
from service.session_evidence_rag import SessionEvidenceSettings
from service.session_evidence_ranking import (rerank_candidates, RerankBudgetError,
                                             CONTEXT_INSTRUCTION, CITATION_INSTRUCTION)


class Counter:
    def count_text(self, text):
        return len(text.split()) if text in {CONTEXT_INSTRUCTION, CITATION_INSTRUCTION} else len(text)


class RankingAPI:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, documents, top_n, *, instruct):
        self.calls.append({"documents": list(documents), "top_n": top_n, "instruct": instruct})
        # 每批相对分数刻意偏向早期批次；整体最终排序只取决于条目号。
        indexes = sorted(range(len(documents)), key=lambda i: -int(documents[i][:4]))[:top_n]
        score = .99 if len(self.calls) == 1 else .01
        return [{"index": i, "relevance_score": score} for i in indexes]


class RerankPagingTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_384_long_candidates_are_scored_then_finalists_compete_together(self):
        for layer, top_k in (("context", 8), ("citation", 5)):
            with self.subTest(layer=layer):
                api, audit = RankingAPI(), {}
                documents = [f"{i:04d}" + "原" * 1020 for i in range(384)]
                result = await rerank_candidates(api, Counter(), "研究", documents, top_k,
                    layer=layer, settings=SessionEvidenceSettings(), audit=audit)
                self.assertEqual([row["index"] for row in result], list(range(383, 383-top_k, -1)))
                self.assertGreater(audit["rerank_rounds"], 1)
                self.assertEqual(audit["rerank_scored_candidate_count"], 384)
                self.assertEqual(audit["rerank_status"], "completed")
                self.assertEqual(audit["rerank_calls"][-1]["phase"], "final")
                self.assertTrue(all(call["input_tokens"] <= 110000 for call in audit["rerank_calls"]))
                scored = {i for call in audit["rerank_calls"] if call["round"] == 1 for i in call["candidate_indices"]}
                self.assertEqual(scored, set(range(384)))
                self.assertTrue(all(call["top_n"] == top_k for call in api.calls))

    async def test_short_384_candidates_use_one_global_request(self):
        api, audit = RankingAPI(), {}
        result = await rerank_candidates(api, Counter(), "研究", [f"{i:04d}" for i in range(384)], 8,
            layer="context", settings=SessionEvidenceSettings(), audit=audit)
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(len(api.calls[0]["documents"]), 384)
        self.assertEqual([row["index"] for row in result], list(range(383, 375, -1)))

    async def test_single_input_limits_reject_without_silent_truncation(self):
        for query, documents in (("长" * 4001, ["0001"]), ("研究", ["0001" + "长" * 4001])):
            api, audit = RankingAPI(), {}
            with self.assertRaises(RerankBudgetError):
                await rerank_candidates(api, Counter(), query, documents, 1, layer="context",
                    settings=SessionEvidenceSettings(), audit=audit)
            self.assertEqual(api.calls, [])
            self.assertEqual(audit["rerank_scored_candidate_count"], 0)

    async def test_request_budget_too_small_for_topk_is_explicit_failure(self):
        api, audit = RankingAPI(), {}
        with self.assertRaisesRegex(RerankBudgetError, "统一TopK"):
            await rerank_candidates(api, Counter(), "研究", [f"{i:04d}" + "长" * 500 for i in range(30)], 8,
                layer="context", settings=SessionEvidenceSettings(rerank_max_request_tokens=1000), audit=audit)
        self.assertEqual(api.calls, [])

    async def test_max_documents_limit_also_batches_and_preserves_global_indexes(self):
        api, audit = RankingAPI(), {}
        result = await rerank_candidates(api, Counter(), "研究", [f"{i:04d}" for i in range(90)], 5,
            layer="citation", settings=SessionEvidenceSettings(rerank_max_documents=20), audit=audit)
        self.assertEqual([row["index"] for row in result], [89, 88, 87, 86, 85])
        self.assertTrue(all(call["candidate_count"] <= 20 for call in audit["rerank_calls"]))

    async def test_invalid_indexes_fail_instead_of_binding_wrong_chunks(self):
        from unittest.mock import AsyncMock
        api, audit = RankingAPI(), {}
        api.rerank = AsyncMock(return_value=[{"index": 0}, {"index": 0}])
        with self.assertRaisesRegex(ValueError, "编号"):
            await rerank_candidates(api, Counter(), "研究", ["0000", "0001"], 2, layer="context",
                settings=SessionEvidenceSettings(), audit=audit)
        self.assertEqual(audit["rerank_calls"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
