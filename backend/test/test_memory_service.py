"""长期记忆原子事实、来源校验和原始证据回填的离线测试。"""

import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from service.memory_contracts import MemoryHit
from service.memory_service import (
    MemoryInfrastructureError,
    MemoryProviderQuotaError,
    MemoryService,
    _fact_extraction_batches,
    _normalize_atomic_facts,
    _parse_summary_json,
    _select_diverse_hits,
    _select_relevant_raw_messages,
)


def message(message_id, content, created_at=None, role="user"):
    return SimpleNamespace(
        message_id=message_id,
        id=message_id,
        content=content,
        created_at=created_at or datetime(2024, 1, 1, 8, 0),
        role=role,
    )


def hit(fragment_id, session_id, memory_type="atomic_fact", content=None, metadata=None):
    return MemoryHit(
        memory_id=f"00000000-0000-0000-0000-{int(fragment_id):012d}",
        fragment_id=str(fragment_id),
        memory_type=memory_type,
        content=content or f"fact {fragment_id}",
        score=1.0 - int(fragment_id) * 0.01,
        session_id=session_id,
        metadata=metadata or {},
    )


class SummaryParsingTests(unittest.TestCase):
    def test_legacy_summary_defaults_to_empty_atomic_facts(self):
        result = _parse_summary_json('{"summary":"旧版摘要"}')
        self.assertEqual(result["atomic_facts"], [])
        self.assertEqual(result["user_preferences"], {})

    def test_atomic_facts_require_a_real_source_message(self):
        messages = [
            message(
                "m1",
                "我刚开始带领4名工程师。",
                datetime(2023, 5, 11, 2, 2),
            )
        ]
        facts = [
            {
                "fact": "用户刚开始带领4名工程师",
                "subject": "用户",
                "predicate": "带领工程师数量",
                "value": 4,
                "source_message_ids": ["hallucinated-id"],
                "exact_quote": "我刚开始带领4名工程师。",
            },
            {
                "fact": "用户住在火星",
                "source_message_ids": ["hallucinated-id"],
            },
        ]

        result = _normalize_atomic_facts(facts, messages)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_message_ids"], ["m1"])
        self.assertEqual(result[0]["source_times"], ["2023-05-11T02:02:00"])
        self.assertEqual(result[0]["exact_quote"], "我刚开始带领4名工程师。")

    def test_summary_prompt_contains_message_ids_and_times(self):
        service = MemoryService()
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "summary": "用户带领工程师",
                "key_insights": [],
                "user_preferences": {},
                "topics": [],
                "atomic_facts": [{
                    "fact": "用户带领4名工程师",
                    "source_message_ids": ["m1"],
                    "exact_quote": "带领4名工程师",
                }],
            }, ensure_ascii=False)))]
        )
        create = Mock(return_value=response)
        service._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        result = service.summarize_conversation([
            message("m1", "我带领4名工程师", datetime(2023, 5, 11, 2, 2))
        ])

        request = create.call_args.kwargs
        prompt = request["messages"][1]["content"]
        self.assertIn("message_id=m1", prompt)
        self.assertIn("2023-05-11T02:02:00", prompt)
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(result["atomic_facts"][0]["source_message_ids"], ["m1"])

    def test_fact_extraction_splits_oversized_messages_into_bounded_batches(self):
        batches = _fact_extraction_batches(
            [message("m1", "A" * 7000)],
            max_chars=2000,
        )

        self.assertGreater(len(batches), 1)
        self.assertTrue(all(len(batch) <= 2000 for batch in batches))
        self.assertTrue(all("message_id=m1" in batch for batch in batches))

    def test_claude_summary_retries_429_and_extracts_facts_separately(self):
        summary_json = json.dumps({
            "summary": "用户带领工程师",
            "key_insights": [],
            "user_preferences": {},
            "topics": [],
        }, ensure_ascii=False)
        fact_json = json.dumps({
            "atomic_facts": [{
                "fact": "用户带领5名工程师",
                "source_message_ids": ["m1"],
                "exact_quote": "带领5名工程师",
            }],
        }, ensure_ascii=False)

        with (
            patch.dict(os.environ, {
                "MEMORY_SUMMARY_PROVIDER": "claude-relay",
                "CLAUDE_RELAY_API_KEY": "test-key",
                "CLAUDE_RELAY_BASE_URL": "https://claude.test/v1",
                "MEMORY_CLAUDE_REQUEST_ATTEMPTS": "2",
            }),
            patch(
                "service.memory_service.claude_messages_completion",
                side_effect=[
                    RuntimeError("Claude Messages HTTP 429: concurrency limit exceeded"),
                    summary_json,
                    fact_json,
                ],
            ) as completion,
            patch("service.memory_service.time.sleep") as sleep,
        ):
            result = MemoryService().summarize_conversation([
                message("m1", "我现在带领5名工程师")
            ])

        self.assertEqual(completion.call_count, 3)
        sleep.assert_called_once_with(2)
        self.assertEqual(result["summary_provider"], "claude-relay")
        self.assertEqual(result["fact_batch_count"], 1)
        self.assertEqual(result["atomic_facts"][0]["source_message_ids"], ["m1"])

    def test_claude_daily_limit_is_not_retried(self):
        with (
            patch.dict(os.environ, {
                "MEMORY_SUMMARY_PROVIDER": "claude-relay",
                "CLAUDE_RELAY_API_KEY": "test-key",
                "CLAUDE_RELAY_BASE_URL": "https://claude.test/v1",
                "MEMORY_CLAUDE_REQUEST_ATTEMPTS": "5",
            }),
            patch(
                "service.memory_service.claude_messages_completion",
                side_effect=RuntimeError("HTTP 429: DAILY_LIMIT_EXCEEDED"),
            ) as completion,
            patch("service.memory_service.time.sleep") as sleep,
        ):
            with self.assertRaises(MemoryProviderQuotaError):
                MemoryService().summarize_conversation([message("m1", "hello")])

        self.assertEqual(completion.call_count, 1)
        sleep.assert_not_called()

    def test_atomic_fact_is_written_as_an_independent_vector_document(self):
        service = MemoryService()
        service._milvus = object()
        collection = Mock()
        summary_data = {
            "summary": "用户带领一个工程团队",
            "key_insights": [],
            "topics": [],
            "user_preferences": {},
            "atomic_facts": [{
                "fact": "用户带领4名工程师",
                "value": 4,
                "source_message_ids": ["m1"],
                "source_times": ["2023-05-11T02:02:00"],
            }],
        }

        with (
            patch(
                "service.memory_service.generate_embedding",
                return_value=[[0.1], [0.2]],
            ) as embedding,
            patch("pymilvus.Collection", return_value=collection),
        ):
            ids = service._store_memory_vectors(
                "memory-1", "user-1", "session-1", summary_data
            )

        self.assertIn("memory-1_atomic_fact_0", ids)
        embedding.assert_called_once_with([
            "用户带领一个工程团队",
            "用户带领4名工程师",
        ])
        inserted = collection.upsert.call_args.args[0]
        fact_index = inserted[0].index("memory-1_atomic_fact_0")
        self.assertEqual(inserted[3][fact_index], "atomic_fact")
        metadata = json.loads(inserted[5][fact_index])
        self.assertEqual(metadata["source_message_ids"], ["m1"])


class RetrievalCompositionTests(unittest.TestCase):
    def test_query_embedding_failure_is_an_infrastructure_error(self):
        service = MemoryService()
        service._milvus = object()
        with (
            patch("service.memory_service.generate_embedding", return_value=None),
            patch("pymilvus.utility.has_collection", return_value=True),
        ):
            with self.assertRaisesRegex(
                MemoryInfrastructureError, "查询 embedding 生成失败"
            ):
                service.retrieve_memories("user-1", "team size", 5)

    def test_vector_search_pushes_memory_type_filter_into_milvus(self):
        service = MemoryService()
        service._milvus = object()
        collection = Mock()
        collection.search.return_value = []
        with (
            patch("service.memory_service.generate_embedding", return_value=[0.1]),
            patch("pymilvus.utility.has_collection", return_value=True),
            patch("pymilvus.Collection", return_value=collection),
        ):
            service.retrieve_memories(
                "user-1",
                "team size",
                5,
                memory_types=("atomic_fact", "preference"),
            )

        expression = collection.search.call_args.kwargs["expr"]
        self.assertIn('user_id == "user-1"', expression)
        self.assertIn('memory_type in ["atomic_fact", "preference"]', expression)

    def test_diverse_hits_cover_sessions_before_reusing_one(self):
        selected = _select_diverse_hits([
            hit(1, "session-a"),
            hit(2, "session-a"),
            hit(3, "session-b"),
        ], 2)
        self.assertEqual([item.fragment_id for item in selected], ["1", "3"])

    def test_raw_selection_prioritizes_fact_source_then_orders_by_time(self):
        messages = [
            message("m1", "我刚开始带4名工程师", datetime(2023, 5, 11)),
            message("m2", "我现在带5名工程师", datetime(2023, 10, 24)),
            message("m3", "无关的旅行计划", datetime(2023, 11, 1)),
        ]
        selected = _select_relevant_raw_messages(
            "刚开始和现在分别带几名工程师",
            ["用户带领工程师数量发生变化"],
            messages,
            preferred_message_ids=["m2", "m1"],
            max_messages=2,
        )
        self.assertEqual([item.message_id for item in selected], ["m1", "m2"])

    def test_context_keeps_fact_metadata_and_appends_raw_evidence(self):
        service = MemoryService()
        candidates = [
            hit(1, "session-a", metadata={
                "source_message_ids": ["m1"],
                "source_times": ["2023-05-11T02:02:00"],
            }),
            hit(2, "session-b", memory_type="preference", content="偏好健康晚餐"),
        ]
        provider = Mock()
        provider.search.return_value = candidates
        raw = [{
            "message_id": "m1",
            "role": "user",
            "created_at": "2023-05-11T02:02:00",
            "content": "我刚开始带4名工程师",
        }]

        with (
            patch("service.memory_service.get_memory_provider", return_value=provider),
            patch.object(service, "_load_raw_evidence", return_value=raw),
        ):
            context = service.build_memory_context("user-id", "我带几个人", 2)

        self.assertIn("原子事实；来源时间=2023-05-11T02:02:00；来源消息=m1", context)
        self.assertIn("用户偏好: 偏好健康晚餐", context)
        self.assertIn("[原始证据]", context)
        self.assertIn("我刚开始带4名工程师", context)


if __name__ == "__main__":
    unittest.main()
