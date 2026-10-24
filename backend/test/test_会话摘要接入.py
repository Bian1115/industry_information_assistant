"""会话摘要的缓存与业务入口机制测试；数据库、模型和 Redis 均使用替身。"""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import UUID
from datetime import datetime, timedelta, timezone

from redis.exceptions import ConnectionError as RedisConnectionError


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from models.chat import ChatMessage
from service.conversation_service import ConversationService
from service.research_context_service import ResearchContextService
from service.semantic_context_service import PreparedSemanticContext, SemanticContextError
from service.session_memory_service import RedisSessionMemoryService, WorkingMemoryContext


class CharCounter:
    """用字符数替代 token，只验证接入层是否错误使用旧预算。"""

    def count_text(self, text):
        return len(text or "")

    def count_messages(self, messages):
        return sum(self.count_text(message["content"]) for message in messages)


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.command = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def watch(self, _key):
        pass

    def unwatch(self):
        pass

    def get(self, key):
        return self.redis.get(key)

    def multi(self):
        pass

    def set(self, key, value, *, ex):
        self.command = (key, value, ex)

    def execute(self):
        if self.command:
            key, value, ttl = self.command
            self.redis.values[key] = value
            self.redis.ttls[key] = ttl
        return [True]


class FakeRedis:
    def __init__(self, unavailable=False):
        self.values = {}
        self.ttls = {}
        self.unavailable = unavailable

    def get(self, key):
        if self.unavailable:
            raise RedisConnectionError("测试 Redis 不可用")
        return self.values.get(key)

    def pipeline(self):
        return FakePipeline(self)


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *_):
        return self

    def order_by(self, *_):
        return self

    def first(self):
        return self.rows[0] if self.rows else None


class FakeDb:
    def __init__(self, message_id):
        self.latest = SimpleNamespace(
            id=UUID(message_id),
            created_at=datetime(2026, 10, 5, tzinfo=timezone.utc) + timedelta(seconds=10),
        )
        self.session = SimpleNamespace(semantic_context_version=1)

    def query(self, entity):
        return FakeQuery([self.latest] if entity is ChatMessage else [self.session])


def full_history():
    # 超过旧 5000 / 3000 token 及 20 / 12 条门槛，内容必须完整透传。
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"原文 {index}：" + "证据" * 220}
        for index in range(32)
    ]


def prepared_context(*, fingerprint="配置甲", version=1, messages=None):
    messages = full_history() if messages is None else messages
    tokens = CharCounter().count_messages(messages)
    return PreparedSemanticContext(
        messages=messages,
        original_tokens=tokens,
        compacted_tokens=tokens,
        summarized_messages=0,
        semantic_context_version=version,
        last_message_id=str(UUID(int=33)),
        metrics={"triggered": False, "keep_messages": 10, "history_tokens": tokens},
        config_fingerprint=fingerprint,
    )


class SessionSummaryIntegrationTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "SEMANTIC_CONTEXT_ENABLED": "true",
            "CONTEXT_COMPRESSION_ENABLED": "1",
            "CONTEXT_ABLATION_CONFIG": "",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.user_id = str(UUID(int=100))
        self.session_id = str(UUID(int=200))
        self.redis = FakeRedis()
        self.memory = RedisSessionMemoryService(
            redis_client=self.redis,
            token_counter=CharCounter(),
            enabled=True,
            ttl_seconds=600,
            storage_token_limit=5000,
            storage_max_messages=20,
        )
        self.prepared = prepared_context()
        self.db = FakeDb(self.prepared.last_message_id)

    def read(self, **kwargs):
        return self.memory.get_context(
            self.db, user_id=self.user_id, session_id=self.session_id, **kwargs,
        )

    def test_complete_history_is_cached_without_legacy_budget_and_hit_still_prepares(self):
        with patch("service.semantic_context_service.prepare_semantic_context", return_value=self.prepared) as prepare, \
                patch("service.session_memory_service.compact_messages", side_effect=AssertionError("不应执行旧裁剪")):
            first = self.read(token_limit=1, max_messages=1, current_query="补充政策证据")
            second = self.read(token_limit=1, max_messages=1, current_query="补充政策证据")

        self.assertGreater(self.prepared.compacted_tokens, 5000)
        self.assertGreater(len(first.messages), 20)
        self.assertEqual(first.messages, self.prepared.messages)
        self.assertEqual(second.messages, self.prepared.messages)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(prepare.call_args.kwargs["current_query"], "补充政策证据")
        payload = json.loads(self.redis.values[self.memory._key(self.user_id, self.session_id)])
        self.assertEqual(payload["messages"], self.prepared.messages)
        self.assertEqual(payload["config_fingerprint"], self.prepared.config_fingerprint)
        self.assertEqual(first.summary_metrics, self.prepared.metrics)

    def test_configuration_change_invalidates_cache_even_with_same_messages_and_cursor(self):
        changed = prepared_context(fingerprint="配置乙")
        with patch("service.semantic_context_service.prepare_semantic_context", side_effect=[self.prepared, changed, changed]):
            first = self.read()
            changed_read = self.read()
            cached_read = self.read()
        self.assertFalse(first.cache_hit)
        self.assertFalse(changed_read.cache_hit)
        self.assertTrue(cached_read.cache_hit)
        self.assertEqual(changed_read.config_fingerprint, "配置乙")

    def test_excluded_turn_changes_cache_identity(self):
        first_turn = str(UUID(int=300))
        second_turn = str(UUID(int=301))
        with patch("service.semantic_context_service.prepare_semantic_context", return_value=self.prepared) as prepare:
            first = self.read(exclude_turn_id=first_turn)
            second = self.read(exclude_turn_id=second_turn)
            repeat = self.read(exclude_turn_id=second_turn)
        self.assertFalse(first.cache_hit)
        self.assertFalse(second.cache_hit)
        self.assertTrue(repeat.cache_hit)
        self.assertEqual(prepare.call_args.kwargs["exclude_turn_id"], second_turn)

    def test_summary_version_change_invalidates_cache(self):
        next_version = prepared_context(version=2)
        with patch("service.semantic_context_service.prepare_semantic_context", side_effect=[self.prepared, next_version]):
            self.read()
            result = self.read()
        self.assertFalse(result.cache_hit)
        self.assertEqual(result.semantic_context_version, 2)

    def test_redis_failure_keeps_complete_prepared_history(self):
        self.memory.redis = FakeRedis(unavailable=True)
        with patch("service.semantic_context_service.prepare_semantic_context", return_value=self.prepared):
            result = self.read()
        self.assertEqual(result.messages, self.prepared.messages)
        self.assertFalse(result.cache_hit)

    def test_assistant_cache_refresh_disables_summary_generation(self):
        with patch("service.semantic_context_service.prepare_semantic_context", return_value=self.prepared) as prepare:
            self.memory.refresh(self.db, user_id=self.user_id, session_id=self.session_id)
        prepare.assert_called_once()
        self.assertFalse(prepare.call_args.kwargs["allow_summarization"])

    def test_conversation_passes_current_query_to_session_summary(self):
        service = ConversationService.__new__(ConversationService)
        service.require_session = Mock()
        service.session_memory = Mock()
        service.session_memory.get_context.return_value = SimpleNamespace(messages=self.prepared.messages)
        service.token_limit = 5000
        service.max_messages = 20
        result = service.get_prompt_history(
            Mock(), self.session_id, user_id=self.user_id,
            exclude_turn_id="当前轮", current_query="补充政策证据",
        )
        self.assertEqual(result, self.prepared.messages)
        self.assertEqual(service.session_memory.get_context.call_args.kwargs["current_query"], "补充政策证据")

    def test_conversation_summary_error_propagates_without_old_postgres_fallback(self):
        service = ConversationService.__new__(ConversationService)
        service.require_session = Mock()
        service.session_memory = Mock()
        service.session_memory.get_context.side_effect = SemanticContextError("摘要失败")
        service.token_limit = 5000
        service.max_messages = 20
        db = Mock()
        with self.assertRaisesRegex(SemanticContextError, "摘要失败"):
            service.get_prompt_history(db, self.session_id, user_id=self.user_id)
        db.query.assert_not_called()

    def test_research_history_load_passes_current_query_and_closes_database(self):
        service = ResearchContextService.__new__(ResearchContextService)
        service.session_memory = Mock()
        service.session_memory.get_context.return_value = self.prepared
        service.history_token_limit = 3000
        db = Mock()
        with patch("service.research_context_service.SessionLocal", return_value=db):
            result = service._load_history(
                user_id=self.user_id, session_id=self.session_id,
                current_turn_id="当前轮", current_query="补充政策证据",
            )
        self.assertIs(result, self.prepared)
        self.assertEqual(service.session_memory.get_context.call_args.kwargs["current_query"], "补充政策证据")
        self.assertEqual(service.session_memory.get_context.call_args.kwargs["exclude_turn_id"], "当前轮")
        db.close.assert_called_once_with()

    def test_research_build_keeps_full_history_and_exposes_summary_metrics(self):
        service = ResearchContextService.__new__(ResearchContextService)
        service.token_counter = CharCounter()
        service.history_token_limit = 3000
        service.memory_token_limit = 2000
        working = WorkingMemoryContext(
            messages=self.prepared.messages,
            original_tokens=self.prepared.original_tokens,
            compacted_tokens=self.prepared.compacted_tokens,
            summarized_messages=0,
            pruned_tool_results=0,
            cache_hit=True,
            last_message_id=self.prepared.last_message_id,
            semantic_context_version=1,
            summary_metrics=self.prepared.metrics,
            config_fingerprint=self.prepared.config_fingerprint,
        )
        service._load_history = Mock(return_value=working)
        service._load_memories = Mock(return_value=[])
        service._fit_memories = Mock(return_value=[])
        with patch("service.session_task_service.load_artifact_candidates", return_value=[]), \
                patch("service.research_context_service.compact_messages", side_effect=AssertionError("不应执行旧裁剪")):
            result = service.build(
                user_id=self.user_id, session_id=self.session_id,
                current_turn_id="当前轮", current_query="分析政策影响", resolve_query=False,
            )
        self.assertEqual(result["conversation_context"], self.prepared.messages)
        self.assertGreater(result["token_counts"]["conversation"], 3000)
        self.assertGreater(len(result["conversation_context"]), 12)
        self.assertEqual(result["session_summary_metrics"], self.prepared.metrics)
        self.assertEqual(result["session_summary_config_fingerprint"], self.prepared.config_fingerprint)
        self.assertEqual(service._load_history.call_args.kwargs["current_query"], "分析政策影响")


if __name__ == "__main__":
    unittest.main()
