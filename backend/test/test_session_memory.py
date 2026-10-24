"""Redis 会话工作记忆的离线回归测试。"""

import json
import os
import sys
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from redis.exceptions import ConnectionError as RedisConnectionError
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from models.chat import ChatMessage
from models.chat import ChatSession
from models.user import User
from service.conversation_service import ConversationService
from service.session_memory_service import (
    SCHEMA_VERSION,
    RedisSessionMemoryService,
    WorkingMemoryContext,
)


class FakeTokenCounter:
    def count_text(self, text):
        return len(text or "")

    def count_messages(self, messages):
        return sum(len(item.get("content", "")) for item in messages)

    def encode(self, text):
        return list(text or "")

    def decode(self, tokens):
        return "".join(tokens)


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.command = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def watch(self, _key):
        return None

    def get(self, key):
        return self.client.get(key)

    def unwatch(self):
        return None

    def multi(self):
        return None

    def set(self, key, value, *, ex):
        self.command = (key, ex, value)

    def execute(self):
        if self.command:
            self.client.setex(*self.command)
        return [True]


class FakeRedis:
    def __init__(self, *, fail=False):
        self.values = {}
        self.ttls = {}
        self.fail = fail

    def get(self, key):
        if self.fail:
            raise RedisConnectionError("redis unavailable")
        return self.values.get(key)

    def setex(self, key, ttl, value):
        if self.fail:
            raise RedisConnectionError("redis unavailable")
        self.values[key] = value
        self.ttls[key] = ttl
        return True

    def delete(self, key):
        self.values.pop(key, None)
        return 1

    def ping(self):
        if self.fail:
            raise RedisConnectionError("redis unavailable")
        return True

    def pipeline(self):
        return FakePipeline(self)


class FakeQuery:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, *_):
        return self

    def order_by(self, *_):
        return self

    def limit(self, value):
        return FakeQuery(self.rows[:value])

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return list(self.rows)


class FakeDb:
    def __init__(self, rows, session=None):
        # RedisSessionMemoryService 查询时要求消息已经是时间倒序。
        self.rows = list(rows)
        self.session = session or SimpleNamespace(
            id="owned-session",
            semantic_context_summary=None,
            semantic_context_version=0,
            semantic_context_last_message_id=None,
        )

    def query(self, entity):
        if entity is ChatMessage:
            return FakeQuery(self.rows)
        return FakeQuery([self.session])


def make_message(index, role, content):
    return SimpleNamespace(
        id=uuid.UUID(int=index + 1),
        role=role,
        content=content,
        turn_id=uuid.UUID(int=(index // 2) + 100),
        created_at=datetime(2026, 8, 30, 8, 0) + timedelta(seconds=index),
    )


@patch.dict(os.environ, {"SEMANTIC_CONTEXT_ENABLED": "false", "CONTEXT_COMPRESSION_ENABLED": "1"})
class RedisSessionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.user_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.redis = FakeRedis()
        self.service = RedisSessionMemoryService(
            redis_client=self.redis,
            token_counter=FakeTokenCounter(),
            enabled=True,
            ttl_seconds=600,
            storage_token_limit=1000,
            storage_max_messages=20,
        )

    def test_cache_miss_rebuilds_and_second_read_hits(self):
        rows = [
            make_message(3, "assistant", "第二轮回答"),
            make_message(2, "user", "第二轮问题"),
            make_message(1, "assistant", "第一轮回答"),
            make_message(0, "user", "第一轮问题"),
        ]
        db = FakeDb(rows)

        first = self.service.get_context(
            db, user_id=self.user_id, session_id=self.session_id
        )
        second = self.service.get_context(
            db, user_id=self.user_id, session_id=self.session_id
        )

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.messages, second.messages)
        self.assertEqual(first.messages[0]["content"], "第一轮问题")
        key = self.service._key(self.user_id, self.session_id)
        self.assertEqual(self.redis.ttls[key], 600)

    def test_new_postgres_message_invalidates_old_cursor(self):
        initial_rows = [
            make_message(1, "assistant", "旧回答"),
            make_message(0, "user", "旧问题"),
        ]
        self.service.get_context(
            FakeDb(initial_rows), user_id=self.user_id, session_id=self.session_id
        )
        new_rows = [
            make_message(3, "assistant", "新回答"),
            make_message(2, "user", "新问题"),
            *initial_rows,
        ]

        rebuilt = self.service.get_context(
            FakeDb(new_rows), user_id=self.user_id, session_id=self.session_id
        )

        self.assertFalse(rebuilt.cache_hit)
        self.assertEqual(rebuilt.messages[-1]["content"], "新回答")

    def test_legacy_excerpt_cache_is_rebuilt_and_replaced(self):
        rows = [
            make_message(1, "assistant", "最新回答"),
            make_message(0, "user", "最新问题"),
        ]
        key = self.service._key(self.user_id, self.session_id)
        legacy = {
            "schema_version": 2,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "last_message_id": str(rows[0].id),
            # 即使旧格式游标较新，也不能阻止新格式写回。
            "last_message_created_at": "2099-01-01T00:00:00.000000",
            "semantic_context_version": 0,
            "messages": [{"role": "system", "content": "[较早对话摘要]\n旧规则摘录"}],
            "summarized_messages": 10,
        }
        self.redis.setex(key, 600, json.dumps(legacy))

        rebuilt = self.service.get_context(
            FakeDb(rows), user_id=self.user_id, session_id=self.session_id,
        )
        cached = self.service.get_context(
            FakeDb(rows), user_id=self.user_id, session_id=self.session_id,
        )

        self.assertFalse(rebuilt.cache_hit)
        self.assertEqual([item["role"] for item in rebuilt.messages], ["user", "assistant"])
        self.assertEqual(rebuilt.messages[-1]["content"], "最新回答")
        self.assertEqual(rebuilt.summarized_messages, 0)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(json.loads(self.redis.get(key))["schema_version"], SCHEMA_VERSION)

    def test_cache_miss_and_hit_do_not_summarize_discarded_turns(self):
        rows = [
            make_message(index, "user" if index % 2 == 0 else "assistant", f"原文{index}")
            for index in reversed(range(30))
        ]
        db = FakeDb(rows)
        first = self.service.get_context(db, user_id=self.user_id, session_id=self.session_id)
        second = self.service.get_context(db, user_id=self.user_id, session_id=self.session_id)

        self.assertEqual([item["content"] for item in first.messages], [f"原文{i}" for i in range(10, 30)])
        self.assertEqual(first.summarized_messages, 0)
        self.assertEqual(first.messages, second.messages)
        self.assertTrue(second.cache_hit)

    def test_redis_failure_falls_back_to_postgres(self):
        service = RedisSessionMemoryService(
            redis_client=FakeRedis(fail=True),
            token_counter=FakeTokenCounter(),
            enabled=True,
            ttl_seconds=600,
            storage_token_limit=1000,
            storage_max_messages=20,
        )
        result = service.get_context(
            FakeDb([make_message(0, "user", "仍然可以回答")]),
            user_id=self.user_id,
            session_id=self.session_id,
        )

        self.assertFalse(result.cache_hit)
        self.assertEqual(result.messages[0]["content"], "仍然可以回答")

    def test_older_rebuild_cannot_overwrite_newer_payload(self):
        key = self.service._key(self.user_id, self.session_id)
        newer = {
            "schema_version": SCHEMA_VERSION,
            "messages": [],
            "last_message_created_at": "2026-08-30T10:00:00.000000",
            "last_message_id": "bbbb",
        }
        older = {
            "schema_version": SCHEMA_VERSION,
            "messages": [],
            "last_message_created_at": "2026-08-30T09:00:00.000000",
            "last_message_id": "aaaa",
        }
        self.redis.setex(key, 600, json.dumps(newer))

        self.assertTrue(self.service._write_payload(key, older))
        self.assertEqual(json.loads(self.redis.get(key)), newer)

    def test_invalidate_removes_session_key(self):
        key = self.service._key(self.user_id, self.session_id)
        self.redis.setex(key, 600, "{}")

        self.service.invalidate(user_id=self.user_id, session_id=self.session_id)

        self.assertIsNone(self.redis.get(key))

    def test_semantic_summary_is_injected_and_version_invalidates_cache(self):
        session = SimpleNamespace(
            id=uuid.UUID(self.session_id),
            semantic_context_summary={
                "schema_version": 1,
                "version": 1,
                "prompt_version": "semantic-context-v1",
                "model": "fake-model",
                "overview": "用户正在分析储能市场",
                "items": [{
                    "id": "ctx-1",
                    "category": "constraint",
                    "text": "不使用预测值",
                    "source_message_ids": ["m-old"],
                    "evidence_quote": "不要预测值",
                    "source_role": "user",
                    "status": "active",
                    "superseded_by": None,
                    "confidence": 1.0,
                }],
                "covered_until_message_id": None,
                "updated_at": "2026-08-30T12:00:00",
            },
            semantic_context_version=1,
            semantic_context_last_message_id=None,
        )
        db = FakeDb([make_message(0, "user", "继续分析")], session=session)

        first = self.service.get_context(
            db, user_id=self.user_id, session_id=self.session_id
        )
        cached = self.service.get_context(
            db, user_id=self.user_id, session_id=self.session_id
        )
        session.semantic_context_version = 2
        session.semantic_context_summary["version"] = 2
        session.semantic_context_summary["items"][0]["text"] = "不使用任何预测值"
        rebuilt = self.service.get_context(
            db, user_id=self.user_id, session_id=self.session_id
        )

        self.assertFalse(first.cache_hit)
        self.assertTrue(cached.cache_hit)
        self.assertFalse(rebuilt.cache_hit)
        self.assertIn("不使用预测值", first.messages[0]["content"])
        self.assertIn("不使用任何预测值", rebuilt.messages[0]["content"])
        self.assertEqual(rebuilt.semantic_context_version, 2)


@unittest.skipUnless(
    os.getenv("RUN_REDIS_INTEGRATION_TESTS") == "1",
    "set RUN_REDIS_INTEGRATION_TESTS=1 to exercise PostgreSQL + Redis",
)
@patch.dict(os.environ, {"SEMANTIC_CONTEXT_ENABLED": "false", "CONTEXT_COMPRESSION_ENABLED": "1"})
class PostgreSQLRedisSessionMemoryTests(unittest.TestCase):
    def setUp(self):
        from core.database import SessionLocal

        self.db = SessionLocal()
        suffix = uuid.uuid4().hex[:10]
        self.user_id = uuid.uuid4()
        self.session_id = uuid.uuid4()
        self.turn_id = uuid.uuid4()
        self.db.add_all([
            User(
                id=self.user_id,
                username=f"short_memory_{suffix}",
                email=f"{suffix}@short-memory.test",
                hashed_password="test",
            ),
            ChatSession(
                id=self.session_id,
                user_id=self.user_id,
                title="短期记忆集成测试",
            ),
        ])
        self.db.commit()
        self.conversations = ConversationService()
        self.memory = self.conversations.session_memory
        if not self.memory.ping():
            self.skipTest("Redis is unavailable")

    def tearDown(self):
        self.memory.invalidate(
            user_id=str(self.user_id), session_id=str(self.session_id)
        )
        self.db.rollback()
        user = self.db.query(User).filter(User.id == self.user_id).first()
        if user:
            self.db.delete(user)
            self.db.commit()
        self.db.close()

    def test_completed_turn_refreshes_redis_and_is_reused(self):
        session = self.conversations.require_session(
            self.db, str(self.session_id), str(self.user_id)
        )
        self.conversations.save_user_message(
            self.db, session, str(self.turn_id), "分析储能市场"
        )
        assistant = self.conversations.save_assistant_message(
            self.db,
            str(self.session_id),
            str(self.user_id),
            str(self.turn_id),
            "先明确市场口径。",
        )

        context = self.memory.get_context(
            self.db,
            user_id=str(self.user_id),
            session_id=str(self.session_id),
        )
        key = self.memory._key(str(self.user_id), str(self.session_id))

        self.assertTrue(context.cache_hit)
        self.assertEqual(context.last_message_id, str(assistant.id))
        self.assertEqual(context.messages[-1]["content"], "先明确市场口径。")
        self.assertGreater(self.memory.redis.ttl(key), 0)

        prompt_history = self.conversations.get_prompt_history(
            self.db,
            str(self.session_id),
            user_id=str(self.user_id),
        )
        self.assertEqual(prompt_history[-1]["content"], "先明确市场口径。")


class DeepResearchWorkingMemoryTests(unittest.TestCase):
    def test_research_history_reads_the_same_redis_working_memory(self):
        from unittest.mock import Mock, patch
        from service.research_context_service import ResearchContextService

        expected = WorkingMemoryContext(
            messages=[{"role": "user", "content": "上一轮问题"}],
            original_tokens=6,
            compacted_tokens=6,
            summarized_messages=0,
            pruned_tool_results=0,
            cache_hit=True,
            last_message_id="message-1",
        )
        fake_db = Mock()
        fake_db.close = Mock()
        memory = Mock()
        memory.get_context.return_value = expected
        service = ResearchContextService.__new__(ResearchContextService)
        service.session_memory = memory
        service.history_token_limit = 3000

        with patch(
            "service.research_context_service.SessionLocal",
            return_value=fake_db,
        ):
            result = service._load_history(
                user_id=str(uuid.uuid4()),
                session_id=str(uuid.uuid4()),
                current_turn_id=str(uuid.uuid4()),
            )

        self.assertIs(result, expected)
        self.assertEqual(memory.get_context.call_args.kwargs["token_limit"], 3000)
        self.assertEqual(memory.get_context.call_args.kwargs["max_messages"], 12)
        fake_db.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
