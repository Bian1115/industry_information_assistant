"""隔离 PostgreSQL schema 验证摘要提交、原文完整性和并发；模型使用固定替身。"""

import os
from dataclasses import replace
import sys
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "app"))

from models.chat import ChatMessage, ChatSession
from models.user import User
from service.semantic_context_service import (
    SemanticContextError, SemanticContextService, execute_semantic_context_job,
    enqueue_semantic_context_if_due, prepare_semantic_context,
)
from service.session_summarization import SessionSummaryConfig
from test_会话滚动摘要 import FakeTokenCounter


@unittest.skipUnless(os.getenv("TEST_SESSION_SUMMARY_POSTGRES") == "1", "显式启用隔离PostgreSQL摘要测试")
class SessionSummaryPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dotenv import dotenv_values
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import URL
        from sqlalchemy.orm import sessionmaker
        values = {**dotenv_values(BACKEND / ".env"), **os.environ}
        url = URL.create(
            "postgresql+psycopg2", username=values.get("POSTGRES_USER", "postgres"),
            password=values.get("POSTGRES_PASSWORD", "postgres123"),
            host=values.get("POSTGRES_HOST", "localhost"), port=int(values.get("POSTGRES_PORT", "5432")),
            database=values.get("POSTGRES_DB", "industry_assistant"),
        )
        cls.schema = "test_session_summary_" + uuid.uuid4().hex
        cls.admin = create_engine(url)
        with cls.admin.begin() as db:
            db.execute(text(f'CREATE SCHEMA "{cls.schema}"'))
        cls.engine = create_engine(url, connect_args={"options": f"-csearch_path={cls.schema}"})
        cls.sessions = sessionmaker(bind=cls.engine)
        with cls.engine.begin() as db:
            User.__table__.create(db)
            ChatSession.__table__.create(db)
            ChatMessage.__table__.create(db)

    @classmethod
    def tearDownClass(cls):
        from sqlalchemy import text
        cls.engine.dispose()
        with cls.admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{cls.schema}" CASCADE'))
        cls.admin.dispose()

    def setUp(self):
        self.user_id, self.session_id = uuid.uuid4(), uuid.uuid4()
        self.message_ids = [uuid.uuid4() for _ in range(20)]
        self.config = SessionSummaryConfig(context_window=20000, trigger_fraction=0.02,
            max_output_tokens=128)
        self.prompts = []
        self.service = SemanticContextService(config=self.config, token_counter=FakeTokenCounter(),
            completion=lambda prompt: self.prompts.append(prompt) or "有效摘要")
        with self.sessions.begin() as db:
            suffix = uuid.uuid4().hex
            db.add(User(id=self.user_id, username=suffix, email=suffix + "@summary.test", hashed_password="test"))
            db.flush()
            db.add(ChatSession(id=self.session_id, user_id=self.user_id, title="会话摘要机制测试"))
            db.flush()
            for index, message_id in enumerate(self.message_ids):
                db.add(ChatMessage(id=message_id, session_id=self.session_id,
                    turn_id=uuid.uuid4(), role="user" if index % 2 == 0 else "assistant",
                    content=f"M{index:03d}" + "x" * 16,
                    created_at=datetime(2026, 10, 5, tzinfo=timezone.utc) + timedelta(seconds=index)))
        self.env_patch = patch.dict(os.environ, {"CONTEXT_COMPRESSION_ENABLED": "1", "SEMANTIC_CONTEXT_ENABLED": "true", "CONTEXT_ABLATION_CONFIG": ""})
        self.env_patch.start()
        self.service_patch = patch("service.semantic_context_service.get_semantic_context_service", return_value=self.service)
        self.service_patch.start()
        self.addCleanup(self.service_patch.stop)
        self.addCleanup(self.env_patch.stop)

    def prepare(self, **kwargs):
        with self.sessions() as db:
            return prepare_semantic_context(db, user_id=str(self.user_id), session_id=str(self.session_id), **kwargs)

    def test_commit_summary_and_cursor_together_without_deleting_raw_messages(self):
        result = self.prepare()
        self.assertEqual(result.summarized_messages, 10)
        self.assertEqual(len(result.messages), 11)
        self.assertTrue(result.metrics["triggered"])
        with self.sessions() as db:
            session = db.get(ChatSession, self.session_id)
            self.assertEqual(session.semantic_context_version, 1)
            self.assertEqual(session.semantic_context_last_message_id, self.message_ids[9])
            self.assertEqual(session.semantic_context_summary["schema_version"], 2)
            self.assertIn("after_tokens", session.semantic_context_summary["metrics"])
            rows = db.query(ChatMessage).filter(ChatMessage.session_id == self.session_id).order_by(ChatMessage.created_at).all()
            self.assertEqual(len(rows), 20)
            self.assertEqual([row.content for row in rows], [f"M{i:03d}" + "x" * 16 for i in range(20)])
        again = self.prepare()
        self.assertEqual(result.messages, again.messages)
        self.assertEqual(len(self.prompts), 1)

    def test_pending_current_query_counts_as_one_of_ten_kept_messages(self):
        result = self.prepare(current_query="新问题")
        self.assertEqual(result.summarized_messages, 11)
        self.assertEqual(len(result.messages), 10)  # 摘要＋9条历史，本轮query由调用方追加。
        self.assertEqual(result.metrics["kept_messages"], 10)
        self.assertNotIn("新问题", self.prompts[0])

    def test_failed_summary_never_advances_cursor_or_changes_state(self):
        self.service.completion = lambda _: (_ for _ in ()).throw(RuntimeError("模拟失败"))
        with self.assertRaisesRegex(SemanticContextError, "覆盖游标未更新"):
            self.prepare()
        with self.sessions() as db:
            session = db.get(ChatSession, self.session_id)
            self.assertEqual(session.semantic_context_version, 0)
            self.assertIsNone(session.semantic_context_summary)
            self.assertIsNone(session.semantic_context_last_message_id)
            self.assertEqual(db.query(ChatMessage).filter(ChatMessage.session_id == self.session_id).count(), 20)

    def test_old_v1_cursor_is_not_used_to_skip_original_messages(self):
        with self.sessions.begin() as db:
            session = db.get(ChatSession, self.session_id)
            session.semantic_context_summary = {"schema_version": 1, "overview": "不完整旧摘要"}
            session.semantic_context_version = 5
            session.semantic_context_last_message_id = self.message_ids[18]
        result = self.prepare()
        self.assertEqual(result.semantic_context_version, 6)
        self.assertIn("M000", self.prompts[0])
        self.assertNotIn("不完整旧摘要", self.prompts[0])

    def test_two_concurrent_requests_generate_only_one_summary(self):
        barrier, lock = threading.Barrier(2), threading.Lock()
        calls = []
        def completion(prompt):
            with lock:
                calls.append(prompt)
            time.sleep(0.05)
            return "有效摘要"
        self.service.completion = completion
        def request(_):
            barrier.wait(timeout=5)
            return self.prepare()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(request, range(2)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(results[0].messages, results[1].messages)
        with self.sessions() as db:
            self.assertEqual(db.get(ChatSession, self.session_id).semantic_context_version, 1)

    def test_old_queued_job_cannot_overwrite_rolling_summary(self):
        self.prepare()
        with self.sessions() as db:
            session = db.get(ChatSession, self.session_id)
            previous = dict(session.semantic_context_summary)
            execute_semantic_context_job(db, SimpleNamespace(id=uuid.uuid4(), user_id=self.user_id,
                session_id=self.session_id, payload={"message_ids": [str(self.message_ids[-1])]}))
            self.assertIsNone(enqueue_semantic_context_if_due(db, str(self.user_id), str(self.session_id)))
            db.commit()
            self.assertEqual(db.get(ChatSession, self.session_id).semantic_context_summary, previous)
            self.assertEqual(db.get(ChatSession, self.session_id).semantic_context_last_message_id, self.message_ids[9])

    def test_other_user_cannot_read_or_summarize_this_session(self):
        with self.sessions() as db:
            with self.assertRaises(LookupError):
                prepare_semantic_context(db, user_id=str(uuid.uuid4()), session_id=str(self.session_id))
        self.assertEqual(self.prompts, [])

    def test_refresh_mode_returns_all_uncovered_messages_without_model_calls(self):
        result = self.prepare(allow_summarization=False)
        self.assertEqual(len(result.messages), 20)
        self.assertFalse(result.metrics["triggered"])
        self.assertEqual(self.prompts, [])

    def test_excluded_current_turn_is_an_upper_boundary_not_a_cursor_hole(self):
        self.service.config = replace(self.config, trigger_fraction=0.005)
        with self.sessions() as db:
            turn_id = db.get(ChatMessage, self.message_ids[12]).turn_id
        result = self.prepare(exclude_turn_id=str(turn_id), current_query="当前问题")
        self.assertIn("M000", self.prompts[0])
        self.assertNotIn("M012", self.prompts[0])
        self.assertFalse(any("M019" in item["content"] for item in result.messages))
        with self.sessions() as db:
            # 前缀只有3条；不能跨过第13条当前消息去摘要它之后的其他轮。
            self.assertEqual(db.get(ChatSession, self.session_id).semantic_context_last_message_id, self.message_ids[2])

    def test_earlier_turn_retry_does_not_receive_future_summary_or_rewind_shared_cursor(self):
        self.prepare()
        with self.sessions() as db:
            turn_id = db.get(ChatMessage, self.message_ids[5]).turn_id
            before = dict(db.get(ChatSession, self.session_id).semantic_context_summary)
        result = self.prepare(exclude_turn_id=str(turn_id), current_query="重试早期问题")
        self.assertEqual(len(result.messages), 5)
        self.assertTrue(result.metrics["request_scoped_view"])
        self.assertFalse(any("有效摘要" in item["content"] for item in result.messages))
        with self.sessions() as db:
            self.assertEqual(db.get(ChatSession, self.session_id).semantic_context_summary, before)
            self.assertEqual(db.get(ChatSession, self.session_id).semantic_context_last_message_id, self.message_ids[9])

    def test_request_scoped_summary_is_not_persisted_over_newer_shared_state(self):
        self.prepare()
        with self.sessions.begin() as db:
            session = db.get(ChatSession, self.session_id)
            state = dict(session.semantic_context_summary)
            state["covered_until_message_id"] = str(self.message_ids[-1])
            state["covered_until_created_at"] = db.get(ChatMessage, self.message_ids[-1]).created_at.isoformat(timespec="microseconds")
            session.semantic_context_summary = state
            session.semantic_context_last_message_id = self.message_ids[-1]
            turn_id = db.get(ChatMessage, self.message_ids[15]).turn_id
        self.service.config = replace(self.config, trigger_fraction=0.01)
        result = self.prepare(exclude_turn_id=str(turn_id), current_query="早期问题")
        self.assertTrue(result.metrics["triggered"])
        self.assertFalse(result.metrics["persisted"])
        with self.sessions() as db:
            session = db.get(ChatSession, self.session_id)
            self.assertEqual(session.semantic_context_version, 1)
            self.assertEqual(session.semantic_context_last_message_id, self.message_ids[-1])
            self.assertEqual(session.semantic_context_summary, state)

    def test_existing_summary_alone_can_be_compressed_without_advancing_raw_cursor(self):
        self.prepare()
        with self.sessions.begin() as db:
            session = db.get(ChatSession, self.session_id)
            state = dict(session.semantic_context_summary)
            state["summary"] = "旧摘要" * 200
            session.semantic_context_summary = state
        result = self.prepare()
        self.assertTrue(result.metrics["triggered"])
        self.assertEqual(result.summarized_messages, 0)
        with self.sessions() as db:
            session = db.get(ChatSession, self.session_id)
            self.assertEqual(session.semantic_context_version, 2)
            self.assertEqual(session.semantic_context_last_message_id, self.message_ids[9])


if __name__ == "__main__":
    unittest.main()
