"""会话 TTL 的边界、消息续期、访问控制、级联清理与调度回归。"""

import asyncio
import importlib.util
import os
import sys
import tempfile
import unittest
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from sqlalchemy import JSON, MetaData, create_engine, event, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import sessionmaker

from core.time_utils import session_expiry_time, utc_now
from models.chat import ChatAttachment, ChatMessage, ChatSession, LongTermMemory, MemoryJob
from models.user import User
from models.session_evidence import SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk, SessionEvidencePurge
from models.research import ResearchRun
from service.conversation_service import ConversationNotFoundError, ConversationService
from service.session_memory_service import RedisSessionMemoryService
from service.session_retention_service import cleanup_expired_sessions

MODELS = (User, ChatSession, ChatMessage, ChatAttachment, LongTermMemory, MemoryJob,
          SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk, SessionEvidencePurge, ResearchRun)


class SessionExpiryTests(unittest.TestCase):
    def setUp(self):
        self.now = utc_now()
        self.engine = self.create_engine()
        metadata = MetaData()
        for model in MODELS:
            table = model.__table__.to_metadata(metadata)
            if self.engine.dialect.name == "sqlite":
                for column in table.columns:
                    if isinstance(column.type, (JSONB, ARRAY)):
                        column.type = JSON()
                for index in table.indexes:
                    where = index.dialect_options["postgresql"].get("where")
                    if where is not None:
                        index.dialect_options["sqlite"]["where"] = where
        metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, autoflush=False)
        self.db = self.factory()
        self.user = User(id=uuid.uuid4(), username=str(uuid.uuid4()),
                         email=f"{uuid.uuid4()}@example.invalid", hashed_password="测试")
        self.db.add(self.user)
        self.db.commit()
        self.session = self.new_session()
        self.conversation = ConversationService.__new__(ConversationService)
        self.conversation.session_memory = Mock()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.memory = Mock()
        for mocker in (
            patch("service.session_retention_service.SessionLocal", self.factory),
            patch("service.session_memory_service.get_session_memory_service", return_value=self.memory),
            patch("service.memory_service.enqueue_memory_if_due"),
            patch("service.semantic_context_service.enqueue_semantic_context_if_due"),
        ):
            mocker.start()
            self.addCleanup(mocker.stop)

    def create_engine(self):
        engine = create_engine("sqlite:///:memory:")
        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, record):
            connection.execute("PRAGMA foreign_keys=ON")
        return engine

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def new_session(self, **kwargs):
        session = ChatSession(id=uuid.uuid4(), user_id=self.user.id, title="测试会话", **kwargs)
        self.db.add(session)
        self.db.commit()
        return session

    def expire(self):
        self.session.expires_at = self.now
        self.db.commit()

    def cleanup(self, **kwargs):
        removed = cleanup_expired_sessions(now=self.now, **kwargs)
        self.db.expire_all()
        return removed

    def test_new_session_has_24_hour_ttl(self):
        self.assertAlmostEqual((self.session.expires_at - self.session.created_at).total_seconds(), 86400, delta=1)

    def test_new_user_message_renews_ttl(self):
        self.session.expires_at = self.now + timedelta(minutes=1)
        self.db.commit()
        self.conversation.save_user_message(self.db, self.session, str(uuid.uuid4()), "新消息")
        self.assertEqual(self.session.expires_at, session_expiry_time(self.session.updated_at))

    def test_duplicate_turn_does_not_renew_ttl(self):
        turn = str(uuid.uuid4())
        first = self.conversation.save_user_message(self.db, self.session, turn, "原消息")
        expiry = self.session.expires_at
        duplicate = self.conversation.save_user_message(self.db, self.session, turn, "重试")
        self.assertEqual(first.id, duplicate.id)
        self.assertEqual(self.session.expires_at, expiry)

    def test_assistant_message_renews_in_same_transaction(self):
        original_expiry = self.now + timedelta(minutes=1)
        self.session.expires_at = original_expiry
        self.db.commit()
        self.conversation.stage_assistant_message(
            self.db, session_id=str(self.session.id), user_id=str(self.user.id),
            turn_id=str(uuid.uuid4()), content="助手消息")
        self.assertEqual(self.session.expires_at, session_expiry_time(self.session.updated_at))
        self.db.rollback()
        self.assertEqual(self.session.expires_at, original_expiry)
        self.assertEqual(self.db.query(ChatMessage).count(), 0)

    def test_assistant_save_commits_renewal(self):
        self.conversation.save_assistant_message(
            self.db, str(self.session.id), str(self.user.id), str(uuid.uuid4()), "已交付")
        self.db.expire_all()
        self.assertEqual(self.session.expires_at, session_expiry_time(self.session.updated_at))
        self.assertEqual(self.db.query(ChatMessage).count(), 1)

    def test_expired_session_cannot_read_or_resume_writes(self):
        self.expire()
        with self.assertRaises(ConversationNotFoundError):
            self.conversation.require_session(self.db, str(self.session.id), str(self.user.id))
        with self.assertRaises(ConversationNotFoundError):
            self.conversation.save_user_message(self.db, self.session, str(uuid.uuid4()), "不能复活")

    def test_redis_cannot_bypass_postgres_expiry(self):
        self.expire()
        memory = RedisSessionMemoryService.__new__(RedisSessionMemoryService)
        with self.assertRaises(LookupError):
            memory._require_owned_session(self.db, user_id=str(self.user.id), session_id=str(self.session.id))

    def test_list_and_detail_hide_expired_session_before_cleanup(self):
        from fastapi import HTTPException
        from router.session_router import get_messages, get_session, get_sessions
        self.expire()
        live = self.new_session()
        rows = asyncio.run(get_sessions(limit=50, offset=0, session_type=None, current_user=self.user, db=self.db))
        self.assertEqual([row.id for row in rows], [str(live.id)])
        for function in (get_session, get_messages):
            with self.subTest(function=function.__name__), self.assertRaises(HTTPException) as error:
                asyncio.run(function(session_id=str(self.session.id), current_user=self.user, db=self.db))
            self.assertEqual(error.exception.status_code, 404)

    def test_title_and_summary_updates_do_not_renew(self):
        expiry = self.session.expires_at
        self.session.title = "新标题"
        self.session.semantic_context_summary = {"summary": "后台更新"}
        self.db.commit()
        self.assertEqual(self.session.expires_at, expiry)

    def test_cleanup_at_exact_boundary_cascades_and_invalidates(self):
        self.expire()
        message = ChatMessage(id=uuid.uuid4(), session_id=self.session.id, role="user", content="过期原文")
        self.db.add(message)
        self.db.flush()
        file = Path(self.temporary.name) / "过期附件.txt"
        file.write_text("附件正文", encoding="utf-8")
        self.db.add(ChatAttachment(session_id=self.session.id, message_id=message.id, user_id=self.user.id,
                                   filename=file.name, file_type="txt", file_size=12, file_path=str(file)))
        self.db.add(MemoryJob(user_id=self.user.id, session_id=self.session.id, operation_type="add",
                              idempotency_key=str(uuid.uuid4())))
        memory = LongTermMemory(user_id=self.user.id, session_id=self.session.id, summary="长期偏好",
                                source_start_message_id=message.id, source_end_message_id=message.id)
        self.db.add(memory)
        self.db.commit()
        session_id, user_id = str(self.session.id), str(self.user.id)
        self.assertEqual(self.cleanup(), 1)
        for model in (ChatSession, ChatMessage, ChatAttachment, MemoryJob):
            self.assertEqual(self.db.query(model).count(), 0)
        self.assertFalse(file.exists())
        self.memory.invalidate.assert_called_once_with(user_id=user_id, session_id=session_id)
        preserved = self.db.query(LongTermMemory).one()
        self.assertIsNone(preserved.session_id)
        self.assertIsNone(preserved.source_end_message_id)

    def test_fresh_session_and_renewed_old_session_survive(self):
        self.session.created_at = self.now - timedelta(days=10)
        self.session.expires_at = self.now + timedelta(seconds=1)
        self.db.commit()
        self.assertEqual(self.cleanup(), 0)
        self.memory.invalidate.assert_not_called()

    def test_recent_summary_does_not_protect_expired_session(self):
        self.expire()
        self.session.semantic_context_summary = {"summary": "刚生成"}
        self.db.commit()
        self.assertEqual(self.cleanup(), 1)

    def test_cleanup_is_batched(self):
        self.expire()
        self.new_session(expires_at=self.now - timedelta(seconds=1))
        self.assertEqual(self.cleanup(batch_size=1), 1)
        self.assertEqual(self.db.query(ChatSession).count(), 1)
        self.assertEqual(self.cleanup(batch_size=1), 1)

    def test_redis_failure_does_not_rollback_database_cleanup(self):
        self.expire()
        self.memory.invalidate.side_effect = RuntimeError("模拟 Redis 不可用")
        self.assertEqual(self.cleanup(), 1)
        self.assertEqual(self.db.query(ChatSession).count(), 0)


@unittest.skipUnless(os.getenv("RUN_SESSION_TTL_POSTGRES") == "1", "显式启用隔离 PostgreSQL 回归")
class PostgreSQLSessionExpiryTests(SessionExpiryTests):
    def create_engine(self):
        from core.database import engine as configured_engine
        self.schema = "会话ttl验证_" + uuid.uuid4().hex
        with configured_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{self.schema}"'))
        def drop_schema():
            with configured_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{self.schema}" CASCADE'))
        self.addCleanup(drop_schema)
        return create_engine(configured_engine.url,
                             connect_args={"options": f'-csearch_path={self.schema}'})

    def test_locked_session_is_skipped_until_transaction_finishes(self):
        self.expire()
        with self.factory() as locker:
            locker.query(ChatSession).filter(ChatSession.id == self.session.id).with_for_update().one()
            self.assertEqual(self.cleanup(), 0)
        self.assertEqual(self.cleanup(), 1)


@unittest.skipUnless(os.getenv("RUN_SESSION_TTL_POSTGRES") == "1", "显式启用隔离 PostgreSQL 迁移回归")
class PostgreSQLMigrationTests(unittest.TestCase):
    def test_backfill_timezone_default_and_idempotence(self):
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        from core.database import engine
        path = Path(__file__).resolve().parents[1] / "alembic/versions/20261006_0012_会话24小时过期.py"
        spec = importlib.util.spec_from_file_location("session_ttl_migration", path)
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        schema = "会话过期迁移验证_" + uuid.uuid4().hex
        now = utc_now()
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            for timezone in (False, True):
                with self.subTest(timezone=timezone):
                    kind = "TIMESTAMP WITH TIME ZONE" if timezone else "TIMESTAMP WITHOUT TIME ZONE"
                    connection.execute(text(f'CREATE TABLE chat_sessions (id UUID PRIMARY KEY, created_at {kind}, updated_at {kind})'))
                    connection.execute(text(f'CREATE TABLE chat_messages (session_id UUID, created_at {kind})'))
                    session_id, empty_id = uuid.uuid4(), uuid.uuid4()
                    created = now - timedelta(days=3)
                    latest = now - timedelta(hours=1)
                    connection.execute(text('INSERT INTO chat_sessions VALUES (:id, :created, :updated)'),
                                       [{'id': session_id, 'created': created if timezone else created.replace(tzinfo=None),
                                         'updated': now if timezone else now.replace(tzinfo=None)},
                                        {'id': empty_id, 'created': created if timezone else created.replace(tzinfo=None),
                                         'updated': now if timezone else now.replace(tzinfo=None)}])
                    connection.execute(text('INSERT INTO chat_messages VALUES (:id, :time)'),
                                       {'id': session_id, 'time': latest if timezone else latest.replace(tzinfo=None)})
                    connection.execute(text("SET LOCAL TIME ZONE 'Asia/Singapore'"))
                    migration.op = Operations(MigrationContext.configure(connection))
                    migration.upgrade()
                    rows = dict(connection.execute(text('SELECT id, expires_at FROM chat_sessions')).all())
                    self.assertEqual(rows[session_id], session_expiry_time(latest))
                    self.assertEqual(rows[empty_id], session_expiry_time(created))
                    renewed = now + timedelta(days=2)
                    connection.execute(text('UPDATE chat_sessions SET expires_at=:time WHERE id=:id'), {'id': session_id, 'time': renewed})
                    migration.upgrade()
                    self.assertEqual(connection.execute(text('SELECT expires_at FROM chat_sessions WHERE id=:id'), {'id': session_id}).scalar_one(), renewed)
                    connection.execute(text('INSERT INTO chat_sessions(id) VALUES (:id)'), {'id': uuid.uuid4()})
                    self.assertEqual(connection.execute(text('SELECT count(*) FROM chat_sessions WHERE expires_at IS NULL')).scalar(), 0)
                    migration.downgrade()
                    connection.execute(text('DROP TABLE chat_messages, chat_sessions'))
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))


class SessionExpirySchedulerTests(unittest.IsolatedAsyncioTestCase):
    def test_cleanup_registered_every_minute(self):
        from service.scheduler_service import SchedulerService
        scheduler = object.__new__(SchedulerService)
        scheduler._scheduler = MagicMock()
        scheduler._scheduler.running = False
        scheduler.start()
        job = next(call for call in scheduler._scheduler.add_job.call_args_list
                   if call.kwargs.get("id") == "expired_sessions_cleanup")
        self.assertEqual(job.args[1].interval.total_seconds(), 60)
        self.assertEqual(job.kwargs["max_instances"], 1)

    async def test_startup_cleans_expired_sessions(self):
        from service.scheduler_service import init_scheduler
        scheduler = MagicMock()
        scheduler._cleanup_web_reader_cache = AsyncMock()
        scheduler._cleanup_expired_sessions = AsyncMock()
        with patch("service.scheduler_service.get_scheduler_service", return_value=scheduler):
            await init_scheduler()
        scheduler._cleanup_expired_sessions.assert_awaited_once()
        self.assertEqual(scheduler.mock_calls[-1][0], "start")


if __name__ == "__main__":
    unittest.main()
