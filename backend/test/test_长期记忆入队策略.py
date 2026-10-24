"""长期记忆及时入队、增量与空闲批处理的离线机制回归。"""

import sys
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from sqlalchemy import JSON, MetaData, create_engine
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import sessionmaker

from models.chat import ChatMessage, ChatSession, LongTermMemory, MemoryJob
from models.user import User
from core.time_utils import utc_now
from service import memory_service
from service.conversation_service import ConversationService
from service.memory_trigger_policy import priority_memory_trigger


class PriorityPolicyTests(unittest.TestCase):
    def test_explicit_memory_requests(self):
        for content in ("请记住我的项目负责人是李明。", "记住：报告要先给结论", "请记住：我今天完成了项目交付。",
                        "Please remember my preferred language is Chinese."):
            with self.subTest(content=content):
                self.assertEqual(priority_memory_trigger([{"role": "user", "content": content}]), "explicit_remember")

    def test_stable_preferences(self):
        for content in ("我喜欢中文表格。", "我更喜欢简洁的回答。", "我不喜欢过多表情。", "以后请用中文回答。",
                        "从现在起，报告都先给结论。", "I prefer concise answers."):
            with self.subTest(content=content):
                self.assertEqual(priority_memory_trigger([{"role": "user", "content": content}]), "stable_preference")

    def test_important_corrections(self):
        for content in ("更正一下：预算是20万元。", "把研究时间范围改为2022—2026年，其他要求不变。",
                        "预算改为20万元。", "不是李明，是王芳。", "Correction: my address is now Shanghai."):
            with self.subTest(content=content):
                self.assertEqual(priority_memory_trigger([{"role": "user", "content": content}]), "important_correction")

    def test_transient_quotes_negations_and_questions_do_not_trigger(self):
        for content in ("谢谢", "这次请用中文表格。", "我今天喜欢吃面。", "I prefer a table for this answer.",
                        "不要记住我喜欢什么。", "请不要记住：我的预算是20万元。", "Don't remember my address.",
                        "他说：我喜欢中文。", "“请记住我的名字”", '> 请记住我的名字\n只是引用。',
                        '```\n请记住我的名字\n```', '```\n请记住我的名字', "我喜欢什么颜色？",
                        "请解释把预算改为20万元的影响。", "记住是什么意思？"):
            with self.subTest(content=content):
                self.assertIsNone(priority_memory_trigger([{"role": "user", "content": content}]))

    def test_only_user_messages_trigger(self):
        for role in ("assistant", "system", "tool"):
            with self.subTest(role=role):
                self.assertIsNone(priority_memory_trigger([{"role": role, "content": "请记住预算改为20万元"}]))


class MemoryEnqueueTests(unittest.TestCase):
    """运行真实 ORM 查询/事务；SQLite 仅替代 PostgreSQL 存储，不连接任何外部服务。"""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        for model in (User, ChatSession, ChatMessage, LongTermMemory, MemoryJob):
            table = model.__table__.to_metadata(metadata)
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
        self.user_id = uuid.uuid4()
        self.db.add(User(id=self.user_id, username="test", email="test@example.invalid", hashed_password="test"))
        self.session = self.new_session()
        self.db.commit()
        self.counter = SimpleNamespace(count_messages=lambda rows: sum(len(row["content"]) for row in rows))
        service = memory_service.MemoryService.__new__(memory_service.MemoryService)
        service.token_counter = self.counter
        self.stack = []
        for mocker in (
            patch.object(memory_service, "SessionLocal", self.factory),
            patch.object(memory_service, "get_memory_service", return_value=service),
            patch.object(memory_service, "MEMORY_TOKEN_THRESHOLD", 2000),
            patch.object(memory_service, "MEMORY_IDLE_SECONDS", 300),
        ):
            mocker.start()
            self.stack.append(mocker)
        self.conversation = ConversationService.__new__(ConversationService)
        self.conversation.session_memory = Mock()

    def tearDown(self):
        for mocker in reversed(self.stack):
            mocker.stop()
        self.db.close()
        self.engine.dispose()

    def new_session(self):
        session = ChatSession(id=uuid.uuid4(), user_id=self.user_id, title="测试会话")
        self.db.add(session)
        return session

    def add_message(self, text="普通内容", *, role="user", age=0, session=None):
        message = ChatMessage(id=uuid.uuid4(), session_id=(session or self.session).id,
                              turn_id=uuid.uuid4(), role=role, content=text,
                              created_at=datetime.utcnow() - timedelta(seconds=age))
        self.db.add(message)
        self.db.commit()
        return message

    def enqueue(self, **kwargs):
        return memory_service.enqueue_memory_if_due(self.db, str(self.user_id), str(self.session.id), **kwargs)

    def consume(self, message, status="ready", *, legacy=False):
        self.db.add(LongTermMemory(id=uuid.uuid4(), user_id=self.user_id, session_id=message.session_id,
                                   summary="已消费", index_status=status,
                                   source_end_message_id=None if legacy else message.id))
        self.db.commit()

    def test_short_important_user_message_commits_job_before_assistant(self):
        turn = str(uuid.uuid4())
        message = self.conversation.save_user_message(self.db, self.session, turn, "请记住我的名字是李明。")
        job = self.db.query(MemoryJob).one()
        self.assertEqual(job.payload["message_ids"], [str(message.id)])
        self.assertEqual(job.payload["trigger_reason"], "explicit_remember")
        duplicate = self.conversation.save_user_message(self.db, self.session, turn, "请记住我的名字是李明。")
        self.assertEqual(duplicate.id, message.id)
        self.assertEqual(self.db.query(MemoryJob).count(), 1)

    def test_message_and_job_roll_back_together_on_commit_failure(self):
        with patch.object(self.db, "commit", side_effect=RuntimeError("模拟提交失败")):
            with self.assertRaisesRegex(RuntimeError, "提交失败"):
                self.conversation.save_user_message(self.db, self.session, str(uuid.uuid4()), "请记住我的名字。")
        self.db.rollback()
        self.assertEqual(self.db.query(ChatMessage).count(), 0)
        self.assertEqual(self.db.query(MemoryJob).count(), 0)

    def test_preferences_and_corrections_bypass_token_counter(self):
        for text, reason in (("我喜欢中文表格", "stable_preference"), ("把预算改为20万元", "important_correction")):
            with self.subTest(text=text):
                self.session = self.new_session()
                self.add_message(text)
                with patch.object(memory_service, "get_memory_service", side_effect=AssertionError("不应计数")):
                    job = self.enqueue(urgent_only=True)
                self.db.commit()
                self.assertEqual(job.payload["trigger_reason"], reason)

    def test_ordinary_user_message_waits_for_assistant_or_idle(self):
        self.conversation.save_user_message(self.db, self.session, str(uuid.uuid4()), "普通文本" * 1000)
        self.assertEqual(self.db.query(MemoryJob).count(), 0)
        job = self.enqueue()
        self.assertEqual(job.payload["trigger_reason"], "incremental_tokens")

    def test_incremental_threshold_is_strict_and_excludes_consumed_messages(self):
        old = self.add_message("旧" * 3000, age=1000)
        self.consume(old)
        self.add_message("新" * 2000)
        self.assertIsNone(self.enqueue())
        self.add_message("加")
        job = self.enqueue()
        self.assertEqual(job.payload["trigger_reason"], "incremental_tokens")
        self.assertNotIn(str(old.id), job.payload["message_ids"])

    def test_idle_scan_flushes_short_conversation_without_new_request(self):
        message = self.add_message(age=301)
        self.assertEqual(memory_service.enqueue_idle_memory_sessions(), 1)
        self.db.expire_all()
        job = self.db.query(MemoryJob).one()
        self.assertEqual(job.payload["trigger_reason"], "idle")
        self.assertEqual(job.payload["message_ids"], [str(message.id)])
        self.assertEqual(memory_service.enqueue_idle_memory_sessions(), 0)

    def test_new_activity_postpones_idle_processing(self):
        self.add_message(age=600)
        self.add_message("新的普通消息", age=10)
        self.assertEqual(memory_service.enqueue_idle_memory_sessions(), 0)
        self.assertIsNone(self.enqueue(idle_before=datetime.utcnow() - timedelta(seconds=300)))

    def test_idle_scanner_rechecks_messages_after_candidate_selection(self):
        self.add_message(age=600)
        enqueue = memory_service.enqueue_memory_if_due
        def concurrent_message(*args, **kwargs):
            self.add_message("扫描后刚到的新消息")
            return enqueue(*args, **kwargs)
        with patch.object(memory_service, "enqueue_memory_if_due", side_effect=concurrent_message):
            self.assertEqual(memory_service.enqueue_idle_memory_sessions(), 0)

    def test_consumed_deleted_legacy_and_dead_sessions_do_not_starve_idle_queue(self):
        for status in ("ready", "deleting", "deleted"):
            self.session = self.new_session()
            self.consume(self.add_message(age=900), status)
        self.session = self.new_session()
        self.consume(self.add_message(age=900), legacy=True)
        self.session = self.new_session()
        self.add_message("请记住我的名字", age=900)
        job = self.enqueue()
        job.status = "dead"
        self.db.commit()
        self.session = self.new_session()
        target = self.add_message(age=600)
        self.assertEqual(memory_service.enqueue_idle_memory_sessions(batch_size=1), 1)
        self.db.expire_all()
        pending = self.db.query(MemoryJob).filter(MemoryJob.status == "pending").one()
        self.assertEqual(pending.payload["message_ids"], [str(target.id)])

    def test_idle_can_be_disabled(self):
        self.add_message(age=10000)
        with patch.object(memory_service, "MEMORY_IDLE_SECONDS", 0):
            self.assertEqual(memory_service.enqueue_idle_memory_sessions(), 0)

    def test_aware_message_and_legacy_naive_cutoff_can_be_compared(self):
        message = self.add_message(age=600)
        # 与 PostgreSQL 返回 timestamptz 的真实故障条件一致。
        message.created_at = utc_now() - timedelta(seconds=600)
        self.db.commit()
        self.db.refresh(message)
        self.assertEqual(message.created_at.utcoffset(), timedelta(0))
        job = self.enqueue(idle_before=datetime.utcnow() - timedelta(seconds=300))
        self.assertEqual(job.payload["trigger_reason"], "idle")
        self.assertEqual(job.next_attempt_at.utcoffset(), timedelta(0))

    def test_beijing_time_is_compared_as_an_instant_at_idle_boundary(self):
        beijing = timezone(timedelta(hours=8))
        cutoff = datetime(2026, 9, 24, 13, 0, tzinfo=timezone.utc)
        for delta, expected in ((-1, "idle"), (0, "idle"), (1, None)):
            with self.subTest(delta_seconds=delta):
                self.session = self.new_session()
                message = self.add_message()
                message.created_at = (cutoff + timedelta(seconds=delta)).astimezone(beijing)
                self.db.commit()
                self.db.refresh(message)
                job = self.enqueue(idle_before=cutoff)
                self.assertEqual(job.payload["trigger_reason"] if job else None, expected)

    def test_persisted_memory_job_timestamps_are_aware_utc(self):
        self.add_message("请记住我的名字")
        job = self.enqueue()
        self.db.commit()
        self.db.refresh(job)
        for field in ("created_at", "updated_at", "next_attempt_at"):
            self.assertEqual(getattr(job, field).utcoffset(), timedelta(0))

    def test_automatic_priority_does_not_revive_dead_job(self):
        self.add_message("请记住预算为20万元")
        job = self.enqueue()
        job.status = "dead"
        self.db.commit()
        self.assertEqual(self.enqueue().status, "dead")
        self.assertEqual(self.enqueue(force=True).status, "pending")

    def test_active_job_payload_is_not_overwritten_by_new_correction(self):
        old = self.add_message("请记住预算为20万元", age=30)
        job = self.enqueue()
        self.db.commit()
        new = self.add_message("把预算改为30万元")
        self.assertEqual(self.enqueue().id, job.id)
        self.assertEqual(job.payload["message_ids"], [str(old.id)])
        self.consume(old)
        job.status = "succeeded"
        self.db.commit()
        followup = self.enqueue()
        self.assertEqual(followup.payload["message_ids"], [str(new.id)])
        self.assertEqual(followup.payload["trigger_reason"], "important_correction")

    def test_cross_user_enqueue_is_rejected(self):
        self.add_message("请记住我的名字")
        with self.assertRaises(LookupError):
            memory_service.enqueue_memory_if_due(self.db, str(uuid.uuid4()), str(self.session.id))

    def test_important_tail_does_not_skip_unprocessed_prefix(self):
        first = self.add_message("旧" * 100, age=10)
        second = self.add_message("请记住预算为20万元")
        with patch.object(memory_service, "MEMORY_SUMMARY_MAX_CHARS", 110):
            job = self.enqueue()
        self.assertEqual(job.payload["message_ids"], [str(first.id)])
        self.assertTrue(job.payload["force_followup"])
        self.consume(first)
        job.status = "succeeded"
        self.db.commit()
        followup = self.enqueue()
        self.assertEqual(followup.payload["message_ids"], [str(second.id)])

    def test_worker_scans_then_consumes_persisted_jobs(self):
        calls = []
        with (patch.object(memory_service, "enqueue_idle_memory_sessions", side_effect=lambda n: calls.append("scan")),
              patch.object(memory_service, "_claim_memory_jobs", side_effect=lambda n: calls.append("claim") or ["job"]),
              patch.object(memory_service, "_execute_memory_job", side_effect=lambda value: calls.append(value))):
            self.assertEqual(memory_service.process_memory_jobs(10), 1)
        self.assertEqual(calls, ["scan", "claim", "job"])

    def test_scan_failure_does_not_block_existing_outbox(self):
        with (patch.object(memory_service, "enqueue_idle_memory_sessions", side_effect=RuntimeError("扫描失败")),
              patch.object(memory_service, "_claim_memory_jobs", return_value=["job"]),
              patch.object(memory_service, "_execute_memory_job") as execute):
            self.assertEqual(memory_service.process_memory_jobs(), 1)
        execute.assert_called_once_with("job")


if __name__ == "__main__":
    unittest.main()
