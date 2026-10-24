"""澄清轮原话复用、同条助手升级与事务幂等的机制回归。"""
from pathlib import Path
from datetime import datetime, timedelta
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import Mock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app"), str(BACKEND / "test")]
import test_会话24小时过期 as expiry_fixture
from models.chat import ChatMessage
from models.research import ResearchRun
from service.conversation_service import ConversationNotFoundError
from eval.scripts.随机题多轮会话实验 import reuse_clarified_user, completed_resume_prefix, verify_restart_prefix, validate_task_rebuild_row


class ClarificationDeliveryTests(unittest.TestCase):
    def setUp(self):
        fixture = self.fixture = expiry_fixture.SessionExpiryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.addCleanup(fixture.tearDown)
        self.turn, self.run = str(uuid.uuid4()), str(uuid.uuid4())
        self.user = fixture.conversation.save_user_message(fixture.db, fixture.session, self.turn, "原追问",
            execution_mode="research", run_id=self.run, research_status="running")
        self.old = fixture.conversation.save_assistant_message(fixture.db, str(fixture.session.id),
            str(fixture.user.id), self.turn, "请明确上一轮", execution_mode="chat", research_status="clarification",
            references_data={"session_task": {"status": "clarify"}})

    def stage(self, **overrides):
        f = self.fixture
        arguments = dict(session_id=str(f.session.id), user_id=str(f.user.id), turn_id=self.turn,
            content="本轮报告", execution_mode="research", run_id=self.run, research_status="completed",
            references_data={"research_contract": {"language": "中文"}}, replace_clarification=True)
        arguments.update(overrides)
        return f.conversation.stage_assistant_message(f.db, **arguments)

    def test_upgrade_keeps_id_query_and_archives_clarification(self):
        old_id, old_created = self.old.id, self.old.created_at
        upgraded = self.stage()
        self.fixture.db.commit()
        self.assertEqual(upgraded.id, old_id)
        self.assertEqual(upgraded.created_at, old_created)
        self.assertEqual(upgraded.content, "本轮报告")
        self.assertEqual(upgraded.run_id, self.run)
        self.assertEqual(upgraded.references_data["previous_clarification"]["content"], "请明确上一轮")
        self.assertEqual(upgraded.references_data["previous_clarification"]["references_data"]["session_task"]["status"], "clarify")
        self.assertEqual(self.user.content, "原追问")
        self.assertEqual(self.fixture.db.query(ChatMessage).count(), 2)

    def test_rollback_restores_clarification_without_extra_messages(self):
        self.stage()
        self.fixture.db.rollback()
        self.assertEqual(self.old.content, "请明确上一轮")
        self.assertEqual(self.old.research_status, "clarification")
        self.assertIsNone(self.old.run_id)
        self.assertEqual(self.fixture.db.query(ChatMessage).count(), 2)

    def test_default_idempotence_and_completed_report_cannot_be_overwritten(self):
        self.assertEqual(self.stage(replace_clarification=False).content, "请明确上一轮")
        first = self.stage(); self.fixture.db.commit()
        again = self.stage(content="不同的第二次报告")
        self.assertEqual(first.id, again.id)
        self.assertEqual(again.content, "本轮报告")

    def test_unrelated_run_cannot_replace_clarification(self):
        with self.assertRaisesRegex(ValueError, "关联"):
            self.stage(run_id=str(uuid.uuid4()))
        self.assertEqual(self.old.content, "请明确上一轮")

    def test_expiry_and_owner_checks_still_apply(self):
        with self.assertRaises(ConversationNotFoundError):
            self.stage(user_id=str(uuid.uuid4()))
        self.fixture.expire()
        with self.assertRaises(ConversationNotFoundError):
            self.stage()

    def test_restart_prefix_rejects_future_messages(self):
        f = self.fixture
        self.stage(); f.db.commit()
        rows = [{"turn_id": 1, "turn_uuid": self.turn, "run_id": self.run, "query": "原追问", "completed": True}]
        with tempfile.TemporaryDirectory() as temporary, patch("core.database.SessionLocal", f.factory), patch(
            "service.conversation_service.get_conversation_service", return_value=f.conversation):
            path = Path(temporary); (path / "第1轮").mkdir(); (path / "第1轮/研究报告.md").write_text("本轮报告")
            result = verify_restart_prefix(str(f.user.id), str(f.session.id), rows, path)
            self.assertEqual(result["message_count"], 2)
            self.assertEqual(result["future_messages"], 0)
            f.conversation.save_user_message(f.db, f.session, str(uuid.uuid4()), "未来追问")
            with self.assertRaisesRegex(ValueError, "未来消息"):
                verify_restart_prefix(str(f.user.id), str(f.session.id), rows, path)


class ReuseUserTests(unittest.TestCase):
    def setUp(self):
        self.owner, self.session, self.turn, self.run = [str(uuid.uuid4()) for _ in range(4)]
        self.message = SimpleNamespace(id=uuid.uuid4(), content="原追问", research_status="clarification",
                                       execution_mode="chat", run_id=None, created_at="原创建时间")
        self.db = Mock()
        self.db.__enter__ = Mock(return_value=self.db); self.db.__exit__ = Mock(return_value=False)
        self.run_query, self.message_query = Mock(), Mock()
        self.run_query.filter_by.return_value.first.return_value = None
        self.message_query.filter_by.return_value.one.return_value = self.message
        self.db.query.side_effect = lambda model: self.run_query if model is ResearchRun else self.message_query
        self.conversation = Mock()

    def reuse(self, **kwargs):
        with patch("core.database.SessionLocal", return_value=self.db), patch(
            "service.conversation_service.get_conversation_service", return_value=self.conversation):
            return reuse_clarified_user(self.owner, self.session, self.turn, self.run, "原追问", **kwargs)

    def test_readonly_then_activate_reuses_same_message(self):
        record = self.reuse()
        self.assertEqual(record["id"], str(self.message.id))
        self.assertIsNone(self.message.run_id)
        self.db.commit.assert_not_called()
        self.reuse(activate=True)
        self.assertEqual(self.message.run_id, self.run)
        self.assertEqual(self.message.execution_mode, "research")
        self.assertEqual(self.message.content, "原追问")
        self.db.add.assert_not_called()
        self.db.commit.assert_called_once()
        self.conversation.require_session.assert_called_with(self.db, self.session, self.owner, for_update=True)

    def test_existing_research_or_changed_query_is_rejected(self):
        self.run_query.filter_by.return_value.first.return_value = object()
        with self.assertRaisesRegex(ValueError, "已有研究"):
            self.reuse(activate=True)
        self.run_query.filter_by.return_value.first.return_value = None
        self.message.content = "另一条问题"
        with self.assertRaisesRegex(ValueError, "原用户消息"):
            self.reuse(activate=True)
        self.db.commit.assert_not_called()


class RestartSelectionTests(unittest.TestCase):
    def test_restart_from_11_keeps_10_completed_turns(self):
        rows = [{"turn_id": number, "completed": number < 15} for number in range(1, 16)]
        self.assertEqual(completed_resume_prefix({"rows": rows}, 11), rows[:10])
        self.assertEqual(completed_resume_prefix({"rows": rows}), rows[:14])
        self.assertEqual(len(rows), 15)

    def test_invalid_restart_or_incomplete_prefix_is_rejected(self):
        rows = [{"turn_id": 1, "completed": False}, {"turn_id": 2, "completed": False}]
        for number in (0, 3, 2):
            with self.subTest(number=number), self.assertRaises(ValueError):
                completed_resume_prefix({"rows": rows}, number)


class PreGraphFailureRecoveryTests(ReuseUserTests):
    def setUp(self):
        super().setUp()
        self.run_record = SimpleNamespace(user_id=uuid.UUID(self.owner), session_id=self.session,
            status="failed", final_checkpoint_id=None, last_projected_checkpoint_id=None,
            delivered_event_id=None, lease_until=None, next_event_seq=2, fencing_token=1)
        self.run_query.filter_by.return_value.first.return_value = self.run_record
        self.message_query.filter_by.return_value.first.return_value = None
        self.message.research_status = "failed"
        self.message.execution_mode = "research"
        self.message.run_id = self.run

    def test_failed_run_reuses_original_message_and_event_sequence(self):
        record = self.reuse(allow_failed_run=True)
        self.assertEqual(record["previous_run_id"], self.run)
        self.db.commit.assert_not_called()
        self.reuse(activate=True, allow_failed_run=True)
        self.assertEqual(self.message.research_status, "running")
        self.assertEqual(self.message.run_id, self.run)
        self.assertEqual(self.run_record.next_event_seq, 2)
        self.assertEqual(self.run_record.fencing_token, 1)
        self.db.add.assert_not_called()

    def test_run_ownership_checkpoint_delivery_and_lease_block_rebuild(self):
        cases = [("user_id", uuid.uuid4()), ("session_id", str(uuid.uuid4())),
                 ("status", "running"), ("status", "completed"),
                 ("final_checkpoint_id", "checkpoint"), ("last_projected_checkpoint_id", "checkpoint"),
                 ("delivered_event_id", uuid.uuid4()), ("lease_until", datetime.utcnow() + timedelta(seconds=60))]
        for key, value in cases:
            before = getattr(self.run_record, key)
            setattr(self.run_record, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.reuse(activate=True, allow_failed_run=True)
            setattr(self.run_record, key, before)
        self.db.commit.assert_not_called()

    def test_already_delivered_assistant_blocks_rebuild(self):
        self.message_query.filter_by.return_value.first.return_value = SimpleNamespace(research_status="completed")
        with self.assertRaisesRegex(ValueError, "助手交付"):
            self.reuse(activate=True, allow_failed_run=True)
        self.db.commit.assert_not_called()

    def test_only_clarification_rows_are_rebuilt(self):
        for row in ({"status": "clarified"}, {"status": "failed", "error_kind": "clarification"}):
            validate_task_rebuild_row(row)
        for row in ({"status": "completed"}, {"status": "failed", "error_kind": "local_context_budget"},
                    {"status": "running", "error_kind": "clarification"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_task_rebuild_row(row)

    def test_readonly_then_activate_reuses_same_message(self):
        # 父夹具测试针对尚无ResearchRun的澄清轮，已有失败run单独核验。
        self.run_query.filter_by.return_value.first.return_value = None
        self.message.run_id = None
        super().test_readonly_then_activate_reuses_same_message()


if __name__ == "__main__":
    unittest.main()
