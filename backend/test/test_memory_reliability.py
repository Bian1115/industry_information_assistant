"""第一阶段记忆链路的回归测试（仅使用 unittest）。"""

import os
import sys
import threading
import time
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from models.chat import ChatMessage, ChatSession, LongTermMemory, MemoryJob
from models.user import User
from service.chat_service import ChatService
from service.conversation_service import ConversationService
from service.memory_service import MemoryService
from service.memory_contracts import MemoryWriteResult


class EmptyMemorySearchTests(unittest.TestCase):
    def test_empty_authorized_memory_does_not_call_vector_service(self):
        from service.memory_contracts import MemoryScope, MemorySearchOptions
        from service.memory_service import NativeMemoryProvider
        db = Mock()
        db.query.return_value.filter.return_value.first.return_value = None
        vector_service = Mock()
        vector_service.retrieve_memories.side_effect = AssertionError("空记忆不能访问向量服务")
        with patch("service.memory_service.SessionLocal", return_value=db):
            result = NativeMemoryProvider(service=vector_service).search(
                MemoryScope(user_id=str(uuid.uuid4())), "研究", MemorySearchOptions(limit=5))
        self.assertEqual(result, [])
        vector_service.retrieve_memories.assert_not_called()
        db.close.assert_called_once()


class MemoryThresholdTests(unittest.TestCase):
    def setUp(self):
        # estimate/should_compress 不应初始化 OpenAI 或 Milvus 客户端。
        self.service = MemoryService.__new__(MemoryService)
        self.service.token_counter = SimpleNamespace(
            count_text=lambda text: len(text or ""),
            count_messages=lambda messages: sum(
                len(message.get("content", "")) for message in messages
            ),
        )

    def test_threshold_uses_qwen_token_count(self):
        below = [SimpleNamespace(content="中" * 2000)]
        above = [SimpleNamespace(content="中" * 2001)]
        self.assertFalse(self.service.should_compress(below))
        self.assertTrue(self.service.should_compress(above))

    def test_memory_batch_does_not_mark_truncated_messages_as_consumed(self):
        from service.memory_service import _select_memory_batch

        messages = [SimpleNamespace(content="中" * 12000) for _ in range(3)]
        selected = _select_memory_batch(messages)
        self.assertEqual(len(selected), 2)

        # 单条超长消息无法按消息游标拆分，但仍必须保证任务能前进。
        oversized = SimpleNamespace(content="中" * 40000)
        self.assertEqual(_select_memory_batch([oversized]), [oversized])


class MemorySummaryFallbackTests(unittest.TestCase):
    def setUp(self):
        self.service = MemoryService.__new__(MemoryService)
        self.service.model = "qwen-test"
        create = Mock(side_effect=RuntimeError("data_inspection_failed"))
        self.service._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        self.messages = [SimpleNamespace(role="user", content="测试对话")]

    def test_content_moderation_falls_back_to_claude(self):
        claude_result = """```json
{"summary":"Claude摘要","key_insights":["信息"],"user_preferences":{},"topics":["测试"]}
```"""
        environment = {
            "CLAUDE_RELAY_API_KEY": "claude-test-key",
            "CLAUDE_RELAY_BASE_URL": "https://claude.test/v1",
            "CLAUDE_RELAY_MODEL": "claude-opus-5",
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            patch(
                "service.memory_service.claude_messages_completion",
                return_value=claude_result,
            ) as claude,
        ):
            result = self.service.summarize_conversation(self.messages)

        self.assertEqual(result["summary"], "Claude摘要")
        self.assertEqual(result["summary_provider"], "claude-relay")
        self.assertEqual(
            result["fallback_reason"], "dashscope_content_moderation"
        )
        self.assertEqual(claude.call_args.args[:3], (
            "claude-test-key", "https://claude.test/v1", "claude-opus-5"
        ))

    def test_non_moderation_error_does_not_fall_back(self):
        self.service._client.chat.completions.create.side_effect = RuntimeError(
            "temporary network failure"
        )
        with patch(
            "service.memory_service.claude_messages_completion"
        ) as claude:
            with self.assertRaisesRegex(RuntimeError, "temporary network failure"):
                self.service.summarize_conversation(self.messages)
        claude.assert_not_called()


class ChatStreamingTests(unittest.TestCase):
    def test_completion_callback_runs_once(self):
        service = ChatService.__new__(ChatService)
        service.openai_api_key = "test"
        service.openai_base_url = "https://example.invalid"
        service.openai_model = "test-model"

        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(content="你好", reasoning_content=None),
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(content=None, reasoning_content=None),
            )]),
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: iter(chunks))
            )
        )
        completed = []
        with patch("service.chat_service.create_openai_client", return_value=client):
            events = list(service.get_chat_completion(
                question="测试",
                retrieved_content=[],
                history_messages=[],
                user_id=None,
                on_complete=lambda answer, thinking: completed.append((answer, thinking)),
            ))

        self.assertEqual(completed, [("你好", "")])
        self.assertEqual(sum("[DONE]" in event for event in events), 1)


@unittest.skipUnless(
    os.getenv("RUN_DB_INTEGRATION_TESTS") == "1",
    "set RUN_DB_INTEGRATION_TESTS=1 to exercise PostgreSQL",
)
class PostgreSQLConversationTests(unittest.TestCase):
    def setUp(self):
        from core.database import SessionLocal

        self.db = SessionLocal()
        suffix = uuid.uuid4().hex[:10]
        self.user_id = uuid.uuid4()
        self.session_id = uuid.uuid4()
        self.turn_id = uuid.uuid4()
        self.user = User(
            id=self.user_id,
            username=f"memory_test_{suffix}",
            email=f"{suffix}@memory.test",
            hashed_password="test",
        )
        self.session = ChatSession(
            id=self.session_id,
            user_id=self.user_id,
            title="新对话",
        )
        self.db.add_all([self.user, self.session])
        self.db.commit()
        self.service = ConversationService()

    def tearDown(self):
        self.db.rollback()
        user = self.db.query(User).filter(User.id == self.user_id).first()
        if user:
            self.db.delete(user)
            self.db.commit()
        self.db.close()

    def test_turn_is_idempotent_and_short_turn_does_not_enqueue(self):
        session = self.service.require_session(
            self.db, str(self.session_id), str(self.user_id)
        )
        user_message = self.service.save_user_message(
            self.db, session, str(self.turn_id), "测试问题"
        )
        duplicate_user = self.service.save_user_message(
            self.db, session, str(self.turn_id), "测试问题"
        )
        assistant = self.service.save_assistant_message(
            self.db,
            str(self.session_id),
            str(self.user_id),
            str(self.turn_id),
            "测试回答",
        )
        duplicate_assistant = self.service.save_assistant_message(
            self.db,
            str(self.session_id),
            str(self.user_id),
            str(self.turn_id),
            "测试回答",
        )

        self.assertEqual(user_message.id, duplicate_user.id)
        self.assertEqual(assistant.id, duplicate_assistant.id)
        self.assertEqual(
            self.db.query(ChatMessage).filter(
                ChatMessage.session_id == self.session_id
            ).count(),
            2,
        )
        self.assertEqual(
            self.db.query(MemoryJob).filter(
                MemoryJob.session_id == self.session_id
            ).count(),
            0,
        )

    def test_large_turn_enqueues_exactly_one_job(self):
        session = self.service.require_session(
            self.db, str(self.session_id), str(self.user_id)
        )
        self.service.save_user_message(
            self.db, session, str(self.turn_id), "甲" * 6100
        )
        self.service.save_assistant_message(
            self.db,
            str(self.session_id),
            str(self.user_id),
            str(self.turn_id),
            "收到",
        )
        self.service.save_assistant_message(
            self.db,
            str(self.session_id),
            str(self.user_id),
            str(self.turn_id),
            "收到",
        )

        jobs = self.db.query(MemoryJob).filter(
            MemoryJob.session_id == self.session_id
        ).all()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, "pending")
        self.assertEqual(len(jobs[0].payload["message_ids"]), 2)

    def test_force_requeues_dead_add_job(self):
        from service.memory_service import enqueue_memory_if_due

        session = self.service.require_session(
            self.db, str(self.session_id), str(self.user_id)
        )
        self.service.save_user_message(
            self.db, session, str(self.turn_id), "丙" * 6100
        )
        self.service.save_assistant_message(
            self.db,
            str(self.session_id),
            str(self.user_id),
            str(self.turn_id),
            "收到",
        )
        original = self.db.query(MemoryJob).filter(
            MemoryJob.session_id == self.session_id
        ).one()
        original.status = "dead"
        original.attempts = 6
        original.last_error = "test dead letter"
        self.db.commit()

        requeued = enqueue_memory_if_due(
            self.db,
            str(self.user_id),
            str(self.session_id),
            force=True,
        )
        self.db.commit()

        self.assertEqual(requeued.id, original.id)
        self.assertEqual(requeued.status, "pending")
        self.assertEqual(requeued.attempts, 0)
        self.assertIsNone(requeued.last_error)

    def test_concurrent_outbox_conflict_keeps_both_assistant_messages(self):
        from core.database import SessionLocal
        from service.memory_service import enqueue_memory_if_due

        second_turn_id = uuid.uuid4()
        self.db.add_all([
            ChatMessage(
                session_id=self.session_id,
                turn_id=self.turn_id,
                role="user",
                content="并" * 6100,
            ),
            ChatMessage(
                session_id=self.session_id,
                turn_id=second_turn_id,
                role="user",
                content="第二个并发问题",
            ),
        ])
        self.db.commit()

        first_db = SessionLocal()
        first_db.add(ChatMessage(
            session_id=self.session_id,
            turn_id=self.turn_id,
            role="assistant",
            content="并发回答一",
        ))
        first_db.flush()
        enqueue_memory_if_due(first_db, str(self.user_id), str(self.session_id))

        started = threading.Event()
        finished = threading.Event()
        errors = []

        def competing_writer():
            second_db = SessionLocal()
            try:
                second_db.add(ChatMessage(
                    session_id=self.session_id,
                    turn_id=second_turn_id,
                    role="assistant",
                    content="并发回答二",
                ))
                second_db.flush()
                started.set()
                enqueue_memory_if_due(
                    second_db, str(self.user_id), str(self.session_id)
                )
                second_db.commit()
            except Exception as exc:  # pragma: no cover - assertion reports details
                errors.append(exc)
                second_db.rollback()
            finally:
                second_db.close()
                finished.set()

        thread = threading.Thread(target=competing_writer)
        thread.start()
        try:
            self.assertTrue(started.wait(3), "并发写入线程未启动")
            # 让第二个事务进入唯一约束等待，再提交第一个事务。
            time.sleep(0.2)
            first_db.commit()
            self.assertTrue(finished.wait(5), "并发写入线程未结束")
            thread.join()
        finally:
            first_db.rollback()
            first_db.close()

        self.assertEqual(errors, [])
        self.db.expire_all()
        self.assertEqual(
            self.db.query(ChatMessage).filter(
                ChatMessage.session_id == self.session_id,
                ChatMessage.role == "assistant",
            ).count(),
            2,
        )
        self.assertEqual(
            self.db.query(MemoryJob).filter(
                MemoryJob.session_id == self.session_id
            ).count(),
            1,
        )

    def test_deleted_memory_still_advances_consumption_cursor(self):
        from service.memory_service import get_unmemorized_messages

        session = self.service.require_session(
            self.db, str(self.session_id), str(self.user_id)
        )
        self.service.save_user_message(
            self.db, session, str(self.turn_id), "第一轮问题"
        )
        self.service.save_assistant_message(
            self.db,
            str(self.session_id),
            str(self.user_id),
            str(self.turn_id),
            "第一轮回答",
        )
        first_turn = self.db.query(ChatMessage).filter(
            ChatMessage.session_id == self.session_id
        ).order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).all()
        memory = LongTermMemory(
            user_id=self.user_id,
            session_id=self.session_id,
            summary="已删除的第一轮记忆",
            key_insights={"summary": "已删除的第一轮记忆"},
            provider="native",
            source_start_message_id=first_turn[0].id,
            source_end_message_id=first_turn[-1].id,
            index_status="deleted",
            deleted_at=datetime.utcnow(),
        )
        self.db.add(memory)
        self.db.commit()

        next_turn_id = uuid.uuid4()
        self.service.save_user_message(
            self.db, session, str(next_turn_id), "第二轮问题"
        )
        remaining = get_unmemorized_messages(self.db, str(self.session_id))

        self.assertEqual([message.turn_id for message in remaining], [next_turn_id])

    def test_provider_search_honors_session_scope(self):
        from service.memory_contracts import MemoryScope, MemorySearchOptions
        from service.memory_service import NativeMemoryProvider

        other_session = ChatSession(
            id=uuid.uuid4(),
            user_id=self.user_id,
            title="另一个会话",
        )
        self.db.add(other_session)
        self.db.flush()
        scoped_memory = LongTermMemory(
            user_id=self.user_id,
            session_id=self.session_id,
            summary="当前会话记忆",
            key_insights={"summary": "当前会话记忆"},
            provider="native",
            index_status="ready",
        )
        other_memory = LongTermMemory(
            user_id=self.user_id,
            session_id=other_session.id,
            summary="其他会话记忆",
            key_insights={"summary": "其他会话记忆"},
            provider="native",
            index_status="ready",
        )
        self.db.add_all([scoped_memory, other_memory])
        self.db.commit()

        class FakeMemoryService:
            def retrieve_memories(self, user_id, query, top_k):
                return [
                    {
                        "id": "fragment-current",
                        "session_id": str(self_session_id),
                        "memory_type": "summary",
                        "content": "当前会话记忆",
                        "metadata": '{"memory_id":"%s"}' % scoped_memory.id,
                        "score": 0.9,
                    },
                    {
                        "id": "fragment-other",
                        "session_id": str(other_session.id),
                        "memory_type": "summary",
                        "content": "其他会话记忆",
                        "metadata": '{"memory_id":"%s"}' % other_memory.id,
                        "score": 0.8,
                    },
                ]

        self_session_id = self.session_id
        provider = NativeMemoryProvider(service=FakeMemoryService())
        hits = provider.search(
            MemoryScope(
                user_id=str(self.user_id),
                session_id=str(self.session_id),
            ),
            "测试",
            MemorySearchOptions(limit=5),
        )

        self.assertEqual([hit.memory_id for hit in hits], [str(scoped_memory.id)])

    def test_failed_worker_operation_is_scheduled_for_retry(self):
        import service.memory_service as memory_module

        job = MemoryJob(
            operation_type="reset_user",
            provider="native",
            user_id=self.user_id,
            payload={},
            idempotency_key=f"reset-test:{uuid.uuid4()}",
            status="running",
            attempts=1,
            next_attempt_at=datetime.utcnow(),
            locked_at=datetime.utcnow(),
        )
        self.db.add(job)
        self.db.commit()

        class FailingProvider:
            name = "native"

            def reset_user(self, user_id, operation_id):
                raise RuntimeError("expected provider failure")

        previous = memory_module._memory_provider
        memory_module._memory_provider = FailingProvider()
        try:
            memory_module._execute_memory_job(str(job.id))
        finally:
            memory_module._memory_provider = previous

        self.db.expire_all()
        retried = self.db.query(MemoryJob).filter(MemoryJob.id == job.id).one()
        self.assertEqual(retried.status, "retry")
        self.assertIn("expected provider failure", retried.last_error)
        self.assertIsNone(retried.locked_at)

    def test_successful_worker_marks_job_succeeded(self):
        import service.memory_service as memory_module
        from core.database import SessionLocal

        session = self.service.require_session(
            self.db, str(self.session_id), str(self.user_id)
        )
        self.service.save_user_message(
            self.db, session, str(self.turn_id), "乙" * 6100
        )
        self.service.save_assistant_message(
            self.db,
            str(self.session_id),
            str(self.user_id),
            str(self.turn_id),
            "收到",
        )
        job = self.db.query(MemoryJob).filter(
            MemoryJob.session_id == self.session_id
        ).one()
        job.status = "running"
        job.attempts = 1
        job.locked_at = datetime.utcnow()
        self.db.commit()

        class SuccessfulProvider:
            name = "native"

            def add_turns(self, scope, turns, operation_id):
                provider_db = SessionLocal()
                try:
                    memory = LongTermMemory(
                        user_id=uuid.UUID(scope.user_id),
                        session_id=uuid.UUID(scope.session_id),
                        summary="测试记忆",
                        key_insights={"summary": "测试记忆"},
                        token_count=2100,
                        provider="native",
                        source_start_message_id=uuid.UUID(turns[0].message_id),
                        source_end_message_id=uuid.UUID(turns[-1].message_id),
                        index_status="ready",
                    )
                    provider_db.add(memory)
                    provider_db.commit()
                    provider_db.refresh(memory)
                    return MemoryWriteResult(
                        operation_id=operation_id,
                        status="succeeded",
                        memory_ids=[str(memory.id)],
                    )
                finally:
                    provider_db.close()

        previous = memory_module._memory_provider
        memory_module._memory_provider = SuccessfulProvider()
        try:
            memory_module._execute_memory_job(str(job.id))
        finally:
            memory_module._memory_provider = previous

        self.db.expire_all()
        completed = self.db.query(MemoryJob).filter(MemoryJob.id == job.id).one()
        self.assertEqual(completed.status, "succeeded")
        self.assertIsNotNone(completed.memory_id)
        self.assertEqual(
            self.db.query(MemoryJob).filter(
                MemoryJob.session_id == self.session_id
            ).count(),
            1,
        )

    def test_chat_endpoint_persists_and_replays_one_turn(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from core.security import create_access_token
        from router.chat_router import get_services, router as chat_router

        class FakeChatService:
            def retrieve_from_web(self, question):
                return []

            def rerank_documents(self, question, documents):
                return documents

            def get_chat_completion(self, **kwargs):
                kwargs["on_complete"]("服务端回答", "")
                yield 'event: message\ndata: {"role":"assistant","content":"服务端回答","thinking":false}\n\n'
                yield "event: end\ndata: [DONE]\n\n"

        app = FastAPI()
        app.include_router(chat_router)
        app.dependency_overrides[get_services] = lambda: {
            "chat_service": FakeChatService(),
            "default_dataset_id": "unused",
        }
        client = TestClient(app)
        token = create_access_token({"sub": str(self.user_id), "username": self.user.username})
        request = {
            "session_id": str(self.session_id),
            "turn_id": str(self.turn_id),
            "question": "接口测试",
            "search_knowledge": False,
            "search_web": False,
        }
        headers = {"Authorization": f"Bearer {token}"}

        first = client.post("/chat/completion", json=request, headers=headers)
        second = client.post("/chat/completion", json=request, headers=headers)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("服务端回答", first.text)
        self.assertIn('"replayed": true', second.text)
        self.db.expire_all()
        self.assertEqual(
            self.db.query(ChatMessage).filter(
                ChatMessage.session_id == self.session_id
            ).count(),
            2,
        )

    def test_research_endpoint_persists_and_replays_one_turn(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from core.security import create_access_token
        from router.research_router import (
            get_research_service,
            router as research_router,
        )

        class FakeResearchV2Service:
            async def research(self, **kwargs):
                yield (
                    'data: {"type":"research_complete",'
                    '"final_report":"研究报告",'
                    '"references":[{"title":"来源","link":"https://example.test"}]}'
                    "\n\n"
                )
                yield "data: [DONE]\n\n"

        app = FastAPI()
        app.include_router(research_router)
        app.dependency_overrides[get_research_service] = lambda: {
            "research_service": object()
        }
        client = TestClient(app)
        token = create_access_token({
            "sub": str(self.user_id),
            "username": self.user.username,
        })
        request = {
            "session_id": str(self.session_id),
            "turn_id": str(self.turn_id),
            "query": "研究接口测试",
            "version": "v2",
            "search_modes": [],
        }
        headers = {"Authorization": f"Bearer {token}"}

        with patch(
            "router.research_router.get_research_service_v2",
            return_value=FakeResearchV2Service(),
        ):
            first = client.post("/research/stream", json=request, headers=headers)
            second = client.post("/research/stream", json=request, headers=headers)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("研究报告", first.text)
        self.assertIn('"replayed": true', second.text)
        self.db.expire_all()
        messages = self.db.query(ChatMessage).filter(
            ChatMessage.session_id == self.session_id
        ).order_by(ChatMessage.created_at.asc()).all()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1].content, "研究报告")
        self.assertEqual(
            messages[-1].references_data["references"][0]["title"],
            "来源",
        )


if __name__ == "__main__":
    unittest.main()
