"""LLM 会话语义摘要的离线结构化回归测试。"""

import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from service.semantic_context_service import (
    SemanticContextService,
    render_semantic_context,
)


def message(message_id, role, content):
    return SimpleNamespace(
        id=message_id,
        role=role,
        content=content,
        created_at=datetime(2026, 8, 30, 12, 0),
    )


class SemanticContextServiceTests(unittest.TestCase):
    def test_old_structured_summary_still_renders_for_legacy_mode(self):
        rendered = render_semantic_context({
            "overview": "分析市场", "items": [{
                "id": "ctx-old", "category": "constraint", "text": "不使用预测值",
                "source_message_ids": ["m1"], "evidence_quote": "不要预测值",
                "source_role": "user", "status": "active",
            }],
        })
        self.assertIn("不使用预测值", rendered)
        self.assertIn("不是本轮研究事实证据", rendered)

    def test_renderer_actually_excludes_superseded_items(self):
        rendered = render_semantic_context({"items": [{
            "id": "ctx-old", "category": "scope", "text": "只看2024年",
            "source_message_ids": ["m1"], "evidence_quote": "只看2024年",
            "source_role": "user", "status": "superseded", "superseded_by": "ctx-new",
        }]})
        self.assertEqual(rendered, "")

    def test_rolling_summary_uses_langchain_wrapper(self):
        rendered = render_semantic_context({"schema_version": 2, "summary": "用户目标和有效要求"})
        self.assertEqual(rendered, "Here is a summary of the conversation to date:\n\n用户目标和有效要求")

    def test_structured_v1_state_rebuilds_from_original_messages(self):
        from service.session_summarization import SessionSummaryConfig
        from test_session_memory import FakeTokenCounter
        prompts = []
        service = SemanticContextService(
            config=SessionSummaryConfig(),
            token_counter=FakeTokenCounter(),
            completion=lambda prompt: prompts.append(prompt) or "新的完整摘要",
        )
        result = service.update_summary({"schema_version": 1, "overview": "旧概览"},
            [message("m1", "user", "新的原始请求")])
        self.assertEqual(result.schema_version, 2)
        self.assertEqual(result.summary, "新的完整摘要")
        self.assertIn("新的原始请求", prompts[0])
        self.assertNotIn("旧概览", prompts[0])


class SemanticContextWorkerTests(unittest.TestCase):
    def test_memory_worker_processes_semantic_job_without_memory_provider(self):
        from unittest.mock import Mock, patch
        from service import memory_service

        job = SimpleNamespace(
            id=uuid.uuid4(),
            status="running",
            operation_type="semantic_context",
            provider="native",
            user_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            payload={"force_followup": False},
            locked_at=datetime.now(timezone.utc),
            last_error="old",
        )

        class Query:
            def filter(self, *_):
                return self

            def first(self):
                return job

        db = Mock()
        db.query.return_value = Query()
        session = SimpleNamespace(id=job.session_id, user_id=job.user_id)
        cache = Mock()

        with (
            patch.object(memory_service, "SessionLocal", return_value=db),
            patch.object(memory_service, "get_memory_provider") as provider,
            patch(
                "service.semantic_context_service.execute_semantic_context_job",
                return_value=session,
            ) as execute,
            patch(
                "service.semantic_context_service.enqueue_semantic_context_if_due"
            ) as enqueue,
            patch(
                "service.session_memory_service.get_session_memory_service",
                return_value=cache,
            ),
        ):
            memory_service._execute_memory_job(str(job.id))

        provider.assert_not_called()
        execute.assert_called_once_with(db, job)
        enqueue.assert_called_once()
        self.assertEqual(job.status, "succeeded")
        self.assertIsNone(job.locked_at)
        self.assertIsNone(job.last_error)
        db.commit.assert_called_once_with()
        cache.invalidate.assert_called_once_with(
            user_id=str(job.user_id),
            session_id=str(job.session_id),
        )


if __name__ == "__main__":
    unittest.main()
