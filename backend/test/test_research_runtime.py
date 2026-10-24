"""3-8 项改造的无网络回归测试。"""

import asyncio
import copy
import json
import os
import sys
import types
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))
service_package = types.ModuleType("service")
service_package.__path__ = [str(APP_DIR / "service")]
sys.modules.setdefault("service", service_package)

from service.context_compaction import compact_messages
from service.research_control import (
    ResearchCancellationToken,
    ResearchCancelledError,
    cancellation_context,
)
from config.llm_config import get_config
from service.deep_research_v2.state import GRAPH_VERSION
from service.research_events import (
    ResearchEventPersistenceError,
    ResearchEventRecorder,
    ResearchRunLease,
    ResearchRunLeaseConflict,
    ResearchRunLeaseLost,
    acquire_research_run_lease,
    make_research_event,
    release_research_run_lease,
    replay_events,
)
from service.tool_runtime import (
    ToolCallContext,
    ToolDefinition,
    ToolErrorCode,
    ToolRisk,
    ToolRuntime,
    ToolRuntimeError,
)


class SearchArgs(BaseModel):
    query: str
    count: int = 5


class ToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_validates_and_executes(self):
        async def handler(args, context):
            return {"query": args.query, "count": args.count}

        result = await ToolRuntime().execute(
            ToolDefinition("search", "test", SearchArgs, handler),
            {"query": "保险", "count": 3},
        )
        self.assertEqual(result.value, {"query": "保险", "count": 3})
        self.assertGreaterEqual(result.duration_ms, 0)

    async def test_rejects_invalid_arguments_before_handler(self):
        called = False

        async def handler(args, context):
            nonlocal called
            called = True

        with self.assertRaises(ToolRuntimeError) as caught:
            await ToolRuntime().execute(
                ToolDefinition("search", "test", SearchArgs, handler),
                {"count": 3},
            )
        self.assertEqual(caught.exception.code, ToolErrorCode.INVALID_ARGUMENTS.value)
        self.assertFalse(called)


    async def test_timeout_and_approval_are_fail_closed(self):
        async def slow_handler(args, context):
            await asyncio.sleep(1)

        with self.assertRaises(ToolRuntimeError) as timeout:
            await ToolRuntime().execute(
                ToolDefinition("slow", "test", SearchArgs, slow_handler, timeout_seconds=0.01),
                {"query": "x"},
            )
        self.assertEqual(timeout.exception.code, ToolErrorCode.TIMEOUT.value)

        with self.assertRaises(ToolRuntimeError) as approval:
            await ToolRuntime().execute(
                ToolDefinition(
                    "write",
                    "test",
                    SearchArgs,
                    slow_handler,
                    risk=ToolRisk.WRITE,
                ),
                {"query": "x"},
            )
        self.assertEqual(approval.exception.code, ToolErrorCode.APPROVAL_REQUIRED.value)

    async def test_local_cancellation_stops_before_execution(self):
        token = ResearchCancellationToken(run_id="run-1", session_id="session-1")
        token.cancel()
        called = False

        async def handler(args, context):
            nonlocal called
            called = True

        with self.assertRaises(ToolRuntimeError) as caught:
            await ToolRuntime().execute(
                ToolDefinition("search", "test", SearchArgs, handler),
                {"query": "x"},
                ToolCallContext(cancellation_token=token),
            )
        self.assertEqual(caught.exception.code, ToolErrorCode.CANCELLED.value)
        self.assertFalse(called)


class LlmCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_failure_switches_to_fallback_model(self):
        import httpx
        from openai import APITimeoutError
        from service.deep_research_v2.agents import base as base_module
        from service.deep_research_v2.agents.base import BaseAgent

        requested_models = []
        requested_max_tokens = []

        class Completions:
            async def create(self, **kwargs):
                requested_models.append(kwargs["model"])
                requested_max_tokens.append(kwargs["max_tokens"])
                if len(requested_models) == 1:
                    raise APITimeoutError(request=httpx.Request("POST", "https://example.test"))
                message = types.SimpleNamespace(content="ok")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice], usage=None)

        class DummyAgent(BaseAgent):
            async def process(self, state):
                return state

        agent = DummyAgent.__new__(DummyAgent)
        agent.name = "Dummy"
        agent.model = "primary-model"
        agent.logger = __import__("logging").getLogger("test.llm-fallback")
        agent.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))

        env = {
            "LLM_AGENT_MAX_ATTEMPTS": "2",
            "LLM_FALLBACK_MODEL": "fallback-model",
            "LLM_MAX_OUTPUT_TOKENS": "3000",
        }
        with patch.dict(os.environ, env), \
             patch.object(base_module, "record_llm_call"), \
             patch.object(base_module, "flush_pending"), \
             patch.object(base_module.random, "uniform", return_value=0), \
             patch.object(base_module.asyncio, "sleep", return_value=None):
            result = await agent.call_llm("system", "user", json_mode=False)

        self.assertEqual(result, "ok")
        self.assertEqual(requested_models, ["primary-model", "fallback-model"])
        self.assertEqual(requested_max_tokens, [3000, 3000])

    async def test_async_llm_request_is_reaped_on_cancel(self):
        import logging
        from service.deep_research_v2.agents.base import BaseAgent

        cancelled = asyncio.Event()

        class Completions:
            async def create(self, **kwargs):
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        class DummyAgent(BaseAgent):
            async def process(self, state):
                return state

        agent = DummyAgent.__new__(DummyAgent)
        agent.name = "Dummy"
        agent.model = "fake-model"
        agent.logger = logging.getLogger("test.llm-cancel")
        agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=Completions())
        )
        token = ResearchCancellationToken("run-llm", "session-llm")

        async def cancel_soon():
            await asyncio.sleep(0.01)
            token.cancel()

        with cancellation_context(token):
            cancel_task = asyncio.create_task(cancel_soon())
            with self.assertRaises(ResearchCancelledError):
                await agent.call_llm("system", "user")
            await cancel_task

        self.assertTrue(cancelled.is_set())


class ContextCompactionTests(unittest.TestCase):
    def test_keeps_recent_turn_without_summarizing_older_turns(self):
        class FakeTokenCounter:
            @staticmethod
            def encode(text):
                return list(text or "")

            @staticmethod
            def decode(tokens):
                return "".join(tokens)

            @staticmethod
            def count_text(text):
                return len(text or "")

            @staticmethod
            def count_messages(messages):
                return sum(len(message.get("content", "")) for message in messages)

        messages = []
        for index in range(8):
            messages.extend([
                {"role": "user", "content": f"问题{index} " + "很长 " * 80},
                {"role": "assistant", "content": f"回答{index} " + "内容 " * 80},
            ])
        result = compact_messages(
            messages,
            token_limit=260,
            max_messages=8,
            token_counter=FakeTokenCounter(),
        )
        rendered = "\n".join(item["content"] for item in result.messages)

        self.assertLessEqual(result.compacted_tokens, 260)
        self.assertEqual(result.summarized_messages, 0)
        self.assertIn("问题7", rendered)
        self.assertIn("回答7", rendered)
        self.assertNotIn("较早对话摘要", rendered)
        self.assertNotIn("问题0", rendered)
        self.assertEqual([item["role"] for item in result.messages], ["user", "assistant"])

    def test_preserves_first_system_message_when_message_count_is_limited(self):
        class FakeTokenCounter:
            @staticmethod
            def encode(text):
                return list(text or "")

            @staticmethod
            def decode(tokens):
                return "".join(tokens)

            @staticmethod
            def count_text(text):
                return len(text or "")

            @staticmethod
            def count_messages(messages):
                return sum(len(message.get("content", "")) for message in messages)

        messages = [{"role": "system", "content": "必须保留的语义摘要"}]
        for index in range(6):
            messages.extend([
                {"role": "user", "content": f"问题{index}"},
                {"role": "assistant", "content": f"回答{index}"},
            ])

        result = compact_messages(
            messages,
            token_limit=1000,
            max_messages=4,
            token_counter=FakeTokenCounter(),
        )

        # 预算为四条也不必填满：语义摘要 + 最新完整问答，共三条。
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.messages[0]["role"], "system")
        self.assertIn("必须保留的语义摘要", result.messages[0]["content"])
        self.assertEqual(result.messages[-1]["content"], "回答5")


class MultiTurnResearchContextTests(unittest.TestCase):
    def test_initial_state_uses_resolved_query_and_keeps_audit_context(self):
        from service.deep_research_v2.state import create_initial_state

        state = create_initial_state(
            "继续分析海外机会",
            "session-1",
            ["example.gov.cn"],
            research_context={
                "original_query": "继续分析海外机会",
                "resolved_query": "分析中国储能企业未来三年的海外机会",
                "conversation_context": [
                    {"role": "user", "content": "研究中国储能市场"},
                ],
                "memory_context": [
                    {
                        "memory_id": "memory-1",
                        "memory_type": "summary",
                        "content": "用户关注中国储能市场",
                    },
                ],
                "used_memory_ids": ["memory-1"],
                "is_follow_up": True,
            },
        )

        self.assertEqual(
            state["query"], "分析中国储能企业未来三年的海外机会"
        )
        self.assertEqual(state["original_query"], "继续分析海外机会")
        self.assertEqual(
            state["context_resolution"]["used_memory_ids"], ["memory-1"]
        )

    def test_planning_context_marks_history_as_non_evidence(self):
        from service.research_context_service import render_planning_context

        rendered = render_planning_context({
            "conversation_context": [
                {"role": "assistant", "content": "上一轮报告"},
            ],
            "memory_context": [
                {
                    "memory_id": "memory-1",
                    "memory_type": "summary",
                    "content": "历史摘要",
                },
            ],
        })

        self.assertIn("不是本轮事实证据", rendered)
        self.assertIn("memory_id=memory-1", rendered)

    def test_query_resolution_filters_unretrieved_memory_ids(self):
        from service.research_context_service import (
            ResearchContext,
            ResearchContextService,
        )

        response = types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=json.dumps({
                "topic_relation": "continue", "depends_on_history": True,
                "operations": [{"kind": "research", "quote": "继续分析海外机会"}],
                "resolved_query": "分析中国储能企业的海外机会",
                "used_memory_ids": ["memory-1", "fabricated-memory"],
                "assumptions": ["海外指中国企业出海市场"],
                "unresolved_references": [],
            }, ensure_ascii=False))
        )])
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **_: response)
            )
        )
        context = ResearchContext(
            original_query="继续分析海外机会",
            resolved_query="继续分析海外机会",
            memory_context=[{
                "memory_id": "memory-1",
                "memory_type": "summary",
                "content": "用户此前研究中国储能市场",
            }],
        )

        with patch(
            "service.research_context_service.create_openai_client",
            return_value=client,
        ):
            ResearchContextService.__new__(ResearchContextService)._resolve_query(
                context
            )

        self.assertTrue(context.is_follow_up)
        self.assertEqual(
            context.resolved_query, "分析中国储能企业的海外机会"
        )
        self.assertEqual(context.used_memory_ids, ["memory-1"])

    def test_architect_requires_current_run_fact_verification(self):
        from service.deep_research_v2.agents.architect import ChiefArchitect

        self.assertIn("所有影响研究结论的事实性内容", ChiefArchitect.PLANNING_PROMPT)
        self.assertIn("重新检索、交叉验证", ChiefArchitect.PLANNING_PROMPT)


class ResearchEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_recording_has_monotonic_sequence_and_replay_shape(self):
        recorder = ResearchEventRecorder(
            session_id="session-1",
            run_id="run-1",
            turn_id=None,
            persist=False,
        )
        first = recorder.record({"type": "research_start", "query": "市场"})
        second = recorder.record({"type": "phase", "phase": "researching"})

        self.assertEqual([first.seq, second.seq], [1, 2])
        self.assertEqual(first.to_wire()["data"], {"query": "市场"})
        self.assertEqual(first.to_wire()["query"], "市场")

        stored = types.SimpleNamespace(
            event_id=first.event_id,
            event_type=first.type,
            session_id=first.session_id,
            run_id=first.run_id,
            turn_id=None,
            seq=first.seq,
            occurred_at=datetime.fromisoformat(first.timestamp.replace("Z", "+00:00")).replace(tzinfo=None),
            payload=first.data,
        )
        replayed = list(replay_events([stored]))
        self.assertEqual(replayed[0]["type"], "research_start")
        self.assertEqual(replayed[0]["seq"], 1)
        self.assertEqual(replayed[0]["data"], {"query": "市场"})

    async def test_service_transcript_snapshot_without_real_llm(self):
        """服务层把图的 custom 事件逐条落账再发 SSE，最后才发完成事件。"""
        from service.deep_research_v2 import service as service_module

        service = service_module.DeepResearchV2Service.__new__(
            service_module.DeepResearchV2Service
        )
        service.config = get_config()
        service.checkpointer = object()

        class FakeGraph:
            async def astream(self, graph_input, config=None, **kwargs):
                yield {"type": "phase", "phase": "planning", "content": "规划"}
                yield {"type": "phase", "phase": "researching", "content": "搜索"}

            async def aget_state(self, config):
                return types.SimpleNamespace(
                    next=(),
                    config={"configurable": {"checkpoint_id": "cp-1"}},
                    values={"research": {
                        "status": "completed",
                        "final_report": "报告",
                        "allowed_domains": ["example.gov.cn"],
                    }},
                )

        def fake_commit(snapshot):
            return make_research_event(
                {"type": "research_complete", "final_report": "报告"},
                session_id="session-1", run_id="run-1", turn_id=None, seq=4,
            )

        with (
            patch.dict(os.environ, {"RESEARCH_ALLOWED_DOMAINS": "example.gov.cn"}),
            patch.object(service_module, "build_graph", return_value=FakeGraph()),
            patch.object(service_module, "create_agents", return_value={}),
            patch.object(service_module, "make_commit_result", return_value=fake_commit),
            patch("service.deep_research_v2.followup_context.latest_parent_run", return_value=None),
        ):
            chunks = []
            async for chunk in service.research(
                query="行业问题",
                session_id="session-1",
                user_id="user-1",
                run_id="run-1",
                persist_events=False,
                allowed_domains=["example.gov.cn"],
            ):
                chunks.append(chunk)

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks[:-1]
        ]
        self.assertEqual(
            [(item["seq"], item["type"]) for item in payloads],
            [(1, "research_start"), (2, "phase"), (3, "phase"), (4, "research_complete")],
        )
        self.assertEqual(payloads[0]["data"]["graph_version"], GRAPH_VERSION)
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

    async def test_persistence_failure_closes_stream_without_sending_event(self):
        """事件落账失败必须停止执行并关闭流，不能当普通进度消息丢掉。"""
        from service.deep_research_v2 import service as service_module

        service = service_module.DeepResearchV2Service.__new__(
            service_module.DeepResearchV2Service
        )
        service.config = get_config()
        service.checkpointer = object()

        class FakeGraph:
            async def astream(self, graph_input, config=None, **kwargs):
                yield {"type": "phase", "phase": "planning"}

            async def aget_state(self, config):
                return types.SimpleNamespace(next=("write",), config={}, values={})

        lease = ResearchRunLease(
            run_id="run-fail-closed",
            session_id="session-fail-closed",
            owner_id="owner-1",
            fencing_token=1,
            lease_until=datetime.utcnow() + timedelta(minutes=1),
        )

        with (
            patch.dict(os.environ, {"RESEARCH_ALLOWED_DOMAINS": "example.gov.cn", "SESSION_EVIDENCE_ENABLED": "0"}),
            patch.object(service_module, "build_graph", return_value=FakeGraph()),
            patch.object(service_module, "create_agents", return_value={}),
            patch.object(service_module, "make_commit_result", return_value=lambda s: None),
            patch.object(service_module, "for_execution", return_value=types.SimpleNamespace(conn=None)),
            patch.object(service_module, "StorageCoordinator", return_value=types.SimpleNamespace(
                seal=AsyncMock(side_effect=lambda value, **kwargs: value))),
            patch("service.deep_research_v2.followup_context.latest_parent_run", return_value=None),
            patch.object(
                ResearchEventRecorder,
                "record",
                side_effect=ResearchEventPersistenceError("injected db failure"),
            ),
            patch.object(
                service_module, "release_research_run_lease", return_value=True
            ),
        ):
            chunks = []
            with self.assertRaises(ResearchEventPersistenceError):
                async for chunk in service.research(
                    query="行业问题",
                    session_id=lease.session_id,
                    user_id="user-1",
                    run_id=lease.run_id,
                    lease=lease,
                    allowed_domains=["example.gov.cn"],
                ):
                    chunks.append(chunk)

        self.assertEqual(chunks, [])


@unittest.skipUnless(
    os.getenv("RUN_DB_INTEGRATION_TESTS") == "1",
    "set RUN_DB_INTEGRATION_TESTS=1 to exercise PostgreSQL",
)
class ResearchEventPostgreSQLTests(unittest.TestCase):
    def setUp(self):
        self.run_id = f"event-reliability-{uuid.uuid4()}"
        self.session_id = f"session-{uuid.uuid4()}"

    def tearDown(self):
        from core.database import SessionLocal
        from models.research import AgentEvent, ResearchCheckpoint, ResearchRun

        db = SessionLocal()
        try:
            db.query(ResearchCheckpoint).filter(
                ResearchCheckpoint.run_id == self.run_id
            ).delete(synchronize_session=False)
            db.query(AgentEvent).filter(
                AgentEvent.run_id == self.run_id
            ).delete(synchronize_session=False)
            db.query(ResearchRun).filter(
                ResearchRun.run_id == self.run_id
            ).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()

    def test_concurrent_recording_allocates_unique_monotonic_sequences(self):
        from core.database import SessionLocal
        from models.research import AgentEvent

        lease = acquire_research_run_lease(
            run_id=self.run_id,
            session_id=self.session_id,
            user_id=None,
            owner_id="owner-a",
        )
        recorder = ResearchEventRecorder(
            session_id=self.session_id,
            run_id=self.run_id,
            persist=True,
            lease=lease,
        )
        with ThreadPoolExecutor(max_workers=8) as pool:
            envelopes = list(pool.map(
                lambda index: recorder.record({"type": "phase", "index": index}),
                range(20),
            ))

        self.assertEqual(sorted(event.seq for event in envelopes), list(range(1, 21)))
        db = SessionLocal()
        try:
            rows = db.query(AgentEvent).filter(
                AgentEvent.run_id == self.run_id
            ).order_by(AgentEvent.seq).all()
            self.assertEqual([row.seq for row in rows], list(range(1, 21)))
        finally:
            db.close()
        self.assertTrue(release_research_run_lease(lease, "paused"))

    def test_lease_conflict_and_fencing_reject_stale_writer(self):
        lease_a = acquire_research_run_lease(
            run_id=self.run_id,
            session_id=self.session_id,
            user_id=None,
            owner_id="owner-a",
        )
        with self.assertRaises(ResearchRunLeaseConflict):
            acquire_research_run_lease(
                run_id=self.run_id,
                session_id=self.session_id,
                user_id=None,
                owner_id="owner-b",
            )

        self.assertTrue(release_research_run_lease(lease_a, "paused"))
        lease_b = acquire_research_run_lease(
            run_id=self.run_id,
            session_id=self.session_id,
            user_id=None,
            owner_id="owner-b",
        )
        self.assertGreater(lease_b.fencing_token, lease_a.fencing_token)

        stale = ResearchEventRecorder(
            session_id=self.session_id,
            run_id=self.run_id,
            persist=True,
            lease=lease_a,
        )
        with self.assertRaises(ResearchRunLeaseLost):
            stale.record({"type": "phase", "phase": "stale"})

        current = ResearchEventRecorder(
            session_id=self.session_id,
            run_id=self.run_id,
            persist=True,
            lease=lease_b,
        )
        self.assertEqual(current.record({"type": "phase"}).seq, 1)
        self.assertTrue(release_research_run_lease(lease_b, "paused"))

    def test_checkpoint_and_checkpoint_event_commit_atomically(self):
        from core.database import SessionLocal
        from models.research import AgentEvent, ResearchCheckpoint
        from service.checkpoint_service import CheckpointService

        lease = acquire_research_run_lease(
            run_id=self.run_id,
            session_id=self.session_id,
            user_id=None,
            owner_id="owner-checkpoint",
        )
        recorder = ResearchEventRecorder(
            session_id=self.session_id,
            run_id=self.run_id,
            persist=True,
            lease=lease,
        )
        state = {
            "session_id": self.session_id,
            "query": "测试原子检查点",
            "phase": "researching",
            "iteration": 0,
            "resume_cursor": {
                "version": 1,
                "next_stage": "analyzing",
                "exact": True,
            },
            "_run_id": self.run_id,
            "_lease_owner_id": lease.owner_id,
            "_fencing_token": lease.fencing_token,
        }
        result = CheckpointService().save_checkpoint(
            self.session_id,
            state,
            event_recorder=recorder,
            event_payload={
                "type": "checkpoint_saved",
                "resume_stage": "analyzing",
            },
        )
        self.assertIsInstance(result, tuple)

        db = SessionLocal()
        try:
            checkpoint = db.query(ResearchCheckpoint).filter(
                ResearchCheckpoint.run_id == self.run_id
            ).one()
            event = db.query(AgentEvent).filter(
                AgentEvent.run_id == self.run_id
            ).one()
            self.assertEqual(checkpoint.last_event_seq, event.seq)
            self.assertEqual(event.event_type, "checkpoint_saved")
        finally:
            db.close()
        self.assertTrue(release_research_run_lease(lease, "paused"))

    def test_checkpoint_rolls_back_when_event_insert_fails(self):
        from core.database import SessionLocal
        from models.research import ResearchCheckpoint
        from service.checkpoint_service import CheckpointService

        lease = acquire_research_run_lease(
            run_id=self.run_id,
            session_id=self.session_id,
            user_id=None,
            owner_id="owner-rollback",
        )
        recorder = ResearchEventRecorder(
            session_id=self.session_id,
            run_id=self.run_id,
            persist=True,
            lease=lease,
        )
        state = {
            "session_id": self.session_id,
            "query": "测试事务回滚",
            "phase": "planning",
            "_run_id": self.run_id,
            "_lease_owner_id": lease.owner_id,
            "_fencing_token": lease.fencing_token,
        }
        with patch.object(
            recorder,
            "record_in_session",
            side_effect=RuntimeError("injected event insert failure"),
        ):
            result = CheckpointService().save_checkpoint(
                self.session_id,
                state,
                event_recorder=recorder,
                event_payload={"type": "checkpoint_saved"},
            )
        self.assertIsNone(result)

        db = SessionLocal()
        try:
            checkpoint = db.query(ResearchCheckpoint).filter(
                ResearchCheckpoint.run_id == self.run_id
            ).first()
            self.assertIsNone(checkpoint)
        finally:
            db.close()
        self.assertTrue(release_research_run_lease(lease, "paused"))


class CheckpointCursorTests(unittest.TestCase):
    """历史快照的只读读取。新运行的执行位置由 LangGraph 检查点负责，
    对应覆盖见 backend/eval/tests/test_单主链路图.py。"""

    def test_checkpoint_uses_latest_event_sequence_for_its_run(self):
        from service.checkpoint_service import CheckpointService

        class Query:
            def filter(self, *args):
                return self

            def scalar(self):
                return 17

        class Db:
            def query(self, *args):
                return Query()

        self.assertEqual(CheckpointService._latest_event_seq(Db(), "run-17"), 17)
        self.assertEqual(CheckpointService._latest_event_seq(Db(), None), 0)

    def test_checkpoint_wire_shape_exposes_event_cursor(self):
        from models.research import ResearchCheckpoint

        checkpoint = ResearchCheckpoint(
            session_id="session-1",
            run_id="run-1",
            last_event_seq=9,
            query="行业问题",
            phase="researching",
            iteration=1,
            state_json={
                "resume_cursor": {
                    "version": 1,
                    "next_stage": "analyzing",
                    "exact": True,
                },
            },
        )
        payload = checkpoint.to_dict()
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["last_event_seq"], 9)
        self.assertEqual(payload["resume_stage"], "analyzing")
        self.assertEqual(payload["checkpoint_version"], 1)
        self.assertTrue(payload["exact_resume"])
        # 历史快照必须明确标记为旧版本，避免被当成新图可续跑的状态。
        self.assertTrue(payload["legacy_snapshot"])


# 阶段游标与旧工作流的精确续跑用例已删除：执行位置改由 LangGraph 检查点
# 负责，首跑/续跑、返工路由、空报告与预算恢复的覆盖见
# backend/eval/tests/test_单主链路图.py。


if __name__ == "__main__":
    unittest.main()
