"""DRB 执行器与当前研究服务契约回归；不连接数据库、Redis 或模型。"""

import asyncio
import io
import json
import os
import sys
import unittest
from contextlib import asynccontextmanager, nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
               str(Path(__file__).resolve().parents[2] / "app")]

from eval.scripts import run_deepresearch_bench as drb


TASK = {"id": 1, "prompt": "比较研究方法", "language": "zh", "topic": "Science"}


def frame(event_type, **payload):
    return "data: " + json.dumps({"type": event_type, "data": payload}, ensure_ascii=False) + "\n\n"


class StreamService:
    """有明确新接口签名，无 graph 属性，确保不会误用旧接口。"""

    def __init__(self, chunks=(), wait=False):
        self.chunks = chunks
        self.wait = wait
        self.calls = []
        self.closed = 0

    async def research(self, *, query, session_id, user_id, run_id, resume=False):
        self.calls.append(dict(query=query, session_id=session_id,
                               user_id=user_id, run_id=run_id, resume=resume))
        try:
            for chunk in self.chunks:
                yield chunk
            if self.wait:
                await asyncio.Event().wait()
        finally:
            self.closed += 1


class RunnerContractTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, service, timeout=2, no_refs=False):
        return await drb.run_one(service, TASK, "接口回归", "20260925_120000", timeout, no_refs)

    async def test_sse_completion_preserves_report_quality_and_run_identity(self):
        service = StreamService([
            frame("research_start"),
            frame("research_complete", final_report="# 报告\n完整正文",
                  references=[{"url": "https://who.int/a", "title": "原文"}],
                  quality_score=8, quality_score_status="valid", quality_verdict="pass",
                  citation_verification="not_executed", completion_reason="approved",
                  allowed_domains=["who.int"], checkpoint_id="checkpoint-1"),
            "data: [DONE]\n\n",
        ])
        record = await self.run_case(service)
        self.assertIsNone(record["_meta"]["error"])
        self.assertIn("完整正文", record["article"])
        self.assertIn("https://who.int/a", record["article"])
        self.assertEqual(record["_meta"]["quality_score"], 8)
        self.assertEqual(record["_meta"]["quality_verdict"], "pass")
        self.assertEqual(record["_meta"]["citation_verification"], "not_executed")
        self.assertEqual(record["_meta"]["checkpoint_id"], "checkpoint-1")
        self.assertEqual(record["_meta"]["run_id"], "20260925_120000")
        self.assertEqual(record["_meta"]["research_run_id"], service.calls[0]["run_id"])
        self.assertNotEqual(service.calls[0]["run_id"], record["_meta"]["run_id"])
        self.assertEqual(service.closed, 1)
        await self.run_case(service)
        self.assertNotEqual(service.calls[0]["run_id"], service.calls[1]["run_id"])

    async def test_flat_event_and_no_refs_remain_supported(self):
        event = {"type": "research_complete", "final_report": "正文",
                 "references": [{"url": "https://who.int/a"}]}
        record = await self.run_case(StreamService(["data: " + json.dumps(event) + "\n\n"]),
                                     no_refs=True)
        self.assertEqual(record["article"], "正文")
        self.assertIsNone(record["_meta"]["error"])

    async def test_terminal_failures_are_recorded_and_stream_is_closed(self):
        for event_type in ("error", "research_paused", "research_cancelled"):
            with self.subTest(event_type=event_type):
                service = StreamService([frame(event_type, content="注入终态错误")])
                record = await self.run_case(service)
                self.assertEqual(record["article"], "")
                self.assertIn(event_type, record["_meta"]["error"])
                self.assertIn("注入终态错误", record["_meta"]["error"])
                self.assertEqual(service.closed, 1)

    async def test_missing_or_empty_completion_cannot_be_success(self):
        cases = [[], ["data: [DONE]\n\n"], [frame("research_complete", final_report=" ")]]
        for chunks in cases:
            with self.subTest(chunks=chunks):
                record = await self.run_case(StreamService(chunks))
                self.assertEqual(record["article"], "")
                self.assertIn("research_complete", record["_meta"]["error"])

    async def test_timeout_closes_stream_and_returns_failure(self):
        service = StreamService(wait=True)
        record = await self.run_case(service, timeout=0.02)
        self.assertEqual(record["article"], "")
        self.assertIn("timeout", record["_meta"]["error"])
        self.assertEqual(service.closed, 1)

    async def test_malformed_event_and_post_completion_error_are_failures(self):
        cases = [["data: invalid-json\n\n"], ["data: []\n\n"],
                 [frame("research_complete", final_report="正文"), frame("error", content="交付错误")]]
        for chunks in cases:
            with self.subTest(chunks=chunks):
                service = StreamService(chunks)
                record = await self.run_case(service)
                self.assertEqual(record["article"], "")
                self.assertIsNotNone(record["_meta"]["error"])
                self.assertEqual(service.closed, 1)

    async def test_batch_closes_service_and_surfaces_output_write_failure(self):
        lifecycle = []

        @asynccontextmanager
        async def factory(max_iterations):
            lifecycle.append(("open", max_iterations))
            try:
                yield StreamService([frame("research_complete", final_report="正文")])
            finally:
                lifecycle.append(("close", max_iterations))

        args = SimpleNamespace(model_name="接口回归", lang="zh", topic=None, limit=1,
                               ids=None, require_pass=False, concurrency=1, timeout=2,
                               no_refs=True, dry_run=False, max_iterations=2)
        with patch.object(drb, "load_queries", return_value=[TASK]), patch.object(
            drb, "load_done", return_value={}
        ), patch.object(drb, "benchmark_service", factory), patch.object(
            drb, "append_result", side_effect=OSError("注入写盘失败")
        ), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "注入写盘失败"):
                await drb.run_bench(args)
        self.assertEqual(lifecycle, [("open", 2), ("close", 2)])


class CurrentServiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_source_configuration_allows_whole_web_and_closes_pool(self):
        from service.deep_research_v2 import checkpointer as checkpoint_module
        from service.deep_research_v2 import service as service_module

        lifecycle = []

        @asynccontextmanager
        async def lifespan(app):
            lifecycle.append("open")
            try:
                yield object()
            finally:
                lifecycle.append("close")

        with patch.object(checkpoint_module, "checkpointer_lifespan", lifespan), patch.dict(
            os.environ, {"RESEARCH_ALLOWED_DOMAINS": ""}
        ), patch.object(service_module, "create_agents") as create_agents:
            async with drb.benchmark_service(2) as service:
                self.assertEqual(service.resolve_allowed_domains(), [])
        create_agents.assert_not_called()
        self.assertEqual(lifecycle, ["open", "close"])

    async def test_checkpoint_lifecycle_and_config_override_are_isolated(self):
        from service.deep_research_v2 import checkpointer as checkpoint_module
        from service.deep_research_v2 import service as service_module
        from config.llm_config import get_config

        shared = get_config()
        old_iterations = shared.research.max_iterations
        saver = object()
        lifecycle = []

        @asynccontextmanager
        async def lifespan(app):
            lifecycle.append("open")
            try:
                yield saver
            finally:
                lifecycle.append("close")

        with patch.object(checkpoint_module, "checkpointer_lifespan", lifespan), patch.dict(
            os.environ, {"RESEARCH_ALLOWED_DOMAINS": "who.int"}
        ):
            with self.assertRaisesRegex(RuntimeError, "模拟批次中断"):
                async with drb.benchmark_service(1) as service:
                    self.assertIsInstance(service, service_module.DeepResearchV2Service)
                    self.assertIs(service.checkpointer, saver)
                    self.assertEqual(service.config.research.max_iterations, 1)
                    self.assertIsNot(service.config, shared)
                    self.assertEqual(shared.research.max_iterations, old_iterations)
                    raise RuntimeError("模拟批次中断")
        self.assertEqual(lifecycle, ["open", "close"])

    async def test_runner_executes_actual_service_and_graph(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from service.deep_research_v2 import service as service_module
        from service.deep_research_v2.delivery import build_complete_event, checkpoint_id_of
        from service.deep_research_v2.state import ResearchPhase
        from service.research_events import ResearchEventRecorder
        from service import research_control

        class Agent:
            llm_call_count = 0
            llm_total_tokens = 0

            def __init__(self, name):
                self.name = name

            async def process(self, state):
                if self.name == "writer":
                    state["final_report"] = "# 报告\n由真实主图交付"
                elif self.name == "critic":
                    state["quality_verdict"] = "pass"
                    state["quality_score"] = 8
                    state["quality_score_status"] = "valid"
                    state["phase"] = ResearchPhase.COMPLETED.value
                return state

        agents = {role: Agent(role) for role in ("architect", "scout", "data_analyst", "writer", "critic")}

        def recorder(**kwargs):
            return ResearchEventRecorder(**{**kwargs, "persist": False, "lease": None})

        def commit_factory(**kwargs):
            def commit(snapshot):
                self.assertEqual(snapshot.next, ())
                return kwargs["recorder"].record(build_complete_event(
                    snapshot.values["research"], run_id=kwargs["run_id"],
                    checkpoint_id=checkpoint_id_of(snapshot)))
            return commit

        service = service_module.DeepResearchV2Service(checkpointer=InMemorySaver())
        offline_service = SimpleNamespace(research=lambda **kwargs: service.research(**kwargs, persist_events=False))
        with patch.dict(os.environ, {"RESEARCH_ALLOWED_DOMAINS": "who.int"}), patch.object(
            service_module, "create_agents", return_value=agents
        ), patch.object(service_module, "acquire_research_run_lease", return_value=None), patch.object(
            service_module, "ResearchEventRecorder", side_effect=recorder
        ), patch.object(service_module, "make_commit_result", side_effect=commit_factory), patch.object(
            service_module, "trace_context", side_effect=lambda **kwargs: nullcontext()
        ), patch.object(service_module, "flush_pending"), patch.object(
            research_control, "cache", Mock(get=Mock(return_value=None))
        ):
            record = await drb.run_one(offline_service, TASK, "真实图回归", "20260925_120000", 5, True)
        self.assertIsNone(record["_meta"]["error"])
        self.assertIn("由真实主图交付", record["article"])
        self.assertEqual(record["_meta"]["quality_verdict"], "pass")
        state = await service.read_state(record["_meta"]["research_run_id"])
        self.assertEqual(state["pending_nodes"], [])
        self.assertEqual(state["research"]["final_report"], record["article"])


if __name__ == "__main__":
    unittest.main()
