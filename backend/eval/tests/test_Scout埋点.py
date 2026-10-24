"""使用真实 Langfuse SDK 和内存导出器验证 Scout；不请求外部服务。"""

import asyncio
import copy
import hashlib
import json
import logging
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND / "app"))
sys.path.insert(0, str(BACKEND))

import httpx
from eval.runtime import langfuse_tracing as tracing
from eval.runtime import tracer
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.state import create_initial_state
from service.deep_research_v2.tool_contracts import WebReadBatchToolArgs, WebSearchToolArgs
from service.tool_runtime import ToolCallContext, ToolRuntime
from service.web_reader_service import WebReadResult

CAPTURED_OBSERVATIONS = []
URL = "https://example.gov.cn/report"
CONTENT = "研究原始正文。" * 2000 + "正文尾部标记"
REAL_ASYNC_CLIENT = httpx.AsyncClient


def make_state(run_id="run-1"):
    state = create_initial_state("测试研究", "session-test", ["example.gov.cn"])
    state.update(_run_id=run_id, phase="researching", outline=[{
        "id": "sec_1", "title": "测试章节", "description": "测试范围",
        "search_queries": ["测试检索"], "status": "pending",
    }], hypotheses=[{"id": "h_1", "content": "待验证假设"}])
    return state


def make_scout():
    scout = DeepScout.__new__(DeepScout)
    scout.name, scout.model = "DeepScout", "fixture-model"
    scout.logger = logging.getLogger("test.scout.tracing")
    scout.tool_runtime = ToolRuntime()
    scout.tool_definitions = scout._build_tool_definitions()
    scout._fetch_stock_data_if_relevant = AsyncMock()
    scout.search_api_key = "test-secret-must-not-be-exported"
    scout.search_cache, scout.fact_fingerprints = {}, {}
    scout._search_rate_lock = asyncio.Lock()
    scout._last_search_at = scout._min_search_interval = 0
    scout._search_circuit_open_until = scout._search_failures = 0
    scout._search_circuit_threshold, scout._search_circuit_seconds = 3, 60
    scout._max_queries_per_section, scout._max_total_searches = 1, 20
    scout._max_deep_queries_per_section = 1
    scout.web_reader = SimpleNamespace(read_many=AsyncMock(return_value=[
        WebReadResult(url=url, final_url=url, content=CONTENT + ("不同来源的独立正文。" if index else ""),
                      fetch_status="success", extractor="fixture", fetched_at=1)
        for index, url in enumerate((URL, URL + "2"))
    ]))
    good = {"fact_id": "f1", "content": "被保留的事实", "source_url": URL,
            "source_type": "official", "data_points": []}
    analysis = {
        "extracted_facts": [good, {**good, "fact_id": "f2"},
                            {**good, "fact_id": "f3", "source_url": "https://fabricated.example"}],
        "hypothesis_evidence": [{
            "fact_id": "f1", "hypothesis_id": "h_1", "evidence_type": "supports",
            "evidence_summary": "支持该假设", "source_url": URL, "evidence_quote": "研究原始正文。",
        }],
    }
    scout.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(analysis, ensure_ascii=False)))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )))))
    return scout


class ScoutObservationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from langfuse import Langfuse
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        cls.exporter = InMemorySpanExporter()
        cls.provider = TracerProvider()
        cls.client = Langfuse(
            public_key="pk-lf-scout-offline-test", secret_key="sk-lf-offline-test",
            base_url="http://127.0.0.1:1", tracer_provider=cls.provider,
            span_exporter=cls.exporter,
        )

    @classmethod
    def tearDownClass(cls):
        cls.client.shutdown()

    def setUp(self):
        self.exporter.clear()
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(tracing, "get_client", return_value=self.client))
        self.stack.enter_context(patch.object(tracer, "_enqueue"))
        self.stack.enter_context(patch.object(tracer, "TRACE_ENABLED", True))
        self.addCleanup(self.stack.close)

    def spans(self):
        self.client.flush()
        return self.exporter.get_finished_spans()

    def one(self, name):
        return next(span for span in self.spans() if span.name == name)

    def payload(self, name, kind="output"):
        return json.loads(self.one(name).attributes[f"langfuse.observation.{kind}"])

    def tearDown(self):
        CAPTURED_OBSERVATIONS.append({
            "test": self.id(), "observations": [{
                "name": span.name, "trace_id": f"{span.context.trace_id:032x}",
                "span_id": f"{span.context.span_id:016x}",
                "parent_id": f"{span.parent.span_id:016x}" if span.parent else None,
                "attributes": dict(span.attributes),
            } for span in self.spans()],
        })

    def mock_http(self, responses=None):
        payload = {"code": 200, "extra_provider_field": "原始响应字段", "data": {"webPages": {"value": [
            {"url": url, "name": "官方报告", "summary": "检索摘要" * 700, "siteName": "官方"}
            for url in (URL, URL + "2")
        ]}}}
        queue = list(responses or [(200, payload)])
        def handler(request):
            status, data = queue.pop(0)
            return httpx.Response(status, json=data, request=request)
        transport = httpx.MockTransport(handler)
        return patch("service.deep_research_v2.agents.scout.httpx.AsyncClient",
                     side_effect=lambda *a, **kw: REAL_ASYNC_CLIENT(transport=transport))

    async def test_process_exports_original_sources_without_fact_extraction(self):
        scout, state = make_scout(), make_state()
        with self.mock_http(), tracer.trace_context(session_id=state["session_id"], state=state):
            result = await scout.process(state)
        output = self.payload("DeepScout.process")
        self.assertIs(result, state)
        self.assertEqual(output["state_after"]["facts"], state["facts"])
        self.assertEqual(len(output["changes"]["facts"]["added"]), 0)
        self.assertEqual(len(output["changes"]["hypothesis_evidence"]["added"]), 0)
        scout.client.chat.completions.create.assert_not_awaited()
        self.assertEqual(len(output["changes"]["raw_sources"]["added"]), 2)
        self.assertNotIn("evidence_ledger", output["state_after"])
        self.assertNotIn("evidence_ledger", state)
        self.assertEqual(state["facts"], [])
        self.assertEqual(output["changes"]["search_budget"]["used_delta"], 1)
        self.assertEqual(output["state_after"]["raw_sources"][0]["related_sections"], ["sec_1"])
        self.assertNotIn("fabricated.example", json.dumps(output))
        root = self.one("DeepScout.process")
        for span in self.spans():
            self.assertEqual(span.context.trace_id, root.context.trace_id)
        self.assertNotIn("DeepScout", [span.name for span in self.spans()])
        self.assertEqual(self.one("Bocha.http").parent.span_id, self.one("Bocha.search").context.span_id)
        state["raw_sources"][0]["title"] = "后续阶段修改"
        self.assertNotEqual(self.payload("DeepScout.process")["state_after"]["raw_sources"][0].get("title"), "后续阶段修改")

    async def test_full_search_and_reader_payloads_are_not_preview_truncated(self):
        scout, state = make_scout(), make_state()
        with self.mock_http(), tracer.trace_context(session_id=state["session_id"], state=state):
            await scout.process(state)
        raw = self.payload("Bocha.http")
        self.assertEqual(raw["body"]["extra_provider_field"], "原始响应字段")
        self.assertEqual(self.payload("Bocha.search")[0]["summary"], "检索摘要" * 700)
        self.assertEqual(self.payload("WebReader.read_many")[0]["content"], CONTENT)
        self.assertEqual(self.payload("WebReader.read_many", "input")["limit"], 6)
        self.assertEqual(self.payload("Bocha.http", "input")["body"]["query"], "site:example.gov.cn 测试检索")
        self.assertNotIn(scout.search_api_key, str([dict(s.attributes) for s in self.spans()]))

    async def test_supplementary_process_exports_new_state_and_existing_budget(self):
        scout, state = make_scout(), make_state()
        state.update(phase="re_researching", pending_research_tasks=[{
            "query": "补证检索", "target_section": "sec_1", "issue_id": "issue_1",
        }])
        state["search_budget"].update(used=7, remaining=13)
        with self.mock_http(), tracer.trace_context(session_id=state["session_id"], state=state):
            await scout.process(state)
        output = self.payload("DeepScout.process")
        self.assertEqual(output["changes"]["search_budget"]["before"]["used"], 7)
        self.assertEqual(output["changes"]["search_budget"]["after"]["used"], 8)
        self.assertEqual(len(output["changes"]["facts"]["added"]), 0)
        self.assertEqual(len(output["changes"]["raw_sources"]["added"]), 2)
        scout.client.chat.completions.create.assert_not_awaited()
        self.assertEqual(output["status"], "completed")

    async def test_cache_hit_exports_results_without_extra_http(self):
        scout = make_scout()
        context = ToolCallContext(session_id="s", run_id="r")
        cached = [{"url": URL, "summary": CONTENT}]
        scout.search_cache[scout._search_cache_key(WebSearchToolArgs(query="cached"))] = cached
        result = await scout._run_web_search_tool(WebSearchToolArgs(query="cached"), context)
        self.assertIs(result, cached)
        self.assertEqual(self.payload("Bocha.search"), cached)
        self.assertNotIn("Bocha.http", [span.name for span in self.spans()])

    async def test_each_retry_exports_original_response_and_counts_one_query(self):
        scout, state = make_scout(), make_state()
        responses = [(429, {"code": 429, "message": "限流详情"}),
                     (200, {"code": 200, "data": {"webPages": {"value": []}}})]
        with self.mock_http(responses), patch("service.deep_research_v2.agents.scout.asyncio.sleep", new=AsyncMock()):
            await scout._execute_search("重试", state=state)
        attempts = [span for span in self.spans() if span.name == "Bocha.http"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].attributes["langfuse.observation.level"], "ERROR")
        self.assertEqual(state["search_budget"]["used"], 1)

    async def test_failed_reader_result_keeps_failure_and_full_payload(self):
        scout = make_scout()
        scout.web_reader.read_many.return_value = [WebReadResult(
            url=URL, final_url=URL, fetch_status="failed", failure_category="http_403", http_status=403,
        )]
        result = await scout._run_web_read_tool(
            WebReadBatchToolArgs(items=[{"url": URL, "summary": CONTENT}]),
            ToolCallContext(session_id="s", run_id="r"),
        )
        self.assertEqual(self.payload("WebReader.read_many"), result)
        self.assertEqual(result[0]["failure_category"], "http_403")

    async def test_business_error_and_cancellation_preserve_partial_state(self):
        for error in (RuntimeError("业务错误"), asyncio.CancelledError("取消")):
            scout, state = make_scout(), make_state()
            async def fail(state, section):
                state["facts"].append({"id": "partial", "content": "尚未完成来源门禁"})
                raise error
            scout._research_section = fail
            with self.assertRaises(type(error)) as caught:
                await scout.process(state)
            self.assertIs(caught.exception, error)
        roots = [s for s in self.spans() if s.name == "DeepScout.process"]
        self.assertEqual(len(roots), 2)
        for root in roots:
            output = json.loads(root.attributes["langfuse.observation.output"])
            self.assertEqual(output["status"], "interrupted")
            self.assertEqual(output["validation_boundary"], "partial_state")
            self.assertEqual(root.attributes["langfuse.observation.level"], "ERROR")

    async def test_concurrent_runs_in_same_session_are_isolated(self):
        @tracing.trace_scout_process
        async def work(self, state):
            await asyncio.sleep(0)
            tracing.record_generation({"agent_name": "nested", "success": True}, "system", "input", state["_run_id"])
            return state
        await asyncio.gather(work(None, make_state("run-a")), work(None, make_state("run-b")))
        roots = [s for s in self.spans() if s.name == "DeepScout.process"]
        self.assertEqual(len({s.context.trace_id for s in roots}), 2)
        for root in roots:
            child = next(s for s in self.spans() if s.name == "nested" and s.context.trace_id == root.context.trace_id)
            self.assertEqual(child.parent.span_id, root.context.span_id)


class TracingIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_http_tracing_preserves_response_without_extra_parsing(self):
        response = Mock()
        client = SimpleNamespace(post=AsyncMock(return_value=response))
        with patch.dict("os.environ", {"LANGFUSE_ENABLED": "false"}):
            result = await tracing.trace_bocha_request(
                client, URL, headers={"Authorization": "secret"}, json={"query": "x"}, attempt=1,
            )
        self.assertIs(result, response)
        response.json.assert_not_called()

    async def test_client_configuration_uses_base_url_and_legacy_host_fallback(self):
        env = {"LANGFUSE_ENABLED": "true", "EVAL_TRACE_ENABLED": "true",
               "LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test",
               "LANGFUSE_HOST": "https://legacy.example", "LANGFUSE_BASE_URL": "https://current.example/"}
        for expected in ("https://current.example", "https://legacy.example"):
            with patch.dict("os.environ", env), patch.object(tracing, "_checked", False), \
                 patch.object(tracing, "_client", None), patch("langfuse.Langfuse", autospec=True) as factory:
                self.assertIs(tracing.get_client(), factory.return_value)
                self.assertEqual(factory.call_args.kwargs["base_url"], expected)
            env["LANGFUSE_BASE_URL"] = ""

    async def test_missing_credentials_disable_export_with_visible_warning(self):
        with patch.dict("os.environ", {"LANGFUSE_ENABLED": "true", "EVAL_TRACE_ENABLED": "true",
                                       "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}), \
             patch.object(tracing, "_checked", False), patch.object(tracing, "_client", None), \
             self.assertLogs(tracing.logger, level="WARNING"):
            self.assertIsNone(tracing.get_client())

    async def test_update_failure_is_isolated(self):
        span = Mock()
        span.update.side_effect = RuntimeError("导出失败")
        with self.assertLogs(tracing.logger, level="WARNING"):
            tracing.SafeObservation(span).update(output={"facts": ["仍然有效"]})
        span.update.assert_called_once()

    async def test_disabled_tracing_does_not_snapshot_or_import_sdk(self):
        sentinel = object()
        @tracing.trace_scout_process
        async def work(self, state):
            return sentinel
        with patch.dict("os.environ", {"LANGFUSE_ENABLED": "false"}), patch.object(tracing, "_snapshot") as snapshot:
            self.assertIs(await work(None, {}), sentinel)
        snapshot.assert_not_called()

    async def test_broken_exporter_cannot_change_business_result(self):
        @tracing.trace_scout_process
        async def work(self, state):
            state["facts"].append({"id": "kept"})
            return state
        client = SimpleNamespace(start_as_current_observation=lambda **kw: (_ for _ in ()).throw(RuntimeError("SDK failure")))
        state = make_state()
        with patch.object(tracing, "get_client", return_value=client):
            self.assertIs(await work(None, state), state)
        self.assertEqual(state["facts"], [{"id": "kept"}])

    async def test_old_sdk_rejected_and_pinned_sdk_has_required_interfaces(self):
        from langfuse import Langfuse
        from importlib.metadata import version
        self.assertEqual(version("langfuse"), tracing.TESTED_SDK_VERSION)
        tracing.validate_sdk(Langfuse, version("langfuse"))
        with self.assertRaisesRegex(RuntimeError, "v4"):
            tracing.validate_sdk(Langfuse, "2.60.3")
        with self.assertRaisesRegex(RuntimeError, "缺少接口"):
            tracing.validate_sdk(object(), "4.15.1")

    async def test_updated_removed_evidence_and_budget_changes(self):
        before = tracing._snapshot(make_state())
        before["facts"] = [{"id": "f1", "content": "旧事实"}, {"id": "f2"}]
        after = copy.deepcopy(before)
        after["facts"] = [{"id": "f1", "content": "更新事实"}]
        after["hypothesis_evidence"] = [{"id": "h1", "fact_id": "f1"}]
        after["search_budget"].update(used=20, remaining=0, skipped=2)
        delta = tracing._changes(before, after)
        self.assertEqual(delta["facts"]["updated"], after["facts"])
        self.assertEqual(delta["facts"]["removed_ids"], ["f2"])
        self.assertEqual(delta["hypothesis_evidence"]["added"], after["hypothesis_evidence"])
        self.assertEqual(delta["search_budget"]["skipped_delta"], 2)


if __name__ == "__main__":
    unittest.main()
