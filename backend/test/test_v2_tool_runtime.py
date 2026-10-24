"""DeepResearch V2 外部工具统一经过 ToolRuntime 的回归测试。"""

import asyncio
import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from pydantic import BaseModel


BACKEND_DIR = Path(__file__).resolve().parents[1]
APP_DIR = BACKEND_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from service.code_sandbox import (  # noqa: E402
    CodeSandboxProvider,
    SandboxExecutionResult,
)
from service.deep_research_v2.agents.scout import DeepScout  # noqa: E402
from service.deep_research_v2.agents.wizard import CodeWizard  # noqa: E402
from service.tool_runtime import (  # noqa: E402
    ToolErrorCode,
    ToolCallContext,
    ToolDefinition,
    ToolExecutionResult,
    ToolRuntime,
    ToolRuntimeError,
)
from service.web_reader_service import WebReadResult  # noqa: E402


class RecordingRuntime:
    def __init__(self, values):
        self.values = values
        self.calls = []

    async def execute(self, definition, raw_args, context=None):
        self.calls.append((definition, raw_args, context))
        return ToolExecutionResult(
            call_id=f"call-{len(self.calls)}",
            tool_name=definition.name,
            value=self.values[definition.name],
            duration_ms=1,
        )


class FakeSandboxProvider(CodeSandboxProvider):
    async def execute(self, code: str) -> SandboxExecutionResult:
        return SandboxExecutionResult(
            success=True,
            output=code,
            provider="fake-sandbox",
        )


def make_scout(runtime):
    scout = DeepScout.__new__(DeepScout)
    scout.name = "DeepScout"
    scout.logger = logging.getLogger("test.v2-tool-runtime.scout")
    scout.tool_runtime = runtime
    scout.tool_definitions = scout._build_tool_definitions()
    return scout


def make_wizard(runtime):
    wizard = CodeWizard.__new__(CodeWizard)
    wizard.name = "CodeWizard"
    wizard.logger = logging.getLogger("test.v2-tool-runtime.wizard")
    wizard.sandbox_provider = FakeSandboxProvider()
    wizard.tool_runtime = runtime
    wizard.tool_definitions = wizard._build_tool_definitions()
    wizard._save_debug_log = lambda *args, **kwargs: None
    wizard._save_debug_info = lambda *args, **kwargs: None
    return wizard


class V2ScoutToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_accepts_missing_dates_and_preserves_known_dates(self):
        """上游日期缺失时，整批搜索结果仍可通过真实工具契约校验。"""
        scout = make_scout(ToolRuntime())
        scout.search_cache = {}
        scout.search_api_key = "test-key"
        scout._search_circuit_open_until = 0
        scout._search_rate_lock = asyncio.Lock()
        scout._last_search_at = 0
        scout._min_search_interval = 0
        dates = [
            ({"datePublished": "2026-09-01", "dateLastCrawled": "2026-09-25"}, "2026-09-01"),
            ({"datePublished": None, "dateLastCrawled": "2026-09-25"}, "2026-09-25"),
            ({}, ""),
            ({"datePublished": "", "dateLastCrawled": ""}, ""),
            ({"dateLastCrawled": None}, ""),
            ({"datePublished": None, "dateLastCrawled": None}, ""),
        ]
        webpages = [
            {"url": f"https://example.org/{i}", "name": "文献", "snippet": "摘要", **fields}
            for i, (fields, _) in enumerate(dates)
        ]
        response = httpx.Response(200, json={
            "code": 200, "data": {"webPages": {"value": webpages}},
        })
        with patch(
            "service.deep_research_v2.agents.scout.trace_bocha_request",
            new=AsyncMock(return_value=response),
        ) as request:
            result = await scout.tool_runtime.execute(
                scout.tool_definitions["web_search"], {"query": "日期回归测试", "count": 6},
            )
            cached = await scout.tool_runtime.execute(
                scout.tool_definitions["web_search"], {"query": "日期回归测试", "count": 6},
            )
        self.assertEqual([item["date"] for item in result.value], [expected for _, expected in dates])
        self.assertEqual(cached.value, result.value)
        request.assert_awaited_once()

    async def test_scoped_search_runs_once_per_domain_through_the_runtime(self):
        """本地知识库与行情旁路已移除；Scout 只剩联网检索与正文读取两个工具。"""
        runtime = RecordingRuntime({
            "v2_web_search": [
                {"url": "https://a.example/1", "title": "web"},
                {"url": "https://b.example/2", "title": "web"},
            ],
        })
        scout = make_scout(runtime)
        self.assertEqual(
            sorted(scout.tool_definitions), ["web_read", "web_search"]
        )
        state = {
            "session_id": "session-1",
            "_run_id": "run-1",
            "_user_id": "user-1",
            "allowed_domains": ["a.example", "b.example"],
            "messages": [],
        }

        web = await scout._execute_search("行业规模", count=5, state=state)

        # 每个允许网站各发一次请求，返回结果按域名过滤后去重。
        self.assertEqual(
            [call[1]["query"] for call in runtime.calls],
            ["site:a.example 行业规模", "site:b.example 行业规模"],
        )
        self.assertEqual(
            sorted(item["url"] for item in web),
            ["https://a.example/1", "https://b.example/2"],
        )
        for _, _, context in runtime.calls:
            self.assertEqual(context.session_id, "session-1")
            self.assertEqual(context.run_id, "run-1")
            self.assertEqual(context.user_id, "user-1")
            self.assertEqual(context.metadata["agent_name"], "DeepScout")

    async def test_web_read_uses_runtime_and_keeps_snapshot_behavior(self):
        content = "可追溯正文" * 100
        runtime = RecordingRuntime({
            "v2_web_fulltext_read": [WebReadResult(
                url="https://official.example/report",
                final_url="https://official.example/report",
                title="官方报告",
                content=content,
                fetch_status="success",
                extractor="fixture",
                fetched_at=1.0,
            ).to_dict()]
        })
        scout = make_scout(runtime)
        scout.web_reader = object()
        scout._update_web_reader_stats = lambda state: None
        state = {
            "session_id": "session-2",
            "_run_id": "run-2",
            "raw_sources": [],
            "allowed_domains": ["official.example"],
        }
        results = [{
            "url": "https://official.example/report",
            "title": "搜索结果",
            "summary": "摘要",
        }]

        await scout._enrich_results_with_full_text(
            state, results, section_id="sec_1", limit=1
        )

        self.assertEqual(runtime.calls[0][0].name, "v2_web_fulltext_read")
        self.assertTrue(results[0]["reader_verified"])
        self.assertEqual(state["raw_sources"][0]["content"], content)
        self.assertEqual(state["raw_sources"][0]["related_sections"], ["sec_1"])

    async def test_strict_search_schema_rejects_empty_query_before_adapter(self):
        scout = make_scout(ToolRuntime())
        state = {"allowed_domains": ["a.example"], "messages": []}
        # 空关键词在入口就被拒绝，不会发出没有主题的查询。
        with self.assertRaises(ValueError):
            await scout._execute_search("", count=5, state=state)
        # 仅有 site 条件、清理后没有主题词时同样拒绝。
        with self.assertRaises(ValueError):
            await scout._execute_search("site:a.example", count=5, state=state)
        with self.assertRaises(ToolRuntimeError) as caught:
            await scout._execute_search_once("", count=5)
        self.assertEqual(caught.exception.code, ToolErrorCode.INVALID_ARGUMENTS.value)


class V2WizardToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_sandbox_execution_uses_dangerous_runtime_definition(self):
        runtime = RecordingRuntime({
            "v2_sandbox_execute": {
                "success": True,
                "output": "ok",
                "error": None,
                "charts": [],
                "sandbox_provider": "fake-sandbox",
                "error_code": None,
            }
        })
        wizard = make_wizard(runtime)
        state = {
            "session_id": "session-3",
            "_run_id": "run-3",
            "_user_id": "user-3",
        }

        result = await wizard._execute_code(
            "data = {'x': [1, 2]}\nprint(data)\nvalue = 1\nvalue += 1",
            state=state,
        )

        definition, raw_args, context = runtime.calls[0]
        self.assertEqual(definition.name, "v2_sandbox_execute")
        self.assertEqual(definition.risk.value, "dangerous")
        self.assertIn("print(data)", raw_args["code"])
        self.assertTrue(context.approved)
        self.assertEqual(context.run_id, "run-3")
        self.assertTrue(result["success"])

    async def test_actual_runtime_validates_sandbox_result(self):
        wizard = make_wizard(ToolRuntime())
        state = {"session_id": "session-4", "_run_id": "run-4"}
        result = await wizard._execute_code(
            "data = {'x': [1, 2]}\nprint(data)\nvalue = 1\nvalue += 1",
            state=state,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["sandbox_provider"], "fake-sandbox")


class CancellationProbe:
    """独立于 Redis 的取消监听，记录生命周期以验证没有后台残留。"""

    def __init__(self):
        self.requested = asyncio.Event()
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.task = None

    def raise_if_cancelled(self):
        return None

    async def wait(self):
        self.task = asyncio.current_task()
        self.started.set()
        try:
            await self.requested.wait()
        finally:
            await asyncio.sleep(0)
            self.closed.set()


class NoArgs(BaseModel):
    pass


class ToolRuntimeTaskCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def _assert_cancel_cleanup(self, trigger, expected_error, expected_code=None):
        token = CancellationProbe()
        handler_started = asyncio.Event()
        handler_closed = asyncio.Event()
        handler_tasks = []

        async def handler(_args, _context):
            handler_tasks.append(asyncio.current_task())
            handler_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                # HTTP 客户端关闭也可能 await；execute 必须等到关闭完成。
                await asyncio.sleep(0)
                handler_closed.set()

        definition = ToolDefinition(
            name="cleanup-probe", description="取消清理测试", args_model=NoArgs,
            handler=handler, timeout_seconds=0.03 if trigger == "runtime_timeout" else 10,
        )
        task = asyncio.create_task(ToolRuntime().execute(
            definition, {}, ToolCallContext(cancellation_token=token),
        ))
        try:
            await asyncio.wait_for(handler_started.wait(), 1)
            await asyncio.wait_for(token.started.wait(), 1)
            if trigger == "outer_cancel":
                task.cancel()
                awaited = task
            elif trigger == "outer_timeout":
                awaited = asyncio.wait_for(task, 0.01)
            else:
                if trigger == "token":
                    token.requested.set()
                awaited = task
            with self.assertRaises(expected_error) as caught:
                await awaited
            if expected_code is not None:
                self.assertEqual(caught.exception.code, expected_code.value)
            self.assertTrue(handler_closed.is_set(), "execute 返回前应关闭工具执行")
            self.assertTrue(token.closed.is_set(), "execute 返回前应关闭取消监听")
            self.assertTrue(handler_tasks[0].done())
            self.assertTrue(token.task.done())
        finally:
            # 若实现回归，测试本身也不会泄漏后台任务影响后续用例。
            pending = [child for child in (task, token.task, *handler_tasks) if child is not None]
            for child in pending:
                if not child.done():
                    child.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def test_external_cancel_drains_execution_and_token_waiter(self):
        await self._assert_cancel_cleanup("outer_cancel", asyncio.CancelledError)

    async def test_outer_wait_for_timeout_drains_execution_and_token_waiter(self):
        await self._assert_cancel_cleanup("outer_timeout", asyncio.TimeoutError)

    async def test_tool_timeout_drains_execution_and_token_waiter(self):
        await self._assert_cancel_cleanup("runtime_timeout", ToolRuntimeError, ToolErrorCode.TIMEOUT)

    async def test_token_cancellation_preserves_error_code_and_drains_execution(self):
        await self._assert_cancel_cleanup("token", ToolRuntimeError, ToolErrorCode.CANCELLED)


if __name__ == "__main__":
    unittest.main()
