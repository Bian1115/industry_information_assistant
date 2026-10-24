"""搜索后端适配、错误归因、缓存及运行冻结的回归测试。"""

import asyncio
import json
import logging
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from service.aliyun_search_mcp import AliyunSearchMCP, SearchMCPError, normalize_mcp_search_result
from service.deep_research_v2.agents.scout import DeepScout
from service.tool_runtime import ToolRuntime, ToolCallContext

SCHEMA = {"type": "object", "properties": {"query": {"type": "string", "minLength": 2,
                                                       "maxLength": 500}},
          "required": ["query"], "additionalProperties": False}
PAGES = [{"url": "https://example.org/paper", "title": "研究", "snippet": "研究摘要",
          "hostname": "研究机构"}]


def raw_result(pages):
    return {"content": [{"type": "text", "text": json.dumps({"pages": pages})}], "isError": False}


def make_scout(provider="aliyun_mcp"):
    scout = DeepScout.__new__(DeepScout)
    scout.name = "DeepScout"
    scout.logger = logging.getLogger("test.aliyun-mcp")
    scout.search_provider = provider
    scout.aliyun_search_mcp = AliyunSearchMCP("test-key")
    scout.search_cache = {}
    scout.search_cache_enabled = True
    scout.tool_runtime = ToolRuntime()
    scout.tool_definitions = scout._build_tool_definitions()
    scout._max_total_searches = 20
    return scout


class NormalizeTests(unittest.TestCase):
    def test_text_json_and_missing_dates(self):
        rows, total = normalize_mcp_search_result(raw_result(PAGES), 10)
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["date"], "")
        self.assertEqual(rows[0]["summary"], "研究摘要")
        self.assertEqual(rows[0]["site_name"], "研究机构")

    def test_structured_content_deduplicates_filters_and_keeps_rank(self):
        pages = [*PAGES, *PAGES, {"url": "javascript:alert(1)", "snippet": "bad"},
                 {"url": "https://example.org/2", "summary": "second"}]
        rows, total = normalize_mcp_search_result({"structuredContent": {"pages": pages}}, 2)
        self.assertEqual(total, 4)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["search_rank"], 4)
        self.assertEqual(len(normalize_mcp_search_result(raw_result(pages), 1)[0]), 1)

    def test_empty_is_valid_but_unknown_and_tool_error_raise(self):
        self.assertEqual(normalize_mcp_search_result(raw_result([]), 10), ([], 0))
        for raw in ({"content": []}, {"isError": True}, {"structuredContent": {"other": []}}):
            with self.assertRaises(SearchMCPError):
                normalize_mcp_search_result(raw, 10)


class GraphConfigTests(unittest.TestCase):
    def test_graph_injects_selected_backend_and_export_omits_mcp_key(self):
        from config.llm_config import LLMConfig
        from service.deep_research_v2 import graph
        config = LLMConfig(api_key="fake-model-key", search_api_key="fake-bocha-key",
                           search_provider="aliyun_mcp", aliyun_search_mcp_api_key="fake-mcp-secret")
        with patch.object(graph, "DeepScout") as factory:
            graph.create_agents(config)
        self.assertEqual(factory.call_args.kwargs["search_provider"], "aliyun_mcp")
        self.assertEqual(factory.call_args.kwargs["aliyun_search_mcp_api_key"], "fake-mcp-secret")
        exported = config.to_dict()
        self.assertTrue(exported["aliyun_search_mcp_key_configured"])
        self.assertNotIn("fake-mcp-secret", json.dumps(exported))


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def fake_session(self, adapter, *, name="search_pro", raw=None):
        client = AsyncMock()
        client.list_tools.return_value = ListToolsResult(tools=[Tool(name=name, input_schema=SCHEMA)])
        result = raw or raw_result(PAGES)
        client.call_tool.return_value = CallToolResult.model_validate(result)
        @asynccontextmanager
        async def session():
            yield client
        return patch.object(adapter, "_session", session), client

    async def test_query_only_call_and_raw_capture_without_key(self):
        adapter = AliyunSearchMCP("test-secret-do-not-log")
        mocked, client = self.fake_session(adapter)
        with mocked:
            rows = await adapter.search("医学研究", 5)
        client.call_tool.assert_awaited_once_with("search_pro", {"query": "医学研究"})
        self.assertEqual(len(rows), 1)
        self.assertFalse(adapter.last_call["server_supports_count"])
        self.assertIn("raw_response", adapter.last_call)
        self.assertNotIn("test-secret", json.dumps(adapter.last_call))

    async def test_invalid_schema_and_missing_tool_make_no_search_call(self):
        adapter = AliyunSearchMCP("test")
        mocked, client = self.fake_session(adapter)
        with mocked, self.assertRaises(SearchMCPError):
            await adapter.search("字", 5)
        client.call_tool.assert_not_awaited()
        mocked, client = self.fake_session(adapter, name="other_tool")
        with mocked, self.assertRaises(SearchMCPError):
            await adapter.search("医学研究", 5)
        client.call_tool.assert_not_awaited()

    async def test_tool_error_raises_and_is_not_retried(self):
        adapter = AliyunSearchMCP("test")
        mocked, client = self.fake_session(adapter, raw={"isError": True, "content": []})
        with mocked, self.assertRaises(SearchMCPError):
            await adapter.search("医学研究", 5)
        self.assertEqual(client.call_tool.await_count, 1)
        self.assertEqual(adapter.last_call["status"], "error")

    async def test_missing_key_reports_configuration_error(self):
        adapter = AliyunSearchMCP("")
        with self.assertRaises(SearchMCPError):
            await adapter.discover_tools()

    async def test_timeout_and_cancellation_are_not_suppressed(self):
        adapter = AliyunSearchMCP("test", timeout_seconds=.01)
        mocked, client = self.fake_session(adapter)
        async def slow(*args, **kwargs):
            await asyncio.sleep(1)
        client.call_tool.side_effect = slow
        with mocked, self.assertRaises(TimeoutError):
            await adapter.search("医学研究", 5)
        client.call_tool.side_effect = asyncio.CancelledError()
        with mocked, self.assertRaises(asyncio.CancelledError):
            await adapter.search("医学研究", 5)


class ScoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_tool_contract_and_cache_identity(self):
        scout = make_scout()
        scout.aliyun_search_mcp.search = AsyncMock(return_value=PAGES)
        definition = scout.tool_definitions["web_search"]
        for count in (1, 1, 2):
            result = await scout.tool_runtime.execute(definition, {"query": "医学研究", "count": count})
            self.assertEqual(result.value[0]["url"], PAGES[0]["url"])
        self.assertEqual(scout.aliyun_search_mcp.search.await_count, 2)
        old_keys = set(scout.search_cache)
        scout.search_provider = "bocha"
        from service.deep_research_v2.tool_contracts import WebSearchToolArgs
        self.assertNotIn(scout._search_cache_key(WebSearchToolArgs(query="医学研究", count=1)), old_keys)

    async def test_mcp_uses_domain_filter_and_report_budget(self):
        scout = make_scout()
        scout.aliyun_search_mcp.search = AsyncMock(return_value=[
            *PAGES, {"url": "https://other.org/paper", "title": "域外结果", "snippet": "摘要"},
        ])
        state = {"messages": [], "allowed_domains": ["example.org"], "search_budget": {"limit": 1}}
        rows = await scout._execute_search("医学研究", state=state)
        self.assertEqual([r["url"] for r in rows], [PAGES[0]["url"]])
        self.assertEqual(state["search_budget"]["used"], 1)
        self.assertEqual(state["search_budget"]["queries"][0]["provider"], "aliyun_mcp")
        self.assertEqual(state["search_backend"]["provider"], "aliyun_mcp")
        self.assertEqual(await scout._execute_search("另一查询", state=state), [])
        scout.aliyun_search_mcp.search.assert_awaited_once()

    async def test_resumed_and_legacy_runs_cannot_mix_backends(self):
        scout = make_scout()
        for state in ({"messages": [], "search_backend": {"provider": "bocha"}},
                      {"messages": [], "search_budget": {"used": 1}}):
            with self.assertRaises(ValueError):
                await scout._execute_search_once("医学研究", state=state)

    async def test_mcp_failure_does_not_fall_back_to_bocha(self):
        scout = make_scout()
        scout.aliyun_search_mcp.search = AsyncMock(side_effect=SearchMCPError("工具失败"))
        scout._run_bocha_search_tool = AsyncMock()
        with self.assertRaises(SearchMCPError):
            await scout._execute_search_once("医学研究")
        scout._run_bocha_search_tool.assert_not_awaited()
        self.assertEqual(scout.last_search_diagnostics["status"], "error")

    async def test_mcp_tracing_preserves_raw_result_and_omits_authentication(self):
        import os
        import tempfile
        scout = make_scout()
        scout.aliyun_search_mcp.last_call = {"raw_response": raw_result(PAGES), "status": "ok"}
        scout.aliyun_search_mcp.search = AsyncMock(return_value=PAGES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "本地轨迹.jsonl"
            with patch.dict(os.environ, {"EVAL_TRACE_LOCAL_FILE": str(path), "LANGFUSE_ENABLED": "false"}):
                await scout.tool_runtime.execute(scout.tool_definitions["web_search"],
                    {"query": "医学研究"}, ToolCallContext(run_id="run-test", session_id="session-test"))
            content = path.read_text(encoding="utf-8")
        events = [json.loads(line) for line in content.splitlines()]
        names = {e["name"] for e in events}
        self.assertIn("AliyunMCP.call", names)
        self.assertIn("AliyunMCP.search", names)
        self.assertNotIn("Bocha.search", names)
        self.assertNotIn("test-key", content)
        self.assertTrue(any(e.get("output", {}).get("raw_response") for e in events
                            if isinstance(e.get("output"), dict)))


if __name__ == "__main__":
    unittest.main()
