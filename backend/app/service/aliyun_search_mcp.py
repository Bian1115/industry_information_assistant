"""百炼 EnhancedSearch MCP → DeepScout 的统一搜索结果。

MCP SDK 仅在使用阿里后端时导入；每次调用在同一协程内打开、关闭会话，
避免 transport 的 task group 跨 LangGraph 节点退出。认证头不进入诊断数据。
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit


ALIYUN_SEARCH_MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/EnhancedSearch/mcp"
SEARCH_PROVIDERS = {"bocha", "aliyun_mcp"}
MCP_CONNECTION_MODE = "legacy"


class SearchMCPError(RuntimeError):
    """配置、上游工具或返回格式错误；禁止静默切换搜索服务。"""


def normalize_mcp_search_result(raw: dict[str, Any], count: int) -> tuple[list[dict], int]:
    """解析已实测的 pages JSON；也支持同结构的 structuredContent。

    摘要只用于筛选候选来源，后续仍由现有 WebReader 读取正文。
    无日期就保留空字符串，不把抓取时间伪装成发布日期。
    """
    if raw.get("isError") or raw.get("is_error"):
        raise SearchMCPError("阿里 MCP 工具返回 isError=true，请检查原始工具结果")
    payloads = []
    structured = raw.get("structuredContent", raw.get("structured_content"))
    if isinstance(structured, dict):
        payloads.append(structured)
    for block in raw.get("content", []):
        if block.get("type") == "text":
            try:
                payloads.append(json.loads(block.get("text", "")))
            except (TypeError, ValueError):
                continue
    pages = next((p["pages"] for p in payloads if isinstance(p, dict)
                  and isinstance(p.get("pages"), list)), None)
    if pages is None:
        raise SearchMCPError("阿里 MCP 结果缺少 pages 数组，请检查原始结果，更新适配器")

    rows, seen = [], set()
    for rank, page in enumerate(pages, 1):
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "").strip()
        try:
            parts = urlsplit(url)
        except ValueError:
            continue
        snippet = str(page.get("snippet") or "")
        summary = str(page.get("summary") or snippet)
        if parts.scheme not in {"http", "https"} or not parts.hostname or url in seen or not summary:
            continue
        seen.add(url)
        rows.append({
            "url": url, "title": str(page.get("title") or ""),
            "summary": summary, "snippet": snippet,
            "site_name": str(page.get("hostname") or parts.hostname),
            "date": str(page.get("datePublished") or page.get("publish_time") or ""),
            "search_provider": "aliyun_mcp", "search_rank": rank,
        })
    return rows[:count], len(pages)


class AliyunSearchMCP:
    def __init__(self, api_key: str, *, url: str = ALIYUN_SEARCH_MCP_URL,
                 tool_name: str = "search_pro", timeout_seconds: float = 60):
        self._api_key = api_key
        self.url = url
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        self.last_call: dict[str, Any] = {}

    @asynccontextmanager
    async def _session(self):
        if not self._api_key:
            raise SearchMCPError("缺少 ALIYUN_SEARCH_MCP_API_KEY 或 DASHSCOPE_API_KEY")
        try:
            import httpx2
            from mcp import Client
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise SearchMCPError("请在后端 Python 环境安装 mcp>=2.3.0,<3") from exc
        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx2.Timeout(15, read=self.timeout_seconds),
        ) as http_client:
            async with Client(
                streamable_http_client(self.url, http_client=http_client),
                # 阿里实测协商 2024-11-05；直接 initialize，避免 v2 discover 探测触发 HTTP 500。
                read_timeout_seconds=self.timeout_seconds, cache=None, mode=MCP_CONNECTION_MODE,
            ) as client:
                yield client

    @staticmethod
    async def _tools(client) -> list[dict[str, Any]]:
        tools, cursor, seen = [], None, set()
        while True:
            result = await client.list_tools(cursor=cursor)
            tools.extend(t.model_dump(mode="json", by_alias=True, exclude_none=True)
                         for t in result.tools)
            cursor = result.next_cursor
            if not cursor:
                return tools
            if cursor in seen:
                raise SearchMCPError("MCP tools/list 返回重复游标")
            seen.add(cursor)

    async def discover_tools(self) -> list[dict[str, Any]]:
        async with asyncio.timeout(self.timeout_seconds):
            async with self._session() as client:
                return await self._tools(client)

    async def search(self, query: str, count: int = 10) -> list[dict]:
        started = time.perf_counter()
        self.last_call = {
            "provider": "aliyun_mcp", "endpoint": self.url, "tool_name": self.tool_name,
            "query": query, "requested_count": count, "status": "running",
            "connection_mode": MCP_CONNECTION_MODE,
            "cache_hit": False, "search_tool_calls": 0,
        }
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with self._session() as client:
                    tools = await self._tools(client)
                    tool = next((t for t in tools if t["name"] == self.tool_name), None)
                    if tool is None:
                        raise SearchMCPError(f"服务未提供工具 {self.tool_name}，请运行工具发现命令")
                    schema = tool["inputSchema"]
                    # EnhancedSearch 当前只有 query；不发送服务端未声明的 count。
                    arguments = {"query": query}
                    from jsonschema import validate, ValidationError
                    try:
                        validate(arguments, schema)
                    except ValidationError as exc:
                        raise SearchMCPError("搜索参数不符合服务端 inputSchema") from exc
                    self.last_call.update(arguments=arguments, input_schema=schema,
                                          server_supports_count="count" in schema.get("properties", {}))
                    self.last_call["search_tool_calls"] = 1
                    result = await client.call_tool(self.tool_name, arguments)
                    raw = result.model_dump(mode="json", by_alias=True, exclude_none=True)
                    self.last_call["raw_response"] = raw
                    rows, upstream_count = normalize_mcp_search_result(raw, count)
                    self.last_call.update(status="ok", upstream_count=upstream_count,
                                          returned_count=len(rows))
                    return rows
        except asyncio.CancelledError:
            self.last_call.update(status="cancelled", error_type="CancelledError")
            raise
        except Exception as exc:
            self.last_call.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            self.last_call["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
