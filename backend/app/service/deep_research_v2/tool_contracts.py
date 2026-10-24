"""DeepResearch V2 外部工具的统一运行时契约。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel

from service.tool_runtime import ToolCallContext


class StrictToolArgs(BaseModel):
    """V2 工具入参默认拒绝模型额外生成的未知字段。"""

    model_config = ConfigDict(extra="forbid")


class WebSearchToolArgs(StrictToolArgs):
    query: str = Field(min_length=1, max_length=500)
    count: int = Field(default=10, ge=1, le=20)


class SearchResultItem(BaseModel):
    """联网检索结果的最小形状，保留来源特有字段。"""

    model_config = ConfigDict(extra="allow")

    url: str = Field(min_length=1)
    title: str = ""
    summary: str = ""
    snippet: str = ""
    site_name: str = ""
    date: str = ""


class SearchResults(RootModel[List[SearchResultItem]]):
    pass


class WebReadBatchToolArgs(StrictToolArgs):
    items: List[Dict[str, Any]]
    limit: int = Field(default=3, ge=1, le=10)
    concurrency: int = Field(default=2, ge=1, le=5)


class WebReadResultItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str
    final_url: str
    title: str = ""
    content: str = ""
    published_at: str = ""
    content_type: str = ""
    extractor: str = ""
    fetch_status: str = "failed"
    failure_category: Optional[str] = None
    http_status: Optional[int] = None
    recovery_action: str = ""
    attempts: Any = 1
    error: Optional[str] = None
    from_cache: bool = False
    fetched_at: Any = 0.0


class WebReadResults(RootModel[List[WebReadResultItem]]):
    pass


class SandboxExecutionToolArgs(StrictToolArgs):
    code: str = Field(min_length=1, max_length=200_000)


class SandboxExecutionToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    output: str = ""
    error: Optional[str] = None
    charts: List[str] = Field(default_factory=list)
    sandbox_provider: str = "unknown"
    error_code: Optional[str] = None


def build_tool_call_context(
    state: Optional[Dict[str, Any]],
    *,
    agent_name: str,
    approved: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolCallContext:
    """从共享 ResearchState 生成统一的工具调用关联上下文。"""

    state = state or {}
    tool_metadata = {"agent_name": agent_name}
    tool_metadata.update(metadata or {})
    return ToolCallContext(
        session_id=str(state.get("session_id") or "") or None,
        run_id=str(state.get("_run_id") or state.get("session_id") or "") or None,
        user_id=str(state.get("_user_id") or "") or None,
        approved=approved,
        metadata=tool_metadata,
    )
