"""Deterministic fake tools for Agent Harness fault-injection evaluation.

The fixtures in this module never access the network, database, or filesystem.
Every observable side effect is kept in ``FakeToolState`` so tests can assert
whether a handler was reached, cancelled, or allowed to run concurrently.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from service.tool_runtime import (
    ToolCallContext,
    ToolDefinition,
    ToolRisk,
)


_MISSING = object()


def _arg_value(args: Any, name: str, default: Any = _MISSING) -> Any:
    """Read from raw legacy dicts or validated Pydantic arguments.

    The legacy direct-call baseline passes the original dictionary exactly as
    the historical ToolExecutor did. The full Harness passes a validated model.
    This adapter only normalizes representation; it performs no validation.
    """

    if isinstance(args, dict):
        if name in args:
            return args[name]
    elif hasattr(args, name):
        return getattr(args, name)
    if default is not _MISSING:
        return default
    raise KeyError(name)


class FakeDependencyError(RuntimeError):
    """Stable exception used to simulate a downstream dependency failure."""


class EchoArgs(BaseModel):
    text: str = Field(min_length=1, max_length=1_000)
    repeat: int = Field(default=1, ge=1, le=10)


class EchoResult(BaseModel):
    text: str
    length: int = Field(ge=0)


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    count: int = Field(default=3, ge=1, le=20)


class SearchItem(BaseModel):
    rank: int = Field(ge=1)
    title: str
    url: str
    snippet: str


class SearchResult(BaseModel):
    query: str
    results: List[SearchItem]


class DelayArgs(BaseModel):
    delay_ms: int = Field(default=100, ge=0, le=60_000)
    value: str = "done"


class DelayResult(BaseModel):
    value: str
    delay_ms: int = Field(ge=0)


class WriteArgs(BaseModel):
    record_id: int = Field(ge=1)
    value: str = Field(min_length=1, max_length=1_000)


class WriteResult(BaseModel):
    success: bool
    record_id: int = Field(ge=1)


class FailureArgs(BaseModel):
    message: str = Field(default="fake dependency unavailable", min_length=1)


class SerialArgs(BaseModel):
    delay_ms: int = Field(default=10, ge=0, le=5_000)
    sequence: int = Field(ge=0)


class SerialResult(BaseModel):
    sequence: int = Field(ge=0)


class LargeResultArgs(BaseModel):
    item_count: int = Field(default=100, ge=1, le=10_000)


class LargeResult(BaseModel):
    items: List[int]


class ContextProbeArgs(BaseModel):
    label: str = Field(default="probe", min_length=1, max_length=100)


class ContextProbeResult(BaseModel):
    label: str
    session_id: str | None
    run_id: str | None
    user_id: str | None


class StrictResult(BaseModel):
    status: Literal["ok"]
    payload: str


@dataclass
class FakeToolState:
    """All mutable state produced by the fake handlers."""

    call_counts: Counter[str] = field(default_factory=Counter)
    cancellation_counts: Counter[str] = field(default_factory=Counter)
    side_effect_counts: Counter[str] = field(default_factory=Counter)
    active_counts: Counter[str] = field(default_factory=Counter)
    max_active_counts: Counter[str] = field(default_factory=Counter)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, tool_name: str, event: str, **details: Any) -> None:
        self.events.append({"tool": tool_name, "event": event, **details})

    def reset(self) -> None:
        self.call_counts.clear()
        self.cancellation_counts.clear()
        self.side_effect_counts.clear()
        self.active_counts.clear()
        self.max_active_counts.clear()
        self.events.clear()


class HarnessFakeTools:
    """Factory for deterministic ToolDefinitions used by Harness evaluations."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 0.02,
        cancellation_timeout_seconds: float = 5.0,
    ) -> None:
        self.state = FakeToolState()
        self.timeout_seconds = timeout_seconds
        self.cancellation_timeout_seconds = cancellation_timeout_seconds

    def definitions(self) -> Dict[str, ToolDefinition]:
        """Return a new registry backed by this factory's observable state."""

        return {
            "fake_echo_success": ToolDefinition(
                name="fake_echo_success",
                description="Return deterministic text for valid-call tests.",
                args_model=EchoArgs,
                handler=self._echo_success,
                result_model=EchoResult,
            ),
            "fake_search_success": ToolDefinition(
                name="fake_search_success",
                description="Return deterministic search-shaped results.",
                args_model=SearchArgs,
                handler=self._search_success,
                result_model=SearchResult,
            ),
            "fake_context_probe": ToolDefinition(
                name="fake_context_probe",
                description="Echo run/session/user identifiers from ToolCallContext.",
                args_model=ContextProbeArgs,
                handler=self._context_probe,
                result_model=ContextProbeResult,
            ),
            "fake_slow_timeout": ToolDefinition(
                name="fake_slow_timeout",
                description="Sleep beyond the configured runtime deadline.",
                args_model=DelayArgs,
                handler=self._slow_timeout,
                result_model=DelayResult,
                timeout_seconds=self.timeout_seconds,
            ),
            "fake_cancellable": ToolDefinition(
                name="fake_cancellable",
                description="Remain in flight until completed or externally cancelled.",
                args_model=DelayArgs,
                handler=self._cancellable,
                result_model=DelayResult,
                timeout_seconds=self.cancellation_timeout_seconds,
            ),
            "fake_invalid_result": ToolDefinition(
                name="fake_invalid_result",
                description="Return a payload that violates its declared result schema.",
                args_model=EchoArgs,
                handler=self._invalid_result,
                result_model=StrictResult,
            ),
            "fake_write_side_effect": ToolDefinition(
                name="fake_write_side_effect",
                description="Increment an in-memory counter when an approved write runs.",
                args_model=WriteArgs,
                handler=self._write_side_effect,
                result_model=WriteResult,
                risk=ToolRisk.WRITE,
            ),
            "fake_dangerous_side_effect": ToolDefinition(
                name="fake_dangerous_side_effect",
                description="Increment an in-memory counter for a dangerous operation.",
                args_model=WriteArgs,
                handler=self._dangerous_side_effect,
                result_model=WriteResult,
                risk=ToolRisk.DANGEROUS,
            ),
            "fake_dependency_failure": ToolDefinition(
                name="fake_dependency_failure",
                description="Raise a deterministic downstream dependency exception.",
                args_model=FailureArgs,
                handler=self._dependency_failure,
            ),
            "fake_serial_counter": ToolDefinition(
                name="fake_serial_counter",
                description="Measure whether non-concurrency-safe calls are serialized.",
                args_model=SerialArgs,
                handler=self._serial_counter,
                result_model=SerialResult,
                timeout_seconds=5.0,
                concurrency_safe=False,
            ),
            "fake_large_result": ToolDefinition(
                name="fake_large_result",
                description="Return a deterministic large result for payload overhead tests.",
                args_model=LargeResultArgs,
                handler=self._large_result,
                result_model=LargeResult,
            ),
        }

    def _begin(self, name: str) -> None:
        self.state.call_counts[name] += 1
        self.state.record(name, "started", call=self.state.call_counts[name])

    def _echo_success(self, args: Any, context: ToolCallContext) -> Dict[str, Any]:
        name = "fake_echo_success"
        self._begin(name)
        text = _arg_value(args, "text") * _arg_value(args, "repeat", 1)
        self.state.record(name, "completed")
        return {"text": text, "length": len(text)}

    async def _search_success(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        name = "fake_search_success"
        self._begin(name)
        query = _arg_value(args, "query")
        count = _arg_value(args, "count", 3)
        results = [
            {
                "rank": rank,
                "title": f"{query}-result-{rank}",
                "url": f"https://example.test/search/{rank}",
                "snippet": f"deterministic snippet {rank} for {query}",
            }
            for rank in range(1, count + 1)
        ]
        self.state.record(name, "completed", result_count=len(results))
        return {"query": query, "results": results}

    def _context_probe(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        name = "fake_context_probe"
        self._begin(name)
        self.state.record(name, "completed", run_id=context.run_id)
        return {
            "label": _arg_value(args, "label", "probe"),
            "session_id": context.session_id,
            "run_id": context.run_id,
            "user_id": context.user_id,
        }

    async def _slow_timeout(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        name = "fake_slow_timeout"
        self._begin(name)
        delay_ms = _arg_value(args, "delay_ms", 100)
        try:
            await asyncio.sleep(delay_ms / 1_000)
        except asyncio.CancelledError:
            self.state.cancellation_counts[name] += 1
            self.state.record(name, "cancelled")
            raise
        self.state.record(name, "completed")
        return {
            "value": _arg_value(args, "value", "done"),
            "delay_ms": delay_ms,
        }

    async def _cancellable(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        name = "fake_cancellable"
        self._begin(name)
        delay_ms = _arg_value(args, "delay_ms", 100)
        try:
            await asyncio.sleep(delay_ms / 1_000)
        except asyncio.CancelledError:
            self.state.cancellation_counts[name] += 1
            self.state.record(name, "cancelled")
            raise
        self.state.record(name, "completed")
        return {
            "value": _arg_value(args, "value", "done"),
            "delay_ms": delay_ms,
        }

    def _invalid_result(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        name = "fake_invalid_result"
        self._begin(name)
        self.state.record(name, "returned_invalid_payload")
        return {"status": "not-ok", "missing_payload": _arg_value(args, "text")}

    def _run_side_effect(self, name: str, args: Any) -> Dict[str, Any]:
        self._begin(name)
        self.state.side_effect_counts[name] += 1
        record_id = _arg_value(args, "record_id")
        self.state.record(name, "side_effect", record_id=record_id)
        return {"success": True, "record_id": record_id}

    def _write_side_effect(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        return self._run_side_effect("fake_write_side_effect", args)

    def _dangerous_side_effect(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        return self._run_side_effect("fake_dangerous_side_effect", args)

    def _dependency_failure(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        name = "fake_dependency_failure"
        self._begin(name)
        message = _arg_value(args, "message", "fake dependency unavailable")
        self.state.record(name, "failed", message=message)
        raise FakeDependencyError(message)

    async def _serial_counter(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        name = "fake_serial_counter"
        self._begin(name)
        sequence = _arg_value(args, "sequence")
        delay_ms = _arg_value(args, "delay_ms", 10)
        self.state.active_counts[name] += 1
        self.state.max_active_counts[name] = max(
            self.state.max_active_counts[name], self.state.active_counts[name]
        )
        try:
            await asyncio.sleep(delay_ms / 1_000)
            self.state.record(name, "completed", sequence=sequence)
            return {"sequence": sequence}
        finally:
            self.state.active_counts[name] -= 1

    def _large_result(
        self, args: Any, context: ToolCallContext
    ) -> Dict[str, Any]:
        name = "fake_large_result"
        self._begin(name)
        items = list(range(_arg_value(args, "item_count", 100)))
        self.state.record(name, "completed", item_count=len(items))
        return {"items": items}
