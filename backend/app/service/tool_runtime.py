# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""所有 Agent 工具共用的安全执行管线。"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Type

from pydantic import BaseModel, ValidationError
from service.agent_history import current_history_scope, record_agent_message

from service.research_control import (
    ResearchCancellationToken,
    ResearchCancelledError,
    get_current_cancellation_token,
)

logger = logging.getLogger(__name__)


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    DANGEROUS = "dangerous"


class ToolErrorCode(str, Enum):
    INVALID_ARGUMENTS = "invalid_arguments"
    APPROVAL_REQUIRED = "approval_required"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    EXECUTION_FAILED = "execution_failed"
    INVALID_RESULT = "invalid_result"


class ToolRuntimeError(RuntimeError):
    def __init__(self, code: ToolErrorCode, message: str):
        super().__init__(message)
        self.code = code.value


@dataclass
class ToolCallContext:
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    user_id: Optional[str] = None
    approved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancellation_token: Optional[ResearchCancellationToken] = None


ToolHandler = Callable[[BaseModel, ToolCallContext], Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: Type[BaseModel]
    handler: ToolHandler
    result_model: Optional[Type[BaseModel]] = None
    timeout_seconds: float = 30.0
    risk: ToolRisk = ToolRisk.READ_ONLY
    concurrency_safe: bool = True


class ToolMiddleware(Protocol):
    async def before(self, definition: ToolDefinition, args: BaseModel, context: ToolCallContext) -> None: ...
    async def after(self, definition: ToolDefinition, result: Any, context: ToolCallContext) -> None: ...
    async def on_error(self, definition: ToolDefinition, error: BaseException, context: ToolCallContext) -> None: ...


class LoggingToolMiddleware:
    async def before(self, definition: ToolDefinition, args: BaseModel, context: ToolCallContext) -> None:
        logger.info("[Tool] start name=%s run_id=%s", definition.name, context.run_id)

    async def after(self, definition: ToolDefinition, result: Any, context: ToolCallContext) -> None:
        logger.info("[Tool] complete name=%s run_id=%s", definition.name, context.run_id)

    async def on_error(self, definition: ToolDefinition, error: BaseException, context: ToolCallContext) -> None:
        logger.warning(
            "[Tool] failed name=%s run_id=%s error=%s",
            definition.name,
            context.run_id,
            type(error).__name__,
        )


@dataclass
class ToolExecutionResult:
    call_id: str
    tool_name: str
    value: Any
    duration_ms: int


class ToolRuntime:
    """统一完成校验、审批、取消、超时和执行日志。"""

    def __init__(self, middleware: Optional[List[ToolMiddleware]] = None):
        self.middleware = middleware or [LoggingToolMiddleware()]
        self._locks: Dict[str, asyncio.Lock] = {}

    async def execute(
        self,
        definition: ToolDefinition,
        raw_args: Dict[str, Any],
        context: Optional[ToolCallContext] = None,
    ) -> ToolExecutionResult:
        context = context or ToolCallContext()
        context.cancellation_token = (
            context.cancellation_token or get_current_cancellation_token()
        )
        call_id = str(uuid.uuid4())
        started = time.monotonic()
        history_scope = current_history_scope()
        agent_name = context.metadata.get("agent_name")
        if history_scope is not None and not agent_name:
            raise ValueError("研究工具缺少 agent_name，拒绝写入错误的历史")

        async def record(kind, payload):
            if history_scope is not None:
                await record_agent_message(agent_name, kind, payload, call_id)

        await record("tool_input", {"tool_name": definition.name, "arguments": raw_args})

        try:
            try:
                args = definition.args_model.model_validate(raw_args)
            except ValidationError as exc:
                raise ToolRuntimeError(
                    ToolErrorCode.INVALID_ARGUMENTS,
                    f"工具 {definition.name} 参数不合法: {exc}",
                ) from exc

            if definition.risk in {ToolRisk.WRITE, ToolRisk.DANGEROUS} and not context.approved:
                raise ToolRuntimeError(
                    ToolErrorCode.APPROVAL_REQUIRED,
                    f"工具 {definition.name} 属于高风险操作，需要用户确认",
                )

            if context.cancellation_token:
                context.cancellation_token.raise_if_cancelled()

            for item in self.middleware:
                await item.before(definition, args, context)
            await record("tool_dispatch", {"tool_name": definition.name, "arguments": args.model_dump(mode="json")})

            async def invoke() -> Any:
                value = definition.handler(args, context)
                if inspect.isawaitable(value):
                    value = await value
                return value

            async def invoke_with_policy() -> Any:
                if definition.concurrency_safe:
                    return await invoke()
                lock = self._locks.setdefault(definition.name, asyncio.Lock())
                async with lock:
                    return await invoke()

            execution = asyncio.create_task(invoke_with_policy())
            cancellation_waiter: Optional[asyncio.Task] = None
            try:
                if context.cancellation_token:
                    cancellation_waiter = asyncio.create_task(context.cancellation_token.wait())

                waiters = {execution}
                if cancellation_waiter:
                    waiters.add(cancellation_waiter)
                done, _ = await asyncio.wait(
                    waiters,
                    timeout=definition.timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if cancellation_waiter and cancellation_waiter in done:
                    raise ToolRuntimeError(ToolErrorCode.CANCELLED, "工具执行已取消")
                if execution not in done:
                    raise ToolRuntimeError(
                        ToolErrorCode.TIMEOUT,
                        f"工具 {definition.name} 超过 {definition.timeout_seconds:g} 秒",
                    )

                value = await execution
            finally:
                # asyncio.wait 不会在调用方被取消时连带取消子任务。
                # 所有出口都先回收 HTTP/工具执行与取消监听，避免超时或
                # 外层 wait_for 取消后继续后台请求、占锁或轮询 Redis。
                children = [execution]
                if cancellation_waiter is not None:
                    children.append(cancellation_waiter)
                for child in children:
                    if not child.done():
                        child.cancel()
                await asyncio.gather(*children, return_exceptions=True)

            await record("tool_raw_result", {"tool_name": definition.name, "value": value})
            if definition.result_model is not None:
                try:
                    value = definition.result_model.model_validate(value).model_dump()
                except ValidationError as exc:
                    raise ToolRuntimeError(
                        ToolErrorCode.INVALID_RESULT,
                        f"工具 {definition.name} 返回格式不合法: {exc}",
                    ) from exc

            for item in reversed(self.middleware):
                await item.after(definition, value, context)
            await record("tool_output", {"tool_name": definition.name, "value": value})

            return ToolExecutionResult(
                call_id=call_id,
                tool_name=definition.name,
                value=value,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except ResearchCancelledError as exc:
            error = ToolRuntimeError(ToolErrorCode.CANCELLED, str(exc))
            await record("tool_error", {"tool_name": definition.name, "error_type": type(error).__name__, "message": str(error)})
            for item in reversed(self.middleware):
                await item.on_error(definition, error, context)
            raise error from exc
        except BaseException as exc:
            await record("tool_error", {"tool_name": definition.name, "error_type": type(exc).__name__, "message": str(exc)})
            for item in reversed(self.middleware):
                await item.on_error(definition, exc, context)
            raise
