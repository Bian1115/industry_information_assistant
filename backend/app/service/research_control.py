# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""研究任务的 run_id 级取消控制。

Redis 负责跨进程传递取消信号，本地 ``asyncio.Event`` 负责让当前进程中
正在等待 LLM/工具的协程立即醒来。旧的 session_id 取消键仍保留兼容。
"""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

from core.redis_client import cache


CANCEL_KEY_PREFIX = "research:cancel:run:"
LEGACY_CANCEL_KEY_PREFIX = "research:cancel:"
_tokens: Dict[str, "ResearchCancellationToken"] = {}
_current_token: contextvars.ContextVar[Optional["ResearchCancellationToken"]] = (
    contextvars.ContextVar("research_cancellation_token", default=None)
)


class ResearchCancelledError(asyncio.CancelledError):
    """带稳定错误码的用户取消异常。"""

    code = "research_cancelled"


@dataclass
class ResearchCancellationToken:
    run_id: str
    session_id: str
    _event: asyncio.Event = field(default_factory=asyncio.Event)

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        if self._event.is_set():
            return True
        result = cache.get(f"{CANCEL_KEY_PREFIX}{self.run_id}")
        if result and result.get("cancelled", False):
            self._event.set()
            return True
        # 兼容旧客户端按 session_id 发出的取消请求。
        legacy = cache.get(f"{LEGACY_CANCEL_KEY_PREFIX}{self.session_id}")
        if legacy and legacy.get("cancelled", False):
            self._event.set()
            return True
        return False

    async def wait(self, poll_interval: float = 0.25) -> None:
        while not self.is_cancelled():
            try:
                await asyncio.wait_for(self._event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                continue

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise ResearchCancelledError("研究已被用户取消")


def create_cancellation_token(run_id: str, session_id: str) -> ResearchCancellationToken:
    token = ResearchCancellationToken(run_id=run_id, session_id=session_id)
    _tokens[run_id] = token
    return token


def release_cancellation_token(run_id: str) -> None:
    _tokens.pop(run_id, None)


def request_cancel(run_id: str, session_id: Optional[str] = None) -> None:
    cache.set(
        f"{CANCEL_KEY_PREFIX}{run_id}",
        {"cancelled": True, "run_id": run_id, "session_id": session_id},
        expire=300,
    )
    token = _tokens.get(run_id)
    if token is not None:
        token.cancel()


def request_legacy_session_cancel(session_id: str) -> None:
    cache.set(
        f"{LEGACY_CANCEL_KEY_PREFIX}{session_id}",
        {"cancelled": True, "session_id": session_id},
        expire=300,
    )
    for token in list(_tokens.values()):
        if token.session_id == session_id:
            token.cancel()


def clear_cancel_flag(run_id: str, session_id: Optional[str] = None) -> None:
    cache.delete(f"{CANCEL_KEY_PREFIX}{run_id}")
    if session_id:
        cache.delete(f"{LEGACY_CANCEL_KEY_PREFIX}{session_id}")


def is_research_cancelled(run_id: str, session_id: Optional[str] = None) -> bool:
    token = _tokens.get(run_id)
    if token is not None:
        return token.is_cancelled()
    result = cache.get(f"{CANCEL_KEY_PREFIX}{run_id}")
    if result and result.get("cancelled", False):
        return True
    if session_id:
        legacy = cache.get(f"{LEGACY_CANCEL_KEY_PREFIX}{session_id}")
        return bool(legacy and legacy.get("cancelled", False))
    return False


def get_current_cancellation_token() -> Optional[ResearchCancellationToken]:
    return _current_token.get()


@contextmanager
def cancellation_context(token: ResearchCancellationToken) -> Iterator[None]:
    marker = _current_token.set(token)
    try:
        yield
    finally:
        try:
            _current_token.reset(marker)
        except ValueError:
            # 异步生成器可能由 Python 在另一个 Context 中执行 aclose（例如
            # wait_for 超时或消费者中断）。ContextVar token 不能跨 Context
            # reset；原 Task 已结束，此处只需清理当前终结器上下文，避免产生
            # “Task exception was never retrieved”并掩盖真正的超时原因。
            if _current_token.get() is token:
                _current_token.set(None)
