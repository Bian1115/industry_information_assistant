# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""记忆系统的稳定边界。

聊天、路由和任务调度只依赖这里的数据结构与协议。具体实现可以是当前
PostgreSQL + Milvus 方案，也可以在后续替换为 LangMem 等 Provider。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence


MemoryRole = Literal["user", "assistant", "system"]
MemoryOperationStatus = Literal["pending", "running", "retry", "succeeded", "dead"]


@dataclass(frozen=True)
class MemoryScope:
    user_id: str
    session_id: Optional[str] = None


@dataclass(frozen=True)
class MemoryTurn:
    message_id: str
    role: MemoryRole
    content: str
    created_at: datetime


@dataclass(frozen=True)
class MemorySearchOptions:
    limit: int = 5
    memory_types: Optional[Sequence[str]] = None
    session_id: Optional[str] = None
    min_score: Optional[float] = None


@dataclass(frozen=True)
class MemoryHit:
    memory_id: str
    fragment_id: str
    memory_type: str
    content: str
    score: float
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryPatch:
    summary: Optional[str] = None
    key_insights: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryWriteResult:
    operation_id: str
    status: MemoryOperationStatus
    memory_ids: List[str] = field(default_factory=list)


class MemoryProvider(Protocol):
    """所有记忆 Provider 都必须实现的统一接口。"""

    name: str

    def add_turns(
        self,
        scope: MemoryScope,
        turns: Sequence[MemoryTurn],
        operation_id: str,
    ) -> MemoryWriteResult:
        ...

    def search(
        self,
        scope: MemoryScope,
        query: str,
        options: MemorySearchOptions,
    ) -> List[MemoryHit]:
        ...

    def update(
        self,
        scope: MemoryScope,
        memory_id: str,
        patch: MemoryPatch,
        operation_id: str,
    ) -> MemoryWriteResult:
        ...

    def delete(
        self,
        scope: MemoryScope,
        memory_id: str,
        operation_id: str,
    ) -> None:
        ...

    def reset_user(self, user_id: str, operation_id: str) -> None:
        ...
