"""按研究 run、智能体、子任务保存的 PostgreSQL 工作历史。

ContextVar 只传递本次执行身份，不把数据库连接放进 LangGraph 状态。
一次工具调用由 call_id 关联输入/输出；并行返回顺序与发起顺序分别保留。
历史服务仅清洗 PostgreSQL 不接受的空字符并记录审计，摘要覆盖游标只影响后续输入的组装。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy import select, update, func, text
from sqlalchemy.dialects.postgresql import insert

from core.database import SessionLocal
from models.agent_history import AgentContext, AgentMessage
from models.research import ResearchRun
from service.context_policy import storage_context_key

logger = logging.getLogger(__name__)


def sanitize_history_payload(value: Any) -> tuple[Any, int]:
    """复制 JSON 值并仅移除实际 NUL；键不改名，避免静默合并不同字段。"""
    if isinstance(value, str):
        return value.replace("\x00", ""), value.count("\x00")
    if isinstance(value, (list, tuple)):
        items = [sanitize_history_payload(item) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    if isinstance(value, dict):
        result, removed = {}, 0
        for key, item in value.items():
            if isinstance(key, str) and "\x00" in key:
                raise ValueError("工作历史 JSON 字段名含空字符，不能安全改名")
            result[key], count = sanitize_history_payload(item)
            removed += count
        return result, removed
    return value, 0


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def exception_details(exc: BaseException) -> dict:
    details = {"error_type": type(exc).__name__, "error_module": type(exc).__module__, "message": str(exc)}
    if exc.__cause__ is not None:
        details.update(cause_type=type(exc.__cause__).__name__, cause_message=str(exc.__cause__))
    if getattr(exc, "summary_phase", None):
        details["phase"] = exc.summary_phase
    return details


class AgentHistoryError(RuntimeError):
    """持久化失败必须中止本节点，不能伪装成已记录。"""
    code = "agent_history_persistence_failed"


@dataclass(frozen=True)
class HistoryScope:
    store: Any
    run_id: str
    session_id: str
    user_id: str | None
    node: str = ""
    context_key: str = "main"
    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    contract: dict = field(default_factory=dict)
    fencing_token: int | None = None
    context_policy: dict | None = None


_scope: ContextVar[HistoryScope | None] = ContextVar("agent_history_scope", default=None)


def current_history_scope() -> HistoryScope | None:
    return _scope.get()


@contextmanager
def bind_agent_history(scope: HistoryScope | None):
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


@contextmanager
def agent_subtask(context_key: str):
    """并行子任务必须给出稳定业务 key；恢复时复用，不按请求序号猜测。"""
    scope = _scope.get()
    with bind_agent_history(replace(scope, context_key=context_key) if scope else None):
        yield


def session_history_scope_id(scope):
    """只用于识别旧版会话共享记录，不作为新模型历史的读取边界。"""
    if not scope.session_id:
        raise ValueError("会话历史要求session_id")
    return "session:" + content_hash([str(scope.user_id) if scope.user_id else None, scope.session_id])


def research_history_scope_id(scope):
    if not scope.run_id or not scope.session_id:
        raise ValueError("研究内历史要求run_id和session_id")
    return "research:" + content_hash([
        str(scope.user_id) if scope.user_id else None, scope.session_id, scope.run_id,
    ])


class PostgresAgentHistory:
    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    @staticmethod
    def _authorize(db, scope):
        # 与租约接管共用 run 行锁，防止验证后旧执行器仍然提交摘要。
        run = db.execute(select(ResearchRun).where(ResearchRun.run_id == scope.run_id).with_for_update()).scalar_one_or_none()
        owner = str(run.user_id) if run and run.user_id else None
        if not run or run.session_id != scope.session_id or owner != (str(scope.user_id) if scope.user_id else None):
            raise PermissionError("智能体历史不属于当前研究运行/用户")
        if scope.fencing_token is not None and run.fencing_token != scope.fencing_token:
            raise AgentHistoryError("智能体执行租约已被新执行器接管")

    def _context(self, db, scope, agent):
        self._authorize(db, scope)
        if not agent or len(agent) > 64 or not scope.context_key or len(scope.context_key) > 128:
            raise ValueError("智能体或子任务标识不合法")
        context_key = storage_context_key(scope)
        history_scope_id = research_history_scope_id(scope)
        # 同一研究/智能体的创建、旧记录导入和写入共用事务锁。
        lock_key = int.from_bytes(hashlib.sha256(canonical_json(
            [history_scope_id, agent, context_key]).encode()).digest()[:8], "big", signed=True)
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})
        context_id = uuid.uuid5(uuid.NAMESPACE_URL, canonical_json([history_scope_id, agent, context_key]))
        db.execute(insert(AgentContext).values(
            id=context_id, run_id=scope.run_id, agent_name=agent,
            history_scope_id=history_scope_id, context_key=context_key,
        ).on_conflict_do_nothing(index_elements=["history_scope_id", "agent_name", "context_key"]))
        ctx = db.execute(select(AgentContext).where(
            AgentContext.history_scope_id == history_scope_id,
            AgentContext.agent_name == agent,
            AgentContext.context_key == context_key,
        ).with_for_update()).scalar_one()
        self._adopt_legacy_contexts(db, scope, agent, context_key, ctx)
        return ctx

    def _adopt_legacy_contexts(self, db, scope, agent, context_key, ctx):
        """仅导入当前run的旧记录；混合多个run的旧摘要保留在旧context供审计。"""
        legacy = db.execute(select(AgentContext).where(
            AgentContext.id != ctx.id,
            AgentContext.history_scope_id.in_([
                "run:" + scope.run_id, session_history_scope_id(scope),
            ]),
            AgentContext.agent_name == agent, AgentContext.context_key == context_key,
            AgentContext.id.in_(select(AgentMessage.context_id).where(
                AgentMessage.run_id == scope.run_id)),
        ).order_by(AgentContext.created_at, AgentContext.id)
            .with_for_update()).scalars().all()
        for old in legacy:
            rows = db.execute(select(AgentMessage).where(
                AgentMessage.context_id == old.id, AgentMessage.run_id == scope.run_id,
            ).order_by(AgentMessage.seq)).scalars().all()
            total = db.execute(select(func.count()).select_from(AgentMessage).where(
                AgentMessage.context_id == old.id)).scalar_one()
            offset = ctx.next_seq
            # 只有独立旧run且完整序号不变，才允许沿用其摘要和覆盖游标。
            preserve_summary = (offset == 0 and ctx.summary_version == 0
                and old.history_scope_id == "run:" + scope.run_id
                and total == len(rows) == old.next_seq
                and [row.seq for row in rows] == list(range(1, len(rows) + 1)))
            for number, row in enumerate(rows, start=1):
                row.context_id, row.seq = ctx.id, offset + number
            ctx.next_seq += len(rows)
            source_ids = {str(row.id) for row in rows}
            ctx.offloaded_source_ids = sorted(set(ctx.offloaded_source_ids) |
                (set(old.offloaded_source_ids) & source_ids))
            if preserve_summary:
                ctx.summary, ctx.summary_version = old.summary, old.summary_version
                ctx.covered_seq, ctx.summary_metadata = old.covered_seq, old.summary_metadata
            # 原记录UUID和payload不变；旧context及其旧摘要不删除。
            db.flush()

    @staticmethod
    def _message(row):
        return {
            "source_id": str(row.id), "seq": row.seq, "kind": row.kind,
            "call_id": str(row.call_id), "invocation_id": str(row.invocation_id),
            "node": row.node, "payload": row.payload, "content_hash": row.content_hash,
            "run_id": row.run_id,
        }

    def _append(self, scope, agent, kind, payload, call_id, event_id):
        # 复制且校验 JSON，调用者后续修改对象不会改变快照。
        payload = json.loads(canonical_json(payload))
        cleaned, removed = sanitize_history_payload(payload)
        if removed:
            logger.warning(
                "工作历史空字符清洗 run=%s agent=%s kind=%s call_id=%s removed=%d raw_sha256=%s",
                scope.run_id, agent, kind, call_id, removed, content_hash(payload),
            )
        payload = cleaned
        digest = content_hash(payload)
        with self.session_factory.begin() as db:
            ctx = self._context(db, scope, agent)
            # 原子分配序号；同一上下文的并发写在此行锁上串行化。
            seq = db.execute(update(AgentContext).where(AgentContext.id == ctx.id).values(
                next_seq=AgentContext.next_seq + 1, updated_at=func.now(),
            ).returning(AgentContext.next_seq)).scalar_one()
            existing = db.get(AgentMessage, uuid.UUID(event_id))
            if existing:
                if existing.context_id != ctx.id or existing.run_id != scope.run_id or existing.content_hash != digest or existing.kind != kind or str(existing.call_id) != call_id:
                    raise ValueError("历史幂等键对应的原文发生变化")
                # 重复提交不消耗一个新序号。
                db.execute(update(AgentContext).where(AgentContext.id == ctx.id).values(next_seq=seq - 1))
                return self._message(existing)
            row = AgentMessage(
                id=uuid.UUID(event_id), context_id=ctx.id, run_id=scope.run_id, seq=seq, kind=kind,
                call_id=uuid.UUID(call_id), invocation_id=uuid.UUID(scope.invocation_id),
                node=scope.node, payload=payload, content_hash=digest,
            )
            db.add(row)
            db.flush()
            return self._message(row)

    async def append(self, scope, agent, kind, payload, call_id, *, event_id=None):
        try:
            return await asyncio.to_thread(self._append, scope, agent, kind, payload, str(call_id), event_id or str(uuid.uuid4()))
        except (PermissionError, ValueError):
            raise
        except Exception as exc:
            raise AgentHistoryError("智能体工作历史保存失败") from exc

    def _load(self, scope, agent, replay_only=False):
        with self.session_factory.begin() as db:
            ctx = self._context(db, scope, agent)
            # 一致视图：短暂锁定 context，与 append/save_summary 共用同一锁。
            db.refresh(ctx, with_for_update=True)
            query = select(AgentMessage).where(
                AgentMessage.context_id == ctx.id,
                AgentMessage.run_id == scope.run_id,
            )
            if replay_only:
                query = query.where(AgentMessage.kind.in_([
                    "llm_input", "llm_output", "llm_intermediate", "history_read", "llm_error",
                    "tool_input", "tool_output", "tool_error",
                    "input_history_view",  # 旧版缓存保留审计，不直接作为对话回放。
                ]), AgentMessage.seq > ctx.covered_seq)
            rows = db.execute(query.order_by(AgentMessage.seq)).scalars().all()
            run_ids = {row.run_id for row in rows}
            statuses = dict(db.execute(select(ResearchRun.run_id, ResearchRun.status).where(
                ResearchRun.run_id.in_(run_ids))).all()) if run_ids else {}
            return {
                "context_id": str(ctx.id), "summary": ctx.summary,
                "summary_version": ctx.summary_version, "covered_seq": ctx.covered_seq,
                "summary_metadata": ctx.summary_metadata,
                "offloaded_source_ids": ctx.offloaded_source_ids,
                "run_statuses": statuses,
                "messages": [self._message(row) for row in rows],
            }

    async def load(self, scope, agent, *, replay_only=False):
        return await asyncio.to_thread(self._load, scope, agent, replay_only)

    async def load_tool_sources(self, scope, agent, *, through_seq):
        """仅供本地引用匹配回查已覆盖的工具结果，不把旧正文重新加入模型上下文。"""
        def read():
            with self.session_factory.begin() as db:
                ctx = self._context(db, scope, agent)
                rows = db.execute(select(AgentMessage).where(
                    AgentMessage.context_id == ctx.id,
                    AgentMessage.run_id == scope.run_id,
                    AgentMessage.kind == "tool_output",
                    AgentMessage.seq <= through_seq,
                ).order_by(AgentMessage.seq)).scalars().all()
                return [self._message(row) for row in rows]
        return await asyncio.to_thread(read)

    async def read_source(self, scope, agent, source_id):
        """内部受控回读：仅当前run、当前agent/subtask的原文。"""
        def read():
            with self.session_factory.begin() as db:
                ctx = self._context(db, scope, agent)
                row = db.execute(select(AgentMessage).where(
                    AgentMessage.id == uuid.UUID(source_id),
                    AgentMessage.context_id == ctx.id,
                    AgentMessage.run_id == scope.run_id,
                )).scalar_one_or_none()
                if row is None:
                    raise PermissionError("无法读取其他上下文的原文")
                return self._message(row)
        return await asyncio.to_thread(read)

    async def save_summary(self, scope, agent, *, expected_version, covered_seq, summary, metadata):
        def save():
            with self.session_factory.begin() as db:
                ctx = self._context(db, scope, agent)
                result = db.execute(update(AgentContext).where(
                    AgentContext.id == ctx.id,
                    AgentContext.summary_version == expected_version,
                    AgentContext.covered_seq < covered_seq,
                    AgentContext.next_seq >= covered_seq,
                ).values(summary=summary, covered_seq=covered_seq,
                         summary_version=expected_version + 1,
                         summary_metadata=metadata, updated_at=func.now()))
                return result.rowcount == 1
        return await asyncio.to_thread(save)

    async def mark_offloaded(self, scope, agent, source_ids):
        def save():
            with self.session_factory.begin() as db:
                ctx = self._context(db, scope, agent)
                db.refresh(ctx, with_for_update=True)
                ctx.offloaded_source_ids = sorted(set(ctx.offloaded_source_ids) | set(source_ids))
        await asyncio.to_thread(save)

    async def update_summary_metadata(self, scope, agent, *, expected_version, expected_summary_hash, metadata):
        """核验只注释已经保存的同一版本正文，不覆盖并发产生的新摘要。"""
        def save():
            with self.session_factory.begin() as db:
                ctx = self._context(db, scope, agent)
                db.refresh(ctx, with_for_update=True)
                if ctx.summary_version != expected_version or content_hash(ctx.summary) != expected_summary_hash:
                    return False
                ctx.summary_metadata = json.loads(canonical_json(metadata))
                return True
        return await asyncio.to_thread(save)


async def record_agent_message(agent, kind, payload, call_id, *, scope=None):
    scope = scope or current_history_scope()
    if scope is None:
        return None  # 无研究运行身份的离线调用由调用方显式注入 scope。
    try:
        return await scope.store.append(scope, agent, kind, payload, call_id)
    except AgentHistoryError:
        raise
    except Exception as exc:
        raise AgentHistoryError("智能体历史保存或归属校验失败") from exc
