# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""模型调用前准备会话滚动摘要；原始消息始终留在 PostgreSQL。"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from core.time_utils import as_utc, utc_now
from models.chat import ChatMessage, ChatSession, MemoryJob
from service.session_summarization import SessionSummarizer, SessionSummaryConfig, summary_message
from prompts.session_summary import PROMPT_VERSION as ROLLING_PROMPT_VERSION


logger = logging.getLogger(__name__)

SEMANTIC_JOB_TYPE = "semantic_context"
ACTIVE_JOB_STATUSES = ("pending", "running", "retry")
LEGACY_PROMPT_VERSION = "semantic-context-v1"

SemanticCategory = Literal[
    "goal",
    "scope",
    "constraint",
    "fact",
    "decision",
    "open_question",
    "preference",
]


class SemanticContextItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: SemanticCategory
    text: str
    source_message_ids: List[str]
    evidence_quote: str
    source_role: Literal["user", "assistant"]
    status: Literal["active", "superseded"] = "active"
    superseded_by: Optional[str] = None
    confidence: float = 1.0


class SemanticContextSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    version: int = 0
    prompt_version: str = LEGACY_PROMPT_VERSION
    model: str = ""
    overview: str = ""
    items: List[SemanticContextItem] = Field(default_factory=list)
    covered_until_message_id: Optional[str] = None
    updated_at: Optional[str] = None


def semantic_context_enabled() -> bool:
    from service.context_policy import session_context_compression_enabled
    if not session_context_compression_enabled():
        return False
    value = os.getenv("SEMANTIC_CONTEXT_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def _as_uuid(value: str) -> UUID:
    return UUID(str(value))


def _message_id(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("id") or message.get("message_id") or "")
    return str(getattr(message, "id", ""))


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", ""))


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", ""))


def parse_semantic_summary(value: Any) -> SemanticContextSummary:
    if not value:
        return SemanticContextSummary()
    try:
        return SemanticContextSummary.model_validate(value)
    except ValidationError as exc:
        logger.warning("已有会话语义摘要格式无效，重新开始抽取: %s", exc)
        return SemanticContextSummary()


def render_semantic_context(value: Any) -> str:
    """把结构化摘要渲染为明确标注为非事实证据的 Prompt 上下文。"""
    if isinstance(value, dict) and value.get("schema_version") == 2:
        text = str(value.get("summary") or "").strip()
        return summary_message(text)["content"] if text else ""
    summary = parse_semantic_summary(value)
    active = [item for item in summary.items if item.status == "active"]
    if not active:
        return ""

    labels = {
        "goal": "当前目标",
        "scope": "研究范围",
        "constraint": "有效约束",
        "fact": "历史陈述",
        "decision": "用户决策",
        "open_question": "待确认问题",
        "preference": "用户偏好",
    }
    lines = [
        "[会话语义摘要：仅用于理解任务和指代，不是本轮研究事实证据]",
    ]
    for category in labels:
        selected = [item for item in active if item.category == category]
        if not selected:
            continue
        lines.append(f"{labels[category]}：")
        for item in selected:
            sources = ",".join(item.source_message_ids)
            lines.append(f"- {item.text} [来源消息: {sources}]")
    return "\n".join(lines)


class RollingSemanticSummary(BaseModel):
    """同一份摘要及覆盖游标原子提交；不删除原始消息。"""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2] = 2
    version: int = 0
    prompt_version: str = ROLLING_PROMPT_VERSION
    model: str = ""
    summary: str = ""
    covered_until_message_id: Optional[str] = None
    covered_until_created_at: Optional[str] = None
    updated_at: Optional[str] = None
    config_fingerprint: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)


class SemanticContextService(SessionSummarizer):
    """与 SummarizationMiddleware 一样生成完整新摘要，替换旧摘要。"""

    def update_summary(self, previous_value, messages):
        previous = _rolling_summary(previous_value)
        working = _working_messages(previous, messages)
        if not working:
            return previous
        text, metrics = self.summarize(working)
        last = _record(messages[-1]) if messages else None
        return RollingSemanticSummary(
            version=previous.version + 1, model=self.config.model, summary=text,
            covered_until_message_id=last["id"] if last else previous.covered_until_message_id,
            covered_until_created_at=last["created_at"] if last else previous.covered_until_created_at,
            updated_at=utc_now().isoformat(timespec="seconds"),
            config_fingerprint=self.config.fingerprint, metrics=metrics,
        )


def _rolling_summary(value: Any) -> RollingSemanticSummary:
    # v1 条目摘要没有完整原文窗口。首次启用新策略时从 PostgreSQL 原文重建，
    # 不沿用旧覆盖游标，避免将被旧条目抽取遗漏的信息永久跳过。
    if isinstance(value, RollingSemanticSummary):
        return value
    if isinstance(value, dict) and value.get("schema_version") == 2:
        return RollingSemanticSummary.model_validate(value)
    return RollingSemanticSummary()


def _record(message: Any) -> Dict[str, Any]:
    value = message.get("created_at") if isinstance(message, dict) else getattr(message, "created_at", None)
    if isinstance(value, datetime):
        created_at = as_utc(value).isoformat(timespec="microseconds")
    else:
        created_at = str(value or "")
    return {
        "id": _message_id(message), "created_at": created_at,
        "role": _message_role(message), "content": _message_content(message),
    }


def _working_messages(summary: RollingSemanticSummary, rows: Sequence[Any]) -> List[Dict[str, Any]]:
    records = sorted((_record(row) for row in rows), key=lambda item: (item["created_at"], item["id"]))
    if summary.covered_until_message_id and summary.covered_until_created_at:
        cursor = (summary.covered_until_created_at, summary.covered_until_message_id)
        records = [item for item in records if (item["created_at"], item["id"]) > cursor]
    if summary.summary:
        return [summary_message(summary.summary), *records]
    return records


_semantic_service: Optional[SemanticContextService] = None


def get_semantic_context_service() -> SemanticContextService:
    global _semantic_service
    config = SessionSummaryConfig.from_env()
    if _semantic_service is None or _semantic_service.config.fingerprint != config.fingerprint:
        _semantic_service = SemanticContextService(config=config)
    return _semantic_service


@dataclass(frozen=True)
class PreparedSemanticContext:
    messages: List[Dict[str, str]]
    original_tokens: int
    compacted_tokens: int
    summarized_messages: int
    semantic_context_version: int
    last_message_id: Optional[str]
    metrics: Dict[str, Any]
    config_fingerprint: str


class SemanticContextError(RuntimeError):
    """摘要失败必须阻止无历史降级；账本及覆盖游标不变。"""


def _session_rows(db: Session, session_id: UUID, exclude_turn_id: Optional[str]) -> List[ChatMessage]:
    query = db.query(ChatMessage).filter(ChatMessage.session_id == session_id)
    rows = query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).all()
    if exclude_turn_id:
        turn_id = _as_uuid(exclude_turn_id)
        current = [row for row in rows if row.turn_id == turn_id]
        if current:
            boundary = min((_record(row)["created_at"], _message_id(row)) for row in current)
            # 历史只读到当前轮最早消息之前；不能跨过被排除的轮次推进单一游标。
            return [row for row in rows if (_record(row)["created_at"], _message_id(row)) < boundary]
        return [row for row in rows if row.turn_id != turn_id]
    return rows


def _state_for_rows(state: RollingSemanticSummary, rows: Sequence[Any]) -> tuple[RollingSemanticSummary, bool]:
    if state.covered_until_message_id and state.covered_until_created_at:
        cursor = (state.covered_until_created_at, state.covered_until_message_id)
        end = (_record(rows[-1])["created_at"], _message_id(rows[-1])) if rows else None
        if end is None or cursor > end:
            # 较早轮重试：共享摘要包含该请求之后的消息，只能从原文建临时视图。
            # 其摘要可用于本次请求，但不得回退/覆盖已经更靠后的共享游标。
            return RollingSemanticSummary(version=state.version), False
    return state, True


def prepare_semantic_context(
    db: Session, *, user_id: str, session_id: str,
    exclude_turn_id: Optional[str] = None, current_query: str = "",
    allow_summarization: bool = True,
) -> PreparedSemanticContext:
    """读摘要＋全部未覆盖原文，在下次模型调用前判断阈值并持久化。

    调用方需使用已提交消息的请求/读取事务；只在新摘要成功后提交派生状态。
    当前 query 参与触发和近期消息数计算，但不重复返回给历史消费者。
    """
    try:
        service = get_semantic_context_service()
    except Exception as exc:
        raise SemanticContextError(f"会话摘要配置无效: {exc}") from exc
    session = db.query(ChatSession).filter(
        ChatSession.id == _as_uuid(session_id), ChatSession.user_id == _as_uuid(user_id),
    ).first()
    if not session:
        raise LookupError("会话不存在或不属于当前用户")
    rows = _session_rows(db, session.id, exclude_turn_id)
    state, persist_allowed = _state_for_rows(
        _rolling_summary(getattr(session, "semantic_context_summary", None)), rows,
    )
    working = _working_messages(state, rows)
    before_tokens = service.token_counter.count_messages(working) if working else 0
    request_messages = [*working, *([{"role": "user", "content": current_query, "_current": True}] if current_query else [])]
    generated, covered = False, []
    metrics = dict(state.metrics)

    if allow_summarization and service.should_summarize(request_messages):
        prefix, _ = service.partition(request_messages)
        if prefix:
            if db.new or db.dirty or db.deleted:
                raise SemanticContextError("摘要准备事务包含未提交变更，请先提交消息事务")
            # 同一会话串行化摘要，锁后重新读取，避免并发请求二次覆盖旧版本。
            session = db.query(ChatSession).filter(
                ChatSession.id == _as_uuid(session_id), ChatSession.user_id == _as_uuid(user_id),
            ).populate_existing().with_for_update().first()
            if not session:
                raise LookupError("会话不存在或不属于当前用户")
            rows = _session_rows(db, session.id, exclude_turn_id)
            state, persist_allowed = _state_for_rows(
                _rolling_summary(getattr(session, "semantic_context_summary", None)), rows,
            )
            working = _working_messages(state, rows)
            before_tokens = service.token_counter.count_messages(working) if working else 0
            request_messages = [*working, *([{"role": "user", "content": current_query, "_current": True}] if current_query else [])]
            prefix, tail = service.partition(request_messages)
            covered = [item for item in prefix if item.get("id")]
            if prefix and service.should_summarize(request_messages):
                try:
                    last = covered[-1] if covered else {
                        "id": state.covered_until_message_id, "created_at": state.covered_until_created_at,
                    }
                    if not last["id"] or not last["created_at"]:
                        raise ValueError("摘要覆盖游标缺失，不能提交新的压缩状态")
                    text, metrics = service.summarize(prefix)
                    metrics.update(
                        input_tokens=service.token_counter.count_messages(request_messages),
                        kept_messages=len(tail), covered_message_ids=[item["id"] for item in covered],
                        context_window=service.config.context_window, trigger_tokens=service.config.trigger_tokens,
                        persisted=persist_allowed,
                    )
                    state = RollingSemanticSummary(
                        version=int(getattr(session, "semantic_context_version", 0) or 0) + int(persist_allowed),
                        model=service.config.model, summary=text,
                        covered_until_message_id=last["id"], covered_until_created_at=last["created_at"],
                        updated_at=utc_now().isoformat(timespec="seconds"),
                        config_fingerprint=service.config.fingerprint, metrics=metrics,
                    )
                    working = [summary_message(text), *[item for item in tail if not item.get("_current")]]
                    metrics["after_tokens"] = service.token_counter.count_messages(
                        [*working, *([{"role": "user", "content": current_query}] if current_query else [])]
                    )
                    state.metrics = metrics
                    if persist_allowed:
                        session.semantic_context_summary = state.model_dump(mode="json")
                        session.semantic_context_version = state.version
                        session.semantic_context_last_message_id = _as_uuid(last["id"])
                        session.semantic_context_model = state.model
                        session.semantic_context_status = "ready"
                        session.semantic_context_last_error = None
                        session.semantic_context_updated_at = utc_now()
                        db.flush()
                    db.commit()
                    generated = True
                    logger.info("会话滚动摘要 session=%s metrics=%s", session_id, json.dumps(metrics, ensure_ascii=False))
                except Exception as exc:
                    db.rollback()
                    raise SemanticContextError(f"会话摘要失败，原始消息和覆盖游标未更新: {exc}") from exc
            else:
                metrics = dict(state.metrics)
                db.commit()  # 释放重新检查后已不需要摘要的行锁。

    messages = [{"role": item["role"], "content": item["content"]} for item in working]
    after_tokens = service.token_counter.count_messages(messages) if messages else 0
    metrics.update(
        triggered=generated, history_tokens=after_tokens,
        request_tokens=service.token_counter.count_messages(
            [*messages, *([{"role": "user", "content": current_query}] if current_query else [])]
        ), context_window=service.config.context_window,
        trigger_tokens=service.config.trigger_tokens, keep_messages=service.config.keep_messages,
        trigger_context_window=service.config.trigger_context_window,
        trigger_fraction=service.config.trigger_fraction, config_fingerprint=service.config.fingerprint,
        token_count_scope="会话工作消息及本轮query；不含业务模板、知识库资料和报告候选",
        request_scoped_view=not persist_allowed,
    )
    last_id = _message_id(rows[-1]) if rows else None
    return PreparedSemanticContext(
        messages=messages, original_tokens=before_tokens, compacted_tokens=after_tokens,
        summarized_messages=len(covered) if generated else 0,
        semantic_context_version=state.version, last_message_id=last_id,
        metrics=metrics, config_fingerprint=service.config.fingerprint,
    )


def enqueue_semantic_context_if_due(db, user_id, session_id, *, force=False):
    """滚动摘要改为模型调用前执行；不再创建回应后的异步摘要任务。"""
    return None


def execute_semantic_context_job(db: Session, job: MemoryJob) -> ChatSession:
    """旧 Outbox 任务安全退役，禁止覆盖新策略摘要或推进覆盖游标。"""
    session = db.query(ChatSession).filter(
        ChatSession.id == job.session_id, ChatSession.user_id == job.user_id,
    ).first()
    if not session:
        raise RuntimeError("语义摘要任务的会话与用户不匹配")
    logger.info("忽略旧会话摘要任务 %s；摘要现由模型调用前准备", job.id)
    return session
