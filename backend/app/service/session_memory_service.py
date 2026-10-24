# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""Redis 会话工作记忆。

PostgreSQL ``chat_messages`` 始终是唯一事实来源；Redis 只保存可重建的
当前会话上下文。摘要模式缓存完整的“摘要＋未覆盖原文”，不额外裁剪。
缓存过期、被淘汰或 Redis 不可用时，
调用方会直接从 PostgreSQL 重建，不能因为缓存故障阻断聊天或研究请求。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from redis import Redis
from redis.exceptions import RedisError, WatchError
from sqlalchemy import or_
from sqlalchemy.orm import Query, Session

from core.redis_client import get_redis_client
from core.time_utils import utc_now
from models.chat import ChatMessage, ChatSession
from service.context_compaction import CompactionResult, compact_messages
from service.qwen_token_counter import QwenTokenCounter, get_qwen_token_counter
from service.semantic_context_service import render_semantic_context, semantic_context_enabled


logger = logging.getLogger(__name__)

# 旧版本缓存缺少摘要配置与排除轮次标识，升级后从 PostgreSQL 重建。
SCHEMA_VERSION = 4
KEY_PREFIX = "session_working_memory:v1"


@dataclass(frozen=True)
class WorkingMemoryContext:
    messages: List[Dict[str, str]]
    original_tokens: int
    compacted_tokens: int
    summarized_messages: int
    pruned_tool_results: int
    cache_hit: bool
    last_message_id: Optional[str]
    semantic_context_version: int = 0
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""


class RedisSessionMemoryService:
    """构建、缓存并读取当前会话的短期工作记忆。"""

    def __init__(
        self,
        *,
        redis_client: Optional[Redis] = None,
        token_counter: Optional[QwenTokenCounter] = None,
        enabled: Optional[bool] = None,
        ttl_seconds: Optional[int] = None,
        storage_token_limit: Optional[int] = None,
        storage_max_messages: Optional[int] = None,
    ) -> None:
        enabled_value = os.getenv("SHORT_TERM_MEMORY_ENABLED", "true").strip().lower()
        self.enabled = (
            enabled
            if enabled is not None
            else enabled_value not in {"0", "false", "no", "off"}
        )
        self.ttl_seconds = max(
            60,
            ttl_seconds
            if ttl_seconds is not None
            else int(os.getenv("SHORT_TERM_MEMORY_TTL_SECONDS", "86400")),
        )
        self.storage_token_limit = max(
            256,
            storage_token_limit
            if storage_token_limit is not None
            else int(os.getenv("SHORT_TERM_MEMORY_TOKEN_LIMIT", "5000")),
        )
        self.storage_max_messages = max(
            2,
            storage_max_messages
            if storage_max_messages is not None
            else int(os.getenv("SHORT_TERM_MEMORY_MAX_MESSAGES", "20")),
        )
        self.redis = redis_client or get_redis_client()
        self.token_counter = token_counter or get_qwen_token_counter()
        self._redis_retry_after = 0.0
        self._redis_backoff_seconds = max(
            1,
            int(os.getenv("SHORT_TERM_MEMORY_REDIS_BACKOFF_SECONDS", "30")),
        )

    @staticmethod
    def _as_uuid(value: str) -> UUID:
        return UUID(str(value))

    @staticmethod
    def _cursor(message: Optional[ChatMessage]) -> tuple[Optional[str], Optional[str]]:
        if not message:
            return None, None
        created_at = message.created_at or datetime.min
        return str(message.id), created_at.isoformat(timespec="microseconds")

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{KEY_PREFIX}:{user_id}:{session_id}"

    def _message_query(
        self,
        db: Session,
        session_id: str,
        *,
        exclude_turn_id: Optional[str] = None,
    ) -> Query:
        query = db.query(ChatMessage).filter(
            ChatMessage.session_id == self._as_uuid(session_id),
            ChatMessage.role.in_(("user", "assistant", "system")),
        )
        if exclude_turn_id:
            turn_uuid = self._as_uuid(exclude_turn_id)
            query = query.filter(or_(
                ChatMessage.turn_id.is_(None),
                ChatMessage.turn_id != turn_uuid,
            ))
        return query

    def _require_owned_session(
        self,
        db: Session,
        *,
        user_id: str,
        session_id: str,
    ) -> ChatSession:
        owned = db.query(ChatSession).filter(
            ChatSession.id == self._as_uuid(session_id),
            ChatSession.user_id == self._as_uuid(user_id),
            ChatSession.expires_at > utc_now(),
        ).first()
        if not owned:
            raise LookupError("会话不存在或不属于当前用户")
        return owned

    def _mark_redis_unavailable(self, exc: Exception) -> None:
        self._redis_retry_after = time.monotonic() + self._redis_backoff_seconds
        logger.warning("Redis 短期记忆不可用，降级为 PostgreSQL 重建: %s", exc)

    def _redis_ready(self) -> bool:
        return self.enabled and time.monotonic() >= self._redis_retry_after

    def _read_payload(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._redis_ready():
            return None
        try:
            raw = self.redis.get(key)
            if not raw:
                return None
            payload = json.loads(raw)
            if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
                return None
            messages = payload.get("messages")
            if not isinstance(messages, list):
                return None
            return payload
        except (RedisError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._mark_redis_unavailable(exc)
            return None

    @staticmethod
    def _payload_is_newer(current: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        current_cursor = (
            str(current.get("last_message_created_at") or ""),
            str(current.get("last_message_id") or ""),
            int(current.get("semantic_context_version") or 0),
        )
        candidate_cursor = (
            str(candidate.get("last_message_created_at") or ""),
            str(candidate.get("last_message_id") or ""),
            int(candidate.get("semantic_context_version") or 0),
        )
        return current_cursor > candidate_cursor

    def _write_payload(self, key: str, payload: Dict[str, Any]) -> bool:
        """通过 WATCH 防止较旧的并发重建覆盖较新的会话快照。"""
        if not self._redis_ready():
            return False
        try:
            for _ in range(3):
                with self.redis.pipeline() as pipeline:
                    try:
                        pipeline.watch(key)
                        raw = pipeline.get(key)
                        if raw:
                            current = json.loads(raw)
                            if (
                                isinstance(current, dict)
                                and current.get("schema_version") == SCHEMA_VERSION
                                and self._payload_is_newer(
                                    current, payload
                                )
                            ):
                                pipeline.unwatch()
                                return True
                        pipeline.multi()
                        pipeline.set(
                            key,
                            json.dumps(payload, ensure_ascii=False),
                            ex=self.ttl_seconds,
                        )
                        pipeline.execute()
                        return True
                    except WatchError:
                        continue
            return False
        except (RedisError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._mark_redis_unavailable(exc)
            return False

    def _build_from_postgres(
        self,
        db: Session,
        *,
        user_id: str,
        session_id: str,
        exclude_turn_id: Optional[str] = None,
        session: Optional[ChatSession] = None,
        latest_message: Optional[ChatMessage] = None,
    ) -> tuple[Dict[str, Any], CompactionResult]:
        session = session or self._require_owned_session(
            db,
            user_id=user_id,
            session_id=session_id,
        )
        query = self._message_query(
            db,
            session_id,
            exclude_turn_id=exclude_turn_id,
        )
        semantic_summary = getattr(session, "semantic_context_summary", None)
        semantic_rendered = render_semantic_context(semantic_summary)
        # 即使摘要已经覆盖旧消息，仍保留最近原文作为校验与降级依据；
        # 语义摘要负责延伸更早状态，而不是替代 PostgreSQL 原始消息。
        rows = query.order_by(
            ChatMessage.created_at.desc(),
            ChatMessage.id.desc(),
        ).limit(max(self.storage_max_messages * 6, self.storage_max_messages)).all()
        chronological = [
            {"role": row.role, "content": row.content}
            for row in reversed(rows)
        ]
        if semantic_rendered:
            chronological.insert(0, {"role": "system", "content": semantic_rendered})
        compacted = compact_messages(
            chronological,
            token_limit=self.storage_token_limit,
            max_messages=self.storage_max_messages,
            token_counter=self.token_counter,
        )
        latest = latest_message or (rows[0] if rows else None)
        last_message_id, last_message_created_at = self._cursor(latest)
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "user_id": str(user_id),
            "session_id": str(session_id),
            "messages": compacted.messages,
            "last_message_id": last_message_id,
            "last_message_created_at": last_message_created_at,
            "source_message_count": len(rows),
            "original_tokens": compacted.original_tokens,
            "compacted_tokens": compacted.compacted_tokens,
            "summarized_messages": compacted.summarized_messages,
            "pruned_tool_results": compacted.pruned_tool_results,
            "semantic_context_version": int(
                getattr(session, "semantic_context_version", 0) or 0
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return payload, compacted

    def get_context(
        self,
        db: Session,
        *,
        user_id: str,
        session_id: str,
        exclude_turn_id: Optional[str] = None,
        token_limit: Optional[int] = None,
        max_messages: Optional[int] = None,
        force_rebuild: bool = False,
        current_query: str = "",
        allow_summarization: bool = True,
    ) -> WorkingMemoryContext:
        """读取当前会话工作记忆；缓存陈旧时从 PostgreSQL 重建。"""
        session = self._require_owned_session(
            db,
            user_id=user_id,
            session_id=session_id,
        )
        query = self._message_query(
            db,
            session_id,
            exclude_turn_id=exclude_turn_id,
        )
        from service.context_policy import session_context_compression_enabled
        if not session_context_compression_enabled():
            # 压缩缓存可能已丢掉旧轮次；关闭时从原始账本全量读取且不注入旧摘要。
            rows = query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).all()
            messages = [{"role": row.role, "content": row.content} for row in rows]
            tokens = self.token_counter.count_messages(messages) if messages else 0
            last_id, _ = self._cursor(rows[-1] if rows else None)
            return WorkingMemoryContext(messages, tokens, tokens, 0, 0, False, last_id, 0)
        if semantic_context_enabled():
            # 每次在消费上下文前判定阈值。Redis 仅缓存完整结果，不能用旧窗口
            # 裁掉尚未进入摘要的消息，也不能让缓存命中跳过本轮阈值检查。
            from service.semantic_context_service import prepare_semantic_context

            prepared = prepare_semantic_context(
                db,
                user_id=str(user_id),
                session_id=str(session_id),
                exclude_turn_id=exclude_turn_id,
                current_query=current_query,
                allow_summarization=allow_summarization,
            )
            latest = query.order_by(
                ChatMessage.created_at.desc(), ChatMessage.id.desc()
            ).first()
            _, last_message_created_at = self._cursor(latest)
            identity_parts = (
                prepared.config_fingerprint,
                str(exclude_turn_id or ""),
                str(prepared.last_message_id or ""),
                str(prepared.semantic_context_version),
            )
            cache_identity = hashlib.sha256(
                json.dumps(identity_parts, ensure_ascii=False).encode()
            ).hexdigest()
            key = self._key(user_id, session_id)
            cached = None if force_rebuild else self._read_payload(key)
            cache_hit = bool(
                cached
                and str(cached.get("user_id")) == str(user_id)
                and str(cached.get("session_id")) == str(session_id)
                and cached.get("cache_identity") == cache_identity
                and cached.get("messages") == prepared.messages
            )
            if not cache_hit:
                self._write_payload(key, {
                    "schema_version": SCHEMA_VERSION,
                    "user_id": str(user_id),
                    "session_id": str(session_id),
                    "messages": prepared.messages,
                    "last_message_id": prepared.last_message_id,
                    "last_message_created_at": last_message_created_at,
                    "original_tokens": prepared.original_tokens,
                    "compacted_tokens": prepared.compacted_tokens,
                    "summarized_messages": prepared.summarized_messages,
                    "pruned_tool_results": 0,
                    "semantic_context_version": prepared.semantic_context_version,
                    "summary_metrics": prepared.metrics,
                    "config_fingerprint": prepared.config_fingerprint,
                    "exclude_turn_id": str(exclude_turn_id or ""),
                    "cache_identity": cache_identity,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
            return WorkingMemoryContext(
                messages=prepared.messages,
                original_tokens=prepared.original_tokens,
                compacted_tokens=prepared.compacted_tokens,
                summarized_messages=prepared.summarized_messages,
                pruned_tool_results=0,
                cache_hit=cache_hit,
                last_message_id=prepared.last_message_id,
                semantic_context_version=prepared.semantic_context_version,
                summary_metrics=prepared.metrics,
                config_fingerprint=prepared.config_fingerprint,
            )
        latest = query.order_by(
            ChatMessage.created_at.desc(), ChatMessage.id.desc()
        ).first()
        latest_id, _ = self._cursor(latest)
        semantic_context_version = int(
            getattr(session, "semantic_context_version", 0) or 0
        )
        key = self._key(user_id, session_id)

        payload = None if force_rebuild else self._read_payload(key)
        cache_hit = bool(
            payload
            and str(payload.get("user_id")) == str(user_id)
            and str(payload.get("session_id")) == str(session_id)
            and payload.get("last_message_id") == latest_id
            and not payload.get("cache_identity")  # 不能复用滚动摘要模式的完整缓存。
            and int(payload.get("semantic_context_version") or 0)
            == semantic_context_version
        )
        if not cache_hit:
            payload, _ = self._build_from_postgres(
                db,
                user_id=user_id,
                session_id=session_id,
                exclude_turn_id=exclude_turn_id,
                session=session,
                latest_message=latest,
            )
            self._write_payload(key, payload)

        requested_limit = token_limit or self.storage_token_limit
        requested_max = max_messages or self.storage_max_messages
        fitted = compact_messages(
            list(payload.get("messages") or []),
            token_limit=requested_limit,
            max_messages=requested_max,
            token_counter=self.token_counter,
        )
        return WorkingMemoryContext(
            messages=fitted.messages,
            original_tokens=int(payload.get("original_tokens") or fitted.original_tokens),
            compacted_tokens=fitted.compacted_tokens,
            summarized_messages=int(
                payload.get("summarized_messages") or fitted.summarized_messages
            ),
            pruned_tool_results=int(
                payload.get("pruned_tool_results") or fitted.pruned_tool_results
            ),
            cache_hit=cache_hit,
            last_message_id=payload.get("last_message_id"),
            semantic_context_version=int(
                payload.get("semantic_context_version") or 0
            ),
        )

    def refresh(self, db: Session, *, user_id: str, session_id: str) -> None:
        """在一轮助手消息落库后，用最新 PostgreSQL 快照刷新 Redis。"""
        if not self.enabled:
            return
        try:
            self.get_context(
                db,
                user_id=user_id,
                session_id=session_id,
                force_rebuild=True,
                allow_summarization=False,
            )
        except Exception as exc:
            # 短期记忆是可重建缓存，不能影响消息持久化主链路。
            logger.warning("刷新 Redis 会话工作记忆失败，已忽略: %s", exc)

    def invalidate(self, *, user_id: str, session_id: str) -> None:
        if not self._redis_ready():
            return
        try:
            self.redis.delete(self._key(user_id, session_id))
        except RedisError as exc:
            self._mark_redis_unavailable(exc)

    def ping(self) -> bool:
        if not self._redis_ready():
            return False
        try:
            return bool(self.redis.ping())
        except RedisError as exc:
            self._mark_redis_unavailable(exc)
            return False


_session_memory_service: Optional[RedisSessionMemoryService] = None


def get_session_memory_service() -> RedisSessionMemoryService:
    global _session_memory_service
    if _session_memory_service is None:
        _session_memory_service = RedisSessionMemoryService()
    return _session_memory_service
