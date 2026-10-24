# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""长期记忆：统一 Provider、Native 实现与 PostgreSQL Outbox。"""

import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy import and_, exists, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.database import SessionLocal
from core.time_utils import as_utc, utc_now
from models.chat import ChatMessage, ChatSession, LongTermMemory, MemoryJob
from service.claude_client import messages_completion as claude_messages_completion
from service.embedding_service import generate_embedding
from service.llm_client import create_openai_client
from service.memory_contracts import (
    MemoryHit,
    MemoryPatch,
    MemoryProvider,
    MemoryScope,
    MemorySearchOptions,
    MemoryTurn,
    MemoryWriteResult,
)
from service.milvus_service import MilvusService, get_milvus_service
from service.qwen_token_counter import get_qwen_token_counter
from service.memory_trigger_policy import priority_memory_trigger

load_dotenv()
logger = logging.getLogger(__name__)

MEMORY_TOKEN_THRESHOLD = int(os.getenv("MEMORY_TOKEN_THRESHOLD", "2000"))
MEMORY_IDLE_SECONDS = max(0, int(os.getenv("MEMORY_IDLE_SECONDS", "300")))
MEMORY_SUMMARY_MAX_CHARS = int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "30000"))
MEMORY_FACT_BATCH_MAX_CHARS = int(
    os.getenv("MEMORY_FACT_BATCH_MAX_CHARS", "6000")
)
MEMORY_FACT_BATCH_MAX_FACTS = int(
    os.getenv("MEMORY_FACT_BATCH_MAX_FACTS", "12")
)
MEMORY_MAX_ATOMIC_FACTS = int(os.getenv("MEMORY_MAX_ATOMIC_FACTS", "40"))
MEMORY_COLLECTION_NAME = "long_term_memories"
MEMORY_PROVIDER_NAME = os.getenv("MEMORY_PROVIDER", "native").lower()
ACTIVE_JOB_STATUSES = ("pending", "running", "retry")
RETRY_DELAYS_SECONDS = (60, 300, 900, 3600, 21600)


class MemoryInfrastructureError(RuntimeError):
    """上游基础设施不可用；评测任务应立即停止而不是继续污染结果。"""

    abort_run = True


class MemoryProviderQuotaError(MemoryInfrastructureError):
    """模型 Provider 的硬额度或账户容量已经耗尽。"""


def _is_hard_provider_error(exc: Exception) -> bool:
    normalized = str(exc).lower()
    return any(token in normalized for token in (
        "daily_limit_exceeded",
        "usage_limit_exceeded",
        "daily usage limit exceeded",
        "billing_error",
        "no available accounts",
    ))


def _is_transient_provider_error(exc: Exception) -> bool:
    normalized = str(exc).lower()
    return any(token in normalized for token in (
        "http 429",
        "concurrency limit exceeded",
        "rate_limit",
        "timed out",
        "timeout",
        "server disconnected",
        "connection reset",
        "connection refused",
        "connection error",
        "remoteprotocolerror",
        "unexpected_eof",
        "incomplete chunked read",
    ))


def _is_content_moderation_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return (
        "data_inspection_failed" in error_text
        or "inappropriate content" in error_text
    )


def _parse_json_object(result_text: str) -> Dict[str, Any]:
    normalized = result_text.strip()
    if normalized.startswith("```json"):
        normalized = normalized[7:]
    elif normalized.startswith("```"):
        normalized = normalized[3:]
    if normalized.endswith("```"):
        normalized = normalized[:-3]

    result = json.loads(normalized.strip())
    if not isinstance(result, dict):
        raise ValueError("模型输出必须是 JSON object")
    return result


def _parse_summary_json(result_text: str) -> Dict[str, Any]:
    result = _parse_json_object(result_text)
    if not isinstance(result.get("summary"), str) or not result["summary"].strip():
        raise ValueError("记忆总结缺少 summary")
    result.setdefault("key_insights", [])
    result.setdefault("user_preferences", {})
    result.setdefault("topics", [])
    result.setdefault("atomic_facts", [])
    for field in ("key_insights", "topics", "atomic_facts"):
        if not isinstance(result[field], list):
            result[field] = []
    if not isinstance(result["user_preferences"], dict):
        result["user_preferences"] = {}
    return result


def _is_transient_claude_error(exc: Exception) -> bool:
    return not _is_hard_provider_error(exc) and _is_transient_provider_error(exc)


def _fact_extraction_batches(
    messages: Sequence[Any],
    *,
    max_chars: int = MEMORY_FACT_BATCH_MAX_CHARS,
) -> List[str]:
    """将对话切成可独立抽取的小批次。

    超长单条消息也会分块，但保留原 message_id 和时间，便于
    后续用原始完整消息验证 exact_quote。
    """
    if max_chars < 1000:
        raise ValueError("MEMORY_FACT_BATCH_MAX_CHARS 不能小于 1000")

    rendered_chunks: List[str] = []
    content_limit = max(500, max_chars - 300)
    for message in messages:
        content = str(getattr(message, "content", ""))
        parts = [
            content[index:index + content_limit]
            for index in range(0, len(content), content_limit)
        ] or [""]
        for part_index, part in enumerate(parts, 1):
            chunk_suffix = (
                f"; chunk={part_index}/{len(parts)}" if len(parts) > 1 else ""
            )
            rendered_chunks.append(
                f"[message_id={_message_id(message)}; "
                f"time={_message_time(message)}{chunk_suffix}] "
                f"{_message_role(message)}: {part}"
            )

    batches: List[str] = []
    current: List[str] = []
    current_chars = 0
    for chunk in rendered_chunks:
        extra = len(chunk) + (1 if current else 0)
        if current and current_chars + extra > max_chars:
            batches.append("\n".join(current))
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += len(chunk) + (1 if len(current) > 1 else 0)
    if current:
        batches.append("\n".join(current))
    return batches


def _as_uuid(value: str) -> UUID:
    return UUID(str(value))


def _message_id(message: Any) -> str:
    return str(getattr(message, "message_id", None) or getattr(message, "id", ""))


def _message_time(message: Any) -> str:
    value = getattr(message, "created_at", None)
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value or "")


def _message_role(message: Any) -> str:
    return {
        "user": "用户",
        "assistant": "助手",
        "system": "系统",
    }.get(str(getattr(message, "role", "")), "未知")


def _normalize_atomic_facts(
    facts: Any,
    messages: Sequence[Any],
    *,
    max_facts: int = 20,
) -> List[Dict[str, Any]]:
    """只保留可回溯到真实消息的原子事实，并附加可信来源时间。"""
    if not isinstance(facts, list):
        return []

    source_by_id = {
        _message_id(message): message
        for message in messages
        if _message_id(message)
    }
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in facts:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()[:2000]
        if not fact or fact in seen:
            continue

        requested_ids = item.get("source_message_ids") or []
        if not isinstance(requested_ids, list):
            requested_ids = [requested_ids]
        source_ids = [
            str(value) for value in requested_ids if str(value) in source_by_id
        ][:10]

        exact_quote = str(item.get("exact_quote") or "").strip()[:500]
        quote_source_ids = [
            source_id
            for source_id, message in source_by_id.items()
            if exact_quote and exact_quote in str(getattr(message, "content", ""))
        ]
        if exact_quote and not quote_source_ids:
            # 不把无法在原文中定位的“引用”继续作为可信证据传播。
            exact_quote = ""
        for source_id in quote_source_ids:
            if source_id not in source_ids:
                source_ids.append(source_id)
            if len(source_ids) >= 10:
                break
        if not source_ids:
            # 原子事实必须可审计；没有合法来源引用的抽取结果直接丢弃。
            continue

        source_times = [
            _message_time(source_by_id[source_id])
            for source_id in source_ids
            if _message_time(source_by_id[source_id])
        ]
        value = item.get("value")
        normalized.append({
            "fact": fact,
            "subject": str(item.get("subject") or "").strip(),
            "predicate": str(item.get("predicate") or "").strip(),
            "value": value if isinstance(value, (str, int, float, bool)) else None,
            "occurred_at": str(item.get("occurred_at") or "").strip() or None,
            "valid_from": str(item.get("valid_from") or "").strip() or None,
            "source_message_ids": source_ids,
            "source_times": source_times,
            "exact_quote": exact_quote or None,
        })
        seen.add(fact)
        if len(normalized) >= max_facts:
            break
    return normalized


def _query_terms(text: str) -> set[str]:
    normalized = text.lower()
    words = set(re.findall(r"[a-z0-9][a-z0-9_./:-]+", normalized))
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", normalized)
    bigrams = {
        chunk[index:index + 2]
        for chunk in chinese_chunks
        for index in range(max(0, len(chunk) - 1))
    }
    return words | bigrams


def _select_relevant_raw_messages(
    query: str,
    hit_contents: Sequence[str],
    messages: Sequence[Any],
    *,
    preferred_message_ids: Sequence[str] = (),
    max_messages: int = 2,
) -> List[Any]:
    """从命中记忆对应的原始批次中选出少量可核验消息。"""
    preferred = set(preferred_message_ids)
    terms = _query_terms("\n".join([query, *hit_contents]))
    ranked = []
    for index, message in enumerate(messages):
        content = str(getattr(message, "content", ""))
        overlap = len(terms & _query_terms(content))
        preferred_score = 1 if _message_id(message) in preferred else 0
        ranked.append((preferred_score, overlap, -index, message))
    ranked.sort(key=lambda item: item[:3], reverse=True)
    selected = [item[3] for item in ranked if item[0] or item[1]][:max_messages]
    return sorted(
        selected,
        key=lambda message: (
            as_utc(getattr(message, "created_at", None))
            if getattr(message, "created_at", None) is not None
            else datetime.min.replace(tzinfo=timezone.utc),
            _message_id(message),
        ),
    )


def _select_diverse_hits(hits: Sequence[MemoryHit], limit: int) -> List[MemoryHit]:
    """优先覆盖不同会话，再用剩余名额补充同会话的相关事实。"""
    if limit <= 0:
        return []
    unique: List[MemoryHit] = []
    seen_content = set()
    for hit in hits:
        normalized = hit.content.strip()
        if not normalized or normalized in seen_content:
            continue
        seen_content.add(normalized)
        unique.append(hit)

    selected: List[MemoryHit] = []
    selected_ids = set()
    seen_sessions = set()
    for hit in unique:
        session_key = hit.session_id or hit.memory_id
        if session_key in seen_sessions:
            continue
        selected.append(hit)
        selected_ids.add((hit.memory_id, hit.fragment_id))
        seen_sessions.add(session_key)
        if len(selected) >= limit:
            return selected
    for hit in unique:
        if (hit.memory_id, hit.fragment_id) in selected_ids:
            continue
        selected.append(hit)
        if len(selected) >= limit:
            break
    return selected


class MemoryService:
    """Native Provider 的 PostgreSQL + Milvus 底层实现。"""

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = os.getenv(
            "MEMORY_SUMMARY_MODEL",
            os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
        )
        self._client = None
        self._milvus: Optional[MilvusService] = None
        self.token_counter = get_qwen_token_counter()

    @property
    def client(self):
        """LLM 客户端只在 Worker 真正总结记忆时初始化。"""
        if self._client is None:
            self._client = create_openai_client(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @property
    def milvus(self) -> MilvusService:
        if self._milvus is None:
            self._milvus = get_milvus_service()
            self._ensure_memory_collection()
        return self._milvus

    def _ensure_memory_collection(self) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if utility.has_collection(MEMORY_COLLECTION_NAME):
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ]
        collection = Collection(
            name=MEMORY_COLLECTION_NAME,
            schema=CollectionSchema(fields=fields, description="Long-term memories"),
        )
        collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        collection.load()
        logger.info("记忆集合 %s 创建成功", MEMORY_COLLECTION_NAME)

    def estimate_tokens(self, text: str) -> int:
        return self.token_counter.count_text(text) if text else 0

    def should_compress(self, messages: Sequence[Any]) -> bool:
        rendered = [
            {
                "role": _message_role(message),
                "content": str(message.content),
            }
            for message in messages
        ]
        return self.token_counter.count_messages(rendered) > MEMORY_TOKEN_THRESHOLD

    def summarize_conversation(self, messages: Sequence[Any]) -> Dict[str, Any]:
        conversation_text = "\n".join(
            f"[message_id={_message_id(message)}; time={_message_time(message)}] "
            f"{_message_role(message)}: {message.content}"
            for message in messages
        )
        # 保留最新内容；结论和偏好变更通常出现在对话末尾。
        if len(conversation_text) > MEMORY_SUMMARY_MAX_CHARS:
            conversation_text = (
                "...(单条超长消息的较早内容已截断)\n"
                + conversation_text[-MEMORY_SUMMARY_MAX_CHARS:]
            )

        summary_prompt = f"""请分析以下对话，并按JSON格式输出总结信息。
这一阶段不要提取原子事实，只生成简洁摘要、洞察、偏好和主题：

对话内容：
{conversation_text}

请只输出以下格式的JSON：
{{
    "summary": "用2-3句话总结这段对话的主要内容和结论",
    "key_insights": ["从对话中提取的3-5个关键信息或知识点"],
    "user_preferences": {{
        "interests": ["用户感兴趣的领域"],
        "communication_style": "用户偏好的沟通风格",
        "focus_areas": ["用户关注的重点领域"]
    }},
    "topics": ["对话涉及的主题标签"]
}}"""

        fallback_reason = None
        providers_used = set()

        def summarize_with_claude(request_prompt: str, max_tokens: int) -> str:
            claude_api_key = os.getenv("CLAUDE_RELAY_API_KEY")
            claude_base_url = os.getenv("CLAUDE_RELAY_BASE_URL")
            if not claude_api_key or not claude_base_url:
                raise RuntimeError(
                    "Claude 记忆总结需要 "
                    "CLAUDE_RELAY_API_KEY 和 CLAUDE_RELAY_BASE_URL"
                )
            claude_model = os.getenv(
                "MEMORY_SUMMARY_MODEL",
                os.getenv(
                    "MEMORY_FALLBACK_MODEL",
                    os.getenv("CLAUDE_RELAY_MODEL", "claude-opus-5"),
                ),
            )
            attempts = max(1, int(os.getenv("MEMORY_CLAUDE_REQUEST_ATTEMPTS", "5")))
            for attempt in range(1, attempts + 1):
                try:
                    return claude_messages_completion(
                        claude_api_key,
                        claude_base_url,
                        claude_model,
                        request_prompt,
                        max_tokens,
                        system_prompt="你是专业的对话分析助手，只输出合法 JSON。",
                    )
                except Exception as exc:
                    if _is_hard_provider_error(exc):
                        raise MemoryProviderQuotaError(
                            f"Claude 记忆抽取额度不可用：{exc}"
                        ) from exc
                    if attempt == attempts or not _is_transient_claude_error(exc):
                        raise
                    delay = min(2 ** attempt, 20)
                    logger.warning(
                        "Claude 记忆抽取第 %d/%d 次失败，%d 秒后重试：%s",
                        attempt,
                        attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            raise AssertionError("unreachable")

        configured_provider = os.getenv(
            "MEMORY_SUMMARY_PROVIDER", "dashscope"
        ).strip().lower()
        if configured_provider not in {"claude", "claude-relay", "dashscope"}:
            raise ValueError(
                "MEMORY_SUMMARY_PROVIDER 仅支持 dashscope 或 claude-relay，"
                f"当前为 {configured_provider!r}"
            )

        def complete(request_prompt: str, *, max_tokens: int) -> str:
            nonlocal fallback_reason
            if configured_provider in {"claude", "claude-relay"}:
                providers_used.add("claude-relay")
                return summarize_with_claude(request_prompt, max_tokens)
            attempts = max(1, int(os.getenv("MEMORY_DASHSCOPE_REQUEST_ATTEMPTS", "3")))
            for attempt in range(1, attempts + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "你是专业的对话分析助手，只输出合法 JSON。",
                            },
                            {"role": "user", "content": request_prompt},
                        ],
                        temperature=0,
                        max_tokens=max_tokens,
                    )
                    providers_used.add("dashscope")
                    return (response.choices[0].message.content or "").strip()
                except Exception as exc:
                    if _is_content_moderation_error(exc):
                        fallback_provider = os.getenv(
                            "MEMORY_CONTENT_MODERATION_FALLBACK", "claude-relay"
                        ).strip().lower()
                        if fallback_provider in {"", "none", "disabled"}:
                            raise
                        if fallback_provider not in {"claude", "claude-relay"}:
                            raise ValueError(
                                "MEMORY_CONTENT_MODERATION_FALLBACK 仅支持 "
                                "none 或 claude-relay"
                            ) from exc
                        logger.warning("Qwen 内容审核拒绝，改用 Claude 生成记忆摘要")
                        providers_used.add("claude-relay")
                        fallback_reason = "dashscope_content_moderation"
                        return summarize_with_claude(request_prompt, max_tokens)
                    if _is_hard_provider_error(exc):
                        raise MemoryProviderQuotaError(
                            f"DashScope 记忆抽取额度不可用：{exc}"
                        ) from exc
                    if attempt == attempts or not _is_transient_provider_error(exc):
                        if _is_transient_provider_error(exc):
                            raise MemoryInfrastructureError(
                                f"DashScope 记忆抽取连接不可用：{exc}"
                            ) from exc
                        raise
                    delay = min(2 ** attempt, 20)
                    logger.warning(
                        "DashScope 记忆抽取第 %d/%d 次失败，%d 秒后重试：%s",
                        attempt,
                        attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            raise AssertionError("unreachable")

        def complete_json(
            request_prompt: str,
            *,
            max_tokens: int,
            parser: Any,
        ) -> Dict[str, Any]:
            attempts = max(1, int(os.getenv("MEMORY_JSON_REQUEST_ATTEMPTS", "2")))
            parse_prompt = request_prompt
            for attempt in range(1, attempts + 1):
                result_text = complete(parse_prompt, max_tokens=max_tokens)
                try:
                    return parser(result_text)
                except (json.JSONDecodeError, ValueError) as exc:
                    if attempt == attempts:
                        raise
                    logger.warning(
                        "记忆抽取 JSON 第 %d/%d 次解析失败，缩短输出后重试：%s",
                        attempt,
                        attempts,
                        exc,
                    )
                    parse_prompt = (
                        request_prompt
                        + "\n\n上一次响应不是完整合法 JSON。请减少条目和文字长度，"
                        "只返回完整 JSON object，不要使用 Markdown。"
                    )
            raise AssertionError("unreachable")

        summary_tokens = int(os.getenv("MEMORY_SUMMARY_MAX_TOKENS", "2000"))
        result = complete_json(
            summary_prompt,
            max_tokens=summary_tokens,
            parser=_parse_summary_json,
        )

        fact_tokens = int(os.getenv("MEMORY_FACT_MAX_TOKENS", "3000"))
        extracted_facts: List[Dict[str, Any]] = []
        fact_batches = _fact_extraction_batches(messages)
        for batch_number, batch_text in enumerate(fact_batches, 1):
            fact_prompt = f"""从下面第 {batch_number}/{len(fact_batches)} 个对话片段中提取原子事实。
只提取原文明示、可由 exact_quote 直接验证的事实；不要推测时间，不要改写 URL、数字、名称或标题。
每批最多输出 {MEMORY_FACT_BATCH_MAX_FACTS} 条；没有事实时返回空数组。

对话片段：
{batch_text}

只输出合法 JSON object：
{{
    "atomic_facts": [
        {{
            "fact": "可独立理解的单一事实",
            "subject": "事实主体",
            "predicate": "属性或关系",
            "value": "原文中的精确值",
            "occurred_at": "仅当原文明示事件时间时填写，否则为null",
            "valid_from": "仅当原文明示状态生效时间时填写，否则为null",
            "source_message_ids": ["支持该事实的真实message_id"],
            "exact_quote": "能够直接支持事实的短原文片段"
        }}
    ]
}}"""
            fact_result = complete_json(
                fact_prompt,
                max_tokens=fact_tokens,
                parser=_parse_json_object,
            )
            batch_facts = fact_result.get("atomic_facts") or []
            if isinstance(batch_facts, list):
                extracted_facts.extend(
                    item for item in batch_facts if isinstance(item, dict)
                )

        result["atomic_facts"] = _normalize_atomic_facts(
            extracted_facts,
            messages,
            max_facts=MEMORY_MAX_ATOMIC_FACTS,
        )
        result["summary_provider"] = "+".join(sorted(providers_used))
        result["fact_batch_count"] = len(fact_batches)
        if fallback_reason:
            result["fallback_reason"] = fallback_reason
        return result

    def create_memory(
        self,
        db: Session,
        user_id: str,
        session_id: str,
        messages: Sequence[Any],
        *,
        source_start_message_id: Optional[str] = None,
        source_end_message_id: Optional[str] = None,
    ) -> LongTermMemory:
        if not messages:
            raise ValueError("不能从空消息列表创建记忆")

        summary_data = self.summarize_conversation(messages)
        memory = self.prepare_memory_record(
            db,
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            summary_data=summary_data,
            source_start_message_id=source_start_message_id,
            source_end_message_id=source_end_message_id,
        )
        if memory.index_status == "ready" and not memory.deleted_at:
            return memory

        try:
            milvus_ids = self._store_memory_vectors(
                memory_id=str(memory.id),
                user_id=user_id,
                session_id=session_id,
                summary_data=summary_data,
            )
            if not milvus_ids:
                raise RuntimeError("没有生成可索引的记忆向量")
            memory.milvus_ids = milvus_ids
            memory.index_status = "ready"
            memory.last_error = None
            db.commit()
            db.refresh(memory)
            logger.info("创建长期记忆成功: %s", memory.id)
            return memory
        except Exception as exc:
            db.rollback()
            memory = db.query(LongTermMemory).filter(LongTermMemory.id == memory.id).first()
            if memory:
                memory.index_status = "failed"
                memory.last_error = str(exc)[:4000]
                db.commit()
            raise

    def prepare_memory_record(
        self,
        db: Session,
        user_id: str,
        session_id: str,
        messages: Sequence[Any],
        summary_data: Dict[str, Any],
        *,
        source_start_message_id: Optional[str] = None,
        source_end_message_id: Optional[str] = None,
    ) -> LongTermMemory:
        """保存已经抽取好的记忆，但把向量写入留给独立批处理阶段。"""
        if not messages:
            raise ValueError("不能从空消息列表创建记忆")
        if not isinstance(summary_data.get("summary"), str) or not summary_data["summary"].strip():
            raise ValueError("记忆总结缺少 summary")

        end_uuid = _as_uuid(source_end_message_id) if source_end_message_id else None
        existing = None
        if end_uuid:
            existing = db.query(LongTermMemory).filter(
                LongTermMemory.session_id == _as_uuid(session_id),
                LongTermMemory.source_end_message_id == end_uuid,
            ).first()
            if existing and existing.index_status == "ready" and not existing.deleted_at:
                return existing

        total_tokens = self.token_counter.count_messages([
            {
                "role": _message_role(message),
                "content": str(message.content),
            }
            for message in messages
        ])

        memory = existing or LongTermMemory(
            user_id=_as_uuid(user_id),
            session_id=_as_uuid(session_id),
            source_start_message_id=(
                _as_uuid(source_start_message_id) if source_start_message_id else None
            ),
            source_end_message_id=end_uuid,
        )
        memory.summary = summary_data["summary"]
        memory.key_insights = summary_data
        memory.token_count = total_tokens
        memory.provider = "native"
        memory.index_status = "indexing"
        memory.last_error = None
        memory.deleted_at = None
        if not existing:
            db.add(memory)
        db.commit()
        db.refresh(memory)
        return memory

    def _store_memory_vectors(
        self,
        memory_id: str,
        user_id: str,
        session_id: str,
        summary_data: Dict[str, Any],
    ) -> List[str]:
        documents = self.build_memory_vector_documents(
            memory_id=memory_id,
            user_id=user_id,
            session_id=session_id,
            summary_data=summary_data,
        )
        if not documents:
            return []

        # DashScope 支持最多 10 条一批；generate_embedding 会自动分批。
        # 在这里一次传入本会话的全部碎片，避免为每条原子事实重复建立 HTTP/TLS 连接。
        embeddings = generate_embedding([document["content"] for document in documents])
        if not isinstance(embeddings, list) or len(embeddings) != len(documents):
            raise MemoryInfrastructureError(
                f"批量生成 embedding 数量不匹配：期望 {len(documents)}，"
                f"实际 {len(embeddings) if isinstance(embeddings, list) else 0}"
            )
        for document, vector in zip(documents, embeddings):
            if not isinstance(vector, list) or not vector:
                raise MemoryInfrastructureError(
                    f"生成 {document['memory_type']} embedding 失败"
                )
            document["vector"] = vector
        return self.store_preembedded_memory_vectors(memory_id, documents)

    def build_memory_vector_documents(
        self,
        memory_id: str,
        user_id: str,
        session_id: str,
        summary_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """把结构化记忆转换成待向量化文档，不发出网络请求。"""
        documents: List[Dict[str, Any]] = []

        def append_document(doc_id: str, memory_type: str, content: str, metadata: dict) -> None:
            serialized_metadata = json.dumps(
                {"memory_id": memory_id, **metadata},
                ensure_ascii=False,
            )
            if len(serialized_metadata.encode("utf-8")) > 8192:
                # 截断 JSON 会导致检索端无法解析 memory_id；退化为最小合法元数据。
                serialized_metadata = json.dumps({"memory_id": memory_id})
            documents.append({
                "id": doc_id,
                "user_id": user_id,
                "session_id": session_id,
                "memory_type": memory_type,
                "content": content,
                "metadata": serialized_metadata,
            })

        summary = summary_data.get("summary", "").strip()
        if summary:
            append_document(f"{memory_id}_summary", "summary", summary, {})

        for index, insight in enumerate(summary_data.get("key_insights", [])):
            if isinstance(insight, str) and insight.strip():
                append_document(
                    f"{memory_id}_insight_{index}",
                    "insight",
                    insight.strip(),
                    {"index": index},
                )

        for index, fact in enumerate(summary_data.get("atomic_facts", [])):
            if not isinstance(fact, dict):
                continue
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_text:
                continue
            fact_metadata = {
                key: fact.get(key)
                for key in (
                    "subject",
                    "predicate",
                    "value",
                    "occurred_at",
                    "valid_from",
                    "source_message_ids",
                    "source_times",
                    "exact_quote",
                )
                if fact.get(key) not in (None, "", [])
            }
            append_document(
                f"{memory_id}_atomic_fact_{index}",
                "atomic_fact",
                fact_text,
                {"index": index, **fact_metadata},
            )

        topics = [topic for topic in summary_data.get("topics", []) if isinstance(topic, str)]
        if topics:
            topics_text = "用户关注的主题: " + ", ".join(topics)
            append_document(
                f"{memory_id}_topics",
                "topics",
                topics_text,
                {"topics": topics},
            )

        preferences = summary_data.get("user_preferences")
        if isinstance(preferences, dict):
            preference_parts = []
            interests = preferences.get("interests")
            if isinstance(interests, list):
                values = [value for value in interests if isinstance(value, str) and value]
                if values:
                    preference_parts.append("兴趣: " + ", ".join(values))
            communication_style = preferences.get("communication_style")
            if isinstance(communication_style, str) and communication_style:
                preference_parts.append("沟通偏好: " + communication_style)
            focus_areas = preferences.get("focus_areas")
            if isinstance(focus_areas, list):
                values = [value for value in focus_areas if isinstance(value, str) and value]
                if values:
                    preference_parts.append("关注重点: " + ", ".join(values))
            if preference_parts:
                append_document(
                    f"{memory_id}_preference",
                    "preference",
                    "；".join(preference_parts),
                    {"preferences": preferences},
                )

        return documents

    def store_preembedded_memory_vectors(
        self,
        memory_id: str,
        documents: Sequence[Dict[str, Any]],
    ) -> List[str]:
        """把已经生成向量的单条记忆文档原子性写入 Milvus。"""
        from pymilvus import Collection

        if not documents:
            return []
        for document in documents:
            vector = document.get("vector")
            if not isinstance(vector, list) or not vector:
                raise MemoryInfrastructureError(
                    f"{document.get('id', 'unknown')} 缺少 embedding"
                )

        _ = self.milvus
        collection = Collection(MEMORY_COLLECTION_NAME)
        collection.load()
        data = [
            [document["id"] for document in documents],
            [document["user_id"] for document in documents],
            [document["session_id"] for document in documents],
            [document["memory_type"] for document in documents],
            [document["content"][:65535] for document in documents],
            [document["metadata"] for document in documents],
            [document["vector"] for document in documents],
        ]
        # 重试时总结结果可能减少某些 insight。先按记忆前缀清掉旧碎片，
        # 再用确定性 ID upsert，既避免重复，也避免残留已经消失的碎片。
        collection.delete(f'id like "{memory_id}_%"')
        collection.upsert(data)
        collection.flush()
        return [document["id"] for document in documents]

    def delete_vectors(self, vector_ids: Sequence[str]) -> None:
        if not vector_ids:
            return
        from pymilvus import Collection, utility

        _ = self.milvus
        if not utility.has_collection(MEMORY_COLLECTION_NAME):
            return
        collection = Collection(MEMORY_COLLECTION_NAME)
        expression = f"id in {json.dumps(list(vector_ids))}"
        collection.delete(expression)
        collection.flush()

    def delete_memory_vectors(self, memory_id: str) -> None:
        """按确定性 ID 前缀删除一条记忆的全部碎片，包括旧版漏记的 ID。"""
        from pymilvus import Collection, utility

        _ = self.milvus
        if not utility.has_collection(MEMORY_COLLECTION_NAME):
            return
        collection = Collection(MEMORY_COLLECTION_NAME)
        collection.delete(f'id like "{memory_id}_%"')
        collection.flush()

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        memory_types: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        from pymilvus import Collection, utility

        try:
            _ = self.milvus
            if not utility.has_collection(MEMORY_COLLECTION_NAME):
                return []
            query_vector = generate_embedding(query)
            if not query_vector:
                raise MemoryInfrastructureError("查询 embedding 生成失败")

            collection = Collection(MEMORY_COLLECTION_NAME)
            collection.load()
            expression = f"user_id == {json.dumps(user_id)}"
            if memory_types:
                expression += f" and memory_type in {json.dumps(list(memory_types))}"
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                expr=expression,
                output_fields=["id", "session_id", "memory_type", "content", "metadata"],
            )
            return [
                {
                    "id": hit.entity.get("id"),
                    "session_id": hit.entity.get("session_id"),
                    "memory_type": hit.entity.get("memory_type"),
                    "content": hit.entity.get("content"),
                    "metadata": hit.entity.get("metadata"),
                    "score": float(hit.score),
                }
                for hits in results
                for hit in hits
            ]
        except MemoryInfrastructureError:
            raise
        except Exception as exc:
            logger.error("检索记忆失败: %s", exc)
            raise RuntimeError(f"Milvus 记忆检索失败: {exc}") from exc

    def _load_raw_evidence(
        self,
        user_id: str,
        query: str,
        hits: Sequence[MemoryHit],
        *,
        max_messages: int = 4,
        max_chars: int = 2400,
    ) -> List[Dict[str, str]]:
        """根据命中记忆的来源范围回填少量、可审计的原始消息。"""
        hit_contents: Dict[str, List[str]] = defaultdict(list)
        preferred_ids: Dict[str, List[str]] = defaultdict(list)
        memory_ids = set()
        for hit in hits:
            try:
                memory_uuid = _as_uuid(hit.memory_id)
            except ValueError:
                continue
            memory_ids.add(memory_uuid)
            hit_contents[hit.memory_id].append(hit.content)
            source_ids = hit.metadata.get("source_message_ids") or []
            if not isinstance(source_ids, list):
                source_ids = [source_ids]
            preferred_ids[hit.memory_id].extend(str(value) for value in source_ids)
        if not memory_ids:
            return []

        db = SessionLocal()
        try:
            memory_rows = db.query(LongTermMemory).filter(
                LongTermMemory.id.in_(memory_ids),
                LongTermMemory.user_id == _as_uuid(user_id),
                LongTermMemory.index_status == "ready",
                LongTermMemory.deleted_at.is_(None),
            ).all()
            memory_by_id = {str(memory.id): memory for memory in memory_rows}
            ordered_memory_ids = list(dict.fromkeys(hit.memory_id for hit in hits))
            evidence: List[Dict[str, str]] = []
            used_chars = 0
            remaining_messages = max_messages
            for memory_id in ordered_memory_ids:
                memory = memory_by_id.get(memory_id)
                if not memory:
                    continue
                if remaining_messages <= 0 or used_chars >= max_chars:
                    break
                if not memory.session_id:
                    continue

                boundaries = [
                    value
                    for value in (
                        memory.source_start_message_id,
                        memory.source_end_message_id,
                    )
                    if value
                ]
                boundary_messages = {
                    message.id: message
                    for message in db.query(ChatMessage).filter(
                        ChatMessage.id.in_(boundaries),
                        ChatMessage.session_id == memory.session_id,
                    ).all()
                } if boundaries else {}
                start = boundary_messages.get(memory.source_start_message_id)
                end = boundary_messages.get(memory.source_end_message_id)

                message_query = db.query(ChatMessage).filter(
                    ChatMessage.session_id == memory.session_id,
                )
                if start and end:
                    message_query = message_query.filter(
                        ChatMessage.created_at >= start.created_at,
                        ChatMessage.created_at <= end.created_at,
                    )
                else:
                    exact_ids = []
                    for value in preferred_ids[str(memory.id)]:
                        try:
                            exact_ids.append(_as_uuid(value))
                        except ValueError:
                            continue
                    if not exact_ids:
                        continue
                    message_query = message_query.filter(ChatMessage.id.in_(exact_ids))

                messages = message_query.order_by(
                    ChatMessage.created_at.asc(), ChatMessage.id.asc()
                ).all()
                selected = _select_relevant_raw_messages(
                    query,
                    hit_contents[str(memory.id)],
                    messages,
                    preferred_message_ids=preferred_ids[str(memory.id)],
                    max_messages=min(2, remaining_messages),
                )
                for message in selected:
                    available = max_chars - used_chars
                    if available <= 0:
                        break
                    content = str(message.content)[:available]
                    if not content:
                        continue
                    evidence.append({
                        "memory_id": str(memory.id),
                        "session_id": str(memory.session_id),
                        "message_id": str(message.id),
                        "role": str(message.role),
                        "created_at": _message_time(message),
                        "content": content,
                    })
                    used_chars += len(content)
                    remaining_messages -= 1
                    if remaining_messages <= 0:
                        break
            return evidence
        finally:
            db.close()

    def build_memory_context(self, user_id: str, current_query: str, max_memories: int = 3) -> str:
        candidates = get_memory_provider().search(
            MemoryScope(user_id=user_id),
            current_query,
            MemorySearchOptions(limit=max(max_memories * 4, max_memories)),
        )
        hits = _select_diverse_hits(candidates, max_memories)
        if not hits:
            return ""

        labels = {
            "summary": "历史对话摘要",
            "insight": "相关知识点",
            "topics": "相关主题",
            "preference": "用户偏好",
            "atomic_fact": "原子事实",
        }
        lines = ["[相关历史记忆]"]
        for hit in hits:
            source_times = hit.metadata.get("source_times") or []
            if not isinstance(source_times, list):
                source_times = [source_times]
            attributes = []
            if hit.metadata.get("occurred_at"):
                attributes.append(f"事件时间={hit.metadata['occurred_at']}")
            if hit.metadata.get("valid_from"):
                attributes.append(f"生效时间={hit.metadata['valid_from']}")
            if source_times:
                attributes.append(f"来源时间={source_times[0]}")
            source_ids = hit.metadata.get("source_message_ids") or []
            if not isinstance(source_ids, list):
                source_ids = [source_ids]
            if source_ids:
                attributes.append(f"来源消息={','.join(str(value) for value in source_ids)}")
            suffix = f"；{'；'.join(attributes)}" if attributes else ""
            lines.append(
                f"- {labels.get(hit.memory_type, '历史信息')}{suffix}: {hit.content}"
            )

        raw_evidence = self._load_raw_evidence(user_id, current_query, hits)
        if raw_evidence:
            lines.append("[原始证据]")
            for item in raw_evidence:
                lines.append(
                    f"- [{item['created_at']}; message_id={item['message_id']}; "
                    f"role={item['role']}] {item['content']}"
                )
        lines.append("")
        return "\n".join(lines)


class NativeMemoryProvider:
    name = "native"

    def __init__(self, service: Optional[MemoryService] = None):
        self.service = service or get_memory_service()

    def add_turns(
        self,
        scope: MemoryScope,
        turns: Sequence[MemoryTurn],
        operation_id: str,
    ) -> MemoryWriteResult:
        if not scope.session_id:
            raise ValueError("add_turns 需要 session_id")
        if not turns:
            return MemoryWriteResult(operation_id=operation_id, status="succeeded")

        db = SessionLocal()
        try:
            memory = self.service.create_memory(
                db,
                user_id=scope.user_id,
                session_id=scope.session_id,
                messages=turns,
                source_start_message_id=turns[0].message_id,
                source_end_message_id=turns[-1].message_id,
            )
            return MemoryWriteResult(
                operation_id=operation_id,
                status="succeeded",
                memory_ids=[str(memory.id)],
            )
        finally:
            db.close()

    def search(
        self,
        scope: MemoryScope,
        query: str,
        options: MemorySearchOptions,
    ) -> List[MemoryHit]:
        # PostgreSQL是可用记忆的权限账本；新身份没有候选时无需访问向量服务。
        db = SessionLocal()
        try:
            exists = db.query(LongTermMemory.id).filter(
                LongTermMemory.user_id == _as_uuid(scope.user_id),
                LongTermMemory.index_status == "ready",
                LongTermMemory.deleted_at.is_(None),
            ).first()
        finally:
            db.close()
        if exists is None:
            return []
        candidate_limit = max(options.limit * 4, options.limit)
        raw_hits = self.service.retrieve_memories(
            scope.user_id,
            query,
            candidate_limit,
            memory_types=options.memory_types,
        )
        if not raw_hits:
            return []

        parsed: List[tuple] = []
        memory_ids = set()
        for hit in raw_hits:
            try:
                metadata = json.loads(hit.get("metadata") or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            memory_id = metadata.get("memory_id")
            if not memory_id:
                continue
            try:
                memory_ids.add(_as_uuid(memory_id))
            except ValueError:
                continue
            parsed.append((hit, metadata, memory_id))

        db = SessionLocal()
        try:
            query_set = db.query(LongTermMemory).filter(
                LongTermMemory.id.in_(memory_ids),
                LongTermMemory.user_id == _as_uuid(scope.user_id),
                LongTermMemory.index_status == "ready",
                LongTermMemory.deleted_at.is_(None),
            )
            effective_session_id = options.session_id or scope.session_id
            if effective_session_id:
                query_set = query_set.filter(
                    LongTermMemory.session_id == _as_uuid(effective_session_id)
                )
            allowed = {str(memory.id): memory for memory in query_set.all()}
        finally:
            db.close()

        results: List[MemoryHit] = []
        for hit, metadata, memory_id in parsed:
            if memory_id not in allowed:
                continue
            if options.memory_types and hit.get("memory_type") not in options.memory_types:
                continue
            score = float(hit.get("score", 0.0))
            if options.min_score is not None and score < options.min_score:
                continue
            results.append(MemoryHit(
                memory_id=memory_id,
                fragment_id=hit.get("id", ""),
                memory_type=hit.get("memory_type", "unknown"),
                content=hit.get("content", ""),
                score=score,
                session_id=hit.get("session_id"),
                metadata=metadata,
            ))
        return _select_diverse_hits(results, options.limit)

    def update(
        self,
        scope: MemoryScope,
        memory_id: str,
        patch: MemoryPatch,
        operation_id: str,
    ) -> MemoryWriteResult:
        db = SessionLocal()
        try:
            query = db.query(LongTermMemory).filter(
                LongTermMemory.id == _as_uuid(memory_id),
                LongTermMemory.user_id == _as_uuid(scope.user_id),
                LongTermMemory.deleted_at.is_(None),
            )
            if scope.session_id:
                query = query.filter(
                    LongTermMemory.session_id == _as_uuid(scope.session_id)
                )
            memory = query.first()
            if not memory:
                raise LookupError("记忆不存在")
            if patch.summary is not None:
                memory.summary = patch.summary
            if patch.key_insights is not None:
                memory.key_insights = dict(patch.key_insights)
            if patch.metadata:
                memory_data = dict(memory.key_insights or {})
                memory_data["_metadata"] = {
                    **dict(memory_data.get("_metadata") or {}),
                    **patch.metadata,
                }
                memory.key_insights = memory_data
            memory.index_status = "indexing"
            db.commit()

            summary_data = memory.key_insights or {"summary": memory.summary}
            summary_data["summary"] = memory.summary
            memory.milvus_ids = self.service._store_memory_vectors(
                str(memory.id),
                str(memory.user_id),
                str(memory.session_id or ""),
                summary_data,
            )
            memory.index_status = "ready"
            memory.last_error = None
            db.commit()
            return MemoryWriteResult(
                operation_id=operation_id,
                status="succeeded",
                memory_ids=[memory_id],
            )
        except Exception as exc:
            db.rollback()
            query = db.query(LongTermMemory).filter(
                LongTermMemory.id == _as_uuid(memory_id),
                LongTermMemory.user_id == _as_uuid(scope.user_id),
            )
            if scope.session_id:
                query = query.filter(
                    LongTermMemory.session_id == _as_uuid(scope.session_id)
                )
            memory = query.first()
            if memory:
                memory.index_status = "failed"
                memory.last_error = str(exc)[:4000]
                db.commit()
            raise
        finally:
            db.close()

    def delete(self, scope: MemoryScope, memory_id: str, operation_id: str) -> None:
        db = SessionLocal()
        try:
            query = db.query(LongTermMemory).filter(
                LongTermMemory.id == _as_uuid(memory_id),
                LongTermMemory.user_id == _as_uuid(scope.user_id),
            )
            if scope.session_id:
                query = query.filter(
                    LongTermMemory.session_id == _as_uuid(scope.session_id)
                )
            memory = query.first()
            if not memory or memory.index_status == "deleted":
                return
            self.service.delete_memory_vectors(str(memory.id))
            memory.index_status = "deleted"
            memory.deleted_at = memory.deleted_at or utc_now()
            memory.last_error = None
            db.commit()
        finally:
            db.close()

    def reset_user(self, user_id: str, operation_id: str) -> None:
        db = SessionLocal()
        try:
            memory_ids = [
                str(memory.id)
                for memory in db.query(LongTermMemory).filter(
                    LongTermMemory.user_id == _as_uuid(user_id),
                    LongTermMemory.index_status != "deleted",
                ).all()
            ]
        finally:
            db.close()
        for memory_id in memory_ids:
            self.delete(MemoryScope(user_id=user_id), memory_id, operation_id)


_memory_service: Optional[MemoryService] = None
_memory_provider: Optional[MemoryProvider] = None


def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


def get_memory_provider() -> MemoryProvider:
    global _memory_provider
    if _memory_provider is None:
        if MEMORY_PROVIDER_NAME != "native":
            raise RuntimeError(
                f"尚未安装记忆 Provider '{MEMORY_PROVIDER_NAME}'；第一阶段仅支持 native"
            )
        _memory_provider = NativeMemoryProvider()
    return _memory_provider


def _get_unmemorized_messages(db: Session, session_id: str) -> List[ChatMessage]:
    last_memory = db.query(LongTermMemory).filter(
        LongTermMemory.session_id == _as_uuid(session_id),
        # 删除只让记忆不可检索，不应让已经消费过的旧消息再次被自动总结。
        LongTermMemory.index_status.in_(("ready", "deleting", "deleted")),
    ).order_by(LongTermMemory.created_at.desc()).first()

    query = db.query(ChatMessage).filter(ChatMessage.session_id == _as_uuid(session_id))
    if last_memory and last_memory.source_end_message_id:
        end_message = db.query(ChatMessage).filter(
            ChatMessage.id == last_memory.source_end_message_id
        ).first()
        if end_message:
            query = query.filter(or_(
                ChatMessage.created_at > end_message.created_at,
                (
                    (ChatMessage.created_at == end_message.created_at)
                    & (ChatMessage.id > end_message.id)
                ),
            ))
    elif last_memory:
        query = query.filter(ChatMessage.created_at > last_memory.created_at)

    return query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).all()


def _select_memory_batch(messages: Sequence[ChatMessage]) -> List[ChatMessage]:
    """按时间顺序截取一个可总结批次，避免截断后却把旧消息标记为已消费。"""
    selected: List[ChatMessage] = []
    chars = 0
    for message in messages:
        formatted_chars = len(message.content) + 5  # 角色前缀与换行的近似开销
        if selected and chars + formatted_chars > MEMORY_SUMMARY_MAX_CHARS:
            break
        selected.append(message)
        chars += formatted_chars
    return selected


def enqueue_memory_if_due(
    db: Session,
    user_id: str,
    session_id: str,
    *,
    force: bool = False,
    urgent_only: bool = False,
    idle_before: Optional[datetime] = None,
) -> Optional[MemoryJob]:
    """在当前事务中创建 add outbox；调用方负责 commit。"""
    owned_session = db.query(ChatSession.id).filter(
        ChatSession.id == _as_uuid(session_id),
        ChatSession.user_id == _as_uuid(user_id),
    ).first()
    if not owned_session:
        raise LookupError("会话不存在或不属于该用户")

    active = db.query(MemoryJob).filter(
        MemoryJob.session_id == _as_uuid(session_id),
        MemoryJob.operation_type == "add",
        MemoryJob.status.in_(ACTIVE_JOB_STATUSES),
    ).first()
    if active:
        return active

    unmemorized_messages = _get_unmemorized_messages(db, session_id)
    if not unmemorized_messages:
        return None
    trigger_reason = priority_memory_trigger(unmemorized_messages)
    if force:
        trigger_reason = "forced"
    elif not trigger_reason:
        if urgent_only:
            return None
        # 以最后一条真实消息判断空闲，避免仅按扫描时的会话快照误判。
        last_activity = unmemorized_messages[-1].created_at
        if (
            idle_before is not None and last_activity is not None
            and as_utc(last_activity) <= as_utc(idle_before)
        ):
            trigger_reason = "idle"
        elif get_memory_service().should_compress(unmemorized_messages):
            trigger_reason = "incremental_tokens"
        else:
            return None

    messages = _select_memory_batch(unmemorized_messages)
    force_followup = force or len(messages) < len(unmemorized_messages)
    end_message_id = str(messages[-1].id)
    idempotency_key = f"add:{session_id}:{end_message_id}"
    existing = db.query(MemoryJob).filter(
        MemoryJob.idempotency_key == idempotency_key
    ).first()
    if existing:
        # dead 是自动重试的终态；只有显式 create(force=True) 才重新激活。
        if force and existing.status == "dead":
            existing.status = "pending"
            existing.attempts = 0
            existing.next_attempt_at = utc_now()
            existing.locked_at = None
            existing.last_error = None
        return existing

    job = MemoryJob(
        operation_type="add",
        provider=MEMORY_PROVIDER_NAME,
        user_id=_as_uuid(user_id),
        session_id=_as_uuid(session_id),
        payload={
            "message_ids": [str(message.id) for message in messages],
            "force_followup": force_followup,
            "trigger_reason": trigger_reason,
        },
        idempotency_key=idempotency_key,
        status="pending",
        next_attempt_at=utc_now(),
    )
    try:
        # 与助手消息处于同一外层事务；并发唯一键冲突只回滚这个保存点，
        # 不能把已经生成的助手消息也一起回滚。
        with db.begin_nested():
            db.add(job)
            db.flush()
        return job
    except IntegrityError:
        concurrent = db.query(MemoryJob).filter(or_(
            MemoryJob.idempotency_key == idempotency_key,
            (
                (MemoryJob.session_id == _as_uuid(session_id))
                & (MemoryJob.operation_type == "add")
                & (MemoryJob.status.in_(ACTIVE_JOB_STATUSES))
            ),
        )).first()
        if concurrent:
            return concurrent
        raise


def enqueue_delete_memory(db: Session, user_id: str, memory_id: str) -> Optional[MemoryJob]:
    memory = db.query(LongTermMemory).filter(
        LongTermMemory.id == _as_uuid(memory_id),
        LongTermMemory.user_id == _as_uuid(user_id),
        LongTermMemory.index_status != "deleted",
    ).first()
    if not memory:
        return None

    existing = db.query(MemoryJob).filter(
        MemoryJob.idempotency_key == f"delete:{memory_id}"
    ).first()
    memory.index_status = "deleting"
    memory.deleted_at = memory.deleted_at or utc_now()
    if existing:
        if existing.status in ("dead", "succeeded"):
            existing.status = "pending"
            existing.attempts = 0
            existing.next_attempt_at = utc_now()
            existing.locked_at = None
            existing.last_error = None
        return existing

    job = MemoryJob(
        operation_type="delete",
        provider=memory.provider or "native",
        user_id=memory.user_id,
        session_id=memory.session_id,
        memory_id=memory.id,
        payload={},
        idempotency_key=f"delete:{memory_id}",
        status="pending",
        next_attempt_at=utc_now(),
    )
    try:
        with db.begin_nested():
            db.add(job)
            db.flush()
        return job
    except IntegrityError:
        concurrent = db.query(MemoryJob).filter(
            MemoryJob.idempotency_key == f"delete:{memory_id}"
        ).first()
        if concurrent:
            return concurrent
        raise


def _claim_memory_jobs(batch_size: int) -> List[str]:
    db = SessionLocal()
    try:
        now = utc_now()
        stale_before = now - timedelta(minutes=10)
        db.query(MemoryJob).filter(
            MemoryJob.status == "running",
            MemoryJob.locked_at < stale_before,
        ).update({
            MemoryJob.status: "retry",
            MemoryJob.next_attempt_at: now,
            MemoryJob.locked_at: None,
            MemoryJob.last_error: "任务锁超时，已自动恢复",
        }, synchronize_session=False)

        jobs = db.query(MemoryJob).filter(
            MemoryJob.status.in_(("pending", "retry")),
            MemoryJob.next_attempt_at <= now,
        ).order_by(MemoryJob.created_at.asc()).with_for_update(skip_locked=True).limit(batch_size).all()

        for job in jobs:
            job.status = "running"
            job.attempts += 1
            job.locked_at = now
        ids = [str(job.id) for job in jobs]
        db.commit()
        return ids
    finally:
        db.close()


def _execute_memory_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(MemoryJob).filter(MemoryJob.id == _as_uuid(job_id)).first()
        if not job or job.status != "running":
            return

        if job.operation_type == "semantic_context":
            from service.semantic_context_service import (
                enqueue_semantic_context_if_due,
                execute_semantic_context_job,
            )

            session = execute_semantic_context_job(db, job)
            job.status = "succeeded"
            job.locked_at = None
            job.last_error = None
            db.flush()
            enqueue_semantic_context_if_due(
                db,
                str(job.user_id),
                str(job.session_id),
                force=bool(job.payload.get("force_followup")),
            )
            db.commit()
            from service.session_memory_service import get_session_memory_service

            get_session_memory_service().invalidate(
                user_id=str(session.user_id),
                session_id=str(session.id),
            )
            return

        provider = get_memory_provider()
        if provider.name != job.provider:
            raise RuntimeError(f"任务需要 Provider {job.provider}，当前为 {provider.name}")

        if job.operation_type == "add":
            owned_session = db.query(ChatSession.id).filter(
                ChatSession.id == job.session_id,
                ChatSession.user_id == job.user_id,
            ).first()
            if not owned_session:
                raise RuntimeError("记忆任务的会话与用户不匹配")
            message_ids = [_as_uuid(value) for value in job.payload.get("message_ids", [])]
            messages_by_id = {
                message.id: message
                for message in db.query(ChatMessage).filter(
                    ChatMessage.id.in_(message_ids),
                    ChatMessage.session_id == job.session_id,
                ).all()
            }
            messages = [messages_by_id[value] for value in message_ids if value in messages_by_id]
            if len(messages) != len(message_ids):
                raise RuntimeError("记忆任务引用的部分消息已不存在")
            turns = [MemoryTurn(
                message_id=str(message.id),
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            ) for message in messages]
            result = provider.add_turns(
                MemoryScope(user_id=str(job.user_id), session_id=str(job.session_id)),
                turns,
                str(job.id),
            )
            if result.memory_ids:
                job.memory_id = _as_uuid(result.memory_ids[0])
        elif job.operation_type == "update":
            provider.update(
                MemoryScope(user_id=str(job.user_id), session_id=str(job.session_id) if job.session_id else None),
                str(job.memory_id),
                MemoryPatch(**job.payload.get("patch", {})),
                str(job.id),
            )
        elif job.operation_type == "delete":
            provider.delete(
                MemoryScope(user_id=str(job.user_id), session_id=str(job.session_id) if job.session_id else None),
                str(job.memory_id),
                str(job.id),
            )
        elif job.operation_type == "reset_user":
            provider.reset_user(str(job.user_id), str(job.id))
        else:
            raise ValueError(f"未知记忆任务类型: {job.operation_type}")

        job.status = "succeeded"
        job.locked_at = None
        job.last_error = None
        db.flush()
        if job.operation_type == "add" and job.session_id:
            enqueue_memory_if_due(
                db,
                str(job.user_id),
                str(job.session_id),
                force=bool(job.payload.get("force_followup")),
            )
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.query(MemoryJob).filter(MemoryJob.id == _as_uuid(job_id)).first()
        if job:
            attempt_index = max(0, job.attempts - 1)
            if attempt_index >= len(RETRY_DELAYS_SECONDS):
                job.status = "dead"
            else:
                job.status = "retry"
                job.next_attempt_at = utc_now() + timedelta(
                    seconds=RETRY_DELAYS_SECONDS[attempt_index]
                )
            job.locked_at = None
            job.last_error = str(exc)[:4000]
            if job.operation_type == "semantic_context" and job.session_id:
                session = db.query(ChatSession).filter(
                    ChatSession.id == job.session_id,
                    ChatSession.user_id == job.user_id,
                ).first()
                if session:
                    session.semantic_context_status = (
                        "failed" if job.status == "dead" else "retry"
                    )
                    session.semantic_context_last_error = str(exc)[:4000]
            db.commit()
        logger.exception("记忆任务 %s 执行失败", job_id)
    finally:
        db.close()


def enqueue_idle_memory_sessions(batch_size: int = 10) -> int:
    """无需新聊天请求也能处理空闲会话；原文、游标和 Outbox 都以数据库为准。"""
    if MEMORY_IDLE_SECONDS <= 0 or batch_size <= 0:
        return 0
    cutoff = utc_now() - timedelta(seconds=MEMORY_IDLE_SECONDS)
    db = SessionLocal()
    try:
        # 两个相关子查询均利用 session/created_at 索引，避免加载全部会话原文。
        def latest(column):
            return select(column).where(
                ChatMessage.session_id == ChatSession.id,
            ).order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(1).correlate(
                ChatSession
            ).scalar_subquery()

        latest_id = latest(ChatMessage.id)
        latest_time = latest(ChatMessage.created_at)
        consumed = exists().where(
            LongTermMemory.session_id == ChatSession.id,
            LongTermMemory.index_status.in_(("ready", "deleting", "deleted")),
            or_(
                LongTermMemory.source_end_message_id == latest_id,
                and_(
                    LongTermMemory.source_end_message_id.is_(None),
                    LongTermMemory.created_at >= latest_time,
                ),
            ),
        )
        blocked = exists().where(
            MemoryJob.session_id == ChatSession.id,
            MemoryJob.operation_type == "add",
            MemoryJob.status.in_((*ACTIVE_JOB_STATUSES, "dead")),
        )
        sessions = db.query(ChatSession.id, ChatSession.user_id).filter(
            ChatSession.user_id.isnot(None),
            latest_time <= cutoff,
            ~consumed,
            ~blocked,
        ).order_by(latest_time.asc(), ChatSession.id.asc()).limit(batch_size).all()
        enqueued = 0
        for session_id, user_id in sessions:
            try:
                job = enqueue_memory_if_due(
                    db, str(user_id), str(session_id), idle_before=cutoff,
                )
                db.commit()
                if job and job.status in ACTIVE_JOB_STATUSES:
                    enqueued += 1
            except Exception:
                db.rollback()
                logger.exception("空闲会话长期记忆入队失败: %s", session_id)
        return enqueued
    finally:
        db.close()


def process_memory_jobs(batch_size: int = 10) -> int:
    # 复用现有五秒 Worker，不依赖用户再次发消息来触发空闲批处理。
    try:
        enqueue_idle_memory_sessions(batch_size)
    except Exception:
        # 扫描失败不能阻止已经持久化的任务继续执行。
        logger.exception("空闲会话长期记忆扫描失败")
    job_ids = _claim_memory_jobs(batch_size)
    for job_id in job_ids:
        _execute_memory_job(job_id)
    return len(job_ids)


def reconcile_memory_state() -> Dict[str, int]:
    """修复卡住的状态；孤立向量通过确定性 upsert/delete 在下次任务中收敛。"""
    db = SessionLocal()
    try:
        stale_before = utc_now() - timedelta(minutes=10)
        stale_memories = db.query(LongTermMemory).filter(
            LongTermMemory.index_status == "indexing",
            LongTermMemory.updated_at < stale_before,
        ).all()
        for memory in stale_memories:
            memory.index_status = "failed"
            memory.last_error = "索引状态超时，等待对应任务重试"
        db.commit()
        return {"stale_memories": len(stale_memories)}
    finally:
        db.close()


# 兼容旧的内部调用名称；不再由 FastAPI BackgroundTasks 使用。
def get_unmemorized_messages(db: Session, session_id: str) -> List[ChatMessage]:
    return _get_unmemorized_messages(db, session_id)
