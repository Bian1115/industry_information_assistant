# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""对话上下文预算裁剪：保留已有语义摘要和最近轮次，不生成旧对话摘录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from service.qwen_token_counter import QwenTokenCounter, get_qwen_token_counter


@dataclass(frozen=True)
class CompactionResult:
    messages: List[Dict[str, str]]
    original_tokens: int
    compacted_tokens: int
    summarized_messages: int  # 兼容既有统计接口；本层不生成摘要，始终为 0。
    pruned_tool_results: int


def _count_messages(token_counter: Any, messages: List[Dict[str, str]]) -> int:
    return token_counter.count_messages(messages) if messages else 0


def _truncate_tool_result(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return f"{text[:head]} … {text[-tail:]}"


def _group_turns(messages: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    groups: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    for message in messages:
        if message["role"] == "user" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return groups


def compact_messages(
    messages: List[Dict[str, str]],
    *,
    token_limit: int,
    max_messages: int = 20,
    tool_result_chars: int = 4000,
    token_counter: Optional[QwenTokenCounter] = None,
) -> CompactionResult:
    """只执行预算裁剪；较早语义由调用方提供的 LLM 摘要承载。"""
    from service.context_policy import session_context_compression_enabled
    if not session_context_compression_enabled():
        token_counter = token_counter or get_qwen_token_counter()
        original = [{"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
                    for item in messages]
        tokens = _count_messages(token_counter, original)
        return CompactionResult(original, tokens, tokens, 0, 0)
    if token_limit <= 0 or max_messages <= 0:
        return CompactionResult([], 0, 0, 0, 0)
    token_counter = token_counter or get_qwen_token_counter()
    normalized = [
        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
        for item in messages
    ]
    original_tokens = _count_messages(token_counter, normalized)

    pruned_tool_results = 0
    for item in normalized:
        if item["role"] in {"tool", "function"} and len(item["content"]) > tool_result_chars:
            item["content"] = (
                _truncate_tool_result(item["content"], tool_result_chars)
                + "\n[工具结果已裁剪；原始内容仍保存在事件账本中]"
            )
            pruned_tool_results += 1

    if (
        _count_messages(token_counter, normalized) <= token_limit
        and len(normalized) <= max_messages
    ):
        return CompactionResult(
            normalized,
            original_tokens,
            _count_messages(token_counter, normalized),
            0,
            pruned_tool_results,
        )

    system_messages = [item for item in normalized if item["role"] == "system"][:1]
    conversation = [item for item in normalized if item["role"] != "system"]
    groups = _group_turns(conversation)
    # 不再为规则摘录预留预算；已有 system 上下文之外的空间都供近期原文使用。
    tail_budget = max(0, token_limit - _count_messages(token_counter, system_messages))
    tail_message_limit = max_messages - len(system_messages)
    tail_groups: List[List[Dict[str, str]]] = []
    tail_tokens = 0
    tail_count = 0
    for group in reversed(groups):
        group_tokens = _count_messages(token_counter, group)
        if tail_groups and (
            tail_tokens + group_tokens > tail_budget
            or tail_count + len(group) > tail_message_limit
        ):
            break
        tail_groups.append(group)
        tail_tokens += group_tokens
        tail_count += len(group)
    tail_groups.reverse()
    compacted = system_messages + [item for group in tail_groups for item in group]
    # 最终保险：先移除最旧轮次；最新一轮过长时裁剪正文，不生成替代摘要。
    guard = 0
    while _count_messages(token_counter, compacted) > token_limit:
        guard += 1
        if guard > 100:
            break
        conversation_indices = [
            index for index, item in enumerate(compacted) if item["role"] != "system"
        ]
        user_indices = [
            index for index in conversation_indices if compacted[index]["role"] == "user"
        ]
        # 仍有多轮时删除最旧的完整轮次，避免只留下“回答”没有“问题”。
        if len(user_indices) > 1:
            first_user = user_indices[0]
            next_user = user_indices[1]
            del compacted[first_user:next_user]
            continue

        # 只剩最新完整轮次时按 token 精确裁剪最长消息，仍保留每个角色。
        tokenized = [token_counter.encode(item["content"]) for item in compacted]
        total = _count_messages(token_counter, compacted)
        candidates = conversation_indices or list(range(len(tokenized)))
        longest_index = max(candidates, key=lambda index: len(tokenized[index]))
        longest = tokenized[longest_index]
        remove_count = max(1, total - token_limit)
        keep_count = max(8, len(longest) - remove_count)
        if keep_count >= len(longest):
            keep_count = max(1, len(longest) - 1)
        compacted[longest_index]["content"] = token_counter.decode(longest[:keep_count])

    if len(compacted) > max_messages:
        # 语义摘要和安全说明使用 system 角色；最终消息数裁剪不能把它从头部挤掉。
        first_system = next(
            (item for item in compacted if item["role"] == "system"),
            None,
        )
        if first_system is not None and max_messages > 1:
            remaining = [item for item in compacted if item is not first_system]
            compacted = [first_system] + remaining[-(max_messages - 1):]
        else:
            compacted = compacted[-max_messages:]

    compacted_tokens = _count_messages(token_counter, compacted)
    return CompactionResult(
        messages=compacted,
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
        summarized_messages=0,
        pruned_tool_results=pruned_tool_results,
    )
