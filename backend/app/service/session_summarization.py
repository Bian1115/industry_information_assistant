"""会话滚动摘要：按阈值触发并保留近期消息，待摘要历史完整送入模型。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import get_buffer_string

from prompts.session_summary import DEFAULT_SUMMARY_PROMPT, PROMPT_SHA256, PROMPT_VERSION
from service.llm_client import create_openai_client
from service.qwen_token_counter import get_qwen_token_counter


SUMMARY_PREFIX = "Here is a summary of the conversation to date:\n\n"
# 2026-10-05 核对百炼官方模型说明；未知模型必须显式指定窗口。
MODEL_WINDOWS = {
    "qwen3.5-plus": 1_000_000,
    "qwen3.8-max": 1_000_000,
    "qwen3.8-max-0902": 1_000_000,
    "qwen3.8-max-2026-09-02": 1_000_000,
}


@dataclass(frozen=True)
class SessionSummaryConfig:
    model: str = "qwen3.5-plus"
    keep_messages: int = 10
    context_window: int = 1_000_000
    trigger_fraction: float = 0.85
    max_output_tokens: int = 2000
    request_attempts: int = 3
    # 触发按消费端预算计算；摘要请求仍按 context_window 校验。
    trigger_context_window: int | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("SEMANTIC_CONTEXT_MODEL 不能为空")
        for name in ("keep_messages", "context_window", "max_output_tokens", "request_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if not math.isfinite(self.trigger_fraction) or not 0 < self.trigger_fraction <= 1:
            raise ValueError("SEMANTIC_CONTEXT_TRIGGER_FRACTION 必须在 (0, 1] 内")
        if self.max_output_tokens >= self.context_window:
            raise ValueError("摘要输出上限必须小于上下文窗口")
        if self.trigger_context_window is not None and (
            isinstance(self.trigger_context_window, bool)
            or not isinstance(self.trigger_context_window, int)
            or self.trigger_context_window <= 0
        ):
            raise ValueError("trigger_context_window 必须是正整数或 None")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SessionSummaryConfig:
        values = os.environ if env is None else env
        model = values.get("SEMANTIC_CONTEXT_MODEL", "qwen3.5-plus").strip()
        raw_window = values.get("SEMANTIC_CONTEXT_CONTEXT_WINDOW", "").strip()
        if raw_window:
            window = int(raw_window)
        elif model in MODEL_WINDOWS:
            window = MODEL_WINDOWS[model]
        else:
            raise ValueError("未知摘要模型的窗口：请配置 SEMANTIC_CONTEXT_CONTEXT_WINDOW")
        raw_trigger_window = values.get("SEMANTIC_CONTEXT_TRIGGER_WINDOW", "").strip()
        return cls(
            model=model,
            keep_messages=int(values.get("SEMANTIC_CONTEXT_KEEP_MESSAGES", "10")),
            context_window=window,
            trigger_fraction=float(values.get("SEMANTIC_CONTEXT_TRIGGER_FRACTION", "0.85")),
            max_output_tokens=int(values.get("SEMANTIC_CONTEXT_MAX_OUTPUT_TOKENS", "2000")),
            request_attempts=int(values.get("SEMANTIC_CONTEXT_REQUEST_ATTEMPTS", "3")),
            trigger_context_window=int(raw_trigger_window) if raw_trigger_window else None,
        )

    @property
    def trigger_tokens(self) -> int:
        window = self.trigger_context_window if self.trigger_context_window is not None else self.context_window
        return max(1, int(window * self.trigger_fraction))

    @property
    def fingerprint(self) -> str:
        data = {**asdict(self), "prompt_sha256": PROMPT_SHA256}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def summary_message(text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": SUMMARY_PREFIX + text,
        "additional_kwargs": {"lc_source": "summarization"},
    }


def _to_message(item: Mapping[str, Any]):
    role, content = item.get("role", "user"), item.get("content", "")
    kwargs = {"content": content}
    if item.get("id"):
        kwargs["id"] = str(item["id"])
    if item.get("additional_kwargs"):
        kwargs["additional_kwargs"] = dict(item["additional_kwargs"])
    if role == "assistant":
        if item.get("tool_calls"):
            calls = []
            for call in item["tool_calls"]:
                if "function" in call:
                    function = call["function"]
                    arguments = function.get("arguments", "{}")
                    calls.append({
                        "id": call.get("id"), "name": function.get("name", ""),
                        "args": json.loads(arguments) if isinstance(arguments, str) else arguments,
                        "type": "tool_call",
                    })
                else:
                    calls.append(call)
            kwargs["tool_calls"] = calls
        return AIMessage(**kwargs)
    if role == "system":
        return SystemMessage(**kwargs)
    if role in {"tool", "function"}:
        return ToolMessage(**kwargs, tool_call_id=str(item.get("tool_call_id") or "unknown"))
    return HumanMessage(**kwargs)


class SessionSummarizer:
    def __init__(self, config=None, token_counter=None, completion=None) -> None:
        self.config = config or SessionSummaryConfig.from_env()
        self.token_counter = token_counter or get_qwen_token_counter()
        self.completion = completion
        self._client = None

    def should_summarize(self, messages: Sequence[Mapping[str, Any]], current_query: str = "") -> bool:
        working = list(messages)
        if current_query:
            working.append({"role": "user", "content": current_query})
        return self.token_counter.count_messages(working) >= self.config.trigger_tokens

    def partition(self, messages: Sequence[Mapping[str, Any]]) -> tuple[list, list]:
        """按条数切分；切点落在工具结果时，与对应 AI 调用一起保留。"""
        items = list(messages)
        cutoff = max(0, len(items) - self.config.keep_messages)
        if cutoff and items[cutoff].get("role") in {"tool", "function"}:
            tool_ids = set()
            end = cutoff
            while end < len(items) and items[end].get("role") in {"tool", "function"}:
                if items[end].get("tool_call_id"):
                    tool_ids.add(items[end]["tool_call_id"])
                end += 1
            matching = next((
                index for index in range(cutoff - 1, -1, -1)
                if items[index].get("role") == "assistant"
                and any(call.get("id") in tool_ids for call in items[index].get("tool_calls", []))
            ), None)
            if matching is not None:
                cutoff = matching
            else:
                cutoff = end
        return items[:cutoff], items[cutoff:]

    def _complete(self, prompt: str) -> tuple[str, dict]:
        last_error = None
        for attempt in range(self.config.request_attempts):
            try:
                if self.completion is not None:
                    result = self.completion(prompt)
                    if isinstance(result, dict):
                        text, usage = str(result.get("text") or ""), dict(result.get("usage") or {})
                    else:
                        text, usage = str(result or ""), {}
                else:
                    if self._client is None:
                        self._client = create_openai_client(
                            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                            max_retries=0,
                        )
                    response = self._client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0, max_tokens=self.config.max_output_tokens,
                    )
                    text = response.choices[0].message.content or ""
                    usage = response.usage.model_dump() if response.usage is not None else {}
                    if response.choices[0].finish_reason == "length":
                        raise RuntimeError("会话摘要输出被截断，原文和覆盖游标未更新")
                if not text.strip():
                    raise RuntimeError("会话摘要为空，原文和覆盖游标未更新")
                return text.strip(), {**usage, "attempts": attempt + 1}
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"会话摘要生成失败: {last_error}") from last_error

    def summarize(self, messages: Sequence[Mapping[str, Any]]) -> tuple[str, dict]:
        if not messages:
            raise ValueError("没有需要摘要的消息")
        started = time.monotonic()
        source_tokens = self.token_counter.count_messages(messages)
        converted = [_to_message(item) for item in messages]
        prompt = DEFAULT_SUMMARY_PROMPT.format(messages=get_buffer_string(converted, format="xml")).rstrip()
        estimated_prompt_tokens = self.token_counter.count_messages([{"role": "user", "content": prompt}])
        if estimated_prompt_tokens + self.config.max_output_tokens > self.config.context_window:
            raise ValueError("完整摘要请求与输出预留超过配置窗口，请增大摘要模型窗口或近期消息保留数量")
        text, usage = self._complete(prompt)
        summary_tokens = self.token_counter.count_messages([summary_message(text)])
        ratio = summary_tokens / source_tokens if source_tokens else 0.0
        metrics = {
            "model": self.config.model, "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256,
            "source_tokens": source_tokens, "summary_tokens": summary_tokens,
            "summary_input_tokens": source_tokens,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "compression_ratio": ratio, "savings_ratio": 1 - ratio,
            "tokens_saved": source_tokens - summary_tokens,
            "latency_ms": round((time.monotonic() - started) * 1000, 3), "attempts": usage["attempts"],
            "config_fingerprint": self.config.fingerprint,
            "context_window": self.config.context_window,
            "trigger_context_window": self.config.trigger_context_window,
            "trigger_tokens": self.config.trigger_tokens,
        }
        return text, metrics
