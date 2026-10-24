"""仅供隔离实验：将项目Chat Completions调用转到既有Claude Relay，保留原生usage。"""
from contextlib import contextmanager, ExitStack
import os
import time
from unittest.mock import patch

import httpx


class 实验禁用付费向量检索(RuntimeError):
    """触发现有会话证据RAG的词法检索降级，不发送欠费服务请求。"""


def payload_for(kwargs, model):
    system, messages = [], []
    for row in kwargs["messages"]:
        role, content = row["role"], row["content"]
        if role in {"system", "developer"}:
            if not isinstance(content, str):
                raise ValueError("Claude实验只接受文本系统消息")
            system.append(content)
        elif role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})
        else:
            raise ValueError("Claude实验不支持此消息角色")
    if kwargs.get("response_format", {}).get("type") == "json_object":
        system.append("本次调用要求单个JSON对象。只输出JSON对象，不要Markdown代码块。")
    payload = {"model": model, "messages": messages, "max_tokens": kwargs.get("max_tokens", 16000)}
    if system:
        payload["system"] = "\n\n".join(system)
    # 与项目现有Claude客户端相同：Opus服务不接受temperature，不传思考参数。
    return payload


def completion_from(response, requested_model):
    from openai import RateLimitError, InternalServerError, APIStatusError
    from openai.types.chat import ChatCompletion
    if response.is_error:
        cls = RateLimitError if response.status_code == 429 else InternalServerError if response.status_code >= 500 else APIStatusError
        raise cls(f"Claude Relay HTTP {response.status_code}", response=response, body=None)
    body = response.json()
    content = "".join(block.get("text", "") for block in body.get("content", []) if block.get("type") == "text")
    if not content:
        raise ValueError("Claude Relay没有返回文本块")
    raw = body.get("usage") or {}
    usage = None
    if type(raw.get("input_tokens")) is int and type(raw.get("output_tokens")) is int:
        inputs = raw["input_tokens"] + sum(raw.get(key) or 0 for key in ("cache_creation_input_tokens", "cache_read_input_tokens"))
        usage = {"prompt_tokens": inputs, "completion_tokens": raw["output_tokens"],
            "total_tokens": raw.get("total_tokens", inputs + raw["output_tokens"]), "anthropic_raw_usage": raw}
    return ChatCompletion(id=body.get("id", "claude-relay"), created=int(time.time()), object="chat.completion",
        model=body.get("model") or requested_model, usage=usage,
        choices=[{"index": 0, "finish_reason": "length" if body.get("stop_reason") == "max_tokens" else "stop",
                  "message": {"role": "assistant", "content": content}}])


@contextmanager
def claude_profile(*, transport=None, lexical_only=True):
    from openai import APIConnectionError, APITimeoutError
    from openai.resources.chat.completions import Completions, AsyncCompletions
    from service.claude_client import messages_url
    from service.llm_client import default_timeout
    from service.summary_temporary_retrieval import RetrievalAPI
    key, base, model = (os.environ[name] for name in ("CLAUDE_RELAY_API_KEY", "CLAUDE_RELAY_BASE_URL", "CLAUDE_RELAY_MODEL"))
    url = messages_url(base)
    headers = {"Authorization": "Bearer " + key, "anthropic-version": "2023-06-01"}
    async def async_call(client, *args, **kwargs):
        try:
            async with httpx.AsyncClient(timeout=default_timeout(), transport=transport) as native:
                response = await native.post(url, headers=headers, json=payload_for(kwargs, model))
        except httpx.TimeoutException as exc:
            raise APITimeoutError(request=exc.request) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(message="Claude Relay连接失败", request=exc.request) from exc
        return completion_from(response, model)
    def sync_call(client, *args, **kwargs):
        try:
            with httpx.Client(timeout=default_timeout(), transport=transport) as native:
                response = native.post(url, headers=headers, json=payload_for(kwargs, model))
        except httpx.TimeoutException as exc:
            raise APITimeoutError(request=exc.request) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(message="Claude Relay连接失败", request=exc.request) from exc
        return completion_from(response, model)
    async def disabled(*args, **kwargs):
        raise 实验禁用付费向量检索("Claude单题实验使用现有词法检索，不请求欠费的Embedding/Rerank服务")
    with ExitStack() as stack:
        stack.enter_context(patch.object(Completions, "create", sync_call))
        stack.enter_context(patch.object(AsyncCompletions, "create", async_call))
        if lexical_only:
            stack.enter_context(patch.object(RetrievalAPI, "embed", disabled))
            stack.enter_context(patch.object(RetrievalAPI, "rerank", disabled))
        yield
