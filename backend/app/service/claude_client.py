# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""Claude Relay 的原生 Anthropic Messages API 客户端。"""

from typing import Mapping, Optional

import httpx

from service.llm_client import default_timeout


def messages_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return (
        f"{normalized}/messages"
        if normalized.endswith("/v1")
        else f"{normalized}/v1/messages"
    )


def messages_completion(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    system_prompt: Optional[str] = None,
) -> str:
    """调用 Claude 原生接口；Opus 5 不支持时不传 temperature。"""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        payload["system"] = system_prompt

    response = httpx.post(
        messages_url(base_url),
        headers={
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=default_timeout(),
    )
    if response.is_error:
        raise RuntimeError(
            f"Claude Messages HTTP {response.status_code}: {response.text[:1000]}"
        )
    body = response.json()
    text_parts = [
        str(block.get("text", ""))
        for block in body.get("content", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    ]
    result = "".join(text_parts).strip()
    if not result:
        raise RuntimeError("Claude Messages 返回中没有 text 内容")
    return result
