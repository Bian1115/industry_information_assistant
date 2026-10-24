# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
LLM 客户端工厂 —— 统一的超时与重试配置

## 为什么需要它

项目里有 9 处 `OpenAI(api_key=..., base_url=...)`，全都没传 timeout。
openai SDK 的默认值是 read=600 秒 + max_retries=2，
意味着**单次调用最坏要等 1800 秒才放弃**。

这不是理论问题：2026-08-18 跑 DeepResearch Bench 时，
DeepScout 有一次调用实测挂了 3547 秒（近 1 小时），
一道题在搜索阶段耗掉 61 分钟也没走完，整批评测因此报废。
而项目里其他 HTTP 调用（stock_service、bidding_service 等）
都规规矩矩设了 10~30 秒超时，唯独 LLM 客户端漏了。

## 默认值怎么定的

read=300 秒：LeadWriter 的 max_tokens 默认上限为 60000，
留够生成时间；但不至于像 600 秒那样把挂起的请求拖成灾难。

max_retries=1：单次调用最坏 2 × 300 = 600 秒封顶，
相比原来的 1800 秒收敛到三分之一。

写作类 Agent 若确实超时，用环境变量调大即可，不必改代码。

## 环境变量

    LLM_TIMEOUT_CONNECT   建连超时，默认 10 秒
    LLM_TIMEOUT_READ      读响应超时，默认 300 秒
    LLM_TIMEOUT_WRITE     发请求超时，默认 60 秒
    LLM_MAX_RETRIES       重试次数，默认 1
    LLM_AGENT_MAX_ATTEMPTS Agent 显式尝试次数，默认 3（含首次请求）
    LLM_FALLBACK_MODEL    瞬时故障后的备用模型；为空则继续使用原模型
    LLM_MAX_OUTPUT_TOKENS 通用输出默认硬上限，默认 3000；Writer 独立默认 60000
    LLM_MAX_OUTPUT_TOKENS_<AGENT> 按 Agent 覆盖，例如 DEEPSCOUT=4000
"""

import os
import logging
from typing import Optional

import httpx
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        logger.warning(f"[LLMClient] {name} 取值非法，回退默认值 {default}")
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        logger.warning(f"[LLMClient] {name} 取值非法，回退默认值 {default}")
        return default


def default_timeout() -> httpx.Timeout:
    """当前生效的超时配置。"""
    return httpx.Timeout(
        connect=_env_float("LLM_TIMEOUT_CONNECT", 10.0),
        read=_env_float("LLM_TIMEOUT_READ", 300.0),
        write=_env_float("LLM_TIMEOUT_WRITE", 60.0),
        pool=_env_float("LLM_TIMEOUT_WRITE", 60.0),
    )


def default_max_retries() -> int:
    return _env_int("LLM_MAX_RETRIES", 1)


def default_agent_max_attempts() -> int:
    """Agent 应用层显式尝试次数；限制在 1~5，避免误配置造成重试风暴。"""
    return max(1, min(5, _env_int("LLM_AGENT_MAX_ATTEMPTS", 3)))


def default_fallback_model() -> Optional[str]:
    value = os.getenv("LLM_FALLBACK_MODEL", "").strip()
    return value or None


def default_max_output_tokens(agent_name: Optional[str] = None) -> int:
    """按 Agent 限制长输出，兼顾连接稳定与结构化 JSON 完整性。"""
    per_agent_defaults = {
        "ChiefArchitect": 3000,
        "DeepScout": 6000,
        "DataAnalyst": 3000,
        "LeadWriter": 60000,
        "CriticMaster": 6000,
        "CodeWizard": 3000,
    }
    default = per_agent_defaults.get(agent_name or "", 3000)
    upper_limit = 65536 if agent_name == "LeadWriter" else 16000
    if agent_name:
        normalized = "".join(ch if ch.isalnum() else "_" for ch in agent_name).upper()
        specific = os.getenv(f"LLM_MAX_OUTPUT_TOKENS_{normalized}")
        if specific is not None:
            try:
                return max(256, min(upper_limit, int(specific)))
            except (TypeError, ValueError):
                logger.warning(
                    "[LLMClient] LLM_MAX_OUTPUT_TOKENS_%s 取值非法，回退默认值 %s",
                    normalized,
                    default,
                )
    if agent_name == "LeadWriter":
        # Writer 的正文、claims 与引句共享输出；通用 3000 上限不压低其独立默认值。
        # 显式 LLM_MAX_OUTPUT_TOKENS_LEADWRITER 仍可在上方覆盖。
        return default
    return max(256, min(16000, _env_int("LLM_MAX_OUTPUT_TOKENS", default)))


def create_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    *,
    read_timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    **kwargs,
) -> OpenAI:
    """
    创建带超时兜底的 OpenAI 客户端。

    Args:
        api_key: API 密钥
        base_url: 服务地址
        read_timeout: 单独放宽这一个客户端的读超时（秒）。
                      长文写作类 Agent 可以传更大的值，
                      不影响其他调用方。
        max_retries: 覆盖重试次数
        **kwargs: 透传给 OpenAI 构造函数的其他参数

    Returns:
        配置好超时与重试的 OpenAI 客户端
    """
    timeout = default_timeout()
    if read_timeout is not None:
        timeout = httpx.Timeout(
            connect=timeout.connect,
            read=read_timeout,
            write=timeout.write,
            pool=timeout.pool,
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=default_max_retries() if max_retries is None else max_retries,
        **kwargs,
    )


def create_async_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    *,
    read_timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    **kwargs,
) -> AsyncOpenAI:
    """异步客户端；用于需要真正传播取消信号的 Agent 工作流。"""
    timeout = default_timeout()
    if read_timeout is not None:
        timeout = httpx.Timeout(
            connect=timeout.connect,
            read=read_timeout,
            write=timeout.write,
            pool=timeout.pool,
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=default_max_retries() if max_retries is None else max_retries,
        **kwargs,
    )


def describe_config() -> str:
    """给日志/自检用的一行配置摘要。"""
    t = default_timeout()
    worst = (t.read or 0) * (default_max_retries() + 1)
    return (
        f"connect={t.connect}s read={t.read}s write={t.write}s "
        f"retries={default_max_retries()} → 单次调用最坏 {worst:.0f}s"
    )
