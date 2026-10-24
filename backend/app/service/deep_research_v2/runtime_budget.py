"""DeepResearch 阶段预算配置。所有值均可通过环境变量覆盖。"""

from __future__ import annotations

import os
from typing import Dict


DEFAULT_PHASE_SECONDS: Dict[str, int] = {
    "ChiefArchitect": 180,
    "DeepScout": 900,
    "DataAnalyst": 300,
    "CodeWizard": 180,
    "LeadWriter": 600,
    "CriticMaster": 300,
}


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


def phase_budget_seconds(agent_name: str) -> int:
    default = DEFAULT_PHASE_SECONDS.get(agent_name, 300)
    key = f"RESEARCH_BUDGET_{agent_name.upper()}_SECONDS"
    return _positive_int(key, default)


def total_budget_seconds() -> int:
    return _positive_int("RESEARCH_TOTAL_BUDGET_SECONDS", 2400)


def max_llm_tokens() -> int:
    """研究累计 token 上限；0 表示不限制，仍保留消耗统计。"""
    try:
        return max(0, int(os.getenv("RESEARCH_MAX_LLM_TOKENS", "0")))
    except (TypeError, ValueError):
        return 0
