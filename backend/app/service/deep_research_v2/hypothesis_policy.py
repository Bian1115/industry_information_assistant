"""假设机制开关：首跑解析配置，执行及续跑只读取冻结状态。"""

import os
from typing import Any, Mapping, Optional


def resolve_hypothesis_enabled(value: Optional[bool] = None) -> bool:
    if value is not None:
        if not isinstance(value, bool):
            raise ValueError("hypothesis_enabled 必须是布尔值")
        return value
    raw = os.getenv("RESEARCH_HYPOTHESIS_ENABLED", "true").strip().lower()
    if raw in {"true", "1", "yes", "on"}:
        return True
    if raw in {"false", "0", "no", "off"}:
        return False
    raise ValueError("RESEARCH_HYPOTHESIS_ENABLED 必须是有效布尔值")


def hypotheses_enabled(state: Mapping[str, Any]) -> bool:
    """旧检查点缺少开关时保持历史行为；不受当前环境默认值影响。"""
    value = state.get("hypothesis_enabled", True)
    if not isinstance(value, bool):
        raise ValueError("已保存的 hypothesis_enabled 必须是布尔值")
    return value
