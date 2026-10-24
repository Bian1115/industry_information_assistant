"""DeepResearch 阶段级精确续跑游标。

检查点只在阶段完成后提交。恢复时从 ``next_stage`` 继续，因此已经提交的
阶段不会再次执行；若进程在阶段中途退出，则只重跑该阶段。
"""

from typing import Any, Dict


CURSOR_VERSION = 1
STAGES = (
    "planning",
    "researching",
    "analyzing",
    "writing",
    "reviewing",
    "re_researching",
    "revising",
    "completed",
)


def _legacy_next_stage(state: Dict[str, Any]) -> str:
    """为没有游标的历史检查点做保守推断。"""
    phase = str(state.get("phase") or "")
    if phase == "completed" or state.get("status") == "completed":
        return "completed"
    if phase in {"re_researching", "revising"}:
        return phase
    if state.get("final_report"):
        return "reviewing"
    if state.get("facts") or state.get("data_points"):
        return "analyzing"
    if state.get("outline"):
        return "researching"
    return "planning"


def ensure_resume_cursor(state: Dict[str, Any]) -> Dict[str, Any]:
    """返回规范化游标，并写回 ``state``。"""
    raw = state.get("resume_cursor")
    if isinstance(raw, dict) and raw.get("next_stage") in STAGES:
        cursor = {
            "version": CURSOR_VERSION,
            "next_stage": raw["next_stage"],
            "completed_stages": list(raw.get("completed_stages") or []),
            "attempts": dict(raw.get("attempts") or {}),
            "exact": bool(raw.get("exact", True)),
        }
    else:
        cursor = {
            "version": CURSOR_VERSION,
            "next_stage": _legacy_next_stage(state),
            "completed_stages": [],
            "attempts": {},
            "exact": False,
        }
    state["resume_cursor"] = cursor
    return cursor


def begin_stage(state: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """标记一次阶段尝试；该信息会随下一个提交点持久化。"""
    cursor = ensure_resume_cursor(state)
    if stage not in STAGES or stage == "completed":
        raise ValueError(f"invalid resumable stage: {stage}")
    if cursor["next_stage"] != stage:
        raise ValueError(
            f"resume cursor expects {cursor['next_stage']}, cannot start {stage}"
        )
    attempts = cursor["attempts"]
    attempts[stage] = int(attempts.get(stage, 0)) + 1
    return cursor


def commit_stage(
    state: Dict[str, Any], completed_stage: str, next_stage: str
) -> Dict[str, Any]:
    """原子检查点写入前推进游标。"""
    if completed_stage not in STAGES or completed_stage == "completed":
        raise ValueError(f"invalid completed stage: {completed_stage}")
    if next_stage not in STAGES:
        raise ValueError(f"invalid next stage: {next_stage}")
    cursor = ensure_resume_cursor(state)
    if cursor["next_stage"] != completed_stage:
        raise ValueError(
            f"resume cursor expects {cursor['next_stage']}, cannot commit {completed_stage}"
        )
    cursor["completed_stages"].append(
        {
            "stage": completed_stage,
            "iteration": int(state.get("iteration", 0) or 0),
        }
    )
    cursor["next_stage"] = next_stage
    return cursor
