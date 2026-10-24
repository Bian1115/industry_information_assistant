"""历史疾病研究记录的只读查询。

独立医疗执行链路下线后，这里只负责按本人归属读出历史快照供展示与导出；
人工纳排与证据纠正的写入路径已随专属入口一并移除。
"""

from uuid import UUID

from models.research import ResearchCheckpoint
from .outputs import public_medical_bundle


def owned_medical_checkpoint(db, run_id: str, user_id: str):
    """按 run_id + 归属读出历史医学快照；不是本人的记录一律返回 None。"""
    try:
        owner = UUID(str(user_id))
    except (ValueError, TypeError, AttributeError):
        return None
    row = (
        db.query(ResearchCheckpoint)
        .filter(
            ResearchCheckpoint.run_id == str(run_id),
            ResearchCheckpoint.user_id == owner,
        )
        .one_or_none()
    )
    if row is None or (row.state_json or {}).get("research_type") != "medical_literature":
        return None
    return row


def medical_checkpoint_view(row) -> dict:
    state = row.state_json or {}
    return {
        **public_medical_bundle(state.get("medical_research") or {}),
        "run_id": row.run_id, "session_id": row.session_id,
        "final_report": row.final_report or state.get("final_report", ""),
        "run_status": row.status,
    }
