# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""研究检查点模型"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    Integer,
    Float,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from core.database import Base


class ResearchCheckpoint(Base):
    """研究检查点模型 - 用于保存和恢复深度研究状态"""
    __tablename__ = "research_checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(64), index=True, nullable=False)  # 研究会话 ID
    # Checkpoint 对应的事件流位置。恢复时先加载快照，再从该序号之后回放。
    run_id = Column(String(64), nullable=True, index=True)
    last_event_seq = Column(Integer, nullable=False, default=0)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    query = Column(Text, nullable=False)  # 原始查询
    phase = Column(String(32), nullable=False)  # planning/researching/analyzing/writing/reviewing/completed
    iteration = Column(Integer, default=0)  # 当前迭代次数
    state_json = Column(JSONB, nullable=False)  # 完整的 ResearchState（后端状态）
    ui_state_json = Column(JSONB)  # 前端 UI 状态（研究步骤、搜索结果、图表等）
    final_report = Column(Text)  # 最终报告内容
    status = Column(String(16), default="running")  # running/paused/completed/failed
    error_message = Column(Text)  # 错误信息（如果失败）

    # ---- 质量评估字段 ----
    # CriticMaster 产出的评分此前只在 SSE 事件里出现一次就丢了，
    # 落库后才能做跨会话的聚合分析和回归对比。
    quality_score = Column(Float, comment="CriticMaster 三维均分 0-100")
    quality_score_status = Column(
        String(48),
        comment="评分来源: valid/valid_after_retry/fallback_invalid_model_output",
    )
    unresolved_issues = Column(Integer, comment="未解决的 critical/major 问题数")
    critic_issues = Column(JSONB, comment="CriticMaster 的完整问题列表")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("run_id", name="uq_research_checkpoints_run_id"),
        Index("ix_research_checkpoints_session_updated", "session_id", "updated_at"),
    )

    # 关系
    user = relationship("User", backref="research_checkpoints")

    def to_dict(self, include_state: bool = False):
        """转换为字典"""
        # 历史快照的阶段游标只做只读展示。新研究的执行位置由 LangGraph
        # 检查点负责，不再从这里推断。
        state = self.state_json or {}
        cursor = state.get("resume_cursor") or {}
        result = {
            "id": str(self.id),
            "session_id": self.session_id,
            "run_id": self.run_id,
            "last_event_seq": self.last_event_seq or 0,
            "user_id": str(self.user_id) if self.user_id else None,
            "query": self.query,
            "phase": self.phase,
            "iteration": self.iteration,
            "status": self.status,
            "error_message": self.error_message,
            "final_report": self.final_report,
            "quality_score": self.quality_score,
            "quality_score_status": self.quality_score_status,
            "unresolved_issues": self.unresolved_issues,
            "legacy_snapshot": True,
            "resume_stage": (
                cursor.get("next_stage")
            ),
            "checkpoint_version": (
                cursor.get("version")
            ),
            "exact_resume": (
                cursor.get("exact", False)
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_state:
            result["state_json"] = self.state_json
            result["ui_state_json"] = self.ui_state_json
        return result


class AgentEvent(Base):
    """一次研究运行的追加式事件账本。"""

    __tablename__ = "agent_events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(64), nullable=False)
    run_id = Column(String(64), nullable=False)
    turn_id = Column(UUID(as_uuid=True), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    seq = Column(Integer, nullable=False)
    event_type = Column(String(64), nullable=False)
    payload = Column(JSONB, nullable=False, default=dict)
    occurred_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_agent_events_run_seq"),
        Index("ix_agent_events_session_created", "session_id", "created_at"),
        Index("ix_agent_events_run_seq", "run_id", "seq"),
        Index("ix_agent_events_user_run", "user_id", "run_id"),
    )

    def to_dict(self):
        return {
            "event_id": str(self.event_id),
            "session_id": self.session_id,
            "run_id": self.run_id,
            "turn_id": str(self.turn_id) if self.turn_id else None,
            "seq": self.seq,
            "type": self.event_type,
            "data": self.payload or {},
            "timestamp": self.occurred_at.isoformat() if self.occurred_at else None,
        }


class ResearchRun(Base):
    """研究运行的单一执行权、fencing token 与原子事件序号。"""

    __tablename__ = "research_runs"

    run_id = Column(String(64), primary_key=True)
    session_id = Column(String(64), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    next_event_seq = Column(Integer, nullable=False, default=0)
    owner_id = Column(String(64), nullable=True)
    lease_until = Column(DateTime, nullable=True)
    fencing_token = Column(Integer, nullable=False, default=0)
    status = Column(String(16), nullable=False, default="idle")
    # ---- LangGraph 主链路接管执行后的轻量投影字段 ----
    # 完整业务状态保存在 LangGraph 检查点里；这里只留定位与对账所需的少量信息，
    # 不再复制一份状态索引表。
    graph_version = Column(String(32), nullable=True, comment="执行该运行的图版本")
    final_checkpoint_id = Column(
        String(64), nullable=True, comment="已完成幂等交付的终态 checkpoint",
    )
    last_projected_checkpoint_id = Column(
        String(64), nullable=True, comment="轻量展示状态已投影到的 checkpoint",
    )
    delivered_event_id = Column(
        UUID(as_uuid=True), nullable=True, comment="终态交付产生的完成事件",
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_research_runs_user_status", "user_id", "status"),
        Index("ix_research_runs_lease_until", "lease_until"),
        # 后台对账扫描"未终结或投影落后"的运行时按这两列筛选。
        Index(
            "ix_research_runs_projection",
            "status",
            "last_projected_checkpoint_id",
        ),
    )
