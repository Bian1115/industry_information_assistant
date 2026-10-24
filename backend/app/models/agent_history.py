"""智能体独立工作历史；原文追加保存，摘要只更新模型视图。"""

import uuid

from sqlalchemy import Column, String, Text, DateTime, BigInteger, Integer, ForeignKey, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID, JSONB

from core.database import Base


class AgentContext(Base):
    __tablename__ = "agent_contexts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 新上下文按run隔离；旧会话共享上下文保留用于历史审计。
    run_id = Column(String(64), ForeignKey("research_runs.run_id", ondelete="SET NULL"), nullable=True)
    history_scope_id = Column(String(128), nullable=False)
    agent_name = Column(String(64), nullable=False)
    context_key = Column(String(128), nullable=False, default="main")
    next_seq = Column(BigInteger, nullable=False, default=0, server_default="0")
    summary = Column(Text, nullable=False, default="", server_default="")
    summary_version = Column(Integer, nullable=False, default=0, server_default="0")
    covered_seq = Column(BigInteger, nullable=False, default=0, server_default="0")
    summary_metadata = Column(JSONB, nullable=False, default=dict, server_default="{}")
    offloaded_source_ids = Column(JSONB, nullable=False, default=list, server_default="[]")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (UniqueConstraint("history_scope_id", "agent_name", "context_key", name="uq_agent_context_scope"),)


class AgentMessage(Base):
    __tablename__ = "agent_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    context_id = Column(UUID(as_uuid=True), ForeignKey("agent_contexts.id", ondelete="CASCADE"), nullable=False)
    # 不随共享context的创建run变化；用于逐run审计和导出。
    run_id = Column(String(64), nullable=False, index=True)
    seq = Column(BigInteger, nullable=False)
    kind = Column(String(32), nullable=False)
    call_id = Column(UUID(as_uuid=True), nullable=False)
    invocation_id = Column(UUID(as_uuid=True), nullable=False)
    node = Column(String(64), nullable=False)
    payload = Column(JSONB, nullable=False)
    content_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (UniqueConstraint("context_id", "seq", name="uq_agent_message_seq"),)
