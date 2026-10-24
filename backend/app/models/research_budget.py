"""独立于图节点提交的研究消耗；不保存提示词或证据正文。"""
from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from core.database import Base


class ResearchBudget(Base):
    __tablename__ = "research_budgets"
    run_id = Column(String(64), ForeignKey("research_runs.run_id", ondelete="CASCADE"), primary_key=True)
    policy = Column(JSONB, nullable=False)
    baseline = Column(JSONB, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ResearchBudgetAttempt(Base):
    __tablename__ = "research_budget_attempts"
    attempt_id = Column(String(36), primary_key=True)
    run_id = Column(String(64), ForeignKey("research_budgets.run_id", ondelete="CASCADE"), nullable=False)
    owner_id = Column(String(64), nullable=False)
    fencing_token = Column(Integer, nullable=False)
    node = Column(String(32), nullable=False)
    phase = Column(String(32), nullable=False)
    agent = Column(String(64), nullable=False)
    started_at = Column(DateTime, nullable=False)
    lease_until = Column(DateTime, nullable=False)
    elapsed_seconds = Column(Float, nullable=False, default=0)
    status = Column(String(16), nullable=False)
    fallback_tokens = Column(Integer, nullable=False, default=0)
    fallback_calls = Column(Integer, nullable=False, default=0)
    __table_args__ = (Index("ix_research_budget_attempts_run", "run_id"),)


class ResearchBudgetCall(Base):
    __tablename__ = "research_budget_calls"
    call_id = Column(String(36), primary_key=True)
    attempt_id = Column(String(36), ForeignKey("research_budget_attempts.attempt_id", ondelete="CASCADE"), nullable=False)
    model = Column(String(128), nullable=False)
    reserved_tokens = Column(Integer, nullable=False)
    actual_tokens = Column(Integer, nullable=True)
    status = Column(String(16), nullable=False)
    __table_args__ = (Index("ix_research_budget_calls_attempt", "attempt_id"),)
