"""会话作用域的临时证据；正文版本不变，关闭后的清理任务独立保留。"""
from sqlalchemy import Column, String, Text, Integer, ForeignKey, Index, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from core.database import Base
from core.time_utils import UTCDateTime, utc_now

DATA = JSON().with_variant(JSONB(), "postgresql")


class SessionEvidenceSpace(Base):
    __tablename__ = "session_evidence_spaces"
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), primary_key=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="active")
    generation = Column(Integer, nullable=False, default=1)
    revision = Column(Integer, nullable=False, default=0)
    ttl_mode = Column(String(16), nullable=False, default="idle")
    ttl_seconds = Column(Integer, nullable=False, default=86400)
    active_topic_id = Column(String(64), nullable=True)
    expires_at = Column(UTCDateTime(), nullable=False, index=True)
    created_at = Column(UTCDateTime(), nullable=False, default=utc_now)


class SessionEvidenceDocument(Base):
    __tablename__ = "session_evidence_documents"
    id = Column(String(64), primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("session_evidence_spaces.session_id", ondelete="CASCADE"), nullable=False)
    topic_id = Column(String(64), nullable=False)
    document_id = Column(String(64), nullable=False)
    payload = Column(DATA, nullable=False)
    __table_args__ = (Index("ix_session_evidence_documents_scope", "session_id", "topic_id", "document_id"),)


class SessionEvidenceChunk(Base):
    __tablename__ = "session_evidence_chunks"
    id = Column(String(64), primary_key=True)
    document_key = Column(String(64), ForeignKey("session_evidence_documents.id", ondelete="CASCADE"), nullable=False)
    unit_type = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    locator = Column(DATA, nullable=False)
    index_status = Column(String(16), nullable=False, default="pending")
    __table_args__ = (Index("ix_session_evidence_chunks_document", "document_key", "unit_type"),)


class SessionEvidencePurge(Base):
    __tablename__ = "session_evidence_purges"
    session_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), nullable=False)
    run_ids = Column(DATA, nullable=False, default=list)
    status = Column(String(20), nullable=False, default="pending")
    attempts = Column(Integer, nullable=False, default=0)
    last_error = Column(String(160), nullable=True)
    created_at = Column(UTCDateTime(), nullable=False, default=utc_now)
    completed_at = Column(UTCDateTime(), nullable=True)
