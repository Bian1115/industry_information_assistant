# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""聊天相关模型"""
import uuid
from sqlalchemy import (
    Column,
    String,
    Text,
    ForeignKey,
    Integer,
    BigInteger,
    Index,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from core.database import Base
from core.time_utils import UTCDateTime, session_expiry_time, utc_now


class ChatAttachment(Base):
    """聊天附件模型"""
    __tablename__ = "chat_attachments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, txt, image, etc.
    file_size = Column(BigInteger, nullable=False)
    file_path = Column(String(500), nullable=False)  # 文件存储路径
    content_text = Column(Text)  # 提取的文本内容（用于问答）
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text)  # 处理错误信息
    created_at = Column(UTCDateTime(), default=utc_now)
    updated_at = Column(UTCDateTime(), default=utc_now, onupdate=utc_now)

    # 关系
    message = relationship("ChatMessage", back_populates="attachments")
    session = relationship("ChatSession", back_populates="attachments")


class ChatSession(Base):
    """聊天会话模型"""
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    title = Column(String(255), default="新对话")
    session_type = Column(String(50), default="chat")  # chat, deepsearch
    # 摘要与标题更新不会续期；只有新消息写入会刷新此时间。
    expires_at = Column(UTCDateTime(), nullable=False, default=session_expiry_time, index=True)
    # LLM 语义摘要是可重建的会话派生状态；原始消息仍以 chat_messages 为准。
    semantic_context_summary = Column(JSONB)
    semantic_context_version = Column(Integer, nullable=False, default=0)
    semantic_context_last_message_id = Column(UUID(as_uuid=True), nullable=True)
    semantic_context_model = Column(String(64), nullable=True)
    semantic_context_status = Column(String(20), nullable=False, default="empty")
    semantic_context_last_error = Column(Text, nullable=True)
    semantic_context_updated_at = Column(UTCDateTime(), nullable=True)
    created_at = Column(UTCDateTime(), default=utc_now)
    updated_at = Column(UTCDateTime(), default=utc_now, onupdate=utc_now)

    # 关系
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    memories = relationship("LongTermMemory", back_populates="session")
    attachments = relationship("ChatAttachment", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """聊天消息模型"""
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    # 同一轮问答共用一个 turn_id。旧数据允许为空；新聊天请求必须提供。
    turn_id = Column(UUID(as_uuid=True), nullable=True)
    execution_mode = Column(String(16), nullable=False, default="chat")
    run_id = Column(String(64), nullable=True)
    research_status = Column(String(16), nullable=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    thinking = Column(Text)  # 思考过程
    references_data = Column(JSONB)  # 引用的文档
    image_results = Column(JSONB)  # 图片搜索结果
    created_at = Column(UTCDateTime(), default=utc_now)

    # 关系
    session = relationship("ChatSession", back_populates="messages")
    attachments = relationship("ChatAttachment", back_populates="message")

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "turn_id",
            "role",
            name="uq_chat_messages_session_turn_role",
        ),
        Index("ix_chat_messages_session_created", "session_id", "created_at"),
        Index("ix_chat_messages_run_id", "run_id"),
    )


class LongTermMemory(Base):
    """长期记忆模型"""
    __tablename__ = "long_term_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True)
    summary = Column(Text, nullable=False)  # 记忆摘要
    key_insights = Column(JSONB)  # 关键洞察
    milvus_ids = Column(ARRAY(Text))  # Milvus 中的向量 ID
    token_count = Column(Integer)
    provider = Column(String(32), nullable=False, default="native")
    source_start_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "chat_messages.id",
            name="fk_long_term_memories_source_start_message",
            ondelete="SET NULL",
        ),
        nullable=True,
    )
    source_end_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "chat_messages.id",
            name="fk_long_term_memories_source_end_message",
            ondelete="SET NULL",
        ),
        nullable=True,
    )
    index_status = Column(String(20), nullable=False, default="indexing")
    last_error = Column(Text)
    deleted_at = Column(UTCDateTime())
    created_at = Column(UTCDateTime(), default=utc_now)
    updated_at = Column(UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now)

    # 关系
    user = relationship("User", back_populates="memories")
    session = relationship("ChatSession", back_populates="memories")

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "source_end_message_id",
            name="uq_long_term_memories_session_source_end",
        ),
        Index("ix_long_term_memories_user_status", "user_id", "index_status"),
    )


class MemoryJob(Base):
    """持久化的记忆操作任务（PostgreSQL Outbox）。"""

    __tablename__ = "memory_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    operation_type = Column(String(20), nullable=False)  # add/update/delete/reset_user
    provider = Column(String(32), nullable=False, default="native")
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=True,
    )
    memory_id = Column(
        UUID(as_uuid=True),
        ForeignKey("long_term_memories.id", ondelete="SET NULL"),
        nullable=True,
    )
    payload = Column(JSONB, nullable=False, default=dict)
    idempotency_key = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    attempts = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(UTCDateTime(), nullable=False, default=utc_now)
    locked_at = Column(UTCDateTime())
    last_error = Column(Text)
    created_at = Column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at = Column(UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now)

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_memory_jobs_idempotency_key"),
        Index("ix_memory_jobs_status_next_attempt", "status", "next_attempt_at"),
        Index(
            "uq_memory_jobs_active_add_session",
            "session_id",
            unique=True,
            postgresql_where=text(
                "operation_type = 'add' AND status IN ('pending', 'running', 'retry')"
            ),
        ),
        Index(
            "uq_memory_jobs_active_semantic_context_session",
            "session_id",
            unique=True,
            postgresql_where=text(
                "operation_type = 'semantic_context' "
                "AND status IN ('pending', 'running', 'retry')"
            ),
        ),
    )
