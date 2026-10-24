# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""以 PostgreSQL 为唯一数据源的会话读写服务。"""

import logging
from copy import deepcopy
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.time_utils import session_expiry_time, utc_now
from models.chat import ChatMessage, ChatSession
from service.context_compaction import compact_messages
from service.qwen_token_counter import get_qwen_token_counter
from service.session_memory_service import get_session_memory_service


logger = logging.getLogger(__name__)


class ConversationNotFoundError(LookupError):
    pass


class ConversationService:
    def __init__(self, token_limit: int = 5000, max_messages: int = 20):
        self.token_limit = token_limit
        self.max_messages = max_messages
        self.token_counter = get_qwen_token_counter()
        self.session_memory = get_session_memory_service()

    def require_session(self, db: Session, session_id: str, user_id: str, *, for_update: bool = False) -> ChatSession:
        try:
            session_uuid = UUID(str(session_id))
            user_uuid = UUID(str(user_id))
        except ValueError as exc:
            raise ConversationNotFoundError("无效的会话ID格式") from exc

        query = db.query(ChatSession).filter(
            ChatSession.id == session_uuid,
            ChatSession.user_id == user_uuid,
            ChatSession.expires_at > utc_now(),
        )
        if for_update:
            # 与过期清理互斥，且重新读取调用方可能持有的旧 ORM 实例。
            query = query.populate_existing().with_for_update()
        session = query.first()
        if not session:
            raise ConversationNotFoundError("会话不存在、已过期或不属于当前用户")
        return session

    def get_turn_message(
        self,
        db: Session,
        session_id: str,
        turn_id: str,
        role: str,
    ) -> Optional[ChatMessage]:
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == UUID(str(session_id)),
            ChatMessage.turn_id == UUID(str(turn_id)),
            ChatMessage.role == role,
        ).first()

    def save_user_message(
        self,
        db: Session,
        session: ChatSession,
        turn_id: str,
        content: str,
        *,
        execution_mode: str = "chat",
        run_id: Optional[str] = None,
        research_status: Optional[str] = None,
    ) -> ChatMessage:
        session = self.require_session(db, str(session.id), str(session.user_id), for_update=True)
        existing = self.get_turn_message(db, str(session.id), turn_id, "user")
        if existing:
            return existing

        message = ChatMessage(
            session_id=session.id,
            turn_id=UUID(str(turn_id)),
            execution_mode=execution_mode,
            run_id=run_id,
            research_status=research_status,
            role="user",
            content=content,
        )
        db.add(message)
        session.updated_at = utc_now()
        session.expires_at = session_expiry_time(session.updated_at)
        if session.title == "新对话":
            session.title = content[:20] + ("..." if len(content) > 20 else "")

        try:
            from service.memory_service import enqueue_memory_if_due

            db.flush()
            # 重要用户信息随原文一起入队，不等待长回答或研究任务结束。
            enqueue_memory_if_due(
                db, user_id=str(session.user_id), session_id=str(session.id),
                urgent_only=True,
            )
            db.commit()
        except IntegrityError:
            # 并发重试可能同时插入；唯一约束决定最终只有一条。
            db.rollback()
            existing = self.get_turn_message(db, str(session.id), turn_id, "user")
            if existing:
                return existing
            raise

        db.refresh(message)
        return message

    def stage_assistant_message(
        self,
        db: Session,
        *,
        session_id: str,
        user_id: str,
        turn_id: str,
        content: str,
        thinking: Optional[str] = None,
        references_data: Optional[dict] = None,
        image_results: Optional[list] = None,
        execution_mode: str = "chat",
        run_id: Optional[str] = None,
        research_status: Optional[str] = None,
        replace_clarification: bool = False,
    ) -> ChatMessage:
        """把助手消息写进调用方的事务，但**不提交**。

        研究终态交付需要让报告、运行状态和完成事件在同一个事务里落库。
        自带 commit 的入口做不到这一点：依次调用几个各自提交的方法，
        只是看起来像原子，实际上中间崩溃就会留下半份交付。
        """
        session = self.require_session(db, session_id, user_id, for_update=True)
        existing = self.get_turn_message(db, session_id, turn_id, "assistant")
        replacing = bool(existing and replace_clarification and execution_mode == "research"
                         and research_status == "completed" and run_id
                         and existing.execution_mode == "chat"
                         and existing.research_status == "clarification" and existing.run_id is None)
        if existing and not replacing:
            return existing
        if replacing:
            user_message = self.get_turn_message(db, session_id, turn_id, "user")
            if not user_message or user_message.run_id != run_id or user_message.execution_mode != "research":
                raise ValueError("澄清消息没有关联本轮研究用户消息，拒绝替换交付")
            previous = {"id": str(existing.id), "content": existing.content,
                        "thinking": existing.thinking, "references_data": deepcopy(existing.references_data),
                        "execution_mode": existing.execution_mode, "run_id": existing.run_id,
                        "research_status": existing.research_status, "created_at": str(existing.created_at)}
            message = existing
            message.content, message.thinking = content, thinking
            message.references_data = {**(references_data or {}), "previous_clarification": previous}
            message.image_results = image_results
            message.execution_mode, message.run_id, message.research_status = execution_mode, run_id, research_status
        else:
            message = ChatMessage(
                session_id=session.id,
                turn_id=UUID(str(turn_id)),
                execution_mode=execution_mode,
                run_id=run_id,
                research_status=research_status,
                role="assistant",
                content=content,
                thinking=thinking,
                references_data=references_data,
                image_results=image_results,
            )
            db.add(message)
        session.updated_at = utc_now()
        session.expires_at = session_expiry_time(session.updated_at)
        db.flush()

        # 助手消息与 outbox 任务在同一事务提交，避免消息已保存但任务丢失。
        from service.memory_service import enqueue_memory_if_due
        from service.semantic_context_service import enqueue_semantic_context_if_due

        enqueue_memory_if_due(db, user_id=str(user_id), session_id=str(session_id))
        enqueue_semantic_context_if_due(
            db,
            user_id=str(user_id),
            session_id=str(session_id),
        )
        return message

    def save_assistant_message(
        self,
        db: Session,
        session_id: str,
        user_id: str,
        turn_id: str,
        content: str,
        *,
        thinking: Optional[str] = None,
        references_data: Optional[dict] = None,
        image_results: Optional[list] = None,
        execution_mode: str = "chat",
        run_id: Optional[str] = None,
        research_status: Optional[str] = None,
    ) -> ChatMessage:
        try:
            message = self.stage_assistant_message(
                db,
                session_id=session_id,
                user_id=user_id,
                turn_id=turn_id,
                content=content,
                thinking=thinking,
                references_data=references_data,
                image_results=image_results,
                execution_mode=execution_mode,
                run_id=run_id,
                research_status=research_status,
            )
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = self.get_turn_message(db, session_id, turn_id, "assistant")
            if existing:
                return existing
            raise

        db.refresh(message)
        # Redis 仅保存可重建的会话工作记忆；失败不会回滚已持久化的消息。
        self.session_memory.refresh(
            db,
            user_id=str(user_id),
            session_id=str(session_id),
        )
        return message

    def get_prompt_history(
        self,
        db: Session,
        session_id: str,
        *,
        user_id: str,
        exclude_turn_id: Optional[str] = None,
        current_query: str = "",
    ) -> List[Dict[str, str]]:
        # 即使 Redis 服务内部异常后走本地回退，也必须先锁定用户归属。
        self.require_session(db, session_id, user_id)
        try:
            return self.session_memory.get_context(
                db,
                user_id=str(user_id),
                session_id=str(session_id),
                exclude_turn_id=exclude_turn_id,
                token_limit=self.token_limit,
                max_messages=self.max_messages,
                current_query=current_query,
            ).messages
        except Exception as exc:
            from service.semantic_context_service import semantic_context_enabled

            if semantic_context_enabled():
                # 摘要失败不能退回旧的 120 条候选和 5000 token 裁剪；否则会
                # 在摘要覆盖游标没有推进时，静默丢掉尚未压缩的历史。
                raise
            # Redis 会话记忆是派生缓存；任何异常都退回原 PostgreSQL 热路径。
            logger.warning("读取 Redis 会话工作记忆失败，使用 PostgreSQL: %s", exc)

        query = db.query(ChatMessage).filter(
            ChatMessage.session_id == UUID(str(session_id)),
            ChatMessage.role.in_(["user", "assistant", "system"]),
        )
        if exclude_turn_id:
            turn_uuid = UUID(str(exclude_turn_id))
            query = query.filter(or_(
                ChatMessage.turn_id.is_(None),
                ChatMessage.turn_id != turn_uuid,
            ))

        # 多取候选，随后按完整对话轮次压缩；原消息始终保留在 PostgreSQL。
        messages = query.order_by(
            ChatMessage.created_at.desc(),
            ChatMessage.id.desc(),
        ).limit(max(self.max_messages * 6, self.max_messages)).all()
        chronological = [
            {"role": message.role, "content": message.content}
            for message in reversed(messages)
        ]
        return compact_messages(
            chronological,
            token_limit=self.token_limit,
            max_messages=self.max_messages,
            token_counter=self.token_counter,
        ).messages


_conversation_service: Optional[ConversationService] = None


def get_conversation_service() -> ConversationService:
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service
