"""清理已过期的 PostgreSQL 会话和其派生缓存。"""

import logging
from datetime import datetime
from pathlib import Path

from core.database import SessionLocal
from core.time_utils import as_utc, utc_now
from models.chat import ChatSession

logger = logging.getLogger(__name__)


def cleanup_expired_sessions(*, now: datetime | None = None, batch_size: int = 100) -> int:
    """每轮最多删除 100 个会话；消息、附件、Outbox 由现有关系级联清理。

    行锁避免多进程重复处理或与消息续期冲突。数据库提交成功后再删除缓存
    和附件文件；长期记忆沿用既有会话删除语义，仅解除会话关联。
    """
    cutoff = as_utc(now) if now is not None else utc_now()
    with SessionLocal() as db:
        sessions = (db.query(ChatSession)
                    .filter(ChatSession.expires_at <= cutoff)
                    .order_by(ChatSession.expires_at, ChatSession.id)
                    .limit(batch_size)
                    .with_for_update(skip_locked=True).all())
        removed = []
        for session in sessions:
            from service.session_evidence_lifecycle import close_evidence_in_transaction
            close_evidence_in_transaction(db, user_id=session.user_id, session_id=session.id)
            removed.append((str(session.user_id), str(session.id),
                            [attachment.file_path for attachment in session.attachments]))
            db.delete(session)
        db.commit()

    if removed:
        from service.session_memory_service import get_session_memory_service

        for user_id, session_id, files in removed:
            try:
                get_session_memory_service().invalidate(user_id=user_id, session_id=session_id)
            except Exception:
                logger.exception("过期会话 %s 的 Redis 缓存清理失败，等待缓存 TTL", session_id)
            for file_path in files:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except OSError:
                    logger.exception("过期会话 %s 的附件文件清理失败", session_id)
    return len(removed)
