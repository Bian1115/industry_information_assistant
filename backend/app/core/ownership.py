"""HTTP 私有资源的归属查询；资源 ID 只定位，登录用户决定访问范围。"""

from uuid import UUID

from fastapi import HTTPException

from core.time_utils import utc_now
from models.chat import ChatAttachment, ChatSession
from models.knowledge import Document, KnowledgeBase


def resource_uuid(value, label="资源"):
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(400, f"无效的{label}ID格式") from exc


def require_owned_session(db, session_id, user_id):
    session = db.query(ChatSession).filter(
        ChatSession.id == resource_uuid(session_id, "会话"),
        ChatSession.user_id == resource_uuid(user_id, "用户"),
        ChatSession.expires_at > utc_now(),
    ).first()
    if session is None:
        raise HTTPException(404, "会话不存在")
    return session


def owned_attachments(db, user_id):
    owner = resource_uuid(user_id, "用户")
    # 两层归属同时成立；历史匿名附件或错误绑定的附件不自动继承会话权限。
    return db.query(ChatAttachment).join(ChatSession).filter(
        ChatAttachment.user_id == owner,
        ChatSession.user_id == owner,
        ChatSession.expires_at > utc_now(),
    )


def require_owned_attachment(db, attachment_id, user_id, *, session_id=None):
    query = owned_attachments(db, user_id).filter(
        ChatAttachment.id == resource_uuid(attachment_id, "附件"),
    )
    if session_id is not None:
        query = query.filter(ChatAttachment.session_id == resource_uuid(session_id, "会话"))
    attachment = query.first()
    if attachment is None:
        raise HTTPException(404, "附件不存在")
    return attachment


def require_owned_knowledge_base(db, kb_id, user_id):
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == resource_uuid(kb_id, "知识库"),
        KnowledgeBase.user_id == resource_uuid(user_id, "用户"),
    ).first()
    if kb is None:
        raise HTTPException(404, "知识库不存在")
    return kb


def owned_documents(db, kb_id, user_id):
    owner = resource_uuid(user_id, "用户")
    return db.query(Document).join(KnowledgeBase).filter(
        Document.knowledge_base_id == resource_uuid(kb_id, "知识库"),
        Document.user_id == owner,
        KnowledgeBase.user_id == owner,
    )
