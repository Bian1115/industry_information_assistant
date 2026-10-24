# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""聊天路由：PostgreSQL 是会话与消息的唯一数据源。"""

import asyncio
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.status import HTTP_200_OK

from core.database import SessionLocal, get_db
from core.ownership import owned_documents, require_owned_attachment
from models.chat import ChatMessage
from models.knowledge import KnowledgeBase
from models.user import User
from router.auth_router import get_current_user_required
from schemas import ChatRequest, ChatWithAttachmentsRequest
from service import ChatService, DocumentService, ServiceConfig, WebSearchService
from service.conversation_service import (
    ConversationNotFoundError,
    get_conversation_service,
)
from service.retrieval_service import hybrid_retrieve

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def get_services() -> Dict[str, Any]:
    config = ServiceConfig.get_api_config()
    document_service = DocumentService(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )
    web_service = WebSearchService(api_key=config.get("bochaai_api_key"))
    return {
        "chat_service": ChatService(document_service, web_service),
        "default_dataset_id": config["default_dataset_id"],
    }


def retrieve_from_user_knowledge_bases(
    db: Session,
    user: User,
    question: str,
    kb_ids: Optional[List[str]] = None,
    top_k_per_kb: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user.id)
    if kb_ids:
        try:
            query = query.filter(KnowledgeBase.id.in_([UUID(value) for value in kb_ids]))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="kb_ids 包含无效 UUID") from exc

    documents: List[Dict[str, Any]] = []
    for knowledge_base in query.all():
        # 私有集合内的遗留或错误归属切片也不能进入用户上下文。
        allowed_ids = {
            hashlib.md5(document.filename.encode()).hexdigest()
            for document in owned_documents(db, knowledge_base.id, user.id).all()
        }
        if not allowed_ids:
            continue
        try:
            hits = hybrid_retrieve(
                kb_id=str(knowledge_base.id),
                question=question,
                top_k=top_k_per_kb,
                allow_benchmark=False,
            )
        except Exception as exc:
            logger.error("检索知识库 '%s' 失败: %s", knowledge_base.name, exc)
            continue

        for item in hits:
            if item.get("document_id") not in allowed_ids:
                continue
            documents.append({
                "id": len(documents) + 1,
                "content": item["content_with_weight"],
                "content_with_weight": item["content_with_weight"],
                "source": "knowledge",
                "title": item.get("document_name"),
                "document_id": item.get("document_id"),
                "document_name": item.get("document_name"),
                "kb_name": knowledge_base.name,
                "chunk_id": item.get("chunk_id"),
                "rrf_score": item.get("rrf_score", 0.0),
                "weight": item.get("rrf_score", item.get("score", 0.0)),
            })
    return documents


def _prepare_turn(
    db: Session,
    current_user: User,
    session_id: str,
    turn_id: str,
    question: str,
) -> tuple[List[Dict[str, str]], Optional[ChatMessage]]:
    conversations = get_conversation_service()
    try:
        session = conversations.require_session(db, session_id, str(current_user.id))
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    user_message = conversations.save_user_message(db, session, turn_id, question)
    if user_message.content != question:
        raise HTTPException(status_code=409, detail="turn_id 已用于另一条用户消息")

    completed = conversations.get_turn_message(db, session_id, turn_id, "assistant")
    if completed:
        return [], completed
    history = conversations.get_prompt_history(
        db,
        session_id,
        user_id=str(current_user.id),
        exclude_turn_id=turn_id,
        current_query=question,
    )
    return history, completed


def _replay_chat_message(message: ChatMessage):
    payload = {
        "role": "assistant",
        "content": message.content,
        "thinking": False,
        "replayed": True,
    }
    yield f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    if message.thinking:
        thinking_payload = {
            "role": "assistant",
            "content": message.thinking,
            "thinking": True,
            "replayed": True,
        }
        yield f"event: message\ndata: {json.dumps(thinking_payload, ensure_ascii=False)}\n\n"
    references = (message.references_data or {}).get("references", [])
    yield f"event: message\ndata: {json.dumps({'documents': references}, ensure_ascii=False)}\n\n"
    yield "event: end\ndata: [DONE]\n\n"


def _completion_response(
    *,
    chat_service: ChatService,
    current_user: User,
    session_id: str,
    turn_id: str,
    question_for_model: str,
    history: List[Dict[str, str]],
    documents: List[Dict[str, Any]],
) -> StreamingResponse:
    user_id = str(current_user.id)

    def persist(answer: str, thinking: str) -> None:
        db = SessionLocal()
        try:
            get_conversation_service().save_assistant_message(
                db,
                session_id=session_id,
                user_id=user_id,
                turn_id=turn_id,
                content=answer,
                thinking=thinking or None,
                references_data={"references": documents} if documents else None,
            )
        finally:
            db.close()

    async def generate_response():
        for chunk in chat_service.get_chat_completion(
            question=question_for_model,
            retrieved_content=documents,
            history_messages=history,
            user_id=user_id,
            on_complete=persist,
        ):
            yield chunk

    return StreamingResponse(generate_response(), media_type="text/event-stream")


@router.post("/completion", status_code=HTTP_200_OK)
async def chat_completion(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_required),
    services: Dict[str, Any] = Depends(get_services),
):
    session_id = str(request.session_id)
    turn_id = str(request.turn_id)
    history, completed = _prepare_turn(
        db,
        current_user,
        session_id,
        turn_id,
        request.question,
    )
    if completed:
        return StreamingResponse(_replay_chat_message(completed), media_type="text/event-stream")

    chat_service: ChatService = services["chat_service"]
    knowledge_documents = (
        retrieve_from_user_knowledge_bases(
            db,
            current_user,
            request.question,
            request.kb_ids,
        )
        if request.search_knowledge
        else []
    )
    web_documents = (
        chat_service.retrieve_from_web(request.question) if request.search_web else []
    )
    documents = await asyncio.to_thread(chat_service.rerank_documents,
        request.question,
        knowledge_documents + web_documents,
    )
    return _completion_response(
        chat_service=chat_service,
        current_user=current_user,
        session_id=session_id,
        turn_id=turn_id,
        question_for_model=request.question,
        history=history,
        documents=documents,
    )


@router.post("/completion/v3", status_code=HTTP_200_OK)
async def chat_completion_with_attachments(
    request: ChatWithAttachmentsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_required),
    services: Dict[str, Any] = Depends(get_services),
):
    session_id = str(request.session_id)
    turn_id = str(request.turn_id)
    # 附件先整体授权，再持久化本轮消息；非法绑定不能污染会话或进入模型上下文。
    attachments = [
        require_owned_attachment(db, attachment_id, current_user.id, session_id=session_id)
        for attachment_id in request.attachment_ids or []
    ]
    history, completed = _prepare_turn(
        db,
        current_user,
        session_id,
        turn_id,
        request.question,
    )
    if completed:
        return StreamingResponse(_replay_chat_message(completed), media_type="text/event-stream")

    attachment_contents = []
    for attachment in attachments:
        if attachment and attachment.status == "completed" and attachment.content_text:
            attachment_contents.append({
                "filename": attachment.filename,
                "content": attachment.content_text[:10000],
            })

    chat_service: ChatService = services["chat_service"]
    knowledge_documents = (
        retrieve_from_user_knowledge_bases(
            db,
            current_user,
            request.question,
            request.kb_ids,
        )
        if request.search_knowledge
        else []
    )
    web_documents = (
        chat_service.retrieve_from_web(request.question) if request.search_web else []
    )
    documents = await asyncio.to_thread(chat_service.rerank_documents,
        request.question,
        knowledge_documents + web_documents,
    )

    question_for_model = request.question
    if attachment_contents:
        attachment_text = "\n\n".join(
            f"--- {item['filename']} ---\n{item['content']}" for item in attachment_contents
        )
        question_for_model = (
            f"用户问题：{request.question}\n\n"
            f"请结合以下上传的附件内容回答：\n{attachment_text}"
        )

    return _completion_response(
        chat_service=chat_service,
        current_user=current_user,
        session_id=session_id,
        turn_id=turn_id,
        question_for_model=question_for_model,
        history=history,
        documents=documents,
    )
