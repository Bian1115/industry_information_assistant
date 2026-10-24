# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""旧文档 API 的知识库适配层：必须登录并显式选择本人的知识库。"""

import asyncio
import hashlib
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from core.database import get_db
from core.ownership import owned_documents, require_owned_knowledge_base, resource_uuid
from models.knowledge import Document
from models.user import User
from router.auth_router import get_current_user_required
from router import knowledge_router
from schemas.document import (
    DeleteDocumentsRequest, RetrieveDocumentsRequest, UploadDocumentResponse,
    DocumentListResponse, DeleteDocumentsResponse,
)
from service.retrieval_service import hybrid_retrieve, rerank_knowledge_candidates

router = APIRouter(prefix="/documents", tags=["documents"])


def selected_documents(db, kb_id, user_id, document_ids):
    """所有指定文档必须属于目标知识库与当前用户，批量操作先整体授权。"""
    ids = {resource_uuid(value, "文档") for value in document_ids}
    docs = owned_documents(db, kb_id, user_id).filter(Document.id.in_(ids)).all()
    if len(docs) != len(ids):
        raise HTTPException(404, "文档不存在")
    return docs


@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    kb_id: UUID = Query(..., description="当前用户的知识库ID"),
    file: UploadFile = File(...),
    index_name: Optional[str] = Query(None, deprecated=True),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    if index_name is not None:
        raise HTTPException(400, "不再支持指定 index_name，请使用 kb_id")
    uploaded = await knowledge_router.upload_document(
        kb_id=str(kb_id), background_tasks=background_tasks,
        file=file, current_user=current_user, db=db,
    )
    return UploadDocumentResponse(
        status=uploaded.status, message=uploaded.message,
        document_id=uploaded.id, document=uploaded.model_dump(),
    )


@router.get("/list", response_model=DocumentListResponse)
async def get_document_list(
    kb_id: UUID = Query(..., description="当前用户的知识库ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    keywords: Optional[str] = Query(None),
    document_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    kb = require_owned_knowledge_base(db, kb_id, current_user.id)
    query = owned_documents(db, kb.id, current_user.id)
    if document_id is not None:
        docs = selected_documents(db, kb.id, current_user.id, [document_id])
        query = query.filter(Document.id == docs[0].id)
    if keywords:
        query = query.filter(Document.filename.contains(keywords, autoescape=True))
    total = query.count()
    docs = query.order_by(Document.created_at.desc(), Document.id).offset((page - 1) * page_size).limit(page_size).all()
    return {"code": 0, "data": {"docs": [
        {**knowledge_router.doc_to_response(doc).model_dump(mode="json"),
         "name": doc.filename, "type": doc.file_type, "size": doc.file_size}
        for doc in docs
    ], "total": total}}


@router.post("/delete", response_model=DeleteDocumentsResponse)
async def delete_documents(
    request: DeleteDocumentsRequest,
    kb_id: UUID = Query(..., description="当前用户的知识库ID"),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    kb = require_owned_knowledge_base(db, kb_id, current_user.id)
    if not request.document_ids:
        raise HTTPException(400, "document_ids 不能为空")
    docs = selected_documents(db, kb.id, current_user.id, request.document_ids)
    # 整批校验结束后才开始删除，避免混入其他用户 ID 时先删除本人的部分数据。
    ids = [str(doc.id) for doc in docs]
    for doc_id in ids:
        await knowledge_router.delete_document(str(kb.id), doc_id, current_user, db)
    return {"code": 0, "message": "文档已删除", "data": {"document_ids": ids}}


@router.post("/retrieve")
async def retrieve_documents(
    request: RetrieveDocumentsRequest,
    kb_id: UUID = Query(..., description="当前用户的知识库ID"),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    kb = require_owned_knowledge_base(db, kb_id, current_user.id)
    if request.document_ids is None:
        docs = owned_documents(db, kb.id, current_user.id).all()
    else:
        docs = selected_documents(db, kb.id, current_user.id, request.document_ids)
    if not docs:
        return {"code": 0, "data": {"chunks": [], "total": 0}}
    # 现有 DocMind 切片使用 md5(文件名) 作为 doc_id；只返回已授权本地文档的切片。
    allowed = {hashlib.md5(doc.filename.encode()).hexdigest() for doc in docs}
    hits = await asyncio.to_thread(hybrid_retrieve, str(kb.id), request.question, allow_benchmark=False)
    chunks = [hit for hit in hits if hit.get("document_id") in allowed]
    chunks = await rerank_knowledge_candidates(request.question, chunks)
    return {"code": 0, "data": {"chunks": chunks, "total": len(chunks)}}
