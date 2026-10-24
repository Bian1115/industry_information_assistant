# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""长期记忆管理路由。"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.database import get_db
from models.chat import ChatSession, LongTermMemory, MemoryJob
from models.user import User
from router.auth_router import get_current_user_required
from service.memory_contracts import MemoryScope, MemorySearchOptions
from service.memory_service import (
    enqueue_delete_memory,
    enqueue_memory_if_due,
    get_memory_provider,
    get_memory_service,
)

router = APIRouter(prefix="/memories", tags=["长期记忆"])


class MemoryResponse(BaseModel):
    id: str
    session_id: Optional[str]
    summary: str
    key_insights: Optional[dict]
    token_count: Optional[int]
    provider: str
    index_status: str
    created_at: datetime


class MemoryListResponse(BaseModel):
    memories: List[MemoryResponse] = Field(default_factory=list)
    total: int = 0


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)


class MemorySearchResult(BaseModel):
    id: str
    fragment_id: str
    session_id: Optional[str]
    memory_type: str
    content: str
    score: float


class CreateMemoryRequest(BaseModel):
    session_id: UUID


class MemoryOperationResponse(BaseModel):
    operation_id: str
    status: str


class MemoryJobResponse(BaseModel):
    operation_id: str
    operation_type: str
    status: str
    attempts: int
    memory_id: Optional[str]
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime


def _memory_response(memory: LongTermMemory) -> MemoryResponse:
    return MemoryResponse(
        id=str(memory.id),
        session_id=str(memory.session_id) if memory.session_id else None,
        summary=memory.summary,
        key_insights=memory.key_insights,
        token_count=memory.token_count,
        provider=memory.provider,
        index_status=memory.index_status,
        created_at=memory.created_at,
    )


@router.get("", response_model=MemoryListResponse)
async def get_memories(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    query = db.query(LongTermMemory).filter(
        LongTermMemory.user_id == current_user.id,
        LongTermMemory.index_status == "ready",
        LongTermMemory.deleted_at.is_(None),
    )
    memories = query.order_by(LongTermMemory.created_at.desc()).offset(offset).limit(limit).all()
    return MemoryListResponse(
        memories=[_memory_response(memory) for memory in memories],
        total=query.count(),
    )


@router.post("/search", response_model=List[MemorySearchResult])
async def search_memories(
    request: MemorySearchRequest,
    current_user: User = Depends(get_current_user_required),
):
    hits = get_memory_provider().search(
        MemoryScope(user_id=str(current_user.id)),
        request.query,
        MemorySearchOptions(limit=request.top_k),
    )
    return [MemorySearchResult(
        id=hit.memory_id,
        fragment_id=hit.fragment_id,
        session_id=hit.session_id,
        memory_type=hit.memory_type,
        content=hit.content,
        score=hit.score,
    ) for hit in hits]


@router.post(
    "/create",
    response_model=MemoryOperationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_memory_from_session(
    request: CreateMemoryRequest,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    session = db.query(ChatSession).filter(
        ChatSession.id == request.session_id,
        ChatSession.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    job = enqueue_memory_if_due(
        db,
        user_id=str(current_user.id),
        session_id=str(request.session_id),
        force=True,
    )
    if not job:
        raise HTTPException(status_code=400, detail="会话没有尚未生成记忆的消息")
    db.commit()
    db.refresh(job)
    return MemoryOperationResponse(operation_id=str(job.id), status=job.status)


@router.get("/jobs/{operation_id}", response_model=MemoryJobResponse)
async def get_memory_job(
    operation_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    job = db.query(MemoryJob).filter(
        MemoryJob.id == operation_id,
        MemoryJob.user_id == current_user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="记忆任务不存在")
    return MemoryJobResponse(
        operation_id=str(job.id),
        operation_type=job.operation_type,
        status=job.status,
        attempts=job.attempts,
        memory_id=str(job.memory_id) if job.memory_id else None,
        last_error=job.last_error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/context/{query}", response_model=dict)
async def get_memory_context(
    query: str,
    current_user: User = Depends(get_current_user_required),
):
    return {"context": get_memory_service().build_memory_context(
        user_id=str(current_user.id),
        current_query=query,
        max_memories=3,
    )}


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    memory = db.query(LongTermMemory).filter(
        LongTermMemory.id == memory_id,
        LongTermMemory.user_id == current_user.id,
        LongTermMemory.index_status == "ready",
        LongTermMemory.deleted_at.is_(None),
    ).first()
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return _memory_response(memory)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    job = enqueue_delete_memory(db, str(current_user.id), str(memory_id))
    if not job:
        raise HTTPException(status_code=404, detail="记忆不存在")
    db.commit()
    return None
