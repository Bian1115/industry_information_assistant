# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""历史疾病文献研究成果的只读查询与导出。

独立医疗编排已随单主链路改造下线：这里不再创建新任务，也不再提供人工
纳排与证据纠正入口。历史报告不随执行链路一起删除，仍可按本人归属读取和
导出，直到历史数据迁移完成。
"""

from typing import Literal
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from core.database import get_db
from models.user import User
from router.auth_router import get_current_user_required
from service.medical_research.outputs import export_medical
from service.medical_research.storage import (
    medical_checkpoint_view,
    owned_medical_checkpoint,
)

router = APIRouter()


def require_result(db: Session, run_id, user_id):
    row = owned_medical_checkpoint(db, str(run_id), str(user_id))
    if row is None:
        raise HTTPException(status_code=404, detail="未找到本人的历史疾病研究记录")
    return row


@router.get("/{run_id}")
def get_medical_result(
    run_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """读取一次历史疾病文献研究的公开结果。"""
    view = medical_checkpoint_view(require_result(db, run_id, current_user.id))
    view["read_only"] = True
    view["legacy_pipeline"] = True
    return view


@router.get("/{run_id}/papers/{paper_id:path}")
def get_medical_paper(
    run_id: UUID,
    paper_id: str,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """读取该历史研究中的单篇文献记录。"""
    row = require_result(db, run_id, current_user.id)
    bundle = (row.state_json or {}).get("medical_research") or {}
    paper = next(
        (p for p in bundle.get("papers", []) if p.get("paper_id") == paper_id), None
    )
    if paper is None:
        raise HTTPException(status_code=404, detail="文献不属于本次研究")
    return paper


@router.get("/{run_id}/export")
def export_medical_result(
    run_id: UUID,
    format: Literal[
        "markdown", "csv", "bibtex", "ris", "json", "html", "docx"
    ] = Query("markdown"),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """导出历史疾病文献研究成果。"""
    row = require_result(db, run_id, current_user.id)
    bundle = (row.state_json or {}).get("medical_research") or {}
    content, media_type, filename = export_medical(
        bundle, row.final_report or "", format
    )
    return Response(content=content, media_type=media_type, headers={
        "Content-Disposition": "attachment; filename*=UTF-8''" + quote(filename),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, no-store",
    })
