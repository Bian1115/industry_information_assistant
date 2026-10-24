# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""深度研究接口：唯一 LangGraph 主链路的执行、恢复、事件与历史读取。

新任务统一进入同一张图，不再有 research_type 分流，也不再保留第二套执行
引擎。历史研究（旧完整快照）保留只读查询入口，不参与新运行的读写路径。
"""

import asyncio
import os
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator
from sqlalchemy.orm import Session
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_409_CONFLICT,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from core.database import SessionLocal, get_db
from models.chat import ChatMessage
from models.research import AgentEvent, ResearchRun
from models.user import User
from router.auth_router import get_current_user_required
from service.conversation_service import ConversationNotFoundError, get_conversation_service
from service.deep_research_v2.service import (
    DeepResearchV2Service,
    LegacyRunNotResumableError,
)
from service.deep_research_v2.source_scope import SourceScopeError
from service.deep_research_v2.citation_verification import CitationVerificationOptions
from service.deep_research_v2.progress import pending_progress
from service.research_context_service import get_research_context_service
from service.session_task_service import fallback_task, SessionTaskBudgetExceeded
from service.context_policy import AGENTS, AgentCompressionSettings
from service.research_control import (
    is_research_cancelled as is_run_cancelled,
    request_cancel,
)
from service.research_events import (
    ResearchEventPersistenceError,
    ResearchRunLeaseConflict,
    ResearchRunLeaseLost,
    acquire_research_run_lease,
    release_research_run_lease,
    replay_events,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResearchRouter")

router = APIRouter(prefix="/research", tags=["research"])


class ResearchRequest(BaseModel):
    """深度研究请求。默认全网；后端配置限定范围时请求只能在范围内收窄。"""

    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {
            "query": "整理阿尔茨海默病近期治疗研究进展，比较主要研究结果与局限。",
            "session_id": "0f1291cf-0f57-42c1-bff0-8d90a909488b",
            "turn_id": "8b2dd7e7-89b2-4cdd-8f14-6c2480f5a6c7",
        }
    })

    query: str = Field(min_length=1, max_length=4000)
    session_id: UUID
    turn_id: UUID
    allowed_domains: Optional[List[str]] = Field(default=None, max_length=50)
    citation_verification: Optional[CitationVerificationOptions] = Field(
        default=None, description="引用核验；未传默认从最终报告独立提取论断，检查遗漏引用、预测与事实混淆及来源支持性，Scout关。默认仅审计；report_failure_action可选audit/delete/rewrite_once，改写后失败仅记录，不中断交付；运行期间固定"
    )
    hypothesis_enabled: Optional[StrictBool] = Field(
        default=None, description="假设机制开关；未传使用后端默认值，运行期间固定"
    )
    agent_compression: Optional[Dict[str, AgentCompressionSettings]] = Field(
        default=None, description="按 Agent 独立设置历史引用化、外置与摘要；只覆盖显式传入字段，运行期间固定"
    )

    @field_validator("agent_compression")
    @classmethod
    def validate_agents(cls, value):
        if value is not None and set(value) - set(AGENTS):
            raise ValueError("逐 Agent 压缩配置包含未知 Agent")
        return value


def get_research_service(request: Request) -> DeepResearchV2Service:
    """使用应用生命周期持有的 LangGraph 检查点构造服务。"""
    checkpointer = getattr(request.app.state, "research_checkpointer", None)
    if checkpointer is None:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="研究检查点尚未就绪，请稍后重试",
        )
    return DeepResearchV2Service(checkpointer=checkpointer)


def _replay_research_message(message: ChatMessage) -> AsyncGenerator[str, None]:
    """同一 turn 已有交付结果时直接回放，不重复执行研究。"""

    async def generator():
        if message.research_status == "clarification":
            yield f"data: {json.dumps({'type': 'session_clarification', 'content': message.content, 'question': message.content, 'run_id': str(message.turn_id or '')}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        event = {
            "type": "research_complete",
            "final_report": message.content,
            "references": (message.references_data or {}).get("references", []),
            "replayed": True,
            "run_id": str(message.run_id or message.turn_id),
        }
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return generator()


async def _stream_with_lease(
    stream: AsyncGenerator[str, None],
    run_lease,
    *,
    on_failure,
) -> AsyncGenerator[str, None]:
    """统一收尾：消费者异常或断线时释放租约，不把未落账事件发出去。"""
    try:
        async for chunk in stream:
            yield chunk
    except (ResearchEventPersistenceError, ResearchRunLeaseLost):
        logger.exception("研究事件未可靠提交，终止流")
        await asyncio.to_thread(on_failure)
        raise
    finally:
        try:
            if hasattr(stream, "aclose"):
                await stream.aclose()
        finally:
            # service 正常收尾时已按真实终态释放；这里只兜底消费者侧异常，
            # 旧 token 下的重复释放不会覆盖已写入的终态。
            await asyncio.to_thread(release_research_run_lease, run_lease, "paused")


@router.post("/stream", status_code=HTTP_200_OK)
async def stream_research(
    request: ResearchRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """发起一次深度研究，按 SSE 流式返回过程与结果。"""
    session_id = str(request.session_id)
    turn_id = str(request.turn_id)
    user_id = str(current_user.id)
    service = get_research_service(http_request)

    conversations = get_conversation_service()
    try:
        session = conversations.require_session(db, session_id, user_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        allowed_domains = service.resolve_allowed_domains(request.allowed_domains)
    except (SourceScopeError, ValueError) as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    user_message = conversations.save_user_message(
        db, session, turn_id, request.query,
        execution_mode="research", run_id=turn_id, research_status="running",
    )
    if user_message.content != request.query:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="turn_id 已用于另一条用户消息")

    completed = conversations.get_turn_message(db, session_id, turn_id, "assistant")
    if completed:
        return StreamingResponse(
            _replay_research_message(completed), media_type="text/event-stream"
        )

    try:
        research_context = await asyncio.to_thread(
            get_research_context_service().build,
            user_id=user_id, session_id=session_id,
            current_turn_id=turn_id, current_query=request.query,
            resolve_query=True,
        )
    except SessionTaskBudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        from service.semantic_context_service import SemanticContextError, semantic_context_enabled
        if isinstance(exc, SemanticContextError) or semantic_context_enabled():
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        logger.warning("构建多轮上下文失败，使用原问题: %s", exc)
        research_context = {
            "original_query": request.query,
            "resolved_query": request.query,
            "conversation_context": [],
            "memory_context": [],
            "session_task": fallback_task(request.query, "会话上下文构建失败"),
        }

    task = research_context.get("session_task") or {}
    if task.get("status") == "clarify":
        question = task["clarification_question"]
        user_message.research_status = "clarification"
        user_message.execution_mode = "chat"
        user_message.run_id = None
        assistant = conversations.save_assistant_message(db, session_id, user_id, turn_id, question,
            execution_mode="chat", research_status="clarification",
            references_data={"session_task": task, "session_task_metrics": research_context.get("task_metrics", {})})
        return StreamingResponse(_replay_research_message(assistant), media_type="text/event-stream")

    try:
        run_lease = await asyncio.to_thread(
            acquire_research_run_lease,
            run_id=turn_id, session_id=session_id, user_id=user_id,
        )
    except ResearchRunLeaseConflict as exc:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail=str(exc)) from exc

    stream = service.research(
        query=request.query, session_id=session_id, user_id=user_id,
        run_id=turn_id, turn_id=turn_id,
        research_context=research_context,
        allowed_domains=request.allowed_domains,
        lease=run_lease,
        **({"citation_verification": request.citation_verification.model_dump()}
           if request.citation_verification is not None else {}),
        **({"hypothesis_enabled": request.hypothesis_enabled}
           if request.hypothesis_enabled is not None else {}),
        **({"agent_compression": {name: settings.model_dump(exclude_unset=True)
                                  for name, settings in request.agent_compression.items()}}
           if request.agent_compression is not None else {}),
    )
    return StreamingResponse(
        _stream_with_lease(
            stream, run_lease,
            on_failure=lambda: _mark_turn_status(session_id, turn_id, "failed"),
        ),
        media_type="text/event-stream",
    )


@router.post("/resume/{run_id}", status_code=HTTP_200_OK)
async def resume_research(
    run_id: UUID,
    http_request: Request,
    after_seq: int = Query(0, ge=0, description="客户端已应用的最后事件序号"),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """从图检查点继续一次技术中断的研究（仅限本人的运行）。"""
    run_id = str(run_id)
    user_id = str(current_user.id)
    service = get_research_service(http_request)
    run = _owned_run(db, run_id, user_id)

    state = await service.read_state(run_id)
    if (state.get("storage_reconciliation") or {}).get("blocked"):
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail=state["storage_reconciliation"])
    if not state["exists"]:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="该运行没有可恢复的图状态；旧版本运行请重新发起研究",
        )
    if not state["resumable"] and not state["delivery_ready"]:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=("图已执行结束，请直接读取研究结果"
                    if state["status"] == "completed" else
                    "研究尚未完成，但检查点没有待执行节点或可恢复的节点结果，请核查断点"),
        )

    if state["delivery_ready"] and getattr(run, "delivered_event_id", None):
        # 已交付只重放账本，不获取新租约或重复进入图执行/交付事务。
        replay_payloads = _pending_events(db, run_id, current_user.id, after_seq)

        async def replay_completed() -> AsyncGenerator[str, None]:
            for payload in replay_payloads:
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(replay_completed(), media_type="text/event-stream")

    original = (
        db.query(ChatMessage)
        .filter(ChatMessage.run_id == run_id, ChatMessage.role == "user")
        .first()
    )
    try:
        run_lease = await asyncio.to_thread(
            acquire_research_run_lease,
            run_id=run_id, session_id=run.session_id, user_id=user_id,
        )
    except ResearchRunLeaseConflict as exc:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail=str(exc)) from exc

    replay_payloads = _pending_events(db, run_id, current_user.id, after_seq)

    async def generate() -> AsyncGenerator[str, None]:
        # 先补发客户端游标之后已落账的事件，再继续执行。
        # 前端仍按 (run_id, seq) 去重，因此重复拉取是安全的。
        for payload in replay_payloads:
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        try:
            async for chunk in service.research(
                query=original.content if original else "",
                session_id=run.session_id, user_id=user_id, run_id=run_id,
                turn_id=str(original.turn_id) if original and original.turn_id else None,
                resume=True, lease=run_lease,
            ):
                yield chunk
        except LegacyRunNotResumableError as exc:
            yield "data: " + json.dumps({
                "type": "error", "run_id": run_id, "content": str(exc),
                "error_code": "legacy_run_not_resumable",
            }, ensure_ascii=False) + "\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _stream_with_lease(
            generate(), run_lease,
            on_failure=lambda: None,
        ),
        media_type="text/event-stream",
    )


@router.post("/cancel/{session_id}", status_code=HTTP_200_OK)
async def cancel_research(
    session_id: str,
    run_id: str = Query(..., description="本次研究运行 ID"),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """取消正在进行的研究任务。"""
    try:
        get_conversation_service().require_session(db, session_id, str(current_user.id))
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    request_cancel(run_id, session_id)
    logger.info("Research cancellation requested: session=%s run=%s", session_id, run_id)
    return {"success": True, "message": "Research cancellation requested", "run_id": run_id}


@router.get("/runs/{run_id}/state", status_code=HTTP_200_OK)
async def get_run_state(
    run_id: UUID,
    http_request: Request,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """读取本人某次研究的图状态（业务底稿与执行位置）。"""
    run_id = str(run_id)
    run = _owned_run(db, run_id, str(current_user.id))
    service = get_research_service(http_request)
    state = await service.read_state(run_id)
    research = state.pop("research")
    state["pipeline"] = "graph"
    if not state.get("exists") and not run.graph_version:
        from service.medical_research.storage import owned_medical_checkpoint
        if owned_medical_checkpoint(db, run_id, str(current_user.id)) is not None:
            state["pipeline"] = "legacy"
    # 检查点可能停在失败节点之前；执行状态以运行表为准。
    state["run_status"] = run.status
    state["error"] = (state.get("storage_reconciliation") or {}).get("reason", "")
    if run.status == "failed" and not state["error"]:
        failure = db.query(AgentEvent).filter(
            AgentEvent.run_id == run_id,
            AgentEvent.user_id == current_user.id,
            AgentEvent.event_type == "error",
        ).order_by(AgentEvent.seq.desc()).first()
        if failure:
            state["error"] = (failure.payload or {}).get("content", "研究执行失败")
    state["graph_version"] = state.get("graph_version") or run.graph_version or ""
    state["query"] = research.get("original_query") or research.get("query", "")
    state["original_query"] = state["query"]
    state["resolved_query"] = research.get("resolved_query") or research.get("query", "")
    # 已完成但尚未交付也可通过同一入口补偿；已交付则只有事件重放。
    state["delivery_pending"] = bool(state.get("delivery_ready") and not getattr(run, "delivered_event_id", None))
    state["resumable"] = bool(state.get("resumable") or state["delivery_pending"])
    # 页面恢复以检查点里的资料和报告为业务底稿；进度事件另按 seq 重放。
    state["draft"] = {
        "outline": research.get("outline", []),
        "facts": research.get("facts", []),
        "references": research.get("references", []),
        "knowledge_graph": research.get("knowledge_graph", {}),
        "final_report": research.get("final_report", ""),
        "claims": research.get("claims", []),
        "critic_feedback": research.get("critic_feedback", []),
        "writer_revision_records": research.get("writer_revision_records", []),
        "quality_gates": research.get("quality_gates", {}),
        "node_attempts": research.get("node_attempts", {}),
        "raw_sources": research.get("raw_sources", []),
        "search_budget": research.get("search_budget", {}),
        "web_reader_attempts": research.get("web_reader_attempts", []),
    }
    # 进度账本可在节点内部更新；仅用于展示，绝不覆盖 draft 或恢复位置。
    state["progress"] = None
    if state.get("pending_nodes") and run.status != "completed":
        latest = db.query(AgentEvent).filter(
            AgentEvent.run_id == run_id,
            AgentEvent.user_id == current_user.id,
            AgentEvent.payload.has_key("progress"),
        ).order_by(AgentEvent.seq.desc()).first()
        if latest:
            state["progress"] = pending_progress(state, latest.payload or {})
    return {"success": True, "state": state}


@router.get("/runs/{run_id}/events", status_code=HTTP_200_OK)
async def get_research_events(
    run_id: str,
    after_seq: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=2000),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """读取本人某次研究的事件，可用于刷新后重放和故障排查。"""
    rows = (
        db.query(AgentEvent)
        .filter(
            AgentEvent.run_id == run_id,
            AgentEvent.user_id == current_user.id,
            AgentEvent.seq > after_seq,
        )
        .order_by(AgentEvent.seq.asc())
        .limit(limit + 1)
        .all()
    )
    has_more = len(rows) > limit
    events = rows[:limit]
    return {
        "run_id": run_id,
        "events": list(replay_events(events)),
        "last_seq": events[-1].seq if events else after_seq,
        "has_more": has_more,
    }


# ============ 历史研究：只读 ============

@router.get("/legacy/checkpoints", status_code=HTTP_200_OK)
async def list_legacy_checkpoints(
    status: Optional[str] = Query(None, description="过滤状态: running/paused/completed/failed"),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user_required),
):
    """列出本人的历史研究快照。

    这些是 LangGraph 接管执行之前写入的完整业务快照。新运行不再写入，
    也不能从这里续跑；保留入口只为让历史报告仍可读取。
    """
    from service.checkpoint_service import get_checkpoint_service

    checkpoints = get_checkpoint_service().list_checkpoints(
        user_id=str(current_user.id), status=status, limit=limit,
    )
    return {
        "success": True,
        "read_only": True,
        "checkpoints": checkpoints,
        "total": len(checkpoints),
    }


@router.get("/legacy/checkpoint/{session_id}", status_code=HTTP_200_OK)
async def get_legacy_checkpoint(
    session_id: str,
    run_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user_required),
):
    """读取本人的历史研究快照全文（只读，不能作为新运行的恢复依据）。"""
    from service.checkpoint_service import get_checkpoint_service

    full = get_checkpoint_service().load_full_checkpoint(
        session_id, user_id=str(current_user.id), run_id=run_id,
    )
    if not full:
        return {"success": False, "message": "No legacy checkpoint found"}
    return {"success": True, "read_only": True, "checkpoint": full}


# ============ 内部工具 ============

def _owned_run(db: Session, run_id: str, user_id: str) -> ResearchRun:
    """thread_id 只负责定位；访问权限一律按 research_runs 的归属校验。"""
    run = (
        db.query(ResearchRun)
        .filter(ResearchRun.run_id == run_id, ResearchRun.user_id == UUID(user_id))
        .one_or_none()
    )
    if run is None:
        raise HTTPException(status_code=404, detail="未找到本人的研究运行")
    return run


def _pending_events(
    db: Session, run_id: str, user_id: Any, after_seq: int
) -> List[Dict[str, Any]]:
    rows = (
        db.query(AgentEvent)
        .filter(
            AgentEvent.run_id == run_id,
            AgentEvent.user_id == user_id,
            AgentEvent.seq > after_seq,
        )
        .order_by(AgentEvent.seq.asc())
        .all()
    )
    return list(replay_events(rows))


def _mark_turn_status(session_id: str, turn_id: str, status: str) -> None:
    db = SessionLocal()
    try:
        db.query(ChatMessage).filter(
            ChatMessage.session_id == UUID(session_id),
            ChatMessage.turn_id == UUID(turn_id),
        ).update({ChatMessage.research_status: status})
        db.commit()
    finally:
        db.close()


def is_research_cancelled(session_id: str) -> bool:
    """兼容旧调用方的会话级取消查询。"""
    return is_run_cancelled(session_id, session_id)


# 历史疾病文献成果的只读查询与导出沿用研究路由的认证体系。
from router.medical_research_router import router as medical_history_router  # noqa: E402

router.include_router(
    medical_history_router, prefix="/legacy/medical", tags=["历史疾病文献研究（只读）"]
)
