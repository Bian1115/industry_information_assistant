# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""研究终态的幂等业务交付。

图检查点是执行位置与业务状态的权威来源；本模块负责把"图已经跑完"翻译成
"用户可见的交付"：最终报告、参考文献、运行终态和一条完成事件。

三件事的一致性不是靠跨组件原子提交实现的（图检查点由 LangGraph 自己提交，
不在下面这个事务里），而是靠：
  1. 交付本身在**一个** SQLAlchemy 事务内完成；
  2. 以 (run_id, checkpoint_id, research_complete) 作为幂等标识，重复交付
     直接返回已有完成事件，不产生第二份报告或第二个完成序号；
  3. 图已完成但交付未提交时，可在下次打开报告或后台对账时从终态检查点补齐。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.exc import IntegrityError

from service.research_events import (
    ResearchEventEnvelope,
    ResearchEventRecorder,
    ResearchRunLease,
    ResearchRunLeaseLost,
)

from .state import GRAPH_VERSION, citation_verification_status

logger = logging.getLogger("DeepResearchDelivery")


class ResearchDeliveryError(RuntimeError):
    """终态交付失败；调用方不得据此对外宣布完成。"""


def checkpoint_id_of(snapshot: Any) -> str:
    """从快照配置中取稳定的 checkpoint 标识。"""
    config = getattr(snapshot, "config", None) or {}
    checkpoint_id = str(
        (config.get("configurable") or {}).get("checkpoint_id") or ""
    ).strip()
    if not checkpoint_id:
        raise ResearchDeliveryError("图快照缺少 checkpoint_id，拒绝交付")
    return checkpoint_id


def build_ui_references(research: Dict[str, Any]) -> List[Dict[str, Any]]:
    """构建前端友好的参考文献列表；只做字段整理，不新增来源。"""
    facts = [f for f in (research.get("facts") or []) if isinstance(f, dict)]
    result: List[Dict[str, Any]] = []
    for index, ref in enumerate(research.get("references") or []):
        if not isinstance(ref, dict):
            continue
        url = str(ref.get("url") or ref.get("link") or "")
        fact = next(
            (f for f in facts if str(f.get("source_url") or "") == url), None
        )
        title = str(ref.get("source") or ref.get("marker") or "")
        if not title and fact:
            content = str(fact.get("content") or "")
            title = content[:50] + "..." if len(content) > 50 else content
        result.append({
            "id": ref.get("id", index + 1),
            "title": title or f"来源 {index + 1}",
            "link": url,
            "content": str((fact or {}).get("content") or "")[:200],
            "source": "web",
        })
    return result


def build_complete_event(
    research: Dict[str, Any],
    *,
    run_id: str,
    checkpoint_id: str,
) -> Dict[str, Any]:
    """组装完成事件载荷。

    这里刻意区分三件不同的事：执行已结束（status/completion_reason）、
    报告审核结论（quality_verdict）、以及引用核验是否执行过
    （citation_verification 默认 not_executed，开启后返回真实审计）。任何一项都不能替代另一项。
    """
    return {
        "type": "research_complete",
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "graph_version": research.get("graph_version") or GRAPH_VERSION,
        "final_report": str(research.get("final_report") or ""),
        "references": build_ui_references(research),
        "completion_reason": str(research.get("completion_reason") or ""),
        "quality_score": research.get("quality_score", 0.0),
        "quality_score_status": research.get("quality_score_status", "not_reviewed"),
        "quality_score_basis": research.get("quality_score_basis", "not_reviewed"),
        "quality_dimension_scores": research.get("quality_dimension_scores", {}),
        "quality_verdict": research.get("quality_verdict", "not_reviewed"),
        "citation_verification": research.get(
            "citation_verification"
        ) or citation_verification_status(),
        "unresolved_issues": research.get("unresolved_issues", 0),
        "critic_feedback": research.get("critic_feedback", []),
        "writer_revision_records": research.get("writer_revision_records", []),
        "citation_verification_config": research.get("citation_verification_config", {"scout_facts": False, "report_claims": False}),
        "scout_citation_verification": research.get("scout_citation_verification") or citation_verification_status(),
        "citation_repair": research.get("citation_repair", {}),
        "iterations": research.get("iteration", 0),
        "facts_count": len(research.get("facts") or []),
        "allowed_domains": list(research.get("allowed_domains") or []),
        "hypotheses": research.get("hypotheses", []),
        "hypothesis_enabled": research.get("hypothesis_enabled", True),
        "hypothesis_evidence": research.get("hypothesis_evidence", []),
        "conclusion_evidence_map": research.get("conclusion_evidence_map", []),
        "claims": research.get("claims", []),
        "verified_citation_bindings": research.get("verified_citation_bindings", []),
        "retrieved_citation_bindings": research.get("retrieved_citation_bindings", []),
        "citation_retrieval_status": research.get("citation_retrieval_status", {}),
        "evidence_rag_metrics": research.get("evidence_rag_metrics", []),
        "evidence_space": research.get("evidence_space", {}),
        "research_contract": research.get("research_contract", {}),
        "execution_plan": research.get("execution_plan", "replan"),
        "session_task": research.get("session_task", {}),
        "session_task_metrics": research.get("session_task_metrics", {}),
    }


def make_commit_result(
    *,
    recorder: ResearchEventRecorder,
    lease: Optional[ResearchRunLease],
    run_id: str,
    session_id: str,
    user_id: str,
    turn_id: Optional[str],
    session_factory: Optional[Callable[[], Any]] = None,
) -> Callable[[Any], "ResearchEventEnvelope"]:
    """构造闭包形式的 ``commit_result``。

    权限信息（run / session / user / 租约）来自这里的闭包，不从模型输出读取：
    图状态里任何自称的用户或会话都不得影响写入目标。
    """

    def commit_result(snapshot: Any) -> ResearchEventEnvelope:
        from core.database import SessionLocal
        from models.chat import ChatMessage
        from models.research import AgentEvent, ResearchRun
        from service.conversation_service import get_conversation_service

        factory = session_factory or SessionLocal
        checkpoint_id = checkpoint_id_of(snapshot)
        research = dict((getattr(snapshot, "values", None) or {}).get("research") or {})
        report = str(research.get("final_report") or "").strip()
        if research.get("status") != "completed" or not report:
            raise ResearchDeliveryError("图未产生完整交付结果，拒绝交付")

        payload = build_complete_event(
            research, run_id=run_id, checkpoint_id=checkpoint_id
        )
        db = factory()
        try:
            # 1) 锁定运行行，确认本执行器仍然持有有效租约与 fencing token。
            run = (
                db.query(ResearchRun)
                .filter(ResearchRun.run_id == run_id)
                .with_for_update()
                .one_or_none()
            )
            if run is None:
                raise ResearchDeliveryError(f"run {run_id} 不存在，拒绝交付")
            if lease is not None:
                if (
                    run.owner_id != lease.owner_id
                    or int(run.fencing_token or 0) != int(lease.fencing_token)
                    or run.lease_until is None
                    or run.lease_until <= datetime.utcnow()
                ):
                    raise ResearchRunLeaseLost(
                        f"run {run_id} 的执行租约已失效，拒绝提交终态交付"
                    )

            # 2) 幂等：同一终态只交付一次。已交付则原样返回既有完成事件。
            delivered = (
                db.query(AgentEvent)
                .filter(
                    AgentEvent.run_id == run_id,
                    AgentEvent.event_type == "research_complete",
                )
                .order_by(AgentEvent.seq.asc())
                .first()
            )
            if delivered is not None:
                existing_checkpoint = str(
                    (delivered.payload or {}).get("checkpoint_id") or ""
                )
                if existing_checkpoint and existing_checkpoint != checkpoint_id:
                    # 同一 run 出现第二个终态：不静默覆盖，交由人工核对。
                    logger.warning(
                        "[交付] run=%s 已按 checkpoint=%s 交付，忽略 checkpoint=%s",
                        run_id, existing_checkpoint, checkpoint_id,
                    )
                db.rollback()
                return ResearchEventRecorder._from_row(delivered)

            # 3) 同一事务内写入最终报告、运行终态与完成事件。
            conversations = get_conversation_service()
            if turn_id:
                from service.session_task_service import outline_sections
                conversations.stage_assistant_message(
                    db,
                    session_id=session_id,
                    user_id=user_id,
                    turn_id=turn_id,
                    content=report,
                    references_data={"references": payload["references"], "research_contract": payload["research_contract"],
                                     "execution_plan": payload["execution_plan"], "session_task_metrics": payload["session_task_metrics"],
                                     "outline_sections": outline_sections(research.get("outline") or []),
                                     "outline_checkpoint_id": checkpoint_id},
                    execution_mode="research",
                    run_id=run_id,
                    research_status="completed",
                    replace_clarification=True,
                )
                db.query(ChatMessage).filter(
                    ChatMessage.run_id == run_id,
                    ChatMessage.role == "user",
                ).update({ChatMessage.research_status: "completed"})

            envelope = recorder.record_in_session(db, payload)

            run.status = "completed"
            run.graph_version = payload["graph_version"]
            run.final_checkpoint_id = checkpoint_id
            run.last_projected_checkpoint_id = checkpoint_id
            run.delivered_event_id = uuid.UUID(envelope.event_id)
            run.updated_at = datetime.utcnow()

            db.commit()
            return envelope
        except IntegrityError:
            # 唯一索引兜住了并发的第二次交付：回读已提交的那一条。
            db.rollback()
            delivered = (
                db.query(AgentEvent)
                .filter(
                    AgentEvent.run_id == run_id,
                    AgentEvent.event_type == "research_complete",
                )
                .order_by(AgentEvent.seq.asc())
                .first()
            )
            if delivered is None:
                raise
            return ResearchEventRecorder._from_row(delivered)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    return commit_result
