"""证据结束事务与幂等清理。独立任务在会话行删除后仍能完成外部资源删除。"""
from datetime import timedelta
from pathlib import Path
import os
import shutil
import uuid

from sqlalchemy import select, delete, update
from core.database import SessionLocal
from core.time_utils import utc_now, as_utc
from models.chat import ChatSession
from models.session_evidence import (SessionEvidenceSpace, SessionEvidenceDocument,
                                     SessionEvidenceChunk, SessionEvidencePurge)
from models.research import ResearchRun, ResearchCheckpoint, AgentEvent
from models.agent_history import AgentContext, AgentMessage


def evidence_cache_root():
    import tempfile
    return Path(os.getenv("SESSION_EVIDENCE_CACHE_DIR", str(Path(tempfile.gettempdir()) / "industry-session-evidence-cache")))


def close_evidence_in_transaction(db, *, user_id, session_id):
    uid, sid = uuid.UUID(str(user_id)), uuid.UUID(str(session_id))
    space = db.get(SessionEvidenceSpace, sid)
    if space is not None and space.user_id != uid:
        raise ValueError("临时证据空间归属不匹配")
    if space is None:
        # 保留关闭墓碑，使已结束的会话不能被旧运行或新索引任务复活。
        space = SessionEvidenceSpace(session_id=sid, user_id=uid, status="closed", generation=1,
                                     revision=0, expires_at=utc_now(), ttl_mode="idle")
        db.add(space)
    elif space.status == "active":
        space.status = "closed"
        space.generation += 1
    runs = db.execute(select(ResearchRun).where(ResearchRun.session_id == str(sid),
                                              ResearchRun.user_id == uid)).scalars().all()
    run_ids = [r.run_id for r in runs]
    for run in runs:
        if run.status not in {"completed", "cancelled"}:
            run.status = "cancelled"
        run.fencing_token += 1
        run.owner_id = None
        run.lease_until = None
    job = db.get(SessionEvidencePurge, str(sid))
    if job is None:
        job = SessionEvidencePurge(session_id=str(sid), user_id=str(uid), run_ids=run_ids,
                                   status="pending", attempts=0)
        db.add(job)
    else:
        combined = list(dict.fromkeys([*job.run_ids, *run_ids]))
        if set(combined) != set(job.run_ids):
            job.status = "pending"
            job.completed_at = None
        job.run_ids = combined
    db.flush()
    return {"status": "closed", "purge_status": job.status, "generation": space.generation}


def close_session_evidence(user_id, session_id, *, session_factory=SessionLocal):
    uid, sid = uuid.UUID(str(user_id)), uuid.UUID(str(session_id))
    with session_factory() as db:
        chat = db.execute(select(ChatSession).where(ChatSession.id == sid, ChatSession.user_id == uid)
                          .with_for_update()).scalar_one_or_none()
        if chat is None:
            raise ValueError("会话不存在")
        result = close_evidence_in_transaction(db, user_id=uid, session_id=sid)
        run_ids = list(db.get(SessionEvidencePurge, str(sid)).run_ids)
        db.commit()
    from service.research_control import request_cancel
    for run_id in run_ids:
        try:
            request_cancel(run_id, str(sid))
        except Exception:
            # 数据库关闭和fence已生效；外部取消通知由清理任务重试。
            pass
    return result


def delete_checkpoint_threads(run_ids):
    if not run_ids:
        return
    from langgraph.checkpoint.postgres import PostgresSaver
    from service.deep_research_v2.checkpointer import checkpoint_database_url
    with PostgresSaver.from_conn_string(checkpoint_database_url()) as saver:
        for run_id in run_ids:
            saver.delete_thread(run_id)


def process_evidence_purges(*, session_factory=SessionLocal, vectors=None, checkpoint_delete=None, batch_size=20, now=utc_now):
    from service.session_evidence_vectors import SessionEvidenceVectors
    vectors = vectors or SessionEvidenceVectors()
    checkpoint_delete = checkpoint_delete or delete_checkpoint_threads
    completed = 0
    for _ in range(batch_size):
        with session_factory() as db:
            job = db.execute(select(SessionEvidencePurge).where(SessionEvidencePurge.status == "pending")
                             .order_by(SessionEvidencePurge.created_at)
                             .limit(1).with_for_update(skip_locked=True)).scalar_one_or_none()
            if job is None:
                break
            job.attempts += 1
            try:
                from service.research_control import request_cancel
                for run_id in job.run_ids:
                    request_cancel(run_id, job.session_id)
                vectors.delete_session(job.user_id, job.session_id)
                checkpoint_delete(job.run_ids)
                uid, sid = uuid.UUID(job.user_id), uuid.UUID(job.session_id)
                # 关闭任务已固定归属；不按URL删除其他会话的证据。
                keys = select(SessionEvidenceDocument.id).where(SessionEvidenceDocument.session_id == sid)
                db.execute(delete(SessionEvidenceChunk).where(SessionEvidenceChunk.document_key.in_(keys)))
                db.execute(delete(SessionEvidenceDocument).where(SessionEvidenceDocument.session_id == sid))
                contexts = list(db.execute(select(AgentMessage.context_id).where(AgentMessage.run_id.in_(job.run_ids))).scalars())
                db.execute(delete(AgentMessage).where(AgentMessage.run_id.in_(job.run_ids)))
                db.execute(delete(AgentContext).where(AgentContext.id.in_(contexts)))
                db.execute(delete(ResearchCheckpoint).where(ResearchCheckpoint.session_id == job.session_id,
                                                           ResearchCheckpoint.user_id == uid))
                db.execute(delete(AgentEvent).where(AgentEvent.session_id == job.session_id, AgentEvent.user_id == uid))
                # 统计可以保留；输入/输出预览也包含证据原文，随会话清理。
                from models.eval_trace import AgentRun
                db.execute(update(AgentRun).where(AgentRun.session_id == job.session_id, AgentRun.user_id == uid)
                           .values(input_preview=None, output_preview=None, error_message=None))
                path = evidence_cache_root() / str(sid)
                if path.exists():
                    shutil.rmtree(path)
                from service.session_memory_service import get_session_memory_service
                get_session_memory_service().invalidate(user_id=job.user_id, session_id=job.session_id)
                job.status = "purged"
                job.completed_at = now()
                job.last_error = None
                db.commit()
                completed += 1
            except Exception as exc:
                db.rollback()
                job = db.get(SessionEvidencePurge, job.session_id)
                job.attempts += 1
                job.last_error = type(exc).__name__
                db.commit()
                break  # 保留可重试任务，避免同一轮反复请求不可用服务。
    return completed


def cleanup_expired_evidence(*, session_factory=SessionLocal, now=utc_now, batch_size=100):
    with session_factory() as db:
        spaces = db.execute(select(SessionEvidenceSpace).where(SessionEvidenceSpace.status == "active")
                            .where(SessionEvidenceSpace.expires_at <= now())
                            .order_by(SessionEvidenceSpace.expires_at)
                            .limit(batch_size).with_for_update(skip_locked=True)).scalars().all()
        closed = 0
        for space in spaces:
            chat = db.get(ChatSession, space.session_id)
            expires = as_utc(space.expires_at)
            if chat is not None and space.ttl_mode == "idle":
                # 延续已提交新消息的期限；摘要、检索不影响这个时间。
                message_time = as_utc(chat.expires_at) - timedelta(hours=24)
                expires = message_time + timedelta(seconds=space.ttl_seconds)
                if expires > as_utc(space.expires_at) and message_time < as_utc(space.expires_at):
                    space.expires_at = expires
                else:
                    expires = as_utc(space.expires_at)
            if chat is None or expires <= now():
                close_evidence_in_transaction(db, user_id=space.user_id, session_id=space.session_id)
                closed += 1
        db.commit()
        return closed
