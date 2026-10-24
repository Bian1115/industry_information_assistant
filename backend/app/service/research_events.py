# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""DeepResearch V2 的版本化事件协议、运行租约和 PostgreSQL 事件账本。"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, or_
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

RESEARCH_EVENT_VERSION = 1
DEFAULT_LEASE_SECONDS = 60
EVENT_PERSIST_RETRIES = 3


class ResearchEventPersistenceError(RuntimeError):
    """事件无法持久化时阻止继续发送和推进研究状态。"""


class ResearchRunLeaseConflict(RuntimeError):
    """同一 run 已由另一个仍存活的执行器持有。"""


class ResearchRunLeaseLost(RuntimeError):
    """执行器的 fencing token 已过期，禁止旧执行器继续写入。"""


@dataclass(frozen=True)
class ResearchRunLease:
    run_id: str
    session_id: str
    owner_id: str
    fencing_token: int
    lease_until: datetime
    user_id: Optional[str] = None


class ResearchEventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = RESEARCH_EVENT_VERSION
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    session_id: str
    run_id: str
    turn_id: Optional[str] = None
    seq: int = Field(ge=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: Dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> Dict[str, Any]:
        """新协议放在 data 中，同时平铺旧字段，保证旧页面不中断。"""
        wire = self.model_dump()
        for key, value in self.data.items():
            if key not in wire:
                wire[key] = value
        return wire


def _as_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def make_research_event(
    raw_event: Dict[str, Any],
    *,
    session_id: str,
    run_id: str,
    turn_id: Optional[str],
    seq: int,
    event_id: Optional[str] = None,
) -> ResearchEventEnvelope:
    raw = dict(raw_event)
    event_type = str(raw.pop("type", "unknown"))
    timestamp = str(raw.pop("timestamp", "") or datetime.now(timezone.utc).isoformat())
    for metadata_key in ("version", "event_id", "session_id", "run_id", "turn_id", "seq"):
        raw.pop(metadata_key, None)
    return ResearchEventEnvelope(
        event_id=event_id or str(uuid.uuid4()),
        type=event_type,
        session_id=session_id,
        run_id=run_id,
        turn_id=turn_id,
        seq=seq,
        timestamp=timestamp,
        data=raw,
    )


def acquire_research_run_lease(
    *,
    run_id: str,
    session_id: str,
    user_id: Optional[str],
    owner_id: Optional[str] = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> ResearchRunLease:
    """原子获取单一执行权；租约过期后 token 加一，使旧执行器失效。"""
    from core.database import SessionLocal
    from models.research import ResearchRun

    owner_id = owner_id or str(uuid.uuid4())
    user_uuid = _as_uuid(user_id)
    now = datetime.utcnow()
    lease_until = now + timedelta(seconds=lease_seconds)
    db = SessionLocal()
    try:
        excluded = postgresql_insert(ResearchRun).excluded
        same_user = or_(ResearchRun.user_id.is_(None), ResearchRun.user_id == user_uuid)
        available = or_(
            ResearchRun.owner_id == owner_id,
            ResearchRun.owner_id.is_(None),
            ResearchRun.lease_until.is_(None),
            ResearchRun.lease_until < now,
        )
        statement = (
            postgresql_insert(ResearchRun)
            .values(
                run_id=run_id,
                session_id=session_id,
                user_id=user_uuid,
                next_event_seq=0,
                owner_id=owner_id,
                lease_until=lease_until,
                fencing_token=1,
                status="running",
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=[ResearchRun.run_id],
                set_={
                    "session_id": excluded.session_id,
                    "user_id": excluded.user_id,
                    "owner_id": excluded.owner_id,
                    "lease_until": excluded.lease_until,
                    "fencing_token": ResearchRun.fencing_token + 1,
                    "status": "running",
                    "updated_at": now,
                },
                where=and_(same_user, available),
            )
            .returning(ResearchRun.fencing_token, ResearchRun.lease_until)
        )
        row = db.execute(statement).first()
        db.commit()
        if not row:
            raise ResearchRunLeaseConflict(f"run {run_id} 已由其他执行器持有")
        return ResearchRunLease(
            run_id=run_id,
            session_id=session_id,
            owner_id=owner_id,
            fencing_token=int(row.fencing_token),
            lease_until=row.lease_until,
            user_id=str(user_uuid) if user_uuid else None,
        )
    except ResearchRunLeaseConflict:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def renew_research_run_lease(
    lease: ResearchRunLease,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> bool:
    from core.database import SessionLocal
    from models.research import ResearchRun

    now = datetime.utcnow()
    db = SessionLocal()
    try:
        updated = (
            db.query(ResearchRun)
            .filter(
                ResearchRun.run_id == lease.run_id,
                ResearchRun.owner_id == lease.owner_id,
                ResearchRun.fencing_token == lease.fencing_token,
            )
            .update({
                ResearchRun.lease_until: now + timedelta(seconds=lease_seconds),
                ResearchRun.updated_at: now,
            })
        )
        db.commit()
        return updated == 1
    except Exception:
        db.rollback()
        logger.exception("[ResearchEvents] 续租失败: run=%s", lease.run_id)
        return False
    finally:
        db.close()


def release_research_run_lease(lease: ResearchRunLease, status: str) -> bool:
    from core.database import SessionLocal
    from models.research import ResearchRun

    db = SessionLocal()
    try:
        updated = (
            db.query(ResearchRun)
            .filter(
                ResearchRun.run_id == lease.run_id,
                ResearchRun.owner_id == lease.owner_id,
                ResearchRun.fencing_token == lease.fencing_token,
            )
            .update({
                ResearchRun.owner_id: None,
                ResearchRun.lease_until: None,
                ResearchRun.status: status,
                ResearchRun.updated_at: datetime.utcnow(),
            })
        )
        db.commit()
        return updated == 1
    except Exception:
        db.rollback()
        logger.exception("[ResearchEvents] 释放租约失败: run=%s", lease.run_id)
        return False
    finally:
        db.close()


class ResearchEventRecorder:
    """以数据库行锁原子分配 seq，并在提交成功后返回可发送事件。"""

    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        turn_id: Optional[str] = None,
        user_id: Optional[str] = None,
        persist: bool = True,
        lease: Optional[ResearchRunLease] = None,
    ):
        self.session_id = session_id
        self.run_id = run_id
        self.turn_id = turn_id
        self.user_id = user_id
        self.persist = persist
        self.lease = lease
        self._seq = 0
        if persist and not lease:
            raise ValueError("持久化事件必须提供有效的 ResearchRunLease")

    @staticmethod
    def _from_row(event: Any) -> ResearchEventEnvelope:
        return ResearchEventEnvelope(
            event_id=str(event.event_id),
            type=event.event_type,
            session_id=event.session_id,
            run_id=event.run_id,
            turn_id=str(event.turn_id) if event.turn_id else None,
            seq=event.seq,
            timestamp=event.occurred_at.replace(tzinfo=timezone.utc).isoformat(),
            data=event.payload or {},
        )

    def _record_in_session(
        self,
        db: Any,
        raw_event: Dict[str, Any],
        event_id: str,
    ) -> ResearchEventEnvelope:
        from models.research import AgentEvent, ResearchRun

        existing = db.query(AgentEvent).filter(
            AgentEvent.event_id == uuid.UUID(event_id)
        ).first()
        if existing:
            return self._from_row(existing)

        run = (
            db.query(ResearchRun)
            .filter(
                ResearchRun.run_id == self.run_id,
                ResearchRun.owner_id == self.lease.owner_id,
                ResearchRun.fencing_token == self.lease.fencing_token,
                ResearchRun.lease_until > datetime.utcnow(),
            )
            .with_for_update()
            .one_or_none()
        )
        if not run:
            raise ResearchRunLeaseLost(
                f"run {self.run_id} 的执行租约已失效，拒绝写入事件"
            )

        run.next_event_seq += 1
        run.updated_at = datetime.utcnow()
        envelope = make_research_event(
            raw_event,
            session_id=self.session_id,
            run_id=self.run_id,
            turn_id=self.turn_id,
            seq=run.next_event_seq,
            event_id=event_id,
        )
        db.add(AgentEvent(
            event_id=uuid.UUID(envelope.event_id),
            session_id=envelope.session_id,
            run_id=envelope.run_id,
            turn_id=_as_uuid(self.turn_id),
            user_id=_as_uuid(self.user_id),
            seq=envelope.seq,
            event_type=envelope.type,
            payload=envelope.data,
            occurred_at=datetime.fromisoformat(
                envelope.timestamp.replace("Z", "+00:00")
            ).replace(tzinfo=None),
        ))
        self._seq = envelope.seq
        return envelope

    def record_in_session(
        self,
        db: Any,
        raw_event: Dict[str, Any],
        event_id: Optional[str] = None,
    ) -> ResearchEventEnvelope:
        """加入调用方事务；由调用方统一提交事件及关联状态。"""
        if not self.persist:
            return self.record(raw_event)
        return self._record_in_session(db, raw_event, event_id or str(uuid.uuid4()))

    def record(self, raw_event: Dict[str, Any]) -> ResearchEventEnvelope:
        if not self.persist:
            self._seq += 1
            return make_research_event(
                raw_event,
                session_id=self.session_id,
                run_id=self.run_id,
                turn_id=self.turn_id,
                seq=self._seq,
            )

        from core.database import SessionLocal

        event_id = str(uuid.uuid4())
        last_error: Optional[BaseException] = None
        for attempt in range(EVENT_PERSIST_RETRIES):
            db = SessionLocal()
            try:
                envelope = self._record_in_session(db, raw_event, event_id)
                db.commit()
                return envelope
            except ResearchRunLeaseLost:
                db.rollback()
                raise
            except SQLAlchemyError as exc:
                db.rollback()
                last_error = exc
                if attempt + 1 < EVENT_PERSIST_RETRIES:
                    time.sleep(0.05 * (2 ** attempt))
            except Exception as exc:
                db.rollback()
                last_error = exc
                break
            finally:
                db.close()

        raise ResearchEventPersistenceError(
            f"事件持久化失败，未发送: run={self.run_id}, event_id={event_id}"
        ) from last_error


def replay_events(events: Iterable[Any]) -> Iterable[Dict[str, Any]]:
    for event in events:
        yield ResearchEventRecorder._from_row(event).to_wire()
