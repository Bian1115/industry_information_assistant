"""将完成通知随执行产物提交，并从已提交账本发送。

节点通知绑定完整 checkpoint，章节通知绑定 task pending write。
事件账本同时作为事务发件箱；不新增另一份待发送状态。
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from service.research_events import (
    ResearchEventEnvelope, ResearchEventPersistenceError, make_research_event,
)

EVENTS_CHANNEL = "checkpoint_events"


def requires_commit(event: dict) -> bool:
    content = event.get("content")
    return event.get("type") in {"section_content", "report_draft", "node_completed"} or (
        event.get("type") == "research_step" and isinstance(content, dict)
        and content.get("status") == "completed"
    )


def event_intent(event: dict) -> dict:
    # ID 随产物保存。aput 重试或后续快照再次包含此意图时沿用同一 ID。
    return {"event_id": str(uuid4()), "event": deepcopy(event)}


def envelope_from_row(row) -> ResearchEventEnvelope:
    occurred = row["occurred_at"]
    return ResearchEventEnvelope(
        event_id=str(row["event_id"]), type=row["event_type"],
        session_id=row["session_id"], run_id=row["run_id"],
        turn_id=str(row["turn_id"]) if row["turn_id"] else None,
        seq=row["seq"], timestamp=occurred.replace(tzinfo=timezone.utc).isoformat(),
        data=row["payload"],
    )


async def append_events(cur, lease, intents, reference, *, turn_id=None) -> int:
    """调用方已持有 run 行锁；本函数不打开连接、不提交事务。"""
    inserted = 0
    for intent in intents:
        event = make_research_event(
            intent["event"], session_id=lease.session_id, run_id=lease.run_id,
            turn_id=turn_id, seq=1, event_id=intent["event_id"],
        )
        await cur.execute("SELECT * FROM agent_events WHERE event_id=%s", (event.event_id,))
        existing = await cur.fetchone()
        if existing:
            payload = dict(existing["payload"])
            payload.pop("persistence", None)
            if (existing["run_id"] != lease.run_id or existing["session_id"] != lease.session_id
                    or existing["event_type"] != event.type or payload != event.data):
                raise ResearchEventPersistenceError("完成事件 ID 与已保存内容冲突，拒绝提交产物")
            continue
        await cur.execute(
            "UPDATE research_runs SET next_event_seq=next_event_seq+1, "
            "updated_at=clock_timestamp() AT TIME ZONE 'UTC' WHERE run_id=%s RETURNING next_event_seq",
            (lease.run_id,),
        )
        event.seq = (await cur.fetchone())["next_event_seq"]
        payload = {**event.data, "persistence": reference}
        occurred = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
        if occurred.tzinfo is not None:
            occurred = occurred.astimezone(timezone.utc).replace(tzinfo=None)
        await cur.execute(
            "INSERT INTO agent_events (event_id,session_id,run_id,turn_id,user_id,seq,event_type,"
            "payload,occurred_at,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (event.event_id, lease.session_id, lease.run_id, turn_id, lease.user_id, event.seq,
             event.type, Jsonb(payload), occurred, datetime.now(timezone.utc).replace(tzinfo=None)),
        )
        inserted += 1
    return inserted


async def read_committed(conn, lease, after_seq, *, limit=500):
    from .checkpointer import shared_connection
    async with shared_connection(conn) as connection:
        async with connection.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM agent_events WHERE run_id=%s AND session_id=%s "
                "AND user_id IS NOT DISTINCT FROM %s::uuid AND seq>%s ORDER BY seq LIMIT %s",
                (lease.run_id, lease.session_id, lease.user_id, after_seq, limit),
            )
            return [envelope_from_row(row) for row in await cur.fetchall()]


async def committed_stream(stream, recorder, saver, *, after_seq):
    """合并实时进度与事务提交唤醒，始终按账本序号输出。

通知仅负责唤醒；发送前重新读取已提交数据，不依赖通知携带的事件副本。
"""
    if not recorder.persist:
        async for event in stream:
            yield recorder.record(event)
        return

    queue: asyncio.Queue = asyncio.Queue()
    def notify():
        queue.put_nowait(("committed", None))

    saver.on_events_committed = notify

    async def pump():
        try:
            async for event in stream:
                await queue.put(("progress", event))
        except Exception as exc:
            await queue.put(("error", exc))
        else:
            await queue.put(("end", None))

    producer = asyncio.create_task(pump())
    try:
        while True:
            kind, value = await queue.get()
            if kind == "progress":
                if requires_commit(value):
                    raise ResearchEventPersistenceError("完成事件绕过产物事务，拒绝直接发送")
                await asyncio.to_thread(recorder.record, value)
            while batch := await read_committed(saver.conn, recorder.lease, after_seq):
                for event in batch:
                    if event.seq != after_seq + 1:
                        raise ResearchEventPersistenceError("已提交事件账本存在序号缺口")
                    after_seq = event.seq
                    yield event
            if kind == "error":
                raise value
            if kind == "end":
                return
    finally:
        if saver.on_events_committed is notify:
            saver.on_events_committed = None
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer
