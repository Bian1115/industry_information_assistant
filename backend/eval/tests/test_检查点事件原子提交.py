"""隔离 PostgreSQL 验证 checkpoint/task 产物与完成事件原子提交。"""
import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "app"), str(Path(__file__).parent)]

from psycopg import sql
from psycopg.conninfo import make_conninfo, conninfo_to_dict
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END

from models.research import ResearchRun, AgentEvent
from service.research_events import ResearchRunLease, ResearchRunLeaseLost, ResearchEventRecorder
from service.deep_research_v2 import checkpointer as module
from service.deep_research_v2.checkpointer import LeaseFencedPostgresSaver, build_pool, checkpoint_database_url
from service.deep_research_v2.checkpoint_events import event_intent, read_committed, committed_stream
from service.deep_research_v2.graph import GraphState, make_node, graph_config, pack_state
from test_单主链路图 import passthrough_stage
from test_Writer章节检查点 import Scenario, initial_state, graph_for


@unittest.skipUnless(os.getenv("RUN_ATOMIC_EVENTS_PG") == "1", "仅在隔离 PostgreSQL 回归中启用")
class AtomicCheckpointEvents(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.schema = "原子事件测试_" + uuid.uuid4().hex[:20]
        self.admin = build_pool()
        await self.admin.open()
        await self.admin.wait()
        self.addAsyncCleanup(self.cleanup)
        async with self.admin.connection() as conn:
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.created = True
        self.audit("created")
        self.pool = build_pool(make_conninfo(checkpoint_database_url(), options="-c search_path=" + self.schema))
        await self.pool.open()
        await self.pool.wait()
        self.run_id, self.uid, self.sid = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        self.lease = ResearchRunLease(self.run_id, self.sid, "隔离执行器", 1,
            datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=10), self.uid)
        async with self.pool.connection() as conn:
            await conn.execute("CREATE TABLE users (id uuid PRIMARY KEY)")
            for model in (ResearchRun, AgentEvent):
                await conn.execute(str(CreateTable(model.__table__).compile(dialect=postgresql.dialect())))
            await AsyncPostgresSaver(conn).setup()
            await conn.execute("INSERT INTO users VALUES (%s)", (self.uid,))
            await conn.execute(
                "INSERT INTO research_runs (run_id,session_id,user_id,owner_id,fencing_token,lease_until,"
                "next_event_seq,status,created_at,updated_at) VALUES(%s,%s,%s,%s,1,%s,0,'running',now(),now())",
                (self.run_id, self.sid, self.uid, self.lease.owner_id, self.lease.lease_until))
        self.saver = LeaseFencedPostgresSaver(self.pool, lease=self.lease)
        self.notifications = []
        self.saver.on_events_committed = lambda: self.notifications.append(True)
        info = conninfo_to_dict(checkpoint_database_url())
        self.engine = create_engine(URL.create("postgresql+psycopg", username=info.get("user"),
            password=info.get("password"), host=info.get("host"), port=int(info.get("port", 5432)),
            database=info["dbname"]), connect_args={"options": "-c search_path=" + self.schema})
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def audit(self, status):
        if path := os.getenv("ATOMIC_EVENTS_AUDIT"):
            import json
            with Path(path).open("a") as f:
                f.write(json.dumps({"test": self.id(), "schema": self.schema, "status": status}, ensure_ascii=False) + "\n")

    async def cleanup(self):
        if hasattr(self, "engine"):
            self.engine.dispose()
        if hasattr(self, "pool"):
            await self.pool.close()
        if getattr(self, "created", False):
            async with self.admin.connection() as conn:
                await conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))
            self.audit("cleaned")
        await self.admin.close()

    def checkpoint(self, intents=()):
        cp = empty_checkpoint()
        cp["channel_values"] = {"research": {"value": 1}, "checkpoint_events": list(intents)}
        cp["channel_versions"] = {"research": "1", "checkpoint_events": "1"}
        return cp

    async def save(self, cp, config=None):
        return await self.saver.aput(config or {"configurable": {"thread_id": self.run_id, "checkpoint_ns": ""}},
            cp, {"source": "update", "step": 1, "parents": {}}, cp["channel_versions"])

    async def counts(self):
        async with self.pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT (SELECT count(*) FROM checkpoints) AS checkpoints, "
                "(SELECT count(*) FROM checkpoint_blobs) AS blobs, "
                "(SELECT count(*) FROM checkpoint_writes) AS writes, "
                "(SELECT count(*) FROM agent_events) AS events, next_event_seq "
                "FROM research_runs WHERE run_id=%s", (self.run_id,))).fetchone()
            return row

    async def reject_events(self):
        async with self.pool.connection() as conn:
            await conn.execute("CREATE FUNCTION 拒绝事件() RETURNS trigger LANGUAGE plpgsql AS $$ "
                               "BEGIN RAISE EXCEPTION '注入事件写入失败'; END; $$")
            await conn.execute("CREATE TRIGGER 拒绝事件 BEFORE INSERT ON agent_events "
                               "FOR EACH ROW EXECUTE FUNCTION 拒绝事件()")

    async def allow_events(self):
        async with self.pool.connection() as conn:
            await conn.execute("DROP TRIGGER 拒绝事件 ON agent_events")

    async def test_checkpoint_event_failure_rolls_back_snapshot_blobs_and_seq(self):
        intent = event_intent({"type": "node_completed", "node": "research"})
        cp = self.checkpoint([intent])
        await self.reject_events()
        with self.assertRaisesRegex(Exception, "注入事件写入失败"):
            await self.save(cp)
        self.assertEqual(await self.counts(), {"checkpoints": 0, "blobs": 0, "writes": 0, "events": 0, "next_event_seq": 0})
        self.assertEqual(self.notifications, [])
        await self.allow_events()
        await self.save(cp)
        events = await read_committed(self.pool, self.lease, 0)
        self.assertEqual((events[0].event_id, events[0].seq), (intent["event_id"], 1))
        self.assertEqual(events[0].data["persistence"]["checkpoint_id"], cp["id"])
        self.assertEqual(len(self.notifications), 1)

    async def test_retry_and_later_snapshot_reuse_original_id_and_seq(self):
        intent = event_intent({"type": "node_completed", "node": "research"})
        cp = self.checkpoint([intent])
        config = await self.save(cp)
        await self.save(cp)
        await self.save(self.checkpoint([intent]), config)
        events = await read_committed(self.pool, self.lease, 0)
        self.assertEqual([(e.event_id, e.seq) for e in events], [(intent["event_id"], 1)])
        self.assertEqual((await self.counts())["next_event_seq"], 1)
        self.assertEqual(len(self.notifications), 1)

    async def test_chapter_event_failure_rolls_back_task_result_and_seq(self):
        config = await self.save(self.checkpoint())
        intent = event_intent({"type": "section_content", "content": {"section_id": "第一章", "content": "正文"}})
        artifact = {"version": 1, "input_hash": "固定输入", "content": "正文", "checkpoint_events": [intent]}
        await self.reject_events()
        with self.assertRaisesRegex(Exception, "注入事件写入失败"):
            await self.saver.aput_writes(config, [("__return__", artifact)], "章节任务")
        counts = await self.counts()
        self.assertEqual((counts["writes"], counts["events"], counts["next_event_seq"]), (0, 0, 0))
        await self.allow_events()
        await self.saver.aput_writes(config, [("__return__", artifact)], "章节任务")
        await self.saver.aput_writes(config, [("__return__", artifact)], "章节任务")
        events = await read_committed(self.pool, self.lease, 0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["persistence"]["kind"], "task_result")
        self.assertEqual(events[0].data["persistence"]["task_id"], "章节任务")
        self.assertEqual((await self.saver.aget_tuple(config)).pending_writes[0][2], artifact)

    async def test_node_pending_writes_do_not_announce_checkpoint_completion(self):
        config = await self.save(self.checkpoint())
        intent = event_intent({"type": "node_completed", "node": "research"})
        await self.saver.aput_writes(config, [("checkpoint_events", [intent])], "节点任务")
        self.assertEqual(await read_committed(self.pool, self.lease, 0), [])
        cp = self.checkpoint([intent])
        await self.save(cp, config)
        self.assertEqual((await read_committed(self.pool, self.lease, 0))[0].data["persistence"]["checkpoint_id"], cp["id"])

    async def test_changed_task_candidate_does_not_publish_events_for_unsaved_result(self):
        config = await self.save(self.checkpoint())
        first = {"content": "原正文", "checkpoint_events": [event_intent({"type": "section_content", "content": "原正文"})]}
        await self.saver.aput_writes(config, [("__return__", first)], "章节任务")
        changed = {"content": "新正文", "checkpoint_events": [event_intent({"type": "section_content", "content": "新正文"})]}
        with self.assertRaisesRegex(Exception, "章节任务结果已存在"):
            await self.saver.aput_writes(config, [("__return__", changed)], "章节任务")
        self.assertEqual((await self.counts())["events"], 1)
        self.assertEqual((await self.saver.aget_tuple(config)).pending_writes[0][2], first)

    async def test_changed_task_body_with_same_events_is_also_rejected(self):
        config = await self.save(self.checkpoint())
        first = {"content": "原正文", "checkpoint_events": [event_intent({"type": "section_content", "content": "原正文"})]}
        await self.saver.aput_writes(config, [("__return__", first)], "章节任务")
        with self.assertRaisesRegex(Exception, "章节任务结果已存在"):
            await self.saver.aput_writes(config, [("__return__", {**first, "content": "另一正文"})], "章节任务")
        self.assertEqual((await self.counts())["events"], 1)
        self.assertEqual((await self.saver.aget_tuple(config)).pending_writes[0][2], first)

    async def test_final_lease_check_failure_rolls_back_everything(self):
        original = module.require_lease

        async def expire(cur, lease, *, lock=True):
            await original(cur, lease, lock=lock)
            if not lock:
                raise ResearchRunLeaseLost("注入提交前租约失效")

        with patch.object(module, "require_lease", expire), self.assertRaises(ResearchRunLeaseLost):
            await self.save(self.checkpoint([event_intent({"type": "node_completed"})]))
        self.assertEqual((await self.counts())["checkpoints"], 0)
        self.assertEqual((await self.counts())["events"], 0)
        self.assertEqual((await self.counts())["next_event_seq"], 0)
        self.assertEqual(self.notifications, [])

    async def test_reusing_event_id_with_changed_content_rejects_new_snapshot(self):
        intent = event_intent({"type": "node_completed", "node": "research"})
        await self.save(self.checkpoint([intent]))
        changed = {**intent, "event": {"type": "node_completed", "node": "write"}}
        with self.assertRaisesRegex(Exception, "ID.*冲突"):
            await self.save(self.checkpoint([changed]))
        self.assertEqual((await self.counts())["checkpoints"], 1)
        self.assertEqual((await self.counts())["next_event_seq"], 1)

    def recorder(self):
        return ResearchEventRecorder(session_id=self.sid, run_id=self.run_id,
            user_id=self.uid, lease=self.lease)

    async def test_commit_wakes_sse_without_waiting_for_more_progress(self):
        stopped = asyncio.Event()

        async def stream():
            try:
                await self.save(self.checkpoint([event_intent({"type": "node_completed"})]))
                await asyncio.Event().wait()
                yield {}
            finally:
                stopped.set()

        output = committed_stream(stream(), self.recorder(), self.saver, after_seq=0)
        event = await asyncio.wait_for(anext(output), 5)
        self.assertEqual(event.seq, 1)
        await output.aclose()
        self.assertTrue(stopped.is_set())
        self.assertIsNone(self.saver.on_events_committed)

    async def test_scout_resume_keeps_original_completion_and_does_not_rerun(self):
        calls = {"research": 0, "analyze": 0}

        async def scout(work):
            calls["research"] += 1
            work["_emit_event"]({"type": "research_step", "content": {"step_type": "research", "status": "completed"}})
            return work

        async def analyze(work):
            calls["analyze"] += 1
            if calls["analyze"] == 1:
                raise RuntimeError("注入后继节点失败")
            return work

        builder = StateGraph(GraphState)
        for name, action, phase in [("research", scout, "researching"), ("analyze", analyze, "analyzing")]:
            builder.add_node(name, make_node(action, phase, name, run_id=self.run_id,
                user_id=self.uid, session_id=self.sid, execute_stage=passthrough_stage))
        builder.add_edge(START, "research"); builder.add_edge("research", "analyze"); builder.add_edge("analyze", END)
        graph = builder.compile(checkpointer=self.saver)

        async def collect(value):
            with patch("core.database.SessionLocal", self.factory):
                return [e async for e in committed_stream(graph.astream(value, graph_config(self.run_id),
                    stream_mode="custom", durability="sync"), self.recorder(), self.saver, after_seq=0)]

        with self.assertRaisesRegex(RuntimeError, "后继节点"):
            await collect(pack_state({"node_attempts": {}}))
        before = [e for e in await read_committed(self.pool, self.lease, 0) if e.type == "node_completed"]
        await collect(None)
        after = [e for e in await read_committed(self.pool, self.lease, 0)
                 if e.type == "node_completed" and e.data["node"] == "research"]
        self.assertEqual(calls, {"research": 1, "analyze": 2})
        self.assertEqual([(e.event_id, e.seq) for e in before], [(e.event_id, e.seq) for e in after])

    async def test_writer_restores_chapter_results_without_new_chapter_events(self):
        scenario = Scenario(fail_section="sec_3")

        async def collect(value):
            graph = graph_for(self.saver, scenario.writer(), self.run_id)
            with patch("core.database.SessionLocal", self.factory):
                return [e async for e in committed_stream(graph.astream(value, graph_config(self.run_id),
                    stream_mode="custom", durability="sync"), self.recorder(), self.saver, after_seq=0)]

        with self.assertRaisesRegex(RuntimeError, "sec_3"):
            await collect(pack_state(initial_state()))
        before = [e for e in await read_committed(self.pool, self.lease, 0) if e.type == "section_content"]
        self.assertEqual(len(before), 2)
        await collect(None)
        all_events = []
        while batch := await read_committed(self.pool, self.lease, all_events[-1].seq if all_events else 0):
            all_events.extend(batch)
        after = [e for e in all_events if e.type == "section_content"]
        self.assertEqual(len(after), 3)
        self.assertEqual([(e.event_id, e.seq) for e in before], [(e.event_id, e.seq) for e in after[:2]])
        self.assertEqual(scenario.calls, {"sec_1": 1, "sec_2": 1, "sec_3": 2, "synthesis": 1})
        self.assertEqual([e.seq for e in all_events], list(range(1, len(all_events) + 1)))


if __name__ == "__main__":
    unittest.main()
