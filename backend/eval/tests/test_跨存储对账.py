"""恢复依赖、派生索引及真实PostgreSQL提交窗口；不调用外部模型。"""
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import uuid

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / "app"), str(Path(__file__).parent)]
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql
from psycopg import sql
from psycopg.conninfo import make_conninfo
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from service.deep_research_v2.checkpointer import LeaseFencedPostgresSaver, build_pool, checkpoint_database_url
from service.deep_research_v2.storage_reconciliation import StorageCoordinator, StorageConsistencyError, digest, PROTOCOL
from service.deep_research_v2.budget_ledger import PostgresRuntimeBudget
from service.deep_research_v2.graph import build_graph, graph_config, pack_state
from service.deep_research_v2.service import DeepResearchV2Service
from service.research_events import ResearchRunLease
from service.session_evidence_store import SessionEvidenceStore, EvidenceExpiredError
from models.chat import ChatSession, ChatMessage
from models.research import ResearchRun, AgentEvent
from models.agent_history import AgentContext, AgentMessage
from models.research_budget import ResearchBudget, ResearchBudgetAttempt, ResearchBudgetCall
from models.session_evidence import SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk
import test_会话临时双层RAG as fixtures
from test_单主链路图 import make_agents, passthrough_stage
from test_Writer章节检查点 import Scenario, initial_state


def snapshot(research, config=None, next_nodes=("write",)):
    return SimpleNamespace(values={"research": research}, config=config or {}, next=next_nodes, tasks=())


class IndexReconciliation(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.TemporaryRAGTests.asyncSetUp
    asyncTearDown = fixtures.TemporaryRAGTests.asyncTearDown
    create_engine = fixtures.TemporaryRAGTests.create_engine
    rows = fixtures.TemporaryRAGTests.rows

    def fake_inventory(self):
        def existing(rows):
            if self.vectors.fail:
                raise ConnectionError("注入索引不可用")
            return set(self.vectors.rows)
        self.vectors.existing_ids = existing

    async def test_missing_corrupt_and_extra_chunks_rebuilt_idempotently(self):
        self.fake_inventory()
        original = self.rows()
        with self.factory() as db:
            rows = db.execute(select(SessionEvidenceChunk)).scalars().all()
            db.delete(rows[0]); rows[1].content = "篡改chunk"
            db.add(SessionEvidenceChunk(id="额外索引", document_key=rows[1].document_key,
                unit_type="sentence", content="游离数据", locator={}))
            db.commit()
        self.rag.externalize(self.state)
        result = await self.rag.reconcile(self.state)
        self.assertEqual(result[0]["repaired_chunks"], 3)
        self.assertEqual({r["id"]:r["content"] for r in original}, {r["id"]:r["content"] for r in self.rows()})
        self.assertEqual((await self.rag.reconcile(self.state))[0]["repaired_chunks"], 0)

    async def test_indexed_flags_do_not_hide_lost_vectors(self):
        self.fake_inventory(); self.vectors.rows.clear(); self.api.embedded.clear()
        self.rag.externalize(self.state)
        result = await self.rag.reconcile(self.state)
        self.assertGreater(result[0]["missing_vectors"], 0)
        self.assertEqual(len(self.api.embedded), len(self.rows()))
        self.assertEqual(result[0]["index_status"], "indexed")

    async def test_vector_outage_is_explicit_lexical_degradation(self):
        self.fake_inventory(); self.vectors.fail = True
        self.rag.externalize(self.state)
        result = await self.rag.reconcile(self.state)
        self.assertEqual(result[0]["index_status"], "lexical_only:ConnectionError")
        await self.rag.prepare_stage(self.state, "writing", "test")
        self.assertNotIn("content", self.state["raw_sources"][0])

    async def test_reference_rejects_different_valid_document(self):
        self.rag.externalize(self.state)
        changed = fixtures.source("完全不同的市场正文。" * 40)
        with self.factory() as db:
            row = db.execute(select(SessionEvidenceDocument)).scalar_one()
            row.payload = changed; db.commit()
        with self.assertRaisesRegex(ValueError, "不存在|哈希|同一正文"):
            await self.rag.prepare_stage(self.state, "writing", "test")

    async def test_corrupt_source_never_rebuilt_as_index(self):
        self.fake_inventory(); self.rag.externalize(self.state)
        with self.factory() as db:
            row = db.execute(select(SessionEvidenceDocument)).scalar_one()
            row.payload = {**row.payload, "content": "正文被改写"}; db.commit()
        with self.assertRaisesRegex(EvidenceExpiredError, "哈希"):
            await self.rag.reconcile(self.state)

    async def test_generation_change_cannot_recreate_old_index(self):
        self.fake_inventory(); self.rag.externalize(self.state)
        with self.factory() as db:
            row = db.get(SessionEvidenceSpace, self.sid); row.generation += 1; db.commit()
        with self.assertRaisesRegex(EvidenceExpiredError, "generation"):
            await self.rag.reconcile(self.state)

    async def test_unindexable_original_remains_explicitly_unindexed(self):
        self.fake_inventory()
        long_source=fixtures.source("无法分句的长正文"*400)
        self.state["raw_sources"]=[long_source]
        await self.rag.ingest(self.state)
        self.rag.externalize(self.state)
        result=await self.rag.reconcile(self.state)
        self.assertEqual(result[0]["repaired_chunks"],0)
        await self.rag.prepare_stage(self.state, "writing", "test")
        self.assertNotIn("content",self.state["raw_sources"][0])
        self.assertFalse(self.state["raw_sources"][0]["rag_usable"])


class VectorInventory(unittest.TestCase):
    def test_actual_inventory_is_scoped_batched_and_rejects_wrong_document(self):
        from service.session_evidence_vectors import SessionEvidenceVectors
        vectors=SessionEvidenceVectors()
        uid,sid,topic=map(str,(uuid.uuid4(),uuid.uuid4(),uuid.uuid4()))
        rows=[{"id":str(i),"unit_type":"sentence","metadata":{"user_id":uid,"session_id":sid,
            "topic_id":topic,"evidence_space_generation":1,"document_id":"doc_1"}} for i in range(300)]
        queries=[]
        class Collection:
            def query(self,expr,**kwargs):
                queries.append(expr)
                ids=json.loads(expr.split(" and id in ",1)[1])
                return [{"id":i,"document_id":"wrong" if i=="1" else "doc_1","unit_type":"sentence"} for i in ids]
        with patch.object(vectors,"_collection",return_value=Collection()):
            found=vectors.existing_ids(rows)
        self.assertEqual(found,{str(i) for i in range(300)}-{"1"})
        self.assertEqual(len(queries),2)
        self.assertTrue(all(uid in q and sid in q and topic in q and "generation == 1" in q for q in queries))

    def test_missing_collection_is_rebuildable(self):
        from service.session_evidence_vectors import SessionEvidenceVectors
        vectors=SessionEvidenceVectors()
        with patch.object(vectors,"_collection",return_value=None):
            self.assertEqual(vectors.existing_ids([]),set())


@unittest.skipUnless(os.getenv("RUN_STORAGE_RECONCILIATION_PG") == "1", "仅在专用隔离schema运行器中启用")
class PostgreSQLReconciliation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.schema = "storage_reconcile_test_" + uuid.uuid4().hex
        self.created=False
        self.addAsyncCleanup(self.cleanup_schema)
        self.admin_pool = build_pool(make_conninfo(checkpoint_database_url(), connect_timeout=5))
        await self.admin_pool.open(); await self.admin_pool.wait()
        async with self.admin_pool.connection() as conn:
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
            self.created=True
            self.audit_schema("created")
        self.pool = build_pool(make_conninfo(checkpoint_database_url(), options="-c search_path=" + self.schema, connect_timeout=5))
        await self.pool.open(); await self.pool.wait()
        async with self.pool.connection() as conn:
            await conn.execute("CREATE TABLE users (id uuid PRIMARY KEY)")
            for model in (ChatSession, ChatMessage, ResearchRun, AgentEvent, AgentContext, AgentMessage,
                          ResearchBudget, ResearchBudgetAttempt, ResearchBudgetCall,
                          SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk):
                await conn.execute(str(CreateTable(model.__table__).compile(dialect=postgresql.dialect())))
            await AsyncPostgresSaver(conn).setup()
        from sqlalchemy.engine import URL
        from psycopg.conninfo import conninfo_to_dict
        info = conninfo_to_dict(checkpoint_database_url())
        self.engine = create_engine(URL.create("postgresql+psycopg", username=info.get("user"), password=info.get("password"),
            host=info.get("host"), port=int(info.get("port", 5432)), database=info["dbname"]),
            connect_args={"options": "-c search_path=" + self.schema, "connect_timeout": 5})
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.uid, self.sid = uuid.uuid4(), uuid.uuid4()
        self.run_id, self.message_id = str(uuid.uuid4()), uuid.uuid4()
        now = datetime.now(timezone.utc)
        self.lease = ResearchRunLease(self.run_id, str(self.sid), "隔离测试执行器", 1, (now + timedelta(minutes=10)).replace(tzinfo=None), str(self.uid))
        async with self.pool.connection() as conn:
            await conn.execute("INSERT INTO users VALUES(%s)", (self.uid,))
        with self.factory() as db:
            db.add(ChatSession(id=self.sid, user_id=self.uid, expires_at=now+timedelta(hours=24)))
            db.add(ResearchRun(run_id=self.run_id, session_id=str(self.sid), user_id=self.uid,
                owner_id=self.lease.owner_id, fencing_token=1, lease_until=self.lease.lease_until))
            db.commit()
            context = AgentContext(run_id=self.run_id, history_scope_id=self.run_id, agent_name="LeadWriter", context_key="main")
            db.add(context); db.flush()
            db.add(AgentMessage(id=self.message_id, context_id=context.id, run_id=self.run_id, seq=1,
                kind="output", call_id=uuid.uuid4(), invocation_id=uuid.uuid4(), node="research", payload={"observed":1}, content_hash=digest({"observed":1})))
            db.commit()
        self.store = SessionEvidenceStore(self.factory)
        space = self.store.ensure(self.uid, self.sid)
        self.topic = str(uuid.uuid4())
        source = fixtures.source()
        self.store.ingest(self.uid, self.sid, self.topic, [source], fixtures.Counter(), 1024)
        self.state = {"storage_protocol": PROTOCOL, "graph_version": "single-main-v5-session-evidence", "status":"running",
            "raw_sources":[{k:v for k,v in source.items() if k not in {"content", "sentence_index"}} | {"evidence_document_ref":True}],
            "evidence_space":{**space, "revision":1, "user_id":str(self.uid), "session_id":str(self.sid), "topic_id":self.topic}}
        self.storage = StorageCoordinator(self.pool, self.run_id, lease=self.lease)
        self.saver = LeaseFencedPostgresSaver(self.pool, lease=self.lease)
        self.saver.storage_coordinator = self.storage
        self.budget = PostgresRuntimeBudget(self.pool, self.lease)
        self.attempt, _ = await self.budget.begin({"_node":"write", "node_attempts":{"write":1}}, "writing", "LeadWriter")
        self.call = await self.budget.reserve(self.attempt, "模拟模型", 120)
        self.state["runtime_budget"] = {"ledger_version":1}
        self.sealed = await self.storage.seal(self.state, node="research")

    def audit_schema(self,status):
        path=os.getenv("STORAGE_RECONCILIATION_AUDIT")
        if path:
            with Path(path).open("a") as stream:
                stream.write(json.dumps({"test":self.id(),"schema":self.schema,"status":status},ensure_ascii=False)+"\n")

    async def cleanup_schema(self):
        if hasattr(self, "engine"): self.engine.dispose()
        if hasattr(self, "pool"): await self.pool.close()
        if hasattr(self, "admin_pool") and self.created:
            async with self.admin_pool.connection() as conn:
                await conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))
                self.audit_schema("cleaned")
        if hasattr(self,"admin_pool"):
            await self.admin_pool.close()

    async def put(self, value=None):
        cp = empty_checkpoint()
        cp["channel_values"] = {"research": value if value is not None else self.sealed}
        cp["channel_versions"] = {"research":"1"}
        config={"configurable":{"thread_id":self.run_id,"checkpoint_ns":""}}
        return await self.saver.aput(config, cp, {"source":"input", "step":0, "parents":{}}, {"research":"1"})

    async def inspect(self, value=None, **kwargs):
        return await self.storage.inspect(snapshot(value if value is not None else self.sealed), **kwargs)

    async def test_manifest_commits_and_late_budget_settlement_remains_valid(self):
        config = await self.put()
        await self.budget.complete(self.attempt, self.call, {"total_tokens":80})
        saved = await self.saver.aget_tuple(config)
        self.assertEqual(saved.checkpoint["channel_values"]["research"], self.sealed)
        self.assertFalse((await self.inspect())["blocked"])

    async def test_history_delete_between_seal_and_commit_rolls_back_checkpoint(self):
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM agent_messages WHERE id=%s", (self.message_id,))
        with self.assertRaisesRegex(StorageConsistencyError, "Agent历史"):
            await self.put()
        self.assertIsNone(await self.saver.aget_tuple(graph_config(self.run_id)))

    async def test_deleted_budget_call_blocks_pending_write(self):
        config = await self.put()
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM research_budget_calls WHERE call_id=%s", (self.call,))
        with self.assertRaisesRegex(StorageConsistencyError, "预算请求"):
            await self.saver.aput_writes(config, [("research", self.sealed)], "节点结果")
        self.assertEqual((await self.saver.aget_tuple(config)).pending_writes, [])

    async def test_valid_replacement_body_still_fails_manifest_hash(self):
        # 即使攻击/损坏同时修改payload自带的哈希，清单的正文版本仍不能被覆盖。
        with self.factory() as db:
            row = db.execute(select(SessionEvidenceDocument)).scalar_one()
            row.payload = fixtures.source("替换的全新正文。"*60); db.commit()
        self.assertTrue((await self.inspect())["blocked"])

    async def test_expired_and_changed_generation_are_blocked(self):
        with self.factory() as db:
            row = db.get(SessionEvidenceSpace, self.sid); row.generation += 1; db.commit()
        self.assertTrue((await self.inspect())["blocked"])
        with self.factory() as db:
            row = db.get(SessionEvidenceSpace, self.sid); row.generation -= 1
            row.expires_at = datetime.now(timezone.utc)-timedelta(seconds=1); db.commit()
        self.assertTrue((await self.inspect())["blocked"])

    async def test_idle_ttl_honors_prior_new_message_but_never_revives_expired_space(self):
        now=datetime.now(timezone.utc)
        with self.factory() as db:
            space=db.get(SessionEvidenceSpace,self.sid)
            space.expires_at=now-timedelta(seconds=10)
            chat=db.get(ChatSession,self.sid)
            chat.expires_at=now+timedelta(hours=24)-timedelta(seconds=20)
            db.commit()
        self.assertFalse((await self.inspect())["blocked"])
        with self.factory() as db:
            self.assertLess(db.get(SessionEvidenceSpace,self.sid).expires_at,now)
            chat=db.get(ChatSession,self.sid); chat.expires_at=now+timedelta(hours=24); db.commit()
        self.assertTrue((await self.inspect())["blocked"])
    async def test_orphan_history_is_observation_not_node_completion(self):
        with self.factory() as db:
            old = db.get(AgentMessage, self.message_id)
            db.add(AgentMessage(context_id=old.context_id, run_id=self.run_id, seq=2, kind="output", call_id=uuid.uuid4(),
                invocation_id=uuid.uuid4(), node="write", payload={"report":"尚未提交"}, content_hash=digest({"report":"尚未提交"})))
            db.commit()
        result = await self.inspect()
        self.assertFalse(result["blocked"]); self.assertEqual(result["uncommitted_history_rows"],1)
        self.assertNotIn("final_report", self.sealed)
        self.assertEqual(self.sealed["status"], "running")

    async def test_event_gap_is_blocked_without_fabricating_events(self):
        with self.factory() as db:
            run = db.get(ResearchRun,self.run_id); run.next_event_seq=2
            db.add(AgentEvent(session_id=str(self.sid), run_id=self.run_id, user_id=self.uid, seq=2,
                event_type="observation", payload={"content":"进度"})); db.commit()
        self.assertTrue((await self.inspect())["blocked"])

    async def test_cross_user_and_artifact_mutation_are_blocked(self):
        changed=deepcopy(self.sealed); changed["status"]="completed"
        self.assertTrue((await self.inspect(changed))["blocked"])
        changed=deepcopy(self.state); changed["evidence_space"]["user_id"]=str(uuid.uuid4())
        with self.assertRaisesRegex(StorageConsistencyError,"当前用户"):
            await self.storage.seal(changed)

    async def test_empty_missing_and_malformed_manifest_cannot_downgrade_to_legacy(self):
        for value in ({}, None, "损坏", {"kind":"research"}):
            with self.subTest(manifest=value):
                changed=deepcopy(self.sealed); changed["storage_manifest"]=value
                self.assertTrue((await self.inspect(changed))["blocked"])
        changed=deepcopy(self.sealed); changed.pop("storage_manifest")
        self.assertTrue((await self.inspect(changed))["blocked"])

    async def test_legacy_checkpoint_is_explicitly_partial_not_false_proof(self):
        old=deepcopy(self.state); old.pop("storage_protocol")
        result=await self.inspect(old)
        self.assertEqual(result["status"],"legacy_checked")
        self.assertTrue(result["legacy_limit"])

    async def test_settled_usage_cannot_be_reduced_after_checkpoint(self):
        await self.budget.complete(self.attempt,self.call,{"total_tokens":80})
        sealed=await self.storage.seal(self.state)
        async with self.pool.connection() as conn:
            await conn.execute("UPDATE research_budget_calls SET actual_tokens=1 WHERE call_id=%s",(self.call,))
        result=await self.inspect(sealed)
        self.assertTrue(result["blocked"])
        self.assertIn("预算请求",result["reason"])

    def rag(self):
        from service.session_evidence_rag import SessionEvidenceRAG
        vectors=fixtures.Vectors()
        vectors.existing_ids=lambda rows:set(vectors.rows)
        return SessionEvidenceRAG(str(self.uid),str(self.sid),store=self.store,api=fixtures.API(),
                                  vectors=vectors,counter=fixtures.Counter())

    async def seed_graph(self, *, stop=False):
        from service.deep_research_v2.state import create_initial_state
        rag=self.rag()
        graph=build_graph(make_agents(),self.saver,run_id=self.run_id,user_id=str(self.uid),session_id=str(self.sid),
            execute_stage=passthrough_stage,storage_coordinator=self.storage,evidence_service=rag,
            interrupt_before=["write"] if stop else None)
        state=create_initial_state("恢复依赖核对",str(self.sid),[])
        state.update(raw_sources=self.state["raw_sources"],evidence_space=self.state["evidence_space"])
        state=await self.storage.seal(state)
        await graph.ainvoke(pack_state(state),graph_config(self.run_id),durability="sync")
        return graph,rag

    async def test_service_state_blocks_and_resume_calls_no_agents_when_body_missing(self):
        from service.deep_research_v2 import service as module
        _,rag=await self.seed_graph(stop=True)
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM session_evidence_documents WHERE session_id=%s",(self.sid,))
        agents=make_agents()
        service=DeepResearchV2Service(LeaseFencedPostgresSaver(self.pool))
        with patch.object(module,"create_agents",return_value=agents), patch("core.database.SessionLocal",self.factory), patch(
            "service.session_evidence_rag.SessionEvidenceRAG",return_value=rag):
            state=await service.read_state(self.run_id)
            self.assertFalse(state["resumable"]); self.assertTrue(state["storage_reconciliation"]["blocked"])
            chunks=[chunk async for chunk in service.research(query="恢复",session_id=str(self.sid),user_id=str(self.uid),
                        run_id=self.run_id,resume=True,lease=self.lease)]
        events=[json.loads(c.removeprefix("data: ").strip()) for c in chunks if "[DONE]" not in c]
        self.assertTrue(any(e.get("error_code")=="research_storage_inconsistent" for e in events))
        self.assertFalse(any(agent.calls for agent in agents.values()))

    async def test_service_terminal_compensation_reconciles_then_delivers_without_agents(self):
        from service.deep_research_v2 import service as module
        _,rag=await self.seed_graph()
        agents=make_agents()
        service=DeepResearchV2Service(LeaseFencedPostgresSaver(self.pool))
        with patch.object(module,"create_agents",return_value=agents), patch("core.database.SessionLocal",self.factory), patch(
            "service.session_evidence_rag.SessionEvidenceRAG",return_value=rag):
            chunks=[c async for c in service.research(query="补交付",session_id=str(self.sid),user_id=str(self.uid),
                          run_id=self.run_id,resume=True,lease=self.lease)]
            state=await service.read_state(self.run_id)
        events=[json.loads(c.removeprefix("data: ").strip()) for c in chunks if "[DONE]" not in c]
        self.assertEqual([e["type"] for e in events],["storage_reconciled","research_complete"])
        self.assertEqual(state["storage_reconciliation"]["delivery"],"delivered")
        self.assertFalse(any(agent.calls for agent in agents.values()))

    async def test_pending_node_result_is_checked_and_resumed_without_rewriting(self):
        from service.deep_research_v2.state import create_initial_state
        class StopSaver(LeaseFencedPostgresSaver):
            stopped=False
            async def aput(inner, config, checkpoint, metadata, new_versions):
                if not inner.stopped and "branch:to:review" in checkpoint["channel_values"]:
                    inner.stopped=True
                    raise RuntimeError("注入节点结果已存但后继checkpoint未提交")
                return await super().aput(config, checkpoint, metadata, new_versions)
        saver=StopSaver(self.pool,lease=self.lease); saver.storage_coordinator=self.storage
        rag=self.rag()
        def graph_for(agents):
            return build_graph(agents,saver,run_id=self.run_id,user_id=str(self.uid),session_id=str(self.sid),
                execute_stage=passthrough_stage,evidence_service=rag,storage_coordinator=self.storage)
        first=graph_for(make_agents())
        state=create_initial_state("待合并节点",str(self.sid),[])
        state.update(raw_sources=self.state["raw_sources"],evidence_space=self.state["evidence_space"])
        with self.assertRaisesRegex(RuntimeError,"后继checkpoint"):
            await first.ainvoke(pack_state(await self.storage.seal(state)),graph_config(self.run_id),durability="sync")
        snap=await first.aget_state(graph_config(self.run_id)); raw=await saver.aget_tuple(snap.config)
        self.assertTrue(any(ch=="research" for _,ch,_ in raw.pending_writes))
        self.assertFalse((await self.storage.inspect(snap,raw))["blocked"])
        damaged=deepcopy(raw)
        for _,channel,value in damaged.pending_writes:
            if channel=="research": value["final_report"]="篡改待合并正文"; break
        self.assertTrue((await self.storage.inspect(snap,damaged))["blocked"])
        agents=make_agents(); resumed=graph_for(agents)
        result=await resumed.ainvoke(None,graph_config(self.run_id),durability="sync")
        self.assertEqual(result["research"]["status"],"completed")
        self.assertEqual(agents["writer"].calls,[])

    async def test_saved_section_is_checked_and_reused_with_new_graph(self):
        scenario=Scenario(fail_section="sec_3")
        from service.session_evidence_rag import SessionEvidenceRAG
        rag=SessionEvidenceRAG(str(self.uid),str(self.sid),store=self.store,api=fixtures.API(),
                               vectors=fixtures.Vectors(),counter=fixtures.Counter())
        async def graph_for():
            return build_graph(make_agents(writer=scenario.writer()), self.saver, run_id=self.run_id,
                user_id=str(self.uid), session_id=str(self.sid), execute_stage=passthrough_stage,
                storage_coordinator=self.storage, evidence_service=rag)
        first=await graph_for()
        state=initial_state()
        # 本用例把正文替换为 example.com 夹具，检索范围也须对应同一来源。
        state["allowed_domains"] = ["example.com"]
        state.update(raw_sources=self.state["raw_sources"], evidence_space=self.state["evidence_space"])
        state=await self.storage.seal(state)
        with self.assertRaisesRegex(RuntimeError,"sec_3"):
            await first.ainvoke(pack_state(state), graph_config(self.run_id), durability="sync")
        snap=await first.aget_state(graph_config(self.run_id))
        raw=await self.saver.aget_tuple(snap.config)
        chapters=[v for _,ch,v in raw.pending_writes if ch=="__return__" and isinstance(v,dict) and v.get("version")==1]
        self.assertEqual(len(chapters),2)
        self.assertTrue(all(c.get("storage_manifest") for c in chapters))
        self.assertFalse((await self.storage.inspect(snap,raw))["blocked"])
        damaged=deepcopy(raw)
        for _,ch,value in damaged.pending_writes:
            if ch=="__return__" and isinstance(value,dict) and value.get("version")==1:
                value["content"]="篡改章节"; break
        self.assertTrue((await self.storage.inspect(snap,damaged))["blocked"])
        resumed=await graph_for()
        result=await resumed.ainvoke(None, graph_config(self.run_id), durability="sync")
        self.assertEqual(result["research"]["status"],"completed")
        self.assertEqual(scenario.calls["sec_1"],1); self.assertEqual(scenario.calls["sec_2"],1)

    async def test_terminal_event_conflict_is_blocked(self):
        state=deepcopy(self.state); state.update(status="completed",final_report="真实报告")
        sealed=await self.storage.seal(state,node="finish")
        snap=snapshot(sealed, {"configurable":{"checkpoint_id":"终态编号"}}, ())
        self.assertEqual((await self.storage.inspect(snap))["delivery"],"pending")
        with self.factory() as db:
            event=AgentEvent(session_id=str(self.sid),run_id=self.run_id,user_id=self.uid,seq=1,
                event_type="research_complete",payload={"checkpoint_id":"终态编号","final_report":"另一份报告"})
            db.add(event); db.flush()
            run=db.get(ResearchRun,self.run_id); run.delivered_event_id=event.event_id
            run.final_checkpoint_id="终态编号"; run.next_event_seq=1; db.commit()
        self.assertTrue((await self.storage.inspect(snap))["blocked"])


if __name__ == "__main__":
    unittest.main()
