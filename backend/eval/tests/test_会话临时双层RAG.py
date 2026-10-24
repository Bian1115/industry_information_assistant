"""临时证据真实SQL存储＋注入检索后端；不调用付费模型或修改业务数据库。"""
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND)]
from sqlalchemy import create_engine, select, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

@compiles(JSONB, "sqlite")
def jsonb_sqlite(type_, compiler, **kwargs):
    return "JSON"

import models
from models.user import User
from models.chat import ChatSession
from models.session_evidence import SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk, SessionEvidencePurge
from models.research import ResearchRun, ResearchCheckpoint, AgentEvent
from models.agent_history import AgentContext, AgentMessage
from models.eval_trace import AgentRun
from service.web_document import index_document, assess_body
from service.session_evidence_chunks import chunk_document
from service.session_evidence_store import SessionEvidenceStore, EvidenceExpiredError
from service.session_evidence_rag import (SessionEvidenceRAG, SessionEvidenceSettings, bind_session_evidence,
                                         evidence_prompt, fit_evidence_prompt, BEGIN, END)
from service.session_evidence_ranking import CONTEXT_INSTRUCTION, CITATION_INSTRUCTION
from service.session_evidence_lifecycle import (close_session_evidence, process_evidence_purges, cleanup_expired_evidence)


class Counter:
    def count_text(self, text):
        # 英文排序指令不能按字符数冒充token，否则短正文384条也会被测试计数器误判超预算。
        return len(text.split()) if text in {CONTEXT_INSTRUCTION, CITATION_INSTRUCTION} else len(text)
    def count_messages(self, messages): return sum(len(m["content"]) for m in messages)


class API:
    def __init__(self): self.embedded = []; self.closed = False; self.reranked = []
    async def embed(self, texts):
        self.embedded.extend(texts)
        return [[float(1 + t.count("增长")), float(1 + t.count("下降"))] for t in texts]
    async def rerank(self, query, documents, top_n, *, instruct=None):
        self.reranked.append({"query": query, "documents": list(documents), "top_n": top_n, "instruct": instruct})
        indexes = sorted(range(len(documents)), key=lambda i: (-sum(w in documents[i] for w in query.split()), i))[:top_n]
        return [{"index": i, "relevance_score": .8} for i in indexes]
    async def close(self): self.closed = True


class Vectors:
    dimensions = 2
    def __init__(self): self.rows = {}; self.fail = False; self.searches = []
    def upsert(self, rows, vectors):
        if self.fail: raise ConnectionError("offline")
        self.rows.update({r["id"]: (deepcopy(r), v) for r, v in zip(rows, vectors)})
    def search(self, user_id, session_id, topic_id, generation, vector, kinds, limit, *, offset=0):
        self.searches.append({"limit": limit, "offset": offset, "kinds": set(kinds)})
        if self.fail: raise ConnectionError("offline")
        valid = [(cid, sum(x*y for x,y in zip(v,vector))) for cid,(r,v) in self.rows.items()
                 if r["metadata"]["user_id"] == user_id and r["metadata"]["session_id"] == session_id
                 and r["metadata"]["topic_id"] == topic_id and r["unit_type"] in kinds
                 and r["metadata"]["evidence_space_generation"] == generation]
        return [cid for cid,_ in sorted(valid, key=lambda item: (-item[1], item[0]))[offset:offset + limit]]
    def delete_session(self, user_id, session_id):
        if self.fail: raise ConnectionError("offline")
        self.rows = {cid: (r,v) for cid,(r,v) in self.rows.items() if not (
            r["metadata"]["user_id"] == user_id and r["metadata"]["session_id"] == session_id)}


def source(text=None, url="https://example.com/report"):
    text = text or "# 市场分析\n\n苹果销量增长。增长只适用于2025年的中国市场。\n\n2026年销量下降。此前增长不能外推到2026年。\n\n| 年份 | 销量（万台） |\n| --- | --- |\n| 2025 | 100 |\n| 2026 | 90 |\n注：仅统计中国市场。\n"
    text, quality = assess_body(text, "市场分析")
    return {"id": "source_1", "url": url, "title": "市场分析", "content": text,
            "body_quality": quality, **index_document(text, url)}


class TemporaryRAGTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = datetime(2026, 10, 8, 10, tzinfo=timezone.utc)
        self.engine = self.create_engine()
        for model in (User, ChatSession, SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk,
                      SessionEvidencePurge, ResearchRun, ResearchCheckpoint, AgentEvent, AgentContext, AgentMessage, AgentRun):
            model.__table__.create(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.uid, self.sid = uuid.uuid4(), uuid.uuid4()
        with self.factory() as db:
            db.add(User(id=self.uid, username="测试", email="test@example.com", hashed_password="unused")); db.flush()
            db.add(ChatSession(id=self.sid, user_id=self.uid, expires_at=self.now+timedelta(hours=24)))
            db.commit()
        self.store = SessionEvidenceStore(self.factory, now=lambda:self.now)
        self.api, self.vectors, self.counter = API(), Vectors(), Counter()
        self.rag = SessionEvidenceRAG(str(self.uid), str(self.sid), store=self.store, api=self.api,
                                      vectors=self.vectors, counter=self.counter,
                                      settings=SessionEvidenceSettings(context_tokens=12000, citation_tokens=6000))
        self.state = {"session_id":str(self.sid), "original_query":"苹果销量增长原因", "query":"苹果销量增长原因",
                      "resolved_query":"苹果销量增长原因", "raw_sources":[source()], "allowed_domains":[], "claims":[]}
        self.original_source = deepcopy(self.state["raw_sources"][0])
        await self.rag.attach(self.state, run_id=str(uuid.uuid4()))
        await self.rag.ingest(self.state)

    async def asyncTearDown(self):
        await self.rag.close(); self.engine.dispose()

    def create_engine(self):
        engine=create_engine("sqlite://", connect_args={"check_same_thread":False}, poolclass=StaticPool)
        @event.listens_for(engine,"connect")
        def fk(connection,_): connection.execute("PRAGMA foreign_keys=ON")
        return engine

    def rows(self, kinds=None):
        return self.store.chunks(self.uid, self.sid, self.rag.topic(self.state), kinds=kinds)

    async def test_chunk_metadata_and_adjacent_ids(self):
        rows = self.rows()
        ids = {r["id"] for r in rows}
        for row in rows:
            meta = row["metadata"]
            self.assertEqual(meta["document_id"], self.state["raw_sources"][0]["document_id"])
            self.assertIn("prev_chunk_id", meta); self.assertIn("next_chunk_id", meta)
            self.assertTrue(meta["paragraph_ids"]); self.assertTrue(meta["sentence_ids"])
            self.assertTrue(meta["prev_chunk_id"] is None or meta["prev_chunk_id"] in ids)
            self.assertTrue(meta["next_chunk_id"] is None or meta["next_chunk_id"] in ids)
        for row in rows:
            if row["unit_type"] == "sentence": self.assertIn(row["metadata"]["parent_chunk_id"], ids)

    async def test_tables_never_enter_plain_sentence_chunks(self):
        tables = self.rows({"table"})
        self.assertEqual(len(tables),1)
        self.assertIn("销量（万台）", tables[0]["content"])
        self.assertIn("仅统计中国市场", tables[0]["content"])
        self.assertEqual(tables[0]["metadata"]["row_range"], [1,2])
        for row in self.rows({"sentence","paragraph"}): self.assertNotIn("| 2025 |", row["content"])
        self.assertIn("context", tables[0]["metadata"]["adjacent_chunk_ids"])
        self.assertIn("citation", tables[0]["metadata"]["adjacent_chunk_ids"])

    async def test_long_table_preserves_every_row_and_header(self):
        text="| 年份 | 指标 |\n| --- | --- |\n"+"".join(f"| {i} | 数值{i} |\n" for i in range(20))+"注：单位为万元。\n"
        rows=chunk_document(source(text), user_id=self.uid, session_id=self.sid,
                            topic_id=self.rag.topic(self.state), counter=self.counter, chunk_tokens=100)
        tables=[r for r in rows if r["unit_type"]=="table"]
        self.assertGreater(len(tables),1)
        self.assertEqual(sum(len(r["metadata"]["row_spans"]) for r in tables),20)
        for row in tables:
            self.assertIn("| 年份 | 指标 |",row["content"]); self.assertIn("单位为万元",row["content"])
            self.assertLessEqual(len(row["content"]),100)

    async def test_retrieval_layers_use_different_units(self):
        for layer, kinds in (("context",{"paragraph","table","record","section"}),("citation",{"sentence","table","record"})):
            rows,audit=await self.rag.retrieve(self.state,"苹果销量增长",layer=layer,neighbors=True)
            self.assertTrue(rows); self.assertTrue(all(r["unit_type"] in kinds for r in rows))
            self.assertLessEqual(audit["selected_tokens"],audit["budget_tokens"])

    async def test_first_three_fused_pages_are_reranked_and_fourth_page_is_excluded(self):
        from unittest.mock import AsyncMock
        self.state["raw_sources"].append(source("\n\n".join(
            f"苹果销量增长条目{i:04d}。" for i in range(520)), url="https://example.com/paged"))
        await self.rag.ingest(self.state)
        self.rag.settings = SessionEvidenceSettings(context_tokens=100000, citation_tokens=100000)
        def reverse(query, documents, top_n, *, instruct=None):
            return [{"index": i, "relevance_score": .9} for i in range(len(documents)-1, len(documents)-top_n-1, -1)]
        self.api.rerank = AsyncMock(side_effect=reverse)
        for layer, top_k in (("context", 8), ("citation", 5)):
            self.vectors.searches.clear(); self.api.rerank.reset_mock()
            rows, audit = await self.rag.retrieve(self.state, "苹果销量增长", layer=layer)
            self.assertEqual([p["count"] for p in audit["coarse_pages"]], [128, 128, 128])
            self.assertEqual(audit["coarse_count"], 384)
            self.assertEqual([(s["limit"], s["offset"]) for s in self.vectors.searches],
                             [(128, 0), (128, 128), (128, 256)])
            self.assertEqual(len(self.api.rerank.await_args.args[1]), 384)
            self.assertEqual(self.api.rerank.await_args.args[2], top_k)
            self.assertEqual(audit["rerank_status"], "completed")
            self.assertEqual(audit["rerank_scored_candidate_count"], 384)
            self.assertEqual(rows[0]["id"], audit["coarse_pages"][-1]["chunk_ids"][-1])
            self.assertEqual(len(rows), top_k)
            self.assertEqual(len(set(cid for p in audit["coarse_pages"] for cid in p["chunk_ids"])), 384)
            expected = "research question" if layer == "context" else "verifying the claim"
            self.assertIn(expected, self.api.rerank.await_args.kwargs["instruct"])

    async def test_short_corpus_has_no_duplicate_or_padded_pages(self):
        rows, audit = await self.rag.retrieve(self.state, "苹果", layer="context")
        self.assertEqual(len(self.vectors.searches), 1)
        self.assertEqual(audit["coarse_page_count"], 1)
        self.assertLess(audit["coarse_pages"][0]["count"], 128)
        self.assertEqual(len(self.api.reranked[-1]["documents"]), audit["coarse_count"])
        self.assertTrue(rows)

    async def test_later_dense_page_failure_keeps_completed_pages(self):
        self.state["raw_sources"].append(source("\n\n".join(
            f"苹果销量增长条目{i:04d}。" for i in range(400)), url="https://example.com/partial"))
        await self.rag.ingest(self.state)
        search = self.vectors.search
        def failed_page(*args, **kwargs):
            if kwargs["offset"] == 128:
                raise ConnectionError("分页服务中断")
            return search(*args, **kwargs)
        self.vectors.search = failed_page
        rows, audit = await self.rag.retrieve(self.state, "苹果增长", layer="context")
        self.assertTrue(rows)
        self.assertIn("partial_dense:ConnectionError", audit["mode"])
        self.assertEqual(audit["dense_count"], 128)
        self.assertEqual(audit["coarse_count"], 384)
        self.assertEqual(audit["rerank_status"], "completed")

    async def test_rerank_failure_is_recorded_and_falls_back_to_coarse_order(self):
        from unittest.mock import AsyncMock
        self.api.rerank = AsyncMock(side_effect=RuntimeError("精排服务中断"))
        rows, audit = await self.rag.retrieve(self.state, "苹果增长", layer="context")
        self.assertTrue(rows)
        self.assertEqual(audit["rerank_status"], "failed")
        self.assertIn("rrf_without_rerank:RuntimeError", audit["mode"])
        self.assertEqual(audit["rerank_seed_ids"][0], audit["coarse_pages"][0]["chunk_ids"][0])
        self.assertEqual(audit["rerank_calls"][0]["status"], "failed")

    async def test_long_query_is_not_truncated_or_sent_to_rerank(self):
        from unittest.mock import AsyncMock
        self.api.rerank = AsyncMock()
        rows, audit = await self.rag.retrieve(self.state, "苹果" * 2100, layer="context")
        self.api.rerank.assert_not_awaited()
        self.assertTrue(rows)
        self.assertEqual(audit["rerank_error_type"], "RerankBudgetError")
        self.assertIn("query", audit["rerank_error_reason"])
        self.assertEqual(audit["rerank_scored_candidate_count"], 0)

    async def test_pagination_settings_and_empty_space_audit(self):
        with patch.dict(os.environ, {"SESSION_EVIDENCE_COARSE_PAGE_SIZE": "128", "SESSION_EVIDENCE_COARSE_PAGES": "3"}):
            settings = SessionEvidenceSettings.from_env()
        self.assertEqual(settings.coarse_page_size * settings.coarse_pages, 384)
        with self.assertRaises(ValueError):
            SessionEvidenceSettings(coarse_page_size=1, coarse_pages=1)
        new = {"original_query": "空主题", "query": "空主题", "raw_sources": [],
               "session_task": {"topic_relation": "new"}}
        await self.rag.attach(new, run_id=str(uuid.uuid4()))
        rows, audit = await self.rag.retrieve(new, "苹果", layer="citation")
        self.assertEqual(rows, [])
        self.assertEqual(audit["coarse_pages"], [])
        self.assertEqual(audit["rerank_status"], "not_needed")

    async def test_neighbor_read_is_scoped_and_same_document(self):
        row=next(r for r in self.rows({"paragraph"}) if r["metadata"]["next_chunk_id"])
        packet=self.store.adjacent(self.uid,self.sid,self.rag.topic(self.state),row["id"],radius=1)
        self.assertTrue(packet["neighbors"])
        self.assertTrue(all(r["metadata"]["document_id"]==row["metadata"]["document_id"] for r in packet["neighbors"]))
        with self.assertRaises(ValueError): self.store.adjacent(self.uid,self.sid,"another_topic",row["id"])
        with self.assertRaises(ValueError): self.store.adjacent(uuid.uuid4(),self.sid,self.rag.topic(self.state),row["id"])

    async def test_exact_original_read_for_table_and_sentence(self):
        rows=self.rows({"sentence","table"})
        refs=await self.rag.references(self.state,rows)
        self.assertEqual([r["quote"] for r in refs],[r["content"] for r in rows])
        with self.assertRaises(ValueError): await self.rag.resolve_units(self.state,"wrong_document",[rows[0]["id"]])

    async def test_index_reuse_does_not_reembed(self):
        before=len(self.api.embedded)
        await self.rag.ingest(self.state)
        self.assertEqual(before,len(self.api.embedded))
        self.assertEqual(self.state["evidence_rag_metrics"][-1]["added_chunks"],0)

    async def test_document_versions_do_not_overwrite_old_citations(self):
        newer=source(self.original_source["content"].replace("销量增长", "销量下降"))
        self.state["raw_sources"].append(newer)
        await self.rag.ingest(self.state)
        docs=self.store.documents(self.uid,self.sid,self.rag.topic(self.state))
        self.assertEqual(len(docs),2)
        self.assertEqual({d["document_id"] for d in docs},{d["document_id"] for d in self.state["raw_sources"]})

    async def test_ttl_boundary_and_read_does_not_extend(self):
        original=self.store.ensure(self.uid,self.sid)["expires_at"]
        self.now+=timedelta(hours=23)
        self.assertEqual(original,self.store.ensure(self.uid,self.sid)["expires_at"])
        self.now+=timedelta(hours=1)
        with self.assertRaises(EvidenceExpiredError): self.rows()

    async def test_new_message_before_deadline_extends(self):
        self.now+=timedelta(hours=23)
        with self.factory() as db:
            db.get(ChatSession,self.sid).expires_at=self.now+timedelta(hours=24); db.commit()
        self.now+=timedelta(hours=2)
        self.assertTrue(self.rows())
        self.assertEqual(self.store.ensure(self.uid,self.sid)["expires_at"], (self.now+timedelta(hours=22)).isoformat())

    async def test_absolute_ttl_does_not_slide(self):
        with self.factory() as db:
            db.get(SessionEvidenceSpace,self.sid).ttl_mode="absolute"
            db.get(ChatSession,self.sid).expires_at=self.now+timedelta(hours=48); db.commit()
        self.now+=timedelta(hours=24)
        with self.assertRaises(EvidenceExpiredError): self.rows()

    async def test_close_immediately_rejects_and_generation_blocks_commit(self):
        old=self.state["evidence_space"]["generation"]
        result=close_session_evidence(self.uid,self.sid,session_factory=self.factory)
        self.assertEqual(result["status"],"closed")
        with self.assertRaises(EvidenceExpiredError): self.rows()
        with self.assertRaises(EvidenceExpiredError): self.store.mark_indexed(self.uid,self.sid,old,[])
        with self.assertRaises(EvidenceExpiredError): await self.rag.ingest(self.state)

    async def test_purge_retry_and_full_scope_cleanup(self):
        with self.factory() as db:
            db.add(ResearchRun(run_id="r1",session_id=str(self.sid),user_id=self.uid,status="running",fencing_token=1,next_event_seq=0))
            db.commit()
        close_session_evidence(self.uid,self.sid,session_factory=self.factory)
        self.vectors.fail=True
        with patch("service.research_control.request_cancel"):
            self.assertEqual(process_evidence_purges(session_factory=self.factory,vectors=self.vectors,checkpoint_delete=lambda ids:None),0)
            self.vectors.fail=False
            calls=[]
            with patch("service.session_memory_service.get_session_memory_service"):
                self.assertEqual(process_evidence_purges(session_factory=self.factory,vectors=self.vectors,checkpoint_delete=lambda ids:calls.extend(ids)),1)
        self.assertEqual(calls,["r1"])
        with self.factory() as db:
            self.assertEqual(db.execute(select(SessionEvidenceDocument)).scalars().all(),[])
            self.assertEqual(db.execute(select(SessionEvidenceChunk)).scalars().all(),[])
            self.assertEqual(db.get(SessionEvidencePurge,str(self.sid)).status,"purged")
        self.assertEqual(self.vectors.rows,{})

    async def test_expiry_cleanup_queues_purge(self):
        self.now+=timedelta(hours=24)
        self.assertEqual(cleanup_expired_evidence(session_factory=self.factory,now=lambda:self.now),1)
        with self.factory() as db: self.assertEqual(db.get(SessionEvidencePurge,str(self.sid)).status,"pending")

    async def test_new_service_reuses_chunks_without_restoring_full_text(self):
        await self.rag.commit_stage(self.state)
        reloaded=json.loads(json.dumps(self.state))
        second=SessionEvidenceRAG(str(self.uid),str(self.sid),store=self.store,api=API(),vectors=self.vectors,counter=self.counter)
        with patch.object(self.store, "documents", side_effect=AssertionError("禁止恢复来源全文")):
            await second.prepare_stage(reloaded, "writing", "test")
            prompt = await second.evidence_prompt(reloaded, "苹果销量")
        self.assertIn("苹果销量增长", prompt)
        self.assertNotIn("content",reloaded["raw_sources"][0])
        self.assertNotIn("sentence_index",reloaded["raw_sources"][0])
        await second.close()


    async def test_32_turns_reuse_session_and_immutable_ids(self):
        ids={r["id"] for r in self.rows()}
        for turn in range(32):
            self.now+=timedelta(minutes=5)
            with self.factory() as db:
                db.get(ChatSession,self.sid).expires_at=self.now+timedelta(hours=24); db.commit()
            await self.rag.prepare_stage(self.state, "writing", "test")
            self.state["session_task"]={"topic_relation":"continue","resolved_query":f"苹果销量增长，追问{turn}"}
            with bind_session_evidence(self.rag):
                prompt=await evidence_prompt(self.state)
            self.assertIn(BEGIN,prompt)
            self.assertLessEqual(self.counter.count_text(prompt),self.rag.settings.context_tokens+2000)
            await self.rag.commit_stage(self.state)
            self.assertEqual(ids,{r["id"] for r in self.rows()})
        self.assertEqual(self.state["evidence_space"]["topic_id"],self.rag.topic(self.state))

    async def test_failed_vector_backend_uses_bounded_lexical_view(self):
        self.vectors.fail=True
        rows,audit=await self.rag.retrieve(self.state,"苹果增长",layer="context",neighbors=True)
        self.assertTrue(rows); self.assertIn("lexical_only",audit["mode"])
        self.assertLessEqual(audit["selected_tokens"],audit["budget_tokens"])

    async def test_atomic_claim_triggers_fine_retrieval_and_invalid_id_is_preserved(self):
        self.state["claims"]=[{"claim":"2025年中国市场苹果销量增长", "evidence":[]},
                              {"claim":"销量下降", "evidence":[{"document_id":"fake","evidence_unit_ids":["fake"]}]}]
        claims=await self.rag.claim_inputs(self.state)
        self.assertTrue(claims[0]["references"])
        self.assertEqual(claims[1]["binding_errors"],["invalid_evidence_unit_ids_or_document"])
        self.assertTrue(any(m.get("layer")=="citation" for m in self.state["evidence_rag_metrics"]))

    async def test_request_budget_drops_whole_units(self):
        with bind_session_evidence(self.rag):
            prompt=await evidence_prompt(self.state)
            with patch("service.agent_context.scoped_input_budget",return_value=6000):
                result=fit_evidence_prompt("系统",prompt,None,"LeadWriter",1000)
        payload=json.loads(result.split(BEGIN)[1].split(END)[0])
        originals={r["id"]:r["content"] for r in self.rows()}
        self.assertLessEqual(self.counter.count_messages([{"content":"系统"},{"content":result}]),6000)
        for row in payload["evidence_units"]: self.assertEqual(row["text"],originals[row["id"]])

    async def test_new_topic_isolated_and_return_to_previous_reuses(self):
        old=deepcopy(self.state)
        new={"original_query":"新主题", "query":"新主题", "raw_sources":[], "session_task":{"topic_relation":"new"}}
        await self.rag.attach(new,run_id=str(uuid.uuid4()))
        rows,_=await self.rag.retrieve(new,"苹果增长",layer="context")
        self.assertEqual(rows,[])
        await self.rag.attach(new,parent=old)
        rows,_=await self.rag.retrieve(new,"苹果增长",layer="context")
        self.assertTrue(rows)

    async def test_graph_node_uses_chunks_and_keeps_only_document_refs(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from service.deep_research_v2.graph import build_graph,graph_config,pack_state
        from service.deep_research_v2.state import create_initial_state
        from test_单主链路图 import make_agents,passthrough_stage
        captured=[]
        class Writer:
            async def process(writer,state):
                self.assertNotIn("content",state["raw_sources"][0])
                captured.append(await evidence_prompt(state))
                state["final_report"]="测试报告正文"
                return state
        initial=create_initial_state("苹果增长",str(self.sid),[])
        initial["raw_sources"]=deepcopy(self.state["raw_sources"])
        initial["evidence_space"]=deepcopy(self.state["evidence_space"])
        initial["execution_plan"]="edit"
        initial["writing_mode"]="user_edit"
        graph=build_graph(make_agents(writer=Writer()),InMemorySaver(),run_id=str(uuid.uuid4()),
                          user_id=str(self.uid),session_id=str(self.sid),execute_stage=passthrough_stage,
                          evidence_service=self.rag)
        config=graph_config(str(uuid.uuid4()))
        result=await graph.ainvoke(pack_state(initial),config)
        self.assertEqual(result["research"]["status"],"completed")
        self.assertIn(BEGIN,captured[0])
        self.assertNotIn("content",result["research"]["raw_sources"][0])
        snapshot=await graph.aget_state(config)
        self.assertTrue(snapshot.values["research"]["raw_sources"][0]["evidence_document_ref"])

    async def test_first_checkpoint_keeps_nested_parent_as_source_refs(self):
        new=source("另一份报告。该报告的证据块应直接提供给写作。",url="https://example.com/parent")
        self.state["session_parent"]={"raw_sources":[new],"evidence_space":deepcopy(self.state["evidence_space"])}
        await self.rag.commit_stage(self.state)
        self.assertNotIn("content",self.state["session_parent"]["raw_sources"][0])
        self.assertTrue(self.state["session_parent"]["raw_sources"][0]["rag_usable"])
        self.assertEqual(self.store.documents(self.uid,self.sid,self.rag.topic(self.state),document_ids=[new["document_id"]])[0]["content"],new["content"])


    async def test_table_unit_id_survives_normalization_and_claim_verification(self):
        from service.deep_research_v2.writer_claims import normalize_claims
        from service.deep_research_v2.citation_verification import verify_stage_citations
        from unittest.mock import AsyncMock
        table=self.rows({"table"})[0]
        self.state["claims"]=normalize_claims([{"claim":"2025年销量为100万台","evidence":[{
            "document_id":table["metadata"]["document_id"],"evidence_unit_ids":[table["id"]]}]}])
        self.state["citation_verification_config"]={"report_claims":True}
        extraction = {"status": "completed", "units_total": 1, "claims": [{
            "claim_id": "claim_1", "claim": "2025年销量为100万台", "section_id": "sec_1",
            "assertion_kind": "fact", "report_quote": "2025年销量为100万台",
            "citation_urls": [table["metadata"]["source_url"]], "binding_errors": []}]}
        verifier=AsyncMock(return_value={"status":"incomplete","claims":[]})
        async def select_all(claim, refs, settings):
            # 本用例只验证表格证据编号经过核验入口后不丢失；选句模型单独注入，
            # 不让集成回归访问真实模型或依赖本机 tokenizer 下载。
            claim["evidence_selection"] = {"status": "completed", "test_selection": "all"}
            return [{**ref, "sentence_selection": {"status": "completed", "test_selection": "all"}}
                    for ref in refs]
        with bind_session_evidence(self.rag), \
                patch.object(self.rag, "retrieve", AsyncMock(return_value=([table], {}))), \
                patch("service.deep_research_v2.report_claim_extraction.extract_report_claims", AsyncMock(return_value=extraction)), \
                patch("service.deep_research_v2.report_sentence_evidence.select_report_sentences", select_all), \
                patch("service.deep_research_v2.citation_verification.verify_citations",verifier):
            await verify_stage_citations(self.state,"report_claims")
        refs=verifier.call_args.args[0][0]["references"]
        table_ref = next(ref for ref in refs if table["id"] in ref["evidence_unit_ids"])
        self.assertEqual(table_ref["quote"],table["content"])
        self.assertIn("销量（万台）",table_ref["quote"])
        self.assertEqual(self.state["verified_citation_bindings"],[])

    async def test_unindexable_source_stays_in_store_and_state_records_error(self):
        new=source("字"*2000+"。",url="https://example.com/long")
        self.state["raw_sources"].append(new)
        await self.rag.ingest(self.state)
        self.assertTrue(self.state["evidence_rag_metrics"][-1]["chunk_errors"])
        await self.rag.commit_stage(self.state)
        self.assertNotIn("content",self.state["raw_sources"][-1])
        self.assertFalse(self.state["raw_sources"][-1]["rag_usable"])
        saved = self.store.documents(self.uid,self.sid,self.rag.topic(self.state),document_ids=[new["document_id"]])
        self.assertEqual(saved[0]["content"],new["content"])


    async def test_rewrite_can_only_reuse_original_table_unit_ids(self):
        from service.deep_research_v2.citation_repair import parse_edits
        table = self.rows({"table"})[0]
        ref = {"document_id": table["metadata"]["document_id"], "evidence_unit_ids": [table["id"]]}
        packet = [{"unit_id": "u", "text": "原文。", "claims": [{"evidence": [ref]}]}]
        response = {"edits": [{"unit_id": "u", "content": "2025年销量为100万台。", "claims": [{
            "claim": "2025年销量为100万台。", "report_quote": "2025年销量为100万台。", "evidence": [deepcopy(ref)]}]}]}
        self.assertEqual(len(parse_edits(response, packet)), 1)
        response["edits"][0]["claims"][0]["evidence"][0]["evidence_unit_ids"] = ["other"]
        with self.assertRaises(ValueError): parse_edits(response, packet)

    async def test_current_resolved_query_and_retrieval_metrics(self):
        self.state["session_task"] = {"resolved_query": "2026年销量下降", "topic_relation": "continue"}
        with bind_session_evidence(self.rag):
            prompt = await evidence_prompt(self.state)
        audit = self.state["evidence_rag_metrics"][-1]
        self.assertEqual(audit["query"], "2026年销量下降")
        self.assertGreater(audit["corpus_characters"], 0)
        self.assertLessEqual(audit["selection_char_fraction"], 1)
        rows = self.rows({"table"})
        refs = await self.rag.references(self.state, rows)
        self.assertEqual(refs[0]["source_url"], "https://example.com/report")

    async def test_close_sends_cancel_without_waiting_for_purge(self):
        with self.factory() as db:
            db.add(ResearchRun(run_id="cancel-now", session_id=str(self.sid), user_id=self.uid,
                               status="running", fencing_token=1, next_event_seq=0))
            db.commit()
        with patch("service.research_control.request_cancel") as cancel:
            close_session_evidence(self.uid, self.sid, session_factory=self.factory)
        cancel.assert_called_once_with("cancel-now", str(self.sid))
        with self.factory() as db:
            self.assertEqual(db.get(SessionEvidencePurge, str(self.sid)).status, "pending")

    async def test_fine_recall_runs_without_claiming_verification_when_nli_disabled(self):
        from service.deep_research_v2.citation_verification import verify_stage_citations
        self.state["claims"] = [{"claim": "2026年销量下降", "evidence": []}]
        self.state["citation_verification_config"] = {"report_claims": False}
        with bind_session_evidence(self.rag), patch("service.nli_verifier.verify_claims") as nli:
            result = await verify_stage_citations(self.state, "report_claims")
        nli.assert_not_called()
        self.assertEqual(result["status"], "not_executed")
        self.assertEqual(self.state["citation_retrieval_status"]["status"], "completed")
        self.assertTrue(self.state["retrieved_citation_bindings"][0]["evidence"])
        self.assertEqual(self.state["verified_citation_bindings"], [])

    async def test_writer_section_task_uses_refs_without_restoring_full_text(self):
        from langgraph.graph import StateGraph, START, END as GRAPH_END
        from langgraph.checkpoint.memory import InMemorySaver
        from service.deep_research_v2.graph import GraphState, pack_state, graph_config
        from service.deep_research_v2.agents import writer as module
        import logging
        writer = module.LeadWriter.__new__(module.LeadWriter)
        writer.name, writer.model = "LeadWriter", "测试模型"
        writer.logger = logging.getLogger("test.rag.section")
        captured = []
        async def write(work, section):
            self.assertNotIn("content", work["raw_sources"][0])
            work.setdefault("draft_sections", {})[section["id"]] = "本章节正文。"
            work.setdefault("section_claims", {})[section["id"]] = []
        writer._write_section = write
        task = module.task
        def recording_task(**kwargs):
            decorate = task(**kwargs)
            def recording(fn):
                async def observed(payload):
                    captured.append(deepcopy(payload))
                    return await fn(payload)
                return decorate(observed)
            return recording
        initial = deepcopy(self.state)
        initial["outline"] = [{"id": "sec_1", "title": "章节", "status": "pending"}]
        self.rag.externalize(initial)
        async def node(graph_state):
            work = deepcopy(graph_state["research"])
            await self.rag.prepare_stage(work, "writing", "test")
            with bind_session_evidence(self.rag):
                await writer._write_checkpointed_section(work, work["outline"][0])
            self.rag.externalize(work)
            return pack_state(work)
        graph = StateGraph(GraphState); graph.add_node("write", node)
        graph.add_edge(START, "write"); graph.add_edge("write", GRAPH_END)
        compiled = graph.compile(checkpointer=InMemorySaver())
        with patch.object(module, "task", recording_task):
            result = await compiled.ainvoke(pack_state(initial), graph_config(str(uuid.uuid4())))
        self.assertEqual(result["research"]["draft_sections"]["sec_1"], "本章节正文。")
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0]["research"]["raw_sources"][0]["evidence_document_ref"])
        self.assertNotIn("content", captured[0]["research"]["raw_sources"][0])

    async def test_legacy_table_row_citation_is_upgraded_with_header(self):
        from service.web_document import sentence_records
        src = self.original_source
        row = next(row for row in sentence_records(src) if "2025 | 100" in row["text"])
        self.state["claims"] = [{"claim": "2025年销量为100万台", "evidence": [{
            "document_id": src["document_id"], "sentence_ids": [row["id"]]}]}]
        inputs = await self.rag.claim_inputs(self.state)
        declared = [ref for ref in inputs[0]["references"] if not ref.get("retrieval_candidate")]
        self.assertEqual(len(declared), 1)
        self.assertEqual(declared[0]["unit_type"], "table")
        self.assertIn("销量（万台）", declared[0]["quote"])
        self.assertIn("仅统计中国市场", declared[0]["quote"])

    async def test_malformed_legacy_table_is_not_scored_without_header(self):
        from service.web_document import sentence_records
        src = source("| 年份 | 销量 |\n| --- | --- |\n| 2025 | 100 | 错列 |\n", url="https://example.com/bad-table")
        self.state["raw_sources"].append(src)
        await self.rag.ingest(self.state)
        row = next(row for row in sentence_records(src) if "错列" in row["text"])
        self.state["claims"] = [{"claim": "2025年销量为100万台", "evidence": [{
            "document_id": src["document_id"], "sentence_ids": [row["id"]]}]}]
        inputs = await self.rag.claim_inputs(self.state)
        self.assertIn("invalid_evidence_unit_ids_or_document", inputs[0]["binding_errors"])
        self.assertFalse(any("错列" in ref["quote"] for ref in inputs[0]["references"]))

    async def test_legacy_plain_sentence_remains_exact(self):
        from service.web_document import sentence_records
        src = self.original_source
        row = next(row for row in sentence_records(src) if row["text"] == "苹果销量增长。")
        self.state["claims"] = [{"claim": row["text"], "evidence": [{
            "document_id": src["document_id"], "sentence_ids": [row["id"]]}]}]
        inputs = await self.rag.claim_inputs(self.state)
        self.assertEqual(inputs[0]["references"][0]["quote"], row["text"])
        self.assertEqual(inputs[0]["binding_errors"], [])

    async def test_long_table_neighbors_follow_rows_instead_of_repeated_header(self):
        text = "表1 销量\n| 年份 | 销量 |\n| --- | --- |\n" + "\n".join(f"| {2000+i} | {i} |" for i in range(20)) + "\n注：单位为万台。"
        rows = chunk_document(source(text), user_id=str(self.uid), session_id=str(self.sid),
                              topic_id=self.rag.topic(self.state), counter=self.counter, chunk_tokens=100)
        tables = sorted((r for r in rows if r["unit_type"] == "table"), key=lambda r:r["metadata"]["row_range"][0])
        self.assertGreater(len(tables), 2)
        for i, row in enumerate(tables):
            for layer in ("context", "citation"):
                adjacent = row["metadata"]["adjacent_chunk_ids"][layer]
                self.assertEqual(adjacent["next_chunk_id"], tables[i+1]["id"] if i+1<len(tables) else None)
                self.assertEqual(adjacent["prev_chunk_id"], tables[i-1]["id"] if i else None)


@unittest.skipUnless(os.getenv("RUN_SESSION_EVIDENCE_POSTGRES")=="1","需显式启用隔离PostgreSQL回归")
class PostgreSQLTemporaryRAGTests(TemporaryRAGTests):
    def create_engine(self):
        from sqlalchemy import text
        from core.database import engine as configured
        schema="临时证据验证_"+uuid.uuid4().hex
        with configured.begin() as db: db.execute(text(f'CREATE SCHEMA "{schema}"'))
        def cleanup():
            with configured.begin() as db: db.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        self.addCleanup(cleanup)
        return create_engine(configured.url,connect_args={"options":f'-csearch_path={schema}'})


@unittest.skipUnless(os.getenv("RUN_SESSION_EVIDENCE_MILVUS")=="1","需显式启用独立Milvus集合回归")
class MilvusAdapterTests(unittest.TestCase):
    def test_native_search_pagination_three_pages_and_partial_fourth(self):
        import math
        from pymilvus import utility
        from service.session_evidence_vectors import SessionEvidenceVectors
        from service.session_evidence_chunks import stable_id
        index = SessionEvidenceVectors(2, "独立分页验证_" + uuid.uuid4().hex)
        uid, sid, topic = [str(uuid.uuid4()) for _ in range(3)]
        rows = [{"id": stable_id(f"分页{i}"), "unit_type": "sentence", "metadata": {
            "user_id": uid, "session_id": sid, "topic_id": topic, "document_id": "paged_doc",
            "evidence_space_generation": 1}} for i in range(400)]
        try:
            index.upsert(rows, [[math.cos(i * .002), math.sin(i * .002)] for i in range(400)])
            pages = [index.search(uid, sid, topic, 1, [1., 0.], {"sentence"}, 128, offset=n * 128)
                     for n in range(4)]
            self.assertEqual([len(page) for page in pages], [128, 128, 128, 16])
            self.assertEqual([cid for page in pages for cid in page], [row["id"] for row in rows])
            self.assertEqual(index.search(uid, sid, topic, 1, [1., 0.], {"paragraph"}, 128), [])
        finally:
            if utility.has_collection(index.collection_name, using=index.alias):
                utility.drop_collection(index.collection_name, using=index.alias)

    def test_native_upsert_scope_generation_and_delete(self):
        from service.session_evidence_vectors import SessionEvidenceVectors
        from service.session_evidence_chunks import stable_id
        from pymilvus import utility
        index=SessionEvidenceVectors(2,"独立机制验证_"+uuid.uuid4().hex)
        uid,sid,other_sid,topic=[str(uuid.uuid4()) for _ in range(4)]
        def row(name,session,generation):
            return {"id":stable_id(name),"unit_type":"sentence","metadata":{
                "user_id":uid,"session_id":session,"topic_id":topic,"document_id":"doc_example",
                "evidence_space_generation":generation}}
        a,b,c=row("a",sid,1),row("b",other_sid,1),row("c",sid,2)
        try:
            index.upsert([a,b,c],[[1.,0.],[1.,0.],[1.,0.]])
            self.assertEqual(index.search(uid,sid,topic,1,[1.,0.],{"sentence"},10),[a["id"]])
            self.assertEqual(index.search(str(uuid.uuid4()),sid,topic,1,[1.,0.],{"sentence"},10),[])
            index.delete_session(uid,sid)
            self.assertEqual(index.search(uid,sid,topic,1,[1.,0.],{"sentence"},10),[])
            self.assertEqual(index.search(uid,other_sid,topic,1,[1.,0.],{"sentence"},10),[b["id"]])
        finally:
            if utility.has_collection(index.collection_name,using=index.alias):
                utility.drop_collection(index.collection_name,using=index.alias)


if __name__=="__main__": unittest.main()
