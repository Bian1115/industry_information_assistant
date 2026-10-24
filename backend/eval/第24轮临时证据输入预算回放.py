"""用冻结的第24轮失败材料回放Writer输入构造；不生成答案、不调用收费API。"""
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sys
import time
import uuid

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "eval/tests")]
from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")
logging.disable(logging.CRITICAL)

from test_会话临时双层RAG import API, Vectors, jsonb_sqlite
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from models.user import User
from models.chat import ChatSession
from models.session_evidence import SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk, SessionEvidencePurge
from service.session_evidence_store import SessionEvidenceStore
from service.session_evidence_rag import SessionEvidenceRAG, SessionEvidenceSettings, bind_session_evidence, fit_evidence_prompt, BEGIN, END
from service.qwen_token_counter import get_qwen_token_counter
from service.agent_history import HistoryScope
from service.agent_context import protected_agent_messages, scoped_input_budget
from service.deep_research_v2.agents.writer import LeadWriter


class CapturedInput(Exception):
    pass


async def main():
    started = time.monotonic()
    original_root = ROOT / "backend/eval/data/第20题32轮待合并结果恢复续跑-20261008-184009"
    source_root = original_root / "真实运行/第24轮"
    output = ROOT / "backend/eval/data/会话临时双层RAG实现验证-20261008"
    if (output / "第24轮失败材料输入预算回放结果.json").exists():
        output = output / ("第24轮预算补充回放-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((source_root / "研究状态.json").read_text())
    history = json.loads((source_root / "工作历史.json").read_text())
    failed = next(row["payload"] for row in history if row["kind"] == "llm_input" and
                  row["call_id"] == "be31baa2-a510-4f40-ae80-48ec703d2a88")
    prior = json.loads((original_root / "第24轮超限输入构成与压缩分析.json").read_text())
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    counter = get_qwen_token_counter()
    uid, sid, now = uuid.uuid4(), uuid.uuid4(), datetime.now(timezone.utc)
    for model in (User, ChatSession, SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk, SessionEvidencePurge):
        model.__table__.create(engine)
    with factory() as db:
        db.add(User(id=uid, username="离线机制回放", email="replay@example.invalid", hashed_password="unused")); db.flush()
        db.add(ChatSession(id=sid, user_id=uid, expires_at=now + timedelta(hours=24))); db.commit()
    state["session_id"] = str(sid)
    state.pop("evidence_space", None)
    state["evidence_rag_metrics"] = []
    rag = SessionEvidenceRAG(str(uid), str(sid), settings=SessionEvidenceSettings(),
            store=SessionEvidenceStore(factory), api=API(), vectors=Vectors(), counter=counter)
    try:
        await rag.attach(state, run_id=str(uuid.uuid4()))
        await rag.ingest(state)
        scope = HistoryScope(store=None, run_id="离线输入预算回放", session_id=str(sid), user_id=str(uid),
                             node="revise", contract=failed["contract"], context_policy=state.get("context_policy"))
        baseline = counter.count_messages(protected_agent_messages(scope, failed["system"], failed["content"]))
        captured = {}
        writer = LeadWriter.__new__(LeadWriter)
        writer.name, writer.model = "LeadWriter", "离线构造测试"
        writer.logger = logging.getLogger("replay.rag")
        async def capture(**kwargs):
            prompt = fit_evidence_prompt(kwargs["system_prompt"], kwargs["user_prompt"], scope, "LeadWriter", kwargs["max_tokens"])
            captured.update(system=kwargs["system_prompt"], content=prompt)
            raise CapturedInput()
        writer.call_llm = capture
        with bind_session_evidence(rag):
            try:
                await writer._edit_existing_report(deepcopy(state))
            except CapturedInput:
                pass
        new_tokens = counter.count_messages(protected_agent_messages(scope, captured["system"], captured["content"]))
        evidence = captured["content"].split(BEGIN, 1)[1].split(END, 1)[0]
        payload = json.loads(evidence)
        with factory() as db:
            docs = list(db.execute(select(SessionEvidenceDocument)).scalars())
            chunks = list(db.execute(select(SessionEvidenceChunk)).scalars())
        failed_docs = [doc.payload.get("rag_chunk_error") for doc in docs if doc.payload.get("rag_chunk_error")]
        counts = {}
        for chunk in chunks:
            counts[chunk.unit_type] = counts.get(chunk.unit_type, 0) + 1
        result = {"date": datetime.now().astimezone().isoformat(), "scope": "冻结真实失败材料的离线输入预算机制回放，未生成答案，未运行真实32轮",
            "original_state": str(source_root / "研究状态.json"), "original_history": str(source_root / "工作历史.json"),
            "tokenizer": counter.model_name, "settings": rag.settings.model_dump(),
            "retrieval_backend": "注入二维向量及词法精排，仅验证预算/定位；不测量语义召回质量", "paid_api_calls": 0,
            "documents": len(docs), "chunks": len(chunks), "chunk_types": counts,
            "unindexable_documents": len(failed_docs), "chunk_errors": failed_docs,
            "frozen_original_input_tokens": baseline, "prior_recorded_protected_tokens": prior["protected_current_input_tokens"],
            "prior_recorded_final_context_tokens": prior["final_context_tokens"], "prior_recorded_input_cap": prior["business_input_cap"],
            "new_protected_input_tokens": new_tokens, "protected_input_reduction_fraction": 1 - new_tokens / baseline,
            "prior_full_evidence_text_tokens": next(p["text_tokens"] for p in prior["parts"] if p["part"] == "可用证据"),
            "new_evidence_payload_text_tokens": counter.count_text(evidence), "selected_units": len(payload["evidence_units"]),
            "input_cap": scoped_input_budget(scope, "LeadWriter", writer.max_tokens),
            "headroom_tokens": scoped_input_budget(scope, "LeadWriter", writer.max_tokens) - new_tokens,
            "prompt_budget": rag.last_prompt_budget, "ingest_audit": state["evidence_rag_metrics"],
            "duration_seconds": time.monotonic() - started,
            "bounds": "只测量当前受保护输入，不包含恢复后的历史摘要和最近消息；未评估答案质量、引用支持率、召回率、真实embedding/rerank/NLI费用或30+轮容量。"}
        (output / "第24轮RAG编辑输入.json").write_text(json.dumps(captured, ensure_ascii=False, indent=2))
        (output / "第24轮失败材料输入预算回放结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in {"ingest_audit", "chunk_errors"}}, ensure_ascii=False, indent=2))
    finally:
        await rag.close()
        engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
