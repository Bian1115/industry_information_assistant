"""追问父成果的授权定位及白名单继承；不共享父运行的可变执行状态。"""
from copy import deepcopy
import hashlib
import json
import uuid
import os
from pydantic import BaseModel, ConfigDict, Field

from .source_scope import url_in_scope
from .hypothesis_policy import hypotheses_enabled

ARTIFACT_FIELDS = (
    'outline', 'research_contract', 'facts', 'raw_sources', 'data_points',
    'hypotheses', 'hypothesis_evidence', 'key_entities', 'research_questions',
    'draft_sections', 'section_claims', 'synthesis_claims', 'claims',
    'final_report', 'references', 'insights', 'conclusion_evidence_map',
    'evidence_space',
)


class SessionInheritanceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    enabled: bool = True
    planning_context_tokens: int = Field(default=3000, ge=512, le=12000)

    @classmethod
    def from_env(cls):
        raw = os.getenv("SESSION_RESEARCH_INHERITANCE_ENABLED", "1")
        if raw not in {"0", "1"}:
            raise ValueError("SESSION_RESEARCH_INHERITANCE_ENABLED 只能为0或1")
        return cls(enabled=raw == "1", planning_context_tokens=int(
            os.getenv("SESSION_RESEARCH_CONTEXT_TOKENS", "3000")))


def planning_context(parent, parent_run_id, token_limit=3000):
    """有界成果索引；完整报告/正文留在业务快照，不当成新事实证据。"""
    from service.qwen_token_counter import get_qwen_token_counter
    counter = get_qwen_token_counter()
    result = {"parent_run_id": parent_run_id, "report_hash": parent["report_hash"],
              "previous_query": str(parent.get("query") or "")[:1600],
              "previous_query_total_chars": len(str(parent.get("query") or "")),
              "note": "旧报告仅用于理解追问；章节正文为节选，不代表全部内容。事实须回到继承的原文证据。",
              "outline": [], "section_excerpts": [], "omitted_items": 0}
    while counter.count_text(json.dumps(result, ensure_ascii=False)) > token_limit - 64:
        if not result["previous_query"]:
            raise ValueError("父研究上下文固定元数据超过预算")
        result["previous_query"] = result["previous_query"][:len(result["previous_query"]) // 2]
    title_by_id = {s.get("id"): s.get("title", "") for s in parent.get("outline") or []}
    candidates = {
        "outline": [{"id": s.get("id"), "title": s.get("title"), "description": str(s.get("description") or "")[:300]}
                    for s in parent.get("outline") or []],
        "section_excerpts": [{"id": key, "title": title_by_id.get(key, ""), "text": str(text)[:500],
                              "total_chars": len(str(text))}
                             for key, text in (parent.get("draft_sections") or {}).items()],
    }
    if not candidates["section_excerpts"] and parent.get("final_report"):
        candidates["section_excerpts"] = [{"id": "report", "text": parent["final_report"][:800],
                                           "total_chars": len(parent["final_report"])}]
    for key, items in candidates.items():
        for item in items:
            result[key].append(item)
            if counter.count_text(json.dumps(result, ensure_ascii=False)) > token_limit - 64:
                result[key].pop()
                result["omitted_items"] += 1
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def latest_parent_run(user_id, session_id, current_run_id):
    """只定位同一用户、同一会话的已交付研究；不接收模型给出的run_id。"""
    from core.database import SessionLocal
    from models.chat import ChatMessage, ChatSession
    with SessionLocal() as db:
        row = (db.query(ChatMessage).join(ChatSession, ChatSession.id == ChatMessage.session_id)
               .filter(ChatSession.user_id == uuid.UUID(user_id),
                       ChatSession.id == uuid.UUID(session_id),
                       ChatMessage.role == 'assistant', ChatMessage.execution_mode == 'research',
                       ChatMessage.research_status == 'completed', ChatMessage.run_id.isnot(None),
                       ChatMessage.run_id != current_run_id)
               .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).first())
        return str(row.run_id) if row else None


def parent_artifacts(snapshot, session_id):
    parent = (snapshot.values or {}).get('research') or {}
    if parent.get('session_id') != session_id or parent.get('status') != 'completed' or snapshot.next:
        raise ValueError('父研究不属于当前会话或尚未完成')
    if not str(parent.get('final_report') or '').strip():
        raise ValueError('父研究没有可修订的报告')
    result = {key: deepcopy(parent.get(key)) for key in ARTIFACT_FIELDS if key in parent}
    result['query'] = parent.get('query', '')
    result['active_constraints'] = deepcopy(parent.get('active_constraints') or {})
    result['allowed_domains'] = deepcopy(parent.get('allowed_domains') or [])
    result['previous_quality'] = {key: deepcopy(parent.get(key)) for key in
                                  ('quality_verdict', 'completion_reason', 'unresolved_issues', 'critic_feedback')}
    result['checkpoint_id'] = ((snapshot.config or {}).get('configurable') or {}).get('checkpoint_id')
    result['report_hash'] = digest(parent['final_report'])
    result['artifacts_hash'] = digest({key: result.get(key) for key in ARTIFACT_FIELDS})
    return result


def inherit_artifacts(state, parent, *, replan=False):
    """继承来源引用和成果元数据，正文仍由会话RAG提供。"""
    fields = ARTIFACT_FIELDS if not replan else ('facts', 'raw_sources', 'data_points', 'key_entities')
    for key in fields:
        if key in parent:
            state[key] = deepcopy(parent[key])
    # 新运行收窄来源范围时，旧资料也必须符合当前范围。
    sources = [s for s in state.get('raw_sources', [])
               if url_in_scope(str(s.get('url') or s.get('source_url') or ''), state['allowed_domains'])]
    source_ids = {s.get('id') for s in sources}
    for source in sources:
        if source.get('evidence_document_ref'):
            # 入库与恢复依赖对账负责检查引用；此处不恢复或重算来源全文。
            for key in ('content', 'full_content', 'sentence_index'):
                source.pop(key, None)
            continue
        stored_hash = source.get('content_sha256')
        content = str(source.get('content') or source.get('full_content') or '')
        if source.get('document_version'):
            from service.web_document import sentence_records
            if not sentence_records(source):
                raise ValueError('继承的来源正文编号或哈希不匹配')
        elif stored_hash and hashlib.sha256(content.encode()).hexdigest() != stored_hash:
            raise ValueError('继承的来源正文哈希不匹配')
    state['raw_sources'] = sources
    state['facts'] = [f for f in state.get('facts', []) if f.get('source_id') in source_ids]
    fact_ids = {f.get('id') for f in state['facts']}
    state['hypothesis_evidence'] = [e for e in state.get('hypothesis_evidence', []) if e.get('fact_id') in fact_ids]
    if not hypotheses_enabled(state):
        state['hypotheses'] = []
        state['hypothesis_evidence'] = []
        state['conclusion_evidence_map'] = []
    # 派生数据需重新计算，防止依赖已被来源过滤掉的输入。
    state['data_points'] = [p for p in state.get('data_points', [])
                            if p.get('source_id') in source_ids and p.get('origin') != 'derived']
    state['references'] = [r for r in state.get('references', [])
                           if url_in_scope(str(r.get('url') or r.get('link') or ''), state['allowed_domains'])]
    state['inherited_evidence_manifest'] = {
        'parent_run_id': state.get('parent_run_id'), 'parent_report_hash': parent['report_hash'],
        'sources': [{'id': s.get('id'), 'sha256': digest(s)} for s in sources],
        'fact_ids': sorted(str(x) for x in fact_ids),
    }
    if replan:
        # 旧章节编号与新大纲尚未建立对应关系；禁止同名编号意外命中。
        for fact in state['facts']:
            fact['related_sections'] = []
