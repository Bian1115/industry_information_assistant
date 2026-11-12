"""DeepResearch 的任务信息、证据视图和报告基础检查。

LLM 负责理解、提炼和写作；本模块负责不可交给概率模型的部分：
保留任务与格式要求、确认来源读取成功，以及检查正文链接的来源范围。

本模块不判定论断与原文的语义支持性；可选审计由 citation_verification
独立执行。它不参与这里的发布门禁，也不自动删除或修复报告。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping
from urllib.parse import urlsplit


from .evidence import (
    LOW_QUALITY_DOMAINS, normalize_url,
    normalize_evidence_state, fact_views, data_point_views,
    source_indexes, resolve_source,
)
from .source_scope import url_in_scope


def candidate_source_score(item: Mapping[str, Any]) -> int:
    """搜索候选的确定性优先级：原始/官方来源先读，文库自媒体最后读。"""
    url = str(item.get("url") or "")
    host = (urlsplit(url).hostname or "").lower()
    label = f"{item.get('site_name', '')} {item.get('title', '')}".lower()
    if any(host == domain or host.endswith(f".{domain}") for domain in LOW_QUALITY_DOMAINS):
        return -100
    score = 0
    if host.endswith(".gov.cn") or host.endswith(".edu.cn"):
        score += 80
    if any(token in label for token in ("官网", "官方", "年报", "annual report", "监管", "统计局")):
        score += 50
    if url.lower().endswith(".pdf"):
        score += 20
    if any(token in label for token in ("文库", "论坛", "自媒体", "博客")):
        score -= 30
    return score


def build_task_context(planning_result: Mapping[str, Any], query: str) -> Dict[str, Any]:
    """只保留任务原话和明确格式要求，不生成量化契约或覆盖验收项。"""
    raw = planning_result.get("research_contract")
    raw = raw if isinstance(raw, Mapping) else {}
    constraints = raw.get("active_constraints")
    constraints = constraints if isinstance(constraints, Mapping) else {}
    limit = constraints.get("max_body_chars")
    context = {
        "version": 1, "query": query,
        "active_constraints": {"max_body_chars": limit}
        if type(limit) is int and 100 <= limit <= 100000 else {},
    }
    for field in ("language", "output_format"):
        if isinstance(raw.get(field), str) and raw[field].strip():
            context[field] = raw[field].strip()
    return context

def bind_verified_evidence(state: Dict[str, Any]) -> Dict[str, int]:
    """规范化来源引用并筛选事实/数据；证据信息按需查询，不另存账本。"""
    normalize_evidence_state(state)
    facts = [fact for fact in state.get("facts", []) or [] if isinstance(fact, dict)]
    views = fact_views(state)
    kept = []
    for fact, view in zip(facts, views):
        if view["reader_verified"]:
            fact["reader_verified"] = True
            kept.append(fact)
    state["facts"] = kept
    state["data_points"] = [point for point in data_point_views(state) if point["reader_verified"]]
    return {"facts_kept": len(kept), "facts_removed": len(facts) - len(kept),
            "evidence_entries": len(kept)}




MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\((https?://[^)\s]+)\)")
def evaluate_report_gate(state: Mapping[str, Any]) -> Dict[str, Any]:
    """写作后的确定性检查：报告非空，且正文链接都落在允许的来源网站内。

    这里**不做**引用核验：不判断论断是否被原文支持，也不判断该 URL 是否
    真的被完整读取过。它只回答"报告是不是空的"和"有没有引用到范围外的
    网站"，因此 passed 只能说明这两件事，不能被展示成"引用核验通过"。
    """
    report = str(state.get("final_report") or "")
    domains = list(state.get("allowed_domains") or [])
    cited = {normalize_url(url) for _, url in MARKDOWN_LINK.findall(report)}
    cited.discard("")
    out_of_scope = sorted(url for url in cited if not url_in_scope(url, domains))
    blocking = []
    if not report.strip():
        blocking.append("报告为空")
    if out_of_scope:
        blocking.append(f"正文引用了 {len(out_of_scope)} 个来源范围外的链接")
    return {
        "gate": "report",
        "scope": "report_non_empty_and_source_scope",
        "citation_verification": "not_executed",
        "passed": not blocking,
        "blocking_reasons": blocking,
        "citations_total": len(cited),
        "citations_in_scope": len(cited) - len(out_of_scope),
        "out_of_scope_citations": out_of_scope,
    }


def deterministic_critic_issues(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """把确定性门禁的阻塞原因转成审核问题；不包含任何引用核验结论。"""
    issues: List[Dict[str, Any]] = []
    gates = [evaluate_report_gate(state)]
    for gate in gates:
        for index, reason in enumerate(gate.get("blocking_reasons", []), 1):
            digest = hashlib.sha1(f"{gate['gate']}:{reason}".encode()).hexdigest()[:8]
            issues.append({
                "issue_id": f"rule_{digest}",
                "target": {"target_type": "existing_section", "chapter_id": None,
                           "chapter_title": "全文", "specific_location": gate["gate"],
                           "insert_after_chapter_id": None, "proposed_chapter_title": None},
                "issue_type": "instruction_violation" if "引用" in reason else "content_omission",
                "severity": "critical",
                "description": reason,
                "evidence": str(gate),
                "modification": "完善报告正文或修正来源链接后再提交报告。",
                "requires_new_search": True,
                "search_query": f"{str(state.get('query') or '').strip()} {reason} 官方原始资料".strip(),
                "rule_based": True,
            })
    return issues


def evidence_fingerprint(state: Mapping[str, Any]) -> str:
    """由事实、实际正文及假设关系生成指纹，不依赖缓存账本或陈旧哈希。"""
    indexes = source_indexes(state)
    facts = []
    for fact in fact_views(state, verified_only=True):
        source = resolve_source(fact, indexes) or {}
        facts.append((str(fact.get("id") or ""), str(fact.get("content") or ""),
                      str(fact.get("source_id") or ""), normalize_url(fact.get("source_url")),
                      source.get("content_sha256", "") if source.get("evidence_document_ref") else
                      hashlib.sha256(str(source.get("content") or "").encode()).hexdigest()))
    fact_ids = {item[0] for item in facts}
    relations = sorted((str(r.get("fact_id") or ""), str(r.get("hypothesis_id") or ""),
                        str(r.get("evidence_type") or ""), str(r.get("evidence_quote") or ""))
                       for r in state.get("hypothesis_evidence", []) or []
                       if isinstance(r, dict) and str(r.get("fact_id") or "") in fact_ids)
    import json
    documents = sorted((str(source.get("document_id") or source.get("id") or ""),
                        normalize_url(source.get("url") or source.get("source_url")),
                        source.get("content_sha256", "") if source.get("evidence_document_ref") else
                        hashlib.sha256(str(source.get("content") or "").encode()).hexdigest(),
                        sorted(str(section) for section in source.get("related_sections") or []))
                       for source in state.get("raw_sources") or [] if isinstance(source, dict))
    payload = json.dumps({"facts": sorted(facts), "relations": relations, "documents": documents}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
