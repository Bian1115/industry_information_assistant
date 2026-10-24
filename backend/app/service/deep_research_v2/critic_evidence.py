"""为报告复审回读全部现有论断的引用片段，不重新检索或执行支持性核验。"""

from copy import deepcopy
import hashlib
import json

from service.session_evidence_rag import current_session_evidence
from .citation_verification import _numbered_report_references
from .report_sentence_evidence import narrow_reference


async def build_critic_evidence(state):
    report = str(state.get("final_report") or "")
    report_hash = hashlib.sha256(report.encode()).hexdigest()
    extraction = state.get("report_claim_extraction") or {}
    current_extraction = extraction.get("report_sha256") == report_hash
    claims = (state.get("report_verification_claims") or []) if current_extraction else (state.get("claims") or [])
    audit = state.get("citation_verification") or {}
    current_audit = audit.get("report_sha256") == report_hash
    audit_rows = {row.get("claim_id"): row for row in audit.get("claims") or []} if current_audit else {}
    packet = {
        "report_sha256": report_hash,
        "claim_scope": "final_report_extracted_claims" if current_extraction else "writer_declared_claims_only",
        "extraction_status": "stale" if extraction.get("report_sha256") and not current_extraction else
                             extraction.get("status", "not_executed"),
        "all_claims_extracted": False,
        "verification_status": audit.get("status", "not_executed") if current_audit else "not_current",
        "claims": [], "excerpts": [],
        "instruction": "全部现有论断及引用原文仅供理解报告和复审旧问题；无片段或未核验不能推断为支持。独立抽取也不保证没有遗漏。",
    }
    documents = {}
    for source in state.get("raw_sources") or []:
        if source.get("document_id"):
            documents.setdefault(source["document_id"], []).append(source)
    rag = current_session_evidence()
    resolved_cache, excerpt_ids = {}, {}

    async def read_reference(ref):
        key = json.dumps(ref, ensure_ascii=False, sort_keys=True)
        if key in resolved_cache:
            return resolved_cache[key]
        if not isinstance(ref, dict):
            return [], ["invalid_citation_structure"]
        try:
            if rag is not None:
                if ref.get("evidence_unit_ids"):
                    values = await rag.resolve_units(state, ref.get("document_id"), ref["evidence_unit_ids"])
                else:
                    values = await rag.sentence_references(state, ref.get("document_id"), ref.get("sentence_ids"))
                errors = [] if values else ["citation_excerpt_unavailable"]
            else:
                # 单元编号必须由原证据库解析，不能信任模型自报的quote或另造分块。
                if ref.get("evidence_unit_ids"):
                    values, errors = [], ["evidence_unit_store_unavailable"]
                elif ref.get("sentence_selection") and ref.get("source_spans"):
                    from service.web_document import sentence_records
                    from service.summary_evidence_units import render_spans
                    sources = documents.get(ref.get("document_id"), [])
                    if len(sources) != 1 or not sentence_records(sources[0]):
                        raise ValueError("选句原文不可用")
                    source = sources[0]
                    spans = ref["source_spans"]
                    if any(type(s.get("start")) is not int or type(s.get("end")) is not int
                           or not 0 <= s["start"] < s["end"] <= len(source["content"]) for s in spans):
                        raise ValueError("选句原文位置无效")
                    values, errors = [{**ref, "source_id": source["id"],
                        "source_url": source.get("url") or source.get("source_url", ""),
                        "source_title": source.get("title", ""),
                        "quote": render_spans(source["content"], spans)}], []
                else:
                    values, errors = _numbered_report_references(ref, documents)
                    for value in values:
                        source = documents[value["document_id"]][0]
                        value.update(source_url=source.get("url") or source.get("source_url", ""),
                                     source_title=source.get("title", ""))
            if ref.get("sentence_selection") and not errors:
                if len(values) != 1:
                    raise ValueError("选句的父证据块不唯一")
                values = [narrow_reference(values[0], ref["sentence_selection"])]
        except (ValueError, KeyError, TypeError):
            values, errors = [], ["citation_excerpt_unavailable"]
        resolved_cache[key] = values, errors
        return values, errors

    for index, claim in enumerate(claims, 1):
        row = {key: deepcopy(claim[key]) for key in
               ("claim", "report_quote", "section_id", "report_start", "report_end", "assertion_kind", "citation_urls")
               if key in claim}
        row["claim_id"] = claim.get("claim_id") or f"writer_claim_{index}"
        quote = claim.get("report_quote") or claim.get("claim") or ""
        row["located_in_current_report"] = bool(quote and quote in report)
        row["binding_errors"] = list(claim.get("binding_errors") or [])
        row["evidence_ids"], row["unavailable_references"] = [], []
        checked = audit_rows.get(claim.get("claim_id"), {})
        if checked.get("claim") == claim.get("claim"):
            row["verification_status"] = checked.get("status", "not_executed")
        else:
            row["verification_status"] = "not_executed"
        for ref in claim.get("evidence") or []:
            excerpts, errors = await read_reference(ref)
            if errors:
                row["unavailable_references"].append({"reference": deepcopy(ref), "errors": errors})
            for excerpt in excerpts:
                value = {key: deepcopy(excerpt[key]) for key in
                         ("document_id", "source_id", "source_url", "source_title", "sentence_ids",
                          "evidence_unit_ids", "source_spans", "unit_type", "quote", "sentence_selection") if key in excerpt}
                identity = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if identity not in excerpt_ids:
                    ident = f"excerpt_{len(excerpt_ids) + 1}"
                    excerpt_ids[identity] = ident
                    packet["excerpts"].append({"evidence_id": ident, **value})
                if excerpt_ids[identity] not in row["evidence_ids"]:
                    row["evidence_ids"].append(excerpt_ids[identity])
        packet["claims"].append(row)
    packet["claims_count"] = len(packet["claims"])
    packet["excerpts_count"] = len(packet["excerpts"])
    return packet
