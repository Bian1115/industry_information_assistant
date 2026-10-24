"""独立报告论断抽取、引用覆盖与支持性核验；与报告整体评分分离。"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, StrictBool, model_validator

from service.context_policy import NLISettings, runtime_nli_settings
from service.web_document import quote_matches_sentences, resolve_sentences
from .evidence import build_evidence_view, resolve_source, source_indexes


class CitationVerificationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scout_facts: StrictBool = False
    report_claims: StrictBool = False
    report_failure_action: Literal["audit", "delete", "rewrite_once"] = "audit"

    @model_validator(mode="after")
    def validate_action(self):
        if self.report_failure_action != "audit" and not self.report_claims:
            raise ValueError("正文失败处理要求 report_claims=true")
        return self


def resolve_citation_verification_options(value: Mapping[str, Any] | None = None) -> CitationVerificationOptions:
    """仅首跑解析部署默认；显式请求和历史检查点保持各自的开关语义。"""
    if value is not None:
        return CitationVerificationOptions.model_validate(value)
    raw = os.getenv("RESEARCH_WRITER_CITATION_VERIFICATION_ENABLED", "true").strip().lower()
    if raw in {"true", "1", "yes", "on"}:
        enabled = True
    elif raw in {"false", "0", "no", "off"}:
        enabled = False
    else:
        raise ValueError("RESEARCH_WRITER_CITATION_VERIFICATION_ENABLED 必须是有效布尔值")
    return CitationVerificationOptions(report_claims=enabled)


def not_executed() -> dict:
    return {"status": "not_executed", "reason": "引用核验开关关闭",
            "all_claims_checked": False, "declared_claims_checked": False}


async def verify_citations(claims: list[dict], *, enabled: bool = False,
                           settings: NLISettings | None = None) -> dict:
    """通用按调用开关。关闭时不加载模型、不核验、不产生通过率。

    claims 的 references 必须先由调用方从保存的原文解析。此接口只作审计，
    不删除事实或改写报告；NLI 支持性也不代表现实真实性或论断抽取完整性。
    """
    if not isinstance(enabled, bool):
        raise ValueError("enabled 必须为布尔值")
    if not enabled:
        return not_executed()
    fingerprint = hashlib.sha256(json.dumps(claims, ensure_ascii=False,
                                            sort_keys=True).encode()).hexdigest()
    base = {"enabled": True, "audit_only": True, "input_sha256": fingerprint,
            "all_claims_checked": False, "declared_claims_checked": False,
            "scope": "declared_claims_only", "claims_count": len(claims), "needs_review": True}
    if not claims:
        return {**base, "status": "incomplete", "execution_status": "incomplete",
                "reason": "没有可核验的论断声明", "claims": []}
    # 延迟导入；关闭的路径不初始化核验器，也不调用模型API。
    from service.nli_verifier import verify_claims, execution_error
    try:
        active_settings = settings or runtime_nli_settings()
        # 长报告按引用对数量分批，不截断或丢弃尾部论断。
        batches, batch, count = [], [], 0
        for claim in claims:
            size = len(claim.get("references", []))
            if batch and count + size > active_settings.max_pairs:
                batches.append(batch)
                batch, count = [], 0
            batch.append(claim)
            count += size
        if batch:
            batches.append(batch)
        reports = [await verify_claims(batch, active_settings) for batch in batches]
        report = dict(reports[0])
        if len(reports) > 1:
            report.update(
                claims=[row for part in reports for row in part.get("claims", [])],
                execution_status="completed" if all(part.get("execution_status") == "completed" for part in reports) else "incomplete",
                batches=[{k: v for k, v in part.items() if k != "claims"} for part in reports],
                api_calls=[call for part in reports for call in part.get("api_calls", [])],
                duration_seconds=sum(part.get("duration_seconds", 0) for part in reports),
                pair_count=sum(part.get("pair_count", 0) for part in reports),
                window_count=sum(part.get("window_count", 0) for part in reports),
            )
    except Exception as exc:
        return {**base, "status": "error", "execution_status": execution_error(exc),
                "reason": "引用核验未完成", "claims": []}
    rows = report.get("claims", [])
    completed = report.get("execution_status") == "completed" and len(rows) == len(claims)
    for item, row in zip(claims, rows):
        row["claim_id"] = item.get("claim_id", "")
        row["binding_errors"] = item.get("binding_errors", [])
        if row["binding_errors"]:
            status = "missing_citation" if "missing_inline_citation" in row["binding_errors"] else "invalid_citation"
            row.update(status=status, content_status="unverified", needs_review=True,
                       confidence=None, review_reasons=list(dict.fromkeys(
                           [*row.get("review_reasons", []), status, *row["binding_errors"]])))
        if any(e.get("prediction_as_fact") for e in row.get("evidence", [])):
            row.update(status="prediction_as_fact", content_status="insufficient_support", needs_review=True, confidence=None,
                       review_reasons=list(dict.fromkeys([*row.get("review_reasons", []), "prediction_as_fact"])))
        for field in ("assertion_kind", "report_quote", "evidence_selection"):
            if field in item:
                row[field] = item[field]
        # 原文保存在证据库；状态与审计只保存引用位置及模型结果。
        for ref, evidence in zip(item.get("references", []), row.get("evidence", [])):
            evidence.pop("quote", None)
            evidence.update({k: ref[k] for k in ("document_id", "sentence_ids", "fact_id", "evidence_unit_ids",
                                               "unit_type", "source_spans", "retrieval_candidate", "source_url", "source_title", "sentence_selection") if k in ref})
        required = set(item.get("required_document_ids") or [])
        if required and row.get("status") == "supported":
            supported_documents = {ref["document_id"] for ref, evidence in zip(
                item.get("references", []), row.get("evidence", []))
                if evidence.get("status") == "scored" and
                (evidence.get("scores") or {}).get("entailment", 0) >= active_settings.entailment_threshold}
            if required - supported_documents:
                row.update(status="unsupported_citation", content_status="insufficient_support", needs_review=True, confidence=None,
                    unsupported_document_ids=sorted(required - supported_documents),
                    review_reasons=list(dict.fromkeys([*row.get("review_reasons", []), "unsupported_citation"])))
    supported = sum(row.get("status") == "supported" for row in rows)
    return {**report, **base, "claims": rows,
            "status": "passed" if completed and supported == len(claims) else "rejected" if completed else "incomplete",
            "declared_claims_checked": completed, "supported_count": supported,
            "needs_review": not completed or any(row.get("needs_review") for row in rows),
            "review_count": sum(bool(row.get("needs_review")) for row in rows),
            "review_claim_indexes": [i for i, row in enumerate(rows) if row.get("needs_review")]}


def _reference(fact: Mapping[str, Any], source: Mapping[str, Any], quote: str) -> dict:
    return {"source_id": source["id"], "fact_id": fact["id"], "quote": quote,
            "document_id": fact.get("document_id", ""),
            "sentence_ids": fact.get("sentence_ids", [])}


def _numbered_report_references(citation: Mapping[str, Any], documents: dict) -> tuple[list[dict], list[str]]:
    """只从校验过的正文索引回读；不使用 Writer 返回的引句文本。

    连续原句保持原始间隔组成证据；不相邻原句分组，避免拼出不存在的原文。
    """
    document_id = citation.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        return [], ["missing_document_id"]
    sources = documents.get(document_id, [])
    if len(sources) != 1:
        return [], ["unknown_or_ambiguous_document_id"]
    source = sources[0]
    if not isinstance(source.get("id"), str) or not source["id"]:
        return [], ["missing_source_id"]
    rows = resolve_sentences(source, document_id, citation.get("sentence_ids"))
    if not rows:
        return [], ["invalid_sentence_ids_or_document"]
    groups = []
    for row in rows:
        if groups and int(row["id"][1:]) == int(groups[-1][-1]["id"][1:]) + 1:
            groups[-1].append(row)
        else:
            groups.append([row])
    text = str(source.get("content") or "")
    return [{"source_id": source["id"], "document_id": document_id,
             "sentence_ids": [row["id"] for row in group],
             "quote": text[group[0]["start"]:group[-1]["end"]]}
            for group in groups], []


def _claims_for_stage(state: Mapping[str, Any], stage: str) -> list[dict]:
    """Scout输入与历史Writer声明的兼容解析；生产报告核验使用独立抽取入口。"""
    indexes = source_indexes(state)
    facts = {str(fact["id"]): fact for fact in build_evidence_view(state)}
    claims = []
    if stage == "scout_facts":
        # 不用正文开头充当证据；历史未编号事实按缺失绑定记录。
        for fact in state.get("facts", []):
            source = resolve_source(fact, indexes) or {}
            rows = resolve_sentences(source, fact.get("document_id"), fact.get("sentence_ids"))
            valid = str(fact.get("id")) in facts and bool(rows)
            claims.append({"claim_id": fact.get("id", ""), "claim": fact.get("content", ""),
                           "references": [_reference(fact, source, row["text"]) for row in rows] if valid else [],
                           "binding_errors": [] if valid else ["missing_numbered_evidence"]})
    else:
        documents = {}
        for source in state.get("raw_sources", []):
            if isinstance(source, dict) and isinstance(source.get("document_id"), str) and source["document_id"]:
                documents.setdefault(source["document_id"], []).append(source)
        for index, item in enumerate(state.get("claims", [])):
            refs, errors = [], list(item.get("binding_errors") or [])
            for citation in item.get("evidence", []):
                if not isinstance(citation, dict):
                    errors.append("invalid_citation_structure")
                    continue
                if "document_id" in citation or "sentence_ids" in citation:
                    resolved, binding_errors = _numbered_report_references(citation, documents)
                    refs.extend(resolved)
                    errors.extend(binding_errors)
                    continue
                # 仅兼容旧检查点已经保存的 evidence_id/quote 声明。
                # 新 Writer 输出由 normalize_claims 统一为编号，不进入此分支。
                fact = facts.get(str(citation.get("evidence_id")))
                source = resolve_source(fact, indexes) if fact else None
                quote = citation.get("quote")
                if not source or not isinstance(quote, str) or not quote.strip():
                    errors.append("unknown_fact_or_empty_quote")
                    continue
                if not quote_matches_sentences(source, fact.get("document_id"),
                                               fact.get("sentence_ids"), quote):
                    errors.append("quote_not_in_bound_sentences")
                    continue
                refs.append(_reference(fact, source, quote))
            claims.append({"claim_id": f"report_claim_{index + 1}", "claim": item["claim"],
                           "section": item.get("section_id"), "references": refs,
                           "binding_errors": errors})
    return claims


async def verify_stage_citations(state: dict, stage: str) -> dict:
    """可从各业务入口调用。旧检查点无配置时明确按关闭处理。"""
    if stage not in {"scout_facts", "report_claims"}:
        raise ValueError(f"未知引用核验阶段: {stage}")
    options = CitationVerificationOptions.model_validate(state.get("citation_verification_config") or {})
    enabled = getattr(options, stage)
    from service.session_evidence_rag import current_session_evidence
    rag = current_session_evidence()
    extraction = None
    settings = (NLISettings.model_validate(state.get("citation_verification_settings") or {})
                if enabled else None)
    if stage == "report_claims" and enabled:
        from .report_claim_extraction import extract_report_claims
        from .report_citation_inputs import report_claim_inputs
        # 先清除旧版结果；失败也不能沿用上一版报告的通过状态或 Writer 声明。
        state["report_verification_claims"] = []
        state["report_claim_extraction"] = {"status": "incomplete"}
        state["verified_citation_bindings"] = []
        try:
            extraction = await extract_report_claims(state, settings)
            state["report_claim_extraction"] = extraction
            state["report_verification_claims"] = extraction["claims"]
            claims = await report_claim_inputs(state, extraction["claims"], rag, settings=settings)
        except Exception as exc:
            from service.nli_verifier import execution_error
            result = {"stage": stage, "status": "error", "execution_status": execution_error(exc),
                "reason": "最终报告独立论断抽取、证据定位或证据选句失败", "error_type": type(exc).__name__,
                "scope": "final_report_extracted_claims", "all_claims_checked": False,
                "declared_claims_checked": False, "extracted_claims_checked": False,
                "needs_review": True, "claims": [], "report_sha256": hashlib.sha256(
                    str(state.get("final_report") or "").encode()).hexdigest()}
            state["citation_verification"] = result
            return result
    else:
        claims = (await rag.claim_inputs(state) if stage == "report_claims" and rag is not None
                  else _claims_for_stage(state, stage) if enabled else [])
    if rag is not None and stage == "report_claims":
        # 精细召回独立于NLI开关；关闭核验时返回可回读定位，不将候选标为已支持。
        locator_fields = ("source_id", "source_url", "source_title", "document_id", "sentence_ids",
                          "evidence_unit_ids", "source_spans", "unit_type", "retrieval_candidate", "sentence_selection")
        state["retrieved_citation_bindings"] = [{"claim_id": item["claim_id"], "claim": item["claim"],
            "binding_errors": item["binding_errors"], "evidence": [
                {key: ref[key] for key in locator_fields if key in ref} for ref in item["references"]]}
            for item in claims]
        state["citation_retrieval_status"] = {"status": "completed", "claims_count": len(claims),
            "nli_enabled": enabled, "scope": "current_session_topic", "verification": "not_executed" if not enabled else "requested"}
        state["verified_citation_bindings"] = []
    result = await verify_citations(claims, enabled=enabled, settings=settings)
    if extraction is not None:
        complete = extraction["status"] == "completed"
        checked = complete and result.get("execution_status") == "completed"
        if not complete:
            result.update(status="incomplete", execution_status="incomplete", needs_review=True)
        elif not claims:
            # 每个单元均显式判为无外部论断，区别于未抽取/漏返回单元。
            result.update(status="passed", execution_status="completed", claims=[], needs_review=False,
                          supported_count=0, review_count=0, review_claim_indexes=[])
            checked = True
        result.update(scope="final_report_extracted_claims", extracted_claims_checked=checked,
            evidence_locator="llm_selected_sentence_ids_nli_v1",
            declared_claims_checked=False, extraction_status=extraction["status"],
            extraction_units_total=extraction["units_total"], claims_count=len(claims),
            all_claims_checked=False,
            missing_citation_count=sum(row.get("status") == "missing_citation" for row in result.get("claims", [])),
            prediction_as_fact_count=sum(row.get("status") == "prediction_as_fact" for row in result.get("claims", [])))
        result["claim_extraction"] = {key: value for key, value in extraction.items() if key != "claims"}
    if rag is not None and stage == "report_claims" and enabled:
        state["verified_citation_bindings"] = [
            {"claim_id": row["claim_id"], "claim": row["claim"], "evidence": [
                {key: e[key] for key in ("source_id", "source_url", "source_title", "document_id", "sentence_ids", "evidence_unit_ids", "source_spans", "unit_type", "sentence_selection") if key in e}
                for e in row.get("evidence", []) if e.get("status") == "scored" and
                (e.get("scores") or {}).get("entailment", 0) >= settings.entailment_threshold]}
            for row in result.get("claims", []) if row.get("status") == "supported"]
    result["stage"] = stage
    if stage == "report_claims":
        result["report_sha256"] = hashlib.sha256(str(state.get("final_report") or "").encode()).hexdigest()
    state["scout_citation_verification" if stage == "scout_facts" else "citation_verification"] = result
    return result
