"""从最终报告实际引用的网站定位原文；其他材料不能替代被引用来源通过核验。"""
from __future__ import annotations

from .evidence import normalize_url
from .source_scope import url_in_scope


async def report_claim_inputs(state: dict, claims: list[dict], rag=None, *, settings=None) -> list[dict]:
    from service.context_policy import NLISettings
    from .report_sentence_evidence import select_report_sentences
    settings = settings or NLISettings()
    from service.qwen_token_counter import get_qwen_token_counter
    from service.session_evidence_chunks import chunk_document
    from service.summary_temporary_retrieval import tokenize
    by_url = {}
    for source in state.get("raw_sources") or []:
        url = normalize_url(source.get("url") or source.get("source_url"))
        if url:
            by_url.setdefault(url, []).append(source)
    result, cached_chunks = [], {}
    for claim in claims:
        claim["evidence"] = []
        refs, errors = [], list(claim.get("binding_errors") or [])
        sources = []
        for url in claim["citation_urls"]:
            if not url_in_scope(url, state.get("allowed_domains") or []):
                errors.append("citation_out_of_scope")
                continue
            found = by_url.get(url, [])
            if len(found) != 1 or not found[0].get("document_id"):
                errors.append("unknown_or_ambiguous_cited_source")
            else:
                sources.append(found[0])
        documents = {source["document_id"] for source in sources}
        if rag is not None and documents:
            rows, audit = await rag.retrieve(state, claim["claim"], layer="citation", neighbors=True,
                                             document_ids=documents)
            audit["scope"] = "report_cited_documents_only"
            state.setdefault("evidence_rag_metrics", []).append(audit)
            refs = await rag.references(state, rows)
            # 即使底层检索返回越界结果，也不让其成为被引来源的证据。
            refs = [ref for ref in refs if ref.get("document_id") in documents]
        elif documents:
            terms = set(tokenize(claim["claim"]))
            for source in sources:
                document = source["document_id"]
                if document not in cached_chunks:
                    try:
                        cached_chunks[document] = chunk_document(source, user_id=state.get("_user_id", "local"),
                            session_id=state.get("session_id", "local"), topic_id="report_verification",
                            counter=get_qwen_token_counter())
                    except ValueError:
                        cached_chunks[document] = []
                chunks = [row for row in cached_chunks[document] if row["unit_type"] in {"paragraph", "table", "record"}]
                ranked = sorted(chunks, key=lambda row: (-len(terms & set(tokenize(row["content"]))),
                                                          row["metadata"]["char_start"]))
                # 有界词法回退保留完整段落/表格，不截取数字附近的短语。
                for row in ranked[:4]:
                    meta = row["metadata"]
                    refs.append({"source_id": source["id"], "document_id": document,
                        "source_url": source.get("url") or source.get("source_url"),
                        "source_title": source.get("title", ""), "quote": row["content"],
                        "sentence_ids": meta["sentence_ids"], "source_spans": meta["source_spans"],
                        "unit_type": row["unit_type"]})
        represented = {ref["document_id"] for ref in refs}
        if documents - represented:
            errors.append("cited_source_evidence_unavailable")
        refs = await select_report_sentences(claim, refs, settings)
        if represented - {ref["document_id"] for ref in refs}:
            errors.append("cited_source_sentences_not_selected")
        claim["binding_errors"] = list(dict.fromkeys(errors))
        claim["evidence"] = [
            {"document_id": ref["document_id"],
             **({"evidence_unit_ids": ref["evidence_unit_ids"]} if ref.get("evidence_unit_ids")
                else {"sentence_ids": ref["sentence_ids"]}),
             "source_spans": ref.get("source_spans", []),
             "sentence_selection": ref["sentence_selection"]}
            for ref in refs]
        result.append({"claim_id": claim["claim_id"], "claim": claim["claim"],
            "section": claim["section_id"], "references": refs, "binding_errors": claim["binding_errors"],
            "required_document_ids": sorted(documents),
            "assertion_kind": claim["assertion_kind"], "report_quote": claim["report_quote"],
            "evidence_selection": claim["evidence_selection"]})
    return result
