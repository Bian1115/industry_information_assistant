"""Writer 的论断与原文引用声明；不进行事实支持性核验。"""

from copy import deepcopy
import json
from typing import Any, Mapping

from .evidence import build_evidence_view
from service.web_document import resolve_sentences


CLAIM_INSTRUCTIONS = """## 论断与引用证据声明（claims）
除正文和原有字段外，必须返回 claims 数组，覆盖本次输出中所有需要外部证据的事实论断。
每项结构为 {"claim": "可独立核验的原子论断", "report_quote": "本次正文中包含该论断的完整原段落或表格行，逐字复制并保留行内引用", "evidence": [{"document_id": "已有正文编号", "sentence_ids": ["已有原句编号"]}]}。
1. claim 必须对应本次输出，保留主体、时间、数值、单位和限定条件；复合命题拆开，不把同一句的多个独立事实合并。
2. 输入 documents 集中保存来源原句。document_id 使用提供的正文编号；sentence_ids 只能使用该正文 original_sentences 中实际展示的 id，不得创建新编号或混用其他正文的句子编号。直接依据原文形成论述，保留原文的时间、单位和限定条件。
3. 引用只返回 document_id 和非空 sentence_ids，不返回 quote、原句文本、evidence_id 或来源正文。后续由程序按编号取回原句核验；Writer 仍需阅读提供的原句以选择证据。
4. 一个 claim 可以关联多个正文及多条原句；找不到对应编号或历史资料未编号时仍保留 claim，并返回 evidence: []。程序推导的数值也不得伪装为来源原句。
5. claims 是待核验的声明，不代表已经验证；它不能替代正文行内引用。参考文献由程序从正文实际使用的引用生成，不需要模型重复返回来源列表。
6. 只收集本次实际输出的论断，不复述参考素材中未写入正文的事实。没有外部事实时返回 claims: []。
7. report_quote 用于定位需要改写或删除的正文，必须逐字出现在本次输出中；同一段的多条论断使用相同 report_quote。不得返回章节标题或整篇报告作为定位文本。
8. 若输入是临时RAG的evidence_units，引用改用 {"document_id":"单元中的正文编号","evidence_unit_ids":["单元的id"]}。这些单元可能是段落、原句或带表头的表格；程序直接使用对应证据块文本，并进一步逐论断检索相关证据块。不要把chunk编号填写成sentence_ids，也不自行拆散表格。
"""


SYNTHESIS_CLAIM_INSTRUCTIONS = """## 首尾论断的章节依据（claims）
claims 覆盖执行摘要、核心结论和未来展望中需要外部证据的原子事实论断。
每项返回 claim、report_quote、source_claim_ids：
- report_quote 是本次首尾输出中包含论断的完整原段落，逐字复制并保留行内引用。
- source_claim_ids 引用本次章节论断索引中的编号，例如 ["C1", "C2"]；综合多条论断时列出所有依据。编号只在本次调用有效，不得引用历史调用中的编号。
- 依据编号必须与本次输出实际表达的论断对应，保留主体、时间、数值、单位、范围和限定条件；不能因为共享一个段落就引用该段的其他无关论断。
- 只能归纳章节中已有事实，不新增数据、来源、已完成的验证或提高结论确定性；章节未解决的问题继续保留不确定性。
- 本次没有提供来源原文或原句编号，不返回 document_id、sentence_ids 或假设证据编号。程序根据章节已有论断解析证据，再按配置核验；章节依据和继承编号均不代表核验通过。
- 找不到章节依据时保留论断并返回 source_claim_ids: []；没有外部事实时返回 claims: []。不要重复收集章节正文的论断。
"""


def build_synthesis_claim_index(state: Mapping[str, Any]) -> dict[str, dict]:
    """本次调用的章节论断快照；只索引仍能定位到当前正文的声明。

    C 编号由程序分配，不修改章节声明。原文、事实和旧首尾声明不进入索引。
    """
    drafts = state.get("draft_sections") or {}
    declared = state.get("section_claims") or {}
    result = {}
    for section in state.get("outline") or []:
        section_id = section.get("id")
        content = drafts.get(section_id)
        if not isinstance(section_id, str) or not isinstance(content, str) or not content.strip():
            continue
        for index, original in enumerate(declared.get(section_id) or []):
            if not isinstance(original, dict):
                continue
            normalized = normalize_claims([original])
            if not normalized:
                continue
            location = normalized[0].get("report_quote") or normalized[0]["claim"]
            if location not in content:
                continue
            result[f"C{len(result) + 1}"] = {
                "section_id": section_id,
                "claim_index": index,
                "declaration": {**normalized[0], "binding_errors": deepcopy(original.get("binding_errors") or [])},
            }
    return result


def format_synthesis_claim_index(index: Mapping[str, dict]) -> str:
    """模型仅接收论断编号、章节和文本；证据编号留给程序继承。"""
    return json.dumps([
        {"id": key, "section_id": item["section_id"], "claim": item["declaration"]["claim"]}
        for key, item in index.items()
    ], ensure_ascii=False, separators=(",", ":"))


def bind_synthesis_claims(
    value: Any, state: Mapping[str, Any], output_parts: list[str], *,
    claim_index: Mapping[str, dict] | None = None,
) -> list[dict]:
    """按本次章节论断编号继承证据；兼容已有的章节原段落声明。

    不读取原文、不信任模型新报的证据编号，不将继承当作支持性核验。
    """
    if not isinstance(value, list):
        return []
    claim_index = build_synthesis_claim_index(state) if claim_index is None else claim_index
    drafts = state.get("draft_sections") or {}
    sections = {section.get("id") for section in state.get("outline") or []}
    declared = state.get("section_claims") or {}
    result = []
    for raw in value:
        claims = normalize_claims([raw])
        if not claims:
            continue
        claim = claims[0]
        claim["evidence"] = []
        errors, source_refs, seen_evidence = [], [], set()
        report_quote = claim.get("report_quote", "")
        if not report_quote or not any(report_quote in part for part in output_parts):
            errors.append("synthesis_report_quote_not_in_output")
        if "source_claim_ids" in raw:
            ids = raw.get("source_claim_ids")
            if not isinstance(ids, list) or not ids:
                errors.append("missing_synthesis_source_claim_ids")
                ids = []
            accepted = []
            for source_id in ids:
                if not isinstance(source_id, str) or not source_id:
                    errors.append("invalid_synthesis_source_claim_id")
                    continue
                source = claim_index.get(source_id)
                if source is None:
                    errors.append("unknown_synthesis_source_claim_id")
                    continue
                if source_id in accepted:
                    continue
                accepted.append(source_id)
                source_refs.append({"section_id": source["section_id"], "claim_index": source["claim_index"]})
                original = source["declaration"]
                errors.extend(original.get("binding_errors") or [])
                evidence = original["evidence"]
                if not evidence:
                    errors.append("synthesis_source_claim_without_evidence")
                for citation in evidence:
                    key = json.dumps(citation, ensure_ascii=False, sort_keys=True)
                    if key not in seen_evidence:
                        seen_evidence.add(key)
                        claim["evidence"].append(deepcopy(citation))
            claim["source_claim_ids"] = accepted
            claim["source_claim_refs"] = source_refs
            claim["binding_errors"] = list(dict.fromkeys(errors))
            result.append(claim)
            continue
        # 兼容旧模型结果；显式给出空/错误编号时不使用段落匹配兜底。
        anchors = raw.get("source_quotes")
        if not isinstance(anchors, list) or not anchors:
            errors.append("missing_synthesis_source_quotes")
            anchors = []
        for anchor in anchors:
            if not isinstance(anchor, dict):
                errors.append("invalid_synthesis_source_quote")
                continue
            section_id, quote = anchor.get("section_id"), anchor.get("quote")
            if not isinstance(section_id, str) or section_id not in sections:
                errors.append("unknown_synthesis_source_section")
                continue
            if not isinstance(quote, str) or not quote.strip() or quote not in drafts.get(section_id, ""):
                errors.append("synthesis_source_quote_not_in_section")
                continue
            matched = False
            for index, original in enumerate(declared.get(section_id) or []):
                if not isinstance(original, dict):
                    continue
                location = original.get("report_quote")
                # 旧声明无正文定位时，仅接受论断本身逐字出现在依据中的情况。
                location = location if isinstance(location, str) and location.strip() else original.get("claim")
                if not isinstance(location, str) or not location.strip() or location not in quote:
                    continue
                matched = True
                ref = {"section_id": section_id, "claim_index": index}
                if ref not in source_refs:
                    source_refs.append(ref)
                errors.extend(original.get("binding_errors") or [])
                normalized = normalize_claims([original])
                evidence = normalized[0]["evidence"] if normalized else []
                if not evidence:
                    errors.append("synthesis_source_claim_without_evidence")
                for citation in evidence:
                    key = json.dumps(citation, ensure_ascii=False, sort_keys=True)
                    if key not in seen_evidence:
                        seen_evidence.add(key)
                        claim["evidence"].append(deepcopy(citation))
            if not matched:
                errors.append("synthesis_source_quote_without_declared_claims")
        claim["source_claim_refs"] = source_refs
        claim["binding_errors"] = list(dict.fromkeys(errors))
        result.append(claim)
    return result


def format_claim_evidence(state: Mapping[str, Any], fact_ids=None) -> str:
    """每条事实保留独立绑定；原句按正文编号和原句编号只展示一次。

    这是 Writer 的临时输入视图，不改写原始正文、事实或历史检查点。
    未编号的历史短正文独立放入 legacy_sources，不能冒充编号原句。
    """
    selected = set(fact_ids) if fact_ids is not None else None
    facts, documents, legacy_sources = [], {}, {}
    seen_sentences = set()
    for fact in build_evidence_view(state):
        evidence_id = str(fact.get("evidence_id") or "")
        if not evidence_id or (selected is not None and evidence_id not in selected):
            continue
        source_id = str(fact.get("source_id") or "")
        document_id = str(fact.get("document_id") or "")
        sentences = fact.get("original_sentences") or []
        facts.append({
            "evidence_id": evidence_id,
            "content": fact.get("content", ""),
            "source_id": source_id,
            "document_id": document_id,
            "sentence_ids": [row["id"] for row in sentences],
        })
        if not document_id:
            if fact.get("excerpt"):
                legacy_sources.setdefault(source_id, {
                    "source_id": source_id,
                    "source_url": fact.get("source_url", ""),
                    "source_title": fact.get("source_title", ""),
                    "text": fact["excerpt"],
                })
            continue
        document = documents.setdefault(document_id, {
            "document_id": document_id,
            "source_id": source_id,
            "source_url": fact.get("source_url", ""),
            "source_title": fact.get("source_title", ""),
            "original_sentences": [],
        })
        for row in sentences:
            key = (document_id, row["id"])
            if key not in seen_sentences:
                document["original_sentences"].append({"id": row["id"], "text": row["text"]})
                seen_sentences.add(key)
    payload = {"facts": facts, "documents": list(documents.values())}
    if legacy_sources:
        payload["legacy_sources"] = list(legacy_sources.values())
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def format_source_evidence(state: Mapping[str, Any], query="", *, section_id=None):
    """未绑定会话RAG时的有界原文视图；无需先生成事实摘要。

    复用正文chunk切分，再按完整单元组装原句，避免将表格拆成缺少表头的散句。
    这是非持久化调用的本地词法回退，不冒充向量或模型精排。
    """
    from service.qwen_token_counter import get_qwen_token_counter
    from service.session_evidence_chunks import chunk_document
    from service.session_evidence_rag import SessionEvidenceSettings
    from service.summary_temporary_retrieval import tokenize
    from service.web_document import sentence_records, source_usable
    from .source_scope import url_in_scope

    counter = get_qwen_token_counter()
    settings = SessionEvidenceSettings.from_env()
    terms = set(tokenize(query))
    candidates, sources = [], {}
    for original in state.get("raw_sources") or []:
        if (not source_usable(original)
                or not url_in_scope(original.get("url") or original.get("source_url") or "",
                                    state.get("allowed_domains") or [])):
            continue
        if section_id is not None and section_id not in (original.get("related_sections") or []):
            continue
        source = dict(original)
        if not sentence_records(source):
            continue
        source.setdefault("id", source["document_id"])
        try:
            chunks = chunk_document(source, user_id=state.get("_user_id", "local"),
                session_id=state.get("session_id", "local"), topic_id="local",
                counter=counter, chunk_tokens=settings.chunk_tokens)
        except ValueError:
            continue
        sources[source["document_id"]] = source
        candidates.extend(row for row in chunks if row["unit_type"] in {"paragraph", "table", "record"})
    candidates.sort(key=lambda row: (-len(terms.intersection(tokenize(row["content"]))),
                                    row["metadata"]["document_id"], row["metadata"]["char_start"]))
    documents, selected = {}, 0
    for row in candidates:
        if selected >= settings.context_top_k:
            break
        source = sources[row["metadata"]["document_id"]]
        proposal = deepcopy(documents)
        document = proposal.setdefault(source["document_id"], {
            "document_id": source["document_id"], "source_id": source["id"],
            "source_url": source.get("url") or source.get("source_url", ""),
            "source_title": source.get("title", ""), "original_sentences": [],
        })
        seen = {sentence["id"] for sentence in document["original_sentences"]}
        ids = set(row["metadata"]["sentence_ids"]) - seen
        document["original_sentences"].extend({"id": sentence["id"], "text": sentence["text"]}
            for sentence in sentence_records(source) if sentence["id"] in ids)
        payload = {"facts": [], "documents": list(proposal.values()), "retrieval_mode": "local_source_view"}
        if counter.count_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) <= settings.context_tokens:
            documents = proposal
            selected += 1
    return json.dumps({"facts": [], "documents": list(documents.values()), "retrieval_mode": "local_source_view"},
                      ensure_ascii=False, separators=(",", ":"))


def format_section_revision_evidence(state: Mapping[str, Any], section_id: str, fact_ids) -> str:
    """修订只展示本章事实和当前章正文仍能定位的原引用，不按问题或 URL 扩展。

    原章节声明可能没有对应事实或缺少章节关联，因此按精确文档/原句编号
    回读其来源；来源准入和编号检查仍由 resolve_sentences 执行。
    """
    payload = json.loads(format_claim_evidence(state, fact_ids))
    documents = {item["document_id"]: item for item in payload["documents"]}
    sources = {source.get("document_id"): source for source in state.get("raw_sources") or []
               if isinstance(source, dict) and source.get("document_id")}
    seen = {(doc["document_id"], row["id"])
            for doc in documents.values() for row in doc["original_sentences"]}
    for item in build_synthesis_claim_index(state).values():
        if item["section_id"] != section_id:
            continue
        for ref in item["declaration"]["evidence"]:
            document_id = ref["document_id"]
            if not isinstance(document_id, str):
                continue
            source = sources.get(document_id)
            if source is None:
                continue
            rows = resolve_sentences(source, document_id, ref["sentence_ids"])
            if not rows:
                continue
            document = documents.setdefault(document_id, {
                "document_id": document_id,
                "source_id": source.get("id", ""),
                "source_url": source.get("url") or source.get("source_url", ""),
                "source_title": source.get("title", ""),
                "original_sentences": [],
            })
            for row in rows:
                key = (document_id, row["id"])
                if key not in seen:
                    document["original_sentences"].append({"id": row["id"], "text": row["text"]})
                    seen.add(key)
    payload["documents"] = list(documents.values())
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def normalize_claims(value: Any) -> list[dict]:
    """整理新输出为正文/原句编号引用，不保存模型重复输出的原句。

    缺失/非法容器兼容为 []，不据此推断正文没有事实或已完整抽取。
    未知或非法编号仍保留，留待核验器明确诊断，不推断引用有效。
    """
    result = []
    if not isinstance(value, list):
        return result
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("claim"), str):
            continue
        claim = item["claim"].strip()
        if not claim:
            continue
        evidence = []
        for ref in item.get("evidence", []) if isinstance(item.get("evidence"), list) else []:
            if not isinstance(ref, dict):
                continue
            evidence.append({"document_id": deepcopy(ref.get("document_id", "")),
                             "sentence_ids": deepcopy(ref.get("sentence_ids", [])),
                             **({"evidence_unit_ids": deepcopy(ref["evidence_unit_ids"])} if "evidence_unit_ids" in ref else {})})
        normalized = {"claim": claim, "evidence": evidence}
        if isinstance(item.get("report_quote"), str) and item["report_quote"].strip():
            normalized["report_quote"] = item["report_quote"].strip()
        result.append(normalized)
    return result


def collect_report_claims(state: Mapping[str, Any]) -> list[dict]:
    """按正文顺序汇总当前版本，section_id 由程序赋值而非模型提供。"""
    result = []
    sections = state.get("section_claims") or {}
    for section in state.get("outline") or []:
        section_id = section.get("id")
        if not (state.get("draft_sections") or {}).get(section_id):
            continue
        result.extend({**deepcopy(item), "section_id": section_id}
                      for item in sections.get(section_id, []))
    result.extend({**deepcopy(item), "section_id": "__synthesis__"}
                  for item in state.get("synthesis_claims") or [])
    return result
