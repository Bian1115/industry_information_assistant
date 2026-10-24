"""会话临时双层检索；状态只存来源引用，写作与核验直接消费证据块。"""
import asyncio
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import os
import time

from pydantic import BaseModel, Field, model_validator
from service.session_evidence_chunks import stable_id
from service.session_evidence_store import SessionEvidenceStore, EvidenceExpiredError
from service.session_evidence_vectors import SessionEvidenceVectors
from service.session_evidence_ranking import rerank_candidates, RerankBudgetError
from service.summary_temporary_retrieval import TemporaryRetrievalSettings, RetrievalAPI, tokenize

_current = ContextVar("session_evidence_rag", default=None)
BEGIN = "<session_evidence_view>"
END = "</session_evidence_view>"


class SessionEvidenceSettings(BaseModel):
    enabled: bool = True
    ttl_seconds: int = Field(default=86400, ge=60)
    ttl_mode: str = "idle"
    chunk_tokens: int = Field(default=1024, ge=64, le=2048)
    context_tokens: int = Field(default=8192, ge=128)
    citation_tokens: int = Field(default=2000, ge=128)
    context_top_k: int = Field(default=8, ge=1, le=60)
    citation_top_k: int = Field(default=5, ge=1, le=60)
    coarse_page_size: int = Field(default=128, ge=1, le=512)
    coarse_pages: int = Field(default=3, ge=1, le=16)
    rrf_k: int = Field(default=60, ge=1)
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = Field(default=1024, ge=64, le=2048)
    rerank_model: str = "qwen3-rerank"
    rerank_max_input_tokens: int = Field(default=4000, ge=64, le=4000)
    rerank_max_request_tokens: int = Field(default=110000, ge=128, le=120000)
    rerank_max_documents: int = Field(default=500, ge=1, le=500)
    neighbor_radius: int = Field(default=1, ge=0, le=2)
    max_chunks: int = Field(default=20000, ge=1)
    history_reserve_tokens: int = Field(default=4096, ge=0)

    @model_validator(mode="after")
    def validate_retrieval(self):
        if max(self.context_top_k, self.citation_top_k) > self.coarse_page_size * self.coarse_pages:
            raise ValueError("精排TopK不能超过粗排分页候选上限")
        return self

    @classmethod
    def from_env(cls):
        values = {}
        for name, field in cls.model_fields.items():
            raw = os.getenv("SESSION_EVIDENCE_" + name.upper())
            if raw is not None:
                if field.annotation is bool:
                    if raw not in {"0", "1", "true", "false"}:
                        raise ValueError("SESSION_EVIDENCE_ENABLED需要布尔值")
                    values[name] = raw in {"1", "true"}
                elif field.annotation is int:
                    values[name] = int(raw)
                else:
                    values[name] = raw
        result = cls(**values)
        if result.ttl_mode not in {"idle", "absolute"}:
            raise ValueError("临时证据TTL模式只允许idle或absolute")
        return result


@contextmanager
def bind_session_evidence(service):
    token = _current.set(service)
    try:
        yield service
    finally:
        _current.reset(token)


def current_session_evidence():
    return _current.get()


class SessionEvidenceRAG:
    def __init__(self, user_id, session_id, *, settings=None, store=None, api=None, vectors=None, counter=None):
        self.user_id, self.session_id = str(user_id), str(session_id)
        self.settings = settings or SessionEvidenceSettings.from_env()
        self.store = store or SessionEvidenceStore(ttl_seconds=self.settings.ttl_seconds, ttl_mode=self.settings.ttl_mode)
        retrieval = TemporaryRetrievalSettings(chunk_tokens=self.settings.chunk_tokens,
            embedding_model=self.settings.embedding_model, embedding_dimensions=self.settings.embedding_dimensions,
            rerank_model=self.settings.rerank_model)
        self.api = api or RetrievalAPI(retrieval)
        self.vectors = vectors or SessionEvidenceVectors(retrieval.embedding_dimensions, retrieval.embedding_model)
        if counter is None:
            from service.qwen_token_counter import get_qwen_token_counter
            counter = get_qwen_token_counter()
        self.counter = counter

    def query(self, state):
        return ((state.get("session_task") or {}).get("resolved_query") or
                state.get("resolved_query") or state.get("query") or state["original_query"])

    def topic(self, state):
        return state["evidence_space"]["topic_id"]

    async def attach(self, state, *, parent=None, run_id=None):
        space = await asyncio.to_thread(self.store.ensure, self.user_id, self.session_id)
        old = (parent or {}).get("evidence_space") or state.get("evidence_space") or {}
        relation = (state.get("session_task") or {}).get("topic_relation")
        topic = old.get("topic_id") or (space.get("active_topic_id") if relation == "continue" else None) or str(state.get("parent_run_id") or run_id or
                                          stable_id(self.session_id, self.query(state)))
        await asyncio.to_thread(self.store.activate_topic, self.user_id, self.session_id, topic)
        state["evidence_space"] = {**space, "user_id": self.user_id, "session_id": self.session_id,
                                   "topic_id": topic, "protocol_version": "session-evidence-v1"}
        state.setdefault("evidence_rag_metrics", [])
        return state

    async def reconcile(self, state, pending_artifacts=()):
        """校验正文后补建派生chunk，并核实Milvus实际存在而非只信indexed标志。"""
        from service.deep_research_v2.storage_reconciliation import document_refs
        groups = {}
        for space, source in document_refs(state):
            groups.setdefault((space["topic_id"], space["generation"]), []).append(source["document_id"])
        for artifact in pending_artifacts:
            for dep in (artifact.get("storage_manifest") or {}).get("dependencies", {}).get("documents", []):
                if dep["user_id"] != self.user_id or dep["session_id"] != self.session_id:
                    raise EvidenceExpiredError("章节依赖不属于当前用户和会话")
                groups.setdefault((dep["topic_id"],dep["generation"]), []).append(dep["document_id"])
        result = []
        for (topic, generation), ids in groups.items():
            audit = await asyncio.to_thread(self.store.reconcile_chunks, self.user_id, self.session_id,
                topic, generation, ids, self.counter, self.settings.chunk_tokens)
            rows = audit.pop("rows")
            try:
                present = await asyncio.to_thread(self.vectors.existing_ids, rows)
                missing = [row["id"] for row in rows if row["id"] not in present]
                if missing:
                    await asyncio.to_thread(self.store.mark_pending, self.user_id, self.session_id, generation, missing)
                work = {"evidence_space": {"topic_id": topic}, "raw_sources": []}
                await self.ingest(work)
                audit.update(missing_vectors=len(missing), index_status=work["evidence_rag_metrics"][-1]["index_status"])
            except EvidenceExpiredError:
                raise
            except Exception as exc:
                # 只允许索引基础设施降级；正文、generation和chunk重建失败必须抛出。
                audit.update(index_status="lexical_only:" + type(exc).__name__)
            result.append(audit)
        self.last_reconciliation = result
        return result

    async def ingest(self, state, *, sources=None):
        value = await asyncio.to_thread(self.store.ingest, self.user_id, self.session_id, self.topic(state),
                                        (state.get("raw_sources") or []) if sources is None else sources,
                                        self.counter, self.settings.chunk_tokens)
        state["raw_sources"] = value.pop("sources")
        state["evidence_space"].update({k: value[k] for k in ("generation", "revision")})
        pending = value.pop("pending")
        status = "reused"
        if pending:
            try:
                if hasattr(self.vectors, "prepare"):
                    await asyncio.to_thread(self.vectors.prepare)
                vectors = await self.api.embed([r["content"] for r in pending])
                for vector in vectors:
                    if len(vector) != self.vectors.dimensions or not all(math.isfinite(x) for x in vector):
                        raise ValueError("临时证据embedding维度或数值不合法")
                await asyncio.to_thread(self.vectors.upsert, pending, vectors)
                try:
                    await asyncio.to_thread(self.store.mark_indexed, self.user_id, self.session_id,
                                             value["generation"], [r["id"] for r in pending])
                except EvidenceExpiredError:
                    await asyncio.to_thread(self.vectors.delete_session, self.user_id, self.session_id)
                    raise
                status = "indexed"
            except EvidenceExpiredError:
                raise
            except Exception as exc:
                # 正文已可恢复；记录基础设施失败，检索走有预算的词法视图。
                status = "lexical_only:" + type(exc).__name__
        state.setdefault("evidence_rag_metrics", []).append({"event": "ingest", **value,
            "pending_chunks": len(pending), "index_status": status})

    async def prepare_stage(self, state, phase, run_id):
        if not state.get("evidence_space"):
            await self.attach(state, run_id=run_id)
        await self.ingest(state)
        if phase == "init":
            rows, audit = await self.retrieve(state, self.query(state), layer="context", neighbors=True)
            state["research_material_context"] = rows
            state["evidence_rag_metrics"].append(audit)

    async def commit_stage(self, state):
        if state.get("session_parent"):
            await self.ingest(state["session_parent"])
        await self.ingest(state)
        self.externalize(state)

    def externalize(self, state):
        # 兼容旧运行的边界清理；新来源在入库后即替换为引用，不再恢复全文。
        def strip(work):
            for source in work.get("raw_sources") or []:
                if source.get("document_id") and source.get("content"):
                    from service.web_document import source_available
                    source["rag_usable"] = source_available(source) and not bool(source.get("rag_chunk_error"))
                    source["evidence_document_ref"] = True
                    for key in ("content", "full_content", "sentence_index"):
                        source.pop(key, None)
            work.pop("research_material_context", None)
            if work.get("session_parent"):
                strip(work["session_parent"])
        strip(state)
        return state

    async def retrieve(self, state, query, *, layer, neighbors=False, document_ids=None):
        started = time.monotonic()
        if layer not in {"context", "citation"}:
            raise ValueError("非法临时证据检索层")
        page_size, page_limit = self.settings.coarse_page_size, self.settings.coarse_pages
        candidate_limit = page_size * page_limit
        retrieval_audit = {"coarse_page_size": page_size, "coarse_pages_requested": page_limit,
                           "coarse_candidate_limit": candidate_limit, "coarse_pages": [],
                           "coarse_page_count": 0, "dense_pages": [], "lexical_pages": [],
                           "rerank_status": "not_needed", "rerank_candidate_count": 0,
                           "rerank_scored_candidate_count": 0, "rerank_calls": [],
                           "embedding_model": self.settings.embedding_model,
                           "embedding_dimensions": self.settings.embedding_dimensions,
                           "rerank_model": self.settings.rerank_model}
        kinds = {"paragraph", "section", "table", "record"} if layer == "context" else {"sentence", "table", "record"}
        rows = await asyncio.to_thread(self.store.chunks, self.user_id, self.session_id,
                                       self.topic(state), kinds=kinds, limit=self.settings.max_chunks)
        from service.deep_research_v2.source_scope import url_in_scope
        rows = [r for r in rows if url_in_scope(r["metadata"].get("source_url", ""), state.get("allowed_domains") or [])]
        if document_ids is not None:
            rows = [r for r in rows if r["metadata"]["document_id"] in document_ids]
        count = len(rows)
        budget = self.settings.context_tokens if layer == "context" else self.settings.citation_tokens
        top_k = self.settings.context_top_k if layer == "context" else self.settings.citation_top_k
        if not rows:
            return [], {"event": "retrieval", "layer": layer, "query": query, "corpus_chunks": 0,
                        "selected_ids": [], "selected_tokens": 0, "budget_tokens": budget,
                        "coarse_count": 0, "dense_count": 0, "lexical_count": 0, **retrieval_audit}
        terms = [Counter(tokenize(r["content"])) for r in rows]
        df = Counter(t for row in terms for t in row)
        average = sum(sum(row.values()) for row in terms) / count
        tokens = set(tokenize(query))
        scores = []
        for row in terms:
            length = sum(row.values())
            score = sum(math.log(1 + (count - df[t] + .5) / (df[t] + .5)) * row[t] * 2.5 /
                        (row[t] + 1.5 * (.25 + .75 * length / max(1, average))) for t in tokens if row[t])
            scores.append(score)
        lexical = sorted((i for i in range(count) if scores[i] > 0),
                         key=lambda i: (-scores[i], rows[i]["id"]))[:candidate_limit]
        retrieval_audit["lexical_pages"] = [
            {"page": offset // page_size + 1, "offset": offset, "count": len(lexical[offset:offset + page_size])}
            for offset in range(0, len(lexical), page_size)]
        by_id = {r["id"]: i for i, r in enumerate(rows)}
        dense = []
        mode = "hybrid_rrf"
        try:
            vector = (await self.api.embed([query]))[0]
            seen_dense = set()
            for page in range(page_limit):
                ids = await asyncio.to_thread(self.vectors.search, self.user_id, self.session_id, self.topic(state),
                    state["evidence_space"]["generation"], vector, kinds, page_size, offset=page * page_size)
                before = len(dense)
                for cid in ids:
                    if cid in by_id and cid not in seen_dense:
                        dense.append(by_id[cid]); seen_dense.add(cid)
                retrieval_audit["dense_pages"].append({"page": page + 1, "offset": page * page_size,
                    "count": len(ids), "scoped_unique_count": len(dense) - before})
                if len(ids) < page_size:
                    break
        except Exception as exc:
            mode = ("partial_dense:" if dense else "lexical_only:") + type(exc).__name__
        fused = {}
        for ranking in (dense, lexical):
            for rank, i in enumerate(ranking, 1):
                fused[i] = fused.get(i, 0) + 1 / (self.settings.rrf_k + rank)
        fused_ranking = sorted(fused, key=lambda i: (-fused[i], rows[i]["id"]))
        coarse = []
        for page in range(page_limit):
            indexes = fused_ranking[page * page_size:(page + 1) * page_size]
            if not indexes:
                break
            coarse.extend(indexes)
            retrieval_audit["coarse_pages"].append({"page": page + 1, "offset": page * page_size,
                "count": len(indexes), "chunk_ids": [rows[i]["id"] for i in indexes]})
        retrieval_audit["coarse_page_count"] = len(retrieval_audit["coarse_pages"])
        retrieval_audit["fused_candidate_count"] = len(fused_ranking)
        retrieval_audit["rrf_k"] = self.settings.rrf_k
        selected = coarse[:top_k]
        if coarse:
            try:
                ranking = await rerank_candidates(self.api, self.counter, query,
                    [rows[i]["content"] for i in coarse], min(top_k, len(coarse)),
                    layer=layer, settings=self.settings, audit=retrieval_audit)
                selected = [coarse[r["index"]] for r in ranking]
            except Exception as exc:
                mode += ";rrf_without_rerank:" + type(exc).__name__
                retrieval_audit.update(rerank_status="failed", rerank_error_type=type(exc).__name__)
                if isinstance(exc, RerankBudgetError):
                    retrieval_audit["rerank_error_reason"] = str(exc)
        retrieval_audit["rerank_seed_ids"] = [rows[i]["id"] for i in selected]
        ordered = [rows[i] for i in selected]
        if neighbors and self.settings.neighbor_radius:
            seen = {r["id"] for r in ordered}
            seeds = list(ordered)
            for seed in seeds:
                for direction in ("prev_chunk_id", "next_chunk_id"):
                    previous = seed
                    for _ in range(self.settings.neighbor_radius):
                        cid = previous["metadata"].get("adjacent_chunk_ids", {}).get(layer, {}).get(direction)
                        if cid not in by_id:
                            break
                        neighbor = rows[by_id[cid]]
                        if neighbor["metadata"]["document_id"] != seed["metadata"]["document_id"]:
                            break
                        if cid not in seen:
                            ordered.append(neighbor); seen.add(cid)
                        previous = neighbor
        result = []
        for row in ordered:
            if self.counter.count_text(json.dumps(result + [row], ensure_ascii=False)) <= budget:
                result.append(row)
        # 回读时再次鉴权，防止检索过程中会话被关闭；不信任Milvus返回的ID。
        valid = await asyncio.to_thread(self.store.chunks, self.user_id, self.session_id,
                                        self.topic(state), ids=[r["id"] for r in result], limit=self.settings.max_chunks)
        valid_ids = {r["id"] for r in valid}
        result = [r for r in result if r["id"] in valid_ids]
        audit = {"event": "retrieval", "layer": layer, "query": query, "turn_id": state.get("turn_id"),
                 "task_hash": state.get("session_task_hash"), "corpus_chunks": count, "dense_count": len(dense),
                 "lexical_count": len(lexical), "coarse_count": len(coarse), "mode": mode,
                 **retrieval_audit,
                 "selected_ids": [r["id"] for r in result], "selected_tokens": self.counter.count_text(json.dumps(result, ensure_ascii=False)),
                 "budget_tokens": budget, "duration_seconds": time.monotonic() - started}
        corpus_chars = sum(len(r["content"]) for r in rows)
        selected_chars = sum(len(r["content"]) for r in result)
        audit.update(corpus_characters=corpus_chars, selected_characters=selected_chars,
                     selection_char_fraction=selected_chars / corpus_chars if corpus_chars else 0.0)
        return result, audit

    async def resolve_units(self, state, document_id, unit_ids):
        if not isinstance(unit_ids, list) or not unit_ids or any(not isinstance(i, str) for i in unit_ids):
            raise ValueError("结构化证据编号无效")
        rows = await asyncio.to_thread(self.store.chunks, self.user_id, self.session_id, self.topic(state), ids=unit_ids)
        if len(rows) != len(set(unit_ids)) or any(r["metadata"]["document_id"] != document_id for r in rows):
            raise ValueError("证据单元不属于当前正文或会话")
        return await self.references(state, rows)

    async def references(self, state, rows):
        """证据文本直接来自RAG块，不再加载全文并按字符偏移重建或比对。"""
        refs = []
        for row in rows:
            meta = row["metadata"]
            refs.append({"source_id": meta["source_id"], "document_id": meta["document_id"],
                         "source_url": meta.get("source_url", ""), "source_title": meta.get("source_title", ""),
                         "sentence_ids": meta["sentence_ids"], "evidence_unit_ids": [row["id"]],
                         "source_spans": meta["source_spans"], "unit_type": row["unit_type"], "quote": row["content"]})
        return refs

    async def sentence_references(self, state, document_id, sentence_ids):
        """历史句号引用映射到现有证据块；表格仍使用带表头的完整单元。"""
        if (not isinstance(document_id, str) or not document_id or not isinstance(sentence_ids, list)
                or not sentence_ids or any(not isinstance(sid, str) for sid in sentence_ids)
                or len(sentence_ids) != len(set(sentence_ids))):
            raise ValueError("历史证据编号无效")
        rows = await asyncio.to_thread(self.store.chunks, self.user_id, self.session_id,
            self.topic(state), kinds={"sentence", "table", "record"}, limit=self.settings.max_chunks)
        wanted = set(sentence_ids)
        rows = [row for row in rows if row["metadata"]["document_id"] == document_id
                and wanted.intersection(row["metadata"]["sentence_ids"])]
        if wanted - {sid for row in rows for sid in row["metadata"]["sentence_ids"]}:
            raise ValueError("历史句号引用没有对应的证据块")
        rows.sort(key=lambda row: (row["metadata"]["char_start"], row["id"]))
        return await self.references(state, rows)

    async def evidence_prompt(self, state, query, *, layer="context"):
        rows, audit = await self.retrieve(state, query, layer=layer, neighbors=True)
        state.setdefault("evidence_rag_metrics", []).append(audit)
        self._prompt_metrics = state["evidence_rag_metrics"]
        # 文本和定位按单元提供；fact正文仅为索引，避免把其全部支持句重新组装。
        payload = {"evidence_units": [{"id": r["id"], "text": r["content"], **r["metadata"]} for r in rows],
                   "instruction": "原子论断可引用document_id与evidence_unit_ids；只选实际提供的编号，引用文本直接取对应证据块。表格保持整块。"}
        return BEGIN + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + END

    async def claim_inputs(self, state):
        result = []
        for i, claim in enumerate(state.get("claims") or []):
            refs, errors = [], list(claim.get("binding_errors") or [])
            for citation in claim.get("evidence") or []:
                if not isinstance(citation, dict):
                    errors.append("invalid_citation_structure"); continue
                try:
                    if citation.get("evidence_unit_ids"):
                        refs.extend(await self.resolve_units(state, citation.get("document_id"), citation["evidence_unit_ids"]))
                    else:
                        refs.extend(await self.sentence_references(state, citation.get("document_id"),
                                                                   citation.get("sentence_ids")))
                except (ValueError, KeyError):
                    errors.append("invalid_evidence_unit_ids_or_document")
            preferred = {r["document_id"] for r in refs}
            # 原绑定直接取证据块，再检索整个当前主题；反证不因首层未命中而被排除。
            rows, audit = await self.retrieve(state, claim["claim"], layer="citation", neighbors=True)
            audit["seed_document_ids"] = sorted(preferred)
            state.setdefault("evidence_rag_metrics", []).append(audit)
            candidates = await self.references(state, rows)
            for ref in candidates:
                ref["retrieval_candidate"] = True
            seen = {(r["document_id"], r["quote"]) for r in refs}
            refs.extend(r for r in candidates if (r["document_id"], r["quote"]) not in seen)
            result.append({"claim_id": claim.get("claim_id") or f"claim_{i}", "claim": claim["claim"],
                           "references": refs, "binding_errors": list(dict.fromkeys(errors))})
        return result

    async def close(self):
        await self.api.close()


async def evidence_prompt(state, query=None, fact_ids=None):
    service = current_session_evidence()
    if service is None:
        from service.deep_research_v2.writer_claims import format_claim_evidence, format_source_evidence
        if not state.get("facts"):
            return format_source_evidence(state, query or state.get("query") or "")
        # 兼容已保存事实的历史运行；新研究直接使用来源正文。
        return format_claim_evidence(state, fact_ids)
    return await service.evidence_prompt(state, query or service.query(state))


async def revision_evidence_prompt(state, section_id, fact_ids, *, query=None):
    service = current_session_evidence()
    if service is None:
        from service.deep_research_v2.writer_claims import format_section_revision_evidence, format_source_evidence
        if not state.get("facts"):
            return format_source_evidence(state, (state.get("query") or "") + "\n" + (query or ""), section_id=section_id)
        return format_section_revision_evidence(state, section_id, fact_ids)
    section = next((s for s in state.get("outline", []) if s.get("id") == section_id), {})
    query = service.query(state) + "\n" + section.get("title", "") + "\n" + section.get("description", "") + "\n" + (query or "")
    return await service.evidence_prompt(state, query)


def fit_evidence_prompt(system, prompt, scope, agent_name, output_tokens):
    service = current_session_evidence()
    if service is None or BEGIN not in prompt:
        return prompt
    from service.agent_context import protected_agent_messages, scoped_input_budget, ContextBudgetExceeded
    budget = scoped_input_budget(scope, agent_name, output_tokens)
    target = budget - service.settings.history_reserve_tokens
    original_tokens = service.counter.count_messages(protected_agent_messages(scope, system, prompt))
    # 每次只移除完整单元；不截断原句、表格或最新任务。
    while service.counter.count_messages(protected_agent_messages(scope, system, prompt)) > target:
        start = prompt.index(BEGIN) + len(BEGIN)
        end = prompt.index(END, start)
        payload = json.loads(prompt[start:end])
        if not payload["evidence_units"]:
            actual = service.counter.count_messages(protected_agent_messages(scope, system, prompt))
            if actual > budget:
                raise ContextBudgetExceeded(budget, actual)
            break
        payload["evidence_units"].pop()
        prompt = prompt[:start] + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + prompt[end:]
    service.last_prompt_budget = {"original_tokens": original_tokens,
        "prepared_tokens": service.counter.count_messages(protected_agent_messages(scope, system, prompt)),
        "input_budget": budget, "history_reserve_tokens": service.settings.history_reserve_tokens}
    if hasattr(service, "_prompt_metrics"):
        service._prompt_metrics.append({"event": "prompt_budget", **service.last_prompt_budget,
            "retained_token_fraction": service.last_prompt_budget["prepared_tokens"] / max(1, original_tokens)})
    return prompt
