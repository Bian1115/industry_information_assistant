"""分层精排与输入预算组批；跨请求候选在同一次最终精排中比较。"""
import math
import time


CONTEXT_INSTRUCTION = (
    "Given the current research question and section requirements, retrieve passages "
    "that provide relevant facts, explanations, comparisons, and constraints. Respect "
    "the requested entities, time period, geography, and scope. Include relevant "
    "contrary findings as well as supporting findings."
)
CITATION_INSTRUCTION = (
    "Retrieve evidence relevant to verifying the claim, including support, negation, "
    "corrections, and necessary conditions. Match the entities, time, numbers, units, "
    "and scope. Do not prefer supporting evidence over contrary evidence."
)


class RerankBudgetError(ValueError):
    pass


async def rerank_candidates(api, counter, query, documents, top_k, *, layer, settings, audit):
    """所有候选至少精排一次；预算允许时直接整体精排，否则逐轮缩减候选。"""
    instruction = CONTEXT_INSTRUCTION if layer == "context" else CITATION_INSTRUCTION
    query_tokens = counter.count_text(query)
    instruction_tokens = counter.count_text(instruction)
    document_tokens = [counter.count_text(text) for text in documents]
    audit.update(rerank_status="running", rerank_candidate_count=len(documents),
                 rerank_model=settings.rerank_model, rerank_calls=[],
                 rerank_scored_candidate_count=0, rerank_query_tokens=query_tokens,
                 rerank_token_count_kind="local_estimate", rerank_rounds=0)
    if query_tokens + instruction_tokens > settings.rerank_max_input_tokens:
        raise RerankBudgetError("精排query与任务指令超过单条输入预算，未截断当前任务")
    if any(tokens > settings.rerank_max_input_tokens for tokens in document_tokens):
        raise RerankBudgetError("精排候选超过单条输入预算，未截断原文")
    # 为任务指令也预留每个query/document对的输入空间。
    costs = [query_tokens + instruction_tokens + tokens for tokens in document_tokens]
    if any(cost > settings.rerank_max_request_tokens for cost in costs):
        raise RerankBudgetError("单个query/document对超过整批精排预算")

    active = list(range(len(documents)))
    scored = set()
    while active:
        batches, batch, used = [], [], 0
        for index in active:
            if batch and (used + costs[index] > settings.rerank_max_request_tokens
                          or len(batch) >= settings.rerank_max_documents):
                batches.append(batch)
                batch, used = [], 0
            batch.append(index)
            used += costs[index]
        if batch:
            batches.append(batch)
        final = len(batches) == 1
        if not final and all(len(batch) <= top_k for batch in batches):
            raise RerankBudgetError("精排预算无法容纳统一TopK比较，未按跨请求相对分数拼接排名")
        audit["rerank_rounds"] += 1
        winners = []
        for number, batch in enumerate(batches, 1):
            size = min(top_k, len(batch))
            call = {"round": audit["rerank_rounds"], "batch": number,
                    "phase": "final" if final else "reduce", "candidate_indices": batch,
                    "candidate_count": len(batch), "input_tokens": sum(costs[i] for i in batch),
                    "top_n": size, "status": "running"}
            audit["rerank_calls"].append(call)
            started = time.monotonic()
            try:
                ranking = await api.rerank(query, [documents[i] for i in batch], size,
                                           instruct=instruction)
                api_metadata = getattr(api, "last_rerank_metadata", {})
                if api_metadata:
                    call["api_usage"] = api_metadata.get("usage", {})
                    call["api_request_id"] = api_metadata.get("request_id")
                indexes = [row["index"] for row in ranking]
                if (len(indexes) != size or len(set(indexes)) != len(indexes)
                        or any(type(i) is not int or not 0 <= i < len(batch) for i in indexes)):
                    raise ValueError("精排返回编号无效")
                if any("relevance_score" in row and (
                        isinstance(row["relevance_score"], bool)
                        or not isinstance(row["relevance_score"], (int, float))
                        or not math.isfinite(row["relevance_score"])) for row in ranking):
                    raise ValueError("精排相关性分数无效")
                mapped = [{**row, "index": batch[row["index"]]} for row in ranking]
                call.update(status="completed", selected_indices=[row["index"] for row in mapped])
                scored.update(batch)
                audit["rerank_scored_candidate_count"] = len(scored)
            except Exception as exc:
                call.update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                call["duration_seconds"] = time.monotonic() - started
            if final:
                audit["rerank_status"] = "completed"
                return mapped
            winners.extend(row["index"] for row in mapped)
        active = winners
    audit["rerank_status"] = "not_needed"
    return []
