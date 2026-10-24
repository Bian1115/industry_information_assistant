"""按完整消息的 tokenizer 预算分批提取；正文和原句编号始终留在程序中。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable
import uuid

from service.agent_context import (
    ContextBudgetExceeded, evidence_batch_input, protected_agent_messages,
    scoped_input_budget,
)
from service.agent_history import canonical_json, current_history_scope, record_agent_message
from service.qwen_token_counter import get_qwen_token_counter
from service.web_document import sentence_records, source_usable


BATCH_INSTRUCTION = (
    "\n本次只提取当前独立正文批次中的证据，不需要回读历史。"
    "不同批次由程序合并；只引用本批实际展示的正文编号和原句编号。"
)


@dataclass
class EvidenceBatch:
    prompt: str
    tokens: int
    # 精确 URL、正文编号、原句编号绑定；不允许引用另一批未展示的句子。
    bindings: dict[tuple[str, str], set[str]]


def render_materials(results, units):
    grouped = {}
    for index, row in units:
        grouped.setdefault(index, []).append(row)
    parts, bindings = [], {}
    for index, rows in grouped.items():
        source = results[index]
        url, document_id = str(source.get("url") or ""), str(source.get("document_id") or "")
        bindings.setdefault((url, document_id), set()).update(row["id"] for row in rows)
        parts.append(
            f"[{index + 1}] {source.get('title', 'N/A')}\n"
            f"URL: {url}\n来源: {source.get('site_name', 'N/A')}\n"
            f"日期: {source.get('date', 'N/A')}\n搜索摘要: {str(source.get('summary') or '')[:500]}\n"
            f"已编号正文（已读取正文）: 正文编号: {document_id}；质量: {source.get('body_quality', {}).get('status', 'unknown')}\n"
            + "\n".join(f"[{row['id']}] {row['text']}" for row in rows)
        )
    return "\n\n".join(parts), bindings


def plan_evidence_batches(agent, results, prompt_builder: Callable[[str], str], system_prompt="", *, counter=None):
    """覆盖全部合格编号正文，按完整原句切分，不再受每页 12000 字展示上限影响。"""
    scope = current_history_scope()
    counter = counter or get_qwen_token_counter()
    budget = scoped_input_budget(scope, agent.name, min(16000, agent._output_limit()))
    units = []
    for index, source in enumerate(results):
        document = {**source, "content": source.get("full_content", source.get("content", ""))}
        rows = sentence_records(document) if source_usable(document) else []
        source["visible_sentence_ids"] = []
        units.extend((index, row) for row in rows)

    def build(selected):
        material, bindings = render_materials(results, selected)
        prompt = prompt_builder(material) + BATCH_INSTRUCTION
        tokens = counter.count_messages(protected_agent_messages(scope, system_prompt, prompt))
        return EvidenceBatch(prompt, tokens, bindings)

    empty = build([])
    if empty.tokens > budget:
        # 任务/假设/契约本身已超限，不能通过删除正文或指令假装成功。
        raise ContextBudgetExceeded(budget, empty.tokens)
    batches, start = [], 0
    while start < len(units):
        # 同文档跨批次保留一条相邻原句，减少指代和表格说明被切开的影响。
        overlap = units[start - 1:start] if start and units[start - 1][0] == units[start][0] else []
        first = build(overlap + units[start:start + 1])
        if first.tokens > budget and overlap:
            overlap = []
            first = build(units[start:start + 1])
        if first.tokens > budget:
            # 保留完整原句和编号；不把一条过长句子静默截断成虚假引用。
            raise ContextBudgetExceeded(budget, first.tokens)
        best, end = first, start + 1
        low, high = end + 1, len(units)
        while low <= high:
            middle = (low + high) // 2
            candidate = build(overlap + units[start:middle])
            if candidate.tokens <= budget:
                best, end, low = candidate, middle, middle + 1
            else:
                high = middle - 1
        batches.append(best)
        start = end
    return batches, budget


def _unique(values):
    result, seen = [], set()
    for value in values:
        key = canonical_json(value)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def merge_batch_analysis(merged, analysis, batch, fact_keys):
    """先核对本批编号，再合并事实与关系；不做语义支持性核验。"""
    raw_facts = analysis.get("extracted_facts", [])
    if not isinstance(raw_facts, list):
        raise ValueError("正文提取响应 extracted_facts 必须为数组")
    counts = Counter(str(fact.get("fact_id") or "") for fact in raw_facts if isinstance(fact, dict))
    local_ids, rejected = {}, 0
    for fact in raw_facts:
        if not isinstance(fact, dict):
            rejected += 1
            continue
        key = (str(fact.get("source_url") or ""), str(fact.get("document_id") or ""))
        ids = fact.get("sentence_ids")
        allowed = batch.bindings.get(key, set())
        if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in allowed for i in ids)
                or len(ids) != len(set(ids)) or not str(fact.get("content") or "").strip()):
            rejected += 1
            continue
        fact_key = (*key, str(fact["content"]).strip())
        existing = fact_keys.get(fact_key)
        local_id = str(fact.get("fact_id") or "")
        if existing is None:
            # 在整个合并结果中从 0 连续编号；批次信息独立写入运行记录。
            existing = {**fact, "fact_id": f"f_{len(merged['extracted_facts'])}",
                        "sentence_ids": list(ids)}
            fact_keys[fact_key] = existing
            merged["extracted_facts"].append(existing)
        else:
            existing["sentence_ids"] = sorted(set(existing["sentence_ids"]) | set(ids))
            if isinstance(fact.get("data_points"), list):
                previous_points = existing.get("data_points")
                existing["data_points"] = _unique(
                    (previous_points if isinstance(previous_points, list) else []) + fact["data_points"])
        if local_id and counts[local_id] == 1:
            local_ids[local_id] = existing["fact_id"]
    relations = analysis.get("hypothesis_evidence", [])
    for relation in relations if isinstance(relations, list) else []:
        if isinstance(relation, dict) and str(relation.get("fact_id") or "") in local_ids:
            merged.setdefault("hypothesis_evidence", []).append(
                {**relation, "fact_id": local_ids[str(relation["fact_id"])]})
    for key, value in analysis.items():
        if key in {"extracted_facts", "hypothesis_evidence"} or key.startswith("_"):
            continue
        if isinstance(value, list):
            merged[key] = _unique(merged.get(key, []) + value)
        elif isinstance(value, str) and value:
            previous = merged.get(key, "")
            merged[key] = previous + "\n" + value if previous and value != previous else value
    if "hypothesis_evidence" in merged:
        merged["hypothesis_evidence"] = _unique(merged["hypothesis_evidence"])
    return rejected


async def analyze_evidence_batches(agent, results, prompt_builder, system_prompt=""):
    batches, budget = plan_evidence_batches(agent, results, prompt_builder, system_prompt)
    merged, fact_keys = {"extracted_facts": []}, {}
    call_id = str(uuid.uuid4())
    await record_agent_message(agent.name, "evidence_batch_plan", {
        "batch_count": len(batches), "budget": budget,
        "input_tokens": [batch.tokens for batch in batches],
        "sentence_appearances": sum(sum(len(ids) for ids in batch.bindings.values()) for batch in batches),
        "unique_sentence_count": len({(url, doc, ident) for batch in batches
                                       for (url, doc), ids in batch.bindings.items() for ident in ids}),
        "history_mode": "independent_extraction",
    }, call_id)
    # 全批成功才将可见句编号交给入库；失败不返回不完整的分析冒充成功。
    for index, batch in enumerate(batches, 1):
        try:
            with evidence_batch_input():
                response = await agent.call_llm(system_prompt=system_prompt, user_prompt=batch.prompt,
                                                json_mode=True, temperature=0.2)
            parsed = agent.parse_json_response(response)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("extracted_facts"), list):
                raise ValueError("正文提取响应缺少有效 extracted_facts，不能跳过本批")
            rejected = merge_batch_analysis(merged, parsed, batch, fact_keys)
            await record_agent_message(agent.name, "evidence_batch_complete", {
                "batch_index": index, "batch_count": len(batches), "input_tokens": batch.tokens,
                "budget": budget, "rejected_fact_count": rejected,
                "fact_count": len(parsed["extracted_facts"]),
            }, call_id)
        except BaseException as exc:
            await record_agent_message(agent.name, "evidence_batch_error", {
                "batch_index": index, "batch_count": len(batches), "error_type": type(exc).__name__,
                "message": str(exc),
            }, call_id)
            raise
    visible = {}
    for batch in batches:
        for key, ids in batch.bindings.items():
            visible.setdefault(key, set()).update(ids)
    for source in results:
        key = (str(source.get("url") or ""), str(source.get("document_id") or ""))
        source["visible_sentence_ids"] = sorted(visible.get(key, set()))
    return merged
