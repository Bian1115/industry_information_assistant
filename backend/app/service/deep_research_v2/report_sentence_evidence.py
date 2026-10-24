"""报告与摘要共用证据句选择协议；块内编号只解析成实际提供的原句。"""
from copy import deepcopy
import hashlib
import json

from service.summary_evidence_units import split_evidence_units, render_spans
from service.summary_sentence_evidence import selection_batches, selection_messages, parse_selection

VERSION = "report-selected-sentences-v1"


async def call_selector(messages, settings):
    from .report_claim_extraction import call_report_model
    return await call_report_model(messages, settings, component="report_evidence_sentence_selection")


def narrow_reference(reference, selection):
    """按程序保存的块内位置回读，不将未选择的句子或整个父块重新带入。"""
    block = reference["quote"]
    if selection.get("block_sha256") != hashlib.sha256(block.encode()).hexdigest():
        raise ValueError("选句对应的证据块版本已变化")
    spans = selection.get("spans")
    if not isinstance(spans, list) or not spans or any(
        not isinstance(span, dict) or type(span.get("start")) is not int or type(span.get("end")) is not int
        or not 0 <= span["start"] < span["end"] <= len(block) for span in spans
    ):
        raise ValueError("选句位置无效，不能回退到整块原文")
    return {**reference, "quote": render_spans(block, spans), "sentence_selection": deepcopy(selection)}


async def select_report_sentences(claim, references, settings, *, call=None, counter=None):
    from service.qwen_token_counter import get_qwen_token_counter
    counter = counter or get_qwen_token_counter()
    audit = {"version": VERSION, "phase": "evidence_selection", "batches": [], "selected_ids": []}
    claim["evidence_selection"] = audit
    candidates, blocks = [], {}
    for reference in references:
        # source_id在共用协议中标识实际提供的块，不把网页ID当成块内句子编号。
        payload = json.dumps({key: reference.get(key) for key in
            ("document_id", "evidence_unit_ids", "sentence_ids", "source_spans", "quote")},
            ensure_ascii=False, sort_keys=True)
        block_id = "block_" + hashlib.sha256(payload.encode()).hexdigest()[:24]
        if block_id in blocks:
            continue
        blocks[block_id] = reference
        for i, piece in enumerate(split_evidence_units(reference["quote"], 0, len(reference["quote"]),
                                      lambda text: counter.count_text(text) <= 384), 1):
            candidates.append({"source_id": block_id, "sentence_id": f"s{i:04d}", **piece})
    audit["candidate_sentences_count"] = len(candidates)
    chosen = set()
    try:
        # 与摘要选句使用相同消息结构、分批器和严格编号解析器。
        input_budget = min(16000, settings.max_length, 262144 - settings.output_tokens - settings.safety_tokens)
        for index, batch in enumerate(selection_batches(claim, candidates, counter, input_budget), 1):
            messages = selection_messages(claim, batch)
            output, metadata = await (call or call_selector)(messages, settings)
            entry = {"batch_index": index, "input_sha256": hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "raw_output": output, "call": metadata}
            audit["batches"].append(entry)
            selected = parse_selection(output, claim, batch)
            entry["selected_ids"] = [{"source_id": item["source_id"], "sentence_id": item["sentence_id"]}
                                     for item in selected]
            chosen.update((item["source_id"], item["sentence_id"]) for item in selected)
        result = []
        for block_id, reference in blocks.items():
            selected = [item for item in candidates if item["source_id"] == block_id
                        and (block_id, item["sentence_id"]) in chosen]
            if not selected:
                continue
            spans = {(span["start"], span["end"]) for item in selected
                     for span in item.get("source_spans") or [{"start": item["start"], "end": item["end"]}]}
            selection = {"version": VERSION, "block_id": block_id,
                "block_sha256": hashlib.sha256(reference["quote"].encode()).hexdigest(),
                "sentence_ids": [item["sentence_id"] for item in selected],
                "spans": [{"start": start, "end": end} for start, end in sorted(spans)]}
            audit["selected_ids"].append({"source_id": block_id, "sentence_ids": selection["sentence_ids"]})
            result.append(narrow_reference(reference, selection))
        audit.update(phase="completed", selected_references_count=len(result))
        return result
    except Exception as exc:
        audit.update(phase="error", error_type=type(exc).__name__, error=str(exc))
        raise
