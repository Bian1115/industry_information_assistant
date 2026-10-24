"""编号并保存摘要模型实际输入的消息片段，不以未提供的历史原文作替代证据。"""
from __future__ import annotations

import json

from service.agent_history import canonical_json, content_hash, record_agent_message

CONTEXT_KIND = "summary_context_source"
CONTEXT_SCOPE = "summary_input_context"


async def number_input_context(request, scope, agent_name, call_id, source_rows):
    """新消息沿用已有ID；输入的旧/中间摘要另存快照并把ID放入实际请求。

    候选仅包含本请求提供的字符范围；完整快照仅用于定位，不扩大候选。
    格式修复沿用original_request，不以待修复的输出证明自身。
    """
    payload = json.loads(request[1]["content"])
    data = payload.get("original_request", payload)
    sources, rows = {}, {}
    for fragment in data.get("new_records", []):
        sid, text = fragment["source_id"], fragment["text"]
        row = source_rows.get(sid)
        if row is None or row["content_hash"] != content_hash(row["payload"]):
            raise ValueError("摘要上下文消息编号或快照哈希无效")
        from service.agent_context import source_text
        start = fragment.get("start", 0)
        if type(start) is not int or start < 0 or source_text(row)[start:start + len(text)] != text:
            raise ValueError("摘要上下文片段与对应消息原文不一致")
        sources.setdefault(sid, []).append({"start": start, "text": text})
        rows[sid] = row

    async def snapshot(text, label, existing_id=None):
        if existing_id:
            row = await scope.store.read_source(scope, agent_name, existing_id)
            if (row["kind"] != CONTEXT_KIND or row["content_hash"] != content_hash(row["payload"])
                    or row["payload"].get("content") != text):
                raise ValueError("摘要输入上下文快照与实际请求不一致")
        else:
            row = await record_agent_message(agent_name, CONTEXT_KIND, {
                "content": text, "context_role": label, "verification_scope": CONTEXT_SCOPE,
                "generation_call_id": call_id,
            }, call_id, scope=scope)
        sid = row["source_id"]
        rows[sid] = row
        sources[sid] = [{"start": 0, "text": text}]
        return sid

    previous = data.get("previous_summary")
    if previous:
        data["previous_summary_source_id"] = await snapshot(
            previous, "previous_summary", data.get("previous_summary_source_id"))
    parts = data.get("summary_parts", [])
    ids = data.get("summary_part_source_ids", [])
    if ids and len(ids) != len(parts):
        raise ValueError("中间摘要上下文编号数量不一致")
    if parts:
        data["summary_part_source_ids"] = [await snapshot(text, "summary_part", ids[i] if ids else None)
                                            for i, text in enumerate(parts)]
    contract = data.get("latest_contract")
    if contract:
        data["latest_contract_source_id"] = await snapshot(
            canonical_json(contract), "latest_contract", data.get("latest_contract_source_id"))
    request[1]["content"] = canonical_json(payload)
    return {"sources": sources, "rows": rows, "input_hash": content_hash(request),
            "verification_scope": CONTEXT_SCOPE}


def locate_context_quote(ref, context, *, max_chars=2400):
    """只从实际提供的片段补相邻句，禁止定位后扩到未输入的消息正文。"""
    from service.agent_context import source_text
    from service.verification_checks import locate_quote, source_type
    sid = ref["source_id"]
    pieces = [p for p in context["sources"][sid]
              if p["start"] <= ref["start"] and ref["end"] <= p["start"] + len(p["text"])]
    if not pieces:
        raise ValueError("引用位置超出生成摘要时实际提供的上下文")
    piece = max(pieces, key=lambda p: len(p["text"]))
    start = piece["start"]
    positioned = locate_quote({**ref, "start": ref["start"] - start, "end": ref["end"] - start},
                              piece["text"], source_type(context["rows"][sid]), max_chars=max_chars)
    for key in ("start", "end", "context_start", "context_end"):
        positioned[key] += start
    positioned.update(verification_scope=CONTEXT_SCOPE,
                      source_text_hash=content_hash(source_text(context["rows"][sid])))
    return positioned
