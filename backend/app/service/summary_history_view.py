"""将摘要实际读取的处理后历史保存为独立快照；原始记录及覆盖游标仍由原历史决定。"""
from service.agent_history import content_hash, record_agent_message

VIEW_KIND = "summary_view_source"
VIEW_VERSION = "summary-history-view-v1"


async def prepare_summary_records(agent, scope, records, views, counter, call_id):
    from service.agent_context import source_text
    from service.verification_checks import source_type

    processed, mapping = [], []
    raw_tokens = view_tokens = 0
    for row in records:
        if row["content_hash"] != content_hash(row["payload"]):
            raise ValueError("摘要原始来源快照哈希不一致")
        original, view = source_text(row), views[row["source_id"]]
        raw_tokens += counter.count_text(original)
        view_tokens += counter.count_text(view)
        if view == original:
            processed.append(row)
            continue
        snapshot = await record_agent_message(agent.name, VIEW_KIND, {
            "content": view, "view_version": VIEW_VERSION,
            "original_source_id": row["source_id"], "original_source_hash": row["content_hash"],
            "original_seq": row["seq"], "original_kind": row["kind"],
            "original_source_type": source_type(row),
        }, call_id, scope=scope)
        processed.append(snapshot)
        mapping.append({"source_id": snapshot["source_id"], "source_hash": snapshot["content_hash"],
                        "original_source_id": row["source_id"], "original_source_hash": row["content_hash"],
                        "original_seq": row["seq"], "original_kind": row["kind"]})
    metadata = {
        "version": VIEW_VERSION, "mode": "processed_history", "raw_tokens": raw_tokens,
        "view_tokens": view_tokens, "saved_tokens": raw_tokens - view_tokens, "source_mapping": mapping,
        "input_hash": content_hash([(r["source_id"], r["content_hash"]) for r in processed]),
    }
    await record_agent_message(agent.name, "summary_input_prepared", metadata, call_id, scope=scope)
    return processed, metadata
