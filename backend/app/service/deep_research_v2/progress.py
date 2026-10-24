"""工作台计数是进度观察，不改变图检查点或恢复位置。"""


def progress_counts(state):
    return {
        "queries": len((state.get("search_budget") or {}).get("queries") or []),
        "sources": len(state.get("raw_sources") or []),
        "facts": len(state.get("facts") or []),
        "references": len(state.get("references") or []),
    }


def pending_progress(state, payload):
    """只接受当前未提交节点的观察；提交后以检查点为准，允许清理后计数下降。"""
    node = payload.get("node")
    counts = payload.get("progress")
    if node not in state.get("pending_nodes", []) or not isinstance(counts, dict):
        return None
    committed = state["draft"]["node_attempts"].get(node, 0)
    attempt = payload.get("attempt", 0)
    if not isinstance(attempt, int) or attempt <= committed:
        return None
    if not all(type(counts.get(key)) is int and counts[key] >= 0
               for key in ("queries", "sources", "facts", "references")):
        return None
    return {"counts": counts, "node": node, "attempt": attempt, "basis": "in_progress"}
