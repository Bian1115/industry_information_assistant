"""只派生旧模型输入的回放视图；原始消息、哈希和全局摘要覆盖游标不变。"""
from __future__ import annotations

import re
import json

from service.agent_history import canonical_json, content_hash, record_agent_message

VERSION = "input-view-v2"
# 只识别程序已有的资料栏目；不在任务要求等任意位置删除相同文本。
MATERIAL_START = re.compile(r"(?m)^(?:已读取正文[:：] ?|(?:搜索)?摘要[:：] ?|## 文档内容\r?\n)")
BODY_FIELDS = {"content", "full_content", "summary", "snippet", "text"}
INPUT_MATERIAL_FIELDS = {"source_evidence", "claim_evidence", "original_content", "original_report", "draft_sections"}
OUTPUT_ARTIFACT_FIELDS = {"section_content", "content", "revised_content", "executive_summary", "conclusions",
                          "outlook", "extracted_facts", "entities_discovered", "outline"}


def reference_agent_materials(row, text, counter, min_tokens, *, output=False, material_fields=None):
    """引用自身已持久化的材料字段，适用于没有网页工具记录的 Writer/Architect。"""
    fields = (material_fields if material_fields is not None else
              OUTPUT_ARTIFACT_FIELDS if output else INPUT_MATERIAL_FIELDS)
    spans = []
    decoder = json.JSONDecoder()
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                # 逐项解码根对象，确保不匹配字符串内部或嵌套的同名字段。
                cursor = len(text) - len(stripped) + 1
                while cursor < len(text):
                    while cursor < len(text) and text[cursor] in " \r\n\t,":
                        cursor += 1
                    if text[cursor:cursor + 1] == "}":
                        break
                    key, cursor = decoder.raw_decode(text, cursor)
                    while text[cursor] in " \r\n\t":
                        cursor += 1
                    if text[cursor] != ":":
                        break
                    cursor += 1
                    while text[cursor] in " \r\n\t":
                        cursor += 1
                    start = cursor
                    material, cursor = decoder.raw_decode(text, cursor)
                    if key in fields and isinstance(material, (str, list, dict)):
                        spans.append((start, cursor, [key], True))
        except (ValueError, IndexError):
            spans = []
    if not output:
        for heading, following, field in (
                ("原始报告", "审核反馈", "original_report"),
                ("原章节正文", "本章需要处理的审核反馈", "original_content"),
                ("各章节内容", "假设与可审计证据账本", "draft_sections")):
            match = re.search(r"(?m)^## " + heading + r"\r?\n", text)
            if match:
                end = re.search(r"(?m)^## " + following + r"\r?$", text[match.end():])
                if end:
                    spans.append((match.end(), match.end() + end.start(), [field], False))
        search = re.search(r"(?m)^## 搜索结果\r?\n", text)
        if search:
            end = re.search(r"(?m)^## 任务", text[search.end():])
            if end:
                spans.append((search.end(), search.end() + end.start(), ["search_results"], False))
        for match in re.finditer(r"(?m)^#{2,3} 可逐字引用的原文片段[^\n]*\n", text):
            start = match.end()
            while start < len(text) and text[start].isspace():
                start += 1
            try:
                _, end = decoder.raw_decode(text, start)
                spans.append((start, end, ["claim_evidence"], False))
            except ValueError:
                pass
    replacements = []
    for start, end, path, json_value in spans:
        piece = text[start:end]
        marker = f"[资料已保存；source_id={row['source_id']}；原文回读start={start}；chars={end-start}；按需分段回读]"
        replacement = json.dumps(marker, ensure_ascii=False) if json_value else marker
        if counter.count_text(piece) < min_tokens or counter.count_text(replacement) >= counter.count_text(piece):
            continue
        replacements.append({"input_start": start, "input_end": end, "source_id": row["source_id"],
                             "source_hash": row["content_hash"], "field_path": path,
                             "start": start, "end": end, "marker": replacement})
    result = text
    for ref in reversed(sorted(replacements, key=lambda item: item["input_start"])):
        result = result[:ref["input_start"]] + ref["marker"] + result[ref["input_end"]:]
    return result, replacements


def body_fields(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in BODY_FIELDS and isinstance(child, str):
                yield (*path, key), child
            elif isinstance(child, (list, dict)):
                yield from body_fields(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from body_fields(child, (*path, index))


def reference_materials(text, tools, counter, min_tokens):
    """匹配工具原文的完整内容或连续前缀（兼容Scout的4500字符展示）。"""
    replacements = []
    occupied_until = 0
    for match in MATERIAL_START.finditer(text):
        start = match.end()
        if start < occupied_until:
            continue
        best = None
        for row in tools:
            if row["content_hash"] != content_hash(row["payload"]):
                raise ValueError("资料来源快照哈希不一致")
            for path, body in body_fields(row["payload"]):
                # 短前缀不构成可靠资料定位；只取逐字一致的连续片段。
                if len(body) < 64 or not text.startswith(body[:64], start):
                    continue
                length = 64
                while length < len(body) and start + length < len(text) and body[length] == text[start + length]:
                    length += 1
                piece = body[:length]
                if counter.count_text(piece) < min_tokens:
                    continue
                if best is None or length > best[0]:
                    best = length, row, path, piece
        if best is None:
            continue
        length, row, path, piece = best
        raw = canonical_json(row["payload"])
        encoded = canonical_json(piece)[1:-1]
        source_start = raw.find(encoded)
        if source_start < 0:
            continue
        marker = f"[资料已保存；source_id={row['source_id']}；原文回读start={source_start}；需要时通过_history_read读取]"
        if counter.count_text(marker) >= counter.count_text(piece):
            continue
        replacements.append({"input_start": start, "input_end": start + length,
                             "source_id": row["source_id"], "source_hash": row["content_hash"],
                             "field_path": list(path), "start": source_start,
                             "end": source_start + len(encoded), "marker": marker})
        occupied_until = start + length
    result = text
    for item in reversed(replacements):
        result = result[:item["input_start"]] + item["marker"] + result[item["input_end"]:]
    return result, replacements


async def prepare_input_views(agent, scope, rows, completed, views, counter, policy, call_id, *, covered_seq=0,
                              reference_all=False):
    """调用方确认整体阈值后派生历史引用；原文与未完成输入保持完整。"""
    from service.agent_context import source_text

    settings = policy.input_history
    stats = {"version": VERSION, "referenced_source_ids": []}
    if settings is None or not (settings.reference_materials and (policy.artifact_offload or policy.agent_compression is not None)
                                and policy.history_read_max_calls != 0):
        return stats
    # 全量遍历时仅保留“引用比原文短”的检查，不再按单段token门槛筛选。
    min_tokens = 0 if reference_all else settings.min_reference_tokens
    tools = [r for r in rows if r["kind"] == "tool_output" and r["call_id"] in completed]
    eligible = [r for r in rows if r["kind"] == "llm_input" and r["call_id"] in completed
                and not r.get("interrupted")]
    if covered_seq > 0 and any(MATERIAL_START.search(source_text(r)) for r in eligible):
        archived = await scope.store.load_tool_sources(scope, agent.name, through_seq=covered_seq)
        tools = [*archived, *tools]
        stats["covered_tool_source_count"] = len(archived)
    verified_sources = {}
    for row in eligible:
        if row["content_hash"] != content_hash(row["payload"]):
            raise ValueError("旧模型输入快照哈希不一致")
        original = source_text(row)
        view, references = (reference_agent_materials(row, original, counter, min_tokens)
                            if policy.agent_compression is not None else (original, []))
        if not references:
            view, references = reference_materials(
                original, [t for t in tools if t["seq"] < row["seq"]], counter,
                min_tokens,
            )
        if not references:
            continue
        # 来源需通过同run/agent/subtask的实际读取权限检查。
        for ref in references:
            source_id = ref["source_id"]
            if source_id not in verified_sources:
                verified_sources[source_id] = await scope.store.read_source(scope, agent.name, source_id)
            source = verified_sources[source_id]
            if source["content_hash"] != ref["source_hash"] or source["content_hash"] != content_hash(source["payload"]):
                raise ValueError("回读资料与引用化来源哈希不一致")
        views[row["source_id"]] = view
        stats["referenced_source_ids"].append(row["source_id"])
        await record_agent_message(agent.name, "input_history_references", {
            "version": VERSION, "source_id": row["source_id"], "source_hash": row["content_hash"],
            "config_hash": policy.fingerprint, "references": references,
            "tokens_before": counter.count_text(original), "tokens_after": counter.count_text(view),
        }, call_id, scope=scope)
    return stats
