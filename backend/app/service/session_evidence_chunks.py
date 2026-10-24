"""不可变来源的父子chunk及双视图相邻关系；表格从分句前独立识别。"""
from collections import defaultdict
import hashlib
import re

from service.summary_evidence_units import split_evidence_units
from service.web_document import sentence_records, source_usable

VERSION = "session-evidence-native-tables-v2"


def stable_id(*parts):
    return hashlib.sha256("\n".join(map(str, parts)).encode()).hexdigest()


def chunk_document(source, *, user_id, session_id, topic_id, counter, chunk_tokens=1024):
    if not source_usable(source) or not sentence_records(source):
        raise ValueError("临时证据需要准入通过、哈希和原句编号一致的正文")
    full = source["content"]
    key = stable_id(session_id, topic_id, source["document_id"])
    headings = [(m.start(), len(m[1]), m[2].strip()) for m in
                re.finditer(r"^(#{1,6})\s+(.+?)\s*$", full, re.M)]
    paragraphs = [(m.start(), m.end()) for m in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", full)]
    original = sentence_records(source)
    fits = lambda text: counter.count_text(text) <= chunk_tokens
    pieces = split_evidence_units(full, 0, len(full), fits)
    result = []
    groups = defaultdict(list)
    for piece in pieces:
        if not fits(piece["text"]):
            raise ValueError("单句或结构化单元超过chunk预算，未截断原文")
        left, right = piece["start"], piece["end"]
        path = []
        section_start = 0
        for offset, depth, title in headings:
            if offset > left:
                break
            path = path[:depth - 1] + [title]
            section_start = offset
        pids = ["p_" + stable_id(source["document_id"], a, b)[:16]
                for a, b in paragraphs if a < right and b > left]
        spans = piece.get("source_spans") or [{"start": left, "end": right}]
        sentence_ids = [row["id"] for row in original if any(
            row["start"] < span["end"] and row["end"] > span["start"] for span in spans)]
        cid = stable_id(key, VERSION, piece["unit_type"], spans)
        metadata = {"user_id": str(user_id), "session_id": str(session_id), "topic_id": topic_id,
                    "document_id": source["document_id"], "content_sha256": source["content_sha256"],
                    "source_id": source["id"], "source_url": source.get("url", ""),
                    "source_title": source.get("title", ""), "chunk_id": cid,
                    "related_sections": list(source.get("related_sections") or []),
                    "unit_type": piece["unit_type"], "source_section_id": "sec_" + stable_id(key, section_start)[:16],
                    "heading_path": path, "paragraph_ids": pids, "sentence_ids": sentence_ids,
                    "char_start": left, "char_end": right, "source_spans": spans, "chunker_version": VERSION,
                    "parent_chunk_id": None}
        if piece["unit_type"] in {"table", "record"}:
            anchor = piece.get("table_start", piece.get("fixed_prefix"))
            metadata.update(table_id="table_" + stable_id(key, anchor)[:16],
                            header_spans=piece.get("fixed_prefix", []), note_spans=piece.get("fixed_suffix", []),
                            row_spans=piece.get("body_rows", []),
                            row_range=[r["start"] for r in piece.get("body_rows", [])],
                            quote_kind="assembled_source_spans")
            if piece["unit_type"] == "table":
                metadata.update(header_status=piece["header_status"], column_count=piece["column_count"],
                                table_requires_whole=piece["table_requires_whole"])
        row = {"id": cid, "document_key": key, "unit_type": piece["unit_type"],
               "content": piece["text"], "metadata": metadata}
        result.append(row)
        if piece["unit_type"] == "sentence":
            groups[(metadata["source_section_id"], tuple(pids))].append(row)
    # 父段落沿完整原句组装，长段落分片；不把表格塞回父段正文。
    parents = []
    for rows in groups.values():
        batch = []
        def flush():
            if not batch:
                return
            spans = [s for r in batch for s in r["metadata"]["source_spans"]]
            cid = stable_id(key, VERSION, "paragraph", spans)
            meta = {**batch[0]["metadata"], "chunk_id": cid, "unit_type": "paragraph",
                    "related_sections": list(batch[0]["metadata"]["related_sections"]),
                    "source_spans": spans, "char_end": batch[-1]["metadata"]["char_end"],
                    "sentence_ids": list(dict.fromkeys(s for r in batch for s in r["metadata"]["sentence_ids"]))}
            parents.append({"id": cid, "document_key": key, "unit_type": "paragraph",
                            "content": "\n".join(r["content"] for r in batch), "metadata": meta})
            for r in batch:
                r["metadata"]["parent_chunk_id"] = cid
        for row in rows:
            if batch and not fits("\n".join([*(r["content"] for r in batch), row["content"]])):
                flush(); batch = []
            batch.append(row)
        flush()
    result.extend(parents)
    table_rows = defaultdict(int)
    for row in sorted(result, key=lambda r: r["metadata"]["char_start"]):
        meta = row["metadata"]
        if row["unit_type"] == "table":
            count = len(meta["row_spans"])
            first = table_rows[meta["table_id"]] + 1
            meta["body_row_range"] = [first, first + count - 1] if count else []
            # 次级表头在原生 Markdown 中可能属于正文行；无法确认时不把其行号
            # 冒充语义上的数据行号，整表仍保留供引用。
            meta["row_range"] = meta["body_row_range"] if meta["header_status"] != "ambiguous" else []
            meta["row_ids"] = ["row_" + stable_id(key, span)[:16] for span in meta["row_spans"]]
            table_rows[meta["table_id"]] += count
    # 表格共用一份chunk，两种视图的邻居分别登记。
    def source_order(row):
        meta = row["metadata"]
        # 长表每片重复原表头/脚注，char_start相同；邻居必须沿实际数据行排列。
        body = meta.get("row_spans") or []
        return (min(span["start"] for span in body) if body else meta["char_start"], row["id"])
    for layer, kinds in (("context", {"paragraph", "table", "record"}),
                         ("citation", {"sentence", "table", "record"})):
        siblings = sorted((r for r in result if r["unit_type"] in kinds),
                          key=source_order)
        for i, row in enumerate(siblings):
            neighbors = {"prev_chunk_id": siblings[i - 1]["id"] if i else None,
                         "next_chunk_id": siblings[i + 1]["id"] if i + 1 < len(siblings) else None}
            meta = row["metadata"]
            meta.setdefault("chunk_index_by_layer", {})[layer] = i
            meta.setdefault("adjacent_chunk_ids", {})[layer] = neighbors
            if layer == "context" or row["unit_type"] == "sentence":
                meta.update(neighbors)
    return result
