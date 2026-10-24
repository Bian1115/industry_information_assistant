"""固定HTML离线复放当前读取器；确定性评分，不允许网络或模型调用。"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import gzip
import hashlib
import importlib.util
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import statistics
import sys
import tarfile
import time
import unicodedata
from zoneinfo import ZoneInfo

import markdown
import numpy as np
from apted import APTED, Config
from bs4 import BeautifulSoup
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment


REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUN = REPO / "backend/eval/data/网页读取链路公开基准评估-20261018-首轮"
TOKEN = re.compile(r"[\u3400-\u9fff]|[^\W_]+", re.UNICODE)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def tokens(text):
    return TOKEN.findall(norm(text))


def overlap(pred, ref):
    p, r = Counter(pred), Counter(ref)
    n = sum((p & r).values())
    pn, rn = sum(p.values()), sum(r.values())
    if not rn:
        return {"precision": 1.0 if not pn else 0.0, "recall": 1.0, "f1": 1.0 if not pn else 0.0}
    precision, recall = n / pn if pn else 0.0, n / rn
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def render(markup):
    html = markdown.markdown(markup, extensions=["tables", "fenced_code", "md_in_html"])
    soup = BeautifulSoup(html, "lxml")
    for node in soup(["script", "style"]):
        node.decompose()
    for image in soup.find_all("img"):
        image.replace_with(image.get("alt", ""))
    return soup


def visible(markup):
    return norm(render(markup).get_text(" ", strip=True))


def text_scores(pred, ref):
    pv, rv = visible(pred), visible(ref)
    pt, rt = tokens(pv), tokens(rv)
    score = overlap(pt, rt)
    score["reference_external_token_ratio"] = 1 - score["precision"] if pt else None
    score["ordered_5gram"] = overlap([tuple(pt[i:i + 5]) for i in range(max(0, len(pt) - 4))],
                                      [tuple(rt[i:i + 5]) for i in range(max(0, len(rt) - 4))]) if len(rt) >= 5 else None
    score["edit_similarity"] = float(Levenshtein.normalized_similarity(pv, rv))
    score["pred_tokens"], score["ref_tokens"] = len(pt), len(rt)
    return score


def official_words(pred, ref):
    pt, rt = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", ref.lower())
    if not rt:
        value = 1.0 if not pt else 0.0
        return {"precision": value, "recall": value, "f1": value}
    return overlap(pt, rt)


def snippets(pred, gt):
    result = {}
    for name in ["with", "without"]:
        seq = gt.get(name, []) or []
        found = [s for s in seq if s.lower() in pred.lower()]
        result[name] = {"n": len(seq), "found": len(found), "rate": len(found) / len(seq) if seq else None,
                        "hits": found, "misses": [s for s in seq if s not in found]}
    return result


def clean_annotations(text):
    stats = {}
    text, stats["cc_attributes"] = re.subn(
        r"\s+cc-[\w-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", "", text, flags=re.I)
    text, stats["annotation_style"] = re.subn(
        r"<style\b[^>]*\bid\s*=\s*['\"]cc-extraStyle['\"][^>]*>.*?</style\s*>", "", text, flags=re.I | re.S)
    counts = [0]
    def clean_class(match):
        values = match.group(2).split()
        kept = [v for v in values if v != "mark-selected" and not v.startswith("cc-")]
        counts[0] += len(values) - len(kept)
        return "class=" + match.group(1) + " ".join(kept) + match.group(1)
    text = re.sub(r"\bclass\s*=\s*(['\"])(.*?)\1", clean_class, text, flags=re.S | re.I)
    stats["annotation_classes"] = counts[0]
    return text, stats


@dataclass
class TableNode:
    tag: str
    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    children: list = field(default_factory=list)


class TableCosts(Config):
    def __init__(self, structure_only=False):
        self.structure_only = structure_only
    def rename(self, a, b):
        if (a.tag, a.rowspan, a.colspan) != (b.tag, b.rowspan, b.colspan):
            return 1.0
        return 0.0 if self.structure_only else float(Levenshtein.normalized_distance(a.text, b.text))


def table_record(table):
    root = TableNode("table")
    cells, grid = [], {}
    rows = [r for r in table.find_all("tr") if r.find_parent("table") is table]
    for i, row in enumerate(rows):
        row_node = TableNode("tr")
        root.children.append(row_node)
        col = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (i, col) in grid:
                col += 1
            text = norm(cell.get_text(" ", strip=True))
            def span(name):
                try:
                    return max(1, min(int(cell.get(name, 1)), 1000))
                except (ValueError, TypeError):
                    return 1
            rs, cs = span("rowspan"), span("colspan")
            row_node.children.append(TableNode("cell", text, rs, cs))
            if text:
                cells.append(text)
            if len(grid) + rs * cs > 200000:
                raise ValueError("表格展开超过20万格")
            for x in range(i, min(i + rs, len(rows))):
                for y in range(col, col + cs):
                    grid[x, y] = text
            col += cs
    h, w = len(rows), max((j + 1 for _, j in grid), default=0)
    horizontal = [(grid.get((i, j), ""), grid.get((i, j + 1), ""))
                  for i in range(h) for j in range(max(0, w - 1))
                  if grid.get((i, j)) or grid.get((i, j + 1))]
    vertical = [(grid.get((i, j), ""), grid.get((i + 1, j), ""))
                for i in range(max(0, h - 1)) for j in range(w)
                if grid.get((i, j)) or grid.get((i + 1, j))]
    return {"tree": root, "cells": cells, "tokens": tokens(" ".join(cells)),
            "rows": h, "columns": w, "horizontal": horizontal, "vertical": vertical,
            "nodes": 1 + len(rows) + sum(len(r.children) for r in root.children),
            "html": str(table)}


def extract_tables(markup):
    clean = re.sub(r"```.*?```|~~~.*?~~~", "", markup, flags=re.S)
    soup = render(clean)
    # 旧网页常以嵌套table布局整篇文章；取叶子表，避免把布局容器当作数据表。
    result = [table_record(t) for t in soup.find_all("table") if t.find("table") is None and t.find("tr")]
    # Trafilatura原生Markdown会输出没有GFM表头分隔线的管道表；它仍有明确行列。
    runs, current = [], []
    for line in clean.splitlines() + [""]:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            current.append(line.strip())
        elif current:
            runs.append(current); current = []
    for run in runs:
        if len(run) < 2 or any(re.fullmatch(r"[\s|:\-]+", line) for line in run):
            continue
        table_soup = BeautifulSoup("<table></table>", "lxml")
        table = table_soup.find("table")
        for line in run:
            row = table_soup.new_tag("tr"); table.append(row)
            for value in re.split(r"(?<!\\)\|", line[1:-1]):
                cell = table_soup.new_tag("td")
                cell.string = render(value.replace("\\|", "|")).get_text(" ", strip=True)
                row.append(cell)
        result.append(table_record(table))
    return result


def alarm_handler(*_):
    raise TimeoutError("单表树编辑距离超过3秒")


def table_teds(a, b, structure=False):
    if max(a["nodes"], b["nodes"]) > 250:
        return None, "超过250节点"
    try:
        signal.setitimer(signal.ITIMER_REAL, 3.0)
        dist = APTED(a["tree"], b["tree"], TableCosts(structure)).compute_edit_distance()
        return max(0.0, 1 - dist / max(a["nodes"], b["nodes"])), None
    except TimeoutError as exc:
        return None, str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def table_scores(pred_text, ref_tables, include_grid=True):
    preds = extract_tables(pred_text)
    if not ref_tables:
        return {"reference_tables": 0, "predicted_tables": len(preds), "eligible": False}
    all_ref_cells = [x for t in ref_tables for x in t["cells"]]
    all_pred_cells = [x for t in preds for x in t["cells"]]
    result = {"reference_tables": len(ref_tables), "predicted_tables": len(preds), "eligible": True,
              "cells": overlap(all_pred_cells, all_ref_cells), "cells_reference_n": len(all_ref_cells),
              "table_text": overlap([x for t in preds for x in t["tokens"]], [x for t in ref_tables for x in t["tokens"]]),
              "table_text_in_whole_output": overlap(tokens(visible(pred_text)), [x for t in ref_tables for x in t["tokens"]])}
    pairs = []
    if preds:
        matrix = np.array([[overlap(p["tokens"], r["tokens"])["f1"] for r in ref_tables] for p in preds])
        pi, ri = linear_sum_assignment(-matrix)
        pairs = [(int(p), int(r), float(matrix[p, r])) for p, r in zip(pi, ri) if matrix[p, r] >= 0.1]
    horizontal_p, horizontal_r, vertical_p, vertical_r = [], [], [], []
    teds_values, struct_values, skipped, matching = [], [], [], []
    matched_p, matched_r = {x[0] for x in pairs}, {x[1] for x in pairs}
    for p, r, similarity in pairs:
        a, b = preds[p], ref_tables[r]
        # 加入表格匹配身份，防止不同表格的相同单元格关系互相抵消。
        horizontal_p.extend((r, *x) for x in a["horizontal"])
        horizontal_r.extend((r, *x) for x in b["horizontal"])
        vertical_p.extend((r, *x) for x in a["vertical"])
        vertical_r.extend((r, *x) for x in b["vertical"])
        tv, why = table_teds(a, b)
        sv, swhy = table_teds(a, b, True)
        if tv is None or sv is None:
            skipped.append({"prediction": p, "reference": r, "reason": why or swhy})
        else:
            teds_values.append(tv); struct_values.append(sv)
        matching.append({"prediction": p, "reference": r, "token_f1": similarity,
                         "reference_shape": [b["rows"], b["columns"]], "prediction_shape": [a["rows"], a["columns"]],
                         "reference_nodes": b["nodes"], "prediction_nodes": a["nodes"], "teds_like": tv, "structure_teds_like": sv})
    for r, table in enumerate(ref_tables):
        if r not in matched_r:
            horizontal_r.extend((r, *x) for x in table["horizontal"])
            vertical_r.extend((r, *x) for x in table["vertical"])
            teds_values.append(0.0); struct_values.append(0.0)
    for p, table in enumerate(preds):
        if p not in matched_p:
            horizontal_p.extend(("extra" + str(p), *x) for x in table["horizontal"])
            vertical_p.extend(("extra" + str(p), *x) for x in table["vertical"])
    result.update({"matched_tables": len(pairs), "table_recall": len(pairs) / len(ref_tables),
                   "horizontal_relations": overlap(horizontal_p, horizontal_r), "vertical_relations": overlap(vertical_p, vertical_r),
                   "horizontal_reference_n": len(horizontal_r), "vertical_reference_n": len(vertical_r),
                   "teds_like": statistics.mean(teds_values) if teds_values and not skipped else None,
                   "structure_teds_like": statistics.mean(struct_values) if struct_values and not skipped else None,
                   "teds_pairs_evaluated": len(teds_values), "teds_skipped": skipped, "matches": matching})
    if include_grid:
        grid_ref = [t for t in ref_tables if t["rows"] >= 2 and t["columns"] >= 2 and len(t["cells"]) >= 4]
        result["grid_tables"] = table_scores(pred_text, grid_ref, False)
    return result


def validate_metrics():
    assert overlap(tokens("a b c"), tokens("a b c"))["f1"] == 1
    assert overlap(tokens("a b"), tokens("a b c"))["recall"] < 1
    assert overlap(tokens("a b c noise"), tokens("a b c"))["precision"] < 1
    a = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    b = "<table><tr><td>B</td><td>A</td></tr><tr><td>D</td><td>C</td></tr></table>"
    identity = table_scores(a, extract_tables(a))
    swapped = table_scores(b, extract_tables(a))
    missing = table_scores("A B C D", extract_tables(a))
    assert identity["teds_like"] == 1.0
    assert swapped["cells"]["recall"] == 1.0 and swapped["horizontal_relations"]["f1"] < 1
    assert swapped["teds_like"] < 1
    assert missing["table_recall"] == 0.0 and missing["cells"]["recall"] == 0.0
    headerless = table_scores("| A | B |\n| C | D |", extract_tables(a))
    assert headerless["cells"]["recall"] == 1.0 and headerless["teds_like"] == 1.0
    assert table_scores("no table", [])["eligible"] is False
    sample = '<html><style id="cc-extraStyle">x</style><p cc-select="true" class="body mark-selected cc-unloaded">hi</p></html>'
    cleaned, counts = clean_annotations(sample)
    assert "cc-select" not in cleaned and "mark-selected" not in cleaned and "hi" in cleaned
    return {"passed": True, "cases": ["文本同一/漏失/噪声", "表格同一", "相同单元格换列", "表格变为散文", "无表格不计满分", "移除标注泄漏"],
            "identity_table": identity, "swapped_columns": swapped, "flattened_table": missing, "annotation_cleaning": counts}


def load_samples(root):
    rows = []
    raw = root / "原始数据"
    for i, line in enumerate((raw / "WebMainBench校准545页.jsonl").open(encoding="utf-8"), 1):
        obj = json.loads(line)
        rows.append({"id": "WebMainBench-" + obj["track_id"], "dataset": "WebMainBench545", "source_row": i,
                     "url": obj.get("url", ""), "html": obj["html"], "reference": obj["groundtruth_content"],
                     "meta": obj.get("meta") or {}, "reference_format": "markdown"})
    assert len(rows) == 545
    with tarfile.open(raw / "WCXB原始仓库.tar.gz") as tar:
        base = tar.getnames()[0]
        metadata = json.load(tar.extractfile(base + "/metadata.json"))
        groups = defaultdict(list)
        for file_id, item in metadata["files"].items():
            if item["split"] == "dev":
                groups[item["page_type"]].append(file_id)
        chosen, allocation = [], {}
        for i, name in enumerate(sorted(groups)):
            n = 29 if i < 4 else 28
            rng = random.Random(20261018 + i)
            candidates = sorted(groups[name]); rng.shuffle(candidates)
            selected = sorted(candidates[:n]); chosen.extend(selected)
            allocation[name] = {"population": len(candidates), "selected": len(selected), "seed": 20261018 + i, "ids": selected}
        assert len(chosen) == 200
        # gzip tar按文件往返随机读取会重复解压整个压缩包；另开顺序流只取抽中项。
        wanted = {f"{base}/dev/{kind}/{file_id}{suffix}"
                  for file_id in chosen for kind, suffix in [("ground-truth", ".json"), ("html", ".html.gz")]}
        blobs = {}
        with tarfile.open(raw / "WCXB原始仓库.tar.gz", mode="r|gz") as stream:
            for member in stream:
                if member.name in wanted:
                    blobs[member.name] = stream.extractfile(member).read()
        for file_id in sorted(chosen):
            gt = json.loads(blobs[f"{base}/dev/ground-truth/{file_id}.json"])
            html = gzip.decompress(blobs[f"{base}/dev/html/{file_id}.html.gz"]).decode("utf-8")
            rows.append({"id": "WCXB-" + file_id, "dataset": "WCXB分层200", "url": gt["url"], "html": html,
                         "reference": gt["ground_truth"]["main_content"], "snippets": gt["ground_truth"],
                         "meta": metadata["files"][file_id], "reference_format": "plaintext"})
    dump(root / "分层抽样配置.json", {"method": "7类近似等额，固定种子，先抽样后评分", "split": "dev", "n": 200,
                                       "allocation": allocation, "note": "均衡样本均值不能直接当作原dev分布均值，另报按类型总体权重校正值"})
    return rows


def freeze_config(root):
    public = json.loads((root / "可公开读取参数.json").read_text())
    for key in ["WEB_READER_MIN_CHARS", "WEB_READER_MAX_CHARS", "WEB_READER_MAX_BYTES", "WEB_READER_ENABLED"]:
        if key in public:
            os.environ[key] = public[key]["value"]
    os.environ["WEB_READER_CACHE_DIR"] = str(root / "隔离缓存")
    os.environ["CRAWL4AI_ENDPOINT"] = ""
    os.environ["WEB_READER_OCR_ENDPOINT"] = ""
    sys.path.insert(0, str(root / "冻结代码"))
    from service.web_reader_service import WebReaderService, HTML_EXTRACTION_VERSION
    from service.web_document_structure import limit_body_preserving_tables
    class ReplayReader(WebReaderService):
        def set_sample(self, html, url):
            self.sample_html = html
            self.sample_url = url
            self.full_extracted = None
            self.browser_attempts = 0
            self.before_validation = None
        async def _download_with_retry(self, url):
            return url, "text/html; charset=utf-8", self.sample_html.encode("utf-8"), 1
        async def _read_with_crawl4ai(self, url, title):
            self.browser_attempts += 1
            return None
        def _save_html_snapshot(self, body, url, content_type, kind="http_response"):
            return {"kind": kind, "sha256": sha(body), "bytes": len(body), "path": "已保存到冻结输入"}
        def _extract_html(self, html, url):
            if self.full_extracted is None:
                budget = self.max_chars
                try:
                    self.max_chars = 100000000
                    self.full_extracted = super()._extract_html(html, url)
                finally:
                    self.max_chars = budget
            result = dict(self.full_extracted)
            result["content"] = limit_body_preserving_tables(result["content"], self.max_chars)
            return result
        def _validate_body(self, result, page_title=""):
            self.before_validation = result.to_dict()
            return super()._validate_body(result, page_title)
    reader = ReplayReader()
    config = {"date": "2026-10-18", "timezone": "Asia/Singapore", "html_extraction_version": HTML_EXTRACTION_VERSION,
              "min_chars": reader.min_chars, "max_chars": reader.max_chars,
              "public_env": public, "python": sys.version, "executable": sys.executable,
              "packages": {name: version(name) for name in ["trafilatura", "beautifulsoup4", "lxml", "httpx", "Markdown", "rapidfuzz", "apted", "scipy", "numpy"]},
              "modes": ["unlimited_pre_gate", "production_pre_gate", "production_cleaned", "accepted_output"],
              "network": "socket connect blocked; frozen download substituted; browser fallback returns unavailable", "paid_api_calls": 0,
              "metrics_protocol": "项目确定性诊断v1；WCXB另报官方word F1/snippet；不复现WebMainBench官方综合榜单",
              "table_match": "Hungarian最大化词元F1，匹配阈值0.1", "teds_limits": {"max_nodes": 250, "seconds_per_pair_per_metric": 3}}
    dump(root / "评估配置.json", config)
    return reader, WebReaderService, config


def install_network_guard():
    attempts = []
    def denied(*args, **kwargs):
        attempts.append(str(args[1:])[:200])
        raise RuntimeError("离线评估禁止任何网络连接")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    return attempts


async def execute(root, limit=None):
    signal.signal(signal.SIGALRM, alarm_handler)
    logging.basicConfig(level=logging.ERROR)
    validation = validate_metrics()
    rows = load_samples(root)
    dump(root / "指标机制验证.json", validation)
    reader, original_class, config = freeze_config(root)
    network_attempts = install_network_guard()
    manifest, results = [], []
    out_dir = root / "逐页结果"
    inputs = root / "冻结输入"
    out_dir.mkdir(exist_ok=True); inputs.mkdir(exist_ok=True)
    start = time.time()
    if limit:
        rows = rows[:limit]
    progress = root / ("校验运行进度.jsonl" if limit else "执行进度.jsonl")
    with progress.open("a", encoding="utf-8") as stream:
        for index, sample in enumerate(rows, 1):
            page_start = time.perf_counter()
            sid = sample["id"]
            folder = out_dir / sid
            existing = folder / "页面结果.json"
            if existing.exists():
                result = json.loads(existing.read_text())
                results.append(result); manifest.append(result["input"])
                continue
            folder.mkdir(exist_ok=True)
            html, annotation_stats = clean_annotations(sample["html"])
            input_bytes = html.encode("utf-8")
            input_path = inputs / ("输入-" + sid + ".html.gz")
            with gzip.open(input_path, "wb") as f:
                f.write(input_bytes)
            (folder / "参考正文.txt").write_text(sample["reference"], encoding="utf-8")
            reader.set_sample(html, sample["url"])
            prod = await reader._read_uncached(sample["url"], "")
            extraction_seconds = time.perf_counter() - page_start
            before_gate = reader.before_validation or prod.to_dict()
            raw_full = (reader.full_extracted or {}).get("content", "")
            # 放宽字符预算的诊断只复用相同抽取结果，不再调用抽取器。
            texts = {"unlimited_pre_gate": raw_full, "production_pre_gate": before_gate.get("content", ""),
                     "production_cleaned": prod.content, "accepted_output": prod.content if prod.success else ""}
            filenames = {"unlimited_pre_gate": "放宽上限正文.txt", "production_pre_gate": "生产门禁前正文.txt",
                         "production_cleaned": "生产清理后正文.txt", "accepted_output": "最终可接纳正文.txt"}
            for mode, text in texts.items():
                (folder / filenames[mode]).write_text(text, encoding="utf-8")
            ref = sample["reference"]
            reference_for_scoring = ref if sample["reference_format"] == "markdown" else __import__("html").escape(ref)
            ref_tables, ref_error = [], None
            if sample["dataset"] == "WebMainBench545":
                try:
                    ref_tables = extract_tables(ref)
                except Exception as exc:
                    ref_error = str(exc)
            scores, memo = {}, {}
            for mode, pred in texts.items():
                key = sha(pred.encode())
                if key in memo:
                    scores[mode] = deepcopy(memo[key]); continue
                entry = {"text": text_scores(pred, reference_for_scoring)}
                if sample["dataset"] == "WCXB分层200":
                    entry["official_wcxb_words"] = official_words(pred, ref)
                    entry["snippets"] = snippets(pred, sample["snippets"])
                else:
                    try:
                        entry["table"] = table_scores(pred, ref_tables) if ref_error is None else {"error": ref_error}
                    except Exception as exc:
                        entry["table"] = {"error": str(exc)}
                scores[mode] = entry; memo[key] = deepcopy(entry)
            provenance = {"id": sid, "dataset": sample["dataset"], "source_row": sample.get("source_row"),
                          "source_html_sha256": sha(sample["html"].encode()), "clean_html_sha256": sha(input_bytes),
                          "reference_sha256": sha(ref.encode()), "clean_html_path": str(input_path.relative_to(root)),
                          "annotation_removed": annotation_stats}
            result = {"id": sid, "dataset": sample["dataset"], "url": sample["url"], "meta": sample["meta"], "input": provenance,
                      "extraction_diagnostics": (reader.full_extracted or {}).get("content_selection", {}),
                      "reference_chars": len(ref), "reference_tables_detected": len(ref_tables), "reference_table_error": ref_error,
                      "reference_snippet_integrity": snippets(ref, sample["snippets"]) if "snippets" in sample else None,
                      "production_result": prod.to_dict(), "chars": {m: len(t) for m, t in texts.items()},
                      "truncated": len(raw_full) > len(texts["production_pre_gate"]),
                      "browser_fallback_requested_but_not_executed": reader.browser_attempts,
                      "extract_and_gate_seconds": extraction_seconds, "total_scoring_seconds": time.perf_counter() - page_start,
                      "scores": scores}
            # 验证复用完整抽取后再截断与原生产函数直接输出一致，保留三个例子。
            if index <= 3:
                direct = original_class._extract_html(reader, html, sample["url"])
                cached = reader._extract_html(html, sample["url"])
                equal = direct == cached
                result["production_extraction_equivalence"] = equal
                if not equal:
                    raise AssertionError("离线适配与生产提取函数不等价")
            dump(existing, result)
            manifest.append(provenance); results.append(result)
            stream.write(json.dumps({"index": index, "id": sid, "accepted": prod.success, "error": prod.failure_category,
                                     "seconds": result["total_scoring_seconds"]}, ensure_ascii=False) + "\n"); stream.flush()
            if index % 25 == 0 or index == len(rows):
                print(json.dumps({"completed": index, "total": len(rows), "elapsed_seconds": round(time.time() - start, 1),
                                  "latest": sid}, ensure_ascii=False), flush=True)
    dump(root / ("校验样本清单.json" if limit else "冻结样本清单.json"), manifest)
    dump(root / ("校验网络审计.json" if limit else "网络与费用审计.json"),
         {"network_guard": "socket connect/create_connection blocked", "network_connection_attempts": network_attempts,
          "paid_api_calls": 0, "browser_executions": 0, "ocr_executions": 0, "llm_scoring_calls": 0,
          "downloads": "运行前仅公开数据与开源Python依赖；依赖及下载流量未计价", "completed_pages": len(results)})
    if not limit:
        dump(root / "逐页指标汇总.json", results)
    print("完成", len(results), "页；网络连接尝试", len(network_attempts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(execute(args.run_dir.resolve(), args.limit))
