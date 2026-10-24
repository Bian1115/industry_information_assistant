"""医学研究的公开视图与可携带交付包；不依赖数据库或模型。"""

from __future__ import annotations

import copy
import csv
import html
import io
import json
import re
from urllib.parse import urlsplit


PUBLIC_KEYS = (
    "protocol", "papers", "screening", "evidence", "search_log", "stats",
    "limitations", "review", "changes", "query_terms", "report_version",
    "human_reviews", "evidence_corrections", "status", "generated_at", "agent_steps",
)


def public_medical_bundle(bundle: dict) -> dict:
    """避免将全文或模型内部状态重复塞入事件流，保留可回查的证据片段。"""
    result = {key: copy.deepcopy(bundle[key]) for key in PUBLIC_KEYS if key in bundle}
    for paper in result.get("papers", []):
        for key in ("full_text", "content", "raw_response", "xml"):
            paper.pop(key, None)
        paper["sections"] = [
            {key: section[key] for key in ("section_id", "title", "page", "table") if key in section}
            for section in paper.get("sections", []) if isinstance(section, dict)
        ]
    return result


def safe_url(value: str) -> str:
    try:
        return value if urlsplit(value).scheme.lower() in {"http", "https"} else ""
    except (TypeError, ValueError):
        return ""


def medical_references(bundle: dict) -> list[dict]:
    included = {
        row.get("paper_id") for row in bundle.get("screening", [])
        if row.get("decision") != "exclude"
    }
    return [
        {
            "id": index + 1, "title": paper.get("title", "未命名文献"),
            "link": safe_url(paper.get("url", "")),
            "url": safe_url(paper.get("url", "")),
            "content": paper.get("abstract", "")[:500],
            "source": paper.get("source", "literature"),
            "paper_id": paper.get("paper_id"), "doi": paper.get("doi"),
            "pmid": paper.get("pmid"), "content_level": paper.get("content_level"),
        }
        for index, paper in enumerate(
            p for p in bundle.get("papers", [])
            if not bundle.get("screening") or p.get("paper_id") in included
        )
    ]


def _text(value) -> str:
    if value is None:
        return "未报告"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _csv_cell(value) -> str:
    text = _text(value)
    # 表格软件不应把论文文本解释为公式；保留其原始显示内容。
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text


def _bib(value) -> str:
    replacements = {"\\": "\\textbackslash{} ", "{": "\\{", "}": "\\}", "\n": " ", "\r": " "}
    return "".join(replacements.get(char, char) for char in _text(value))


def export_medical(bundle: dict, report: str, format: str) -> tuple[bytes, str, str]:
    """返回内容、媒体类型和中文文件名；报告正文不被当成 HTML 执行。"""
    scope = bundle.get("protocol") or {}
    disease = re.sub(r"[^\w\u4e00-\u9fff-]", "_", str(scope.get("disease") or "疾病"))[:60]
    cutoff = str(scope.get("end_date") or "未指定日期")
    stem = f"{disease}研究进展-{cutoff}"
    stale = bool((bundle.get("review") or {}).get("report_stale"))
    heading = "人工筛选或提取已变更，本文尚未重新生成。\n\n" if stale else ""
    body = heading + (report or "尚未生成报告。")
    if format == "markdown":
        return body.encode(), "text/markdown; charset=utf-8", stem + ".md"
    if format == "json":
        payload = {**public_medical_bundle(bundle), "final_report": body}
        return json.dumps(payload, ensure_ascii=False, indent=2).encode(), "application/json", stem + ".json"
    if format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        fields = {
            "evidence_id": "证据编号", "paper_id": "文献编号", "study_id": "研究编号",
            "study_design": "研究设计", "population": "人群", "sample_size": "样本量",
            "intervention": "干预或暴露", "comparator": "对照", "outcome": "结局",
            "effect": "结果或效应量", "confidence_interval": "置信区间", "follow_up": "随访",
            "safety": "安全性", "limitations": "局限", "quote": "原文片段",
            "section_id": "原文位置", "content_level": "证据内容层级",
            "verification_status": "原文核验状态", "semantic_status": "模型语义核验",
            "field_semantic_checks": "逐字段语义核验",
        }
        papers = {p.get("paper_id"): p for p in bundle.get("papers", [])}
        decisions = {r.get("paper_id"): r.get("decision") for r in bundle.get("screening", [])}
        writer.writerow([*fields.values(), "文献标题", "DOI", "PMID", "来源链接", "当前纳排决定", "检索截止日", "报告需更新"])
        for row in bundle.get("evidence", []):
            paper = papers.get(row.get("paper_id"), {})
            writer.writerow([_csv_cell(row.get(key)) for key in fields] + [
                _csv_cell(paper.get(key)) for key in ("title", "doi", "pmid", "url")
            ] + [_csv_cell(decisions.get(row.get("paper_id"))), cutoff, "是" if stale else "否"])
        return stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8", stem + "-证据表.csv"
    if format in {"bibtex", "ris"}:
        included = {r["paper_id"] for r in medical_references(bundle)}
        entries = []
        for index, paper in enumerate(bundle.get("papers", []), 1):
            if paper.get("paper_id") not in included:
                continue
            authors = paper.get("authors") or []
            if format == "bibtex":
                values = {
                    "title": paper.get("title"), "author": " and ".join(authors),
                    "journal": paper.get("journal"), "year": str(paper.get("publication_date") or "")[:4],
                    "doi": paper.get("doi"), "url": safe_url(paper.get("url", "")),
                }
                entries.append("@article{medical" + str(index) + ",\n" + ",\n".join(
                    "  " + key + " = {" + _bib(value) + "}" for key, value in values.items() if value
                ) + "\n}")
            else:
                values = [("TY", "JOUR"), ("ID", paper.get("paper_id")), ("TI", paper.get("title"))]
                values += [("AU", author) for author in authors]
                values += [("JO", paper.get("journal")), ("PY", paper.get("publication_date")),
                           ("DO", paper.get("doi")), ("UR", safe_url(paper.get("url", "")))]
                entries.append("\n".join(tag + "  - " + _text(value).replace("\n", " ") for tag, value in values if value) + "\nER  -")
        ext = "bib" if format == "bibtex" else "ris"
        return "\n\n".join(entries).encode(), "text/plain; charset=utf-8", stem + "." + ext
    if format == "html":
        document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{max-width:1000px;margin:40px auto;padding:0 24px;color:#162c37;font:16px/1.8 system-ui,sans-serif}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}}@media print{{body{{margin:0;font-size:11pt}}}}@page{{size:A4;margin:18mm}}</style>
<body><h1>{title}</h1><p>检索截止：{date}。可通过浏览器打印保存为 PDF。</p><pre>{body}</pre></body></html>""".format(title=html.escape(stem), date=html.escape(cutoff), body=html.escape(body))
        return document.encode(), "text/html; charset=utf-8", stem + ".html"
    if format == "docx":
        from docx import Document
        from docx.shared import Pt
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        document = Document()
        style = document.styles["Normal"]
        style.font.name = "Calibri"
        style.font.size = Pt(10)
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        if stale:
            document.add_paragraph(heading.strip(), style="Caption")
        table_lines = []

        def flush_table():
            if not table_lines:
                return
            rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in table_lines]
            rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in row)]
            if rows:
                table = document.add_table(rows=0, cols=max(map(len, rows)))
                table.style = "Table Grid"
                for i, row in enumerate(rows):
                    cells = table.add_row().cells
                    for j, value in enumerate(row):
                        cells[j].text = value
                    if i == 0:
                        props = table.rows[0]._tr.get_or_add_trPr()
                        props.append(OxmlElement("w:tblHeader"))
            table_lines.clear()

        for line in (report or "尚未生成报告。").splitlines():
            if line.strip().startswith("|") and line.strip().endswith("|"):
                table_lines.append(line)
                continue
            flush_table()
            match = re.match(r"^(#{1,4})\s+(.+)", line)
            if match:
                document.add_heading(match.group(2), level=len(match.group(1)))
            elif line.strip():
                document.add_paragraph(line)
        flush_table()
        stream = io.BytesIO()
        document.save(stream)
        return stream.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", stem + ".docx"
    raise ValueError("不支持的导出格式")
