"""静态网页到证据chunk的结构保真回归，不访问网络。"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
from service.web_reader_service import WebReaderService
from service.web_document import assess_body, index_document
from service.web_document_structure import limit_body_preserving_tables
from service.session_evidence_chunks import chunk_document


class Counter:
    def count_text(self, text): return len(text)


class HTMLTableTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        environment = patch.dict("os.environ", {"WEB_READER_CACHE_DIR": folder.name})
        environment.start(); self.addCleanup(environment.stop)
        self.reader = WebReaderService()

    def extract(self, body):
        return self.reader._extract_html("<html><head><title>报告</title></head><body><nav>菜单</nav><article>" + body + "</article></body></html>", "https://example.com/report")["content"]

    def chunks(self, text, chunk_tokens=1024):
        text, quality = assess_body(text)
        source = {"id": "s1", "url": "https://example.com/report", "content": text,
                  "body_quality": quality, **index_document(text, "https://example.com/report")}
        return chunk_document(source, user_id="u", session_id="s", topic_id="t", counter=Counter(), chunk_tokens=chunk_tokens)

    def test_html_table_header_notes_and_heading_reach_chunks(self):
        text = self.extract('<h2>市场</h2><p>统计范围为中国。年度销售数据如下。</p><table><caption>表1 销量</caption><tr><th>年份</th><th>销量（万台）</th></tr><tr><td>2025</td><td>100</td></tr><tr><td>2026</td><td>90</td></tr></table><p>注：只统计中国。</p><p>这是后文。后文另作段落。</p>')
        self.assertNotIn("菜单", text)
        rows = self.chunks(text)
        tables = [r for r in rows if r["unit_type"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["metadata"]["heading_path"], ["市场"])
        self.assertEqual(tables[0]["metadata"]["row_range"], [1, 2])
        self.assertIn("销量（万台）", tables[0]["content"])
        self.assertIn("注：只统计中国", tables[0]["content"])
        self.assertFalse(any("2025 | 100" in r["content"] for r in rows if r["unit_type"] != "table"))

    def test_merged_headers_and_rowspans_preserve_column_mapping(self):
        text = self.extract('<table><thead><tr><th rowspan="2">年份</th><th colspan="2">销量</th></tr><tr><th>中国</th><th>美国</th></tr></thead><tbody><tr><td rowspan="2">2025</td><td>100</td><td>80</td></tr><tr><td>90</td><td>70</td></tr></tbody></table>')
        self.assertIn("| 年份 | 销量 |  |", text)
        self.assertIn("|  | 中国 | 美国 |", text)
        self.assertIn("|  | 90 | 70 |", text)
        tables = [row for row in self.chunks(text) if row["unit_type"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["metadata"]["header_status"], "ambiguous")
        self.assertEqual(tables[0]["metadata"]["row_range"], [])
        self.assertTrue(all(value in tables[0]["content"] for value in ("年份", "销量", "中国", "美国", "2025", "100", "90", "70")))
        with self.assertRaisesRegex(ValueError, "整表超出预算"):
            self.chunks(text, chunk_tokens=65)

    def test_missing_header_does_not_discard_first_data_row(self):
        text = self.extract('<table><tr><td>2025</td><td>100</td></tr><tr><td>2026</td><td>90</td></tr></table>')
        self.assertNotIn("列1", text)
        self.assertIn("| 2025 | 100 |", text)
        table = next(row for row in self.chunks(text) if row["unit_type"] == "table")
        self.assertEqual(table["metadata"]["header_status"], "absent")
        self.assertEqual(table["metadata"]["header_spans"], [])
        self.assertEqual(table["metadata"]["row_range"], [1, 2])

    def test_pipe_in_cell_remains_one_column(self):
        text = self.extract('<table><tr><th>项目</th><th>数值</th></tr><tr><td>A|B</td><td>3.14</td></tr></table>')
        self.assertIn("A\\|B", text)
        self.assertEqual(len(self.chunks(text)[0]["metadata"]["row_spans"]), 1)

    def test_reader_limit_does_not_cut_table_or_note(self):
        text = "前文。\n\n| 年份 | 数值 |\n| --- | --- |\n| 2025 | 100 |\n| 2026 | 90 |\n注：单位为万元。\n\n后文很长。"
        limited = limit_body_preserving_tables(text, text.index("2025") + 2)
        self.assertIn("2026 | 90", limited)
        self.assertTrue(limited.endswith("注：单位为万元。"))
        self.assertEqual(self.chunks(limited)[-1]["unit_type"], "paragraph")
        self.assertFalse("后文" in limited)

    def test_missing_extractor_does_not_bypass_body_selection(self):
        with patch("service.web_reader_service.trafilatura", None):
            with self.assertRaisesRegex(RuntimeError, "Trafilatura"):
                self.extract('<h1>章节</h1><table><tr><th>年份</th><th>数值</th></tr><tr><td>2025</td><td>100</td></tr></table>')
