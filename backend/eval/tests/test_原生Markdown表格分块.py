"""原生 Markdown 到表格证据的边界、表头上下文及原文位置验证。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
from service.summary_evidence_units import split_evidence_units, structured_windows, validate_spans
from service.session_evidence_chunks import chunk_document
from service.web_document import assess_body, index_document
from service.web_document_structure import limit_body_preserving_tables


CAPTION_HEADER = "| 表1 销量 |  |\n|---|---|\n| 年份 | 销量（万台） |\n|---|---|"
HEADER = "| 年份 | 数值 |\n|---|---|"


class Counter:
    def count_text(self, text):
        return len(text)


def units(text, budget=10000):
    return split_evidence_units(text, 0, len(text), lambda value: len(value) <= budget)


def chunks(text, budget=10000):
    text, quality = assess_body(text)
    source = {"id": "s", "url": "https://example.org/table", "content": text,
              "body_quality": quality, **index_document(text, "https://example.org/table")}
    return [row for row in chunk_document(source, user_id="u", session_id="s", topic_id="t",
                                          counter=Counter(), chunk_tokens=budget)
            if row["unit_type"] == "table"]


class NativeMarkdownTableTests(unittest.TestCase):
    def test_caption_and_repeated_separators_are_one_table(self):
        text = CAPTION_HEADER + "\n| 2025 | 100 |\n| 2026 | 90 |\n\n注：仅统计中国。"
        got = units(text)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["header_status"], "explicit")
        self.assertEqual(len(got[0]["body_rows"]), 2)
        self.assertEqual(got[0]["fixed_prefix"][0]["role"], "caption")
        validate_spans(text, got[0])
        row = chunks(text)[0]
        self.assertEqual(row["metadata"]["row_range"], [1, 2])
        self.assertIn("注：仅统计中国", row["content"])

    def test_long_native_table_repeats_all_header_levels_and_notes(self):
        text = CAPTION_HEADER + "\n" + "\n".join(f"| {2000+i} | {100+i} |" for i in range(30)) + "\n\n注：单位为万台。"
        got = units(text, 125)
        self.assertGreater(len(got), 1)
        seen = []
        for item in got:
            self.assertTrue(item["text"].startswith(CAPTION_HEADER))
            self.assertIn("单位为万台", item["text"])
            self.assertLessEqual(len(item["text"]), 125)
            validate_spans(text, item)
            seen.extend(span["start"] for span in item["body_rows"])
        self.assertEqual(len(seen), 30)
        self.assertEqual(len(set(seen)), 30)
        rows = chunks(text, 125)
        self.assertEqual(rows[0]["metadata"]["row_range"][0], 1)
        self.assertEqual(rows[-1]["metadata"]["row_range"][-1], 30)

    def test_headless_table_keeps_first_data_row_and_invents_no_header(self):
        text = "| 2025 | 100 |\n| 2026 | 90 |"
        item = units(text)[0]
        self.assertEqual(item["header_status"], "absent")
        self.assertEqual(item["fixed_prefix"], [])
        self.assertEqual(len(item["body_rows"]), 2)
        self.assertEqual(item["text"], text)
        self.assertNotIn("列1", chunks(text)[0]["content"])

    def test_headless_tables_have_distinct_ids_and_separate_row_ranges(self):
        text = "| 2025 | 100 |\n| 2026 | 90 |\n\n| 2030 | 50 |\n| 2031 | 60 |"
        rows = chunks(text)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["metadata"]["table_id"], rows[1]["metadata"]["table_id"])
        self.assertEqual([row["metadata"]["row_range"] for row in rows], [[1, 2], [1, 2]])

    def test_multilevel_headers_and_blank_spans_stay_in_one_context(self):
        text = "| 年份 | 销量 |  |\n|---|---|---|\n|  | 中国 | 美国 |\n| 2025 | 100 | 80 |\n|  | 90 | 70 |"
        item = units(text)[0]
        self.assertTrue(item["table_requires_whole"])
        self.assertEqual(item["header_status"], "ambiguous")
        self.assertEqual(item["text"], text)
        self.assertEqual(chunks(text)[0]["metadata"]["row_range"], [])
        self.assertNotIn("2025 | 90", item["text"])
        with self.assertRaisesRegex(ValueError, "整表超出预算"):
            structured_windows(item, text, lambda value: len(value) <= 65)

    def test_blank_data_cell_does_not_get_filled_or_detached_from_previous_row(self):
        text = HEADER + "\n| 2025 | 100 |\n|  | 90 |"
        self.assertTrue(units(text)[0]["table_requires_whole"])
        self.assertEqual(units(text)[0]["text"], text)
        with self.assertRaisesRegex(ValueError, "整表超出预算"):
            units(text, len(HEADER) + 18)

    def test_text_only_leading_row_is_not_guessed_to_be_data_or_header(self):
        text = "| 指标 | 区域 |\n|---|---|\n| 国内 | 海外 |\n| A | B |"
        self.assertEqual(units(text)[0]["header_status"], "ambiguous")
        self.assertEqual(len(units(text)), 1)
        self.assertEqual(chunks(text)[0]["metadata"]["row_range"], [])

    def test_table_example_inside_code_fence_is_not_a_data_table(self):
        text = "```markdown\n" + HEADER + "\n| 2025 | 100 |\n```"
        self.assertTrue(all(item["unit_type"] == "sentence" for item in units(text)))

    def test_secondary_pack_keeps_native_caption_and_exact_original_spans(self):
        text = CAPTION_HEADER + "\n" + "\n".join(f"| {2000+i} | {100+i} |" for i in range(12))
        original = units(text)[0]
        narrowed = structured_windows(original, text, lambda value: len(value) <= 110)
        self.assertGreater(len(narrowed), 1)
        self.assertEqual(sum(len(item["body_rows"]) for item in narrowed), 12)
        for item in narrowed:
            self.assertTrue(item["text"].startswith(CAPTION_HEADER))
            validate_spans(text, item)

    def test_output_budget_keeps_headless_table_and_note_after_blank_line(self):
        text = "前文。\n\n| 2025 | 100 |\n| 2026 | 90 |\n\n注：单位为万元。\n\n后文。"
        limited = limit_body_preserving_tables(text, text.index("2025") + 2)
        self.assertIn("| 2026 | 90 |", limited)
        self.assertTrue(limited.endswith("注：单位为万元。"))
        self.assertNotIn("后文", limited)

    def test_output_budget_does_not_treat_code_example_as_real_table(self):
        text = "```markdown\n" + HEADER + "\n| 2025 | 100 |\n```"
        budget = text.index("2025") + 2
        self.assertEqual(limit_body_preserving_tables(text, budget), text[:budget].rstrip())


if __name__ == "__main__":
    unittest.main()
