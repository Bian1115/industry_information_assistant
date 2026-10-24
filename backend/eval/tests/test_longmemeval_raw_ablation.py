"""LongMemEval-S Raw Session 对照实验纯离线测试。"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_longmemeval_raw_ablation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_longmemeval_raw_ablation", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_row():
    return {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "What happened?",
        "answer": "LEAK_SENTINEL",
        "question_date": "2024/01/03 (Wed) 10:00",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": [
            "2024/01/01 (Mon) 10:00",
            "2024/01/02 (Tue) 10:00",
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
            ],
            [{"role": "user", "content": "second question"}],
        ],
        "answer_session_ids": ["s2", "s1"],
    }


class SelectionAndRenderingTests(unittest.TestCase):
    def test_oracle_preserves_annotated_order_and_raw_turns(self):
        sessions = MODULE.select_oracle_sessions(make_row())
        self.assertEqual([item.source_session_id for item in sessions], ["s2", "s1"])
        rendered = MODULE.render_raw_context(sessions, max_context_chars=10_000)
        self.assertEqual(rendered.session_count, 2)
        self.assertEqual(rendered.turn_count, 3)
        self.assertIn("date=2024/01/02 (Tue) 10:00", rendered.text)
        self.assertLess(rendered.text.index("second question"), rendered.text.index("first question"))

    def test_retrieved_selection_preserves_frozen_first_hit_order(self):
        row = make_row()
        sessions = MODULE.select_retrieved_sessions(
            row,
            {"question_id": "q1", "retrieved_session_ids": ["s1", "s2", "s1"]},
        )
        self.assertEqual([item.source_session_id for item in sessions], ["s1", "s2"])

    def test_duplicate_source_id_is_rejected_when_selected(self):
        row = make_row()
        row["haystack_session_ids"] = ["duplicate", "duplicate"]
        with self.assertRaisesRegex(ValueError, "无法仅凭 source_session_id"):
            MODULE.select_sessions_by_source_ids(row, ["duplicate"])

    def test_context_limit_fails_without_truncation(self):
        sessions = MODULE.select_oracle_sessions(make_row())
        with self.assertRaisesRegex(ValueError, "拒绝静默截断"):
            MODULE.render_raw_context(sessions, max_context_chars=10)

    def test_answer_prompt_does_not_read_reference_answer_field(self):
        row = make_row()
        rendered = MODULE.render_raw_context(
            MODULE.select_oracle_sessions(row), max_context_chars=10_000
        )
        prompt = MODULE.build_raw_answer_prompt(row, rendered)
        self.assertNotIn("LEAK_SENTINEL", prompt)
        self.assertIn(row["question"], prompt)

    def test_renderer_is_deterministic(self):
        sessions = MODULE.select_oracle_sessions(make_row())
        first = MODULE.render_raw_context(sessions, max_context_chars=10_000)
        second = MODULE.render_raw_context(sessions, max_context_chars=10_000)
        self.assertEqual(first, second)


class SnapshotAndComparisonTests(unittest.TestCase):
    def test_atomic_jsonl_replaces_incomplete_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "snapshot.jsonl"
            path.write_text('{"old": true}\n', encoding="utf-8")
            MODULE.write_jsonl_atomic(path, [{"question_id": "q1"}])
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(
                rows,
                [{"question_id": "q1"}],
            )

    def test_snapshot_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "questions.jsonl"
            path.write_text(json.dumps({"question_id": "q1"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "拒绝混用基线"):
                MODULE.validate_retrieval_snapshot(path, "not-the-real-sha")

    def test_mcnemar_counts_paired_flips(self):
        result = MODULE.mcnemar_exact(
            [False, False, True, True],
            [True, False, False, True],
        )
        self.assertEqual(result["left_only_correct"], 1)
        self.assertEqual(result["right_only_correct"], 1)
        self.assertEqual(result["discordant"], 2)
        self.assertEqual(result["p_value"], 1.0)

    def test_paired_bootstrap_is_deterministic(self):
        first = MODULE.paired_bootstrap_ci(
            [False, False, True], [True, True, True], samples=100
        )
        second = MODULE.paired_bootstrap_ci(
            [False, False, True], [True, True, True], samples=100
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
