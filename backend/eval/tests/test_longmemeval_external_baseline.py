"""Tests for the external LongMemEval baseline adapter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_longmemeval_external_baseline import (  # noqa: E402
    ExternalEvidence,
    ExternalMemoryAdapter,
    retrieved_session_ids,
)


class FakeProvider:
    provider = "fake"

    def __init__(self):
        self.calls = []

    def request(self, operation, **payload):
        self.calls.append((operation, payload))
        if operation == "add":
            return {"stored_items": 2}
        if operation == "finalize_scope":
            return {"finalized": True}
        raise AssertionError(operation)


class ExternalMemoryAdapterTest(unittest.TestCase):
    def test_ingestion_checkpoint_prevents_duplicate_adds(self):
        row = {
            "question_id": "q1",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2023/01/01 10:00", "2023/01/02 10:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "one"}],
                [{"role": "assistant", "content": "two"}],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            first_provider = FakeProvider()
            first = ExternalMemoryAdapter(
                first_provider, checkpoint, retries=1, verbose_sessions=False
            ).ensure_index(row)
            self.assertEqual(first.status, "created")
            self.assertEqual(first.ready_sessions, 2)
            self.assertEqual(first.stored_items, 4)
            self.assertEqual(
                [call[0] for call in first_provider.calls],
                ["add", "add", "finalize_scope"],
            )

            resumed_provider = FakeProvider()
            resumed = ExternalMemoryAdapter(
                resumed_provider, checkpoint, retries=1, verbose_sessions=False
            ).ensure_index(row)
            self.assertEqual(resumed.status, "reused")
            self.assertEqual(
                [call[0] for call in resumed_provider.calls],
                ["finalize_scope"],
            )

    def test_retrieved_session_ids_preserve_rank_and_deduplicate(self):
        evidence = [
            ExternalEvidence("m1", "f1", "edge", "fact 1", 1.0, ["s2", "s1"], {}, 1),
            ExternalEvidence("m2", "f2", "edge", "fact 2", 0.5, ["s1", "s3"], {}, 2),
        ]
        self.assertEqual(retrieved_session_ids(evidence), ["s2", "s1", "s3"])

    def test_search_retries_transient_provider_error(self):
        class FlakySearchProvider(FakeProvider):
            def __init__(self):
                super().__init__()
                self.search_attempts = 0

            def request(self, operation, **payload):
                self.calls.append((operation, payload))
                if operation != "search":
                    raise AssertionError(operation)
                self.search_attempts += 1
                if self.search_attempts == 1:
                    raise TimeoutError("temporary")
                return {
                    "evidence": [{
                        "memory_id": "m1",
                        "fragment_id": "f1",
                        "memory_type": "fact",
                        "content": "remembered",
                        "score": 0.9,
                        "source_session_ids": ["s1"],
                        "metadata": {},
                        "rank": 1,
                    }]
                }

        with tempfile.TemporaryDirectory() as directory:
            provider = FlakySearchProvider()
            adapter = ExternalMemoryAdapter(
                provider,
                Path(directory) / "checkpoint.json",
                retries=2,
                verbose_sessions=False,
            )
            evidence = adapter.search(
                {"question_id": "q1", "question": "what?"}, top_k=5
            )
        self.assertEqual(provider.search_attempts, 2)
        self.assertEqual([item.content for item in evidence], ["remembered"])

    def test_duplicate_session_ids_get_unique_ingestion_ids_and_are_finalized(self):
        row = {
            "question_id": "q-duplicate",
            "haystack_session_ids": ["s1", "s1"],
            "haystack_dates": ["2023/01/01 10:00", "2023/01/01 10:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "same"}],
                [{"role": "user", "content": "same"}],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider()
            outcome = ExternalMemoryAdapter(
                provider,
                Path(directory) / "checkpoint.json",
                retries=1,
                verbose_sessions=False,
            ).ensure_index(row)
        self.assertEqual(outcome.total_sessions, 2)
        self.assertEqual(outcome.ready_sessions, 2)
        self.assertEqual(
            [call[0] for call in provider.calls], ["add", "add", "finalize_scope"]
        )
        add_payloads = [payload for operation, payload in provider.calls if operation == "add"]
        self.assertEqual(
            [payload["ingestion_session_id"] for payload in add_payloads],
            ["s1", "s1__occurrence_2"],
        )
        self.assertEqual(
            [payload["source_session_id"] for payload in add_payloads], ["s1", "s1"]
        )

    def test_duplicate_session_id_with_different_content_is_preserved(self):
        row = {
            "question_id": "q-conflict",
            "haystack_session_ids": ["s1", "s1"],
            "haystack_dates": ["2023/01/01 10:00", "2023/01/01 10:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "different"}],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider()
            outcome = ExternalMemoryAdapter(
                provider,
                Path(directory) / "checkpoint.json",
                retries=1,
                verbose_sessions=False,
            ).ensure_index(row)
        self.assertEqual(outcome.total_sessions, 2)
        add_payloads = [payload for operation, payload in provider.calls if operation == "add"]
        self.assertEqual(
            [payload["messages"][0]["content"] for payload in add_payloads],
            ["first", "different"],
        )


if __name__ == "__main__":
    unittest.main()
