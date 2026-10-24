"""DashScope Batch 记忆索引流水线的离线测试。"""

import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rebuild_longmemeval_batch_index.py"
)
SPEC = importlib.util.spec_from_file_location("rebuild_longmemeval_batch_index", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EmbeddingBatchTests(unittest.TestCase):
    def test_embedding_requests_use_ten_inputs_per_request(self):
        keys = [f"key-{index}" for index in range(23)]
        requests, groups = MODULE.embedding_requests(
            keys,
            {key: f"text {key}" for key in keys},
            "text-embedding-v4",
            1,
        )

        self.assertEqual([len(item["body"]["input"]) for item in requests], [10, 10, 3])
        self.assertEqual(sum(len(group) for group in groups.values()), 23)
        self.assertTrue(all(item["url"] == "/v1/embeddings" for item in requests))

    def test_float32_embedding_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache = MODULE.EmbeddingCache(Path(temporary_dir) / "cache.sqlite3")
            cache.put("content-sha", [0.125, -0.25, 0.5])
            cache.commit()

            self.assertEqual(cache.keys(), {"content-sha"})
            self.assertEqual(cache.get("content-sha"), [0.125, -0.25, 0.5])


class BatchRunnerTests(unittest.TestCase):
    def test_batch_runner_checkpoints_and_downloads_completed_output(self):
        output = b'{"custom_id":"item-1","response":{"status_code":200,"body":{}}}\n'
        files = SimpleNamespace(
            create=lambda **_: SimpleNamespace(id="file-input"),
            content=lambda _: SimpleNamespace(content=output),
        )
        batches = SimpleNamespace(
            create=lambda **_: SimpleNamespace(id="batch-1", status="validating"),
            retrieve=lambda _: SimpleNamespace(
                model_dump=lambda **__: {
                    "status": "completed",
                    "output_file_id": "file-output",
                    "error_file_id": None,
                    "request_counts": {"total": 1, "completed": 1, "failed": 0},
                }
            ),
        )
        client = SimpleNamespace(files=files, batches=batches)

        with tempfile.TemporaryDirectory() as temporary_dir:
            runner = MODULE.BatchRunner(client, Path(temporary_dir), poll_seconds=0)
            rows = runner.run(
                "test-batch",
                "/v1/chat/completions",
                [{"custom_id": "item-1", "method": "POST", "url": "/v1/chat/completions", "body": {}}],
            )

            self.assertEqual(rows[0]["custom_id"], "item-1")
            self.assertTrue((Path(temporary_dir) / "test-batch.batch.json").exists())
            self.assertTrue((Path(temporary_dir) / "test-batch.output.jsonl").exists())


class ExtractionTests(unittest.TestCase):
    def test_one_request_combines_summary_and_atomic_facts(self):
        turn = MODULE.MemoryTurn(
            message_id="message-1",
            role="user",
            content="我带领4名工程师",
            created_at=datetime(2024, 1, 1, 8, 0),
        )
        task = MODULE.SessionTask(
            custom_id="q01-s000",
            question_id="q1",
            source_session_id="s1",
            session_index=0,
            user_id="user-1",
            session_id="session-1",
            turns=[turn],
        )

        request = MODULE.extraction_request(task, "qwen-plus-test")
        prompt = request["body"]["messages"][1]["content"]

        self.assertIn('"summary"', prompt)
        self.assertIn('"atomic_facts"', prompt)
        self.assertIn("message_id=message-1", prompt)
        self.assertIn("json", request["body"]["messages"][0]["content"])
        self.assertEqual(request["body"]["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
