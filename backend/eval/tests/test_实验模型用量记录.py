"""实验HTTP用量记录：只使用MockTransport，不发起真实网络调用。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2] / "app")]
import httpx
from eval.scripts.记录模型调用用量 import Recorder, summarize
from eval.scripts.运行Critic职责分离五题配对消融 import citation_metrics, experiment_model_aliases


class UsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_complete_usage_and_group_are_preserved_without_request_body(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / "用量.jsonl")
            response = {"id": "req-1", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=response))) as client:
                with recorder.installed(), recorder.group("公共首稿"):
                    await client.post("https://example.test/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "不保存原文"}]})
            self.assertEqual(recorder.summary()["total_tokens"], 15)
            self.assertEqual(recorder.rows[0]["group"], "公共首稿")
            self.assertNotIn("不保存原文", recorder.path.read_text())

    async def test_failed_request_with_unknown_usage_does_not_become_zero_total(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / "用量.jsonl")
            def fail(req):
                raise httpx.ReadTimeout("不得写出敏感异常文本", request=req)
            async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
                with recorder.installed(), self.assertRaises(httpx.ReadTimeout):
                    await client.post("https://example.test/chat/completions", json={"model": "test", "messages": []})
            self.assertIsNone(recorder.summary()["total_tokens"])
            self.assertEqual(recorder.summary()["failed_calls"], 1)
            self.assertNotIn("不得写出敏感", recorder.path.read_text())

    async def test_non_model_http_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / "用量.jsonl")
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={}))) as client:
                with recorder.installed():
                    await client.get("https://example.test/page")
                    await client.post("https://example.test/search", json={"query": "题目"})
            self.assertEqual(recorder.summary()["calls"], 0)

    def test_sync_embedding_keeps_missing_output_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / "用量.jsonl")
            with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={
                    "usage": {"prompt_tokens": 12, "total_tokens": 12}}))) as client:
                with recorder.installed():
                    client.post("https://example.test/embeddings", json={"model": "embedding", "input": ["资料"]})
            self.assertEqual(recorder.rows[0]["kind"], "embedding")
            self.assertEqual(recorder.summary()["total_tokens"], 12)
            self.assertIsNone(recorder.summary()["output_tokens"])

    def test_shared_prefix_is_not_double_counted_as_actual_usage(self):
        rows = [{"group": group, "component": "test", "model": "m", "kind": "language_model",
                 "status": "completed", "input_tokens": amount, "output_tokens": 0, "total_tokens": amount}
                for group, amount in [("公共首稿", 10), ("开启审核", 5)]]
        self.assertEqual(summarize(rows)["total_tokens"], 15)
        self.assertEqual(summarize([r for r in rows if r["group"] == "公共首稿"])["total_tokens"], 10)

    def test_invalid_usage_values_remain_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / "用量.jsonl")
            req = httpx.Request("POST", "https://example.test/chat/completions", json={"model": "m", "messages": []})
            row = recorder.start(req)
            recorder.finish(row, httpx.Response(200, json={"usage": {"total_tokens": "10", "prompt_tokens": True}}, request=req))
            self.assertIsNone(recorder.summary()["total_tokens"])

    def test_partial_citation_execution_does_not_generate_success_rate(self):
        result = citation_metrics({"citation_verification": {"status": "incomplete", "claims_count": 10,
            "supported_count": 3, "extracted_claims_checked": False}})
        self.assertIsNone(result["supported_rate"])

    def test_completed_citation_rates_use_extracted_claim_denominator(self):
        result = citation_metrics({"citation_verification": {"status": "rejected", "claims_count": 10,
            "supported_count": 6, "missing_citation_count": 2, "prediction_as_fact_count": 1,
            "extracted_claims_checked": True}})
        self.assertEqual(result["supported_rate"], .6)
        self.assertEqual(result["missing_citation_rate"], .2)

    async def test_experiment_alias_is_explicit_and_records_actual_requested_model(self):
        from openai import AsyncOpenAI
        requested, thinking = [], []
        def response(req):
            payload = json.loads(req.content)
            requested.append(payload["model"])
            thinking.append(payload.get("enable_thinking"))
            return httpx.Response(200, json={"id": "r", "object": "chat.completion", "created": 1,
                "model": requested[-1], "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / "用量.jsonl")
            async with AsyncOpenAI(api_key="test", base_url="https://example.test/v1",
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(response))) as client:
                with recorder.installed(), experiment_model_aliases({"qwen3-max": "qwen3.8-max"}):
                    await client.chat.completions.create(model="qwen3-max", messages=[])
            self.assertEqual(requested, ["qwen3.8-max"])
            self.assertEqual(thinking, [False])
            self.assertEqual(recorder.rows[0]["model"], "qwen3.8-max")


if __name__ == "__main__":
    unittest.main()
