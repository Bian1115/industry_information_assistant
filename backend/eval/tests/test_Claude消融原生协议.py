"""Claude切换的协议、用量和禁止欠费服务请求机制；不访问真实模型。"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2] / "app")]
import httpx
from openai import AsyncOpenAI, OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from eval.scripts.Claude消融原生协议适配 import claude_profile, completion_from, 实验禁用付费向量检索
from eval.scripts.记录模型调用用量 import Recorder


class ClaudeProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"CLAUDE_RELAY_API_KEY": "test-secret", "CLAUDE_RELAY_BASE_URL": "https://example.test/v1", "CLAUDE_RELAY_MODEL": "claude-test"})
        self.environment.start(); self.addCleanup(self.environment.stop)

    async def test_async_bridge_uses_native_model_and_counts_cache_once(self):
        requests = []
        def response(request):
            requests.append(json.loads(request.content))
            self.assertEqual(request.url.path, "/v1/messages")
            return httpx.Response(200, json={"id": "r", "model": "claude-test", "content": [{"type": "thinking", "thinking": "不当成输出正文"}, {"type": "text", "text": '{"ok":true}'}], "usage": {"input_tokens": 7, "output_tokens": 2, "cache_creation_input_tokens": 11, "cache_read_input_tokens": 13}})
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / "用量.jsonl")
            async with AsyncOpenAI(api_key="unused", base_url="https://never-request.test") as client:
                with recorder.installed(), claude_profile(transport=httpx.MockTransport(response)):
                    result = await client.chat.completions.create(model="qwen3-max", messages=[{"role": "system", "content": "系统"}, {"role": "user", "content": "问题"}], temperature=0, response_format={"type": "json_object"})
            self.assertEqual(result.choices[0].message.content, '{"ok":true}')
            self.assertEqual(result.usage.prompt_tokens, 31)
            self.assertEqual(recorder.summary()["total_tokens"], 33)
            self.assertEqual(recorder.rows[0]["model"], "claude-test")
            self.assertNotIn("test-secret", recorder.path.read_text())
        self.assertNotIn("temperature", requests[0]);self.assertNotIn("response_format", requests[0])
        self.assertIn("JSON", requests[0]["system"])

    def test_sync_bridge_handles_nli_client_and_preserves_missing_usage(self):
        with OpenAI(api_key="unused", base_url="https://never-request.test") as client:
            with claude_profile(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]}))):
                result = client.chat.completions.create(model="qwen3-max", messages=[{"role": "user", "content": "NLI"}])
        self.assertIsNone(result.usage)

    async def test_embedding_and_rerank_are_blocked_before_network(self):
        from service.summary_temporary_retrieval import RetrievalAPI, TemporaryRetrievalSettings
        api = RetrievalAPI(TemporaryRetrievalSettings())
        with claude_profile():
            with self.assertRaises(实验禁用付费向量检索): await api.embed(["证据"])
            with self.assertRaises(实验禁用付费向量检索): await api.rerank("问题", ["证据"], 1)
        self.assertIsNone(api.client)

    def test_native_rate_limit_is_explicit_retryable_error(self):
        request = httpx.Request("POST", "https://example.test/v1/messages")
        with self.assertRaises(RateLimitError):
            completion_from(httpx.Response(429, json={}, request=request), "claude-test")

    async def test_native_connection_and_timeout_reach_existing_agent_retry_types(self):
        for native_error, sdk_error in [(httpx.ConnectError, APIConnectionError), (httpx.ReadTimeout, APITimeoutError)]:
            def fail(request):
                raise native_error("测试错误", request=request)
            async with AsyncOpenAI(api_key="unused", base_url="https://never-request.test") as client:
                with claude_profile(transport=httpx.MockTransport(fail), lexical_only=False):
                    with self.assertRaises(sdk_error):
                        await client.chat.completions.create(model="claude-test", messages=[{"role":"user","content":"问题"}])

    def test_hybrid_profile_preserves_original_retrieval_methods(self):
        from service.summary_temporary_retrieval import RetrievalAPI
        original=RetrievalAPI.embed
        with claude_profile(lexical_only=False):
            self.assertIs(RetrievalAPI.embed,original)
