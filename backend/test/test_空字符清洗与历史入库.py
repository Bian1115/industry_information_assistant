"""空字符清洗、正文准入及独立 PostgreSQL 空间中的真实历史写入回归。"""

import copy
import json
import os
import tempfile
import time
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from sqlalchemy import text
from sqlalchemy.exc import DataError

from service.agent_history import content_hash, sanitize_history_payload
from service.web_reader_service import WebReaderService, WebReadResult
from service.web_document import content_hash as body_hash
import test_智能体独立历史 as history_tests


class NullSanitizationTests(unittest.TestCase):
    def test_nested_values_preserve_normal_text_and_original(self):
        raw = {"value": [{"content": "中\x00文\n列\t值\r\n😀"}],
               "literal": r"\u0000", "number": 3, "flag": True, "empty": None}
        before = copy.deepcopy(raw)
        cleaned, removed = sanitize_history_payload(raw)
        self.assertEqual(cleaned["value"][0]["content"], "中文\n列\t值\r\n😀")
        self.assertEqual(cleaned["literal"], r"\u0000")
        self.assertEqual(removed, 1)
        self.assertEqual(raw, before)
        self.assertEqual(sanitize_history_payload(cleaned), (cleaned, 0))

    def test_invalid_keys_are_rejected_without_colliding(self):
        with self.assertRaisesRegex(ValueError, "字段名含空字符"):
            sanitize_history_payload({"a": 1, "a\x00": 2})


class NullReaderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        env = patch.dict(os.environ, {"WEB_READER_CACHE_DIR": temporary.name,
                                    "WEB_READER_MIN_CHARS": "20"})
        env.start()
        self.addCleanup(env.stop)
        self.reader = WebReaderService()

    async def test_pdf_nulls_removed_before_cache_and_hash(self):
        raw = "[第1页]\n博\x00物馆的教育功能值得研究。\n国内外现状与发展趋势。\x00"
        with patch.object(self.reader, "_download", AsyncMock(return_value=(
                "https://example.org/report.pdf", "application/pdf", b"%PDF"))), \
                patch.object(self.reader, "_extract_pdf", return_value=raw):
            result = await self.reader._read_uncached("https://example.org/report.pdf", "")
        self.assertTrue(result.success)
        self.assertEqual(result.content, raw.replace("\x00", ""))
        self.assertEqual(result.body_quality["content_sha256"], body_hash(result.content))
        audit = result.body_quality["text_sanitization"]
        self.assertEqual(audit["removed_null_chars"], 2)
        self.assertEqual(audit["raw_content_sha256"], body_hash(raw))
        key = self.reader._cache_key(result.url)
        self.reader._save_cache(key, result)
        cached = self.reader._load_cache(key)
        self.assertEqual(cached.content, result.content)
        self.assertEqual(cached.body_quality["text_sanitization"], audit)

    def test_old_cache_is_cleaned_without_rewriting_original(self):
        raw = "博\x00物馆开展教育活动。\n不同国家有不同的实践路径。"
        key = self.reader._cache_key("https://example.org/old")
        self.reader._save_cache(key, WebReadResult(url="https://example.org/old", final_url="https://example.org/old",
            content=raw, fetch_status="success", fetched_at=time.time()))
        original = self.reader._cache_path(key).read_bytes()
        cached = self.reader._load_cache(key)
        self.assertNotIn("\x00", cached.content)
        self.assertEqual(self.reader._memory_cache[key].content, raw)
        self.assertEqual(self.reader._cache_path(key).read_bytes(), original)

    def test_null_only_old_cache_cannot_be_valid_evidence(self):
        result = self.reader._validate_body(WebReadResult(url="https://example.org/empty", final_url="https://example.org/empty",
            content="\x00" * 100, fetch_status="success"))
        self.assertFalse(result.success)
        self.assertEqual(result.fetch_status, "failed")
        self.assertEqual(result.error, "empty_body")

    async def test_null_padding_cannot_satisfy_minimum_body_length(self):
        with patch.object(self.reader, "_download", AsyncMock(return_value=(
                "https://example.org/report.pdf", "application/pdf", b"%PDF"))), \
                patch.object(self.reader, "_extract_pdf", return_value="短文" + "\x00" * 100):
            result = await self.reader._read_uncached("https://example.org/report.pdf", "")
        self.assertFalse(result.success)
        self.assertEqual(result.failure_category, "content_too_short")

    def test_browser_or_ocr_result_is_cleaned_by_shared_validation(self):
        raw = "博\x00物馆开展教育活动。\n多种教学方式服务不同年龄群体。"
        result = self.reader._validate_body(WebReadResult(url="https://example.org/rendered", final_url="https://example.org/rendered",
            content=raw, extractor="ocr-sidecar", fetch_status="success"))
        self.assertTrue(result.success)
        self.assertEqual(result.content, raw.replace("\x00", ""))

    def test_old_cache_or_fallback_null_padding_cannot_pass_body_gate(self):
        result = self.reader._validate_body(WebReadResult(url="https://example.org/padded",
            final_url="https://example.org/padded", content="短文" + "\x00" * 100,
            fetch_status="success"))
        self.assertFalse(result.success)
        self.assertEqual(result.failure_category, "content_too_short")


class NullHistoryPostgresTests(history_tests.PostgresHistoryTests):
    # 沿用现有测试的 UUID 独立 schema，结束后删除，生产运行表不受影响。
    async def test_database_rejects_raw_but_history_persists_cleaned_payload(self):
        raw = {"tool_name": "v2_web_fulltext_read", "value": [
            {"url": "https://example.org/report.pdf", "content": "博\x00物馆\n年龄\t人数😀"}],
            "literal": r"\u0000"}
        with self.assertRaises(DataError):
            with self.engine.begin() as db:
                db.execute(text("SELECT CAST(:payload AS JSONB)"),
                           {"payload": json.dumps(raw)})
        call, event = str(uuid.uuid4()), str(uuid.uuid4())
        with self.assertLogs("service.agent_history", level="WARNING") as audit:
            row = await self.store.append(self.scope, "DeepScout", "tool_raw_result", raw,
                                          call, event_id=event)
        self.assertIn("removed=1", audit.output[0])
        self.assertIn(content_hash(raw), audit.output[0])
        self.assertEqual(row["payload"]["value"][0]["content"], "博物馆\n年龄\t人数😀")
        self.assertEqual(row["payload"]["literal"], r"\u0000")
        self.assertEqual(row["content_hash"], content_hash(row["payload"]))
        again = await self.store.append(self.scope, "DeepScout", "tool_raw_result",
                                       row["payload"], call, event_id=event)
        self.assertEqual(row, again)
        loaded = (await self.store.load(self.scope, "DeepScout"))["messages"]
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0], row)
        self.assertIn("\x00", raw["value"][0]["content"])

    async def test_real_archived_reader_payload_can_be_saved(self):
        fixture = os.getenv("NULL_FIX_REPLAY_FILE")
        if not fixture:
            self.skipTest("未提供原始缓存复现输入")
        from pathlib import Path
        payload = json.loads(Path(fixture).read_text())
        cleaned, removed = sanitize_history_payload(payload)
        self.assertGreater(removed, 0)
        row = await self.store.append(self.scope, "DeepScout", "tool_raw_result",
                                      payload, str(uuid.uuid4()))
        self.assertEqual(row["payload"], cleaned)
        self.assertEqual(row["content_hash"], content_hash(cleaned))


if __name__ == "__main__":
    unittest.main()
