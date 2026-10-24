"""本地网页读取、全文证据接入和安全门禁回归测试。"""

import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
sys.path.insert(0, str(APP_DIR))

from service.deep_research_v2.agents.scout import DeepScout  # noqa: E402
from service.deep_research_v2.agents.writer import (  # noqa: E402
    InsufficientEvidenceError,
    LeadWriter,
)
from service.web_reader_service import (  # noqa: E402
    UnsafeURL,
    WebReaderService,
    WebReadResult,
)


class WebReaderUnitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {
            "WEB_READER_CACHE_DIR": self.temporary_dir.name,
            "WEB_READER_MIN_CHARS": "20",
        })
        self.environment.start()
        self.reader = WebReaderService()

    def tearDown(self):
        self.environment.stop()
        self.temporary_dir.cleanup()

    def test_static_html_extracts_main_content_and_metadata(self):
        html = """
        <html><head><title>年度报告</title>
        <meta property="article:published_time" content="2026-08-01" />
        </head><body><nav>菜单</nav><article>
        <h1>年度报告</h1><p>这是经过审计的公司年度报告正文。</p>
        <p>总资产为一百亿元，相关数据可追溯。</p>
        </article></body></html>
        """
        result = self.reader._extract_html(html, "https://corp.example/report")
        self.assertEqual(result["title"], "年度报告")
        self.assertEqual(result["published_at"], "2026-08-01")
        self.assertIn("总资产", result["content"])
        self.assertEqual(result["extractor"], "trafilatura")

    async def test_local_and_credential_urls_are_rejected(self):
        with self.assertRaises(UnsafeURL):
            self.reader.normalize_url("http://user:pass@example.com/report")
        result = await self.reader.read_url("http://127.0.0.1/admin")
        self.assertEqual(result.fetch_status, "rejected")
        self.assertIn("内网", result.error)

    def test_crawl4ai_response_shapes_are_supported(self):
        self.assertEqual(
            self.reader._crawl4ai_item({
                "results": [{"html": "<article>正文</article>"}]
            })["html"],
            "<article>正文</article>",
        )

    async def test_http_statuses_have_separate_failure_categories(self):
        for status in (403, 404, 429):
            with self.subTest(status=status):
                request = httpx.Request("GET", f"https://example.com/{status}")
                response = httpx.Response(status, request=request)
                error = httpx.HTTPStatusError(
                    f"HTTP {status}", request=request, response=response
                )
                with patch.object(
                    self.reader, "_download", AsyncMock(side_effect=error)
                ), patch.object(
                    self.reader, "_read_with_crawl4ai", AsyncMock(return_value=None)
                ):
                    result = await self.reader._read_uncached(
                        str(request.url), "测试页面"
                    )
                self.assertEqual(result.failure_category, f"http_{status}")
                self.assertEqual(result.http_status, status)
                self.assertEqual(result.fetch_status, "failed")

    async def test_timeout_has_separate_failure_category(self):
        request = httpx.Request("GET", "https://example.com/slow")
        error = httpx.ReadTimeout("读取超时", request=request)
        with patch.object(
            self.reader, "_download", AsyncMock(side_effect=error)
        ), patch.object(
            self.reader, "_read_with_crawl4ai", AsyncMock(return_value=None)
        ):
            result = await self.reader._read_uncached(str(request.url), "慢页面")
        self.assertEqual(result.failure_category, "timeout")

    async def test_403_uses_browser_recovery_when_available(self):
        request = httpx.Request("GET", "https://example.com/protected")
        response = httpx.Response(403, request=request)
        error = httpx.HTTPStatusError(
            "HTTP 403", request=request, response=response
        )
        browser_result = WebReadResult(
            url=str(request.url), final_url=str(request.url),
            content="浏览器渲染后的完整正文" * 100,
            extractor="crawl4ai-sidecar", fetch_status="success",
        )
        with patch.object(
            self.reader, "_download", AsyncMock(side_effect=error)
        ), patch.object(
            self.reader, "_read_with_crawl4ai",
            AsyncMock(return_value=browser_result),
        ):
            result = await self.reader._read_uncached(
                str(request.url), "受保护页面"
            )
        self.assertTrue(result.success)
        self.assertEqual(result.recovery_action, "browser_after_http_error")

    async def test_dynamic_page_is_distinguished_from_short_content(self):
        dynamic_html = b"""
        <html><body><div id="root"></div>
        <script src="runtime.js"></script><script src="app.js"></script>
        </body></html>
        """
        with patch.object(
            self.reader,
            "_download",
            AsyncMock(return_value=(
                "https://example.com/app", "text/html; charset=utf-8", dynamic_html
            )),
        ), patch.object(
            self.reader, "_read_with_crawl4ai", AsyncMock(return_value=None)
        ):
            dynamic = await self.reader._read_uncached(
                "https://example.com/app", "动态页面"
            )
        self.assertEqual(dynamic.failure_category, "dynamic_page")

        short_html = "<html><body><article>简讯</article></body></html>".encode()
        with patch.object(
            self.reader,
            "_download",
            AsyncMock(return_value=(
                "https://example.com/brief", "text/html; charset=utf-8", short_html
            )),
        ), patch.object(
            self.reader, "_read_with_crawl4ai", AsyncMock(return_value=None)
        ):
            short = await self.reader._read_uncached(
                "https://example.com/brief", "简讯"
            )
        self.assertEqual(short.failure_category, "content_too_short")

    async def test_pdf_without_text_layer_has_separate_category(self):
        with patch.object(
            self.reader,
            "_download",
            AsyncMock(return_value=(
                "https://example.com/scan.pdf", "application/pdf", b"%PDF-scan"
            )),
        ), patch.object(self.reader, "_extract_pdf", return_value=""), patch.object(
            self.reader, "_read_with_crawl4ai", AsyncMock(return_value=None)
        ):
            result = await self.reader._read_uncached(
                "https://example.com/scan.pdf", "扫描报告"
            )
        self.assertEqual(result.failure_category, "pdf_no_text_layer")


class _FakeReader:
    async def read_many(self, items, limit=3, concurrency=2):
        item = items[0]
        return [WebReadResult(
            url=item["url"], final_url=item["url"], title=item["title"],
            content="可追溯的完整正文" * 100, extractor="trafilatura",
            fetch_status="success", fetched_at=time.time(),
        )]


class ScoutFullTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_text_is_attached_and_persisted_as_raw_source(self):
        scout = DeepScout.__new__(DeepScout)
        scout.name = "DeepScout"
        scout.logger = logging.getLogger("test.scout.reader")
        scout.web_reader = _FakeReader()
        results = [{
            "url": "https://official.example/report",
            "title": "官方报告",
            "summary": "摘要",
        }]
        state = {"raw_sources": [], "allowed_domains": ["official.example"]}

        await scout._enrich_results_with_full_text(
            state, results, section_id="sec_1", limit=3
        )

        self.assertTrue(results[0]["reader_verified"])
        self.assertIn("完整正文", results[0]["full_content"])
        self.assertEqual(state["raw_sources"][0]["related_sections"], ["sec_1"])
        self.assertEqual(state["raw_sources"][0]["extractor"], "trafilatura")

    def test_evidence_coverage_is_calculated_by_section(self):
        state = {
            "outline": [{"id": "sec_1"}, {"id": "sec_2"}],
            "facts": [{"related_sections": ["sec_1"]}],
            "raw_sources": [{"content": "正文", "related_sections": ["sec_1"]}],
        }
        DeepScout._update_evidence_coverage(state)
        self.assertEqual(state["evidence_coverage"]["fact_section_ratio"], 0.5)
        self.assertEqual(state["evidence_coverage"]["full_text_section_ratio"], 0.5)

    def test_reader_failures_are_counted_by_unique_url_and_category(self):
        state = {
            "outline": [],
            "facts": [],
            "raw_sources": [],
            "web_reader_attempts": [
                {
                    "url": "https://example.com/retry",
                    "fetch_status": "failed",
                    "failure_category": "http_403",
                    "http_status": 403,
                },
                {
                    "url": "https://example.com/retry",
                    "fetch_status": "failed",
                    "failure_category": "http_429",
                    "http_status": 429,
                },
                {
                    "url": "https://example.com/slow",
                    "fetch_status": "failed",
                    "failure_category": "timeout",
                },
                {
                    "url": "https://example.com/good",
                    "fetch_status": "success",
                },
            ],
        }
        DeepScout._update_evidence_coverage(state)
        stats = state["web_reader_stats"]
        self.assertEqual(stats["urls_total"], 3)
        self.assertEqual(stats["urls_invalid"], 2)
        self.assertEqual(stats["failure_categories"]["http_403"], 0)
        self.assertEqual(stats["failure_categories"]["http_429"], 1)
        self.assertEqual(stats["failure_categories"]["timeout"], 1)
        self.assertEqual(stats["invalid_ratio"], 0.667)
        self.assertEqual(state["evidence_coverage"]["web_reader"], stats)


class WriterEvidenceGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_writer_stops_when_no_traceable_fact_is_available(self):
        # 当前Writer只在没有可追溯事实时提前停止；全文覆盖率改为审计指标。
        writer = LeadWriter.__new__(LeadWriter)
        writer.name = "LeadWriter"
        writer.logger = logging.getLogger("test.writer.reader")
        emitted = []
        state = {
            "facts": [],
            "evidence_coverage": {
                "sections_total": 4,
                "sections_with_full_text": 1,
                "full_text_section_ratio": 0.25,
            },
            "errors": [],
            "messages": [],
            "allowed_domains": ["official.example"],
            # 进度事件直接发往图的 custom 流，不再累积在 state["messages"] 里。
            "_emit_event": emitted.append,
        }
        with self.assertRaises(InsufficientEvidenceError):
            await writer._write_report(state)
        self.assertEqual(
            emitted[0]["content"]["error_code"],
            "insufficient_evidence",
        )


if __name__ == "__main__":
    unittest.main()
