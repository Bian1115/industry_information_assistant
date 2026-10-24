"""正文选择与可回溯 HTML 的离线集成回归，不访问网络。"""

import hashlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
from service.web_reader_service import WebReaderService


URL = "https://example.org/report"
HTML = """<html><head><title>年度报告</title>
<meta property="article:published_time" content="2026-10-17"></head><body>
<nav>首页 联系我们</nav><article><h1>年度报告</h1>
<p>本报告统计了中国市场的年度销售情况，所有数据均使用统一口径，金额单位为万元。</p>
<p>报告记录了两个年度的实际数值，可用于比较变化情况，结果不包含其他地区的数据。</p>
<table><tr><th>年份</th><th>销售额</th></tr>
<tr><td>2025</td><td>100</td></tr><tr><td>2026</td><td>120</td></tr></table>
</article><footer>版权所有</footer></body></html>"""


class HTMLReaderPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        environment = patch.dict(os.environ, {
            "WEB_READER_CACHE_DIR": temporary.name,
            "WEB_READER_MIN_CHARS": "20",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.reader = WebReaderService()

    def test_article_main_and_table_do_not_bypass_trafilatura(self):
        for body in ("<article>旧文章区域</article>", "<main>旧主体区域</main>",
                     "<table><tr><td>侧栏表格</td></tr></table>"):
            with self.subTest(body=body):
                html = "<html><body><header>保留给提取器判断</header>" + body + "</body></html>"
                with patch("service.web_reader_service.trafilatura.extract", return_value="提取器选择的正文") as extract:
                    result = self.reader._extract_html(html, URL)
                self.assertEqual(extract.call_args.args[0], html)
                self.assertTrue(extract.call_args.kwargs["include_tables"])
                self.assertEqual(extract.call_args.kwargs["output_format"], "markdown")
                self.assertEqual(result["content"], "提取器选择的正文")
                self.assertEqual(result["extractor"], "trafilatura")

    def test_empty_or_missing_extractor_does_not_accept_body_text(self):
        with patch("service.web_reader_service.trafilatura.extract", return_value=None):
            self.assertEqual(self.reader._extract_html(HTML, URL)["content"], "")
        with patch("service.web_reader_service.trafilatura", None):
            with self.assertRaisesRegex(RuntimeError, "Trafilatura"):
                self.reader._extract_html(HTML, URL)

    def test_real_extraction_keeps_data_table_and_excludes_navigation(self):
        result = self.reader._extract_html(HTML, URL)
        self.assertIn("2025 | 100", result["content"])
        self.assertIn("2026 | 120", result["content"])
        self.assertIn("销售额", result["content"])
        self.assertNotIn("联系我们", result["content"])
        self.assertEqual(result["title"], "年度报告")
        self.assertEqual(result["published_at"], "2026-10-17")

    async def test_original_legacy_bytes_survive_decoding_and_quality_check(self):
        body = HTML.replace("<head>", '<head><meta charset="gbk">').encode("gbk")
        with patch.object(self.reader, "_download", AsyncMock(return_value=(URL, "text/html; charset=gbk", body))), \
                patch.object(self.reader, "_read_with_crawl4ai", AsyncMock(return_value=None)):
            result = await self.reader._read_uncached(URL, "年度报告")
        self.assertTrue(result.success)
        snapshot = result.body_quality["html_snapshots"][0]
        self.assertEqual(Path(snapshot["path"]).read_bytes(), body)
        self.assertEqual(snapshot["sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(snapshot["kind"], "http_response")
        self.assertIn("中国市场", result.content)

    async def test_failed_extraction_still_has_original_html_reference(self):
        body = HTML.encode()
        with patch.object(self.reader, "_download", AsyncMock(return_value=(URL, "text/html", body))), \
                patch.object(self.reader, "_extract_html", side_effect=ValueError("模拟提取失败")):
            result = await self.reader._read_uncached(URL, "年度报告")
        self.assertEqual(result.failure_category, "extraction_error")
        self.assertEqual(Path(result.body_quality["html_snapshots"][0]["path"]).read_bytes(), body)

    async def test_rejected_body_keeps_html_for_diagnostics(self):
        html = "<html><title>人机验证</title><body>请完成人机验证。</body></html>"
        with patch.object(self.reader, "_download", AsyncMock(return_value=(URL, "text/html", html.encode()))), \
                patch("service.web_reader_service.trafilatura.extract", return_value="请完成人机验证。"), \
                patch.object(self.reader, "_read_with_crawl4ai", AsyncMock(return_value=None)):
            result = await self.reader._read_uncached(URL, "年度报告")
        self.assertEqual(result.failure_category, "body_quality_rejected")
        self.assertEqual(Path(result.body_quality["html_snapshots"][0]["path"]).read_text(), html)

    async def test_disk_cache_retains_snapshot_without_refetch(self):
        download = AsyncMock(return_value=(URL, "text/html; charset=utf-8", HTML.encode()))
        with patch.object(self.reader, "validate_url", AsyncMock(return_value=URL)), \
                patch.object(self.reader, "_download", download):
            first = await self.reader.read_url(URL, "年度报告")
            self.reader._memory_cache.clear()
            cached = await self.reader.read_url(URL, "年度报告")
        self.assertTrue(cached.from_cache)
        self.assertEqual(cached.body_quality["html_snapshots"], first.body_quality["html_snapshots"])
        self.assertTrue(Path(cached.body_quality["html_snapshots"][0]["path"]).exists())
        download.assert_awaited_once()

    def test_cache_key_does_not_reuse_old_extraction(self):
        self.assertNotEqual(self.reader._cache_key(URL), hashlib.sha256(URL.encode()).hexdigest())

    def test_archive_io_failure_is_visible_and_does_not_abort_extraction(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError("模拟只读目录")), \
                self.assertLogs("service.web_reader_service", level="WARNING"):
            snapshot = self.reader._save_html_snapshot(HTML.encode(), URL, "text/html")
        self.assertEqual(snapshot["path"], "")
        self.assertIn("模拟只读目录", snapshot["error"])
        self.assertEqual(self.reader._extract_html(HTML, URL)["extractor"], "trafilatura")

    def test_browser_dom_uses_same_extractor_and_is_saved(self):
        result = self.reader._extract_browser_html(HTML, URL, "年度报告", "crawl4ai-sidecar")
        result = self.reader._validate_body(result)
        self.assertTrue(result.success)
        self.assertEqual(result.extractor, "crawl4ai-sidecar+trafilatura")
        snapshot = result.body_quality["html_snapshots"][0]
        self.assertEqual(snapshot["kind"], "browser_dom")
        self.assertEqual(Path(snapshot["path"]).read_text(), HTML)
        self.assertIn("2026 | 120", result.content)

    def test_native_markdown_does_not_restore_filtered_rows_or_other_tables(self):
        selected = "| 年份 | 数值 |\n| --- | --- |\n| 2025 | 100 |"
        with patch("service.web_reader_service.trafilatura.extract", return_value=selected):
            result = self.reader._extract_html(HTML, URL)
        self.assertEqual(result["content"], selected)
        self.assertNotIn("2026", result["content"])
        self.assertNotIn("中国市场", result["content"])

    def test_native_markdown_is_not_reconstructed_from_original_tables(self):
        original = "<html><body><table><tr><td>AB</td><td>C</td></tr></table></body></html>"
        selected = "| A | BC |\n| --- | --- |\n| D | 12 |"
        with patch("service.web_reader_service.trafilatura.extract", return_value=selected):
            text = self.reader._extract_html(original, URL)["content"]
        self.assertEqual(text, selected)
        self.assertNotIn("| AB | C |", text)

    async def test_unsuccessful_browser_recovery_still_keeps_dom_reference(self):
        request = httpx.Request("GET", URL)
        response = httpx.Response(403, request=request)
        error = httpx.HTTPStatusError("Forbidden", request=request, response=response)
        with patch.object(self.reader, "_extract_html", side_effect=ValueError("模拟浏览器提取失败")):
            browser_result = self.reader._extract_browser_html(HTML, URL, "年度报告", "crawl4ai")
        with patch.object(self.reader, "_download", AsyncMock(side_effect=error)), \
                patch.object(self.reader, "_read_with_crawl4ai", AsyncMock(return_value=browser_result)):
            result = await self.reader._read_uncached(URL, "年度报告")
        self.assertEqual(result.failure_category, "http_403")
        snapshot = result.body_quality["html_snapshots"][0]
        self.assertEqual(Path(snapshot["path"]).read_text(), HTML)

    async def test_sidecar_markdown_cannot_bypass_html_selection(self):
        self.reader.crawl4ai_endpoint = "http://sidecar.example"
        response = httpx.Response(200, json={"results": [{
            "html": HTML, "markdown": {"raw_markdown": "不应采用的浏览器导航输出"},
        }]}, request=httpx.Request("POST", "http://sidecar.example/crawl"))
        with patch("service.web_reader_service.httpx.AsyncClient") as client:
            client.return_value.__aenter__.return_value.post = AsyncMock(return_value=response)
            result = await self.reader._read_with_crawl4ai(URL, "年度报告")
        self.assertEqual(result.extractor, "crawl4ai-sidecar+trafilatura")
        self.assertNotIn("不应采用", result.content)
        self.assertIn("中国市场", result.content)

    async def test_browser_recovery_keeps_both_static_response_and_dom(self):
        shell = b'<html><body><div id="root"></div><script src="app.js"></script></body></html>'
        browser_result = self.reader._extract_browser_html(HTML, URL, "年度报告", "crawl4ai")
        with patch.object(self.reader, "_download", AsyncMock(return_value=(URL, "text/html", shell))), \
                patch.object(self.reader, "_read_with_crawl4ai", AsyncMock(return_value=browser_result)):
            result = await self.reader._read_uncached(URL, "年度报告")
        self.assertTrue(result.success)
        snapshots = result.body_quality["html_snapshots"]
        self.assertEqual([s["kind"] for s in snapshots], ["http_response", "browser_dom"])
        self.assertEqual(Path(snapshots[0]["path"]).read_bytes(), shell)
        self.assertEqual(Path(snapshots[1]["path"]).read_text(), HTML)

    def test_missing_or_oversized_browser_dom_is_not_accepted(self):
        self.assertIsNone(self.reader._extract_browser_html("", URL, "", "crawl4ai"))
        self.reader.max_bytes = 5
        self.assertIsNone(self.reader._extract_browser_html(HTML, URL, "", "crawl4ai"))

    def test_snapshot_cleanup_respects_ttl_and_ignores_unrelated_files(self):
        self.reader.cache_ttl = 100
        old = Path(self.reader._save_html_snapshot(b"old", URL, "text/html")["path"])
        fresh = Path(self.reader._save_html_snapshot(b"fresh", URL, "text/html")["path"])
        unrelated = old.parent / "人工归档.html"
        unrelated.write_text("保留")
        os.utime(old, (1000, 1000))
        os.utime(fresh, (1095, 1095))
        with patch("service.web_reader_service.time.time", return_value=1100):
            self.assertEqual(self.reader.cleanup_expired_cache(), 1)
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())
        self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
