"""验证码、脚本、缓存及引用门禁的离线回归；不访问外网或调用模型。"""

import copy
import json
import logging
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from service.web_document import (
    BODY_VERSION, QUALITY_VERSION, assess_body, content_hash, index_document,
    resolve_sentences, source_usable,
)
from service.web_reader_service import WebReadResult, WebReaderService
from service.deep_research_v2.agents.scout import DeepScout

URL = "https://example.org/article"
ARTICLE = "\n".join([
    "This study evaluates human verification methods using a controlled experiment. "
    "The authors describe the participants, measurement procedure and limitations. "
    "The results provide evidence for the reported comparison under these conditions."
] * 4)
SCRIPT = "function a(e){var r=document.cookie;return window.location.href+typeof e+r;}" * 80
CHALLENGE = ("# Are you a robot?\n"
             "Please confirm you are a human by completing the captcha challenge below.\n"
             "Verification successful. Waiting for example.org to respond\n" + SCRIPT)


def old_source(body):
    return {"id": "source_1", "url": URL, "title": "Research article", "content": body,
            "body_quality": {"version": BODY_VERSION, "status": "passed",
                             "content_sha256": content_hash(body)}, **index_document(body, URL)}


def scout():
    agent = DeepScout.__new__(DeepScout)
    agent.name = "DeepScout"
    agent.logger = logging.getLogger("test.abnormal-body")
    agent.add_message = lambda *args, **kwargs: None
    agent.parse_json_response = json.loads
    agent.call_llm = AsyncMock()
    return agent


class AbnormalBodyRuleTests(unittest.TestCase):
    def test_large_challenge_cannot_pass_by_length_or_script_narration(self):
        body, quality = assess_body(CHALLENGE, "Research article")
        self.assertEqual(body, CHALLENGE)
        self.assertEqual(quality["status"], "rejected")
        self.assertIn("captcha_challenge", quality["reasons"])
        self.assertEqual(quality["quality_version"], QUALITY_VERSION)
        self.assertLess(quality["metrics"]["narrative_ratio"], 0.1)

    def test_chinese_challenge_and_direct_instruction_are_rejected(self):
        for text in ("# 人机验证\n请完成下方的验证码。", "Please verify that you are human."):
            with self.subTest(text=text):
                self.assertEqual(assess_body(text)[1]["reasons"], ["captcha_challenge"])

    def test_script_body_without_captcha_keyword_is_rejected(self):
        self.assertEqual(assess_body(SCRIPT)[1]["reasons"], ["script_body"])

    def test_long_prose_tables_and_formula_are_not_rejected_by_length(self):
        cases = [(ARTICLE * 30, "passed"),
                 ("\n".join(["| model | accuracy | 0.98 |"] * 150), "passed"),
                 # 原有规则对纯公式页保留待复核，不能把它归类为脚本或验证码。
                 ("\n".join(["x_i = alpha + beta * y_i; E = mc^2"] * 150), "uncertain")]
        for text, expected in cases:
            with self.subTest(chars=len(text)):
                cleaned, quality = assess_body(text)
                self.assertEqual(cleaned, text)
                self.assertEqual(quality["status"], expected)
                self.assertFalse(set(quality["reasons"]) & {"captcha_challenge", "script_body", "script_contamination"})

    def test_article_discussing_captcha_with_quoted_instructions_is_usable(self):
        text = "# Are you a robot?\n" + ARTICLE + "\n> Please verify that you are human."
        self.assertEqual(assess_body(text, "Are you a robot?")[1]["status"], "passed")

    def test_legitimate_fenced_javascript_is_usable_with_article(self):
        self.assertEqual(assess_body(ARTICLE + "\n```javascript\n" + SCRIPT + "\n```")[1]["status"], "passed")

    def test_large_fenced_example_does_not_turn_article_into_navigation(self):
        text = ARTICLE + "\n## Implementation\n```javascript\n" + SCRIPT * 5 + "\n```\n## Discussion\n" + ARTICLE
        self.assertEqual(assess_body(text)[1]["status"], "passed")

    def test_fenced_script_without_article_is_not_research_body(self):
        self.assertEqual(assess_body("```javascript\n" + SCRIPT + "\n```")[1]["reasons"], ["script_body"])

    def test_mixed_challenge_and_article_requires_review(self):
        quality = assess_body("# Are you a robot?\nPlease verify you are human.\n" + ARTICLE)[1]
        self.assertEqual(quality["status"], "uncertain")
        self.assertIn("mixed_challenge_content", quality["reasons"])

    def test_mixed_unfenced_script_and_article_cannot_be_used_directly(self):
        text = ARTICLE + "\n" + SCRIPT
        self.assertEqual(assess_body(text)[1]["status"], "uncertain")
        self.assertFalse(source_usable(old_source(text)))

    def test_old_passed_checkpoint_cannot_supply_valid_script_sentence_ids(self):
        source = old_source(CHALLENGE)
        original = copy.deepcopy(source)
        self.assertFalse(source_usable(source))
        self.assertEqual(resolve_sentences(source, source["document_id"], ["s000001"]), [])
        self.assertEqual(source, original)

    def test_clean_old_checkpoint_preserves_ids_and_offsets(self):
        source = old_source(ARTICLE)
        original = copy.deepcopy(source)
        self.assertTrue(source_usable(source))
        self.assertTrue(resolve_sentences(source, source["document_id"], ["s000001"]))
        self.assertEqual(source, original)

    def test_changed_content_cannot_reuse_current_quality_stamp(self):
        source = old_source(ARTICLE)
        source["body_quality"] = assess_body(ARTICLE)[1]
        source["content"] = SCRIPT
        self.assertFalse(source_usable(source))


class AbnormalBodyReaderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        env = patch.dict(os.environ, {"WEB_READER_CACHE_DIR": folder.name, "WEB_READER_MIN_CHARS": "20"})
        env.start()
        self.addCleanup(env.stop)
        self.reader = WebReaderService()

    def test_memory_and_disk_cache_recheck_old_passed_without_rewriting_original(self):
        value = WebReadResult(url=URL, final_url=URL, content=CHALLENGE,
                              body_quality=old_source(CHALLENGE)["body_quality"],
                              fetch_status="success", fetched_at=time.time())
        key = self.reader._cache_key(URL)
        self.reader._save_cache(key, value)
        saved = self.reader._cache_path(key).read_bytes()
        for disk_only in (False, True):
            with self.subTest(disk_only=disk_only):
                if disk_only:
                    self.reader._memory_cache.clear()
                result = self.reader._load_cache(key)
                self.assertFalse(result.success)
                self.assertEqual(result.failure_category, "body_quality_rejected")
                self.assertIn("captcha_challenge", result.error)
                self.assertEqual(self.reader._cache_path(key).read_bytes(), saved)
                self.assertEqual(self.reader._memory_cache[key].fetch_status, "success")

    def test_cached_actual_page_title_survives_quality_upgrade(self):
        quality = old_source("Please wait.")["body_quality"]
        quality["page_title"] = "Are you a robot?"
        value = WebReadResult(url=URL, final_url=URL, title="Research article", content="Please wait.",
                              fetch_status="success", body_quality=quality)
        result = self.reader._validate_body(value)
        self.assertFalse(result.success)
        self.assertIn("captcha_challenge", result.error)
        self.assertEqual(result.body_quality["page_title"], "Are you a robot?")

    async def test_browser_403_and_timeout_recovery_cannot_bypass_gate(self):
        request = httpx.Request("GET", URL)
        errors = [httpx.HTTPStatusError("403", request=request, response=httpx.Response(403, request=request)),
                  httpx.ReadTimeout("timeout", request=request)]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                value = WebReadResult(url=URL, final_url=URL, content=CHALLENGE, fetch_status="success")
                with patch.object(self.reader, "_download_with_retry", AsyncMock(side_effect=error)), \
                        patch.object(self.reader, "_read_with_crawl4ai", AsyncMock(return_value=value)):
                    result = await self.reader._read_uncached(URL, "Research article")
                self.assertFalse(result.success)
                self.assertEqual(result.failure_category, "body_quality_rejected")

    async def test_http_200_short_challenge_has_explicit_failure_reason(self):
        html = b"<html><head><title>Are you a robot?</title></head><body><h1>Are you a robot?</h1><p>Please verify you are human.</p></body></html>"
        self.reader.min_chars = 500
        with patch.object(self.reader, "_download_with_retry", AsyncMock(return_value=(URL, "text/html", html, 1))), \
                patch.object(self.reader, "_read_with_crawl4ai", AsyncMock(return_value=None)):
            result = await self.reader._read_uncached(URL, "Research article")
        self.assertFalse(result.success)
        self.assertEqual(result.failure_category, "body_quality_rejected")
        self.assertIn("captcha_challenge", result.error)

    def test_static_html_keeps_article_and_removes_script_style_nodes(self):
        html = '<html><body><article><p>' + ARTICLE + '</p></article><script>' + SCRIPT + '</script><style>.x{display:none}</style></body></html>'
        result = self.reader._extract_html(html, URL)
        self.assertIn("controlled experiment", result["content"])
        self.assertNotIn("document.cookie", result["content"])
        self.assertEqual(assess_body(result["content"])[1]["status"], "passed")


class AbnormalBodyScoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_clear_challenge_is_rejected_without_model_review(self):
        agent = scout()
        result = WebReadResult(url=URL, final_url=URL, content=CHALLENGE,
                               body_quality=old_source(CHALLENGE)["body_quality"], fetch_status="success")
        await agent._check_source_body(result, {})
        self.assertFalse(result.success)
        agent.call_llm.assert_not_awaited()

    async def test_model_cannot_approve_remaining_script_contamination(self):
        for selected in (["b000005"], [f"b{i:06d}" for i in range(1, 6)]):
            with self.subTest(selected=selected):
                agent = scout()
                agent.call_llm.return_value = json.dumps({"status": "passed", "body_block_ids": selected})
                result = WebReadResult(url=URL, final_url=URL, content=ARTICLE + "\n" + SCRIPT, fetch_status="success")
                await agent._check_source_body(result, {})
                self.assertFalse(source_usable({"content": result.content, "body_quality": result.body_quality}))
                self.assertIn(result.body_quality["status"], {"uncertain", "rejected"})

    async def test_model_may_select_clean_original_article_blocks(self):
        agent = scout()
        agent.call_llm.return_value = json.dumps({"status": "partial", "body_block_ids": [f"b{i:06d}" for i in range(1, 5)]})
        result = WebReadResult(url=URL, final_url=URL, content=ARTICLE + "\n" + SCRIPT, fetch_status="success")
        await agent._check_source_body(result, {})
        self.assertEqual(result.content, ARTICLE)
        self.assertEqual(result.body_quality["status"], "partial")
        self.assertEqual(result.body_quality["content_sha256"], content_hash(ARTICLE))
        self.assertTrue(source_usable({"content": result.content, "body_quality": result.body_quality}))

    async def test_rejected_source_is_recorded_and_next_clean_source_is_ingested(self):
        agent = scout()
        agent.web_reader = object()
        agent._ensure_tool_runtime = lambda: None
        agent.tool_definitions = {"web_read": object()}
        bad = WebReadResult(url=URL, final_url=URL, content=CHALLENGE, fetch_status="success")
        good = WebReadResult(url=URL + "/clean", final_url=URL + "/clean", content=ARTICLE, fetch_status="success")
        agent.tool_runtime = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(value=[bad.to_dict(), good.to_dict()])))
        state = {"allowed_domains": ["example.org"], "raw_sources": [], "facts": [], "data_points": [], "hypotheses": []}
        items = [{"url": bad.url}, {"url": good.url}]
        await agent._enrich_results_with_full_text(state, items, section_id="sec1")
        self.assertFalse(items[0]["reader_verified"])
        self.assertTrue(items[1]["reader_verified"])
        self.assertEqual(len(state["raw_sources"]), 1)
        self.assertEqual(state["web_reader_attempts"][0]["failure_category"], "body_quality_rejected")
        self.assertEqual(state["raw_sources"][0]["content"], ARTICLE)
        agent.call_llm.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
