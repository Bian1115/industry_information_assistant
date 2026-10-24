"""正文准入、句子定位、Scout 入库及 Writer 原句传递的离线回归。"""

import copy
import json
import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from service.web_document import (
    assess_body, content_hash, format_numbered_body, index_document,
    resolve_sentences, sentence_records, split_sentences, quote_matches_sentences,
)
from service.web_reader_service import WebReadResult, WebReaderService
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.state import create_initial_state
from service.deep_research_v2.evidence import build_evidence_view, normalize_evidence_state
from service.deep_research_v2.quality_pipeline import bind_verified_evidence
from service.deep_research_v2.writer_claims import format_claim_evidence

URL = "https://example.gov.cn/report"
BODY = "项目于2023年投运。\n服务人口为6000人。\n统计范围仅包含城区。"
LIST = "淘豆网\n* 日阅读TOP20\n" + "\n".join(f"![文档标题{i}](https://example.org/{i}.png)\n上传人:作者{i}" for i in range(20))


def source(text=BODY, url=URL, ident="source_1"):
    return {"id": ident, "url": url, "title": "项目报告", "content": text,
            "body_quality": {"status": "passed"}, **index_document(text, url)}


def scout():
    value = DeepScout.__new__(DeepScout)
    value.name = "DeepScout"
    value.logger = logging.getLogger("test.numbered-body")
    value.add_message = lambda *args, **kwargs: None
    value.parse_json_response = json.loads
    return value


def fixture():
    doc = source()
    item = {**doc, "full_content": BODY, "reader_verified": True}
    format_numbered_body(item)
    state = {"raw_sources": [doc], "facts": [], "data_points": [], "hypotheses": [], "hypothesis_evidence": []}
    raw = {"fact_id": "f1", "content": "项目于2023年投入运营", "source_url": URL,
           "document_id": doc["document_id"], "sentence_ids": ["s000001"]}
    return state, item, raw


class BodyQualityTests(unittest.TestCase):
    def test_markdown_recommendation_list_is_rejected(self):
        self.assertEqual(assess_body(LIST)[1]["status"], "rejected")

    def test_normal_article_recommendation_tail_is_removed(self):
        body = "这是项目的正式正文，介绍建设过程及运营指标。" * 20
        cleaned, quality = assess_body(body + "\n相关推荐\n其他文档甲\n其他文档乙")
        self.assertEqual(cleaned, body)
        self.assertEqual(quality["status"], "passed")
        self.assertIn("recommendation_tail_removed", quality["reasons"])

    def test_keyword_inside_normal_paragraph_is_not_rejected(self):
        body = "本报告推荐使用生物处理技术，并讨论了适用条件。"
        self.assertEqual(assess_body(body)[1]["status"], "passed")

    def test_http_200_error_page_is_rejected(self):
        self.assertEqual(assess_body("Access denied\nPlease contact administrator", "Access denied")[1]["status"], "rejected")

    def test_preview_is_uncertain_and_table_is_usable(self):
        self.assertEqual(assess_body("仅供预览\n" + BODY)[1]["status"], "uncertain")
        self.assertEqual(assess_body("\n".join("| 年份 | 收入 |" for _ in range(10)))[1]["status"], "passed")

    def test_reader_rechecks_old_cached_success(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"WEB_READER_CACHE_DIR": folder}):
            reader = WebReaderService()
            result = WebReadResult(url=URL, final_url=URL, content=LIST, fetch_status="success", fetched_at=1000)
            reader._save_cache(reader._cache_key(URL), result)
            with patch("service.web_reader_service.time.time", return_value=1001):
                cached = reader._load_cache(reader._cache_key(URL))
            self.assertFalse(cached.success)
            self.assertEqual(cached.failure_category, "body_quality_rejected")

    def test_html_fallback_excludes_recommendation_module(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"WEB_READER_CACHE_DIR": folder}), patch("service.web_reader_service.trafilatura", None):
            result = WebReaderService()._extract_html(
                '<html><body><nav>导航</nav><article>项目2023年投运。</article><div class="recommend-list">推荐甲乙</div></body></html>', URL)
        self.assertIn("2023年", result["content"])
        self.assertNotIn("推荐甲乙", result["content"])

    def test_target_title_mismatch_is_flagged_for_model_review(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"WEB_READER_CACHE_DIR": folder}):
            result = WebReaderService()._validate_body(WebReadResult(
                url=URL, final_url=URL, content=BODY, title="污水处理报告", fetch_status="success"), "小学数学教案")
        self.assertEqual(result.body_quality["status"], "uncertain")
        self.assertIn("target_title_mismatch", result.body_quality["reasons"])


class SentenceIndexTests(unittest.TestCase):
    def test_original_offsets_cover_chinese_english_decimals_and_table(self):
        text = 'Dr. Smith measured 3.14 mg. Results improved.\n项目2023年投运！服务6000人。\n| 数值 | 3.14 |'
        rows = split_sentences(text)
        self.assertEqual([text[r["start"]:r["end"]] for r in rows], [
            'Dr. Smith measured 3.14 mg.', 'Results improved.', '项目2023年投运！', '服务6000人。', '| 数值 | 3.14 |'])

    def test_duplicate_sentences_have_distinct_positions(self):
        doc = source("重复句。\n重复句。")
        rows = sentence_records(doc)
        self.assertNotEqual(rows[0]["id"], rows[1]["id"])
        self.assertNotEqual(rows[0]["start"], rows[1]["start"])
        self.assertEqual(rows[0]["text"], rows[1]["text"])

    def test_url_query_punctuation_does_not_split_url(self):
        text = "详情见 https://example.org/read?item=1。下一句。"
        self.assertEqual([text[r["start"]:r["end"]] for r in split_sentences(text)],
                         ["详情见 https://example.org/read?item=1。", "下一句。"])

    def test_ids_are_stable_but_change_when_document_changes(self):
        self.assertEqual(index_document(BODY, URL), index_document(BODY, URL))
        self.assertNotEqual(index_document(BODY, URL)["document_id"], index_document(BODY + "修订。", URL)["document_id"])
        self.assertNotEqual(index_document(BODY, URL)["document_id"], index_document(BODY, URL + "2")["document_id"])

    def test_modified_content_or_index_cannot_resolve_quotes(self):
        for field in ("content", "sentence_index"):
            doc = source()
            if field == "content":
                doc[field] += "变化"
            else:
                doc[field][0]["end"] -= 1
            self.assertEqual(resolve_sentences(doc, doc["document_id"], ["s000001"]), [])

    def test_quote_cannot_bridge_unselected_sentence(self):
        doc = source()
        quote = "项目于2023年投运。\n服务人口为6000人。"
        self.assertTrue(quote_matches_sentences(doc, doc["document_id"], ["s000001", "s000002"], quote))
        self.assertFalse(quote_matches_sentences(doc, doc["document_id"], ["s000001", "s000003"], quote))
        self.assertFalse(quote_matches_sentences(doc, doc["document_id"], ["s000001"], "项目2023年投运"))

    def test_long_document_selection_reaches_relevant_tail(self):
        text = ("无关背景。\n" * 1000) + "项目2023年投运，服务人口6000人。"
        doc = source(text)
        item = {**doc, "full_content": text}
        packet = format_numbered_body(item, "服务人口6000", max_chars=200)
        self.assertIn("服务人口6000人", packet)
        self.assertLessEqual(len(packet), 400)
        self.assertIn(doc["sentence_index"][-1]["id"], item["visible_sentence_ids"])


class FactBindingTests(unittest.TestCase):
    def ingest(self, state, item, raw):
        return scout()._ingest_search_analysis(state, {"extracted_facts": [raw]}, [item], related_sections=["sec1"])

    def test_fact_and_data_point_require_valid_numbered_evidence(self):
        state, item, raw = fixture()
        raw["data_points"] = [{"name": "投运年份", "value": 2023}]
        self.assertEqual(self.ingest(state, item, raw)["facts_added"], 1)
        fact = state["facts"][0]
        self.assertEqual(fact["document_id"], item["document_id"])
        self.assertEqual(fact["sentence_ids"], ["s000001"])
        self.assertEqual(fact["evidence_status"], "located_not_semantically_verified")
        self.assertEqual(bind_verified_evidence(state)["facts_kept"], 1)
        self.assertEqual(len(state["data_points"]), 1)

    def test_missing_wrong_document_or_unknown_sentence_is_rejected(self):
        for changes in ({"sentence_ids": []}, {"document_id": "invented"}, {"sentence_ids": ["s999999"]}, {"sentence_ids": ["s000001", "s000001"]}):
            state, item, raw = fixture()
            raw.update(changes)
            self.assertEqual(self.ingest(state, item, raw)["facts_added"], 0)

    def test_existing_but_unshown_sentence_is_rejected(self):
        state, item, raw = fixture()
        item["visible_sentence_ids"] = ["s000002"]
        self.assertEqual(self.ingest(state, item, raw)["facts_added"], 0)

    def test_rejected_source_cannot_admit_facts_even_if_reader_flag_is_true(self):
        state, item, raw = fixture()
        state["raw_sources"][0]["body_quality"]["status"] = "rejected"
        self.assertEqual(self.ingest(state, item, raw)["facts_added"], 0)

    def test_legacy_polluted_checkpoint_cannot_pass_source_gate(self):
        state = {"raw_sources": [{"id": "old", "url": URL, "content": LIST}],
                 "facts": [{"id": "f1", "source_id": "old", "content": "旧事实"}]}
        self.assertEqual(bind_verified_evidence(state)["facts_kept"], 0)

    def test_checkpoint_roundtrip_preserves_binding_and_dedup(self):
        state, item, raw = fixture()
        self.ingest(state, item, raw)
        normalize_evidence_state(state)
        state = json.loads(json.dumps(state))
        self.assertEqual(self.ingest(state, item, raw)["facts_reused"], 1)
        packet = json.loads(format_claim_evidence(state))
        self.assertEqual(packet["documents"][0]["original_sentences"][0]["text"], "项目于2023年投运。")

    def test_writer_receives_distinct_fact_supports_not_source_prefix(self):
        text = "无关导言。\n" * 300 + BODY
        doc = source(text)
        state = {"raw_sources": [doc], "facts": [
            {"id": "f1", "source_id": "source_1", "document_id": doc["document_id"], "sentence_ids": ["s000301"], "content": "2023年投运"},
            {"id": "f2", "source_id": "source_1", "document_id": doc["document_id"], "sentence_ids": ["s000302"], "content": "服务6000人"},
        ]}
        packet = json.loads(format_claim_evidence(state))
        self.assertEqual(len(packet["facts"]), 2)
        self.assertEqual(packet["facts"][0]["sentence_ids"], ["s000301"])
        self.assertEqual(packet["facts"][1]["sentence_ids"], ["s000302"])
        self.assertEqual([s["text"] for s in packet["documents"][0]["original_sentences"]],
                         ["项目于2023年投运。", "服务人口为6000人。"])
        self.assertNotIn("无关导言", json.dumps(packet, ensure_ascii=False))
        self.assertEqual(build_evidence_view(state)[0]["original_sentences"][0]["start"], text.index("项目"))

    def test_nonadjacent_supports_do_not_turn_intervening_text_into_quote(self):
        doc = source()
        state = {"raw_sources": [doc], "facts": [{"id": "f1", "source_id": "source_1",
                 "document_id": doc["document_id"], "sentence_ids": ["s000001", "s000003"], "content": "投运及范围"}]}
        packet = json.loads(format_claim_evidence(state))["documents"][0]
        self.assertEqual(len(packet["original_sentences"]), 2)
        self.assertNotIn("6000", json.dumps(packet, ensure_ascii=False))

    def test_fact_id_from_another_source_cannot_authorize_data_point(self):
        state, item, raw = fixture()
        self.ingest(state, item, raw)
        other = source(url=URL + "2", ident="source_2")
        state["raw_sources"].append(other)
        state["data_points"] = [{"id": "dp1", "source_id": "source_2", "fact_id": state["facts"][0]["id"], "name": "年份", "value": 2023}]
        bind_verified_evidence(state)
        self.assertEqual(state["data_points"], [])


class QualityModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_writer_only_receives_admitted_facts_and_their_ids(self):
        state = create_initial_state("项目运营", "test", ["example.gov.cn"])
        good = source()
        bad = source("污染页面内容。", URL + "2", "source_2")
        bad["body_quality"]["status"] = "rejected"
        state["raw_sources"] = [good, bad]
        state["facts"] = [{"id": "f_good", "content": "已准入的事实", "source_id": good["id"],
                           "document_id": good["document_id"], "sentence_ids": ["s000001"], "related_sections": ["sec1"]},
                          {"id": "f_bad", "content": "不应出现在Writer输入中的事实", "source_id": bad["id"],
                           "document_id": bad["document_id"], "sentence_ids": ["s000001"], "related_sections": ["sec1"]}]
        writer = LeadWriter.__new__(LeadWriter)
        writer.name = "LeadWriter"
        writer.logger = logging.getLogger("test.numbered-writer")
        writer.add_message = lambda *args, **kwargs: None
        writer.parse_json_response = json.loads
        writer.call_llm = AsyncMock(return_value=json.dumps({"content": "章节正文。", "claims": [], "key_points": [], "citations": []}))
        await writer._write_section(state, {"id": "sec1", "title": "运营", "status": "pending"})
        prompt = writer.call_llm.call_args.kwargs["user_prompt"]
        self.assertNotIn("### 相关事实", prompt)
        self.assertIn('"evidence_id":"f_good"', prompt)
        self.assertIn("项目于2023年投运", prompt)
        self.assertNotIn("不应出现在Writer输入中的事实", prompt)

    async def test_enrich_ingest_writer_roundtrip_and_rejected_page_diagnostics(self):
        agent = scout()
        agent.call_llm = AsyncMock()
        agent.web_reader = object()
        agent._ensure_tool_runtime = lambda: None
        agent.tool_definitions = {"web_read": object()}
        good = WebReadResult(url=URL, final_url=URL, title="项目报告", content=BODY, fetch_status="success", fetched_at=1000)
        bad_url = URL + "-bad"
        bad = WebReadResult(url=bad_url, final_url=bad_url, content=LIST, fetch_status="success", fetched_at=1000)
        agent.tool_runtime = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(value=[good.to_dict(), bad.to_dict()])))
        state = {"allowed_domains": ["example.gov.cn"], "raw_sources": [], "facts": [], "data_points": [], "hypotheses": []}
        items = [{"url": URL}, {"url": bad_url}]
        await agent._enrich_results_with_full_text(state, items, section_id="sec1")
        self.assertEqual(len(state["raw_sources"]), 1)
        self.assertTrue(items[0]["reader_verified"])
        self.assertFalse(items[1]["reader_verified"])
        self.assertEqual(state["web_reader_attempts"][1]["failure_category"], "body_quality_rejected")
        format_numbered_body(items[0], "投运")
        raw = {"fact_id": "f1", "content": "2023年投运", "source_url": URL,
               "document_id": items[0]["document_id"], "sentence_ids": ["s000001"]}
        agent._ingest_search_analysis(state, {"extracted_facts": [raw]}, items, related_sections=["sec1"])
        bind_verified_evidence(state)
        saved = json.loads(json.dumps(state))
        self.assertEqual(json.loads(format_claim_evidence(saved))["documents"][0]["original_sentences"][0]["text"], "项目于2023年投运。")
        agent.call_llm.assert_not_awaited()

    async def test_browser_http_recovery_cannot_bypass_quality_gate(self):
        import httpx
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"WEB_READER_CACHE_DIR": folder, "WEB_READER_MIN_CHARS": "20"}):
            reader = WebReaderService()
            request = httpx.Request("GET", URL)
            error = httpx.HTTPStatusError("403", request=request, response=httpx.Response(403, request=request))
            recovered = WebReadResult(url=URL, final_url=URL, content=LIST, fetch_status="success")
            with patch.object(reader, "_download_with_retry", AsyncMock(side_effect=error)), patch.object(reader, "_read_with_crawl4ai", AsyncMock(return_value=recovered)):
                result = await reader._read_uncached(URL, "目标报告")
        self.assertFalse(result.success)
        self.assertEqual(result.failure_category, "body_quality_rejected")

    async def test_clear_recommendation_list_needs_no_model(self):
        agent = scout()
        agent.call_llm = AsyncMock()
        result = WebReadResult(url=URL, final_url=URL, content=LIST, fetch_status="success")
        await agent._check_source_body(result, {})
        self.assertFalse(result.success)
        agent.call_llm.assert_not_awaited()

    async def test_preview_model_selects_original_blocks_before_numbering(self):
        agent = scout()
        agent.call_llm = AsyncMock(return_value=json.dumps({"status": "partial", "reason": "只有预览正文", "body_block_ids": ["b000002"]}))
        result = WebReadResult(url=URL, final_url=URL, content="仅供预览\n项目于2023年投运。\n登录后查看", fetch_status="success")
        await agent._check_source_body(result, {})
        self.assertTrue(result.success)
        self.assertEqual(result.content, "项目于2023年投运。")
        self.assertEqual(result.body_quality["status"], "partial")
        agent.call_llm.assert_awaited_once()

    async def test_invalid_model_block_or_failed_model_leaves_page_unverified(self):
        for response in ('{"status":"passed","body_block_ids":["invented"]}', 'invalid-json'):
            agent = scout()
            agent.call_llm = AsyncMock(return_value=response)
            result = WebReadResult(url=URL, final_url=URL, content="仅供预览\n" + BODY, fetch_status="success")
            await agent._check_source_body(result, {})
            self.assertTrue(result.success)
            self.assertEqual(result.body_quality["status"], "uncertain")
            self.assertIsNone(result.failure_category)
            self.assertEqual(result.body_quality["model_review"]["error_category"], "invalid_response")
            self.assertIn("quality_review_failed", result.body_quality["reasons"])

    async def test_model_cannot_approve_a_selected_login_prompt(self):
        agent = scout()
        agent.call_llm = AsyncMock(return_value=json.dumps({"status": "partial", "body_block_ids": ["b000002"]}))
        result = WebReadResult(url=URL, final_url=URL, content="仅供预览\n登录后查看", fetch_status="success")
        await agent._check_source_body(result, {})
        self.assertFalse(result.success)

    async def test_legitimate_duplicate_requires_review_not_automatic_rejection(self):
        agent = scout()
        agent.call_llm = AsyncMock(return_value=json.dumps({"status": "passed", "reason": "转载同一报告", "body_block_ids": ["b000001", "b000002", "b000003"]}))
        result = WebReadResult(url=URL + "2", final_url=URL + "2", content=BODY, fetch_status="success")
        await agent._check_source_body(result, {"raw_sources": [source()]})
        self.assertTrue(result.success)
        agent.call_llm.assert_awaited_once()

    async def test_all_scout_prompts_use_numbered_body_and_request_ids(self):
        agent = scout()
        agent.call_llm = AsyncMock(return_value='{"extracted_facts":[]}')
        _, item, _ = fixture()
        await agent._analyze_search_results("项目", {"title": "运营"}, [copy.deepcopy(item)], [])
        await agent._analyze_deep_search_results("项目", "运营", [copy.deepcopy(item)], "source_tracing", [])
        await agent._analyze_supplementary_results("项目", "运营", [copy.deepcopy(item)], [])
        for call in agent.call_llm.call_args_list:
            prompt = call.kwargs["user_prompt"]
            self.assertIn(item["document_id"], prompt)
            self.assertIn("[s000001] 项目于2023年投运。", prompt)
            self.assertIn('"sentence_ids"', prompt)


if __name__ == "__main__":
    unittest.main()
