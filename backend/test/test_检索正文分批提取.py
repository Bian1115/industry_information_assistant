"""真实本地 Qwen tokenizer + 模拟模型，覆盖超长正文、完整请求预算和编号合并。"""

import copy
import json
import logging
import os
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "app"))
sys.path.insert(0, str(BACKEND / "test"))

from service.agent_context import (
    ContextBudgetExceeded, evidence_batch_input, is_evidence_batch_input,
    prepare_agent_messages, protected_agent_messages,
)
from service.agent_history import HistoryScope, bind_agent_history
from service.context_policy import ContextPolicy, policy_for_scope
from service.qwen_token_counter import QwenTokenCounter
from service.web_document import index_document, sentence_records
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.evidence_batches import (
    EvidenceBatch, analyze_evidence_batches, merge_batch_analysis, plan_evidence_batches,
)
from test_智能体独立历史 import MemoryStore


def sources(count=2, sentences=100):
    result = []
    for number in range(count):
        url = f"https://example.gov.cn/report/{number}"
        body = "\n".join(f"第{i}项指标属于第{number}份报告，统计地区为城区，统计期为2024年，服务人口达到6000人。"
                         for i in range(sentences))
        result.append({"url": url, "title": f"官方报告{number}", "site_name": "官方",
                       "body_quality": {"status": "passed"}, "full_content": body,
                       "content": body, "reader_verified": True, **index_document(body, url)})
    return result


def agent():
    value = DeepScout.__new__(DeepScout)
    value.name, value.logger = "DeepScout", logging.getLogger("test.evidence-batches")
    value.parse_json_response = json.loads
    value.call_llm = AsyncMock(return_value='{"extracted_facts": []}')
    return value


class EvidenceBatchTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = QwenTokenCounter(model_name=str(BACKEND / ".cache/qwen-tokenizer"))
        cls.counter.count_text("仅加载本地 tokenizer")

    def setUp(self):
        self.patches = [
            patch("service.agent_context.get_qwen_token_counter", return_value=self.counter),
            patch("service.deep_research_v2.evidence_batches.get_qwen_token_counter", return_value=self.counter),
            patch("service.qwen_token_counter.get_qwen_token_counter", return_value=self.counter),
            patch.dict(os.environ, {"AGENT_CONTEXT_MAX_INPUT": "2600", "AGENT_CONTEXT_WINDOW": "32768",
                                    "AGENT_CONTEXT_SAFETY_TOKENS": "1024"}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    async def test_all_three_routes_cover_all_sentences_and_obey_budget(self):
        for route in ("initial", "deep", "supplementary"):
            for enabled in (True, False):
                with self.subTest(route=route, hypotheses=enabled):
                    value, docs = agent(), sources()
                    before = copy.deepcopy(docs)
                    if route == "initial":
                        await value._analyze_search_results("研究问题", {"title": "项目指标"}, docs,
                                                           [{"id": "h1", "content": "人口增长"}], enabled)
                    elif route == "deep":
                        await value._analyze_deep_search_results("研究问题", "指标", docs, "source_tracing",
                                                                [{"id": "h1", "content": "人口增长"}], enabled)
                    else:
                        await value._analyze_supplementary_results("研究问题", "指标", docs,
                                                                  [{"id": "h1", "content": "人口增长"}], enabled)
                    self.assertGreater(value.call_llm.await_count, 1)
                    seen = []
                    for call in value.call_llm.call_args_list:
                        kwargs = call.kwargs
                        self.assertLessEqual(self.counter.count_messages(protected_agent_messages(
                            None, kwargs["system_prompt"], kwargs["user_prompt"])), 2600)
                        material = kwargs["user_prompt"].split("## 搜索结果\n", 1)[1].split("## 任务", 1)[0]
                        seen.extend(re.findall(r"\[(s\d+)\]", material))
                    self.assertGreaterEqual(len(seen), 200)
                    self.assertEqual(set(seen), {f"s{i:06d}" for i in range(1, 101)})
                    for doc, original in zip(docs, before):
                        self.assertEqual(doc["full_content"], original["full_content"])
                        self.assertEqual(len(doc["visible_sentence_ids"]), 100)
                        self.assertEqual(sentence_records(doc), sentence_records(original))

    async def test_single_small_body_keeps_one_call(self):
        value, docs = agent(), sources(1, 2)
        await value._analyze_search_results("问题", {"title": "项目"}, docs)
        value.call_llm.assert_awaited_once()

    async def test_numbering_and_body_beyond_old_12000_char_limit_are_retained(self):
        docs, value = sources(1, 400), agent()
        self.assertGreater(len(docs[0]["content"]), 12000)
        await analyze_evidence_batches(value, docs, lambda materials: "任务\n" + materials)
        prompts = "\n".join(call.kwargs["user_prompt"] for call in value.call_llm.call_args_list)
        self.assertIn("[s000400]", prompts)
        self.assertEqual(len(docs[0]["visible_sentence_ids"]), 400)

    async def test_frozen_contract_is_counted_and_cannot_be_removed(self):
        scope = HistoryScope(MemoryStore(), "test", "session", None, contract={"query": "完整冻结研究问题" * 120})
        with bind_agent_history(scope):
            batches, budget = plan_evidence_batches(agent(), sources(), lambda materials: "任务\n" + materials)
        self.assertGreater(len(batches), 1)
        for batch in batches:
            messages = protected_agent_messages(scope, "", batch.prompt)
            self.assertIn("完整冻结研究问题", messages[0]["content"])
            self.assertEqual(batch.tokens, self.counter.count_messages(messages))
            self.assertLessEqual(batch.tokens, budget)

    async def test_frozen_policy_budget_wins_over_environment(self):
        scope = HistoryScope(MemoryStore(), "test", "session", None)
        config = policy_for_scope(scope).config_payload()
        config["stage_budgets"]["DeepScout"]["max_input"] = 1800
        snapshot = ContextPolicy.model_validate(config).snapshot()
        scope = replace(scope, context_policy=snapshot)
        with bind_agent_history(scope):
            batches, budget = plan_evidence_batches(agent(), sources(), lambda text: text)
        self.assertEqual(budget, 1800)
        self.assertTrue(all(batch.tokens <= 1800 for batch in batches))

    async def test_overlarge_fixed_instructions_fail_before_any_llm_call(self):
        value = agent()
        with self.assertRaises(ContextBudgetExceeded):
            await analyze_evidence_batches(value, sources(), lambda text: "固定研究要求" * 3000 + text)
        value.call_llm.assert_not_awaited()

    async def test_overlarge_single_sentence_is_not_silently_truncated(self):
        value = agent()
        docs = sources(1, 1)
        docs[0]["content"] = docs[0]["full_content"] = "超长无标点原句" * 3000
        docs[0].update(index_document(docs[0]["content"], docs[0]["url"]))
        with self.assertRaises(ContextBudgetExceeded):
            await analyze_evidence_batches(value, docs, lambda text: text)
        value.call_llm.assert_not_awaited()

    async def test_failed_middle_batch_does_not_return_partial_success(self):
        value, docs = agent(), sources()
        value.call_llm.side_effect = ['{"extracted_facts": []}', RuntimeError("模拟第二批失败")]
        with self.assertRaisesRegex(RuntimeError, "第二批失败"):
            await analyze_evidence_batches(value, docs, lambda text: text)
        self.assertEqual(value.call_llm.await_count, 2)
        self.assertTrue(all(doc["visible_sentence_ids"] == [] for doc in docs))
        self.assertFalse(is_evidence_batch_input())

    async def test_invalid_json_schema_cannot_count_as_a_successful_empty_batch(self):
        value = agent()
        value.call_llm.return_value = '{}'
        with self.assertRaisesRegex(ValueError, "extracted_facts"):
            await analyze_evidence_batches(value, sources(1, 2), lambda text: text)

    async def test_quality_rejected_or_tampered_documents_are_never_shown(self):
        docs = sources()
        docs[0]["body_quality"]["status"] = "rejected"
        docs[1]["full_content"] += "缓存已被篡改"
        value = agent()
        result = await analyze_evidence_batches(value, docs, lambda text: text)
        self.assertEqual(result["extracted_facts"], [])
        value.call_llm.assert_not_awaited()

    async def test_huge_history_is_saved_but_not_replayed_into_extraction(self):
        store = MemoryStore()
        scope = HistoryScope(store, "test", "session", None, contract={"query": "任务保持"})
        await store.append(scope, "DeepScout", "tool_output", {"content": "历史正文" * 100000}, "old")
        value = agent()
        # 使用真实 BaseAgent.call_llm，只有最底层付费模型调用被替换。
        value.call_llm = DeepScout.call_llm.__get__(value)
        sent = []
        async def model(*args, **kwargs):
            messages = kwargs["prepared_messages"]
            self.assertLessEqual(self.counter.count_messages(messages), 2600)
            self.assertEqual(len(messages), 2)
            self.assertNotIn("历史正文", messages[0]["content"] + messages[1]["content"])
            sent.append(messages)
            return '{"extracted_facts": []}'
        value._call_llm_impl = AsyncMock(side_effect=model)
        with bind_agent_history(scope):
            await analyze_evidence_batches(value, sources(), lambda text: "任务\n" + text)
        saved = await store.load(scope, "DeepScout")
        self.assertEqual(saved["messages"][0]["payload"]["content"], "历史正文" * 100000)
        self.assertEqual(sum(row["kind"] == "llm_input" for row in saved["messages"]), len(sent))
        contexts = [row["payload"] for row in saved["messages"] if row["kind"] == "context_view"]
        self.assertTrue(all(row["input_mode"] == "evidence_batch" and not row["history_replay"] for row in contexts))
        self.assertFalse(is_evidence_batch_input())

    async def test_ordinary_scout_calls_keep_existing_history_replay(self):
        store = MemoryStore()
        scope = HistoryScope(store, "test", "session", None)
        await store.append(scope, "DeepScout", "llm_output", {"content": "已有研究结论"}, "old")
        with bind_agent_history(scope):
            messages = await prepare_agent_messages(agent(), scope, "系统", "后续计划", "new", 6000)
        self.assertEqual(len(messages), 3)
        self.assertIn("已有研究结论", messages[1]["content"])

    async def test_independent_batches_reject_unplanned_history_read(self):
        value, store = agent(), MemoryStore()
        value.call_llm = DeepScout.call_llm.__get__(value)
        value._call_llm_impl = AsyncMock(return_value='{"_history_read":{"source_id":"old"}}')
        with bind_agent_history(HistoryScope(store, "test", "session", None)), evidence_batch_input():
            with self.assertRaisesRegex(ValueError, "不允许追加历史回读"):
                await value.call_llm("系统", "本批正文")
        self.assertEqual(value._call_llm_impl.await_count, 1)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.merged, self.keys = {"extracted_facts": []}, {}
        self.batch1 = EvidenceBatch("", 100, {("https://a", "doc_a"): {"s000001"}})
        self.batch2 = EvidenceBatch("", 100, {("https://a", "doc_a"): {"s000002"}})

    def analysis(self, sentence, content="同一事实"):
        return {"extracted_facts": [{"fact_id": "f_1", "content": content, "source_url": "https://a",
                                    "document_id": "doc_a", "sentence_ids": [sentence]}],
                "hypothesis_evidence": [{"fact_id": "f_1", "hypothesis_id": "h1", "evidence_type": "supports"}]}

    def test_duplicate_facts_merge_support_and_relations(self):
        merge_batch_analysis(self.merged, self.analysis("s000001"), self.batch1, self.keys)
        merge_batch_analysis(self.merged, self.analysis("s000002"), self.batch2, self.keys)
        self.assertEqual(len(self.merged["extracted_facts"]), 1)
        self.assertEqual(self.merged["extracted_facts"][0]["fact_id"], "f_0")
        self.assertEqual(self.merged["extracted_facts"][0]["sentence_ids"], ["s000001", "s000002"])
        self.assertEqual(len(self.merged["hypothesis_evidence"]), 1)

    def test_repeated_local_fact_id_across_batches_is_not_a_collision(self):
        merge_batch_analysis(self.merged, self.analysis("s000001", "事实甲"), self.batch1, self.keys)
        merge_batch_analysis(self.merged, self.analysis("s000002", "事实乙"), self.batch2, self.keys)
        ids = [fact["fact_id"] for fact in self.merged["extracted_facts"]]
        self.assertEqual(ids, ["f_0", "f_1"])
        self.assertEqual([row["fact_id"] for row in self.merged["hypothesis_evidence"]], ids)

    def test_deduplicated_fact_keeps_id_and_does_not_leave_numbering_gaps(self):
        merge_batch_analysis(self.merged, self.analysis("s000001"), self.batch1, self.keys)
        merge_batch_analysis(self.merged, self.analysis("s000002"), self.batch2, self.keys)
        merge_batch_analysis(self.merged, self.analysis("s000002", "新的事实"), self.batch2, self.keys)
        self.assertEqual([fact["fact_id"] for fact in self.merged["extracted_facts"]], ["f_0", "f_1"])
        self.assertEqual([relation["fact_id"] for relation in self.merged["hypothesis_evidence"]], ["f_0", "f_1"])

    def test_wrong_batch_sentence_or_url_cannot_be_merged(self):
        self.assertEqual(merge_batch_analysis(self.merged, self.analysis("s000002"), self.batch1, self.keys), 1)
        wrong = self.analysis("s000001")
        wrong["extracted_facts"][0]["source_url"] = "https://b"
        self.assertEqual(merge_batch_analysis(self.merged, wrong, self.batch1, self.keys), 1)
        self.assertEqual(self.merged["extracted_facts"], [])
        self.assertFalse(self.merged.get("hypothesis_evidence"))

    def test_ambiguous_duplicate_id_inside_same_batch_has_no_relation_mapping(self):
        analysis = self.analysis("s000001")
        analysis["extracted_facts"].append({**analysis["extracted_facts"][0], "content": "另一事实"})
        merge_batch_analysis(self.merged, analysis, self.batch1, self.keys)
        self.assertEqual(len(self.merged["extracted_facts"]), 2)
        self.assertEqual(len({fact["fact_id"] for fact in self.merged["extracted_facts"]}), 2)
        self.assertFalse(self.merged.get("hypothesis_evidence"))


if __name__ == "__main__":
    unittest.main()
