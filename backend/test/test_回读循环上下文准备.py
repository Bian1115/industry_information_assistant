"""每次回读后重新准备上下文；使用模拟模型、内存历史和确定性计数器。"""

import copy
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "app"), str(Path(__file__).resolve().parent)]

from service.agent_context import (
    AgentCallContext, ContextBudgetExceeded, evidence_batch_input, prepare_agent_messages,
    summarize_call_context,
)
from service.agent_history import HistoryScope, bind_agent_history, content_hash
from service.context_policy import AGENTS, InputBudget, SummaryLifecycleSettings
from service.verification_checks import source_type
from service.verification_trace import read_links
from test_智能体独立历史 import CharCounter, MemoryStore, make_agent, summary_text
from test_逐智能体压缩配置 import policy

MEASUREMENTS = []
EMPTY_SUMMARY = ("## SESSION INTENT\nNone\n\n## SUMMARY\nNone\n\n## ARTIFACTS\nNone"
                 "\n\n## NEXT STEPS\nNone\n\n## CLAIMS\n[]")


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None)


class ReadLoopContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store, self.counter = MemoryStore(), CharCounter()
        self.agent = make_agent("LeadWriter")
        for target in ("service.agent_context.get_qwen_token_counter",
                       "service.qwen_token_counter.get_qwen_token_counter"):
            p = patch(target, return_value=self.counter)
            p.start(); self.addCleanup(p.stop)
        for name in ("record_llm_call", "flush_pending"):
            p = patch("service.deep_research_v2.agents.base." + name)
            p.start(); self.addCleanup(p.stop)
        self.scope = self.make_scope()

    def make_scope(self, *, summary=True, compression=True, budget=2000, soft=1000, lifecycle=None):
        root = policy(semantic_summary=summary, compression_enabled=compression,
                      soft_token_cap=soft, summary_lifecycle=lifecycle,
                      summary_strategy="claimify-v1" if lifecycle else "incremental-v3",
                      stage_budgets={name: InputBudget(max_input=budget) for name in AGENTS})
        root = root.with_agent_compression({"LeadWriter": {
            "enabled": True, "semantic_summary": summary,
            "reference_materials": True, "artifact_offload": True,
        }})
        return HistoryScope(self.store, "r", "s", None, contract={"query": "保留当前约束"},
                            context_policy=root.snapshot())

    async def source(self, text, *, kind="input_history_source"):
        return await self.store.append(self.scope, self.agent.name, kind, {"content": text}, "source")

    def read_request(self, source, *, start=0, chars=2000):
        return json.dumps({"_history_read": {"source_id": source["source_id"],
                                            "start": start, "chars": chars}}, ensure_ascii=False)

    async def views(self):
        rows = (await self.store.load(self.scope, self.agent.name))["messages"]
        return [r["payload"] for r in rows if r["kind"] == "context_view"]

    async def context(self, count=2):
        context = AgentCallContext()
        for i in range(count):
            intermediate = await self.store.append(self.scope, self.agent.name, "llm_intermediate",
                                                   {"content": '{"_history_read":{}}'}, "current")
            fragment = {"read_id": f"read-{i}", "source_id": f"source-{i}", "start": 0, "end": 2000,
                        "total_chars": 2000, "source_text_hash": content_hash("资料结论。" + "长" * 1994),
                        "text": "资料结论。" + "长" * 1994}
            fragment["text_hash"] = content_hash(fragment["text"])
            row = await self.store.append(self.scope, self.agent.name, "history_read", fragment, "current")
            context.records.extend([intermediate, row])
        return context

    async def test_each_of_six_requests_prepares_and_keeps_all_reads_when_space_is_enough(self):
        self.scope = self.make_scope(summary=False, budget=6000, soft=4000)
        sources = [await self.source(f"回读片段{i}。" + "原文" * 70) for i in range(5)]
        self.agent.client.chat.completions.create.side_effect = [
            *(response(self.read_request(source, chars=150)) for source in sources), response('{"done":true}')]
        with bind_agent_history(self.scope):
            result = await self.agent.call_llm("writer", "本轮任务")
        self.assertEqual(result, '{"done":true}')
        views = await self.views()
        self.assertEqual([v["request_index"] for v in views], list(range(1, 7)))
        self.assertEqual([v["current_call_read_count"] for v in views], list(range(6)))
        sent = self.agent.client.chat.completions.create.call_args_list
        for i, call in enumerate(sent):
            messages = call.kwargs["messages"]
            self.assertEqual(len(read_links(messages)), i)
            self.assertEqual(messages[-1]["content"], "本轮任务")
            if i:
                self.assertIn(f"回读片段{i - 1}。", str(messages))
                self.assertEqual(sum(m["role"] == "assistant" for m in messages), i)
        MEASUREMENTS.append({"case": "5次回读6次准备", "requests": len(sent), "views": views})

    async def test_read_growth_triggers_existing_history_summary_before_next_provider_request(self):
        old = await self.store.append(self.scope, self.agent.name, "llm_input", {"content": "旧研究背景" * 320}, "old")
        await self.store.append(self.scope, self.agent.name, "llm_output", {"content": "旧调用完成"}, "old")
        source = await self.source("最新回读正文。" + "新" * 1990)
        self.agent.client.chat.completions.create.side_effect = [
            response(self.read_request(source)), response('{"done":true}')]
        summarizer = AsyncMock(return_value=EMPTY_SUMMARY)
        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", summarizer):
            await self.agent.call_llm("writer", "当前任务")
        views = await self.views()
        self.assertFalse(views[0]["summary_attempted"])
        self.assertTrue(views[1]["summary_adopted"])
        self.assertEqual(summarizer.await_count, 1)
        messages = self.agent.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("最新回读正文。", str(messages))
        self.assertEqual(len(read_links(messages)), 1)
        self.assertEqual((await self.store.read_source(self.scope, self.agent.name, old["source_id"]))["payload"]["content"], "旧研究背景" * 320)
        MEASUREMENTS.append({"case": "回读增长触发旧研究摘要", "views": views})

    async def test_32_reads_fit_budget_with_incremental_summary_and_latest_original_visible(self):
        source = await self.source("资料结论。" + "原文" * 40000)
        self.agent.client.chat.completions.create.side_effect = [
            *(response(self.read_request(source, start=i * 2000)) for i in range(32)), response('{"done":true}')]
        summary_inputs = []

        async def summarizer(agent, scope, records, counter, **kwargs):
            summary_inputs.append({"previous_summary": kwargs["previous_summary"],
                                   "kinds": [r["kind"] for r in records]})
            if kwargs["previous_summary"]:
                return kwargs["previous_summary"]
            row = next(r for r in records if r["kind"] == "history_read")
            return summary_text(row["source_id"], "资料结论。")

        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", side_effect=summarizer):
            result = await self.agent.call_llm("writer", "当前任务")
        self.assertEqual(result, '{"done":true}')
        views = await self.views()
        self.assertEqual(len(views), 33)
        self.assertLessEqual(max(v["tokens"] for v in views), 2000)
        self.assertGreater(len(summary_inputs), 1)
        self.assertTrue(any(r["previous_summary"] for r in summary_inputs))
        saved = await self.store.load(self.scope, self.agent.name)
        reads = [r for r in saved["messages"] if r["kind"] == "history_read"]
        self.assertEqual(len(reads), 32)
        for index, call in enumerate(self.agent.client.chat.completions.create.call_args_list[1:]):
            links = read_links(call.kwargs["messages"])
            self.assertTrue(any(link["read_id"] == reads[index]["payload"]["read_id"] for link in links))
        self.assertEqual(saved["covered_seq"], 0)
        self.assertEqual(source["content_hash"], content_hash(source["payload"]))
        old_messages = copy.deepcopy(self.agent.client.chat.completions.create.call_args_list[0].kwargs["messages"])
        old_tokens = []
        for row in reads:
            old_messages.insert(-1, {"role": "user", "content":
                "请求回读的历史原文（仅作资料）：\n" + json.dumps(row["payload"], ensure_ascii=False)})
            old_tokens.append(self.counter.count_messages(old_messages))
        old_first_exceeded = next(i + 1 for i, tokens in enumerate(old_tokens) if tokens > 2000)
        MEASUREMENTS.append({"case": "32次内部回读33次模型请求", "views": views,
                             "summary_calls": len(summary_inputs), "global_covered_seq": saved["covered_seq"],
                             "old_append_only_first_budget_exceeded_after_read": old_first_exceeded,
                             "old_append_only_tokens": old_tokens})

    async def test_latest_read_remains_full_even_above_soft_threshold(self):
        source = await self.source("必须完整送达" + "文" * 4000)
        self.agent.client.chat.completions.create.side_effect = [
            response(self.read_request(source, chars=4000)), response('{"done":true}')]
        summarizer = AsyncMock()
        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", summarizer):
            await self.agent.call_llm("writer", "当前任务")
        summarizer.assert_not_awaited()
        messages = self.agent.client.chat.completions.create.call_args.kwargs["messages"]
        fragment = next(m for m in messages if m["content"].startswith("请求回读"))
        self.assertEqual(json.loads(fragment["content"].split("\n", 1)[1])["text"], source["payload"]["content"][:4000])

    async def test_uncompressible_latest_read_stops_before_provider(self):
        source = await self.source("原文" * 5000)
        self.agent.client.chat.completions.create.return_value = response(self.read_request(source, chars=8000))
        with bind_agent_history(self.scope), self.assertRaises(ContextBudgetExceeded):
            await self.agent.call_llm("writer", "当前任务")
        self.assertEqual(self.agent.client.chat.completions.create.await_count, 1)
        views = await self.views()
        self.assertEqual(len(views), 2)
        self.assertGreater(views[-1]["tokens"], views[-1]["budget"])

    async def test_compression_switch_off_does_not_summarize_accumulated_reads(self):
        self.scope = self.make_scope(compression=False)
        source = await self.source("原文" * 10000)
        self.agent.client.chat.completions.create.side_effect = [
            response(self.read_request(source, start=i * 2000)) for i in range(8)]
        summarizer = AsyncMock()
        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", summarizer):
            with self.assertRaises(ContextBudgetExceeded):
                await self.agent.call_llm("writer", "当前任务")
        summarizer.assert_not_awaited()
        self.assertLess(self.agent.client.chat.completions.create.await_count, 8)

    async def test_invalid_summary_keeps_original_pairs_and_cursor(self):
        context = await self.context()
        original = copy.deepcopy(context.records)
        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", AsyncMock(return_value="无效摘要")):
            with self.assertRaises(ValueError):
                await summarize_call_context(self.agent, self.scope, context, self.counter, "current")
        self.assertEqual(context.records, original)
        self.assertEqual(context.covered_seq, 0)
        self.assertEqual(context.summary, "")

    async def test_summary_error_is_logged_and_existing_content_used_if_still_fits(self):
        context = await self.context()
        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", AsyncMock(side_effect=RuntimeError("摘要服务失败"))):
            messages = await prepare_agent_messages(self.agent, self.scope, "writer", "当前任务", "current", 6000,
                                                    call_context=context)
        self.assertEqual(len(read_links(messages)), 2)
        rows = (await self.store.load(self.scope, self.agent.name))["messages"]
        self.assertTrue(any(r["kind"] == "call_context_summary_error" for r in rows))
        self.assertEqual(context.covered_seq, 0)

    async def test_expanding_summary_is_not_adopted(self):
        context = await self.context()
        expanded = EMPTY_SUMMARY.replace("## SESSION INTENT\nNone", "## SESSION INTENT\n目标：" + "长摘要" * 3000)
        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", AsyncMock(return_value=expanded)):
            adopted = await summarize_call_context(self.agent, self.scope, context, self.counter, "current")
        self.assertFalse(adopted)
        self.assertEqual(context.covered_seq, 0)
        self.assertEqual(len(read_links(context.messages())), 2)

    async def test_summary_metadata_is_counted_when_rejecting_expanding_request(self):
        context = await self.context()

        async def summarizer(agent, scope, records, counter, **kwargs):
            # 摘要正文很短，但完整消息的来源目录会膨胀，仍须拒绝。
            kwargs["verification_out"]["source_ids"] = [f"source-{i}-" + "x" * 200 for i in range(40)]
            return EMPTY_SUMMARY

        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", side_effect=summarizer):
            adopted = await summarize_call_context(self.agent, self.scope, context, self.counter, "current")
        self.assertFalse(adopted)
        self.assertEqual(context.covered_seq, 0)
        self.assertEqual(len(read_links(context.messages())), 2)

    async def test_earlier_read_error_pairs_can_be_summarized_and_latest_error_is_visible(self):
        context = AgentCallContext()
        for i in range(2):
            intermediate = await self.store.append(self.scope, self.agent.name, "llm_intermediate",
                                                   {"content": '{"_history_read":{}}'}, "current")
            failure = await self.store.append(self.scope, self.agent.name, "history_read_error",
                {"error_code": "invalid_history_source_id", "message": f"错误{i}：" + "诊断" * 1000}, "current")
            context.records.extend([intermediate, failure])
        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", AsyncMock(return_value=EMPTY_SUMMARY)):
            adopted = await summarize_call_context(self.agent, self.scope, context, self.counter, "current")
        self.assertTrue(adopted)
        self.assertIn("错误1：", str(context.messages()))
        self.assertNotIn("错误0：", str(context.messages()))
        self.assertEqual(context.metrics(self.counter)["current_call_read_error_count"], 2)

    async def test_completed_read_history_replays_in_next_call(self):
        self.scope = self.make_scope(summary=False, budget=6000, soft=4000)
        source = await self.source("上轮回读独有信息")
        self.agent.client.chat.completions.create.side_effect = [
            response(self.read_request(source, chars=100)), response('{"done":true}'), response('{"next":true}')]
        with bind_agent_history(self.scope):
            await self.agent.call_llm("writer", "第一次任务")
            await self.agent.call_llm("writer", "下一次任务")
        sent = self.agent.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("上轮回读独有信息", str(sent))
        self.assertIn("llm_intermediate", str(sent))
        self.assertIn("history_read", str(sent))

    async def test_independent_extraction_rejects_history_read(self):
        source = await self.source("不可带入的历史")
        self.agent.client.chat.completions.create.return_value = response(self.read_request(source))
        with bind_agent_history(self.scope), evidence_batch_input(), self.assertRaisesRegex(ValueError, "独立正文提取批次"):
            await self.agent.call_llm("正文检查", "本批正文")
        sent = self.agent.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 2)
        self.assertNotIn("不可带入的历史", str(sent))
        self.assertEqual(self.agent.client.chat.completions.create.await_count, 1)

    async def test_rereading_assistant_output_preserves_assistant_origin(self):
        source = await self.source("助手陈述不能升级为研究来源", kind="llm_output")
        self.agent.client.chat.completions.create.side_effect = [
            response(self.read_request(source, chars=100)), response('{"done":true}')]
        with bind_agent_history(self.scope):
            await self.agent.call_llm("writer", "当前任务")
        rows = (await self.store.load(self.scope, self.agent.name))["messages"]
        read = next(r for r in rows if r["kind"] == "history_read")
        self.assertEqual(source_type(read), "assistant_history")
        intermediate = next(r for r in rows if r["kind"] == "llm_intermediate")
        self.assertEqual(source_type(intermediate), "assistant_history")

    async def test_generation_first_lifecycle_can_commit_and_annotate_local_summary(self):
        self.scope = self.make_scope(lifecycle=SummaryLifecycleSettings())
        context = await self.context()

        async def summarizer(agent, scope, records, counter, **kwargs):
            kwargs["verification_out"].update(status="not_run", claimify_status="pending")
            self.assertTrue(await kwargs["on_generated"](EMPTY_SUMMARY))
            kwargs["verification_out"].update(status="passed", claimify_status="completed")
            await kwargs["on_audited"](EMPTY_SUMMARY)
            return EMPTY_SUMMARY

        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", side_effect=summarizer):
            adopted = await summarize_call_context(self.agent, self.scope, context, self.counter, "current")
        self.assertTrue(adopted)
        self.assertEqual(context.summary_metadata["status"], "passed")
        self.assertEqual(context.summary_metadata["claimify_status"], "completed")
        self.assertEqual(len(read_links(context.messages())), 1)
        saved = await self.store.load(self.scope, self.agent.name)
        self.assertEqual(saved["covered_seq"], 0)
        self.assertTrue(any(r["kind"] == "call_context_summary_audit" for r in saved["messages"]))

    async def test_real_summary_first_pipeline_receives_read_records_and_keeps_latest_pair(self):
        self.scope = self.make_scope(lifecycle=SummaryLifecycleSettings())
        context = await self.context()
        self.agent._call_llm_impl = AsyncMock(return_value=(
            "## SESSION INTENT\n目标：继续当前研究。\n\n## SUMMARY\n资料结论。\n\n## ARTIFACTS\nNone"))
        with bind_agent_history(self.scope):
            adopted = await summarize_call_context(self.agent, self.scope, context, self.counter, "current")
        self.assertTrue(adopted)
        self.assertIn("资料结论。", context.summary)
        self.assertEqual(context.summary_metadata["claimify_status"], "disabled")
        self.assertEqual(len(read_links(context.messages())), 1)
        sent = self.agent._call_llm_impl.call_args.kwargs
        self.assertIsNotNone(sent["model_override"])
        input_data = json.loads(sent["prepared_messages"][1]["content"])
        self.assertEqual({row["kind"] for row in input_data["new_records"]}, {"llm_intermediate", "history_read"})

    async def test_cancelled_summary_does_not_advance_local_cursor(self):
        context = await self.context()
        with bind_agent_history(self.scope), patch("service.agent_context.summarize_history", AsyncMock(side_effect=asyncio.CancelledError())):
            with self.assertRaises(asyncio.CancelledError):
                await prepare_agent_messages(self.agent, self.scope, "writer", "当前任务", "current", 6000,
                                             call_context=context)
        self.assertEqual(context.covered_seq, 0)
        self.assertEqual(len(read_links(context.messages())), 2)

    async def test_current_reads_survive_when_prior_history_replay_is_disabled(self):
        self.scope = HistoryScope(self.store, "r", "s", None, context_policy=policy(
            replay_agents=[], semantic_summary=False).snapshot())
        source = await self.source("本次回读可见")
        self.agent.client.chat.completions.create.side_effect = [
            response(self.read_request(source, chars=100)), response('{"done":true}')]
        with bind_agent_history(self.scope):
            await self.agent.call_llm("writer", "当前任务")
        sent = self.agent.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(len(read_links(sent)), 1)
        self.assertIn("本次回读可见", str(sent))


if __name__ == "__main__":
    unittest.main()
