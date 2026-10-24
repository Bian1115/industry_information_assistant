"""多轮改写结构、输入隔离、运行边界及压缩观测回归；不调用真实模型。"""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND/'app'), str(BACKEND/'test')]
from eval.multiturn import Task, load_tasks, research_context, run_dialogue, compression_points, first_summary_replay


def fixture():
    return {"schema_version":1, "task_id":"drb-mt-025", "split":"development",
            "source":{"benchmark":"DeepResearch Bench", "id":25,"topic":"Health","language":"zh",
                      "prompt":"原题不直接传模型", "prompt_sha256":hashlib.sha256("原题不直接传模型".encode()).hexdigest(),
                      "revision":"a"*40,"url":"https://example.org","file_sha256":"b"*64},
            "as_of":"2026-09-27","rewrite_notes":"机制样例",
            "turns":[{"turn_id":1,"user_query":"先研究甲","depends_on":[],"objective":"隐藏评价意图"},
                     {"turn_id":2,"user_query":"继续比较乙","depends_on":[1],"objective":"后续测试"}],
            "probes":[{"probe_id":"隐含真值","introduced_turn":1,"check_turns":[2],"kind":"constraint",
                       "criterion":"不得提供给生成模型","annotation_status":"user_constraint"}]}


class SchemaTests(unittest.TestCase):
    def test_thirty_and_longer_turns_have_no_fixed_cap(self):
        for count in (30, 120):
            data = fixture()
            data['turns'] = [{'turn_id': i, 'user_query': f'第{i}轮', 'depends_on': [i-1] if i > 1 else [],
                              'objective': '结构回归'} for i in range(1, count + 1)]
            self.assertEqual(len(Task.model_validate(data).turns), count)

    def test_bad_hash_future_dependency_and_unknown_field_rejected(self):
        for mutate in [lambda d:d["source"].update(prompt="篡改"),
                       lambda d:d["turns"][0].update(depends_on=[2]),
                       lambda d:d.update(answer="泄漏"),
                       lambda d:d["probes"][0].update(check_turns=[3])]:
            data = fixture(); mutate(data)
            with self.assertRaises(ValueError):
                Task.model_validate(data)

    def test_original_id_cannot_cross_splits(self):
        a,b = fixture(),fixture()
        b.update(task_id="drb-mt-099",split="test")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"数据.jsonl"
            path.write_text(json.dumps(a)+"\n"+json.dumps(b))
            with self.assertRaisesRegex(ValueError,"跨集合"):
                load_tasks([path])

    def test_context_is_independent_copy_and_contains_only_past_dialogue(self):
        history=[{"role":"user","content":"先前问题"}]
        context=research_context(history,"新问题")
        context["conversation_context"][0]["content"]="修改"
        self.assertEqual(history[0]["content"],"先前问题")
        self.assertNotIn("probes",context)

    def test_old_telemetry_is_unknown_not_false(self):
        row={"agent_name":"DeepScout","context_key":"main","call_id":"c","seq":1,"kind":"context_view","payload":{"tokens":50}}
        self.assertIsNone(compression_points([row])[0]["summary_adopted"])

    def test_snapshot_contains_no_future_records_or_labels(self):
        common={"agent_name":"DeepScout","context_key":"main","call_id":"c"}
        rows=[{**common,"kind":"llm_output","seq":1,"payload":{"content":"先前资料"}},
              {**common,"kind":"llm_input","seq":2,"payload":{"system":"sys","content":"next","contract":{}}},
              {**common,"kind":"context_view","seq":3,"payload":{"summary_attempted":True,"summary_version_before":0,"summary_adopted":True}},
              {**common,"kind":"llm_request","seq":4,"payload":{"request":{"model":"frozen-model"}}},
              {**common,"kind":"llm_output","seq":5,"payload":{"content":"未来回答"}}]
        result=first_summary_replay(rows,"checkpoint")["tasks"][0]
        self.assertEqual([r["seq"] for r in result["records"]],[1])
        self.assertEqual(result["next_task"],"next")
        self.assertTrue(compression_points(rows)[0]["summary_used_in_request"])

    def test_incomplete_cross_run_history_cannot_be_exported_as_full_replay(self):
        common={'agent_name':'ChiefArchitect','context_key':'main','call_id':'c','run_id':'run2'}
        rows=[{**common,'kind':'llm_input','seq':5,'payload':{'system':'sys','content':'next'}},
              {**common,'kind':'context_view','seq':6,'payload':{'summary_attempted':True,
                'summary_version_before':0,'history_run_ids':['run1']}},
              {**common,'kind':'llm_request','seq':7,'payload':{'request':{'model':'test'}}}]
        self.assertIsNone(first_summary_replay(rows,'checkpoint'))


class FakeService:
    def __init__(self, fail=False, wait=False):
        self.calls=[]; self.fail=fail; self.wait=wait; self.closed=0

    async def research(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        try:
            if self.wait:
                await asyncio.Event().wait()
            event={"type":"error","message":"故意失败"} if self.fail else {"type":"research_complete","final_report":"回答"+str(len(self.calls))}
            yield "data: "+json.dumps(event)+"\n\n"
        finally:
            self.closed+=1


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_turns_share_session_but_not_run_and_no_hidden_inputs(self):
        service=FakeService()
        with tempfile.TemporaryDirectory() as tmp:
            result=await run_dialogue(service,Task.model_validate(fixture()),Path(tmp)/"运行",timeout_seconds=1,collect_records=AsyncMock(return_value=[]))
        self.assertTrue(result["completed"])
        a,b=service.calls
        self.assertEqual(a["session_id"],b["session_id"])
        self.assertNotEqual(a["run_id"],b["run_id"])
        self.assertEqual(a["research_context"]["conversation_context"],[])
        self.assertEqual(b["research_context"]["conversation_context"],[{"role":"user","content":"先研究甲"},{"role":"assistant","content":"回答1"}])
        self.assertNotIn("隐含真值",str(service.calls))
        self.assertNotIn("原题不直接传模型",str(service.calls))
        self.assertEqual(service.closed,2)

    async def test_failure_stops_followups_and_saves_raw_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/"运行"
            result=await run_dialogue(FakeService(fail=True),Task.model_validate(fixture()),directory,timeout_seconds=1,collect_records=AsyncMock(return_value=[]))
            self.assertTrue((directory/"逐轮结果.json").exists())
        self.assertFalse(result["completed"])
        self.assertEqual(result["unexecuted_turns"],[2])

    async def test_timeout_closes_stream_and_is_not_success(self):
        service=FakeService(wait=True)
        with tempfile.TemporaryDirectory() as tmp:
            result=await run_dialogue(service,Task.model_validate(fixture()),Path(tmp)/"运行",timeout_seconds=.01,collect_records=AsyncMock(return_value=[]))
        self.assertFalse(result["completed"])
        self.assertEqual(service.closed,1)

    async def test_telemetry_failure_not_misreported_as_no_compression(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=await run_dialogue(FakeService(),Task.model_validate(fixture()),Path(tmp)/"运行",timeout_seconds=1,collect_records=AsyncMock(side_effect=RuntimeError("无遥测")))
        self.assertEqual(result["turns"][0]["telemetry_status"],"error")
        self.assertFalse(result["observation_complete"])


class CompressionTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_attempt_and_adoption_have_distinct_records(self):
        from test_上下文消融配置 import NamespacedStore, runnable_policy, CharCounter, make_agent, summary_text
        from service.agent_context import prepare_agent_messages, ContextBudgetExceeded
        from service.agent_history import HistoryScope, bind_agent_history
        import uuid
        for success in (True, False):
            store=NamespacedStore()
            scope=HistoryScope(store,"run","session",None,context_policy=runnable_policy(semantic_summary=True).snapshot())
            row=await store.append(scope,"DeepScout","llm_output",{"content":"保留这个结论。"*2000},str(uuid.uuid4()))
            summarizer=AsyncMock(return_value=summary_text(row["source_id"]),side_effect=None if success else ValueError("模拟摘要失败"))
            with bind_agent_history(scope), patch("service.agent_context.get_qwen_token_counter",return_value=CharCounter()), patch("service.agent_context.scoped_input_budget",return_value=1000), patch("service.agent_context.summarize_history",summarizer):
                if success:
                    await prepare_agent_messages(make_agent(),scope,"s","q","current",6000)
                else:
                    with self.assertRaises(ContextBudgetExceeded):
                        await prepare_agent_messages(make_agent(),scope,"s","q","current",6000)
            view=next(r["payload"] for r in (await store.load(scope,"DeepScout"))["messages"] if r["kind"]=="context_view")
            self.assertTrue(view["summary_attempted"])
            self.assertEqual(view["summary_adopted"],success)
            self.assertEqual(view["summary_version_after"],int(success))

    async def test_offload_and_summary_disabled_are_distinguished(self):
        from test_上下文消融配置 import NamespacedStore, runnable_policy, CharCounter, make_agent
        from service.agent_context import prepare_agent_messages
        from service.agent_history import HistoryScope, bind_agent_history
        import uuid
        store=NamespacedStore()
        scope=HistoryScope(store,"run","session",None,context_policy=runnable_policy().snapshot())
        for i in range(4):
            call=str(uuid.uuid4())
            await store.append(scope,"DeepScout","tool_input",{"q":str(i)},call)
            await store.append(scope,"DeepScout","tool_output",{"body":"全文"*2500},call)
        with bind_agent_history(scope), patch("service.agent_context.get_qwen_token_counter",return_value=CharCounter()), patch("service.agent_context.scoped_input_budget",return_value=6000):
            await prepare_agent_messages(make_agent(),scope,"s","q","current",6000)
        view=next(r["payload"] for r in (await store.load(scope,"DeepScout"))["messages"] if r["kind"]=="context_view")
        self.assertGreater(view["tokens_before_compression"],view["tokens_after_early_offload"])
        self.assertFalse(view["summary_attempted"])
        self.assertFalse(view["summary_adopted"])
        self.assertEqual(view["summary_version_before"],view["summary_version_after"])


if __name__ == "__main__":
    unittest.main()
