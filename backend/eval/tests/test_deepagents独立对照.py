"""真实deepagents图配合模型/网页桩；不发外部模型或搜索请求。"""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration, LLMResult
from eval.baselines.deepagents_runner import Settings, Journal, UsageLedger, WebTools, BudgetExceeded, run_task, load_tasks, final_article
from eval.scripts.run_deepagents_bench import prepare_output, export_results


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    cursor: int = 0

    @property
    def _llm_type(self):
        return "deepagents-offline-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.responses[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def response(content="", tool=None, args=None):
    return AIMessage(content=content, tool_calls=[{"name":tool,"args":args,"id":f"call-{tool}"}] if tool else [],
                     usage_metadata={"input_tokens":10,"output_tokens":5,"total_tokens":15})


class IndependentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name)
        self.settings = Settings(allowed_domains=["example.org"])
        self.search = Mock()
        self.search.search.return_value = {"data":{}}
        self.search.extract_search_results.return_value = [
            {"url":"https://example.org/paper","snippet":"证据"},
            {"url":"https://elsewhere.org/paper","snippet":"域外"}]
        item = Mock(success=True, final_url="https://example.org/paper")
        item.to_dict.return_value = {"content":"研究结果原文。","url":"https://example.org/paper"}
        self.reader = Mock(read_url=AsyncMock(return_value=item))

    async def test_real_graph_search_read_virtual_file_and_report(self):
        model = ScriptedModel(responses=[
            response(tool="search_web", args={"query":"研究","domain":"example.org"}),
            response(tool="read_web", args={"url":"https://example.org/paper"}),
            response(tool="write_file", args={"file_path":"/证据.md","content":"已读取研究原文。"}),
            response("结论见[原文](https://example.org/paper)。")])
        result = await run_task({"id":1,"prompt":"研究问题"}, self.settings, self.path, "研究任务",
                                model=model, search=self.search, reader=self.reader)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["search_requests"], 1)
        self.assertEqual(result["page_reads"], 1)
        self.assertEqual(result["metrics"]["model_calls"], 4)
        self.assertEqual(result["metrics"]["reported_input_tokens"], 40)
        self.assertTrue(result["metrics"]["usage_complete"])
        self.assertIsNone(result["metrics"]["cost_estimate"])
        state = json.loads((self.path/"最终状态.json").read_text())
        self.assertIn("/证据.md", state["files"])
        self.assertFalse((self.path/"证据.md").exists())
        self.assertNotIn("service.deep_research_v2.graph", sys.modules)

    async def test_model_call_limit_stops_actual_model(self):
        model = ScriptedModel(responses=[response(tool="search_web",args={"query":"研究","domain":"example.org"}),response("不应执行")])
        settings = self.settings.model_copy(update={"max_model_calls":1})
        result = await run_task({"id":1,"prompt":"研究问题"},settings,self.path,"研究",
                                model=model,search=self.search,reader=self.reader)
        self.assertEqual(result["status"],"failed")
        self.assertEqual(model.cursor,1)
        self.assertEqual(result["error"],"BudgetExceeded")

    async def test_scoped_tools_and_limits(self):
        settings=self.settings.model_copy(update={"max_search_requests":1,"max_page_reads":1})
        gateway=WebTools(settings,Journal(self.path/"调用.jsonl"),self.search,self.reader)
        blocked=json.loads(await gateway.read_web("https://elsewhere.org"))
        self.assertIn("error",blocked)
        self.reader.read_url.assert_not_awaited()
        value=json.loads(await gateway.search_web("site:elsewhere.org 研究","example.org"))
        self.assertEqual(len(value["results"]),1)
        self.search.search.assert_called_once_with("site:example.org 研究")
        with self.assertRaises(BudgetExceeded):
            await gateway.search_web("研究","example.org")
        await gateway.read_web("https://example.org/paper")
        with self.assertRaises(BudgetExceeded):
            await gateway.read_web("https://example.org/next")

    async def test_open_web_search_does_not_require_a_domain(self):
        gateway = WebTools(self.settings.model_copy(update={"allowed_domains": []}),
                           Journal(self.path / "全网调用.jsonl"), self.search, self.reader)
        result = json.loads(await gateway.search_web("site:elsewhere.org 研究"))
        self.assertEqual(len(result["results"]), 2)
        self.search.search.assert_called_once_with("研究")

    async def test_citation_tool_binds_only_read_snapshot_quotes(self):
        settings = self.settings.model_copy(update={"citation_verification": True})
        gateway = WebTools(settings, Journal(self.path / "核验调用.jsonl"), self.search, self.reader)
        gateway.documents["https://example.org/paper"] = {"final_url": "https://example.org/paper", "content": "研究结果原文。"}
        async def verifier(request, response):
            data = json.loads(request.read_text())
            self.assertEqual(data["claims"][0]["references"][0]["quote"], "研究结果原文。")
            self.assertEqual(data["claims"][1]["references"], [])
            self.assertEqual(data["claims"][2]["references"], [])
            return {"status": "passed", "claims": [{"status": "supported"} for _ in data["claims"]]}
        gateway._run_verifier = verifier
        result = json.loads(await gateway.verify_citations([
            {"claim": "有结果", "references": [{"url": "https://example.org/paper", "quote": "研究结果原文。"}]},
            {"claim": "编造引文", "references": [{"url": "https://example.org/paper", "quote": "虚构内容"}]},
            {"claim": "未读来源", "references": [{"url": "https://example.org/unread", "quote": "研究结果原文。"}]}]))
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["supported_count"], 1)
        self.assertEqual(result["claims"][1]["status"], "invalid_citation")
        self.assertFalse(result["all_claims_checked"])
        self.assertTrue((self.path / "引用核验请求-001.json").exists())

    async def test_citation_worker_failure_is_visible_and_report_survives(self):
        settings = self.settings.model_copy(update={"citation_verification": True})
        gateway = WebTools(settings, Journal(self.path / "异常核验调用.jsonl"), self.search, self.reader)
        gateway._run_verifier = AsyncMock(side_effect=RuntimeError("模拟异常"))
        result = json.loads(await gateway.verify_citations([{"claim": "事实", "references": []}]))
        self.assertEqual(result["execution_status"], "error")
        self.assertEqual(result["supported_count"], 0)

    async def test_missing_usage_is_not_zero_cost(self):
        model=ScriptedModel(responses=[AIMessage(content="报告")])
        result=await run_task({"id":1,"prompt":"研究"},self.settings,self.path,"研究",
                              model=model,search=self.search,reader=self.reader)
        self.assertEqual(result["status"],"completed")
        self.assertFalse(result["metrics"]["usage_complete"])
        self.assertEqual(result["metrics"]["calls_missing_usage"],1)
        self.assertIsNone(result["metrics"]["cost_estimate"])

    async def test_timeout_saves_failure(self):
        from unittest.mock import patch
        graph=Mock()
        async def hang(*args,**kwargs):
            await asyncio.Event().wait()
        graph.ainvoke=hang
        settings=self.settings.model_copy(update={"timeout_seconds":.01})
        with patch("eval.baselines.deepagents_runner.build_agent",return_value=graph):
            result=await run_task({"id":1,"prompt":"研究"},settings,self.path,"研究",model=Mock(),search=self.search,reader=self.reader)
        self.assertEqual(result["status"],"timeout")
        self.assertTrue((self.path/"结果.json").exists())

    async def test_failed_artifact_write_does_not_export_success_article(self):
        from unittest.mock import patch
        from eval.baselines.deepagents_runner import write_json
        def fail_state(path, value):
            if path.name == "最终状态.json":
                raise OSError("模拟磁盘失败")
            write_json(path, value)
        with patch("eval.baselines.deepagents_runner.write_json", side_effect=fail_state):
            result = await run_task({"id":1,"prompt":"研究"},self.settings,self.path,"研究",
                                    model=ScriptedModel(responses=[response("不应导出的报告")]),
                                    search=self.search,reader=self.reader)
        self.assertEqual(result["status"],"failed")
        self.assertEqual(result["article"],"")

    async def test_cancelled_task_saves_partial_usage_and_propagates(self):
        from unittest.mock import patch
        graph=Mock(ainvoke=AsyncMock(side_effect=asyncio.CancelledError))
        with patch("eval.baselines.deepagents_runner.build_agent",return_value=graph), self.assertRaises(asyncio.CancelledError):
            await run_task({"id":1,"prompt":"研究"},self.settings,self.path,"研究",
                           model=Mock(),search=self.search,reader=self.reader)
        result=json.loads((self.path/"结果.json").read_text())
        self.assertEqual(result["status"],"interrupted")
        self.assertFalse(result["metrics"]["usage_complete"])

    def test_prices_calculate_only_from_reported_usage(self):
        settings=self.settings.model_copy(update={"input_price_per_million":2.,"output_price_per_million":6.,"search_price_per_request":.01})
        ledger=UsageLedger(settings,Journal(self.path/"调用.jsonl"))
        ledger.on_chat_model_start({},[],run_id="one")
        ledger.on_llm_end(LLMResult(generations=[[ChatGeneration(message=response("报告"))]]),run_id="one")
        self.assertAlmostEqual(ledger.summary(2)["cost_estimate"],.02005)

    def test_resume_requires_identical_manifest_and_no_overwrite(self):
        output=self.path/"运行"
        prepare_output(output,{"hash":"a"})
        prepare_output(output,{"hash":"a"},True)
        with self.assertRaises(ValueError): prepare_output(output,{"hash":"b"},True)
        with self.assertRaises(ValueError): prepare_output(output,{"hash":"a"})

    def test_resume_exports_latest_failure_or_success_without_duplicates(self):
        for attempt,status in [(1,"failed"),(2,"completed"),(3,"failed")]:
            folder=self.path/"题目-1"/f"尝试-{attempt:04d}"
            folder.mkdir(parents=True)
            (folder/"结果.json").write_text(json.dumps({"id":1,"status":status,"article":"报告" if status=="completed" else ""}))
        rows=export_results(self.path,[{"id":1}])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["status"],"completed")

    def test_input_validation_and_empty_report(self):
        path=self.path/"题目.jsonl"
        path.write_text('{"id":1,"prompt":"研究","language":"zh"}\n')
        self.assertEqual(len(load_tasks(path,ids={"1"})),1)
        with self.assertRaises(ValueError): load_tasks(path,ids={"2"})
        with self.assertRaises(ValueError): final_article({"messages":[AIMessage(content="")]})
        with self.assertRaises(ValueError): Settings(max_output_tokens=32768)
        with self.assertRaises(ValueError): Settings(input_price_per_million=-1)


if __name__ == "__main__":
    unittest.main()
