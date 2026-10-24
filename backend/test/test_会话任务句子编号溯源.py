"""编号溯源与在线执行解耦的回归；使用模型、数据库和父图替身。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "test"), str(BACKEND / "eval/tests")]
from service.session_task_service import query_sentence_index, validate_decision, commit_requirements
from service.research_context_service import ResearchContext, ResearchContextService
from test_会话智能路由 import candidate, parent
from test_会话成果继承 import service
from test_智能体独立历史 import CharCounter

QUERY = "请将正文缩短到1200字。参考文献单列，不计正文。"

def numbered_decision(ids=None):
    return {"topic_relation": "continue", "depends_on_history": True,
            "target_artifact_ref": "report:one", "resolved_query": "修改已有工程报告，" + QUERY,
            "operations": [{"kind": "edit", "source_sentence_ids": ["S1", "S2"] if ids is None else ids}],
            "requirement_changes": [{"op": "set", "field": "max_body_chars", "value": 1200,
                                     "source_sentence_ids": ["S1"]}],
            "revision_instructions": [QUERY]}

def resolve_response(raw, query=QUERY):
    context = ResearchContext(query, query, artifact_candidates=[candidate()],
                              task_trace={"sentence_index": query_sentence_index(query, "turn-15")})
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))))
    resolver = ResearchContextService.__new__(ResearchContextService)
    resolver.token_counter = CharCounter()
    with patch("service.research_context_service.create_openai_client", return_value=client):
        resolver._resolve_query(context)
    return context, client.chat.completions.create


class SentenceIndexTests(unittest.TestCase):
    def test_ids_offsets_and_original_text_are_deterministic(self):
        query = "  请缩短正文。\n参考文献单列； 保留版本!  "
        index = query_sentence_index(query, "turn-a")
        self.assertEqual([x["id"] for x in index["sentences"]], ["S1", "S2", "S3"])
        self.assertEqual([x["text"] for x in index["sentences"]], ["请缩短正文。", "参考文献单列；", "保留版本!"])
        self.assertTrue(all(query[x["start"]:x["end"]] == x["text"] for x in index["sentences"]))
        self.assertEqual(index, query_sentence_index(query, "turn-a"))
        self.assertEqual(index["query_sha256"], hashlib.sha256(query.encode()).hexdigest())
        self.assertNotEqual(index["turn_id"], query_sentence_index(query, "turn-b")["turn_id"])

    def test_urls_code_quoted_punctuation_and_decimal_are_not_split(self):
        query = "检查 https://example.com/v1.2?a=1;b=2。保留 `x; y?`。不要改变“连接。状态”。\n```ts\na();\nb();\n```\n目标1.5秒. Next step."
        sentences = query_sentence_index(query)["sentences"]
        self.assertEqual(len(sentences), 6)
        self.assertIn("a=1;b=2", sentences[0]["text"])
        self.assertEqual(sentences[3]["text"], "```ts\na();\nb();\n```")
        self.assertTrue(all(query[x["start"]:x["end"]] == x["text"] for x in sentences))


class TraceExecutionTests(unittest.TestCase):
    def test_numbered_operations_commit_without_quote(self):
        value = numbered_decision()
        result = validate_decision(value, QUERY, [candidate()], turn_id="turn-15")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["execution_plan"], "edit")
        self.assertEqual(result["source_trace"]["status"], "resolved")
        self.assertEqual(result["source_trace"]["turn_id"], "turn-15")
        base = {"contract_revision": 1, "active_constraints": {"max_body_chars": 2400}, "language": "中文"}
        before = copy.deepcopy(base)
        committed = commit_requirements(base, result, QUERY)
        self.assertEqual(committed["active_constraints"]["max_body_chars"], 1200)
        self.assertEqual(committed["language"], "中文")
        self.assertFalse(committed["contract_update"]["source_quote_check"])
        self.assertEqual(committed["contract_update"]["source_trace"], result["source_trace"])
        self.assertEqual(base, before)

    def test_missing_unknown_malformed_and_duplicate_ids_do_not_change_execution(self):
        for ids in ([], ["S999"], None, "S1", {"bad": True}, [3, "S1", "S1"]):
            with self.subTest(ids=ids):
                value = numbered_decision(ids)
                value["operations"][0]["source_sentence_ids"] = ids
                value["requirement_changes"][0]["source_sentence_ids"] = ids
                before = copy.deepcopy(value)
                result = validate_decision(value, QUERY, [candidate()])
                self.assertEqual(result["status"], "ready")
                self.assertEqual(result["execution_plan"], "edit")
                self.assertEqual(result["source_trace"]["status"], "issues")
                self.assertEqual(result["source_trace"]["issue_item_count"], 2)
                committed = commit_requirements({"contract_revision": 0, "active_constraints": {}}, result, QUERY)
                self.assertEqual(committed["active_constraints"]["max_body_chars"], 1200)
                self.assertEqual(value, before)

    def test_legacy_quotes_are_recorded_but_never_gate_execution(self):
        for quote in ("改写过的引用", "", None, 123, {"bad": True}, "长" * 5000):
            with self.subTest(quote_type=type(quote).__name__):
                value = numbered_decision()
                for group in ("operations", "requirement_changes"):
                    value[group][0]["quote"] = quote
                    value[group][0].pop("source_sentence_ids")
                result = validate_decision(value, QUERY, [candidate()])
                self.assertEqual(result["source_trace"]["entries"][0]["legacy_quote"], quote)
                committed = commit_requirements({"active_constraints": {}}, result, QUERY)
                self.assertEqual(committed["active_constraints"]["max_body_chars"], 1200)

    def test_remove_constraint_does_not_depend_on_provenance(self):
        value = numbered_decision([])
        value["requirement_changes"] = [{"op": "remove", "field": "max_body_chars", "source_sentence_ids": ["S999"]}]
        result = validate_decision(value, "取消字数限制", [candidate()])
        self.assertEqual(commit_requirements({"active_constraints": {"max_body_chars": 2400}}, result,
                                             "取消字数限制")["active_constraints"], {})

    def test_turn15_style_evidence_followup_is_not_blocked_by_missing_quotes(self):
        query = "现在请针对官方 TypeScript SDK 定向补一到两处直接相关的文档或源码，把上一轮待核实的服务入口、配置方式和状态处理核实清楚。已有 Python 资料可以作为历史对照，但不能替代 TypeScript 的实现依据。"
        value = numbered_decision([])
        value.update(resolved_query="核对既有MCP报告的TypeScript SDK实现依据", requirement_changes=[],
                     operations=[{"kind": "evidence", "source_sentence_ids": ["S999"]}],
                     revision_instructions=[query], search_tasks=[{"query": "MCP TypeScript SDK state handling",
                        "target_section": "sec_0", "evidence_goal": "核对服务入口和状态管理"}])
        result = validate_decision(value, query, [candidate()])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["execution_plan"], "augment")
        self.assertEqual(result["source_trace"]["status"], "issues")

    def test_core_types_targets_and_search_tasks_remain_required(self):
        for mutation in (lambda v: v["requirement_changes"][0].update(value=True),
                         lambda v: v["operations"][0].update(kind="unknown"),
                         lambda v: v.update(target_artifact_ref="other-owner"),
                         lambda v: v["operations"][0].update(kind="evidence")):
            value = numbered_decision(["S999"]); mutation(value)
            with self.assertRaises(ValueError): validate_decision(value, QUERY, [candidate()])


class TracePersistenceTests(unittest.TestCase):
    def test_first_turn_without_classifier_still_preserves_sentence_index(self):
        resolver = ResearchContextService.__new__(ResearchContextService)
        resolver.token_counter = CharCounter(); resolver.history_token_limit = 3000; resolver.memory_token_limit = 2000
        resolver._load_history = Mock(return_value=SimpleNamespace(messages=[], cache_hit=False,
            last_message_id=None, compacted_tokens=0, original_tokens=0, summarized_messages=0,
            pruned_tool_results=0, semantic_context_version=0))
        resolver._load_memories = Mock(return_value=[])
        with patch("service.session_task_service.load_artifact_candidates", return_value=[]), patch(
            "service.semantic_context_service.semantic_context_enabled", return_value=True):
            result = resolver.build(user_id="user", session_id="session", current_turn_id="turn-first",
                                    current_query=QUERY, resolve_query=False)
        trace = result["task_metrics"]["task_trace"]
        self.assertFalse(trace["model_called"])
        self.assertIsNone(trace["raw_response"])
        self.assertEqual(trace["sentence_index"]["turn_id"], "turn-first")
        self.assertEqual(len(trace["sentence_index"]["sentences"]), 2)

    def test_success_keeps_raw_model_output_and_turn_scoped_index(self):
        raw = "  ```json\n" + json.dumps(numbered_decision(["S999"]), ensure_ascii=False) + "\n```  "
        context, call = resolve_response(raw)
        self.assertEqual(context.session_task["status"], "ready")
        self.assertEqual(context.task_trace["raw_response"], raw)
        self.assertEqual(context.task_trace["parsed_response"], numbered_decision(["S999"]))
        self.assertEqual(context.task_metrics["task_trace"]["sentence_index"]["turn_id"], "turn-15")
        self.assertGreater(context.task_metrics["source_trace_issue_items"], 0)
        prompt = call.call_args.kwargs["messages"][1]["content"]
        current = call.call_args.kwargs["messages"][-1]["content"]
        self.assertIn("本轮程序分句", current)
        self.assertIn(QUERY, current)
        self.assertIn("不因此选择uncertain", prompt)
        schema = json.loads(prompt.split("仅返回符合以下schema的JSON，不输出代码围栏：\n", 1)[1])
        self.assertNotIn("quote", schema["$defs"]["TaskOperation"]["properties"])
        self.assertNotIn("quote", schema["$defs"]["RequirementChange"]["properties"])
        self.assertEqual(context.to_dict()["task_metrics"]["task_trace"]["raw_response"], raw)

    def test_bad_json_and_core_schema_errors_keep_raw_output(self):
        for raw in ("```json\n{broken}\n```", '{"wrong":true}'):
            with self.subTest(raw=raw):
                context, _ = resolve_response(raw, "这份报告缩短")
                self.assertEqual(context.session_task["status"], "clarify")
                self.assertEqual(context.task_metrics["task_trace"]["raw_response"], raw)
                self.assertIn("validation_error", context.task_trace)
                self.assertEqual(context.task_trace["sentence_index"]["sentences"][0]["text"], "这份报告缩短")


class NumberedGraphInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_entry_revalidation_and_requirement_commit_are_nonblocking(self):
        context, _ = resolve_response(json.dumps(numbered_decision(["S999"]), ensure_ascii=False))
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=parent()))
        with patch("service.session_task_service.authorize_artifact") as authorize, patch(
            "service.qwen_token_counter.get_qwen_token_counter", return_value=CharCounter()
        ), patch.dict(os.environ, {"SESSION_RESEARCH_INHERITANCE_ENABLED": "1", "CONTEXT_ABLATION_CONFIG": "", "AGENT_COMPRESSION_CONFIG": ""}):
            result = await service()._graph_input(graph, {}, resume=False, query=QUERY,
                session_id="session", user_id="user", run_id="child", allowed_domains=None,
                research_context=context.to_dict())
        state = result["research"]
        self.assertEqual(state["execution_plan"], "edit")
        self.assertEqual(state["active_constraints"]["max_body_chars"], 1200)
        self.assertEqual(state["session_task_metrics"]["task_trace"]["raw_response"], context.task_trace["raw_response"])
        self.assertEqual(state["session_task"]["source_trace"]["status"], "issues")
        self.assertEqual(state["session_task"]["source_trace"]["turn_id"], "turn-15")
        authorize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
