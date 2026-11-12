"""补搜仅执行结构化任务，并保留问题与章节的对应关系。"""

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.state import create_initial_state, ResearchPhase


def make_task(issue_id="issue_1", section="sec_1", query="示例公司 年报 营收"):
    return {
        "issue_id": issue_id,
        "target_section": section,
        "query": query,
        "acceptance": "新增全文验证证据并通过对应质量门禁",
    }


def make_state():
    state = create_initial_state("示例公司经营情况", "补搜任务执行测试", ["example.gov.cn"])
    state["phase"] = ResearchPhase.RE_RESEARCHING.value
    return state


def make_scout():
    scout = DeepScout.__new__(DeepScout)
    scout.name = "DeepScout"
    scout.logger = logging.getLogger("test.补搜任务执行")
    scout.add_message = Mock()
    scout._execute_search = AsyncMock(return_value=[{
        "url": "https://example.gov.cn/report",
        "reader_verified": True,
        "full_content": "示例公司经营情况的原始报告。",
    }])
    scout._enrich_results_with_full_text = AsyncMock()
    scout._analyze_supplementary_results = AsyncMock(return_value={"extracted_facts": []})
    scout._ingest_search_analysis = Mock()
    scout._store_collected_sources = AsyncMock()
    scout._emit_search_results_event = Mock()
    scout._update_evidence_coverage = Mock()
    return scout


class SupplementaryTaskExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_query_preserves_each_issue_and_section(self):
        state, scout = make_state(), make_scout()
        state["pending_research_tasks"] = [
            make_task("issue_1", "sec_1"),
            make_task("issue_2", "sec_2/sec_3"),
        ]
        state["search_budget"].update(used=7, remaining=13)
        budget = state["search_budget"]

        result = await scout._supplementary_research(state)

        self.assertIs(result, state)
        self.assertEqual(scout._execute_search.await_count, 2)
        for search in scout._execute_search.await_args_list:
            self.assertEqual(search.args, ("示例公司 年报 营收",))
            self.assertIs(search.kwargs["state"], state)
            self.assertEqual(search.kwargs["count"], 8)
        self.assertIs(state["search_budget"], budget)
        self.assertEqual(budget["used"], 7)
        self.assertEqual(budget["remaining"], 13)
        self.assertEqual(
            [call.kwargs["section_id"]
             for call in scout._enrich_results_with_full_text.await_args_list],
            ["sec_1", "sec_2"],
        )
        ingested = scout._store_collected_sources.await_args_list
        self.assertEqual([call.args[2] for call in ingested],
                         [["sec_1"], ["sec_2", "sec_3"]])
        self.assertEqual(
            [call.kwargs for call in ingested],
            [{"is_supplementary": True, "critic_issue_id": "issue_1"},
             {"is_supplementary": True, "critic_issue_id": "issue_2"}],
        )
        scout._analyze_supplementary_results.assert_not_awaited()
        scout._ingest_search_analysis.assert_not_called()
        self.assertEqual(state["pending_research_tasks"], [])
        # 下一步由图的固定边决定（re_research -> analyze -> write），
        # Scout 不再自行把阶段改写成 writing。
        self.assertEqual(state["phase"], ResearchPhase.RE_RESEARCHING.value)

    async def test_legacy_queries_alone_cannot_start_supplementary_search(self):
        state, scout = make_state(), make_scout()
        state["pending_search_queries"] = ["旧入口中的检索词"]
        with self.assertRaisesRegex(ValueError, "pending_research_tasks"):
            await scout._supplementary_research(state)
        scout._execute_search.assert_not_awaited()
        self.assertEqual(state["phase"], ResearchPhase.RE_RESEARCHING.value)

    async def test_legacy_queries_are_ignored_when_tasks_exist(self):
        state, scout = make_state(), make_scout()
        state["pending_search_queries"] = ["旧入口中的检索词"]
        state["pending_research_tasks"] = [make_task(query="当前任务检索词")]
        await scout._supplementary_research(state)
        self.assertEqual(scout._execute_search.await_count, 1)
        self.assertEqual(scout._execute_search.await_args.args, ("当前任务检索词",))

    async def test_missing_empty_or_invalid_task_list_is_rejected(self):
        for value in (None, [], {}, "补搜任务", (make_task(),)):
            with self.subTest(value=value):
                state, scout = make_state(), make_scout()
                state["pending_research_tasks"] = value
                with self.assertRaisesRegex(ValueError, "pending_research_tasks"):
                    await scout._supplementary_research(state)
                scout._execute_search.assert_not_awaited()
        state, scout = make_state(), make_scout()
        del state["pending_research_tasks"]
        with self.assertRaisesRegex(ValueError, "pending_research_tasks"):
            await scout._supplementary_research(state)
        scout._execute_search.assert_not_awaited()

    async def test_invalid_query_fails_before_any_task_executes(self):
        for query in (None, False, 3, [], {}, "", " \t\n"):
            with self.subTest(query=query):
                state, scout = make_state(), make_scout()
                state["pending_research_tasks"] = [make_task(), make_task(query=query)]
                with self.assertRaisesRegex(ValueError, "第 2 个补搜任务的 query"):
                    await scout._supplementary_research(state)
                scout._execute_search.assert_not_awaited()
                self.assertEqual(len(state["pending_research_tasks"]), 2)
        state, scout = make_state(), make_scout()
        task = make_task()
        del task["query"]
        state["pending_research_tasks"] = [task]
        with self.assertRaisesRegex(ValueError, "query 必须是非空字符串"):
            await scout._supplementary_research(state)
        scout._execute_search.assert_not_awaited()

    async def test_non_object_task_fails_before_any_search(self):
        state, scout = make_state(), make_scout()
        state["pending_research_tasks"] = [make_task(), "无效任务"]
        with self.assertRaisesRegex(ValueError, "第 2 个补搜任务必须是对象"):
            await scout._supplementary_research(state)
        scout._execute_search.assert_not_awaited()

    async def test_task_query_is_trimmed_without_rewriting_input(self):
        state, scout = make_state(), make_scout()
        task = make_task(query="  示例公司 年报 营收\n")
        state["pending_research_tasks"] = [task]
        await scout._supplementary_research(state)
        self.assertEqual(scout._execute_search.await_args.args, ("示例公司 年报 营收",))
        self.assertEqual(task["query"], "  示例公司 年报 营收\n")

    async def test_supplementary_search_executes_all_tasks_before_cleanup(self):
        state, scout = make_state(), make_scout()
        state["pending_research_tasks"] = [
            make_task(f"issue_{index}", f"sec_{index}", f"公司{index} 年报")
            for index in range(12)
        ]
        await scout._supplementary_research(state)
        self.assertEqual(scout._execute_search.await_count, 12)
        self.assertEqual(
            [call.args[0] for call in scout._execute_search.await_args_list],
            [f"公司{index} 年报" for index in range(12)],
        )
        self.assertEqual(state["pending_research_tasks"], [])

    def test_initial_state_has_only_structured_research_tasks(self):
        state = make_state()
        self.assertEqual(state["pending_research_tasks"], [])
        self.assertNotIn("pending_search_queries", state)


if __name__ == "__main__":
    unittest.main()
