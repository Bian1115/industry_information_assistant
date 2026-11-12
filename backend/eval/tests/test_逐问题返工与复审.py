"""按当前审核问题返工、整篇复审与章节恢复的离线机制测试；不调用外部模型。"""

from copy import deepcopy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "app"), str(Path(__file__).parent)]

from service.deep_research_v2.issue_workflow import (
    record_writer_responses,
)
from service.deep_research_v2.state import reopen_sections_for_rewrite
from service.deep_research_v2.review_contract import QUALITY_SCORE_BASIS
from service.deep_research_v2.graph import pack_state
from test_Writer论断引用结构 import make_state, make_writer, synthesis, DOCUMENT, URL, ORIGINAL
import test_quality_score as quality_tests
from test_补搜任务执行 import make_scout


QUOTE = "公司2025年营业收入为100亿元。"
CONTENT = QUOTE + f"[年度报告]({URL})" + "保留统计口径并解释适用范围。" * 60
OBSERVATIONS = []


def issue(ident="问题收入", section="sec_1", **overrides):
    title = {"sec_1": "收入", "sec_2": "风险", "全局": "执行摘要"}[section]
    row = {"issue_id": ident, "issue_type": "logical_error", "severity": "major",
           "target": {"target_type": "existing_section", "chapter_id": None if section == "全局" else section,
                      "chapter_title": title, "specific_location": "收入段",
                      "insert_after_chapter_id": None, "proposed_chapter_title": None},
           "description": "公司收入99亿元，需报告准确收入及统计范围", "evidence": "原文为100亿元，旧句为99亿元",
           "modification": "修正收入，并保留范围限制；收入为100亿元，并说明统计范围",
           "requires_new_search": False, "search_query": "",
           "resolved_section_ids": ["sec_1", "sec_2", "__synthesis__"] if section == "全局" else [section]}
    row.update(overrides)
    return row


def response(ident="问题收入", **overrides):
    row = {"issue_id": ident, "status": "addressed", "revised_quote": QUOTE,
           "change_description": "将99修为100，保留统计范围", "evidence_summary": "年报支持收入100亿元",
           "remaining_gaps": "无剩余缺口", "acceptance_check": "数字已修正，范围维持原文限定",
           "evidence": [{"document_id": DOCUMENT, "sentence_ids": ["s000001"]}]}
    row.update(overrides)
    return row


def work(*issues):
    state = make_state()
    state.update(iteration=1, phase="revising", critic_feedback=list(issues or [issue()]))
    state["facts"] = []
    state["raw_sources"][0]["related_sections"] = ["sec_1"]
    state["draft_sections"] = {"sec_1": "公司收入99亿元。" + "原有分析。" * 100, "sec_2": "无需改动。" * 100}
    for section in state["outline"]:
        section["status"] = "drafted"
    state["final_report"] = "## 1 收入\n" + QUOTE + "\n## 2 风险\n当前报告其余正文。"
    return state


def verdict(issues=None):
    return quality_tests.review(80, issues)


class WriterIssueTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_edit_revision_entry_also_returns_issue_trace(self):
        state = work()
        state["writing_mode"] = "user_edit"
        result = {"sections": [
            {"id": "sec_1", "title": "收入", "content": CONTENT, "claims": [], "issue_responses": [response()]},
            {"id": "sec_2", "title": "风险", "content": state["draft_sections"]["sec_2"], "claims": []}],
            "executive_summary": QUOTE, "conclusions": ["保留范围"], "outlook": "仍有不确定性", "claims": []}
        writer = make_writer(result)
        await writer.process(state)
        self.assertIn("按章返工清单", writer.call_llm.await_args.kwargs["user_prompt"])
        self.assertEqual(state["writer_revision_records"][0]["status"], "addressed")
        self.assertEqual(state["critic_feedback"][0]["issue_id"], "问题收入")

    async def test_direct_revision_records_location_evidence_and_waits_for_review(self):
        state = work()
        writer = make_writer({"revised_content": CONTENT, "claims": [], "issue_responses": [response()]}, synthesis([]))
        unchanged = state["draft_sections"]["sec_2"]
        await writer._revise_report(state)
        prompt = writer.call_llm.await_args_list[0].kwargs["user_prompt"]
        for value in ("问题收入", "公司收入99亿元。", "报告准确收入及统计范围", "原文为100亿元", "修正收入", "收入为100亿元，并说明统计范围", ORIGINAL):
            self.assertIn(value, prompt)
        self.assertEqual(state["draft_sections"]["sec_2"], unchanged)
        self.assertEqual(state["critic_feedback"][0]["issue_id"], "问题收入")
        self.assertEqual(state["critic_feedback"][0]["writer_status"], "addressed")
        row = state["writer_revision_records"][0]
        self.assertEqual(row["validation_errors"], [])
        self.assertEqual(row["evidence_excerpts"][0]["text"], ORIGINAL)
        self.assertIn(row["revised_quote"], state["final_report"])

    async def test_supplementary_rewrite_uses_same_issue_context_and_preserves_old_content(self):
        state = work(issue(requires_new_search=True, search_query="公司收入年度报告"))
        original = state["draft_sections"]["sec_1"]
        state["phase"] = "re_researching"
        state["pending_research_tasks"] = [{"issue_id": "问题收入", "target_section": "sec_1", "query": "公司收入年度报告", "acceptance": "收入与统计范围均有依据"}]
        scout = make_scout()
        await scout._supplementary_research(state)
        self.assertEqual(state["revision_original_sections"]["sec_1"], original)
        writer = make_writer({"content": CONTENT, "claims": [], "issue_responses": [response()]})
        await writer._write_section(state, state["outline"][0])
        prompt = writer.call_llm.await_args.kwargs["user_prompt"]
        for value in ("judgment_basis", "acceptance_criteria", "remaining_gaps", "sources_collected", "收入与统计范围均有依据", original):
            self.assertIn(value, prompt)
        self.assertEqual(state["writer_revision_records"][0]["status"], "addressed")
        self.assertEqual(state["critic_feedback"][0]["issue_id"], "问题收入")

    async def test_mixed_search_and_text_issues_reopen_both_chapters(self):
        state = work(issue(requires_new_search=True, search_query="公司年报"), issue("范围问题", "sec_2"))
        state["phase"] = "re_researching"
        state["pending_research_tasks"] = [{"issue_id": "问题收入", "target_section": "sec_1", "query": "公司年报"}]
        await make_scout()._supplementary_research(state)
        self.assertTrue(all(s["status"] == "pending" for s in state["outline"]))
        self.assertEqual(set(state["revision_original_sections"]), {"sec_1", "sec_2"})

    async def test_no_search_evidence_is_explicit_and_does_not_close_issue(self):
        state = work(issue(requires_new_search=True, search_query="缺失数据"))
        state["pending_research_tasks"] = [{"issue_id": "问题收入", "target_section": "sec_1", "query": "缺失数据"}]
        scout = make_scout()
        scout._execute_search = AsyncMock(return_value=[])
        await scout._supplementary_research(state)
        self.assertEqual(state["issue_search_results"]["问题收入"]["status"], "no_usable_evidence")
        self.assertEqual(state["critic_feedback"][0]["issue_id"], "问题收入")

    async def test_global_issue_is_also_passed_to_synthesis_with_trace(self):
        state = work(issue(section="全局"))
        state["draft_sections"]["sec_1"] = CONTENT
        state["section_claims"] = {"sec_1": [{"claim": QUOTE, "report_quote": QUOTE,
            "evidence": [{"document_id": DOCUMENT, "sentence_ids": ["s000001"]}]}]}
        row = response(evidence=[{"source_claim_ids": ["C1"]}], evidence_summary="首尾继承正文，不新增事实")
        result = synthesis([])
        result["issue_responses"] = [row]
        writer = make_writer(result)
        await writer._synthesize_report(state)
        self.assertIn("摘要、结论与展望的按问题返工清单", writer.call_llm.await_args.kwargs["user_prompt"])
        self.assertEqual(state["writer_revision_records"][0]["section_id"], "__synthesis__")
        self.assertEqual(state["writer_revision_records"][0]["evidence_excerpts"][0]["kind"], "section_claim_reference")
        self.assertEqual(state["critic_feedback"][0]["issue_id"], "问题收入")


class TraceValidationTests(unittest.TestCase):
    packet = {"documents": [{"document_id": DOCUMENT, "source_url": URL,
                             "original_sentences": [{"id": "s000001", "text": ORIGINAL}]}]}

    def test_forged_quote_or_evidence_and_legacy_self_report_cannot_mark_addressed(self):
        cases = [
            {"addressed_issues": ["问题收入"]},
            {"issue_responses": [response(revised_quote="正文没有这句")]},
            {"issue_responses": [response(evidence=[{"document_id": DOCUMENT, "sentence_ids": ["s999999"]}])]},
            {"issue_responses": [response(), response()]},
            {"issue_responses": [response(evidence=[{"document_id": [], "sentence_ids": ["s000001"]}])]},
            {"issue_responses": [response(evidence=[{"document_id": [], "source_claim_ids": ["C1"]}])]},
        ]
        for result in cases:
            with self.subTest(result=result):
                state = work()
                rows = record_writer_responses(state, "sec_1", CONTENT, result, self.packet)
                self.assertEqual(rows[0]["status"], "not_reported")
                self.assertTrue(rows[0]["validation_errors"])
                self.assertEqual(state["critic_feedback"][0]["issue_id"], "问题收入")

    def test_rag_units_keep_program_resolved_excerpts_and_idempotent_records(self):
        state = work()
        packet = {"evidence_units": [{"id": "原文单元", "document_id": DOCUMENT, "text": ORIGINAL, "source_url": URL}]}
        result = {"issue_responses": [response(evidence=[{"document_id": DOCUMENT, "evidence_unit_ids": ["原文单元"]}])]}
        for _ in range(2):
            record_writer_responses(state, "sec_1", CONTENT, result, packet)
        self.assertEqual(len(state["writer_revision_records"]), 1)
        self.assertEqual(state["writer_revision_records"][0]["evidence_excerpts"][0]["text"], ORIGINAL)
        self.assertEqual(state["critic_feedback"][0]["writer_status"], "addressed")

    def test_partial_global_response_does_not_claim_all_chapters_addressed(self):
        state = work(issue(section="全局"))
        record_writer_responses(state, "sec_1", CONTENT, {"issue_responses": [response()]}, self.packet)
        self.assertEqual(state["critic_feedback"][0]["writer_status"], "partial")
        record_writer_responses(state, "sec_2", CONTENT, {}, self.packet)
        self.assertEqual(state["critic_feedback"][0]["writer_status"], "not_reported")


class RevisionCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_section_restore_keeps_issue_records_without_duplicate_generation(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from test_Writer章节检查点 import Scenario, initial_state, graph_for, collect
        state = initial_state()
        state.update(iteration=1, critic_feedback=[issue()])
        scenario, saver = Scenario(fail_section="sec_3"), InMemorySaver()

        def writer():
            agent = scenario.writer()
            original = agent.call_llm
            async def call(**kwargs):
                data = json.loads(await original(**kwargs))
                if "content" in data and "sec_1正文" in data["content"]:
                    data["issue_responses"] = [response(revised_quote=data["content"])]
                return json.dumps(data, ensure_ascii=False)
            agent.call_llm = call
            return agent

        with self.assertRaisesRegex(RuntimeError, "sec_3"):
            await collect(graph_for(saver, writer()), pack_state(state))
        graph = graph_for(saver, writer())
        await collect(graph, None)
        from service.deep_research_v2.graph import graph_config
        result = (await graph.aget_state(graph_config("章节恢复测试"))).values["research"]
        self.assertEqual(scenario.calls["sec_1"], 1)
        self.assertEqual(len(result["writer_revision_records"]), 1)
        self.assertEqual(result["writer_revision_records"][0]["issue_id"], "问题收入")
        self.assertEqual(result["writer_revision_records"][0]["status"], "addressed")
        self.assertEqual(result["critic_feedback"][0]["issue_id"], "问题收入")


class RealAgentGraphTests(unittest.IsolatedAsyncioTestCase):
    async def scenario(self, search):
        from langgraph.checkpoint.memory import InMemorySaver
        from service.deep_research_v2.graph import build_graph, graph_config
        from test_单主链路图 import make_agents, passthrough_stage
        state = work()
        state.update(iteration=0, critic_feedback=[], execution_plan="edit")
        revised = {"claims": [], "issue_responses": [response()],
                   "content" if search else "revised_content": CONTENT}
        writer = make_writer(synthesis([]), revised, synthesis([]))
        critic = quality_tests.CriticReviewFlowTests._make_critic()
        critic._review_content = AsyncMock(side_effect=[
            verdict(issues=[issue(requires_new_search=search, search_query="公司收入及统计范围" if search else "")]),
            verdict(),
        ])
        scout = make_scout()
        graph = build_graph(make_agents(writer=writer, critic=critic, scout=scout), InMemorySaver(),
            run_id="补搜问题图" if search else "修订问题图", user_id="user", session_id="test-session",
            execute_stage=passthrough_stage)
        config = graph_config("补搜问题图" if search else "修订问题图")
        result = (await graph.ainvoke(pack_state(state), config))["research"]
        self.assertEqual(result["completion_reason"], "passed")
        self.assertEqual(result["critic_feedback"], [])
        self.assertEqual(result["critic_history"][0][0]["issue_id"], "问题收入")
        self.assertEqual(result["critic_history"][0][0]["writer_status"], "addressed")
        self.assertEqual(result["writer_revision_records"][0]["evidence_excerpts"][0]["text"], ORIGINAL)
        self.assertEqual(critic._review_content.await_count, 2)
        self.assertEqual(scout._execute_search.await_count, 1 if search else 0)
        OBSERVATIONS.append({"path": "补搜后重写" if search else "直接修订", "completion_reason": result["completion_reason"],
            "critic_feedback": deepcopy(result["critic_feedback"]),
            "writer_revision_records": deepcopy(result["writer_revision_records"]),
            "issue_search_results": deepcopy(result["issue_search_results"]),
            "boundary": "真实Agent流程与内存检查点，模型和搜索均为替身"})

    async def test_real_critic_writer_graph_direct_revision(self):
        await self.scenario(False)

    async def test_real_critic_writer_scout_graph_supplementary_rewrite(self):
        await self.scenario(True)


if __name__ == "__main__":
    unittest.main()
