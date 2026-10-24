"""按问题返工、逐项复查与章节恢复的离线机制测试；不调用外部模型。"""

from copy import deepcopy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "app"), str(Path(__file__).parent)]

from service.deep_research_v2.issue_workflow import (
    issue_ledger, reconcile_reviews, record_writer_responses, review_packet,
)
from service.deep_research_v2.state import reopen_sections_for_rewrite
from service.deep_research_v2.graph import pack_state
from test_Writer论断引用结构 import make_state, make_writer, synthesis, DOCUMENT, URL, ORIGINAL
import test_quality_score as quality_tests
from test_补搜任务执行 import make_scout


QUOTE = "公司2025年营业收入为100亿元。"
CONTENT = QUOTE + f"[年度报告]({URL})" + "保留统计口径并解释适用范围。" * 60
OBSERVATIONS = []


def issue(ident="问题收入", section="sec_1", **overrides):
    row = {"id": ident, "target_section": section, "issue_type": "logic_error", "severity": "major",
           "location": "收入段", "report_quote": "公司收入99亿元。", "requirement": "报告准确收入及统计范围",
           "description": "收入数字与所列原文不一致", "evidence": "原文为100亿元，旧句为99亿元",
           "suggestion": "修正收入，并保留范围限制", "acceptance_criteria": "收入为100亿元，并说明统计范围",
           "requires_new_search": False, "search_query": "", "resolved": False}
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
    state["final_report"] = QUOTE + "当前报告其余正文。"
    return state


def verdict(issues=None, reviews=None):
    return {"overall_assessment": {"quality_score": 9, "verdict": "pass"},
            "issues": issues or [], "issue_reviews": reviews or []}


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
        self.assertFalse(state["critic_feedback"][0]["resolved"])

    async def test_direct_revision_records_location_evidence_and_waits_for_review(self):
        state = work()
        writer = make_writer({"revised_content": CONTENT, "claims": [], "issue_responses": [response()]}, synthesis([]))
        unchanged = state["draft_sections"]["sec_2"]
        await writer._revise_report(state)
        prompt = writer.call_llm.await_args_list[0].kwargs["user_prompt"]
        for value in ("问题收入", "公司收入99亿元。", "报告准确收入及统计范围", "原文为100亿元", "修正收入", "收入为100亿元，并说明统计范围", ORIGINAL):
            self.assertIn(value, prompt)
        self.assertEqual(state["draft_sections"]["sec_2"], unchanged)
        self.assertFalse(state["critic_feedback"][0]["resolved"])
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
        self.assertFalse(state["critic_feedback"][0]["resolved"])

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
        self.assertFalse(state["critic_feedback"][0]["resolved"])

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
        self.assertFalse(state["critic_feedback"][0]["resolved"])


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
                self.assertFalse(state["critic_feedback"][0]["resolved"])

    def test_rag_units_keep_program_resolved_excerpts_and_idempotent_records(self):
        state = work()
        packet = {"evidence_units": [{"id": "原文单元", "document_id": DOCUMENT, "text": ORIGINAL, "source_url": URL}]}
        result = {"issue_responses": [response(evidence=[{"document_id": DOCUMENT, "evidence_unit_ids": ["原文单元"]}])]}
        for _ in range(2):
            record_writer_responses(state, "sec_1", CONTENT, result, packet)
        self.assertEqual(len(state["writer_revision_records"]), 1)
        self.assertEqual(state["writer_revision_records"][0]["evidence_excerpts"][0]["text"], ORIGINAL)
        self.assertEqual(state["critic_feedback"][0]["writer_status"], "addressed")

    def test_review_packet_detects_stale_or_removed_revision_location(self):
        state = work()
        record_writer_responses(state, "sec_1", CONTENT, {"issue_responses": [response()]}, self.packet)
        state["final_report"] = "合稿后这句已经被替换。"
        self.assertFalse(json.loads(review_packet(state))[0]["located_in_current_report"])

    def test_legacy_writer_resolved_flag_requires_explicit_review(self):
        state = work(issue(resolved=True))
        row = issue_ledger(state)[0]
        self.assertFalse(row["resolved"])
        self.assertEqual(row["writer_status"], "addressed")

    def test_partial_global_response_does_not_claim_all_chapters_addressed(self):
        state = work(issue(section="全局"))
        record_writer_responses(state, "sec_1", CONTENT, {"issue_responses": [response()]}, self.packet)
        self.assertEqual(state["critic_feedback"][0]["writer_status"], "partial")
        record_writer_responses(state, "sec_2", CONTENT, {}, self.packet)
        self.assertEqual(state["critic_feedback"][0]["writer_status"], "not_reported")


class CriticIssueTests(unittest.IsolatedAsyncioTestCase):
    async def process(self, state, result):
        critic = quality_tests.CriticReviewFlowTests._make_critic()
        critic._review_content = AsyncMock(return_value=result)
        state["phase"] = "reviewing"
        await critic.process(state)
        return state

    async def test_omitted_old_issue_is_retained_and_blocks_pass(self):
        state = work()
        await self.process(state, verdict())
        self.assertEqual(state["critic_feedback"][0]["id"], "问题收入")
        self.assertFalse(state["critic_feedback"][0]["resolved"])
        self.assertEqual(state["quality_verdict"], "needs_revision")
        self.assertLessEqual(state["quality_score"], 6)

    async def test_confirm_repair_and_dismiss_wrong_critique_are_distinct(self):
        for status in ("resolved", "dismissed"):
            with self.subTest(status=status):
                state = work()
                original = state["final_report"]
                await self.process(state, verdict(reviews=[{"issue_id": "问题收入", "status": status,
                    "report_quote": QUOTE, "reason": "逐项对照当前正文，收入和统计范围符合验收；撤销时确认原批评误读正文"}]))
                self.assertTrue(state["critic_feedback"][0]["resolved"])
                self.assertEqual(state["critic_feedback"][0]["review_status"], status)
                self.assertEqual(state["quality_verdict"], "pass")
                self.assertEqual(state["final_report"], original)

    async def test_each_old_issue_requires_its_own_review(self):
        state = work(issue(), issue("第二问题", "sec_2", severity="minor"))
        await self.process(state, verdict(reviews=[{"issue_id": "问题收入", "status": "resolved", "report_quote": QUOTE, "reason": "满足验收"}]))
        self.assertTrue(state["critic_feedback"][0]["resolved"])
        self.assertFalse(state["critic_feedback"][1]["resolved"])
        self.assertEqual(state["quality_verdict"], "needs_revision")

    async def test_nonempty_report_quote_does_not_require_exact_match(self):
        for status in ("resolved", "dismissed"):
            for quote in ("收入段已说明100亿元收入及统计范围", f"“{QUOTE}”（sec_1）", "收入已修正……统计范围已说明"):
                with self.subTest(status=status, quote=quote):
                    state = work()
                    self.assertNotIn(quote, state["final_report"])
                    await self.process(state, verdict(reviews=[{"issue_id": "问题收入", "status": status,
                        "report_quote": quote, "reason": "当前收入和统计范围已满足该条验收"}]))
                    self.assertTrue(state["critic_feedback"][0]["resolved"])
                    self.assertEqual(state["critic_feedback"][0]["review_status"], status)
                    self.assertEqual(state["quality_verdict"], "pass")
                    self.assertEqual(state["phase"], "completed")

    async def test_empty_report_quote_or_reason_still_prevents_closure(self):
        for status in ("resolved", "dismissed"):
            for quote, reason in ((None, "已满足验收"), ("", "已满足验收"), ("   ", "已满足验收"),
                                  (QUOTE, None), (QUOTE, ""), (QUOTE, "   ")):
                with self.subTest(status=status, quote=quote, reason=reason):
                    state = work()
                    await self.process(state, verdict(reviews=[{"issue_id": "问题收入", "status": status,
                        "report_quote": quote, "reason": reason}]))
                    self.assertFalse(state["critic_feedback"][0]["resolved"])
                    self.assertEqual(state["quality_verdict"], "needs_revision")

    async def test_more_sources_and_high_score_do_not_implicitly_resolve(self):
        state = work()
        state["raw_sources"].append({"document_id": "新增原文", "content": "不相关新内容"})
        state["critic_feedback"][0]["evidence_fingerprint"] = "旧指纹"
        await self.process(state, verdict())
        self.assertFalse(state["critic_feedback"][0]["resolved"])

    async def test_repeated_issue_keeps_id_even_when_model_renumbers(self):
        state = work()
        await self.process(state, verdict(issues=[issue("模型重编号")]))
        self.assertEqual(len(state["critic_feedback"]), 1)
        self.assertEqual(state["critic_feedback"][0]["id"], "问题收入")

    async def test_unresolved_old_issue_keeps_search_query_and_exact_acceptance(self):
        state = work(issue(requires_new_search=True, search_query="原始统计范围"))
        await self.process(state, verdict(reviews=[{"issue_id": "问题收入", "status": "unresolved", "reason": "仍缺统计范围"}]))
        task = state["pending_research_tasks"][0]
        self.assertEqual(task["issue_id"], "问题收入")
        self.assertEqual(task["acceptance"], "收入为100亿元，并说明统计范围")

    async def test_critic_prompt_contains_prior_issues_writer_evidence_and_acceptance(self):
        state = work()
        record_writer_responses(state, "sec_1", CONTENT, {"issue_responses": [response()]}, TraceValidationTests.packet)
        critic = quality_tests.CriticReviewFlowTests._make_critic()
        critic.call_llm = AsyncMock(return_value=json.dumps(verdict(), ensure_ascii=False))
        await critic._review_content(state)
        prompt = critic.call_llm.await_args.kwargs["user_prompt"]
        for token in ("问题收入", "收入为100亿元，并说明统计范围", "revised_quote", ORIGINAL, "issue_reviews"):
            self.assertIn(token, prompt)


class ReviewContractTests(unittest.TestCase):
    def test_unknown_duplicate_and_closed_ids_are_rejected(self):
        cases = [[{"issue_id": "未知", "status": "resolved"}],
                 [{"issue_id": "问题收入", "status": "unresolved"}] * 2,
                 [{"issue_id": "问题收入", "status": []}],
                 [{"issue_id": [], "status": "resolved"}]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                reconcile_reviews(work(), verdict(reviews=rows))

    def test_closed_id_cannot_be_reused_for_a_new_issue(self):
        state = work(issue(review_status="resolved", resolved=True))
        ledger, active = reconcile_reviews(state, verdict(issues=[issue(description="新的不同问题")]))
        self.assertEqual(len(ledger), 2)
        self.assertNotEqual(active[0]["id"], "问题收入")
        self.assertTrue(ledger[0]["resolved"])

    def test_duplicate_new_issue_ids_cannot_silently_drop_an_issue(self):
        with self.assertRaisesRegex(ValueError, "重复问题ID"):
            reconcile_reviews(work(), verdict(issues=[issue(), issue(description="不同问题")]))

    def test_conflicting_closed_review_and_open_issue_does_not_close(self):
        state = work()
        _, active = reconcile_reviews(state, verdict(issues=[issue()], reviews=[{
            "issue_id": "问题收入", "status": "resolved", "report_quote": QUOTE, "reason": "已修复"}]))
        self.assertEqual(active[0]["id"], "问题收入")


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
        self.assertFalse(result["critic_feedback"][0]["resolved"])


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
            verdict(reviews=[{"issue_id": "问题收入", "status": "resolved", "report_quote": f"收入段：{QUOTE}（sec_1）",
                              "reason": "当前收入句已修正为100亿元，保留原文统计范围，满足该条验收。"}]),
        ])
        scout = make_scout()
        graph = build_graph(make_agents(writer=writer, critic=critic, scout=scout), InMemorySaver(),
            run_id="补搜问题图" if search else "修订问题图", user_id="user", session_id="test-session",
            execute_stage=passthrough_stage)
        config = graph_config("补搜问题图" if search else "修订问题图")
        result = (await graph.ainvoke(pack_state(state), config))["research"]
        self.assertEqual(result["completion_reason"], "passed")
        self.assertEqual(result["critic_feedback"][0]["id"], "问题收入")
        self.assertEqual(result["critic_feedback"][0]["writer_status"], "addressed")
        self.assertEqual(result["critic_feedback"][0]["review_status"], "resolved")
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
