"""三维百分制审核、真实章节定位和60分动态路由的离线回归测试。"""

from copy import deepcopy
import json
import logging
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "eval/scripts")]
from service.deep_research_v2.agents.critic import CriticMaster
from service.deep_research_v2.review_contract import (
    DIMENSIONS, QUALITY_SCORE_BASIS, normalize_quality_score, normalize_review_result,
    validate_critic_issues, bind_issue_targets, materialize_new_sections,
)
from service.deep_research_v2.graph import route_after_review, finish, pack_state
import run_deepresearch_bench as drb
from run_deepresearch_bench import valid_quality_score, display_quality_score, is_hard_pass


def issue(ident="ISSUE-001", severity="major", search=False, **overrides):
    result = {
        "issue_id": ident, "issue_type": "insufficient_depth", "severity": severity,
        "target": {"target_type": "existing_section", "chapter_id": "1",
                   "chapter_title": "年度数据", "specific_location": "收入分析段",
                   "insert_after_chapter_id": None, "proposed_chapter_title": None},
        "description": "只列收入，没有说明增长原因", "evidence": "收入段未比较各业务贡献",
        "modification": "比较各业务收入贡献，解释增长原因与不确定性",
        "requires_new_search": search, "search_query": "2026 公司年度报告 各业务收入" if search else "",
    }
    result.update(overrides)
    return result


def review(score=80, issues=None, **assessment):
    return {"overall_assessment": {"quality_score": score, "verdict": "pass",
            "dimension_scores": dict.fromkeys(DIMENSIONS, score), "summary": "说明收入增长原因", **assessment},
            "issues": deepcopy(issues or []), "strength_points": []}


class QualityScoreValidationTests(unittest.TestCase):
    def test_number_range_includes_zero_and_fraction_but_rejects_coercion(self):
        for value in (0, 59.99, 60, 100):
            self.assertEqual(normalize_quality_score(value), value)
        for value in (-1, 101, "60", True, None, float("nan"), float("inf")):
            self.assertIsNone(normalize_quality_score(value))

    def test_mean_is_exact_and_not_capped_for_critical(self):
        result = review(1, [issue(severity="critical")], dimension_scores=dict(zip(DIMENSIONS, [85, 75, 80])))
        normalize_review_result(result)
        self.assertEqual(result["overall_assessment"]["quality_score"], 80)
        self.assertEqual(result["overall_assessment"]["raw_quality_score"], 1)
        self.assertEqual(result["overall_assessment"]["verdict"], "major_issues")

    def test_decimal_rounding_and_zero(self):
        for scores, expected in (([59.99, 60, 60], 60), ([0, 0, 0], 0), ([60, 61, 62], 61)):
            result = review(dimension_scores=dict(zip(DIMENSIONS, scores)))
            normalize_review_result(result)
            self.assertEqual(result["overall_assessment"]["quality_score"], expected)

    def test_missing_or_old_dimensions_fail_without_guessed_scores(self):
        for dimensions in ({}, {"logical_coherence": 8, "requirement_coverage": 8, "freshness": 8},
                           {**dict.fromkeys(DIMENSIONS, 80), "instruction_following": 80},
                           {**dict.fromkeys(DIMENSIONS, 80), "depth": "80"}):
            with self.assertRaises(ValueError):
                normalize_review_result(review(dimension_scores=dimensions))

    def test_new_statistics_use_percentage_scale_and_explicit_basis(self):
        self.assertEqual(valid_quality_score(0), 0)
        self.assertEqual(valid_quality_score(100), 100)
        self.assertIsNone(valid_quality_score("60"))
        self.assertEqual(display_quality_score({"quality_score": 80, "quality_score_status": "derived_from_dimensions"}), "80（分维度）")
        row = {"article": "报告", "_meta": {"quality_score": 60, "quality_verdict": "pass", "quality_score_basis": QUALITY_SCORE_BASIS}}
        self.assertTrue(is_hard_pass(row))
        row["_meta"]["quality_score"] = 59.99
        self.assertFalse(is_hard_pass(row))
        row["_meta"].update(quality_score=80, quality_score_basis="report_review_3d_v2")
        self.assertFalse(is_hard_pass(row))


class CriticIssueContractTests(unittest.TestCase):
    def test_invalid_issue_fields_are_rejected(self):
        cases = [{"issue_id": ""}, {"issue_type": "logic_error"}, {"severity": "fatal"},
                 {"requires_new_search": "true"}, {"requires_new_search": True, "search_query": " "},
                 {"search_query": "未要求补搜不能带查询"}, {"modification": ""}, {"target": {}}]
        for override in cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_critic_issues(review(issues=[issue(**override)]))
        with self.assertRaises(ValueError):
            validate_critic_issues(review(issues=[issue(), issue()]))
        with self.assertRaises(ValueError):
            validate_critic_issues(review(issues=[{"id": "旧格式", "requires_new_search": False}]))

    def test_report_number_title_and_internal_id_bind_consistently(self):
        state = CriticReviewFlowTests._make_state()
        for chapter_id in ("1", "sec_7", None):
            row = issue()
            row["target"]["chapter_id"] = chapter_id
            bind_issue_targets(state, [row])
            self.assertEqual(row["resolved_section_ids"], ["sec_7"])
        for overrides in ({"chapter_id": "99"}, {"chapter_title": "不存在的标题"}):
            row = issue()
            row["target"].update(overrides)
            with self.assertRaises(ValueError):
                bind_issue_targets(state, [row])

    def test_subsection_binds_to_owning_writer_section_and_fences_are_ignored(self):
        state = CriticReviewFlowTests._make_state()
        state["final_report"] += '\n### 1.1 业务贡献\n业务分析\n```text\n## 2 假标题\n```'
        row = issue()
        row["target"].update(chapter_id="1.1", chapter_title="业务贡献")
        bind_issue_targets(state, [row])
        self.assertEqual(row["resolved_section_ids"], ["sec_7"])
        row["target"].update(chapter_id="2", chapter_title="假标题")
        with self.assertRaises(ValueError):
            bind_issue_targets(state, [row])

    def test_new_sections_share_anchor_keep_order_and_receive_distinct_ids(self):
        state = CriticReviewFlowTests._make_state()
        rows = []
        for i in range(2):
            row = issue(ident=f"新增{i}")
            row["target"] = {"target_type": "new_section", "chapter_id": None, "chapter_title": None,
                "specific_location": None, "insert_after_chapter_id": "1", "proposed_chapter_title": f"风险{i}"}
            rows.append(row)
        bind_issue_targets(state, rows)
        state["critic_feedback"] = rows
        materialize_new_sections(state)
        self.assertEqual([s["title"] for s in state["outline"]], ["年度数据", "风险0", "风险1"])
        self.assertEqual(len({r["resolved_section_ids"][0] for r in rows}), 2)
        self.assertTrue(all(s["status"] == "pending" for s in state["outline"][1:]))


class CriticReviewFlowTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _make_state(**overrides):
        state = {"phase": "reviewing", "query": "解释年度收入增长",
                 "outline": [{"id": "sec_7", "title": "年度数据", "description": "解释收入", "status": "drafted"}],
                 "draft_sections": {"sec_7": "收入分析"}, "final_report": "## 1 年度数据\n收入分析",
                 "critic_feedback": [], "iteration": 0, "max_iterations": 2}
        state.update(overrides)
        return state

    @staticmethod
    def _make_critic():
        critic = CriticMaster.__new__(CriticMaster)
        critic.name = "CriticMaster"
        critic.logger = logging.getLogger("test.critic")
        critic.add_message = Mock()
        critic.parse_json_response = json.loads
        return critic

    async def test_threshold_severity_search_and_limit_matrix(self):
        for score in (0, 59.99, 60, 100):
            for severity in (None, "minor", "major", "critical"):
                for search in (False, True):
                    for iteration in (0, 2):
                        with self.subTest(score=score, severity=severity, search=search, iteration=iteration):
                            rows = [issue(severity=severity, search=search)] if severity else []
                            critic = self._make_critic()
                            critic._review_content = AsyncMock(return_value=review(score, rows, verdict="pass"))
                            state = self._make_state(iteration=iteration)
                            await critic.process(state)
                            passed = score >= 60 and severity not in {"major", "critical"}
                            self.assertEqual(state["quality_score"], score)
                            self.assertEqual(state["quality_verdict"] == "pass", passed)
                            route = "finish" if passed or iteration == 2 else ("re_research" if rows and search else "revise")
                            self.assertEqual(route_after_review(pack_state(state)), route)
                            if route == "re_research":
                                self.assertEqual(state["pending_research_tasks"][0]["target_section"], "sec_7")
                            else:
                                self.assertEqual(state["pending_research_tasks"], [])
                            if route == "finish":
                                completed = finish(pack_state(state))["research"]
                                self.assertEqual(completed["completion_reason"], "passed" if passed else "revision_limit")

    async def test_latest_review_replaces_old_issues_and_clears_search_cache(self):
        critic = self._make_critic()
        critic._review_content = AsyncMock(return_value=review(60))
        old = issue()
        state = self._make_state(quality_score_basis=QUALITY_SCORE_BASIS, critic_feedback=[old],
                                issue_search_results={"ISSUE-001": {"status": "sources_collected"}})
        await critic.process(state)
        self.assertEqual(state["critic_feedback"], [])
        self.assertEqual(state["critic_history"], [[old]])
        self.assertEqual(state["issue_search_results"], {})
        self.assertEqual(state["quality_verdict"], "pass")

    async def test_only_three_complete_inputs_and_no_recursive_placeholder_expansion(self):
        state = self._make_state(query="用户原话保留 {{report}}", final_report="最新完整报告" * 6000,
                                data_points=["不应发送的数据点"], critic_feedback=["不应发送的旧问题"],
                                research_contract={"secret": "不应发送的契约"})
        critic = self._make_critic()
        async def respond(**kwargs):
            from service.agent_context import is_isolated_agent_input
            self.assertTrue(is_isolated_agent_input())
            return json.dumps(review())
        critic.call_llm = AsyncMock(side_effect=respond)
        await critic._review_content(state)
        prompt = critic.call_llm.await_args.kwargs["user_prompt"]
        self.assertIn(state["query"], prompt)
        self.assertIn(state["final_report"], prompt)
        for forbidden in ("不应发送的数据点", "不应发送的旧问题", "不应发送的契约", "issue_reviews"):
            self.assertNotIn(forbidden, prompt)
        self.assertEqual(critic.call_llm.await_count, 1)
        from service.agent_context import is_isolated_agent_input
        self.assertFalse(is_isolated_agent_input())

    async def test_invalid_dimensions_abort_without_fabricated_fallback_score(self):
        critic = self._make_critic()
        critic.call_llm = AsyncMock(return_value=json.dumps(review(dimension_scores={})))
        with self.assertRaises(ValueError):
            await critic._review_content(self._make_state())
        self.assertEqual(critic.call_llm.await_count, 1)

    async def test_all_explicit_search_tasks_are_retained_beyond_ten(self):
        critic = self._make_critic()
        critic._review_content = AsyncMock(return_value=review(50, [issue(ident=f"问题{i}", search=True) for i in range(12)]))
        state = self._make_state()
        await critic.process(state)
        self.assertEqual(len(state["pending_research_tasks"]), 12)

    async def test_rule_issue_does_not_reduce_mean_and_requires_revision(self):
        critic = self._make_critic()
        critic._review_content = AsyncMock(return_value=review(95))
        state = self._make_state(allowed_domains=["official.example"], final_report="## 1 年度数据\n[数据](https://other.example/report)")
        await critic.process(state)
        self.assertEqual(state["quality_score"], 95)
        self.assertEqual(state["quality_verdict"], "major_issues")
        self.assertEqual(state["phase"], "re_researching")


class RunRetentionTests(unittest.TestCase):
    def test_new_runs_include_question_count_and_increment_from_legacy_history(self):
        with tempfile.TemporaryDirectory() as temporary_dir, \
                patch.object(drb, "RAW_DIR", temporary_dir):
            legacy = Path(temporary_dir) / "trial-5.jsonl"
            with legacy.open("w", encoding="utf-8") as handle:
                for question_id in range(1, 6):
                    handle.write(json.dumps({"id": question_id, "article": "ok"}) + "\n")

            second = drb.reserve_run_name(
                "industry-assistant-v1", 5, "20260820_160000"
            )
            third = drb.reserve_run_name(
                "industry-assistant-v1", 5, "20260820_160001"
            )

            self.assertEqual(second, "industry-ass-5q-run002-20260820_160000")
            self.assertEqual(third, "industry-ass-5q-run003-20260820_160001")
            self.assertTrue(Path(drb.output_path(second)).exists())
            self.assertTrue(Path(drb.output_path(third)).exists())
            self.assertLessEqual(len(drb.make_session_id(second, "20260820_160000", "100")), 64)

    def test_report_name_uses_chinese_question_count_and_run_number(self):
        with tempfile.TemporaryDirectory() as temporary_dir, \
                tempfile.TemporaryDirectory() as docs_dir, \
                patch.object(drb, "RAW_DIR", temporary_dir), \
                patch.object(drb, "DOCS_DIR", docs_dir), \
                patch.object(drb, "_fetch_trace_stats", return_value=None):
            run_name = drb.reserve_run_name("test", 1, "20260820_160000")
            drb.append_result(run_name, {
                "id": 1,
                "prompt": "test",
                "article": "answer",
                "_meta": {
                    "article_chars": 6,
                    "references_count": 0,
                    "duration_s": 1,
                    "quality_score": 5,
                },
            })

            report = drb.write_report(run_name, "2026-08-20 16:00")

            self.assertIn("1题-第1次", Path(report).name)
            self.assertTrue(Path(report).exists())



if __name__ == "__main__":
    unittest.main()
