"""集中式评估目录中的 CriticMaster 评分约束与 DRB 统计回归测试。"""

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
SCRIPTS_DIR = BACKEND_DIR / "eval" / "scripts"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from service.deep_research_v2.agents.critic import (  # noqa: E402
    CriticMaster,
    derive_quality_score,
    normalize_quality_score,
    validate_critic_issues,
)
from service.deep_research_v2.quality_pipeline import (  # noqa: E402
    deterministic_critic_issues,
)
from service.deep_research_v2.state import ResearchPhase  # noqa: E402
from run_deepresearch_bench import (  # noqa: E402
    display_quality_score,
    is_hard_pass,
    valid_quality_score,
)
import run_deepresearch_bench as drb  # noqa: E402


class CriticIssueContractTests(unittest.TestCase):
    def setUp(self):
        self.critic = CriticMaster.__new__(CriticMaster)

    def test_explicit_new_search_is_independent_of_issue_type_and_ratio(self):
        tasks = self.critic._build_research_tasks({
            "issues": [
                {
                    "id": "核验原始数据",
                    "target_section": "sec_2",
                    "issue_type": "unverified/hallucination",
                    "severity": "minor",
                    "requires_new_search": True,
                    "search_query": "  Brand Finance official Ping An 2024  ",
                    "acceptance_criteria": "核对官方原始数据的年份和统计口径",
                },
                *[
                    {
                        "issue_type": "logic_error",
                        "severity": "major",
                        "requires_new_search": False,
                    }
                    for _ in range(9)
                ],
            ],
        })

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["issue_id"], "核验原始数据")
        self.assertEqual(tasks[0]["target_section"], "sec_2")
        self.assertEqual(tasks[0]["query"], "Brand Finance official Ping An 2024")
        self.assertEqual(
            tasks[0]["acceptance"],
            "核对官方原始数据的年份和统计口径",
        )

    def test_false_never_triggers_search_even_with_query_or_missing_aspects(self):
        tasks = self.critic._build_research_tasks({
            "issues": [{
                "issue_type": "missing_source/incomplete/outdated",
                "severity": "critical",
                "requires_new_search": False,
                "search_query": "已有来源可补充引用格式，无需再次检索",
            }],
            "missing_aspects": ["市场规模", "公司年报"],
        })

        self.assertEqual(tasks, [])
        self.assertEqual(
            self.critic._build_research_tasks({
                "issues": [],
                "missing_aspects": ["仅列出遗漏方面不能产生补搜任务"],
            }),
            [],
        )

    def test_query_is_preserved_as_one_explicit_task_without_splitting(self):
        query = (
            '"Ping An 2019 annual report total assets";\n'
            '"AIA 2023 annual report financial summary"'
        )
        tasks = self.critic._build_research_tasks({
            "issues": [{
                "id": "issue_1",
                "requires_new_search": True,
                "search_query": query,
            }],
        })

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["query"], query)

    def test_empty_issues_and_false_without_query_are_valid(self):
        validate_critic_issues({"issues": []})
        validate_critic_issues({"issues": [{"requires_new_search": False}]})
        self.assertEqual(
            self.critic._build_research_tasks({
                "issues": [{"requires_new_search": False}],
            }),
            [],
        )

    def test_rejects_malformed_result_or_issues_container(self):
        for result in (
            None,
            [],
            "not an object",
            {},
            {"issues": None},
            {"issues": {}},
            {"issues": "not a list"},
            {"issues": ()},
            {"issues": [None]},
            {"issues": ["not an issue object"]},
            {"issues": [[]]},
        ):
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    validate_critic_issues(result)

    def test_requires_new_search_must_be_present_and_strictly_boolean(self):
        issues = [{"search_query": "公司年报"}]
        issues.extend({"requires_new_search": value, "search_query": "公司年报"}
                      for value in (None, "true", "false", 1, 0, [], {}))
        for issue in issues:
            with self.subTest(issue=issue):
                with self.assertRaises(ValueError):
                    validate_critic_issues({"issues": [issue]})

    def test_true_requires_nonempty_string_query_without_text_fallbacks(self):
        issues = [{"requires_new_search": True}]
        issues.extend({"requires_new_search": True, "search_query": value}
                      for value in (None, "", " \t\n", 1, False, [], {}))
        for issue in issues:
            issue.update({
                "suggestion": "查找公司官网年度报告核验总资产",
                "description": "这两段文字都不能代替缺失的 search_query",
            })
            with self.subTest(issue=issue):
                with self.assertRaises(ValueError):
                    self.critic._build_research_tasks({"issues": [issue]})

    def test_task_limit_preserves_first_ten_explicit_issues_and_their_identity(self):
        issues = [{"requires_new_search": False}]
        issues.extend({
            "id": f"issue_{index}",
            "target_section": f"sec_{index}",
            "requires_new_search": True,
            "search_query": f"查询 {index}",
        } for index in range(12))

        tasks = self.critic._build_research_tasks({"issues": issues})

        self.assertEqual(len(tasks), 10)
        self.assertEqual(
            [(task["issue_id"], task["target_section"], task["query"])
             for task in tasks],
            [(f"issue_{index}", f"sec_{index}", f"查询 {index}")
             for index in range(10)],
        )

    def test_deterministic_gate_issues_produce_queries_for_the_current_question(self):
        query = "甲公司2025年营收"
        issues = deterministic_critic_issues({
            "query": query,
            "facts": [],
            "data_points": [],
            "references": [],
            "allowed_domains": ["official.example.cn"],
            "final_report": "甲公司2025年营收[来源](https://outside.example.cn/report)。",
        })

        tasks = self.critic._build_research_tasks({"issues": issues})

        self.assertTrue(issues)
        self.assertTrue(tasks)
        self.assertEqual(len(tasks), len(issues))
        for task, issue in zip(tasks, issues):
            with self.subTest(issue_id=issue["id"]):
                self.assertEqual(task["issue_id"], issue["id"])
                self.assertIn(query, task["query"])
                self.assertTrue(task["query"].strip())

    def test_invalid_issue_after_ten_tasks_is_still_rejected(self):
        issues = [{
            "requires_new_search": True,
            "search_query": f"查询 {index}",
        } for index in range(10)]
        issues.append({"requires_new_search": "true", "search_query": "不能截断逃过校验"})

        with self.assertRaises(ValueError):
            self.critic._build_research_tasks({"issues": issues})


class QualityScoreValidationTests(unittest.TestCase):
    def test_critic_accepts_only_integer_scores_in_range(self):
        self.assertEqual(normalize_quality_score(1), 1)
        self.assertEqual(normalize_quality_score("7"), 7)
        self.assertEqual(normalize_quality_score(10.0), 10)

        for invalid in (-1, 0, 7.5, 11, None, True, "无法评分", float("inf")):
            with self.subTest(invalid=invalid):
                self.assertIsNone(normalize_quality_score(invalid))

    def test_review_normalization_repairs_inconsistent_pass(self):
        result = {
            "overall_assessment": {
                "quality_score": "4",
                "verdict": "pass",
            }
        }

        self.assertTrue(CriticMaster._normalize_review_result(result))
        self.assertEqual(result["overall_assessment"]["quality_score"], 4)
        self.assertEqual(result["overall_assessment"]["verdict"], "needs_revision")
        self.assertEqual(result["overall_assessment"]["score_status"], "valid")

    def test_drb_statistics_reject_historical_sentinel_scores(self):
        self.assertIsNone(valid_quality_score(-1))
        self.assertIsNone(valid_quality_score(0))
        self.assertEqual(valid_quality_score(4), 4.0)
        self.assertEqual(display_quality_score({"quality_score": -1}), "无效(-1)")
        self.assertEqual(
            display_quality_score({
                "quality_score": 1,
                "quality_score_status": "fallback_invalid_model_output",
            }),
            "1（降级）",
        )
        self.assertEqual(
            display_quality_score({
                "quality_score": 6,
                "quality_score_status": "derived_from_dimensions",
            }),
            "6（分维度）",
        )

    def test_dimension_scores_are_deterministically_aggregated(self):
        result = {
            "overall_assessment": {
                "quality_score": -1,
                "verdict": "needs_revision",
                "dimension_scores": {
                    "logical_coherence": 7,
                    "requirement_coverage": 9,
                    "freshness": 5,
                },
            },
            "issues": [{"severity": "major"}],
        }

        derived = derive_quality_score(result)
        self.assertEqual(derived["score"], 6)  # needs_revision 最高 6 分
        self.assertEqual(derived["status"], "derived_from_dimensions")
        self.assertTrue(CriticMaster._normalize_review_result(result))
        self.assertEqual(result["overall_assessment"]["quality_score"], 6)
        self.assertEqual(
            result["overall_assessment"]["score_status"],
            "derived_from_dimensions",
        )


class RunRetentionTests(unittest.TestCase):
    def test_hard_pass_requires_native_pass_verdict_and_score_at_least_seven(self):
        base = {"article": "answer", "_meta": {"quality_score": 7}}
        self.assertFalse(is_hard_pass(base))
        base["_meta"]["quality_verdict"] = "needs_revision"
        self.assertFalse(is_hard_pass(base))
        base["_meta"]["quality_verdict"] = "pass"
        self.assertTrue(is_hard_pass(base))
        base["_meta"]["quality_score"] = 6
        self.assertFalse(is_hard_pass(base))

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


class CriticReviewFlowTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _make_state(**overrides):
        state = {
            "phase": ResearchPhase.REVIEWING.value,
            "query": "核验公司年度报告数据",
            "outline": [{"id": "sec_1", "title": "年度数据", "status": "done"}],
            "final_report": "待审核的完整报告",
            "facts": [],
            "data_points": [],
            "references": [],
            "critic_feedback": [],
            "iteration": 0,
            "max_iterations": 2,
        }
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

    async def test_llm_schema_errors_do_not_enter_score_repair(self):
        malformed_results = [
            {"overall_assessment": {"quality_score": 8, "verdict": "pass"}},
            {"issues": None},
            {"issues": [{}]},
            {"issues": [{"requires_new_search": "true", "search_query": "年报"}]},
            {"issues": [{"requires_new_search": True, "search_query": " "}]},
        ]
        for result in malformed_results:
            with self.subTest(result=result):
                critic = self._make_critic()
                result.setdefault("overall_assessment", {
                    "quality_score": -1,
                    "verdict": "pass",
                })
                critic.call_llm = AsyncMock(return_value=json.dumps(result))

                with self.assertRaises(ValueError):
                    await critic._review_content(self._make_state())

                critic.call_llm.assert_awaited_once()

    async def test_valid_issue_schema_allows_score_only_repair_response(self):
        critic = self._make_critic()
        critic.call_llm = AsyncMock(side_effect=[
            json.dumps({
                "overall_assessment": {
                    "quality_score": -1,
                    "verdict": "needs_revision",
                    "summary": "现有摘要需保留，只修复评分",
                },
                "issues": [],
            }),
            json.dumps({
                "overall_assessment": {
                    "quality_score": 4,
                    "verdict": "needs_revision",
                },
            }),
        ])

        result = await critic._review_content(self._make_state())

        self.assertEqual(critic.call_llm.await_count, 2)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["overall_assessment"]["quality_score"], 4)
        self.assertEqual(result["overall_assessment"]["score_status"], "valid_after_retry")
        self.assertEqual(result["overall_assessment"]["summary"], "现有摘要需保留，只修复评分")

    async def test_failed_score_repair_keeps_original_research_issue_fields(self):
        critic = self._make_critic()
        issue = {
            "id": "待补证据",
            "target_section": "sec_1",
            "severity": "major",
            "requires_new_search": True,
            "search_query": "甲公司2025年营收 年度报告",
        }
        critic.call_llm = AsyncMock(side_effect=[
            json.dumps({
                "overall_assessment": {
                    "quality_score": -1,
                    "verdict": "needs_revision",
                    "summary": "保留首轮审核结论和补搜任务",
                },
                "issues": [issue],
            }),
            json.dumps({
                "overall_assessment": {
                    "quality_score": "仍然无法评分",
                    "verdict": "needs_revision",
                },
            }),
        ])

        # 非空 issues 正常可以派生总分；注入派生不可用故障，覆盖最终降级分支。
        with patch(
            "service.deep_research_v2.agents.critic.derive_quality_score",
            return_value=None,
        ):
            result = await critic._review_content(self._make_state())

        self.assertEqual(critic.call_llm.await_count, 2)
        self.assertEqual(result["overall_assessment"]["quality_score"], 1)
        self.assertEqual(
            result["overall_assessment"]["score_status"],
            "fallback_invalid_model_output",
        )
        self.assertEqual(result["issues"], [issue])
        tasks = critic._build_research_tasks(result)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["issue_id"], issue["id"])
        self.assertEqual(tasks[0]["target_section"], issue["target_section"])
        self.assertEqual(tasks[0]["query"], issue["search_query"])

    async def test_process_validates_before_pass_or_iteration_limit_can_complete(self):
        cases = [
            ("pass", 0, None),
            ("pass", 0, [{"requires_new_search": "false"}]),
            ("needs_revision", 2, [{"requires_new_search": True, "search_query": " "}]),
        ]
        for verdict, iteration, issues in cases:
            with self.subTest(verdict=verdict, iteration=iteration, issues=issues):
                critic = self._make_critic()
                result = {"overall_assessment": {
                    "quality_score": 8 if verdict == "pass" else 4,
                    "verdict": verdict,
                }}
                if issues is not None:
                    result["issues"] = issues
                critic._review_content = AsyncMock(return_value=result)
                state = self._make_state(iteration=iteration)

                with patch(
                    "service.deep_research_v2.agents.critic.deterministic_critic_issues",
                    return_value=[],
                ), self.assertRaises(ValueError):
                    await critic.process(state)

                self.assertEqual(state["phase"], ResearchPhase.REVIEWING.value)
                self.assertEqual(state["iteration"], iteration)
                self.assertNotIn("quality_verdict", state)
                self.assertFalse(state.get("pending_research_tasks"))

    async def test_process_validates_rule_issues_as_well_as_model_issues(self):
        critic = self._make_critic()
        critic._review_content = AsyncMock(return_value={
            "overall_assessment": {"quality_score": 8, "verdict": "pass"},
            "issues": [],
        })
        state = self._make_state()
        invalid_rule = {
            "issue_type": "missing_source",
            "severity": "major",
            "requires_new_search": True,
        }

        with patch(
            "service.deep_research_v2.agents.critic.deterministic_critic_issues",
            return_value=[invalid_rule],
        ), self.assertRaises(ValueError):
            await critic.process(state)

        self.assertEqual(state["phase"], ResearchPhase.REVIEWING.value)
        self.assertEqual(state["critic_feedback"], [])

    async def test_process_tasks_reference_saved_issue_ids_including_generated_ids(self):
        critic = self._make_critic()
        critic._review_content = AsyncMock(return_value={
            "overall_assessment": {"quality_score": 6, "verdict": "needs_revision"},
            "issues": [
                {
                    "id": "原始问题",
                    "target_section": "sec_1",
                    "issue_type": "unverified",
                    "severity": "minor",
                    "requires_new_search": True,
                    "search_query": "  公司年度报告  ",
                },
                {
                    "id": "仅修订文字",
                    "issue_type": "missing_source",
                    "severity": "major",
                    "requires_new_search": False,
                    "search_query": "不得补搜这一条",
                },
                {
                    "target_section": "sec_2",
                    "issue_type": "bias",
                    "severity": "minor",
                    "requires_new_search": True,
                    "search_query": "公司年度报告",
                },
            ],
        })
        state = self._make_state()

        with patch(
            "service.deep_research_v2.agents.critic.deterministic_critic_issues",
            return_value=[],
        ):
            returned = await critic.process(state)

        self.assertIs(returned, state)
        self.assertEqual(state["phase"], ResearchPhase.RE_RESEARCHING.value)
        self.assertEqual(state["iteration"], 1)
        tasks = state["pending_research_tasks"]
        feedback = state["critic_feedback"]
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["issue_id"], feedback[0]["id"])
        self.assertEqual(tasks[0]["issue_id"], "原始问题")
        self.assertTrue(feedback[2]["id"])
        self.assertEqual(tasks[1]["issue_id"], feedback[2]["id"])
        self.assertNotEqual(tasks[0]["issue_id"], tasks[1]["issue_id"])
        self.assertEqual([task["target_section"] for task in tasks], ["sec_1", "sec_2"])
        self.assertEqual([task["query"] for task in tasks], ["公司年度报告"] * 2)
        self.assertNotIn("pending_search_queries", state)

    async def test_explicit_search_cannot_be_swallowed_by_pass_verdict(self):
        for iteration in (0, 2):
            with self.subTest(iteration=iteration):
                critic = self._make_critic()
                critic._review_content = AsyncMock(return_value={
                    "overall_assessment": {"quality_score": 9, "verdict": "pass"},
                    "issues": [{
                        "id": "仍需核验",
                        "severity": "minor",
                        "requires_new_search": True,
                        "search_query": "甲公司2025年营收 年度报告",
                    }],
                })
                state = self._make_state(iteration=iteration)

                with patch(
                    "service.deep_research_v2.agents.critic.deterministic_critic_issues",
                    return_value=[],
                ):
                    await critic.process(state)

                self.assertEqual(state["quality_verdict"], "needs_revision")
                self.assertEqual(state["quality_score"], 6)
                self.assertFalse(state["critic_feedback"][0]["resolved"])
                review_messages = [
                    call.args[2]
                    for call in critic.add_message.call_args_list
                    if call.args[1] == "review"
                ]
                self.assertEqual(review_messages[0]["verdict"], "needs_revision")
                self.assertEqual(review_messages[0]["quality_score"], 6)
                if iteration == 0:
                    self.assertEqual(state["phase"], ResearchPhase.RE_RESEARCHING.value)
                    self.assertEqual(state["iteration"], 1)
                    self.assertEqual(
                        state["pending_research_tasks"][0]["issue_id"],
                        "仍需核验",
                    )
                else:
                    self.assertEqual(state["phase"], ResearchPhase.COMPLETED.value)
                    self.assertEqual(state["iteration"], 2)
                    self.assertFalse(state["pending_research_tasks"])
                    self.assertTrue(any(
                        call.args[1] == "warning"
                        for call in critic.add_message.call_args_list
                    ))

    async def test_process_false_only_feedback_routes_to_revision(self):
        critic = self._make_critic()
        critic._review_content = AsyncMock(return_value={
            "overall_assessment": {"quality_score": 5, "verdict": "needs_revision"},
            "issues": [{
                "issue_type": "missing_source",
                "severity": "major",
                "requires_new_search": False,
                "search_query": "即使提供查询，也只按明确布尔决策路由",
            }],
            "missing_aspects": ["遗漏方面不能独立创建任务"],
        })
        state = self._make_state()

        with patch(
            "service.deep_research_v2.agents.critic.deterministic_critic_issues",
            return_value=[],
        ):
            await critic.process(state)

        self.assertEqual(state["phase"], ResearchPhase.REVISING.value)
        self.assertEqual(state["iteration"], 1)
        self.assertFalse(state.get("pending_research_tasks"))
        self.assertNotIn("pending_search_queries", state)

    async def test_reviews_latest_full_report_and_derives_invalid_score(self):
        critic = CriticMaster.__new__(CriticMaster)
        critic.logger = logging.getLogger("test.critic")
        critic.call_llm = AsyncMock(return_value=json.dumps({
                "overall_assessment": {
                    "quality_score": -1,
                    "verdict": "major_issues",
                    "summary": "无法评分",
                },
                "issues": [{
                    "severity": "critical",
                    "description": "证据不足",
                    "requires_new_search": True,
                    "search_query": "测试报告关键数据的原始来源",
                }],
            }, ensure_ascii=False))
        critic.parse_json_response = lambda response: json.loads(response)

        state = {
            "query": "测试问题",
            "outline": [{"id": "sec_1", "title": "章节", "status": "done"}],
            "draft_sections": {"sec_1": "OLD_DRAFT_SHOULD_NOT_BE_REVIEWED"},
            "final_report": "LATEST_REPORT_" + "完整内容" * 3000,
            "facts": [],
            "data_points": [],
            "references": [],
        }

        result = await critic._review_content(state)

        self.assertEqual(critic.call_llm.await_count, 1)
        first_prompt = critic.call_llm.await_args_list[0].kwargs["user_prompt"]
        self.assertIn("LATEST_REPORT_", first_prompt)
        self.assertIn("完整内容" * 3000, first_prompt)
        self.assertNotIn("OLD_DRAFT_SHOULD_NOT_BE_REVIEWED", first_prompt)
        self.assertEqual(result["overall_assessment"]["quality_score"], 4)
        self.assertEqual(result["overall_assessment"]["raw_quality_score"], -1)
        self.assertEqual(result["issues"][0]["description"], "证据不足")
        self.assertEqual(
            result["overall_assessment"]["score_status"],
            "derived_from_issue_severity",
        )


if __name__ == "__main__":
    unittest.main()
