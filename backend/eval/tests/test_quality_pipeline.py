"""深度研究证据门禁、引用清理与按章修订回归测试。"""

import json
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
sys.path.insert(0, str(APP_DIR))

from service.deep_research_v2.agents.scout import (  # noqa: E402
    DeepScout,
    expand_search_queries,
)
from service.deep_research_v2.agents.writer import (  # noqa: E402
    InsufficientEvidenceError,
    LeadWriter,
)
from service.deep_research_v2.deterministic_analysis import (  # noqa: E402
    calculate_cagr,
    run_deterministic_analysis,
)
from service.deep_research_v2.quality_pipeline import (  # noqa: E402
    bind_verified_evidence,
    build_task_context,
    evaluate_report_gate,
)
from service.deep_research_v2.evidence import build_evidence_view
from service.deep_research_v2.state import ResearchPhase  # noqa: E402


def make_writer():
    writer = LeadWriter.__new__(LeadWriter)
    writer.name = "LeadWriter"
    writer.logger = logging.getLogger("test.writer")
    return writer


class EvidenceGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_zero_sources_stops_formal_report_generation(self):
        writer = make_writer()
        # 进度事件直接发往图的 custom 流。
        emitted = []
        state = {
            "facts": [], "errors": [],
            "allowed_domains": ["example.gov.cn"],
            "_emit_event": emitted.append,
        }

        with self.assertRaises(InsufficientEvidenceError):
            await writer._write_report(state)

        self.assertIn("未获得任何可用原文", state["errors"][0])
        self.assertEqual(emitted[0]["content"]["error_code"], "insufficient_evidence")

    def test_unknown_and_placeholder_links_are_not_kept(self):
        writer = make_writer()
        known = {"https://official.example.cn/report"}
        content = (
            "[official](https://official.example.cn/report) "
            "[fake](https://example.com/a) "
            "[invented](https://unknown.example.cn/x)"
        )

        cleaned = writer._sanitize_citation_links(content, known)

        self.assertIn("[official](https://official.example.cn/report)", cleaned)
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("https://unknown.example.cn", cleaned)
        self.assertEqual(cleaned.count("来源待核验"), 2)

class SectionRevisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_revision_preserves_unaffected_sections_and_full_report(self):
        writer = make_writer()
        old_section = "旧内容" * 200
        untouched = "第二章保留内容" * 100
        source_url = "https://official.example.cn/report"
        revised = (
            "官方报告显示总资产达到 100 亿元"
            f"（[官方报告]({source_url})）。" + "新内容" * 200
        )
        writer.call_llm = AsyncMock(side_effect=[
            json.dumps({
                "revised_content": revised,
                "addressed_issues": ["issue_a"],
                "unable_to_address": [],
                "citations": [],
            }, ensure_ascii=False),
            json.dumps({
                "executive_summary": "摘要",
                "conclusions": ["结论"],
                "outlook": "展望",
                "references": [],
            }, ensure_ascii=False),
        ])
        writer.parse_json_response = lambda response: json.loads(response)
        state = {
            "query": "test",
            "phase": ResearchPhase.REVISING.value,
            "allowed_domains": ["official.example.cn"],
            "errors": [],
            "outline": [
                {"id": "sec_1", "title": "第一章", "status": "drafted"},
                {"id": "sec_2", "title": "第二章", "status": "drafted"},
            ],
            "draft_sections": {"sec_1": old_section, "sec_2": untouched},
            "final_report": "OLD_FULL_REPORT",
            "facts": [{
                "content": "补充事实",
                "source_name": "官方",
                "source_url": source_url,
                "related_sections": ["sec_1"],
            }],
            "raw_sources": [{
                "source_url": source_url,
                "content": "官方报告显示该公司总资产达到 100 亿元。",
            }],
            "evidence_ledger": [{
                "source_url": source_url,
                "reader_verified": True,
                "excerpt": "官方报告显示该公司总资产达到 100 亿元。",
            }],
            "quality_gates": {},
            "references": [],
            "critic_feedback": [{
                "issue_id": "issue_a",
                "resolved_section_ids": ["sec_1"],
                "target": {"chapter_id": "sec_1"},
                "evidence": "原文比较",
                "requires_new_search": False,
                "severity": "major",
                "description": "证据不足",
                "modification": "补充官方来源",
                "resolved": False,
            }],
            "messages": [],
        }

        await writer._revise_report(state)

        self.assertEqual(state["draft_sections"]["sec_1"], revised)
        self.assertEqual(state["draft_sections"]["sec_2"], untouched)
        self.assertIn(revised, state["final_report"])
        self.assertIn(untouched, state["final_report"])
        self.assertNotEqual(state["final_report"], revised)
        self.assertEqual(state["critic_feedback"][0]["issue_id"], "issue_a")
        self.assertEqual(state["writer_revision_records"][0]["status"], "not_reported")
        # 旧 addressed_issues 自报不提供定位和依据，不能关闭审核问题。
        self.assertTrue(state["quality_gates"]["report"]["passed"])
        # 已移除引用核验：修订只剩"逐章改写 + 首尾合成"两次模型调用。
        self.assertEqual(writer.call_llm.await_count, 2)


class ScoutCoverageTests(unittest.IsolatedAsyncioTestCase):
    def test_outline_search_string_is_split_into_independent_queries(self):
        combined = (
            "全球保险公司综合实力排名 2023”， “Fortune Global 500 insurance 2023”， "
            "“Brand Finance insurance brands 2023”， “世界十大保险公司排名”"
        )

        self.assertEqual(expand_search_queries([combined]), [
            "全球保险公司综合实力排名 2023",
            "Fortune Global 500 insurance 2023",
            "Brand Finance insurance brands 2023",
            "世界十大保险公司排名",
        ])

    async def test_supplementary_extraction_receives_exact_urls_and_dates(self):
        from service.web_document import index_document
        scout = DeepScout.__new__(DeepScout)
        scout.name = "DeepScout"
        scout.logger = logging.getLogger("test.scout")
        scout.call_llm = AsyncMock(return_value='{"extracted_facts": []}')
        scout.parse_json_response = lambda response: {"extracted_facts": []}

        await scout._analyze_supplementary_results(
            "比较保险公司",
            "Ping An annual report",
            [{
                "title": "2023 Annual Report",
                "url": "https://group.pingan.com/resource/annual-report-2023.pdf",
                "site_name": "Ping An",
                "date": "2024-03-21",
                "summary": "The group published its financial results.",
                "full_content": "The group published its financial results.",
                "reader_verified": True,
                "body_quality": {"status": "passed"},
                **index_document("The group published its financial results.",
                                 "https://group.pingan.com/resource/annual-report-2023.pdf"),
            }],
        )

        prompt = scout.call_llm.await_args.kwargs["user_prompt"]
        self.assertIn(
            "https://group.pingan.com/resource/annual-report-2023.pdf",
            prompt,
        )
        self.assertIn("2024-03-21", prompt)
        self.assertIn("必须逐字复制", prompt)

    async def test_all_pending_sections_are_researched_in_batches(self):
        scout = DeepScout.__new__(DeepScout)
        scout.name = "DeepScout"
        scout.logger = logging.getLogger("test.scout")
        scout._research_section = AsyncMock()
        scout._emit_search_results_event = lambda state: None
        scout.web_reader = None
        state = {
            "phase": ResearchPhase.RESEARCHING.value,
            "allowed_domains": ["example.gov.cn"],
            "outline": [
                {"id": f"sec_{i}", "title": str(i), "status": "pending"}
                for i in range(1, 8)
            ],
            "facts": [],
            "messages": [],
        }

        await scout.process(state)

        self.assertEqual(scout._research_section.await_count, 7)

    async def test_section_search_budget_prevents_recursive_query_explosion(self):
        scout = DeepScout.__new__(DeepScout)
        scout.name = "DeepScout"
        scout.logger = logging.getLogger("test.scout")
        scout._max_queries_per_section = 3
        scout._max_deep_queries_per_section = 1
        scout.add_message = lambda *args, **kwargs: None

        async def search(query, count=10, **kwargs):
            return [{
                "title": query,
                "url": f"https://example.gov.cn/{abs(hash(query))}",
                "site_name": "官方",
            }]

        async def enrich(state, results, section_id, limit):
            for result in results:
                result["reader_verified"] = True
                result["full_content"] = "已验证正文" * 300

        scout._execute_search = AsyncMock(side_effect=search)
        scout._enrich_results_with_full_text = AsyncMock(side_effect=enrich)
        scout._analyze_search_results = AsyncMock(return_value={
            "extracted_facts": [],
            "entities_discovered": [],
            "hypothesis_evidence": [],
            "key_insights": [],
            "source_tracing_queries": ["官方来源一", "官方来源二"],
            "follow_up_queries": ["普通线索一", "普通线索二"],
        })
        scout._execute_deep_search = AsyncMock()
        state = {
            "query": "复杂研究题",
            "search_web": True,
            "search_local": False,
            "hypotheses": [],
            "facts": [],
            "data_points": [],
            "insights": [],
            "iteration": 0,
            "max_iterations": 2,
            "messages": [],
        }
        section = {
            "id": "sec_1",
            "title": "章节",
            "search_queries": ["查询一", "查询二", "查询三", "查询四"],
        }

        await scout._research_section(state, section)

        self.assertEqual(scout._execute_search.await_count, 2)
        self.assertEqual([call.args[0] for call in scout._execute_search.await_args_list], ["查询一", "查询二"])
        # 原文直写已取消事实分析驱动的自动追搜，不因旧分析字段递归扩张。
        scout._analyze_search_results.assert_not_awaited()
        scout._execute_deep_search.assert_not_awaited()

    async def test_report_search_budget_blocks_the_twenty_first_search(self):
        scout = DeepScout.__new__(DeepScout)
        scout.name = "DeepScout"
        scout.logger = logging.getLogger("test.scout")
        scout._max_total_searches = 20
        scout.tool_definitions = {"web_search": object()}
        scout.tool_runtime = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(value=[]))
        )
        emitted = []
        scout.add_message = lambda state, event_type, content: emitted.append({
            "type": event_type,
            "content": content,
        })
        # 预算按"实际发出的一次搜索请求"计数；单个允许网站时一次入口调用
        # 正好发出一次请求。
        state = {"allowed_domains": ["example.gov.cn"]}

        for index in range(21):
            await scout._execute_search(f"查询{index}", state=state)

        self.assertEqual(scout.tool_runtime.execute.await_count, 20)
        self.assertEqual(state["search_budget"]["limit"], 20)
        self.assertEqual(state["search_budget"]["used"], 20)
        self.assertEqual(state["search_budget"]["remaining"], 0)
        self.assertTrue(state["search_budget"]["exhausted"])
        self.assertEqual(state["search_budget"]["skipped"], 1)
        self.assertEqual(len(state["search_budget"]["queries"]), 20)

    async def test_report_search_budget_preserves_lower_experiment_limit(self):
        scout = DeepScout.__new__(DeepScout)
        scout.name = "DeepScout"
        scout.logger = logging.getLogger("test.scout")
        scout._max_total_searches = 20
        scout.tool_definitions = {"web_search": object()}
        scout.tool_runtime = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(value=[]))
        )
        scout.add_message = lambda state, event_type, content: None
        state = {
            "messages": [],
            "allowed_domains": ["example.gov.cn"],
            "search_budget": {"limit": 12, "used": 0},
        }

        for index in range(13):
            await scout._execute_search(f"查询{index}", state=state)

        self.assertEqual(scout.tool_runtime.execute.await_count, 12)
        self.assertEqual(state["search_budget"]["limit"], 12)
        self.assertEqual(state["search_budget"]["used"], 12)
        self.assertEqual(state["search_budget"]["skipped"], 1)


class IndustrialQualityPipelineTests(unittest.TestCase):
    def test_task_context_ignores_generated_quantitative_requirements(self):
        context = build_task_context({"research_contract": {
            "required_metrics": [{"name": "总资产", "entity_scope": "公司", "years": "2024"}],
            "required_entities": ["模型自选公司"],
            "active_constraints": {"max_body_chars": 2000},
        }}, "比较公司实力")
        self.assertEqual(context["query"], "比较公司实力")
        self.assertEqual(context["active_constraints"], {"max_body_chars": 2000})
        for key in ("required_metrics", "required_entities", "critical_sections", "acceptance"):
            self.assertNotIn(key, context)

    def test_search_summary_fact_is_removed_and_full_text_fact_is_bound(self):
        state = {
            "raw_sources": [{
                "id": "source_1", "source_url": "https://example.gov.cn/report",
                "content": "原始报告正文" * 200, "extractor": "trafilatura",
                "title": "官方报告",
            }],
            "facts": [
                {"id": "keep", "content": "可验证事实", "source_url": "https://example.gov.cn/report", "source_type": "official", "related_sections": ["sec_1"]},
                {"id": "drop", "content": "仅摘要事实", "source_url": "https://docin.com/p/1", "source_type": "report", "related_sections": ["sec_1"]},
            ],
            "data_points": [
                {"name": "总资产", "value": 10, "source": "https://example.gov.cn/report"},
                {"name": "收入", "value": 20, "source": "https://docin.com/p/1"},
            ],
        }
        stats = bind_verified_evidence(state)
        self.assertEqual(stats["facts_removed"], 1)
        self.assertEqual([fact["id"] for fact in state["facts"]], ["keep"])
        self.assertEqual(len(build_evidence_view(state)), 1)
        self.assertNotIn("evidence_ledger", state)
        self.assertEqual([point["name"] for point in state["data_points"]], ["总资产"])

    def test_report_gate_covers_source_scope(self):
        state = {
            "research_contract": {
                "critical_sections": ["sec_1"],
                "required_metrics": [{"name": "总资产", "critical": True}],
            },
            "allowed_domains": ["example.gov.cn"],
            "raw_sources": [{"id": "src", "url": "https://example.gov.cn/a", "content": "公司总资产100亿元"}],
            "facts": [{"id": "f", "source_id": "src", "content": "公司总资产100亿元", "related_sections": ["sec_1"]}],
            "data_points": [{"id": "dp", "source_id": "src", "name": "公司总资产"}],
            "calculation_checks": {"passed": True, "blocking_reasons": []},
            "final_report": "数据来自[官方报告](https://example.gov.cn/a)",
        }
        gate = evaluate_report_gate(state)
        self.assertTrue(gate["passed"])
        # 发布门禁只回答非空与来源范围，不代表做过引用核验。
        self.assertEqual(gate["citation_verification"], "not_executed")
        state["final_report"] += "，另见[范围外来源](https://bad.example/x)"
        self.assertFalse(evaluate_report_gate(state)["passed"])

    def test_deterministic_cagr_and_weight_check(self):
        self.assertAlmostEqual(calculate_cagr(100, 121, 2), 0.1)
        state = {
            "research_contract": {"required_calculations": []},
            "data_points": [
                {"name": "质量权重", "value": 60, "unit": "%", "role": "weight"},
                {"name": "增长权重", "value": 30, "unit": "%", "role": "weight"},
            ],
        }
        result = run_deterministic_analysis(state)
        self.assertFalse(result["passed"])
        self.assertIn("权重合计", result["blocking_reasons"][0])

    def test_network_weights_do_not_trigger_score_normalization_gate(self):
        state = {
            "research_contract": {"required_calculations": []},
            "data_points": [
                {"name": "银行间网络边权重", "value": 15},
                {"name": "非银机构网络边权重", "value": 25},
            ],
        }
        result = run_deterministic_analysis(state)
        self.assertTrue(result["passed"])

    def test_deterministic_weighted_score_blocks_incomplete_entity(self):
        state = {
            "research_contract": {
                "required_entities": ["甲公司", "乙公司"],
                "required_calculations": [{
                    "id": "score_1",
                    "type": "weighted_score",
                    "output": "综合得分",
                    "weights": {"规模": 0.6, "增长": 0.4},
                }],
            },
            "data_points": [
                {"entity": "甲公司", "name": "规模", "value": 80},
                {"entity": "甲公司", "name": "增长", "value": 90},
                {"entity": "乙公司", "name": "规模", "value": 70},
            ],
        }

        result = run_deterministic_analysis(state)

        scores = [point for point in result["derived_points"] if point["name"] == "综合得分"]
        self.assertEqual([(point["entity"], point["value"]) for point in scores], [("甲公司", 84.0)])
        self.assertFalse(result["passed"])
        self.assertTrue(any("乙公司 缺少评分输入" in reason for reason in result["blocking_reasons"]))

    def test_deterministic_ranking_and_cross_entity_values_do_not_conflict(self):
        state = {
            "research_contract": {
                "required_entities": ["甲公司", "乙公司"],
                "required_calculations": [{
                    "id": "rank_1",
                    "type": "ranking",
                    "inputs": ["总资产"],
                    "output": "资产排名",
                }],
            },
            "data_points": [
                {"entity": "甲公司", "name": "总资产", "year": "2025", "value": 100},
                {"entity": "乙公司", "name": "总资产", "year": "2025", "value": 200},
            ],
        }

        result = run_deterministic_analysis(state)

        ranks = [point for point in result["derived_points"] if point["name"] == "资产排名"]
        self.assertEqual([(point["entity"], point["value"]) for point in ranks], [("乙公司", 1), ("甲公司", 2)])
        self.assertEqual(result["conflicting_values"], [])

        rerun = run_deterministic_analysis(state)
        rerun_ranks = [point for point in rerun["derived_points"] if point["name"] == "资产排名"]
        all_ranks = [point for point in state["data_points"] if point["name"] == "资产排名"]
        self.assertEqual(len(rerun_ranks), 2)
        self.assertEqual(len(all_ranks), 2)


if __name__ == "__main__":
    unittest.main()
