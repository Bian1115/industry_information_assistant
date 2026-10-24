"""删除量化契约及门禁后，旧坏字段不阻断规划、写作或触发覆盖补搜。"""
import json
import logging
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from service.deep_research_v2.agents.architect import ChiefArchitect
from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.quality_pipeline import build_task_context, deterministic_critic_issues
from service.deep_research_v2.contract_updates import retain_committed_contract


class RemovedCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def test_planning_ignores_model_metrics_and_preserves_explicit_format(self):
        agent = ChiefArchitect.__new__(ChiefArchitect)
        agent.name = "ChiefArchitect"
        agent.logger = logging.getLogger("removed-coverage-test")
        agent.add_message = Mock()
        agent.call_llm = AsyncMock(return_value=json.dumps({
            "outline": [{"id": f"s{i}", "title": f"研究章节{i}"} for i in range(4)],
            "research_contract": {"required_metrics": [{"name": "准确率", "entity_scope": "算法", "years": "2019–2024"}],
                                  "active_constraints": {"max_body_chars": 2000}, "output_format": "表格"},
            "hypotheses": [],
        }))
        state = {"query": "比较算法的输入与准确率，2000字以内，用表格", "errors": []}
        await agent._initial_planning(state)
        self.assertEqual(agent.call_llm.await_count, 1)
        self.assertNotIn('"required_metrics"', agent.call_llm.call_args.kwargs["user_prompt"])
        self.assertNotIn("required_metrics", state["research_contract"])
        self.assertEqual(state["active_constraints"], {"max_body_chars": 2000})
        self.assertEqual(state["research_contract"]["output_format"], "表格")

    async def test_stale_missing_coverage_does_not_block_writer_even_with_old_env_flags(self):
        writer = LeadWriter.__new__(LeadWriter)
        writer.name = "LeadWriter"
        writer.add_message = Mock()
        writer._write_section = AsyncMock()
        writer._synthesize_report = AsyncMock()
        state = {"query": "比较算法", "facts": [{"id": "f1", "content": "已有事实"}],
                 "outline": [{"id": "s1", "title": "算法比较"}], "errors": [],
                 "research_contract": {"critical_sections": ["missing"], "required_metrics": [{"name": "准确率", "entity_scope": list("算法"), "years": list("2019–2024")}]},
                 "calculation_checks": {"blocking_reasons": ["旧计算缺失"]}}
        with patch.dict(os.environ, {"RESEARCH_CONTRACT_ENABLED": "true", "RESEARCH_GATE_COVERAGE_ENABLED": "true", "RESEARCH_GATE_REQUIRE_DECLARED_METRICS": "true"}):
            await writer._write_report(state)
        writer._write_section.assert_awaited_once()
        writer._synthesize_report.assert_awaited_once()
        self.assertEqual(state["phase"], "reviewing")
        self.assertNotIn("prewrite", state.get("quality_gates", {}))

    def test_stale_metrics_do_not_produce_critic_coverage_research_tasks(self):
        state = {"research_contract": {"critical_sections": ["missing"], "required_metrics": [{"name": "准确率", "years": list("2019–2024")}]},
                 "final_report": "已有研究正文", "allowed_domains": []}
        self.assertEqual(deterministic_critic_issues(state), [])

    def test_committed_user_requirements_survive_without_system_acceptance_metadata(self):
        committed = {"contract_update": {"status": "committed"}, "required_entities": ["用户指定A"],
                     "active_constraints": {"max_body_chars": 1000}, "acceptance": {"critical_metric_coverage": 1.0},
                     "critical_sections": ["old"], "coverage_check": [{"requirement_id": "old"}]}
        result = retain_committed_contract(build_task_context({}, "研究A"), committed, [])
        self.assertEqual(result["required_entities"], ["用户指定A"])
        self.assertEqual(result["active_constraints"], {"max_body_chars": 1000})
        for field in ("acceptance", "critical_sections", "coverage_check"):
            self.assertNotIn(field, result)
        self.assertIn("acceptance", committed)


if __name__ == '__main__':
    unittest.main()
