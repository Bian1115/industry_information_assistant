"""三字段规划的离线执行验证；替代已删除的逐条要求校验测试。"""
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from test_追问契约状态更新 import architect, state_v2
from service.deep_research_v2.quality_pipeline import build_task_context


def minimal_plan():
    return {
        'outline': [{'id': f'sec_{i}', 'title': f'章节{i}',
                     'description': '研究范围与证据需求', 'search_queries': ['检索词']}
                    for i in range(1, 5)],
        'research_contract': {'critical_questions': ['有哪些可靠证据？'],
                              'required_entities': ['A'], 'required_metrics': [],
                              'required_calculations': [], 'active_constraints': {'max_body_chars': 3000},
                              'time_range': '2020年以来', 'language': '中文', 'output_format': '不用表格'},
        'hypotheses': [{'id': 'h_1', 'content': '因素A可能影响结果B'}],
    }


class PlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_field_output_completes_and_drives_contract_and_ui(self):
        output = minimal_plan()
        agent = architect(output)
        state = state_v2('研究A，2020年以来，不超过3000字，中文，不用表格')
        await agent._initial_planning(state)
        self.assertEqual(agent.call_llm.await_count, 1)
        self.assertEqual(state['active_constraints'], {'max_body_chars': 3000})
        self.assertNotIn('time_range', state['research_contract'])
        self.assertEqual(state['research_contract']['output_format'], '不用表格')
        self.assertEqual(state['research_questions'], [state['query']])
        self.assertEqual(state['key_entities'], [])
        self.assertEqual(state['hypotheses'][0]['status'], 'unverified')
        self.assertEqual(state['hypotheses'][0]['evidence_inconclusive'], [])
        event = next(c.args[2] for c in agent.add_message.call_args_list if c.args[1] == 'outline')
        self.assertEqual(event['research_questions'], state['research_questions'])
        self.assertNotIn('requirement_responses', event)
        self.assertTrue(all('requirement_responses' not in x for x in state['outline']))
        self.assertNotIn('explicit_requirements', state['research_contract'])

    async def test_no_authority_or_coverage_retry_for_ordinary_constraints(self):
        output = minimal_plan()
        output['hypotheses'] = []
        agent = architect(output)
        state = state_v2('比较三家公司，2020年以来，从三个维度分析')
        await agent._initial_planning(state)
        self.assertEqual(agent.call_llm.await_count, 1)
        self.assertEqual(len(state['outline']), 4)
        self.assertNotIn('explicit_requirements', state['research_contract'])
        self.assertNotIn('requirement_responses', agent.call_llm.call_args.kwargs['user_prompt'])

    async def test_deprecated_extra_fields_do_not_override_contract(self):
        output = minimal_plan()
        output.update(requirement_responses=[{'id': 'req_bad', 'original_requirement': '旧要求'}],
                      research_questions=['旧问题'], key_entities=[{'name': '旧实体'}])
        output['research_contract']['required_entities'] = []
        agent = architect(output)
        state = state_v2()
        await agent._initial_planning(state)
        self.assertEqual(state['key_entities'], [])
        self.assertEqual(state['research_questions'], [state['query']])
        self.assertNotIn('explicit_requirements', state['research_contract'])

    async def test_bad_structure_gets_local_retry(self):
        output = minimal_plan()
        for key, value in [('research_contract', '不是对象'), ('outline', 3), ('hypotheses', None)]:
            with self.subTest(key=key):
                broken = deepcopy(output)
                broken[key] = value
                agent = architect(output)
                agent.call_llm = AsyncMock(side_effect=[json.dumps(broken), json.dumps(output)])
                await agent._initial_planning(state_v2())
                self.assertEqual(agent.call_llm.await_count, 2)

    async def test_repeated_bad_structure_does_not_commit_plan(self):
        output = minimal_plan()
        output['hypotheses'] = '错误类型'
        agent = architect(output)
        state = state_v2()
        before = deepcopy(state.get('research_contract'))
        await agent._initial_planning(state)
        self.assertEqual(agent.call_llm.await_count, 3)
        self.assertIn('Failed to generate research plan after retries', state['errors'])
        self.assertEqual(state.get('research_contract'), before)

    async def test_model_cannot_initialize_verified_hypothesis(self):
        output = minimal_plan()
        output['hypotheses'][0].update(status='confirmed', evidence_for=['编造证据'])
        state = state_v2()
        await architect(output)._initial_planning(state)
        self.assertEqual(state['hypotheses'][0]['status'], 'unverified')
        self.assertEqual(state['hypotheses'][0]['evidence_for'], [])

    def test_empty_new_contract_entities_do_not_use_legacy_fallback(self):
        output = minimal_plan()
        output['research_contract']['required_entities'] = []
        output['key_entities'] = [{'name': '模型自选对象'}]
        contract = build_task_context(output, '任务')
        self.assertNotIn('required_entities', contract)

    def test_system_policy_not_overridden_by_model(self):
        output = minimal_plan()
        output['research_contract']['source_policy'] = {'search_summaries_are_evidence': True}
        contract = build_task_context(output, '任务')
        self.assertNotIn('source_policy', contract)
