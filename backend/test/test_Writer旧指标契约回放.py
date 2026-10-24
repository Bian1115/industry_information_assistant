"""旧检查点的异常指标容器不重新注入写作，用户原话和明确要求仍保留。"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND/'app'), str(BACKEND/'test')]
from test_Writer证据去重与程序参考文献 import state_fixture, writer_fixture


class LegacyMetricReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_string_and_character_array_metrics_are_not_reinjected(self):
        for entities, years in [('非接触式感知算法', '2019–2024'),
                                (list('非接触式感知算法'), list('2019–2024'))]:
            with self.subTest(entities=entities, years=years):
                state = state_fixture()
                state['query'] = '整理非接触式感知算法，并比较输入信号与准确率。'
                state['outline'][0]['requirement_ids'] = ['user_1']
                state['research_contract'] = {
                    'required_metrics': [{'id': 'metric_old', 'name': '旧自动验收项',
                                          'entity_scope': entities, 'years': years,
                                          'section_id': 'sec_1'}],
                    'explicit_requirements': [{'id': 'user_1',
                                              'original_requirement': '用户明确要求用表格比较输入信号',
                                              'planned_response': '在本章提供比较表格'}],
                }
                before = json.dumps(state['research_contract'], ensure_ascii=False)
                writer = writer_fixture()
                writer.call_llm = AsyncMock(return_value=json.dumps({'content': '已有事实与来源。', 'claims': []}))
                writer.parse_json_response = json.loads
                await writer._write_section(state, state['outline'][0])
                prompt = writer.call_llm.call_args.kwargs['user_prompt']
                self.assertIn(state['query'], prompt)
                self.assertIn('用户明确要求用表格比较输入信号', prompt)
                self.assertNotIn('旧自动验收项', prompt)
                self.assertNotIn('[指标契约', prompt)
                self.assertNotIn('2,0,1,9', prompt)
                self.assertEqual(json.dumps(state['research_contract'], ensure_ascii=False), before)


if __name__ == '__main__':
    unittest.main()
