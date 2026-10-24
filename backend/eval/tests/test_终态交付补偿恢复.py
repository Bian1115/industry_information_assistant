"""终态交付补偿及已交付事件回放的HTTP机制测试；不连接真实业务数据库。"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'app'), str(Path(__file__).parent)]
from service import research_control
from service.deep_research_v2 import service as service_module
from service.deep_research_v2.delivery import ResearchDeliveryError, build_complete_event, checkpoint_id_of
import test_恢复入口待合并节点结果 as entry_fixture


class TerminalDeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.entry = entry_fixture.ResumeEndpointTests()
        self.entry.setUp()

    def terminal_snapshot(self):
        state = self.entry.snapshot(status='completed', result=False)
        state.tasks = ()
        state.config = {'configurable': {'thread_id': self.entry.run_id, 'checkpoint_id': '已提交终态检查点'}}
        return state

    async def test_terminal_report_is_delivered_without_graph_execution(self):
        entry = self.entry
        state = self.terminal_snapshot()
        commits = []

        def factory(**kwargs):
            def commit(snapshot):
                commits.append(snapshot)
                return kwargs['recorder'].record(build_complete_event(
                    snapshot.values['research'], run_id=entry.run_id, checkpoint_id=checkpoint_id_of(snapshot)))
            return commit

        proxy = SimpleNamespace(read_state=entry.service.read_state,
            research=lambda **kwargs: entry.service.research(**kwargs, persist_events=False))
        with patch.object(service_module, 'make_commit_result', side_effect=factory), patch.object(
            service_module, 'release_research_run_lease', return_value=True
        ), patch.object(research_control, 'cache', Mock(get=Mock(return_value=None))), patch.object(
            entry.service, '_graph_input', AsyncMock(side_effect=AssertionError('终态不得重建图输入'))
        ) as graph_input:
            view = (await entry.request(state, 'GET', proxy=proxy)).json()['state']
            self.assertTrue(view['delivery_ready'])
            self.assertTrue(view['delivery_pending'])
            self.assertTrue(view['resumable'])
            response = await entry.request(state, proxy=proxy)
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            self.assertEqual([event['type'] for event in events], ['research_complete'])
            self.assertEqual(events[0]['data']['final_report'], '已保存的写作结果')
            self.assertEqual(len(commits), 1)
            self.assertEqual(commits[0].config['configurable']['checkpoint_id'], '已提交终态检查点')
            self.assertTrue(response.text.endswith('data: [DONE]\n\n'))
            graph_input.assert_not_awaited()

    async def test_delivery_failure_does_not_emit_completion(self):
        entry = self.entry
        proxy = SimpleNamespace(read_state=entry.service.read_state,
            research=lambda **kwargs: entry.service.research(**kwargs, persist_events=False))
        with patch.object(service_module, 'make_commit_result', return_value=Mock(
            side_effect=ResearchDeliveryError('注入交付事务失败'))), patch.object(
            service_module, 'release_research_run_lease', return_value=True
        ), patch.object(research_control, 'cache', Mock(get=Mock(return_value=None))):
            response = await entry.request(self.terminal_snapshot(), proxy=proxy)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('research_complete', response.text)
        self.assertIn('research_delivery_failed', response.text)

    async def test_delivered_report_replays_existing_event_without_lease_or_execution(self):
        entry = self.entry
        entry.run.delivered_event_id = str(uuid4())
        entry.run.status = 'completed'
        state = self.terminal_snapshot()
        view = (await entry.request(state, 'GET')).json()['state']
        self.assertFalse(view['delivery_pending'])
        self.assertFalse(view['resumable'])
        event = {'type': 'research_complete', 'seq': 42, 'run_id': entry.run_id, 'data': {'final_report': '已有报告'}}
        response = await entry.request(state, replay_payloads=[event])
        self.assertEqual(response.status_code, 200)
        self.assertIn('"seq": 42', response.text)
        self.assertIn('已有报告', response.text)
        entry.acquire.assert_not_called()
        self.assertEqual(entry.executions, [])

    async def test_incomplete_terminal_snapshot_is_not_delivered(self):
        for missing in ('report', 'checkpoint_id', 'committed_tasks'):
            with self.subTest(missing=missing):
                state = self.terminal_snapshot()
                if missing == 'report':
                    state.values['research']['final_report'] = ' '
                elif missing == 'checkpoint_id':
                    state.config = {}
                else:
                    state.tasks = self.entry.snapshot(status='completed').tasks
                response = await self.entry.request(state)
                self.assertEqual(response.status_code, 400)
                self.entry.acquire.assert_not_called()
        self.assertEqual(self.entry.executions, [])

    async def test_terminal_compensation_still_requires_owner_and_lease(self):
        state = self.terminal_snapshot()
        self.assertEqual((await self.entry.request(state, owned=False)).status_code, 404)
        self.assertEqual((await self.entry.request(state, lease_conflict=True)).status_code, 409)
        self.assertEqual(self.entry.executions, [])


if __name__ == '__main__':
    unittest.main()
