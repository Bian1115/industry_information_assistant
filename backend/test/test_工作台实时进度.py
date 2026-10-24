"""阶段中途可见计数、检查点隔离与旧进度淘汰。"""
import asyncio
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from test_研究记录读取 import ResearchResultTests
from eval.tests.test_单主链路图 import make_agents, passthrough_stage
from langgraph.checkpoint.memory import InMemorySaver
from service.deep_research_v2.graph import build_graph, graph_config, pack_state
from service.deep_research_v2.agents.base import BaseAgent
from service.deep_research_v2.state import create_initial_state
from service.deep_research_v2.progress import pending_progress


class ProgressApiTests(ResearchResultTests):
    def setUp(self):
        super().setUp()
        self.state['pending_nodes'] = ['research']
        self.payload = {'node': 'research', 'attempt': 1, 'progress': {
            'queries': 3, 'sources': 2, 'facts': 8, 'references': 0}}
        self.db.query.return_value.filter.return_value.order_by.return_value.first.return_value = SimpleNamespace(payload=self.payload)

    def test_failed_run_returns_saved_sources_and_authoritative_status(self):
        # 父类错误字段断言需要其原始错误数据。
        self.payload['content'] = '已记录的执行错误'
        super().test_failed_run_returns_saved_sources_and_authoritative_status()

    def test_live_counts_do_not_overwrite_resume_draft(self):
        state = self.fetch(status='running').json()['state']
        self.assertEqual(state['progress']['counts']['facts'], 8)
        self.assertEqual(len(state['draft']['facts']), 1)
        self.assertEqual(state['pending_nodes'], ['research'])
        self.assertTrue(state['resumable'])

    def test_committed_or_previous_node_progress_is_ignored(self):
        for pending, attempts in [(['analyze'], {}), (['research'], {'research': 1})]:
            self.state.update(pending_nodes=pending)
            self.state['research']['node_attempts'] = attempts
            self.assertIsNone(self.fetch(status='running').json()['state']['progress'])

    def test_completed_run_ignores_progress(self):
        self.assertIsNone(self.fetch(status='completed').json()['state']['progress'])

    def test_legacy_events_without_progress_remain_readable(self):
        self.payload.pop('progress')
        self.assertIsNone(self.fetch(status='running').json()['state']['progress'])

    def test_invalid_counts_are_rejected(self):
        for invalid in [-1, '8', True, None]:
            self.payload['progress']['facts'] = invalid
            self.assertIsNone(self.fetch(status='running').json()['state']['progress'])


class LiveGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_counts_visible_while_scout_has_not_committed(self):
        release = asyncio.Event()
        observed = asyncio.Event()
        events = []
        fake = SimpleNamespace(name='DeepScout', logger=logging.getLogger(__name__))

        class Scout:
            async def process(self, state):
                state['search_budget'] = {'queries': [{'query': 'q'}]}
                state['raw_sources'] = [{'id': 's1', 'content': '正文'}]
                state['facts'] = [{'id': 'f1'}]
                BaseAgent.add_message(fake, state, 'research_progress', {})
                await release.wait()
                return state

        graph = build_graph(make_agents(scout=Scout()), InMemorySaver(),
                            run_id='progress-test', user_id='u', session_id='s',
                            execute_stage=passthrough_stage)
        config = graph_config('progress-test')

        async def consume():
            async for event in graph.astream(pack_state(create_initial_state('q', 's', ['who.int'])),
                                             config=config, stream_mode='custom', durability='sync'):
                events.append(event)
                if event['type'] == 'research_progress':
                    observed.set()

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(observed.wait(), 5)
            snapshot = await graph.aget_state(config)
            self.assertEqual(snapshot.values['research']['facts'], [])
            self.assertEqual(snapshot.next, ('research',))
            state = {'pending_nodes': list(snapshot.next), 'draft': snapshot.values['research']}
            live = next(e for e in events if e['type'] == 'research_progress')
            self.assertEqual(pending_progress(state, live)['counts'], {
                'queries': 1, 'sources': 1, 'facts': 1, 'references': 0})
            # 事件保存的是数值快照，后续执行不会更改历史观察。
            release.set()
            await asyncio.wait_for(task, 5)
            saved = await graph.aget_state(config)
            self.assertEqual(len(saved.values['research']['facts']), 1)
            self.assertIsNone(pending_progress({'pending_nodes': [], 'draft': saved.values['research']}, live))
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
