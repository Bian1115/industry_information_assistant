"""恢复 HTTP 入口与图状态接口的回归；无真实模型、数据库或搜索调用。"""

from contextlib import ExitStack
from datetime import datetime, timedelta
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / 'app'), str(Path(__file__).parent)]

from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Interrupt, PregelTask
from service import research_control
from service.deep_research_v2 import service as service_module
from service.deep_research_v2.delivery import build_complete_event, checkpoint_id_of
from service.deep_research_v2.graph import graph_config
from service.deep_research_v2.service import DeepResearchV2Service
from service.deep_research_v2.state import GRAPH_VERSION, create_initial_state
from service.research_events import ResearchRunLease, ResearchRunLeaseConflict
from test_单主链路图 import make_agents
from test_待合并节点结果恢复 import StopBeforeNextCheckpoint

routes = importlib.import_module('router.research_router')


class ResumeEndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.run_id, self.session_id, self.turn_id = map(str, (uuid4(), uuid4(), uuid4()))
        self.user = SimpleNamespace(id=uuid4())
        self.run = SimpleNamespace(session_id=self.session_id, status='failed', graph_version=GRAPH_VERSION)
        self.db = Mock()
        self.db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
            content='原始研究问题', turn_id=self.turn_id)
        self.db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        self.app = FastAPI()
        self.app.include_router(routes.router)
        self.app.dependency_overrides[routes.get_db] = lambda: self.db
        self.app.dependency_overrides[routes.get_current_user_required] = lambda: self.user
        self.service = DeepResearchV2Service(checkpointer=InMemorySaver())
        self.executions = []
        self.lease = ResearchRunLease(
            run_id=self.run_id, session_id=self.session_id, owner_id='测试执行器', fencing_token=1,
            lease_until=datetime.utcnow() + timedelta(minutes=1), user_id=str(self.user.id))

    def snapshot(self, *, next_nodes=(), result=True, error=None, interrupts=(), status='running', exists=True):
        research = create_initial_state('原始研究问题', self.session_id, [])
        research.update(status=status, phase='reviewing', final_report='已保存的写作结果')
        task = PregelTask('写作任务', 'write', (), error=error, interrupts=interrupts,
                          result={'research': research} if result else None)
        return SimpleNamespace(values={'research': research} if exists else {},
                               next=next_nodes, tasks=(task,))

    async def stream(self, **kwargs):
        self.executions.append(kwargs)
        yield 'data: {"type": "research_resumed"}\n\n'
        yield 'data: [DONE]\n\n'

    async def request(self, snapshot, method='POST', *, proxy=None, owned=True, lease_conflict=False, replay_payloads=None):
        with ExitStack() as stack:
            if snapshot is not None:
                self.graph = SimpleNamespace(aget_state=AsyncMock(return_value=snapshot))
                stack.enter_context(patch.object(service_module, 'build_graph', return_value=self.graph))
                stack.enter_context(patch.object(service_module, 'create_agents', return_value={}))
            proxy = proxy or SimpleNamespace(read_state=self.service.read_state, research=self.stream)
            stack.enter_context(patch.object(routes, 'get_research_service', return_value=proxy))
            stack.enter_context(patch.object(routes, '_owned_run', return_value=self.run,
                side_effect=None if owned else HTTPException(404, '未找到本人的研究运行')))
            self.acquire = stack.enter_context(patch.object(routes, 'acquire_research_run_lease',
                return_value=self.lease, side_effect=ResearchRunLeaseConflict('已有执行器') if lease_conflict else None))
            stack.enter_context(patch.object(routes, 'release_research_run_lease', return_value=True))
            stack.enter_context(patch.object(routes, '_pending_events', return_value=replay_payloads or []))
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url='http://test') as client:
                path = f'/research/resume/{self.run_id}' if method == 'POST' else f'/research/runs/{self.run_id}/state'
                return await client.request(method, path)

    async def test_saved_write_with_empty_next_is_admitted_and_exposed_as_resumable(self):
        state = self.snapshot()
        response = await self.request(state, 'GET')
        self.assertEqual(response.status_code, 200)
        view = response.json()['state']
        self.assertEqual(view['pending_nodes'], [])
        self.assertTrue(view['resumable'])
        self.assertEqual(view['saved_result_nodes'], ['write'])
        self.assertEqual(view['draft']['final_report'], '已保存的写作结果')
        response = await self.request(state)
        self.assertEqual(response.status_code, 200)
        self.assertIn('research_resumed', response.text)
        self.assertEqual(len(self.executions), 1)
        self.assertTrue(self.executions[0]['resume'])
        self.assertEqual(self.executions[0]['query'], '原始研究问题')

    async def test_normal_pending_node_still_resumes(self):
        state = self.snapshot(next_nodes=('review',), result=False)
        self.assertTrue((await self.request(state, 'GET')).json()['state']['resumable'])
        self.assertEqual((await self.request(state)).status_code, 200)

    async def test_storage_inconsistency_blocks_resume_and_exposes_reason(self):
        state = await self.service.read_state(self.run_id)
        state.update(exists=True, resumable=False, delivery_ready=False, status="running")
        state["storage_reconciliation"] = {"status":"blocked", "blocked":True,
            "error_code":"research_storage_inconsistent", "reason":"检查点引用的临时正文不存在"}
        proxy=SimpleNamespace(read_state=AsyncMock(return_value=state),research=self.stream)
        response=await self.request(None,proxy=proxy)
        self.assertEqual(response.status_code,409)
        self.assertEqual(response.json()["detail"]["reason"],state["storage_reconciliation"]["reason"])
        self.acquire.assert_not_called()
        self.db.query.return_value.filter.return_value.order_by.return_value.first.return_value = SimpleNamespace(
            payload={"content":"上次运行的网络错误"})
        response=await self.request(None,"GET",proxy=proxy)
        self.assertFalse(response.json()["state"]["resumable"])
        self.assertEqual(response.json()["state"]["error"],state["storage_reconciliation"]["reason"])

    async def test_unfinished_without_successful_result_is_not_misreported_completed(self):
        cases = [self.snapshot(result=False), self.snapshot(error='节点失败'),
                 self.snapshot(interrupts=(Interrupt('等待输入', id='测试中断'),))]
        for state in cases:
            with self.subTest(task=state.tasks[0]):
                response = await self.request(state)
                self.assertEqual(response.status_code, 400)
                self.assertIn('研究尚未完成', response.json()['detail'])
                self.assertNotIn('图已执行结束', response.json()['detail'])
                self.acquire.assert_not_called()
                self.assertFalse((await self.request(state, 'GET')).json()['state']['resumable'])
        self.assertEqual(self.executions, [])

    async def test_completed_graph_is_not_reexecuted_even_with_saved_result(self):
        state = self.snapshot(status='completed')
        self.assertFalse((await self.request(state, 'GET')).json()['state']['resumable'])
        response = await self.request(state)
        self.assertEqual(response.status_code, 400)
        self.assertIn('图已执行结束', response.json()['detail'])
        self.acquire.assert_not_called()
        self.assertEqual(self.executions, [])

    async def test_missing_graph_state_is_not_resumable(self):
        state = self.snapshot(exists=False)
        self.assertFalse((await self.request(state, 'GET')).json()['state']['resumable'])
        response = await self.request(state)
        self.assertEqual(response.status_code, 400)
        self.assertIn('没有可恢复的图状态', response.json()['detail'])
        self.acquire.assert_not_called()

    async def test_owner_check_precedes_reading_saved_results(self):
        response = await self.request(self.snapshot(), owned=False)
        self.assertEqual(response.status_code, 404)
        self.graph.aget_state.assert_not_awaited()
        self.acquire.assert_not_called()

    async def test_saved_result_does_not_bypass_execution_lease(self):
        response = await self.request(self.snapshot(), lease_conflict=True)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.executions, [])

    async def test_real_graph_saved_write_resumes_through_http_without_rewriting(self):
        saver = StopBeforeNextCheckpoint('review')
        first_agents, resumed_agents = make_agents(), make_agents()
        service = DeepResearchV2Service(checkpointer=saver)
        delivered = []

        def commit_factory(**kwargs):
            def commit(snapshot):
                self.assertEqual(snapshot.values['research']['status'], 'completed')
                delivered.append(snapshot)
                return kwargs['recorder'].record(build_complete_event(
                    snapshot.values['research'], run_id=self.run_id, checkpoint_id=checkpoint_id_of(snapshot)))
            return commit

        with patch.dict(os.environ, {'CONTEXT_ABLATION_CONFIG': '', 'AGENT_COMPRESSION_CONFIG': '',
                                     'AGENT_SUMMARY_STRATEGY': 'incremental-v3'}), patch.object(
            service_module, 'make_commit_result', side_effect=commit_factory
        ), patch.object(research_control, 'cache', Mock(get=Mock(return_value=None))), patch.object(
            service_module, 'release_research_run_lease', return_value=True
        ):
            with patch.object(service_module, 'create_agents', return_value=first_agents):
                first = [chunk async for chunk in service.research(
                    query='原始研究问题', session_id=self.session_id, user_id=str(self.user.id), run_id=self.run_id,
                    persist_events=False, allowed_domains=[])]
            self.assertTrue(saver.failed)
            self.assertIn('注入下一检查点写入中断', first[-2])
            self.assertEqual(delivered, [])

            proxy = SimpleNamespace(read_state=service.read_state,
                research=lambda **kwargs: service.research(**kwargs, persist_events=False))
            with patch.object(service_module, 'create_agents', return_value=resumed_agents):
                view = (await self.request(None, 'GET', proxy=proxy)).json()['state']
                self.assertEqual(view['pending_nodes'], [])
                self.assertEqual(view['saved_result_nodes'], ['write'])
                self.assertTrue(view['resumable'])
                response = await self.request(None, proxy=proxy)
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            self.assertEqual([event['phase'] for event in events if event['type'] == 'phase'], ['reviewing'])
            self.assertEqual(events[-1]['type'], 'research_complete')
            self.assertEqual(len(delivered), 1)
            self.assertEqual(len(first_agents['writer'].calls), 1)
            self.assertEqual(resumed_agents['writer'].calls, [])
            self.assertEqual(len(resumed_agents['critic'].calls), 1)
            final = await service_module.build_graph(
                resumed_agents, saver, run_id=self.run_id, user_id=None, session_id=self.session_id,
                execute_stage=service_module.make_execute_stage(run_id=self.run_id, session_id=self.session_id)
            ).aget_state(graph_config(self.run_id))
            self.assertEqual(final.next, ())
            self.assertEqual(final.values['research']['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
