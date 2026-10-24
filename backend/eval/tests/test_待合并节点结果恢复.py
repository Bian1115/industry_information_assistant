"""待合并节点结果的服务恢复；真实LangGraph、内存检查点，不调用模型或业务库。"""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / 'app'), str(Path(__file__).parent)]

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Interrupt, PregelTask
from service.deep_research_v2 import service as service_module
from service.deep_research_v2.delivery import build_complete_event, checkpoint_id_of
from service.deep_research_v2.graph import graph_config
from service.deep_research_v2.service import (
    DeepResearchV2Service, LegacyRunNotResumableError, ResearchStatePausedError,
)
from service.deep_research_v2.state import create_initial_state
from service import research_control
from test_单主链路图 import make_agents


RUN = '待合并结果恢复测试'
SESSION = '恢复测试会话'


def snapshot(*, next_nodes=(), tasks=(), research=None):
    return SimpleNamespace(
        values={'research': research or create_initial_state('原始研究问题', SESSION, [])},
        next=next_nodes, tasks=tasks,
    )


def saved_task(*, result=None, error=None, interrupts=()):
    return PregelTask('测试任务', 'analyze', (), error=error, interrupts=interrupts, result=result)


async def resume_input(state):
    graph = SimpleNamespace(aget_state=AsyncMock(return_value=state))
    return await DeepResearchV2Service()._graph_input(
        graph, graph_config(RUN), resume=True, query='不应重新提交的输入',
        session_id=SESSION, user_id='测试用户', run_id=RUN,
        research_context=None, allowed_domains=None,
    )


class ResumeAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_pending_node_still_resumes(self):
        self.assertIsNone(await resume_input(snapshot(next_nodes=('write',))))

    async def test_empty_next_with_saved_result_resumes_without_editing_snapshot(self):
        research = create_initial_state('原始研究问题', SESSION, [])
        research.update(phase='analyzing', status='running', completion_reason='not_completed')
        state = snapshot(research=research, tasks=(saved_task(result={'research': research}),))
        before = deepcopy(state)
        self.assertIsNone(await resume_input(state))
        self.assertEqual(state.values, before.values)
        self.assertEqual(state.next, ())
        self.assertEqual(state.tasks, before.tasks)

    async def test_completed_business_state_is_not_resumed_even_with_saved_result(self):
        research = create_initial_state('原始研究问题', SESSION, [])
        research.update(status='completed', completion_reason='passed', final_report='已完成正文')
        for tasks in ((), (saved_task(result={'research': research}),)):
            with self.subTest(has_result=bool(tasks)), self.assertRaisesRegex(ValueError, '图已执行结束'):
                await resume_input(snapshot(research=research, tasks=tasks))

    async def test_empty_unfinished_checkpoint_is_paused_instead_of_reported_finished(self):
        with self.assertRaisesRegex(ResearchStatePausedError, '研究尚未完成'):
            await resume_input(snapshot())

    async def test_missing_result_error_and_interrupt_are_not_saved_success(self):
        cases = [saved_task(), saved_task(result={'research': {}}, error=RuntimeError('节点失败')),
                 saved_task(result={'research': {}}, interrupts=(Interrupt('等待输入', id='测试中断'),))]
        for task in cases:
            with self.subTest(task=task), self.assertRaises(ResearchStatePausedError):
                await resume_input(snapshot(tasks=(task,)))

    async def test_missing_graph_state_remains_non_resumable(self):
        with self.assertRaisesRegex(LegacyRunNotResumableError, '没有可恢复的图状态'):
            await resume_input(SimpleNamespace(values={}, next=(), tasks=(saved_task(result={}),)))

    async def test_saved_results_do_not_bypass_graph_version_check(self):
        research = create_initial_state('原始研究问题', SESSION, [])
        research['graph_version'] = '不兼容旧图'
        with self.assertRaisesRegex(LegacyRunNotResumableError, '图版本'):
            await resume_input(snapshot(research=research, tasks=(saved_task(result={'research': research}),)))

    async def test_saved_results_do_not_bypass_session_task_hash_check(self):
        research = create_initial_state('原始研究问题', SESSION, [])
        research.update(session_task={'execution_plan': 'augment'}, session_task_hash='错误哈希')
        with self.assertRaisesRegex(ValueError, '任务决策快照哈希'):
            await resume_input(snapshot(research=research, tasks=(saved_task(result={'research': research}),)))


class StopBeforeNextCheckpoint(InMemorySaver):
    """结果写入pending writes之后、后继步完整检查点写入之前失败一次。"""

    def __init__(self, destination):
        super().__init__()
        self.destination = destination
        self.failed = False

    async def aput(self, config, checkpoint, metadata, new_versions):
        if not self.failed and f'branch:to:{self.destination}' in checkpoint['channel_values']:
            self.failed = True
            raise RuntimeError('注入下一检查点写入中断')
        return await super().aput(config, checkpoint, metadata, new_versions)


class SavedResultGraphIntegration(unittest.IsolatedAsyncioTestCase):
    async def check_recovery(self, destination, completed_node, completed_agent, resumed_phases):
        saver = StopBeforeNextCheckpoint(destination)
        first_agents, resumed_agents = make_agents(), make_agents()
        delivered = []

        def commit_factory(**kwargs):
            def commit(state):
                self.assertEqual(state.values['research']['status'], 'completed')
                self.assertEqual(state.next, ())
                delivered.append(state)
                return kwargs['recorder'].record(build_complete_event(
                    state.values['research'], run_id=RUN, checkpoint_id=checkpoint_id_of(state)))
            return commit

        async def collect(service, *, resume):
            return [chunk async for chunk in service.research(
                query='原始研究问题', session_id=SESSION, user_id='测试用户', run_id=RUN,
                resume=resume, persist_events=False, allowed_domains=[],
            )]

        with patch.dict(os.environ, {'CONTEXT_ABLATION_CONFIG': '', 'AGENT_COMPRESSION_CONFIG': '',
                                     'AGENT_SUMMARY_STRATEGY': 'incremental-v3'}), patch.object(
            service_module, 'create_agents', side_effect=[first_agents, resumed_agents]
        ), patch.object(service_module, 'make_commit_result', side_effect=commit_factory), patch.object(
            research_control, 'cache', Mock(get=Mock(return_value=None))
        ):
            first = await collect(DeepResearchV2Service(checkpointer=saver), resume=False)
            first_events = [json.loads(chunk[6:]) for chunk in first if chunk != 'data: [DONE]\n\n']
            self.assertTrue(saver.failed)
            self.assertEqual(first_events[-1]['type'], 'error')
            self.assertIn('下一检查点写入中断', first_events[-1]['content'])
            self.assertEqual(delivered, [])

            graph = service_module.build_graph(
                resumed_agents, saver, run_id=RUN, user_id='测试用户', session_id=SESSION,
                execute_stage=service_module.make_execute_stage(
                    run_id=RUN, session_id=SESSION,
                    agents_of=service_module.agents_by_phase(resumed_agents)),
            )
            state = await graph.aget_state(graph_config(RUN))
            self.assertEqual(state.next, ())
            self.assertEqual(state.values['research']['status'], 'running')
            self.assertTrue(any(task.name == completed_node and task.result is not None for task in state.tasks))
            stored = await saver.aget_tuple(graph_config(RUN))
            self.assertIn('research', {channel for _, channel, _ in stored.pending_writes})
            self.assertIn(f'branch:to:{destination}', {channel for _, channel, _ in stored.pending_writes})

            # 新服务、新Agent实例复用序列化检查点；不能依赖旧Agent中的计算缓存。
            resumed = await collect(DeepResearchV2Service(checkpointer=saver), resume=True)
            events = [json.loads(chunk[6:]) for chunk in resumed if chunk != 'data: [DONE]\n\n']
            self.assertEqual(events[0]['type'], 'research_resumed')
            self.assertEqual([e['phase'] for e in events if e['type'] == 'phase'], resumed_phases)
            self.assertEqual(events[-1]['type'], 'research_complete')
            self.assertEqual(events[-1]['completion_reason'], 'passed')
            self.assertEqual(len(delivered), 1)
            self.assertEqual(len(first_agents[completed_agent].calls), 1)
            self.assertEqual(resumed_agents[completed_agent].calls, [])
            for agents in (first_agents, resumed_agents):
                for role, agent in agents.items():
                    self.assertLessEqual(len(agent.calls), 1, role)
            self.assertEqual(sum(len(a['data_analyst'].calls) for a in (first_agents, resumed_agents)), 1)
            self.assertEqual(sum(len(a['writer'].calls) for a in (first_agents, resumed_agents)), 1)
            self.assertEqual(sum(len(a['critic'].calls) for a in (first_agents, resumed_agents)), 1)
            final = delivered[0].values['research']
            self.assertEqual(final['query'], '原始研究问题')
            self.assertEqual(final['session_id'], SESSION)
            self.assertEqual(final['node_attempts'][completed_node], 1)
            self.assertEqual(resumed[-1], 'data: [DONE]\n\n')

    async def test_saved_analysis_advances_to_write_without_repeating_analysis(self):
        await self.check_recovery('write', 'analyze', 'data_analyst', ['writing', 'reviewing'])

    async def test_saved_writer_advances_to_review_without_repeating_writer(self):
        await self.check_recovery('review', 'write', 'writer', ['reviewing'])


if __name__ == '__main__':
    unittest.main()
