"""仅固定开发集可串行30轮；模拟服务，不创建数据库身份或调用API。"""
import asyncio
from contextlib import asynccontextmanager, redirect_stdout
from copy import deepcopy
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

BACKEND=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(BACKEND), str(BACKEND/'app')]
from eval.multiturn import load_tasks
from eval.scripts.采集真实追问开发集 import execute, dialogue_groups, validate_single_session_dataset
from service.verification_budget import task_budget

DATASET=Path('/Users/bxy/projects/DeepResearch医疗研报助手/评测集/DeepResearch多轮开发集-5题-20260927.jsonl')


class Service:
    def __init__(self, fail_at=None): self.calls=[]; self.fail_at=fail_at
    def resolve_allowed_domains(self): return []
    async def research(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        event=({'type':'error','data':{'content':'故意失败'}} if len(self.calls)==self.fail_at else
               {'type':'research_complete','data':{'final_report':'实际回答'+str(len(self.calls))}})
        yield 'data: '+json.dumps(event, ensure_ascii=False)+'\n\n'


class SingleSessionTests(unittest.IsolatedAsyncioTestCase):
    async def run_fake(self, single, fail_at=None):
        tasks=load_tasks([DATASET]);service=Service(fail_at)
        @asynccontextmanager
        async def benchmark(_): yield service
        budget={'max_seconds':60,'max_requests':100,'max_tokens_by_model':{'test':100000}}
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp)/'采集', dataset=DATASET, single_session_development=single)
            def build_context(**kwargs):
                return {'original_query': kwargs['current_query'], 'resolved_query': kwargs['current_query'],
                        'conversation_context': [], 'builder_input': kwargs,
                        'session_task': {'status': 'fallback', 'execution_plan': 'replan'}}
            with patch('service.research_context_service.get_research_context_service',
                       return_value=SimpleNamespace(build=build_context)), patch('eval.scripts.采集真实追问开发集.create_identity', return_value='owner'), \
                 patch('eval.scripts.采集真实追问开发集.create_session', side_effect=['session'+str(i) for i in range(5)]) as session, \
                 patch('eval.scripts.采集真实追问开发集.save_user'), \
                 patch('eval.scripts.采集真实追问开发集.export', new=AsyncMock()), \
                 patch('eval.scripts.run_deepresearch_bench.benchmark_service', benchmark), \
                 patch('service.verification_budget.task_budget', wraps=task_budget) as ledger, redirect_stdout(io.StringIO()):
                result=await execute(args, tasks, budget, {})
            return service.calls, result, session.call_count, ledger.call_count

    async def test_thirty_turns_share_session_history_and_one_budget(self):
        calls,result,sessions,ledgers=await self.run_fake(True)
        self.assertEqual((len(calls),len(result),sessions,ledgers),(30,1,1,1))
        self.assertEqual(len({c['session_id'] for c in calls}),1)
        self.assertEqual(len({c['run_id'] for c in calls}),30)
        for call in calls:
            context = call['research_context']
            self.assertEqual(context['builder_input']['session_id'], call['session_id'])
            self.assertEqual(context['builder_input']['current_turn_id'], call['turn_id'])
            self.assertEqual(context['builder_input']['current_query'], call['query'])
            self.assertEqual(context['session_task']['execution_plan'], 'replan')
        self.assertEqual([r['turn_id'] for r in result[0]['turns']],list(range(1,31)))
        self.assertEqual([(r['source_id'],r['source_turn_id']) for r in result[0]['turns'][::6]],
                         [(8,1),(25,1),(26,1),(27,1),(28,1)])
        self.assertNotIn('pending_source_annotation',str(calls))
        self.assertTrue(all(c['user_id']=='owner' and c['turn_id'] for c in calls))

    async def test_default_still_uses_five_independent_sessions(self):
        calls,result,sessions,ledgers=await self.run_fake(False)
        self.assertEqual((len(result),sessions,ledgers),(5,5,5))
        self.assertEqual(len({c['session_id'] for c in calls}),5)
        self.assertTrue(all(calls[i]['research_context']['conversation_context']==[] for i in (0,6,12,18,24)))

    async def test_failure_stops_entire_combined_dialogue(self):
        calls,result,_,_=await self.run_fake(True,7)
        self.assertEqual(len(calls),7)
        self.assertEqual(result[0]['completed_turns'],6)
        self.assertEqual(result[0]['unexecuted_turns'],list(range(8,31)))

    async def test_recovery_reuses_failed_run_without_duplicate_user_or_lost_budget(self):
        tasks = load_tasks([DATASET])
        first, resumed = Service(5), Service()
        selected = [first]
        @asynccontextmanager
        async def benchmark(_):
            yield selected[0]
        builder = SimpleNamespace(build=lambda **kwargs: {'history_metrics': {}, 'session_task': {}})
        budget = {'max_seconds':600,'max_requests':100,'max_tokens_by_model':{'test':100000}}
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp)/'原运行'
            new = Path(tmp)/'恢复运行'
            args = SimpleNamespace(output=old, dataset=DATASET, single_session_development=True)
            manifest = {'date':datetime.now().isoformat(),'status':'finished',
                        'recovery':{'original_started_at':(datetime.now()-timedelta(seconds=5)).isoformat()}}
            with patch('service.research_context_service.get_research_context_service', return_value=builder), \
                 patch('eval.scripts.采集真实追问开发集.create_identity', return_value='owner') as identity, \
                 patch('eval.scripts.采集真实追问开发集.create_session', return_value='session') as session, \
                 patch('eval.scripts.采集真实追问开发集.save_user') as user, \
                 patch('eval.scripts.采集真实追问开发集.export', new=AsyncMock()), \
                 patch('eval.scripts.run_deepresearch_bench.benchmark_service', benchmark), redirect_stdout(io.StringIO()):
                prior = await execute(args, tasks, budget, manifest)
                prior[0]['cost']['requests'] = [{'model':'test','input_reserved':123,'output_reserved':456,
                    'usage':{'prompt_tokens':123,'completion_tokens':456},'cost_usd':None,'execution_status':'completed'}]
                prior[0]['cost']['tokens_reserved_by_model'] = {'test':579}
                (old/'采集进度.json').write_text(json.dumps(prior,ensure_ascii=False))
                original = (old/'合并开发集30轮/第5轮/轮次结果.json').read_bytes()
                selected[0] = resumed
                args = SimpleNamespace(output=new, dataset=DATASET, single_session_development=True, resume_from=old)
                result = await execute(args, tasks, budget, {})
            self.assertEqual((identity.call_count,session.call_count,user.call_count),(1,1,30))
            self.assertEqual(len(resumed.calls),26)
            self.assertTrue(resumed.calls[0]['resume'])
            self.assertEqual(resumed.calls[0]['run_id'],prior[0]['turns'][-1]['run_id'])
            self.assertEqual(resumed.calls[0]['turn_id'],prior[0]['turns'][-1]['turn_uuid'])
            self.assertEqual({c['session_id'] for c in resumed.calls},{'session'})
            self.assertTrue(all(not c['resume'] for c in resumed.calls[1:]))
            self.assertEqual(result[0]['completed_turns'],30)
            self.assertEqual(result[0]['cost']['tokens_reserved_by_model'],{'test':579})
            self.assertEqual(len(result[0]['cost']['requests']),1)
            self.assertGreaterEqual(result[0]['cost']['duration_seconds'],5)
            self.assertEqual((old/'合并开发集30轮/第5轮/轮次结果.json').read_bytes(),original)
            self.assertFalse(json.loads((new/'合并开发集30轮/第5轮/恢复前轮次结果.json').read_text())['completed'])

    def test_only_fixed_dataset_can_enable_combined_mode(self):
        tasks=load_tasks([DATASET])
        validate_single_session_dataset(DATASET,tasks)
        with self.assertRaises(ValueError):validate_single_session_dataset(DATASET,list(reversed(tasks)))
        with tempfile.TemporaryDirectory() as tmp:
            changed=Path(tmp)/'其他开发集.jsonl';changed.write_bytes(DATASET.read_bytes()+b'\n')
            with self.assertRaises(ValueError):validate_single_session_dataset(changed,tasks)


if __name__=='__main__':unittest.main()
