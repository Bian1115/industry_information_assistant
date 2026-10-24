#!/usr/bin/env python3
"""真实Agent独立历史链路的同源下一步任务回放；不读取人工标注，不在线搜索。"""
import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys

BACKEND=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(BACKEND),str(BACKEND/'app')]
from service.agent_history import HistoryScope,bind_agent_history,content_hash
from service.context_policy import read_policy
from service.deep_research_v2.agents.base import BaseAgent
from service.verification_budget import task_budget
from eval.verification.replay import ReplayStore


class ReplayAgent(BaseAgent):
    async def process(self,state):
        raise NotImplementedError('回放只执行冻结的下一步任务，不在线重搜')


async def execute(snapshot, policies, output, repeats, budget_config):
    rows=[]
    plan = [{'task_id':task['task_id'], 'group':p.group, 'repeat':repeat,
             'run_id':content_hash([task['task_id'],p.fingerprint,repeat,content_hash(snapshot)]),
             'summary_enabled':p.semantic_summary}
            for task in snapshot['tasks'] for p in policies for repeat in range(repeats)]
    (output/'运行计划.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    for task in snapshot['tasks']:
        # 使用白名单复制；人工事实清单、离线标签和评测答案没有进入生成输入的通路。
        allowed={'task_id','model','records','contract','system_prompt','next_task','agent_name'}
        if set(task)-allowed:
            raise ValueError('回放任务包含未知字段；标注文件须交独立评分器')
        for policy in policies:
            if policy.policy_version=='legacy-runtime-v1':
                raise ValueError('改造前基线必须在其冻结源码中另行运行，不用当前运行捕获冒充')
            for repeat in range(repeats):
                rid=content_hash([task['task_id'],policy.fingerprint,repeat,content_hash(snapshot)])
                directory=output/f'任务-{len(rows)+1}-{policy.group}-重复{repeat+1}'
                store=ReplayStore(directory,task['records'])
                scope=HistoryScope(store,rid,rid,None,contract=task.get('contract',{}),context_policy=policy.snapshot())
                agent=ReplayAgent(task.get('agent_name','DeepScout'),'冻结下一步任务回放',os.environ['LLM_API_KEY'],os.environ['LLM_BASE_URL'],task['model'])
                result={'task_id':task['task_id'],'run_id':rid,'group':policy.group,'repeat':repeat,
                        'config':policy.snapshot(),'snapshot_hash':content_hash(snapshot),'completed':False,'final_body':''}
                with task_budget(budget_config) as ledger:
                    try:
                        async with asyncio.timeout(budget_config['max_seconds']):
                            with bind_agent_history(scope):
                                result['final_body']=await agent.call_llm(task['system_prompt'],task['next_task'],temperature=0.0)
                        result['completed']=True
                    except Exception as exc:
                        result.update(error_type=type(exc).__name__,error=str(exc))
                    finally:
                        result['cost']=ledger.report()
                        await agent.client.close()
                events=store.data['messages']
                stages={r['call_id']:r['payload']['stage'] for r in events if r['kind']=='summary_stage'}
                read_calls={r['call_id'] for r in events if r['kind']=='history_read'}
                for ticket in result['cost']['requests']:
                    ticket['stage']=stages.get(ticket['call_id'],'main_task_with_read' if ticket['call_id'] in read_calls else 'main_task')
                result['cost']['local_nli_seconds']=sum(r['payload'].get('duration_seconds',0) for r in events if r['kind']=='summary_verification')
                result['cost']['stage_timings']=[r['payload'] for r in events if r['kind']=='summary_stage_end']
                used={(r['call_id'],r['payload']['attempt']) for r in events if r['kind']=='llm_usage'}
                result.update(summary_enabled=policy.semantic_summary,
                              summary_triggered=any(r['kind']=='summary_stage' for r in events),
                              summary_delivered=bool(store.data['summary']),summary_body=store.data['summary'],
                              fallback='old_summary' if store.data['summary'] else 'original_history',
                              reads=[r['payload'] for r in events if r['kind']=='history_read'],
                              requests=[{'sequence':r['seq'],**r['payload'],
                                         'execution_status':'completed' if (r['call_id'],r['payload']['attempt']) in used else 'failed'} for r in events if r['kind']=='llm_request'],
                              verification_events=[r['payload'] for r in events if r['kind']=='summary_verification'])
                (directory/'运行结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
                rows.append(result)
                (output/'全部运行结果.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--config',type=Path,action='append',required=True)
    parser.add_argument('--budget',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1 or args.output.exists() or args.report.exists():
        parser.error('重复次数须为正且不得覆盖历史结果或报告')
    snapshot=json.loads(args.snapshot.read_text())
    budget=json.loads(args.budget.read_text())
    if budget.get('max_requests',0)<=0 or budget.get('max_seconds',0)<=0 or not budget.get('max_tokens_by_model'):
        parser.error('缺少统一整任务预算')
    policies=[read_policy(p) for p in args.config]
    for policy in policies:
        policy.require_supported()
    if len({p.fingerprint for p in policies})!=len(policies):
        parser.error('配置重复')
    args.output.mkdir(parents=True,exist_ok=False)
    error=None
    try:
        rows=asyncio.run(execute(snapshot,policies,args.output,args.repeats,budget))
    except BaseException as exc:
        error={'error_type':type(exc).__name__,'message':str(exc)}
        (args.output/'运行中断.json').write_text(json.dumps(error,ensure_ascii=False,indent=2))
        raise
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text('# 核验链路同源回放运行记录与分析\n\n'
            +f'评估日期：{datetime.now().isoformat()}。对象：固定下一步任务；并非在线检索或整份研究报告的完整图。\n\n'
            +f'配置口径：各组共用快照SHA256 {content_hash(snapshot)}、统一任务预算、{args.repeats}次重复。\n\n'
            +f'原始结果目录：[{args.output.name}]({args.output.resolve()})；每个任务/组/重复包含原始历史与运行结果。\n\n'
            +'核心指标、逐题成败和失败样例见全部运行结果.json；完整性错误另存运行中断.json。\n\n'
            +'已测量：实际执行状态、阶段记录与API usage（如供应商返回）。未定价和本地资源费用保留未知。'
            +'间接推断：请求关联可用于验证回读实际可见性。尚未评估：人审事实保留、阈值准确率与正式因果收益。'
            +'结论：本文件记录回放执行，不以NLI通过代替任务正确。下一步对产物盲审后使用独立评分器计算第7节指标。\n'
            +f'\n顶层中断：{json.dumps(error,ensure_ascii=False)}\n')


if __name__=='__main__':
    main()
