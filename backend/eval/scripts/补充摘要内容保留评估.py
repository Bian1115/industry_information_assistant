"""复核生成前参考引句，将非连续引句还原为真实来源跨度，再评审内容保留。"""
import argparse
import asyncio
import importlib.util
import json
import re
import sys
import time
import uuid
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / 'app')]
spec = importlib.util.spec_from_file_location('summary_evaluation', Path(__file__).with_name('评估已有轨迹摘要与引用.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
from config.llm_config import LLMConfig, get_agent_model
from eval.verification.replay import ReplayStore
from service.agent_history import HistoryScope, bind_agent_history, canonical_json, record_agent_message, exception_details
from service.context_policy import legacy_policy
from service.verification_budget import task_budget
from service.qwen_token_counter import get_qwen_token_counter
from service.agent_context import scoped_input_budget


def save(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def recover_reference(check, context):
    sid, quote = check['source_id'], check['quote']
    pieces = context['sources'].get(sid, [])
    if quote and any(quote in p['text'] for p in pieces):
        return check, {'id':check['id'], 'status':'exact'}
    lines = [s for s in quote.splitlines() if s.strip()]
    # 模型引用省略中间句时，程序取包含所有逐字原句的真实连续区间；不拼接伪原文。
    for p in pieces:
        matches = [p['text'].find(line) for line in lines]
        if lines and all(i >= 0 for i in matches) and matches == sorted(matches):
            start, end = matches[0], matches[-1] + len(lines[-1])
            repaired = {**check, 'quote': p['text'][start:end]}
            return repaired, {'id':check['id'], 'status':'expanded_to_real_contiguous_span', 'original_quote':quote,
                              'start':p['start']+start,'end':p['start']+end,'recovered_quote':repaired['quote']}
    # 仅空白排版不同：建立原字符索引，不改变实体、数字、否定或标点。
    target = ''.join(c for c in quote if not c.isspace())
    for p in pieces:
        indexes = [i for i,c in enumerate(p['text']) if not c.isspace()]
        compact = ''.join(p['text'][i] for i in indexes)
        pos = compact.find(target) if target else -1
        if pos >= 0 and compact.find(target,pos+1)<0:
            start,end=indexes[pos],indexes[pos+len(target)-1]+1
            repaired={**check,'quote':p['text'][start:end]}
            return repaired,{'id':check['id'],'status':'whitespace_mapped_to_original','original_quote':quote,
                             'start':p['start']+start,'end':p['start']+end,'recovered_quote':repaired['quote']}
    candidates = [other for other, values in context['sources'].items() if quote and any(quote in p['text'] for p in values)]
    if len(candidates)==1:
        return {**check,'source_id':candidates[0]},{'id':check['id'],'status':'unique_exact_source_corrected',
              'original_source_id':sid,'source_id':candidates[0]}
    return None, {'id':check['id'],'status':'unresolved_reference','original':check}


async def main(source):
    out=source/'补充内容保留评估';out.mkdir(exist_ok=False)
    model,config,counter,policy=get_agent_model('scout'),LLMConfig(),get_qwen_token_counter(),legacy_policy()
    results=[]
    for topic in [16,31,35,46,49]:
        d=source/f'题目{topic}'
        # 主回放独立运行，等待该题摘要和生成前参考原始输出落盘。
        tick=time.monotonic()
        while not (d/'新生成摘要.md').exists() or not list(d.glob('返回-*-quality_reference.txt')):
            if time.monotonic()-tick>900: raise TimeoutError('等待主回放产物超时')
            await asyncio.sleep(2)
        dest=out/f'题目{topic}';dest.mkdir()
        context=json.loads((d/'上下文编号映射.json').read_text());prose=(d/'新生成摘要.md').read_text()
        raw=json.loads(next(d.glob('返回-*-quality_reference.txt')).read_text())
        checks,repairs=[],[]
        for check in raw['checkpoints']:
            fixed,audit=recover_reference(check,context);repairs.append(audit)
            if fixed is not None:checks.append(fixed)
        save(dest/'参考引句定位复核.json',{'original_count':len(raw['checkpoints']),'accepted_count':len(checks),
              'checkpoints':checks,'audit':repairs,'scope':'reference_authoring_only; production_citation_checks_unchanged'})
        result={'topic':topic,'reference_count':len(raw['checkpoints']),'accepted_reference_count':len(checks),
                'reference_unresolved':len(raw['checkpoints'])-len(checks),'reference_repairs':[r['status'] for r in repairs if r['status']!='exact']}
        store=ReplayStore(dest/'独立工作历史',[])
        scope=HistoryScope(store,str(uuid.uuid4()),str(uuid.uuid4()),None,context_policy=policy.snapshot())
        agent=module.EvaluationAgent('DeepScout','摘要效果独立评审',config.api_key,config.base_url,model)
        evidence=[{'source_id':sid,'text':p['text']} for sid,pieces in context['sources'].items() for p in pieces]
        request=[{'role':'system','content':module.QUALITY_PROMPT},{'role':'user','content':canonical_json(
                 {'summary':prose,'checkpoints':checks,'context':evidence})}]
        save(dest/'实际评审输入.json',request)
        with task_budget({'max_requests':3,'max_seconds':180,'max_tokens_by_model':{model:100000}}) as ledger:
            try:
                tokens=counter.count_messages(request)
                if tokens>scoped_input_budget(scope,agent.name,6000,summary=True):raise ValueError('质量评审输入超预算')
                with bind_agent_history(scope):
                    call_id=str(uuid.uuid4())
                    await record_agent_message(agent.name,'summary_stage',{'stage':'quality_coverage_supplement'},call_id)
                    async with asyncio.timeout(120):
                        answer=await agent._call_llm_impl(request[0]['content'],request[1]['content'],True,0.0,6000,
                            history_call_id=call_id,prepared_messages=request,model_override=model)
                (dest/'评审原始输出.txt').write_text(answer)
                data=json.loads(answer);entries=data['coverage']
                if len(entries)!=len(checks) or {x['id'] for x in entries}!={x['id'] for x in checks}:raise ValueError('覆盖评审编号或数量无效')
                for item in entries:
                    if item['status'] not in {'preserved','partial','missing','contradicted'}:raise ValueError('覆盖状态无效')
                    if item['status']!='missing' and (not item['summary_quote'] or item['summary_quote'] not in prose):raise ValueError('覆盖评审引句不在摘要')
                for issue in data.get('issues',[]):
                    if not issue.get('summary_quote') or issue['summary_quote'] not in prose:raise ValueError('问题评审引句不在摘要')
                save(dest/'内容保留评审.json',data)
                counts=dict(module.Counter(x['status'] for x in entries))
                result.update(status='completed',coverage_counts=counts,preserved_rate=counts.get('preserved',0)/len(checks),
                    critical_counts=dict(module.Counter(x['status'] for x in entries if next(c for c in checks if c['id']==x['id'])['importance']=='critical')),
                    issues=data.get('issues',[]))
            except Exception as exc:result.update(status='error',error=exception_details(exc))
            finally:
                result['cost']=ledger.report();save(dest/'运行结果.json',result);await agent.client.close()
        results.append(result);save(out/'汇总结果.json',results)
        print(json.dumps({k:v for k,v in result.items() if k not in {'cost','issues'}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    asyncio.run(main(parser.parse_args().source))
