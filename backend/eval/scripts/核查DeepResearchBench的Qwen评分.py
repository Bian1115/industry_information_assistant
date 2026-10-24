#!/usr/bin/env python3
"""核查完整Qwen结果，以Decimal独立复算官方权重；不修改评分。"""
import hashlib
import importlib.util
import json
from collections import Counter
from decimal import Decimal as D
from pathlib import Path

SCRIPTS=Path(__file__).parent
spec=importlib.util.spec_from_file_location('bench',SCRIPTS/'DeepResearchBench参考报告双裁判一致性.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def independent_weighted(t, scores):
    s={r['criterion_id']:D(str(r['score'])) for r in scores};dims={}
    for dim in b.DIMS:
        cs=[c for c in t['criteria'] if c['dimension']==dim]
        assert all(c['weight']>0 for c in cs)
        dims[dim]=sum(s[c['criterion_id']]*D(str(c['weight'])) for c in cs)/sum(D(str(c['weight'])) for c in cs)
    assert abs(sum(D(str(t['dimension_weight'][dim])) for dim in b.DIMS)-1)<D('1e-12')
    dims['total']=sum(dims[dim]*D(str(t['dimension_weight'][dim])) for dim in b.DIMS)
    return dims

def run():
    p=b.OUT;cfg=b.read(p/'评估配置.json');tasks=b.read(p/'冻结输入.json');summary=b.read(p/'一致性汇总.json')
    assert len(tasks)==30 and sum(len(t['criteria']) for t in tasks)==758
    assert b.sha(p/'冻结输入.json')==cfg['input_sha256']
    assert b.sha(p/'双裁判提示词快照.txt')==cfg['prompt_sha256']
    table={r['task_id']:r for r in summary['tasks']};records=[];quotes=Counter();models=Counter();numeric_count=0
    for t in tasks:
        tid=t['id'];ownp=p/'助手独立评分'/f'题{tid}.json'
        assert b.sha(ownp)==cfg['assistant_file_sha256'][ownp.name]
        own=independent_weighted(t,b.read(ownp)['scores']);runs=[]
        for n in [1,2]:
            raw=b.read(p/'裁判原始记录'/f'题{tid}-细则-qwen-{n}.json')
            assert raw.get('result') is not None,f'题{tid}重复{n}尚未完成'
            expected=b.prompt(t,n)
            assert raw['prompt']==expected and raw['prompt_sha256']==hashlib.sha256(expected.encode()).hexdigest()
            assert raw['request_options']==cfg['qwen_request_options']
            assert raw['completion_event']=='[DONE]' and raw['finish_reason']=='stop'
            assert json.loads(raw['raw_response'])==raw['result']
            b.validate(t,raw['result']);models[raw['response_model']]+=1
            numeric_count+=len(raw['result']['scores'])
            for r in raw['result']['scores']:
                q=r['evidence_quote'];quotes['empty' if not q else ('matched' if q in t['article'] else 'unmatched')]+=1
            runs.append(independent_weighted(t,raw['result']['scores']))
        assert table[tid]['counts']['qwen']==2
        mean={dim:(runs[0][dim]+runs[1][dim])/2 for dim in b.DIMS+['total']}
        for dim in b.DIMS+['total']:
            assert abs(float(mean[dim])-table[tid]['scores']['qwen'][dim])<1e-10
            assert abs(float(own[dim])-table[tid]['scores']['assistant'][dim])<1e-10
        records.append({'task_id':tid,'assistant':float(own['total']),'qwen_repeat_1':float(runs[0]['total']),
                        'qwen_repeat_2':float(runs[1]['total']),'qwen_mean':float(mean['total'])})
    assert numeric_count==1516
    assert models=={cfg['models']['qwen']:60}
    assert summary['agreement']['qwen']['total']['n']==30
    report={'checked_at':b.now(),'tasks':30,'valid_qwen_calls':60,'numeric_scores':numeric_count,'frozen_assistant_checks':30,
            'prompt_and_parameters_checks':60,'native_json_and_full_stream_checks':60,'response_models':dict(models),
            'independent_decimal_weight_checks':30,'quote_audit':dict(quotes),'totals':records,
            'limitations':'引文连续匹配只证明定位；分数一致性不能证明事实正确或裁判准确。'}
    b.dump(p/'Qwen最终核查.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='totals'},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
