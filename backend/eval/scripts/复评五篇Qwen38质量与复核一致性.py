#!/usr/bin/env python3
"""更换质量裁判；新增十份Qwen评分，逐字复用此前十份Claude评分。"""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PREVIOUS=ROOT/'backend/eval/data/五篇报告fewshot复评与复核一致性-20261006'
spec=importlib.util.spec_from_file_location('quality_agreement',Path(__file__).with_name('复评五篇质量与复核一致性.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
DIMS=a.DIMS
read,dump,now=a.read,a.dump,a.now

def prepare(out):
    prior=read(PREVIOUS/'一致性汇总.json')
    if read(PREVIOUS/'运行状态.json')['status']!='完成':raise ValueError('上次双裁判复评未完成')
    if read(a.REVIEW)!=read(PREVIOUS/'复核评分快照.json'):raise ValueError('复核参照已变化，需重新确定比较口径')
    a.prepare(out)
    tasks=read(out/'冻结报告输入.json');oldtasks=read(PREVIOUS/'冻结报告输入.json')
    if tasks!=oldtasks:raise ValueError('当前原文与上次few-shot复评原文不完全相同')
    m=a.evaluator();options=m.quality_request_options('qwen3.8-max')
    if (PREVIOUS/'质量评分提示词快照.txt').read_text()!=m.QUALITY:raise ValueError('提示词改变，不能作为仅模型替换的对照')
    claude=[r for r in prior['new_records'] if r['model']=='claude']
    if len(claude)!=10:raise ValueError('必须沿用十份完整Claude评分')
    dump(out/'沿用Claude质量评分.json',claude)
    provenance=[]
    for r in claude:
        source=PREVIOUS/'裁判原始记录'/f"{r['key']}.json"
        raw=read(source)
        a.validate_quality(raw['result'])
        if raw['result']!=r['result']:raise ValueError('Claude原始记录与汇总不一致')
        target=out/'沿用Claude原始记录'/source.name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        provenance.append({'key':r['key'],'source':str(source),'copied_to':str(target),'sha256':a.digest(source),'new_api_call':False})
    dump(out/'沿用Claude来源与指纹.json',provenance)
    dump(out/'上次fewshot一致性汇总.json',prior)
    cfg=read(out/'评估配置.json')
    cfg.update({'created_at':now(),'scope':'仅重跑Qwen质量五篇各两次；沿用上次Claude十份评分及旧版逐条标签',
        'models':{'qwen':'qwen3.8-max','claude':'claude-opus-5'},'expected_calls':10,'inherited_claude_scores':10,
        'temperature':{'qwen':0.6,'claude':'沿用上次评分，无新请求'},'qwen_request_options':options,
        'qwen_max_concurrency':2,'previous_evaluation':str(PREVIOUS),
        'changes_vs_previous':['Qwen模型从qwen3-max替换为qwen3.8-max','Qwen温度0.3改为0.6，启用xhigh思考与seed=1234','严格JSON Schema，完整结束与原生JSON校验；不对质量分数做语法修复'],
        'quality_prompt_unchanged':True,'notice':'新Qwen十份、Claude沿用十份；双裁判均分含跨时间结果。模型和思考/采样/输出约束同时变化，不能单独归因模型。复核不是多人独立金标。'})
    dump(out/'评估配置.json',cfg)
    dump(out/'运行状态.json',{'status':'准备完成','expected':10,'completed':0,'inherited_claude':10,'failed':[]})
    print('已冻结Qwen3.8十次请求及Claude十份沿用评分',out,flush=True)

def prompt_for(m,t,repeat):
    return m.QUALITY+'\n本次重复编号'+str(repeat)+'\n用户任务：'+t['query']+'\n完整正文：\n'+t['body']

def run(out):
    m=a.evaluator();m.OUT=out;cfg=read(out/'评估配置.json')
    if cfg['quality_prompt']['sha256']!=m.quality_prompt_metadata()['sha256']:raise ValueError('冻结提示词与运行代码不同')
    tasks=read(out/'冻结报告输入.json');done=[];failed=[];started=now()
    def work(t,repeat):
        key=f"题{t['id']}-质量-qwen-{repeat}"
        result=m.call('qwen',prompt_for(m,t,repeat),key,validator=a.validate_quality,request_options=cfg['qwen_request_options'])
        return {'topic':t['id'],'model':'qwen','repeat':repeat,'key':key,'result':result}
    def save():
        dump(out/'新Qwen质量评分.json',sorted(done,key=lambda r:(r['topic'],r['repeat'])))
        dump(out/'运行状态.json',{'status':'运行中','started_at':started,'expected':10,'completed':len(done),'inherited_claude':10,'failed':failed})
    # Check cache identities before any new request; never silently reuse a different model.
    for t in tasks:
        for n in [1,2]:m.read_matching_cache(out/'裁判原始记录'/f"题{t['id']}-质量-qwen-{n}.json",prompt_for(m,t,n),cfg['qwen_request_options'])
    save()
    # First production request also verifies endpoint compatibility; it counts toward ten.
    try:first=work(tasks[0],1)
    except Exception as ex:
        dump(out/'运行状态.json',{'status':'未完成','started_at':started,'finished_at':now(),'expected':10,'completed':0,'inherited_claude':10,'failed':[str(ex)]})
        raise
    done.append(first);save()
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(work,t,n) for t in tasks for n in [1,2] if (t['id'],n)!=(tasks[0]['id'],1)]
        for f in cf.as_completed(futures):
            try:done.append(f.result())
            except Exception as ex:failed.append(str(ex))
            save();print('进度',len(done),'/10；失败',len(failed),flush=True)
    dump(out/'运行状态.json',{'status':'完成' if len(done)==10 and not failed else '未完成','started_at':started,'finished_at':now(),'expected':10,'completed':len(done),'inherited_claude':10,'failed':failed})
    if failed:raise RuntimeError('有未完成评分，可在同一冻结目录重试')

def analyze(out):
    if read(out/'运行状态.json')['status']!='完成':raise ValueError('评分尚未完成')
    cfg=read(out/'评估配置.json');tasks=read(out/'冻结报告输入.json');ref=read(out/'复核评分快照.json')['annotations']
    qwen=read(out/'新Qwen质量评分.json');claude=read(out/'沿用Claude质量评分.json');new=qwen+claude
    expected={(t['id'],model,n) for t in tasks for model in ['qwen','claude'] for n in [1,2]}
    if len(new)!=20 or {(r['topic'],r['model'],r['repeat']) for r in new}!=expected:raise ValueError('两模型各两次评分不完整或重复')
    m=a.evaluator();audit=[];calls=[]
    for r in new:
        a.validate_quality(r['result']);t=next(t for t in tasks if t['id']==r['topic'])
        p=out/('裁判原始记录' if r['model']=='qwen' else '沿用Claude原始记录')/f"{r['key']}.json";raw=read(p)
        prompt=prompt_for(m,t,r['repeat'])
        if raw['prompt']!=prompt or raw['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():raise ValueError('原始输入与冻结输入不同')
        a.validate_quality(raw['result'])
        if raw['result']!=r['result']:raise ValueError('原始评分与汇总不同')
        if r['model']=='qwen':
            if raw['requested_model']!='qwen3.8-max' or raw['request_options']!=cfg['qwen_request_options']:raise ValueError('Qwen请求模型或参数错误')
            if raw['completion_event']!='[DONE]' or raw['finish_reason']!='stop' or json.loads(raw['raw_response'])!=r['result']:raise ValueError('Qwen评分不是完整原生JSON输出')
            calls.append({'key':r['key'],'response_model':raw['response_model'],'usage':raw['usage'],'elapsed_s':raw['elapsed_s'],'errors':raw['errors'],'thinking_characters':len(raw['reasoning_content'])})
        audit.append({'key':r['key'],'source':'新调用' if r['model']=='qwen' else '沿用','frozen_prompt_match':True,'raw_score_and_reason_match':True})
    dump(out/'输入与评分完整性核查.json',{'date':now(),'valid_scores':20,'new_qwen_scores':10,'inherited_claude_scores':10,'checks':audit})
    means={}
    for t in tasks:
        tid=str(t['id']);means[tid]={}
        for model in ['qwen','claude']:
            group=[r for r in new if r['topic']==t['id'] and r['model']==model]
            means[tid][model]={d:statistics.mean(r['result']['dimensions'][d]['score'] for r in group) for d in DIMS}
        means[tid]['ensemble']={d:(means[tid]['qwen'][d]+means[tid]['claude'][d])/2 for d in DIMS}
    old=read(out/'上次fewshot一致性汇总.json');rows=[];comparison={}
    for model in ['qwen','claude','ensemble']:
        oldpairs=[];newpairs=[];old_errors={};new_errors={}
        for t in tasks:
            tid=str(t['id']);old_errors[tid]=[];new_errors[tid]=[]
            for d in DIMS:
                before=old['new_means'][tid][model][d];after=means[tid][model][d];reference=ref['quality'][tid][d]
                oldpairs.append((before,reference));newpairs.append((after,reference))
                old_errors[tid].append(abs(before-reference));new_errors[tid].append(abs(after-reference))
                rows.append({'topic':t['id'],'model':model,'dimension':d,'review':reference,'old':before,'new':after,'old_error':before-reference,'new_error':after-reference})
        bydim={d:{'old':a.agreement([(r['old'],r['review']) for r in rows if r['model']==model and r['dimension']==d]),
                   'new':a.agreement([(r['new'],r['review']) for r in rows if r['model']==model and r['dimension']==d])} for d in DIMS}
        comparison[model]={'old':a.agreement(oldpairs),'new':a.agreement(newpairs),'by_dimension':bydim,'delta':a.task_bootstrap_delta(old_errors,new_errors)}
        if model!='ensemble':
            gaps=[]
            for t in tasks:
                group=sorted([r for r in new if r['topic']==t['id'] and r['model']==model],key=lambda r:r['repeat'])
                gaps.extend(abs(group[0]['result']['dimensions'][d]['score']-group[1]['result']['dimensions'][d]['score']) for d in DIMS)
            comparison[model]['new_repeat_mean_absolute_difference']=statistics.mean(gaps)
            comparison[model]['old_repeat_mean_absolute_difference']=old['quality_comparison'][model]['new_repeat_mean_absolute_difference']
    inherited=a.inherited_comparison(read(out/'沿用逐条模型数据.json'),ref,read(out/'沿用预审判断.json'))
    dump(out/'逐条沿用结果与复核一致性.json',inherited)
    machine={'date':now(),'sample_unit':'5篇×4维=20个均分坐标；重复不增加独立报告数','quality_comparison':comparison,'score_rows':rows,'new_means':means,'new_records':new,
        'old_new_inter_judge_mean_absolute_gap':{'old':old['old_new_inter_judge_mean_absolute_gap']['new'],'new':statistics.mean(abs(means[str(t['id'])]['qwen'][d]-means[str(t['id'])]['claude'][d]) for t in tasks for d in DIMS)},
        'inherited_groups':inherited['groups'],'calls':{'new_valid_scores':10,'inherited_claude_scores':10,'records':calls,'known_tokens':sum(r['usage'].get('total_tokens',0) for r in calls),
            'saved_failed_attempts':len(list((out/'失败尝试').glob('*.json'))),'notice':'沿用Claude无新请求；失败尝试另存，消耗未包含于成功记录Token合计'},
        'limitations':['仅五篇，维度/重复不独立；复核不是独立多人金标','Claude跨时间沿用；逐条结果未重跑','同时改变模型、思考、温度和输出约束，不能纯因果归因模型','两次重复的编号不同；重复差异不等同固定输入纯采样方差','裁判输入仅正文，复核曾结合来源资料，证据条件不完全相同','API别名可能随服务更新；以原始响应model字段追踪实际版本']}
    dump(out/'一致性汇总.json',machine)
    print(json.dumps({k:{'old':v['old'],'new':v['new'],'delta':v['delta']} for k,v in comparison.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','analyze']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();{'prepare':prepare,'run':run,'analyze':analyze}[args.mode](args.out.resolve())
