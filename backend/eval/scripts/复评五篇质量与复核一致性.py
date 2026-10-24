#!/usr/bin/env python3
"""固定五篇原文，盲跑few-shot质量裁判并与冻结复核结果对照。"""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
WORKBENCH=Path('/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/五篇报告人工评分工作台-20261004.html')
REVIEW=WORKBENCH.parent/'五篇报告复核评分-20261005.json'
OLD=ROOT/'backend/eval/data/五篇报告正文与Scout独立评估-20261003-234200'
DIMS=['depth','comprehensiveness','readability','requirements']
NAMES={'depth':'深度','comprehensiveness':'全面性','readability':'可读性','requirements':'需求覆盖'}

def now():return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
def read(p):return json.loads(p.read_text())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def evaluator():
    spec=importlib.util.spec_from_file_location('frozen_quality_evaluator',ROOT/'backend/eval/scripts/评估五篇正文与Scout.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def validate_quality(x):
    if not isinstance(x,dict) or not isinstance(x.get('dimensions'),dict):raise ValueError('四维质量评分缺失或字段不正确')
    dims=x['dimensions'];metadata={'strengths','weaknesses','fact_risks','overall_comment'}
    if not set(DIMS)<=set(dims) or set(dims)-set(DIMS)-metadata:raise ValueError('四维质量评分缺失或字段不正确')
    for d in DIMS:
        v=dims[d]
        if not isinstance(v,dict):raise ValueError('维度输出必须包含score和reason')
        score=v.get('score')
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=10:raise ValueError('质量分数须为0–10有限数值')
        if not isinstance(v.get('reason'),str) or not v['reason'].strip():raise ValueError('质量评分须有理由')
    # Some valid responses put commentary inside dimensions. Move only known
    # metadata; retain every score/reason and the original API text in the record.
    moved=sorted(set(dims)&metadata)
    if any(k in x and x[k]!=dims[k] for k in moved):raise ValueError('点评字段在两个位置存在冲突')
    if moved:
        original_keys=list(dims)
        for k in moved:x[k]=dims.pop(k)
        x['_format_normalization']={'operation':'将嵌套附加点评移至顶层，分数与理由不变','original_dimensions_keys':original_keys,'moved_fields':moved}

def recover_streamed_quality(text):
    """Recover only fully decoded native JSON dimension objects; never repair them."""
    match=re.match(r'\s*(?:```(?:json)?\s*)?\{\s*"dimensions"\s*:\s*\{',text)
    if not match:raise ValueError('中断响应缺少可完整解码的四维对象')
    decoder=json.JSONDecoder();pos=match.end();dims={}
    for _ in range(4):
        pos=re.match(r'\s*',text[pos:]).end()+pos
        key,pos=decoder.raw_decode(text,pos)
        if key not in DIMS or key in dims:raise ValueError('中断响应维度键不正确')
        colon=re.match(r'\s*:\s*',text[pos:])
        if not colon:raise ValueError('中断响应维度分隔符缺失')
        value,pos=decoder.raw_decode(text,pos+colon.end())
        dims[key]=value
        delimiter=re.match(r'\s*([,}])',text[pos:])
        if not delimiter or (len(dims)<4 and delimiter.group(1)!=','):raise ValueError('中断响应维度对象尚未完整输出')
        pos+=delimiter.end()
    packet={'dimensions':dims};validate_quality(packet)
    packet['_quality_packet_recovery']={'complete_dimension_objects':4,'method':'原生JSON解码完整四维对象，不使用JSON语法修复补全分数或理由','excluded':'四维对象之后的异常或缺失附加点评','cause':'待标注'}
    return packet

def prepare(out):
    if out.exists() and any(out.iterdir()):raise ValueError('新复评必须使用空目录')
    m=evaluator();html=WORKBENCH.read_text();d=json.loads(re.search(r'<script id="dataset" type="application/json">(.*?)</script>',html,re.S).group(1))
    embedded=json.loads(re.search(r'<script id="review" type="application/json">(.*?)</script>',html,re.S).group(1));review=read(REVIEW)
    if embedded['annotations']!=review['annotations']:raise ValueError('工作台与已保存复核评分不一致，不能静默选择比较对象')
    hashes={x['id']:x['sha256'] for x in review['report_hashes']}
    tasks=[]
    for t in d['tasks']:
        if hashlib.sha256(t['report'].encode()).hexdigest()!=hashes[t['id']]:raise ValueError('复核与待评报告指纹不同')
        tasks.append({k:t[k] for k in ['id','title','query','body','report','report_sha256','folder']})
    if {t['id'] for t in tasks}!={35,31,46,16,49}:raise ValueError('必须是既有五篇固定报告')
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'冻结报告输入.json',tasks);dump(out/'复核评分快照.json',review);dump(out/'沿用逐条模型数据.json',d)
    dump(out/'旧质量评分.json',d['summary']['quality'])
    old_pre=[]
    for t in tasks:
        old_pre.extend(read(OLD/'裁判原始记录'/f"题{t['id']}-审核检出.json")['result']['pre_claims'])
    dump(out/'沿用预审判断.json',old_pre)
    dump(out/'评估配置.json',{'created_at':now(),'scope':'只重跑四维质量：两模型各2次；逐条事实/Scout/预审/来源/覆盖/Critic沿用旧版并与复核比较',
       'models':{'qwen':m.quality_request_options()['model'],'claude':'claude-opus-5'},'qwen_request_options':m.quality_request_options(),'repeats_per_model':2,'expected_calls':20,
       'temperature':{'qwen':m.quality_request_options()['temperature'],'claude':'服务默认，与旧版一致'},'claude_max_concurrency':1,
       'report_input':'与旧质量评分相同的body（参考文献章节之前），不附复核标签或分数',
       'quality_prompt':m.quality_prompt_metadata(),'evidence_cutoff':'2026-10-03','comparison_reference':str(REVIEW),
       'review_id':review['review_context']['review_id'],'review_snapshot_sha256':digest(REVIEW),
       'workbench_snapshot_sha256':digest(WORKBENCH),'previous_evaluation':str(OLD),
       'score_agreement_protocol':{'primary':['MAE','RMSE','signed_bias','within_1_point'],'denominator':'5篇×4维=20个均分坐标；两次重复先按模型/题目/维度平均','unknown':'逐条标签包含unknown的全量一致率与双方可判定的一致率分别报告','bootstrap':'以任务为单位的5^5穷举经验bootstrap；仅5篇，区间为描述性','blinding':'复核分数不提供给裁判'},
       'notice':'复核是既有评分参照，不等同多名独立标注员金标；没有重新生成研究报告；虚构few-shot不使用真实产出内容'})
    (out/'质量评分提示词快照.txt').write_text(m.QUALITY)
    (out/'虚构样例快照.json').write_bytes(m.QUALITY_EXAMPLES_BYTES)
    dump(out/'运行状态.json',{'status':'准备完成','expected':20,'completed':0,'failed':[]})
    print('已冻结五篇原文和复核参照',out,flush=True)

def run(out):
    m=evaluator();config=read(out/'评估配置.json');m.OUT=out
    if config['quality_prompt']['sha256']!=m.quality_prompt_metadata()['sha256']:raise ValueError('运行提示词与冻结版本不同')
    # Pin the same API aliases used by the original two judges, independent of other runs.
    os.environ['OPENAI_MODEL']=config['models']['qwen'];os.environ['CLAUDE_RELAY_MODEL']=config['models']['claude']
    tasks=read(out/'冻结报告输入.json');jobs=[];done=[];failed=[]
    def work(t,model,repeat):
        key=f"题{t['id']}-质量-{model}-{repeat}"
        prompt=m.QUALITY+'\n本次重复编号'+str(repeat)+'\n用户任务：'+t['query']+'\n完整正文：\n'+t['body']
        options=config.get('qwen_request_options') if model=='qwen' else None
        reply=m.call(model,prompt,key,7000,validate_quality,None if options and options.get('response_format') else recover_streamed_quality,request_options=options)
        return {'topic':t['id'],'model':model,'repeat':repeat,'key':key,'result':reply}
    started=now();dump(out/'运行状态.json',{'status':'运行中','started_at':started,'expected':20,'completed':0,'failed':[]})
    # Dedicated pools prevent waiting Claude calls from occupying Qwen workers.
    with cf.ThreadPoolExecutor(max_workers=2) as qwen_pool, cf.ThreadPoolExecutor(max_workers=1) as claude_pool:
        for t in tasks:
            for model in ['qwen','claude']:
                pool=qwen_pool if model=='qwen' else claude_pool
                for repeat in [1,2]:jobs.append(pool.submit(work,t,model,repeat))
        for f in cf.as_completed(jobs):
            try:done.append(f.result())
            except Exception as ex:failed.append(str(ex))
            dump(out/'新质量评分.json',sorted(done,key=lambda x:(x['topic'],x['model'],x['repeat'])))
            dump(out/'运行状态.json',{'status':'运行中','started_at':started,'expected':20,'completed':len(done),'failed':failed})
            print('进度',len(done),'/20；失败',len(failed),flush=True)
    status={'status':'完成' if len(done)==20 and not failed else '未完成','started_at':started,'finished_at':now(),'expected':20,'completed':len(done),'failed':failed}
    dump(out/'运行状态.json',status)
    if status['status']!='完成':raise RuntimeError('存在未完成评分，可在同一冻结目录重试')

def ranks(values):
    ordered=sorted(range(len(values)),key=lambda i:values[i]);result=[0.0]*len(values);i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[ordered[j]]==values[ordered[i]]:j+=1
        for k in range(i,j):result[ordered[k]]=(i+j+1)/2
        i=j
    return result

def correlation(a,b):
    am=statistics.mean(a);bm=statistics.mean(b);num=sum((x-am)*(y-bm) for x,y in zip(a,b));den=math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
    return num/den if den else None

def agreement(pairs):
    if not pairs:return {'n':0}
    predicted=[p[0] for p in pairs];review=[p[1] for p in pairs];errors=[a-b for a,b in pairs];n=len(errors)
    return {'n':n,'mae':statistics.mean(abs(x) for x in errors),'rmse':math.sqrt(statistics.mean(x*x for x in errors)),
        'signed_bias':statistics.mean(errors),'exact_count':sum(abs(x)<1e-9 for x in errors),'within_half_count':sum(abs(x)<=.5+1e-9 for x in errors),
        'within_1_count':sum(abs(x)<=1+1e-9 for x in errors),'within_1_percent':100*sum(abs(x)<=1+1e-9 for x in errors)/n,
        'pearson':correlation(predicted,review),'spearman':correlation(ranks(predicted),ranks(review))}

def categorical(pairs):
    confusion={}
    for predicted,review in pairs:
        confusion.setdefault(review,{});confusion[review][predicted]=confusion[review].get(predicted,0)+1
    known=[p for p in pairs if all(x not in ['unknown','uncertain',None] for x in p)]
    return {'n':len(pairs),'exact':sum(p==r for p,r in pairs),'percent':100*sum(p==r for p,r in pairs)/len(pairs) if pairs else None,
        'both_determinate_n':len(known),'both_determinate_exact':sum(p==r for p,r in known),'both_determinate_percent':100*sum(p==r for p,r in known)/len(known) if known else None,
        'reference_unknown':sum(r in ['unknown','uncertain'] for p,r in pairs),'prediction_unknown':sum(p in ['unknown','uncertain'] for p,r in pairs),'confusion_reference_rows':confusion}

def quantile(sorted_values,p):
    index=(len(sorted_values)-1)*p;lo=math.floor(index);hi=math.ceil(index);return sorted_values[lo]+(sorted_values[hi]-sorted_values[lo])*(index-lo)

def task_bootstrap_delta(old_errors,new_errors):
    ids=list(old_errors)
    if set(ids)!=set(new_errors):raise ValueError('任务不配对')
    deltas={i:statistics.mean(new_errors[i])-statistics.mean(old_errors[i]) for i in ids}
    samples=sorted(statistics.mean(deltas[i] for i in draw) for draw in itertools.product(ids,repeat=len(ids)))
    return {'delta_mae_new_minus_old':statistics.mean(deltas.values()),'task_cluster_bootstrap_ci95':[quantile(samples,.025),quantile(samples,.975)],'tasks':len(ids),'resamples':len(samples)}

def inherited_comparison(d,annotations,old_pre=None):
    eligible=[r for r in d['rows'] if r['kind']!='body' or r['type']=='factual'];items=annotations['items'];out={}
    for group,rows in [('全部适用论断',eligible),('有引用论断',[r for r in eligible if r['refs']]),('正文事实',[r for r in eligible if r['kind']=='body']),('Scout事实',[r for r in eligible if r['kind']=='scout']),('预审论断',[r for r in eligible if r['kind']=='pre'])]:
        pairs=[(r['judgment']['joint_support'],items[r['id']]['support']) for r in rows]
        out[group]=categorical(pairs)
    ref_pairs=[];ref_details=[]
    for r in eligible:
        labels={v['id']:v['label'] for v in r['judgment'].get('ref_support',[])}
        for i,ref in enumerate(r['refs']):
            rid=r['id']+'-ref-'+str(i)
            if ref['id'] not in labels:raise ValueError(f'{rid}缺少旧版引用关系标签，不能按顺序猜测')
            predicted=labels[ref['id']];reference=items[rid]['support']
            ref_pairs.append((predicted,reference))
            ref_details.append({'id':rid,'claim_id':r['id'],'source_id':ref['id'],'old_support':predicted,'review_support':reference,'agree':predicted==reference})
    out['逐条引用关系']=categorical(ref_pairs)
    out['需求覆盖']=categorical([(r['coverage'],items[r['id']]['coverage']) for r in d['coverage']])
    out['Critic问题有效性']=categorical([((r.get('model_review') or {}).get('validity','unknown'),items[r['id']]['validity']) for r in d['audits']])
    out['来源质量']=agreement([(r['score'],float(items[r['id']]['score'])) for r in d['sources']])
    out['Scout重复信息']=categorical([('yes' if r.get('model_duplicate') else 'no',items[r['id']]['duplicate']) for r in eligible if r['kind']=='scout'])
    if old_pre is not None:
        mapping={'true':'yes','false':'no','unknown':'unknown'}
        out['预审明确错误判断']=categorical([(mapping.get(str(r['error']).lower(),'unknown'),items[r['id']]['error']) for r in old_pre])
        positives=[r for r in old_pre if items[r['id']]['error']=='yes' and items[r['id']]['detected'] in ['yes','no']]
        out['首轮检出判断（复核确认错误）']=categorical([('yes' if r.get('detected') is True else 'no',items[r['id']]['detected']) for r in positives])
    details=[{'id':r['id'],'topic':r['topic'],'kind':r['kind'],'cited':bool(r['refs']),'old_support':r['judgment']['joint_support'],'review_support':items[r['id']]['support'],'agree':r['judgment']['joint_support']==items[r['id']]['support']} for r in eligible]
    return {'scope':'旧版逐条结果沿用；没有使用few-shot重新裁判这些标签','groups':out,'details':details,'reference_details':ref_details}

def analyze(out):
    if read(out/'运行状态.json')['status']!='完成':raise ValueError('评分尚未全部完成')
    new=read(out/'新质量评分.json');old=read(out/'旧质量评分.json');review=read(out/'复核评分快照.json')['annotations'];tasks=read(out/'冻结报告输入.json')
    expected={(t['id'],m,n) for t in tasks for m in ['qwen','claude'] for n in [1,2]}
    if len(new)!=20 or {(r['topic'],r['model'],r['repeat']) for r in new}!=expected:raise ValueError('20份质量评分的题目/模型/重复编号不完整')
    audit=[];recoveries=[];prompt_base=(out/'质量评分提示词快照.txt').read_text()
    for r in new:
        validate_quality(r['result']);t=next(t for t in tasks if t['id']==r['topic']);raw=read(out/'裁判原始记录'/f"{r['key']}.json")
        expected_prompt=prompt_base+'\n本次重复编号'+str(r['repeat'])+'\n用户任务：'+t['query']+'\n完整正文：\n'+t['body']
        if raw['prompt']!=expected_prompt or raw['prompt_sha256']!=hashlib.sha256(expected_prompt.encode()).hexdigest():raise ValueError('评分实际输入与冻结报告/提示词不一致')
        validate_quality(raw['result'])
        if raw['result']['dimensions']!=r['result']['dimensions']:raise ValueError('评分汇总与原始记录不一致')
        if raw.get('recovered_complete_quality_dimensions'):
            if recover_streamed_quality(raw['raw_response'])['dimensions']!=r['result']['dimensions']:raise ValueError('恢复评分不能从原始完整JSON维度对象重现')
            recoveries.append({'key':r['key'],'transport_complete':raw.get('transport_complete',True),'cause':r['result'].get('_quality_packet_recovery',r['result'].get('_transport_recovery',{})).get('cause','网络读取超时'),
                'native_dimension_objects':4,'scores':{d:r['result']['dimensions'][d]['score'] for d in DIMS},'raw_record':str((out/'裁判原始记录'/f"{r['key']}.json").resolve())})
        audit.append({'key':r['key'],'frozen_prompt_match':True,'raw_score_and_reason_match':True,'score_dimensions':4,'transport_complete':raw.get('transport_complete',True)})
    dump(out/'输入与评分完整性核查.json',{'date':now(),'expected_scores':20,'valid_scores':len(audit),'checks':audit,'note':'实际请求逐字等于冻结提示词、重复编号、用户任务和正文的组合；复核标签未拼入输入'})
    dump(out/'完整维度恢复核查.json',{'date':now(),'rule':'保留既有成功返回；异常响应只接收可原生JSON完整解码的四维对象，分数与理由不补全、不依据复核筛选','recovered':recoveries})
    means={};records={}
    for t in tasks:
        tid=t['id'];means[tid]={};records[tid]={}
        for model in ['qwen','claude']:
            group=[r for r in new if r['topic']==tid and r['model']==model]
            if {r['repeat'] for r in group}!={1,2} or len(group)!=2:raise ValueError('每模型每题必须正好两次评分')
            means[tid][model]={d:statistics.mean(r['result']['dimensions'][d]['score'] for r in group) for d in DIMS};records[tid][model]=group
        means[tid]['ensemble']={d:statistics.mean(means[tid][m][d] for m in ['qwen','claude']) for d in DIMS}
    rows=[];comparison={}
    for model in ['qwen','claude','ensemble']:
        old_pairs=[];new_pairs=[];old_errors={};new_errors={};by_dim={}
        for t in tasks:
            tid=t['id'];baseline=next(x for x in old if x['id']==tid);old_errors[tid]=[];new_errors[tid]=[]
            for d in DIMS:
                ref=review['quality'][str(tid)][d];before=baseline['dimensions'][d]['mean'] if model=='ensemble' else baseline['models'][model]['dimensions'][d];after=means[tid][model][d]
                old_pairs.append((before,ref));new_pairs.append((after,ref));old_errors[tid].append(abs(before-ref));new_errors[tid].append(abs(after-ref))
                rows.append({'topic':tid,'model':model,'dimension':d,'review':ref,'old':before,'new':after,'old_error':before-ref,'new_error':after-ref})
        for dim in DIMS:
            selected=[r for r in rows if r['model']==model and r['dimension']==dim];by_dim[dim]={'old':agreement([(r['old'],r['review']) for r in selected]),'new':agreement([(r['new'],r['review']) for r in selected])}
        comparison[model]={'old':agreement(old_pairs),'new':agreement(new_pairs),'by_dimension':by_dim,'delta':task_bootstrap_delta(old_errors,new_errors)}
        if model!='ensemble':
            diffs=[abs(records[t['id']][model][0]['result']['dimensions'][d]['score']-records[t['id']][model][1]['result']['dimensions'][d]['score']) for t in tasks for d in DIMS]
            comparison[model]['new_repeat_mean_absolute_difference']=statistics.mean(diffs)
            old_diffs=[]
            for baseline in old:
                group=[r for r in baseline['records'] if r['model']==model]
                if len(group)!=2:raise ValueError('旧版重复数不是2，不能直接对比')
                old_diffs.extend(abs(group[0]['dimensions'][d]['score']-group[1]['dimensions'][d]['score']) for d in DIMS)
            comparison[model]['old_repeat_mean_absolute_difference']=statistics.mean(old_diffs)
    inherited=inherited_comparison(read(out/'沿用逐条模型数据.json'),review,read(out/'沿用预审判断.json'))
    dump(out/'逐条沿用结果与复核一致性.json',inherited)
    machine={'date':now(),'sample_unit':'5篇×4维=20个均分坐标；重复调用不增加独立报告数','quality_comparison':comparison,'score_rows':rows,'new_means':means,'new_records':new,
      'old_new_inter_judge_mean_absolute_gap':{'old':statistics.mean(abs(t['models']['qwen']['dimensions'][d]-t['models']['claude']['dimensions'][d]) for t in old for d in DIMS),'new':statistics.mean(abs(means[t['id']]['qwen'][d]-means[t['id']]['claude'][d]) for t in tasks for d in DIMS)},
      'inherited_groups':inherited['groups'],'limitations':['复核参照不是独立多人金标','仅5篇报告，维度与重复调用不独立','与旧版是不同时间重复测量，模型随机性及服务版本可能变化；不能纯因果归因few-shot','相关性不能替代分数接近','逐条结果未重跑，不能声称few-shot改善逐条标签']}
    usage=[]
    for p in sorted((out/'裁判原始记录').glob('*.json')):
        x=read(p);u=x.get('usage',{});total=u.get('total_tokens')
        if total is None and 'input_tokens' in u and 'output_tokens' in u:total=u['input_tokens']+u['output_tokens']
        if x.get('transport_complete') is False:total=None
        usage.append({'key':x['key'],'response_model':x.get('response_model'),'total_tokens':total,'usage':u,'elapsed_s':x.get('elapsed_s'),'errors':x.get('errors',[]),'transport_complete':x.get('transport_complete',True)})
    machine['calls']={'successful':len(new),'known_tokens':sum(x['total_tokens'] or 0 for x in usage),'unknown_usage_calls':sum(x['total_tokens'] is None for x in usage),
      'format_normalized_results':sum('_format_normalization' in r['result'] for r in new),
      'transport_recovered_quality_results':sum('_transport_recovery' in r['result'] for r in new),
      'malformed_metadata_recovered_quality_results':sum(r['result'].get('_quality_packet_recovery',{}).get('cause')=='附加输出格式异常' for r in new),
      'saved_failed_attempts':len(list((out/'失败尝试').glob('*.json'))),
      'notice':'successful为有效评分数量，包含缓存复用；不是HTTP请求总次数。失败或中断请求的服务端消耗可能不可见。','records':usage}
    dump(out/'一致性汇总.json',machine)
    print(json.dumps({'quality':{k:{'old_mae':v['old']['mae'],'new_mae':v['new']['mae'],'new_within_1':v['new']['within_1_percent']} for k,v in comparison.items()},'inherited':{k:{f:v.get(f) for f in ['n','percent','mae']} for k,v in inherited['groups'].items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','analyze']);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    {'prepare':prepare,'run':run,'analyze':analyze}[args.mode](args.out.resolve())
