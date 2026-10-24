#!/usr/bin/env python3
"""汇总可追溯初评，并生成不依赖网络的人工评分工作台。"""
from pathlib import Path
import json, hashlib, statistics, math, html, re, csv, argparse
from collections import Counter
import markdown
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'backend/eval/data/五篇报告正文与Scout独立评估-20261003-234200'
DOC=ROOT/'docs/五篇报告正文与Scout独立评估分析-20261004.md'
PAGE=ROOT/'docs/五篇报告人工评分工作台-20261004.html'
DATE='2026-10-04'
TITLES={35:'市政污水与雨污协同',31:'中外博物馆教育',46:'房地产可持续发展',16:'非接触式感知算法',49:'正畸与医美共同需求'}
DIMS={'depth':'深度','comprehensiveness':'全面性','readability':'可读性','requirements':'要求覆盖'}
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def rate(a,b):return {'numerator':a,'denominator':b,'percent':round(a/b*100,2) if b else None}
def fmt(m):return f"{m['percent']:.1f}% ({m['numerator']}/{m['denominator']})" if m['percent'] is not None else '不适用（分母为0）'
def load_result(key):
    p=OUT/'裁判原始记录'/f'{key}.json'
    return read(p).get('result') if p.exists() else None
def prepare():
    tasks=read(OUT/'冻结输入.json');rows=read(OUT/'论断与证据材料.json');nli={r['id']:r for r in read(OUT/'NLI窗口筛查.json')};external=read(OUT/'外部核查与评审复核.json')
    first={};second={}
    for p in sorted((OUT/'裁判原始记录').glob('*-支持-*.json')):
        for r in (read(p).get('result') or {}).get('items',[]):first[r['id']]=r
    for p in sorted((OUT/'裁判原始记录').glob('*-复核-*.json')):
        for r in (read(p).get('result') or {}).get('items',[]):second[r['id']]=r
    date_calibrated={}
    for p in sorted((OUT/'裁判原始记录').glob('*-日期校准-*.json')):
        for r in (read(p).get('result') or {}).get('items',[]):date_calibrated[r['id']]=r
    overrides={
      '35-scout-009':('unknown','只有大连转载页标题及网页发布日期，无法判断文件制定层级；不能据转载机关证明与省级规范矛盾。'),
      '31-pre-044':('full','2026年发布的2025年统计是过去期统计，不能当未来预测；数字与保存资料一致。'),
      '31-pre-045':('full','2026年公报可以统计2025年末从业人员，首判时间逻辑错误。'),
      '31-pre-052':('full','7188是2025年末实际统计，并非未来预测。'),
      '46-scout-015':('contradicted','原来源标题是十五五规划纲要，Scout改成规划建议，并把页面日期当建议发布时间。'),
      '35-body-053':('partial','旧报批稿能证明稿件存在，不能证明截至2026年尚未实施；官方标准已于2022年实施。'),
      '35-pre-012':('partial','旧报批稿不足以证明当前未实施；官方标准已于2022年实施。'),
      '31-body-034':('contradicted','来源和定向核查证明88.6%为免费开放占比，改变为国有占比。'),
      '49-body-009':('partial','19–23等分组无法精确得出20–33岁占比，且原表文本分母未清晰可核。'),
      '49-body-017':('contradicted','按原分组累加的比例不能改写成年龄段内女性比例；原数据起点19岁。')}
    unknown_ids={'35-scout-033','35-scout-034','35-body-083','35-body-084','35-body-085'}
    confirmed={cid:c for c in external['cases'] if c['status'].startswith('confirmed') for cid in c.get('claim_ids',[])}
    for c in external['cases']:
        for cid in c.get('additional_error_ids',[]):confirmed[cid]=c
    for r in rows:
        r['initial_judgment']=first.get(r['id']);r['second_judgment']=second.get(r['id'])
        r['date_calibrated_judgment']=date_calibrated.get(r['id'])
        j=date_calibrated.get(r['id']) or second.get(r['id']) or first.get(r['id'])
        if not j:j={'joint_support':'unsupported' if not r['refs'] else 'unknown','ref_support':[],'reason':'正文未标引用。' if not r['refs'] else '未获得模型判定。','confidence':1 if not r['refs'] else 0}
        r['judgment']=dict(j)
        if r['id'] in unknown_ids:r['judgment'].update(joint_support='unknown',reason='保存的.doc为二进制乱码，无法验证；不把取证失败算成反事实矛盾。',reviewer_override=True)
        if r['id'] in overrides:
            label,reason=overrides[r['id']];r['judgment'].update(joint_support=label,reason=reason,reviewer_override=True)
        r['confirmed_error']=r['id'] in confirmed
        r['external_cases']=[c for c in external['cases'] if r['id'] in c.get('claim_ids',[])+c.get('additional_error_ids',[])]
        nr=nli.get(r['id']);r['nli']={k:v for k,v in nr.items() if k!='windows'} if nr else None
    # 外核率不把纯建议与语境待裁决算事实错误；统计按出现计，家族数分开。
    metrics=[];qual=[];cover=[];sources=[];audits=[]
    for t in tasks:
        tid=t['id']; rr=[r for r in rows if r['topic']==tid];body=[r for r in rr if r['kind']=='body' and r['type']=='factual'];scout=[r for r in rr if r['kind']=='scout'];pre=[r for r in rr if r['kind']=='pre']
        checkpoints=read(Path(t['folder'])/'完整检查点历史.json')
        first_feedback=next(x['checkpoint']['channel_values']['research']['critic_feedback'] for x in reversed(checkpoints) if x['checkpoint']['channel_values'].get('research',{}).get('critic_feedback'))
        first_signatures={(i.get('location'),i.get('description')) for i in first_feedback}
        t['first_critic_issue_ids']=[f"{tid}-{i['audit_id']}" for i in t['critic_issues'] if (i.get('location'),i.get('description')) in first_signatures]
        cq=load_result(f'题{tid}-Scout覆盖与重复') or {'facets':[],'duplicate_groups':[]}
        facts={r['original_id']:r for r in scout}
        for i,f in enumerate(cq['facets']):
            f.update(id=f'{tid}-facet-{i+1:02d}',topic=tid,kind='facet')
            if tid==46 and f['facet']=='未来十年时间路径':
                f.update(initial_coverage=f['coverage'],coverage='partial',reviewer_override=True,reason='2025年的历史节点和2026–2030的五年规划不能构成截至2026评估时点的未来十年路径；缺2031–2036阶段或情景。')
            validids=[x for x in f.get('fact_ids',[]) if x in facts]
            f['supporting_fact_ids']=[facts[x]['id'] for x in validids]
            f['effective_full']=f['coverage']=='full' and any(facts[x]['judgment']['joint_support']=='full' and not facts[x]['confirmed_error'] for x in validids)
            cover.append(f)
        dup=set();dupmap={}
        for i,g in enumerate(cq['duplicate_groups']):
            ids=[x for x in g['ids'] if x in facts]
            for x in ids[1:]:dup.add(x);dupmap[x]={'of':facts[ids[0]]['id'],'reason':g['reason']}
        for r in scout:r['model_duplicate']=dupmap.get(r['original_id'])
        sq=load_result(f'题{tid}-来源质量') or {'sources':[]};sm={x['id']:x for x in sq['sources']}
        for s in t['sources']:
            q=sm.get(s['id'],{'score':None,'primary':False,'tier':'未判定','reason':'未获得裁判输出'})
            if s['id']=='source_a129b947':q={**q,'score':1,'primary':False,'reason':'保存正文是二进制乱码；域名权威不能替代可读内容。'}
            sources.append({'id':f"{tid}-source-{s['id']}",'topic':tid,'kind':'source','original_id':s['id'],'title':s['title'],'url':s['url'],'content':s['content'],'score':q.get('score'),'primary':q.get('primary'),'tier':q.get('tier'),'reason':q.get('reason'),'risk':q.get('risk','')})
        sourceitems=[s for s in sources if s['topic']==tid];slookup={s['original_id']:s for s in sourceitems}
        audit=load_result(f'题{tid}-审核检出') or {'pre_claims':[],'critic_issues':[]}
        ap={a['id']:a for a in audit['pre_claims']};ip={i['audit_id']:i for i in audit['critic_issues']}
        for issue in t['critic_issues']:
            iid=f"{tid}-{issue['audit_id']}"
            audits.append({**issue,'id':iid,'topic':tid,'kind':'audit','first_review':iid in t['first_critic_issue_ids'],'model_review':ip.get(issue['audit_id'])})
        for r in pre:r['audit_model']=ap.get(r['id'])
        # 明确核查过的预审错误家族：不使用模型把2025历史数据当预测的伪金标。
        gold=[r for r in pre if r['confirmed_error']]
        for r in gold:
            r['verified_audit_error']=True
            r['verified_first_detected']=False
            r['verified_any_detected']=r['id']=='49-pre-029'
        bfull=sum(r['judgment']['joint_support']=='full' for r in body)
        cited=[r for r in body if r['refs']];bcover=sum(r['judgment']['joint_support']=='full' and r['citations_explicit'] for r in body)
        pairs=[(r,ref) for r in cited for ref in r['refs']]
        def reflabel(r,ref):
            if r['judgment'].get('reviewer_override') and len(r['refs'])==1:return r['judgment']['joint_support']
            return next((x['label'] for x in r['judgment'].get('ref_support',[]) if x['id']==ref['id']),'unknown')
        scoutpairs=[(r,ref) for r in scout for ref in r['refs']]
        primaryfact=sum(any(slookup.get(a['id'],{}).get('primary') for a in r['refs']) for r in scout)
        scorevalues=[s['score'] for s in sourceitems if s['score'] is not None]
        criticvalid=sum((a['model_review'] or {}).get('validity')=='valid' for a in audits if a['topic']==tid)
        criticunknown=sum((a['model_review'] or {}).get('validity')=='uncertain' for a in audits if a['topic']==tid)
        row={'id':tid,'title':TITLES[tid],'body_claims':len(body),'body_cited_claims':len(cited),'body_support_counts':dict(Counter(r['judgment']['joint_support'] for r in body)),
          'citation_support':rate(sum(reflabel(r,a)=='full' for r,a in pairs),len(pairs)),
          'effective_evidence_coverage':rate(bcover,len(body)),
          'unsupported_strict':rate(len(body)-bcover,len(body)),
          'unsupported_no_or_none':rate(sum(r['judgment']['joint_support'] in ['unsupported','contradicted'] for r in body),len(body)),
          'confirmed_error_lower_bound':rate(sum(r['confirmed_error'] for r in body),len(body)),
          'scout_facts':len(scout),'scout_support_counts':dict(Counter(r['judgment']['joint_support'] for r in scout)),
          'scout_support':rate(sum(r['judgment']['joint_support']=='full' for r in scout),len(scout)),
          'scout_locator':rate(sum(r['locator_valid'] for r in scout),len(scout)),
          'scout_evidence_correctness':rate(sum(a['locator_valid'] and reflabel(r,a)=='full' for r,a in scoutpairs),len(scoutpairs)),
          'scout_coverage':rate(sum(f['effective_full'] for f in cq['facets']),len(cq['facets'])),
          'scout_duplicate':rate(len(dup),len(scout)),
          'source_score_mean':round(statistics.mean(scorevalues),3) if scorevalues else None,
          'source_records':len(sourceitems),'source_score_counts':dict(Counter(s['score'] for s in sourceitems)),
          'primary_source_fact_share':rate(primaryfact,len(scout)),
          'audit_first_confirmed_recall':rate(sum(r.get('verified_first_detected',False) for r in gold),len(gold)),
          'audit_any_confirmed_recall':rate(sum(r.get('verified_any_detected',False) for r in gold),len(gold)),
          'audit_model_precision':rate(criticvalid,len(t['critic_issues'])-criticunknown),'audit_uncertain':criticunknown,
          'audit_pre_sample':len(pre),'critic_issues':len(t['critic_issues'])}
        metrics.append(row)
        quality={'id':tid,'models':{},'dimensions':{},'records':[]}
        for model in ['qwen','claude']:
            records=[]
            for rep in [1,2]:
                q=load_result(f'题{tid}-质量-{model}-{rep}')
                if q:
                    normalized={d:q.get('dimensions',{}).get(d) for d in DIMS}
                    if all(isinstance(x,dict) and isinstance(x.get('score'),(int,float)) for x in normalized.values()):records.append({'model':model,'repeat':rep,'dimensions':normalized,'strengths':q.get('strengths',[]),'weaknesses':q.get('weaknesses',[])})
            quality['records']+=records
            quality['models'][model]={'n':len(records),'dimensions':{d:round(statistics.mean(r['dimensions'][d]['score'] for r in records),3) if records else None for d in DIMS}}
        for d in DIMS:
            scores=[r['dimensions'][d]['score'] for r in quality['records']]
            quality['dimensions'][d]={'mean':round(statistics.mean(scores),3) if scores else None,'sd':round(statistics.stdev(scores),3) if len(scores)>1 else None,'range':[min(scores),max(scores)] if scores else [],'n':len(scores)}
        quality['overall']=round(statistics.mean(v['mean'] for v in quality['dimensions'].values()),3) if quality['records'] else None
        qual.append(quality)
        t['title']=TITLES[tid];t['report_html']=markdown.markdown(html.escape(t['report']),extensions=['tables','fenced_code'])
    disagreement=sum(first[k]['joint_support']!=second[k]['joint_support'] for k in second if k in first)
    eligible=[r for r in rows if r['kind']!='body' or r.get('type')=='factual']
    nr=[r for r in eligible if r.get('nli') and r['refs']]
    config=read(OUT/'评估配置.json')
    summary={'metrics':metrics,'quality':qual,'initial_second_disagreement':rate(disagreement,len(second)),'nli_vs_final_agreement':rate(sum(r['nli']['label']==r['judgment']['joint_support'] for r in nr),len(nr)),'nli_counts':dict(Counter(r['nli']['label'] for r in eligible if r.get('nli'))),'nli_model':config['nli']['id'],'pending_human':True}
    old_nli_path=OUT/'NLI旧版窗口筛查.json'
    if old_nli_path.exists():
        old={r['id']:r for r in read(old_nli_path)}
        comparable=[r for r in nr if r['id'] in old]
        summary['nli_previous_comparison']={'previous_model':'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli','agreement':rate(sum(old[r['id']]['label']==r['nli']['label'] for r in comparable),len(comparable)),
            'by_topic':{str(t['id']):{'old_counts':dict(Counter(old[r['id']]['label'] for r in comparable if r['topic']==t['id'])),'new_counts':dict(Counter(r['nli']['label'] for r in comparable if r['topic']==t['id'])),'changed':sum(old[r['id']]['label']!=r['nli']['label'] for r in comparable if r['topic']==t['id'])} for t in tasks}}
    dump(OUT/'逐条初评与复核.json',rows);dump(OUT/'指标汇总.json',summary)
    dump(OUT/'审核问题初评.json',audits);dump(OUT/'覆盖与来源质量初评.json',{'coverage':cover,'sources':[{k:v for k,v in s.items() if k!='content'} for s in sources]})
    # 全部人工字段空白，数据集指纹固定，支持独立导出与再次计算。
    dataset_id=hashlib.sha256(''.join(t['report_sha256'] for t in tasks).encode()).hexdigest()[:20]
    dataset={'dataset_id':dataset_id,'date':DATE,'tasks':tasks,'rows':rows,'coverage':cover,'sources':sources,'audits':audits,'summary':summary,'external':external,'dimensions':DIMS}
    dump(OUT/'评分网页数据.json',dataset)
    return dataset

HTML=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五篇报告 · 人工评估工作台</title><style>
:root{--ink:#19283e;--muted:#63728a;--line:#dce4ed;--blue:#255dc1;--soft:#eef4fd;--orange:#925515}*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}button,select,input,textarea{font:inherit;color:inherit}button{cursor:pointer}header{background:#172942;color:white;padding:24px 32px}header h1{font-size:25px;margin:0}header p{margin:4px 0;color:#bfcee2;font-size:13px}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:9px;margin-top:15px}.toolbar button{border:1px solid #70809b;background:#233b5c;color:white;border-radius:6px;padding:5px 12px}.saved{margin-left:auto;font-size:12px}main{max-width:1500px;margin:auto;padding:22px 28px}.topics{display:flex;gap:8px;flex-wrap:wrap}.topics button{background:white;border:1px solid var(--line);padding:10px 15px;border-radius:8px}.topics .active{background:var(--blue);color:white;border-color:var(--blue)}.query{margin:15px 0 18px;padding:14px 18px;background:white;border-left:3px solid var(--blue);font-size:14px}.tabs{display:flex;gap:5px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:20px}.tabs button{border:0;background:transparent;padding:12px 17px;color:var(--muted)}.tabs .active{color:var(--blue);border-bottom:3px solid var(--blue);font-weight:600}.box{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.scorelabel{display:block;color:var(--muted);font-size:13px}.scorebox input{width:100%;padding:9px;border:1px solid var(--line);border-radius:7px;margin-top:8px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.metric{background:var(--soft);border-radius:7px;padding:11px 14px}.metric b{font-size:20px;display:block}.metric small,.muted{color:var(--muted)}.notice{font-size:13px;background:#fff7e9;padding:12px 15px;border-radius:7px;color:#755125}.filter{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}.filter input{flex:1;min-width:150px}.filter input,.filter select{padding:8px;border:1px solid var(--line);border-radius:7px;background:white}.layout{display:grid;grid-template-columns:minmax(300px,40%) minmax(400px,60%);gap:16px;align-items:start}.list{background:white;border:1px solid var(--line);border-radius:9px;max-height:750px;overflow:auto}.item{width:100%;padding:14px;text-align:left;border:0;border-bottom:1px solid var(--line);background:white;display:block}.item.active{background:var(--soft);box-shadow:inset 3px 0 var(--blue)}.item p{margin:4px 0;font-size:14px}.item small{color:var(--muted)}.tag{display:inline-block;background:#edf1f7;padding:1px 7px;border-radius:5px;font-size:11px;margin-right:5px}.done{color:#26744f;background:#e7f5ed}.detail{background:white;border:1px solid var(--line);border-radius:9px;padding:20px;min-height:320px}.detail h3{margin:0 0 9px;font-size:18px}.detail select,.detail textarea,.detail input{border:1px solid var(--line);border-radius:6px;padding:7px;margin:5px 0;width:100%}.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}.evidence{border-left:3px solid #9cb5db;background:#f6f8fc;padding:12px;margin:12px 0;font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:440px;overflow:auto}a{color:var(--blue);overflow-wrap:anywhere}details{margin:10px 0}summary{cursor:pointer;color:var(--blue)}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fc;padding:12px;max-height:500px;overflow:auto}.model{overflow-wrap:anywhere;min-width:0;border:1px dashed #ba9a68;background:#fff8ed;padding:12px;font-size:13px;margin:12px 0}.pager{display:flex;gap:12px;align-items:center;margin:12px 0}.pager button{background:white;border:1px solid var(--line);border-radius:5px;padding:5px 10px}.report{max-height:800px;overflow:auto;padding:15px;font-size:14px}.report h2{font-size:20px}.report h3{font-size:16px}.report table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}.report td,.report th{border:1px solid var(--line);padding:8px}.hide{display:none}.foot{font-size:12px;color:var(--muted);margin:30px 0}#toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);background:#172942;color:white;padding:10px 20px;border-radius:8px;display:none;z-index:9}@media(max-width:900px){header{padding:20px}main{padding:15px}.grid{grid-template-columns:1fr 1fr}.metrics{grid-template-columns:1fr 1fr}.layout{display:block}.list{max-height:360px;margin-bottom:15px}.fields{grid-template-columns:1fr}.saved{margin-left:0}}@media print{header,.topics,.tabs,.filter,.pager,.toolbar{display:none}.layout{display:block}.report,.evidence,pre,.list{max-height:none}.model{display:block}}
</style></head><body><header><h1>五篇报告 · 人工评估工作台</h1><p>正文质量 / 证据支持 / Scout / 审核 · 截止资料日期 2026-10-03 · 本地离线可用</p><div class="toolbar"><button id="export">导出评分 JSON</button><button id="csv">导出逐条 CSV</button><button id="import">导入评分</button><input id="file" type="file" accept="application/json" hidden><label><input id="blind" type="checkbox"> 显示模型初评</label><span class="saved" id="saved">人工评分留空，尚未开始</span></div></header><main><div class="topics" id="topics"></div><div class="query" id="query"></div><nav class="tabs" id="tabs"></nav><div id="content"></div><p class="foot">评分只保存在当前浏览器；建议每轮完成后导出 JSON 备份。模型初评、外部核查和人工意见分别保存。待标注项不按错误计数。来源忠实度不等于现实事实正确性。</p></main><div id="toast" role="status"></div><script id="dataset" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('dataset').textContent), KEY='research-review-'+D.dataset_id;
let A={quality:{},items:{}}, topic=D.tasks[0].id, tab='quality', page=0, selected=null, query='', filter='all', showModel=false;
try{const x=JSON.parse(localStorage.getItem(KEY));if(x&&x.dataset_id===D.dataset_id)A=x.annotations}catch(e){}
const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={full:'完整支持',partial:'部分支持',unsupported:'无支持',contradicted:'证据明确相反',unknown:'无法判定'}, yesno={yes:'是',no:'否',unknown:'无法判定'}, covered={full:'充分覆盖',partial:'部分覆盖',absent:'未覆盖'}, valid={valid:'有效问题',invalid:'误报',uncertain:'待核实'};
function safeUrl(u){return /^https?:\/\//.test(u)?u:'#'}
function toast(t){$('toast').textContent=t;$('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',2600)}
function save(){try{localStorage.setItem(KEY,JSON.stringify({dataset_id:D.dataset_id,annotations:A}));$('saved').textContent='已在本地保存 · '+new Date().toLocaleTimeString()}catch(e){$('saved').textContent='本地存储失败，请立即导出 JSON';toast('浏览器不允许保存，请导出备份')}}
function setItem(id,key,value){A.items[id]??={};A.items[id][key]=value;save();updateMetrics()}
function opts(map,value){return '<option value="">待评分</option>'+Object.entries(map).map(([k,v])=>`<option value="${k}" ${String(value)===k?'selected':''}>${v}</option>`).join('')}
function field(id,key,title,map){return `<label>${title}<select data-id="${esc(id)}" data-key="${key}">${opts(map,(A.items[id]||{})[key])}</select></label>`}
function bindFields(){document.querySelectorAll('[data-key]').forEach(e=>e.addEventListener('change',()=>setItem(e.dataset.id,e.dataset.key,e.value)))}
function pct(a,b){return b?`${(100*a/b).toFixed(1)}% <small>(${a}/${b})</small>`:'— <small>尚无评分</small>'}
function hm(rows,key,test){const marked=rows.filter(r=>(A.items[r.id]||{})[key]);return {s:marked.filter(r=>test((A.items[r.id]||{})[key],r)).length,n:marked.length,total:rows.length}}
function metric(name,r){return `<div class="metric"><span>${name}</span><b>${pct(r.s,r.n)}</b><small>已判 ${r.n}/${r.total}；分母为已判项</small></div>`}
function humanStats(){const rows=D.rows.filter(r=>r.topic===topic),body=rows.filter(r=>r.kind==='body'&&r.type==='factual'),scout=rows.filter(r=>r.kind==='scout'),pre=rows.filter(r=>r.kind==='pre'),cov=D.coverage.filter(r=>r.topic===topic),src=D.sources.filter(r=>r.topic===topic),audit=D.audits.filter(r=>r.topic===topic);
 const b=hm(body,'support',(v,r)=>v==='full'&&r.citations_explicit),bu=hm(body,'support',(v,r)=>v!=='full'||!r.citations_explicit),be=hm(body,'error',v=>v==='yes'),s=hm(scout,'support',v=>v==='full'),dup=hm(scout,'duplicate',v=>v==='yes'),c=hm(cov,'coverage',v=>v==='full'),cr=hm(audit,'validity',v=>v==='valid');
 const pairs=body.flatMap(r=>r.refs.map((x,i)=>({id:r.id+'-ref-'+i}))),sp=scout.flatMap(r=>r.refs.map((x,i)=>({id:r.id+'-ref-'+i}))); const spmarked=sp.filter(r=>(A.items[r.id]||{}).support&&(A.items[r.id]||{}).locator);const spevidence={s:spmarked.filter(r=>A.items[r.id].support==='full'&&A.items[r.id].locator==='yes').length,n:spmarked.length,total:sp.length};
 const rec=pre.filter(r=>(A.items[r.id]||{}).error==='yes'&&(A.items[r.id]||{}).detected);const marks=src.filter(r=>(A.items[r.id]||{}).score);let out=metric('正文有效证据覆盖',b)+metric('正文严格无支持',bu)+metric('正文明确错误 / 矛盾',be)+metric('逐条引用支持',hm(pairs,'support',v=>v==='full'))+metric('Scout 事实支持',s)+metric('Scout 证据正确率',spevidence)+metric('Scout 需求要点覆盖',c)+metric('Scout 重复信息',dup)+metric('Critic 问题有效率',cr)+`<div class="metric"><span>预审错误检出率</span><b>${pct(rec.filter(r=>A.items[r.id].detected==='yes').length,rec.length)}</b><small>仅已标为错误且已判断检出的预审样本 ${rec.length}/${pre.length}</small></div>`+`<div class="metric"><span>来源质量均分</span><b>${marks.length?(marks.reduce((a,r)=>a+Number(A.items[r.id].score),0)/marks.length).toFixed(2):'—'} <small>/ 5</small></b><small>已判 ${marks.length}/${src.length}</small></div>`;
 return '<div class="metrics">'+out+'</div>'}
function updateMetrics(){if($('humanmetrics'))$('humanmetrics').innerHTML=humanStats()}
function render(){const task=D.tasks.find(t=>t.id===topic);$('topics').innerHTML=D.tasks.map(t=>`<button class="${t.id===topic?'active':''}" data-topic="${t.id}">${t.id} · ${esc(t.title)}</button>`).join('');document.querySelectorAll('[data-topic]').forEach(b=>b.onclick=()=>{topic=Number(b.dataset.topic);page=0;selected=null;render()});$('query').innerHTML='<b>原始任务</b> · '+esc(task.query);
 const tabs={quality:'正文质量',body:'正文事实',scout:'Scout 事实',facet:'需求覆盖',pre:'预审检出',audit:'Critic 问题',source:'来源质量'};$('tabs').innerHTML=Object.entries(tabs).map(([k,v])=>`<button data-tab="${k}" class="${tab===k?'active':''}">${v}</button>`).join('');document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{tab=b.dataset.tab;page=0;selected=null;query='';filter='all';render()});
 if(tab==='quality')renderQuality(task);else renderRows();}
function renderQuality(task){const q=A.quality[topic]||{},model=D.summary.quality.find(q=>q.id===topic),m=D.summary.metrics.find(x=>x.id===topic);$('content').innerHTML=`<div class="notice">建议先阅读报告，独立打四项 0–10 分，再打开模型初评。0–2 严重缺失，3–4 较差，5–6 基本可用，7–8 较好，9–10 充分且严谨。</div><div class="box"><div class="grid">${Object.entries(D.dimensions).map(([d,n])=>`<label class="scorebox"><span class="scorelabel">${n} / 10</span><input type="number" min="0" max="10" step="0.5" value="${q[d]??''}" data-quality="${d}" placeholder="留空表示待评分"></label>`).join('')}</div><p class="muted">深度看机制、比较、反证和推导；全面性看视角与证据；可读性看结构与重复；要求覆盖看原始需求的限定。</p><label>评审备注<textarea id="qnote" rows="3" style="width:100%;border:1px solid #dce4ed;border-radius:6px">${esc(q.note||'')}</textarea></label></div><details class="box" open><summary>阅读完整报告</summary><article class="report">${task.report_html}</article></details><div class="box"><h3>你的逐条评分指标</h3><div id="humanmetrics">${humanStats()}</div></div>${showModel?`<div class="model"><b>模型初评（非人工金标）</b><p>各维度均分：${Object.entries(model.dimensions).map(([d,x])=>`${D.dimensions[d]} ${x.mean??'待完成'}，SD ${x.sd??'—'}`).join(' · ')}</p><p>Qwen：${JSON.stringify(model.models.qwen.dimensions)}<br>Claude：${JSON.stringify(model.models.claude.dimensions)}</p><p>正文有效证据覆盖 ${m.effective_evidence_coverage.percent}%；已核实错误下限 ${m.confirmed_error_lower_bound.percent}%；Scout 完整支持 ${m.scout_support.percent}%</p>${model.records.map(r=>`<details><summary>${r.model} · 第${r.repeat}次 · 理由</summary>${Object.entries(r.dimensions).map(([d,v])=>`<p><b>${D.dimensions[d]} ${v.score}</b> · ${esc(v.reason)}</p>`).join('')}</details>`).join('')}</div>`:''}`;
 document.querySelectorAll('[data-quality]').forEach(e=>e.onchange=()=>{if(e.value!==''&&(!Number.isFinite(Number(e.value))||Number(e.value)<0||Number(e.value)>10)){toast('分数须在 0–10');e.value=q[e.dataset.quality]??'';return}A.quality[topic]??={};A.quality[topic][e.dataset.quality]=e.value===''?null:Number(e.value);save()});$('qnote').onchange=()=>{A.quality[topic]??={};A.quality[topic].note=$('qnote').value;save()};}
function allRows(){if(['body','scout','pre'].includes(tab))return D.rows.filter(r=>r.topic===topic&&r.kind===tab&&(tab!=='body'||r.type==='factual'));if(tab==='facet')return D.coverage.filter(r=>r.topic===topic);if(tab==='audit')return D.audits.filter(r=>r.topic===topic);return D.sources.filter(r=>r.topic===topic)}
function textOf(r){return r.claim||r.facet||r.title||r.description}
function isDone(r){const a=A.items[r.id]||{};return tab==='source'?!!a.score:tab==='facet'?!!a.coverage:tab==='audit'?!!a.validity:!!a.support}
function renderRows(){let rows=allRows().filter(r=>!query||JSON.stringify([textOf(r),r.id,r.section]).toLowerCase().includes(query.toLowerCase()));if(filter==='pending')rows=rows.filter(r=>!isDone(r));if(filter==='marked')rows=rows.filter(isDone);if(filter==='important')rows=rows.filter(r=>r.important);if(filter==='nocite')rows=rows.filter(r=>!r.citations_explicit);const pages=Math.max(1,Math.ceil(rows.length/20));page=Math.min(page,pages-1);const view=rows.slice(page*20,page*20+20);if(!view.some(r=>r.id===selected))selected=view[0]?.id;
 $('content').innerHTML=`<div class="notice">${tab==='pre'?'这里是第一次审核前的 198 条 Writer 结构化论断样本；先判断是否错误，再核对 Critic 是否检出同一错误。不是完整报告错误召回率。':'完整支持须保留对象、时间、数值、分母和条件。部分支持、证据相反、证据不可用分别标注。'}</div><div class="filter"><input id="search" placeholder="搜索论断、编号或章节" value="${esc(query)}"><select id="filter"><option value="all">全部</option><option value="pending">待评分</option><option value="marked">已评分</option><option value="important">核心论断（模型抽取）</option><option value="nocite">未标引用</option></select><span>${rows.length} 项</span></div><div class="pager"><button id="prev">上一页</button><span>${page+1} / ${pages}</span><button id="next">下一页</button></div><div class="layout"><div class="list">${view.map(r=>`<button class="item ${r.id===selected?'active':''}" data-row="${esc(r.id)}"><small>${esc(r.id)} · ${esc(r.section||'')}</small>${isDone(r)?'<span class="tag done">已评分</span>':''}<p>${esc(textOf(r))}</p></button>`).join('')||'<p style="padding:15px">没有符合条件的条目。</p>'}</div><div id="detail" class="detail"></div></div><details class="box"><summary>查看你的指标与标注进度</summary><div id="humanmetrics">${humanStats()}</div></details>`;
 $('filter').value=filter;$('filter').onchange=()=>{filter=$('filter').value;page=0;renderRows()};$('search').oninput=()=>{query=$('search').value;page=0;const pos=$('search').selectionStart;renderRows();$('search').focus();$('search').setSelectionRange(pos,pos)};$('prev').onclick=()=>{page=Math.max(0,page-1);renderRows()};$('next').onclick=()=>{page=Math.min(pages-1,page+1);renderRows()};document.querySelectorAll('[data-row]').forEach(b=>b.onclick=()=>{selected=b.dataset.row;renderRows()});renderDetail(view.find(r=>r.id===selected));}
function renderDetail(r){if(!r){$('detail').textContent='请选择条目';return}const a=A.items[r.id]||{};let out=`<small class="muted">${esc(r.id)}</small><h3>${esc(textOf(r))}</h3>`;
 if(['body','scout','pre'].includes(tab)){out+=`<div class="fields">${field(r.id,'support','来源对论断的支持',labels)}${field(r.id,'error','论断有明确错误 / 矛盾？',yesno)}${tab==='scout'?field(r.id,'duplicate','相对其他 Scout 事实重复？',yesno):''}${tab==='pre'?field(r.id,'detected','首次 Critic 检出了同一错误？',yesno):''}</div>`;
 if(tab==='scout')out+=`<label>若重复，重复于哪条编号？<input data-id="${r.id}" data-key="duplicate_of" value="${esc(a.duplicate_of||'')}" placeholder="如 31-scout-050"></label>`;
 if(tab==='pre')out+=`<label>匹配的审核问题编号<input data-id="${r.id}" data-key="matched_issue" value="${esc(a.matched_issue||'')}" placeholder="必须指出同一错误，只有要求加引用不算检出分母错误"></label>`;
 if(r.paragraph)out+=`<details><summary>正文上下文 / 原始段落</summary><pre>${esc(r.paragraph)}</pre></details>`;
 if(!r.refs.length)out+='<p class="notice">正文没有明确引用。写作时保留的内部证据可用于复核，但不能算已发表正文的引用覆盖。</p>';
 r.refs.forEach((ref,i)=>{const rid=r.id+'-ref-'+i;out+=`<div class="box"><a href="${esc(safeUrl(ref.url))}" target="_blank" rel="noreferrer">${esc(ref.title||ref.url)}</a><small class="muted"> · ${esc(ref.document_id||'')} · ${ref.locator_valid?'哈希/位置可核':'定位不可核'}</small><div class="fields">${field(rid,'support','这一条引用是否支持？',labels)}${field(rid,'locator','来源身份和引用定位正确？',yesno)}</div>${ref.windows.map(w=>`<div class="evidence">${esc(w.quote)}<br><small>原文位置 ${w.start}–${w.end}；${esc(w.sentence_ids.join(', '))}</small></div>`).join('')}${fullSource(ref.id)}</div>`});
 if(r.latent_refs?.length)out+=`<details><summary>内部保留证据（不计正文引用）</summary>${r.latent_refs.map(ref=>`<p>${esc(ref.title)}</p><pre>${esc(ref.windows.map(w=>w.quote).join('\n'))}</pre>`).join('')}</details>`;
 if(tab==='pre'){out+=`<details><summary>第一次审核前全文</summary><pre>${esc(JSON.stringify(D.tasks.find(t=>t.id===topic).pre_review_draft,null,2))}</pre></details><details><summary>查看本题首次 Critic 问题（检出匹配限定于此）</summary><pre>${esc(JSON.stringify(D.audits.filter(x=>x.topic===topic&&x.first_review).map(x=>({id:x.id,location:x.location,description:x.description})),null,2))}</pre></details>`}
 if(showModel)out+=`<div class="model"><b>模型初评：${labels[r.judgment.joint_support]}</b><p>${esc(r.judgment.reason)}</p><p>NLI 筛查 · ${esc(r.nli?.model||D.summary.nli_model)}：${esc(labels[r.nli?.label]||r.nli?.label||'无')} ${r.nli?.model==='qwen3-max'?'（语义分类，无概率阈值）':`；支持最大分 ${r.nli?.support_max?.toFixed(3)??'—'}；矛盾最大分 ${r.nli?.contradiction_max?.toFixed(3)??'—'}`}</p>${r.nli?.reason?`<p>${esc(r.nli.reason)}</p><details><summary>NLI 逐来源判定</summary><pre>${esc(JSON.stringify(r.nli.ref_labels,null,2))}</pre></details>`:''}${r.model_duplicate?'<p>模型认为重复于 '+esc(r.model_duplicate.of)+'</p>':''}${r.external_cases?.length?'<p>定向复核：'+r.external_cases.map(c=>esc(c.summary)).join('<br>')+'</p>':''}</div>`;
 }else if(tab==='facet'){out+=field(r.id,'coverage','Scout 证据是否覆盖这项原始需求？',covered);out+=`<p>对应 Scout 事实：</p>${r.supporting_fact_ids.map(id=>`<pre>${esc(id)} · ${esc(D.rows.find(x=>x.id===id)?.claim)}</pre>`).join('')}`;if(showModel)out+=`<div class="model">初评 ${covered[r.coverage]} · ${esc(r.reason)}；通过支持核查后充分覆盖=${r.effective_full}</div>`;
 }else if(tab==='audit'){out+=`<p><b>原文位置</b>：${esc(r.location)}</p><p>${esc(r.description)}</p><p><b>Critic 依据</b>：${esc(r.evidence)}</p><div class="fields">${field(r.id,'validity','这个问题是否有效？',valid)}${field(r.id,'category','问题类型',{factual:'事实错误',missing_citation:'引用遗漏',scope:'范围扩大',other:'其他'})}</div><details><summary>完整报告供对照</summary><article class="report">${D.tasks.find(t=>t.id===topic).report_html}</article></details><details><summary>初版章节供对照</summary><pre>${esc(JSON.stringify(D.tasks.find(t=>t.id===topic).pre_review_draft,null,2))}</pre></details>`;if(showModel)out+=`<div class="model">${esc(JSON.stringify(r.model_review))}</div>`;
 }else{out+=`<a href="${esc(safeUrl(r.url))}" target="_blank" rel="noreferrer">${esc(r.url)}</a><p class="muted">5 官方一手/原始论文；4 预印本/机构摘要/限定厂商资料；3 专业二手报道；2 商业转述或未核原作者的上传资料；1 营销/UGC/无法辨源或证据不可读。</p><div class="fields">${field(r.id,'score','来源质量 / 5',{'1':'1','2':'2','3':'3','4':'4','5':'5'})}${field(r.id,'primary','是否为本论断的一手来源？',yesno)}</div><details><summary>阅读完整保存正文</summary><pre>${esc(r.content)}</pre></details>`;if(showModel)out+=`<div class="model">初评 ${r.score}/5 · ${esc(r.tier)}<p>${esc(r.reason)}</p><p>${esc(r.risk)}</p></div>`}
 out+=`<label>你的评审备注<textarea rows="3" data-id="${esc(r.id)}" data-key="note">${esc(a.note||'')}</textarea></label>`;$('detail').innerHTML=out;bindFields();}
function fullSource(id){const src=D.sources.find(s=>s.topic===topic&&s.original_id===id);return src?`<details><summary>展开完整保存正文，核查检索窗口是否遗漏</summary><pre>${esc(src.content)}</pre></details>`:''}
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('export').onclick=()=>download('五篇报告人工评分-'+new Date().toISOString().slice(0,10)+'.json',JSON.stringify({schema:'research-review-v1',dataset_id:D.dataset_id,exported_at:new Date().toISOString(),report_hashes:D.tasks.map(t=>({id:t.id,sha256:t.report_sha256})),annotations:A},null,2),'application/json');
$('csv').onclick=()=>{const fields=['id','support','locator','error','duplicate','duplicate_of','coverage','validity','category','detected','matched_issue','score','primary','note'];const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';const lines=[fields.map(q).join(','),...Object.entries(A.items).map(([id,a])=>fields.map(f=>q(f==='id'?id:a[f])).join(','))];download('五篇报告逐条人工评分.csv','\ufeff'+lines.join('\r\n'),'text/csv;charset=utf-8')};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{try{const x=JSON.parse(await $('file').files[0].text());if(x.schema!=='research-review-v1'||x.dataset_id!==D.dataset_id)throw Error('评分文件与当前五篇报告的指纹不一致');if(!x.annotations?.items||!x.annotations?.quality)throw Error('缺少评分字段');const allowed=new Set([...D.rows,...D.coverage,...D.sources,...D.audits].map(r=>r.id));D.rows.forEach(r=>r.refs.forEach((ref,i)=>allowed.add(r.id+'-ref-'+i)));for(const [id,a] of Object.entries(x.annotations.items)){if(!allowed.has(id)||!a||typeof a!=='object')throw Error('条目编号不属于当前评估集');}for(const [id,q] of Object.entries(x.annotations.quality)){if(!D.tasks.some(t=>String(t.id)===id))throw Error('题目编号错误');for(const d of Object.keys(D.dimensions)){if(q[d]!=null&&(typeof q[d]!=='number'||!Number.isFinite(q[d])||q[d]<0||q[d]>10))throw Error('质量分数须为0–10');}}A=x.annotations;save();render();toast('评分已导入')}catch(e){toast('导入失败：'+e.message)}$('file').value=''};$('blind').onchange=()=>{showModel=$('blind').checked;render()};render();
</script></body></html>'''

def write_page(data):
    encoded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page=HTML.replace('__DATA__',encoded)
    page=page.replace('截止资料日期 2026-10-03 · 本地离线可用',f"截止资料日期 2026-10-03 · 评估版本 {DATE} · NLI {html.escape(data['summary']['nli_model'])} · 本地离线可用")
    PAGE.write_text(page)

def write_report(data):
    ms=data['summary']['metrics'];qs=data['summary']['quality']
    usage=read(OUT/'裁判调用统计.json') if (OUT/'裁判调用统计.json').exists() else {}
    def table(headers,rr):return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in rr)+'\n'
    qwen_nli=data['summary']['nli_model']=='qwen3-max'
    nli_method=('证据先核对来源哈希、文档/句号位置，再用Qwen3-max做独立NLI语义分类。逐来源给full/partial/unsupported/contradicted/unknown，再给联合标签；full对应蕴含，contradicted对应矛盾，partial/unsupported对应中立，unknown对应无法判定。温度0，不使用0.85/0.5阈值或虚构分类概率。输入保留全部既有证据窗口，重叠片段合并而不截断；不输入以前的评分。741条有引用论断实际调用模型，218条无引用论断记unknown并跳过调用。正文和Scout的最终支持指标继续采用此前逐条语义裁判、盲审和定向复核结果；NLI是辅助筛查，不能直接覆盖已复核指标。' if qwen_nli else '证据先核对来源哈希、文档/句号位置，再用本地mDeBERTa NLI筛查，并由Qwen逐条语义裁判。NLI revision为8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c，支持阈值0.85、矛盾阈值0.5；不截断过长输入。')
    opening=('本次于2026-10-05按用户要求将NLI改为Qwen3-max并重跑。沿用2026-10-04的正文质量、支持复核及外部核查；资料判断截止仍为2026-10-03，原报告与旧评估保留。' if qwen_nli else '本次复评于2026-10-03夜间开始、10-04完成，资料判断截止2026-10-03。')
    lines=[f'# 五篇报告正文与 Scout 独立评估分析（{DATE}）','',
    opening+'五篇报告表达普遍清楚，但证据链和任务关键问题的完成程度不足。尤其需要优先修复第31题的统计分母、第49题的共同需求推断、第35题的现行标准与PPP规则。下面是模型初评及助手定向复核，**不是人工金标，不是官方 DeepResearch Bench 排名**。人工评分和一致性仍待你在网页中补充。','',
    f'- [人工评分网页]({PAGE})：单文件，可离线打开，模型分数默认隐藏。',
    f'- [机器可读指标]({OUT}/指标汇总.json)；[逐条论断和证据]({OUT}/逐条初评与复核.json)；[外部核查记录]({OUT}/外部核查与评审复核.json)。',
    f'- 标准来自[医疗多agent评估.md](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/医疗多agent评估.md)；[冻结输入与原文快照]({OUT}/冻结输入.json)保存最终检查点中的来源正文、Scout事实、Writer论断及Critic问题。','',
    '## 评估对象与方法','',
    '仅评估用户指定的五篇最终报告：35、31、46、16、49。31题沿用此前成功版本，其余采用取消研究累计token上限后的版本。本次没有重新生成研究报告，也没有更改既有轨迹。原运行总计184次LLM调用、4,447,639 token；这不是评价裁判调用的成本。并发运行与31题旧预算使这些样本不能用于公平预算消融比较。','',
    table(['题目','最终正文','Scout事实','原始来源记录','预审结构化样本'],[(f"{m['id']} {m['title']}",f"[{m['body_claims']}条事实论断]({t['folder']}/研究报告.md)",m['scout_facts'],m['source_records'],m['audit_pre_sample']) for m,t in zip(ms,data['tasks'])]),
    '正文抽取覆盖187个内容段落，含执行摘要、表格、正文、结论；参考文献目录不进入论断分母。共573条抽取条目，511条为事实论断，62条为建议、假设或结构项。重复出现的事实按出现次数计算，避免摘要和结论的漏引被隐藏。Writer自动输出的claims不是全正文分母。段落覆盖检查无遗漏，但不保证每个段内原子论断都已抽全；46-body-010的原句锚点跨过了Markdown引用，保留原段落便于人工核对。31-body-137的抽取增加了原文未明确写出的“文物机构”限定，已单列语境风险。','',
    '质量由Qwen3-max和Claude-opus-5（接口返回的模型名，不能保证服务别名底层版本可固定）各重复2次，四维0–10分分别评分；没有参考报告比较或顺序交换，因此不报相对参考分数。Qwen温度0.3；Claude服务拒绝0.3，取消该字段使用服务默认值。Claude并发限制导致失败，随后串行重试；原始提示词、返回文本、usage和失败原因全部保留。','',
    f"本次评价裁判取得完整响应的成功调用共{usage.get('successful_calls')}次，已知token合计{usage.get('known_tokens'):,}，不设置总token预算。失败或中断请求若在服务端计费，未进入这项已知合计。记录见[裁判调用统计]({OUT}/裁判调用统计.json)。对17条日期风险候选又补充明确截止日期的校准，保留每层判定。",'',
    nli_method+'对非完整支持候选此前已做一次盲审，最后对日期逻辑、乱码及高影响案例由助手复核。来源支持主要使用运行时保存快照；正文引用采用每个URL全文的词面top6邻接句段，辅以同URL的写作句号定位。窗口可能遗漏证据，unknown必须保留，不能冒充穷尽全文核查。','',
    '## 指标口径','',
    table(['指标','本次分子 / 分母'],[
      ('引用支持率','完整支持的事实论断–引用URL关系 / 正文全部事实论断–引用URL关系；同一论断多源分别评。'),
      ('有效证据覆盖率','正文有显式引用且联合证据完整支持的事实论断出现次数 / 全部正文事实论断。'),
      ('严格无支持论断率','1−有效证据覆盖率；含未引用、部分支持、相反及无法判定，不能称为事实幻觉率。'),
      ('明确错误／矛盾率','定向核查已确认的错误出现次数 / 全部正文事实论断，报告为已发现错误下限；其余未穷尽现实核查。'),
      ('审核错误检出率','Critic首轮指出同一已确认预审错误 / 已确认预审错误样本；另报全轨迹累计检出，不能外推全报告召回。'),
      ('Scout事实支持率','保存证据完整支持的Scout事实 / 最终Scout事实；复合fact需全部重要内容支持。'),
      ('Scout证据覆盖率','需求要点被充分覆盖且至少有通过支持核查的相关fact / 原始任务需求要点；背景或代理为部分覆盖。'),
      ('Scout证据正确率','哈希位置身份可核且完整支持的fact–来源关系 / 全部fact–来源关系；这里衡量取证忠实度，现实正确率未穷尽核查。'),
      ('重复信息比例','同主体/时间/条件的语义重复组中，除首条外的事实数 / 最终Scout事实数；不同数字/条件不算重复。'),
      ('来源质量','逐来源1–5分均分；另报依赖一手来源的Scout事实比例。原始来源记录均入分母，未按正文引用数量加权。')]),
    '## 正文质量','',
    table(['题目','深度','全面性','可读性','要求覆盖','四维等权均分'],[(q['id'],*[f"{q['dimensions'][d]['mean']:.2f}" for d in DIMS],f"{q['overall']:.2f}") for q in qs]),
    '上述均分只表示裁判判断。两模型存在明显系统差异，需对照各自均分和理由：','',
    table(['题目','Qwen 深/全/读/要求','Claude 深/全/读/要求','最大模型均分差'],[(q['id'],' / '.join(str(q['models']['qwen']['dimensions'][d]) for d in DIMS),' / '.join(str(q['models']['claude']['dimensions'][d]) for d in DIMS),f"{max(abs(q['models']['qwen']['dimensions'][d]-q['models']['claude']['dimensions'][d]) for d in DIMS):.2f}") for q in qs]),
    '网页保留每次分数、理由、标准差及范围。Qwen与Writer/Critic共享模型家族，可能更偏好类似表达；Claude较重视可比性、证据与推导深度。简单平均不能替代人工校准。2次重复不足以推断稳定性，尚未完成N的收益消融或人工一致性测量。','',
    '## 正文证据与审核','',
    '汇总按论断/引用关系数量加权：引用支持198/296=66.9%；正文有效证据覆盖197/511=38.6%；严格未充分支持314/511=61.4%；已定向核实的错误8/511=1.6%（下限）。Scout完整支持198/250=79.2%。这些分母不相同，不能互相替代。','',
    table(['题目','引用支持','有效证据覆盖','严格无支持','已确认错误下限'],[(m['id'],fmt(m['citation_support']),fmt(m['effective_evidence_coverage']),fmt(m['unsupported_strict']),fmt(m['confirmed_error_lower_bound'])) for m in ms]),
    '严格无支持包括unknown和部分支持，故保守。另将完全缺乏支持/证据相反与取证不确定性分开：','',
    table(['题目','完整','部分','无支持','相反','无法判断'],[(m['id'],*[m['body_support_counts'].get(k,0) for k in ['full','partial','unsupported','contradicted','unknown']]) for m in ms]),
    '明确错误下限不是总体真实错误率的无偏估计；0表示本次没有确认，不表示整篇无错误。已经确认的错误见下文逐题案例及外部核查记录。未支持结论、危险建议与确定事实错误分别保留。','',
    table(['题目','首轮检出（已确认预审样本）','全轨迹累计检出同一错误','Critic问题有效率（模型复核）','问题待核实'],[(m['id'],fmt(m['audit_first_confirmed_recall']),fmt(m['audit_any_confirmed_recall']),fmt(m['audit_model_precision']),m['audit_uncertain']) for m in ms]),
    '预审材料198条不是穷尽错误金标。已确认预审样本只有4条：35-pre-012、31-pre-008、31-pre-049、49-pre-029；前三条未被Critic针对同一语义错误指出，49的年龄/分母问题在后续审核被指出，首轮没有指出。同一错误在不同句出现仍计不同论断，错误家族仅3类。审核器能发现很多漏引，但对统计总体、现行标准以及自身记忆中的政策时间较弱。不能拿最终残留错误作原始错误分母，更不能使用resolved字段证明修正成功。','',
    '审核初评模型也有误判：曾把2026年发布的2025年统计当未来预测，将8.22万亿元的已注明纠错认成真实错误，以及在unknown窗口上制造“金标”。这些原始输出保留供校准，未纳入已确认错误检出分母。Critic问题有效率仍是模型估计，涉及不同轮次稿件；网页同时展示初版和最终正文，对中间轮次可回到原检查点核对，因此尚不作为审核准确率金标。','',
    '## Scout 组件质量','',
    table(['题目','事实完整支持','有效要点覆盖','证据正确率（取证忠实度）','重复比例','来源均分/5','依赖一手来源事实'],[(m['id'],fmt(m['scout_support']),fmt(m['scout_coverage']),fmt(m['scout_evidence_correctness']),fmt(m['scout_duplicate']),m['source_score_mean'],fmt(m['primary_source_fact_share'])) for m in ms]),
    'fact的证据身份与句号位置核对率：'+ '；'.join(f"{m['id']}题 {fmt(m['scout_locator'])}" for m in ms)+'。即使定位通过，也不保证句段支持、原文正确或时效有效。来源质量1–5层级是定性裁判评分，输入为标题、URL、发布日期及开头1200字符，可能被导航或摘要不足影响，并非全文权威性审计；官方域名中被乱码污染的内容按1分处理；论文摘要、厂商参数、新闻与营销不能混作同等强证据。','',
    '证据支持与现实正确性可以相反：第46题Scout将二手文中的“8.22亿元”透明注明为笔误并改成“8.22万亿元”。严格按原快照核验会认为引文单位不一致，但[金融监管部门原始绿色信贷说明](https://www.nfra.gov.cn/chinese/files/2018/8F21409B87D84147BE72C19C24C59DE9.pdf)及[中国政府网转载统计](https://app.www.gov.cn/govdata/gov/201802/21/420724/article.html)支持万亿元口径。因此此条保留为引文不一致，不能当现实错误；更好的做法是替换为一手来源并保留纠错说明。','',
    '## 逐题结论与失败案例','',
    '### 35 市政污水与雨污协同','',
    '优点是能够围绕厂网河、财政拨付、智慧调度和应急形成统一框架，政策工具与案例较易阅读。短板是几乎没有元/吨、元/km·年、调蓄容积或全生命周期折现成本比较；维护、改造和应急更多是概念层建议。摘要“不显著增加财政投入”和“系统性降低成本”没有预算或对照结果支撑。把监管职能整合进同时持有资产和运营的水务集团，也未厘清采购与独立监管责任。','',
    '高影响错误是声称GB/T 23023仍未实施，[官方标准平台](https://std.samr.gov.cn//gb/search/gbDetailed?id=EB58F4DA90DEB2A2E05397BE0A0A7D33)显示2022年已实施。建议对未来新项目使用政府可行性缺口补助，又没有处理[2023年PPP新机制](https://app.www.gov.cn/govdata/gov/202311/08/509155/article.html)限制及存量过渡边界。温州.doc快照是二进制乱码却passed，不能证明“已率先”实施预算归口。应先修复读取、更新政策，再做一页具体KPI、扣款公式、实施责任与成本方案。','',
    '### 31 中外博物馆教育','',
    '优点是国内/国外、分类、技术和中国未来路径结构齐全，有馆校合作政策依据。缺口是国外主要依赖美国、英国、日本各一个材料，若干材料来自2013–2014年；“国家级行政层级”和“国家一级评估等级”靠近混用；基层及非国有馆教育特征多是未经实证比较的概括。中国未来建议尚缺资源配置、学习成效与人员培训实施指标。','',
    '最明确的是把88.6%的免费开放比例当成国有比例，再得出非国有11.4%。[国家文物局统计发布的新华社记录](https://www.xinhuanet.com/politics/2019-05/18/c_1124512026.htm)支持的是免费开放分母。7188数量有官方依据，但16.4亿观众及18.4万从业人员属于[全部文物机构口径](https://zwgk.mct.gov.cn/zfxxgkml/tjxx/202606/t20260602_966073.html)，不宜直接作为博物馆专属数据。报告质量比文风所呈现的严谨度低，应重写分类与统计表。','',
    '### 46 房地产可持续发展','',
    '政策与绿色金融材料丰富，基本覆盖政策、资金和国家导向，但核心动力过度聚焦绿色转型，人口、收入、存量去化、房企债务重组、土地财政与地区分化不足。未来十年没有阶段路径或场景，正文仍使用2024–2034窗口，和2026评估时点不一致；2017年农行不良率不能外推2025年全行业绿色资产稳健。绿色贷款总额也不能当房地产专项融资规模。','',
    'Critic声称十五五规划建议尚未发布属于自身误报，[国务院公报](https://www.gov.cn/gongbao/2025/issue_12386/material/gwygb202531.pdf)及[北京市政府转载](https://www.bjstb.gov.cn/bjtb/ywdt/gwyxx/1356572/index.html)证明2025年10月已通过。Scout还把“纲要”误写成“建议”，提示文件名称、正文、发布日期需要联合核验。推荐改为现行政策与未来情景分栏，补充住房需求/库存/债务模型，按不同主体分析资金可得性。','',
    '### 16 非接触式感知算法','',
    '能整理毫米波、近红外、工业近场感知等输入与性能，明确若干闭集局限，技术表易读。但是“做得最好”的算法没有同任务基准排名或选择标准；99%分类准确率、10nm分辨率、相对精度提升和工业传感器参数并不是可直接比较的一张排行榜。缺少跨被试/跨环境划分、样本量、数据集及可复现实验协议；电容+IMU融合和植入式动物遥测也需要明确是否符合用户“非接触”的目标。','',
    'Wi-Fi CSI与视频rPPG等路线在正文不足，结论却新增相关论断。所谓连续生理信号量化数据严重缺失应限定为本次检索素材；[MultiPhys原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC11722562/)已有MAE/RMSE指标。建议按人体活动/心率/工业位移分成三个任务表，给原始论文、数据集、评价协议、输入、误差与适用条件。','',
    '### 49 正畸与医美共同需求','',
    '优点是承认缺乏个体层面交叉统计，整理了美观动机、社交媒体及未来整合路径。核心需求“全球20–30岁女性共同需求比重”并未回答；两个市场女性多，不等于同一女性同时有两项需求。19–33、26–55和20–35岁的数据不能统一写成20–30；表中73%又改变分母。现有原表文本总和不自洽，必须核查原图，不应继续精确推算。','',
    '报告一面承认没有整合满意度/依从性实证，一面在结论肯定“整合可提升满意度和治疗依从性”，存在无支持的效益结论。“口腔美学”案例也不等于跨口腔与皮肤/注射医美的联合诊疗。监管障碍需引用具体法域：[中国现有医疗美容管理规则](https://www.nhc.gov.cn/wjw/c100221/202201/d7e8fa33a26b425da98d69fb04191699.shtml)已规定登记科目与医师条件。建议明确“交集比例无法由当前数据估计”，定义同一受访者共同需求的分子、目标总体分母，设计分国分年龄问卷或用户行为研究，融合可能性作为待验证假设。','',
    '## 评估器局限与下一步','',
    f"初评与再次盲审标签变化 {fmt(data['summary']['initial_second_disagreement'])}（仅首判非完整支持候选，带选择偏差）；NLI与最终语义初评严格同标签比例 {fmt(data['summary']['nli_vs_final_agreement'])}。后者是机器判定的一致率，不能称NLI准确率。"+('本版NLI与最终支持裁判同用Qwen3-max，分歧只能衡量不同提示词及复核口径的差异，不是不同模型的独立验证。' if qwen_nli else '')+'NLI仅用于筛查，不能直接宣布虚假事实。','',
    '已测量：四维两模型重复评分、正文和Scout保存来源的支持判定、需求要点覆盖、重复组、来源分层、预审样本与Critic轨迹。间接推断：可操作性、现实共同需求与外推风险。尚未测量：人工一致率、人类裁决下的幻觉率和审核准确率、所有来源现实真实性、全报告审核召回、同预算单Agent收益及组件消融。本次没有编造这些指标。','',
    '优先级：先修复读取乱码和现行政策/统计分母，再防止摘要与结论新增无引用推断，最后提高任务关键证据的覆盖。建议先在网页独立评5篇四维质量，并优先标核心论断和分歧项；导出JSON后可计算人工—模型分数差、支持标签一致率及审核检出指标。评分页允许展开完整来源、导入/导出、逐条备注，待标项不会被当作错误。','',
    f'原始裁判记录目录：`{OUT}/裁判原始记录`；NLI完整窗口文件：[NLI窗口筛查.json]({OUT}/NLI窗口筛查.json)。评价脚本：[评估五篇正文与Scout.py]({ROOT}/backend/eval/scripts/评估五篇正文与Scout.py)，生成脚本：[生成五篇评估分析与评分页.py]({ROOT}/backend/eval/scripts/生成五篇评估分析与评分页.py)。','']
    if qwen_nli:
        comparison=data['summary']['nli_previous_comparison']
        selected={r['id']:r for r in data['rows']}
        lines+=['## NLI替换与复跑结果','',f"新旧NLI同标签比例：{fmt(comparison['agreement'])}，仅比较741条有显式来源的论断；标签变化不是错误纠正率。旧版按窗口最大概率并采用矛盾优先规则，新版由Qwen联合读取每个来源的全部保存窗口，并增加partial类别；模型和聚合口径同时变化，不能作纯模型胜负结论。",'',
                table(['题目','旧NLI标签计数','Qwen NLI标签计数','标签变化条数'],[(k,json.dumps(v['old_counts'],ensure_ascii=False),json.dumps(v['new_counts'],ensure_ascii=False),v['changed']) for k,v in comparison['by_topic'].items()]),
                '重点样例保留NLI与最终复核的不同口径：GB/T旧报批稿不能证明当前仍未实施；乱码只能无法判断；忠实于原文错误数字与现实数字正确是两回事。下面的差异不自动更新最终支持标签或明确错误计数。','',
                table(['样例','Qwen NLI','此前最终支持','NLI理由'],[(cid,selected[cid]['nli']['label'],selected[cid]['judgment']['joint_support'],selected[cid]['nli']['reason'].replace('|','/')) for cid in ['35-body-053','35-body-083','31-body-034','46-scout-033','16-body-053','49-body-009']]),
                f"本轮新增Qwen NLI成功调用{usage.get('new_nli_calls')}次、已知token {usage.get('new_nli_tokens',0):,}；继承旧评估{usage.get('inherited_calls')}次调用。没有累计token预算。所有请求提示词、完整返回、usage、失败尝试及原文位置均落盘。",'',
                f"[NLI运行状态]({OUT}/NLI运行状态.json)；[新旧差异与一致性核对]({OUT}/NLI替换核对.json)；[旧版评估]({ROOT}/docs/五篇报告正文与Scout独立评估分析-20261004.md)。评分网页沿用同一报告指纹与论断编号，可导入此前导出的人工评分JSON。",'']
    DOC.write_text('\n'.join(lines))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--date',default=DATE);ap.add_argument('--doc',type=Path);ap.add_argument('--page',type=Path);args=ap.parse_args()
    if args.out:OUT=args.out.resolve()
    DATE=args.date
    if args.doc:DOC=args.doc.resolve()
    if args.page:PAGE=args.page.resolve()
    data=prepare();write_page(data);write_report(data);print(DOC);print(PAGE)
