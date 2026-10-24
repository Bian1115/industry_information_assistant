#!/usr/bin/env python3
"""离线逐题HTML、汇总与一致性分析；缺失评分不插补。可监测本次运行并刷新页面。"""
import hashlib, html, json, math, re, sys, time, os, importlib.util
from datetime import datetime
from pathlib import Path
from html.parser import HTMLParser
import markdown
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007'
PAGES=Path('/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007')
DOC=ROOT/'docs/DeepResearchBench参考报告30题评分一致性与分歧分析-20261007.md'
DIMS=['comprehensiveness','insight','instruction_following','readability']
NAMES=dict(zip(DIMS,['全面性','洞察力','指令遵循','可读性']))
LABELS={'assistant':'助手第一轮','qwen':'Qwen3.8-max','claude':'Claude-opus-5','ensemble':'双裁判均分'}
E=lambda x:html.escape(str(x),quote=True)
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def dump(p,v):
    p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));tmp.replace(p)
class SafeHTML(HTMLParser):
    tags=set('h1 h2 h3 h4 h5 h6 p ul ol li strong em a blockquote pre code table thead tbody tr th td br hr del sup sub'.split())
    def __init__(self):super().__init__();self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag not in self.tags:return
        att=''
        if tag=='a':
            href=dict(attrs).get('href','')
            if href.startswith(('http://','https://','#')):att=f' href="{E(href)}" target="_blank" rel="noopener noreferrer"'
        self.parts.append('<'+tag+att+'>')
    def handle_endtag(self,tag):
        if tag in self.tags:self.parts.append('</'+tag+'>')
    def handle_data(self,s):self.parts.append(html.escape(s,quote=False))
def md(s):
    p=SafeHTML();p.feed(markdown.markdown(s,extensions=['tables','fenced_code']));return ''.join(p.parts)
def weighted(t,scores):
    scores={r['criterion_id']:r['score'] for r in scores};ds={}
    for d in DIMS:
        cs=[c for c in t['criteria'] if c['dimension']==d]
        ds[d]=sum(scores[c['criterion_id']]*c['weight'] for c in cs)/sum(c['weight'] for c in cs)
    ds['total']=sum(ds[d]*t['dimension_weight'][d] for d in DIMS)
    return ds
def metrics(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);diff=x-y
    def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None
    def rank(a):
        return np.array([np.mean(np.flatnonzero(np.sort(a)==v))+1 for v in a])
    den=np.var(x)+np.var(y)+(np.mean(x)-np.mean(y))**2
    return {'n':len(x),'mae':float(np.mean(abs(diff))),'rmse':float(np.sqrt(np.mean(diff**2))),
            'bias':float(np.mean(diff)),'within_0_5':float(np.mean(abs(diff)<=.5+1e-12)),
            'within_1':float(np.mean(abs(diff)<=1+1e-12)),'pearson':corr(x,y),'spearman':corr(rank(x),rank(y)),
            'exact_match':float(np.mean(abs(diff)<=1e-12)),'gap_at_least_2':float(np.mean(abs(diff)>=2-1e-12)),
            'ccc':float(2*np.mean((x-x.mean())*(y-y.mean()))/den) if den>0 and len(x)>1 else None,
            'judge_mean':float(x.mean()),'assistant_mean':float(y.mean())}
def bootstrap(x,y):
    # 以题目而非细则/重复为独立抽样单位，仅用于总分的MAE和偏差。
    rng=np.random.default_rng(20261007);a=np.asarray(x)-np.asarray(y)
    if len(a)<2:return None
    s=a[rng.integers(0,len(a),(4000,len(a)))];return {k:[float(z) for z in np.quantile(v,[.025,.975])] for k,v in [('mae',np.mean(abs(s),axis=1)),('bias',np.mean(s,axis=1))]}
CSS='''*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#1e293b;font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}a{color:#2764ad;text-decoration:none}a:hover{text-decoration:underline}header{background:#12283f;color:#fff;padding:28px max(24px,calc((100vw - 1180px)/2));}header a{color:#b9d9ff}.eyebrow{font-size:12px;letter-spacing:2px;color:#a8c4df}h1{font-size:28px;line-height:1.4;margin:10px 0}h2{font-size:21px;margin:26px 0 15px}h3{font-size:17px;margin:8px 0}main{max-width:1180px;margin:auto;padding:24px}.notice{border-left:4px solid #e1ab44;background:#fff6e2;padding:15px 20px;border-radius:6px;margin:16px 0}.muted{color:#64748b;font-size:13px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,section,details.criterion{background:#fff;border:1px solid #dce2ea;border-radius:10px;padding:20px;margin-bottom:14px}.number{font-size:31px;font-weight:650;color:#153b61}.badge{display:inline-block;background:#e8f1fa;color:#255d92;padding:2px 8px;border-radius:5px;font-size:12px}.pending{background:#f1f2f4;color:#75808c}.warn{background:#fff0da;color:#975b10}.good{background:#e7f5ee;color:#227147}.actions{display:flex;gap:12px;flex-wrap:wrap;position:sticky;top:0;background:#f4f5f7ee;padding:12px 0;z-index:2}button,input,select{border:1px solid #bccbdc;border-radius:6px;padding:8px 12px;background:#fff;font:inherit;cursor:pointer}input{cursor:text;min-width:240px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{background:#edf3f8;text-align:left;white-space:nowrap}td,th{padding:9px 12px;border-bottom:1px solid #e2e8f0;vertical-align:top}.scores td{white-space:nowrap}.positive{color:#bb6411}.negative{color:#2b73a0}summary{cursor:pointer;list-style:none}summary::-webkit-details-marker{display:none}.criterion summary{display:grid;grid-template-columns:1fr auto;gap:15px;align-items:start}.criterion summary h3:before{content:"＋ ";color:#3675af}details[open] summary h3:before{content:"－ "}.mini{display:flex;gap:8px;flex-wrap:wrap}.mini span{min-width:105px}.review-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:14px}.review{padding:14px;border:1px solid #e1e6ef;border-radius:8px;background:#fafbfc}blockquote{margin:12px 0;padding:10px 15px;background:#edf3f8;border-left:3px solid #91adc7;font-size:13px;overflow-wrap:anywhere}.report{font-size:15px;overflow-wrap:anywhere}.report h1{font-size:25px}.report table{display:block;overflow:auto;max-width:100%}.report td{min-width:100px}.report pre{overflow:auto;background:#f4f5f7;padding:12px}.report img{max-width:100%}.split{display:grid;grid-template-columns:1fr 1fr;gap:16px}.hidden{display:none!important}footer{max-width:1180px;margin:auto;padding:15px 24px 35px;color:#738195;font-size:12px}.plot{max-width:450px;width:100%;height:auto}.report p{margin:12px 0}code{background:#eef1f5;padding:2px 5px}nav a{margin-right:12px}@media(max-width:800px){.grid,.review-grid,.split{grid-template-columns:1fr 1fr}.criterion summary{grid-template-columns:1fr}main{padding:14px}h1{font-size:24px}}@media(max-width:500px){.grid,.review-grid,.split{grid-template-columns:1fr}}@media print{.actions,button,input,select{display:none}header{background:white;color:black}body{background:white}details{break-inside:avoid}main{max-width:none}a{color:black}}'''
METHOD='助手第一轮由当前对话模型独立给出逐条分数，先于裁判结果冻结。阅读采用结构逐段浏览与按标准定位补读，较长段落有截取，未做全文逐字或外部事实核验；它是模型评分参照，并非人类金标。双裁判接收完整官方标准和完整正文，不能看到助手评分。'
def shell(title,body):return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{E(title)}</title><style>{CSS}</style></head><body>{body}<footer>DeepResearch Bench · 固定随机种子 20261007 · 本页可离线使用 · 分数范围 0–10 · 刷新页面查看新结果</footer></body></html>'
def fmt(v):return '待评分' if v is None else f'{v:.2f}'
def statfmt(v):return '不适用' if v is None else f'{v:.3f}'
def snippet(t,q):
    if not q:return '<span class="muted">该要素缺失或未提供引文。</span>'
    pos=t['article'].find(q)
    if pos<0:return '<blockquote>'+E(q)+'</blockquote><span class="badge warn">引文未在原文连续匹配，需人工复核</span>'
    context=t['article'][max(0,pos-100):pos+len(q)+130]
    return '<blockquote>'+E(q)+'</blockquote><details><summary class="muted">查看引文附近原文</summary><pre style="white-space:pre-wrap">'+E(context)+'</pre></details>'
def metric_table(rows):
    b='<div class="scroll"><table><tr><th>比较（已完成题目）</th><th>N题</th><th>MAE</th><th>偏差</th><th>±1比例</th><th>Pearson</th><th>Spearman</th><th>CCC</th></tr>'
    for k,v in rows.items():
        m=v['total'];b+='<tr><td>'+E(LABELS.get(k,k))+' vs 助手</td><td>'+str(m['n'])+'</td>'+''.join('<td>'+ (f"{m[x]:.1%}" if x=='within_1' else statfmt(m[x]))+'</td>' for x in ['mae','bias','within_1','pearson','spearman','ccc'])+'</tr>'
    return b+'</table></div>'
def qwen_plot(rows):
    rows=[r for r in rows if r['counts']['qwen']==2]
    if not rows:return ''
    # 相同0–10坐标轴；点链接逐题页面，可通过悬停读取未四舍五入的差值。
    parts=['<svg viewBox="0 0 500 460" role="img" aria-label="助手和Qwen总分一致性散点图" style="max-width:560px;width:100%">',
           '<rect x="55" y="25" width="400" height="380" fill="#f8fafc" stroke="#cbd5e1"/>']
    for k in range(0,11,2):
        x=55+40*k;y=405-38*k
        parts.append(f'<path d="M {x} 25 V 405 M 55 {y} H 455" stroke="#e2e8f0"/><text x="{x}" y="426" text-anchor="middle" font-size="12">{k}</text><text x="43" y="{y+4}" text-anchor="end" font-size="12">{k}</text>')
    parts.append('<path d="M55 405 L455 25" stroke="#8095ad" stroke-dasharray="5 4"/><text x="255" y="451" text-anchor="middle" font-size="13">助手第一轮总分</text><text x="17" y="225" text-anchor="middle" transform="rotate(-90 17 225)" font-size="13">Qwen两次均分</text>')
    for r in rows:
        a=r['scores']['assistant']['total'];q=r['scores']['qwen']['total'];i=r['task_id']
        color='#2678ac' if r['language']=='zh' else '#b96e35'
        parts.append(f'<a href="题{i}.html"><circle cx="{55+40*a:.2f}" cy="{405-38*q:.2f}" r="5" fill="{color}" fill-opacity=".8" stroke="white"><title>题{i} · {r["language"]} · 助手 {a:.3f} · Qwen {q:.3f} · 差值 {q-a:+.3f}</title></circle></a>')
    return '<section><h2 style="margin-top:0">Qwen与助手总分对照</h2><p class="muted">虚线表示两者同分；悬停查看题号和差值，点击点进入逐题页面。蓝色为中文，棕色为英文；只显示Qwen完成两次的题目。</p>'+''.join(parts)+'</svg></section>'
def recover_null_metadata(tasks):
    """按原始ID与数字确认完整评分，首个合格输出优先，不按分数选择。"""
    spec=importlib.util.spec_from_file_location('bench_validation',Path(__file__).with_name('DeepResearchBench参考报告双裁判一致性.py'))
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    ts={t['id']:t for t in tasks}
    for f in sorted((OUT/'失败尝试').glob('*claude*.json')):
        d=read(f);key=d['key'];dest=OUT/'裁判原始记录'/f'{key}.json';old=read(dest,{})
        if old.get('result') is not None or d.get('completion_event')!='message_stop':continue
        match=re.fullmatch(r'题(\d+)-细则-claude-([12])',key)
        if not match:continue
        tid,n=map(int,match.groups());t=ts[tid]
        try:
            result,processing=v._fmt.parse_response(d['raw_response'])
            v.validate(t,result)
            prompt=(OUT/'双裁判完整输入'/f'题{tid}-重复{n}.txt').read_text()
            if hashlib.sha256(prompt.encode()).hexdigest()!=d['prompt_sha256']:continue
        except (ValueError,AssertionError,KeyError,TypeError):continue
        recovered={**d,'prompt':prompt,'result':result,'errors':old.get('errors',[]),
                   'criterion_packet_complete':True,'format_processing':processing,'normalization':{'processing':processing,'source_failed_attempt':str(f)}}
        dump(dest,recovered)
        audit=read(OUT/'格式恢复核查记录.json',read(OUT/'空元数据格式恢复记录.json',[]));audit.append({'key':key,'source':str(f),'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'normalization':recovered['normalization']});dump(OUT/'格式恢复核查记录.json',audit)
def generate():
    PAGES.mkdir(parents=True,exist_ok=True)
    tasks=read(OUT/'冻结输入.json');cfg=read(OUT/'评估配置.json');state=read(OUT/'运行状态.json',{})
    qstate=read(OUT/'Qwen续跑状态.json',{})
    review=read(OUT/'Qwen分歧归因复核.json',{})
    recover_null_metadata(tasks)
    spec=importlib.util.spec_from_file_location('bench_validation',Path(__file__).with_name('DeepResearchBench参考报告双裁判一致性.py'))
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    ts={t['id']:t for t in tasks};rec=[];rejected=[]
    for f in sorted((OUT/'裁判原始记录').glob('*.json')):
        try:
            r=read(f);match=re.fullmatch(r'题(\d+)-细则-(qwen|claude)-([12])',r['key'])
            if r.get('result') is None or not match:continue
            tid,model,n=int(match[1]),match[2],int(match[3]);t=ts[tid]
            expected=v.prompt(t,n)
            assert r['prompt']==expected and r['prompt_sha256']==hashlib.sha256(expected.encode()).hexdigest(),'输入不匹配'
            v.validate(t,r['result'])
            if model=='qwen':
                assert r['request_options']==cfg['qwen_request_options'],'Qwen参数不匹配'
                assert r.get('completion_event')=='[DONE]' and r.get('finish_reason')=='stop','Qwen流不完整'
                assert json.loads(r['raw_response'])==r['result'],'Qwen原生分数不匹配'
            else:
                assert r.get('completion_event')=='message_stop','Claude流不完整'
                assert v._fmt.parse_response(r['raw_response'])[0]==r['result'],'Claude原始分数不匹配'
            rec.append({'task_id':tid,'model':model,'repeat':n,'key':r['key'],'result':r['result']})
        except (AssertionError,KeyError,ValueError,TypeError) as ex:
            rejected.append({'file':str(f),'error':str(ex)})
    dump(OUT/'原始响应完整性核查.json',{'accepted':len(rec),'rejected':rejected})
    dump(OUT/'核查后有效双裁判评分.json',rec)
    by={};summaries=[];complete={k:[] for k in ['qwen','claude','ensemble']};allmeans={};audit=[]
    account_blocked=qstate.get('status')=='账户异常阻塞' if qstate else any('Arrearage' in str(x) for x in state.get('failed',[]))
    qcount=sum(r['model']=='qwen' for r in rec)
    block_notice=(f'Qwen本次续跑仍返回账户异常 Arrearage，有效评分{qcount}/60；缺失分数保留为空。' if account_blocked else
                  f'Qwen续跑：{qstate.get("status","未启动")}，有效评分{qcount}/60；按原冻结配置补齐。')
    for r in rec:by[(r['task_id'],r['model'],r['repeat'])]=r
    for t in tasks:
        i=t['id'];p=OUT/'助手独立评分'/f'题{i}.json';a=read(p)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['assistant_file_sha256'][p.name]
        own=weighted(t,a['scores']);ownmap={r['criterion_id']:r['score'] for r in a['scores']};cm={};aggregates={'assistant':own};counts={}
        for model in ['qwen','claude']:
            rr=[by[(i,model,n)]['result']['scores'] for n in [1,2] if (i,model,n) in by];counts[model]=len(rr)
            if not rr:continue
            maps=[{r['criterion_id']:r['score'] for r in rs} for rs in rr]
            cm[model]={cid:sum(s[cid] for s in maps)/len(maps) for cid in ownmap}
            aggregates[model]=weighted(t,[{'criterion_id':k,'score':v} for k,v in cm[model].items()])
            if len(rr)==2:complete[model].append(i)
        if all(counts[m]==2 for m in ['qwen','claude']):
            cm['ensemble']={cid:(cm['qwen'][cid]+cm['claude'][cid])/2 for cid in ownmap}
            aggregates['ensemble']=weighted(t,[{'criterion_id':k,'score':v} for k,v in cm['ensemble'].items()]);complete['ensemble'].append(i)
        allmeans[i]={'task':t,'ownmap':ownmap,'criterion_means':cm,'aggregates':aggregates}
        summaries.append({'task_id':i,'prompt':t['prompt'],'language':t['language'],'counts':counts,'scores':aggregates})
        totalcards=''
        for k in ['assistant','qwen','claude','ensemble']:
            count='已冻结 · '+str(len(t['criteria']))+'条细则' if k=='assistant' else ('两个裁判均完成' if k=='ensemble' and k in aggregates else (f"{counts.get(k,0)}/2 次完成" if k!='ensemble' else '须两裁判各完成两次'))
            totalcards+=f'<div class="card"><div class="muted">{LABELS[k]}</div><div class="number">{fmt(aggregates.get(k,{}).get("total"))}</div><div class="muted">{count}</div></div>'
        dimrows=''
        for d in DIMS:
            dimrows+='<tr><td>'+NAMES[d]+f" · {t['dimension_weight'][d]:.0%}"+'</td>'+''.join('<td>'+fmt(aggregates.get(k,{}).get(d))+'</td>' for k in ['assistant','qwen','claude','ensemble'])+'</tr>'
        notes=''
        for d in DIMS:
            x=a['dimension_analysis'][d];notes+=f'<div class="card"><h3>{NAMES[d]} · {own[d]:.2f}</h3><p>{E(x["reason"])}</p>{snippet(t,x["evidence_quote"])}</div>'
        comparison='ensemble' if 'ensemble' in cm else ('qwen' if counts.get('qwen')==2 else None)
        comparison_label=LABELS[comparison] if comparison else '双裁判均分'
        comparison_section=''
        if counts.get('qwen')==2:
            impacts=[]
            for c in t['criteria']:
                cid=c['criterion_id'];d=c['dimension'];ew=c['weight']/sum(x['weight'] for x in t['criteria'] if x['dimension']==d)*t['dimension_weight'][d]
                gap=cm['qwen'][cid]-ownmap[cid]
                impacts.append((abs(gap*ew),c,ownmap[cid],cm['qwen'][cid],gap,gap*ew))
            impact_rows=''.join(f'<tr><td>{E(c["criterion"])}</td><td>{aa:.2f}</td><td>{qq:.2f}</td><td>{gg:+.2f}</td><td>{ww:+.3f}</td></tr>' for _,c,aa,qq,gg,ww in sorted(impacts,key=lambda x:x[0],reverse=True)[:3])
            comparison_section=f'<section><h2 style="margin-top:0">本题Qwen与助手的分歧</h2><p>总分差值：<b>{aggregates["qwen"]["total"]-own["total"]:+.3f}</b>。下列3条按对总分贡献的绝对值排序；正数表示Qwen较高，负数表示助手较高。逐条理由和原文证据见下方完整评分。</p><div class="scroll"><table><tr><th>官方细则</th><th>助手</th><th>Qwen两次均分</th><th>差值</th><th>对总分贡献</th></tr>{impact_rows}</table></div></section>'
            task_note=review.get('by_task',{}).get(str(i))
            if task_note:comparison_section+='<section><h2 style="margin-top:0">助手的分歧复核说明</h2>'+md(task_note)+'<p class="muted">此说明在看到裁判结果后补充；第一轮分数和理由保持冻结，复核解释不作为独立金标。</p></section>'
        cards=''
        for c in t['criteria']:
            cid=c['criterion_id'];d=c['dimension'];diff=cm.get(comparison,{}).get(cid)
            delta=None if diff is None else diff-ownmap[cid]
            cw=sum(z['weight'] for z in t['criteria'] if z['dimension']==d)
            mini=''.join(f'<span class="badge {"pending" if k not in cm and k!="assistant" else ""}">{LABELS[k]} {fmt(ownmap[cid] if k=="assistant" else cm.get(k,{}).get(cid))}</span>' for k in ['assistant','qwen','claude','ensemble'])
            reviews=''
            for model in ['qwen','claude']:
                for n in [1,2]:
                    r=by.get((i,model,n));b='尚未获得有效评分。'
                    if r:
                        z=next(z for z in r['result']['scores'] if z['criterion_id']==cid)
                        b='<b>'+fmt(z['score'])+'</b><p>'+E(z['reason'])+'</p>'+snippet(t,z['evidence_quote'])
                        audit.append({'task_id':i,'model':model,'repeat':n,'criterion_id':cid,'quote_status':'空引文' if not z['evidence_quote'] else ('连续匹配' if z['evidence_quote'] in t['article'] else '未匹配')})
                    reviews+=f'<div class="review"><h3>{LABELS[model]} · 第{n}次</h3>{b}</div>'
            cards+=f'<details class="criterion" data-gap="{abs(delta) if delta is not None else -1}"><summary><div><span class="muted">{E(cid)} · {NAMES[d]} · 细则权重 {c["weight"]} · 总分有效权重 {c["weight"]/cw*t["dimension_weight"][d]:.1%}</span><h3>{E(c["criterion"])}</h3><div class="mini">{mini}</div></div><span class="badge {"warn" if delta is not None and abs(delta)>1 else "pending"}">{comparison_label}−助手 {"待计算" if delta is None else f"{delta:+.2f}"}</span></summary><p><b>官方解释：</b>{E(c["explanation"])}</p><p class="muted">助手逐条分数已冻结；理由见上方「{NAMES[d]}」维度分析，同一维度理由不会被冒充为每条独立理由。</p><div class="review-grid">{reviews}</div></details>'
        mismatch='' if t['prompt']==t['reference_prompt'] else '<div class="notice"><b>原始数据提示词不一致。</b>官方查询：'+E(t['prompt'])+'<br>参考报告原提示词：'+E(t['reference_prompt'])+'<br>本次保留原始细则，按其评分；请结合两份提示词理解指令遵循分数。</div>'
        judge_overviews='<h2>裁判整体评语与待核查风险</h2>'
        for model in ['qwen','claude']:
            for n in [1,2]:
                packet=by.get((i,model,n))
                if not packet:continue
                result=packet['result'];raw=read(OUT/'裁判原始记录'/f"{packet['key']}.json",{})
                lists=''
                for field,label in [('strengths','优点'),('weaknesses','缺口'),('fact_risks','待核查事实风险（未作外部事实核验）')]:
                    items=result[field]
                    lists+='<h3>'+label+'</h3>'+('<ul>'+''.join('<li>'+E(x)+'</li>' for x in items)+'</ul>' if items else '<p class="muted">未列示。</p>')
                judge_overviews+=f'<details class="card"><summary><b>{LABELS[model]} · 第{n}次整体评语</b></summary><p>{E(result["overall_comment"])}</p>{lists}<p class="muted">响应模型：{E(raw.get("response_model",model))} · 流结束：{E(raw.get("completion_event","未记录"))}</p></details>'
        navigation='<nav><a href="总览.html">← 30题总览</a><a href="#criteria">逐条评分</a><a href="#reference">完整参考报告</a></nav>'
        b=f'<header><div class="eyebrow">DEEPRESEARCH BENCH · 参考报告评分</div><h1>第{i}题</h1>{navigation}</header><main><section><h2 style="margin-top:0">原始任务</h2><p>{E(t["prompt"])}</p><span class="badge">{E(t["language"])} · {len(t["criteria"])}条官方标准</span></section>{mismatch}<div class="grid">{totalcards}</div><div class="notice">{METHOD}<br>{E(block_notice)}单裁判只完成一次时显示暂存均分，不进入完整比较统计。</div><section><h2 style="margin-top:0">官方权重汇总</h2><div class="scroll"><table><tr><th>维度与权重</th><th>助手</th><th>Qwen评分（n/2）</th><th>Claude评分（n/2）</th><th>双裁判</th></tr>{dimrows}</table></div><p class="muted">维度分 = Σ(细则分×细则权重)/Σ细则权重；总分 = Σ(维度分×官方维度权重)。裁判各均两次后，再等权平均。差值 = 双裁判 − 助手。</p></section><h2>助手第一轮分析</h2><div class="split">{notes}</div>{judge_overviews}{comparison_section}<div class="actions"><button onclick="document.querySelectorAll(\'.criterion\').forEach(x=>x.open=true)">展开全部细则</button><button onclick="document.querySelectorAll(\'.criterion\').forEach(x=>x.open=false)">折叠细则</button><button id="gap" onclick="toggleGap()">仅看 |差值| &gt; 1</button><button onclick="window.print()">打印 / 保存PDF</button></div><h2 id="criteria">逐条官方标准与评分</h2>{cards}<section id="reference"><h2 style="margin-top:0">完整参考报告</h2><p class="muted">官方清洗后article全文，未截短；SHA-256: {t["report_sha256"]}</p><div class="report">{md(t["article"])}</div><details><summary>查看完整原始Markdown</summary><pre style="white-space:pre-wrap">{E(t["article"])}</pre></details></section></main><script>let onlyGap=false;function toggleGap(){{onlyGap=!onlyGap;document.querySelectorAll(\'.criterion\').forEach(x=>x.classList.toggle(\'hidden\',onlyGap&&Number(x.dataset.gap)<=1));document.getElementById(\'gap\').textContent=onlyGap?\'显示全部细则\':\'仅看 |差值| > 1\';}}</script>'
        (PAGES/f'题{i}.html').write_text(shell(f'第{i}题 · Bench参考报告评分',b))
    analysis={};gaps=[];model_gaps={}
    for model,ids in complete.items():
        if not ids:continue
        x=[allmeans[i]['aggregates'][model]['total'] for i in ids];y=[allmeans[i]['aggregates']['assistant']['total'] for i in ids]
        dims={d:metrics([allmeans[i]['aggregates'][model][d] for i in ids],[allmeans[i]['aggregates']['assistant'][d] for i in ids]) for d in DIMS}
        xx=[];yy=[]
        for i in ids:
            for cid,v in allmeans[i]['criterion_means'][model].items():xx.append(v);yy.append(allmeans[i]['ownmap'][cid])
        analysis[model]={'task_ids':ids,'total':metrics(x,y),'total_bootstrap_ci95':bootstrap(x,y),'dimensions':dims,'criteria':metrics(xx,yy)}
        model_gaps[model]=[]
        for i in ids:
            data=allmeans[i];t=data['task']
            for c in t['criteria']:
                cid=c['criterion_id'];a=data['ownmap'][cid];z=data['criterion_means'][model][cid];d=c['dimension']
                ew=c['weight']/sum(x['weight'] for x in t['criteria'] if x['dimension']==d)*t['dimension_weight'][d]
                model_gaps[model].append({'task_id':i,'criterion_id':cid,'criterion':c['criterion'],'dimension':d,'assistant':a,'judge':z,'delta':z-a,'effective_weight':ew,'weighted_contribution':(z-a)*ew})
    for i in complete['ensemble']:
        data=allmeans[i];t=data['task']
        for c in t['criteria']:
            cid=c['criterion_id'];a=data['ownmap'][cid];v=data['criterion_means']['ensemble'][cid];d=c['dimension']
            ew=c['weight']/sum(x['weight'] for x in t['criteria'] if x['dimension']==d)*t['dimension_weight'][d]
            gaps.append({'task_id':i,'criterion_id':cid,'criterion':c['criterion'],'dimension':d,'assistant':a,'ensemble':v,'delta':v-a,'weighted_contribution':(v-a)*ew})
    repeat={}
    for model in ['qwen','claude']:
        ids=complete[model]
        if not ids:continue
        xs=[];ys=[]
        for i in ids:
            t=allmeans[i]['task'];xs.append(weighted(t,by[(i,model,1)]['result']['scores'])['total']);ys.append(weighted(t,by[(i,model,2)]['result']['scores'])['total'])
        repeat[model]=metrics(xs,ys)
    inter=None
    if complete['ensemble']:
        ids=complete['ensemble'];inter=metrics([allmeans[i]['aggregates']['qwen']['total'] for i in ids],[allmeans[i]['aggregates']['claude']['total'] for i in ids])
    stamp=datetime.now().astimezone().isoformat();status='完整双裁判完成' if len(complete['ensemble'])==30 else ('Qwen30题各两次评分完成；Claude及双裁判仍按已完成子集统计' if len(complete['qwen'])==30 else '双裁判未完成；一致性只列已完成子集，不能作为30题结论')
    machine={'updated_at':stamp,'status':status,'tasks':summaries,'agreement':analysis,'repeat_stability':repeat,'inter_judge':inter,'full_dual_tasks':complete['ensemble'],
             'largest_criterion_disagreements':sorted(gaps,key=lambda r:abs(r['delta']),reverse=True),
             'largest_weighted_contributions':sorted(gaps,key=lambda r:abs(r['weighted_contribution']),reverse=True),
             'disagreements_by_model':{k:sorted(v,key=lambda r:abs(r['weighted_contribution']),reverse=True) for k,v in model_gaps.items()},'quote_audit':audit,'method':METHOD}
    dump(OUT/'一致性汇总.json',machine);dump(PAGES/'评分数据.json',machine)
    dump(OUT/'双裁判综合状态.json',{'updated_at':stamp,'status':status,'assistant_tasks':30,'valid_judge_calls':len(rec),
         'qwen_calls':sum(r['model']=='qwen' for r in rec),'claude_calls':sum(r['model']=='claude' for r in rec),
         'complete_qwen_tasks':len(complete['qwen']),'complete_claude_tasks':len(complete['claude']),
         'complete_dual_tasks':len(complete['ensemble']),'qwen_resumption':qstate,
         'original_runner_notice':'原始运行状态只计该进程收集的评分，可能保留历史Qwen账户阻塞说明；综合进度按有效原始记录重新核查。'})
    report=['# DeepResearchBench参考报告30题评分一致性与分歧分析',f'评估日期：2026-10-07；最近刷新：{stamp}。',f'状态：**{status}**。',
            '固定种子20261007，从官方100题总体均匀随机抽30题，保留首次抽样。ID按抽样顺序：'+', '.join(str(t['id']) for t in tasks)+'。',
            '样本：30份清洗后参考报告（18中文、12英文），758条官方标准。来源为[官方DeepResearch Bench](https://github.com/Ayanami0730/deep_research_bench)，提交469cce54ea7f6a63c163d3d9fec879cf289ec484；原始数据指纹见随机抽样清单。',
            METHOD,'## 数据、模型与汇总口径','Qwen3.8-max：temperature=0.6，enable_thinking=true，reasoning_effort=xhigh，seed=1234，未设置输出token上限；初始调度并发2，本次续跑调度并发6（详见续跑记录）。Claude-opus-5：服务默认采样，输出上限12000，并发1。每模型每题两次，目标120份有效响应。保留当前系统虚构评分锚点，改为完整官方逐题细则及原生JSON完整性校验。',
            '格式补充（同一运行）：Claude出现额外空元数据或JSON引号未转义。沿用当前系统的JSON语法修复，同时要求原始criterion_id与数字score逐项齐全且与解析结果完全一致；完整流结束、全部官方细则和理由必须通过校验。额外null元数据可删除，数字分数不得补造或改变。每个重复编号取首个通过此规则的完整响应，原始失败包及格式核查记录另存。Qwen仍要求严格原生JSON。',
            '所有0–10分按官方细则和维度权重计算。双裁判先分别均两次，再等权平均；不将报告与其自身比较求0.5相对分。重复编号不同，稳定性包含编号提示变化。',
            f'已保存有效响应：{len(rec)}/120；助手：30/30；完整Qwen题：{len(complete["qwen"])}/30；完整Claude题：{len(complete["claude"])}/30；完整双裁判题：{len(complete["ensemble"])}/30。',
            '## 已测量一致性','MAE/RMSE衡量0–10分距离；偏差=裁判−助手；±1为绝对差≤1。Pearson/Spearman衡量线性/排序关联，CCC同时考虑尺度和位置一致性。总分以题为样本，维度和细则不可作为额外独立题目。总分MAE/偏差95%区间按题bootstrap4000次，种子20261007。',
            '| 比较 | 完整题数 | MAE | RMSE | 偏差 | ±1比例 | Pearson | Spearman | CCC |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model,v in analysis.items():
        m=v['total'];report.append('| '+LABELS[model]+' vs 助手 | '+str(m['n'])+' | '+' | '.join(statfmt(m[x]) for x in ['mae','rmse','bias'])+f' | {m["within_1"]:.1%} | '+' | '.join(statfmt(m[x]) for x in ['pearson','spearman','ccc'])+' |')
    if not analysis:report.append('尚未获得完整两次评分的单裁判题目，不能计算上述比较；缺失没有填0。')
    if 'ensemble' in analysis:
        report+=[f'## 当前完整双裁判子集的分歧维度（N={len(complete["ensemble"])}，不可代表30题）','| 维度 | MAE | 偏差 | ±1比例 |','|---|---:|---:|---:|']
        for d,m in sorted(analysis['ensemble']['dimensions'].items(),key=lambda z:z[1]['mae'],reverse=True):report.append(f'| {NAMES[d]} | {m["mae"]:.3f} | {m["bias"]:+.3f} | {m["within_1"]:.1%} |')
        report+=['## 当前双裁判子集的最大逐条分歧','| 题号 | 细则 | 助手 | 双裁判 | 差值 | 对总分贡献 |','|---|---|---:|---:|---:|---:|']
        for r in sorted(gaps,key=lambda r:abs(r['weighted_contribution']),reverse=True)[:20]:report.append(f'| {r["task_id"]} | {r["criterion"]} | {r["assistant"]:.2f} | {r["ensemble"]:.2f} | {r["delta"]:+.2f} | {r["weighted_contribution"]:+.3f} |')
    else:report+=['## 双裁判分歧主要方面：尚未评估',f'{block_notice} 完整双裁判一致性与分歧主因尚不能判断；第一轮观察到的报告缺口不是裁判分歧结论。']
    if 'qwen' in analysis:
        report+=['## Qwen与助手的分歧维度','| 维度 | 完整题数 | MAE | 偏差 | ±1比例 |','|---|---:|---:|---:|---:|']
        for d,m in sorted(analysis['qwen']['dimensions'].items(),key=lambda z:z[1]['mae'],reverse=True):report.append(f'| {NAMES[d]} | {m["n"]} | {m["mae"]:.3f} | {m["bias"]:+.3f} | {m["within_1"]:.1%} |')
        cmstat=analysis['qwen']['criteria']
        report+=['## Qwen细则级一致性',f'已完成题目的{cmstat["n"]}条细则：MAE={cmstat["mae"]:.3f}，偏差={cmstat["bias"]:+.3f}；完全同分{cmstat["exact_match"]:.1%}，差值≤0.5分{cmstat["within_0_5"]:.1%}，差值≤1分{cmstat["within_1"]:.1%}，差值≥2分{cmstat["gap_at_least_2"]:.1%}。各细则等权统计，采用Qwen两次均分；758条不视为758份独立报告，也不能代替官方权重总分。']
        ci=analysis['qwen']['total_bootstrap_ci95']
        if ci:report.append(f'Qwen总分MAE的95%题目bootstrap区间：[{ci["mae"][0]:.3f}, {ci["mae"][1]:.3f}]；偏差区间：[{ci["bias"][0]:+.3f}, {ci["bias"][1]:+.3f}]。')
        report+=['## Qwen最大逐条分歧（按对总分贡献绝对值排序）','| 题号/页面 | 细则 | 助手 | Qwen均分 | 差值 | 对总分贡献 |','|---|---|---:|---:|---:|---:|']
        for r in sorted(model_gaps['qwen'],key=lambda r:abs(r['weighted_contribution']),reverse=True)[:20]:report.append(f'| [题{r["task_id"]}]({PAGES}/题{r["task_id"]}.html) | {r["criterion"]} | {r["assistant"]:.2f} | {r["judge"]:.2f} | {r["delta"]:+.2f} | {r["weighted_contribution"]:+.3f} |')
        report+=['## Qwen重复评分稳定性',f'每題两次总分之间MAE={repeat["qwen"]["mae"]:.3f}，CCC={statfmt(repeat["qwen"]["ccc"])}；重复编号提示不同，不能将这两次视作独立的60题。']
    review=read(OUT/'Qwen分歧归因复核.json',{})
    if review:
        report+=['## Qwen分歧归因复核',review['analysis_markdown']]
    if qstate:
        report+=['## 2026-10-07 Qwen续跑记录',f'本轮续跑开始：{qstate["started_at"]}；最近更新：{qstate["updated_at"]}；状态：{qstate["status"]}。仅续跑Qwen，不重抽题、不调整助手评分，不向裁判提供助手结果。模型、提示词、输入、评分尺度和聚合方式均沿用已冻结配置。',
                 f'执行调度调整：冻结配置原并发2；本轮续跑并发{qstate.get("workers",2)}。首批7份有效评分复用；并发提高时尚未完成的流不计入评分，按原请求重新生成。这是吞吐调度变更，采样参数和评分口径不变；续跑历史与执行配置另存。',
                 f'[续跑状态]({OUT}/Qwen续跑状态.json) · [续跑历史]({OUT}/Qwen续跑历史.json) · [调度与冻结参数]({OUT}/Qwen续跑执行配置.json) · [续跑日志]({OUT}/Qwen续跑日志.txt) · [本次有效评分]({OUT}/Qwen续跑评分.json)']
    report+=['## 逐题第一轮结果','| 题号/页面 | 语言 | 助手总分 | 全面性 | 洞察力 | 指令遵循 | 可读性 |','|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        z=r['scores']['assistant'];report.append(f'| [题{r["task_id"]}]({PAGES}/题{r["task_id"]}.html) | {r["language"]} | '+ ' | '.join(fmt(z[d]) for d in ['total']+DIMS)+' |')
    report+=['## 失败样例与第一轮观察','第14题的参考正文保留摘要待撰写占位，末尾句中停止，尚未完成未来5–10年突破团队和技术预测。第72题期刊限定与MPRA、Diva Portal等未证明期刊身份来源有冲突。第10题的临界点及第26题的定量模型主要为框架，实际求解不足。第16题具体算法与准确率的可比对应不足。详情以逐题官方标准及页面第一轮理由为准。',
            '题1存在原始数据内部不一致：query是“所有阶层”，细则与参考报告保留“9阶层”要求。页面同时展示两份提示词，保留官方细则评分，指令遵循分数需结合该冲突阅读。',
            '## 局限性与边界','助手评分不是独立多人金标；助手与双裁判阅读流程不同。没有外部检索核实报告事实；所谓证据不足和待核查风险不等于已证明事实错误。历史参考报告按题目及报告内时间锚点评估，不因2026年评估日期机械扣旧数据分。30题来自全语言总体，语言/主题分组属于探索，不应以小子集下强结论。缺失、超时与账户错误不作为低分；引文自动连续匹配只核查引文定位，不证明论断真实。',
            '## 原始记录与可复现文件',f'- [30题HTML总览]({PAGES}/总览.html)',f'- [冻结输入]({OUT}/冻结输入.json)',f'- [随机抽样与指纹]({OUT}/随机抽样清单.json)',f'- [配置及助手冻结指纹]({OUT}/评估配置.json)',f'- [双裁判原始响应目录]({OUT}/裁判原始记录)',f'- [机器可读一致性汇总]({OUT}/一致性汇总.json)',f'- [Qwen最终格式与权重复算核查]({OUT}/Qwen最终核查.json)',f'- [30页完整性核查]({OUT}/Qwen页面最终核查.json)',f'- [Qwen分歧归因与来源指纹]({OUT}/Qwen分歧归因复核.json)',f'- [综合运行状态]({OUT}/双裁判综合状态.json)',f'- [原始Claude进程状态（不含独立Qwen续跑）]({OUT}/运行状态.json)',
            '## 结论与下一步',('完整双裁判30题已完成，一致性指标和分歧均可按此样本读取。' if len(complete['ensemble'])==30 else
                                   ('Qwen已完成30题各两次评分，可报告Qwen与助手的一致性；完整双裁判结论仍待Claude补齐。' if len(complete['qwen'])==30 else f'Qwen完成{len(complete["qwen"])}题各两次评分；完整30题的一致性与双裁判主分歧尚待补齐。')),
            f'单独续跑Qwen（复用匹配缓存，不并行启动第二个Qwen续跑）：\n```sh\ncd "{ROOT}"\nbackend/.venv/bin/python backend/eval/scripts/续跑DeepResearchBench的Qwen评分.py\nbackend/.venv/bin/python backend/eval/scripts/生成DeepResearchBench评分页面与分析.py\n```']
    DOC.parent.mkdir(exist_ok=True,parents=True)
    DOC.write_text(re.sub(r'(\|)\n\n(?=\|)',r'\1\n','\n\n'.join(report)))
    (PAGES/'评估报告.md').write_text(DOC.read_text())
    rows=''
    for pos,r in enumerate(summaries,1):
        z=r['scores'];delta=z.get('ensemble',{}).get('total');delta=None if delta is None else delta-z['assistant']['total']
        qdelta=z.get('qwen',{}).get('total');qdelta=None if qdelta is None or r['counts']['qwen']!=2 else qdelta-z['assistant']['total']
        rows+=f'<tr data-id="{r["task_id"]}"><td>{pos}</td><td><a href="题{r["task_id"]}.html"><b>题{r["task_id"]}</b></a></td><td style="min-width:320px"><a href="题{r["task_id"]}.html">{E(r["prompt"][:115])}</a></td><td>{r["language"]}</td>'+''.join('<td>'+fmt(z.get(k,{}).get('total'))+'</td>' for k in ['assistant','qwen','claude','ensemble'])+f'<td>{"待计算" if qdelta is None else f"{qdelta:+.2f}"}</td><td>{"待计算" if delta is None else f"{delta:+.2f}"}</td><td>Q {r["counts"]["qwen"]}/2 · C {r["counts"]["claude"]}/2</td></tr>'
    review_html='<section><h2>Qwen分歧归因复核</h2>'+md(review['analysis_markdown'])+'</section>' if review else ''
    b=f'<header><div class="eyebrow">DEEPRESEARCH BENCH · 随机30题评估</div><h1>参考报告评分与双裁判一致性</h1><p>逐题官方标准 · 助手第一轮已冻结 · 裁判结果可追溯</p></header><main><div class="grid"><div class="card"><div class="muted">助手评分</div><div class="number">30 / 30</div></div><div class="card"><div class="muted">官方细则</div><div class="number">758</div></div><div class="card"><div class="muted">有效裁判调用</div><div class="number">{len(rec)} / 120</div></div><div class="card"><div class="muted">完整双裁判题目</div><div class="number">{len(complete["ensemble"])} / 30</div></div></div><div class="notice"><b>{E(status)}</b><br>{E(block_notice)} Claude评分随完成追加。最近刷新：{E(stamp)}。</div><section><h2 style="margin-top:0">如何读这些分数</h2><p>{METHOD}</p><p>先按官方权重逐题汇总；每位裁判各评两次，取均分后再等权合并。0–10分，差值正数表示双裁判高于助手。点击题号进入完整标准、评分理由、原文证据和全文。</p><p><a href="评估报告.md">分析报告（Markdown）</a> · <a href="评分数据.json">机器可读评分数据</a> · <a href="https://github.com/Ayanami0730/deep_research_bench" target="_blank">官方Bench来源</a></p></section><section><h2 style="margin-top:0">评分一致性（样本数见N）</h2>{metric_table(analysis) if analysis else "尚无完整两次评分的裁判题目；一致性尚未测量。"}<p class="muted">各行N=30时代表完整30题；N&lt;30时仅代表已完成子集。MAE越低、CCC越高表示更接近；相关性高也可能存在系统偏差。</p></section>{qwen_plot(summaries)}{review_html}<div class="actions"><input id="search" placeholder="搜索题号、主题或语言" oninput="filter()"><button onclick="location.reload()">刷新裁判结果</button></div><section><h2 style="margin-top:0">30题评分清单</h2><div class="scroll"><table id="tasks"><thead><tr><th>抽样序</th><th>页面</th><th>任务</th><th>语言</th><th>助手</th><th>Qwen评分（n/2）</th><th>Claude评分（n/2）</th><th>双裁判</th><th>Qwen−助手</th><th>双裁判−助手</th><th>完成次数</th></tr></thead><tbody>{rows}</tbody></table></div></section></main><script>function filter(){{let q=document.getElementById("search").value.toLowerCase();document.querySelectorAll("#tasks tbody tr").forEach(x=>x.classList.toggle("hidden",!x.textContent.toLowerCase().includes(q)));}}</script>'
    (PAGES/'总览.html').write_text(shell('30题参考报告评分总览',b))
    dump(OUT/'页面生成核查.json',{'updated_at':stamp,'pages':30,'criterion_count':758,'html_files':[str(PAGES/f'题{t["id"]}.html') for t in tasks],'input_characters':sum(len(t['article']) for t in tasks),'quote_matches':sum(r['quote_status']=='连续匹配' for r in audit),'quote_mismatches':sum(r['quote_status']=='未匹配' for r in audit),'quote_empty':sum(r['quote_status']=='空引文' for r in audit),'frozen_assistant_checks':30})
    print('页面已刷新：30题；有效裁判响应',len(rec),'；完整双裁判题',len(complete['ensemble']),flush=True)
if __name__=='__main__':
    generate()
    if len(sys.argv)>1 and sys.argv[1]=='watch':
        pids=[int(x) for x in sys.argv[2:]]
        def signatures():
            return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in [OUT/'运行状态.json',OUT/'Qwen续跑状态.json',OUT/'Qwen分歧归因复核.json'])
        old=signatures()
        while True:
            time.sleep(10)
            cur=signatures()
            if cur!=old:generate();old=cur
            alive=[]
            for pid in pids:
                try:os.kill(pid,0);alive.append(pid)
                except ProcessLookupError:pass
            pids=alive
            if not pids:generate();break
