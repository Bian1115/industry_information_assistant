"""将真实摘要离线核验结果生成可离线保存标注的单文件HTML及中文报告。"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def build(directory):
    manifest = read(directory / '轨迹与真实触发点清单.json')
    config = read(directory / '运行配置.json')
    data = {'version': 'natural-summary-review-v1', 'created_at': datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
            'dataset_id': hashlib.sha256(json.dumps([manifest, config['code_hashes']], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'topics': manifest['topics'], 'samples': [], 'atoms': [], 'parents': [], 'coverage': []}
    for spec in manifest['samples']:
        p = directory / spec['id']
        result = read(p / '运行结果.json', {})
        freeze = read(p / '冻结摘要及生成输入.json')
        context = read(p / '实际输入上下文.json', {'sources': {}, 'rows': {}})
        audit = read(p / '完整引用核验.json', {})
        catalog = read(p / '摘要分句目录.json', [])
        review = read(p / '原子论断盲评模型初评.json', {'rows': []})
        model = {r['claim_id']: r for r in review['rows']}
        reports = {r['claim_id']: r for r in audit.get('nli_reports', [])}
        supplement_files = sorted(p.glob('内容保留分批补评*/补评结果.json'))
        supplements = [read(f) for f in supplement_files]
        completed = [f for f in supplement_files if read(f).get('status') == 'completed']
        quality_directory = completed[-1].parent if completed else p
        supplement = read(quality_directory / '补评结果.json', {})
        sample = {**spec, 'summary': freeze['summary'], 'result': result, 'quality_supplement': supplement,
                  'all_quality_supplements': supplements,
                  'sources': [{'id': sid, 'kind': context['rows'].get(sid, {}).get('kind'),
                               'seq': context['rows'].get(sid, {}).get('seq'), 'pieces': pieces}
                              for sid, pieces in context['sources'].items()],
                  'generation_input_prepared': freeze['summary_input_prepared'],
                  'is_hierarchical': len(freeze['generation_requests']) > 1,
                  'upstream_requests': [r['payload']['request']['messages'][1]['content'] for r in freeze['generation_requests'][:-1]],
                  'summary_catalog': catalog}
        data['samples'].append(sample)
        for i, atom in enumerate(audit.get('mapped_claims', []), 1):
            rep = reports.get(atom['claim_id'], {})
            nli = rep.get('report', {})
            status = nli.get('claims', [{}])[0].get('status') if nli.get('claims') else 'execution_error'
            data['atoms'].append({'id': f"{spec['id']}-a{i:03d}", 'sample_id': spec['id'], 'topic': spec['topic'],
                'claim_id': atom['claim_id'], 'claim': atom['claim'], 'parent_id': atom['parent_claim_id'],
                'parent': atom['origin']['parent_claim'], 'section': atom['section'], 'system_status': status,
                'references': atom.get('references', []), 'candidate_references': atom.get('candidate_references', []),
                'windows': nli.get('all_windows', []), 'nli': nli.get('claims', []),
                'error': atom.get('error') or rep.get('error'), 'retrieval_used': atom.get('retrieval_used', False),
                'model_review': model.get(atom['claim_id']), 'human': {}})
        children = {}
        for a in data['atoms']:
            if a['sample_id'] == spec['id']: children.setdefault(a['parent_id'], []).append(a['id'])
        traces = {a['parent_claim_id']: a['extraction'] for a in audit.get('claimify', {}).get('audit', [])}
        for i, parent in enumerate(audit.get('binding', {}).get('claims', []), 1):
            trace = traces.get(parent['claim_id'], {})
            ids = children.get(parent['claim_id'], [])
            data['parents'].append({'id': f"{spec['id']}-p{i:03d}", 'sample_id': spec['id'], 'topic': spec['topic'],
                'claim': parent['claim'], 'section': parent['section'], 'parent_id': parent['claim_id'],
                'system_status': 'extracted' if ids else 'excluded_or_unresolved', 'atomic_ids': ids,
                'source_ids': parent['source_ids'], 'trace': trace, 'human': {}})
        reference = read(quality_directory / '重要信息参考清单.json', {'checkpoints': []})
        quality = read(quality_directory / '摘要内容保留模型初评.json', {'coverage': [], 'issues': []})
        judged = {r['id']: r for r in quality['coverage']}
        sample['quality_issues'] = quality.get('issues', [])
        for checkpoint in reference['checkpoints']:
            pred = judged.get(checkpoint['id'])
            data['coverage'].append({'id': f"{spec['id']}-{checkpoint['id']}", 'sample_id': spec['id'], 'topic': spec['topic'],
                'claim': checkpoint['statement'], 'category': checkpoint['category'], 'importance': checkpoint['importance'],
                'references': checkpoint['references'], 'system_status': pred['status'] if pred else 'not_run',
                'model_review': pred, 'human': {}})
    return data


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>自然摘要 · 引用与信息保留审核</title>
<style>
:root{--nav:#142338;--ink:#203047;--muted:#6b7b8e;--line:#dbe3ec;--bg:#f3f6fa;--blue:#1665cc;--light:#e9f1ff;--green:#17715e;--red:#b74340;--amber:#a0650b}*{box-sizing:border-box}body{margin:0;background:var(--bg);font:14px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink)}button,select,input,textarea{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible{outline:3px solid #77aaff;outline-offset:2px}.top{background:var(--nav);color:white;padding:23px 30px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{font-size:23px;font-weight:650;letter-spacing:.3px}.sub{color:#b5c4d7;font-size:12px;margin-top:5px}.actions{display:flex;gap:9px;flex-wrap:wrap}.btn{border:1px solid #cad6e4;background:white;color:var(--ink);padding:8px 13px;border-radius:7px}.primary{background:#2675df;color:white;border-color:#2675df}.top .btn{background:#22354e;color:white;border-color:#435674}.top .primary{background:#2675df;border-color:#2675df}.overview{padding:19px 30px 0;display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.metric{background:white;border:1px solid var(--line);border-radius:10px;padding:15px 19px}.metric b{display:block;font-size:25px}.metric span{color:var(--muted);font-size:12px}.notice{margin:15px 30px 18px;border-left:3px solid var(--blue);background:#eaf1fb;padding:11px 15px;font-size:13px}.shell{display:grid;grid-template-columns:330px minmax(0,1fr);margin:0 30px 30px;gap:20px}.sidebar,.panel{background:white;border:1px solid var(--line);border-radius:11px;overflow:hidden}.controls{padding:15px;border-bottom:1px solid var(--line)}.controlrow{display:flex;gap:8px;margin-top:8px}input[type=search],select{min-width:0;width:100%;border:1px solid #ccd7e3;border-radius:6px;background:white;padding:8px 9px}.tabs{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:12px}.tab{padding:6px 8px;border:0;border-radius:5px;background:#f0f3f7;color:var(--muted);font-size:12px}.tab.active{color:var(--blue);background:var(--light);font-weight:650}.listnote{padding:8px 15px;color:var(--muted);font-size:12px}.rows{overflow:auto;height:calc(100vh - 480px);min-height:310px}.row{display:block;width:100%;text-align:left;background:white;padding:13px 16px;border:0;border-top:1px solid #edf0f4;color:var(--ink)}.row.active{background:#edf4ff;box-shadow:inset 3px 0 var(--blue)}.row .rid{font-size:11px;color:var(--muted);margin-bottom:5px}.row .text{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;font-size:13px}.badge{display:inline-block;border-radius:4px;padding:1px 6px;font-size:11px;margin-right:5px;background:#edf1f6;color:#65758a}.badge.green{background:#e5f4ee;color:var(--green)}.badge.red{background:#fceeed;color:var(--red)}.badge.amber{background:#fff2db;color:var(--amber)}.head{padding:18px 23px;display:flex;justify-content:space-between;gap:14px;align-items:center;border-bottom:1px solid var(--line)}.eyebrow{font-size:11px;color:var(--muted);letter-spacing:.5px}.title{font-size:18px;font-weight:650;margin:2px 0}.pane{padding:20px 23px}.block{border:1px solid var(--line);border-radius:9px;margin-bottom:17px;overflow:hidden}.blockhead{padding:10px 14px;font-weight:600;background:#f7f9fc;border-bottom:1px solid var(--line);font-size:13px}.blockbody{padding:14px}.claim{font-size:17px;line-height:1.9;font-weight:520}.parent{color:var(--muted);font-size:13px;margin-top:11px}.evidence{background:#fbfcfe;border-left:3px solid #c5d6ed;padding:11px 13px;margin-bottom:10px}.meta{font:11px/1.5 ui-monospace,SFMono-Regular,monospace;color:var(--muted);overflow-wrap:anywhere;margin-bottom:7px}.textpre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.8}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}.fieldset{margin-bottom:12px}.fieldset label{display:block;font-size:12px;font-weight:650;margin-bottom:5px}.fieldset select{padding:7px 8px}textarea{resize:vertical;width:100%;min-height:84px;border:1px solid #ccd7e3;border-radius:6px;padding:9px}.footer{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:12px}.save{color:var(--green);font-size:12px}.empty{color:var(--muted);padding:22px}.small{font-size:12px;color:var(--muted)}details{margin:10px 0}summary{cursor:pointer;font-size:13px;color:var(--blue)}pre{overflow:auto;background:#f2f5fa;padding:12px;font:12px/1.6 ui-monospace,monospace;max-height:400px}.summarytext{white-space:pre-wrap;font-size:13px;line-height:1.85}.pred{background:#fff8eb;border:1px solid #f1dfbc;border-radius:7px;padding:12px;font-size:13px;margin-bottom:14px}.statusbar{color:var(--muted);font-size:12px;padding:0 30px 16px}.toggle{display:flex;align-items:center;gap:6px;font-size:12px}.toggle input{width:16px;height:16px}#toast{position:fixed;right:25px;bottom:25px;background:#22354e;color:white;padding:11px 18px;border-radius:9px;display:none;z-index:5}.scope{color:var(--muted);font-size:12px;max-width:590px}@media(max-width:1000px){.shell{grid-template-columns:280px minmax(0,1fr);margin:0 16px 20px;gap:12px}.overview{padding:16px 16px 0}.top{padding:18px 16px}.notice{margin:12px 16px}.pair{grid-template-columns:1fr}.metric{padding:12px}.head,.pane{padding:15px}}@media(max-width:700px){.top{display:block}.actions{margin-top:13px}.overview{grid-template-columns:repeat(3,1fr)}.shell{display:block}.sidebar{margin-bottom:12px}.rows{height:230px;min-height:200px}.brand{font-size:19px}}
</style></head><body>
<header class="top"><div><div class="brand">自然摘要 · 人工审核</div><div class="sub">最新中文五题真实触发点 / 引用支持 · 抽取完整性 · 信息保留</div></div><div class="actions"><button class="btn" id="importBtn">导入标注</button><button class="btn primary" id="exportBtn">导出标注 JSON</button><button class="btn" id="exportAll">下载审核数据</button><input hidden type="file" id="importFile" accept="application/json"></div></header>
<section class="overview" id="metrics"></section>
<div class="notice">这是冻结轨迹的离线审计：核验不改写摘要，也不修改原研究会话。引用通过率只描述已抽取论断；信息遗漏单独审核。模型初评不是人工金标，网页不会替你填写裁决。</div>
<main class="shell"><aside class="sidebar"><div class="controls"><div class="tabs" id="tabs"></div><select aria-label="选择摘要" id="sampleFilter"></select><div class="controlrow"><select id="reviewFilter" aria-label="审核进度"><option value="all">全部条目</option><option value="pending">待审核</option><option value="reviewed">已审核</option></select><select id="riskFilter" aria-label="系统状态筛选"><option value="all">全部系统状态</option></select></div><input type="search" id="search" placeholder="搜索论断、数字或资源" aria-label="搜索条目" style="margin-top:8px"><label class="toggle" style="margin-top:11px"><input type="checkbox" id="showPred">显示系统结论与模型初评</label></div><div class="listnote" id="listnote"></div><div class="rows" id="rows"></div></aside><section class="panel"><div class="head"><div><div class="eyebrow" id="eyebrow"></div><div class="title" id="title"></div></div><div class="actions"><button class="btn" id="prev">上一条</button><button class="btn" id="next">下一条</button></div></div><div class="pane" id="detail"></div></section></main>
<div class="statusbar" id="statusbar"></div><div id="toast" role="status"></div><div id="exportDialog" role="dialog" aria-modal="true" aria-label="标注JSON导出" style="display:none;position:fixed;inset:0;background:#14233890;z-index:20;padding:5vh 8vw"><div style="background:white;border-radius:12px;padding:22px;max-width:900px;margin:auto"><div class="title">标注 JSON 已生成</div><div class="small">如果浏览器没有开始下载，可复制下方内容，保存为 .json 文件。</div><textarea id="exportText" aria-label="导出JSON内容" readonly style="height:55vh;font:12px/1.5 monospace;margin:14px 0"></textarea><div class="actions"><button class="btn primary" id="copyExport">复制 JSON</button><button class="btn" id="closeExport">关闭</button></div></div></div>
<script type="application/json" id="dataset">__DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('dataset').textContent),KEY='natural-summary-review:'+D.dataset_id;
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={supported:'充分支持',unknown:'信息不足',contradiction:'矛盾',execution_error:'执行错误',preserved:'完整保留',partial:'部分保留',missing:'遗漏',contradicted:'含义冲突',not_run:'未评审',extracted:'已抽取',excluded_or_unresolved:'未抽取 / 未决',insufficient:'支持不足',uncertain:'待裁决',faithful:'保真',changed:'改坏父句'};
let tab='atoms',selected=null,visible=[],ann={};
try{ann=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){toast('浏览器本地标注不可用，请及时导出 JSON')}
const all=[...D.atoms,...D.parents,...D.coverage],lookup=new Map(all.map(r=>[r.id,r])),samples=new Map(D.samples.map(s=>[s.id,s]));
const choices={support:[['supported','充分支持'],['insufficient','必要信息不足'],['contradicted','原文明确反对'],['uncertain','待裁决']],selected_support:[['supported','所选证据充分支持'],['insufficient','所选证据不足'],['contradicted','所选证据明确反对'],['uncertain','待裁决']],fidelity:[['faithful','子论断忠于父句'],['changed','子论断改坏父句'],['uncertain','待裁决']],verify_type:[['fact','研究事实'],['operation','已发生操作 / 结果'],['requirement','用户要求'],['plan','计划 / 待办'],['resource','资源标识'],['mixed','混合内容'],['nonfact','无可核验内容']],extraction:[['complete','需要的事实均已抽取'],['partial','漏掉部分事实'],['wrongly_excluded','事实被错误排除'],['valid_excluded','合理排除'],['uncertain','待裁决']],coverage:[['preserved','完整保留'],['partial','部分保留'],['missing','遗漏'],['contradicted','含义冲突'],['uncertain','待裁决']],importance:[['critical','关键'],['useful','有用'],['not_important','无需保留'],['uncertain','待裁决']]};
function reviewed(r){const a=ann[r.id]||{};return tab==='atoms'?!!(a.support&&a.fidelity&&a.selected_support):tab==='parents'?!!(a.extraction&&a.verify_type):!!(a.coverage&&a.importance)}
function countReviewed(r,type){const a=ann[r.id]||{};return type==='atoms'?!!(a.support&&a.fidelity&&a.selected_support):type==='parents'?!!(a.extraction&&a.verify_type):!!(a.coverage&&a.importance)}
function toast(s){$('toast').textContent=s;$('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',3500)}
function badge(s){let c=['supported','preserved','faithful'].includes(s)?'green':['contradiction','contradicted','changed','execution_error'].includes(s)?'red':'amber';return `<span class="badge ${c}">${esc(names[s]||s)}</span>`}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ann));$('saved')&&($('saved').textContent='已保存至本浏览器')}catch(e){toast('本地保存失败，请导出标注 JSON')}metrics();renderList()}
function metrics(){let n=all.filter(r=>countReviewed(r,D.atoms.includes(r)?'atoms':D.parents.includes(r)?'parents':'coverage')).length;let supported=D.atoms.filter(r=>r.system_status==='supported').length;$('metrics').innerHTML=[['5','冻结题目 / 1题未触发'],[D.samples.length,'真实摘要 / 2个智能体'],[D.atoms.length,`原子论断 · 系统支持 ${supported}`],[D.coverage.length,'重要信息检查点'],[n+'/'+all.length,'人工审核完成']].map(([v,t])=>`<div class="metric"><b>${esc(v)}</b><span>${esc(t)}</span></div>`).join('')}
function initTabs(){$('tabs').innerHTML=[['atoms','原子论断'],['parents','摘要父句'],['coverage','信息保留'],['sources','原始输入']].map(([v,l])=>`<button class="tab ${tab===v?'active':''}" data-tab="${v}">${l}</button>`).join('');$('tabs').querySelectorAll('button').forEach(b=>b.onclick=()=>{tab=b.dataset.tab;selected=null;initTabs();riskOptions();renderList();renderDetail()})}
function riskOptions(){let statuses=tab==='sources'?[]:[...new Set(D[tab].map(r=>r.system_status))];$('riskFilter').innerHTML='<option value="all">全部系统状态</option>'+statuses.map(s=>`<option value="${esc(s)}">${esc(names[s]||s)}</option>`).join('');$('riskFilter').disabled=tab==='sources';$('reviewFilter').disabled=tab==='sources'}
function filters(){let sid=$('sampleFilter').value,q=$('search').value.toLowerCase(),r=$('reviewFilter').value,risk=$('riskFilter').value;if(tab==='sources')return D.samples.filter(s=>(sid==='all'||sid===s.id)&&(!q||s.summary.toLowerCase().includes(q))).map(s=>({id:s.id,sample_id:s.id,claim:s.summary,system_status:'source',topic:s.topic}));return D[tab].filter(x=>(sid==='all'||x.sample_id===sid)&&(!q||[x.claim,x.parent,x.id].join(' ').toLowerCase().includes(q))&&(r==='all'||(r==='reviewed')===reviewed(x))&&(risk==='all'||x.system_status===risk))}
function renderList(){visible=filters();if(!visible.some(r=>r.id===selected))selected=visible[0]?.id||null;$('listnote').textContent=`${visible.length} 条${tab==='sources'?'摘要输入':'审核项'} · ${tab==='sources'?'第35题没有自然摘要':visible.filter(reviewed).length+' 条已完成裁决'}`;$('rows').innerHTML=visible.map(r=>`<button class="row ${r.id===selected?'active':''}" data-id="${esc(r.id)}"><div class="rid">${esc(r.id)}</div><div class="text">${esc(tab==='sources'?samples.get(r.id).agent+' · 第'+r.topic+'题':r.claim)}</div><div style="margin-top:7px">${tab!=='sources'?'<span class="badge '+(reviewed(r)?'green':'')+'">'+(reviewed(r)?'已审核':'待审核')+'</span>':''}${$('showPred').checked&&tab!=='sources'?badge(r.system_status):''}</div></button>`).join('')||'<div class="empty">当前筛选没有条目。</div>';$('rows').querySelectorAll('.row').forEach(b=>b.onclick=()=>{selected=b.dataset.id;renderList();renderDetail()})}
function evidence(refs){if(!refs?.length)return '<div class="small">没有选中的证据。可展开下方完整实际输入自行核对；执行错误与信息不足应分别判断。</div>';return refs.map(r=>`<div class="evidence"><div class="meta">${esc(r.source_id)} · ${esc(r.sentence_id||r.unit_id||'')} · 字符 ${esc(r.start)}–${esc(r.end)} ${r.source_spans?'· 组合片段':''}</div><div class="textpre">${esc(r.quote||r.text)}</div>${r.source_spans?'<details><summary>原始片段与位置</summary><pre>'+esc(JSON.stringify(r.source_spans,null,2))+'</pre></details>':''}</div>`).join('')}
function sourceHtml(s){return s.sources.map(r=>`<details><summary>${esc(r.kind||'输入片段')} · ${esc(r.id)} · seq ${esc(r.seq)}</summary>${r.pieces.map(p=>`<div class="evidence"><div class="meta">实际提供范围 ${p.start}–${p.start+p.text.length}</div><div class="textpre">${esc(p.text)}</div></div>`).join('')}</details>`).join('')}
function block(title,body){return `<div class="block"><div class="blockhead">${title}</div><div class="blockbody">${body}</div></div>`}
function field(r,key,label){let a=ann[r.id]||{};return `<div class="fieldset"><label for="f-${key}">${label}</label><select id="f-${key}" data-key="${key}"><option value="">未裁决</option>${choices[key].map(([v,l])=>`<option value="${v}" ${a[key]===v?'selected':''}>${l}</option>`).join('')}</select></div>`}
function renderDetail(){const r=visible.find(x=>x.id===selected);if(!r){$('title').textContent='没有可显示的条目';$('eyebrow').textContent='';$('detail').innerHTML='<div class="empty">可调整筛选条件。</div>';return}const s=samples.get(r.sample_id),pred=$('showPred').checked;$('title').textContent=tab==='sources'?'摘要与实际生成输入':tab==='atoms'?'引用支持与父子保真':tab==='parents'?'摘要事实抽取完整性':'重要信息保留';$('eyebrow').textContent=`第${r.topic}题 / ${s.agent} / 摘要版本${s.version_in_agent} / ${r.id}`;let html='';
html+=`<div class="scope" style="margin-bottom:15px">核对范围：${s.is_hierarchical?'最终合并请求中的中间摘要与任务约束。上游分块输入仅供追溯，不能倒算为最终合并直接读到的证据。':'本次摘要生成请求实际提供的消息片段与任务约束。'} 字符位置使用 Python Unicode 下标。</div>`;
if(tab==='sources'){html+=block('原始已提交摘要',`<div class="summarytext">${esc(s.summary)}</div>`);html+=block('摘要调用与压缩数据',`<div class="small">压缩前后的全部上下文变化包含外置，不能全部归因于摘要。</div><pre>${esc(JSON.stringify({summary_tokens:s.result.summary_tokens,final_input_tokens:s.result.source_tokens,ratio:s.result.summary_to_final_input_ratio,processed_view:s.generation_input_prepared},null,2))}</pre>`);html+=block('最终生成请求实际输入',sourceHtml(s));if(s.upstream_requests.length)html+=block('上游分块请求（追溯范围）',s.upstream_requests.map((q,i)=>`<details><summary>分块请求 ${i+1}</summary><pre>${esc(q)}</pre></details>`).join(''));$('detail').innerHTML=html;return}
html+=block(tab==='coverage'?'待保留的重要信息':'审核对象',`<div class="claim">${esc(r.claim)}</div>${r.parent?'<div class="parent">父分句：'+esc(r.parent)+'</div>':''}${r.importance?'<div class="small" style="margin-top:8px">模型编制清单 · '+esc(r.category)+' · '+esc(r.importance)+'</div>':''}`);
if(pred){html+=`<div class="pred">${badge(r.system_status)} ${r.retrieval_used?'<span class="badge">使用补检索</span>':''}<div style="margin-top:7px">${esc(r.model_review?.reason||'')}</div>${r.model_review?.fidelity?'<div>模型初评：'+badge(r.model_review.fidelity)+badge(r.model_review.support)+'</div>':''}<div class="small">系统标签与模型初评都可被人工改判。未校准 NLI 分数不是事实正确概率。</div></div>`}
if(tab==='atoms'){html+=block('本轮最终选中原文证据',evidence(r.references));if(pred&&r.windows.length)html+=block('NLI 各窗口结果',r.windows.map(w=>`<details><summary>${esc(w.source_id)} · ${esc(w.status)} · 支持 ${Number(w.scores?.entailment||0).toFixed(4)} / 矛盾 ${Number(w.scores?.contradiction||0).toFixed(4)}</summary><div class="textpre">${esc(w.quote)}</div><pre>${esc(JSON.stringify(w.scores,null,2))}</pre></details>`).join(''));if(r.error)html+=block('执行异常',`<pre>${esc(JSON.stringify(r.error,null,2))}</pre>`)}
if(tab==='parents'){html+=block('抽取出的子论断',r.atomic_ids.length?r.atomic_ids.map(id=>`<div class="evidence"><div class="meta">${esc(id)}</div>${esc(lookup.get(id).claim)}</div>`).join(''):'<div class="small">该父句没有形成可核验子论断；请判断是否包含被漏掉的事实或操作记录。</div>');if(pred)html+=block('Claimify 筛选与拆解记录',`<details><summary>查看抽取过程</summary><pre>${esc(JSON.stringify(r.trace,null,2))}</pre></details>`)}
if(tab==='coverage'){html+=block('清单对应的实际输入原文',evidence(r.references));const ids=r.model_review?.summary_sentence_ids||[];html+=block('对应摘要内容',`<div class="summarytext">${esc(s.summary)}</div>${pred&&ids.length?'<div class="small" style="margin-top:9px">模型定位：'+esc(ids.join(', '))+'</div>':''}`)}
html+=block('人工裁决',`<div class="pair">${tab==='atoms'?field(r,'support','完整实际输入是否支持子论断')+field(r,'selected_support','最终所选证据是否充分支持')+field(r,'fidelity','子论断是否保留父句含义'):tab==='parents'?field(r,'verify_type','父句内容类型')+field(r,'extraction','该父句抽取是否完整'):field(r,'coverage','摘要保留情况')+field(r,'importance','是否值得保留')}</div><label class="small" for="notes">备注、缺失限定或修正建议</label><textarea id="notes" placeholder="填写判断依据；不必重复系统结论">${esc(ann[r.id]?.notes||'')}</textarea><div class="footer"><span class="save" id="saved">标注仅保存至本浏览器；导出 JSON 可长期留存</span><button class="btn primary" id="nextPending">下一条待审核</button></div>`);
html+=`<details><summary>展开完整实际输入，检查选证据是否遗漏</summary>${sourceHtml(s)}</details><details><summary>查看完整原始摘要</summary><div class="summarytext">${esc(s.summary)}</div></details>`;$('detail').innerHTML=html;
$('detail').querySelectorAll('select[data-key]').forEach(el=>el.onchange=()=>{ann[r.id]={...ann[r.id],[el.dataset.key]:el.value,updated_at:new Date().toISOString()};const before=selected;persist();if(selected!==before)renderDetail()});$('notes').oninput=()=>{ann[r.id]={...ann[r.id],notes:$('notes').value,updated_at:new Date().toISOString()};persist()};$('nextPending').onclick=()=>{const pending=visible.filter(x=>!reviewed(x));const current=visible.findIndex(x=>x.id===selected);selected=pending.find(x=>visible.indexOf(x)>current)?.id||pending[0]?.id||selected;renderList();renderDetail()}}
function download(obj,name){const blob=new Blob([JSON.stringify(obj,null,2)],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function exportAnnotations(){const obj={schema:D.version,dataset_id:D.dataset_id,exported_at:new Date().toISOString(),annotations:all.map(r=>({id:r.id,human:ann[r.id]||{}}))};$('exportText').value=JSON.stringify(obj,null,2);$('exportDialog').style.display='block';download(obj,'自然摘要人工审核标注.json')}
$('closeExport').onclick=()=>{$('exportDialog').style.display='none'};$('copyExport').onclick=async()=>{try{await navigator.clipboard.writeText($('exportText').value);toast('已复制标注 JSON')}catch(e){$('exportText').focus();$('exportText').select();const ok=document.execCommand('copy');toast(ok?'已复制标注 JSON':'请手动选择并复制文本')}};
function importAnnotations(obj){if(obj.schema!==D.version||obj.dataset_id!==D.dataset_id||!Array.isArray(obj.annotations))throw Error('数据集身份或格式不一致');const next={},seen=new Set();for(const r of obj.annotations){if(!lookup.has(r.id)||seen.has(r.id)||!r.human||typeof r.human!=='object'||Array.isArray(r.human))throw Error('未知、重复编号或标注格式错误');seen.add(r.id);let allowed=D.atoms.some(a=>a.id===r.id)?['support','selected_support','fidelity','notes','updated_at']:D.parents.some(a=>a.id===r.id)?['verify_type','extraction','notes','updated_at']:['coverage','importance','notes','updated_at'];for(const [k,v] of Object.entries(r.human)){if(!allowed.includes(k)||typeof v!=='string'||(choices[k]&&v!==''&&!choices[k].some(([x])=>x===v)))throw Error('标注字段或值不合法')}next[r.id]=r.human}ann={...ann,...next};persist();renderDetail();return seen.size}
$('sampleFilter').innerHTML='<option value="all">全部 7 份真实摘要</option>'+D.samples.map(s=>`<option value="${s.id}">第${s.topic}题 · ${s.agent} · v${s.version_in_agent}</option>`).join('');
for(const id of ['sampleFilter','reviewFilter','riskFilter','search'])$(id).addEventListener(id==='search'?'input':'change',()=>{renderList();renderDetail()});$('showPred').onchange=()=>{renderList();renderDetail()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);function move(d){let i=visible.findIndex(r=>r.id===selected);selected=visible[Math.max(0,Math.min(visible.length-1,i+d))]?.id;renderList();renderDetail()}
$('exportBtn').onclick=exportAnnotations;$('exportAll').onclick=()=>download(D,'自然摘要完整审核数据.json');$('importBtn').onclick=()=>$('importFile').click();$('importFile').onchange=async()=>{try{const n=importAnnotations(JSON.parse(await $('importFile').files[0].text()));toast(`已导入 ${n} 条标注`)}catch(e){toast('导入失败：'+e.message)}finally{$('importFile').value=''}};
$('statusbar').textContent=`数据集 ${D.dataset_id.slice(0,16)} · ${D.created_at} · 断网可用 · 不自动上传标注`;
window.reviewApp={dataset:D,getAnnotations:()=>ann,importAnnotations,exportAnnotations};initTabs();riskOptions();metrics();renderList();renderDetail();
</script></body></html>'''


def report(data, directory):
    counts = Counter(a['system_status'] for a in data['atoms'])
    coverage = Counter(r['system_status'] for r in data['coverage'])
    usages = [r for s in data['samples'] for cost in [s['result'].get('cost', {}), *[q.get('cost', {}) for q in s['all_quality_supplements']]] for r in cost.get('requests', [])]
    prompt = sum((r.get('usage') or {}).get('prompt_tokens', 0) or 0 for r in usages)
    completion = sum((r.get('usage') or {}).get('completion_tokens', 0) or 0 for r in usages)
    text = f'''# 最新五题真实自然摘要引用核验与人工审核结果及分析

评估日期：{data['created_at']}，Asia/Singapore。

本次固定最新选用中文五题35、31、46、16、49，审核原始已提交摘要。第35题没有触发，其余4题共7次提交；没有重新生成摘要、强制触发或更新真实研究会话。原始轨迹与代码哈希保存在运行配置和触发点清单，核验只做离线审计，不执行失败论断删改闭环。

## 已测量的结果

- 实际摘要：{len(data['samples'])}份；父分句：{len(data['parents'])}；原子论断：{len(data['atoms'])}。
- 原子终态：{dict(counts)}。支持比例为{counts['supported']}/{len(data['atoms'])}；该比例不是精确率或信息保留率。
- 重要信息清单模型初评：{dict(coverage)}，分母{len(data['coverage'])}项。清单在评审调用前仅从来源编制，编制者未读摘要；自然摘要在采集时已存在，本轮没有生成前人工金标。
- 主评测LLM请求{len(usages)}次，已记录prompt {prompt} token、completion {completion} token。embedding/rerank和本地NLI开销独立，不估算总费用。

## 数据与配置口径

复用生产摘要分句编号、Claimify原子化、候选引用编号关联、相关证据选择、本地mDeBERTa NLI及不足时的临时检索。NLI revision为8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c，支持阈值0.85、矛盾阈值0.5，保持any_support_no_contradiction_v1聚合。没有实施此前文档中的联合证据及冲突复核改造。

支持核验只读取最终生成请求实际输入。两份Scout摘要来自多块合并，当前来源包含中间摘要；这能测末次改写的忠实度，不能证明中间摘要相对原始网页的完整性。审核页另保存上游分块输入供追溯，原始轨迹入口保留。Writer输入多含外置后的历史视图；不把其压缩率全部归于摘要。

内容保留和原子盲评均使用同一个已配置摘要模型{read(directory/'运行配置.json')['quality_model']}；盲评调用未提供NLI预测，仍不是独立人工金标。所有human字段为空，需用户人工裁决后复算。被Claimify排除的父句单列，不视为通过。

## 逐摘要结果

|题号/智能体/版本|摘要token|最终输入token|摘要/最终输入|父句/原子|原子终态|保留模型初评|核验状态|
|---|---:|---:|---:|---|---|---|---|
'''
    for s in data['samples']:
        r = s['result']
        parents = sum(x['sample_id'] == s['id'] for x in data['parents'])
        ratio = r.get('summary_to_final_input_ratio')
        text += f"|{s['id']}|{r.get('summary_tokens','—')}|{r.get('source_tokens','—')}|{ratio:.2%}" if ratio is not None else f"|{s['id']}|—|—|—"
        text += f"|{parents}/{r.get('atomic_claim_count',0)}|{r.get('nli_counts',{})}|{s.get('quality_supplement',{}).get('coverage_counts', r.get('coverage_counts',{}))}|{r.get('verification_status')}|\n"
    text += '\n## 失败样例与分析\n\n'
    failures = [r for r in data['atoms'] if r['system_status'] != 'supported']
    for r in failures[:10]:
        text += f"- {r['id']}（{r['system_status']}）：{r['claim']}。模型复核依据：{(r.get('model_review') or {}).get('reason','未形成有效初评，需人工检查')}\n"
    if not failures: text += '本轮已抽取原子论断没有未通过项；这不证明未抽取内容或遗漏不存在。\n'
    losses = [r for r in data['coverage'] if r['system_status'] in {'partial', 'missing', 'contradicted'}]
    text += '\n重要信息保留缺口（模型暂定，待人工）：\n\n'
    for r in losses[:12]: text += f"- {r['id']}（{r['system_status']}）：{r['claim']}。{(r.get('model_review') or {}).get('reason','')}\n"
    errors = [(s['id'], {k: v for k, v in s['result'].items() if 'error' in k}) for s in data['samples'] if any('error' in k for k in s['result'])]
    text += f'''\n执行失败记录：{errors or '无'}。执行错误不转成事实矛盾。补充评审若发生，单独记录调用及实际usage，不覆盖原始失败文件。

## 局限与结论

已测量：真实触发点、来源定位、抽取终态、NLI/检索及模型清单保留结果、实际调用usage。间接推断：复核意见可帮助定位证据组织或抽取问题，仍应人工确认。尚未评估：独立人工金标上的精确率/召回率、全部重要事实召回、外部网页真实性、处理失败论断后的净收益、最终报告质量和在线随机消融。

同题多次摘要相关，7个触发点不是7条独立研究轨迹；31沿用旧100万token预算成功尝试，其余4题无总token限制。自然摘要过短或引用通过不能代替内容完整性，核验也无法找回生成时完全遗漏的内容。

下一步在默认隐藏预测的界面中审核原子支持、父子保真、父句排除和重要信息保留，导出标注JSON。人工裁决后再计算支持判定指标；不要依据未决NLI标签直接整句删除。

## 原始结果与审核入口

- [人工审核HTML]({directory/'自然摘要引用与信息保留人工审核.html'})
- [完整机器审核数据]({directory/'完整人工审核数据.json'})
- [轨迹与触发点清单]({directory/'轨迹与真实触发点清单.json'})
- [运行配置及代码哈希]({directory/'运行配置.json'})
- [逐摘要结果]({directory/'汇总结果.json'})
- [原始资料完整性复核]({directory/'结束完整性复核.json'})

每份摘要目录保存冻结摘要/生成输入、实际输入上下文、重要信息清单、内容保留初评、完整引用核验、NLI窗口、原子盲评和全部LLM请求返回。HTML为独立单文件，支持本地保存标注、按状态筛选、导入/导出；本地存储可能被浏览器清理，完成后应导出JSON。人工字段为空，系统与模型字段单独保存。
'''
    return text


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(); directory = args.directory.resolve()
    data = build(directory)
    (directory / '完整人工审核数据.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    (directory / '自然摘要引用与信息保留人工审核.html').write_text(TEMPLATE.replace('__DATA__', payload))
    root = Path(__file__).resolve().parents[3]
    name = '最新五题真实自然摘要引用核验与人工审核结果及分析-' + directory.name.rsplit('-', 2)[-2] + '-' + directory.name.rsplit('-', 1)[-1] + '.md'
    (root / 'docs' / name).write_text(report(data, directory))
    print(json.dumps({'samples': len(data['samples']), 'atoms': len(data['atoms']), 'parents': len(data['parents']), 'coverage': len(data['coverage']), 'html': str(directory / '自然摘要引用与信息保留人工审核.html'), 'report': str(root / 'docs' / name)}, ensure_ascii=False))


if __name__ == '__main__': main()
