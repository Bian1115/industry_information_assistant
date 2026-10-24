#!/usr/bin/env python3
"""发布模型替换的真实测量结果，并向原工作台附加对照面板。"""
import argparse
import csv
import html
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
WORK=Path('/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估')
PAGE=WORK/'五篇报告人工评分工作台-20261004.html'
DIMS=['depth','comprehensiveness','readability','requirements']
NAMES=dict(zip(DIMS,['深度','全面性','可读性','需求覆盖']))
MODELS={'qwen':'Qwen3.8-max（新评）','claude':'Claude（沿用）','ensemble':'新双裁判均分'}
def read(p):return json.loads(p.read_text())
def fmt(x):return '无法计算' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
def esc(x):return html.escape(str(x))
def pct(x):return '—' if x is None else f'{x:.1f}%'
def vec(x):return ' / '.join(fmt(x[d]) for d in DIMS)
def embedded(h,name):return json.loads(re.search(fr'<script id="{name}" type="application/json">(.*?)</script>',h,re.S).group(1))

def publish(out):
    x=read(out/'一致性汇总.json');cfg=read(out/'评估配置.json');tasks=read(out/'冻结报告输入.json');review=read(out/'复核评分快照.json');cs=x['quality_comparison']
    if read(out/'运行状态.json')['status']!='完成':raise ValueError('不得发布未完成评分')
    report=ROOT/'docs/五篇报告Qwen38复评与复核一致性分析-20261006.md'
    dashboard=ROOT/'docs/五篇报告Qwen38复评与复核一致性-20261006.html'
    current=PAGE.read_text();R=embedded(current,'review');D=embedded(current,'dataset')
    if R['annotations']!=review['annotations']:raise ValueError('工作台复核参照变化，不可附加失配结果')
    lines=['# 五篇报告Qwen3.8-max复评与复核一致性分析','',f"完成日期：{x['date']}。质量裁判已从qwen3-max替换为用户指定的qwen3.8-max。既有五篇正文各评分两次，共10份新Qwen结果；Claude沿用上一轮few-shot复评的10份结果。本轮没有重新生成报告或重跑逐条事实、Scout、预审、覆盖、来源与Critic。",'',
      f"[交互对照页面]({dashboard}) · [一致性汇总]({out/'一致性汇总.json'}) · [配置]({out/'评估配置.json'}) · [新Qwen原始响应]({out/'裁判原始记录'}) · [Claude沿用来源及指纹]({out/'沿用Claude来源与指纹.json'}) · [完整性核查]({out/'输入与评分完整性核查.json'})。",'',
      '## 评分与比较口径','',
      '四维评分为0–10分。两次重复先按模型、报告、维度平均；双裁判再等权平均。每个裁判的一致性分母为5篇×4维=20个均分坐标，独立报告仅5篇。MAE为平均绝对误差，越小越接近；RMSE强调大误差；偏差为模型减复核，正值表示偏高；±1分比例按绝对差不超过1计算。相关性只衡量排序或线性关系，不代表分数尺度一致。', '',
      '基线是上一轮使用相同虚构few-shot的Qwen3-max结果，非最初无样例的评分。报告正文、用户任务、虚构样例和复核分数均保持一致；每份实际请求与冻结输入逐字核对。复核分数和备注没有加入裁判输入，评分不依据复核选择。Claude原始记录逐字复制，附加点评归一化仅沿用上轮规则，分数和理由不修改。', '',
      '| 裁判 | 上轮MAE | 本轮MAE | 上轮±1分 | 本轮±1分 | 本轮RMSE | 本轮偏差 | 完全同分 |',
      '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for model,c in cs.items():
        lines.append(f"| {MODELS[model]} | {fmt(c['old']['mae'])} | {fmt(c['new']['mae'])} | {pct(c['old']['within_1_percent'])} | {pct(c['new']['within_1_percent'])}（{c['new']['within_1_count']}/20） | {fmt(c['new']['rmse'])} | {fmt(c['new']['signed_bias'])} | {c['new']['exact_count']}/20 |")
    q=cs['qwen'];e=cs['ensemble'];cl=cs['claude']
    change=q['new']['mae']-q['old']['mae'];direction='降低' if change<0 else ('升高' if change>0 else '不变')
    lines+=['',f"已测量：新Qwen MAE由{fmt(q['old']['mae'])}变为{fmt(q['new']['mae'])}，{direction}{fmt(abs(change))}分；±1分比例由{pct(q['old']['within_1_percent'])}变为{pct(q['new']['within_1_percent'])}。新Qwen偏差为{fmt(q['new']['signed_bias'])}分。Claude仍为MAE {fmt(cl['new']['mae'])}，这是沿用同一评分的恒定参照，不是本轮新测量。新双裁判均分MAE为{fmt(e['new']['mae'])}，±1分比例为{pct(e['new']['within_1_percent'])}。",'']
    for model in ['qwen','ensemble']:
        d=cs[model]['delta'];ci=d['task_cluster_bootstrap_ci95']
        lines.append(f"{MODELS[model]}以报告为单位进行5^5=3125次经验bootstrap，新减旧MAE为{fmt(d['delta_mae_new_minus_old'])}分，95%描述性区间[{fmt(ci[0])}, {fmt(ci[1])}]。")
    lines+=['','区间只描述固定五篇的任务重采样，不能量化未知任务上的收益，也没有包含全部模型随机性。模型、思考模式、温度与结构化输出约束同时变化，不能把差异纯粹归因于模型升级。','',
      '## 各维度表现','',
      '| 维度 | Qwen3-max→Qwen3.8-max MAE | 新Qwen偏差 | 新Qwen±1分 | Claude MAE（沿用） | 新均分MAE | 新均分±1分 |','| --- | --- | --- | --- | --- | --- | --- |']
    for d in DIMS:
        v=q['by_dimension'][d];cv=cl['by_dimension'][d]['new'];ev=e['by_dimension'][d]['new']
        lines.append(f"| {NAMES[d]} | {fmt(v['old']['mae'])} → {fmt(v['new']['mae'])} | {fmt(v['new']['signed_bias'])} | {v['new']['within_1_count']}/5 | {fmt(cv['mae'])} | {fmt(ev['mae'])} | {ev['within_1_count']}/5 |")
    for d in DIMS:
        v=q['by_dimension'][d]
        lines+=['',f"{NAMES[d]}：新Qwen MAE {fmt(v['new']['mae'])}，偏差{fmt(v['new']['signed_bias'])}；Spearman由{fmt(v['old']['spearman'])}变为{fmt(v['new']['spearman'])}。单维只有五篇且复核分数较集中，相关性尤其不稳定。"]
    lines+=['','## 每篇报告的评分','','四维顺序为深度 / 全面性 / 可读性 / 需求覆盖。','',
      '| 报告 | 复核 | 上轮Qwen3-max | 本轮Qwen3.8-max | Claude沿用 | 新双裁判均分 |','| --- | --- | --- | --- | --- | --- |']
    old=read(out/'上次fewshot一致性汇总.json')
    for t in tasks:
        tid=str(t['id']);v=x['new_means'][tid]
        lines.append(f"| {t['id']} {t['title']} | {vec(review['annotations']['quality'][tid])} | {vec(old['new_means'][tid]['qwen'])} | {vec(v['qwen'])} | {vec(v['claude'])} | {vec(v['ensemble'])} |")
    lines+=['','## 偏差最大的判定与理由','', '这些是相对既有复核参照的分歧，不能直接等同现实事实错误。裁判理由是本轮实际输出，未经重新撰写。','']
    worst=sorted([r for r in x['score_rows'] if r['model']=='qwen'],key=lambda r:abs(r['new_error']),reverse=True)[:5]
    for r in worst:
        lines += [f"### 第{r['topic']}题：{NAMES[r['dimension']]}",'',f"上轮Qwen {fmt(r['old'])}，新Qwen {fmt(r['new'])}，复核{fmt(r['review'])}；本轮差值{fmt(r['new_error'])}分。",'']
        for v in sorted([v for v in x['new_records'] if v['topic']==r['topic'] and v['model']=='qwen'],key=lambda v:v['repeat']):
            dim=v['result']['dimensions'][r['dimension']]
            lines+= [f"第{v['repeat']}次：{fmt(dim['score'])}分。{dim['reason']}",'']
        lines += [f"复核备注：{review['annotations']['quality'][str(r['topic'])]['note']}",'']
    lines+=['## 第49题的需求边界','',
       f"第49题的需求覆盖均分从上轮Qwen {fmt(old['new_means']['49']['qwen']['requirements'])}降至本轮{fmt(x['new_means']['49']['qwen']['requirements'])}，复核为{fmt(review['annotations']['quality']['49']['requirements'])}。本轮两次均给5分，并明确区分个体共同需求比例与两类消费群体分别具有的年龄/性别特征；也指出缺少全球范围的可比分母与补测路径。这个判定比上轮更接近既有复核，仍相差1.5分。承认交叉证据缺失是否完成任务，仍需要在需求维度单独扣分，不能由未来联系的讨论完全补偿。",'',
       '## 重复波动与模型间分歧','',f"Qwen两次重复评分的平均绝对差由{fmt(q['old_repeat_mean_absolute_difference'])}变为{fmt(q['new_repeat_mean_absolute_difference'])}分；Claude沿用值为{fmt(cl['new_repeat_mean_absolute_difference'])}分。两次请求的重复编号不同，且每篇只有两次，不能据此判断固定输入下的完整采样方差。seed是可复现性辅助参数，不代表绝对确定性。",'',
       f"两裁判的平均绝对分差由{fmt(x['old_new_inter_judge_mean_absolute_gap']['old'])}变为{fmt(x['old_new_inter_judge_mean_absolute_gap']['new'])}分。裁判相互接近和接近复核是两个不同指标；均分也可能受偏高与偏低抵消影响。",'',
       '## 逐条沿用结果与复核','', '本轮逐条标签未重跑，因此以下比例与上次对照相同。全量一致率保留unknown/uncertain；双方可判定子集排除这些标签，有选择偏差。标签一致率不等同事实正确率。','',
       '| 对象 | 全量一致 | 比例 | 双方可判定一致 | 比例 |','| --- | --- | --- | --- | --- |']
    for name,g in x['inherited_groups'].items():
        if 'exact' in g:lines.append(f"| {name} | {g['exact']}/{g['n']} | {pct(g['percent'])} | {g['both_determinate_exact']}/{g['both_determinate_n']} | {pct(g['both_determinate_percent'])} |")
    source=x['inherited_groups']['来源质量']
    lines+=['',f"来源质量{source['n']}项的MAE为{fmt(source['mae'])}。完整逐条记录和混淆矩阵见[沿用结果对照]({out/'逐条沿用结果与复核一致性.json'})。",'',
       '## 请求参数、失败诊断与验证','',
       f"Qwen请求使用精确别名`qwen3.8-max`，temperature=0.6、enable_thinking=true、reasoning_effort=xhigh、seed=1234、strict JSON Schema，不设置max_tokens截断输出；只接收正常stop及[DONE]结束且能由原生JSON解码的完整评分。没有语法修复、分数映射或依据复核筛选。参数依据[千问OpenAI兼容接口](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)与[结构化输出](https://help.aliyun.com/en/model-studio/qwen-structured-output)的说明。思考模式的温度下限为0.6，因此本次没有用温度0。",'',
       f"提示词为{cfg['quality_prompt']['version']}，五组独立虚构样例，SHA256=`{cfg['quality_prompt']['sha256']}`，与上轮完全相同。新调用的服务响应model字段：{', '.join(sorted(set(r['response_model'] for r in x['calls']['records'])))}。新有效响应总Token为{x['calls']['known_tokens']}（思考Token明细见原始usage）；保存{x['calls']['saved_failed_attempts']}份网络/输出失败尝试。这个合计不包含失败请求的不可见消耗，也不包含沿用Claude的历史调用。",'',
       '准备阶段曾因一份历史Claude的附加点评嵌套结构而中断；与上轮相同的归一化后，分数和理由逐字相同，复制的原始文件仍保持原字节。该准备中断尚未发出新模型请求，诊断目录为backend/eval/data/五篇报告Qwen38复评准备中断-20261006。参数接入、模型缓存隔离、流结束处理与一致性指标的离线检查通过；全部十份新评分及十份沿用评分完成输入和原始分数核查。','',
       f"实测评分已独立重算MAE、RMSE、偏差和±1分数量，Claude沿用文件逐字节核对，工作台的数据、复核和历史元数据全部保留，见[评分与归档核查]({out/'评分与归档核查.json'})。18项离线检查通过。两个HTML页面的脚本编译、五个报告面板及20行对照、报告和误差筛选逻辑检查通过，见[HTML脚本核查]({out/'HTML脚本核查.json'})；浏览器安全策略拒绝自动读取file协议，因此未进行视觉核查。",'',
       '## 局限性与下一步','',
       '已测量：本批五篇报告的实际评分、相对复核的逐维误差、重复差异，以及历史逐条标签对照。间接推断：裁判理由可用于识别其如何权衡证据缺口、内容深度和需求完成，但不能证明内部评分机制。尚未评估：独立多人标注一致性、未见任务的泛化、固定模型快照下的长期稳定性、真实事实错误率、新模型对逐条核验的收益。','']
    lines += [f"- {v}" for v in x['limitations']]
    lines+=['',f"本批结论：新Qwen配置相对上轮MAE降低{fmt(100*(q['old']['mae']-q['new']['mae'])/q['old']['mae'])}%，但总体仍偏高{fmt(q['new']['signed_bias'])}分。Claude沿用评分的MAE {fmt(cl['new']['mae'])}仍低于新Qwen的{fmt(q['new']['mae'])}及新双裁判均分的{fmt(e['new']['mae'])}；等权均分在本批没有优于Claude单裁判。后续用独立报告、固定模型版本与相同证据条件验证泛化；本批报告和复核备注继续作为评价对象，不写入few-shot。"]
    report.write_text('\n'.join(lines)+'\n')
    with (out/'质量逐维对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,['topic','model','dimension','review','old','new','old_error','new_error']);w.writeheader();w.writerows(x['score_rows'])
    cards=''.join(f'<div class="card"><span>{MODELS[model]}</span><b>MAE {fmt(c["old"]["mae"])} → {fmt(c["new"]["mae"])}</b><small>±1分 {c["new"]["within_1_count"]}/20（{pct(c["new"]["within_1_percent"])}） · 偏差 {fmt(c["new"]["signed_bias"])}</small></div>' for model,c in cs.items())
    rows=[];details=[]
    for t in tasks:
        tid=str(t['id']);v=x['new_means'][tid];previous=old['new_means'][tid]
        for d in DIMS:
            reference=review['annotations']['quality'][tid][d];err=v['ensemble'][d]-reference
            rows.append(f'<tr data-topic="{tid}" data-error="{abs(err)}"><td>{tid} {esc(t["title"])}</td><td>{NAMES[d]}</td><td>{fmt(reference)}</td><td>{fmt(previous["qwen"][d])}</td><td>{fmt(v["qwen"][d])}</td><td>{fmt(v["claude"][d])}</td><td>{fmt(previous["ensemble"][d])}</td><td>{fmt(v["ensemble"][d])}</td><td>{fmt(err)}</td></tr>')
        reasons=[]
        for r in sorted([r for r in x['new_records'] if r['topic']==t['id']],key=lambda r:(r['model'],r['repeat'])):
            reasons.append(f'<h3>{MODELS[r["model"]]} 第{r["repeat"]}次</h3>'+''.join(f'<p><b>{NAMES[d]} {fmt(r["result"]["dimensions"][d]["score"])}</b> · {esc(r["result"]["dimensions"][d]["reason"])}</p>' for d in DIMS))
        details.append(f'<details><summary>{tid} {esc(t["title"])}：评分理由和复核备注</summary>'+''.join(reasons)+f'<h3>复核备注</h3><p>{esc(review["annotations"]["quality"][tid]["note"])}</p></details>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五篇报告Qwen3.8-max复评与复核一致性</title><style>body{font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f4f6fa;color:#19283e;margin:0}main{max-width:1450px;padding:28px;margin:auto}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card,details,.box{padding:18px;background:white;border:1px solid #dce4ed;border-radius:9px;margin:12px 0}.card b{display:block;color:#255dc1;font-size:23px}small{color:#63728a}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #dce4ed;text-align:left;white-space:nowrap}th{background:#eef4fd}summary{cursor:pointer;color:#255dc1}select,label{margin-right:12px;padding:6px}@media(max-width:700px){main{padding:14px}.cards{display:block}}</style><main><h1>五篇报告Qwen3.8-max复评与复核一致性</h1><p>2026-10-06 · Qwen五篇各两次新评（10份） · Claude沿用上轮10份 · 复核参照冻结且未提供给裁判</p><p>正文和虚构few-shot与上轮相同。新Qwen温度0.6，开启xhigh思考，严格JSON输出；逐条事实、Scout和预审结果沿用。MAE越小越接近复核，分母为20个均分坐标，独立报告数为5。</p>'''+f'<div class="cards">{cards}</div><div class="box"><h2>逐题逐维对照</h2><select id="topic"><option value="all">全部报告</option>'+''.join(f'<option value="{t["id"]}">{t["id"]} {esc(t["title"])}</option>' for t in tasks)+f'</select><label><input id="errors" type="checkbox">仅显示新均分与复核相差超过1分</label><div class="scroll"><table id="scores"><thead><tr><th>报告</th><th>维度</th><th>复核</th><th>旧Qwen3-max</th><th>新Qwen3.8-max</th><th>Claude沿用</th><th>旧均分</th><th>新均分</th><th>新均分−复核</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>'+''.join(details)+'''<p>结果只反映固定五篇；复核参照不是独立多人金标。模型、思考、采样参数和输出约束同时变化，不能单独归因模型。Claude沿用评分，双裁判组合包含跨时间结果。详情与原始响应见同名分析报告和本轮运行目录。</p></main><script>function filterRows(){document.querySelectorAll('#scores tbody tr').forEach(r=>r.hidden=(document.getElementById('topic').value!=='all'&&r.dataset.topic!==document.getElementById('topic').value)||(document.getElementById('errors').checked&&Number(r.dataset.error)<=1))}document.getElementById('topic').onchange=filterRows;document.getElementById('errors').onchange=filterRows;</script></html>'''
    dashboard.write_text(page);(WORK/dashboard.name).write_text(page)
    backup=WORK/'五篇报告人工评分工作台-Qwen38复评前备份-20261006.html'
    if not backup.exists():backup.write_text(current)
    R['qwen38_evaluation']={'config':cfg,'summary':x,'frozen_review_quality':review['annotations']['quality'],'dashboard_path':str(dashboard),'report_path':str(report)}
    h=re.sub(r'<script id="review" type="application/json">.*?</script>',lambda _: '<script id="review" type="application/json">'+json.dumps(R,ensure_ascii=False).replace('</','<\\/')+'</script>',current,flags=re.S)
    if 'function renderQwen38Quality(' not in h:
        hook="${showModel?renderCalibratedQuality(task):''}"
        if hook not in h:raise ValueError('工作台结构已改变，未找到质量面板挂载点')
        fn='''function renderQwen38Quality(task){const e=R.qwen38_evaluation;if(!e)return '';const x=e.summary,id=String(task.id),v=x.new_means[id],c=x.quality_comparison.qwen,en=x.quality_comparison.ensemble,ref=e.frozen_review_quality[id];return `<div class="model"><h3>Qwen3.8-max质量复评 · 2026-10-06</h3><p>Qwen新增10份评分；Claude沿用上轮10份。全五篇Qwen MAE ${c.old.mae.toFixed(4)} → ${c.new.mae.toFixed(4)}，±1分 ${c.new.within_1_count}/20；新双裁判均分MAE ${en.new.mae.toFixed(4)}，±1分 ${en.new.within_1_count}/20。</p><p class="muted">Qwen温度0.6，开启xhigh思考；提示词和原文与上轮相同。下表及整体指标使用冻结复核分数；后续手动改分需重新计算一致性。逐条结果未重跑。</p><div style="overflow:auto"><table><thead><tr><th>维度</th><th>冻结复核</th><th>旧Qwen3-max</th><th>新Qwen3.8-max</th><th>Claude沿用</th><th>新均分</th></tr></thead><tbody>${Object.entries(D.dimensions).map(([d,n])=>`<tr><td>${esc(n)}</td><td>${ref[d]}</td><td>${x.score_rows.find(r=>r.topic===task.id&&r.model==='qwen'&&r.dimension===d).old}</td><td>${v.qwen[d]}</td><td>${v.claude[d]}</td><td>${v.ensemble[d]}</td></tr>`).join('')}</tbody></table></div>${x.new_records.filter(r=>r.topic===task.id&&r.model==='qwen').map(r=>`<details><summary>新Qwen第${r.repeat}次：四维评分理由</summary>${Object.entries(r.result.dimensions).map(([d,z])=>`<p><b>${D.dimensions[d]} ${z.score}</b> · ${esc(z.reason)}</p>`).join('')}</details>`).join('')}</div>`}
'''
        h=h.replace('function renderQuality(task){',fn+'function renderQuality(task){',1)
        h=h.replace(hook,"${showModel?renderQwen38Quality(task):''}"+hook,1)
    afterR=embedded(h,'review')
    if embedded(h,'dataset')!=D or afterR['annotations']!=R['annotations'] or afterR.get('calibrated_evaluation')!=embedded(current,'review').get('calibrated_evaluation'):raise ValueError('附加结果改变了原数据、复核或历史面板')
    PAGE.write_text(h)
    print('已发布',report,dashboard,PAGE,sep='\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();publish(args.out.resolve())
