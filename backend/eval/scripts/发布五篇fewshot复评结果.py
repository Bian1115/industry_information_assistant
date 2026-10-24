#!/usr/bin/env python3
"""将已完成的冻结复评生成说明、对照页面及工作台附加面板。"""
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
NAMES={'depth':'深度','comprehensiveness':'全面性','readability':'可读性','requirements':'需求覆盖'}
MODELS={'qwen':'Qwen','claude':'Claude','ensemble':'双裁判均分'}
def read(p):return json.loads(p.read_text())
def fmt(v):return '无法计算' if v is None else f'{v:.4f}'.rstrip('0').rstrip('.')
def esc(v):return html.escape(str(v))
def percent(v):return '—' if v is None else f'{v:.1f}%'

def publish(out):
    x=read(out/'一致性汇总.json');cfg=read(out/'评估配置.json');tasks=read(out/'冻结报告输入.json');review=read(out/'复核评分快照.json');compare=x['quality_comparison']
    report=ROOT/'docs/五篇报告fewshot复评与复核一致性分析-20261006.md'
    dashboard=ROOT/'docs/五篇报告fewshot复评与复核一致性-20261006.html'
    def vector(values):return ' / '.join(fmt(values[d]) for d in DIMS)
    lines=['# 五篇报告fewshot复评与复核一致性分析','',
      f"评估完成日期：{x['date']}。五篇报告正文指纹与复核参照一致；使用五组独立虚构few-shot样例，Qwen3-max和Claude-opus-5各对每篇报告评分两次，共{len(x['new_records'])}次真实质量评分。原文、旧评分和既有复核均保留。逐条事实、Scout、预审、覆盖、来源与Critic结果按用户选择沿用旧版，没有重新调用裁判。",'',
      f"[交互对照页面]({dashboard})；[一致性汇总]({out/'一致性汇总.json'})；[评估配置]({out/'评估配置.json'})；[裁判原始记录]({out/'裁判原始记录'})。复核参照冻结于[复核评分快照]({out/'复核评分快照.json'})，没有放入任何裁判请求。",'',
      '## 四维质量与复核的接近程度','',
      '两次重复先在每个模型、报告、维度内平均，再与复核分数比较。每个裁判或双裁判均分的分母是5篇×4维共20个坐标，重复调用不增加独立报告数。MAE是平均绝对误差，越小越接近；偏差为模型减复核，正值表示整体偏高。相差不超过1分的比例作为可解释的一致性指标。','',
      '| 裁判 | 旧MAE | 新MAE | 旧±1分一致率 | 新±1分一致率 | 新RMSE | 新偏差 |','| --- | --- | --- | --- | --- | --- | --- |']
    for model,c in compare.items():
        lines.append(f"| {MODELS[model]} | {fmt(c['old']['mae'])} | {fmt(c['new']['mae'])} | {percent(c['old']['within_1_percent'])} | {percent(c['new']['within_1_percent'])} | {fmt(c['new']['rmse'])} | {fmt(c['new']['signed_bias'])} |")
    ensemble=compare['ensemble'];delta=ensemble['delta'];direction='下降' if delta['delta_mae_new_minus_old']<0 else ('上升' if delta['delta_mae_new_minus_old']>0 else '未变')
    ci=delta['task_cluster_bootstrap_ci95']
    lines+=['',f"本批未验证稳定的一致性收益：Qwen总体MAE未变，Claude由{fmt(compare['claude']['old']['mae'])}升至{fmt(compare['claude']['new']['mae'])}；双裁判均分虽然MAE略降，±1分一致率却由{percent(ensemble['old']['within_1_percent'])}降至{percent(ensemble['new']['within_1_percent'])}。全面性与需求覆盖的新±1分一致率各只有20%（1/5）。",'']
    lines+=['',f"双裁判均分MAE从{fmt(ensemble['old']['mae'])}变为{fmt(ensemble['new']['mae'])}，{direction}{fmt(abs(delta['delta_mae_new_minus_old']))}分。以任务为单位穷举3125次经验bootstrap，新减旧的MAE差区间为[{fmt(ci[0])}, {fmt(ci[1])}]。这只描述固定五篇的重采样变化；模型随机性、调用时间和服务别名版本也可能改变，不能把差异全部因果归于few-shot。",'',
      f"Qwen新评分的平均偏差为{fmt(compare['qwen']['new']['signed_bias'])}分，Claude为{fmt(compare['claude']['new']['signed_bias'])}分。均分的接近程度必须与各裁判误差同时判断：两个裁判分别偏高和偏低时，均分可能通过偏差抵消接近复核，并不表示每个裁判都完成校准。",'',
      '## 按维度比较','',
      '| 维度 | Qwen旧→新MAE | Claude旧→新MAE | 双裁判旧→新MAE | 双裁判新±1分一致率 |','| --- | --- | --- | --- | --- |']
    for d in DIMS:
        values=[compare[m]['by_dimension'][d] for m in ['qwen','claude','ensemble']]
        lines.append(f"| {NAMES[d]} | "+' | '.join(f"{fmt(v['old']['mae'])} → {fmt(v['new']['mae'])}" for v in values)+f" | {percent(values[-1]['new']['within_1_percent'])} |")
    req49=[r['result']['dimensions']['requirements']['score'] for r in x['new_records'] if r['topic']==49 and r['model']=='qwen']
    lines+=['',f"第49题是明确的评分边界分歧：实际任务要求共同需求比重，报告未给出个体共现比例。Qwen两次需求覆盖分别给{fmt(req49[0])}和{fmt(req49[1])}分，复核为{fmt(review['annotations']['quality']['49']['requirements'])}分。一次理由认为承认证据不足已满足条件性回答，另一次认为核心量化目标未完成。诚实承认缺失能减少编造，却不自动完成用户的量化目标；只有任务明确允许转为缺口及验证方案时，这一替代交付才符合要求。这个边界在虚构样例中已经说明，本批结果显示模型仍未稳定遵循，不能仅靠加入few-shot宣布校准成功。",'']
    lines+=['','## 按报告比较','','四维顺序为深度、全面性、可读性、需求覆盖。','',
      '| 报告 | 复核 | 新Qwen均分 | 新Claude均分 | 新双裁判均分 |','| --- | --- | --- | --- | --- |']
    for t in tasks:
        tid=str(t['id']);means=x['new_means'][tid]
        lines.append(f"| {t['id']} {t['title']} | {vector(review['annotations']['quality'][tid])} | {vector(means['qwen'])} | {vector(means['claude'])} | {vector(means['ensemble'])} |")
    lines+=['','## 重复波动与裁判间差异','']
    for model in ['qwen','claude']:
        c=compare[model];lines.append(f"{MODELS[model]}两次重复的平均绝对差由{fmt(c['old_repeat_mean_absolute_difference'])}变为{fmt(c['new_repeat_mean_absolute_difference'])}分。")
    gap=x['old_new_inter_judge_mean_absolute_gap'];lines+=['',f"两个裁判在20个均分坐标上的平均绝对分差由{fmt(gap['old'])}变为{fmt(gap['new'])}分。裁判相互接近不等于与复核接近，应同时查看上面的MAE。每模型仅两次重复，不能据此确定稳定性分布。",'',
      '## 仍有差异的质量判定','']
    worst=sorted([r for r in x['score_rows'] if r['model']!='ensemble'],key=lambda r:abs(r['new_error']),reverse=True)[:5]
    for r in worst:
        t=next(t for t in tasks if t['id']==r['topic']);records=[v for v in x['new_records'] if v['topic']==r['topic'] and v['model']==r['model']]
        reasons='；'.join(v['result']['dimensions'][r['dimension']]['reason'].replace('\n',' ') for v in records)
        lines+= [f"### 第{r['topic']}题 {MODELS[r['model']]} {NAMES[r['dimension']]}",'',f"新均分{fmt(r['new'])}，复核{fmt(r['review'])}，相差{fmt(r['new_error'])}分。裁判理由：{reasons}",'',f"复核备注：{review['annotations']['quality'][str(r['topic'])]['note']}",'']
    lines+=['## 沿用逐条结果与复核的标签一致性','',
      '以下是历史标签的对照，未使用新few-shot重新评分，不能称为few-shot带来的逐条准确率改善。全量比例保留无法判定标签；双方可判定的比例排除任一方unknown或uncertain，属于经过选择的子集。它们衡量标签一致，不衡量现实事实真值。','',
      '| 对象 | 全量一致 | 比例 | 双方可判定一致 | 比例 | 复核无法判定数 |','| --- | --- | --- | --- | --- | --- |']
    for name,g in x['inherited_groups'].items():
        if 'exact' not in g:continue
        lines.append(f"| {name} | {g['exact']}/{g['n']} | {percent(g['percent'])} | {g['both_determinate_exact']}/{g['both_determinate_n']} | {percent(g['both_determinate_percent'])} | {g['reference_unknown']} |")
    source=x['inherited_groups']['来源质量'];lines+=['',f"来源质量189项，旧评分相对复核MAE为{fmt(source['mae'])}，完全同分{source['exact_count']}/189，相差不超过1分{source['within_1_count']}/189。",'',
      '全部适用论断的比例包含218条没有显式引用的共同无支持标记，有引用的741条更能反映来源语义判断差异。预审错误判断的低全量一致率，主要涉及旧裁判倾向给无错误而复核保留无法判定：应结合混淆矩阵，不能把双方可判定子集的高比例当作对全部样本的可靠结论。首轮检出只比较4条复核确认错误，样本很小且有重复错误，不能外推整篇召回率。','',
      f"完整逐条对照和混淆矩阵见[逐条沿用结果与复核一致性]({out/'逐条沿用结果与复核一致性.json'})。",'',
      '## 配置与局限性','',
      f"提示词版本为{cfg['quality_prompt']['version']}，SHA256为`{cfg['quality_prompt']['sha256']}`。Qwen温度0.3，Claude使用服务默认值，与旧版一致；Claude并发上限为1。20份有效评分中{x['calls']['format_normalized_results']}份做过附加点评结构归一化，{x['calls']['malformed_metadata_recovered_quality_results']}份在附加输出格式异常后提取完整四维，{x['calls']['transport_recovered_quality_results']}份在网络流中断后恢复完整四维对象。另保存{x['calls']['saved_failed_attempts']}份[失败尝试诊断]({out/'失败尝试'})；20份有效评分不是HTTP请求总次数。已知有效响应token合计{x['calls']['known_tokens']}，有{x['calls']['unknown_usage_calls']}次完整token用量不可得；失败或中断请求的服务端消耗不在此合计内。本次不估计价格。",'',
      '已测量：20份真实四维评分、原文与参照指纹、质量分数误差、重复差异及历史逐条标签对照。指标、输出校验与流结束处理的10项离线测试通过。部分Claude响应将附加点评嵌套进dimensions，仅将已知点评字段归一至顶层；分数和理由未改动。运行中新增网络流恢复规则：网络读取超时或附加输出格式异常时，仅当四个维度对象能用原生JSON解码完整读取、且通过评分校验时接收，附加点评或结束事件可缺失，不使用JSON语法修复补全分数或理由；缺少任何完整维度仍重试。采用首个合格结果，不依据复核分数选择，原始响应和中断标记均保留。间接推断：分歧理由反映裁判对证据、深度和需求完成的要求不同。尚未评估：独立多人标注一致性、未见任务上的校准泛化、真实事实错误率和few-shot对逐条判断的收益。既有复核不是独立多人金标，且复核读过更完整的来源与外部核查；质量裁判仅看与旧版相同的报告正文，证据条件不同。','',
      '接收器补上message_stop与[DONE]结束标记处理，并设置600秒接收总时限及文本进度快照；这是源码已确认的等待风险修复，缺少此前挂起时的完整事件日志，不能认定所有超时都由此造成。结束事件定义见[Claude官方流式文档](https://platform.claude.com/docs/en/build-with-claude/streaming)。','',
      '保持本批真实报告作为评价对象，不把其文字或复核理由加入few-shot。后续要验证校准收益，应使用独立任务、固定模型版本和重复条件，再测与独立标注的误差。']
    report.write_text('\n'.join(lines)+'\n')
    fields=['topic','model','dimension','review','old','new','old_error','new_error']
    with (out/'质量逐维对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(x['score_rows'])
    # Standalone table dashboard; source text and actual model reasons remain inspectable.
    cards=''.join(f"<div class=card><span>{MODELS[m]}平均绝对误差</span><b>{fmt(c['old']['mae'])} → {fmt(c['new']['mae'])}</b><small>新±1分一致率 {percent(c['new']['within_1_percent'])}（{c['new']['within_1_count']}/20）</small></div>" for m,c in compare.items())
    table=[]
    for t in tasks:
        for d in DIMS:
            r=next(r for r in x['score_rows'] if r['topic']==t['id'] and r['model']=='ensemble' and r['dimension']==d);newmeans=x['new_means'][str(t['id'])]
            oldtask=next(v for v in read(out/'旧质量评分.json') if v['id']==t['id'])
            table.append(f"<tr data-topic='{t['id']}' data-error='{abs(r['new_error'])}'><td>{t['id']} {esc(t['title'])}</td><td>{NAMES[d]}</td><td>{fmt(r['review'])}</td><td>{fmt(oldtask['models']['qwen']['dimensions'][d])}</td><td>{fmt(oldtask['models']['claude']['dimensions'][d])}</td><td>{fmt(r['old'])}</td><td>{fmt(newmeans['qwen'][d])}</td><td>{fmt(newmeans['claude'][d])}</td><td>{fmt(r['new'])}</td><td>{fmt(r['new_error'])}</td></tr>")
    details=[]
    for t in tasks:
        pieces=[]
        for v in x['new_records']:
            if v['topic']!=t['id']:continue
            pieces.append(f"<h3>{MODELS[v['model']]} 第{v['repeat']}次</h3>"+''.join(f"<p><b>{NAMES[d]} {fmt(v['result']['dimensions'][d]['score'])}</b> · {esc(v['result']['dimensions'][d]['reason'])}</p>" for d in DIMS))
        details.append(f"<details><summary>第{t['id']}题 {esc(t['title'])}：新裁判理由与复核备注</summary>{''.join(pieces)}<h3>复核备注</h3><p>{esc(review['annotations']['quality'][str(t['id'])]['note'])}</p></details>")
    inherited=''.join(f"<tr><td>{esc(name)}</td><td>{g['exact']}/{g['n']}</td><td>{percent(g['percent'])}</td><td>{g['both_determinate_exact']}/{g['both_determinate_n']}</td><td>{percent(g['both_determinate_percent'])}</td></tr>" for name,g in x['inherited_groups'].items() if 'exact' in g)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五篇报告few-shot复评与复核一致性</title><style>body{{font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#19283e;background:#f4f6fa;margin:0}}main{{max-width:1400px;margin:auto;padding:28px}}h1{{font-size:26px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card,details,.box{{background:white;padding:18px;border:1px solid #dce4ed;border-radius:9px;margin:12px 0}}.card b{{display:block;font-size:25px;color:#255dc1}}small{{color:#63728a}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:10px;border-bottom:1px solid #dce4ed;text-align:left;white-space:nowrap}}th{{background:#eef4fd}}select,label{{padding:7px;margin-right:10px}}summary{{cursor:pointer;color:#255dc1}}@media(max-width:700px){{.cards{{display:block}}main{{padding:14px}}}}</style><main><h1>五篇报告few-shot复评与复核一致性</h1><p>2026-10-06 · 两模型各两次，共20次真实质量评分 · 复核分数未提供给裁判 · 原报告与历史结果保留</p><p><small>{x["calls"]["transport_recovered_quality_results"]}份评分的附加点评因网络中断未完整返回；四维分数和理由完整，原始记录与恢复说明均已保存。</small></p><div class="cards">{cards}</div><p>MAE越小越接近复核。这里的复核参照不是独立多人金标；重复和维度不增加独立报告数。模型评分均分不是事实正确率。</p><div class=box><h2>逐题逐维新旧分数</h2><select id=topic><option value=all>全部报告</option>{''.join(f'<option value="{t["id"]}">{t["id"]} {esc(t["title"])}</option>' for t in tasks)}</select><label><input id=errors type=checkbox>仅显示新双裁判均分与复核相差超过1分</label><div class=scroll><table id=scores><thead><tr><th>报告</th><th>维度</th><th>复核</th><th>旧Qwen</th><th>旧Claude</th><th>旧均分</th><th>新Qwen</th><th>新Claude</th><th>新均分</th><th>新均分−复核</th></tr></thead><tbody>{''.join(table)}</tbody></table></div></div>{''.join(details)}<div class=box><h2>沿用逐条结果的标签一致性</h2><p>这些标签未重跑，不能把差异解释为few-shot改善。全量含无法判定；双方可判定子集有选择偏差。</p><div class=scroll><table><thead><tr><th>对象</th><th>全量一致</th><th>比例</th><th>双方可判定一致</th><th>比例</th></tr></thead><tbody>{inherited}</tbody></table></div></div><p>详细误差、混淆矩阵、原始响应与配置保存在本次独立运行目录。</p></main><script>function filterRows(){{document.querySelectorAll('#scores tbody tr').forEach(r=>r.hidden=(document.getElementById('topic').value!=='all'&&r.dataset.topic!==document.getElementById('topic').value)||(document.getElementById('errors').checked&&Number(r.dataset.error)<=1))}}document.getElementById('topic').onchange=filterRows;document.getElementById('errors').onchange=filterRows;</script></html>'''
    dashboard.write_text(page)
    (WORK/dashboard.name).write_text(page)
    # Append results to the current workbench without changing its dataset or annotations.
    h=PAGE.read_text();backup=WORK/'五篇报告人工评分工作台-fewshot复评前备份-20261006.html'
    if not backup.exists():backup.write_text(h)
    R=json.loads(re.search(r'<script id="review" type="application/json">(.*?)</script>',h,re.S).group(1))
    if R['annotations']!=review['annotations']:raise ValueError('工作台已被修改，不能覆盖并附加失配结果')
    R['calibrated_evaluation']={'config':cfg,'summary':x,'dashboard_path':str(dashboard),'report_path':str(report)}
    h=re.sub(r'<script id="review" type="application/json">.*?</script>',lambda m:'<script id="review" type="application/json">'+json.dumps(R,ensure_ascii=False).replace('</','<\\/')+'</script>',h,flags=re.S)
    if 'function renderCalibratedQuality(' not in h:
        fn='''function renderCalibratedQuality(task){const e=R.calibrated_evaluation;if(!e)return '';const x=e.summary,c=x.quality_comparison.ensemble,means=x.new_means[String(task.id)];return `<div class="model"><h3>few-shot质量复评 · 2026-10-06</h3><p>全五篇双裁判均分MAE ${c.old.mae.toFixed(3)} → ${c.new.mae.toFixed(3)}；±1分一致 ${c.new.within_1_count}/20。复核是既有评分参照，逐条结果未重跑。</p><div style="overflow:auto"><table><thead><tr><th>维度</th><th>本页复核</th><th>新Qwen</th><th>新Claude</th><th>新均分</th></tr></thead><tbody>${Object.entries(D.dimensions).map(([d,n])=>`<tr><td>${esc(n)}</td><td>${esc(A.quality[topic]?.[d])}</td><td>${means.qwen[d]}</td><td>${means.claude[d]}</td><td>${means.ensemble[d]}</td></tr>`).join('')}</tbody></table></div><p class="muted">整体一致性基于冻结复核分数；后续手动改分不会自动重新计算。${x.calls.transport_recovered_quality_results}份评分的附加点评因网络中断缺失，四维分数与理由完整。旧模型初评保留在下方。</p>${x.new_records.filter(r=>r.topic===topic).map(r=>`<details><summary>${r.model} 第${r.repeat}次评分理由</summary>${Object.entries(r.result.dimensions).map(([d,v])=>`<p><b>${D.dimensions[d]} ${v.score}</b> · ${esc(v.reason)}</p>`).join('')}</details>`).join('')}</div>`}
'''
        h=h.replace('function renderQuality(task){',fn+'function renderQuality(task){')
        h=h.replace('${showModel?`<div class="model"><b>模型初评', '${showModel?renderCalibratedQuality(task):\'\'}${showModel?`<div class="model"><b>模型初评')
    h=h.replace('c.old.mae.toFixed(3)','c.old.mae.toFixed(4)').replace('c.new.mae.toFixed(3)','c.new.mae.toFixed(4)')
    PAGE.write_text(h)
    print('已发布',report,dashboard,PAGE,sep='\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();publish(a.out.resolve())
