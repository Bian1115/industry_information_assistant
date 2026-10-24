"""聚合冻结轨迹与初评标签，发布可离线使用的HTML及中文分析。"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATE = '2026-10-08'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def dump(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def rate(n, d):
    return {'n': n, 'd': d, 'value': n / d if d else None}

def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * q
    return values[int(x)] + (values[math.ceil(x)] - values[int(x)]) * (x - int(x))

def timing(rows):
    values = [r['duration_ms'] for r in rows if r.get('duration_ms') is not None]
    return {'n': len(values), 'p50_ms': quantile(values, .5), 'p95_ms': quantile(values, .95), 'mean_ms': statistics.mean(values) if values else None, 'max_ms': max(values) if values else None}

def ndcg(grades):
    def dcg(values):
        return sum((2 ** value - 1) / math.log2(i + 2) for i, value in enumerate(values))
    ideal = dcg(sorted(grades, reverse=True))
    return dcg(grades) / ideal if ideal else 0.0

def summarize(queries, pages, facets, batches):
    known_p5, lower, upper, ndcgs = [], [], [], []
    unknown_slots = 0
    for query in queries:
        top = query['results'][:5]
        grades = [r.get('assessment', {}).get('grade') for r in top]
        good = sum(g is not None and g >= 2 for g in grades)
        unknown = sum(g is None for g in grades)
        unknown_slots += unknown
        lower.append(good / 5)
        upper.append((good + unknown) / 5)
        if not unknown:
            known_p5.append(good / 5)
        allgrades = [r.get('assessment', {}).get('grade') for r in query['results'][:10]]
        if all(g is not None for g in allgrades):
            ndcgs.append(ndcg(allgrades))
    counts = Counter(p['assessment']['label'] for p in pages)
    eligible = [p for p in pages if p['assessment']['label'] in {'usable', 'partial', 'abnormal'}]
    tp = sum(p['assessment']['label'] == 'abnormal' and p['program_rejected'] for p in eligible)
    fn = sum(p['assessment']['label'] == 'abnormal' and not p['program_rejected'] for p in eligible)
    fp = sum(p['assessment']['label'] in {'usable', 'partial'} and p['program_rejected'] for p in eligible)
    tn = sum(p['assessment']['label'] in {'usable', 'partial'} and not p['program_rejected'] for p in eligible)
    pending_pages = counts['not_evaluated']
    pending_facets = sum(f['assessment'].get('status') == 'not_evaluated' for f in facets)
    ai_usable = rate(counts['usable'] + counts['partial'], len(pages))
    ai_strict = rate(counts['usable'], len(pages))
    coverage = rate(sum(f['assessment'].get('status') == 'full' and bool(f['assessment'].get('fact_ids')) for f in facets), len(facets))
    if pending_pages:
        ai_usable.update(value=None, pending=pending_pages)
        ai_strict.update(value=None, pending=pending_pages)
    if pending_facets:
        coverage.update(value=None, pending=pending_facets)
    return {
        'queries': len(queries), 'pages': len(pages), 'batches': len(batches),
        'search_complete': rate(sum(q['complete'] for q in queries), len(queries)),
        'search_nonempty': rate(sum(bool(q['results']) for q in queries), len(queries)),
        'p5': statistics.mean(known_p5) if len(known_p5) == len(queries) and queries else None,
        'p5_known_queries': len(known_p5), 'p5_unknown_slots': unknown_slots,
        'p5_lower': statistics.mean(lower) if lower else None,
        'p5_upper': statistics.mean(upper) if upper else None,
        'ndcg_returned_candidates': statistics.mean(ndcgs) if ndcgs else None, 'ndcg_queries': len(ndcgs),
        'fetch_success': rate(sum(p['attempt'].get('fetch_status') == 'success' for p in pages), len(pages)),
        'program_admitted': rate(sum(p['program_admitted'] for p in pages), len(pages)),
        'ai_usable': ai_usable,
        'ai_usable_strict': ai_strict,
        'page_labels': dict(counts), 'sampled_pages': sum(not p['sample_is_full'] and bool(p['content']) for p in pages),
        'content_judged': rate(sum(p['assessment']['label'] not in {'unavailable', 'not_evaluated'} for p in pages), len(pages)),
        'coverage': coverage,
        'coverage_labels': dict(Counter(f['assessment'].get('status') for f in facets)),
        'noise_detector_vs_ai': {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'eligible_pages': len(eligible), 'excluded_pages': len(pages) - len(eligible), 'precision': rate(tp, tp + fp), 'recall': rate(tp, tp + fn), 'false_reject': rate(fp, fp + tn)},
        'search_timing': timing(queries), 'read_batch_timing': timing(batches),
        'failures': dict(Counter(p['attempt'].get('failure_category') or '无抓取失败分类' for p in pages if p['attempt'].get('fetch_status') != 'success')),
        'unmeasured': {'retrieval_recall_at_k': '缺少相关文档池金标', 'canonical_ndcg_at10': '只有返回候选标签，缺少完整独立候选池', 'body_extraction_f1': '轨迹保存抽取正文，缺少原始HTML/PDF与人工正文金标', 'per_page_p95': '读取工具只记录批次起止，没有每URL开始时间', 'real_cost': '没有对齐真实账单和冻结单价'},
    }

def percent(value):
    if isinstance(value, dict):
        v = value['value']
        return f'{100*v:.2f}% ({value["n"]}/{value["d"]})' if v is not None else '不可用'
    return f'{100*value:.2f}%' if value is not None else '不可用'

def ms(value):
    return f'{value/1000:.3f}s' if value is not None else '不可用'

def publish(out):
    out = Path(out)
    config = read(out / '评估配置.json')
    records = read(out / '初评完成记录.json') if (out / '初评完成记录.json').exists() else []
    labels = {kind: {} for kind in ['search', 'body', 'coverage']}
    for record in records:
        for row in record['items']:
            if row['id'] in labels[record['kind']]:
                raise ValueError('重复初评标签')
            labels[record['kind']][row['id']] = row
    correction_path = out / 'Codex复核修正.json'
    correction = read(correction_path) if correction_path.exists() else {'changes': []}
    initial_coverage = Counter(a['status'] for a in labels['coverage'].values())
    initial_body = Counter(a['label'] for a in labels['body'].values())
    for change in correction['changes']:
        kind, ident = change['kind'], change['id']
        current = labels[kind][ident]
        if current != change['before']:
            raise ValueError('复核修正的原标签与本次原始初评不匹配')
        labels[kind][ident] = {**current, **change['after'], 'initial_assessment': current, 'reviewer': change['reviewer'], 'review_date': change['review_date']}
    queries, pages, batches, tasks = [read(out / name) for name in ['冻结检索轨迹.json', '冻结网页读取材料.json', '冻结读取批次.json', '冻结需求与事实证据.json']]
    facets = []
    for query in queries:
        for row in query['results'][:10]:
            row['assessment'] = labels['search'].get(row['id'], {'grade': None, 'reason': '裁判尚未完成', 'quote': ''})
    for page in pages:
        page['assessment'] = labels['body'].get(page['id'], {'label': 'unavailable' if not page['content'].strip() else 'not_evaluated', 'category': 'empty' if not page['content'].strip() else 'unknown', 'reason': '轨迹未取得正文，保留在有效正文率分母' if not page['content'].strip() else '裁判尚未完成', 'quote': ''})
        page['program_rejected'] = page['attempt'].get('failure_category') == 'body_quality_rejected' or (page['attempt'].get('body_quality') or {}).get('status') == 'rejected'
    for task in tasks:
        for row in task['facets']:
            facets.append({**row, 'topic': task['topic'], 'assessment': labels['coverage'].get(row['id'], {'status': 'not_evaluated', 'reason': '裁判尚未完成', 'fact_ids': [], 'quote': ''})})
    per_topic = []
    for task in tasks:
        topic = task['topic']
        metrics = summarize(*[[row for row in data if row['topic'] == topic] for data in [queries, pages, facets, batches]])
        per_topic.append({'topic': topic, 'title': task['title'], 'folder': task['folder'], 'budget': task['search_budget'], 'metrics': metrics})
    summary = summarize(queries, pages, facets, batches)
    summary['codex_review'] = {'changes': len(correction['changes']), 'initial_coverage_labels': dict(initial_coverage), 'initial_body_labels': dict(initial_body), 'final_coverage_labels': summary['coverage_labels'], 'final_body_labels': summary['page_labels'], 'human_gold': False}
    summary['tasks'] = len(tasks)
    summary['topic_macro_p5'] = statistics.mean(p['metrics']['p5'] for p in per_topic) if all(p['metrics']['p5'] is not None for p in per_topic) else None
    summary['topic_macro_coverage'] = statistics.mean(p['metrics']['coverage']['value'] for p in per_topic) if all(p['metrics']['coverage']['value'] is not None for p in per_topic) else None
    summary['topic_macro_ai_usable'] = statistics.mean(p['metrics']['ai_usable']['value'] for p in per_topic) if all(p['metrics']['ai_usable']['value'] is not None for p in per_topic) else None
    usage = Counter()
    judge_times = []
    successful_packets = [path for path in (out / '裁判原始记录').glob('*.json') if read(path).get('result') is not None]
    failed_packets = list((out / '失败尝试').glob('*.json'))
    for path in successful_packets + failed_packets:
        packet = read(path)
        usage.update({k: v for k, v in packet.get('usage', {}).items() if type(v) in {int, float}})
        if packet.get('elapsed_s') is not None:
            judge_times.append(packet['elapsed_s'])
    summary['new_judge_usage'] = {'successful_response_packets': len(successful_packets), 'failed_attempt_packets': len(failed_packets), 'offline_recovered_jobs': sum(bool(r.get('recovered_from')) for r in records), 'usage_fields': dict(usage), 'sum_request_wall_seconds': sum(judge_times), 'notice': '累计已记录成功响应和失败尝试，不重复累计离线恢复引用的响应；可能存在中断未记录的请求用量，真实费用未估算'}
    dump(out / '指标汇总.json', {'configuration': config, 'overall': summary, 'per_topic': per_topic})
    dump(out / '逐条检索相关性初评.json', queries)
    dump(out / '逐页正文质量初评.json', pages)
    dump(out / '逐项需求覆盖初评.json', facets)
    fingerprint = hashlib.sha256(json.dumps({'sources': read(out / '来源文件指纹.json'), 'records': records, 'correction': correction}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    dataset = {'dataset_id': fingerprint, 'date': DATE, 'configuration': config, 'summary': summary, 'per_topic': per_topic, 'queries': queries, 'pages': pages, 'facets': facets, 'tasks': tasks, 'batches': batches, 'run_dir': str(out), 'codex_review': correction}
    dump(out / '评分页面数据.json', dataset)
    template = (ROOT / 'backend/eval/templates/开发集检索读取评估页面.html').read_text(encoding='utf-8')
    embedded = json.dumps(dataset, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = ROOT / 'docs' / f'开发集网络检索与网页读取评估页面-{out.name.split("评估-")[-1]}.html'
    page.write_text(template.replace('__DATASET__', embedded), encoding='utf-8')
    report = ROOT / 'docs' / f'开发集网络检索与网页读取评估结果及分析-{out.name.split("评估-")[-1]}.md'
    state = read(out / '运行状态.json')
    p5 = percent(summary['p5']) if summary['p5'] is not None else f'{percent(summary["p5_lower"])}—{percent(summary["p5_upper"])}（未知标签区间）'
    lines = ['# 开发集网络检索与网页读取评估结果及分析', '', f'评估日期：{DATE}，Asia/Singapore。状态：{state["status"]}。', '', '## 对象、数据与配置口径', '',
        '评估通用研究中文五题35、31、46、16、49的已完成历史轨迹。采用此前五篇报告冻结输入中的逐题目录；五题均有完整工具历史和已提交检查点。31题来自旧预算运行，其他四题取消累计token上限，不能作为公平预算消融。选择已完成运行不代表全部开发集端到端成功率。', '',
        f'共{len(queries)}次搜索、{len(batches)}次正文读取批次、{len(pages)}个按题去重URL、{len(facets)}项独立冻结需求。搜索按单次调用URL去重；重复查询调用保留；读取采用每题URL最后一次已提交尝试，并保留失败URL在分母。', '',
        '新增初评使用qwen3-max、temperature=0，基于保存的搜索标题/摘要、抽取正文和Scout事实原句。判定以冻结资料为准，不重新联网取证、不依赖在线程序标签作为金标。AI初评不是人工金标；部分正文可用性来自抽样，语义支持性也可能误判。页面预填全部已完成标签，保留理由和原文，允许独立复核、持久保存和导出。', '',
        f'原始结果目录：[{out.name}]({out})。', f'交互页面：[{page.name}]({page})。', '',
        '## 原始机器可读结果', '',
        *[f'- [{name}]({out/name})' for name in ['评估配置.json', '来源文件指纹.json', '冻结检索轨迹.json', '冻结网页读取材料.json', '冻结读取批次.json', '冻结需求与事实证据.json', '初评完成记录.json', '逐条检索相关性初评.json', '逐页正文质量初评.json', '逐项需求覆盖初评.json', '指标汇总.json', '运行状态.json']],
        '- 裁判原始记录目录保存完整提示、响应、usage、耗时及错误；失败尝试目录保留格式或网络失败，不能以0补缺失。', '',
        '## 复评与Codex核查记录', '',
        '复评日期：2026-10-08，同一冻结运行内补充。第一阶段严格引句匹配拒绝了一些结构完整的分类响应；后改为分类与引句分别审计，原提示和材料不变。未定位措辞不显示为原文引句，原始失败尝试保留；部分完整响应在本地重新审计后恢复，没有新增模型调用。详细变更见评估配置中的reassessment_change及引句严格校验阶段状态.json。', '',
        f'Codex主代理浏览36项需求初评，回查明显年龄、时间与成本范围误判的保存原句，并核查摘要分类口径。保存{len(correction["changes"])}项修正，原始Qwen响应不改写；这仍是AI复核，不是人工金标。修正记录：[Codex复核修正.json]({correction_path})。', '',
        f'需求full计数：原Qwen {initial_coverage["full"]}/{len(facets)} → 复核后 {summary["coverage"]["n"]}/{len(facets)}。正文原始分类：{json.dumps(dict(initial_body), ensure_ascii=False)}；最终分类见下表。相邻年龄段不能替代20—30岁，过去规划不能充当未来十年；只有调度/维护功能的原句不能自动证明应急成本或收支闭环。', '',
        '## 核心指标', '', '| 指标 | 结果 | 口径 |', '| --- | --- | --- |',
        f'| AI初评 Precision@5 | {p5} | grade≥2为相关；不足5条按缺失位置不相关；跨77调用平均 |',
        f'| 返回候选内nDCG@10 | {summary["ndcg_returned_candidates"]:.4f} | 仅{summary["ndcg_queries"]}个可判调用；IDCG由各自返回候选构成，不能称正式全网nDCG |' if summary['ndcg_returned_candidates'] is not None else '| 返回候选内nDCG@10 | 未完成 | 无可判调用 |',
        f'| AI充分证据需求覆盖率 | {percent(summary["coverage"])} | full且有真实事实原句；实际搜索13—18次，均低于20次逻辑预算 |',
        f'| AI可用正文率（包含partial） | {percent(summary["ai_usable"])} | 分母为所有按题去重待读取URL |',
        f'| AI正常正文率（不含partial） | {percent(summary["ai_usable_strict"])} | usable不声称全文完整，抽样页另列 |',
        f'| 程序抓取成功率 | {percent(summary["fetch_success"])} | 最新fetch_status=success，不能替代人工可用性 |',
        f'| 程序正文准入率 | {percent(summary["program_admitted"])} | success且passed/partial |',
        f'| 工具完整返回率 | {percent(summary["search_complete"])} | 有dispatch/raw_result/output，不等于上游API成功率 |',
        f'| 非空搜索率 | {percent(summary["search_nonempty"])} | 至少1个唯一URL |',
        f'| 搜索工具P50/P95 | {ms(summary["search_timing"]["p50_ms"])} / {ms(summary["search_timing"]["p95_ms"])} | {summary["search_timing"]["n"]}次调用，包含限流/重试与未知缓存状态 |',
        f'| 正文读取批次P50/P95 | {ms(summary["read_batch_timing"]["p50_ms"])} / {ms(summary["read_batch_timing"]["p95_ms"])} | {summary["read_batch_timing"]["n"]}批次，包含并发与混合缓存，不是单页延迟 |',
        '', '以下检测结果以本次AI页面标签为参照，尚未完成独立人工校准：', '',
        f'- 异常页判定Precision：{percent(summary["noise_detector_vs_ai"]["precision"])}；Recall：{percent(summary["noise_detector_vs_ai"]["recall"])}；正常页误拒率：{percent(summary["noise_detector_vs_ai"]["false_reject"])}。',
        f'- 混淆计数：{json.dumps(summary["noise_detector_vs_ai"], ensure_ascii=False)}。',
        f'- 正文标签分布：{json.dumps(summary["page_labels"], ensure_ascii=False)}。超长抽样页面{summary["sampled_pages"]}个。', '',
        '## 逐题分析', '', '| 题目 | 搜索/读取URL | AI P@5 | AI充分覆盖 | AI可用正文 | 程序准入 | 搜索P95 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for item in per_topic:
        m = item['metrics']
        lines.append(f'| {item["topic"]} {item["title"]} | {m["queries"]}/{m["pages"]} | {percent(m["p5"])} | {percent(m["coverage"])} | {percent(m["ai_usable"])} | {percent(m["program_admitted"])} | {ms(m["search_timing"]["p95_ms"])} |')
    for task in tasks:
        topic = task['topic']
        lines.extend(['', f'### {topic} {task["title"]}', '', f'原始轨迹：[{task["folder"]}]({task["folder"]})。实际搜索预算使用{task["search_budget"]["used"]}/{task["search_budget"]["limit"]}。'])
        for facet in [f for f in facets if f['topic'] == topic]:
            a = facet['assessment']
            lines.append(f'- {facet["id"]}，{facet["text"]}：{a["status"]}。{a["reason"]} 支持事实：{", ".join(a.get("fact_ids", [])) or "无"}。')
    lines.extend(['', '## 失败样例与可定位证据', '', '失败URL保留在分母；没有读取正文的页面不能被当作确定异常页金标。以下列出低相关候选、程序放行的异常正文和程序拒绝的正常正文；完整记录见HTML逐条页。'])
    low = sorted([(q, r) for q in queries for r in q['results'][:5] if r['assessment'].get('grade') in {0, 1}], key=lambda pair: pair[1]['assessment']['grade'])
    for query, row in low[:10]:
        lines.append(f'- 检索{row["id"]}：查询“{query["query"]}”，[{row.get("title") or row["url"]}]({row["url"]})。等级{row["assessment"]["grade"]}：{row["assessment"]["reason"]}')
    cases = [p for p in pages if p['assessment']['label'] == 'abnormal' and not p['program_rejected']] + [p for p in pages if p['assessment']['label'] in {'usable', 'partial'} and p['program_rejected']]
    for page_row in cases[:12]:
        lines.append(f'- 正文{page_row["id"]}：[{page_row["title"] or page_row["url"]}]({page_row["url"]})。AI={page_row["assessment"]["label"]}，程序拒绝={page_row["program_rejected"]}。{page_row["assessment"]["reason"]}')
    lines.extend(['', '## 局限性与测量边界', '',
        '已测量：冻结轨迹的搜索/读取数量、工具阶段耗时、程序读取状态和失效类别，以及本次模型对候选相关性、保存正文可用性和事实原句需求覆盖的判定。派生质量指标是AI初评，不能称人工准确率。', '',
        '间接推断：低相关查询、正文污染与需求缺口可能影响报告可信度，页面按原文展示，可作为复核和改进入口；本次没有以新报告或消融证明改进收益。', '',
        '尚未评估：完整相关文档池Recall@K与正式nDCG@10、人工正文抽取Precision/Recall/F1、单页读取P95、真实账单成本、广泛站点泛化和当前版本端到端效果。没有把缺失值填为0。正文抽取F1需要源HTML/PDF与独立正文金标，不能把抽取器自己的输出同时当作金标。', '',
        '该语义裁判使用与历史研究可能相同的模型家族，存在偏好与共享错误；没有双独立人工标注。读取超长正文抽样不能证明完整正文没有尾部污染。样本仅5题，多次查询和URL并非完全独立，P95为该轨迹的描述性值；不同预算、并发、缓存及重试影响延迟，不具备生产SLA解释。', '',
        f'本次裁判统计：{json.dumps(summary["new_judge_usage"], ensure_ascii=False)}。所有usage与单次耗时保留，未对真实费用进行估算。', '',
        '## 结论、简历口径与下一步', '',
        f'对五题历史轨迹进行可追溯的离线评估，覆盖{len(queries)}次搜索与{len(pages)}个待读取URL；AI初评Precision@5为{p5}，充分证据需求覆盖为{percent(summary["coverage"])}，可用正文率为{percent(summary["ai_usable"])}。这些是当前固定材料的测量，不是相较基线的提升幅度。', '',
        '简历可写“构建基于真实开发集轨迹的检索与正文读取评估工作台，支持逐候选相关性、证据覆盖、正文可用性、失效分类和调用时延分析”。引用质量数字必须注明AI初评、5题样本及分母；时延明确区分搜索工具和读取批次。尚无对照，不写提升百分点或成本降低比例。', '',
        '优先复核页面中的低相关候选、程序放行异常正文和缺失需求；由独立评审对跨题样本标注后校准AI分数。下一轮保存原始HTML/PDF与每URL起止计时，以同搜索服务、同预算、同页面建立查询策略和读取策略的配对对照，再补正式nDCG/Recall、正文F1、单页P95及账单。', ''])
    checks_path = out / '最终评估与交付核对.json'
    if checks_path.exists():
        checks = read(checks_path)
        lines.extend(['## 交付核对', '',
            f'核对记录：[{checks_path.name}]({checks_path})。核对时间：{checks.get("checked_at", "未记录")}。', '',
            f'来源文件指纹、标签完整性、引句定位和聚合指标核对：{checks.get("status", "未完成")}。核对结果不等于语义标签准确性金标。', '',
            f'页面交互核对：[{"页面交互核对结果.json"}]({out / "页面交互核对结果.json"})。桌面及手机截图均已检查；筛选、标签修改持久保存、完整评分导出与导入、错误数据指纹拒绝通过，未发现页面脚本错误或整页横向溢出。', '',
            '页面在本地浏览器进行核对，人工标签修改仅写入该浏览器的localStorage；导出JSON才形成可共享复核快照。', ''])
        if checks.get('supplemental_ui_check'):
            first_ui = out / checks['supplemental_ui_check']['first_attempt_file']
            lines.extend([
                f'同日补充机制核对：一次错误指纹检查在异步文件读取完成前即断言提示，未通过记录保留于[{first_ui.name}]({first_ui})。核对脚本增加等待拒绝提示后，复核全部通过；页面功能和评估标签未修改。', ''])
    report.write_text('\n'.join(lines), encoding='utf-8')
    dump(out / '交付文件记录.json', {'dataset_id': fingerprint, 'report': str(report), 'html': str(page), 'html_bytes': page.stat().st_size, 'report_bytes': report.stat().st_size, 'original_sources_preserved': True})
    print(json.dumps({'report': str(report), 'page': str(page), 'p5': summary['p5'], 'coverage': summary['coverage'], 'usable': summary['ai_usable'], 'state': state['status']}, ensure_ascii=False), flush=True)
