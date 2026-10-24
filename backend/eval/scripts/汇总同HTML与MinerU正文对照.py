#!/usr/bin/env python3
"""汇总同HTML、统一盲评口径的最多100份配对结果与完整采集边界。"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('paired_capture', Path(__file__).with_name('对照评估289网址与MinerU.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
read, dump, sha, now = common.read, common.dump, common.sha, common.now


def rate(n, d):
    return {'n': n, 'd': d, 'value': n / d if d else None}


def pct(value):
    return f'{100 * value:.2f}%'


def summary(rows):
    labels = Counter(r['assessment']['label'] for r in rows)
    usable = labels['usable'] + labels['partial']
    return {'count': len(rows), 'usable': usable, 'labels': dict(labels),
            'usable_rate': rate(usable, len(rows)),
            'unknown': labels['unknown'],
            'possible_rate_interval': [usable / len(rows), (usable + labels['unknown']) / len(rows)],
            'unusable_categories': dict(Counter(r['assessment']['category'] for r in rows
                                               if r['assessment']['label'] == 'unusable'))}


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    return values[math.floor(k)] + (values[math.ceil(k)] - values[math.floor(k)]) * (k - math.floor(k))


def run(out):
    sample = read(out / '冻结100份HTML配对样本.json')
    assessed_path = out / '逐条正文配对复核后结果.json'
    if not assessed_path.exists():
        assessed_path = out / '逐条正文盲评结果.json'
    assessed = read(assessed_path)
    all_capture = common.records(out)
    lookup = {(r['page_id'], r['method']): r for r in assessed}
    comparisons = []
    for page in sample:
        pid = page['page_id']
        baseline, mineru = lookup[(pid, 'baseline')], lookup[(pid, 'mineru')]
        stored = read(out / 'MinerU结果' / pid / '抽取结果.json')
        assert baseline['shared_html_sha256'] == mineru['shared_html_sha256'] == stored['shared_html_sha256']
        assert sha(Path(page['shared_html_path']).read_bytes()) == page['shared_html_sha256']
        assert sha(Path(baseline['content_path']).read_text()) == baseline['content_sha256']
        assert sha(Path(mineru['content_path']).read_text()) == mineru['content_sha256']
        unknown = any(x['assessment']['label'] == 'unknown' for x in [baseline, mineru])
        transition = ('unknown' if unknown else
                      'improved' if not baseline['content_usable'] and mineru['content_usable'] else
                      'regressed' if baseline['content_usable'] and not mineru['content_usable'] else
                      'both_usable' if baseline['content_usable'] else 'both_unusable')
        comparisons.append({'page_id': pid, 'topic': page['topic'], 'url': page['url'], 'title': page['title'],
                            'shared_html_path': page['shared_html_path'],
                            'shared_html_sha256': page['shared_html_sha256'],
                            'historical_usable': page['historical_usable'], 'transition': transition,
                            'baseline': baseline, 'mineru': mineru,
                            'mineru_service_state': stored['service_state']})
    submitted = sum(len(read(p)['payload']['files']) for p in (out / 'MinerU服务批次').glob('*/提交意图.json'))
    assert submitted <= 100 and len(sample) <= 100
    assert len(assessed) == 2 * len(sample)
    methods = {name: summary([r for r in assessed if r['method'] == name]) for name in ['baseline', 'mineru']}
    transitions = Counter(r['transition'] for r in comparisons)
    per_topic = []
    for topic in common.TOPICS:
        rr = [r for r in comparisons if r['topic'] == topic]
        per_topic.append({'topic': topic, 'title': common.TOPICS[topic], 'sample_count': len(rr),
                          'baseline_usable': sum(r['baseline']['content_usable'] for r in rr),
                          'mineru_usable': sum(r['mineru']['content_usable'] for r in rr),
                          'transitions': dict(Counter(r['transition'] for r in rr))})
    usage = Counter()
    times = []
    response_models = Counter()
    for path in (out / '正文判定').glob('*.json'):
        packet = read(path)
        response_models[packet.get('actual_model', 'deterministic_empty_rule')] += 1
        usage.update({k: v for k, v in (packet.get('usage') or {}).items() if isinstance(v, (int, float))})
        if packet.get('elapsed_seconds') is not None:
            times.append(packet['elapsed_seconds'])
    config = read(out / '分层抽样配置.json')
    judge_config = read(out / '裁判模型配置.json')
    weighted = {}
    for method in methods:
        weighted[method] = sum(s['population'] / config['html_population'] *
            sum(r['content_usable'] for r in assessed if r['method'] == method and
                r['topic'] == s['topic'] and
                next(p['historical_usable'] for p in sample if p['page_id'] == r['page_id']) == s['historical_usable']) /
            s['sample'] for s in config['strata'] if s['sample'])
    result = {'date': '2026-10-17', 'finished_at': now(),
        'source_urls': 289, 'captured_urls': len(all_capture),
        'html_population': config['html_population'],
        'input_kinds': dict(Counter(r['input_kind'] for r in all_capture)),
        'sample_count': len(sample), 'api_file_submissions_including_pilot': submitted,
        'hard_cap': 100, 'paired_hash_verified': len(comparisons),
        'methods': methods, 'transitions': dict(transitions), 'per_topic': per_topic,
        'absolute_gain_percentage_points': 100 * (methods['mineru']['usable_rate']['value'] - methods['baseline']['usable_rate']['value']),
        'relative_gain': ((methods['mineru']['usable'] - methods['baseline']['usable']) / methods['baseline']['usable'])
                         if methods['baseline']['usable'] else None,
        'stratum_weighted_rates_exploratory': weighted,
        'unknown_outputs': sum(r['assessment']['label'] == 'unknown' for r in assessed),
        'quote_not_located_outputs': sum(r['assessment'].get('quote_status') == 'not_located' for r in assessed),
        'api_failed_outputs': sum(r['mineru_service_state'] != 'done' for r in comparisons),
        'judge': {'requested_model': judge_config['requested_model'],
                  'provider': judge_config['provider'], 'temperature': judge_config['temperature'],
                  'actual_models': dict(response_models),
                  'usage_success_packets': dict(usage), 'request_count': len(times),
                  'request_p50_seconds': quantile(times, .5), 'request_p95_seconds': quantile(times, .95),
                  'failed_attempt_files': len(list((out / '裁判失败尝试').glob('*.json')))},
        'scope': '同一100份HTML的当前读取方案与官方MinerU-HTML服务；不是全部289 URL的新方案可用率',
        'assessment_stage': '配对分歧一致性复核后' if '复核后' in assessed_path.name else '独立盲评初评',
        'assessment_input_path': str(assessed_path),
    }
    review_path = out / '配对分歧复核/Claude原始复核响应.json'
    if review_path.exists():
        review = read(review_path)
        result['judge']['paired_review_requests'] = 1
        result['judge']['paired_review_outputs'] = len(review['result']['items'])
        result['judge']['paired_review_model'] = review['actual_model']
        result['judge']['paired_review_usage'] = review.get('usage', {})
        result['judge']['paired_review_request_seconds'] = review.get('elapsed_seconds')
        assert review['actual_model'] == judge_config['requested_model']
    dump(out / '最终指标汇总.json', result)
    dump(out / '逐页配对结果.json', comparisons)
    dump(out / '改善与退化样例.json', [r for r in comparisons if r['transition'] in {'improved', 'regressed', 'unknown'}])
    shutil.copy2(__file__, out / '冻结汇总脚本.py')
    paths = [out / '冻结289网址.json', out / '冻结100份HTML配对样本.json', out / '分层抽样配置.json',
             out / '评估配置.json', out / '裁判模型配置.json',
             out / '同输入与提交额度完整性检查.json', out / '100份预算保护负向机制检查.json',
             out / '正文盲评提示词.txt', out / '匿名编号方案映射.json', out / '逐条正文盲评结果.json',
             out / '冻结执行脚本.py', out / '冻结MinerU上传脚本.py', out / '冻结盲评脚本.py', out / '冻结汇总脚本.py']
    if (out / '配对分歧复核/Claude原始复核响应.json').exists():
        paths += [assessed_path, out / '配对分歧复核/Claude原始复核响应.json',
                  out / '配对分歧复核/复核修正明细.json']
    dump(out / '最终输入文件指纹.json', [{'path': str(p), 'sha256': sha(p.read_bytes())} for p in paths])
    write_report(out, result, comparisons)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def write_report(out, result, comparisons):
    baseline, mineru = result['methods']['baseline'], result['methods']['mineru']
    n = result['sample_count']
    lines = [f'# 通用研究289网址归档与{n}份同HTML的MinerU正文对照结果及分析', '',
        f'评估日期：2026-10-17，Asia/Singapore。完成时间：{result["finished_at"]}。对象：固定289个历史URL中的本次HTML配对样本。', '',
        f'**同一{n}份HTML、统一盲评口径下，当前方案正文可用率为{baseline["usable"]}/{n}＝{pct(baseline["usable_rate"]["value"])}，官方MinerU-HTML服务为{mineru["usable"]}/{n}＝{pct(mineru["usable_rate"]["value"])}，净变化{result["absolute_gain_percentage_points"]:+.2f}个百分点。**', '',
        f'本次MinerU提交共{result["api_file_submissions_including_pilot"]}份HTML，包含1份接口验证样本；用户上限100份已遵守。正文可用是AI判定估计，不是人工金标或全文完整率。', '',
        '## 数据与配置口径', '',
        f'- 从原通用研究五题35、31、46、16、49的289个URL重新抓取；共归档{result["html_population"]}份HTML，其余类型为`{json.dumps(result["input_kinds"], ensure_ascii=False)}`。',
        '- URL仍有HTML响应不表示目标内容有效；404提示、验证码、登录、重定向错页也留在实际取得HTML的集合中，由统一语义标准判失败。',
        '- 按题号及历史正文可用性标签分层，按HTML总体比例用最大余数分配100份；层内固定随机种子，保留先前已提交的1份验证样本。抽样不依据本次两组效果，完整抽样记录已冻结。',
        '- 当前读取方案冻结了web_reader_service、web_document和结构渲染函数。静态链路被选择时保留静态HTML；原有浏览器回退被选择时保留同次DOM与raw_markdown。',
        '- 基线文本为现有方案机器质量门禁前、统一30000字符上限的抽取输出，保留程序接纳状态另查。MinerU也采用相同字符上限，不把程序接纳或长度当正文可用。',
        '- 两组共用归档HTML；上传的是本地文件，逐页检查两组输入SHA-256一致。静态响应另存原始字节；共享HTML只统一UTF-8与charset声明，基线也读取同一字节。',
        '- MinerU使用官方托管API，model_version=MinerU-HTML；返回main.html及full.md。官方没有在接口中公开精确权重版本及本体/回退标志，因此评价对象是官方服务整体。',
        f'- 全部待评输出打乱并匿名编号，统一使用{result["judge"]["requested_model"]}，provider={result["judge"]["provider"]}、temperature={result["judge"]["temperature"]}；裁判接收同一标准和完整待评文本，不接收方案身份。每个输出最终采用一份有效独立判定，包含全量原始响应、实际模型、用量和原句定位。',
        '- 可用分子为usable+partial：包括实质正文、真实摘要、实质预览；纯导航、登录/验证码、乱码、错页、目录、封面、元信息、下载链接、图链及销售空壳不可用。',
        '- 历史51.56%（149/289）与后续复核47.40%（137/289）采用历史页面与标签；不能与本次100份HTML相减宣称改进。此次比较只在相同100份内成立。', '',
        '同次运行口径补充日期：2026-10-17。用户限定MinerU总量100份（含接口验证1份），本次从231份HTML分层抽取100份。原qwen3-max返回model_not_found，qwen3.8-max被账户Arrearage拦截，未得到有效Qwen标签；用户明确同意使用Claude，因此两组全部统一改用既有Claude接口评分。原始失败记录与旧执行脚本保留。', '',
        '## 原始结果与可追溯路径', '',
        f'- 运行目录：[{out.name}]({out})。',
    ]
    for name in ['最终指标汇总.json', '逐页配对结果.json', '改善与退化样例.json', '冻结289网址.json',
                 '冻结100份HTML配对样本.json', '分层抽样配置.json', '评估配置.json', 'MinerU提交预算.json',
                 '逐条正文盲评结果.json', '匿名编号方案映射.json', '正文盲评提示词.txt', '最终输入文件指纹.json']:
        lines.append(f'- [{name}]({out / name})。')
    if result['assessment_stage'] == '配对分歧一致性复核后':
        for name in ['独立盲评初评指标汇总.json', '独立盲评初评逐页配对结果.json', '逐条正文配对复核后结果.json',
                     '配对分歧复核/Claude原始复核响应.json', '配对分歧复核/复核修正明细.json',
                     '配对分歧复核/摘要重复存在证据.json']:
            lines.append(f'- [{name}]({out / name})。')
    lines += ['', '逐页`逐页采集/<page_id>/`保留原始响应、静态HTML、浏览器DOM/Markdown、共享输入HTML、基线文本与程序状态；`MinerU结果/<page_id>/`保留服务ZIP、主体HTML、Markdown、待评正文和任务状态。`正文判定/`保留裁判原始响应。', '',
        '复现使用本目录的冻结脚本及冻结基线源码；抓取重放不得覆盖历史快照，已提交MinerU批次仅按身份继续查询；程序会拒绝第101份HTML提交。', '',
        '## 核心指标', '',
        '| 指标 | 当前方案 | MinerU-HTML官方服务 |', '| --- | --- | --- |']
    for label, key in [('可用正文或实质摘要', 'usable'), ('unknown未决', 'unknown')]:
        lines.append(f'| {label} | {baseline[key]}/{n} | {mineru[key]}/{n} |')
    lines.append(f'| 正文可用率 | {pct(baseline["usable_rate"]["value"])} | {pct(mineru["usable_rate"]["value"])} |')
    lines.append(f'| 标签分布 | `{json.dumps(baseline["labels"], ensure_ascii=False)}` | `{json.dumps(mineru["labels"], ensure_ascii=False)}` |')
    t = result['transitions']
    lines += ['', f'配对转移：新增可用{t.get("improved", 0)}份，退化{t.get("regressed", 0)}份，两组均可用{t.get("both_usable", 0)}份，两组均不可用{t.get("both_unusable", 0)}份，含未决{t.get("unknown", 0)}对。',
        f'成功验证{result["paired_hash_verified"]}对共享HTML哈希；MinerU服务失败{result["api_failed_outputs"]}份；裁判原句未连续定位{result["quote_not_located_outputs"]}条，未定位引句不作为原文证据。',
        f'按分层总体权重校正的探索性条件可用率：当前方案{pct(result["stratum_weighted_rates_exploratory"]["baseline"])}、MinerU {pct(result["stratum_weighted_rates_exploratory"]["mineru"])}；不外推为全部289 URL的已测量结果。', '',
        '## 同次配对一致性复核', '',
        '复核日期：2026-10-17。独立盲评初评为51/100对53/100；检查仅有的2个新增可用样例后发现同一内容被不一致判定。因此保持正文可用性定义不变，对这2对输出使用相同Claude进行并排匿名复核，补充统一目标年份、摘要存在性判定；初评文件及原响应完整保留。没有重新抓取或新增MinerU提交。',
        '- 31-page-019：两组都取得2027年会内容，预期标题却指向2024年；不能把相同页面改版在一组记错页、另一组豁免，并当作抽取器恢复正文。Claude复核按“同一URL滚动更新首页，以当前首页正文为目标”对两组均判usable，本报告主指标沿用此统一口径。若严格要求2024年目标正文，两组应同时扣除该页，分别为52/100与52/100，净差仍为0；见年份严格匹配敏感性分析.json。',
        '- 49-page-026：两组都保存了同一份人才报告的内容简介；已按连续原句定位证明关键摘要在基线中也存在，不能在一组称销售空壳、另一组称实质摘要。MinerU去掉大量导航属于清理变化，并不自动证明增加了新的实质内容。',
        '本报告主指标采用配对复核后的标签。配对复核与独立盲评是不同评审阶段，不能把初评的2个百分点直接当作效果提升。仍缺少人工金标和全量重复裁判一致性评估。', '',
        '## 逐题结果', '',
        '| 题号 | 样本 | 当前可用 | MinerU可用 | 净增 |', '| --- | --- | --- | --- | --- |']
    for r in result['per_topic']:
        lines.append(f'| {r["topic"]} | {r["sample_count"]} | {r["baseline_usable"]} | {r["mineru_usable"]} | {r["mineru_usable"] - r["baseline_usable"]:+d} |')
    lines += ['']
    for r in result['per_topic']:
        topic_rows = [x for x in comparisons if x['topic'] == r['topic']]
        parts = []
        for method, label in [('baseline', '当前方案'), ('mineru', 'MinerU')]:
            counts = Counter(x[method]['assessment']['category'] for x in topic_rows
                             if x[method]['assessment']['label'] == 'unusable')
            parts.append(label + '主要失败标签' + ('、'.join(f'{key} {n}份' for key, n in counts.most_common(2)) or '无'))
        lines.append(f'- 第{r["topic"]}题（{r["title"]}）：{r["sample_count"]}份中新增可用{r["transitions"].get("improved", 0)}份、退化{r["transitions"].get("regressed", 0)}份；' + '；'.join(parts) + '。')
    lines += ['', '逐题只用于定位差异；每题15至26份左右，不能把个别题的比例当稳定总体性能。', '',
        '## 失败类别', '', '| 类别 | 当前方案不可用数 | MinerU不可用数 |', '| --- | --- | --- |']
    cats = set(baseline['unusable_categories']) | set(mineru['unusable_categories'])
    for cat in sorted(cats, key=lambda k: -(baseline['unusable_categories'].get(k, 0) + mineru['unusable_categories'].get(k, 0))):
        lines.append(f'| {cat} | {baseline["unusable_categories"].get(cat, 0)} | {mineru["unusable_categories"].get(cat, 0)} |')
    lines += ['', '## 改善、退化与持续失败样例', '']
    for transition, label in [('improved', '新增可用'), ('regressed', '退化'), ('both_unusable', '持续失败')]:
        chosen = [r for r in comparisons if r['transition'] == transition][:5]
        for r in chosen:
            lines.append(f'- {label} `{r["page_id"]}`（{r["url"]}）：当前方案{r["baseline"]["assessment"]["label"]}，{r["baseline"]["assessment"]["reason"]}；MinerU {r["mineru"]["assessment"]["label"]}，{r["mineru"]["assessment"]["reason"]}。')
    lines += ['', '## 延迟、用量与接口限制', '',
        f'- MinerU提交{result["api_file_submissions_including_pilot"]}份，最多每分钟50文件，批次结果查询间隔10秒；包含验证样本，未通过重试扩展到第101份。',
        '- 云端排队、分批提交及ZIP下载时间与本地抽取耗时不能直接比速度，未测得可比的单页推理P50/P95。没有读取真实账单，不能宣称实际费用。',
        f'- 独立盲评有效请求{result["judge"]["request_count"]}次，失败尝试文件{result["judge"]["failed_attempt_files"]}个；这些有效响应记录的用量为`{json.dumps(result["judge"]["usage_success_packets"], ensure_ascii=False)}`。',
        f'- 另有同模型配对一致性复核{result["judge"].get("paired_review_requests", 0)}次，复核4份输出；独立保存的用量为`{json.dumps(result["judge"].get("paired_review_usage", {}), ensure_ascii=False)}`，没有重复计入上面的独立盲评用量。', '',
        '- 评分并发由2调整至4，已完成判定全部保留；中断时最多2个未完成请求未收到可归档响应，其实际用量未知，未计入成功请求用量，也没有用它们生成正文标签。调度记录见运行目录`评分并发调整记录.json`。',
        '- 批次44经直接连接补评仍长时间无响应，主动中断1个未完成请求后改用代理，保留完整成功响应及回退来源；该中断请求用量同样未知。原TLS失败及中断不作为正文失败标签，也未重新调用MinerU。',
        '- 额度与同输入检查见`同输入与提交额度完整性检查.json`；`100份预算保护负向机制检查.json`验证第101份会在网络调用前拒绝，没有新增HTML提交。', '',
        '## 局限性与证据边界', '',
        '**已测量**：289 URL采集分类、231份HTML归档数量（以机器汇总为准）、抽样100份的同输入哈希、两组输出、匿名裁判标签与配对转移、API提交总量。',
        '**间接推断**：正文语义可用性为AI裁判估计；分层加权仅估计本次可获得HTML集合的条件可用率，保留1份验证样本带来非纯随机限制。分类理由不能替代人工金标。',
        '**尚未评估**：另外未送MinerU的HTML、MinerU覆盖全部289 URL的端到端正文可用率、固定开源权重本体、字符级抽取Precision/Recall/F1、完整性、表格数字保真、人工一致性、真实费用及长期线上稳定性。',
        '抓取失败、PDF和Office没有送入MinerU-HTML；不能将它们当作MinerU模型抽取失败，也不能从分母删除后仍宣称测了289个URL的全量新方案。', '',
        '## 结论与下一步建议', '',
        f'本次统一口径、配对一致性复核后的结果为{baseline["usable"]}/{n}对{mineru["usable"]}/{n}。本样本未观察到新增可用正文收益；初评的2个百分点来自配对判定不一致。',
        'MinerU可使部分已有正文输出更精简，但去掉导航不等于恢复了原方案缺失的正文。49-page-026从12044字符减至1036字符，关键内容简介两组都已有；这是本页的清理收益示例，尚未做全样本噪声精确率、表格保真或下游报告质量评测。',
        '剩余共同失败以导航、验证码、元信息、乱码和空输出等为主，具体样例表明页面获取、目标定位和附件/图片内容缺失仍限制可用性。仅替换同HTML主体抽取器不足以证明能大幅提升当前URL正文可用率。',
        '建议优先人工复核配对差异与正文/摘要边界，进一步检查表格关键数值；如接入，保留现有抓取及文档格式路由，按正文质量选择候选，并使用独立留出页面验证。继续评测需重新获得用户额度授权，本次不再提交HTML。', '']
    common.report_path(out).write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())


if __name__ == '__main__':
    main()
