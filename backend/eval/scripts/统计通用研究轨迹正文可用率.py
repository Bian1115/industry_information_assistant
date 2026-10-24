#!/usr/bin/env python3
"""离线统计历史正文返回条目的可用率，保留来源与边界复核记录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'backend/eval/data/开发集网络检索与网页读取评估-20261008'
TOPICS = {35: '市政污水与雨污协同', 31: '中外博物馆教育', 46: '房地产可持续发展', 16: '非接触式感知算法', 49: '正畸与医美共同需求'}
# 本次主代理对保存原文的边界复核；不改写历史模型标签。
EXCLUSIONS = {
    '35-page-010': ('attachment_only', '关于印发《辽宁省农村污水处理设施运行维护管理办法（试行）》的通知.pdf', '目标文件只以PDF下载链接出现，未提取附件中的条文正文。'),
    '35-page-017': ('metadata_only', '全文阅读已结束，下载本文需要使用', '仅标题、文档元信息和积分下载提示，没有标准条文或实质预览。'),
    '35-page-018': ('cover_only', 'ICS35.240.50', '文本预览仅标准封面、分类号、标题及发布机构，没有指标条文。'),
    '35-page-019': ('metadata_only', '全文阅读已结束，下载本文需要使用', '仅标题、文档元信息和积分下载提示，没有标准条文或实质预览。'),
    '35-page-031': ('toc_only', '目         录', '保存文字仅为章节目录和页码，没有项目论述、成本、指标或可用正文摘要。'),
    '35-page-043': ('attachment_only', '温州市人民政府关于下达2016年温州市国民经济和社会发展计划的通知.doc', '目标内容只以DOC下载链接出现，未提取附件正文。'),
    '31-page-057': ('metadata_only', '全文阅读已结束，下载本文需要使用', '仅目标标题、界面控件及下载提示，没有日文文章正文或摘要。'),
    '46-page-014': ('toc_only', '第一章 “十四五”规划背景研究', '保存内容为报告章节目录，没有市场规模、政策论述或研究结论。'),
    '46-page-056': ('metadata_only', '【摘要】', '有摘要栏目名，但栏目正文为空，随后为下载提示及文献关联信息。'),
    '49-page-007': ('image_link_only', '2020年中国人口结构（分性别、年龄）', '目标人口结构图只保存图片URL和标题，未保存图中文字或数值；图片链接不算已提取图表内容。'),
    '49-page-027': ('toc_only', '第一章 行业综述', '保存内容为正畸行业报告章节目录，没有章节正文、结果或实质主题摘要。'),
    '49-page-029': ('sales_description_only', '《2026-2032年中国牙科用正畸材料行业市场深度调查及投资规划建议报告》', '保存内容为报告销售简介、目录、通用研究方法和数据来源说明，没有目标报告的行业论述或具体结果。'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def norm(url):
    p = urlsplit(url.strip())
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or '/', p.query, ''))


def rate(n, d):
    return {'n': n, 'd': d, 'value': n / d if d else None}


def percent(r):
    return f'{100 * r["value"]:.2f}% ({r["n"]}/{r["d"]})' if r['d'] else '无分母'


def summarize(rows):
    labels = Counter(r['original_label'] for r in rows)
    usable = sum(r['content_usable'] for r in rows)
    nonempty = sum(r['nonempty'] for r in rows)
    return {
        'returned_count': len(rows), 'nonempty_count': nonempty,
        'usable_count': usable, 'unusable_count': len(rows) - usable,
        'empty_count': len(rows) - nonempty,
        'nonempty_unusable_count': sum(r['nonempty'] and not r['content_usable'] for r in rows),
        'usable_rate': rate(usable, len(rows)), 'nonempty_body_usable_rate': rate(usable, nonempty),
        'nonempty_rate': rate(nonempty, len(rows)), 'original_labels': dict(labels),
        'original_broad_usable_rate': rate(labels['usable'] + labels['partial'], len(rows)),
        'boundary_excluded_count': sum(r['boundary_excluded'] for r in rows),
        'failure_categories': dict(Counter(r['category'] for r in rows if not r['content_usable'])),
    }


def make_report(out, date, config, result, corrections, batch_rows, topic_rows, unique_rows, report_path=None):
    overall, unique = result['overall'], result['per_topic_url_dedup']
    lines = [
        '# 通用研究开发集已有轨迹正文可用率结果及分析', '',
        f'评估日期：{date}，Asia/Singapore。评估对象：此前通用研究中文五题35、31、46、16、49的已完成轨迹。状态：离线统计完成。', '',
        f'**按实际返回条目统计，正文可用率为 {percent(overall["usable_rate"])}。** 分母保留空正文、失败占位、异常页和重复读取；分子要求已提取到目标正文，或含实质内容的真实摘要/预览。仅下载链接、文档封面、章节目录和图片URL不计可用。', '',
        '## 数据与配置口径', '',
        '- 使用2026-10-08冻结评估选定的五题轨迹，不重新搜索、不抓取网页、不新增模型调用。第31题为旧token预算运行，其余四题取消总token上限；这是选择的完成样本，不能代表全部开发集成功率或公平预算消融。',
        '- 77次 `v2_web_search` 返回标题/摘要/URL；正文来自随后57次 `v2_web_fulltext_read`。只对实际读取返回条目统计正文可用率，未选择读取的搜索候选不计正文失败。',
        f'- 读取工具输入累计包含{sum(b["input_candidate_count"] for b in batch_rows)}个候选条目，实际返回302条；输入候选与实际返回数量不同，不能将未返回候选直接计为正文抽取失败。',
        '- 主指标为 Σ可用正文条目 / Σ实际返回条目，对应“返回6条、5条可用→5/6”。同一URL跨批次返回多次，逐次计数；这是使用端实际收到的条目，包含缓存，不是独立网页样本。',
        '- 条件指标另以 `content.strip()` 非空的条目为分母，回答“已经有文本的页面中，有多少可用”。二者的分母不能混用。',
        '- 沿用Qwen3-max（temperature=0）初评及2026-10-08 Codex复核；本次复核partial及可疑目录/元信息边界，排除12个原来计为可用的URL。其余继承旧标签，未进行全量独立重判；结果是AI辅助估计，不是人工金标。',
        '- 可用性指目标内容是否被提取，不等于全文完整、与研究问题充分相关、事实真实或能回答所求比例。真实论文摘要可计可用；只有图表链接但没有数字/OCR不可计已提取图表。通用机构信息索引中的真实机构条目不等同于研究报告的章节目录。',
        f'- 已验证全部{overall["returned_count"]}个返回条目的URL及正文SHA-256与冻结页面一致，因此可按内容复用标签。', '',
        '## 原始结果及可追溯路径', '',
        f'本次原始结果目录：[{out.name}]({out})。', '',
    ]
    for name in ['评估配置.json', '来源文件指纹.json', '正文可用性复核记录.json', '逐条返回正文明细.jsonl', '逐批正文可用率.json', '逐题正文可用率.json', '按题去重正文明细.jsonl', '指标汇总.json']:
        lines.append(f'- [{name}]({out / name})')
    lines += ['', f'历史标签及正文原文：[逐页正文质量初评.json]({SOURCE / "逐页正文质量初评.json"})；历史报告：[开发集网络检索与网页读取评估结果及分析-20261008.md]({ROOT / "docs/开发集网络检索与网页读取评估结果及分析-20261008.md"})。逐条明细包含history_path、call_id、原始记录seq和批内位置，指向原始正文；来源文件指纹保留哈希。', '',
              '## 核心指标', '', '| 指标 | 结果 | 说明 |', '| --- | --- | --- |',
              f'| 返回条目正文可用率（主指标） | {percent(overall["usable_rate"])} | 按读取返回次数统计，含空值与失败 |',
              f'| 非空文本中的正文可用率 | {percent(overall["nonempty_body_usable_rate"])} | 分母只含已返回非空文本 |',
              f'| 非空正文返回率 | {percent(overall["nonempty_rate"])} | 有文字不等于可用 |',
              f'| 按题URL去重后的正文可用率 | {percent(unique["usable_rate"])} | 每题唯一URL，正文与最后冻结版本一致 |',
              f'| 旧宽松标签投影到返回条目 | {percent(overall["original_broad_usable_rate"])} | 原usable+partial全部计入 |',
              f'| 原始工具fetch_status=success | {percent(result["raw_fetch_success"])} | 原始返回状态，尚不能代表正文语义可用 |',
              f'| 各读取批次可用率等权平均 | {result["batch_macro_usable_rate"] * 100:.2f}% | 每批等权，主指标按条目加权 |', '',
              f'302条返回中，{overall["usable_count"]}条可用，{overall["nonempty_unusable_count"]}条虽有文字但不可用，{overall["empty_count"]}条没有正文。重复返回共{result["repeat_return_count"]}条，全部来自缓存；同一条目的可用性判断相同。', '',
              '## 逐题结果及分析', '', '| 题目 | 批次 | 返回条目 | 非空文本 | 可用正文 | 返回条目可用率 | 非空文本中可用率 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for t in topic_rows:
        s = t['summary']
        lines.append(f'| {t["topic"]} {t["title"]} | {t["batches"]} | {s["returned_count"]} | {s["nonempty_count"]} | {s["usable_count"]} | {100*s["usable_rate"]["value"]:.2f}% | {100*s["nonempty_body_usable_rate"]["value"]:.2f}% |')
    lines += ['', '- 第35题：正文失败除空值外，集中在文档平台推荐页、下载链接、标准封面及目录。新增边界排除6条，说明仅文件链接或标题不能证明取得制度/指标内容。',
              '- 第31题：大部分真实文章/摘要保留；日本博物馆研修文档仅积分下载界面，新增排除1条。',
              '- 第46题：政策及新闻正文仍可用；报告目录、空摘要栏目新增排除2条。报告简介中的真实行业论述可以算部分可用，但不能宣称已读取报告全文。',
              '- 第16题：真实论文摘要可以作为可用内容；验证码、导航及错误页不可用。7个缓存重复条目提高了按次数分母和分子，不能当成新增独立页面。',
              '- 第49题：空正文25条、异常或只有图链/目录等内容21条；新增排除3条。保留真实摘要和已抽取的文字数据表，不把图片URL视为已读取图中数字。', '',
              '## 失败分类', '', '| 类别 | 返回次数 |', '| --- | --- |']
    for category, count in sorted(overall['failure_categories'].items(), key=lambda x: -x[1]):
        lines.append(f'| {category} | {count} |')
    lines += ['', '空正文来自HTTP 403/404等错误、网络失败、抽取异常及无内容；已有文字的失败包括推荐/导航、验证码、登录墙、乱码、错误页及本次新增的边界排除。逐条明细另保留原始failure_category，不把页面标签混同HTTP状态。', '',
              '## 本次边界复核与失败样例', '', '| 页面ID | 原标签 | 本次排除原因 |', '| --- | --- | --- |']
    for c in corrections:
        lines.append(f'| {c["page_id"]} | {c["original_label"]} | {c["reason"]} |')
    lines += ['', '复核记录保存每条的URL、正文SHA-256、原文引句和字符起止位置；排除依据是保存原文缺少实质目标内容，不是站点名或付费墙本身。', '',
              '## 逐批口径与6篇示例', '', '| 批次 | 返回 | 非空文本 | 可用 | 可用率 |', '| --- | --- | --- | --- | --- |']
    for b in batch_rows:
        s = b['summary']
        lines.append(f'| {b["batch_id"]} | {s["returned_count"]} | {s["nonempty_count"]} | {s["usable_count"]} | {100*s["usable_rate"]["value"]:.2f}% |')
    examples = [b for b in batch_rows if b['summary']['returned_count'] == 6 and b['summary']['usable_count'] == 5]
    lines += ['', f'实际“返回6条、可用5条”的批次：{", ".join(b["batch_id"] for b in examples)}，各为83.33%。例如16-batch-09含1条空正文。31-batch-11虽然6条都有文字，其中1条为实质空页、1条仅文档元信息，本次计4/6（66.67%）。', '',
              '## 与旧评估的差异', '',
              f'旧报告的51.56%是149/289：按题URL去重，且全部partial都算可用。本次先恢复逐次返回，旧标签口径得到{percent(overall["original_broad_usable_rate"])}；再扣除12个无实质目标内容的边界条目，得到{percent(overall["usable_rate"])}。按同样新可用性标准去重则为{percent(unique["usable_rate"])}。这是分母与可用性口径的变化，不能解释为系统性能上升或下降。历史原文件不覆盖。', '',
              '## 局限性与证据边界', '',
              '- 已测量：五题工具调用、实际返回次数、空/非空正文、URL及内容哈希、缓存重复；逐批及逐题计数可由脚本复现。',
              '- 间接推断：语义正文可用性沿用历史AI标签并补充12条原文边界复核。历史50个超长页面使用抽样，未重新全量独立评审；仍可能误判。所有数字都以该判定口径为条件。',
              '- 尚未评估：人工金标、未读取的搜索候选、原始HTML/PDF对照抽取精确率/召回率、图片/OCR、全部开发集及在线当前网页成功率、正文事实真实性或任务答案充分性。', '',
              '## 运行核查与同次补充复核', '',
              f'补充复核日期：{date}。首次执行在检查输入候选数等于返回数的断言处中断，未生成结果文件；原始数据核查确认输入638条、实际返回302条。调整为逐条验证返回URL属于调用输入，并按实际返回数计分母，随后正式统计完成。此中断不代表抓取失败。', '',
              '同次补充仅修正报告示例、将无正文58条与有文本但实质空页3条分别展示，并记录候选输入量；不改变145/302的正文可用率或12条边界复核结论。', '',
              '## 结论与下一步建议', '',
              f'在这五题已有轨迹中，每实际返回100条，约{100*overall["usable_rate"]["value"]:.1f}条提取到可用正文或实质摘要；其中空值与仅有导航/链接的内容必须保留为失败。后续评估统一记录逐次正文可用率，并同时保留去重率和非空条件可用率；优先针对附件正文、目录/封面、图片型数据和验证码进行抽取改进，再在同一冻结集合上独立人工复核。', '',
              f'复现：`python3 backend/eval/scripts/统计通用研究轨迹正文可用率.py --date {date} --output backend/eval/data/通用研究开发集正文可用率复现-{date.replace("-", "")}`。输出目录必须未存在，防止覆盖历史。', '',]
    report = report_path or ROOT / 'docs' / f'通用研究开发集已有轨迹正文可用率结果及分析-{date.replace("-", "")}.md'
    if report.exists() and report_path is None:
        report = out / '正文可用率结果及分析.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default='2026-10-11')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / f'backend/eval/data/通用研究开发集正文可用率-{args.date.replace("-", "")}'
    # 保留新评估，禁止覆盖历史目录。
    out = out.resolve()
    if out.exists():
        raise ValueError(f'输出目录已存在：{out}')
    pages = read(SOURCE / '逐页正文质量初评.json')
    batches = read(SOURCE / '冻结读取批次.json')
    queries = read(SOURCE / '冻结检索轨迹.json')
    lookup = {(p['topic'], norm(p['url'])): p for p in pages}
    assert len(lookup) == len(pages)
    corrections = []
    for page_id, (category, anchor, reason) in EXCLUSIONS.items():
        p = next(p for p in pages if p['id'] == page_id)
        assert p['assessment']['label'] in {'usable', 'partial'}
        start = p['content'].index(anchor)
        quote = p['content'][start:start + min(180, max(len(anchor), 110))]
        corrections.append({'page_id': page_id, 'topic': p['topic'], 'url': p['url'], 'content_sha256': p['content_sha256'], 'original_label': p['assessment']['label'], 'content_usable': False, 'category': category, 'reason': reason, 'quote': quote, 'quote_start': start, 'quote_end': start + len(quote), 'review_date': args.date, 'reviewer': 'Codex主代理原文边界复核，非人工金标'})

    def decision(p):
        a = p['assessment']
        override = EXCLUSIONS.get(p['id'])
        category = override[0] if override else ('empty' if not p['content'].strip() else a.get('category', a['label']))
        if category == 'empty' and p['content'].strip():
            category = 'empty_page'
        return {'page_id': p['id'], 'topic': p['topic'], 'url': p['url'], 'title': p['title'], 'content_sha256': p['content_sha256'], 'content_chars': len(p['content']), 'nonempty': bool(p['content'].strip()), 'original_label': a['label'], 'original_category': a.get('category'), 'original_reason': a['reason'], 'content_usable': not override and a['label'] in {'usable', 'partial'}, 'category': category, 'reason': override[2] if override else a['reason'], 'boundary_excluded': bool(override), 'assessment_sample_is_full': p['sample_is_full']}

    rows, batch_rows, inputs = [], [], {SOURCE / n for n in ['逐页正文质量初评.json', '冻结读取批次.json', '冻结检索轨迹.json', '评估配置.json', 'Codex复核修正.json', '来源文件指纹.json']}
    original_fingerprints = {r['path']: r['sha256'] for r in read(SOURCE / '来源文件指纹.json')}
    for topic, path in sorted({(b['topic'], b['history_path']) for b in batches}):
        path = Path(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_fingerprints[str(path)], '历史轨迹与冻结时不同'
        inputs.add(path)
        history = read(path)
        groups = defaultdict(dict)
        for record in history:
            if record['kind'].startswith('tool_'):
                assert record['kind'] not in groups[record['call_id']]
                groups[record['call_id']][record['kind']] = record
        for batch in [b for b in batches if b['topic'] == topic]:
            group = groups[batch['call_id']]
            assert group['tool_input']['payload']['tool_name'] == 'v2_web_fulltext_read'
            raw = group['tool_raw_result']
            values = raw['payload']['value']
            assert len(values) == batch['returned_count']
            candidates = group['tool_input']['payload']['arguments']['items']
            input_urls = {norm(item['url']) for item in candidates}
            assert all(norm(value['url']) in input_urls for value in values)
            batch_items = []
            for index, value in enumerate(values):
                page = lookup[(topic, norm(value['url']))]
                content = str(value.get('content') or '')
                assert sha(content) == page['content_sha256'] == sha(page['content']), '相同URL内容版本不同，不能直接复用标签'
                row = {**decision(page), 'batch_id': batch['id'], 'call_id': batch['call_id'], 'return_position': index + 1, 'history_path': str(path), 'history_raw_seq': raw['seq'], 'raw_result_index': index, 'raw_fetch_status': value.get('fetch_status'), 'raw_failure_category': value.get('failure_category'), 'raw_error': value.get('error'), 'from_cache': bool(value.get('from_cache'))}
                assert not row['content_usable'] or row['nonempty']
                rows.append(row)
                batch_items.append(row)
            batch_rows.append({'batch_id': batch['id'], 'topic': topic, 'call_id': batch['call_id'], 'history_path': str(path), 'history_raw_seq': raw['seq'], 'input_candidate_count': len(candidates), 'summary': summarize(batch_items)})
    unique_rows = [decision(p) for p in pages]
    topic_rows = [{'topic': topic, 'title': title, 'batches': sum(b['topic'] == topic for b in batches), 'summary': summarize([r for r in rows if r['topic'] == topic]), 'url_dedup_summary': summarize([r for r in unique_rows if r['topic'] == topic])} for topic, title in TOPICS.items()]
    frequencies = Counter((r['topic'], norm(r['url'])) for r in rows)
    duplicate_rows = sum(n - 1 for n in frequencies.values())
    result = {'date': args.date, 'overall': summarize(rows), 'per_topic_url_dedup': summarize(unique_rows), 'batches': len(batch_rows), 'search_calls': len(queries), 'batch_macro_usable_rate': statistics.mean(b['summary']['usable_rate']['value'] for b in batch_rows), 'raw_fetch_success': rate(sum(r['raw_fetch_status'] == 'success' for r in rows), len(rows)), 'raw_fetch_status_counts': dict(Counter(r['raw_fetch_status'] for r in rows)), 'repeat_return_count': duplicate_rows, 'cached_return_count': sum(r['from_cache'] for r in rows), 'content_hash_matched_count': len(rows), 'boundary_excluded_unique_pages': len(corrections), 'sampling': {'unique_sampled_pages': sum(not p['sample_is_full'] and bool(p['content']) for p in pages), 'returned_sampled_entries': sum(not r['assessment_sample_is_full'] and r['nonempty'] for r in rows)}}
    assert len(rows) == sum(b['returned_count'] for b in batches)
    assert set(frequencies) == set(lookup)
    assert sum(r['summary']['usable_count'] for r in topic_rows) == result['overall']['usable_count']
    assert duplicate_rows == sum(r['from_cache'] for r in rows)
    seen_returns = set()
    for row in rows:
        key = (row['topic'], norm(row['url']))
        assert row['from_cache'] == (key in seen_returns)
        seen_returns.add(key)
    config = {'date': args.date, 'timezone': 'Asia/Singapore', 'source_directory': str(SOURCE), 'topics': list(TOPICS), 'primary_denominator': '所有实际v2_web_fulltext_read返回条目，包括空正文、失败占位、异常页、缓存重复；不按URL去重', 'usable_definition': '提取到目标正文或有实质内容的真实摘要/预览；只含下载链接、文档封面/章节目录、未OCR图片URL、销售简介/通用元信息不可用', 'nonempty_denominator': 'content.strip()非空的返回条目，作为条件指标另报', 'label_reuse': '2026-10-08 Qwen3-max初评及Codex复核；本次12个边界排除；每个条目按题URL与完整正文SHA-256匹配，未匹配必须失败', 'new_model_calls': 0, 'new_web_fetches': 0, 'human_gold': False, 'scope': '五个已完成历史轨迹，不包括未选择读取的搜索候选', 'script_path': str(Path(__file__).resolve()), 'repeat_policy': '主指标每次返回都计数；另提供按题URL去重指标', 'review_limitations': '继承其余历史AI标签，并未全量独立重判；旧评估超长正文抽样，图片内容未评估'}
    inputs.add(Path(__file__).resolve())
    fingerprints = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(inputs)]
    out.mkdir(parents=True)
    dump(out / '评估配置.json', config)
    dump(out / '来源文件指纹.json', fingerprints)
    dump(out / '正文可用性复核记录.json', corrections)
    jsonl(out / '逐条返回正文明细.jsonl', rows)
    jsonl(out / '按题去重正文明细.jsonl', unique_rows)
    dump(out / '逐批正文可用率.json', batch_rows)
    dump(out / '逐题正文可用率.json', topic_rows)
    dump(out / '指标汇总.json', result)
    report = make_report(out, args.date, config, result, corrections, batch_rows, topic_rows, unique_rows)
    print(json.dumps({'report': str(report), 'output': str(out), 'primary': result['overall']['usable_rate'], 'nonempty': result['overall']['nonempty_body_usable_rate'], 'dedup': result['per_topic_url_dedup']['usable_rate'], 'batch_macro': result['batch_macro_usable_rate']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
