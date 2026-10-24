#!/usr/bin/env python3
"""冻结289网址，归档当前读取链路使用的同一份HTML，支持云API配对评估。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import time
import types
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'backend/eval/data/通用研究开发集正文可用率-20261011/按题去重正文明细.jsonl'
TOPICS = {35: '市政污水与雨污协同', 31: '中外博物馆教育', 46: '房地产可持续发展',
          16: '非接触式感知算法', 49: '正畸与医美共同需求'}


def now():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def load_reader(out):
    # 只导入冻结的读取器及其纯函数依赖，避免service.__init__启动整个业务栈。
    package = types.ModuleType('service')
    package.__path__ = [str(out / '冻结基线源码')]
    sys.modules['service'] = package
    path = out / '冻结基线源码/web_reader_service.py'
    spec = importlib.util.spec_from_file_location('service.web_reader_service', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def init(out):
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(s) for s in SOURCE.read_text().splitlines()]
    assert len(rows) == 289 and len({r['url'] for r in rows}) == 289
    dump(out / '冻结289网址.json', rows)
    paths = [SOURCE, Path(__file__).resolve()]
    folder = out / '冻结基线源码'
    folder.mkdir()
    for name in ['web_reader_service.py', 'web_document.py', 'web_document_structure.py']:
        source = ROOT / 'backend/app/service' / name
        shutil.copy2(source, folder / name)
        paths.append(source)
    shutil.copy2(__file__, out / '冻结执行脚本.py')
    dump(out / '来源文件指纹.json', [{'path': str(p), 'sha256': sha(p.read_bytes())} for p in paths])
    config = {
        'created_at': now(), 'timezone': 'Asia/Singapore', 'topics': TOPICS,
        'source_count': 289, 'source_path': str(SOURCE),
        'scope': '固定289个历史请求URL，重新获取页面；并非重跑研究智能体',
        'acquisition': '执行冻结现有_read_uncached，截取它实际选择的静态HTML或浏览器DOM；关闭读取器正文缓存；浏览器cache_mode=bypass',
        'baseline': '冻结的当前读取链路；静态提取和原有浏览器回退选择保持，保存机器质量门禁前的抽取正文供独立语义评审',
        'shared_input': '当前链路选中静态输出时归档静态HTML；选中浏览器输出时归档对应浏览器DOM与同次raw_markdown。MinerU上传该同HTML文件，禁止让云端重新抓原URL',
        'html_encoding': '沿用当前读取器_decode_html；另保存原始响应字节。共享HTML统一编码UTF-8并修正charset声明，避免上传端再次错误解码',
        'format_policy': 'HTML两组配对；PDF、Office等二进制另归档并单列；非HTML不送MinerU-HTML，不混入同HTML分母',
        'whole_url_policy': '289分母另报读取链路覆盖；访问失败保留失败；非HTML用相同的当前文档路由结果共享，明确为组合链路',
        'judge': '统一qwen3-max、temperature=0；盲化输出来源；保留真实摘要/实质预览；仅目录、封面、下载/图片链接、导航、验证码、登录、错页、乱码不可用',
        'history_baselines': {'2026-10-08': '149/289=51.56%', '2026-10-11': '137/289=47.40%'},
        'history_comparability': '历史数值仅作参考；新抓取内容与当前基线版本可能变化，增益仅由本次同HTML两组得出',
        'mineru_api': {'base_url': 'https://mineru.net/api/v4', 'model_version': 'MinerU-HTML',
                       'mode': '本地冻结HTML上传；每批不超过200文件；下载main.html与full.md',
                       'credentials': '仅进程环境或backend/.env中的MINERU_API_TOKEN / MINERU_TOKEN；不写入归档'},
    }
    dump(out / '评估配置.json', config)
    dump(out / 'API资料来源.json', {
        'checked_at': now(), 'official_docs': 'https://mineru.net/doc/docs/index_en/',
        'file_upload_endpoint': '/api/v4/file-urls/batch',
        'query_endpoint': '/api/v4/extract-results/batch/{batch_id}',
        'model_version': 'MinerU-HTML', 'token_required': True,
        'self_host_required': False,
        'note': '免费Agent轻量接口固定pipeline，不能替代MinerU-HTML对照；普通聊天API的其他模型也不是同一个微调模型',
    })
    report(out)
    print(json.dumps({'out': str(out), 'report': str(report_path(out))}, ensure_ascii=False), flush=True)


def report_path(out):
    return ROOT / 'docs' / f'{out.name}结果及分析.md'


def records(out):
    return [read(p) for p in sorted((out / '逐页采集').glob('*/采集结果.json'))]


def report(out):
    rows = records(out)
    config = read(out / '评估配置.json')
    html_rows = [r for r in rows if r.get('shared_html_path')]
    types_count = dict(Counter(r.get('input_kind', 'unknown') for r in rows))
    state = {
        'updated_at': now(), 'expected_urls': 289, 'captured_urls': len(rows),
        'shared_html_count': len(html_rows), 'input_kinds': types_count,
        'baseline_nonempty': sum(bool(r.get('baseline_content_chars')) for r in rows),
        'baseline_fetch_status_counts': dict(Counter(r.get('baseline_fetch_status') for r in rows)),
        'same_input_verified_count': sum(bool(r.get('shared_html_sha256')) for r in html_rows),
        'mineru_completed': len(list((out / 'MinerU结果').glob('*/抽取结果.json'))),
        'judge_completed': len(list((out / '正文判定').glob('*.json'))),
        'credential_configured': bool(os.getenv('MINERU_API_TOKEN') or os.getenv('MINERU_TOKEN')),
    }
    dump(out / '阶段指标汇总.json', state)
    lines = [f'# {out.name}结果及分析', '',
        f'评估日期：2026-10-17，Asia/Singapore。最近更新：{state["updated_at"]}。',
        '评估对象：历史通用研究五题的固定289个URL；当前读取方案与MinerU-HTML在同一份重新采集HTML上的正文可用性。', '',
        '## 数据与配置口径', '',
        '- 两组共用相同的归档HTML/浏览器DOM及SHA-256；MinerU通过文件上传接收同一输入，不自行重新访问原URL。',
        '- 当前读取器及正文结构、质量函数已冻结；保存机器质量门禁前的提取文本供两组统一语义评判。原有程序接纳状态另列，不作为可用性标签。',
        '- 基线选择浏览器结果时，同时归档同一次浏览器返回的DOM和raw_markdown；静态结果被保留时使用静态HTML。',
        '- 可用包括真实正文、实质摘要或部分正文；纯目录、封面、导航、验证、登录、下载/图片链接、错页或乱码不可用；两组用相同盲评口径。',
        '- PDF、Office等非HTML单列。HTML抽取可用率与289个URL组合链路覆盖分别报告，不能混用分母。',
        '- 历史51.56%（149/289）与复核47.40%（137/289）只作参考；当前网页、代码与判定可能变化，不直接据此计算本次增益。', '',
        '## 原始结果及配置', '',
        f'- 运行目录：[{out.name}]({out})。',
        f'- [冻结289网址.json]({out / "冻结289网址.json"})、[评估配置.json]({out / "评估配置.json"})、[来源文件指纹.json]({out / "来源文件指纹.json"})。',
        f'- [阶段指标汇总.json]({out / "阶段指标汇总.json"})；逐页响应字节、静态HTML、浏览器DOM/Markdown、共享HTML、基线文本与SHA-256在`逐页采集/`。',
        f'- [冻结执行脚本.py]({out / "冻结执行脚本.py"})及`冻结基线源码/`保留执行版本。', '',
        '## 核心指标与阶段状态', '',
        f'- 已完成URL采集：{len(rows)}/289；归档可配对HTML：{len(html_rows)}份。',
        f'- 输入类型：`{json.dumps(types_count, ensure_ascii=False)}`。',
        f'- 基线非空输出：{state["baseline_nonempty"]}；这不是语义可用正文数。',
        f'- MinerU已归档结果：{state["mineru_completed"]}；语义判定记录：{state["judge_completed"]}。',
        '- 尚未完成两组统一语义评审时，不报告正文可用率或提升百分点。', '',
        '## 逐题采集', '',
        '| 题号 | 当前已完成URL | 同HTML样本 | 基线非空输出 |', '| --- | --- | --- | --- |']
    for topic in TOPICS:
        rr = [r for r in rows if r['topic'] == topic]
        lines.append(f'| {topic} | {len(rr)} | {sum(bool(r.get("shared_html_path")) for r in rr)} | {sum(bool(r.get("baseline_content_chars")) for r in rr)} |')
    lines += ['', '## 失败样例与局限性', '']
    failed = [r for r in rows if r.get('capture_error') or not r.get('shared_html_path')]
    for r in failed[:8]:
        lines.append(f'- `{r["page_id"]}`：{r.get("capture_error") or r.get("input_kind")}；URL：{r["url"]}。')
    if not failed:
        lines.append('- 当前还没有已归档的失败或非HTML样例；不能据此推断全量访问成功。')
    lines += ['', '- 当前页面可能已失效、改版或重定向；记录最终URL和页面标题，不能将内容变化归因于抽取模型。',
        '- 对受阻页面只保留实际可获得内容；正文选择不能恢复服务器没有提供的目标正文。',
        '- 官方云API未公开精确模型权重版本和回退细节时，只能称为官方MinerU-HTML服务效果，不等同开源固定权重本体。',
        '- AI语义裁判不是人工金标；超长输出的判定抽样与未决条目必须另报。', '',
        '## API与部署结论', '',
        '[官方文档](https://mineru.net/doc/docs/index_en/)明确支持带Token的MinerU-HTML托管API及本地HTML文件上传，不必在本机部署。免费Agent轻量接口固定为pipeline，不能拿来替代本次MinerU-HTML。', '',
        '## 证据边界、结论及下一步', '',
        '**已测量**：以阶段指标为准的真实抓取、输入归档、程序抽取与文件哈希。**间接推断**：质量改善方向及历史AI标签。**尚未评估**：未完成统一语义判定时的正文可用率、MinerU相对增益、人工正文F1及真实账单费用。', '',
        '继续顺序：完成289个URL输入归档 → 上传同一HTML至MinerU-HTML → 保存原始服务结果 → 对两组盲评 → 汇总配对改善/退化、逐类失败与延迟。缺少API Token时可先完成抓取与基线，不把缺少服务凭证记为网页抽取失败。', '']
    report_path(out).write_text('\n'.join(lines), encoding='utf-8')
    return state


def classify(body, content_type, url):
    mime = content_type.lower().split(';')[0].strip()
    head = body[:4096].lstrip().lower()
    if body.startswith(b'%PDF-') or mime == 'application/pdf':
        return 'pdf'
    if body.startswith(bytes.fromhex('d0cf11e0a1b11ae1')) or mime in {
        'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.ms-powerpoint', 'application/vnd.ms-excel'} or body.startswith(b'PK\x03\x04'):
        return 'office_or_archive'
    if mime.startswith('image/'):
        return 'image'
    if 'html' in mime or any(x in head for x in [b'<html', b'<!doctype html', b'<head', b'<body']):
        return 'html'
    if mime.startswith('text/'):
        return 'other_text'
    return 'unknown_binary'


def utf8_html(html):
    # 页面内容不清洗；仅让文件的UTF-8编码与声明一致。两组都使用同样字节。
    import re
    html = re.sub(r'(<meta\b[^>]*charset\s*=\s*["\']?)[\w.:-]+', r'\1utf-8', html, flags=re.I)
    return html


async def capture_one(source, out, module, browser_gate):
    folder = out / '逐页采集' / source['page_id']
    result_path = folder / '采集结果.json'
    if result_path.exists():
        return read(result_path)
    folder.mkdir(parents=True, exist_ok=True)
    reader = module.WebReaderService()
    result = {'page_id': source['page_id'], 'topic': source['topic'], 'url': source['url'],
              'title': source['title'], 'started_at': now(), 'historical_usable': source['content_usable'],
              'historical_label': source['original_label'], 'capture_error': None}
    captured = {}
    original_download = reader._download_with_retry
    original_extract = reader._extract_html

    async def download(url):
        try:
            final, content_type, body, attempts = await original_download(url)
        except Exception as exc:
            captured['download_error'] = type(exc).__name__ + ': ' + str(exc)[:400]
            if isinstance(exc, httpx.HTTPStatusError):
                captured['download_http_status'] = exc.response.status_code
            raise
        kind = classify(body, content_type, final)
        (folder / '原始响应.bin').write_bytes(body)
        captured.update(final_url=final, content_type=content_type, downloaded_bytes=len(body),
                        response_sha256=sha(body), input_kind=kind, attempts=attempts)
        if kind == 'html':
            html = utf8_html(reader._decode_html(body, content_type))
            (folder / '静态HTML.html').write_text(html, encoding='utf-8')
            captured['static_html'] = html
            # 让基线也读取将要上传的完全相同UTF-8字节。
            return final, 'text/html; charset=utf-8', html.encode(), attempts
        return final, content_type, body, attempts

    def extract(html, url):
        value = original_extract(html, url)
        captured['static_extraction'] = value
        return value

    async def browser(url, title):
        if not reader.crawl4ai_endpoint:
            captured['browser_error'] = '当前读取器没有配置Crawl4AI服务'
            return None
        async with browser_gate:
            try:
                async with httpx.AsyncClient(timeout=65, trust_env=False) as client:
                    payload = {'urls': [url], 'crawler_config': {'type': 'CrawlerRunConfig',
                               'params': {'cache_mode': 'bypass', 'page_timeout': 30000}}}
                    if reader.browser_proxy:
                        payload['browser_config'] = {'proxy': reader.browser_proxy}
                    response = await client.post(reader.crawl4ai_endpoint + '/crawl', json=payload)
                    response.raise_for_status()
                    data = response.json()
                dump(folder / '浏览器原始响应.json', data)
                items = data.get('results') or []
                item = items[0] if items else data
                html = item.get('html') or ''
                captured['browser_status_code'] = item.get('status_code')
                captured['browser_success'] = item.get('success')
                captured['browser_final_url'] = item.get('redirected_url') or item.get('url')
                if html:
                    html = utf8_html(html)
                    (folder / '浏览器DOM.html').write_text(html, encoding='utf-8')
                    captured['browser_html'] = html
                markdown = reader._crawl4ai_item(data)
                if markdown:
                    (folder / '浏览器原始正文.txt').write_text(markdown, encoding='utf-8')
                    return module.WebReadResult(url=url, final_url=url, title=title, content=markdown,
                           extractor='crawl4ai-sidecar', fetch_status='success', fetched_at=time.time())
            except Exception as exc:
                captured['browser_error'] = type(exc).__name__ + ': ' + str(exc)[:400]
            return None

    reader._download_with_retry = download
    reader._extract_html = extract
    reader._read_with_crawl4ai = browser
    start = time.perf_counter()
    try:
        baseline = await reader._read_uncached(source['url'], source['title'])
        final = baseline.to_dict()
        (folder / '基线最终正文.txt').write_text(final['content'], encoding='utf-8')
        dump(folder / '基线程序结果.json', final)
        browser_used = str(baseline.recovery_action).startswith('browser')
        html = captured.get('browser_html') if browser_used else captured.get('static_html')
        # 程序质量门禁不会改变核心抽取语义判定，保留门禁前文本用于两组独立评审。
        before = ((folder / '浏览器原始正文.txt').read_text() if browser_used and
                  (folder / '浏览器原始正文.txt').exists() else
                  captured.get('static_extraction', {}).get('content', baseline.content))
        before = before.replace('\x00', '')
        from service.web_document_structure import limit_body_preserving_tables
        before = limit_body_preserving_tables(before.strip(), reader.max_chars)
        (folder / '基线待评正文.txt').write_text(before, encoding='utf-8')
        result.update(baseline_content_chars=len(before), baseline_fetch_status=baseline.fetch_status,
                      baseline_failure_category=baseline.failure_category, baseline_extractor=baseline.extractor,
                      baseline_recovery_action=baseline.recovery_action, baseline_content_sha256=sha(before),
                      input_kind='html' if html else captured.get('input_kind', 'fetch_failed'))
        if html:
            shared = folder / '共享输入HTML.html'
            shared.write_text(html, encoding='utf-8')
            result.update(shared_html_path=str(shared), shared_html_sha256=sha(shared.read_bytes()),
                          shared_html_bytes=shared.stat().st_size,
                          shared_input_source='browser_dom' if browser_used else 'static_html')
            if not browser_used:
                # 保存的HTML重放核心提取，确保没有把清洗后的另一份HTML当输入。
                replay = original_extract(shared.read_text(), captured.get('final_url') or source['url'])
                assert replay.get('content', '') == captured['static_extraction']['content']
        elif browser_used:
            result['capture_error'] = '浏览器返回正文但未返回DOM，不能加入同HTML配对'
        elif result['input_kind'] == 'fetch_failed':
            result['capture_error'] = baseline.error or captured.get('download_error')
    except Exception as exc:
        result.update(capture_error=type(exc).__name__ + ': ' + str(exc)[:500],
                      input_kind=captured.get('input_kind', 'capture_failed'))
    result.update({k: v for k, v in captured.items() if k not in {
        'static_html', 'browser_html', 'static_extraction', 'input_kind'}})
    result.update(elapsed_seconds=round(time.perf_counter() - start, 3), finished_at=now())
    dump(result_path, result)
    return result


async def capture(out, limit, concurrency):
    module = load_reader(out)
    source = read(out / '冻结289网址.json')
    if limit:
        source = source[:limit]
    gate, browser_gate = asyncio.Semaphore(concurrency), asyncio.Semaphore(2)
    completed = 0

    async def work(row):
        nonlocal completed
        async with gate:
            result = await capture_one(row, out, module, browser_gate)
        completed += 1
        report(out)
        print(f'采集 {completed}/{len(source)} {row["page_id"]} {result.get("input_kind")} '
              f'HTML={bool(result.get("shared_html_path"))} 基线字符={result.get("baseline_content_chars", 0)}', flush=True)

    await asyncio.gather(*(work(row) for row in source))
    state = report(out)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'capture', 'report'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--concurrency', type=int, default=6)
    args = parser.parse_args()
    load_dotenv(ROOT / 'backend/.env')
    # 独立运行目录只保存评测缓存，避免污染线上读取器缓存。
    os.environ['WEB_READER_CACHE_DIR'] = str(args.out.resolve() / '隔离缓存')
    logging.basicConfig(level=logging.WARNING)
    if args.action == 'init':
        init(args.out.resolve())
    elif args.action == 'capture':
        asyncio.run(capture(args.out.resolve(), args.limit, args.concurrency))
    else:
        print(json.dumps(report(args.out.resolve()), ensure_ascii=False))


if __name__ == '__main__':
    main()
