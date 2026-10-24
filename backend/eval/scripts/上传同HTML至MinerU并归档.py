#!/usr/bin/env python3
"""使用官方MinerU-HTML托管API上传冻结HTML，支持已提交批次断点继续。"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile
from datetime import datetime

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('paired_capture', Path(__file__).with_name('对照评估289网址与MinerU.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
read, dump, sha, now = common.read, common.dump, common.sha, common.now


class MinerUError(RuntimeError):
    pass


MAX_HTML_FILES = 100


def submitted_count(out):
    # 按提交文件次数保守计数，连结果未知的提交意图也占用预算。
    return sum(len(read(p)['payload']['files']) for p in
               (out / 'MinerU服务批次').glob('*/提交意图.json'))


def freeze_sample(out, candidates):
    path = out / '冻结100份HTML配对样本.json'
    if path.exists():
        selected = read(path)
        lookup = {r['page_id']: r for r in candidates}
        for row in selected:
            assert row['shared_html_sha256'] == lookup[row['page_id']]['shared_html_sha256']
        return selected
    if len(common.records(out)) != 289:
        raise MinerUError('先完成289网址采集，再冻结最多100份分层样本')
    from collections import defaultdict
    strata = defaultdict(list)
    for row in candidates:
        strata[(row['topic'], row['historical_usable'])].append(row)
    target = min(MAX_HTML_FILES, len(candidates))
    quotas = {k: int(target * len(v) / len(candidates)) for k, v in strata.items()}
    order = sorted(strata, key=lambda k: (-(target * len(strata[k]) / len(candidates) - quotas[k]), k))
    for key in order[:target - sum(quotas.values())]:
        quotas[key] += 1
    pinned = set()
    for p in (out / 'MinerU服务批次').glob('*/批次身份.json'):
        pinned.update(read(p)['page_ids'])
    selected, details = [], []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: r['page_id'])
        random.Random('20261017:' + str(key)).shuffle(group)
        group.sort(key=lambda r: r['page_id'] not in pinned)
        if sum(r['page_id'] in pinned for r in group) > quotas[key]:
            raise MinerUError('接口验证样本超过该层配额，请显式调整抽样设计')
        selected.extend(group[:quotas[key]])
        details.append({'topic': key[0], 'historical_usable': key[1],
                        'population': len(group), 'sample': quotas[key]})
    assert len(selected) == target and len({r['page_id'] for r in selected}) == target
    assert pinned <= {r['page_id'] for r in selected}
    dump(path, selected)
    dump(out / '分层抽样配置.json', {
        'frozen_at': now(), 'hard_cap_including_pilot': MAX_HTML_FILES,
        'html_population': len(candidates), 'sample_count': target,
        'strata': details, 'seed': '20261017:<topic,historical_usable>',
        'pinned_pilot_ids': sorted(pinned),
        'method': '按题号和2026-10-11历史正文标签分层，最大余数比例分配；层内固定种子抽样，保留已提交接口验证样本',
        'limitation': '仅代表本次实际获取HTML的条件集合；不代表全部289 URL；保留1个非随机接口验证样本',
    })
    return selected


async def submission_rate_gate(out, count):
    # 用户截图限制：创建任务接口合计每分钟50个文件。
    while True:
        events = [(datetime.fromisoformat(read(p)['at']).timestamp(), len(read(p)['payload']['files']))
                  for p in (out / 'MinerU服务批次').glob('*/提交意图.json')]
        recent = [(at, n) for at, n in events if time.time() - at < 62]
        if sum(n for _, n in recent) + count <= 50:
            return
        await asyncio.sleep(min(10, max(1, min(at for at, _ in recent) + 62 - time.time())))


async def request_json(client, method, url, token, dest, payload=None):
    start = time.perf_counter()
    response = await client.request(method, url, headers={'Authorization': 'Bearer ' + token}, json=payload)
    safe_text = response.text.replace(token, '[凭证已隐藏]')
    try:
        data = json.loads(safe_text)
    except ValueError:
        data = {'non_json_response': safe_text[:5000]}
    dump(dest, {'received_at': now(), 'http_status': response.status_code,
                'elapsed_seconds': round(time.perf_counter() - start, 3), 'response': data})
    if response.status_code != 200 or data.get('code') != 0:
        raise MinerUError(f'HTTP {response.status_code} code={data.get("code")} msg={str(data.get("msg") or data.get("message") or data.get("non_json_response"))[:300]}')
    return data['data']


async def save_result(item, row, out, client):
    folder = out / 'MinerU结果' / row['page_id']
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / '抽取结果.json'
    if dest.exists():
        return
    result = {'page_id': row['page_id'], 'topic': row['topic'], 'source_url': row['url'],
              'shared_html_path': row['shared_html_path'], 'shared_html_sha256': row['shared_html_sha256'],
              'service': 'https://mineru.net/api/v4', 'model_version': 'MinerU-HTML',
              'precise_weight_version': '官方未在接口结果中公开，未推断',
              'fallback_observability': '服务未公开本体与回退标志，按服务整体评价',
              'service_state': item.get('state'), 'archived_at': now()}
    dump(folder / '服务任务结果.json', item)
    content = ''
    start = time.perf_counter()
    if item.get('state') == 'done':
        response = await client.get(item['full_zip_url'])
        response.raise_for_status()
        if len(response.content) > 200 * 1024 * 1024:
            raise MinerUError('结果ZIP超过200MB，停止解压')
        (folder / '服务原始结果.zip').write_bytes(response.content)
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            selected = {}
            for info in archive.infolist():
                leaf = Path(info.filename).name
                if leaf in {'full.md', 'main.html'}:
                    if info.file_size > 30 * 1024 * 1024:
                        raise MinerUError('单个正文结果超过30MB')
                    selected[leaf] = archive.read(info).decode('utf-8-sig', errors='replace')
            if 'main.html' in selected:
                (folder / '提取主体HTML.html').write_text(selected['main.html'], encoding='utf-8')
            if 'full.md' in selected:
                (folder / '原始全文Markdown.md').write_text(selected['full.md'], encoding='utf-8')
                content = selected['full.md']
            elif selected.get('main.html'):
                from bs4 import BeautifulSoup
                from service.web_document_structure import render_html_body
                content = render_html_body(BeautifulSoup(selected['main.html'], 'lxml'))
            else:
                raise MinerUError('API标记完成但ZIP缺少full.md和main.html')
    else:
        result['error'] = item.get('err_msg') or '服务抽取失败'
    result['original_content_chars'] = len(content)
    from service.web_document_structure import limit_body_preserving_tables
    content = limit_body_preserving_tables(content.replace('\x00', '').strip(), 30000)
    (folder / '待评正文.txt').write_text(content, encoding='utf-8')
    result.update(content_chars=len(content), content_sha256=sha(content),
                  result_download_seconds=round(time.perf_counter() - start, 3))
    dump(dest, result)


async def poll_batch(manifest_path, out, client, token, base, max_wait):
    manifest = read(manifest_path)
    rows = {r['page_id']: r for r in common.records(out)}
    batch_folder = manifest_path.parent
    start = time.monotonic()
    while True:
        data = await request_json(client, 'GET', base + '/extract-results/batch/' + manifest['batch_id'],
                                  token, batch_folder / '最近服务状态.json')
        items = data.get('extract_result') or []
        if isinstance(items, dict):
            items = list(items.values())
        mapped = {str(item.get('data_id') or Path(item.get('file_name') or '').stem): item for item in items}
        finished = 0
        for page_id in manifest['page_ids']:
            item = mapped.get(page_id)
            if item and item.get('state') in {'done', 'failed'}:
                await save_result(item, rows[page_id], out, client)
                finished += 1
        common.report(out)
        print(f'MinerU批次 {manifest_path.parent.name} 完成 {finished}/{len(manifest["page_ids"])}', flush=True)
        if finished == len(manifest['page_ids']):
            dump(batch_folder / '批次完成.json', {'completed_at': now(), 'batch_id': manifest['batch_id'],
                                                  'count': finished})
            return
        if time.monotonic() - start >= max_wait:
            raise MinerUError('轮询达到本次等待上限；批次身份已保存，可继续查询，禁止重新提交')
        await asyncio.sleep(10)


async def run(out, limit, batch_size, max_wait):
    token = os.getenv('MINERU_API_TOKEN') or os.getenv('MINERU_TOKEN')
    if not token:
        raise MinerUError('缺少MINERU_API_TOKEN或MINERU_TOKEN')
    base = 'https://mineru.net/api/v4'
    common.load_reader(out)
    shutil.copy2(__file__, out / '冻结MinerU上传脚本.py')
    candidates = [r for r in common.records(out) if r.get('shared_html_path')]
    if limit:
        if limit > MAX_HTML_FILES:
            raise MinerUError('用户设置HTML总量上限100，拒绝更大的limit')
        candidates = candidates[:limit]
    else:
        candidates = freeze_sample(out, candidates)
    submitted = set()
    manifests = sorted((out / 'MinerU服务批次').glob('*/批次身份.json'))
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=25), trust_env=True,
                                 follow_redirects=True) as client:
        # 已提交批次先继续查询，不重复申请上传和计算。
        for path in manifests:
            manifest = read(path)
            submitted.update(manifest['page_ids'])
            if not (path.parent / '批次完成.json').exists():
                await poll_batch(path, out, client, token, base, max_wait)
        pending = [r for r in candidates if r['page_id'] not in submitted]
        for start in range(0, len(pending), batch_size):
            group = pending[start:start + batch_size]
            used = submitted_count(out)
            if used + len(group) > MAX_HTML_FILES:
                raise MinerUError(f'提交预算保护：已有{used}份，本批{len(group)}份，将超过100，已拒绝')
            await submission_rate_gate(out, len(group))
            index = len(list((out / 'MinerU服务批次').glob('*/批次身份.json'))) + 1
            folder = out / 'MinerU服务批次' / f'批次{index:03d}'
            intent = folder / '提交意图.json'
            if intent.exists() and not (folder / '批次身份.json').exists():
                raise MinerUError('存在未确认结果的历史提交意图，请先核对服务状态，避免重复提交')
            payload = {'files': [{'name': r['page_id'] + '.html', 'data_id': r['page_id']} for r in group],
                       'model_version': 'MinerU-HTML'}
            dump(intent, {'at': now(), 'payload': payload,
                          'source_hashes': {r['page_id']: r['shared_html_sha256'] for r in group}})
            dump(out / 'MinerU提交预算.json', {'at': now(), 'hard_cap': MAX_HTML_FILES,
                  'reserved_file_submissions': submitted_count(out),
                  'remaining': MAX_HTML_FILES - submitted_count(out), 'includes_pilot': True})
            data = await request_json(client, 'POST', base + '/file-urls/batch', token,
                                      folder / '申请上传原始响应.json', payload)
            urls = data['file_urls']
            if len(urls) != len(group):
                raise MinerUError('上传URL数量与本地文件数量不同')
            manifest = {'batch_id': data['batch_id'], 'created_at': now(),
                        'page_ids': [r['page_id'] for r in group], 'uploaded_page_ids': []}
            identity = folder / '批次身份.json'
            dump(identity, manifest)
            for row, url in zip(group, urls):
                body = Path(row['shared_html_path']).read_bytes()
                assert sha(body) == row['shared_html_sha256']
                response = await client.put(url, content=body)
                dump(folder / f'{row["page_id"]}-上传记录.json',
                     {'at': now(), 'status': response.status_code, 'bytes': len(body), 'sha256': sha(body)})
                response.raise_for_status()
                manifest['uploaded_page_ids'].append(row['page_id'])
                dump(identity, manifest)
            await poll_batch(identity, out, client, token, base, max_wait)
    common.report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--max-wait', type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 50:
        parser.error('batch-size必须在1至50之间，且全部提交总量不超过100')
    if args.limit > MAX_HTML_FILES:
        parser.error('用户授权HTML总量上限100，拒绝limit超过100；未发送网络请求')
    load_dotenv(ROOT / 'backend/.env', override=True)
    try:
        asyncio.run(run(args.out.resolve(), args.limit, args.batch_size, args.max_wait))
    except Exception as exc:
        token = os.getenv('MINERU_API_TOKEN') or os.getenv('MINERU_TOKEN') or '未设置'
        error = (type(exc).__name__ + ': ' + str(exc)).replace(token, '[凭证已隐藏]')
        dump(args.out / 'MinerU服务运行失败.json', {'at': now(), 'error': error,
             'note': '服务配置或运行失败不等同目标网页正文不可用；尚未完成配对判定'})
        print(error, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
