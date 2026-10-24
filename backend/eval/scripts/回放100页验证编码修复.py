#!/usr/bin/env python3
"""用冻结原字节和已有浏览器响应回放读取链路，不进行网络请求。"""
import asyncio
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import types

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'backend/eval/data/通用研究289网址同HTML与MinerU对照-20261017-首轮'
OUT = ROOT / 'backend/eval/data/网页编码修复与三份资源补救验证-20261017'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def replay(row, module, common, phase):
    folder = SOURCE / '逐页采集' / row['page_id']
    meta = json.loads((folder / '采集结果.json').read_text())
    target = OUT / '100页离线回放' / row['page_id'] / phase
    target.mkdir(parents=True, exist_ok=True)
    reader = module.WebReaderService()
    original_extract = reader._extract_html
    captured = {}

    async def download(url):
        path = folder / '原始响应.bin'
        if path.exists():
            body = path.read_bytes()
            assert sha(body) == meta['response_sha256']
            # 原对照实验的共享输入协议：原字节解码后，统一UTF8及meta声明。
            html = common.utf8_html(reader._decode_html(body, meta.get('content_type', '')))
            captured['static_html'] = html
            return meta.get('final_url') or url, 'text/html; charset=utf-8', html.encode(), meta.get('attempts', 1)
        status = meta.get('download_http_status')
        request = httpx.Request('GET', url)
        if status:
            response = httpx.Response(status, request=request)
            raise httpx.HTTPStatusError(f'档案HTTP {status}', request=request, response=response)
        if 'timeout' in str(meta.get('download_error', '')).lower():
            raise httpx.ReadTimeout('档案超时', request=request)
        raise httpx.NetworkError('原抓取无静态响应，回放已记录网络失败', request=request)

    def extract(html, url):
        value = original_extract(html, url)
        captured['static_output'] = value['content']
        return value

    async def browser(url, title):
        path = folder / '浏览器原始正文.txt'
        if not path.exists():
            return None
        text = path.read_text()
        captured['browser_output'] = text
        return module.WebReadResult(url=url, final_url=url, title=title, content=text,
                                    extractor='crawl4ai-sidecar', fetch_status='success')

    reader._download_with_retry = download
    reader._extract_html = extract
    reader._read_with_crawl4ai = browser
    result = await reader._read_uncached(row['url'], row['title'])
    browser_used = result.recovery_action.startswith('browser')
    before = captured.get('browser_output', result.content) if browser_used else captured.get('static_output', result.content)
    from service.web_document_structure import limit_body_preserving_tables
    before = limit_body_preserving_tables(before.replace('\x00', '').strip(), reader.max_chars)
    (target / '待评正文.txt').write_text(before)
    dump(target / '程序结果.json', result.to_dict())
    selected_html = (folder / '浏览器DOM.html').read_bytes().decode() if browser_used else captured.get('static_html', '')
    (target / '选中HTML.html').write_text(selected_html)
    return {'page_id': row['page_id'], 'phase': phase, 'content_path': str(target / '待评正文.txt'),
            'content_sha256': sha(before.encode()), 'chars': len(before), 'fetch_status': result.fetch_status,
            'failure_category': result.failure_category, 'quality_status': result.body_quality.get('status'),
            'recovery_action': result.recovery_action, 'input_source': 'browser_dom' if browser_used else 'static_html',
            'shared_html_path': str(target / '选中HTML.html'), 'shared_html_sha256': sha(selected_html.encode())}


async def main():
    load_dotenv(ROOT / 'backend/.env')
    os.environ['WEB_READER_CACHE_DIR'] = str(OUT / '回放隔离缓存')
    pkg = types.ModuleType('service')
    pkg.__path__ = [str(ROOT / 'backend/app/service')]
    sys.modules['service'] = pkg
    from service import web_reader_service as new
    audit = []
    for filename in ['web_document.py', 'web_document_structure.py']:
        active = (ROOT/'backend/app/service'/filename).read_text()
        frozen = (SOURCE/'冻结基线源码'/filename).read_text()
        current_tree, frozen_tree = ast.parse(active), ast.parse(frozen)
        # 工作区另有来源引用功能新增辅助函数，读取器未导入它；不覆盖用户改动。
        ignored = []
        if filename == 'web_document.py':
            ignored = [n.name for n in current_tree.body if isinstance(n, ast.FunctionDef) and n.name=='source_available']
            current_tree.body = [n for n in current_tree.body if not (isinstance(n, ast.FunctionDef) and n.name=='source_available')]
        assert ast.dump(current_tree) == ast.dump(frozen_tree), filename
        audit.append({'file': filename, 'active_sha256': sha(active.encode()), 'frozen_sha256': sha(frozen.encode()),
                      'reader_dependencies_ast_equal': True, 'unrelated_added_functions': ignored})
    dump(OUT / '读取器依赖差异审计.json', audit)
    old = load_module('service.frozen_reader', SOURCE / '冻结基线源码/web_reader_service.py')
    common = load_module('capture_common', Path(__file__).with_name('对照评估289网址与MinerU.py'))
    pairs = json.loads((SOURCE / '逐页配对结果.json').read_text())
    results = []
    for pair in pairs:
        before = await replay(pair, old, common, '修复前冻结实现')
        after = await replay(pair, new, common, '修复后实现')
        original = Path(pair['baseline']['content_path']).read_text()
        results.append({'page_id': pair['page_id'], 'url': pair['url'], 'title': pair['title'],
                        'original_label': pair['baseline']['assessment']['label'], 'original_usable': pair['baseline']['content_usable'],
                        'old_replay_matches_original': sha(original.encode()) == before['content_sha256'],
                        'content_changed': before['content_sha256'] != after['content_sha256'],
                        'input_source_changed': before['input_source'] != after['input_source'], 'before': before, 'after': after})
    dump(OUT / '100页编码修复离线回放结果.json', results)
    summary = {'count': len(results), 'old_replay_exact_matches': sum(x['old_replay_matches_original'] for x in results),
               'changed_pages': [x['page_id'] for x in results if x['content_changed']],
               'unchanged_pages': sum(not x['content_changed'] for x in results),
               'route_changes': [x['page_id'] for x in results if x['input_source_changed']],
               'network_requests': 0, 'mineru_calls': 0,
               'reader_settings': {k:getattr(new.WebReaderService(), k) for k in ['min_chars','max_chars']}}
    dump(OUT / '100页编码回放核对汇总.json', summary)
    shutil.copy2(__file__, OUT / '冻结100页回放脚本.py')
    for name, folder in [('修复后读取器源码', ROOT/'backend/app/service'), ('修复前读取器源码', SOURCE/'冻结基线源码')]:
        target = OUT / name
        target.mkdir(exist_ok=True)
        for f in ['web_reader_service.py','web_document.py','web_document_structure.py']:
            shutil.copy2(folder/f, target/f)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    assert summary['count'] == summary['old_replay_exact_matches'] == 100


if __name__ == '__main__':
    asyncio.run(main())
