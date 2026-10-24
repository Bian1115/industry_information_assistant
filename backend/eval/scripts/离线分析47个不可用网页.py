#!/usr/bin/env python3
"""只读冻结抓取，归档 HTML 取证和编码恢复实验；不联网、不调用 MinerU。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'backend/eval/data/通用研究289网址同HTML与MinerU对照-20261017-首轮'
OUT = ROOT / 'backend/eval/data/通用研究47个不可用网页根因分析-20261017'


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    rows = json.loads((OUT / '逐页HTML取证清单.json').read_text())
    spec = importlib.util.spec_from_file_location('capture_common', Path(__file__).with_name('对照评估289网址与MinerU.py'))
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    module = common.load_reader(SOURCE)
    os.environ['WEB_READER_CACHE_DIR'] = str(OUT / '离线机制缓存')
    reader = module.WebReaderService()
    encoding_results = []
    structured = []
    alternate = []
    for row in rows:
        page_id = row['page_id']
        folder = SOURCE / '逐页采集' / page_id
        evidence = OUT / '逐页证据' / page_id
        html = (folder / '共享输入HTML.html').read_bytes().decode('utf-8')
        assert sha(html.encode()) == row['shared_html_sha256']
        soup = BeautifulSoup(html, 'lxml')
        structured_row = {'page_id': page_id, 'json_records': [], 'article_containers': [], 'media': []}
        for node in soup.find_all('script'):
            if node.get('type') == 'application/ld+json' or node.get('id') == '__NEXT_DATA__':
                try:
                    structured_row['json_records'].append(json.loads(node.get_text()))
                except (ValueError, TypeError):
                    structured_row.setdefault('invalid_json_count', 0)
                    structured_row['invalid_json_count'] += 1
                    # 少数站点 JSON-LD 带尾逗号，仅移除该语法错误，不补写内容。
                    try:
                        fixed = re.sub(r',\s*([}\]])', r'\1', node.get_text())
                        structured_row.setdefault('json_records_trailing_comma_only', []).append(json.loads(fixed))
                    except (ValueError, TypeError):
                        pass
        for node in soup.select('article, main, #Detailcontent, #ContentBody, #vsb_content, .v_news_content, .article-content, .text-preview, .doc-content'):
            structured_row['article_containers'].append({'tag': node.name, 'id': node.get('id'), 'class': node.get('class'), 'text': node.get_text(' ', strip=True)})
        for node in soup.find_all(['video', 'source', 'embed', 'object', 'canvas']):
            structured_row['media'].append({'tag': node.name, 'attrs': dict(node.attrs), 'text': node.get_text(' ', strip=True)[:200]})
        structured.append(structured_row)
        # 检查未被选中的历史静态页/浏览器 DOM；这里只重放提取，不请求新页面。
        for filename in ['静态HTML.html', '浏览器DOM.html']:
            path = folder / filename
            if not path.exists() or path.read_bytes().decode('utf-8') == html:
                continue
            text = path.read_bytes().decode('utf-8')
            extracted = reader._extract_html(text, row['url'])
            output_path = evidence / ('未选中' + filename.removesuffix('.html') + '重放正文.txt')
            output_path.write_text(extracted['content'])
            alternate.append({'page_id': page_id, 'html_path': str(path), 'html_sha256': sha(text.encode()),
                              'title': extracted['title'], 'chars': len(extracted['content']), 'content_path': str(output_path)})
        if row['replacement_chars'] < 50:
            continue
        raw_path = folder / '原始响应.bin'
        raw = raw_path.read_bytes()
        metadata = json.loads((folder / '采集结果.json').read_text())
        declarations = re.findall(rb'<meta[^>]{0,300}charset[^>]{0,120}>', raw, re.I)
        assert any(b'gb2312' in item.lower() for item in declarations)
        # 五页均声明 GB2312；用其兼容超集严格解码，不能悄悄丢弃非法字节。
        recovered = raw.decode('gb18030', errors='strict')
        # 仅修正文件编码声明，不改正文、节点或提取器。
        recovered = common.utf8_html(recovered)
        result = reader._extract_html(recovered, row['url'])
        (evidence / '按原始声明恢复编码HTML.html').write_text(recovered)
        (evidence / '恢复编码后基线正文.txt').write_text(result['content'])
        original = (folder / '基线待评正文.txt').read_text()
        enc = {'page_id': page_id, 'url': row['url'], 'expected_title': row['title'],
               'original_content_type': metadata.get('content_type'), 'meta_declarations': [b.decode('ascii') for b in declarations],
               'decode_codec': 'gb18030', 'decode_errors': 'strict', 'raw_response_path': str(raw_path), 'raw_sha256': sha(raw),
               'old_shared_html_sha256': row['shared_html_sha256'], 'recovered_html_sha256': sha(recovered.encode()),
               'old_html_replacements': html.count('\ufffd'), 'recovered_html_replacements': recovered.count('\ufffd'),
               'old_body_replacements': original.count('\ufffd'), 'recovered_body_replacements': result['content'].count('\ufffd'),
               'old_body_chars': len(original), 'recovered_body_chars': len(result['content']), 'recovered_title': result['title'],
               'extractor': result['extractor'], 'recovered_content_sha256': sha(result['content'].encode()),
               'content_path': str(evidence / '恢复编码后基线正文.txt'),
               'html_path': str(evidence / '按原始声明恢复编码HTML.html'),
               'network_requests': 0, 'mineru_submissions': 0,
               'scope': '冻结原始响应字节，仅替换解码步骤后重放冻结基线提取器；不是重新抓取或MinerU修复后结果'}
        encoding_results.append(enc)
    dump(OUT / '结构化正文与媒体取证.json', structured)
    dump(OUT / '未选中HTML离线重放结果.json', alternate)
    dump(OUT / '五页编码恢复机制实验.json', encoding_results)
    shutil.copy2(__file__, OUT / '冻结离线分析脚本.py')
    dump(OUT / '分析配置.json', {'date': '2026-10-17', 'source_run': str(SOURCE), 'scope': '最终配对判定两组均不可用的47页',
         'source_pair_sha256': sha((SOURCE / '逐页配对结果.json').read_bytes()),
         'input_count': len(rows), 'encoding_experiments': len(encoding_results), 'new_fetches': 0, 'mineru_submissions': 0,
         'business_code_changed': False, 'offline_script_sha256': sha(Path(__file__).read_bytes())})
    print(json.dumps({'input_count': len(rows), 'encoding_experiments': len(encoding_results), 'alternate_snapshots': len(alternate),
                      'recovered': [{k: x[k] for k in ['page_id', 'recovered_title', 'recovered_body_chars', 'recovered_body_replacements']} for x in encoding_results]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
