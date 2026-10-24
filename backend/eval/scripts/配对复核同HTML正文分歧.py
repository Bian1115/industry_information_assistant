#!/usr/bin/env python3
"""保留独立盲评，使用同一Claude和原判定标准复核有无实质内容的配对分歧。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import time

import httpx
from dotenv import load_dotenv
from json_repair import repair_json

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('paired_judge', Path(__file__).with_name('盲评同HTML配对正文.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    load_dotenv(ROOT / 'backend/.env', override=True)
    folder = out / '配对分歧复核'
    folder.mkdir(exist_ok=True)
    pairs = module.read(out / '改善与退化样例.json')
    materials, mapping = [], {}
    for i, pair in enumerate(pairs, 1):
        methods = ['baseline', 'mineru']
        random.Random(20261017 + i).shuffle(methods)
        entries = []
        for j, method in enumerate(methods):
            anon = f'配对{i:02d}-文本{j+1}'
            r = pair[method]
            text = Path(r['content_path']).read_text()
            assert module.sha(text) == r['content_sha256']
            entries.append({'id': anon, 'content': text})
            mapping[anon] = {'page_id': pair['page_id'], 'method': method,
                             'original_anonymous_id': r['anonymous_id']}
        materials.append({'pair_id': f'配对{i:02d}', 'url': pair['url'], 'title': pair['title'],
                          'actual_page_title': pair['baseline'].get('actual_page_title'), 'outputs': entries})
    prompt = module.PROMPT.split('输入：')[0] + '''
本轮是同一URL、同一HTML、两个匿名抽取输出的一致性复核。输入不是独立页面列表，而是配对输出。之前的标签与方案身份均不提供。请仍严格使用上面的原始可用性规则，分别给每份输出判定，并检查两份之间的口径一致性。
特别核对：1.预期标题中的年份与本次内容年份不同时，两份都应使用同一目标匹配标准；不能只对其中一份用“滚动更新首页”豁免。2.若两份含相同实质段落，不能在一份里称其空洞销售说明，在另一份里又称其有实质目标摘要；必须核对完整文本中该段是否确实存在。3.区分真正恢复了原方案没有的实质内容，与仅减少导航、图片链接或排版变化；若两份可用性不同，指出足以改变判定的具体内容差异及同一规则依据。
返回严格JSON：{"items":[{"id":"配对编号-文本编号","label":"usable|partial|unusable|unknown","category":"按原规则","reason":"中文依据及与同对另一文本一致性说明","quote":"本份content中的连续原句，至多180字符"}],"pair_analysis":[{"pair_id":"配对编号","same_substantive_content":true,"year_policy":"中文说明","difference_is_actual_content_recovery":false,"reason":"配对差异及依据"}]}。
输入：''' + json.dumps(materials, ensure_ascii=False)
    module.dump(folder / '冻结匿名配对材料.json', materials)
    module.dump(folder / '匿名配对映射.json', mapping)
    (folder / '复核提示词.txt').write_text(prompt)
    shutil.copy2(__file__, folder / '复核执行脚本.py')
    dest = folder / 'Claude原始复核响应.json'
    if dest.exists():
        packet = module.read(dest)
        assert packet['prompt_sha256'] == module.sha(prompt)
        answer = packet['result']
    else:
        base = os.environ['CLAUDE_RELAY_BASE_URL'].rstrip('/')
        url = base + '/messages' if base.endswith('/v1') else base + '/v1/messages'
        token = os.environ['CLAUDE_RELAY_API_KEY']
        payload = {'model': os.environ['CLAUDE_RELAY_MODEL'], 'max_tokens': 5000,
                   'messages': [{'role': 'user', 'content': prompt}]}
        for attempt in range(1, 4):
            start = time.perf_counter()
            response_text = ''
            try:
                with httpx.Client(timeout=httpx.Timeout(180, connect=20), trust_env=True) as client:
                    response = client.post(url, json=payload, headers={
                        'Authorization': 'Bearer ' + token, 'anthropic-version': '2023-06-01'})
                response_text = response.text.replace(token, '[凭证已隐藏]')
                raw = json.loads(response_text)
                if response.status_code != 200 or raw.get('stop_reason') != 'end_turn':
                    raise ValueError('Claude配对复核HTTP或响应完整性异常')
                text = ''.join(b.get('text', '') for b in raw['content'] if b.get('type') == 'text')
                answer = json.loads(repair_json(text))
                module.validate(answer, [{'id': entry['id'], 'content': entry['content']}
                                         for pair in materials for entry in pair['outputs']])
                packet = {'judged_at': module.now(), 'model': payload['model'], 'actual_model': raw.get('model'),
                          'prompt': prompt, 'prompt_sha256': module.sha(prompt), 'raw_response': raw,
                          'result': answer, 'usage': raw.get('usage'),
                          'elapsed_seconds': round(time.perf_counter() - start, 3),
                          'rubric_change': '原始正文可用性定义不变，增加同URL配对一致性核查；不是新增抓取或新HTML处理'}
                module.dump(dest, packet)
                break
            except Exception as exc:
                module.dump(folder / f'失败尝试{attempt}.json', {'error': str(exc).replace(token, '[凭证已隐藏]'),
                            'raw_response': response_text, 'elapsed_seconds': time.perf_counter() - start})
                if attempt == 3:
                    raise
                time.sleep(2)
    initial = module.read(out / '逐条正文盲评结果.json')
    lookup = {(r['page_id'], r['method']): r for r in initial}
    changes = []
    for assessment in answer['items']:
        identity = mapping[assessment['id']]
        row = lookup[(identity['page_id'], identity['method'])]
        changes.append({'page_id': row['page_id'], 'method': row['method'],
                        'before': row['assessment'], 'after': assessment})
        row['independent_initial_assessment'] = row['assessment']
        row['assessment'] = assessment
        row['content_usable'] = assessment['label'] in {'usable', 'partial'}
        row['review_source'] = str(dest)
    module.dump(folder / '复核修正明细.json', changes)
    module.dump(out / '逐条正文配对复核后结果.json', initial)
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
