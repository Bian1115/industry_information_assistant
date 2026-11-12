"""保留历史18道中文题，补12道官方中文原题，生成可追溯的验证集。"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007'
DEFAULT_OUTPUT = ROOT / 'backend/eval/data/DeepResearch中文验证集-20261108'


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def select_tasks(rows, old_ids, seed):
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows) or not set(old_ids) <= by_id.keys():
        raise ValueError('官方题号重复或历史题号缺失')
    retained = [i for i in old_ids if by_id[i]['language'] == 'zh']
    if len(retained) != 18:
        raise ValueError('历史清单不符合18道中文题口径')
    candidates = sorted([r['id'] for r in rows if r['language'] == 'zh' and r['id'] not in old_ids])
    added = random.Random(seed).sample(candidates, 12)
    return [by_id[i] for i in retained + added], retained, added, candidates


def build(source, output, desktop, seed):
    if output.exists() or (desktop and desktop.exists()):
        raise FileExistsError('输出目录已存在；为避免覆盖历史数据，请使用新的目录')
    previous = json.loads((source / '随机抽样清单.json').read_text())
    # 准备程序先验证官方文件，再从这些文件提取原题、细则及参考报告。
    for entry in previous['source_files']:
        if entry['saved_as'] in {'官方题目.jsonl', '官方评分标准.jsonl', '官方清洗后参考报告.jsonl'}:
            actual = hashlib.sha256((source / entry['saved_as']).read_bytes()).hexdigest()
            if actual != entry['sha256']:
                raise ValueError('官方快照校验失败：' + entry['saved_as'])
    rows = read_rows(source / '官方题目.jsonl')
    tasks, retained, added, candidates = select_tasks(rows, previous['sample_order'], seed)
    criteria = {r['id']: r for r in read_rows(source / '官方评分标准.jsonl')}
    references = {r['id']: r for r in read_rows(source / '官方清洗后参考报告.jsonl')}
    reference_prompt_differences = []
    for task in tasks:
        if criteria[task['id']]['prompt'] != task['prompt']:
            raise ValueError(f"题{task['id']}的原题与细则不对应")
        if references[task['id']]['prompt'] != task['prompt']:
            reference_prompt_differences.append({'id': task['id'], 'query_prompt': task['prompt'],
                'reference_prompt': references[task['id']]['prompt'],
                'policy': '保留官方原文差异；本轮只用原题与细则给生成报告评分，不用参考报告作事实答案'})
    output.mkdir(parents=True)
    grading = output / '评分专用材料'
    grading.mkdir()
    files = {'验证题目.jsonl': tasks,
             '评分专用材料/官方评分细则.jsonl': [criteria[t['id']] for t in tasks],
             '评分专用材料/官方参考报告.jsonl': [references[t['id']] for t in tasks]}
    for name, data in files.items():
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data), encoding='utf-8')
    manifest = {
        'dataset_name': 'DeepResearch验证集', 'dataset_version': '中文30题-20261108',
        'created_date': '2026-11-08', 'timezone': 'Asia/Singapore', 'split': 'validation',
        'purpose': '全部30题用于失败归因与开发优化；另行选取30道不重叠测试题',
        'source_directory': str(source), 'source_commit': previous['source_commit'],
        'seed': seed, 'sampling': '保留历史中文题的原顺序；从题号排序后的其余官方中文题中random.Random(seed).sample(...,12)',
        'original_sample_ids': previous['sample_order'], 'retained_chinese_ids': retained,
        'replaced_english_ids': [i for i in previous['sample_order'] if i not in retained],
        'additional_candidate_ids': candidates, 'added_chinese_ids': added,
        'validation_ids': [t['id'] for t in tasks], 'language_counts': {'zh': 30},
        'remaining_chinese_count': sum(r['language'] == 'zh' and r['id'] not in {t['id'] for t in tasks} for r in rows),
        'test_selection_status': '未抽取；相同题库剩余20道中文，若测试集也要30道中文需要扩展题源',
        'prompt_rewritten': False, 'translated': False,
        'historical_exposure': '保留18题曾用于参考报告评分；新增12题仅保证不在旧30题中，不宣称从未用于其他项目实验',
        'reference_prompt_differences': reference_prompt_differences,
        'file_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files},
    }
    save(output / '抽样清单.json', manifest)
    overview = ['# DeepResearch验证集', '', '30道官方中文原题；全部用于验证和调优。正式测试集后续另选，不与本集重复。', '']
    for index, task in enumerate(tasks, 1):
        overview += [f"## {index} 官方题号{task['id']} {task['topic']}", '', task['prompt'], '']
    (output / '题目一览.md').write_text('\n'.join(overview), encoding='utf-8')
    (output / '使用说明.md').write_text(
        '# DeepResearch验证集使用说明\n\n研究服务只读取`验证题目.jsonl`中的原题。'
        '`评分专用材料`仅由离线评分器读取，不传给研究服务。\n\n'
        '本集保留旧30题中的18道中文题，再补12道官方中文原题，未翻译或改写。'
        '全部30题用于失败分析和优化，不再拆20/10。测试集后续另选30题；相同官方题库剩余20道中文，'
        '若测试集也要求全中文，需要补充其他题源。\n\n'
        f'历史实验原始目录：`{source}`。仓库归档：`{output}`。'
        '桌面文件是便于查看和使用的独立副本；修改任一副本后需检查抽样清单中的文件指纹，不能无记录地继续评测。\n', encoding='utf-8')
    if desktop:
        shutil.copytree(output, desktop)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--desktop', type=Path, default=Path.home() / 'Desktop/DeepResearch验证集')
    parser.add_argument('--seed', type=int, default=20261108)
    args = parser.parse_args()
    result = build(args.source, args.output, args.desktop, args.seed)
    print(json.dumps({'output': str(args.output), 'desktop': str(args.desktop),
                      'retained': result['retained_chinese_ids'], 'added': result['added_chinese_ids']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
