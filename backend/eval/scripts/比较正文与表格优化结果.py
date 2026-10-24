"""对同一批冻结HTML的多个本地读取版本作配对统计，不修改历史预测。"""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'backend/eval/data/网页正文与表格优化对照评估-20261018-首轮'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    spec = importlib.util.spec_from_file_location('网页优化评分实现', REPO / 'backend/eval/scripts/评估公开网页读取质量.py')
    metric = importlib.util.module_from_spec(spec); sys.modules[spec.name] = metric; spec.loader.exec_module(metric)
    network_attempts = metric.install_network_guard()
    versions = {}
    for name in ['优化前', '候选初版', '候选收紧版', '候选表格净化版', '优化后']:
        folder = ROOT / name
        rows = json.loads((folder / '逐页指标汇总.json').read_text())
        assert len(rows) == 745 and len({r['id'] for r in rows}) == 745
        summary = json.loads((folder / '核心指标与分层汇总.json').read_text())
        normalized = {key: {'n': 0, 'hits': 0, 'pages': 0, 'pages_with_hit': 0} for key in ['with', 'without']}
        for row in rows:
            if row['dataset'] != 'WCXB分层200':
                continue
            pred = metric.visible((folder / '逐页结果' / row['id'] / '生产门禁前正文.txt').read_text())
            for key in normalized:
                snippets = row['scores']['production_pre_gate']['snippets'][key]
                seq = snippets['hits'] + snippets['misses']; hits = sum(metric.norm(s) in pred for s in seq)
                normalized[key]['n'] += len(seq); normalized[key]['hits'] += hits
                normalized[key]['pages'] += bool(seq); normalized[key]['pages_with_hit'] += hits > 0
        versions[name] = {'rows': {r['id']: r for r in rows}, 'summary': summary, 'normalized_snippets': normalized,
                          'output_size': {dataset: {'over_budget_pages': sum(r['chars']['production_pre_gate'] > 30000 for r in rows if r['dataset'] == dataset),
                                                   'max_chars': max(r['chars']['production_pre_gate'] for r in rows if r['dataset'] == dataset),
                                                   'mean_chars': statistics.mean(r['chars']['production_pre_gate'] for r in rows if r['dataset'] == dataset)}
                                          for dataset in ['WebMainBench545', 'WCXB分层200']},
                          'profiles': dict(Counter(r.get('extraction_diagnostics', {}).get('profile', 'baseline') for r in rows))}
    before, after = versions['优化前']['rows'], versions['优化后']['rows']
    pairs, mismatches = [], []
    for sid, old in before.items():
        new = after[sid]
        if old['input']['clean_html_sha256'] != new['input']['clean_html_sha256'] or old['input']['reference_sha256'] != new['input']['reference_sha256']:
            mismatches.append(sid)
        if not old['reference_chars']:
            continue
        a, b = old['scores']['production_pre_gate']['text'], new['scores']['production_pre_gate']['text']
        pairs.append({'id': sid, 'dataset': old['dataset'], 'url': old['url'], 'meta': old['meta'],
                      'before': a, 'after': b, 'delta': {key: b[key] - a[key] for key in ['precision', 'recall', 'f1']},
                      'before_accepted': old['production_result']['fetch_status'] == 'success',
                      'after_accepted': new['production_result']['fetch_status'] == 'success',
                      'selection': new.get('extraction_diagnostics', {}),
                      'before_table': old['scores']['production_pre_gate'].get('table', {}).get('grid_tables'),
                      'after_table': new['scores']['production_pre_gate'].get('table', {}).get('grid_tables')})
    assert not mismatches
    outcomes = {}
    for ds in ['WebMainBench545', 'WCXB分层200']:
        seq = [p for p in pairs if p['dataset'] == ds]
        outcomes[ds] = {'n': len(seq), 'improved_f1_over_0.5pp': sum(p['delta']['f1'] > .005 for p in seq),
                        'regressed_f1_over_0.5pp': sum(p['delta']['f1'] < -.005 for p in seq),
                        'other': sum(abs(p['delta']['f1']) <= .005 for p in seq),
                        'newly_accepted': [p['id'] for p in seq if p['after_accepted'] and not p['before_accepted']],
                        'no_longer_accepted': [p['id'] for p in seq if p['before_accepted'] and not p['after_accepted']]}
    table_pairs = [p for p in pairs if (p.get('before_table') or {}).get('eligible') and (p.get('after_table') or {}).get('eligible')]
    teds_pairs = [p for p in table_pairs if p['before_table'].get('teds_like') is not None and p['after_table'].get('teds_like') is not None]
    tables = {'paired_grid_pages': len(table_pairs), 'paired_teds_pages': len(teds_pairs),
              'paired_teds_before': statistics.mean(p['before_table']['teds_like'] for p in teds_pairs),
              'paired_teds_after': statistics.mean(p['after_table']['teds_like'] for p in teds_pairs),
              'structure_teds_before': statistics.mean(p['before_table']['structure_teds_like'] for p in teds_pairs),
              'structure_teds_after': statistics.mean(p['after_table']['structure_teds_like'] for p in teds_pairs)}
    result = {'date': '2026-10-18', 'versions': {k: {x: y for x, y in v.items() if x != 'rows'} for k, v in versions.items()},
              'paired_outcomes': outcomes, 'paired_table_metrics': tables, 'input_mismatches': mismatches,
              'network_attempts_during_comparison': network_attempts, 'paid_api_calls': 0,
              'evaluation_boundary': '同批数据用于分析与开发，再作配对回归；不是独立留出测试，不能据此宣称泛化提升。'}
    dump(ROOT / '优化前后对比汇总.json', result)
    dump(ROOT / '逐页配对变化.json', pairs)
    dump(ROOT / '改善与退化候选.json', {'improvements': sorted(pairs, key=lambda p: -p['delta']['f1'])[:20],
                                      'regressions': sorted(pairs, key=lambda p: p['delta']['f1'])[:20]})
    print(json.dumps({'outcomes': outcomes, 'tables': tables, 'top_regressions': [(p['id'], p['delta']['f1']) for p in sorted(pairs, key=lambda p: p['delta']['f1'])[:8]]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
