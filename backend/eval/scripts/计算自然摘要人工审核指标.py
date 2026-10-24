"""按导出的人工裁决复算引用支持和信息保留，未决/未标项不填充为金标。"""
import argparse
import json
from collections import Counter
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def div(a, b): return a / b if b else None


def score(data, annotations):
    if annotations.get('schema') != data['version'] or annotations.get('dataset_id') != data['dataset_id']:
        raise ValueError('人工标注与原始审核数据身份不一致')
    known = {r['id'] for kind in ('atoms', 'parents', 'coverage') for r in data[kind]}
    items = annotations.get('annotations')
    if not isinstance(items, list): raise ValueError('缺少annotations数组')
    if len({r['id'] for r in items}) != len(items) or any(r['id'] not in known for r in items):
        raise ValueError('存在重复或未知标注编号')
    humans = {r['id']: r.get('human', {}) for r in items}
    results = {'date': datetime.now(ZoneInfo('Asia/Singapore')).isoformat(), 'label_origin': 'human',
               'dataset_id': data['dataset_id'], 'total_atoms': len(data['atoms']), 'support_metrics': {}}
    for scope in ('support', 'selected_support'):
        cm, excluded = Counter(), Counter()
        for atom in data['atoms']:
            truth = humans.get(atom['id'], {}).get(scope)
            pred = atom['system_status']
            if truth not in {'supported', 'insufficient', 'contradicted'}:
                excluded['unlabeled_or_uncertain'] += 1; continue
            if pred not in {'supported', 'unknown', 'contradiction'}:
                excluded['execution_error_or_no_nli'] += 1; continue
            # 问题论断为正类；系统未通过为预测正。
            problem, flagged = truth != 'supported', pred != 'supported'
            cm['TP' if problem and flagged else 'FN' if problem else 'FP' if flagged else 'TN'] += 1
        tp, fp, tn, fn = (cm[k] for k in ('TP', 'FP', 'TN', 'FN'))
        results['support_metrics'][scope] = {'positive_class': 'insufficient_or_contradicted',
            'confusion_matrix': dict(cm), 'included': tp+fp+tn+fn, 'excluded': dict(excluded),
            'problem_precision': div(tp, tp+fp), 'problem_recall': div(tp, tp+fn),
            'correct_false_rejection_rate': div(fp, fp+tn), 'problem_false_pass_rate': div(fn, tp+fn)}
    fidelity = Counter(humans.get(r['id'], {}).get('fidelity') or 'unlabeled' for r in data['atoms'])
    extraction = Counter(humans.get(r['id'], {}).get('extraction') or 'unlabeled' for r in data['parents'])
    retention, excluded = Counter(), Counter()
    for r in data['coverage']:
        h = humans.get(r['id'], {}); importance, status = h.get('importance'), h.get('coverage')
        if importance not in {'critical', 'useful'}: excluded['importance_unlabeled_uncertain_or_not_important'] += 1; continue
        if status not in {'preserved', 'partial', 'missing', 'contradicted'}: excluded['coverage_unlabeled_or_uncertain'] += 1; continue
        retention[status] += 1
    results.update(fidelity_counts=dict(fidelity), parent_extraction_counts=dict(extraction),
                   retention={'included': sum(retention.values()), 'counts': dict(retention), 'excluded': dict(excluded),
                              'full_preservation_rate': div(retention['preserved'], sum(retention.values()))},
                   note='指标仅适用于人工已裁决子集；null表示分母为0，不表示0%。同题摘要存在相关性。')
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = score(json.loads(args.data.read_text()), json.loads(args.annotations.read_text()))
    with args.output.open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(args.output)
