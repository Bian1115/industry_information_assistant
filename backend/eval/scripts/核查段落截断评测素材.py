"""只盘点冻结素材与字段，不计算段落截断率，不调用网络或模型。"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'backend/eval/data/段落截断评测素材核查-20261021'
SOURCE = ROOT / 'backend/eval/data/开发集网络检索与网页读取评估-20261008/冻结网页读取材料.json'
PAIRS = ROOT / 'backend/eval/data/网页正文与表格优化对照评估-20261018-首轮/候选表格净化版/逐页结果'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    pages = json.loads(SOURCE.read_text())
    candidates = []
    for p in pages:
        content = p['content']
        candidates.append({
            'id': p['id'], 'topic': p['topic'], 'url': p['url'],
            'source_path': relative(SOURCE), 'source_field': 'content',
            'content_chars': len(content), 'nonempty': bool(content.strip()),
            'sha256': sha(content.encode()),
            'hash_matches': sha(content.encode()) == p['content_sha256'],
            'sample_is_full': p['sample_is_full'],
            'history_path': p['history_path'], 'read_call_id': p.get('read_call_id'),
            'status': '待正文筛选与段落标注' if content.strip() else '空正文，不进入分块分母',
        })
    histories = []
    for path in sorted({p['history_path'] for p in pages}):
        raw = Path(path).read_bytes()
        text = raw.decode()
        rows = json.loads(text)
        histories.append({
            'path': relative(path), 'sha256': sha(raw), 'records': len(rows),
            'kinds': dict(Counter(r['kind'] for r in rows)),
            'field_literal_occurrences': {key: text.count('"' + key + '"') for key in
                ['source_spans', 'paragraph_ids', 'chunk_tokens', 'evidence_units']},
        })
    pairs = []
    for before in sorted(PAIRS.glob('*/放宽上限正文.txt')):
        after = before.parent / '生产门禁前正文.txt'
        if not after.exists():
            continue
        a, b = before.read_text(), after.read_text()
        pairs.append({
            'id': before.parent.name, 'before_path': relative(before), 'after_path': relative(after),
            'before_sha256': sha(before.read_bytes()), 'after_sha256': sha(after.read_bytes()),
            'before_chars': len(a), 'after_chars': len(b),
            'exact_prefix': a.startswith(b), 'shorter': len(b) < len(a),
            'nonempty_before': bool(a.strip()),
        })
    summary = {
        'date': '2026-10-21', 'timezone': 'Asia/Singapore',
        'scope': '指定五题冻结材料与指定公开网页历史运行的素材核查；非全仓轨迹普查',
        'source_path': relative(SOURCE), 'source_sha256': sha(SOURCE.read_bytes()),
        'frozen_records': len(pages), 'nonempty_records': sum(x['nonempty'] for x in candidates),
        'unique_nonempty_content_hashes': len({x['sha256'] for x in candidates if x['nonempty']}),
        'hash_matches': sum(x['hash_matches'] for x in candidates),
        'nonfull_judge_samples': sum(not x['sample_is_full'] for x in candidates),
        'history_files': histories,
        'paired_files': len(pairs), 'exact_prefix_pairs': sum(x['exact_prefix'] for x in pairs),
        'nonprefix_pairs': sum(not x['exact_prefix'] for x in pairs),
        'shorter_prefix_pairs': sum(x['exact_prefix'] and x['shorter'] for x in pairs),
        'measured': '素材条目数、哈希、字段出现次数与文本前缀关系',
        'not_evaluated': ['语义正文资格', '人工段落边界', '段落截断率', '历史chunk还原', '线上总体发生率'],
    }
    for name, rows in [('轨迹候选素材清单.jsonl', candidates), ('预算前后素材清单.jsonl', pairs)]:
        (OUT / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    (OUT / '素材核查汇总.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'history_files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
