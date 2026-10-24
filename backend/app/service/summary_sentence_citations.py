"""程序生成摘要分句编号，模型只关联候选消息编号，原句及位置由程序恢复。"""
from __future__ import annotations

import json

from service.agent_history import content_hash
from service.claimify import parse_prose, split_sentences

VERSION = 'summary-sentence-citations-v1'


def summary_sentence_catalog(prose):
    """编号绑定本次完整摘要；相同原句在不同位置仍有不同编号。"""
    digest = content_hash(prose.strip())
    return [{**row, 'sentence_id': f'sum_{digest[:16]}_{index + 1:04d}', 'summary_hash': digest}
            for index, row in enumerate(split_sentences(parse_prose(prose)))]


def parse_numbered_citations(output, prose, context):
    data = json.loads(output) if isinstance(output, str) else output
    if not isinstance(data, dict) or set(data) != {'citations'} or not isinstance(data['citations'], list):
        raise ValueError('编号引用须仅返回citations数组')
    catalog = summary_sentence_catalog(prose)
    expected = {row['sentence_id']: row for row in catalog}
    selected = {}
    for item in data['citations']:
        if not isinstance(item, dict) or set(item) != {'sentence_id', 'source_ids'}:
            raise ValueError('编号引用只能包含sentence_id及source_ids，不得输出原句、栏目或quote')
        sid, sources = item['sentence_id'], item['source_ids']
        if not isinstance(sid, str) or sid not in expected:
            raise ValueError('未知摘要分句编号，或编号不属于当前摘要')
        if sid in selected:
            raise ValueError('摘要分句编号重复返回')
        if not isinstance(sources, list) or any(not isinstance(s, str) or s not in context['sources'] for s in sources):
            raise ValueError('候选引用超出本次实际提供的消息编号')
        selected[sid] = list(dict.fromkeys(sources))
    missing = set(expected) - set(selected)
    if missing:
        raise ValueError('编号引用遗漏摘要分句：' + ', '.join(sorted(missing)))
    return [{**row, 'source_ids': selected[row['sentence_id']]} for row in catalog]


def merge_numbered_citations(merged, output, prose, context):
    """先校验本批全部映射，再按稳定分句编号合并各证据批次的候选。"""
    rows = parse_numbered_citations(output, prose, context)
    for row in rows:
        sid = row['sentence_id']
        entry = merged.setdefault(sid, {'sentence_id': sid, 'source_ids': []})
        entry['source_ids'] = list(dict.fromkeys(entry['source_ids'] + row['source_ids']))
    return {'citations': [merged[row['sentence_id']] for row in rows]}


def bind_numbered_citations(output, prose, context):
    """无原句匹配或语义修复调用；文本、栏目、跨度全部读取程序目录。"""
    rows = parse_numbered_citations(output, prose, context)
    claims, audit = [], []
    for index, row in enumerate(rows):
        claim_id = 'claim_' + content_hash([row['summary_hash'], row['sentence_id']])[:16]
        claims.append({
            'claim_id': claim_id, 'section': row['section'], 'claim': row['sentence'],
            'source_ids': row['source_ids'], 'summary_sentence_id': row['sentence_id'],
            'summary_hash': row['summary_hash'], 'summary_start': row['start'], 'summary_end': row['end'],
            'summary_original': row['sentence'], 'model_claim': None, 'binding_method': 'sentence_id',
        })
        audit.append({
            'claim_id': claim_id, 'input_index': index, 'status': 'bound', 'binding_method': 'sentence_id',
            'original': {'sentence_id': row['sentence_id'], 'source_ids': row['source_ids']},
            'summary_start': row['start'], 'summary_end': row['end'], 'semantic_calls': 0, 'repair_calls': 0,
        })
    return {'version': VERSION, 'claims': claims, 'unresolved_claims': [], 'binding_audit': audit,
            'excluded': [], 'status': 'bound', 'semantic_calls': 0, 'repair_calls': 0,
            'verification_scope': 'summary_binding_only'}
