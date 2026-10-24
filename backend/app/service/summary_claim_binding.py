"""逐条绑定摘要论断：精确匹配、一次语义判断、一次定向修复；不核验证据。"""
from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from service.agent_history import canonical_json, content_hash, exception_details
from service.claimify import parse_prose, sentence_spans

PROMPTS = Path(__file__).parents[1] / 'prompts'


def candidate_spans(text, claim, limit=8):
    """相似度仅用于给判断模型提供候选，不能自动判定等价。"""
    spans = list(sentence_spans(text))
    ranges = {(0, len(text))}
    ranges.update(spans)
    ranges.update((m.start(), m.end()) for m in re.finditer(r'\S[\s\S]*?(?=\n\s*\n|\Z)', text))
    for i, (start, _) in enumerate(spans):
        for _, end in spans[i:min(i + 12, len(spans))]:
            ranges.add((start, end))
    normalized = lambda s: re.sub(r'\s+', '', s).replace('。', '.')
    target = normalized(claim)
    scored = []
    for start, end in ranges:
        while start < end and text[start].isspace(): start += 1
        while end > start and text[end - 1].isspace(): end -= 1
        if start == end: continue
        score = SequenceMatcher(None, target, normalized(text[start:end]), autojunk=False).ratio()
        scored.append((score, start, end))
    result = []
    used = set()
    for score, start, end in sorted(scored, key=lambda x: (-x[0], x[1], x[2])):
        if (start, end) in used: continue
        used.add((start, end))
        result.append({'candidate_id': f'span_{start}_{end}', 'start': start, 'end': end,
                       'text': text[start:end], 'retrieval_score': score})
        if len(result) == limit: break
    return result


def request_messages(prompt_name, payload):
    return [{'role': 'system', 'content': (PROMPTS / prompt_name).read_text()},
            {'role': 'user', 'content': canonical_json(payload)}]


def decision(output, candidates, *, repair=False, source_ids=None):
    data = json.loads(output)
    expected = {'status', 'candidate_id', 'reason'} | ({'source_ids'} if repair else set())
    if not isinstance(data, dict) or set(data) != expected or not isinstance(data['reason'], str) or not data['reason'].strip():
        raise ValueError('摘要绑定判断输出协议不合法')
    accepted = 'repaired' if repair else 'equivalent'
    statuses = {accepted, 'unresolved'} if repair else {accepted, 'different', 'uncertain'}
    if data['status'] not in statuses:
        raise ValueError('摘要绑定判断状态不合法')
    ids = {c['candidate_id'] for c in candidates}
    if ((data['status'] == accepted and data['candidate_id'] not in ids)
            or (data['status'] != accepted and data['candidate_id'] is not None)):
        raise ValueError('摘要绑定判断使用了未知候选或状态与候选不一致')
    if repair and data['source_ids'] != source_ids:
        raise ValueError('定向绑定修复不能修改来源消息编号')
    return data


async def bind_summary_claims(output, prose, context, call_stage, *, repair=True):
    data = json.loads(output)
    if isinstance(data, dict) and 'citations' in data:
        from service.summary_sentence_citations import bind_numbered_citations
        return bind_numbered_citations(data, prose, context)
    if (not isinstance(data, dict) or set(data) not in ({'claims'}, {'claims', 'excluded'})
            or not isinstance(data['claims'], list) or not isinstance(data.get('excluded', []), list)):
        raise ValueError('直接引用补齐须返回claims数组')
    parts = parse_prose(prose)
    bound, unresolved, log = [], [], []
    for index, raw in enumerate(data['claims']):
        claim_id = 'claim_' + content_hash([index, raw])[:16]
        entry = {'claim_id': claim_id, 'input_index': index, 'original': raw,
                 'semantic_calls': 0, 'repair_calls': 0}
        log.append(entry)
        try:
            if (not isinstance(raw, dict) or set(raw) != {'section', 'claim', 'source_ids'}
                    or raw['section'] not in parts or not isinstance(raw['claim'], str) or not raw['claim'].strip()):
                raise ValueError('论断字段不合法；模型不得返回quote')
            if (not isinstance(raw['source_ids'], list)
                    or any(not isinstance(s, str) or s not in context['sources'] for s in raw['source_ids'])):
                raise ValueError('论断引用了未提供的消息编号')
            ids = list(dict.fromkeys(raw['source_ids']))
            text = parts[raw['section']]
            start = text.find(raw['claim'])
            if start >= 0 and text.find(raw['claim'], start + 1) < 0:
                selected = {'start': start, 'end': start + len(raw['claim']), 'text': raw['claim']}
                method = 'exact'
            else:
                selected = None
                candidates = candidate_spans(text, raw['claim'])
                entry['candidates'] = candidates
                entry['semantic_calls'] = 1
                try:
                    answer = await call_stage(request_messages('摘要论断语义绑定判断.txt', {
                        'claim_id': claim_id, 'section': raw['section'], 'model_claim': raw['claim'],
                        'candidates': candidates,
                    }), 'claim_summary_equivalence')
                    entry['semantic_raw_output'] = answer
                    result = decision(answer, candidates)
                    entry['semantic_decision'] = result
                    if result['status'] == 'equivalent':
                        selected = next(c for c in candidates if c['candidate_id'] == result['candidate_id'])
                        method = 'semantic_equivalent'
                except Exception as exc:
                    entry['semantic_error'] = exception_details(exc)
                if selected is None and repair:
                    entry['repair_calls'] = 1
                    try:
                        answer = await call_stage(request_messages('摘要未绑定论断定向修复.txt', {
                            'claim_id': claim_id, 'section': raw['section'], 'original_claim': raw['claim'],
                            'fixed_source_ids': ids, 'candidates': candidates,
                            'binding_result': entry.get('semantic_decision', entry.get('semantic_error')),
                        }), 'claim_binding_targeted_repair')
                        entry['repair_raw_output'] = answer
                        result = decision(answer, candidates, repair=True, source_ids=ids)
                        entry['repair_decision'] = result
                        if result['status'] == 'repaired':
                            selected = next(c for c in candidates if c['candidate_id'] == result['candidate_id'])
                            method = 'targeted_repair'
                    except Exception as exc:
                        entry['repair_error'] = exception_details(exc)
                if selected is None:
                    raise ValueError('一次语义判断及允许的定向修复后，论断仍无法绑定摘要')
            # 语义绑定保留模型原论断；修复时使用候选原文，均保留可审计的摘要跨度。
            claim = selected['text'] if method == 'targeted_repair' else raw['claim']
            bound.append({'claim_id': claim_id, 'section': raw['section'], 'claim': claim,
                          'source_ids': ids, 'summary_start': selected['start'], 'summary_end': selected['end'],
                          'summary_original': selected['text'], 'model_claim': raw['claim'], 'binding_method': method})
            entry.update(status='bound', binding_method=method, summary_start=selected['start'], summary_end=selected['end'])
        except Exception as exc:
            entry.update(status='unresolved', error=exception_details(exc))
            unresolved.append({'claim_id': claim_id, 'input_index': index, 'original': raw,
                               'reason': entry['error'], 'stage': 'summary_binding'})
    return {'version': 'summary-semantic-binding-v1', 'claims': bound, 'unresolved_claims': unresolved,
            'binding_audit': log, 'excluded': data.get('excluded', []),
            'status': 'incomplete' if unresolved else 'bound',
            'semantic_calls': sum(e['semantic_calls'] for e in log),
            'repair_calls': sum(e['repair_calls'] for e in log),
            'verification_scope': 'summary_binding_only'}


async def verify_bound_claims(binding, context, settings, *, verify=None, call_stage=None):
    """逐条核验证据，绑定失败或单条运行错误不会阻断其他论断。"""
    if verify is None:
        if call_stage is not None:
            from service.summary_sentence_evidence import verify_relevant_claims
            async def verify(claims, context, settings):
                return await verify_relevant_claims(claims, context, settings, call_stage)
        else:
            # 保留旧离线实验的底层入口；当前直接引用入口和CLI传入call_stage。
            from service.summary_message_verification import verify_message_claims
            verify = verify_message_claims
    remaining = list(binding['unresolved_claims'])
    reports = []
    for claim in binding['claims']:
        try:
            result = await verify([claim], context, settings)
            reports.append({'claim_id': claim['claim_id'], 'status': result['status'], 'report': result})
            if result.get('execution_status') not in (None, 'completed'):
                remaining.append({'claim_id': claim['claim_id'], 'stage': 'nli', 'reason': 'incomplete_execution'})
            elif any(c.get('status') in {'unknown', 'missing_evidence', 'unverified'} for c in result.get('claims', [])):
                remaining.append({'claim_id': claim['claim_id'], 'stage': 'evidence', 'reason': 'insufficient_evidence'})
        except Exception as exc:
            selection_audit = getattr(exc, 'evidence_selection_audit', None)
            failure_stage = selection_audit.get('phase', 'nli') if selection_audit else 'nli'
            is_timeout = isinstance(exc, TimeoutError) or bool(selection_audit and
                selection_audit.get('error', {}).get('error_type') == 'TimeoutError')
            reports.append({'claim_id': claim['claim_id'], 'status': 'timeout' if is_timeout else 'error',
                            'error': exception_details(exc),
                            'evidence_selection': selection_audit})
            remaining.append({'claim_id': claim['claim_id'], 'stage': failure_stage, 'reason': exception_details(exc)})
    if remaining or not binding['claims']:
        status, label = 'incomplete', '核验未完成'
    elif all(r['status'] == 'passed' for r in reports):
        status, label = 'passed', '核验通过'
    else:
        status, label = 'rejected', '核验未通过'
    return {'status': status, 'status_label': label, 'binding': binding,
            'nli_reports': reports, 'unresolved_claims': remaining,
            'needs_review': bool(remaining) or status != 'passed'}
