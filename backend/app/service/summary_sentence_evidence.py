"""编号实际输入中的句子，LLM只选编号，程序取回原文后交NLI。"""
from __future__ import annotations

import json
from pathlib import Path

from service.agent_history import canonical_json, exception_details
from service.agent_context import source_text
from service.summary_evidence_units import split_evidence_units, render_spans, validate_spans

VERSION = 'summary-relevant-sentences-v2'
# 32768上下文窗口，预留20000输出及1024余量，与直接引用回放一致。
DEFAULT_INPUT_BUDGET = 11744


class EvidenceSelectionError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.evidence_selection_audit = audit


def sentence_catalog(claim, context, *, fits=None):
    """字符偏移使用Python字符串下标；只读取生成摘要时实际输入的范围。"""
    result = []
    if fits is None:
        def fits(text):
            from service.qwen_token_counter import get_qwen_token_counter
            return get_qwen_token_counter().count_text(text) <= 384
    for sid in dict.fromkeys(claim['source_ids']):
        number = 0
        if sid not in context['sources'] or sid not in context['rows']:
            raise ValueError('证据句选择使用了未提供的source_id')
        full = source_text(context['rows'][sid])
        intervals = []
        for piece in context['sources'][sid]:
            start = piece['start']; end = start + len(piece['text'])
            if type(start) is not int or start < 0 or full[start:end] != piece['text']:
                raise ValueError('证据句选择的输入片段与消息原文不一致')
            intervals.append((start, end))
        # 合并重叠输入片段，不扩大到实际输入之外；同一位置仅编号一次。
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for start, end in merged:
            for unit in split_evidence_units(full, start, end, fits):
                number += 1
                result.append({'source_id': sid, 'sentence_id': f's{number:04d}',
                               **unit})
    return result


def selection_messages(claim, candidates):
    messages = {}
    for c in candidates:
        messages.setdefault(c['source_id'], []).append({'sentence_id': c['sentence_id'],
                                                      'unit_type': c.get('unit_type', 'sentence'), 'text': c['text']})
    payload = {'claim_id': claim['claim_id'], 'claim': claim['claim'],
               'messages': [{'source_id': sid, 'sentences': sentences} for sid, sentences in messages.items()]}
    return [{'role': 'system', 'content': (Path(__file__).parents[1] / 'prompts/摘要相关证据句编号选择.txt').read_text()},
            {'role': 'user', 'content': canonical_json(payload)}]


def selection_batches(claim, candidates, counter, input_budget):
    """只沿句边界分批；不静默丢掉超预算单句或后续候选。"""
    batch = []
    for candidate in candidates:
        if counter.count_messages(selection_messages(claim, batch + [candidate])) <= input_budget:
            batch.append(candidate)
            continue
        if batch:
            yield batch
        if counter.count_messages(selection_messages(claim, [candidate])) > input_budget:
            raise ValueError('单句及论断超过证据选择输入预算，未截断；需要进一步拆分该长句')
        batch = [candidate]
    if batch:
        yield batch


def parse_selection(output, claim, candidates):
    data = json.loads(output)
    if (not isinstance(data, dict) or set(data) != {'claim_id', 'evidence'}
            or data['claim_id'] != claim['claim_id'] or not isinstance(data['evidence'], list)):
        raise ValueError('证据句选择输出协议或claim_id不合法')
    available = {(c['source_id'], c['sentence_id']) for c in candidates}
    sources = {c['source_id'] for c in candidates}
    selected = set()
    for item in data['evidence']:
        if (not isinstance(item, dict) or set(item) != {'source_id', 'sentence_ids'}
                or not isinstance(item['source_id'], str) or not isinstance(item['sentence_ids'], list)):
            raise ValueError('证据句选择只能返回source_id和sentence_ids，不返回quote')
        if item['source_id'] not in sources:
            raise ValueError('证据句选择引用了本批之外的消息')
        for sentence_id in item['sentence_ids']:
            key = (item['source_id'], sentence_id) if isinstance(sentence_id, str) else None
            if key not in available:
                raise ValueError('证据句编号不属于本批提供的消息或句子')
            selected.add(key)
    # 顺序以原始消息位置为准，不采用模型返回的重排。
    return [c for c in candidates if (c['source_id'], c['sentence_id']) in selected]


def selected_context(context, selected):
    """只组合连续被选句；不把中间未选内容或隐藏邻文送入NLI。"""
    intervals = {}; structured = {}
    for c in selected:
        if c.get('source_spans'):
            full = source_text(context['rows'][c['source_id']])
            validate_spans(full, c)
            for span in c['source_spans']:
                cursor = span['start']
                for p in sorted(context['sources'][c['source_id']], key=lambda p: p['start']):
                    if p['start'] > cursor: break
                    cursor = max(cursor, p['start'] + len(p['text']))
                    if cursor >= span['end']: break
                if cursor < span['end']:
                    raise ValueError('表头或记录字段超出摘要实际输入范围')
            structured.setdefault(c['source_id'], []).append({k: v for k, v in c.items() if k != 'source_id'})
            continue
        intervals.setdefault(c['source_id'], []).append((c['start'], c['end']))
    sources = {}
    for sid, ranges in intervals.items():
        full = source_text(context['rows'][sid]); merged = []
        def supplied(start, end):
            cursor = start
            for piece in sorted(context['sources'][sid], key=lambda p: p['start']):
                left, right = piece['start'], piece['start'] + len(piece['text'])
                if left > cursor:
                    break
                if right > cursor:
                    cursor = right
                if cursor >= end:
                    return True
            return False
        for start, end in sorted(set(ranges)):
            if merged and (start <= merged[-1][1] or
                           (not full[merged[-1][1]:start].strip() and supplied(merged[-1][0], end))):
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        sources[sid] = [{'start': start, 'text': full[start:end]} for start, end in merged]
    for sid, pieces in structured.items():
        sources.setdefault(sid, []).extend(pieces)
    return {**context, 'sources': sources, 'rows': {sid: context['rows'][sid] for sid in sources}}


async def verify_relevant_claims(claims, context, settings, call_stage, *, verify=None, counter=None,
                                 input_budget=DEFAULT_INPUT_BUDGET):
    """一次论断的证据选择；候选超预算分批，所有批次完成后才进入NLI。"""
    if len(claims) != 1:
        raise ValueError('相关句核验按论断独立执行')
    claim = claims[0]
    audit = {'version': VERSION, 'claim_id': claim['claim_id'], 'phase': 'evidence_selection',
             'batches': [], 'selected_sentences': []}
    try:
        # 使用当前核验后端计算证据、论断及提示词的实际消息容量。
        model = None
        def fits(text):
            nonlocal model
            if model is None:
                from service.nli_verifier import get_nli
                model = get_nli(settings)
            from service.nli_verifier import pair_fits
            return pair_fits(model, text, claim['claim'], settings)
        candidates = sentence_catalog(claim, context, fits=fits)
        audit['sentence_catalog'] = candidates
        chosen = {}
        if candidates:
            if counter is None:
                from service.qwen_token_counter import get_qwen_token_counter
                counter = get_qwen_token_counter()
            for index, batch in enumerate(selection_batches(claim, candidates, counter, input_budget), 1):
                messages = selection_messages(claim, batch)
                entry = {'batch_index': index, 'input': messages}
                audit['batches'].append(entry)
                output = await call_stage(messages, 'claim_relevant_sentence_selection')
                entry['raw_output'] = output
                selected = parse_selection(output, claim, batch)
                entry['selected_ids'] = [{'source_id': c['source_id'], 'sentence_id': c['sentence_id']} for c in selected]
                for c in selected:
                    chosen[(c['source_id'], c['sentence_id'])] = c
        selected = [c for c in candidates if (c['source_id'], c['sentence_id']) in chosen]
        audit['selected_sentences'] = selected
        narrowed = selected_context(context, selected)
        audit['selected_pieces'] = narrowed['sources']
        if verify is None:
            from service.summary_message_verification import verify_message_claims
            verify = verify_message_claims
        # 无选中句时不回退到整条消息；交NLI汇总模块记为missing_evidence。
        audit['phase'] = 'nli'
        report = await verify([{**claim, 'source_ids': list(narrowed['sources'])}], narrowed, settings)
        audit['phase'] = 'completed'
        report.update(evidence_selection=audit, evidence_locator='llm_selected_sentence_ids_nli_v1',
                      original_source_ids=claim['source_ids'])
        return report
    except Exception as exc:
        audit['error'] = exception_details(exc)
        raise EvidenceSelectionError('相关证据句选择或核验未完成：' + str(exc), audit) from exc
