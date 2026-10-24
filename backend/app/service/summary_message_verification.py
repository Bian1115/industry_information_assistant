"""实验路径：直接输出论断和消息ID，由程序读取输入片段并定位NLI证据。

此模块尚未切换生产摘要生命周期；可供独立轨迹回放使用。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from service.agent_history import canonical_json
from service.claimify import parse_prose, sentence_spans
from service.summary_input_context import CONTEXT_SCOPE

VERSION = 'summary-message-ids-v1'
CITATION_OUTPUT_TOKENS = 20000


async def verify_message_output(output, prose, context, settings, call_stage, *, repair=True, verify=None,
                                claimify_atomic=False, **atomic_options):
    """当前直接引用路径入口：逐条语义绑定与修复，再核验证据并汇总未完成项。"""
    if claimify_atomic:
        from service.summary_atomic_citations import verify_atomic_output
        return await verify_atomic_output(output, prose, context, settings, call_stage,
                                          repair=repair, verify=verify, **atomic_options)
    if atomic_options:
        raise ValueError('原子论断选项仅在claimify_atomic=True时生效')
    data = json.loads(output)
    if isinstance(data, dict) and 'citations' in data:
        raise ValueError('分句编号引用需启用claimify_atomic，由Claimify筛选事实后核验')
    from service.summary_claim_binding import bind_summary_claims, verify_bound_claims
    binding = await bind_summary_claims(output, prose, context, call_stage, repair=repair)
    return await verify_bound_claims(binding, context, settings, verify=verify, call_stage=call_stage)


def summary_batches(prose, max_chars=450):
    """不调用模型分句；把栏目中的连续句段按引用输出预算分批。"""
    for section,text in parse_prose(prose).items():
        if text=='None':
            continue
        start=None
        end=None
        for left,right in sentence_spans(text):
            if start is not None and right-start>max_chars:
                yield {'section':section,'start':start,'end':end,'text':text[start:end]}
                start=None
            if start is None:start=left
            end=right
        if start is not None:
            yield {'section':section,'start':start,'end':end,'text':text[start:end]}


def completion_messages(prose, context, *, summary_sections=None, numbered=True):
    name = '摘要论断与消息编号.txt' if numbered else '摘要论断与消息编号旧版.txt'
    prompt = (Path(__file__).parents[1] / 'prompts' / name).read_text()
    if numbered:
        if summary_sections is not None:
            raise ValueError('编号协议使用完整摘要目录，不能以改写或局部正文重新编号')
        from service.summary_sentence_citations import summary_sentence_catalog
        summary = {'summary_sentences': [
            {'sentence_id': row['sentence_id'], 'section': row['section'], 'text': row['sentence']}
            for row in summary_sentence_catalog(prose)]}
    else:
        summary = {'summary_sections': summary_sections if summary_sections is not None else parse_prose(prose)}
    return [{'role':'system','content':prompt}, {'role':'user','content':canonical_json({
        **summary,
        'context_messages':[{'source_id':sid,'start':piece['start'],'text':piece['text']}
                            for sid,pieces in context['sources'].items() for piece in pieces],
    })}]


def citation_context_batches(prose, context, counter, budget, *, numbered=True):
    """为20000输出预留窗口；只对候选输入分批，不扩展实际摘要输入范围。"""
    batch={}
    def view(sources):return {**context,'sources':sources}
    fits=lambda sources:counter.count_messages(completion_messages(prose,view(sources), numbered=numbered))<=budget
    if not context['sources']:
        if not fits({}):
            raise ValueError('摘要本身已占满引用补齐输入预算')
        yield view({})
        return
    for sid,pieces in context['sources'].items():
        for piece in pieces:
            pending=piece['text'];offset=piece['start']
            while pending:
                candidate={key:list(value) for key,value in batch.items()}
                candidate.setdefault(sid,[]).append({'start':offset,'text':pending})
                if fits(candidate):
                    batch=candidate
                    break
                if batch:
                    yield view(batch)
                    batch={}
                    continue
                low,high=1,len(pending)
                if not fits({sid:[{'start':offset,'text':pending[:1]}]}):
                    raise ValueError('摘要本身已占满引用补齐输入预算')
                while low<high:
                    mid=(low+high+1)//2
                    if fits({sid:[{'start':offset,'text':pending[:mid]}]}):low=mid
                    else:high=mid-1
                yield view({sid:[{'start':offset,'text':pending[:low]}]})
                step=low-min(128,low//4)
                offset+=step;pending=pending[step:]
    if batch:yield view(batch)


def parse_message_claims(output, prose, context, *, summary_sections=None):
    data = json.loads(output)
    if isinstance(data, dict) and 'citations' in data:
        if summary_sections is not None:
            raise ValueError('编号协议使用完整摘要目录')
        from service.summary_sentence_citations import bind_numbered_citations
        result = bind_numbered_citations(data, prose, context)
        return {**result, 'uncovered': [], 'covered_content_ratio': 1.0 if result['claims'] else None}
    parts = summary_sections if summary_sections is not None else parse_prose(prose)
    if (not isinstance(data,dict) or set(data) not in ({'claims'},{'claims','excluded'})
            or not isinstance(data['claims'],list) or not isinstance(data.get('excluded',[]),list)):
        raise ValueError('直接引用补齐须返回claims数组')
    data.setdefault('excluded',[])  # 兼容首轮实验；新提示词不要求模型抄写任务性正文。
    covered = {name:set() for name in parts}
    claims, excluded = [], []
    for kind in ('claims','excluded'):
        for item in data[kind]:
            keys = {'section','claim','source_ids'} if kind=='claims' else {'section','text','reason'}
            if not isinstance(item,dict) or set(item)!=keys or item['section'] not in parts:
                raise ValueError('直接引用补齐字段不合法；模型不得返回quote')
            text = item['claim' if kind=='claims' else 'text']
            section = item['section']
            if not isinstance(text,str) or not text.strip() or text not in parts[section]:
                raise ValueError('论断或排除项不是摘要栏目的连续原文')
            start = parts[section].find(text)
            if parts[section].find(text,start+1)>=0:
                raise ValueError('摘要论断重复出现，无法唯一定位')
            covered[section].update(range(start,start+len(text)))
            if kind=='claims':
                ids = item['source_ids']
                if not isinstance(ids,list) or any(not isinstance(s,str) or s not in context['sources'] for s in ids):
                    raise ValueError('论断引用了不在实际摘要输入中的消息编号')
                claims.append({**item,'source_ids':list(dict.fromkeys(ids)),
                               'summary_start':start,'summary_end':start+len(text)})
            else:
                if not isinstance(item['reason'],str) or not item['reason'].strip():
                    raise ValueError('排除项必须说明原因')
                excluded.append(item)
    total = missed = 0
    uncovered = []
    for section,text in parts.items():
        if text=='None':
            continue
        significant = {i for i,c in enumerate(text) if c.isalnum()}
        total += len(significant)
        left = significant-covered[section]
        missed += len(left)
        if left:
            uncovered.append({'section':section,'uncovered_chars':len(left),
                              'text':''.join(text[i] for i in sorted(left))})
    return {'version':VERSION,'claims':claims,'excluded':excluded,'uncovered':uncovered,
            'covered_content_ratio':(total-missed)/total if total else None}


def source_windows(text, claim, model, settings):
    """完整句段优先；超长单句按字符二分形成token可容纳的连续重叠窗口。"""
    from service.nli_verifier import pair_fits
    fits = lambda s: pair_fits(model, s, claim, settings)
    if not fits(''):
        raise ValueError('论断本身超过NLI输入预算')
    spans = list(sentence_spans(text))
    windows = []
    current = None
    for start,end in spans:
        if not fits(text[start:end]):
            if current:
                windows.append(current)
                current = None
            cursor = start
            while cursor<end:
                low,high = 1,end-cursor
                if not fits(text[cursor:cursor+1]):
                    raise ValueError('单字符证据也无法放入NLI预算')
                while low<high:
                    mid=(low+high+1)//2
                    if fits(text[cursor:cursor+mid]): low=mid
                    else: high=mid-1
                windows.append((cursor,cursor+low))
                if cursor+low==end: break
                cursor += low-min(80,low//4)
        elif current and fits(text[current[0]:end]):
            current=(current[0],end)
        else:
            if current: windows.append(current)
            current=(start,end)
    if current: windows.append(current)
    if not windows and text.strip():
        raise ValueError('非空输入消息没有形成证据窗口')
    return [{'start':start,'end':end,'quote':text[start:end]} for start,end in windows]


def verify_message_claims_sync(claims, context, settings, *, max_windows=4096):
    from service.agent_context import source_text
    from service.nli_verifier import _INFERENCE_LOCK, NLIBusyError, get_nli, aggregate_claims, pair_fits, backend_metadata
    from service.verification_checks import source_type
    if not _INFERENCE_LOCK.acquire(blocking=False):
        raise NLIBusyError('本地NLI正在处理其他请求')
    started = time.monotonic()
    try:
        if settings.model_id != "qwen3-max":
            get_nli(settings)  # 旧冻结配置明确拒绝，不加载旧模型。
        model=get_nli(settings) if any(c['source_ids'] for c in claims) else None
        plan=[]
        for index,claim in enumerate(claims):
            for sid in claim['source_ids']:
                row=context['rows'][sid]
                full=source_text(row)
                for piece in context['sources'][sid]:
                    if piece.get('source_spans'):
                        from service.summary_evidence_units import structured_windows
                        fits = lambda text: pair_fits(model, text, claim['claim'], settings)
                        for window in structured_windows(piece, full, fits):
                            plan.append({'claim_index': index, 'source_id': sid, 'quote': window['text'],
                                         'start': window['start'], 'end': window['end'],
                                         'source_spans': window['source_spans'],
                                         'quote_kind': window['quote_kind'], 'unit_type': window['unit_type'],
                                         'field_fragmented': window.get('field_fragmented', False),
                                         'source_type': source_type(row), 'verification_scope': CONTEXT_SCOPE})
                            if len(plan) > max_windows:
                                raise ValueError('程序定位证据的NLI窗口数超过实验上限，未截断')
                        continue
                    if full[piece['start']:piece['start']+len(piece['text'])]!=piece['text']:
                        raise ValueError('程序读取的输入片段与保存的消息不一致')
                    for window in source_windows(piece['text'],claim['claim'],model,settings):
                        offset=piece['start']
                        plan.append({'claim_index':index,'source_id':sid,'quote':window['quote'],
                            'start':window['start']+offset,'end':window['end']+offset,
                            'source_type':source_type(row),'verification_scope':CONTEXT_SCOPE})
                        if len(plan)>max_windows:
                            raise ValueError('程序定位证据的NLI窗口数超过实验上限，未截断')
        predictions=[]
        for start in range(0,len(plan),settings.max_pairs):
            batch=plan[start:start+settings.max_pairs]
            predictions.extend(model.predict([(w['quote'],claims[w['claim_index']]['claim']) for w in batch],settings))
        all_windows=[{**w,**p} for w,p in zip(plan,predictions)]
        verified=[]
        selected_predictions=[]
        for index,claim in enumerate(claims):
            refs=[]
            for sid in claim['source_ids']:
                windows=[w for w in all_windows if w['claim_index']==index and w['source_id']==sid]
                scored=[w for w in windows if w['status']=='scored']
                if not scored:
                    raise ValueError('消息没有可完成NLI推理的证据窗口')
                # 证据定位允许相关窗口支持；保留最高矛盾窗口，避免只选支持片段。
                chosen=[max(scored,key=lambda w:w['scores']['entailment'])]
                contrary=max(scored,key=lambda w:w['scores']['contradiction'])
                if contrary['scores']['contradiction']>=settings.contradiction_threshold and contrary not in chosen:
                    chosen.append(contrary)
                for window in chosen:
                    refs.append({k:window[k] for k in ('source_id','quote','start','end','source_type','verification_scope',
                                                      'source_spans','quote_kind','unit_type','field_fragmented') if k in window})
                    selected_predictions.append({k:v for k,v in window.items() if k not in ('claim_index','source_id','quote')})
            verified.append({'section':claim['section'],'claim':claim['claim'],'references':refs})
        report=aggregate_claims(verified,selected_predictions,settings)
        if not claims:
            report.update(status='no_claims',needs_review=True)
        report.update(**backend_metadata(model, settings), version=VERSION,verification_scope=CONTEXT_SCOPE,claims_with_program_quotes=verified,
                      all_windows=all_windows,window_count=len(plan),duration_seconds=time.monotonic()-started,
                      evidence_locator='all_selected_message_windows_nli_v1',
                      threshold_calibration='not_calibrated')
        return report
    finally:
        _INFERENCE_LOCK.release()


async def verify_message_claims(claims, context, settings):
    return await asyncio.wait_for(asyncio.to_thread(verify_message_claims_sync,claims,context,settings),
                                  timeout=settings.timeout_seconds)
