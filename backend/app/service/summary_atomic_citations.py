"""论断a和候选引用b → Claimify论断c → 逐条映射b → 不足时临时检索。"""
from __future__ import annotations

import asyncio

from service.agent_history import canonical_json, content_hash, exception_details
from service.claimify import extract_claims, parse_prose
from service.summary_sections import section_names
from service.summary_claim_binding import bind_summary_claims
from service.summary_sentence_citations import merge_numbered_citations
from service.summary_sentence_evidence import (
    DEFAULT_INPUT_BUDGET, sentence_catalog, selection_batches, selection_messages,
    parse_selection, selected_context,
)
from service.summary_temporary_retrieval import TemporaryEvidenceIndex, TemporaryRetrievalSettings, unit_key

VERSION = 'summary-atomic-citations-v2'


async def audit_prose_atomic(agent, scope, prose, context, counter, attempt_id):
    """生产可选审计：先取得a+b，再复用原子论断入口；不改变正文提交协议。"""
    import time
    import uuid
    from config.llm_config import get_summary_model
    from service.agent_history import record_agent_message
    from service.agent_context import scoped_input_budget
    from service.context_policy import policy_for_scope, NLISettings
    from service.summary_message_verification import completion_messages, citation_context_batches, CITATION_OUTPUT_TOKENS

    policy = policy_for_scope(scope)
    budget = scoped_input_budget(scope, agent.name, CITATION_OUTPUT_TOKENS, summary=True)
    max_calls = policy.verification.max_stage_calls if policy.verification else 128
    totals = []

    async def call(messages, stage):
        input_tokens = counter.count_messages(messages)
        if input_tokens > budget or len(totals) >= max_calls:
            raise ValueError('原子论断审计超过输入或阶段调用预算')
        call_id = str(uuid.uuid4()); started = time.monotonic()
        entry = {'stage': stage, 'call_id': call_id, 'phase': 'claimify', 'summary_attempt_id': attempt_id,
                 'model': get_summary_model(), 'input_tokens': input_tokens, 'input_budget': budget,
                 'output_budget': CITATION_OUTPUT_TOKENS, 'prompt_version': VERSION}
        totals.append(entry)
        await record_agent_message(agent.name, 'summary_stage', entry, call_id)
        try:
            output = await agent._call_llm_impl(messages[0]['content'], messages[1]['content'], False,
                0.0, CITATION_OUTPUT_TOKENS, history_call_id=call_id, prepared_messages=messages,
                model_override=entry['model'])
            if counter.count_text(output) > CITATION_OUTPUT_TOKENS:
                raise ValueError('原子论断审计输出超过预算')
            await record_agent_message(agent.name, 'summary_output', {
                **entry, 'content': output, 'output_tokens': counter.count_text(output)}, call_id)
            entry['status'] = 'completed'
            return output
        except BaseException as exc:
            entry.update(status='error', **exception_details(exc))
            await record_agent_message(agent.name, 'summary_stage_error', entry, call_id)
            raise
        finally:
            entry['duration_seconds'] = time.monotonic() - started
            await record_agent_message(agent.name, 'summary_stage_end', entry, call_id)

    merged = {}
    payload = {'citations': []}
    for batch in citation_context_batches(prose, context, counter, budget):
        raw = await call(completion_messages(prose, batch), 'message_id_citation_completion')
        payload = merge_numbered_citations(merged, raw, prose, batch)
    report = await verify_atomic_output(canonical_json(payload), prose, context,
        policy.nli or NLISettings(), call, counter=counter, input_budget=budget, question=scope.contract)
    report.update(stage_totals=totals, citation_scope='summary_input_context',
                  context_input_hash=context['input_hash'], omission_check='not_run',
                  execution_status='incomplete' if report['status'] == 'incomplete' else 'completed')
    return report


async def extract_bound_claims(binding, prose, question, call_stage):
    """只拆已绑定的a；保留父论断和摘要原跨度，不把改写c做子串匹配。"""
    children, audit, excluded = [], [], []
    unresolved = list(binding['unresolved_claims'])
    calls = 0

    async def tracked(messages, stage):
        nonlocal calls
        calls += 1
        return await call_stage(messages, stage)

    # 全摘要仅用于指代与归属消歧，抽取对象仍是当前父论断。
    parts = parse_prose(prose)
    task = canonical_json({'task': question, 'summary_context': parts})
    for parent in binding['claims']:
        target = '\n\n'.join('## ' + section + '\n' +
                              (parent['claim'] if section == parent['section'] else 'None') for section in section_names(parts))
        try:
            extracted = await extract_claims(target, task, tracked)
            audit.append({'parent_claim_id': parent['claim_id'], 'extraction': extracted})
            for trace in extracted['sentences']:
                if trace['status'] == 'cannot_disambiguate':
                    unresolved.append({'parent_claim_id': parent['claim_id'], 'stage': 'claimify',
                                       'reason': 'cannot_disambiguate', 'sentence': trace['sentence']})
                elif trace['status'] == 'no_verifiable_claims':
                    excluded.append({'parent_claim_id': parent['claim_id'], 'sentence': trace['sentence'],
                                     'reason': 'no_verifiable_claims'})
            for number, child in enumerate(extracted['claims']):
                trace = extracted['sentences'][child['sentence_index']]
                origin = {
                    'parent_claim_id': parent['claim_id'], 'parent_claim': parent['claim'],
                    'model_claim': parent['model_claim'], 'section': parent['section'],
                    'summary_start': parent['summary_start'], 'summary_end': parent['summary_end'],
                    'summary_original': parent['summary_original'], 'binding_method': parent['binding_method'],
                    'parent_sentence_start': trace['start'], 'parent_sentence_end': trace['end'],
                    'parent_sentence': trace['sentence'], 'candidate_source_ids': parent['source_ids'],
                    **({k: parent[k] for k in ('summary_sentence_id', 'summary_hash')}
                       if 'summary_sentence_id' in parent else {}),
                }
                claim_id = 'atomic_' + content_hash([parent['claim_id'], number, child])[:16]
                children.append({**child, 'claim_id': claim_id, 'parent_claim_id': parent['claim_id'],
                                 'source_ids': list(parent['source_ids']), 'origin': origin})
        except Exception as exc:
            unresolved.append({'parent_claim_id': parent['claim_id'], 'stage': 'claimify',
                               'reason': exception_details(exc)})
    return {'claims': children, 'unresolved_claims': unresolved, 'excluded': excluded,
            'audit': audit, 'stage_calls': calls}


async def choose_evidence(claim, candidates, call_stage, counter, input_budget, *, phase):
    chosen = set(); audit = []
    for batch in selection_batches(claim, candidates, counter, input_budget):
        output = await call_stage(selection_messages(claim, batch), 'atomic_claim_' + phase)
        selected = parse_selection(output, claim, batch)
        chosen.update(unit_key(u) for u in selected)
        audit.append({'candidate_ids': [{'source_id': u['source_id'], 'sentence_id': u['sentence_id']} for u in batch],
                      'selected_ids': [{'source_id': u['source_id'], 'sentence_id': u['sentence_id']} for u in selected]})
    return [u for u in candidates if unit_key(u) in chosen], audit


def references_for(selected):
    """连续引句与组装表格分别标记；表格用source_spans恢复，不虚构连续quote。"""
    references = []
    for unit in selected:
        ref = {k: unit[k] for k in ('source_id', 'sentence_id', 'start', 'end', 'unit_type',
                                   'source_spans', 'fixed_prefix', 'body_rows', 'fixed_suffix',
                                   'quote_kind', 'field_fragmented') if k in unit}
        ref.update(quote=unit['text'], verification_scope='summary_input_context')
        references.append(ref)
    return references


def insufficient(report):
    if report.get('execution_status') not in (None, 'completed'):
        return False  # 运行错误不能伪装成证据不足后重试。
    return any(c.get('status') in {'missing_evidence', 'unknown', 'unverified'} for c in report.get('claims', []))


async def verify_atomic_output(output, prose, context, settings, call_stage, *, question='', repair=True,
                               verify=None, counter=None, input_budget=DEFAULT_INPUT_BUDGET,
                               retrieval_settings=None, index_factory=None):
    """完整新入口；verify为底层NLI回调，不应再次执行证据选择。"""
    if counter is None:
        from service.qwen_token_counter import get_qwen_token_counter
        counter = get_qwen_token_counter()
    if verify is None:
        from service.summary_message_verification import verify_message_claims
        verify = verify_message_claims
    retrieval_settings = retrieval_settings or TemporaryRetrievalSettings()
    index_factory = index_factory or TemporaryEvidenceIndex
    binding = await bind_summary_claims(output, prose, context, call_stage, repair=repair)
    extraction = await extract_bound_claims(binding, prose, question, call_stage)
    remaining = list(extraction['unresolved_claims']); reports = []; mapped = []
    index = None
    retrieval_audit = {'settings': retrieval_settings.model_dump(), 'built': False,
                       'released': False, 'storage': 'memory_only', 'queries': []}
    try:
        for claim in extraction['claims']:
            phase = 'candidate_mapping'
            entry = {**claim, 'references': [], 'candidate_references': [], 'status': 'pending'}
            mapped.append(entry)
            try:
                fits = lambda text: counter.count_text(text) <= retrieval_settings.chunk_tokens
                candidates = sentence_catalog(claim, context, fits=fits)
                selected, selection_audit = await choose_evidence(
                    claim, candidates, call_stage, counter, input_budget, phase='candidate_mapping')
                entry['candidate_references'] = references_for(selected)
                phase = 'nli'
                narrowed = selected_context(context, selected)
                initial = await verify([{**claim, 'source_ids': list(narrowed['sources'])}], narrowed, settings)
                result = initial
                fallback = None
                if insufficient(initial):
                    phase = 'temporary_retrieval'
                    if index is None:
                        index = index_factory(retrieval_settings, counter=counter)
                        async with asyncio.timeout(retrieval_settings.timeout_seconds):
                            await index.build(context)
                        retrieval_audit.update(built=True, chunk_count=len(index.units),
                            rerank_protocol=getattr(index.backend, 'rerank_protocol', 'injected_backend'))
                    async with asyncio.timeout(retrieval_settings.timeout_seconds):
                        retrieved, fallback = await index.retrieve(claim['claim'])
                    retrieved, extra_audit = await choose_evidence(
                        claim, retrieved, call_stage, counter, input_budget, phase='retrieved_mapping')
                    fallback.update(claim_id=claim['claim_id'], selection_batches=extra_audit)
                    retrieval_audit['queries'].append(fallback)
                    # 追加相关证据；第一次选中的反证/条件不因新的精排被覆盖。
                    combined = {unit_key(u): u for u in selected}
                    combined.update({unit_key(u): u for u in retrieved})
                    selected = list(combined.values())
                    phase = 'nli'
                    if retrieved:
                        narrowed = selected_context(context, selected)
                        result = await verify([{**claim, 'source_ids': list(narrowed['sources'])}], narrowed, settings)
                entry.update(references=references_for(selected), status=result['status'],
                             selection_batches=selection_audit, retrieval_used=fallback is not None)
                result.update(claim_id=claim['claim_id'], origin=claim['origin'],
                              evidence_selection={'selected_sentences': selected, 'initial_batches': selection_audit,
                                                  'retrieval': fallback},
                              evidence_locator=VERSION)
                reports.append({'claim_id': claim['claim_id'], 'parent_claim_id': claim['parent_claim_id'],
                                'status': result['status'], 'report': result,
                                **({'initial_report': initial} if fallback is not None else {})})
                if insufficient(result):
                    remaining.append({'claim_id': claim['claim_id'], 'stage': 'evidence', 'reason': 'insufficient_evidence'})
                elif result.get('execution_status') not in (None, 'completed'):
                    remaining.append({'claim_id': claim['claim_id'], 'stage': 'nli', 'reason': 'incomplete_execution'})
            except Exception as exc:
                entry.update(status='error', error=exception_details(exc))
                reports.append({'claim_id': claim['claim_id'], 'status': 'error', 'error': exception_details(exc), 'stage': phase})
                remaining.append({'claim_id': claim['claim_id'], 'stage': phase, 'reason': exception_details(exc)})
    finally:
        if index is not None:
            await index.close()
            retrieval_audit['released'] = True
    if remaining:
        status = 'incomplete'
    elif not extraction['claims']:
        status = 'no_verifiable_claims'
    else:
        status = 'passed' if all(r['status'] == 'passed' for r in reports) else 'rejected'
    return {'version': VERSION, 'status': status,
            'status_label': {'passed': '核验通过', 'rejected': '核验未通过', 'incomplete': '核验未完成',
                             'no_verifiable_claims': '无可核验论断'}[status],
            'binding': binding, 'claimify': extraction, 'mapped_claims': mapped,
            'nli_reports': reports, 'unresolved_claims': remaining,
            'temporary_retrieval': retrieval_audit, 'needs_review': status in {'incomplete', 'rejected'}}
