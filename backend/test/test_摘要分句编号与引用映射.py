"""分句编号协议、证据批次范围及Claimify集成；不调用真实模型。"""
import json
import unittest
from unittest.mock import AsyncMock

from service.agent_history import canonical_json, content_hash
from service.claimify import parse_prose
from service.context_policy import ContextPolicy, SummaryLifecycleSettings, InputBudget
from service.summary_claim_binding import bind_summary_claims
from service.summary_sentence_citations import (
    summary_sentence_catalog, parse_numbered_citations, merge_numbered_citations, bind_numbered_citations,
)
from service.summary_message_verification import (
    completion_messages, citation_context_batches, verify_message_output,
)
from service.summary_atomic_citations import audit_prose_atomic
from test_摘要原子论断与临时检索 import context, prose, Counter, nli_report, stage_call
from test_智能体独立历史 import MemoryStore, HistoryScope, make_agent, bind_agent_history


def numbered(body, sources=None):
    rows = summary_sentence_catalog(body)
    return canonical_json({'citations': [
        {'sentence_id': row['sentence_id'], 'source_ids': (sources or {}).get(row['sentence_id'], [])}
        for row in rows]})


class NumberedTests(unittest.IsolatedAsyncioTestCase):
    def test_catalog_distinguishes_identical_sentences_and_preserves_offsets(self):
        body = prose('研究纳入200人。\n\n研究纳入200人。')
        rows = summary_sentence_catalog(body)
        self.assertEqual(rows, summary_sentence_catalog(body))
        self.assertEqual(len({r['sentence_id'] for r in rows}), 2)
        self.assertNotEqual(rows[0]['start'], rows[1]['start'])
        parts = parse_prose(body)
        for row in rows:
            self.assertEqual(parts[row['section']][row['start']:row['end']], row['sentence'])

    def test_model_input_catalog_and_output_protocol_do_not_request_original_text(self):
        body = prose('研究纳入200人。')
        request = completion_messages(body, context({'m': '原文。'}))
        payload = json.loads(request[1]['content'])
        self.assertNotIn('summary_sections', payload)
        self.assertEqual(set(payload['summary_sentences'][0]), {'sentence_id', 'section', 'text'})
        self.assertIn('不提取、抄写或改写摘要原句', request[0]['content'])
        self.assertIn('逐一返回目录中的每个sentence_id', request[0]['content'])

    async def test_binding_restores_text_without_any_model_binding_calls(self):
        body = prose('研究纳入200人。研究纳入200人。')
        rows = summary_sentence_catalog(body)
        raw = numbered(body, {rows[1]['sentence_id']: ['m']})
        call = AsyncMock(side_effect=AssertionError('编号绑定不应调用模型'))
        result = await bind_summary_claims(raw, body, context({'m': '研究纳入200人。'}), call)
        call.assert_not_awaited()
        self.assertEqual(result['semantic_calls'], 0)
        self.assertEqual(result['repair_calls'], 0)
        self.assertEqual([c['summary_start'] for c in result['claims']], [r['start'] for r in rows])
        self.assertEqual(result['claims'][1]['source_ids'], ['m'])
        self.assertIsNone(result['claims'][1]['model_claim'])
        self.assertTrue(all(c['binding_method'] == 'sentence_id' for c in result['claims']))

    def test_unknown_or_stale_summary_ids_are_rejected(self):
        body = prose('甲研究。')
        for raw in [numbered(prose('乙研究。')),
                    canonical_json({'citations': [{'sentence_id': 'invented', 'source_ids': []}]})]:
            with self.assertRaises(ValueError): parse_numbered_citations(raw, body, context({}))

    def test_missing_duplicate_or_extra_text_fields_are_rejected(self):
        body = prose('甲研究。乙研究。')
        valid = json.loads(numbered(body))
        variants = [ {'citations': valid['citations'][:1]},
                     {'citations': valid['citations'] + valid['citations'][:1]} ]
        for key in ['claim', 'section', 'quote', 'text']:
            data = json.loads(numbered(body));data['citations'][0][key] = '不得输出'
            variants.append(data)
        variants.append({'claims': [], **valid})
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_numbered_citations(data, body, context({}))

    def test_source_ids_must_belong_to_the_current_evidence_batch(self):
        body = prose('甲研究。');sid = summary_sentence_catalog(body)[0]['sentence_id']
        raw = numbered(body, {sid: ['outside']})
        with self.assertRaises(ValueError): parse_numbered_citations(raw, body, context({'inside': '原文。'}))
        with self.assertRaises(ValueError):
            parse_numbered_citations(numbered(body, {sid: [123]}), body, context({'inside': '原文。'}))

    def test_merging_keeps_empty_entries_and_all_batches_without_partial_invalid_update(self):
        body = prose('甲研究。乙研究。');rows = summary_sentence_catalog(body);merged = {}
        merge_numbered_citations(merged, numbered(body, {rows[0]['sentence_id']: ['a']}), body, context({'a': '甲。'}))
        result = merge_numbered_citations(merged, numbered(body, {rows[0]['sentence_id']: ['b']}), body, context({'b': '乙。'}))
        self.assertEqual(result['citations'][0]['source_ids'], ['a', 'b'])
        self.assertEqual(result['citations'][1]['source_ids'], [])
        before = canonical_json(merged)
        with self.assertRaises(ValueError):
            merge_numbered_citations(merged, numbered(body, {rows[0]['sentence_id']: ['unknown']}), body, context({}))
        self.assertEqual(canonical_json(merged), before)

    def test_empty_context_still_emits_one_request_and_preserves_unsupported_sentences(self):
        body = prose('甲研究。');ctx = context({})
        batches = list(citation_context_batches(body, ctx, Counter(), 10000))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]['sources'], {})
        bound = bind_numbered_citations(numbered(body), body, ctx)
        self.assertEqual(bound['claims'][0]['source_ids'], [])
        self.assertEqual(bound['claims'][0]['claim'], '甲研究。')

    def test_source_chunking_uses_the_same_summary_ids_in_every_request(self):
        body = prose('甲研究。乙研究。');ctx = context({'m': '长原文' * 300})
        counter = Counter();budget = counter.count_messages(completion_messages(body, context({}))) + 180
        batches = list(citation_context_batches(body, ctx, counter, budget))
        self.assertGreater(len(batches), 1)
        expected = [r['sentence_id'] for r in summary_sentence_catalog(body)]
        for batch in batches:
            messages = completion_messages(body, batch)
            self.assertLessEqual(counter.count_messages(messages), budget)
            self.assertEqual([r['sentence_id'] for r in json.loads(messages[1]['content'])['summary_sentences']], expected)

    async def test_numbered_path_requires_claimify_before_nli(self):
        body = prose('待办：回读论文。');verify = AsyncMock()
        with self.assertRaisesRegex(ValueError, 'claimify_atomic'):
            await verify_message_output(numbered(body), body, context({}), None, AsyncMock(), verify=verify)
        verify.assert_not_awaited()

    async def test_numbered_atomic_claims_keep_parent_ids_and_filter_plans(self):
        parent = '研究A纳入200人，并报告准确率90%。'
        children = ['研究A纳入200人。', '研究A报告准确率90%。']
        body = prose(parent + '\n待办：回读论文B。')
        rows = summary_sentence_catalog(body);ctx = context({'m1': children[0], 'm2': children[1]})
        normal = stage_call(children, lambda claim, unit: claim == unit['text'])
        async def call(messages, stage):
            if stage == 'claimify_selection' and 'Sentence:\n\n待办：' in messages[1]['content']:
                return canonical_json({'status': 'no_verifiable_claims', 'sentence': '', 'reason': '纯待办'})
            return await normal(messages, stage)
        verify = AsyncMock(return_value=nli_report('supported'))
        report = await verify_message_output(numbered(body, {rows[0]['sentence_id']: ['m1', 'm2']}),
            body, ctx, None, call, claimify_atomic=True, verify=verify, counter=Counter())
        self.assertEqual(report['status'], 'passed', report['unresolved_claims'])
        self.assertEqual([c['claim'] for c in report['mapped_claims']], children)
        self.assertEqual([[r['source_id'] for r in c['references']] for c in report['mapped_claims']], [['m1'], ['m2']])
        self.assertEqual(verify.await_count, 2)
        for child in report['mapped_claims']:
            self.assertEqual(child['origin']['summary_sentence_id'], rows[0]['sentence_id'])
            self.assertEqual(child['origin']['summary_original'], parent)
        self.assertEqual(report['claimify']['excluded'][0]['sentence'], '待办：回读论文B。')

    async def test_bad_sentence_id_cannot_reach_claimify_or_nli(self):
        body = prose('研究A。');call = AsyncMock();verify = AsyncMock()
        with self.assertRaises(ValueError):
            await verify_message_output(numbered(prose('另一摘要。')), body, context({}), None, call,
                claimify_atomic=True, verify=verify, counter=Counter())
        call.assert_not_awaited();verify.assert_not_awaited()

    async def test_production_atomic_audit_requests_and_receives_only_ids(self):
        from unittest.mock import patch
        child = '研究A纳入200人。';body = prose(child)
        store = MemoryStore();policy = ContextPolicy(group='编号协议测试', semantic_verification=False,
            summary_budget=InputBudget(context_window=65536, max_input=64000),
            summary_lifecycle=SummaryLifecycleSettings(claimify_enabled=True, atomic_citations=True))
        scope = HistoryScope(store, 'numbered', 'session', None, context_policy=policy.snapshot())
        agent = make_agent();ctx = context({'m': child});ctx['input_hash'] = content_hash(ctx['sources'])
        normal = stage_call([child]);raw_outputs = []
        async def llm(*args, **kwargs):
            messages = kwargs['prepared_messages']
            payload = json.loads(messages[1]['content']) if messages[1]['content'].startswith('{') else {}
            if 'summary_sentences' in payload:
                rows = payload['summary_sentences']
                raw = canonical_json({'citations': [{'sentence_id': r['sentence_id'], 'source_ids': ['m']} for r in rows]})
                raw_outputs.append(json.loads(raw));return raw
            if 'Sentence:\n\n' in messages[1]['content']:
                system = messages[0]['content']
                stage = 'claimify_decomposition' if '"claims":[' in system else (
                    'claimify_disambiguation' if 'resolved|cannot_disambiguate' in system else 'claimify_selection')
            else: stage = 'atomic_claim_candidate_mapping'
            return await normal(messages, stage)
        agent._call_llm_impl = AsyncMock(side_effect=llm)
        with bind_agent_history(scope), patch('service.summary_message_verification.verify_message_claims',
                new=AsyncMock(return_value=nli_report('supported'))):
            report = await audit_prose_atomic(agent, scope, body, ctx, Counter(), 'attempt')
        self.assertEqual(report['status'], 'passed', report['unresolved_claims'])
        self.assertEqual(set(raw_outputs[0]['citations'][0]), {'sentence_id', 'source_ids'})
        self.assertEqual(report['binding']['semantic_calls'], 0)
        self.assertEqual(report['mapped_claims'][0]['origin']['summary_original'], child)

    def test_legacy_replay_still_uses_its_explicit_text_protocol(self):
        messages = completion_messages(prose('旧摘要。'), context({}), numbered=False)
        self.assertIn('summary_sections', json.loads(messages[1]['content']))
        self.assertIn('claim必须逐字摘自', messages[0]['content'])


if __name__ == '__main__': unittest.main()
