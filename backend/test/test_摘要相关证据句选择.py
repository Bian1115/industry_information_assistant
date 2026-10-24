import json
import unittest
from unittest.mock import AsyncMock, patch

from service.agent_history import content_hash
from service.context_policy import NLISettings
from service.summary_sentence_evidence import (
    EvidenceSelectionError, parse_selection, sentence_catalog, selected_context,
    selection_batches, selection_messages, verify_relevant_claims,
)
from service.summary_message_verification import verify_message_output
from service.summary_claim_binding import verify_bound_claims
from test_摘要消息编号直接核验 import Model, PROSE


def context(text, pieces=None):
    payload = {'content': text}
    return {'sources': {'m': pieces if pieces is not None else [{'start': 0, 'text': text}]},
            'rows': {'m': {'kind': 'llm_output', 'payload': payload, 'content_hash': content_hash(payload)}}}


def claim(text='研究纳入200人。'):
    return {'claim_id': 'c1', 'section': 'SUMMARY', 'claim': text, 'source_ids': ['m']}


class Counter:
    def count_messages(self, messages):
        data = json.loads(messages[1]['content'])
        return 100 + sum(len(c['text']) for m in data['messages'] for c in m['sentences'])


def answer(messages, predicate=lambda s: True):
    data = json.loads(messages[1]['content'])
    evidence = [{'source_id': m['source_id'], 'sentence_ids': [s['sentence_id'] for s in m['sentences'] if predicate(s)]}
                for m in data['messages']]
    return json.dumps({'claim_id': data['claim_id'], 'evidence': evidence}, ensure_ascii=False)


class SentenceEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_offsets_read_exact_fragment_without_hidden_neighbors(self):
        text = '隐藏前文。该模型来自论文A。准确率90%。隐藏后文。'
        start = text.index('该'); end = text.index('隐藏后')
        ctx = context(text, [{'start': start, 'text': text[start:end]}])
        catalog = sentence_catalog(claim(), ctx)
        self.assertEqual(len(catalog), 2)
        self.assertTrue(all(text[c['start']:c['end']] == c['text'] for c in catalog))
        selected = selected_context(ctx, catalog)
        self.assertEqual(selected['sources']['m'], [{'start': start, 'text': text[start:end]}])

    def test_overlapping_supplied_pieces_do_not_duplicate_or_expand_scope(self):
        ctx = context('甲。乙。丙。外部。', [{'start': 0, 'text': '甲。乙。'}, {'start': 2, 'text': '乙。丙。'}])
        catalog = sentence_catalog(claim(), ctx)
        self.assertEqual([c['text'] for c in catalog], ['甲。', '乙。', '丙。'])

    def test_selection_never_inserts_unselected_middle_sentence(self):
        ctx = context('相关一。无关句。相关二。')
        catalog = sentence_catalog(claim(), ctx)
        narrowed = selected_context(ctx, [catalog[0], catalog[2]])
        self.assertEqual([p['text'] for p in narrowed['sources']['m']], ['相关一。', '相关二。'])

    def test_selected_neighbors_keep_original_whitespace_and_coreference(self):
        text = '论文A采用模型X。\n  该模型准确率90%。'
        ctx = context(text)
        narrowed = selected_context(ctx, sentence_catalog(claim(), ctx))
        self.assertEqual(narrowed['sources']['m'], [{'start': 0, 'text': text}])

    def test_hidden_whitespace_gap_is_not_added_to_selected_evidence(self):
        ctx = context('甲。\n乙。', [{'start': 0, 'text': '甲。'}, {'start': 3, 'text': '乙。'}])
        narrowed = selected_context(ctx, sentence_catalog(claim(), ctx))
        self.assertEqual([p['text'] for p in narrowed['sources']['m']], ['甲。', '乙。'])

    def test_tampered_source_is_rejected_before_model_call(self):
        ctx = context('原句。'); ctx['sources']['m'][0]['text'] = '篡改。'
        with self.assertRaises(ValueError): sentence_catalog(claim(), ctx)

    def test_wrong_claim_unknown_message_sentence_and_quote_are_rejected(self):
        candidates = sentence_catalog(claim(), context('证据。'))
        sid = candidates[0]['sentence_id']
        invalid = [
            {'claim_id': 'other', 'evidence': []},
            {'claim_id': 'c1', 'evidence': [{'source_id': 'other', 'sentence_ids': []}]},
            {'claim_id': 'c1', 'evidence': [{'source_id': 'm', 'sentence_ids': ['unknown']}]},
            {'claim_id': 'c1', 'evidence': [{'source_id': 'm', 'sentence_ids': [sid], 'quote': '改写'}]},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_selection(json.dumps(data), claim(), candidates)

    async def test_only_selected_original_sentences_reach_nli(self):
        ctx = context('模型属于论文A。准确率90%。论文B的准确率80%。')
        call = AsyncMock(side_effect=lambda m, stage: answer(m, lambda s: '论文B' not in s['text']))
        nli = AsyncMock(return_value={'status': 'passed'})
        report = await verify_relevant_claims([claim()], ctx, NLISettings(), call, verify=nli, counter=Counter())
        sent = nli.await_args.args[1]
        self.assertEqual([p['text'] for p in sent['sources']['m']], ['模型属于论文A。准确率90%。'])
        self.assertEqual(report['evidence_locator'], 'llm_selected_sentence_ids_nli_v1')
        self.assertEqual(call.await_args.args[1], 'claim_relevant_sentence_selection')

    async def test_relevant_counterevidence_is_not_dropped(self):
        ctx = context('支持。无关。相反。')
        call = AsyncMock(side_effect=lambda m, stage: answer(m, lambda s: s['text'] != '无关。'))
        with patch('service.nli_verifier.get_nli', return_value=Model()):
            result = await verify_relevant_claims([claim()], ctx, NLISettings(), call, counter=Counter())
        self.assertEqual(result['claims'][0]['status'], 'contradiction')
        self.assertEqual([c['text'] for c in result['evidence_selection']['selected_sentences']], ['支持。', '相反。'])

    async def test_empty_selection_is_missing_evidence_without_full_message_fallback(self):
        ctx = context('无关句。')
        call = AsyncMock(return_value=json.dumps({'claim_id': 'c1', 'evidence': []}))
        with patch('service.nli_verifier.get_nli') as model:
            report = await verify_relevant_claims([claim()], ctx, NLISettings(), call, counter=Counter())
        model.assert_not_called()
        self.assertEqual(report['claims'][0]['status'], 'missing_evidence')
        self.assertEqual(report['window_count'], 0)

    async def test_long_message_batches_keep_all_candidates_and_union_selections(self):
        ctx = context('第一句。第二句。第三句。')
        call = AsyncMock(side_effect=lambda m, stage: answer(m))
        nli = AsyncMock(return_value={'status': 'passed'})
        report = await verify_relevant_claims([claim()], ctx, None, call, verify=nli, counter=Counter(), input_budget=104)
        self.assertEqual(call.await_count, 3)
        self.assertEqual(len(report['evidence_selection']['selected_sentences']), 3)
        self.assertEqual(nli.await_args.args[1]['sources']['m'][0]['text'], '第一句。第二句。第三句。')

    async def test_batch_failure_does_not_use_partial_selection(self):
        ctx = context('第一句。第二句。')
        calls = 0
        async def call(messages, stage):
            nonlocal calls
            calls += 1
            return answer(messages) if calls == 1 else 'invalid JSON'
        nli = AsyncMock()
        with self.assertRaises(EvidenceSelectionError) as raised:
            await verify_relevant_claims([claim()], ctx, None, call, verify=nli, counter=Counter(), input_budget=104)
        nli.assert_not_awaited()
        self.assertEqual(len(raised.exception.evidence_selection_audit['batches']), 2)

    def test_oversize_sentence_is_explicit_error_not_truncated(self):
        candidates = sentence_catalog(claim(), context('很长的句子。'))
        with self.assertRaises(ValueError): list(selection_batches(claim(), candidates, Counter(), 101))

    async def test_current_entry_calls_selector_before_nli(self):
        ctx = context('研究纳入200人。无关句。')
        raw = json.dumps({'claims': [{'section': 'SUMMARY', 'claim': '研究纳入200人。', 'source_ids': ['m']}]})
        call = AsyncMock(side_effect=lambda m, stage: answer(m, lambda s: '200' in s['text']))
        with patch('service.qwen_token_counter.get_qwen_token_counter', return_value=Counter()), \
                patch('service.nli_verifier.get_nli', return_value=Model()):
            result = await verify_message_output(raw, PROSE, ctx, NLISettings(), call)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(call.await_count, 1)
        self.assertEqual(result['nli_reports'][0]['report']['window_count'], 1)

    async def test_failed_selection_keeps_other_claims_running_and_summary_incomplete(self):
        ctx = context('研究纳入200人。')
        claims = [claim(), {**claim(), 'claim_id': 'c2'}]
        binding = {'claims': claims, 'unresolved_claims': []}
        call = AsyncMock(side_effect=lambda m, stage: 'wrong' if json.loads(m[1]['content'])['claim_id'] == 'c1' else answer(m))
        with patch('service.qwen_token_counter.get_qwen_token_counter', return_value=Counter()), \
                patch('service.nli_verifier.get_nli', return_value=Model()):
            result = await verify_bound_claims(binding, ctx, NLISettings(), call_stage=call)
        self.assertEqual(result['status_label'], '核验未完成')
        self.assertEqual(result['nli_reports'][1]['status'], 'passed')
        self.assertIsNotNone(result['nli_reports'][0]['evidence_selection'])

    async def test_no_cited_sources_skips_selector_and_stays_unresolved(self):
        call = AsyncMock()
        result = await verify_relevant_claims([{**claim(), 'source_ids': []}], context('原文。'), NLISettings(), call)
        call.assert_not_awaited()
        self.assertEqual(result['claims'][0]['status'], 'missing_evidence')


if __name__ == '__main__': unittest.main()
