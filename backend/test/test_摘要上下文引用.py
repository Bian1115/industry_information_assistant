"""摘要上下文编号、候选范围和忠实性核验；全部使用模型桩。"""
import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

from test_智能体独立历史 import MemoryStore, HistoryScope, CharCounter, make_agent, bind_agent_history
from test_Claimify摘要论断抽取 import prose, answer, decomposed, responses
from service.agent_context import source_text, parse_summary, previous_summary_sources, summarize_history, REPLAY_KINDS
from service.agent_history import content_hash
from service.claimify import prose_messages, extract_claims, pack_summary
from service.context_policy import ContextPolicy, NLISettings, VerificationSettings
from service.summary_citations import citation_messages, parse_citation_result, summarize_with_citations
from service.summary_input_context import number_input_context, locate_context_quote, CONTEXT_KIND, CONTEXT_SCOPE
from service.nli_verifier import verify_claims_sync


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = MemoryStore()
        self.scope = HistoryScope(self.store, '上下文引用测试', 'session', None,
                                  context_policy=ContextPolicy(group='上下文引用测试', semantic_verification=False).snapshot())
        self.agent = make_agent()
        self.row = await self.store.append(self.scope, self.agent.name, 'llm_output',
                                           {'content': '未提供的前文。研究纳入200人。未提供的后文。'}, 'same-call')

    async def context(self, payload):
        request = prose_messages(payload)
        context = await number_input_context(request, self.scope, self.agent.name, 'generation-call',
                                             {self.row['source_id']: self.row})
        return request, context

    async def test_each_message_has_id_even_with_same_call_id(self):
        other = await self.store.append(self.scope, self.agent.name, 'tool_output', {'result': '网页结果'}, 'same-call')
        request = prose_messages({'new_records': [{'source_id': r['source_id'], 'text': source_text(r)}
                                                   for r in (self.row, other)]})
        context = await number_input_context(request, self.scope, self.agent.name, 'call',
                                             {r['source_id']: r for r in (self.row, other)})
        self.assertEqual(set(context['sources']), {self.row['source_id'], other['source_id']})
        self.assertNotEqual(self.row['source_id'], other['source_id'])

    async def test_previous_and_intermediate_summaries_have_actual_input_ids(self):
        old, middle = prose('旧摘要事实。'), prose('中间摘要事实。')
        request, context = await self.context({'previous_summary': old, 'summary_parts': [middle],
                                                'latest_contract': {'query': '只关注安全性'}})
        payload = json.loads(request[1]['content'])
        self.assertEqual(len(context['sources']), 3)
        self.assertNotIn(self.row['source_id'], context['sources'])
        for sid, text in [(payload['previous_summary_source_id'], old),
                          (payload['summary_part_source_ids'][0], middle)]:
            row = await self.store.read_source(self.scope, self.agent.name, sid)
            self.assertEqual(source_text(row), text)
            self.assertNotIn(row['kind'], REPLAY_KINDS)
            self.assertEqual(row['payload']['generation_call_id'], 'generation-call')
        self.assertEqual(context['input_hash'], content_hash(request))

    async def test_prose_repair_reuses_original_context_and_excludes_invalid_output(self):
        request, first = await self.context({'previous_summary': prose('旧事实。')})
        repair = prose_messages({'original_request': json.loads(request[1]['content']), 'invalid_output': '伪造事实。'})
        second = await number_input_context(repair, self.scope, self.agent.name, 'repair', {})
        self.assertEqual(first['sources'], second['sources'])
        self.assertNotIn('伪造事实', str(second['sources']))
        snapshots = [r for r in (await self.store.load(self.scope, self.agent.name))['messages'] if r['kind'] == CONTEXT_KIND]
        self.assertEqual(len(snapshots), 1)

    async def test_quote_and_adjacent_context_are_bounded_to_actual_fragment(self):
        quote = '研究纳入200人。'
        start = source_text(self.row).index(quote)
        _, context = await self.context({'new_records': [{'source_id': self.row['source_id'], 'start': start, 'text': quote}]})
        item = {'section': 'SUMMARY', 'claim': quote, 'source_ids': list(context['sources']), 'citation_scope': CONTEXT_SCOPE}
        excerpts = [{'source_id': self.row['source_id'], 'start': start, 'text': quote}]
        ref = parse_citation_result(json.dumps({'section': 'SUMMARY', 'claim': quote,
                  'references': [{'source_id': self.row['source_id'], 'quote': quote}]}), item, excerpts,
                  program_positions=True)[0]
        located = locate_context_quote(ref, context)
        self.assertEqual(located['context'], quote)
        self.assertEqual((located['start'], located['end']), (start, start + len(quote)))
        self.assertEqual(located['verification_scope'], CONTEXT_SCOPE)
        self.assertNotIn('未提供', located['context'])
        with self.assertRaisesRegex(ValueError, '超出'):
            locate_context_quote({**ref, 'start': 0, 'end': start}, context)

    async def test_unknown_message_wrong_offset_and_tampered_hash_are_rejected(self):
        cases = [{'source_id': '论文编号', 'text': '研究纳入200人。'},
                 {'source_id': self.row['source_id'], 'text': '研究纳入200人。', 'start': 0}]
        for fragment in cases:
            with self.subTest(fragment=fragment), self.assertRaises(ValueError):
                await self.context({'new_records': [fragment]})
        self.row['payload']['content'] = '被篡改'
        with self.assertRaisesRegex(ValueError, '哈希'):
            await self.context({'new_records': [{'source_id': self.row['source_id'], 'text': '被篡改'}]})

    async def test_unprovided_old_source_and_reformatted_quote_cannot_be_cited(self):
        _, context = await self.context({'previous_summary': prose('研究纳入200人。')})
        sid = next(iter(context['sources']))
        item = {'section': 'SUMMARY', 'claim': '研究纳入200人。', 'source_ids': [sid], 'citation_scope': CONTEXT_SCOPE}
        excerpts = [{'source_id': sid, **p} for p in context['sources'][sid]]
        prompt = citation_messages(item, excerpts)[0]['content']
        self.assertIn('不是对话轮次', prompt)
        for source_id, quote in [(self.row['source_id'], item['claim']), (sid, '研究纳入 200 人。')]:
            with self.subTest(source_id=source_id, quote=quote), self.assertRaises(ValueError):
                parse_citation_result(json.dumps({**item, 'references': [{'source_id': source_id, 'quote': quote}]}),
                                      item, excerpts, program_positions=True)

    async def test_context_snapshot_can_be_reloaded_after_covered_cursor(self):
        claim = '研究纳入200人。'
        request, context = await self.context({'previous_summary': prose(claim)})
        sid = json.loads(request[1]['content'])['previous_summary_source_id']
        quote_start = source_text(context['rows'][sid]).index(claim)
        extraction = await extract_claims(prose(claim), '', AsyncMock(side_effect=responses(claim)))
        summary = pack_summary(prose(claim), extraction, [{'section': 'SUMMARY', 'claim': claim, 'sentence_index': 0,
                    'references': [{'source_id': sid, 'quote': claim, 'start': quote_start, 'end': quote_start + len(claim)}]}])
        await self.store.save_summary(self.scope, self.agent.name, expected_version=0, covered_seq=self.row['seq'],
                                      summary=summary, metadata={})
        saved = await self.store.load(self.scope, self.agent.name)
        self.assertGreater(context['rows'][sid]['seq'], saved['covered_seq'])
        self.assertEqual(await previous_summary_sources(self.scope, self.agent.name, saved), {sid: [claim]})

    async def test_snapshot_id_cannot_be_reused_with_changed_text_or_other_scope(self):
        request, _ = await self.context({'previous_summary': prose('旧事实。')})
        changed = copy.deepcopy(request)
        payload = json.loads(changed[1]['content'])
        payload['previous_summary'] = prose('伪造事实。')
        changed[1]['content'] = json.dumps(payload)
        with self.assertRaises(ValueError):
            await number_input_context(changed, self.scope, self.agent.name, 'call', {})
        with self.assertRaises(PermissionError):
            await number_input_context(copy.deepcopy(request), replace(self.scope, context_key='other'), self.agent.name, 'call', {})

    async def test_hierarchical_final_quotes_only_actual_summary_parts(self):
        claim = '研究纳入200人。'
        rows = [await self.store.append(self.scope, self.agent.name, 'llm_output', {'content': claim + '背景' * 12000}, f'call-{i}')
                for i in range(2)]
        seen = []
        async def llm(system, user, *args, **kwargs):
            stages = [r['payload']['stage'] for r in (await self.store.load(self.scope, self.agent.name))['messages']
                      if r['kind'] == 'summary_stage']
            stage = stages[-1]
            if stage.startswith('summary'):
                return prose(claim)
            if stage == 'claimify_selection': return answer('selected', claim)
            if stage == 'claimify_disambiguation': return answer('resolved', claim)
            if stage == 'claimify_decomposition': return decomposed(claim)
            payload = json.loads(user)
            seen.extend(payload['candidate_excerpts'])
            excerpt = payload['candidate_excerpts'][0]
            return json.dumps({'section': 'SUMMARY', 'claim': claim,
                               'references': [{'source_id': excerpt['source_id'], 'quote': claim}]})
        self.agent._call_llm_impl = AsyncMock(side_effect=llm)
        with patch('service.agent_context.scoped_input_budget', return_value=6000), bind_agent_history(self.scope):
            final = await summarize_history(self.agent, self.scope, rows, CharCounter())
        self.assertTrue(seen)
        self.assertTrue(all(e['source_id'] not in {r['source_id'] for r in rows} for e in seen))
        self.assertTrue(all('背景' not in e['text'] for e in seen))
        stored = (await self.store.load(self.scope, self.agent.name))['messages']
        self.assertTrue(parse_summary(final, {r['source_id']: source_text(r) for r in stored})[1])

    async def test_prepared_audit_without_actual_context_is_rejected(self):
        with bind_agent_history(self.scope), self.assertRaisesRegex(ValueError, '实际输入上下文'):
            await summarize_with_citations(self.agent, self.scope, [self.row], CharCounter(), prepared_prose=prose('研究完成。'))

    async def test_context_positions_are_computed_with_legacy_verification_settings(self):
        claim = '研究纳入200人。'
        scope = replace(self.scope, context_policy=ContextPolicy(group='旧核验配置',
                        verification=VerificationSettings(rules_version='verification-v1', coverage=False,
                                                          key_fields=False, nli_support=False)).snapshot())
        self.agent._call_llm_impl = AsyncMock(side_effect=[prose(claim), *responses(claim),
                json.dumps({'section': 'SUMMARY', 'claim': claim,
                            'references': [{'source_id': self.row['source_id'], 'quote': claim}]})])
        with bind_agent_history(scope):
            final = await summarize_history(self.agent, scope, [self.row], CharCounter())
        ref = parse_summary(final, {self.row['source_id']: source_text(self.row)})[1][0]['references'][0]
        self.assertEqual(source_text(self.row)[ref['start']:ref['end']], claim)


class FidelityNLITests(unittest.TestCase):
    def test_history_fidelity_is_scored_but_research_gate_is_preserved(self):
        quote = '研究纳入200人。'
        ref = {'source_id': 'message', 'quote': quote, 'source_type': 'assistant_history', 'verification_scope': CONTEXT_SCOPE}
        claims = [{'section': 'SUMMARY', 'claim': quote, 'references': [ref]}]
        model = Mock()
        model.predict.return_value = [{'status': 'scored', 'scores': {'entailment': .98, 'neutral': .01, 'contradiction': .01}}]
        with patch('service.nli_verifier.get_nli', return_value=model), \
             patch('service.nli_verifier.reference_windows', return_value=([(quote, 0, len(quote))], None)):
            report = verify_claims_sync(claims, NLISettings())
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['claims'][0]['evidence'][0]['verification_scope'], CONTEXT_SCOPE)
            model.predict.assert_called_once()
            model.reset_mock()
            del ref['verification_scope']
            for source_type in ('assistant_history', 'summary_input'):
                ref['source_type'] = source_type
                self.assertEqual(verify_claims_sync(claims, NLISettings())['status'], 'rejected')
                model.predict.assert_not_called()
