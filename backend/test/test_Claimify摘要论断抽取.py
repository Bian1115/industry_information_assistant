"""Claimify 的协议、来源映射与摘要集成机制测试；模型为确定性桩。"""
import copy
import json
import os
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from test_智能体独立历史 import MemoryStore, CharCounter, HistoryScope, make_agent, bind_agent_history
from service.agent_context import parse_summary, summary_reference_quotes, summarize_history, prepare_agent_messages, ContextBudgetExceeded
from service.claimify import (parse_prose, sentence_spans, split_sentences, extract_claims, excerpt_for,
                              parse_stage, pack_summary, saved_prose, prose_messages, generate_prose, stage_prompt)
from service.context_policy import ContextPolicy, VerificationSettings, NLISettings, legacy_policy
from service.summary_citations import summary_merge_view


def prose(summary, intent="None", steps="None"):
    if steps != "None":
        summary = summary + "\n\n" + steps if summary != "None" else steps
    return f"## SESSION INTENT\n{intent}\n\n## SUMMARY\n{summary}\n\n## ARTIFACTS\nNone"


def answer(status, sentence="", reason="测试结论"):
    return json.dumps({"status": status, "sentence": sentence, "reason": reason}, ensure_ascii=False)


def decomposed(*claims):
    return json.dumps({"claims": list(claims)}, ensure_ascii=False)


def responses(sentence, *claims):
    return [answer("selected", sentence), answer("resolved", sentence), decomposed(*(claims or [sentence]))]


class ProtocolTests(unittest.TestCase):
    def test_sentence_boundaries_keep_decimal_abbreviation_url_and_quotes(self):
        text = 'Dr. Smith measured 3.14 mg. 他表示：“研究未证明有效。”记录位于 /tmp/data.json。见 https://example.com/a.b?q=1 。'
        pieces = [text[a:b] for a, b in sentence_spans(text)]
        self.assertEqual(pieces, ['Dr. Smith measured 3.14 mg.', '他表示：“研究未证明有效。”',
                                  '记录位于 /tmp/data.json。', '见 https://example.com/a.b?q=1 。'])

    def test_chinese_paragraphs_and_newlines(self):
        body = prose('甲研究纳入800人。它没有证明有效！\n下一项研究尚未完成', intent='用户希望比较研究。')
        rows = split_sentences(parse_prose(body))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[2]['sentence'], '它没有证明有效！')
        for row in rows:
            self.assertEqual(parse_prose(body)[row['section']][row['start']:row['end']], row['sentence'])

    def test_context_windows_follow_paper(self):
        rows = split_sentences(parse_prose(prose(''.join(f'第{i}句。' for i in range(15)))))
        selection = excerpt_for(rows, 7, 'selection', '当前句。')
        resolution = excerpt_for(rows, 7, 'disambiguation', '当前句。')
        self.assertIn('第2句。', selection)
        self.assertNotIn('第1句。', selection)
        self.assertIn('第12句。', selection)
        self.assertNotIn('第13句。', selection)
        self.assertNotIn('第8句。', resolution)
        self.assertIn('## SUMMARY', resolution)
        self.assertTrue(resolution.startswith('[...]'))

    def test_model_prompt_only_requests_prose(self):
        request = prose_messages({'new_records': []})
        self.assertIn('自然语言', request[0]['content'])
        self.assertNotIn('allowed_source_ids', request[1]['content'])
        with self.assertRaises(ValueError):
            parse_prose(prose('{"claims":[]}'))
        with self.assertRaises(ValueError):
            parse_prose(prose('事实。') + '\n## CLAIMS\n[]')

    def test_strict_stage_protocol(self):
        for output, stage in [('[]', 'selection'), ('{"claims":[1]}', 'decomposition'),
                              (answer('selected', ''), 'selection'),
                              (answer('cannot_disambiguate', '猜测'), 'disambiguation'),
                              (answer('selected', 'remains unchanged'), 'selection')]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_stage(output, stage)
        self.assertEqual(parse_stage(decomposed(), 'decomposition'), {'claims': []})

    def test_official_prompts_with_adapter(self):
        self.assertIn('partnership between Company X', stage_prompt('selection'))
        self.assertIn('Cannot be decontextualized', stage_prompt('disambiguation'))
        self.assertIn('Essential Context/Clarifications', stage_prompt('decomposition'))

    def test_new_runtime_default_and_explicit_legacy_strategy(self):
        with patch.dict(os.environ, {'AGENT_SUMMARY_NLI_ENABLED': '0'}):
            with patch.dict(os.environ):
                os.environ.pop('AGENT_SUMMARY_STRATEGY', None)
                self.assertEqual(legacy_policy().summary_strategy, 'claimify-v1')
            with patch.dict(os.environ, {'AGENT_SUMMARY_STRATEGY': 'chunk-merge-v1'}):
                self.assertEqual(legacy_policy().summary_strategy, 'chunk-merge-v1')
        self.assertEqual(ContextPolicy(group='新运行').summary_strategy, 'claimify-v1')

    def test_legacy_fenced_claims_remain_readable(self):
        from test_智能体独立历史 import summary_text
        old = summary_text('s1', '旧结论。')
        body, claims = old.rsplit('\n## CLAIMS\n', 1)
        fenced = body + '\n## CLAIMS\n```json\n' + claims + '\n```'
        self.assertIn('[s1]', summary_merge_view(fenced, {'s1': '旧结论。'}))
        self.assertEqual(saved_prose(fenced, {'s1': '旧结论。'}), prose('旧结论。'))


class ExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_filter_disambiguate_decompose_and_skip_unresolved(self):
        text = prose('甲研究纳入800人，体现合作的重要性。它于2023年完成。两家公司的领导推动了两项业务。',
                     intent='建议继续检索。')
        call = AsyncMock(side_effect=[answer('no_verifiable_claims'),
            *responses('甲研究纳入800人。'),
            answer('selected', '它于2023年完成。'), answer('resolved', '甲研究于2023年完成。'),
            decomposed('甲研究于2023年完成。'),
            answer('selected', '两家公司的领导推动了两项业务。'), answer('cannot_disambiguate')])
        result = await extract_claims(text, '比较研究', call)
        self.assertEqual([c['claim'] for c in result['claims']], ['甲研究纳入800人。', '甲研究于2023年完成。'])
        self.assertEqual([s['status'] for s in result['sentences']],
                         ['no_verifiable_claims', 'extracted', 'extracted', 'cannot_disambiguate'])
        self.assertEqual(call.await_count, 9)
        decomposition_calls = [c for c in call.call_args_list if c.args[1] == 'claimify_decomposition']
        self.assertIn('甲研究于2023年完成。', decomposition_calls[1].args[0][1]['content'])
        self.assertEqual(result['claims'][1]['sentence_index'], 2)

    async def test_nonfactual_prefix_does_not_bypass_selection(self):
        sentence = '建议：甲研究纳入800人，应继续检索。'
        call = AsyncMock(side_effect=responses('甲研究纳入800人。'))
        result = await extract_claims(prose(sentence), '', call)
        self.assertEqual(result['claims'][0]['claim'], '甲研究纳入800人。')
        self.assertIn(sentence, call.call_args_list[0].args[0][1]['content'])

    async def test_one_sentence_multiple_attributed_conditional_claims(self):
        sentence = '报告称2023年甲组和乙组在成人中未发现显著改善。'
        claims = ['报告称2023年甲组在成人中未发现显著改善。', '报告称2023年乙组在成人中未发现显著改善。']
        result = await extract_claims(prose(sentence), '', AsyncMock(side_effect=responses(sentence, *claims)))
        self.assertEqual([c['claim'] for c in result['claims']], claims)
        self.assertEqual([c['sentence_index'] for c in result['claims']], [0, 0])

    async def test_malformed_output_repaired_once(self):
        call = AsyncMock(side_effect=['bad', answer('no_verifiable_claims')])
        result = await extract_claims(prose('建议继续。'), '', call)
        self.assertEqual(result['claims'], [])
        self.assertEqual(call.call_args_list[1].args[1], 'claimify_selection_parse_repair')
        call = AsyncMock(return_value='bad')
        with self.assertRaises(ValueError):
            await extract_claims(prose('甲研究完成。'), '', call)
        self.assertEqual(call.await_count, 2)

    async def test_empty_decomposition_is_explicitly_audited(self):
        call = AsyncMock(side_effect=[answer('selected', '文本。'), answer('resolved', '文本。'), decomposed()])
        result = await extract_claims(prose('文本。'), '', call)
        self.assertEqual(result['sentences'][0]['status'], 'no_verifiable_claims')
        self.assertEqual(result['claims'], [])

    async def test_prose_repair_still_requests_text(self):
        call = AsyncMock(side_effect=['{"clauses":[]}', prose('甲研究完成。')])
        result = await generate_prose(prose_messages({}), 'summary_merge', call, {})
        self.assertEqual(result, prose('甲研究完成。'))
        self.assertEqual(call.call_args.args[1], 'summary_merge_prose_repair')


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = MemoryStore()
        self.policy = ContextPolicy(group='Claimify机制', semantic_verification=False)
        self.scope = HistoryScope(self.store, 'claimify-run', 'session', None, context_policy=self.policy.snapshot())
        self.agent = make_agent()
        self.row = await self.store.append(self.scope, self.agent.name, 'llm_output',
                                           {'content': '研究A在2023年纳入800名成人，未发现显著改善。'}, 'source')

    def citation(self, claim, quote=None):
        return json.dumps({'section': 'SUMMARY', 'claim': claim, 'references': [
            {'source_id': self.row['source_id'], 'quote': quote or self.row['payload']['content']} ]}, ensure_ascii=False)

    async def run_summary(self, outputs, **kwargs):
        self.agent._call_llm_impl = AsyncMock(side_effect=outputs)
        with bind_agent_history(self.scope):
            return await summarize_history(self.agent, self.scope, [self.row], CharCounter(), **kwargs)

    async def test_natural_summary_to_claims_citations_and_roundtrip(self):
        text = prose('研究A在2023年纳入800名成人。它未发现显著改善。', intent='用户希望比较研究。')
        fact1, fact2 = '研究A在2023年纳入800名成人。', '研究A在2023年纳入的800名成人中未发现显著改善。'
        report = {}
        result = await self.run_summary([text, answer('no_verifiable_claims'), *responses(fact1),
            answer('selected', '它未发现显著改善。'), answer('resolved', fact2), decomposed(fact2),
            self.citation(fact1), self.citation(fact2)], verification_out=report)
        parts, claims = parse_summary(result, {self.row['source_id']: self.row['payload']['content']})
        self.assertEqual(parts['SUMMARY'], parse_prose(text)['SUMMARY'])
        self.assertNotIn(fact2, parts['SUMMARY'])
        self.assertEqual([c['claim'] for c in claims], [fact1, fact2])
        self.assertEqual(summary_merge_view(result, summary_reference_quotes(result)), text)
        self.assertEqual(saved_prose(result, summary_reference_quotes(result)), text)
        self.assertEqual(report['verification_scope'], 'extracted_claims_only')
        for call in self.agent._call_llm_impl.call_args_list:
            self.assertEqual(call.kwargs['model_override'], 'qwen3-max')
        events = (await self.store.load(self.scope, self.agent.name))['messages']
        self.assertTrue(any(e['kind'] == 'summary_claimify' for e in events))
        stages = [e['payload']['stage'] for e in events if e['kind'] == 'summary_stage']
        self.assertEqual(stages[0], 'summary_merge')
        self.assertGreater(stages.index('citation_completion'), stages.index('claimify_decomposition'))

    async def test_all_filtered_keeps_goals_and_empty_claims(self):
        text = prose('None', intent='用户希望比较研究。', steps='继续检索。')
        result = await self.run_summary([text, answer('no_verifiable_claims'), answer('no_verifiable_claims')])
        self.assertEqual(parse_summary(result, {})[1], [])
        self.assertEqual(saved_prose(result, {}), text)

    async def test_unresolved_is_not_verified_and_is_flagged(self):
        report = {}
        text = prose('两家公司的领导推动了两项业务。')
        result = await self.run_summary([text, answer('selected', '两家公司的领导推动了两项业务。'),
                                         answer('cannot_disambiguate')], verification_out=report)
        self.assertEqual(parse_summary(result, {})[1], [])
        self.assertTrue(report['needs_review'])
        self.assertEqual(report['unresolved_sentence_indexes'], [0])

    async def test_tampered_ledger_and_citation_are_rejected(self):
        claim = '研究A在2023年纳入800名成人。'
        result = await self.run_summary([prose(claim), *responses(claim), self.citation(claim)])
        body, raw = result.rsplit('\n## CLAIMS\n', 1)
        original = json.loads(raw)
        for change in ('span', 'status', 'index', 'quote', 'empty', 'offset'):
            data = copy.deepcopy(original)
            if change == 'span': data['sentences'][0]['end'] -= 1
            if change == 'status': data['sentences'][0]['status'] = 'cannot_disambiguate'
            if change == 'index': data['claims'][0]['sentence_index'] = 99
            if change == 'quote': data['claims'][0]['references'][0]['quote'] = '伪造引用'
            if change == 'empty': data['claims'] = []
            if change == 'offset': data['claims'][0]['references'][0]['start'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_summary(body+'\n## CLAIMS\n'+json.dumps(data), {self.row['source_id']: self.row['payload']['content']})

    async def test_incremental_cites_actual_previous_summary_not_unprovided_old_source(self):
        claim = '研究A在2023年纳入800名成人。'
        old = await self.run_summary([prose(claim), *responses(claim), self.citation(claim)])
        previous_sources = summary_reference_quotes(old)
        self.row = await self.store.append(self.scope, self.agent.name, 'llm_output', {'content': '新增检索没有结果。'}, 'new-source')
        old_sid = next(iter(previous_sources))
        outputs = iter([prose(claim), *responses(claim)])
        async def llm(system, user, *args, **kwargs):
            payload = json.loads(user) if user.startswith('{') else {}
            if 'final_claim' in payload:
                excerpt = next(e for e in payload['candidate_excerpts'] if '## SUMMARY' in e['text'])
                self.assertNotEqual(excerpt['source_id'], old_sid)
                return json.dumps({'section': 'SUMMARY', 'claim': claim, 'references': [
                    {'source_id': excerpt['source_id'], 'quote': claim}]}, ensure_ascii=False)
            return next(outputs)
        self.agent._call_llm_impl = AsyncMock(side_effect=llm)
        with bind_agent_history(self.scope):
            result = await summarize_history(self.agent, self.scope, [self.row], CharCounter(),
                                             previous_summary=old, previous_sources=previous_sources)
        sent = json.loads(self.agent._call_llm_impl.call_args_list[0].args[1])
        self.assertEqual(sent['previous_summary'], prose(claim))
        self.assertNotIn('CLAIMS', sent['previous_summary'])
        self.assertNotIn(old_sid, summary_reference_quotes(result))
        self.assertIn(sent['previous_summary_source_id'], summary_reference_quotes(result))

    async def test_nli_uses_extracted_claim_and_contract_uses_prose(self):
        claim = '研究A在2023年纳入800名成人。'
        self.scope = replace(self.scope, contract={'query': '比较研究'}, context_policy=ContextPolicy(
            group='核验集成', nli=NLISettings(), verification=VerificationSettings(coverage=False)).snapshot())
        self.agent._call_llm_impl = AsyncMock(side_effect=[prose(claim, intent='用户希望比较研究。'),
            answer('no_verifiable_claims'), *responses(claim), self.citation(claim),
            json.dumps({'items': [{'status': 'preserved', 'quote': '用户希望比较研究。'}]})])
        with patch('service.nli_verifier.verify_claims', new=AsyncMock(return_value={
                'status':'passed', 'execution_status':'completed'})) as verify, bind_agent_history(self.scope):
            await summarize_history(self.agent, self.scope, [self.row], CharCounter())
        self.assertEqual(verify.call_args.args[0][0]['claim'], claim)
        contract_input = json.loads(self.agent._call_llm_impl.call_args_list[-1].args[1])
        self.assertIn('用户希望比较研究。', contract_input['body'])

    async def test_stage_limit_stops_before_selection(self):
        self.scope = replace(self.scope, context_policy=ContextPolicy(group='上限测试',
            verification=VerificationSettings(nli_support=False, key_fields=False, max_stage_calls=1)).snapshot())
        with self.assertRaisesRegex(ValueError, '调用数量'):
            await self.run_summary([prose('甲研究完成。')])
        saved = await self.store.load(self.scope, self.agent.name)
        self.assertEqual(saved['covered_seq'], 0)
        self.assertEqual(saved['summary'], '')

    async def test_extraction_failure_preserves_saved_summary_and_cursor(self):
        claim = '研究A在2023年纳入800名成人。'
        old = await self.run_summary([prose(claim), *responses(claim), self.citation(claim)])
        await self.store.save_summary(self.scope, self.agent.name, expected_version=0,
                                     covered_seq=self.row['seq'], summary=old, metadata={})
        await self.store.append(self.scope, self.agent.name, 'llm_output',
                                {'content': '新研究完成。' + '背景' * 4000}, 'new')
        self.agent._call_llm_impl = AsyncMock(side_effect=[prose('新研究完成。'), 'invalid', 'invalid'])
        with patch('service.agent_context.get_qwen_token_counter', return_value=CharCounter()), \
             patch('service.agent_context.scoped_input_budget', side_effect=lambda *a, **kw: 30000 if kw.get('summary') else 1400), \
             bind_agent_history(self.scope), self.assertRaises(ContextBudgetExceeded):
            await prepare_agent_messages(self.agent, self.scope, '系统', '继续', 'current', 6000)
        saved = await self.store.load(self.scope, self.agent.name)
        self.assertEqual((saved['summary'], saved['summary_version'], saved['covered_seq']), (old, 1, self.row['seq']))
        self.assertEqual(self.agent._call_llm_impl.await_count, 3)

    async def test_nli_repair_reextracts_from_repaired_prose(self):
        claim = '研究A在2023年纳入800名成人。'
        self.scope = replace(self.scope, context_policy=ContextPolicy(group='修复集成', nli=NLISettings(),
            verification=VerificationSettings(key_fields=False, coverage=False)).snapshot())
        outputs = [prose(claim), *responses(claim), self.citation(claim)]
        verify = AsyncMock(side_effect=[{'status': 'rejected', 'execution_status': 'completed',
                                       'claims': [{'claim': claim, 'status': 'unknown'}]},
                                       {'status': 'passed', 'execution_status': 'completed'}])
        with patch('service.nli_verifier.verify_claims', verify):
            result = await self.run_summary(outputs + outputs)
        self.assertEqual(parse_summary(result, {self.row['source_id']: self.row['payload']['content']})[1][0]['claim'], claim)
        events = (await self.store.load(self.scope, self.agent.name))['messages']
        stages = [r['payload']['stage'] for r in events if r['kind'] == 'summary_stage']
        self.assertEqual(stages.count('claimify_selection'), 2)
        self.assertEqual(stages.count('summary_repair'), 1)

    async def test_budget_rejects_prompt_without_silent_truncation(self):
        self.agent._call_llm_impl = AsyncMock(return_value=prose('甲研究完成。'))
        with patch('service.agent_context.scoped_input_budget', return_value=1), bind_agent_history(self.scope):
            with self.assertRaises(ValueError):
                await summarize_history(self.agent, self.scope, [self.row], CharCounter())
        self.agent._call_llm_impl.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
