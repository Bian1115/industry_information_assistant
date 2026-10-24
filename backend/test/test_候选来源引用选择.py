"""候选池允许子集；模型与NLI用桩，验证引用定位与门禁控制流。"""
import unittest
from unittest.mock import AsyncMock, patch

from test_智能体独立历史 import MemoryStore, CharCounter, HistoryScope, make_agent, bind_agent_history, summary_text
from test_摘要合并与引用分离 import clause, draft, citation
from service.context_policy import ContextPolicy, VerificationSettings, NLISettings
from service.summary_citations import summarize_with_citations, parse_citation_result
from service.agent_context import parse_summary


class CandidateSourceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = MemoryStore()
        policy = ContextPolicy(group='候选来源机制验证', summary_strategy='chunk-merge-v1', nli=NLISettings(),
            verification=VerificationSettings(coverage=False, key_fields=False,
                                             evidence_context=False, repair_attempts=0))
        self.scope = HistoryScope(self.store, 'run', 'session', None, context_policy=policy.snapshot())
        self.a = await self.store.append(self.scope, 'DeepScout', 'llm_output', {'content': '研究结论。后续记录。'}, 'a')
        self.b = await self.store.append(self.scope, 'DeepScout', 'llm_output', {'content': '其他主题。'}, 'b')
        self.item = clause('研究结论。', self.a['source_id'], self.b['source_id'])
        self.agent = make_agent()

    async def run_summary(self, responses, *, status='passed', records=None, **kwargs):
        self.agent._call_llm_impl = AsyncMock(side_effect=responses)
        verification = AsyncMock(return_value={'status': status, 'execution_status': 'completed'})
        with bind_agent_history(self.scope), patch('service.nli_verifier.verify_claims', verification):
            result = await summarize_with_citations(self.agent, self.scope,
                records if records is not None else [self.a, self.b], CharCounter(), **kwargs)
        return result, verification

    async def test_one_supporting_candidate_is_enough_and_only_selected_ref_reaches_nli(self):
        result, verify = await self.run_summary([
            draft(self.item), citation(self.item, [(self.a['source_id'], '研究结论。')])])
        claims = parse_summary(result, {self.a['source_id']: self.a['payload']['content']})[1]
        self.assertEqual([r['source_id'] for r in claims[0]['references']], [self.a['source_id']])
        checked = verify.call_args.args[0][0]['references']
        self.assertEqual([(r['source_id'], r['start'], r['end']) for r in checked], [(self.a['source_id'], 0, 5)])
        verify.assert_awaited_once()
        self.assertEqual(self.agent._call_llm_impl.await_count, 2)

    async def test_empty_candidate_pool_evidence_fails_without_nli_or_commit(self):
        with self.assertRaisesRegex(ValueError, '候选来源均未提供'):
            await self.run_summary([draft(self.item), citation(self.item, [])])
        saved = await self.store.load(self.scope, 'DeepScout')
        self.assertEqual((saved['summary_version'], saved['covered_seq']), (0, 0))
        self.assertFalse(any(r['kind']=='summary_verification' for r in saved['messages']))

    async def test_locatable_quote_does_not_override_nli_rejection(self):
        with self.assertRaisesRegex(ValueError, '核验拒绝'):
            await self.run_summary([draft(self.item), citation(self.item, [(self.a['source_id'], '研究结论。')])], status='rejected')
        saved = await self.store.load(self.scope, 'DeepScout')
        self.assertEqual((saved['summary_version'], saved['covered_seq']), (0, 0))
        report = next(r['payload'] for r in saved['messages'] if r['kind']=='summary_verification')
        self.assertEqual(report['status'], 'rejected')

    async def test_all_batches_are_seen_even_after_first_candidate_has_quote(self):
        item = clause('研究结论。', self.a['source_id'])
        batches = [[{'source_id': self.a['source_id'], 'start': 0, 'text': '研究结论。'}],
                   [{'source_id': self.a['source_id'], 'start': 5, 'text': '后续记录。'}]]
        with patch('service.summary_citations.citation_batches', return_value=iter(batches)):
            _, verify = await self.run_summary([draft(item),
                citation(item, [(self.a['source_id'], '研究结论。')]),
                citation(item, [(self.a['source_id'], '后续记录。')])])
        self.assertEqual(len(verify.call_args.args[0][0]['references']), 2)
        self.assertEqual(self.agent._call_llm_impl.await_count, 3)

    async def test_empty_first_batch_can_use_later_candidate(self):
        batches = [[{'source_id': self.b['source_id'], 'start': 0, 'text': '其他主题。'}],
                   [{'source_id': self.a['source_id'], 'start': 0, 'text': '研究结论。'}]]
        with patch('service.summary_citations.citation_batches', return_value=iter(batches)):
            _, verify = await self.run_summary([draft(self.item), citation(self.item, []),
                citation(self.item, [(self.a['source_id'], '研究结论。')])])
        self.assertEqual(verify.call_args.args[0][0]['references'][0]['source_id'], self.a['source_id'])

    async def test_unselected_old_candidate_does_not_force_full_text_fallback(self):
        item = clause('其他主题。', self.a['source_id'], self.b['source_id'])
        _, verify = await self.run_summary([draft(item), citation(item, [(self.b['source_id'], '其他主题。')])],
            records=[self.b], previous_sources={self.a['source_id']: ['研究结论。']},
            previous_summary=summary_text(self.a['source_id'], '研究结论。'))
        self.assertEqual(self.agent._call_llm_impl.await_count, 2)
        self.assertEqual(verify.call_args.args[0][0]['references'][0]['source_id'], self.b['source_id'])

    async def test_no_evidence_in_old_quotes_still_rereads_original(self):
        item = clause('后续记录。', self.a['source_id'])
        _, verify = await self.run_summary([draft(item), citation(item, []),
            citation(item, [(self.a['source_id'], '后续记录。')])], records=[self.b],
            previous_sources={self.a['source_id']: ['研究结论。']},
            previous_summary=summary_text(self.a['source_id'], '研究结论。'))
        self.assertEqual(verify.call_args.args[0][0]['references'][0]['start'], 5)
        self.assertEqual(self.agent._call_llm_impl.await_count, 3)

    def test_unknown_candidate_and_fabricated_quote_still_rejected(self):
        excerpts = [{'source_id': self.a['source_id'], 'text': self.a['payload']['content']}]
        for sid, quote in [('invented', '研究结论。'), (self.a['source_id'], '不存在的原句。')]:
            with self.assertRaises(ValueError):
                parse_citation_result(citation(self.item, [(sid, quote)]), self.item, excerpts, program_positions=True)


if __name__ == '__main__':
    unittest.main()
