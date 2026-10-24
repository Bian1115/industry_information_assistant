import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock

from service.summary_claim_binding import bind_summary_claims, verify_bound_claims
from service.summary_message_verification import verify_message_output
from test_摘要消息编号直接核验 import PROSE


def output(claims):
    return json.dumps({'claims': [{'section': 'SUMMARY', 'claim': text, 'source_ids': ['m']} for text in claims]}, ensure_ascii=False)


def choice(messages, status='equivalent', *, repair=False):
    payload = json.loads(messages[1]['content'])
    data = {'status': status, 'candidate_id': payload['candidates'][0]['candidate_id']
            if status in {'equivalent', 'repaired'} else None, 'reason': '测试判断'}
    if repair: data['source_ids'] = payload['fixed_source_ids']
    return json.dumps(data, ensure_ascii=False)


class BindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = {'sources': {'m': [{'start': 0, 'text': '研究纳入200人。'}]},
                        'rows': {'m': {'kind': 'llm_output', 'payload': {'content': '研究纳入200人。'}}}}
        self.verified = AsyncMock(return_value={'status': 'passed', 'execution_status': 'completed'})

    async def test_exact_claim_does_not_call_equivalence(self):
        call = AsyncMock()
        result = await verify_message_output(output(['研究纳入200人。']), PROSE, self.context, None, call, verify=self.verified)
        call.assert_not_awaited()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['binding']['claims'][0]['binding_method'], 'exact')

    async def test_punctuation_difference_calls_model_once_and_is_only_binding_pass(self):
        call = AsyncMock(side_effect=lambda messages, stage: choice(messages))
        result = await verify_message_output(output(['研究纳入200人.']), PROSE, self.context, None, call, verify=self.verified)
        self.assertEqual(call.await_count, 1)
        self.assertEqual(call.await_args.args[1], 'claim_summary_equivalence')
        bound = result['binding']['claims'][0]
        self.assertEqual(bound['summary_original'], '研究纳入200人。')
        self.assertEqual(bound['model_claim'], '研究纳入200人.')
        self.assertEqual(bound['binding_method'], 'semantic_equivalent')
        self.verified.assert_awaited_once()

    async def test_different_claim_unresolved_does_not_block_other_claim(self):
        async def call(messages, stage):
            return choice(messages, 'different' if stage == 'claim_summary_equivalence' else 'unresolved',
                          repair=stage == 'claim_binding_targeted_repair')
        result = await verify_message_output(output(['研究纳入200人。', '研究纳入300人。']), PROSE,
                                             self.context, None, AsyncMock(side_effect=call), verify=self.verified)
        self.assertEqual(result['status_label'], '核验未完成')
        self.assertEqual(len(result['binding']['claims']), 1)
        self.assertEqual(len(result['unresolved_claims']), 1)
        self.verified.assert_awaited_once()
        self.assertEqual(self.verified.await_args.args[0][0]['claim'], '研究纳入200人。')
        self.assertEqual(result['binding']['semantic_calls'], 1)
        self.assertEqual(result['binding']['repair_calls'], 1)

    async def test_targeted_repair_uses_program_original_and_keeps_source_ids(self):
        async def call(messages, stage):
            return choice(messages, 'different' if stage == 'claim_summary_equivalence' else 'repaired',
                          repair=stage == 'claim_binding_targeted_repair')
        bound = await bind_summary_claims(output(['研究纳入300人。']), PROSE, self.context, call)
        self.assertEqual(bound['claims'][0]['claim'], '研究纳入200人。')
        self.assertEqual(bound['claims'][0]['model_claim'], '研究纳入300人。')
        self.assertEqual(bound['claims'][0]['source_ids'], ['m'])
        self.assertEqual(bound['claims'][0]['binding_method'], 'targeted_repair')
        self.assertFalse(bound['unresolved_claims'])

    async def test_uncertain_judgment_is_not_accepted(self):
        call = AsyncMock(side_effect=lambda messages, stage: choice(messages, 'uncertain'))
        result = await bind_summary_claims(output(['研究纳入300人。']), PROSE, self.context, call, repair=False)
        self.assertEqual(len(result['unresolved_claims']), 1)
        self.assertEqual(call.await_count, 1)
        self.assertFalse(result['claims'])

    async def test_forged_candidate_is_not_a_binding(self):
        call = AsyncMock(return_value=json.dumps({'status': 'equivalent', 'candidate_id': 'invented', 'reason': '同义'}))
        result = await bind_summary_claims(output(['研究纳入200人.']), PROSE, self.context, call, repair=False)
        self.assertEqual(result['status'], 'incomplete')
        self.assertIn('semantic_error', result['binding_audit'][0])
        self.assertEqual(call.await_count, 1)

    async def test_repair_cannot_replace_source_ids(self):
        async def call(messages, stage):
            if stage == 'claim_summary_equivalence': return choice(messages, 'different')
            data = json.loads(choice(messages, 'repaired', repair=True)); data['source_ids'] = ['invented']
            return json.dumps(data)
        result = await bind_summary_claims(output(['研究纳入300人。']), PROSE, self.context, call)
        self.assertTrue(result['unresolved_claims'])
        self.assertFalse(result['claims'])

    async def test_invalid_source_is_isolated(self):
        data = json.loads(output(['研究纳入200人。', '研究纳入200人。']))
        data['claims'][1]['source_ids'] = ['outside-context']
        call = AsyncMock()
        result = await verify_message_output(json.dumps(data), PROSE, self.context, None, call, verify=self.verified)
        self.assertEqual(len(result['binding']['claims']), 1)
        self.assertEqual(result['status'], 'incomplete')
        call.assert_not_awaited()
        self.verified.assert_awaited_once()

    async def test_nli_timeout_is_isolated_from_next_claim(self):
        binding = await bind_summary_claims(output(['研究纳入200人。', '研究纳入200人。']), PROSE, self.context, AsyncMock())
        self.verified.side_effect = [TimeoutError(), {'status': 'passed', 'execution_status': 'completed'}]
        result = await verify_bound_claims(binding, self.context, None, verify=self.verified)
        self.assertEqual(self.verified.await_count, 2)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual([r['status'] for r in result['nli_reports']], ['timeout', 'passed'])

    async def test_binding_pass_cannot_override_nli_rejection(self):
        self.verified.return_value = {'status': 'rejected', 'execution_status': 'completed'}
        result = await verify_message_output(output(['研究纳入200人.']), PROSE, self.context, None,
            AsyncMock(side_effect=lambda messages, stage: choice(messages)), verify=self.verified)
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['status_label'], '核验未通过')

    async def test_insufficient_evidence_is_an_unresolved_item(self):
        self.verified.return_value = {'status': 'rejected', 'execution_status': 'completed',
                                     'claims': [{'status': 'unknown'}]}
        result = await verify_message_output(output(['研究纳入200人。']), PROSE, self.context, None,
                                            AsyncMock(), verify=self.verified)
        self.assertEqual(result['status_label'], '核验未完成')
        self.assertEqual(result['unresolved_claims'][0]['stage'], 'evidence')

    async def test_failed_judgment_protocol_gets_no_second_equivalence_call(self):
        call = AsyncMock(return_value='not-json')
        result = await bind_summary_claims(output(['研究纳入200人.']), PROSE, self.context, call)
        self.assertEqual(call.await_count, 2)
        self.assertEqual([c.args[1] for c in call.await_args_list],
                         ['claim_summary_equivalence', 'claim_binding_targeted_repair'])
        self.assertEqual(result['status'], 'incomplete')

    async def test_no_claims_does_not_auto_pass(self):
        result = await verify_message_output(output([]), PROSE, self.context, None, AsyncMock(), verify=self.verified)
        self.assertEqual(result['status_label'], '核验未完成')
        self.verified.assert_not_awaited()

    async def test_history_topic31_has_two_semantic_checks_and_eleven_exact_bindings(self):
        root = Path(__file__).resolve().parents[1] / 'eval/data/消息编号20000token直接核验-20261001-203200/题目31'
        if not root.exists(): self.skipTest('历史回放输入不存在')
        prose = (root / '生成摘要.md').read_text()
        context = json.loads((root / '上下文编号映射.json').read_text())
        raw = (root / '模型原始编号输出-1.txt').read_text()
        call = AsyncMock(side_effect=lambda messages, stage: choice(messages))
        result = await bind_summary_claims(raw, prose, context, call)
        self.assertEqual(len(result['claims']), 13)
        self.assertEqual(result['semantic_calls'], 2)
        self.assertEqual(result['repair_calls'], 0)
        self.assertEqual(sum(c['binding_method'] == 'exact' for c in result['claims']), 11)
        self.assertFalse(result['unresolved_claims'])
