"""契约、正文编号及一次局部修复回归；使用确定性模型桩。"""
import json
import logging
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from test_智能体独立历史 import MemoryStore, CharCounter, HistoryScope, make_agent, bind_agent_history
from test_摘要合并与引用分离 import clause, draft, citation
from service.context_policy import ContextPolicy, VerificationSettings
from service.summary_citations import draft_messages, generate_draft, parse_draft, summarize_with_citations, repaired_draft, parse_citation_result
from service.summary_hierarchy import hierarchical_summary
from service.deep_research_v2.agents.architect import ChiefArchitect


class ContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_planning_builds_contract_from_final_outline_and_current_query(self):
        agent = ChiefArchitect.__new__(ChiefArchitect)
        agent.name = 'ChiefArchitect'
        agent.logger = logging.getLogger('contract-test')
        agent.add_message = Mock()
        plan = {'outline': [{'id': f's{i}', 'title': f'研究章节{i}'} for i in range(4)],
                'research_contract': {'critical_questions': ['哪些方法得到实验验证？'],
                                      'active_constraints': {'max_body_chars': 3000}},
                'hypotheses': []}
        agent.call_llm = AsyncMock(return_value=json.dumps(plan))
        state = {'query': '研究材料配比优化，区分预测与实验验证', 'original_query': '请控制在3000字',
                 'active_constraints': {'max_body_chars': 3000}, 'errors': []}
        await agent._initial_planning(state)
        contract = state['research_contract']
        self.assertEqual(contract['query'], state['query'])
        self.assertEqual(contract['original_query'], state['original_query'])
        self.assertEqual(contract['active_constraints'], {'max_body_chars': 3000})
        self.assertNotIn('critical_sections', contract)
        self.assertEqual(state['research_questions'], [state['query']])


class RepairTests(unittest.IsolatedAsyncioTestCase):
    def test_json_quote_representation_is_reversible_and_exact(self):
        quote = '日期2026年\n标题"研究"'
        text = json.dumps({'summary': quote}, ensure_ascii=False)
        item = clause('研究于2026年发表。', 'body-uuid')
        output = citation(item, [('body-uuid', quote)])
        refs = parse_citation_result(output, item, [{'source_id': 'body-uuid', 'start': 0, 'text': text}], program_positions=True)
        ref = refs[0]
        self.assertEqual(text[ref['start']:ref['end']], ref['quote'])
        self.assertEqual(json.loads('"'+ref['quote']+'"'), quote)
        for candidate in ['前缀'+text, json.dumps({'summary': '日期2025年\n标题"研究"'}, ensure_ascii=False)]:
            with self.assertRaisesRegex(ValueError, '原句不在'):
                parse_citation_result(output, item, [{'source_id': 'body-uuid', 'start': 0, 'text': candidate}], program_positions=True)

    async def test_structured_repair_renders_explicit_task_kind_and_keeps_facts(self):
        data = {'clauses': [
            {'section': 'SUMMARY', 'kind': 'todo', 'claim': '继续检索。', 'source_ids': []},
            {'section': 'SUMMARY', 'kind': 'fact', 'claim': '研究纳入800人。', 'source_ids': ['body-uuid']}]}
        call = AsyncMock(side_effect=['格式错误', json.dumps(data)])
        result = await generate_draft(self.request(), 'summary_chunk', call, {'body-uuid': '原文'})
        self.assertIn('待办：继续检索。', result)
        fact = next(c for c in parse_draft(result, {'body-uuid'}) if c['claim'] == '研究纳入800人。')
        self.assertEqual(fact['source_ids'], ['body-uuid'])
        self.assertEqual(call.await_count, 2)
        data['clauses'].append({'section': 'ARTIFACTS', 'kind': 'fact', 'claim': 'None', 'source_ids': []})
        self.assertIn('## ARTIFACTS\nNone', repaired_draft(json.dumps(data)))
        with self.assertRaisesRegex(ValueError, '全部内容'):
            repaired_draft(json.dumps({'clauses': [data['clauses'][-1]]}))
        data['clauses'][1]['source_ids'] = ['invented']
        with self.assertRaisesRegex(ValueError, '未知来源'):
            parse_draft(repaired_draft(json.dumps(data)), {'body-uuid'})
        data['clauses'][1]['source_ids'] = []
        with self.assertRaisesRegex(ValueError, '事实须携带来源'):
            repaired_draft(json.dumps(data))

    def test_comma_separated_ids_normalize_without_accepting_unknown_ids(self):
        text = draft(clause('事实 [body-uuid, second-id]。'))
        self.assertEqual(parse_draft(text, {'body-uuid', 'second-id'})[0]['source_ids'], ['body-uuid', 'second-id'])
        self.assertEqual(json.loads(draft_messages({'summary_parts': [text]})[1]['content'])['allowed_source_ids'], ['body-uuid', 'second-id'])
        with self.assertRaisesRegex(ValueError, '未知来源'):
            parse_draft(text, {'body-uuid'})
        with self.assertRaisesRegex(ValueError, '分句为空'):
            parse_draft(draft(clause('[body-uuid]。')), {'body-uuid'})

    def test_punctuation_after_source_marker_preserves_claim_and_id(self):
        text = draft(clause('事实 [body-uuid]。'))
        self.assertEqual(parse_draft(text, {'body-uuid'})[0]['claim'], '事实。')
        self.assertEqual(json.loads(draft_messages({'summary_parts': [text]})[1]['content'])['allowed_source_ids'], ['body-uuid'])

    def test_all_bad_lines_are_reported_in_one_parse(self):
        text = draft(clause('尚未核实。'), clause('需继续检索。'), clause('另一事实。','invented'))
        with self.assertRaises(ValueError) as context:
            parse_draft(text, {'body-uuid'})
        self.assertIn('尚未核实', str(context.exception))
        self.assertIn('需继续检索', str(context.exception))
        self.assertIn('invented', str(context.exception))

    def request(self):
        return draft_messages({'new_records': [{'source_id': 'body-uuid', 'text': '研究纳入800人。内部编号 source_1'}],
                               'latest_contract': {'query': '比较研究并撰写报告'}, 'previous_summary': ''})

    async def test_unknown_nested_id_repaired_once_using_exact_original_evidence(self):
        bad = draft(clause('研究纳入800人。', 'source_1'))
        good = draft(clause('研究纳入800人。', 'body-uuid'))
        call = AsyncMock(side_effect=[bad, good])
        result = await generate_draft(self.request(), 'summary_chunk', call, {'body-uuid': '原文'})
        self.assertEqual(result, good)
        self.assertEqual([c.args[1] for c in call.call_args_list], ['summary_chunk', 'summary_chunk_parse_repair'])
        repair = json.loads(call.call_args_list[1].args[0][1]['content'])
        self.assertEqual(repair['original_request'], json.loads(self.request()[1]['content']))
        self.assertIn('source_1', repair['parse_error'])
        self.assertEqual(repair['repair_attempt'], 1)

    async def test_second_parse_failure_does_not_repair_again(self):
        call = AsyncMock(return_value=draft(clause('事实。', 'source_1')))
        with self.assertRaisesRegex(ValueError, '未知来源'):
            await generate_draft(self.request(), 'summary_merge', call, {'body-uuid': '原文'})
        self.assertEqual(call.await_count, 2)

    async def test_only_current_batch_ids_are_allowed_even_when_other_sources_exist(self):
        call = AsyncMock(return_value=draft(clause('事实。', 'other-body')))
        with self.assertRaisesRegex(ValueError, '未知来源'):
            await generate_draft(self.request(), 'summary_chunk', call, {'body-uuid': '原文', 'other-body': '另一批'})
        self.assertEqual(call.await_count, 2)

    async def test_success_needs_no_repair(self):
        call = AsyncMock(return_value=draft(clause('研究纳入800人。', 'body-uuid')))
        await generate_draft(self.request(), 'summary_merge', call, {'body-uuid': '原文'})
        call.assert_awaited_once()

    async def test_contract_and_original_ids_reach_every_chunk_and_merge(self):
        fragments = [{'source_id': 'body-uuid', 'text': '正文'*1500, 'start': n*3000, 'end': (n+1)*3000} for n in range(3)]
        contract = {'query': '比较材料配比方法，写3000字报告'}
        calls = []
        async def call(request, stage):
            payload = json.loads(request[1]['content']); calls.append(stage)
            self.assertEqual(payload['latest_contract'], contract)
            self.assertEqual(payload['allowed_source_ids'], ['body-uuid'])
            return draft(clause('研究纳入800人。', 'body-uuid'))
        budget = max(CharCounter().count_messages(draft_messages({'new_records': [f], 'previous_summary': '', 'latest_contract': contract})) for f in fragments)
        await hierarchical_summary(fragments, '', contract, call, CharCounter(), budget,
                                   ContextPolicy(group='测试'), {'body-uuid': '原文'})
        self.assertEqual(calls, ['summary_chunk']*3+['summary_hierarchical_merge'])

    async def setup_scope(self, **checks):
        store = MemoryStore()
        policy = ContextPolicy(group='修复测试', summary_strategy='chunk-merge-v1', semantic_verification=True,
            verification=VerificationSettings(nli_support=False, key_fields=False, **checks))
        scope = HistoryScope(store, 'run', 'session', None, context_policy=policy.snapshot())
        row = await store.append(scope, 'DeepScout', 'llm_output', {'content': '研究纳入800人。'}, 'c')
        return store, scope, row

    async def test_citation_json_error_repaired_in_place_and_position_still_verified(self):
        store, scope, row = await self.setup_scope()
        item = clause('研究纳入800人。', row['source_id'])
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=[draft(item), '{bad json', citation(item, [(row['source_id'], item['claim'])])])
        with bind_agent_history(scope):
            result = await summarize_with_citations(agent, scope, [row], CharCounter())
        self.assertIn('"start": 0', result)
        rows = (await store.load(scope, 'DeepScout'))['messages']
        self.assertEqual([r['payload']['stage'] for r in rows if r['kind']=='summary_stage'],
                         ['summary_merge', 'citation_completion', 'citation_completion_parse_repair'])

    async def test_structured_draft_repair_reaches_citation_with_unchanged_body_id(self):
        store, scope, row = await self.setup_scope()
        item = clause('研究纳入800人。', row['source_id'])
        repaired = json.dumps({'clauses': [{**item, 'kind': 'fact'}]})
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=['invalid', repaired, citation(item, [(row['source_id'], item['claim'])])])
        with bind_agent_history(scope):
            result = await summarize_with_citations(agent, scope, [row], CharCounter())
        self.assertIn(row['source_id'], result)
        self.assertIn('"start": 0', result)
        self.assertTrue(agent._call_llm_impl.call_args_list[1].args[2])  # 修复启用 JSON 输出。
        payload = json.loads(agent._call_llm_impl.call_args_list[2].args[1])
        self.assertEqual(payload['candidate_excerpts'][0]['source_id'], row['source_id'])

    async def test_repair_shares_call_budget_and_cannot_commit_failure(self):
        store, scope, row = await self.setup_scope(max_stage_calls=1)
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(return_value='invalid')
        with bind_agent_history(scope), self.assertRaisesRegex(ValueError, '调用数量'):
            await summarize_with_citations(agent, scope, [row], CharCounter())
        agent._call_llm_impl.assert_awaited_once()
        saved = await store.load(scope, 'DeepScout')
        self.assertEqual((saved['summary_version'], saved['covered_seq'], saved['summary']), (0, 0, ''))

    async def test_repair_cannot_bypass_input_budget(self):
        request = self.request()
        budget = CharCounter().count_messages(request)
        calls=[]
        async def call(req, stage):
            if CharCounter().count_messages(req)>budget:
                raise ValueError('输入超出摘要器预算')
            calls.append(stage)
            return 'bad output'
        with self.assertRaisesRegex(ValueError, '输入超出'):
            await generate_draft(request, 'summary_chunk', call, {'body-uuid':'原文'})
        self.assertEqual(calls, ['summary_chunk'])


if __name__ == '__main__':
    unittest.main()
