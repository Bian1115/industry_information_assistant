"""三栏目生成、历史待办保留及旧引用位置兼容；不调用真实模型。"""
import json
import unittest
from unittest.mock import AsyncMock

from test_Claimify摘要论断抽取 import prose, responses, answer
from test_智能体独立历史 import MemoryStore, CharCounter, HistoryScope, make_agent, bind_agent_history
from test_摘要合并与引用分离 import clause, draft
from service.agent_history import canonical_json
from service.agent_context import parse_summary, summary_reference_quotes
from service.claimify import (
    parse_prose, saved_prose, prose_messages, generate_prose, extract_claims, pack_summary,
)
from service.summary_citations import summary_merge_view, parse_draft, draft_messages
from service.summary_lifecycle import PROSE_VERSION, pack_prose_summary
from service.summary_message_verification import completion_messages
from service.summary_atomic_citations import verify_atomic_output
from service.summary_sections import SECTIONS
from service.context_policy import ContextPolicy, SummaryLifecycleSettings


def legacy(summary="None", artifacts="None", pending="待办：用户要求回读论文A，尚未完成。"):
    return ("## SESSION INTENT\nNone\n\n## SUMMARY\n" + summary +
            "\n\n## ARTIFACTS\n" + artifacts + "\n\n## NEXT STEPS\n" + pending)


class MigrationTests(unittest.IsolatedAsyncioTestCase):
    def test_current_prompts_use_three_sections_and_preserve_existing_pending_items(self):
        current = prose_messages({})[0]['content']
        self.assertNotIn('## NEXT STEPS', current)
        self.assertIn('未完成事项属于 SUMMARY', current)
        self.assertIn('不得根据任务目标、信息缺口或常识自行规划', current)
        old_strategy = draft_messages({})[0]['content']
        self.assertNotIn('## NEXT STEPS', old_strategy)
        candidate = completion_messages(prose('事实。'), {'sources': {}})[0]['content']
        self.assertIn('由后续Claimify处理', candidate)
        self.assertIn('不提取、抄写或改写摘要原句', candidate)
        self.assertIn('栏目不决定事实性质', candidate)
        self.assertEqual(tuple(parse_prose(prose('待办：用户要求回读论文A，尚未完成。'))), SECTIONS)

    def test_unverified_legacy_summary_keeps_pending_without_promoting_verification(self):
        pending = '待办：用户要求回读论文A，尚未完成。'
        body = legacy('None', pending=pending)
        envelope = {'version': PROSE_VERSION, 'claims': [], 'verification_status': 'unverified'}
        old = body + '\n\n## CLAIMS\n' + canonical_json(envelope)
        parts, claims = parse_summary(old, {})
        self.assertEqual(parts['NEXT STEPS'], pending)
        self.assertEqual(claims, [])
        migrated = saved_prose(old, {})
        self.assertEqual(parse_prose(migrated, allow_legacy=False)['SUMMARY'], pending)
        self.assertNotIn('## NEXT STEPS', migrated)
        self.assertEqual(summary_merge_view(old, {}), migrated)
        self.assertEqual(parse_summary(pack_prose_summary(migrated), {})[1], [])
        self.assertEqual(old, body + '\n\n## CLAIMS\n' + canonical_json(envelope))

    def test_legacy_list_quotes_and_task_lines_survive_merge_with_new_section_order(self):
        fact, artifact, pending_fact = '研究纳入200人。', '已生成 /tmp/a.txt。', '回读论文A失败。'
        todo = '待办：用户要求重试回读论文A，尚未完成。'
        body = legacy(fact, artifact, pending_fact + '\n' + todo)
        claims = [{'section': section, 'claim': text, 'references': [{'source_id': sid, 'quote': text}]}
                  for section, text, sid in [('SUMMARY', fact, 'm1'), ('ARTIFACTS', artifact, 'm2'),
                                             ('NEXT STEPS', pending_fact, 'm3')]]
        old = body + '\n\n## CLAIMS\n' + canonical_json(claims)
        sources = {'m1': fact, 'm2': artifact, 'm3': pending_fact}
        self.assertEqual(parse_summary(old, sources)[1], claims)
        merged = summary_merge_view(old, sources)
        items = parse_draft(merged, sources, allow_legacy=False)
        self.assertEqual([(c['section'], c['claim'], c['source_ids']) for c in items], [
            ('SUMMARY', fact, ['m1']), ('SUMMARY', pending_fact, ['m3']),
            ('SUMMARY', todo, []), ('ARTIFACTS', artifact, ['m2'])])
        self.assertEqual(summary_reference_quotes(old), {s: [t] for s, t in sources.items()})

    async def test_legacy_claimify_ledger_validates_original_offsets_before_migration(self):
        fact, pending = '研究纳入200人。', '待办：用户要求回读论文A，尚未完成。'
        body = legacy(fact, pending=pending)
        extraction = await extract_claims(body, '', AsyncMock(side_effect=[
            *responses(fact), answer('no_verifiable_claims')]))
        self.assertEqual(extraction['sentences'][1]['section'], 'NEXT STEPS')
        claims = [{**extraction['claims'][0], 'references': [
            {'source_id': 'm1', 'quote': fact, 'start': 0, 'end': len(fact)}]}]
        old = pack_summary(body, extraction, claims)
        parts, parsed = parse_summary(old, {'m1': fact})
        self.assertEqual(parsed, claims)
        self.assertEqual(parts['NEXT STEPS'][extraction['sentences'][1]['start']:
                                            extraction['sentences'][1]['end']], pending)
        merged = saved_prose(old, {'m1': fact})
        self.assertEqual(parse_prose(merged)['SUMMARY'], fact + '\n\n' + pending)
        self.assertEqual(summary_reference_quotes(old), {'m1': [fact]})

    async def test_new_generation_repairs_legacy_heading_once_and_keeps_pending(self):
        pending = '待办：用户要求回读论文A，尚未完成。'
        current = prose(pending)
        call = AsyncMock(side_effect=[legacy(pending=pending), current])
        result = await generate_prose(prose_messages({}), 'summary_merge', call, {})
        self.assertEqual(result, current)
        self.assertEqual(call.await_count, 2)
        request = json.loads(call.call_args.args[0][1]['content'])
        self.assertIn('已有未完成事项并入 SUMMARY', request['instruction'])
        with self.assertRaises(ValueError):
            pack_prose_summary(legacy(pending=pending))

    async def test_incremental_generation_receives_migrated_history_and_saves_three_sections(self):
        from service.summary_lifecycle import summarize_prose_first
        pending = '待办：用户要求回读论文A，尚未完成。'
        old = legacy(pending=pending) + '\n\n## CLAIMS\n' + canonical_json({
            'version': PROSE_VERSION, 'claims': [], 'verification_status': 'unverified'})
        store = MemoryStore()
        policy = ContextPolicy(group='三栏目迁移测试', semantic_verification=False,
            summary_lifecycle=SummaryLifecycleSettings(claimify_enabled=False))
        scope = HistoryScope(store, 'migration', 'session', None, context_policy=policy.snapshot())
        agent = make_agent()
        row = await store.append(scope, agent.name, 'llm_output', {'content': '新增检索没有结果。'}, 'new')
        expected = prose('新增检索没有结果。\n\n' + pending)
        agent._call_llm_impl = AsyncMock(return_value=expected)
        commit = AsyncMock(return_value=True)
        with bind_agent_history(scope):
            result = await summarize_prose_first(agent, scope, [row], CharCounter(),
                previous_summary=old, previous_sources={}, on_generated=commit)
        sent = json.loads(agent._call_llm_impl.call_args.args[1])['previous_summary']
        self.assertEqual(parse_prose(sent, allow_legacy=False)['SUMMARY'], pending)
        self.assertNotIn('## NEXT STEPS', sent)
        self.assertEqual(parse_summary(result, {})[0]['SUMMARY'], '新增检索没有结果。\n\n' + pending)
        self.assertEqual(commit.await_args.args[0], result)

    def test_missing_duplicate_reordered_or_extra_sections_are_rejected(self):
        valid = prose('事实。')
        for body in [valid.replace('## ARTIFACTS', '## UNKNOWN'),
                     valid.replace('## ARTIFACTS', '## SUMMARY'),
                     valid.replace('## SESSION INTENT', '## ARTIFACTS'),
                     valid + '\n## EXTRA\nNone']:
            with self.subTest(body=body), self.assertRaises(ValueError):
                parse_prose(body)
        self.assertEqual(parse_draft(draft(clause('待办：回读论文A。')), {}, allow_legacy=False)[0]['section'], 'SUMMARY')

    async def test_legacy_pending_section_remains_auditable_by_atomic_claimify(self):
        fact = '回读论文A失败。'
        body = legacy(pending=fact)
        output = canonical_json({'claims': [{'section': 'NEXT STEPS', 'claim': fact, 'source_ids': []}]})
        call = AsyncMock(side_effect=responses(fact))
        verify = AsyncMock()
        # 无候选证据时不构建真实索引；验证仍到达原子论断，缺证据显式报未完成。
        factory = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('测试禁止真实检索'))
        verify.return_value = {'status': 'incomplete', 'execution_status': 'completed',
                               'claims': [{'status': 'missing_evidence'}]}
        class Counter:
            def count_text(self, text): return len(text)
            def count_messages(self, messages): return sum(len(m['content']) for m in messages)
        report = await verify_atomic_output(output, body, {'sources': {}}, None, call,
            verify=verify, counter=Counter(), index_factory=factory)
        self.assertEqual(report['claimify']['claims'][0]['section'], 'NEXT STEPS')
        self.assertEqual(report['claimify']['claims'][0]['claim'], fact)
        self.assertEqual(call.await_count, 3)
        self.assertEqual(report['status'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
