"""同run续跑摘要递增，跨run隔离及原文回读权限。"""
import json
import sys
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import AsyncMock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'app'), str(Path(__file__).resolve().parent)]
from test_智能体独立历史 import MemoryStore, CharCounter, make_agent, summary_text, combined_summary
from service.agent_history import HistoryScope, bind_agent_history
from service.agent_context import prepare_agent_messages, ContextBudgetExceeded
from service.context_policy import ContextPolicy


class SessionSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_agent_resumes_summary_only_in_same_run(self):
        for name in ('ChiefArchitect', 'LeadWriter'):
            store=MemoryStore()
            policy=ContextPolicy(group='会话接续', summary_strategy='incremental-v3', semantic_verification=False,
                replay_agents=['ChiefArchitect', 'LeadWriter'])
            first=HistoryScope(store, 'run1', 'session1', 'user1', context_policy=policy.snapshot())
            second=replace(first, invocation_id='next-invocation')
            agent=make_agent(name)
            old=await store.append(first, name, 'llm_input', {'content':'旧结论。' + '研究过程' * 2000}, 'old')
            await store.append(first, name, 'llm_output', {'content':'完成'}, 'old')
            summary=AsyncMock(return_value=summary_text(old['source_id'], '旧结论。'))
            with bind_agent_history(first), patch('service.agent_context.get_qwen_token_counter', return_value=CharCounter()), \
                 patch('service.agent_context.scoped_input_budget', return_value=1000), \
                 patch('service.agent_context.summarize_history', summary):
                await prepare_agent_messages(agent, first, '系统', '本轮任务', 'current1', 3000)
            initial=await store.load(second, name)
            self.assertEqual(initial['summary_version'], 1)
            self.assertEqual(initial['covered_seq'], 2)
            self.assertEqual(await store.read_source(second, name, old['source_id']), old)
            new=await store.append(second, name, 'llm_input', {'content':'新结论。' + '新研究过程' * 2000}, 'new')
            await store.append(second, name, 'llm_output', {'content':'完成'}, 'new')
            merged=AsyncMock(return_value=combined_summary([(old['source_id'], '旧结论。'), (new['source_id'], '新结论。')]))
            with bind_agent_history(second), patch('service.agent_context.get_qwen_token_counter', return_value=CharCounter()), \
                 patch('service.agent_context.scoped_input_budget', return_value=1000), \
                 patch('service.agent_context.summarize_history', merged):
                messages=await prepare_agent_messages(agent, second, '系统', '当前新任务', 'current2', 3000)
            self.assertEqual(merged.call_args.kwargs['previous_summary'], initial['summary'])
            final=await store.load(second, name)
            self.assertEqual(final['summary_version'], 2)
            self.assertGreater(final['covered_seq'], initial['covered_seq'])
            self.assertIn('旧结论。', messages[1]['content'])
            self.assertIn('新结论。', messages[1]['content'])
            self.assertEqual(messages[-1]['content'], '当前新任务')
            self.assertEqual((await store.load(replace(second, session_id='other-session'), name))['messages'], [])
            other_name='LeadWriter' if name == 'ChiefArchitect' else 'ChiefArchitect'
            self.assertEqual((await store.load(second, other_name))['messages'], [])
            other_run = replace(second, run_id='run2')
            isolated = await store.load(other_run, name)
            self.assertEqual((isolated['summary'], isolated['summary_version'], isolated['messages']), ('', 0, []))
            with self.assertRaises(PermissionError):
                await store.read_source(other_run, name, old['source_id'])

    async def test_pending_other_run_does_not_enter_current_input(self):
        store=MemoryStore()
        policy=ContextPolicy(group='在途会话', summary_strategy='incremental-v3', semantic_verification=False)
        first=HistoryScope(store, 'run1', 'session', None, context_policy=policy.snapshot())
        old=await store.append(first, 'DeepScout', 'llm_input', {'content':'仍在处理' * 2000}, 'pending')
        # 模拟生产load附带的来源运行状态。
        store.data[store.key(first, 'DeepScout')]['messages'][0]['origin_run_status']='running'
        second=replace(first, run_id='run2', invocation_id='other')
        summarizer=AsyncMock()
        with bind_agent_history(second), patch('service.agent_context.get_qwen_token_counter', return_value=CharCounter()), \
             patch('service.agent_context.scoped_input_budget', return_value=1000), \
             patch('service.agent_context.summarize_history', summarizer):
            messages = await prepare_agent_messages(make_agent(), second, '系统', '新任务', 'current', 3000)
        summarizer.assert_not_awaited()
        self.assertEqual((await store.load(second, 'DeepScout'))['covered_seq'], 0)
        self.assertNotIn('仍在处理', str(messages))


if __name__ == '__main__':
    unittest.main()
