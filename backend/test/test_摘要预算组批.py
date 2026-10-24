"""按预算组批和一次额外合并的机制回归；模型输出使用桩。"""
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from test_智能体独立历史 import MemoryStore, CharCounter, HistoryScope, make_agent, bind_agent_history, content_hash
from test_摘要合并与引用分离 import clause, draft, citation
from service.context_policy import ContextPolicy
from service.summary_citations import summarize_with_citations, draft_messages
from service.summary_hierarchy import BATCH_POLICY, fragments_for_records, pack_batches, hierarchical_summary
from service.verification_checks import messages

PROMPT = (Path(__file__).resolve().parents[1]/'app/prompts/摘要正文生成与合并.txt').read_text()


def build(items, previous=''):
    return draft_messages({'new_records':items, 'previous_summary':previous, 'latest_contract':{}})


def fragment(i, text):
    return {'source_id':'s1', 'kind':'llm_output', 'start':i*1000, 'end':i*1000+len(text), 'text':text}


def policy():
    # 故意保留旧的1块/99层值，确认不会限制新组批，也不会允许无限层级。
    return ContextPolicy(group='摘要组批机制', summary_strategy='chunk-merge-v1', semantic_verification=False,
                         summary_max_chunks=1, summary_max_merge_levels=99)


class PackingTests(unittest.TestCase):
    def test_short_records_share_batch_without_being_split_for_remaining_space(self):
        counter = CharCounter()
        items = [fragment(i, '甲'*n) for i,n in enumerate([100,100,500])]
        budget = counter.count_messages(build([items[2]]))
        batches = pack_batches(items, counter, budget, build)
        self.assertEqual(batches, [items[:2],items[2:]])
        self.assertEqual([x for batch in batches for x in batch],items)
        self.assertTrue(all(counter.count_messages(build(batch))<=budget for batch in batches))

    def test_oversize_item_after_flush_is_rejected(self):
        counter = CharCounter()
        small,large = fragment(0,'短'),fragment(1,'长'*2000)
        with self.assertRaisesRegex(ValueError,'单个片段'):
            pack_batches([small,large],counter,counter.count_messages(build([small])),build)

    def test_long_record_keeps_all_offsets_and_tail(self):
        text = '正文。'*3000+'末尾不能丢失。'
        payload = {'content':text}
        row = {'source_id':'s1','kind':'llm_output','call_id':'c','payload':payload,'content_hash':content_hash(payload)}
        counter = CharCounter()
        pieces = list(fragments_for_records([row],counter,800,lambda f:build([f])))
        self.assertGreater(len(pieces),1)
        self.assertEqual(''.join(p['text'] for p in pieces),text)
        self.assertEqual(pieces[0]['start'],0)
        self.assertEqual(pieces[-1]['end'],len(text))
        for left,right in zip(pieces,pieces[1:]):
            self.assertEqual(left['end'],right['start'])
        self.assertTrue(all(p['source_id']=='s1' and p['call_id']=='c' for p in pieces))
        self.assertTrue(all(counter.count_messages(build([p]))<=800 for p in pieces))

    def test_fixed_prompt_cannot_fit_is_an_explicit_failure(self):
        payload={'content':'原文'}
        row={'source_id':'s1','kind':'llm_output','call_id':'c','payload':payload,'content_hash':content_hash(payload)}
        with self.assertRaisesRegex(ValueError,'固定提示'):
            list(fragments_for_records([row],CharCounter(),1,lambda f:build([f])))


class SummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_more_than_eight_short_records_use_one_request(self):
        pieces=[fragment(i,'短记录。') for i in range(20)]
        call=AsyncMock(return_value=draft(clause('结论。','s1')))
        await hierarchical_summary(pieces,'',{},call,CharCounter(),10000,policy(),{'s1':'原文'})
        call.assert_awaited_once()
        self.assertEqual(len(json.loads(call.call_args.args[0][1]['content'])['new_records']),20)

    async def exercise_merge(self, *, leaf_size, intermediate_size):
        pieces=[fragment(i,'原'*1000) for i in range(8)]
        counter=CharCounter()
        budget=counter.count_messages(build(pieces[:2]))
        calls=[]
        async def call(msgs,stage):
            self.assertLessEqual(counter.count_messages(msgs),budget)
            payload=json.loads(msgs[1]['content'])
            calls.append((stage,payload))
            length=leaf_size if stage=='summary_chunk' else intermediate_size
            return draft(clause('甲'*length+'。','s1'))
        return pieces,counter,budget,calls,call

    async def test_batches_summarized_then_merged_directly(self):
        pieces,counter,budget,calls,call=await self.exercise_merge(leaf_size=5,intermediate_size=5)
        await hierarchical_summary(pieces,'',{},call,counter,budget,policy(),{'s1':'原文'})
        leaves=[p for stage,p in calls if stage=='summary_chunk']
        self.assertGreater(len(leaves),1)
        self.assertLess(len(leaves),len(pieces))
        self.assertEqual([f for p in leaves for f in p['new_records']],pieces)
        self.assertEqual([p['merge_level'] for stage,p in calls if stage=='summary_hierarchical_merge'],[1])

    async def test_one_extra_group_merge_then_final_merge(self):
        pieces,counter,budget,calls,call=await self.exercise_merge(leaf_size=800,intermediate_size=5)
        await hierarchical_summary(pieces,'',{},call,counter,budget,policy(),{'s1':'原文'})
        levels=[p['merge_level'] for stage,p in calls if stage=='summary_hierarchical_merge']
        self.assertGreater(levels.count(1),1)
        self.assertEqual(levels[-1],2)
        self.assertEqual(levels.count(2),1)

    async def test_still_too_long_after_one_group_merge_stops_without_another_call(self):
        pieces,counter,budget,calls,call=await self.exercise_merge(leaf_size=800,intermediate_size=1800)
        with self.assertRaisesRegex(ValueError,'分批合并一次后仍超出'):
            await hierarchical_summary(pieces,'',{},call,counter,budget,policy(),{'s1':'原文'})
        levels=[p['merge_level'] for stage,p in calls if stage=='summary_hierarchical_merge']
        self.assertGreater(len(levels),1)
        self.assertEqual(set(levels),{1})

    async def test_previous_summary_remains_in_merge_input(self):
        pieces,counter,budget,calls,call=await self.exercise_merge(leaf_size=5,intermediate_size=5)
        old=draft(clause('旧要求仍有效。','s1'))
        await hierarchical_summary(pieces,old,{},call,counter,budget,policy(),{'s1':'原文'})
        self.assertTrue(any(old in p.get('summary_parts',[]) for _,p in calls))

    async def test_real_summary_entry_no_longer_rejects_many_records_and_keeps_citations(self):
        store=MemoryStore()
        scope=HistoryScope(store,'r','s',None,context_policy=policy().snapshot())
        rows=[await store.append(scope,'DeepScout','llm_output',{'content':'原句。'},f'c{i}') for i in range(12)]
        item=clause('原句。',rows[0]['source_id'])
        agent=make_agent()
        agent._call_llm_impl=AsyncMock(side_effect=[draft(item),citation(item,[(rows[0]['source_id'],'原句。')])])
        with bind_agent_history(scope):
            result=await summarize_with_citations(agent,scope,rows,CharCounter())
        self.assertIn('## CLAIMS',result)
        self.assertEqual(agent._call_llm_impl.await_count,2)
        first=agent._call_llm_impl.call_args_list[0]
        self.assertEqual(len(json.loads(first.args[1])['new_records']),12)
        events=(await store.load(scope,'DeepScout'))['messages']
        stages=[r['payload'] for r in events if r['kind']=='summary_stage']
        self.assertTrue(all(s['batch_policy']==BATCH_POLICY for s in stages))


if __name__=='__main__':
    unittest.main()
