"""会话成果契约变更与规划输入回归；不依赖执行层追问路由。"""
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'app')]
from service.deep_research_v2.contract_updates import ContractPatch, apply_contract_patch, parent_contract, contract_query
from service.deep_research_v2.state import create_initial_state
from service.deep_research_v2.agents.architect import ChiefArchitect


def parent():
    return {'report_hash':'parent-hash','checkpoint_id':'cp','query':'旧研究问题',
            'research_contract':{'contract_revision':3,'query':'旧研究问题',
                'required_entities':['成年人'],'time_range':'2020—2025',
                'active_constraints':{'max_body_chars':5000}},
            'active_constraints':{'max_body_chars':5000}}


def operation(field='max_body_chars', value=3000, op='replace', quote='字数上限改成3000字'):
    return dict(op=op, field=field, value=value, quote=quote)


def patch_data(*ops):
    return {'base_revision':3,'operations':list(ops)}


def merged(query, *ops, reuse=True):
    return apply_contract_patch(parent_contract(parent()),patch_data(*ops),
        original_query=query,resolved_query='完整本轮需求',instructions=[query],reuse_previous=reuse)


def plan():
    return {'outline':[{'id':f'sec_{i}','title':f'章节{i}'} for i in range(4)],
            'key_entities':[{'name':'模型建议实体'}], 'hypotheses':[],
            'research_contract':{'required_entities':['错误覆盖'],'active_constraints':{'max_body_chars':99999}}}


def architect(output):
    obj=ChiefArchitect.__new__(ChiefArchitect)
    obj.name='ChiefArchitect';obj.logger=logging.getLogger('contract-without-router')
    obj.add_message=Mock();obj.call_llm=AsyncMock(return_value=json.dumps(output,ensure_ascii=False))
    return obj


def state_v2(query='项目运营分析'):
    return create_initial_state(query, 'session', [])


class MergeTests(unittest.TestCase):
    def test_replace_keeps_other_constraints_without_mutating_parent(self):
        baseline=parent_contract(parent());before=deepcopy(baseline)
        new=apply_contract_patch(baseline,patch_data(operation()),original_query='字数上限改成3000字',
            resolved_query='完整需求',instructions=['调整篇幅'])
        self.assertEqual(new['active_constraints'],{'max_body_chars':3000})
        self.assertEqual(new['required_entities'],['成年人'])
        self.assertEqual(new['time_range'],'2020—2025')
        self.assertEqual(new['contract_revision'],4)
        self.assertEqual(baseline,before)

    def test_remove_length_is_not_old_length_inheritance(self):
        new=merged('取消字数限制',operation(op='remove',value=None,quote='取消字数限制'))
        self.assertEqual(new['active_constraints'],{})

    def test_new_topic_uses_empty_baseline(self):
        new=merged('改为研究新能源，中文输出',operation('language','中文',op='add',quote='中文输出'),reuse=False)
        self.assertNotIn('required_entities',new);self.assertNotIn('time_range',new)
        self.assertEqual(new['language'],'中文')

    def test_stale_version_cannot_be_applied_twice(self):
        new=merged('字数上限改成3000字',operation())
        with self.assertRaisesRegex(ValueError,'版本'):
            apply_contract_patch(new,patch_data(operation()),original_query='字数上限改成3000字',
                resolved_query='新需求',instructions=[])

    def test_bad_quote_does_not_mutate_input(self):
        base=parent_contract(parent());before=deepcopy(base)
        with self.assertRaisesRegex(ValueError,'原话'):
            apply_contract_patch(base,patch_data(operation(quote='伪造依据')),original_query='字数上限改成3000字',
                resolved_query='完整需求',instructions=[])
        self.assertEqual(base,before)

    def test_schema_rejects_unknown_fields_and_bad_types(self):
        for op in [operation('source_policy',{}),operation(value=True),operation(value=99),
                   operation('required_entities','A'),operation(op='remove',value=3000),
                   operation('required_metrics',[{'id':'m','name':'量','years':2025}])]:
            with self.subTest(op=op),self.assertRaises(ValueError):
                ContractPatch.model_validate(patch_data(op))

    def test_add_replace_remove_have_distinct_semantics(self):
        for op in [operation(op='add'),operation('language',None,op='remove'),operation('language','中文',op='replace')]:
            with self.subTest(op=op),self.assertRaises(ValueError):merged('字数上限改成3000字',op)

    def test_duplicate_fields_rejected(self):
        with self.assertRaisesRegex(ValueError,'一次'):
            merged('字数上限改成3000字',operation(),operation())

    def test_scope_change_requires_updating_old_metric_scope(self):
        base=parent_contract(parent());base['required_metrics']=[{'id':'m','name':'数量','years':['2020']}]
        with self.assertRaisesRegex(ValueError,'required_metrics'):
            apply_contract_patch(base,patch_data(operation('time_range','2025',quote='仅研究2025年')),
                original_query='仅研究2025年',resolved_query='完整需求',instructions=[])


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_architect_preserves_committed_contract(self):
        state=create_initial_state('字数上限改成3000字','session',[])
        state['research_contract']=merged(state['original_query'],operation())
        state['active_constraints']=deepcopy(state['research_contract']['active_constraints'])
        state['query']=contract_query(state['research_contract'])
        agent=architect(plan());await agent._initial_planning(state)
        self.assertEqual(state['research_contract']['required_entities'],['成年人'])
        self.assertEqual(state['research_contract']['active_constraints'],{'max_body_chars':3000})
        self.assertEqual(state['research_contract']['contract_revision'],4)
        self.assertIn('已提交研究契约',agent.call_llm.call_args.kwargs['user_prompt'])

    async def test_architect_does_not_resurrect_removed_limit(self):
        state=create_initial_state('取消字数限制','session',[])
        state['research_contract']=merged(state['original_query'],operation(op='remove',value=None,quote='取消字数限制'))
        state['active_constraints']={};state['query']=contract_query(state['research_contract'])
        await architect(plan())._initial_planning(state)
        self.assertEqual(state['research_contract']['active_constraints'],{})
        self.assertEqual(state['active_constraints'],{})

    async def test_architect_rejects_legacy_pending_contract_before_model_call(self):
        state=create_initial_state('取消字数限制','session',[])
        state['research_contract']=parent_contract(parent())
        state['research_contract']['pending_contract_update']={'base_contract':parent_contract(parent())}
        before=deepcopy(state);agent=architect(plan())
        with self.assertRaisesRegex(ValueError,'旧追问契约更新已停用'):
            await agent._initial_planning(state)
        agent.call_llm.assert_not_awaited()
        agent.add_message.assert_not_called()
        self.assertEqual(state,before)

    async def test_first_research_plans_without_legacy_quote_protocol(self):
        state=state_v2();agent=architect(plan())
        await agent._initial_planning(state)
        self.assertEqual(agent.call_llm.await_count,1)
        prompt=agent.call_llm.call_args.kwargs['user_prompt']
        self.assertNotIn('followup_contract_update',prompt)
        self.assertNotIn('quote',prompt)
        self.assertNotIn('契约更新任务',prompt)
        self.assertEqual(state['phase'],'planning')


if __name__=='__main__':unittest.main()
