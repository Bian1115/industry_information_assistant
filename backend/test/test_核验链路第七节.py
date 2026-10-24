"""第7节运行链路与独立评分机制测试；不使用付费模型或伪造人工医学真值。"""
import copy
import json
import sys
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_智能体独立历史 import MemoryStore, CharCounter, HistoryScope, make_agent, bind_agent_history, content_hash
from test_摘要合并与引用分离 import clause,draft,citation
from service.context_policy import ContextPolicy,InputBudget,NLISettings,VerificationSettings
from service.verification_checks import locate_quote,check_coverage,check_output_fields,active_constraints,check_latest_contract
from service.summary_hierarchy import fragments_for_records,hierarchical_summary
from service.summary_citations import summarize_with_citations
from service.nli_verifier import verify_claims_sync,NLIBusyError,execution_error,reference_windows
from service.verification_budget import task_budget
from service.verification_trace import read_links,PREFIX
from eval.verification.scoring import digest,score_task,aggregate,validate_gold,nli_metrics,paired_difference


def fact():
    return {'fact_id':'f1','source_id':'s1','start':0,'end':18,'quote':'研究A于2022年纳入800人。',
            'topic':'研究A','entity':'研究A','statement':'研究A于2022年纳入800人。','reason':'任务要求',
            'kind':'fact','fields':['2022','800'],'source_text_hash':content_hash('研究A于2022年纳入800人。')}


def policy(**changes):
    # 本文件原有链路用例保留为v1冻结规则回归；v2顺序单独验证。
    changes = {"rules_version": "verification-v1", "coverage": True, **changes}
    return ContextPolicy(group='新链路机制测试', summary_strategy='chunk-merge-v1',
                         nli=NLISettings(), verification=VerificationSettings(**changes),
                         summary_budget=InputBudget(context_window=32768,max_input=30000,safety_tokens=1024))


class CoverageTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_year_detected_even_if_nli_supports_short_claim(self):
        claims=[{'claim':'研究A纳入800人。','references':[{'source_id':'s1','quote':fact()['quote']}]}]
        call=AsyncMock(return_value=json.dumps({'status':'preserved','claim_indexes':[0],'reason':'桩模拟错误宽松判定'}))
        result=await check_coverage([fact()],claims,call,VerificationSettings(coverage=True))
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(result['items'][0]['fields'][0]['status'],'omitted_or_changed')

    async def test_entity_swap_and_suggestion_do_not_preserve_fact(self):
        for text in ['研究B于2022年纳入800人。','建议：研究A于2022年纳入800人。']:
            call=AsyncMock(return_value=json.dumps({'status':'preserved','claim_indexes':[0]}))
            result=await check_coverage([fact()],[{'claim':text,'references':[{'source_id':'s1'}]}],call,VerificationSettings(coverage=True))
            self.assertEqual(result['status'],'rejected')

    async def test_latest_contract_replaces_old_constraint_without_reinjecting(self):
        old={**fact(),'kind':'constraint','quote':'5000字'}
        call=AsyncMock(return_value=json.dumps([{'fact_id':'f1','active':False,'reason':'最新要求3000字'}]))
        self.assertEqual(await active_constraints([old],{'篇幅':'3000字'},call),[])
        check=AsyncMock(return_value=json.dumps({'items':[{'status':'changed','quote':'5000字','reason':'不是最新要求'}]}))
        report=await check_latest_contract({'篇幅':'3000字'},[{'claim':'5000字'}],check)
        self.assertEqual(report['status'],'rejected')

    def test_output_number_requires_actual_source(self):
        report=check_output_fields([{'claim':'研究A有80人。','references':[{'quote':'研究A有800人。'}]}])
        self.assertEqual(report['status'],'rejected')

    def test_quote_context_keeps_qualifier_and_duplicate_needs_position(self):
        text='仅在男性亚组中观察到风险升高。其他人群未见差异。'
        ref=locate_quote({'source_id':'s','quote':'风险升高'},text,'research_source')
        self.assertIn('仅在男性亚组',ref['context'])
        self.assertIn('未见差异',ref['context'])
        with self.assertRaisesRegex(ValueError,'不唯一'):
            locate_quote({'quote':'结论。'},'结论。结论。','research_source')
        self.assertEqual(locate_quote({'quote':'结论。','start':3,'end':6},'结论。结论。','research_source')['start'],3)

    def test_long_sentence_or_missing_context_is_not_truncated(self):
        model=SimpleNamespace(tokenizer=lambda a,b,**kw:{'input_ids':list(range(len(a)+len(b)))},model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)))
        windows,error=reference_windows({'quote':'尾部','context':'甲'*600+'不成立。','context_complete':True},'结论',model,NLISettings())
        self.assertFalse(windows)
        self.assertEqual(error,'input_too_long')
        windows,error=reference_windows({'quote':'句子','context':'甲。乙。','context_complete':False},'结论',model,NLISettings())
        self.assertEqual(error,'context_incomplete')

    def test_busy_timeout_and_missing_model_are_execution_errors(self):
        self.assertEqual(execution_error(NLIBusyError()),'resource_busy')
        self.assertEqual(execution_error(TimeoutError()),'timeout')
        self.assertEqual(execution_error(FileNotFoundError()),'model_missing')

    def test_long_tail_contradiction_cannot_be_hidden_by_support_window(self):
        class Model:
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=30))
            def tokenizer(self,a,b,**kwargs):
                return {'input_ids':list(range(len(a)+len(b)))}
            def predict(self,pairs,settings):
                return [{'status':'scored','scores':{'entailment':.01 if '不支持' in a else .98,
                        'neutral':.01,'contradiction':.98 if '不支持' in a else .01}} for a,b in pairs]
        claims=[{'section':'SUMMARY','claim':'风险升高。','references':[{'source_id':'s1','quote':'风险升高。',
                 'context':'研究A的男性亚组显示风险升高。'*3+'其他人群的完整分析不支持这一结论。',
                 'context_complete':True,'source_type':'research_source'}]}]
        with patch('service.nli_verifier.get_nli',return_value=Model()):
            report=verify_claims_sync(claims,NLISettings(max_length=30))
        self.assertEqual(report['status'],'rejected')
        self.assertEqual(report['claims'][0]['content_status'],'contradiction')
        self.assertGreater(report['window_count'],1)

    def test_assistant_statement_cannot_validate_research_claim(self):
        model=Mock()
        claims=[{'section':'SUMMARY','claim':'已证实有效。','references':[{'source_id':'s','quote':'已证实有效。','source_type':'assistant_history'}]}]
        with patch('service.nli_verifier.get_nli',return_value=model):
            report=verify_claims_sync(claims,NLISettings())
        self.assertEqual(report['status'],'rejected')
        model.predict.assert_not_called()


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_uses_real_base_path_and_isolates_each_group_repeat(self):
        from eval.scripts.run_verification_replay import execute
        def factory(*args):
            agent=make_agent(response='固定下一步任务产出')
            agent.client.close=AsyncMock()
            return agent
        a=ContextPolicy(group='甲组',semantic_summary=False,semantic_verification=False)
        b=a.model_copy(update={'group':'乙组','artifact_offload':False})
        snapshot={'tasks':[{'task_id':'t','model':'test-model','records':[],
                           'system_prompt':'按材料作答','next_task':'继续','contract':{}}]}
        budget={'max_requests':2,'max_seconds':60,'max_tokens_by_model':{'test-model':20000}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'LLM_API_KEY':'test','LLM_BASE_URL':'http://unused'}), \
                patch('eval.scripts.run_verification_replay.ReplayAgent',side_effect=factory), \
                patch('service.qwen_token_counter.get_qwen_token_counter',return_value=CharCounter()), \
                patch('service.agent_context.get_qwen_token_counter',return_value=CharCounter()), \
                patch('service.deep_research_v2.agents.base.record_llm_call'), \
                patch('service.deep_research_v2.agents.base.flush_pending'):
            rows=await execute(snapshot,[a,b],Path(directory),2,budget)
            self.assertEqual(len(rows),4)
            self.assertEqual(len({r['run_id'] for r in rows}),4)
            self.assertTrue(all(r['completed'] for r in rows))
            self.assertTrue(all(len(r['requests'])==1 for r in rows))
            self.assertEqual(len(list(Path(directory).glob('任务-*/工作历史.json'))),4)
            self.assertTrue((Path(directory)/'运行计划.json').is_file())

    async def test_one_repair_rebuilds_citations_and_rechecks(self):
        store=MemoryStore()
        p=policy(nli_support=False)
        scope=HistoryScope(store,'run','session',None,context_policy=p.snapshot())
        row=await store.append(scope,'DeepScout','tool_output',{'tool_name':'v2_web_fulltext_read','value':{'text':fact()['quote']}},'c1')
        from service.agent_context import source_text
        text=source_text(row)
        start=text.index(fact()['quote'])
        extracted={**fact(),'start':start}
        for key in ['fact_id','source_id','source_text_hash','end']:
            extracted.pop(key,None)
        bad=clause('研究A纳入800人。',row['source_id'])
        good=clause(fact()['quote'],row['source_id'])
        agent=make_agent()
        agent._call_llm_impl=AsyncMock(side_effect=[json.dumps([extracted],ensure_ascii=False),draft(bad),
            citation(bad,[(row['source_id'],fact()['quote'])]),json.dumps({'status':'preserved','claim_indexes':[0]}),
            draft(good),citation(good,[(row['source_id'],fact()['quote'])]),json.dumps({'status':'preserved','claim_indexes':[0]})])
        verification={}
        with bind_agent_history(scope):
            result=await summarize_with_citations(agent,scope,[row],CharCounter(),verification_out=verification)
        self.assertIn(fact()['quote'],result)
        self.assertEqual(verification['attempt'],1)
        stages=[r['payload']['stage'] for r in (await store.load(scope,'DeepScout'))['messages'] if r['kind']=='summary_stage']
        self.assertEqual(stages.count('summary_repair'),1)
        self.assertEqual(stages.count('coverage_check'),2)
        self.assertEqual(stages.count('citation_completion'),2)
        self.assertEqual(verification['omission_check'],'completed')

    async def test_second_rejection_and_infrastructure_failure_do_not_loop(self):
        store=MemoryStore();p=policy(coverage=False,key_fields=False)
        scope=HistoryScope(store,'run','session',None,context_policy=p.snapshot())
        row=await store.append(scope,'DeepScout','llm_output',{'content':'原句。'},'c1')
        item=clause('原句。',row['source_id'])
        agent=make_agent()
        agent._call_llm_impl=AsyncMock(side_effect=[draft(item),citation(item,[(row['source_id'],'原句。')]),draft(item),citation(item,[(row['source_id'],'原句。')])])
        reject={'status':'rejected','execution_status':'completed','claims':[{'status':'unknown','claim':'原句。','evidence':[]}]}
        with bind_agent_history(scope),patch('service.nli_verifier.verify_claims',new=AsyncMock(return_value=reject)),self.assertRaises(ValueError):
            await summarize_with_citations(agent,scope,[row],CharCounter())
        self.assertEqual(agent._call_llm_impl.await_count,4)
        self.assertEqual((await store.load(scope,'DeepScout'))['covered_seq'],0)
        agent._call_llm_impl=AsyncMock(side_effect=[draft(item),citation(item,[(row['source_id'],'原句。')])])
        with bind_agent_history(scope),patch('service.nli_verifier.verify_claims',new=AsyncMock(side_effect=NLIBusyError())),self.assertRaises(ValueError):
            await summarize_with_citations(agent,scope,[row],CharCounter())
        self.assertEqual(agent._call_llm_impl.await_count,2)

    async def test_decontract_does_not_inject_hidden_contract_during_repair(self):
        store=MemoryStore();p=policy(nli_support=False,coverage=False,key_fields=True).model_copy(update={'pinned_contract':False})
        scope=HistoryScope(store,'r','s',None,context_policy=p.snapshot(),contract={'篇幅':'3000字'})
        row=await store.append(scope,'DeepScout','llm_output',{'content':'操作完成。'},'c1')
        item=clause('操作记录：操作完成。',row['source_id'])
        agent=make_agent()
        agent._call_llm_impl=AsyncMock(side_effect=['[]',draft(item),citation(item,[(row['source_id'],'操作完成。')]),json.dumps({'items':[{'status':'omitted','quote':'','reason':'未保留当前要求'}]})])
        with bind_agent_history(scope),self.assertRaises(ValueError):
            await summarize_with_citations(agent,scope,[row],CharCounter())
        calls=agent._call_llm_impl.call_args_list
        self.assertNotIn('3000字',calls[1].args[1])
        self.assertEqual(len(calls),4) # 隐藏契约的检查问题不能流入生成/修复。

    async def test_hierarchy_preserves_tail_and_enforces_levels(self):
        text='甲。'*1500+'末尾不得遗漏。'
        row={'source_id':'s1','kind':'llm_output','payload':{'content':text},'call_id':'c','content_hash':content_hash({'content':text})}
        fragments=list(fragments_for_records([row],CharCounter(),200,lambda f:[{'role':'user','content':f['text']}]))
        self.assertEqual(''.join(f['text'] for f in fragments),text)
        self.assertIn('末尾不得遗漏。',fragments[-1]['text'])
        for left,right in zip(fragments,fragments[1:]):
            self.assertEqual(left['end'],right['start'])
        p=policy().model_copy(update={'summary_max_chunks':100,'summary_max_merge_levels':1})
        call=AsyncMock(return_value=draft(clause('结论。','s1')))
        result=await hierarchical_summary(fragments,'',{},call,CharCounter(),1000,p,{'s1':text})
        self.assertIn('结论。',result)

    def test_task_budget_includes_retries_and_has_no_cross_run_state(self):
        config={'max_requests':2,'max_seconds':60,'max_tokens_by_model':{'m':1000}}
        with task_budget(config) as ledger:
            ledger.reserve('m',100,100)
            ledger.reserve('m',100,100)
            with self.assertRaisesRegex(ValueError,'调用预算'):
                ledger.reserve('m',1,1)
            self.assertIsNone(ledger.report()['usd'])
        with task_budget(config) as ledger:
            self.assertEqual(ledger.requests,[])


class ScoringTests(unittest.TestCase):
    def test_scoring_cli_keeps_unexecuted_checkpoint_targets_in_denominator(self):
        from eval.scripts.score_verification_chain import main
        task,run,j=self.fixture()
        task['checkpoint_annotations']={'g':{'summary_expected':True,
            'summary_input_visible_fact_ids':['f1'],'offloaded_required_fact_ids':['f1']}}
        run['run_id']='done';j['artifact_hash']=digest(run)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            data={'annotation_origin':'synthetic_mechanism','annotation_version':'v1','tasks':[task]}
            documents={'标注.json':data,'产物.json':[run],'人审.json':{'done':j},'计划.json':[
                {'task_id':'t','group':'g','repeat':0,'run_id':'done'},
                {'task_id':'t','group':'g','repeat':1,'run_id':'missing'}]}
            for name,value in documents.items():
                (path/name).write_text(json.dumps(value,ensure_ascii=False))
            argv=['score','--dataset',str(path/'标注.json'),'--runs',str(path/'产物.json'),
                  '--judgments',str(path/'人审.json'),'--plan',str(path/'计划.json'),
                  '--output',str(path/'结果'),'--report',str(path/'分析.md')]
            with patch.object(sys,'argv',argv):
                main()
            result=json.loads((path/'结果/核验链路评分结果.json').read_text())
            metrics=result['aggregate']['g']['metrics']
            self.assertEqual(metrics['summary_retention']['micro']['required'],2)
            self.assertEqual(metrics['read_target_visible']['micro']['required'],2)
            self.assertEqual(metrics['final_fact_use']['micro']['required'],2)

    def fixture(self):
        text=fact()['quote'];f={**fact(),'end':len(text),'needed_before_sequence':10}
        task={'task_id':'t','trajectory_id':'trace','split':'test','facts':[f],
              'sources':{'s1':{'text':text,'text_hash':digest(text)}},
              'constraints':[{'key':'篇幅','type':'length','mode':'max','value':30}]}
        run={'task_id':'t','group':'g','repeat':0,'completed':True,'final_body':'研究A于2022年纳入80人。',
             'summary_enabled':True,'summary_triggered':True,'summary_delivered':True,'summary_body':text,
             'reads':[],'requests':[]}
        judgment={'artifact_hash':digest(run),'origin':'synthetic_mechanism','summary_input_visible_fact_ids':['f1'],
                  'offloaded_required_fact_ids':['f1'], 'summary':{'correct_fact_ids':['f1'],'correct_field_ids':['f1:0','f1:1']},
                  'final':{'correct_fact_ids':[],'correct_field_ids':['f1:0']},'task_accepted':False}
        return task,run,judgment

    def test_summary_correct_final_wrong_and_unfinished_remain_in_denominator(self):
        task,run,j=self.fixture()
        row=score_task(task,run,j)
        self.assertEqual(row['metrics']['summary_retention']['value'],1)
        self.assertEqual(row['metrics']['final_fact_use']['value'],0)
        self.assertEqual(row['metrics']['read_target_visible']['required'],1)
        run['completed']=False;j['artifact_hash']=digest(run)
        row=score_task(task,run,j)
        self.assertEqual(row['metrics']['task_success']['value'],0)
        self.assertEqual(row['metrics']['final_field_use']['correct'],0)
        self.assertEqual(aggregate([row])['g']['failed'],1)

    def test_read_wrong_fragment_not_sent_and_late_all_fail(self):
        task,run,j=self.fixture();text=task['sources']['s1']['text']
        read={'read_id':'r1','source_id':'s1','start':0,'end':len(text),'text':text,
              'source_text_hash':digest(text),'text_hash':digest(text)}
        run['reads']=[read];j['artifact_hash']=digest(run)
        self.assertEqual(score_task(task,run,j)['metrics']['read_target_visible']['correct'],0)
        request={'messages':[{'role':'user','content':PREFIX+json.dumps(read,ensure_ascii=False)}]}
        run['requests']=[{'sequence':11,'request':request,'request_hash':digest(request),'execution_status':'completed'}]
        j['artifact_hash']=digest(run)
        self.assertEqual(score_task(task,run,j)['metrics']['read_target_visible']['correct'],0)
        run['requests'][0]['sequence']=5;j['artifact_hash']=digest(run)
        self.assertEqual(score_task(task,run,j)['metrics']['read_target_visible']['correct'],1)
        run['reads'][0]['end']=3;run['reads'][0]['text_hash']=digest(text[:3]);j['artifact_hash']=digest(run)
        self.assertEqual(score_task(task,run,j)['metrics']['read_target_visible']['correct'],0)

    def test_label_hash_duplicates_and_split_leakage_are_rejected(self):
        task,run,j=self.fixture()
        j['artifact_hash']='wrong'
        with self.assertRaises(ValueError):score_task(task,run,j)
        data={'annotation_origin':'human','annotation_version':'v1','tasks':[task]}
        validate_gold(data)
        other=copy.deepcopy(task);other.update(task_id='t2',split='selection');data['tasks'].append(other)
        with self.assertRaises(ValueError):validate_gold(data)

    def test_availability_errors_not_counted_as_contradiction_or_na(self):
        metrics=nli_metrics([{'label':'supported','execution_status':'resource_busy'},
                             {'label':'insufficient_support','execution_status':'completed','scores':{'entailment':.95,'neutral':.04,'contradiction':.01}}])
        self.assertEqual(metrics['false_accept']['value'],1)
        self.assertEqual(metrics['false_reject']['value'],1)
        self.assertEqual(metrics['availability']['value'],.5)

    def test_pairing_requires_same_task_repeat(self):
        task,run,j=self.fixture();row=score_task(task,run,j)
        other=copy.deepcopy(row);other.update(group='other',repeat=1)
        self.assertEqual(paired_difference([row,other],'g','other')['tasks'],0)

    def test_calibration_cannot_use_same_trajectory_for_selection_and_test(self):
        from eval.scripts.calibrate_nli_thresholds import select_thresholds
        common={'execution_status':'completed','scores':{'entailment':.95,'neutral':.04,'contradiction':.01}}
        data={'annotation_origin':'synthetic_mechanism','annotation_version':'test','model_id':'m','revision':'r','pairs':[
            {**common,'trajectory_id':'a','split':'selection','label':'supported'},
            {**common,'trajectory_id':'b','split':'selection','label':'insufficient_support'},
            {**common,'trajectory_id':'a','split':'test','label':'supported'}]}
        with self.assertRaisesRegex(ValueError,'泄漏'):
            select_thresholds(data)


if __name__=='__main__':
    unittest.main()
