import importlib.util
import math
import json
import tempfile
import os
from unittest.mock import patch
import unittest
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/复评五篇质量与复核一致性.py'
spec=importlib.util.spec_from_file_location('quality_agreement',SCRIPT)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class QualityAgreementTests(unittest.TestCase):
    def test_numeric_agreement_preserves_absolute_error_and_signed_bias(self):
        result=M.agreement([(5,5),(7,5),(4,5)])
        self.assertEqual(result['n'],3)
        self.assertAlmostEqual(result['mae'],1)
        self.assertAlmostEqual(result['rmse'],math.sqrt(5/3))
        self.assertAlmostEqual(result['signed_bias'],1/3)
        self.assertEqual(result['exact_count'],1)
        self.assertEqual(result['within_1_count'],2)
        self.assertIsNone(result['pearson'])  # constant reference is not perfect correlation

    def test_ranking_ties_receive_average_ranks(self):
        self.assertEqual(M.ranks([1,1,3,2]),[1.5,1.5,4,3])
        self.assertAlmostEqual(M.correlation([1,2,3],[3,2,1]),-1)

    def test_unknowns_remain_visible_and_are_excluded_from_determinate_subset(self):
        result=M.categorical([('full','full'),('full','unknown'),('unknown','unknown'),('partial','full')])
        self.assertEqual((result['n'],result['exact']),(4,2))
        self.assertEqual((result['both_determinate_n'],result['both_determinate_exact']),(2,1))
        self.assertEqual(result['reference_unknown'],2)
        self.assertEqual(result['prediction_unknown'],1)
        self.assertEqual(result['confusion_reference_rows']['full'],{'full':1,'partial':1})

    def test_bootstrap_keeps_whole_tasks_paired_and_rejects_unmatched_tasks(self):
        result=M.task_bootstrap_delta({1:[1,1],2:[2,2]},{1:[0,0],2:[3,3]})
        self.assertEqual(result['resamples'],4)
        self.assertEqual(result['tasks'],2)
        self.assertEqual(result['delta_mae_new_minus_old'],0)
        self.assertLess(result['task_cluster_bootstrap_ci95'][0],0)
        self.assertGreater(result['task_cluster_bootstrap_ci95'][1],0)
        with self.assertRaises(ValueError):M.task_bootstrap_delta({1:[1]},{2:[1]})

    def test_invalid_scores_are_not_accepted_as_completed_calls(self):
        def packet(score):return {'dimensions':{d:{'score':score,'reason':'评分理由'} for d in M.DIMS}}
        M.validate_quality(packet(5.5))
        for score in [True,-1,11,float('nan'),'5']:
            with self.subTest(score=score),self.assertRaises(ValueError):M.validate_quality(packet(score))

    def test_nested_commentary_is_normalized_without_changing_scores_or_reasons(self):
        dims={d:{'score':4.5,'reason':'原始评分理由'} for d in M.DIMS}
        packet={'dimensions':dict(dims, strengths=['原始优点'],overall_comment='原始点评')}
        M.validate_quality(packet)
        self.assertEqual(packet['dimensions'],dims)
        self.assertEqual(packet['strengths'],['原始优点'])
        self.assertEqual(packet['overall_comment'],'原始点评')
        self.assertEqual(packet['_format_normalization']['moved_fields'],['overall_comment','strengths'])

    def test_normalization_does_not_fill_missing_scores_or_accept_unknown_dimensions(self):
        packet={'dimensions':{d:{'score':5,'reason':'理由'} for d in M.DIMS}}
        for bad in [dict(packet['dimensions'],other={'score':5}),{d:v for d,v in packet['dimensions'].items() if d!='depth'}]:
            with self.assertRaises(ValueError):M.validate_quality({'dimensions':bad})
        with self.assertRaises(ValueError):M.validate_quality({'dimensions':dict(packet['dimensions'],overall_comment='嵌套'),'overall_comment':'冲突'})

    def test_reference_labels_are_paired_by_source_id_instead_of_model_order(self):
        row={'id':'c1','topic':1,'kind':'body','type':'factual','refs':[{'id':'a'},{'id':'b'}],
             'judgment':{'joint_support':'partial','ref_support':[{'id':'b','label':'partial'},{'id':'a','label':'full'}]}}
        data={'rows':[row],'sources':[],'audits':[],'coverage':[]}
        annotations={'items':{'c1':{'support':'partial'},'c1-ref-0':{'support':'full'},'c1-ref-1':{'support':'partial'}}}
        result=M.inherited_comparison(data,annotations)
        self.assertEqual((result['groups']['逐条引用关系']['n'],result['groups']['逐条引用关系']['exact']),(2,2))
        self.assertEqual(result['reference_details'][0]['source_id'],'a')

    def test_stream_recovery_requires_each_dimension_object_to_be_complete_json(self):
        dimensions={d:{'score':4.5,'reason':'完整理由'} for d in M.DIMS}
        text=json.dumps({'dimensions':dimensions},ensure_ascii=False)[:-2]+', "strengths": ["截断'
        packet=M.recover_streamed_quality(text)
        self.assertEqual(packet['dimensions'],dimensions)
        with self.assertRaises(ValueError):M.recover_streamed_quality(text[:text.index('完整理由',text.index('requirements'))+2])
        with self.assertRaises(ValueError):M.recover_streamed_quality('{"dimensions":{"depth":{"score":4.5,"reason":"未结束')

    def test_claude_message_stop_ends_consumption_before_connection_closes(self):
        base=M.evaluator();packet={'dimensions':{d:{'score':5,'reason':'完整理由'} for d in M.DIMS}}
        class Response:
            status_code=200
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def iter_lines(self):
                yield 'data: '+json.dumps({'type':'content_block_delta','delta':{'text':json.dumps(packet)}})
                yield 'data: '+json.dumps({'type':'message_stop'})
                raise AssertionError('结束标记之后不得等待连接关闭')
        class Client:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def stream(self,*args,**kwargs):return Response()
        with tempfile.TemporaryDirectory() as folder,patch.object(base.httpx,'Client',return_value=Client()),patch.dict(os.environ,{'CLAUDE_RELAY_BASE_URL':'https://unit.test','CLAUDE_RELAY_API_KEY':'fake'}):
            base.OUT=Path(folder)
            result=base.call('claude','测试提示词','结束事件测试',100,M.validate_quality)
            self.assertEqual(result['dimensions'],packet['dimensions'])
            self.assertEqual(json.loads((base.OUT/'裁判原始记录/结束事件测试.json').read_text())['completion_event'],'message_stop')


if __name__=='__main__':unittest.main()
