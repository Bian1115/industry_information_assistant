"""独立虚构评分样例的双裁判接入与历史缓存保护。"""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
SCRIPT=ROOT/'backend/eval/scripts/评估五篇正文与Scout.py'


def load_evaluator():
    spec=importlib.util.spec_from_file_location('synthetic_quality_evaluator',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    # Offline tests do not read runtime credentials or invoke model services.
    with patch('dotenv.load_dotenv'):
        spec.loader.exec_module(module)
    return module


class SyntheticQualityPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator=load_evaluator()

    def test_first_stage_sends_same_examples_to_both_judges_and_freezes_version(self):
        m=self.evaluator
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);task={'id':90001,'query':'只评价这份测试文章','body':'这是待评的虚构测试正文。','paragraphs':[]}
            (out/'冻结输入.json').write_text(json.dumps([task],ensure_ascii=False))
            calls=[]
            def capture(model,prompt,key,max_tokens=12000,validator=None,**kwargs):
                calls.append({'model':model,'prompt':prompt,'key':key,'options':kwargs.get('request_options')})
                return {'dimensions':{}}
            with patch.object(m,'OUT',out),patch.object(m,'call',side_effect=capture):
                m.first_stage()
            self.assertEqual(len(calls),4)
            self.assertEqual({x['model'] for x in calls},{'qwen','claude'})
            self.assertTrue(all(x['options']['model']=='qwen3.8-max' for x in calls if x['model']=='qwen'))
            self.assertTrue(all(x['options'] is None for x in calls if x['model']=='claude'))
            self.assertEqual(len({x['prompt'] for x in calls}),2)  # only repeat marker differs
            for call in calls:
                prefix,target=call['prompt'].split('\n用户任务：',1)
                self.assertIn('<synthetic_quality_examples>',prefix)
                self.assertIn('虚构样例结束',prefix)
                self.assertTrue(target.endswith(task['body']))
                for example in m.QUALITY_EXAMPLES['examples']:
                    self.assertIn(example['report'],call['prompt'])
                    self.assertIn(example['id'],call['prompt'])
            version=json.loads((out/'质量评分提示词版本.json').read_text())
            self.assertEqual(version['example_count'],5)
            self.assertEqual(version['sha256'],hashlib.sha256((out/'质量评分提示词快照.txt').read_bytes()).hexdigest())
            self.assertEqual(json.loads((out/'第一阶段状态.json').read_text())['failed'],[])

    def test_old_quality_prompt_is_rejected_before_any_first_stage_call(self):
        m=self.evaluator
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'裁判原始记录').mkdir()
            (out/'冻结输入.json').write_text(json.dumps([{'id':90001,'query':'虚构问题','body':'虚构正文','paragraphs':[]}]))
            # Last planned request conflicts, so earlier requests must not start either.
            record=out/'裁判原始记录/题90001-质量-claude-2.json'
            record.write_text(json.dumps({'prompt':'旧版无样例提示词','result':{'score':9}}))
            before=record.read_bytes()
            with patch.object(m,'OUT',out),patch.object(m,'call') as call:
                with self.assertRaisesRegex(ValueError,'新的评估目录'):
                    m.first_stage()
                call.assert_not_called()
            self.assertEqual(record.read_bytes(),before)
            self.assertFalse((out/'质量评分提示词版本.json').exists())

    def test_identical_historical_prompt_can_reuse_cache_without_digest(self):
        m=self.evaluator
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'裁判原始记录').mkdir()
            record=out/'裁判原始记录/虚构缓存.json'
            record.write_text(json.dumps({'prompt':'相同虚构提示词','result':{'value':'原结果'}}))
            before=record.read_bytes()
            with patch.object(m,'OUT',out),patch.object(m.httpx,'Client') as client:
                result=m._call('qwen','相同虚构提示词','虚构缓存')
                client.assert_not_called()
            self.assertEqual(result,{'value':'原结果'})
            self.assertEqual(record.read_bytes(),before)

    def test_changed_prompt_never_reuses_or_overwrites_success_or_failed_cache(self):
        m=self.evaluator
        for old_result in ({'value':'旧结果'},None):
            with self.subTest(old_result=old_result),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);(out/'裁判原始记录').mkdir()
                record=out/'裁判原始记录/虚构缓存.json'
                record.write_text(json.dumps({'prompt':'旧提示词','result':old_result}))
                before=record.read_bytes()
                with patch.object(m,'OUT',out),patch.object(m.httpx,'Client') as client:
                    with self.assertRaisesRegex(ValueError,'不同提示词'):
                        m._call('qwen','新提示词加虚构样例','虚构缓存')
                    client.assert_not_called()
                self.assertEqual(record.read_bytes(),before)


if __name__=='__main__':
    unittest.main()
