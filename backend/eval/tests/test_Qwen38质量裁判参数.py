"""Synthetic offline requests verify model isolation, stream handling and cache safety."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/评估五篇正文与Scout.py'
def load():
    s=importlib.util.spec_from_file_location('qwen38_profile_test',SCRIPT);m=importlib.util.module_from_spec(s)
    with patch('dotenv.load_dotenv'):s.loader.exec_module(m)
    return m

class Qwen38ProfileTests(unittest.TestCase):
    def setUp(self):self.m=load()

    def test_profile_uses_exact_alias_and_supported_thinking_parameters(self):
        with patch.dict(os.environ,{},clear=True):options=self.m.quality_request_options()
        self.assertEqual(options['model'],'qwen3.8-max')
        self.assertEqual(options['temperature'],0.6)
        self.assertTrue(options['enable_thinking'])
        self.assertEqual(options['reasoning_effort'],'xhigh')
        self.assertNotIn('thinking_budget',options)
        self.assertIsNone(options['max_tokens'])
        self.assertTrue(options['response_format']['json_schema']['strict'])

    def test_model_or_sampling_change_rejects_identical_prompt_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'虚构评分.json';old=self.m.quality_request_options('qwen3-max')
            p.write_text(json.dumps({'prompt':'虚构输入','request_options':old,'result':{'value':9}}));before=p.read_bytes()
            with self.assertRaisesRegex(ValueError,'不同模型或参数'):
                self.m.read_matching_cache(p,'虚构输入',self.m.quality_request_options())
            self.assertEqual(p.read_bytes(),before)

    def test_structured_stream_preserves_thinking_and_omits_token_cap(self):
        m=self.m;sent=[];result={'dimensions':{d:{'score':5,'reason':'虚构理由'} for d in ['depth','comprehensiveness','readability','requirements']},'strengths':[],'weaknesses':[],'fact_risks':[],'overall_comment':'虚构点评'}
        class Response:
            status_code=200
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def iter_lines(self):
                for delta,finish in [({'reasoning_content':'虚构思考'},None),({'content':json.dumps(result)},'stop')]:
                    yield 'data: '+json.dumps({'model':'qwen3.8-max-test','choices':[{'delta':delta,'finish_reason':finish}]})
                yield 'data: '+json.dumps({'choices':[],'usage':{'total_tokens':30}})
                yield 'data: [DONE]'
        class Client:
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def stream(self,method,url,**kwargs):sent.append(kwargs['json']);return Response()
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'OUT',Path(tmp)),patch.object(m.httpx,'Client',Client),patch.dict(os.environ,{'DASHSCOPE_API_KEY':'synthetic-test-key','OPENAI_MODEL':'qwen3-max'}):
            reply=m.call('qwen','虚构输入','虚构评分',request_options=m.quality_request_options())
            raw=json.loads((Path(tmp)/'裁判原始记录/虚构评分.json').read_text())
            self.assertEqual(reply,result);self.assertEqual(raw['reasoning_content'],'虚构思考')
            self.assertEqual(raw['requested_model'],'qwen3.8-max');self.assertEqual(raw['finish_reason'],'stop')
            self.assertEqual(raw['usage']['total_tokens'],30)
            m.call('qwen','虚构抽取输入','虚构抽取')
            m.call('qwen3-max','虚构NLI输入','虚构NLI')
        self.assertEqual(sent[0]['model'],'qwen3.8-max');self.assertNotIn('max_tokens',sent[0])
        self.assertEqual(sent[1]['model'],'qwen3-max');self.assertEqual(sent[1]['temperature'],0.3)
        self.assertEqual(sent[2]['model'],'qwen3-max');self.assertEqual(sent[2]['temperature'],0)
        self.assertNotIn('enable_thinking',sent[1]);self.assertNotIn('response_format',sent[2])

    def test_historical_quality_profile_remains_available(self):
        # An explicit older quality profile stays available for historical reproduction.
        self.assertEqual(self.m.quality_request_options('qwen3-max'),{'model':'qwen3-max','temperature':0.3})

if __name__=='__main__':unittest.main()
