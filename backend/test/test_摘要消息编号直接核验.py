import json
import types
import unittest
from unittest.mock import patch

from service.context_policy import NLISettings
from service.summary_message_verification import parse_message_claims, source_windows, verify_message_claims_sync, summary_batches


PROSE='## SESSION INTENT\n目标：分析。\n\n## SUMMARY\n研究纳入200人。\n\n## ARTIFACTS\nNone'


class Model:
    model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=512))
    def tokenizer(self, text, claim, truncation=False):
        return {'input_ids':list(range(len(text)+len(claim)+2))}
    def predict(self, pairs, settings):
        return [{'status':'scored','input_tokens':len(text)+len(claim)+2,
                 'scores':({'entailment':.01,'neutral':.01,'contradiction':.98} if '相反' in text
                           else {'entailment':.98,'neutral':.01,'contradiction':.01})} for text,claim in pairs]


class MessageCitationTests(unittest.TestCase):
    def context(self, text='研究纳入200人。',start=0):
        return {'sources':{'m':[{'start':start,'text':text}]},
                'rows':{'m':{'kind':'llm_output','payload':{'content':'前文'[:start]+text+'后文'}}}}
    def output(self, ids=None):
        return json.dumps({'claims':[{'section':'SUMMARY','claim':'研究纳入200人。','source_ids':ids or ['m']}],
                           'excluded':[{'section':'SESSION INTENT','text':'目标：分析。','reason':'目标'}]},ensure_ascii=False)
    def test_message_id_only_and_full_summary_coverage(self):
        result=parse_message_claims(self.output(),PROSE,self.context())
        self.assertEqual(result['covered_content_ratio'],1)
        self.assertEqual(result['claims'][0]['source_ids'],['m'])
    def test_rejects_model_quote_and_unknown_source(self):
        output=json.loads(self.output());output['claims'][0]['quote']='研究纳入200人。'
        with self.assertRaises(ValueError):parse_message_claims(json.dumps(output),PROSE,self.context())
        with self.assertRaises(ValueError):parse_message_claims(self.output(['unprovided']),PROSE,self.context())
    def test_rejects_rewritten_claim(self):
        output=self.output().replace('200','300')
        with self.assertRaises(ValueError):parse_message_claims(output,PROSE,self.context())
    def test_reports_missing_summary_content(self):
        output=json.loads(self.output());output['excluded']=[]
        result=parse_message_claims(json.dumps(output),PROSE,self.context())
        self.assertLess(result['covered_content_ratio'],1)
        self.assertTrue(result['uncovered'])
    def test_summary_batches_keep_exact_sections_and_sentence_boundaries(self):
        batches=list(summary_batches(PROSE,max_chars=6))
        self.assertEqual([b['text'] for b in batches],['目标：分析。','研究纳入200人。'])
        part={'SUMMARY':'研究纳入200人。'}
        output=json.loads(self.output());output['excluded']=[]
        result=parse_message_claims(json.dumps(output),PROSE,self.context(),summary_sections=part)
        self.assertEqual(result['covered_content_ratio'],1)
    def test_long_sentence_windows_are_exact_and_cover_tail(self):
        text='连续原文'*100+'结束。'
        windows=source_windows(text,'论断',Model(),NLISettings(max_length=48))
        self.assertEqual(windows[-1]['end'],len(text))
        covered=set()
        for window in windows:
            self.assertEqual(text[window['start']:window['end']],window['quote'])
            self.assertLessEqual(len(window['quote'])+len('论断')+2,48)
            covered.update(range(window['start'],window['end']))
        self.assertEqual(covered,set(range(len(text))))
    def test_program_reads_only_supplied_fragment_and_preserves_offsets(self):
        context=self.context(start=2)
        claims=parse_message_claims(self.output(),PROSE,context)['claims']
        with patch('service.nli_verifier.get_nli',return_value=Model()):
            result=verify_message_claims_sync(claims,context,NLISettings())
        ref=result['claims_with_program_quotes'][0]['references'][0]
        self.assertEqual(ref['start'],2)
        self.assertEqual(ref['quote'],'研究纳入200人。')
        self.assertEqual(result['status'],'passed')
    def test_contradicting_window_is_not_hidden_by_support_selection(self):
        text='支持。'+'无关。'*30+'相反。'
        context=self.context(text)
        claims=[{'section':'SUMMARY','claim':'研究纳入200人。','source_ids':['m']}]
        with patch('service.nli_verifier.get_nli',return_value=Model()):
            result=verify_message_claims_sync(claims,context,NLISettings(max_length=48))
        self.assertEqual(result['claims'][0]['status'],'contradiction')
        self.assertEqual(result['status'],'rejected')
        self.assertGreater(len(result['claims_with_program_quotes'][0]['references']),1)
    def test_missing_message_and_tampered_fragment_are_rejected(self):
        claims=[{'section':'SUMMARY','claim':'研究纳入200人。','source_ids':['m']}]
        context=self.context();context['rows']['m']['payload']['content']='篡改'
        with patch('service.nli_verifier.get_nli',return_value=Model()):
            with self.assertRaises(ValueError):verify_message_claims_sync(claims,context,NLISettings())
