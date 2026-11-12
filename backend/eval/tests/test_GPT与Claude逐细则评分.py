"""验证裁判替换后的真实请求契约、缓存边界和缺失评分处理。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from eval import 报告双裁判 as scoring
from eval.scripts.构建DeepResearch中文验证集 import select_tasks


def task():
    return {'id': 1, 'prompt': '解释研究结果', 'article': '研究发现增长10%，但样本有限。',
            'report_sha256': 'report-v1', 'generation_status': 'completed', 'online_critic': {'quality_score': 90},
            'dimension_weight': dict(zip(scoring.DIMENSIONS, (0.1, 0.2, 0.3, 0.4))),
            'criteria': [{'criterion_id': d + ':01', 'dimension': d, 'weight': 1} for d in scoring.DIMENSIONS]}


def result():
    return {'scores': [{'criterion_id': d + ':01', 'score': score,
                       'reason': '包含结果与局限', 'evidence_quote': '样本有限'}
                      for d, score in zip(scoring.DIMENSIONS, (5, 6, 7, 8))],
            'strengths': [], 'weaknesses': [], 'fact_risks': [], 'overall_comment': '说明结果与边界'}


def config():
    return {'gpt': {'model': 'gpt-6-astra', 'base_url': 'https://api.openai.com/v1',
                    'reasoning_effort': 'xhigh', 'max_output_tokens': 32000},
            'claude': {'model': 'claude-test', 'base_url': 'https://relay.invalid/v1', 'max_tokens': 12000}}


class JudgeContractTests(unittest.TestCase):
    def test_gpt_uses_responses_schema_and_no_qwen_only_parameters(self):
        payload = scoring.request_parameters('gpt', '待评报告', config())
        self.assertEqual(payload['model'], 'gpt-6-astra')
        self.assertEqual(payload['reasoning'], {'effort': 'xhigh'})
        self.assertTrue(payload['text']['format']['strict'])
        self.assertEqual(payload['text']['format']['schema'], scoring.SCHEMA)
        self.assertFalse(payload['store'])
        for field in ('temperature', 'top_p', 'seed', 'enable_thinking', 'response_format'):
            self.assertNotIn(field, payload)

    def test_completed_response_parses_and_incomplete_refusal_fail(self):
        body = {'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': json.dumps(result())}]}]}
        self.assertEqual(scoring.parse_gpt(body), result())
        body['status'] = 'incomplete'
        with self.assertRaises(ValueError):
            scoring.parse_gpt(body)
        body.update(status='completed', output=[{'type': 'message', 'content': [{'type': 'refusal'}]}])
        with self.assertRaises(ValueError):
            scoring.parse_gpt(body)

    def test_missing_duplicate_bad_scores_and_invented_quotes_rejected(self):
        cases = []
        missing = result(); missing['scores'].pop(); cases.append(missing)
        duplicate = result(); duplicate['scores'][-1] = duplicate['scores'][0]; cases.append(duplicate)
        bad_score = result(); bad_score['scores'][0]['score'] = float('nan'); cases.append(bad_score)
        invented = result(); invented['scores'][0]['evidence_quote'] = '原文不存在的引文'; cases.append(invented)
        for candidate in cases:
            with self.assertRaises((ValueError, AssertionError)):
                scoring.validate_result(task(), candidate)
        self.assertAlmostEqual(scoring.weighted(task(), result())['total'], 7)

    def test_missing_claude_repeat_does_not_produce_ensemble(self):
        packets = [{'provider': p, 'repeat': n, 'status': 'completed', 'scores': scoring.weighted(task(), result())}
                   for p, n in [('gpt', 1), ('gpt', 2), ('claude', 1)]]
        row = scoring.aggregate(task(), packets)
        self.assertIsNone(row['ensemble'])
        self.assertEqual(row['online_critic']['quality_score'], 90)
        packets.append({**packets[-1], 'repeat': 2})
        self.assertAlmostEqual(scoring.aggregate(task(), packets)['ensemble']['total'], 7)

    def test_cache_reuses_only_complete_identical_requests(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(scoring, 'fetch_judgment', return_value=result()) as fetch:
            output = Path(temporary)
            first = scoring.grade_once(task(), 'gpt', 1, '共同细则', config(), output)
            self.assertEqual(first['status'], 'completed')
            scoring.grade_once(task(), 'gpt', 1, '共同细则', config(), output)
            self.assertEqual(fetch.call_count, 1)
            changed = copy.deepcopy(config()); changed['gpt']['reasoning_effort'] = 'high'
            with self.assertRaises(ValueError):
                scoring.grade_once(task(), 'gpt', 1, '共同细则', changed, output)
            altered_task = task(); altered_task['article'] += '另一个结论'
            with self.assertRaises(ValueError):
                scoring.grade_once(altered_task, 'gpt', 1, '共同细则', config(), output)

    def test_failed_response_is_saved_and_only_judging_retried(self):
        def fail(provider, payload, cfg, packet):
            packet['raw_response'] = {'status': 'incomplete', 'output': []}
            raise ValueError('incomplete')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(scoring, 'fetch_judgment', side_effect=fail):
                failed = scoring.grade_once(task(), 'gpt', 1, '细则', config(), output)
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['raw_response']['status'], 'incomplete')
            with patch.object(scoring, 'fetch_judgment', return_value=result()):
                done = scoring.grade_once(task(), 'gpt', 1, '细则', config(), output)
            self.assertEqual(done['status'], 'completed')
            self.assertEqual(len(list(output.rglob('*尝试*.json'))), 2)

    def test_claude_stream_without_end_turn_is_rejected(self):
        events = [{'type': 'message_start', 'message': {'model': 'claude-test', 'usage': {}}},
                  {'type': 'content_block_delta', 'delta': {'text': json.dumps(result())}},
                  {'type': 'message_delta', 'delta': {'stop_reason': 'max_tokens'}},
                  {'type': 'message_stop'}]
        response = Mock(status_code=200)
        response.iter_lines.return_value = ['data: ' + json.dumps(e) for e in events]
        with patch.object(scoring.httpx, 'Client') as client, patch.dict(scoring.os.environ, {'CLAUDE_RELAY_API_KEY': 'fake'}):
            client.return_value.__enter__.return_value.stream.return_value.__enter__.return_value = response
            with self.assertRaises(ValueError):
                scoring.fetch_judgment('claude', {}, config(), {'_started': scoring.time.monotonic()})

    def test_online_scores_are_not_exposed_to_independent_judges(self):
        row = task(); row['online_critic'] = {'secret_score': '不应发送的在线审核结果'}
        self.assertNotIn('不应发送的在线审核结果', scoring.make_prompt(row, 1, '统一标准'))


class ChineseValidationSelectionTests(unittest.TestCase):
    def test_retain_eighteen_add_twelve_without_translation_or_overlap(self):
        rows = [{'id': i, 'language': 'zh' if i <= 50 else 'en', 'prompt': f'原题{i}'} for i in range(1, 101)]
        old = list(range(1, 19)) + list(range(51, 63))
        selected, retained, added, candidates = select_tasks(rows, old, 20261108)
        self.assertEqual(retained, list(range(1, 19)))
        self.assertEqual(len(added), 12)
        self.assertFalse(set(added) & set(old))
        self.assertEqual(len({r['id'] for r in selected}), 30)
        self.assertTrue(all(r['language'] == 'zh' and r in rows for r in selected))
        self.assertEqual(select_tasks(rows, old, 20261108)[0], selected)


if __name__ == '__main__':
    unittest.main()
