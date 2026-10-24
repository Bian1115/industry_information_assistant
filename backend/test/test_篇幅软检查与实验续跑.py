"""超字数仍交付并测量；续跑只接受同一原话、会话和失败前缀。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND), str(BACKEND / 'app'), str(BACKEND / 'test')]
from service.deep_research_v2.report_constraints import check_body_length
from service.deep_research_v2.graph import finish
from eval.scripts.随机题多轮会话实验 import load_resume
from test_正文检查异常分类与全文预算 import writer, writing_state


class AdvisoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_report_is_preserved_without_extra_repair_calls(self):
        state = writing_state(100)
        state['final_report'] = '正文' * 100
        state['draft_sections'] = {'sec_1': state['final_report']}
        before = copy.deepcopy(state['draft_sections'])
        value = writer(); value.call_llm = AsyncMock()
        with patch.dict(os.environ, {'REPORT_BODY_LENGTH_HARD_CHECK_ENABLED': 'false'}):
            await value._fit_report_to_budget(state, '摘要', [], '')
        value.call_llm.assert_not_awaited()
        self.assertEqual(state['draft_sections'], before)
        self.assertEqual(state['final_report'], '正文' * 100)
        self.assertEqual(state['body_length_check']['excess_chars'], 100)
        self.assertEqual(state['body_length_repair']['status'], 'advisory_exceeded')

    def test_finish_records_excess_but_delivers(self):
        state = {'active_constraints': {'max_body_chars': 100}, 'final_report': '正文' * 100,
                 'phase': 'completed', 'quality_verdict': 'pass'}
        with patch.dict(os.environ, {'REPORT_BODY_LENGTH_HARD_CHECK_ENABLED': 'false'}):
            result = finish({'research': state})['research']
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['body_length_check']['exceeded'])
        self.assertEqual(result['final_report'], state['final_report'])

    def test_explicit_hard_mode_still_rejects(self):
        with patch.dict(os.environ, {'REPORT_BODY_LENGTH_HARD_CHECK_ENABLED': 'true'}):
            with self.assertRaises(ValueError):
                check_body_length({'active_constraints': {'max_body_chars': 100}, 'final_report': '正文' * 100})


class ResumeManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.directory = Path(self.temp.name)
        self.dialogue = {'turns': [{'turn_id': 1, 'user_query': '原题'}, {'turn_id': 2, 'user_query': '追问'}]}
        self.dataset = self.directory / '追问.json'; self.dataset.write_text(json.dumps(self.dialogue))
        self.manifest = {'status': 'stopped_at_first_failure', 'session_id': 'same-session', 'evaluation_user_id': 'same-owner',
                         'dataset_sha256': hashlib.sha256(self.dataset.read_bytes()).hexdigest()}
        self.rows = [{'turn_id': 1, 'query': '原题', 'completed': False, 'session_id': 'same-session', 'user_id': 'same-owner'}]

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        (self.directory / '运行清单.json').write_text(json.dumps(self.manifest))
        (self.directory / '逐轮结果.json').write_text(json.dumps(self.rows))
        return load_resume(self.directory, self.dialogue, self.dataset)

    def test_preserves_failed_turn_and_session(self):
        value = self.load()
        self.assertEqual(value['failed_row'], self.rows[0])
        self.assertEqual(value['manifest']['session_id'], 'same-session')

    def test_rejects_changed_dataset_and_wrong_owner(self):
        self.dataset.write_text('改动')
        with self.assertRaisesRegex(ValueError, '哈希'): self.load()
        self.manifest['dataset_sha256'] = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        self.rows[0]['user_id'] = 'other-owner'
        with self.assertRaisesRegex(ValueError, '归属'): self.load()

    def test_rejects_live_run_and_unfinished_prefix(self):
        self.manifest['status'] = 'running'
        with self.assertRaisesRegex(ValueError, '终止'): self.load()
        self.manifest['status'] = 'failed'
        self.rows.append({**self.rows[0], 'turn_id': 2, 'query': '追问'})
        with self.assertRaisesRegex(ValueError, '前缀'): self.load()


if __name__ == '__main__':
    unittest.main()
