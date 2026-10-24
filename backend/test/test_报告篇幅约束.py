"""删除追问路由后，报告交付仍执行有效篇幅约束。"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from service.deep_research_v2.report_constraints import body_character_count, check_body_length
from service.deep_research_v2.graph import finish


class ReportLengthTests(unittest.TestCase):
    def test_links_and_references_do_not_inflate_body_count(self):
        report = '# 标题\n正文 [来源](https://example.org/long-path)\n## 参考文献\n' + '文献' * 100
        self.assertEqual(body_character_count(report), len('标题正文来源'))

    def test_removed_limit_does_not_use_an_old_report_length(self):
        check_body_length({'active_constraints': {}, 'final_report': '正文' * 3000})

    def test_finish_does_not_deliver_report_over_current_limit(self):
        state = {'active_constraints': {'max_body_chars': 100}, 'final_report': '正文' * 100,
                 'phase': 'completed', 'quality_verdict': 'pass'}
        with self.assertRaisesRegex(ValueError, '100字'):
            finish({'research': state})


if __name__ == '__main__':
    unittest.main()
