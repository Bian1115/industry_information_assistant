"""运行摘要机制回归并保留逐项结果；不调用真实模型。"""
import io
import json
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path

backend = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(backend / 'app'), str(backend / 'test')]
modules = sys.argv[1:] or [
    'test_Claimify摘要论断抽取', 'test_摘要事实分句与独立模型', 'test_摘要合并与引用分离',
    'test_摘要预算组批', 'test_摘要先生成与引用程序定位', 'test_候选来源引用选择',
    'test_上下文消融配置', 'test_旧输入历史压缩', 'test_智能体独立历史',
]
run = backend / 'eval/data' / ('Claimify摘要机制验证-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
run.mkdir(parents=True)
items = []


class Result(unittest.TextTestResult):
    def startTest(self, test):
        self.started = time.monotonic()
        super().startTest(test)

    def capture(self, test, status, error=None):
        items.append({'test': test.id(), 'status': status, 'seconds': time.monotonic()-self.started,
                      'error': self._exc_info_to_string(error, test) if error else None})

    def addSuccess(self, test):
        self.capture(test, 'passed')
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.capture(test, 'failed', err)
        super().addFailure(test, err)

    def addError(self, test, err):
        self.capture(test, 'error', err)
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err:
            self.capture(subtest, 'failed', err)
        super().addSubTest(test, subtest, err)

    def addSkip(self, test, reason):
        self.capture(test, 'skipped')
        super().addSkip(test, reason)


stream = io.StringIO()
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=Result).run(suite)
(run / '测试日志.txt').write_text(stream.getvalue())
report = {'date': datetime.now().isoformat(), 'scope': '确定性桩机制测试；未评估真实模型语义质量',
          'modules': modules, 'tests_run': result.testsRun, 'failures': len(result.failures),
          'errors': len(result.errors), 'skipped': len(result.skipped), 'results': items}
(run / '测试结果.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k: report[k] for k in ('tests_run', 'failures', 'errors', 'skipped')}, ensure_ascii=False))
print(run)
for item in items:
    if item['status'] in {'failed', 'error'}:
        print(item['test'], item['error'][-1800:])
sys.exit(0 if result.wasSuccessful() else 1)
