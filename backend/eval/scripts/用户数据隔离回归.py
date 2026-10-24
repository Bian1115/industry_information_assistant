"""离线运行双用户 HTTP 隔离及相关回归，保存逐测试机器可读结果。"""

import contextlib
import json
import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backend/app'))


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.current = None

    def startTest(self, test):
        super().startTest(test)
        self.current = {'id': test.id(), 'status': 'running', 'subtests': []}
        self.started = time.monotonic()
        self.records.append(self.current)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.current['status'] = 'passed'

    def addError(self, test, err):
        super().addError(test, err)
        self.current.update(status='error', detail=self._exc_info_to_string(err, test))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.current.update(status='failed', detail=self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.current.update(status='skipped', reason=reason)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        entry = {'id': str(subtest), 'status': 'passed' if err is None else 'failed'}
        if err is not None:
            entry['detail'] = self._exc_info_to_string(err, test)
            self.current['status'] = 'failed'
        self.current['subtests'].append(entry)

    def stopTest(self, test):
        self.current['duration_seconds'] = round(time.monotonic() - self.started, 6)
        super().stopTest(test)


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'backend/eval/data' / ('用户隔离回归-' + datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d-%H%M%S'))
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory / '后端最终测试结果.json'
    log_file = directory / '后端最终测试日志.txt'
    if result_file.exists() or log_file.exists():
        raise SystemExit('结果文件已存在，请使用新的运行目录，避免覆盖历史结果。')
    from dotenv import load_dotenv
    load_dotenv(ROOT / 'backend/.env')
    # 本运行明确为离线回归，不连接业务 PostgreSQL 或使用外部 AI/索引服务。
    os.environ['RUN_DB_INTEGRATION_TESTS'] = '0'
    patterns = ['test_用户数据完整隔离.py', 'test_会话24小时过期.py', 'test_研究记录读取.py',
                'test_session_memory.py', 'test_memory_service.py', 'test_memory_reliability.py']
    with log_file.open('x', encoding='utf-8') as log, contextlib.redirect_stderr(log), contextlib.redirect_stdout(log):
        suite = unittest.TestSuite()
        for pattern in patterns:
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'backend/test'), pattern=pattern))
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordingResult).run(suite)
    raw = {'date': '2026-10-09', 'recorded_at': datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
           'scope': '离线；真实 JWT、HTTP 路由与 SQLite ORM；外部 AI/DocMind/索引服务替身；PostgreSQL 集成测试跳过',
           'python': sys.version, 'patterns': patterns, 'tests_run': result.testsRun,
           'passed': sum(row['status'] == 'passed' for row in result.records),
           'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
           'subtests': sum(len(row['subtests']) for row in result.records),
           'successful': result.wasSuccessful(), 'records': result.records}
    result_file.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: raw[key] for key in ['tests_run', 'passed', 'failures', 'errors', 'skipped', 'successful']}, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
