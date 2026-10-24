"""保存会话 TTL 及关联机制回归的逐项机器结果；--postgres 启用隔离数据库验证。"""

import argparse
import io
import json
import logging
import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND / ".env")
sys.path[:0] = [str(BACKEND / "app"), str(BACKEND / "test")]


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.rows.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.rows.append({"test": test.id(), "status": "failed", "error": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.rows.append({"test": test.id(), "status": "error", "error": self._exc_info_to_string(err, test)})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.rows.append({"test": test.id(), "status": "skipped", "reason": reason})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres", action="store_true")
    args = parser.parse_args()
    os.environ["RUN_SESSION_TTL_POSTGRES"] = "1" if args.postgres else "0"
    modules = ["test_会话24小时过期", "test_网页缓存过期清理", "test_聊天记忆UTC时间",
               "test_长期记忆入队策略", "test_session_memory", "test_memory_reliability"]
    output = BACKEND / "eval/data" / ("会话24小时过期回归-" + datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True)
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    logging.getLogger().addHandler(handler)
    start = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(
        unittest.defaultTestLoader.loadTestsFromNames(modules))
    payload = {"date": datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(), "postgresql_enabled": args.postgres,
               "postgresql_scope": "中文临时schema；不删除public业务数据", "modules": modules,
               "duration_seconds": round(time.monotonic() - start, 3), "tests_run": result.testsRun,
               "passed": sum(row["status"] == "passed" for row in result.rows),
               "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
               "cases": result.rows}
    (output / "回归结果.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "运行日志.txt").write_text(log.getvalue(), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"}, ensure_ascii=False))
    print("原始结果目录:", output)
    for row in result.rows:
        if row["status"] in {"failed", "error"}:
            print(row["test"], row["error"])
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
