"""以SQLite及注入API验证证据块直读，结果写入指定的新目录。"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "backend/app"), str(ROOT / "backend/eval/tests"), str(ROOT / "backend")]
# 本地机制回归无需外部观测服务，也不启用真实数据库/向量集成测试。
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["EVAL_TRACE_ENABLED"] = "false"

TESTS = [
    "test_会话RAG证据块直读",
    "test_会话临时双层RAG.TemporaryRAGTests",
    "test_跨存储对账.IndexReconciliation",
    "test_跨存储对账.VectorInventory",
    "test_取消写作前事实提炼",
    "test_Writer章节检查点",
    "test_Writer论断引用结构",
    "test_来源关联与派生追溯",
    "test_会话双层RAG分页精排",
    "test_限定网站与终态交付",
]


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--输出目录", required=True, type=Path)
    parser.add_argument("--日期", required=True)
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (args.输出目录 / "测试日志.txt").open("w") as log, redirect_stderr(log), redirect_stdout(log):
        suite = unittest.defaultTestLoader.loadTestsFromNames(TESTS)
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
    data = {"日期": args.日期, "时区": "Asia/Singapore", "测试入口": TESTS,
            "口径": "SQLite内存库、注入检索API；未请求真实模型、业务数据库或真实Milvus",
            "testsRun": result.testsRun, "passed": len(result.passed),
            "failures": [{"test": test.id(), "traceback": trace} for test, trace in result.failures],
            "errors": [{"test": test.id(), "traceback": trace} for test, trace in result.errors],
            "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
            "passed_tests": result.passed, "duration_seconds": time.monotonic() - started}
    (args.输出目录 / "测试结果.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in data.items()
                      if key not in {"passed_tests", "测试入口"}}, ensure_ascii=False))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
