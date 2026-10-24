#!/usr/bin/env python3
"""离线验证 SDK、Scout 和工具埋点，保存逐项结果及真实 SDK 导出的观测。"""

import argparse
import importlib.util
import io
import json
import sys
import types
import unittest
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]


class Results(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.rows.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.rows.append({"test": test.id(), "status": "failed", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.rows.append({"test": test.id(), "status": "error", "detail": self._exc_info_to_string(err, test)})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.rows.append({"test": test.id(), "status": "skipped", "detail": reason})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    today = datetime.now(ZoneInfo("Asia/Shanghai"))
    out = args.output_dir
    if out is None:
        index = 1
        while True:
            out = BACKEND / "eval/data" / f"Scout输入输出埋点评测-{today:%Y%m%d}-{index:02d}"
            if not out.exists():
                break
            index += 1
    out.mkdir(parents=True, exist_ok=False)

    sys.path[:0] = [str(BACKEND / "app"), str(BACKEND)]
    # 隔离测试只跳过聚合 __init__ 中无关的聊天/数据库初始化；
    # Scout、BaseAgent、ToolRuntime、证据门禁和 SDK 均加载真实实现。
    for name in ("service", "service.deep_research_v2", "service.deep_research_v2.agents"):
        package = types.ModuleType(name)
        package.__path__ = [str(BACKEND / "app" / name.replace(".", "/"))]
        sys.modules[name] = package

    modules = []
    suite = unittest.TestSuite()
    for index, path in enumerate((
        BACKEND / "eval/tests/test_Scout埋点.py",
        BACKEND / "test/test_v2_tool_runtime.py",
        BACKEND / "eval/tests/test_hypothesis_evidence_traceability.py",
    )):
        name = f"scout_trace_checks_{index}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            modules.append(module)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        except Exception as exc:
            # unittest 负责将导入失败写入结果，不能因为异常而不生成原始文件。
            def failed_import(error=exc):
                raise error
            suite.addTest(unittest.FunctionTestCase(failed_import, description=str(path)))

    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=Results).run(suite)
    captured = [item for module in modules for item in getattr(module, "CAPTURED_OBSERVATIONS", [])]
    try:
        sdk_version = version("langfuse")
    except PackageNotFoundError:
        sdk_version = None
    payload = {
        "评测日期": today.isoformat(), "sdk_version": sdk_version, "python": sys.version,
        "run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skips": len(result.skipped), "tests": result.rows,
        "scope": "真实 SDK + 内存导出器 + 模拟 Bocha/正文/LLM；跳过聚合包初始化，不验证应用整体启动和云端连通性",
    }
    (out / "评测结果.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out / "观测记录.json").write_text(json.dumps(captured, ensure_ascii=False, indent=2))
    (out / "测试日志.txt").write_text(stream.getvalue())
    print(stream.getvalue())
    print(f"结果目录：{out}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
