"""容器内部的代码执行入口。

安全边界由容器运行时（生产环境为 gVisor）提供；本文件只负责兼容
CodeWizard 的预置变量、限制输出，以及生成结构化结果。
"""

import io
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import collections
import datetime
import math
import re
import statistics


WORKSPACE = Path("/workspace")
CODE_PATH = WORKSPACE / "code.py"
RESULT_PATH = WORKSPACE / "result.json"
CHART_PATH = WORKSPACE / "chart.png"
MAX_CAPTURE_CHARS = 20_000


class LimitedBuffer(io.TextIOBase):
    def __init__(self, limit: int):
        self.limit = limit
        self.parts: list[str] = []
        self.length = 0
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        value = str(value)
        remaining = self.limit - self.length
        if remaining > 0:
            accepted = value[:remaining]
            self.parts.append(accepted)
            self.length += len(accepted)
        if len(value) > remaining:
            self.truncated = True
        return len(value)

    def getvalue(self) -> str:
        value = "".join(self.parts)
        if self.truncated:
            value += "\n...[沙箱输出已截断]"
        return value


def main() -> None:
    stdout = LimitedBuffer(MAX_CAPTURE_CHARS)
    stderr = LimitedBuffer(MAX_CAPTURE_CHARS)
    payload = {"success": False, "output": "", "error": None}

    execution_globals = {
        "__name__": "__sandbox__",
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "pandas": pd,
        "numpy": np,
        "matplotlib": matplotlib,
        "datetime": datetime,
        "math": math,
        "statistics": statistics,
        "json": json,
        "collections": collections,
        "re": re,
    }

    try:
        code = CODE_PATH.read_text(encoding="utf-8")
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "SimHei",
            "Microsoft YaHei",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.figsize"] = [12, 7]
        plt.rcParams["figure.dpi"] = 200

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(code, str(CODE_PATH), "exec"), execution_globals)

        figures = plt.get_fignums()
        if figures:
            figure = plt.figure(figures[-1])
            figure.savefig(
                CHART_PATH,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
            )

        payload = {
            "success": True,
            "output": stdout.getvalue(),
            "error": stderr.getvalue() or None,
        }
    except BaseException as exc:
        payload = {
            "success": False,
            "output": stdout.getvalue(),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
        }
    finally:
        plt.close("all")
        RESULT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
