#!/usr/bin/env python3
"""兼容入口；实现已迁移到 backend/eval/scripts。"""

import runpy
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR))

if __name__ == "__main__":
    runpy.run_module("eval.scripts.run_longmemeval_regression", run_name="__main__")
else:
    from eval.scripts.run_longmemeval_regression import *  # noqa: F401,F403

