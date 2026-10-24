"""兼容入口：AgentRun 实现位于 backend/eval/models.py。"""

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from eval.models import AgentRun  # noqa: E402,F401

__all__ = ["AgentRun"]

