"""兼容入口：实现位于 backend/eval/runtime/profile.py。"""

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[3]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from eval.runtime.profile import *  # noqa: E402,F401,F403
from eval.runtime.profile import _config_path  # noqa: E402,F401

