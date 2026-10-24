"""评测专用的完整埋点副本；未设置输出路径时不改变在线行为。"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_active = contextvars.ContextVar("local_trace_active", default=None)
_errors = 0


def enabled():
    return bool(os.getenv("EVAL_TRACE_LOCAL_FILE"))


def write(event):
    global _errors
    path = os.getenv("EVAL_TRACE_LOCAL_FILE")
    if not path:
        return
    try:
        line = json.dumps({"recorded_at": datetime.now(timezone.utc).isoformat(), **event},
                          ensure_ascii=False, default=str) + "\n"
        with _lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(line)
    except Exception:
        _errors += 1
        logger.exception("本地完整埋点写入失败")


def error_count():
    return _errors


class LocalObservation:
    def __init__(self, name, **fields):
        parent = _active.get() or {}
        self.identity = {"observation_id": str(uuid.uuid4()),
                         "parent_id": parent.get("observation_id"), "name": name,
                         "run_id": fields.get("run_id") or parent.get("run_id"),
                         "session_id": fields.get("session_id") or parent.get("session_id")}
        self.token = _active.set(self.identity)
        write({**fields, **self.identity, "event": "start"})

    def update(self, **values):
        write({**self.identity, "event": "update", **values})

    def close(self):
        write({**self.identity, "event": "end"})
        _active.reset(self.token)
