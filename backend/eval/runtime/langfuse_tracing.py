"""Langfuse v4 旁路埋点：完整业务载荷、运行关联和 Scout 状态变化。"""

from __future__ import annotations

import atexit
import contextvars
import copy
import hashlib
import logging
import os
import threading
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import version
from . import local_trace

logger = logging.getLogger(__name__)
TESTED_SDK_VERSION = "4.15.1"
_client = None
_checked = False
_lock = threading.Lock()
_active_run = contextvars.ContextVar("langfuse_active_run", default=None)


def tracing_enabled():
    def enabled(name, default):
        return os.getenv(name, default).strip().lower() in {"true", "1", "yes", "on"}
    return enabled("EVAL_TRACE_ENABLED", "true") and enabled("LANGFUSE_ENABLED", "false")


def validate_sdk(client, sdk_version):
    """禁止旧版接口静默失效；部署依赖固定在实测版本。"""
    if sdk_version.split(".")[0] != "4":
        raise RuntimeError(f"需要 Langfuse Python SDK v4，当前为 {sdk_version}")
    for method in ("start_as_current_observation", "flush"):
        if not callable(getattr(client, method, None)):
            raise RuntimeError(f"Langfuse SDK 缺少接口 {method}")


def get_client():
    global _client, _checked
    if not tracing_enabled():
        return None
    if _checked:
        return _client
    with _lock:
        if _checked:
            return _client
        _checked = True
        try:
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            if not public_key or not secret_key:
                raise RuntimeError("缺少 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY")
            from langfuse import Langfuse
            sdk_version = version("langfuse")
            validate_sdk(Langfuse, sdk_version)
            _client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=(os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
                          or "https://cloud.langfuse.com").rstrip("/"),
            )
            logger.info("Langfuse 埋点已初始化，SDK=%s", sdk_version)
        except Exception as exc:
            logger.warning("Langfuse 埋点未启用：%s；安装 requirements-Langfuse埋点.txt 后重启", exc)
    return _client


class SafeObservation:
    def __init__(self, span=None, local=None):
        self.span = span
        self.local = local

    def update(self, **values):
        if self.local is not None:
            self.local.update(**values)
        if self.span is not None:
            try:
                # SDK 异步发送前冻结载荷，后续 state 原地修改不能污染历史记录。
                self.span.update(**copy.deepcopy(values))
            except Exception as exc:
                logger.warning("Langfuse 更新失败，研究继续：%s", type(exc).__name__)


@contextmanager
def observation(name, *, as_type="span", input=None, session_id=None, run_id=None,
                metadata=None, **attributes):
    """SDK 错误隔离于业务；业务异常及取消仍原样向外传播。"""
    manager = propagation = token = None
    local = local_trace.LocalObservation(name, as_type=as_type, input=input,
        session_id=session_id, run_id=run_id, metadata=metadata, **attributes) if local_trace.enabled() else None
    handle = SafeObservation(local=local)
    try:
        client = get_client()
        if client is not None:
            from langfuse import propagate_attributes
            active = _active_run.get()
            identity = run_id or active or session_id
            kwargs = {}
            if identity and identity != active:
                kwargs["trace_context"] = {
                    "trace_id": hashlib.sha256(str(identity).encode()).hexdigest()[:32]
                }
            propagation = propagate_attributes(session_id=session_id)
            propagation.__enter__()
            manager = client.start_as_current_observation(
                name=name, as_type=as_type, input=copy.deepcopy(input),
                metadata={**(metadata or {}), "run_id": identity},
                **kwargs, **attributes,
            )
            handle = SafeObservation(manager.__enter__(), local=local)
            token = _active_run.set(identity)
    except Exception as exc:
        logger.warning("Langfuse 创建记录失败，研究继续：%s", type(exc).__name__)
    try:
        yield handle
    except BaseException as exc:
        handle.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if local is not None:
            local.close()
        if token is not None:
            _active_run.reset(token)
        for context in (manager, propagation):
            if context is not None:
                try:
                    context.__exit__(None, None, None)
                except Exception as exc:
                    logger.warning("Langfuse 结束记录失败，研究继续：%s", type(exc).__name__)


def record_generation(record, system_prompt, user_prompt, output, *, run_id=None):
    with observation(
        record.get("agent_name") or "LLM", as_type="generation",
        session_id=record.get("session_id"), run_id=run_id,
        input=[{"role": "system", "content": system_prompt},
               {"role": "user", "content": user_prompt}],
        model=record.get("model"),
        usage_details={key: record[field] for key, field in (
            ("input", "prompt_tokens"), ("output", "completion_tokens"),
            ("total", "total_tokens")) if record.get(field) is not None},
        metadata={key: record.get(key) for key in (
            "phase", "iteration", "parse_ok", "parse_strategy", "duration_ms")},
    ) as span:
        span.update(output=output, level="DEFAULT" if record.get("success") else "ERROR")


_STATE_FIELDS = ("facts", "hypothesis_evidence", "raw_sources", "hypotheses",
                 "data_points", "search_budget", "evidence_coverage")


def _snapshot(state):
    return copy.deepcopy({key: state.get(key, {} if key in {
        "search_budget", "evidence_coverage"} else []) for key in _STATE_FIELDS})


def _changes(before, after):
    changes = {}
    for field, key in (("facts", "id"), ("hypothesis_evidence", "id"),
                       ("raw_sources", "id")):
        old = {str(item.get(key)): item for item in before[field]}
        new = {str(item.get(key)): item for item in after[field]}
        changes[field] = {
            "added": [item for ident, item in new.items() if ident not in old],
            "updated": [item for ident, item in new.items() if ident in old and item != old[ident]],
            "removed_ids": [ident for ident in old if ident not in new],
        }
    old, new = before["search_budget"], after["search_budget"]
    changes["search_budget"] = {"before": old, "after": new}
    for key in ("used", "remaining", "skipped"):
        changes["search_budget"][f"{key}_delta"] = int(new.get(key, 0)) - int(old.get(key, 0))
    return changes


def trace_scout_process(function):
    @wraps(function)
    async def wrapped(self, state, *args, **kwargs):
        if get_client() is None and not local_trace.enabled():
            return await function(self, state, *args, **kwargs)
        try:
            before = _snapshot(state)
            inputs = copy.deepcopy({key: state.get(key) for key in (
                "query", "outline", "hypotheses", "search_web", "search_local",
                "phase", "iteration", "pending_research_tasks")})
            inputs["state_before"] = before
        except Exception as exc:
            logger.warning("Scout 状态快照失败，研究继续：%s", type(exc).__name__)
            return await function(self, state, *args, **kwargs)
        with observation(
            "DeepScout.process", as_type="agent", input=inputs,
            session_id=state.get("session_id"), run_id=state.get("_run_id"),
            metadata={"phase": state.get("phase"), "iteration": state.get("iteration")},
        ) as span:
            result, completed = state, False
            try:
                result = await function(self, state, *args, **kwargs)
                completed = True
                return result
            finally:
                try:
                    after = _snapshot(result)
                    span.update(output={
                        "status": "completed" if completed else "interrupted",
                        "state_after": after,
                        "changes": _changes(before, after),
                        "phase_after": result.get("phase"),
                        "validation_boundary": "process_returned" if completed else "partial_state",
                    })
                except Exception as exc:
                    logger.warning("Scout 状态导出失败，研究继续：%s", type(exc).__name__)
    return wrapped


def trace_scout_tool(name):
    def decorate(function):
        @wraps(function)
        async def wrapped(self, args, runtime_context):
            if get_client() is None and not local_trace.enabled():
                return await function(self, args, runtime_context)
            with observation(
                name, as_type="tool", input=args.model_dump(),
                session_id=runtime_context.session_id, run_id=runtime_context.run_id,
                metadata=runtime_context.metadata,
            ) as span:
                result = await function(self, args, runtime_context)
                span.update(output=result)
                return result
        return wrapped
    return decorate


async def trace_bocha_request(client, url, *, headers, json, attempt):
    """每次 HTTP 尝试保留完整请求体和响应体；认证请求头不进入载荷。"""
    if get_client() is None and not local_trace.enabled():
        return await client.post(url, headers=headers, json=json)
    with observation("Bocha.http", as_type="tool", input={
        "method": "POST", "url": url, "body": json, "attempt": attempt,
    }) as span:
        response = await client.post(url, headers=headers, json=json)
        try:
            body = response.json()
        except ValueError:
            body = response.text
        failed = response.status_code >= 400 or (
            isinstance(body, dict) and str(body.get("code", "200")) != "200"
        )
        span.update(output={"http_status": response.status_code, "body": body},
                    level="ERROR" if failed else "DEFAULT")
        return response


def flush_langfuse():
    """只冲刷已创建的客户端；不因退出进程而启用新连接。"""
    if _client is not None:
        try:
            _client.flush()
        except Exception as exc:
            logger.warning("Langfuse 冲刷失败：%s", type(exc).__name__)


atexit.register(flush_langfuse)
