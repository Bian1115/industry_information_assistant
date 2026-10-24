# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
LLM 调用追踪器

设计原则（按重要性排序）：

1. **绝不影响主流程**。埋点里任何异常都被吞掉，只打日志。
   评估是旁路功能，不能因为追踪写库失败导致用户的研究请求挂掉。

2. **绝不拖慢主流程**。DB 写入交给后台守护线程，
   调用方 put 完立即返回。队列满了就丢弃最旧的记录，
   宁可丢追踪数据也不阻塞 LLM 调用。

3. **上下文靠 contextvars 传递**。BaseAgent.call_llm 拿不到 ResearchState，
   如果要透传 session_id 就得改所有 Agent 的方法签名。
   改用 contextvars 在研究入口设置一次，
   asyncio.to_thread 会自动复制上下文，线程内也能读到。

环境变量：
    EVAL_TRACE_ENABLED         是否开启追踪，默认 true
    EVAL_TRACE_PAYLOAD_CHARS   载荷快照截断长度，默认 2000，设 0 则不存载荷
    EVAL_TRACE_QUEUE_SIZE      内存队列上限，默认 1000
    LANGFUSE_ENABLED           是否额外导出到 Langfuse，默认 false
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL（兼容 LANGFUSE_HOST）
"""

import os
import queue
import atexit
import logging
import threading
import contextvars
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


TRACE_ENABLED = _env_bool("EVAL_TRACE_ENABLED", True)
PAYLOAD_CHARS = _env_int("EVAL_TRACE_PAYLOAD_CHARS", 2000)
QUEUE_SIZE = _env_int("EVAL_TRACE_QUEUE_SIZE", 1000)


def is_trace_enabled() -> bool:
    return TRACE_ENABLED


# ---------------------------------------------------------------------------
# 追踪上下文
# ---------------------------------------------------------------------------

# 存的是对 ResearchState 的引用而不是快照，
# 这样 phase/iteration 能读到调用发生时的实时值。
# _run_simplified 全程原地修改同一个 state dict，不会重新绑定，
# 所以引用不会失效。
_ctx: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "eval_trace_ctx", default=None
)

# 暂存"已完成 LLM 调用但还没解析 JSON"的记录。
#
# 为什么必须用 contextvars 而不是 self._pending：
# Agent 实例是常驻的（graph.__init__ 里创建一次），
# scout 又存在并发调用同一个实例的情况。挂在实例上会互相覆盖，
# 而 contextvars 每个 asyncio.Task 一份拷贝，天然隔离。
_pending: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "eval_trace_pending", default=None
)


class trace_context:
    """
    研究会话级的追踪上下文。

    用法：
        with trace_context(session_id="xxx", user_id="yyy", state=state):
            ...  # 期间所有 call_llm 都会带上这些信息
    """

    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
    ):
        self._payload = {
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "state": state,
        }
        self._token = None

    def __enter__(self):
        self._token = _ctx.set(self._payload)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            try:
                _ctx.reset(self._token)
            except ValueError:
                # 跨任务 reset 会抛 ValueError，忽略即可
                pass
        return False


def _current_context() -> Dict[str, Any]:
    """读取当前上下文，任何异常都退化成空上下文。"""
    try:
        ctx = _ctx.get()
        if not ctx:
            return {}

        state = ctx.get("state") or {}
        phase = None
        iteration = 0
        if isinstance(state, dict):
            phase = state.get("phase")
            iteration = state.get("iteration", 0)

        return {
            "session_id": ctx.get("session_id"),
            "run_id": state.get("_run_id") if isinstance(state, dict) else None,
            "user_id": ctx.get("user_id"),
            "query": ctx.get("query") or (state.get("query") if isinstance(state, dict) else None),
            "phase": phase,
            "iteration": iteration,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 后台写入线程
# ---------------------------------------------------------------------------

_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=QUEUE_SIZE)
_worker: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_dropped = 0


def _write_one(record: Dict[str, Any]) -> None:
    """把一条记录写进 agent_runs 表。"""
    from core.database import SessionLocal
    from eval.models import AgentRun

    db = SessionLocal()
    try:
        db.add(AgentRun(**record))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"[EvalTracer] 写入 agent_runs 失败: {e}")
    finally:
        db.close()


def _worker_loop() -> None:
    while True:
        try:
            record = _queue.get()
        except Exception:
            continue

        if record is None:  # 关闭信号
            break

        try:
            _write_one(record)
        except Exception as e:
            logger.warning(f"[EvalTracer] 后台写入异常: {e}")
        finally:
            try:
                _queue.task_done()
            except Exception:
                pass


def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    with _worker_lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(
            target=_worker_loop,
            name="eval-tracer",
            daemon=True,
        )
        _worker.start()
        logger.info("[EvalTracer] 后台写入线程已启动")


def shutdown_tracer(timeout: float = 5.0) -> None:
    """优雅关闭：把队列里剩下的记录冲刷完。"""
    global _worker
    if _worker is None or not _worker.is_alive():
        return
    try:
        _queue.put_nowait(None)
        _worker.join(timeout=timeout)
    except Exception as e:
        logger.warning(f"[EvalTracer] 关闭失败: {e}")


atexit.register(shutdown_tracer)


# ---------------------------------------------------------------------------
# Langfuse 可选导出（默认关闭）
# ---------------------------------------------------------------------------

from .langfuse_tracing import (  # noqa: E402
    observation,
    record_generation,
    trace_bocha_request,
    trace_scout_process,
    trace_scout_tool,
)


def _export_langfuse(record, system_prompt, user_prompt, output):
    record_generation(
        record, system_prompt, user_prompt, output,
        run_id=_current_context().get("run_id"),
    )


# ---------------------------------------------------------------------------
# 对外埋点接口
# ---------------------------------------------------------------------------

def _truncate(text: Optional[str]) -> Optional[str]:
    if not text or PAYLOAD_CHARS <= 0:
        return None
    if len(text) <= PAYLOAD_CHARS:
        return text
    return text[:PAYLOAD_CHARS] + f"\n...[已截断，原长度 {len(text)}]"


def record_llm_call(
    agent_name: str,
    model: str,
    system_prompt: str = "",
    user_prompt: str = "",
    output: str = "",
    temperature: Optional[float] = None,
    json_mode: Optional[bool] = None,
    duration_ms: int = 0,
    usage: Any = None,
    success: bool = True,
    error: Optional[BaseException] = None,
    parse_ok: Optional[bool] = None,
    parse_strategy: Optional[str] = None,
    defer_parse: bool = False,
) -> None:
    """
    记录一次 LLM 调用。此函数保证不抛异常、不阻塞。

    Args:
        usage: OpenAI 响应里的 usage 对象（有 prompt_tokens 等属性）
        parse_ok / parse_strategy: JSON 解析结果，
            由 parse_json_response 回填，见 base.py
        defer_parse: True 表示这次调用预期后续会解析 JSON，
            先暂存不写库，等 attach_parse_result 补齐解析结果再写，
            这样一次调用只落一行、且带解析质量信息。
    """
    if not TRACE_ENABLED:
        return

    try:
        ctx = _current_context()

        prompt_tokens = completion_tokens = total_tokens = None
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

        user_id = ctx.get("user_id")
        # user_id 存的是字符串，写库前交给 SQLAlchemy 转 UUID；
        # 格式非法时直接置空，不能让脏数据把整条追踪记录搞挂
        if user_id is not None:
            try:
                import uuid as _uuid
                user_id = _uuid.UUID(str(user_id))
            except (ValueError, AttributeError, TypeError):
                user_id = None

        record = {
            "session_id": ctx.get("session_id") or "unknown",
            "user_id": user_id,
            "query": ctx.get("query"),
            "agent_name": agent_name,
            "model": model,
            "phase": ctx.get("phase"),
            "iteration": ctx.get("iteration", 0),
            "temperature": temperature,
            "json_mode": json_mode,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "success": success,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error)[:2000] if error is not None else None,
            "parse_ok": parse_ok,
            "parse_strategy": parse_strategy,
            "input_preview": _truncate(f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}"),
            "output_preview": _truncate(output),
            "created_at": datetime.utcnow(),
        }

        if defer_parse:
            # 先挂起，等 attach_parse_result 补齐解析结果
            _pending.set({"record": record, "payload": (system_prompt, user_prompt, output)})
            return

        _enqueue(record)
        _export_langfuse(record, system_prompt, user_prompt, output)

    except Exception as e:
        # 埋点自身出错绝不能冒泡到业务流程
        logger.debug(f"[EvalTracer] 记录失败（忽略）: {e}")


def attach_parse_result(parse_ok: bool, parse_strategy: str) -> None:
    """
    给暂存的 LLM 调用记录补上 JSON 解析结果并落库。

    没有暂存记录时静默返回（例如 Agent 直接解析了一段非 LLM 产出的文本）。
    """
    if not TRACE_ENABLED:
        return
    try:
        pending = _pending.get()
        if not pending:
            return
        _pending.set(None)

        record = pending["record"]
        record["parse_ok"] = parse_ok
        record["parse_strategy"] = parse_strategy
        _enqueue(record)

        system_prompt, user_prompt, output = pending["payload"]
        _export_langfuse(record, system_prompt, user_prompt, output)
    except Exception as e:
        logger.debug(f"[EvalTracer] 补写解析结果失败（忽略）: {e}")


def flush_pending() -> None:
    """
    冲刷未被解析的暂存记录（parse_ok 留空）。

    覆盖"调用了 LLM 但没走 parse_json_response"的情况，
    避免记录一直挂在上下文里丢失。
    """
    if not TRACE_ENABLED:
        return
    try:
        pending = _pending.get()
        if not pending:
            return
        _pending.set(None)
        _enqueue(pending["record"])
        system_prompt, user_prompt, output = pending["payload"]
        _export_langfuse(pending["record"], system_prompt, user_prompt, output)
    except Exception as e:
        logger.debug(f"[EvalTracer] 冲刷暂存记录失败（忽略）: {e}")


def _enqueue(record: Dict[str, Any]) -> None:
    """投递到后台写入队列。队列满时丢最旧的，绝不阻塞调用方。"""
    try:
        _ensure_worker()
        try:
            _queue.put_nowait(record)
        except queue.Full:
            # 队列满说明写库跟不上产生速度。丢最旧的，保最新的。
            global _dropped
            _dropped += 1
            try:
                _queue.get_nowait()
                _queue.put_nowait(record)
            except Exception:
                pass
            if _dropped % 50 == 1:
                logger.warning(f"[EvalTracer] 队列积压，已丢弃 {_dropped} 条追踪记录")
    except Exception as e:
        logger.debug(f"[EvalTracer] 投递失败（忽略）: {e}")
