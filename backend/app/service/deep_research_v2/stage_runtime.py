# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""节点执行的运行保护层。

``execute_stage`` 是本项目的适配接口，不是 LangGraph 自带能力。图只负责
决定"下一步执行哪个节点"，而"这一步允许跑多久、还能花多少 token、用户
是否已经取消、本执行器是否还持有运行租约"全部由这里负责。

契约：执行传入的 action，并在执行期间持续接入已有的超时、取消、预算、
trace 与租约检查；返回 Agent 结果或向外抛出异常。未完成的节点异常必须
向外传播，不能只把错误写进 ``errors`` 后返回"成功"。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from service.research_control import (
    ResearchCancelledError,
    get_current_cancellation_token,
    is_research_cancelled,
)

from .runtime_budget import max_llm_tokens, phase_budget_seconds, total_budget_seconds
from .budget_ledger import bind_runtime_budget

logger = logging.getLogger("DeepResearchStageRuntime")

# 节点结束后的取消宽限期：先让模型与工具自己收尾并把已耗预算写回状态，
# 超过这个时间才强制取消任务。
CANCEL_GRACE_SECONDS = 10.0

# 取消、超时与预算的轮询间隔。
POLL_SECONDS = 0.5


class StageBudgetExceeded(TimeoutError):
    """阶段、总时长或 token 预算耗尽。"""


class LeaseLostDuringStage(RuntimeError):
    """执行期间失去运行租约，必须停止推进并让上层处理。"""


def consumed_tokens(state: Dict[str, Any]) -> int:
    """独立账本投影包含失败与未知请求预留；兼容历史图内统计。"""
    budget = state.get("runtime_budget") or {}
    return sum(
        int(run.get("charged_tokens", run.get("llm_tokens", 0)) or 0)
        for stats in (budget.get("phases") or {}).values()
        for run in (stats.get("runs") or [])
    )


def ensure_runtime_budget(state: Dict[str, Any]) -> Dict[str, Any]:
    """初始化或修复可跨检查点恢复的运行预算账本。"""
    budget = state.get("runtime_budget")
    if not isinstance(budget, dict):
        budget = {}
    budget["total_limit_seconds"] = total_budget_seconds()
    budget["token_limit"] = max_llm_tokens()
    if not isinstance(budget.get("phases"), dict):
        budget["phases"] = {}
    # elapsed_seconds 是跨节点累计值：LangGraph 每个节点都是独立调用，
    # 不能用"本进程启动至今"当作整个研究的已用时长。
    try:
        budget["elapsed_seconds"] = float(budget.get("elapsed_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        budget["elapsed_seconds"] = 0.0
    state["runtime_budget"] = budget
    return budget


def make_execute_stage(
    *,
    run_id: str,
    session_id: str,
    check_lease: Optional[Callable[[], Awaitable[bool]]] = None,
    agents_of: Optional[Callable[[str], Any]] = None,
    budget_store: Any = None,
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """构造本次运行专用的 ``execute_stage``。

    Args:
        run_id / session_id: 取消标志与租约都按 run 维度判断。
        check_lease: 返回 False 表示本执行器已失去租约，必须停止推进。
        agents_of: 按阶段返回执行该阶段的 Agent，用于统计其 token 消耗。
    """

    async def cancelled() -> bool:
        if is_research_cancelled(run_id, session_id):
            return True
        if check_lease is not None and not await check_lease():
            raise LeaseLostDuringStage(f"run {run_id} 的执行租约已失效")
        return False

    async def execute_stage(
        action: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        work: Dict[str, Any],
        phase: str,
    ) -> Dict[str, Any]:
        # 1) 开始前先检查取消与租约，避免白跑一个昂贵节点。
        if await cancelled():
            raise ResearchCancelledError("研究已取消")

        agent = agents_of(phase) if agents_of else None
        agent_name = getattr(agent, "name", phase)
        attempt_id = None
        if budget_store is not None:
            attempt_id, budget = await budget_store.begin(work, phase, agent_name)
            work["runtime_budget"] = budget
            work["node_attempts"] = dict(budget["node_attempts"])
            work["_attempt"] = work["node_attempts"].get(work.get("_node", phase), 1)
        else:
            budget = ensure_runtime_budget(work)
        phase_limit = budget.get("phase_limits", {}).get(agent_name, phase_budget_seconds(agent_name))
        total_limit = int(budget["total_limit_seconds"])
        token_limit = int(budget["token_limit"])
        phase_stats = budget["phases"].setdefault(
            agent_name, {"limit_seconds": phase_limit, "runs": []}
        )
        phase_stats["limit_seconds"] = phase_limit

        started = time.monotonic()
        elapsed_before = float(budget["elapsed_seconds"])
        tokens_before = int(getattr(agent, "llm_total_tokens", 0) or 0)
        calls_before = int(getattr(agent, "llm_call_count", 0) or 0)
        completed_tokens = consumed_tokens(work)

        async def scoped_action():
            with bind_runtime_budget(budget_store, attempt_id):
                return await action(work)

        task = asyncio.create_task(scoped_action())
        ending = "failed"
        result = None

        async def check_budget():
            nonlocal budget
            phase_elapsed = time.monotonic() - started
            if budget_store is not None:
                budget = await budget_store.observe(attempt_id, phase_elapsed)
                work["runtime_budget"] = budget
                total_elapsed, tokens = budget["elapsed_seconds"], consumed_tokens(work)
            else:
                total_elapsed = elapsed_before + phase_elapsed
                tokens = completed_tokens + int(getattr(agent, "llm_total_tokens", 0) or 0) - tokens_before
            if phase_elapsed > phase_limit:
                raise StageBudgetExceeded(f"{agent_name} 阶段超过 {phase_limit}s 预算")
            if total_elapsed > total_limit:
                raise StageBudgetExceeded(f"研究任务超过 {total_limit}s 总预算")
            if token_limit > 0 and tokens > token_limit:
                raise StageBudgetExceeded(f"研究任务超过 {token_limit} token 预算")

        try:
            while not task.done():
                try:
                    await check_budget()
                except StageBudgetExceeded:
                    token = get_current_cancellation_token()
                    if token:
                        token.cancel()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise

                if await cancelled():
                    token = get_current_cancellation_token()
                    if token:
                        token.cancel()
                    # 先传播取消信号让模型与工具自行收尾，超过宽限期再强制取消。
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(task), timeout=CANCEL_GRACE_SECONDS
                        )
                    except (
                        asyncio.TimeoutError,
                        ResearchCancelledError,
                        asyncio.CancelledError,
                    ):
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    raise ResearchCancelledError("研究已取消")

                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=POLL_SECONDS)
                except asyncio.TimeoutError:
                    continue

            # 2) 取回结果；Agent 抛出的异常在这里原样向外传播。
            result = await task
            if not isinstance(result, dict):
                raise TypeError(f"{phase} 阶段返回了非状态对象: {type(result)!r}")
            # 即使 action 在两个轮询之间完成，也不能跳过租约、取消和预算检查。
            if await cancelled():
                raise ResearchCancelledError("研究已取消")
            await check_budget()
            ending = "completed"
            return result
        except (ResearchCancelledError, asyncio.CancelledError):
            ending = "cancelled"
            raise
        finally:
            # 3) 无论成功、失败还是取消，都记账并回收任务，不把 Agent 留在后台。
            if not task.done():
                token = get_current_cancellation_token()
                if token:
                    token.cancel()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            phase_elapsed = round(time.monotonic() - started, 3)
            calls = int(getattr(agent, "llm_call_count", 0) or 0) - calls_before
            tokens = int(getattr(agent, "llm_total_tokens", 0) or 0) - tokens_before
            if budget_store is not None:
                work["runtime_budget"] = await budget_store.finish(attempt_id, phase_elapsed, ending, tokens, calls)
                if ending == "completed":
                    result["runtime_budget"] = work["runtime_budget"]
                    result["node_attempts"] = work["node_attempts"]
            else:
                phase_stats["runs"].append({
                    "phase": phase, "elapsed_seconds": phase_elapsed, "completed": ending == "completed",
                    "llm_calls": calls, "llm_tokens": tokens,
                })
                budget["elapsed_seconds"] = round(elapsed_before + phase_elapsed, 3)

    return execute_stage
