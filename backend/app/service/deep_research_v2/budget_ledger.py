"""运行消耗独立于图状态提交；请求先预留，未知usage不能当作零消耗。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import uuid

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .checkpointer import lease_transaction, shared_connection
from .runtime_budget import DEFAULT_PHASE_SECONDS, max_llm_tokens, phase_budget_seconds, total_budget_seconds

_scope = ContextVar("research_runtime_budget", default=None)


@contextmanager
def bind_runtime_budget(store, attempt_id):
    token = _scope.set((store, attempt_id) if store is not None else None)
    try:
        yield
    finally:
        _scope.reset(token)


def current_runtime_budget():
    return _scope.get()


def _exceeded(message):
    from .stage_runtime import StageBudgetExceeded
    raise StageBudgetExceeded(message)


async def _snapshot(cur, run_id):
    await cur.execute("SELECT policy, baseline FROM research_budgets WHERE run_id = %s", (run_id,))
    header = await cur.fetchone()
    if header is None:
        return None
    await cur.execute(
        "SELECT a.*, clock_timestamp() AT TIME ZONE 'UTC' AS now, "
        "count(c.call_id) AS call_count, "
        "coalesce(sum(c.actual_tokens), 0) AS known_tokens, "
        "coalesce(sum(coalesce(c.actual_tokens, c.reserved_tokens)), 0) AS charged_tokens, "
        "count(c.call_id) FILTER (WHERE c.actual_tokens IS NULL) AS unknown_calls "
        "FROM research_budget_attempts a LEFT JOIN research_budget_calls c USING (attempt_id) "
        "WHERE a.run_id = %s GROUP BY a.attempt_id ORDER BY a.started_at, a.attempt_id", (run_id,),
    )
    attempts = await cur.fetchall()
    baseline, policy = header["baseline"], header["policy"]
    budget = deepcopy(baseline["runtime_budget"])
    budget.update(ledger_version=1, **policy)
    budget.setdefault("phases", {})
    elapsed = float(budget.get("elapsed_seconds", 0) or 0)
    counts = dict(baseline.get("node_attempts") or {})
    for row in attempts:
        duration = row["elapsed_seconds"]
        if row["status"] == "active":
            # 崩溃精确时刻不可知：以旧租约截止时间为上界，不把整个离线期间算入。
            duration = max(duration, (min(row["now"], row["lease_until"]) - row["started_at"]).total_seconds())
        elapsed += max(0, duration)
        calls = row["call_count"]
        stats = budget["phases"].setdefault(row["agent"], {"runs": []})
        stats["limit_seconds"] = policy["phase_limits"].get(row["agent"], 300)
        stats.setdefault("runs", []).append({
            "attempt_id": row["attempt_id"], "phase": row["phase"], "node": row["node"],
            "elapsed_seconds": round(max(0, duration), 3), "status": row["status"],
            "completed": row["status"] == "completed", "fencing_token": row["fencing_token"],
            "llm_calls": calls if calls else row["fallback_calls"],
            "llm_tokens": row["known_tokens"] if calls else row["fallback_tokens"],
            "charged_tokens": row["charged_tokens"] if calls else row["fallback_tokens"],
            "unknown_usage_calls": row["unknown_calls"],
            "elapsed_basis": "lease_upper_bound" if row["status"] in {"active", "abandoned"} else "measured",
        })
        counts[row["node"]] = counts.get(row["node"], 0) + 1
    budget["elapsed_seconds"] = round(elapsed, 3)
    budget["node_attempts"] = counts
    budget["known_tokens"] = sum(int(r.get("llm_tokens", 0) or 0)
        for s in budget["phases"].values() for r in s.get("runs", []))
    budget["charged_tokens"] = sum(int(r.get("charged_tokens", r.get("llm_tokens", 0)) or 0)
        for s in budget["phases"].values() for r in s.get("runs", []))
    budget["unknown_usage_calls"] = sum(int(r.get("unknown_usage_calls", 0) or 0)
        for s in budget["phases"].values() for r in s.get("runs", []))
    return budget


async def read_runtime_budget(conn, run_id):
    """只读投影；上层须先校验运行归属。"""
    async with shared_connection(conn) as connection:
        async with connection.transaction(), connection.cursor(row_factory=dict_row) as cur:
            # 同一运行所有账本写入也持有该锁，投影不会混合不同事务的结果。
            await cur.execute("SELECT run_id FROM research_runs WHERE run_id = %s FOR SHARE", (run_id,))
            return await _snapshot(cur, run_id)


class PostgresRuntimeBudget:
    def __init__(self, conn, lease):
        self.conn, self.lease = conn, lease

    @staticmethod
    def _check(budget, extra_tokens=0):
        if budget["elapsed_seconds"] >= budget["total_limit_seconds"]:
            _exceeded("研究任务累计时长预算已耗尽（含失败尝试）")
        if budget["token_limit"] > 0 and budget["charged_tokens"] + extra_tokens > budget["token_limit"]:
            _exceeded("研究任务累计token预算不足（含失败及未知usage预留）")

    async def begin(self, work, phase, agent):
        attempt_id = str(uuid.uuid4())
        async with lease_transaction(self.conn, self.lease) as cur:
            await cur.execute("SELECT run_id FROM research_budgets WHERE run_id = %s", (self.lease.run_id,))
            if await cur.fetchone() is None:
                saved = deepcopy(work.get("runtime_budget") or {})
                if saved.get("ledger_version"):
                    raise RuntimeError("检查点已使用独立预算但账本缺失，拒绝重置消耗")
                # 导入旧图里已知的预算一次；旧失败请求未知消耗不能凭空重建。
                saved.setdefault("phases", {})
                saved.setdefault("elapsed_seconds", 0)
                counts = dict(work.get("node_attempts") or {})
                node = work.get("_node", phase)
                counts[node] = max(0, counts.get(node, 1) - 1)
                policy = {"total_limit_seconds": total_budget_seconds(), "token_limit": max_llm_tokens(),
                          "phase_limits": {name: phase_budget_seconds(name) for name in DEFAULT_PHASE_SECONDS}}
                await cur.execute(
                    "INSERT INTO research_budgets(run_id, policy, baseline, created_at) "
                    "VALUES (%s, %s, %s, clock_timestamp() AT TIME ZONE 'UTC')",
                    (self.lease.run_id, Jsonb(policy), Jsonb({"runtime_budget": saved, "node_attempts": counts})),
                )
            await cur.execute(
                "UPDATE research_budget_attempts SET status = 'abandoned', "
                "elapsed_seconds = greatest(elapsed_seconds, 0, "
                "extract(epoch FROM (least(clock_timestamp() AT TIME ZONE 'UTC', lease_until) - started_at))) "
                "WHERE run_id = %s AND status = 'active'", (self.lease.run_id,),
            )
            budget = await _snapshot(cur, self.lease.run_id)
            self._check(budget)
            await cur.execute(
                "INSERT INTO research_budget_attempts(attempt_id, run_id, owner_id, fencing_token, "
                "node, phase, agent, started_at, lease_until, elapsed_seconds, status, fallback_tokens, fallback_calls) "
                "SELECT %s, run_id, owner_id, fencing_token, %s, %s, %s, "
                "clock_timestamp() AT TIME ZONE 'UTC', lease_until, 0, 'active', 0, 0 "
                "FROM research_runs WHERE run_id = %s",
                (attempt_id, work.get("_node", phase), phase, agent, self.lease.run_id),
            )
            budget = await _snapshot(cur, self.lease.run_id)
        return attempt_id, budget

    async def observe(self, attempt_id, elapsed):
        async with lease_transaction(self.conn, self.lease) as cur:
            await cur.execute(
                "UPDATE research_budget_attempts a SET elapsed_seconds = greatest(a.elapsed_seconds, %s), "
                "lease_until = r.lease_until FROM research_runs r "
                "WHERE a.attempt_id = %s AND a.run_id = r.run_id AND a.run_id = %s "
                "AND a.owner_id = %s AND a.fencing_token = %s AND a.status = 'active'",
                (elapsed, attempt_id, self.lease.run_id, self.lease.owner_id, self.lease.fencing_token),
            )
            if cur.rowcount != 1:
                raise RuntimeError("预算物理尝试不存在或已经结束")
            return await _snapshot(cur, self.lease.run_id)

    async def reserve(self, attempt_id, model, tokens):
        call_id = str(uuid.uuid4())
        if type(tokens) is not int or tokens < 0:
            raise ValueError("模型请求预算预留必须为非负整数")
        async with lease_transaction(self.conn, self.lease) as cur:
            await cur.execute(
                "SELECT attempt_id FROM research_budget_attempts WHERE attempt_id = %s AND run_id = %s "
                "AND owner_id = %s AND fencing_token = %s AND status = 'active'",
                (attempt_id, self.lease.run_id, self.lease.owner_id, self.lease.fencing_token),
            )
            if await cur.fetchone() is None:
                raise RuntimeError("不能为已结束或其他执行器的尝试发起请求")
            self._check(await _snapshot(cur, self.lease.run_id), tokens)
            await cur.execute(
                "INSERT INTO research_budget_calls(call_id, attempt_id, model, reserved_tokens, status) "
                "VALUES (%s, %s, %s, %s, 'pending')", (call_id, attempt_id, model, tokens),
            )
        return call_id

    async def complete(self, attempt_id, call_id, usage):
        actual = None
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if type(total) is int and total >= 0:
                actual = total
            elif all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                actual = usage["prompt_tokens"] + usage["completion_tokens"]
        async with shared_connection(self.conn) as connection:
            async with connection.transaction(), connection.cursor(row_factory=dict_row) as cur:
                await self._audit_authorize(cur, attempt_id)
                await cur.execute(
                    "SELECT actual_tokens FROM research_budget_calls WHERE call_id = %s AND attempt_id = %s FOR UPDATE",
                    (call_id, attempt_id),
                )
                old = await cur.fetchone()
                if old is None:
                    raise RuntimeError("模型请求预算预留不存在")
                if old["actual_tokens"] is not None:
                    if actual is not None and old["actual_tokens"] != actual:
                        raise RuntimeError("同一请求的usage不一致，拒绝覆盖预算")
                    return
                await cur.execute(
                    "UPDATE research_budget_calls SET actual_tokens = %s, status = %s WHERE call_id = %s",
                    (actual, "completed" if actual is not None else "unknown", call_id),
                )

    async def _audit_authorize(self, cur, attempt_id):
        # 迟到的 usage 只可结算原请求；不授予旧执行器新的请求或图提交权限。
        await cur.execute("SELECT session_id, user_id FROM research_runs WHERE run_id = %s FOR UPDATE", (self.lease.run_id,))
        row = await cur.fetchone()
        if (row is None or row["session_id"] != self.lease.session_id
                or (str(row["user_id"]) if row["user_id"] else None) != self.lease.user_id):
            raise PermissionError("运行预算不属于当前会话/用户")
        await cur.execute(
            "SELECT attempt_id FROM research_budget_attempts WHERE attempt_id = %s AND run_id = %s "
            "AND owner_id = %s AND fencing_token = %s",
            (attempt_id, self.lease.run_id, self.lease.owner_id, self.lease.fencing_token),
        )
        if await cur.fetchone() is None:
            raise PermissionError("预算尝试不属于本执行器")

    async def finish(self, attempt_id, elapsed, status, tokens, calls):
        async with shared_connection(self.conn) as connection:
            async with connection.transaction(), connection.cursor(row_factory=dict_row) as cur:
                await self._audit_authorize(cur, attempt_id)
                await cur.execute(
                    "UPDATE research_budget_attempts SET elapsed_seconds = greatest(elapsed_seconds, %s), "
                    "status = CASE WHEN status = 'active' THEN %s ELSE status END, "
                    "fallback_tokens = greatest(fallback_tokens, %s), fallback_calls = greatest(fallback_calls, %s) "
                    "WHERE attempt_id = %s", (elapsed, status, max(0, tokens), max(0, calls), attempt_id),
                )
                await cur.execute("UPDATE research_budget_calls SET status = 'unknown' WHERE attempt_id = %s AND status = 'pending'", (attempt_id,))
                return await _snapshot(cur, self.lease.run_id)
