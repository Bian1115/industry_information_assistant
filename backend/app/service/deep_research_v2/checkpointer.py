# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""LangGraph PostgreSQL 检查点的连接生命周期与部署建表。

checkpoint-postgres 使用 psycopg3 异步连接，不复用项目现有的 psycopg2
SQLAlchemy 连接，因此这里单独管理一个连接池，并挂在应用生命周期上。
建表是独立的部署步骤，不在每次请求时执行。
"""

from __future__ import annotations

import logging
import os
import asyncio
from weakref import WeakKeyDictionary
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Optional
from urllib.parse import quote

from psycopg.conninfo import conninfo_to_dict
from psycopg import AsyncConnection
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger("DeepResearchCheckpointer")
_connection_locks = WeakKeyDictionary()
_write_artifacts = ContextVar("checkpoint_write_artifacts", default=())
_write_events = ContextVar("checkpoint_commit_events", default=None)


@asynccontextmanager
async def shared_connection(conn):
    """单连接的测试/嵌入入口也需串行，避免并发事务变成彼此的savepoint。"""
    from langgraph.checkpoint.postgres import _ainternal
    if isinstance(conn, AsyncConnection):
        lock = _connection_locks.setdefault(conn, asyncio.Lock())
        async with lock:
            yield conn
    else:
        async with _ainternal.get_connection(conn) as connection:
            yield connection


def business_database_url() -> str:
    """租约的权威库；返回值只能用于连接，不得写入日志。"""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = quote(os.getenv("POSTGRES_USER", "postgres"), safe="")
    password = quote(os.getenv("POSTGRES_PASSWORD", "postgres123"), safe="")
    database = os.getenv("POSTGRES_DB", "industry_assistant")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def checkpoint_database_url() -> str:
    """检查点使用的 PostgreSQL 连接串；密钥不会被写进日志。

    检查点与 research_runs 必须同库，才能在同一事务内校验 fencing。
    显式连接串可使用不同数据库账号，但不能指向独立实例或不同 search_path。
    """
    explicit = os.getenv("RESEARCH_CHECKPOINT_DATABASE_URL", "").strip()
    if explicit:
        requested = conninfo_to_dict(explicit)
        authority = conninfo_to_dict(business_database_url())
        for key in ("host", "hostaddr", "port", "dbname", "options"):
            default = "5432" if key == "port" else ""
            if requested.get(key, default) != authority.get(key, default):
                raise ValueError("检查点必须与运行租约同库；请使用相同的 host、port、dbname 和 search_path")
        return explicit
    return business_database_url()


async def require_lease(cur, lease, *, lock=True):
    """锁住权威运行行后校验；锁等待结束再用数据库时钟判断到期。"""
    from service.research_events import ResearchRunLeaseLost
    await cur.execute(
        "SELECT session_id, user_id, owner_id, fencing_token FROM research_runs WHERE run_id = %s"
        + (" FOR UPDATE" if lock else ""), (lease.run_id,),
    )
    row = await cur.fetchone()
    if (row is None or row["session_id"] != lease.session_id
            or (str(row["user_id"]) if row["user_id"] else None) != lease.user_id
            or row["owner_id"] != lease.owner_id or row["fencing_token"] != lease.fencing_token):
        raise ResearchRunLeaseLost("执行租约已被接管，拒绝持久化研究结果")
    await cur.execute(
        "SELECT (lease_until AT TIME ZONE 'UTC') > clock_timestamp() AS valid "
        "FROM research_runs WHERE run_id = %s", (lease.run_id,),
    )
    if not (await cur.fetchone())["valid"]:
        raise ResearchRunLeaseLost("执行租约已到期，拒绝持久化研究结果")


@asynccontextmanager
async def lease_transaction(conn, lease):
    """校验、写入及提交共用一个连接和行锁；接管必须等该事务结束。"""
    async with shared_connection(conn) as connection:
        async with connection.transaction(), connection.cursor(binary=True, row_factory=dict_row) as cur:
            await require_lease(cur, lease)
            yield cur
            # 写入期间也可能自然过期；回滚全部 blob、pending writes 和 checkpoint。
            await require_lease(cur, lease, lock=False)


class LeaseFencedPostgresSaver(AsyncPostgresSaver):
    """只为本执行器创建实例；共享连接池，禁止共享可变租约身份。"""

    def __init__(self, conn, *, lease=None, serde=None):
        super().__init__(conn, serde=serde)
        self.lease = lease
        self.storage_coordinator = None
        self.event_turn_id = None
        self.on_events_committed = None

    def for_lease(self, lease):
        return type(self)(self.conn, lease=lease, serde=self.serde)

    def _require_thread(self, config):
        if self.lease is None or config["configurable"]["thread_id"] != self.lease.run_id:
            raise ValueError("检查点写入必须绑定当前 run 的执行租约")

    async def aput(self, config, checkpoint, metadata, new_versions):
        self._require_thread(config)
        from .checkpoint_events import EVENTS_CHANNEL
        token = _write_artifacts.set(tuple(checkpoint.get("channel_values", {}).values()))
        event_token = _write_events.set((
            checkpoint.get("channel_values", {}).get(EVENTS_CHANNEL, []),
            {"kind": "checkpoint", "checkpoint_id": checkpoint["id"],
             "checkpoint_ns": config["configurable"].get("checkpoint_ns", "")},
        ))
        try:
            return await super().aput(config, checkpoint, metadata, new_versions)
        finally:
            _write_events.reset(event_token)
            _write_artifacts.reset(token)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        self._require_thread(config)
        from .checkpoint_events import EVENTS_CHANNEL
        # 节点完成通知等待完整快照；章节 task 返回值与 pending write 一起提交。
        intents = [intent for channel, value in writes
                   if channel == "__return__" and isinstance(value, dict)
                   for intent in value.get(EVENTS_CHANNEL, [])]
        token = _write_artifacts.set(tuple(value for _, value in writes))
        event_token = _write_events.set((intents, {
            "kind": "task_result", "checkpoint_id": config["configurable"]["checkpoint_id"],
            "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
            "task_id": task_id,
        }))
        try:
            return await super().aput_writes(config, writes, task_id, task_path)
        finally:
            _write_events.reset(event_token)
            _write_artifacts.reset(token)

    @asynccontextmanager
    async def _cursor(self, *, pipeline=False):
        if pipeline and self.lease is not None:
            inserted = 0
            # PostgreSQL pipeline 自动提交不能维持业务行锁，写操作改为显式事务。
            async with self.lock, lease_transaction(self.conn, self.lease) as cur:
                from .storage_reconciliation import StorageCoordinator, artifacts
                for value in _write_artifacts.get():
                    for artifact in artifacts(value):
                        if self.storage_coordinator is not None or artifact.get("storage_manifest") or artifact.get("storage_protocol"):
                            coordinator = self.storage_coordinator or StorageCoordinator(self.conn, self.lease.run_id, lease=self.lease)
                            await coordinator.validate(cur, artifact)
                yield cur
                pending = _write_events.get()
                if pending is not None:
                    from .checkpoint_events import append_events
                    intents, reference = pending
                    if intents and reference["kind"] == "task_result":
                        # 上游 saver 的重复 task 写入使用 ON CONFLICT DO NOTHING。
                        # 必须确认实际保留的产物，不能为未被保存的候选结果发完成事件。
                        await cur.execute(
                            "SELECT type,blob FROM checkpoint_writes WHERE thread_id=%s "
                            "AND checkpoint_ns=%s AND checkpoint_id=%s AND task_id=%s AND channel='__return__'",
                            (self.lease.run_id, reference["checkpoint_ns"],
                             reference["checkpoint_id"], reference["task_id"]),
                        )
                        stored = await cur.fetchone()
                        artifact = self.serde.loads_typed((stored["type"], bytes(stored["blob"]))) if stored else None
                        candidate = next((value for value in _write_artifacts.get()
                                          if isinstance(value, dict) and value.get("checkpoint_events") == intents), None)
                        if artifact is None or artifact != candidate:
                            from service.research_events import ResearchEventPersistenceError
                            raise ResearchEventPersistenceError("章节任务结果已存在且完成事件不同，拒绝追加新事件")
                    inserted = await append_events(cur, self.lease, *pending, turn_id=self.event_turn_id)
            # 只有退出 lease_transaction、COMMIT 成功后才唤醒发送器。
            if inserted and self.on_events_committed is not None:
                self.on_events_committed()
        elif isinstance(self.conn, AsyncConnection):
            async with self.lock, shared_connection(self.conn) as connection:
                if pipeline:
                    async with connection.transaction(), connection.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                else:
                    async with connection.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
        else:
            async with super()._cursor(pipeline=pipeline) as cur:
                yield cur


def for_execution(checkpointer, lease):
    """生产入口不得静默退回无 fencing 的 saver。"""
    if not isinstance(checkpointer, AsyncPostgresSaver) or checkpointer.pipe is not None:
        raise ValueError("可靠研究执行必须使用无外部 pipeline 的 PostgreSQL saver")
    return LeaseFencedPostgresSaver(checkpointer.conn, lease=lease, serde=checkpointer.serde)


def build_pool(database_url: Optional[str] = None) -> AsyncConnectionPool:
    """构造尚未打开的连接池；saver 需要按字段名读取数据库行。"""
    return AsyncConnectionPool(
        conninfo=database_url or checkpoint_database_url(),
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )


@asynccontextmanager
async def checkpointer_lifespan(app: Any, database_url: Optional[str] = None):
    """把 saver 和连接池挂在应用上；应与现有 lifespan 合并而不是替换它。"""
    pool = build_pool(database_url)
    await pool.open()
    try:
        await pool.wait()
        app.state.research_checkpointer = LeaseFencedPostgresSaver(pool)
        app.state.research_checkpointer_pool = pool
        logger.info("LangGraph PostgreSQL 检查点连接池已就绪")
        yield app.state.research_checkpointer
    finally:
        app.state.research_checkpointer = None
        app.state.research_checkpointer_pool = None
        await pool.close()
        logger.info("LangGraph PostgreSQL 检查点连接池已关闭")


async def prepare_checkpoint_schema(database_url: Optional[str] = None) -> None:
    """部署时调用一次，创建或迁移 saver 所需的表结构。"""
    async with AsyncPostgresSaver.from_conn_string(
        database_url or checkpoint_database_url()
    ) as saver:
        await saver.setup()
    logger.info("LangGraph 检查点表结构已就绪")


if __name__ == "__main__":  # pragma: no cover - 部署脚本入口
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(prepare_checkpoint_schema())
