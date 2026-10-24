"""Single LangGraph main chain: run projection columns and delivery idempotency.

新增 research_runs 的图版本与投影字段，并为终态交付加数据库级唯一约束：
一个 run 最多只能有一条 research_complete 事件，重复交付不会产生第二份
报告或第二个完成序号。

Revision ID: 20260919_0008
Revises: 20260831_0007
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260919_0008"
down_revision: Optional[str] = "20260831_0007"
branch_labels = None
depends_on = None


NEW_COLUMNS = (
    ("graph_version", sa.Column("graph_version", sa.String(length=32), nullable=True)),
    ("final_checkpoint_id", sa.Column("final_checkpoint_id", sa.String(length=64), nullable=True)),
    (
        "last_projected_checkpoint_id",
        sa.Column("last_projected_checkpoint_id", sa.String(length=64), nullable=True),
    ),
    (
        "delivered_event_id",
        sa.Column("delivered_event_id", postgresql.UUID(as_uuid=True), nullable=True),
    ),
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("research_runs"):
        return

    existing = {column["name"] for column in inspector.get_columns("research_runs")}
    for name, column in NEW_COLUMNS:
        if name not in existing:
            op.add_column("research_runs", column)

    indexes = {index["name"] for index in inspector.get_indexes("research_runs")}
    if "ix_research_runs_projection" not in indexes:
        op.create_index(
            "ix_research_runs_projection",
            "research_runs",
            ["status", "last_projected_checkpoint_id"],
        )

    # 终态交付幂等：一个 run 只允许存在一条 research_complete 事件。
    # 没有这条约束时，"检查重复 -> 再插入"之间仍有并发窗口。
    if inspector.has_table("agent_events"):
        op.execute(sa.text("""
            DELETE FROM agent_events AS duplicated
            USING agent_events AS kept
            WHERE duplicated.event_type = 'research_complete'
              AND kept.event_type = 'research_complete'
              AND duplicated.run_id = kept.run_id
              AND duplicated.seq > kept.seq
        """))
        op.execute(sa.text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_events_run_complete
            ON agent_events (run_id)
            WHERE event_type = 'research_complete'
        """))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("agent_events"):
        op.execute(sa.text("DROP INDEX IF EXISTS uq_agent_events_run_complete"))
    if not inspector.has_table("research_runs"):
        return
    indexes = {index["name"] for index in inspector.get_indexes("research_runs")}
    if "ix_research_runs_projection" in indexes:
        op.drop_index("ix_research_runs_projection", table_name="research_runs")
    existing = {column["name"] for column in inspector.get_columns("research_runs")}
    for name, _ in reversed(NEW_COLUMNS):
        if name in existing:
            op.drop_column("research_runs", name)
