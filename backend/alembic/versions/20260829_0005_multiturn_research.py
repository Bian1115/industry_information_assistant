"""Add explicit per-turn execution metadata.

Revision ID: 20260829_0005
Revises: 20260826_0004
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa


revision: str = "20260829_0005"
down_revision: Optional[str] = "20260826_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "execution_mode" not in columns:
        op.add_column(
            "chat_messages",
            sa.Column(
                "execution_mode",
                sa.String(length=16),
                nullable=False,
                server_default="chat",
            ),
        )
    if "run_id" not in columns:
        op.add_column("chat_messages", sa.Column("run_id", sa.String(length=64)))
    if "research_status" not in columns:
        op.add_column(
            "chat_messages", sa.Column("research_status", sa.String(length=16))
        )

    indexes = {index["name"] for index in inspector.get_indexes("chat_messages")}
    if "ix_chat_messages_run_id" not in indexes:
        op.create_index("ix_chat_messages_run_id", "chat_messages", ["run_id"])

    # AgentEvent 是旧研究轮次的可靠标记；没有事件的旧消息继续视为 chat。
    op.execute(sa.text("""
        UPDATE chat_messages AS message
        SET execution_mode = 'research',
            run_id = event.run_id,
            research_status = CASE
                WHEN event.event_type = 'research_complete' THEN 'completed'
                WHEN event.event_type = 'research_cancelled' THEN 'cancelled'
                WHEN event.event_type = 'error' THEN 'failed'
                ELSE COALESCE(message.research_status, 'running')
            END
        FROM (
            SELECT DISTINCT ON (turn_id)
                turn_id, run_id, event_type
            FROM agent_events
            WHERE turn_id IS NOT NULL
            ORDER BY turn_id, seq DESC
        ) AS event
        WHERE message.turn_id = event.turn_id
    """))

    checkpoint_constraints = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("research_checkpoints")
    }
    if "uq_research_checkpoints_run_id" not in checkpoint_constraints:
        op.create_unique_constraint(
            "uq_research_checkpoints_run_id",
            "research_checkpoints",
            ["run_id"],
        )
    checkpoint_indexes = {
        index["name"]
        for index in inspector.get_indexes("research_checkpoints")
    }
    if "ix_research_checkpoints_session_updated" not in checkpoint_indexes:
        op.create_index(
            "ix_research_checkpoints_session_updated",
            "research_checkpoints",
            ["session_id", "updated_at"],
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    checkpoint_indexes = {
        index["name"]
        for index in inspector.get_indexes("research_checkpoints")
    }
    if "ix_research_checkpoints_session_updated" in checkpoint_indexes:
        op.drop_index(
            "ix_research_checkpoints_session_updated",
            table_name="research_checkpoints",
        )
    checkpoint_constraints = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("research_checkpoints")
    }
    if "uq_research_checkpoints_run_id" in checkpoint_constraints:
        op.drop_constraint(
            "uq_research_checkpoints_run_id",
            "research_checkpoints",
            type_="unique",
        )
    indexes = {index["name"] for index in inspector.get_indexes("chat_messages")}
    if "ix_chat_messages_run_id" in indexes:
        op.drop_index("ix_chat_messages_run_id", table_name="chat_messages")
    columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    for column in ("research_status", "run_id", "execution_mode"):
        if column in columns:
            op.drop_column("chat_messages", column)
