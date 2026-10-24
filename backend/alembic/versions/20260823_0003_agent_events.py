"""Add append-only DeepResearch event ledger.

Revision ID: 20260823_0003
Revises: 20260819_0002
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260823_0003"
down_revision: Optional[str] = "20260819_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("agent_events"):
        return
    op.create_table(
        "agent_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("occurred_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("run_id", "seq", name="uq_agent_events_run_seq"),
    )
    op.create_index("ix_agent_events_session_created", "agent_events", ["session_id", "created_at"])
    op.create_index("ix_agent_events_run_seq", "agent_events", ["run_id", "seq"])
    op.create_index("ix_agent_events_user_run", "agent_events", ["user_id", "run_id"])


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("agent_events"):
        op.drop_table("agent_events")
