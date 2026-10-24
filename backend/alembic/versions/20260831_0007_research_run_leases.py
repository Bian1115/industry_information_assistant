"""Add per-run leases, fencing tokens, and atomic event counters.

Revision ID: 20260831_0007
Revises: 20260830_0006
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260831_0007"
down_revision: Optional[str] = "20260830_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("research_runs"):
        op.create_table(
            "research_runs",
            sa.Column("run_id", sa.String(length=64), nullable=False),
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("next_event_seq", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("owner_id", sa.String(length=64), nullable=True),
            sa.Column("lease_until", sa.DateTime(), nullable=True),
            sa.Column("fencing_token", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("status", sa.String(length=16), nullable=False, server_default="idle"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("run_id"),
        )
        op.create_index("ix_research_runs_session_id", "research_runs", ["session_id"])
        op.create_index("ix_research_runs_user_status", "research_runs", ["user_id", "status"])
        op.create_index("ix_research_runs_lease_until", "research_runs", ["lease_until"])

    # 把旧事件账本的最大序号迁移为各 run 的计数器高水位。
    op.execute(sa.text("""
        INSERT INTO research_runs (
            run_id, session_id, user_id, next_event_seq, fencing_token, status
        )
        SELECT
            event.run_id,
            MIN(event.session_id),
            MIN(event.user_id::text)::uuid,
            MAX(event.seq),
            0,
            CASE
                WHEN BOOL_OR(event.event_type = 'research_complete') THEN 'completed'
                WHEN BOOL_OR(event.event_type = 'research_cancelled') THEN 'cancelled'
                WHEN BOOL_OR(event.event_type = 'error') THEN 'failed'
                ELSE 'paused'
            END
        FROM agent_events AS event
        GROUP BY event.run_id
        ON CONFLICT (run_id) DO UPDATE
        SET next_event_seq = GREATEST(research_runs.next_event_seq, EXCLUDED.next_event_seq),
            updated_at = CURRENT_TIMESTAMP
    """))


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("research_runs"):
        op.drop_table("research_runs")
