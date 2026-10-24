"""Link research checkpoints to an event-ledger cursor.

Revision ID: 20260826_0004
Revises: 20260823_0003
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa


revision: str = "20260826_0004"
down_revision: Optional[str] = "20260823_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("research_checkpoints")}
    if "run_id" not in columns:
        op.add_column(
            "research_checkpoints",
            sa.Column("run_id", sa.String(length=64), nullable=True),
        )
    if "last_event_seq" not in columns:
        op.add_column(
            "research_checkpoints",
            sa.Column(
                "last_event_seq",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
        )

    indexes = {index["name"] for index in inspector.get_indexes("research_checkpoints")}
    if "ix_research_checkpoints_run_id" not in indexes:
        op.create_index(
            "ix_research_checkpoints_run_id",
            "research_checkpoints",
            ["run_id"],
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    indexes = {index["name"] for index in inspector.get_indexes("research_checkpoints")}
    if "ix_research_checkpoints_run_id" in indexes:
        op.drop_index("ix_research_checkpoints_run_id", table_name="research_checkpoints")

    columns = {column["name"] for column in inspector.get_columns("research_checkpoints")}
    if "last_event_seq" in columns:
        op.drop_column("research_checkpoints", "last_event_seq")
    if "run_id" in columns:
        op.drop_column("research_checkpoints", "run_id")
