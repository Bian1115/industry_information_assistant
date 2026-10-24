"""Add versioned LLM semantic context summaries.

Revision ID: 20260830_0006
Revises: 20260829_0005
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260830_0006"
down_revision: Optional[str] = "20260829_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("chat_sessions")}
    additions = {
        "semantic_context_summary": sa.Column(
            "semantic_context_summary", postgresql.JSONB(), nullable=True
        ),
        "semantic_context_version": sa.Column(
            "semantic_context_version",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        "semantic_context_last_message_id": sa.Column(
            "semantic_context_last_message_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        "semantic_context_model": sa.Column(
            "semantic_context_model", sa.String(length=64), nullable=True
        ),
        "semantic_context_status": sa.Column(
            "semantic_context_status",
            sa.String(length=20),
            nullable=False,
            server_default="empty",
        ),
        "semantic_context_last_error": sa.Column(
            "semantic_context_last_error", sa.Text(), nullable=True
        ),
        "semantic_context_updated_at": sa.Column(
            "semantic_context_updated_at", sa.DateTime(), nullable=True
        ),
    }
    for name, column in additions.items():
        if name not in columns:
            op.add_column("chat_sessions", column)

    indexes = {index["name"] for index in inspector.get_indexes("memory_jobs")}
    if "uq_memory_jobs_active_semantic_context_session" not in indexes:
        op.create_index(
            "uq_memory_jobs_active_semantic_context_session",
            "memory_jobs",
            ["session_id"],
            unique=True,
            postgresql_where=sa.text(
                "operation_type = 'semantic_context' "
                "AND status IN ('pending', 'running', 'retry')"
            ),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    indexes = {index["name"] for index in inspector.get_indexes("memory_jobs")}
    if "uq_memory_jobs_active_semantic_context_session" in indexes:
        op.drop_index(
            "uq_memory_jobs_active_semantic_context_session",
            table_name="memory_jobs",
        )

    columns = {column["name"] for column in inspector.get_columns("chat_sessions")}
    for name in (
        "semantic_context_updated_at",
        "semantic_context_last_error",
        "semantic_context_status",
        "semantic_context_model",
        "semantic_context_last_message_id",
        "semantic_context_version",
        "semantic_context_summary",
    ):
        if name in columns:
            op.drop_column("chat_sessions", name)
