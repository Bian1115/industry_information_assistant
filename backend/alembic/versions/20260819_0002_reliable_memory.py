"""Add PostgreSQL conversations idempotency and reliable memory outbox.

Revision ID: 20260819_0002
Revises: 20260819_0001
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260819_0002"
down_revision: Optional[str] = "20260819_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Fresh database: create the complete current ORM schema. Existing databases
    # are stamped at 0001 and follow the explicit ALTER path below.
    if not inspector.has_table("users"):
        from core.database import Base
        import models  # noqa: F401  ensure every table is registered

        Base.metadata.create_all(bind=bind)
        return

    message_columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "turn_id" not in message_columns:
        op.add_column("chat_messages", sa.Column("turn_id", postgresql.UUID(as_uuid=True)))
    message_constraints = {
        constraint["name"] for constraint in inspector.get_unique_constraints("chat_messages")
    }
    if "uq_chat_messages_session_turn_role" not in message_constraints:
        op.create_unique_constraint(
            "uq_chat_messages_session_turn_role",
            "chat_messages",
            ["session_id", "turn_id", "role"],
        )
    message_indexes = {index["name"] for index in inspector.get_indexes("chat_messages")}
    if "ix_chat_messages_session_created" not in message_indexes:
        op.create_index(
            "ix_chat_messages_session_created",
            "chat_messages",
            ["session_id", "created_at"],
        )

    memory_columns = {column["name"] for column in inspector.get_columns("long_term_memories")}
    additions = [
        ("provider", sa.Column("provider", sa.String(32), nullable=False, server_default="native")),
        ("source_start_message_id", sa.Column("source_start_message_id", postgresql.UUID(as_uuid=True))),
        ("source_end_message_id", sa.Column("source_end_message_id", postgresql.UUID(as_uuid=True))),
        ("index_status", sa.Column("index_status", sa.String(20), nullable=False, server_default="ready")),
        ("last_error", sa.Column("last_error", sa.Text())),
        ("deleted_at", sa.Column("deleted_at", sa.DateTime())),
        ("updated_at", sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP"))),
    ]
    for name, column in additions:
        if name not in memory_columns:
            op.add_column("long_term_memories", column)

    foreign_keys = {fk.get("name") for fk in inspector.get_foreign_keys("long_term_memories")}
    if "fk_long_term_memories_source_start_message" not in foreign_keys:
        op.create_foreign_key(
            "fk_long_term_memories_source_start_message",
            "long_term_memories",
            "chat_messages",
            ["source_start_message_id"],
            ["id"],
            ondelete="SET NULL",
        )
    if "fk_long_term_memories_source_end_message" not in foreign_keys:
        op.create_foreign_key(
            "fk_long_term_memories_source_end_message",
            "long_term_memories",
            "chat_messages",
            ["source_end_message_id"],
            ["id"],
            ondelete="SET NULL",
        )

    memory_constraints = {
        constraint["name"] for constraint in inspector.get_unique_constraints("long_term_memories")
    }
    if "uq_long_term_memories_session_source_end" not in memory_constraints:
        op.create_unique_constraint(
            "uq_long_term_memories_session_source_end",
            "long_term_memories",
            ["session_id", "source_end_message_id"],
        )
    memory_indexes = {index["name"] for index in inspector.get_indexes("long_term_memories")}
    if "ix_long_term_memories_user_status" not in memory_indexes:
        op.create_index(
            "ix_long_term_memories_user_status",
            "long_term_memories",
            ["user_id", "index_status"],
        )

    if not inspector.has_table("memory_jobs"):
        op.create_table(
            "memory_jobs",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("operation_type", sa.String(20), nullable=False),
            sa.Column("provider", sa.String(32), nullable=False, server_default="native"),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("session_id", postgresql.UUID(as_uuid=True)),
            sa.Column("memory_id", postgresql.UUID(as_uuid=True)),
            sa.Column("payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("idempotency_key", sa.String(255), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("next_attempt_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("locked_at", sa.DateTime()),
            sa.Column("last_error", sa.Text()),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["memory_id"], ["long_term_memories.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("idempotency_key", name="uq_memory_jobs_idempotency_key"),
        )
        op.create_index(
            "ix_memory_jobs_status_next_attempt",
            "memory_jobs",
            ["status", "next_attempt_at"],
        )
        op.create_index(
            "uq_memory_jobs_active_add_session",
            "memory_jobs",
            ["session_id"],
            unique=True,
            postgresql_where=sa.text(
                "operation_type = 'add' AND status IN ('pending', 'running', 'retry')"
            ),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("memory_jobs"):
        op.drop_table("memory_jobs")

    op.drop_index("ix_long_term_memories_user_status", table_name="long_term_memories")
    op.drop_constraint(
        "uq_long_term_memories_session_source_end",
        "long_term_memories",
        type_="unique",
    )
    op.drop_constraint(
        "fk_long_term_memories_source_end_message",
        "long_term_memories",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_long_term_memories_source_start_message",
        "long_term_memories",
        type_="foreignkey",
    )
    for column in [
        "updated_at",
        "deleted_at",
        "last_error",
        "index_status",
        "source_end_message_id",
        "source_start_message_id",
        "provider",
    ]:
        op.drop_column("long_term_memories", column)

    op.drop_index("ix_chat_messages_session_created", table_name="chat_messages")
    op.drop_constraint(
        "uq_chat_messages_session_turn_role",
        "chat_messages",
        type_="unique",
    )
    op.drop_column("chat_messages", "turn_id")
