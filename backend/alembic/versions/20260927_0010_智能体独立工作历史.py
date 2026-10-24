"""新增智能体上下文和追加式原文历史；不修改聊天与研究检查点。"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = "20260927_0010"
down_revision = "20260924_0009"
branch_labels = None
depends_on = None


def upgrade():
    # 允许部署时先独立准备这两张表，而不顺带执行其他待迁移项。
    inspector = sa.inspect(op.get_bind())
    existing = {name for name in ("agent_contexts", "agent_messages") if inspector.has_table(name)}
    if existing:
        expected = {
            "agent_contexts": {"id", "run_id", "agent_name", "context_key", "next_seq", "summary", "summary_version", "covered_seq", "summary_metadata", "offloaded_source_ids", "created_at", "updated_at"},
            "agent_messages": {"id", "context_id", "seq", "kind", "call_id", "invocation_id", "node", "payload", "content_hash", "created_at"},
        }
        if existing != set(expected) or any({c["name"] for c in inspector.get_columns(table)} != cols for table, cols in expected.items()):
            raise RuntimeError("智能体历史表存在但结构不完整，拒绝跳过迁移")
        for table, unique in [("agent_contexts", ["run_id", "agent_name", "context_key"]), ("agent_messages", ["context_id", "seq"])]:
            if not any(c["column_names"] == unique for c in inspector.get_unique_constraints(table)):
                raise RuntimeError("智能体历史表缺少隔离/排序唯一约束")
        return
    op.create_table(
        "agent_contexts",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("research_runs.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_name", sa.String(64), nullable=False),
        sa.Column("context_key", sa.String(128), nullable=False),
        sa.Column("next_seq", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("summary_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("covered_seq", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("summary_metadata", pg.JSONB(), nullable=False, server_default="{}"),
        sa.Column("offloaded_source_ids", pg.JSONB(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("run_id", "agent_name", "context_key", name="uq_agent_context_scope"),
    )
    op.create_table(
        "agent_messages",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("context_id", pg.UUID(as_uuid=True), sa.ForeignKey("agent_contexts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("call_id", pg.UUID(as_uuid=True), nullable=False),
        sa.Column("invocation_id", pg.UUID(as_uuid=True), nullable=False),
        sa.Column("node", sa.String(64), nullable=False),
        sa.Column("payload", pg.JSONB(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("context_id", "seq", name="uq_agent_message_seq"),
    )


def downgrade():
    op.drop_table("agent_messages")
    op.drop_table("agent_contexts")
