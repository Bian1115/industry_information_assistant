"""会话共享智能体历史；旧上下文由历史服务按授权会话惰性接续。"""
from alembic import op
import sqlalchemy as sa

revision = "20261003_0011"
down_revision = "20260927_0010"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("agent_contexts", sa.Column("history_scope_id", sa.String(128), nullable=True))
    op.add_column("agent_messages", sa.Column("run_id", sa.String(64), nullable=True))
    op.execute("UPDATE agent_contexts SET history_scope_id = 'run:' || run_id")
    op.execute("UPDATE agent_messages m SET run_id = c.run_id FROM agent_contexts c WHERE m.context_id = c.id")
    op.alter_column("agent_contexts", "history_scope_id", nullable=False)
    op.alter_column("agent_messages", "run_id", nullable=False)
    op.create_index("ix_agent_messages_run_id", "agent_messages", ["run_id"])
    op.drop_constraint("uq_agent_context_scope", "agent_contexts", type_="unique")
    op.create_unique_constraint("uq_agent_context_scope", "agent_contexts", ["history_scope_id", "agent_name", "context_key"])
    for fk in sa.inspect(op.get_bind()).get_foreign_keys("agent_contexts"):
        if fk["constrained_columns"] == ["run_id"]:
            op.drop_constraint(fk["name"], "agent_contexts", type_="foreignkey")
    op.alter_column("agent_contexts", "run_id", nullable=True)
    op.create_foreign_key("fk_agent_context_origin_run", "agent_contexts", "research_runs", ["run_id"], ["run_id"], ondelete="SET NULL")


def downgrade():
    # 接续后一个context含多个run，直接恢复旧schema会丢失审计与摘要边界。
    raise RuntimeError("会话历史接续迁移不支持有损自动降级；请从迁移前备份恢复")
