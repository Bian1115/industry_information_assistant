"""会话在最后一次新消息写入 24 小时后过期。"""

from alembic import op
import sqlalchemy as sa

revision = "20261006_0012"
down_revision = "20261003_0011"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    session_columns = {column["name"]: column for column in inspector.get_columns("chat_sessions")}
    if "expires_at" not in session_columns:
        op.add_column("chat_sessions", sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True))
    message_columns = {column["name"]: column for column in inspector.get_columns("chat_messages")}
    # 兼容已手工建表但尚未执行 0009 的本地数据库；旧 naive 时间按 UTC 读取。
    message_time = "m.created_at" if message_columns["created_at"]["type"].timezone else "m.created_at AT TIME ZONE 'UTC'"
    session_time = "s.created_at" if session_columns["created_at"]["type"].timezone else "s.created_at AT TIME ZONE 'UTC'"
    # 旧摘要更新也会改 updated_at；回填必须以原始消息时间为准。
    op.execute(f"""
        UPDATE chat_sessions s SET expires_at = COALESCE(
            (SELECT MAX({message_time}) FROM chat_messages m WHERE m.session_id = s.id),
            {session_time}, CURRENT_TIMESTAMP
        ) + INTERVAL '24 hours'
        WHERE s.expires_at IS NULL
    """)
    op.alter_column("chat_sessions", "expires_at", nullable=False,
                    server_default=sa.text("CURRENT_TIMESTAMP + INTERVAL '24 hours'"))
    if "ix_chat_sessions_expires_at" not in {index["name"] for index in inspector.get_indexes("chat_sessions")}:
        op.create_index("ix_chat_sessions_expires_at", "chat_sessions", ["expires_at"])


def downgrade():
    op.drop_index("ix_chat_sessions_expires_at", table_name="chat_sessions")
    op.drop_column("chat_sessions", "expires_at")
