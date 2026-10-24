"""聊天与记忆时间字段迁移为 timestamptz，已有带时区的列不重解释。

旧 naive 数据的原时区需由部署者显式指定，例如：
alembic -x legacy_timezone=UTC upgrade 20260924_0009
旧应用 datetime.utcnow() 写入值应选择 UTC；本地时间导入数据需先核实。
"""

from zoneinfo import ZoneInfo

from alembic import context, op
import sqlalchemy as sa

revision = "20260924_0009"
down_revision = "20260919_0008"
branch_labels = None
depends_on = None

TIME_COLUMNS = {
    "chat_attachments": ("created_at", "updated_at"),
    "chat_sessions": ("created_at", "updated_at", "semantic_context_updated_at"),
    "chat_messages": ("created_at",),
    "long_term_memories": ("created_at", "updated_at", "deleted_at"),
    "memory_jobs": ("created_at", "updated_at", "next_attempt_at", "locked_at"),
}


def columns_to_convert(inspector, *, to_aware):
    """只选择时区属性需要变化的现存日期列。"""
    result = []
    for table, names in TIME_COLUMNS.items():
        if not inspector.has_table(table):
            continue
        for column in inspector.get_columns(table):
            if column["name"] in names and isinstance(column["type"], sa.DateTime):
                if bool(column["type"].timezone) != to_aware:
                    result.append((table, column))
    return result


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("聊天记忆时间迁移仅适用于 PostgreSQL")
    pending = columns_to_convert(sa.inspect(bind), to_aware=True)
    if not pending:
        return
    legacy_timezone = context.get_x_argument(as_dictionary=True).get("legacy_timezone")
    if not legacy_timezone:
        raise RuntimeError("请核实旧时间原时区，并通过 -x legacy_timezone=UTC（或实际 IANA 时区）显式指定")
    ZoneInfo(legacy_timezone)  # 拒绝非法时区；任何 DDL 执行前完成验证。
    zone_literal = str(sa.literal(legacy_timezone).compile(
        dialect=bind.dialect, compile_kwargs={"literal_binds": True},
    ))
    for table, column in pending:
        quoted = bind.dialect.identifier_preparer.quote(column["name"])
        op.alter_column(
            table, column["name"], existing_type=column["type"],
            type_=sa.DateTime(timezone=True),
            postgresql_using=f"{quoted} AT TIME ZONE {zone_literal}",
        )


def downgrade():
    # 部署前各列可能已有混合类型，无法仅凭版本号恢复原列类型。
    # 禁止把原先就带时区的列误降级，需基于部署备份另行安排回退。
    raise RuntimeError("禁止自动丢弃聊天记忆时间的时区信息；请基于迁移前 schema 备份回退")
