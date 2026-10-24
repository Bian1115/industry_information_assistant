"""聊天与记忆链路使用 aware UTC；旧应用产生的 naive 时间按 UTC 兼容读取。"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator


def utc_now() -> datetime:
    """生成带 UTC 时区的当前时间。"""
    return datetime.now(timezone.utc)


def session_expiry_time(now: datetime | None = None) -> datetime:
    """会话从创建或最后一次新消息写入起保留 24 小时。"""
    return as_utc(now if now is not None else utc_now()) + timedelta(hours=24)


def as_utc(value: datetime) -> datetime:
    """规范化为 UTC，保留真实时刻。

    naive 兼容仅适用于此前由 datetime.utcnow() 生成的旧应用数据。
    外部输入若为本地时间，应先明确原时区，不能直接调用本函数猜测时区。
    """
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class UTCDateTime(TypeDecorator):
    """PostgreSQL 使用 timestamptz，ORM 读写统一返回/传入 aware UTC。

    SQLite 不保存时区：离线测试写入 UTC 墙上时间，读取时再显式补 UTC。
    声明类型不会自动迁移已经存在的数据库列，需运行对应 Alembic 迁移。
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        value = as_utc(value)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value, dialect):
        return as_utc(value) if value is not None else None
