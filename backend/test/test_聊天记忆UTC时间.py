"""时间归一化、数据库类型和历史列迁移的离线回归。"""
import importlib.util
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite
from core.time_utils import UTCDateTime, as_utc, utc_now
from models.chat import ChatAttachment, ChatSession, ChatMessage, LongTermMemory, MemoryJob
from service.memory_service import _select_relevant_raw_messages

MIGRATION = Path(__file__).resolve().parents[1] / 'alembic/versions/20260924_0009_聊天记忆时间统一UTC.py'
spec = importlib.util.spec_from_file_location('utc_migration', MIGRATION)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


class UTCTimeTests(unittest.TestCase):
    def test_clock_and_naive_legacy_are_utc(self):
        self.assertEqual(utc_now().utcoffset(), timedelta(0))
        legacy = datetime(2026, 9, 24, 13)
        self.assertEqual(as_utc(legacy), datetime(2026, 9, 24, 13, tzinfo=timezone.utc))

    def test_offset_conversion_preserves_instant(self):
        original = datetime(2026, 9, 24, 21, tzinfo=timezone(timedelta(hours=8)))
        self.assertEqual(as_utc(original).hour, 13)
        self.assertEqual(as_utc(original).timestamp(), original.timestamp())

    def test_database_binding_and_reading_do_not_use_host_timezone(self):
        column = UTCDateTime()
        local = datetime(2026, 9, 24, 21, tzinfo=timezone(timedelta(hours=8)))
        pg_value = column.process_bind_param(local, postgresql.dialect())
        self.assertEqual(pg_value.hour, 13)
        self.assertEqual(pg_value.utcoffset(), timedelta(0))
        sqlite_value = column.process_bind_param(local, sqlite.dialect())
        self.assertEqual(sqlite_value, datetime(2026, 9, 24, 13))
        self.assertEqual(column.process_result_value(sqlite_value, sqlite.dialect()), pg_value)
        self.assertEqual(column.process_result_value(local, postgresql.dialect()), pg_value)
        self.assertIsNone(column.process_bind_param(None, postgresql.dialect()))
        self.assertIsNone(column.process_result_value(None, postgresql.dialect()))

    def test_model_time_columns_declare_timestamptz(self):
        for model in (ChatAttachment, ChatSession, ChatMessage, LongTermMemory, MemoryJob):
            for column in model.__table__.columns:
                if isinstance(column.type, UTCDateTime):
                    with self.subTest(table=model.__tablename__, column=column.name):
                        self.assertEqual(column.type.compile(dialect=postgresql.dialect()), 'TIMESTAMP WITH TIME ZONE')
                        if column.default is not None:
                            self.assertEqual(column.default.arg(None).utcoffset(), timedelta(0))

    def test_recalled_messages_sort_mixed_legacy_and_aware_times(self):
        rows = [
            SimpleNamespace(id='later', content='UTC', created_at=datetime(2026, 9, 24, 21, tzinfo=timezone(timedelta(hours=8)))),
            SimpleNamespace(id='earlier', content='UTC', created_at=datetime(2026, 9, 24, 12)),
        ]
        result = _select_relevant_raw_messages('UTC', [], rows)
        self.assertEqual([row.id for row in result], ['earlier', 'later'])


class UTCMigrationTests(unittest.TestCase):
    def setUp(self):
        self.bind = SimpleNamespace(dialect=postgresql.dialect())
        self.inspector = Mock()
        self.inspector.has_table.side_effect = lambda table: table == 'chat_messages'
        self.inspector.get_columns.return_value = [{'name': 'created_at', 'type': sa.DateTime(timezone=False)}]

    def test_requires_explicit_legacy_timezone_before_altering(self):
        with patch.object(migration.op, 'get_bind', return_value=self.bind), patch.object(
            migration.sa, 'inspect', return_value=self.inspector
        ), patch.object(migration.context, 'get_x_argument', return_value={}), patch.object(
            migration.op, 'alter_column'
        ) as alter:
            with self.assertRaisesRegex(RuntimeError, '核实旧时间原时区'):
                migration.upgrade()
            alter.assert_not_called()

    def test_utc_conversion_uses_explicit_zone_not_database_session_setting(self):
        with patch.object(migration.op, 'get_bind', return_value=self.bind), patch.object(
            migration.sa, 'inspect', return_value=self.inspector
        ), patch.object(migration.context, 'get_x_argument', return_value={'legacy_timezone': 'UTC'}), patch.object(
            migration.op, 'alter_column'
        ) as alter:
            migration.upgrade()
            self.assertEqual(alter.call_count, 1)
            self.assertEqual(alter.call_args.kwargs['postgresql_using'], "created_at AT TIME ZONE 'UTC'")
            self.assertTrue(alter.call_args.kwargs['type_'].timezone)

    def test_existing_aware_columns_are_not_reinterpreted(self):
        self.inspector.get_columns.return_value = [{'name': 'created_at', 'type': sa.DateTime(timezone=True)}]
        with patch.object(migration.op, 'get_bind', return_value=self.bind), patch.object(
            migration.sa, 'inspect', return_value=self.inspector
        ), patch.object(migration.op, 'alter_column') as alter:
            migration.upgrade()
            alter.assert_not_called()


if __name__ == '__main__':
    unittest.main()
