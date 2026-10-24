"""只准备智能体历史两张表，不推进其他 Alembic 迁移或重启后端。"""

import importlib.util
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")
sys.path.insert(0, str(BACKEND / "app"))

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import inspect, text
from core.database import engine


def main():
    path = BACKEND / "alembic/versions/20260927_0010_智能体独立工作历史.py"
    spec = importlib.util.spec_from_file_location("agent_history_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL lock_timeout = '5s'"))
        conn.execute(text("SET LOCAL statement_timeout = '30s'"))
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
        migration.op = Operations(MigrationContext.configure(conn))
        migration.upgrade()
        after = conn.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
        assert version == after
        result = {"prepared_tables": [name for name in ("agent_contexts", "agent_messages") if inspect(conn).has_table(name)], "alembic_version_unchanged": version}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
