#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
清理无归属的研究检查点（user_id IS NULL）

背景：
    在研究接口加上归属校验之前，research_router 从未把 user_id 传给 service，
    所以那段时期产生的 research_checkpoints 记录，user_id 全部是 NULL。

    加上归属校验后，这些记录任何用户都查不到（SQL 的 WHERE 里带了 user_id 条件），
    成了既占空间又读不出来的僵尸数据。因为无法反推它们原本属于谁，只能删除。

    修复之后新产生的检查点都会带上正确的 user_id，本脚本只需执行一次。

用法：
    cd backend
    python scripts/cleanup_orphan_checkpoints.py          # 预演：只列出待删记录，不改数据库
    python scripts/cleanup_orphan_checkpoints.py --yes    # 确认执行删除

    数据库连接沿用 app/core/database.py 的环境变量：
    POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
"""

import sys
from pathlib import Path

# 让脚本能 import 到 app 下的模块（app 目录是 import 根）
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from core.database import SessionLocal          # noqa: E402
from models.research import ResearchCheckpoint  # noqa: E402


def main() -> int:
    confirmed = "--yes" in sys.argv

    db = SessionLocal()
    try:
        orphans = (
            db.query(ResearchCheckpoint)
            .filter(ResearchCheckpoint.user_id.is_(None))
            .order_by(ResearchCheckpoint.updated_at.desc())
            .all()
        )

        if not orphans:
            print("没有 user_id 为 NULL 的检查点，无需清理。")
            return 0

        print(f"发现 {len(orphans)} 条无归属的检查点：\n")
        print(f"{'session_id':<38} {'status':<10} {'updated_at':<20} query")
        print("-" * 110)
        for cp in orphans:
            updated = cp.updated_at.strftime("%Y-%m-%d %H:%M:%S") if cp.updated_at else "-"
            query_text = (cp.query or "")[:40]
            print(f"{cp.session_id:<38} {(cp.status or '-'):<10} {updated:<20} {query_text}")

        if not confirmed:
            print(f"\n以上 {len(orphans)} 条记录将被删除（含报告全文和搜索结果，不可恢复）。")
            print("当前是预演模式，数据库未做任何改动。确认无误后加 --yes 重新运行：")
            print("    python scripts/cleanup_orphan_checkpoints.py --yes")
            return 0

        deleted = (
            db.query(ResearchCheckpoint)
            .filter(ResearchCheckpoint.user_id.is_(None))
            .delete(synchronize_session=False)
        )
        db.commit()
        print(f"\n已删除 {deleted} 条无归属的检查点。")
        return 0

    except Exception as e:
        db.rollback()
        print(f"清理失败，已回滚，数据库未改动：{e}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
