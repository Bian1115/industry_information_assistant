#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
评估体系 - 数据库迁移

为什么需要这个脚本：项目用 Base.metadata.create_all() 建表（app_main.py:34），
它只会创建"不存在的表"，不会给"已存在的表"加列。
agent_runs 是新表，create_all 能自动建；
但 research_checkpoints 已经存在，新增的 quality_score 等列必须手动 ALTER。

幂等：全部使用 ADD COLUMN IF NOT EXISTS，重复执行安全。

用法：
    cd backend
    python eval/scripts/migrate_eval_schema.py           # 执行迁移
    python eval/scripts/migrate_eval_schema.py --check   # 只检查不修改
"""

import os
import sys
import argparse

# 让脚本能 import 到 app 下的模块（与 cleanup_orphan_checkpoints.py 保持一致的做法）
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "app"))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text  # noqa: E402
from core.database import engine, Base  # noqa: E402


# research_checkpoints 需要补的列
CHECKPOINT_COLUMNS = [
    ("quality_score", "DOUBLE PRECISION"),
    # 评分来源：valid（模型直接给出）/ valid_after_retry（重试后修正）/
    # fallback_invalid_model_output（模型给了越界值，降级取 1）。
    # 没有这一列就无法把"模型认为很差"和"模型输出格式坏了"区分开，
    # 做质量分析时会把降级分误当真实评分。
    ("quality_score_status", "VARCHAR(48)"),
    ("unresolved_issues", "INTEGER"),
    ("critic_issues", "JSONB"),
]


def get_existing_columns(conn, table_name: str):
    rows = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :t"
        ),
        {"t": table_name},
    )
    return {r[0] for r in rows}


def table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text("SELECT to_regclass(:t)"), {"t": f"public.{table_name}"}
    ).scalar()
    return row is not None


def main():
    parser = argparse.ArgumentParser(description="评估体系数据库迁移")
    parser.add_argument("--check", action="store_true", help="只检查，不修改")
    args = parser.parse_args()

    print("=" * 60)
    print("评估体系数据库迁移")
    print("=" * 60)

    # 1. 建新表（agent_runs）
    #    import 模型以触发注册，再让 create_all 建缺失的表
    from eval.models import AgentRun  # noqa: F401

    with engine.connect() as conn:
        agent_runs_exists = table_exists(conn, "agent_runs")

    if agent_runs_exists:
        print("✅ 表 agent_runs 已存在")
    elif args.check:
        print("⚠️  表 agent_runs 不存在（--check 模式，未创建）")
    else:
        Base.metadata.create_all(bind=engine, tables=[AgentRun.__table__])
        print("✅ 已创建表 agent_runs")

    # 2. 给 research_checkpoints 补列
    with engine.connect() as conn:
        if not table_exists(conn, "research_checkpoints"):
            print("⚠️  表 research_checkpoints 不存在，跳过补列")
            print("    （首次启动后端时 create_all 会自动带上新字段建表）")
            return

        existing = get_existing_columns(conn, "research_checkpoints")
        missing = [(c, t) for c, t in CHECKPOINT_COLUMNS if c not in existing]

        if not missing:
            print("✅ research_checkpoints 的质量字段已齐全")
            return

        print(f"发现 {len(missing)} 个缺失字段: {[c for c, _ in missing]}")

        if args.check:
            print("⚠️  --check 模式，未执行 ALTER")
            return

        for col, col_type in missing:
            conn.execute(
                text(
                    f"ALTER TABLE research_checkpoints "
                    f"ADD COLUMN IF NOT EXISTS {col} {col_type}"
                )
            )
            print(f"  ✅ 已添加 {col} {col_type}")

        conn.commit()

    print("=" * 60)
    print("迁移完成")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        sys.exit(1)
