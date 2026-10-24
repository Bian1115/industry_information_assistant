#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
benchmark 数据源适配层 - 查看与自检

用法：
    cd backend
    python eval/scripts/eval_profile.py                    # 列出所有 profile
    python eval/scripts/eval_profile.py --show crud_rag    # 查看某个 profile 详情
    python eval/scripts/eval_profile.py --verify crud_rag  # 自检：确认切换真的生效
"""

import os
import sys
import json
import argparse

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "app"))

from dotenv import load_dotenv

load_dotenv()

from eval.runtime.profile import (  # noqa: E402
    list_profiles, get_profile, use_profile, active_profile, _config_path,
)


def cmd_list():
    profiles = list_profiles()
    print(f"\n配置文件: {_config_path()}")
    print(f"共 {len(profiles)} 个 profile\n")
    print(f"{'名称':<16}{'benchmark':<14}{'检索集合':<20}{'SQL 库':<8} 说明")
    print("-" * 96)
    for p in profiles:
        sql_flag = "✓" if (p.sql_dsn or p.sql_schema_file) else "-"
        print(
            f"{p.name:<16}{(p.benchmark or '-'):<14}"
            f"{(p.kb_collection or '-'):<20}{sql_flag:<8} {p.description[:40]}"
        )
    print("\n用法: with use_profile('<名称>'): ...   或   EVAL_PROFILE=<名称> python ...\n")


def cmd_show(name: str):
    p = get_profile(name)
    print(f"\n=== Profile: {p.name} ===")
    print(json.dumps(p.to_dict(), ensure_ascii=False, indent=2))

    schema = p.resolved_sql_schema()
    if schema:
        print(f"\n--- SQL schema 描述（前 400 字符）---\n{schema[:400]}")
    elif p.sql_schema_file:
        print(f"\n⚠️  schema 文件不存在: {p.sql_schema_file}")
    print()


def cmd_verify(name: str):
    """
    自检：确认切换真的作用到了业务代码上。

    直接调业务函数 kb_collection_name()，而不是读 profile 字段——
    只有前者能证明适配层确实接进了检索链路。
    """
    from service.retrieval_service import kb_collection_name

    fake_kb_id = "11111111-2222-3333-4444-555555555555"
    print(f"\n自检 profile: {name}")
    print("=" * 60)

    baseline = kb_collection_name(fake_kb_id)
    print(f"1. 切换前（生产）    : kb_collection_name() -> {baseline}")
    print(f"   当前 profile      : {active_profile().name}")

    target = get_profile(name)
    if target.name == "production" and name != "production":
        print(f"\n❌ profile '{name}' 不存在")
        return False

    with use_profile(name) as p:
        switched = kb_collection_name(fake_kb_id)
        print(f"\n2. 切换后（{p.name}）")
        print(f"   当前 profile      : {active_profile().name}")
        print(f"   kb_collection_name() -> {switched}")

        if p.kb_collection:
            ok_kb = switched == p.kb_collection
            print(f"   检索集合切换      : {'✅ 生效' if ok_kb else '❌ 未生效'}")
        else:
            ok_kb = switched == baseline
            print(f"   检索集合          : 该 profile 未配置，应保持原值 "
                  f"{'✅' if ok_kb else '❌'}")

        ok_sql = True
        if p.sql_dsn or p.sql_schema_file:
            from eval.runtime.profile import resolve_sql_dsn, resolve_sql_schema
            dsn = resolve_sql_dsn("postgresql://prod/original")
            schema = resolve_sql_schema("原始 schema")

            ok_dsn = (dsn == p.sql_dsn) if p.sql_dsn else True
            print(f"   SQL DSN           : {dsn}  {'✅' if ok_dsn else '❌'}")

            # schema 声明了文件却读不到时，resolve 会回落到原始 schema。
            # 这种情况必须如实报告，否则会误以为切换成功。
            if p.sql_schema_file:
                if schema != "原始 schema":
                    print(f"   SQL schema        : ✅ 已替换（{len(schema)} 字符）")
                    ok_schema = True
                else:
                    print(f"   SQL schema        : ⚠️  文件不存在，已回落到生产 schema")
                    print(f"                       {p.sql_schema_file}")
                    print(f"                       → 下载 benchmark 数据后补上该文件")
                    ok_schema = False
            else:
                ok_schema = True

            ok_sql = ok_dsn and ok_schema

    restored = kb_collection_name(fake_kb_id)
    ok_restore = restored == baseline
    print(f"\n3. 退出上下文后")
    print(f"   当前 profile      : {active_profile().name}")
    print(f"   kb_collection_name() -> {restored}")
    print(f"   自动还原          : {'✅ 已还原' if ok_restore else '❌ 泄漏了！'}")

    passed = ok_kb and ok_sql and ok_restore
    print("\n" + "=" * 60)
    print("✅ 自检通过：适配层已正确接入，且不会污染生产路径"
          if passed else "❌ 自检未通过，见上方标记")
    return passed


def main():
    parser = argparse.ArgumentParser(description="benchmark 数据源适配层")
    parser.add_argument("--show", metavar="NAME", help="查看 profile 详情")
    parser.add_argument("--verify", metavar="NAME", help="自检切换是否生效")
    args = parser.parse_args()

    if args.show:
        cmd_show(args.show)
    elif args.verify:
        sys.exit(0 if cmd_verify(args.verify) else 1)
    else:
        cmd_list()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)
