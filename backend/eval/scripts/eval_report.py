#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
评估体系 - 追踪数据速查

把 agent_runs 和 research_checkpoints 里沉淀的数据变成能看的东西。
这是评估体系的第一个消费端：没有它，埋点数据就只是躺在表里。

用法：
    cd backend
    python eval/scripts/eval_report.py                 # 总览
    python eval/scripts/eval_report.py --days 3        # 只看最近 3 天
    python eval/scripts/eval_report.py --badcase       # 列出低分/失败的会话
    python eval/scripts/eval_report.py --session <id>  # 翻某次研究的完整调用链
"""

import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta

# 表里的时间列是 naive UTC（全项目统一用 datetime.utcnow），
# 这里必须用同类型的值去比较，因此沿用 utcnow 并屏蔽它的弃用告警，
# 避免每次跑报表都刷一屏 DeprecationWarning。
warnings.filterwarnings("ignore", category=DeprecationWarning, module=__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "app"))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import func, case  # noqa: E402
from core.database import SessionLocal  # noqa: E402
from eval.models import AgentRun  # noqa: E402
from models.research import ResearchCheckpoint  # noqa: E402


def _hr(title: str = ""):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def overview(db, since):
    _hr("一、LLM 调用总览")

    total = db.query(func.count(AgentRun.id)).filter(AgentRun.created_at >= since).scalar() or 0
    if total == 0:
        print("暂无追踪数据。确认 EVAL_TRACE_ENABLED 未被关闭，并且跑过至少一次深度研究。")
        return

    failed = db.query(func.count(AgentRun.id)).filter(
        AgentRun.created_at >= since, AgentRun.success.is_(False)
    ).scalar() or 0
    tokens = db.query(func.sum(AgentRun.total_tokens)).filter(AgentRun.created_at >= since).scalar() or 0
    sessions = db.query(func.count(func.distinct(AgentRun.session_id))).filter(
        AgentRun.created_at >= since
    ).scalar() or 0

    print(f"研究会话数    : {sessions}")
    print(f"LLM 调用次数  : {total}")
    print(f"调用失败      : {failed} ({failed / total * 100:.1f}%)")
    print(f"总 token      : {tokens:,}")
    if sessions:
        print(f"单次研究均耗  : {tokens / sessions:,.0f} token / {total / sessions:.1f} 次调用")

    _hr("二、按 Agent 拆分")
    rows = db.query(
        AgentRun.agent_name,
        AgentRun.model,
        func.count(AgentRun.id),
        func.sum(case((AgentRun.success.is_(False), 1), else_=0)),
        func.avg(AgentRun.duration_ms),
        func.sum(AgentRun.total_tokens),
    ).filter(AgentRun.created_at >= since).group_by(
        AgentRun.agent_name, AgentRun.model
    ).order_by(func.count(AgentRun.id).desc()).all()

    print(f"{'Agent':<16}{'模型':<18}{'调用':>6}{'失败':>6}{'均耗时':>9}{'token':>12}")
    print("-" * 72)
    for name, model, cnt, fail, avg_ms, tok in rows:
        print(
            f"{(name or '-'):<16}{(model or '-'):<18}{cnt:>6}{int(fail or 0):>6}"
            f"{(avg_ms or 0) / 1000:>8.1f}s{int(tok or 0):>12,}"
        )

    _hr("三、JSON 解析质量（本项目最脆弱的一环）")
    rows = db.query(
        AgentRun.agent_name,
        AgentRun.parse_strategy,
        func.count(AgentRun.id),
    ).filter(
        AgentRun.created_at >= since,
        AgentRun.parse_strategy.isnot(None),
    ).group_by(AgentRun.agent_name, AgentRun.parse_strategy).all()

    if not rows:
        print("暂无 JSON 解析记录。")
    else:
        by_agent = {}
        for name, strategy, cnt in rows:
            by_agent.setdefault(name, {})[strategy] = cnt

        print(f"{'Agent':<16}{'direct':>8}{'code_bk':>9}{'braces':>8}{'lit_eval':>10}{'failed':>8}{'降级率':>9}")
        print("-" * 72)
        for name, stats in sorted(by_agent.items()):
            d = stats.get("direct", 0)
            c = stats.get("code_block", 0)
            b = stats.get("braces", 0)
            l = stats.get("literal_eval", 0)
            f = stats.get("failed", 0)
            total_p = d + c + b + l + f
            degraded = (c + b + l + f) / total_p * 100 if total_p else 0
            print(f"{name:<16}{d:>8}{c:>9}{b:>8}{l:>10}{f:>8}{degraded:>8.1f}%")
        print("\n提示：direct 之外的都是兜底逻辑救回来的。降级率上升 = 提示词的 JSON 约束在退化。")

    _hr("四、报告质量评分")
    q = db.query(
        func.count(ResearchCheckpoint.id),
        func.avg(ResearchCheckpoint.quality_score),
        func.min(ResearchCheckpoint.quality_score),
        func.max(ResearchCheckpoint.quality_score),
    ).filter(
        ResearchCheckpoint.updated_at >= since,
        ResearchCheckpoint.quality_score.isnot(None),
    ).first()

    cnt, avg_s, min_s, max_s = q
    if not cnt:
        print("暂无质量评分。跑一次完整研究（走到 CriticMaster 审核）后即可看到。")
    else:
        print(f"已评分会话: {cnt}   平均: {avg_s:.2f}   最低: {min_s:.1f}   最高: {max_s:.1f}")
        dist = db.query(
            case(
                (ResearchCheckpoint.quality_score >= 9, "9-10 优秀"),
                (ResearchCheckpoint.quality_score >= 7, "7-8  通过"),
                (ResearchCheckpoint.quality_score >= 5, "5-6  一般"),
                else_="<5   较差",
            ).label("bucket"),
            func.count(ResearchCheckpoint.id),
        ).filter(
            ResearchCheckpoint.updated_at >= since,
            ResearchCheckpoint.quality_score.isnot(None),
        ).group_by("bucket").all()
        for bucket, n in sorted(dist, reverse=True):
            bar = "█" * int(n / max(cnt, 1) * 30)
            print(f"  {bucket:<12}{n:>4}  {bar}")


def badcase(db, since, limit):
    _hr("Badcase：低分或失败的研究会话")

    rows = db.query(ResearchCheckpoint).filter(
        ResearchCheckpoint.updated_at >= since
    ).filter(
        (ResearchCheckpoint.status == "failed")
        | (ResearchCheckpoint.quality_score < 7)
    ).order_by(ResearchCheckpoint.updated_at.desc()).limit(limit).all()

    if not rows:
        print("没有找到 badcase（没有失败会话，也没有评分低于 7 的会话）。")
        return

    for cp in rows:
        print(f"\n[{cp.updated_at:%Y-%m-%d %H:%M}] session={cp.session_id}")
        print(f"  查询   : {(cp.query or '')[:60]}")
        print(f"  状态   : {cp.status}   评分: {cp.quality_score}   未解决问题: {cp.unresolved_issues}")
        if cp.error_message:
            print(f"  错误   : {cp.error_message[:120]}")
        issues = cp.critic_issues or []
        for issue in issues[:3]:
            if isinstance(issue, dict):
                print(
                    f"    - [{issue.get('severity', '?')}/{issue.get('issue_type', '?')}] "
                    f"{str(issue.get('description', ''))[:80]}"
                )
        print(f"  详查   : python eval/scripts/eval_report.py --session {cp.session_id}")


def session_detail(db, session_id):
    _hr(f"会话调用链: {session_id}")

    cp = db.query(ResearchCheckpoint).filter(
        ResearchCheckpoint.session_id == session_id
    ).first()
    if cp:
        print(f"查询: {cp.query}")
        print(f"状态: {cp.status}   阶段: {cp.phase}   迭代: {cp.iteration}   评分: {cp.quality_score}")

    runs = db.query(AgentRun).filter(
        AgentRun.session_id == session_id
    ).order_by(AgentRun.created_at).all()

    if not runs:
        print("\n该会话没有追踪记录。")
        return

    print(f"\n共 {len(runs)} 次 LLM 调用：")
    print(f"{'#':>3}  {'时间':<8}  {'Agent':<16}{'阶段':<16}{'耗时':>8}{'token':>10}  {'解析':<14}")
    print("-" * 78)
    for i, r in enumerate(runs, 1):
        flag = "  ❌FAILED" if not r.success else ""
        parse = r.parse_strategy or "-"
        if r.parse_ok is False:
            parse = f"!{parse}"
        print(
            f"{i:>3}  {r.created_at:%H:%M:%S}  {(r.agent_name or '-'):<16}"
            f"{(r.phase or '-'):<16}{(r.duration_ms or 0) / 1000:>7.1f}s"
            f"{(r.total_tokens or 0):>10,}  {parse:<14}{flag}"
        )
        if not r.success and r.error_message:
            print(f"      └─ {r.error_type}: {r.error_message[:100]}")


def prefix_summary(db, prefix: str):
    """
    按 session_id 前缀聚合——用于分析一整轮 benchmark。

    run_deepresearch_bench.py 的 session_id 形如
        drb_<模型名>_<运行时间戳>_<题号>
    传前缀 drb_<模型名>_<运行时间戳> 就能精确圈出某一轮，
    不会把重跑的历史数据混进来。
    """
    like = prefix + "%"

    _hr(f"整轮汇总: {prefix}*")

    sessions = db.query(
        func.count(func.distinct(AgentRun.session_id)),
        func.count(AgentRun.id),
        func.sum(AgentRun.total_tokens),
        func.sum(case((AgentRun.success.is_(False), 1), else_=0)),
    ).filter(AgentRun.session_id.like(like)).first()

    n_sess, n_calls, tokens, failed = sessions
    if not n_calls:
        print(f"没有匹配 '{prefix}*' 的追踪记录。")
        return

    print(f"会话数    : {n_sess}")
    print(f"调用次数  : {n_calls}   失败 {int(failed or 0)}")
    print(f"总 token  : {int(tokens or 0):,}")

    _hr("按 Agent 拆分")
    # 取整放在 Python 侧：PostgreSQL 的 round(double precision, int) 不存在，
    # 只有 round(numeric, int)，在 SQL 里做需要显式 CAST，不如直接在这里算
    rows = db.query(
        AgentRun.agent_name,
        func.count(AgentRun.id),
        func.avg(AgentRun.duration_ms),
        func.max(AgentRun.duration_ms),
        func.sum(AgentRun.total_tokens),
        func.sum(case((AgentRun.success.is_(False), 1), else_=0)),
    ).filter(AgentRun.session_id.like(like)).group_by(
        AgentRun.agent_name
    ).order_by(func.count(AgentRun.id).desc()).all()

    print(f"{'Agent':<18}{'调用':>6}{'失败':>6}{'均耗时':>10}{'最长':>10}{'token':>12}")
    print("-" * 66)
    for name, cnt, avg_ms, max_ms, tok, fail in rows:
        print(f"{(name or '-'):<18}{cnt:>6}{int(fail or 0):>6}"
              f"{float(avg_ms or 0) / 1000:>9.1f}s"
              f"{float(max_ms or 0) / 1000:>9.1f}s{int(tok or 0):>12,}")

    _hr("各会话进度")
    rows = db.query(
        AgentRun.session_id,
        func.count(AgentRun.id),
        func.sum(AgentRun.total_tokens),
        func.max(AgentRun.phase),
        func.min(AgentRun.created_at),
        func.max(AgentRun.created_at),
    ).filter(AgentRun.session_id.like(like)).group_by(
        AgentRun.session_id
    ).order_by(func.min(AgentRun.created_at)).all()

    print(f"{'会话':<40}{'调用':>6}{'token':>10}{'用时':>9}  阶段")
    print("-" * 78)
    for sid, cnt, tok, phase, first, last in rows:
        mins = (last - first).total_seconds() / 60 if first and last else 0
        print(f"{sid:<40}{cnt:>6}{int(tok or 0):>10,}{mins:>8.1f}m  {phase or '-'}")


def main():
    parser = argparse.ArgumentParser(description="评估追踪数据速查")
    parser.add_argument("--days", type=int, default=7, help="统计窗口天数，默认 7")
    parser.add_argument("--badcase", action="store_true", help="列出 badcase")
    parser.add_argument("--session", type=str, help="查看指定会话的调用链")
    parser.add_argument("--session-prefix", type=str,
                        help="按前缀聚合一整轮，如 drb_trial-5_20260819_104412")
    parser.add_argument("--limit", type=int, default=10, help="badcase 条数上限")
    args = parser.parse_args()

    since = datetime.utcnow() - timedelta(days=args.days)
    db = SessionLocal()
    try:
        if args.session:
            session_detail(db, args.session)
        elif args.session_prefix:
            prefix_summary(db, args.session_prefix)
        elif args.badcase:
            badcase(db, since, args.limit)
        else:
            overview(db, since)
        print()
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)
