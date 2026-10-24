#!/usr/bin/env python3
"""多轮开发集预检/压缩点探测；默认仅预检，不调用模型。"""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
from eval.multiturn import Task, load_tasks, run_dialogue


async def collect_records(run_id):
    def read():
        from core.database import SessionLocal
        from models.agent_history import AgentContext, AgentMessage
        with SessionLocal() as db:
            rows = (db.query(AgentMessage, AgentContext).join(AgentContext, AgentMessage.context_id == AgentContext.id)
                    .filter(AgentMessage.run_id == run_id)
                    .order_by(AgentMessage.created_at, AgentMessage.seq).all())
            return [{"source_id":str(m.id), "seq":m.seq, "kind":m.kind, "run_id":m.run_id,
                     "call_id":str(m.call_id), "invocation_id":str(m.invocation_id),
                     "context_key":c.context_key, "agent_name":c.agent_name, "node":m.node,
                     "created_at":m.created_at.isoformat(), "payload":m.payload, "content_hash":m.content_hash}
                    for m, c in rows]
    return await asyncio.to_thread(read)


async def execute(tasks, output, budget, iterations):
    from eval.scripts.run_deepresearch_bench import benchmark_service
    from service.verification_budget import task_budget
    rows = []
    async with benchmark_service(iterations) as service:
        for task in tasks:
            with task_budget(budget) as ledger:
                try:
                    async with asyncio.timeout(budget["max_seconds"]):
                        row = await run_dialogue(service, task, output / f"任务-{task.source.id}",
                                                 timeout_seconds=budget["max_seconds"], collect_records=collect_records)
                except Exception as exc:
                    path = output / f"任务-{task.source.id}" / "逐轮结果.json"
                    turns = json.loads(path.read_text()) if path.exists() else []
                    row = {"task_id":task.task_id, "completed":False, "turns":turns,
                           "error":str(exc), "error_type":type(exc).__name__, "planned_turns":len(task.turns)}
                row["cost"] = ledger.report()
                rows.append(row)
                (output / "探测结果.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, nargs="+", required=True)
    p.add_argument("--mode", choices=["validate", "pilot"], default="validate")
    p.add_argument("--output", type=Path, required=True, help="新目录，不覆盖既有结果")
    p.add_argument("--report", type=Path, required=True, help="新中文Markdown报告路径")
    p.add_argument("--policy", type=Path, help="探测必须显式指定冻结策略")
    p.add_argument("--budget", type=Path, help="整条对话共享的请求/token/时间预算JSON")
    p.add_argument("--max-iterations", type=int, default=2, choices=range(6))
    args = p.parse_args()
    tasks = load_tasks(args.dataset)
    if args.mode == "pilot" and (not args.policy or not args.budget or any(t.split != "development" for t in tasks)):
        p.error("pilot只用于开发集，必须提供--policy和--budget；正式测试使用冻结后的独立执行方案")
    if args.output.exists() or args.report.exists():
        p.error("输出目录或报告已存在，请选择新路径")
    args.output.mkdir(parents=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"date":datetime.now().isoformat(), "mode":args.mode, "task_count":len(tasks),
                "turn_count":sum(len(t.turns) for t in tasks), "executed":False,
                "datasets":{str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in args.dataset},
                "history_mode":"full_prior_dialogue_new_run", "agent_history_continues_across_turns":False,
                "max_iterations":args.max_iterations, "search_mode":"live_search"}
    source_paths = [BACKEND/'eval/multiturn.py', Path(__file__), BACKEND/'app/service/agent_context.py']
    manifest["source_sha256"] = {str(f.relative_to(BACKEND)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths}
    (args.output / "数据结构.json").write_text(json.dumps(Task.model_json_schema(), ensure_ascii=False, indent=2))
    (args.output / "任务快照.json").write_text(json.dumps([t.model_dump() for t in tasks], ensure_ascii=False, indent=2))
    results = []
    try:
        if args.mode == "pilot":
            from dotenv import load_dotenv
            load_dotenv(BACKEND / ".env", override=False)
            from service.context_policy import read_policy
            from service.verification_budget import TaskBudget
            policy = read_policy(args.policy.resolve())
            policy.require_supported()
            budget = json.loads(args.budget.read_text())
            TaskBudget(budget)
            manifest.update(context_policy=policy.snapshot(), budget=budget)
            snapshot = args.output / "配置快照.json"
            snapshot.write_text(json.dumps(policy.snapshot(), ensure_ascii=False, indent=2))
            os.environ["CONTEXT_ABLATION_CONFIG"] = str(snapshot.resolve())
            manifest["executed"] = True
            (args.output / "运行清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            results = asyncio.run(execute(tasks, args.output, budget, args.max_iterations))
            manifest["completed_tasks"] = sum(r["completed"] for r in results)
        manifest["passed"] = args.mode == "validate" or all(r["completed"] and r.get("observation_complete") for r in results)
    except Exception as exc:
        manifest.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    finally:
        (args.output / "运行清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        lines = ["# DeepResearch多轮开发集预检与探测报告", "", f"日期：{manifest['date']}。模式：{args.mode}。",
                 f"原始结果：[{args.output.name}]({args.output.resolve()})。",
                 f"任务数：{len(tasks)}；计划轮数：{manifest['turn_count']}；执行模型：{manifest['executed']}。",
                 f"预检/运行成功：{manifest['passed']}。", "", "## 逐题范围"]
        lines += [f"- 原题{t.source.id}：{t.source.topic}，{len(t.turns)}轮，{len(t.probes)}个检查项；事实标注待固定原文后完成。" for t in tasks]
        lines += ["", "## 触发观察"]
        for row in results:
            turns = row.get("turns", [])
            points = [(r["turn_id"], v) for r in turns for v in r.get("compression_points", [])]
            adopted = [(n, v["agent_name"], v["call_id"]) for n, v in points if v.get("summary_adopted") is True]
            lines.append(f"- {row['task_id']}：完成={row['completed']}，观测完整={row.get('observation_complete', False)}，已记录输入视图={len(points)}，摘要采用位置={adopted}。")
            for turn in turns:
                if turn.get("error") or turn.get("telemetry_status") != "collected":
                    lines.append(f"  第{turn['turn_id']}轮：运行错误={turn.get('error')}；遥测={turn.get('telemetry_status')}，{turn.get('telemetry_error', '')}。")
        lines += ["", "## 失败样例与边界", f"入口错误：{manifest.get('error', '无')}。",
                  "已测量：validate仅验证结构与哈希；pilot仅报告实际记录的调用及压缩事件。没有遥测时不能推断未触发。",
                  "间接推断：多轮增加材料可能提高压缩触发机会，尚不能认定压缩改善任务质量。",
                  "尚未评估：正式20题、人工事实保留率、阈值校准和候选优劣。实时搜索使各次资料不同。",
                  "当前传递先前对话报告给新运行，智能体工作历史与摘要按研究run隔离；质量仍需独立评估。",
                  "下一步：开发集定位首次摘要后，冻结同源快照和人工事实标注，使用同源回放比较候选。"]
        args.report.write_text("\n".join(lines)+"\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
