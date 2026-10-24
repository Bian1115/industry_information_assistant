#!/usr/bin/env python3
"""只重试既有失败run，调用真实恢复路由；保存新证据，不重跑前缀。"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from uuid import UUID
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
from eval.scripts import 评估开发集节点级恢复 as runner


def read(path):
    return json.loads(Path(path).read_text())


def digest(text):
    return hashlib.sha256((text or "").encode()).hexdigest()


async def worker(folder):
    ident = read(folder / "运行身份.json")
    os.environ["EVAL_TRACE_ENABLED"] = "true"
    os.environ["EVAL_TRACE_PAYLOAD_CHARS"] = "10000000"
    os.environ["EVAL_TRACE_LOCAL_FILE"] = str(folder / "节点恢复完整过程埋点.jsonl")
    from eval.scripts.run_deepresearch_bench import benchmark_service, decode_service_event
    from service.deep_research_v2 import service as module
    from core.database import SessionLocal
    from models.research import AgentEvent
    from models.user import User
    from router.research_router import resume_research
    from fastapi import HTTPException, Request

    runner.install_agents(module, ident["case"], "节点恢复", folder)
    await runner.export_case(folder, "重试前")
    started = time.monotonic()
    status = {"mode": "节点恢复", "started_at": runner.now(), "completed": False,
              "errors": [], "entry": "真实resume_research路由及StreamingResponse迭代器",
              "transport": "进程内直接调用；未测试网络或浏览器"}
    runner.save(folder / "节点恢复执行结果.json", status)
    try:
        async with benchmark_service(ident["max_iterations"]) as service:
            runner.hook_saver(service.checkpointer, ident["case"], "节点恢复", folder)
            runner.save(folder / "恢复请求状态.json", await service.read_state(ident["run_id"]))
            request = Request({"type": "http", "app": SimpleNamespace(
                state=SimpleNamespace(research_checkpointer=service.checkpointer))})
            with SessionLocal() as db:
                user = db.query(User).filter_by(id=UUID(ident["user_id"])).one()
                last = db.query(AgentEvent.seq).filter_by(run_id=ident["run_id"]).order_by(
                    AgentEvent.seq.desc()).first()
                cursor = last[0] if last else 0
                status["after_seq"] = cursor
                route_started = time.monotonic()
                response = await resume_research(UUID(ident["run_id"]), request,
                    after_seq=cursor, current_user=user, db=db)
                status.update(http_status=response.status_code,
                              route_setup_seconds=time.monotonic() - route_started)
                runner.save(folder / "节点恢复执行结果.json", status)
                async for chunk in response.body_iterator:
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode()
                    if "data: [DONE]" in chunk:
                        status["done_received"] = True
                    event = decode_service_event(chunk)
                    if event is None:
                        continue
                    runner.append(folder / "节点恢复研究事件.jsonl", {"event": event})
                    if event["type"] == "research_complete":
                        payload = event.get("data") or event
                        status.update(completed=bool(payload.get("final_report")), delivery=payload)
                        (folder / "研究报告.md").write_text(payload.get("final_report") or "")
                    if event["type"] in {"error", "research_paused", "research_cancelled"}:
                        status["errors"].append(event)
    except HTTPException as exc:
        status.update(http_status=exc.status_code, error=exc.detail,
                      error_type="HTTPException")
    except Exception as exc:
        status.update(error=str(exc), error_type=type(exc).__name__)
        traceback.print_exc()
    finally:
        status.update(duration_seconds=time.monotonic() - started, ended_at=runner.now())
        runner.save(folder / "节点恢复执行结果.json", status)
        await runner.export_case(folder, "最终")


def summarize(folder):
    ident, status = read(folder / "运行身份.json"), read(folder / "节点恢复执行结果.json")
    before, after = read(folder / "重试前工作历史.json"), read(folder / "最终工作历史.json")
    seen = {r["source_id"] for r in before}
    added = [r for r in after if r["source_id"] not in seen]
    runner.save(folder / "本次新增工作历史.json", added)
    usage = [dict(r["payload"], source_id=r["source_id"]) for r in added if r["kind"] == "llm_usage"]
    all_usage = [dict(r["payload"], source_id=r["source_id"]) for r in after if r["kind"] == "llm_usage"]
    requested = {(r["call_id"], r["payload"].get("attempt", 1)) for r in added if r["kind"] == "llm_request"}
    accounted = {(r["call_id"], r["payload"].get("attempt", 1)) for r in added if r["kind"] == "llm_usage"}
    state, delivery = read(folder / "最终研究状态.json"), read(folder / "最终交付状态.json")
    events = read(folder / "最终事件账本.json")
    report = state.get("final_report") or ""
    event_hashes = [digest(r["payload"].get("final_report")) for r in events if r["type"] == "research_complete"]
    from core.database import SessionLocal
    from models.chat import ChatMessage
    with SessionLocal() as db:
        messages = db.query(ChatMessage).filter_by(run_id=ident["run_id"], role="assistant").all()
        message_hashes = [digest(m.content) for m in messages]
    journal = folder / "节点与提交记录.jsonl"
    entries = [json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    nodes = [r.get("node") for r in entries if r["kind"] == "node_start"]
    hashes_match = bool(report) and message_hashes == event_hashes == [digest(report)]
    ok = bool(status["completed"] and delivery["run_status"] == "completed"
              and not state["pending_nodes"] and delivery["assistant_messages"] == delivery["complete_events"] == 1
              and hashes_match)
    baseline = read(folder / "历史基线.json")
    row = {"source_id": ident["source_id"], "case": ident["case"], "run_id": ident["run_id"],
           "recovery_completed": ok, "execution": status,
           "first_resumed_node": nodes[0] if nodes else None, "resume_node_calls": dict(Counter(nodes)),
           "resume_cost": runner.costs(usage), "whole_run_cost": runner.costs(all_usage),
           "usage_missing_requests": [{"call_id": c, "attempt": a} for c, a in sorted(requested - accounted)],
           "llm_error_records": sum(r["kind"] == "llm_error" for r in added),
           "baseline_seconds": baseline.get("duration_seconds"),
           "estimated_saved_seconds": baseline["duration_seconds"] - status["duration_seconds"]
               if ok and baseline.get("duration_seconds") is not None else None,
           "delivery": delivery, "report_sha256": digest(report), "message_sha256": message_hashes,
           "complete_event_sha256": event_hashes, "report_delivery_matches_checkpoint": hashes_match,
           "completion_reason": state.get("completion_reason"), "quality_verdict": state.get("quality_verdict"),
           "child_task_return_writes": sum("__return__" in r.get("channels", []) for r in entries)}
    runner.save(folder / "重试评估结果.json", row)
    return row


def write_report(output, report, rows, manifest):
    lines = ["# 开发集节点恢复失败项重试结果及分析", "",
        f"评估日期：{manifest['date']}（Asia/Singapore）。状态：{manifest['status']}。",
        "复评对象：原五题纯节点恢复评估中失败的26、28题；沿用原run、用户、会话和检查点。",
        "", "## 数据与配置口径", "",
        "开发集五题的首轮问题，本次仅恢复26、28题。没有重跑正常基线、成功题或中断前搜索与写作；没有重新注入故障。",
        "沿用原检查点的上下文配置，max_iterations=2，Scout为qwen-plus，其余主Agent为qwen3-max。评测实例关闭Writer章节task，保持纯节点恢复。",
        "此次直接调用当前真实resume_research路由并消费StreamingResponse，包含归属校验、恢复判定、租约、生产服务、PostgreSQL检查点与交付事务；未测试网络传输及浏览器。",
        "原评测器复制的pending_nodes前置判断已落后于当前生产入口，本次不复用该判断；没有在评测器中补交付或改写检查点。当前源码冻结后执行，详见源码哈希及与上轮差异。",
        f"原评估目录：{manifest['source_output']}。本次原始目录：{output}。",
        "", "## 核心指标", "",
        f"已执行{len(rows)}/2项，成功交付{sum(r['recovery_completed'] for r in rows)}项。",
        "", "| 原题 | 结果 | 恢复耗时/秒 | 新增LLM调用 | 新增费用估算/元 | 恢复节点 |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for r in rows:
        cost = r["resume_cost"]
        lines.append(f"| {r['source_id']} | {'成功' if r['recovery_completed'] else '失败'} | {r['execution']['duration_seconds']:.3f} | {cost['known_calls'] + cost['unknown_pricing_calls']} | {cost['estimated_known_cny']:.6f} | {r['resume_node_calls']} |")
    lines += ["", "费用是按2026-10-07已核对的北京实时非思考公开单价及本次新增usage估算；不是实际账单，未折算缓存、免费额度，未计搜索和本地资源。", "", "## 逐题分析与失败样例", ""]
    for r in rows:
        sid, execution = r["source_id"], r["execution"]
        lines += [f"### 原题{sid}：{r['case']}", "",
            f"run_id：`{r['run_id']}`。入口HTTP状态：{execution.get('http_status')}；首个恢复节点：{r['first_resumed_node']}。",
            f"节点调用：{r['resume_node_calls']}。交付：{r['delivery']}。正文、完成事件与检查点哈希一致：{r['report_delivery_matches_checkpoint']}。",
            f"恢复完成原因：{r['completion_reason']}；流程内质量判断：{r['quality_verdict']}。新增usage缺失请求：{len(r['usage_missing_requests'])}，新增LLM错误记录：{r['llm_error_records']}，章节task返回写入：{r['child_task_return_writes']}。"]
        if not r["recovery_completed"]:
            lines.append(f"失败证据：{execution.get('error') or execution.get('errors')}。")
        if sid == 26:
            lines.append("该断点已持久化Writer节点结果，完整后继检查点尚未提交；读取next为空但tasks包含已保存结果。生产入口现在以统一resumable判定交由LangGraph推进，是否实际成功以上述实测交付为准。")
        else:
            lines.append("该断点图状态已completed，报告已在终态检查点，但尚未写入助手报告与research_complete事件。当前入口将业务终态判为不可恢复；若本次仍被拒绝，重复请求本身无法补齐交付，需实现生产终态交付补偿后再评估。")
        if r["estimated_saved_seconds"] is not None:
            lines.append(f"已有完整基线{r['baseline_seconds']:.3f}秒，减去本次恢复{execution['duration_seconds']:.3f}秒，近似节约{r['estimated_saved_seconds']:.3f}秒。历史运行版本/资料/上下文不同，非严格配对性能对照。")
        else:
            lines.append("本次未计算时间节约：恢复未成功或没有已有完整基线。")
        lines += [""]
    lines += ["## 原始结果与可追溯性", "",
        "本次每题保存运行身份、上轮失败结果、原故障触发记录、历史基线、重试前/最终研究状态、完整检查点历史、工作历史及事件账本；本次新增调用单独导出。",
        "汇总见`重试结果汇总.json`；执行清单见`运行清单.json`；逐题见`题目26-检查点提交前/重试评估结果.json`与`题目28-终态交付提交前/重试评估结果.json`；实际SSE与节点记录保留在各题目录。",
        "", "## 局限性、证据边界与结论", "",
        "已测量：上述实际恢复请求、真实模型新增usage、检查点及数据库交付结果。间接推断：历史基线减本次恢复的近似节约；费用按公开单价估算。尚未评估：多次随机重试稳定性、独立报告质量、真实HTTP/浏览器SSE，以及更细粒度恢复。",
        "本次是上轮运行的补充复评，复评日期为2026-10-08；新口径使用当前真实路由而非评测器内过时的pending_nodes判定，原五题结果和失败证据未覆盖。人工等待重试的间隔不计入恢复耗时，进程初始化/证据导出不计入路由恢复耗时。",
        "下一步建议：对仍缺终态交付补偿的断点增加生产恢复分支，使用既有带租约的幂等交付事务；补充提交前失败、事务后事件重放和重复请求测试，再复评原断点。",
        ""]
    report.write_text("\n".join(lines))


def orchestrate(args):
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env", override=False)
    output, report, source = args.output.resolve(), args.report.resolve(), args.source.resolve()
    if output.exists() or report.exists():
        raise ValueError("输出和报告必须使用新路径")
    output.mkdir(parents=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "配置快照.json", output / "配置快照.json")
    os.environ["CONTEXT_ABLATION_CONFIG"] = str(output / "配置快照.json")
    original = read(source / "运行清单.json")
    for role, model in original["models"].items():
        os.environ["LLM_MODEL_" + role.upper()] = model
    os.environ["RESEARCH_MAX_ITERATIONS"] = "2"
    frozen = output / "运行源码快照" / "backend"
    shutil.copytree(BACKEND / "app", frozen / "app", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(BACKEND / "eval", frozen / "eval", ignore=shutil.ignore_patterns("data", "__pycache__", "benchmark"))
    hashes = {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in frozen.rglob("*") if p.is_file()}
    old_hashes = read(source / "源码哈希.json")
    runner.save(output / "源码哈希.json", hashes)
    runner.save(output / "与上轮源码差异.json", {p: {"before": old_hashes.get(p), "after": h}
        for p, h in hashes.items() if old_hashes.get(p) != h})
    manifest = {"date": datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(),
                "status": "running", "started_at": runner.now(), "source_output": str(source),
                "source_ids": [26, 28], "new_prefix_runs": 0, "new_faults": 0,
                "baseline_rerun": False, "recovery_granularity": "node_only",
                "models": original["models"], "context_policy_hash": original["context_policy_hash"],
                "dataset": original["dataset"], "dataset_sha256": original["dataset_sha256"]}
    runner.save(output / "运行清单.json", manifest)
    rows = []
    write_report(output, report, rows, manifest)
    for sid in [26, 28]:
        old = next(source.glob(f"题目{sid}-*"))
        folder = output / old.name
        folder.mkdir()
        for name in ["运行身份.json", "故障触发.json", "历史基线.json"]:
            shutil.copy2(old / name, folder / name)
        shutil.copy2(old / "恢复评估结果.json", folder / "上轮失败结果.json")
        started = time.monotonic()
        script = frozen / "eval/scripts" / Path(__file__).name
        with (folder / "节点恢复进程日志.txt").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", str(script), "--worker", str(folder)],
                cwd=BACKEND, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            runner.save(output / "当前进程.json", {"pid": process.pid, "source_id": sid, "started_at": runner.now()})
            process.wait(timeout=args.timeout)
        runner.save(folder / "节点恢复进程计时.json", {"returncode": process.returncode,
                    "duration_seconds": time.monotonic() - started, "ended_at": runner.now()})
        if not (folder / "最终研究状态.json").exists():
            asyncio.run(runner.export_case(folder, "最终"))
        row = summarize(folder)
        rows.append(row)
        runner.save(output / "重试结果汇总.json", rows)
        write_report(output, report, rows, manifest)
        print(json.dumps({"source_id": sid, "completed": row["recovery_completed"],
                          "seconds": row["execution"]["duration_seconds"], "cost": row["resume_cost"]}, ensure_ascii=False), flush=True)
    manifest.update(status="finished", finished_at=runner.now(),
                    completed=sum(r["recovery_completed"] for r in rows), executed=len(rows))
    runner.save(output / "运行清单.json", manifest)
    write_report(output, report, rows, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args()
    if args.worker:
        asyncio.run(worker(args.worker.resolve()))
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
