#!/usr/bin/env python3
"""剩余四题统一在Writer节点检查点提交前SIGKILL，使用26题复评冻结源码。"""
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
import signal
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
CASE = "检查点提交前"
IDS = [8, 25, 27, 28]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256((value or "").encode()).hexdigest()


def activate(backend):
    sys.path[:0] = [str(backend), str(backend / "app")]
    from eval.scripts import 评估开发集节点级恢复 as runner
    return runner


async def recover(folder, runner):
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

    runner.install_agents(module, CASE, "节点恢复", folder)
    await runner.export_case(folder, "重试前")
    started = time.monotonic()
    status = {"mode": "节点恢复", "started_at": runner.now(), "completed": False,
              "errors": [], "entry": "26题复评冻结版真实resume_research路由",
              "transport": "进程内StreamingResponse，未测浏览器和网络",
              "lease_wait_seconds": 0, "route_attempts": 0}
    save(folder / "节点恢复执行结果.json", status)
    try:
        async with benchmark_service(2) as service:
            runner.hook_saver(service.checkpointer, CASE, "节点恢复", folder)
            save(folder / "恢复请求状态.json", await service.read_state(ident["run_id"]))
            request = Request({"type": "http", "app": SimpleNamespace(
                state=SimpleNamespace(research_checkpointer=service.checkpointer))})
            with SessionLocal() as db:
                user = db.query(User).filter_by(id=UUID(ident["user_id"])).one()
                last = db.query(AgentEvent.seq).filter_by(run_id=ident["run_id"]).order_by(
                    AgentEvent.seq.desc()).first()
                cursor = last[0] if last else 0
            status["after_seq"] = cursor
            route_started = time.monotonic()
            while True:
                status["route_attempts"] += 1
                try:
                    with SessionLocal() as db:
                        response = await resume_research(UUID(ident["run_id"]), request,
                            after_seq=cursor, current_user=user, db=db)
                    break
                except HTTPException as exc:
                    if exc.status_code != 409 or time.monotonic() - route_started >= 120:
                        raise
                    status["last_lease_conflict"] = str(exc.detail)
                    save(folder / "节点恢复执行结果.json", status)
                    await asyncio.sleep(2)
                    status["lease_wait_seconds"] = time.monotonic() - route_started
            status.update(http_status=response.status_code,
                          route_setup_seconds=time.monotonic() - route_started)
            save(folder / "节点恢复执行结果.json", status)
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
        status.update(http_status=exc.status_code, error=str(exc.detail), error_type="HTTPException")
    except Exception as exc:
        status.update(error=str(exc), error_type=type(exc).__name__)
        traceback.print_exc()
    finally:
        status.update(duration_seconds=time.monotonic() - started, ended_at=runner.now())
        status["wall_duration_seconds"] = (datetime.fromisoformat(status["ended_at"])
            - datetime.fromisoformat(status["started_at"])).total_seconds()
        status["continuation_seconds_excluding_lease_wait"] = max(
            0, status["duration_seconds"] - status["lease_wait_seconds"])
        save(folder / "节点恢复执行结果.json", status)
        await runner.export_case(folder, "最终")


def summarize(folder, runner):
    ident = read(folder / "运行身份.json")
    state = read(folder / "最终研究状态.json")
    delivery = read(folder / "最终交付状态.json")
    status_path = folder / "节点恢复执行结果.json"
    status = read(status_path) if status_path.exists() else {}
    records = read(folder / "最终工作历史.json")
    prefix = read(folder / "中断后工作历史.json")
    prefix_ids = {row["source_id"] for row in prefix}
    added = [row for row in records if row["source_id"] not in prefix_ids]
    save(folder / "恢复新增工作历史.json", added)
    usages = [dict(row["payload"], source_id=row["source_id"]) for row in added if row["kind"] == "llm_usage"]
    all_usages = [dict(row["payload"], source_id=row["source_id"]) for row in records if row["kind"] == "llm_usage"]
    events = read(folder / "最终事件账本.json")
    event_hashes = [digest(row["payload"].get("final_report")) for row in events if row["type"] == "research_complete"]
    from core.database import SessionLocal
    from models.chat import ChatMessage
    with SessionLocal() as db:
        messages = db.query(ChatMessage).filter_by(run_id=ident["run_id"], role="assistant").all()
        message_hashes = [digest(row.content) for row in messages]
    body_hash = digest(state.get("final_report"))
    consistent = bool(state.get("final_report") and message_hashes == event_hashes == [body_hash])
    journal = folder / "节点与提交记录.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    nodes = [row.get("node") for row in entries if row["kind"] == "node_start" and row["mode"] == "节点恢复"]
    fault = read(folder / "故障触发.json") if (folder / "故障触发.json").exists() else None
    ok = bool(fault and status.get("completed") and delivery["run_status"] == "completed"
              and not state["pending_nodes"] and delivery["assistant_messages"] == delivery["complete_events"] == 1
              and consistent)
    baseline = read(folder / "历史基线.json")
    requested = {(row["call_id"], row["payload"].get("attempt", 1)) for row in added if row["kind"] == "llm_request"}
    accounted = {(row["call_id"], row["payload"].get("attempt", 1)) for row in added if row["kind"] == "llm_usage"}
    result = {"source_id": ident["source_id"], "case": CASE, "run_id": ident["run_id"],
              "fault_triggered": bool(fault), "fault": fault, "recovery_completed": ok,
              "recovery_seconds": status.get("duration_seconds"),
              "wall_recovery_seconds": status.get("wall_duration_seconds"),
              "lease_wait_seconds": status.get("lease_wait_seconds"),
              "continuation_seconds_excluding_lease_wait": status.get("continuation_seconds_excluding_lease_wait"),
              "first_resumed_node": nodes[0] if nodes else None,
              "resume_node_calls": dict(Counter(nodes)), "delivery": delivery,
              "report_delivery_matches_checkpoint": consistent, "report_sha256": body_hash,
              "resume_cost": runner.costs(usages), "whole_run_cost": runner.costs(all_usages),
              "usage_missing_requests": [{"call_id": call, "attempt": attempt} for call, attempt in sorted(requested-accounted)],
              "llm_error_records": sum(row["kind"] == "llm_error" for row in added),
              "child_task_return_writes": sum("__return__" in row.get("channels", []) for row in entries),
              "completion_reason": state.get("completion_reason"), "quality_verdict": state.get("quality_verdict"),
              "resume_error": status.get("error") or status.get("errors"),
              "historical_baseline_seconds": baseline.get("duration_seconds"),
              "estimated_saved_seconds": baseline["duration_seconds"]-status["duration_seconds"]
                  if ok and baseline.get("duration_seconds") is not None else None,
              "runtime_budget_final": state.get("research", {}).get("runtime_budget"),
              "search_budget_final": state.get("research", {}).get("search_budget")}
    save(folder / "恢复评估结果.json", result)
    return result


def stats(rows, key):
    values = [row[key] for row in rows if row["recovery_completed"] and row.get(key) is not None]
    return {"successful_samples": len(values), "mean_seconds": statistics.mean(values) if values else None,
            "median_seconds": statistics.median(values) if values else None,
            "min_seconds": min(values) if values else None, "max_seconds": max(values) if values else None}


def write_report(args, manifest, rows, old26):
    planned = manifest["source_ids"]
    mean = stats(rows, "recovery_seconds")
    active = stats(rows, "continuation_seconds_excluding_lease_wait")
    combined = stats(rows + [{"source_id": 26, "recovery_completed": old26["recovery_completed"],
                               "recovery_seconds": old26["execution"]["duration_seconds"]}], "recovery_seconds")
    save(args.output / "耗时统计.json", {"new_four": mean, "new_four_excluding_lease_wait": active,
         "five_including_prior26": combined, "status": manifest["status"],
         "limits": ["26题旧前缀与新四题中断前版本不同，26恢复源码相同；五题为补充汇总", "未新增同版本完整基线", "每题一次，成功样本均值不代替失败率"]})
    def display(value):
        return f"{value:.3f}" if value is not None else "未测得"
    title = "开发集四题写作检查点提交前节点恢复评估结果及分析" if len(planned)==4 else "开发集第28题写作检查点提交前节点恢复补充复评结果及分析"
    lines = ["# " + title, "",
        f"评估日期：{manifest['date']}，Asia/Singapore。状态：{manifest['status']}。",
        f"评估对象：开发集{planned}题统一在Writer节点后继检查点提交前中断并恢复。", "",
        "## 数据与配置口径", "",
        f"只使用开发集首轮，{len(planned)}题独立新评测身份/会话/run，真实模型、真实搜索和PostgreSQL节点检查点。",
        "中断边界与26题相同：Writer节点结果的pending writes已保存、后继aput执行前SIGKILL。关闭章节task，仅节点级恢复；不伪造检查点、不补救交付。",
        "复用26题成功复评的完整冻结应用源码；中断前与恢复采用同一副本。Scout为qwen-plus，其余主Agent及摘要为qwen3-max；max_iterations=2，搜索总上限20、研究运行总预算2400秒、累计token上限0（不限制）。",
        "沿用原上下文策略配置及输出上限，阶段时间预算保持原默认值。所有有效配置写入配置快照和有效运行配置。预算上限相同不要求实际模型调用数、搜索结果或审核返工次数相同。",
        "恢复计时从读取检查点后发起恢复开始，包含服务初始化、租约等待和最终交付；进程导入、证据导出及人为间隔不计入。额外记录剔除409租约冲突等待的续跑时间，未修改或抢占数据库租约。",
        "已有26题为补充参考，保留其原始结果。未新增完整正常基线，旧完整耗时仍为历史对照，不能称为同版本配对的平均省时。", "",
        "## 原始结果路径", "",
        f"- 本次原始目录：`{args.output}`。",
        f"- 冻结版本来源：`{args.source}`。",
        "- 本次逐题结果：`逐题恢复结果.json`；均值：`耗时统计.json`；配置：`配置快照.json`、`有效运行配置.json`；源码：`源码哈希.json`。",
        "- 各题包含运行身份、故障触发、节点提交日志、中断后/最终完整检查点、事件账本、工作历史与模型usage。",
        "- 26题参考：`backend/eval/data/开发集节点恢复失败项重试-20261008-185800/题目26-检查点提交前/重试评估结果.json`。", "",
        "## 核心指标", "",
        f"本次完成{len(rows)}/{len(planned)}项；触发指定故障{sum(row['fault_triggered'] for row in rows)}项，成功交付{sum(row['recovery_completed'] for row in rows)}项。",
        f"本次成功样本恢复均值：{display(mean['mean_seconds'])}秒；中位数：{display(mean['median_seconds'])}秒；样本数：{mean['successful_samples']}。",
        f"剔除租约等待后续跑均值：{display(active['mean_seconds'])}秒。",
        f"包含已有26题的补充汇总均值：{display(combined['mean_seconds'])}秒；成功样本数：{combined['successful_samples']}，不作为新同版本五题统一实验。", "",
        "| 原题 | 指定故障 | 恢复交付 | 含等待/秒 | 租约等待/秒 | 剔除等待/秒 | 恢复节点 | 正文一致 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['source_id']} | {row['fault_triggered']} | {row['recovery_completed']} | {display(row['recovery_seconds'])} | {display(row['lease_wait_seconds'])} | {display(row['continuation_seconds_excluding_lease_wait'])} | {row['resume_node_calls']} | {row['report_delivery_matches_checkpoint']} |")
    lines += ["", "## 逐题分析、失败样例与调用消耗", ""]
    for row in rows:
        lines += [f"### 原题{row['source_id']}", "",
            f"run_id：`{row['run_id']}`；首个恢复节点：{row['first_resumed_node']}；完成原因：{row['completion_reason']}；质量判断：{row['quality_verdict']}。",
            f"恢复阶段已记录模型usage：{row['resume_cost']['known_calls'] + row['resume_cost']['unknown_pricing_calls']}次，输入token {row['resume_cost']['input_tokens']}，输出token {row['resume_cost']['output_tokens']}；缺失usage请求{len(row['usage_missing_requests'])}项。",
            f"章节task返回写入{row['child_task_return_writes']}次；正文与检查点、报告消息及完成事件一致：{row['report_delivery_matches_checkpoint']}。",
            f"失败/错误证据：{row['resume_error'] or '当前未记录恢复错误'}。",
            f"旧完整耗时：{display(row['historical_baseline_seconds'])}秒；历史估计时间差：{display(row['estimated_saved_seconds'])}秒。", ""]
    lines += ["## 局限性、结论与下一步", "",
        "已测量：表中已完成题目的实际故障、中断后检查点、恢复路由、模型usage、恢复时间与最终交付一致性。运行中未完成样本不得记作成功。",
        "间接推断：历史完整耗时减恢复耗时的差值；表内费用计算函数沿用2026-10-07价格快照，费用不是实际账单，也不用于宣称成本降低。",
        "尚未评估：新同版本完整正常基线、严格配对的平均节约、多次故障重复稳定性、独立报告事实质量、HTTP和浏览器性能。",
        "结论以成功样本均值与明确失败数同时呈现；若恢复阶段再次执行搜索/分析/写作，需区分初稿重跑与审核返工。",
        "下一步如需简历平均省时指标，应补同版本完整基线或预先设计完整执行/故障恢复的配对重复实验。", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))


def orchestrate(args):
    repo_backend = BACKEND
    from dotenv import load_dotenv
    load_dotenv(repo_backend / ".env", override=False)
    for attr in ("output", "report", "source", "original"):
        setattr(args, attr, getattr(args, attr).resolve())
    if args.output.exists() or args.report.exists():
        raise ValueError("只允许新增运行目录和报告")
    original_manifest = read(args.original / "运行清单.json")
    dataset = Path(original_manifest["dataset"])
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == original_manifest["dataset_sha256"]
    tasks = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    assert [task["source"]["id"] for task in tasks] == [8, 25, 26, 27, 28]
    args.output.mkdir(parents=True)
    frozen = args.output / "运行源码快照" / "backend"
    shutil.copytree(args.source / "运行源码快照/backend", frozen, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__), frozen / "eval/scripts" / Path(__file__).name)
    shutil.copy2(args.source / "配置快照.json", args.output / "配置快照.json")
    os.environ["CONTEXT_ABLATION_CONFIG"] = str(args.output / "配置快照.json")
    for role, model in original_manifest["models"].items():
        os.environ["LLM_MODEL_" + role.upper()] = model
    overrides = {"LLM_MODEL_SUMMARY": "qwen3-max", "RESEARCH_MAX_ITERATIONS": "2",
        "RESEARCH_MAX_TOTAL_SEARCHES": "20", "RESEARCH_SEARCH_PROVIDER": "bocha",
        "RESEARCH_TOTAL_BUDGET_SECONDS": "2400", "RESEARCH_MAX_LLM_TOKENS": "0",
        "RESEARCH_ALLOWED_DOMAINS": "", "RESEARCH_HYPOTHESIS_ENABLED": "false",
        "SESSION_RESEARCH_INHERITANCE_ENABLED": "1", "SESSION_RESEARCH_CONTEXT_TOKENS": "3000"}
    for name, seconds in {"CHIEFARCHITECT":180, "DEEPSCOUT":900, "DATAANALYST":300,
                           "CODEWIZARD":180, "LEADWRITER":600, "CRITICMASTER":300}.items():
        overrides["RESEARCH_BUDGET_" + name + "_SECONDS"] = str(seconds)
    os.environ.update(overrides)
    runner = activate(frozen)
    from config.llm_config import get_config
    from service.context_policy import selected_policy
    from service.deep_research_v2.runtime_budget import DEFAULT_PHASE_SECONDS, phase_budget_seconds, total_budget_seconds, max_llm_tokens
    from service.deep_research_v2.followup_context import SessionInheritanceSettings
    from eval.scripts.采集真实追问开发集 import create_identity, create_session, save_user
    from core.database import SessionLocal
    from sqlalchemy import text
    with SessionLocal() as db:
        assert db.execute(text("select 1")).scalar() == 1
    config = get_config()
    assert {role:getattr(config.agents,role).model for role in original_manifest["models"]} == original_manifest["models"]
    assert selected_policy().snapshot()["config_hash"] == original_manifest["context_policy_hash"]
    assert config.research.max_iterations == 2 and config.research.max_total_searches == 20
    inheritance = SessionInheritanceSettings.from_env()
    save(args.output / "有效运行配置.json", {"models":{role:getattr(config.agents,role).to_dict() for role in original_manifest["models"]},
        "summary_model":"qwen3-max", "research":vars(config.research), "search_provider":config.search_provider,
        "runtime_budget":{"total_limit_seconds":total_budget_seconds(),"token_limit":max_llm_tokens(),
             "phases":{name:phase_budget_seconds(name) for name in DEFAULT_PHASE_SECONDS}},
        "context_policy_hash":selected_policy().snapshot()["config_hash"],
        "session_inheritance":vars(inheritance), "explicit_non_secret_env_overrides":overrides})
    app_hashes = {str(path.relative_to(frozen)):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (frozen / "app").rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts}
    source_hashes = read(args.source / "源码哈希.json")
    assert all(source_hashes.get(path) == value for path,value in app_hashes.items())
    save(args.output / "源码哈希.json", app_hashes)
    old26 = read(args.source / "题目26-检查点提交前/重试评估结果.json")
    save(args.output / "已有26题参考.json", {"source":str(args.source),"recovery_seconds":old26["execution"]["duration_seconds"],
        "recovery_completed":old26["recovery_completed"],"limits":"旧前缀版本不同；恢复应用源码相同"})
    manifest = {"date":datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(), "status":"running",
        "started_at":runner.now(), "source_ids":args.source_ids, "case":CASE, "source_snapshot":str(args.source),
        "models":original_manifest["models"],"context_policy_hash":original_manifest["context_policy_hash"],
        "dataset":str(dataset),"dataset_sha256":original_manifest["dataset_sha256"],
        "baseline_rerun":False,"recovery_granularity":"node_only","writer_section_tasks_disabled":True,
        "production_app_hashes_match26_retry":True, "timeout_seconds_per_attempt":args.timeout}
    rows=[]
    save(args.output / "运行清单.json", manifest)
    write_report(args,manifest,rows,old26)
    owner = create_identity(args.output.name)
    script = frozen / "eval/scripts" / Path(__file__).name
    try:
        for task in tasks:
            sid = task["source"]["id"]
            if sid not in args.source_ids:
                continue
            folder = args.output / f"题目{sid}-检查点提交前"
            folder.mkdir()
            old_folder = next(args.original.glob(f"题目{sid}-*"))
            for name in ["历史基线.json","历史基线模型调用.json","历史基线事件.json"]:
                if (old_folder/name).is_file():shutil.copy2(old_folder/name,folder/name)
            identity = {"source_id":sid,"case":CASE,"query":task["turns"][0]["user_query"],
                "run_id":str(uuid4()),"turn_uuid":str(uuid4()),"user_id":owner,
                "session_id":create_session(owner,f"同位置节点恢复 原题{sid}"),"max_iterations":2}
            save_user(owner,identity["session_id"],identity["turn_uuid"],identity["run_id"],identity["query"])
            save(folder / "运行身份.json", identity)
            print(json.dumps({"event":"case_start","source_id":sid,"at":runner.now()},ensure_ascii=False),flush=True)
            for mode in ["中断前","节点恢复"]:
                if mode=="节点恢复" and not (folder / "故障触发.json").exists():break
                started=time.monotonic()
                with (folder / (mode+"进程日志.txt")).open("w") as log:
                    process=subprocess.Popen([sys.executable,"-u",str(script),"--worker","--folder",str(folder),"--mode",mode],
                        cwd=repo_backend,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
                    save(args.output / "当前进程.json", {"pid":process.pid,"source_id":sid,"mode":mode,"at":runner.now()})
                    try:process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                        save(folder / (mode+"外部超时.json"),{"seconds":args.timeout,"at":runner.now()})
                save(folder / (mode+"进程计时.json"),{"returncode":process.returncode,"duration_seconds":time.monotonic()-started})
                if mode=="中断前":asyncio.run(runner.export_case(folder,"中断后"))
            asyncio.run(runner.export_case(folder,"最终"))
            row=summarize(folder,runner)
            rows.append(row)
            save(args.output / "逐题恢复结果.json",rows)
            write_report(args,manifest,rows,old26)
            print(json.dumps({"event":"case_end","source_id":sid,"fault_triggered":row["fault_triggered"],
                "recovery_completed":row["recovery_completed"],"recovery_seconds":row["recovery_seconds"]},ensure_ascii=False),flush=True)
        manifest.update(status="finished",finished_at=runner.now())
    except BaseException as exc:
        manifest.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        manifest.update(evaluated_cases=len(rows),fault_triggered=sum(row['fault_triggered'] for row in rows),
                        recovery_completed=sum(row['recovery_completed'] for row in rows))
        save(args.output / "运行清单.json",manifest)
        write_report(args,manifest,rows,old26)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--original",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--timeout",type=int,default=5400)
    parser.add_argument("--source-ids",type=int,nargs="+",default=IDS)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--folder",type=Path)
    parser.add_argument("--mode",choices=["中断前","节点恢复"])
    args=parser.parse_args()
    if not set(args.source_ids)<=set(IDS):parser.error("source-ids只能选8、25、27、28")
    if args.worker:
        runner=activate(BACKEND)
        if args.mode=="中断前":asyncio.run(runner.worker(args))
        else:asyncio.run(recover(args.folder,runner))
    else:
        if not all([args.source,args.original,args.output,args.report]):parser.error("需要source、original、output和report")
        orchestrate(args)


if __name__=="__main__":
    main()
