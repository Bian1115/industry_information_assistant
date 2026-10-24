#!/usr/bin/env python3
"""五题各一次硬中断与原run节点级恢复；复用旧账本，不执行正常基线。"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import types
import uuid

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
CASES = [(8, "部分搜索"), (25, "第三章完成"), (26, "检查点提交前"),
         (27, "检查点提交后"), (28, "终态交付提交前")]
PRICE_SOURCE = "https://help.aliyun.com/zh/model-studio/model-pricing"
# 2026-10-07经官方页和qianwen models info核对，北京实时非思考原价，不扣缓存。
RATES = {"qwen3-max": [(32000, 2.5, 10), (128000, 4, 16), (256000, 7, 28)],
         "qwen-plus": [(128000, .8, 2), (256000, 2.4, 20), (1000000, 4.8, 48)]}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"at": now(), "monotonic": time.monotonic(), **value},
                           ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def price(model, usage):
    a, b = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if a is None or b is None:
        return None
    key = next((k for k in RATES if model == k or model.startswith(k + "-")), None)
    for upper, pin, pout in RATES.get(key, []):
        if a <= upper:
            return (a * pin + b * pout) / 1000000
    return None


def costs(rows):
    total, unknown, inp, out = 0.0, 0, 0, 0
    for row in rows:
        usage = row.get("usage") or row
        inp += usage.get("prompt_tokens") or 0
        out += usage.get("completion_tokens") or 0
        value = price(row.get("model", ""), usage)
        if value is None:
            unknown += 1
        else:
            total += value
    return {"known_calls": len(rows) - unknown, "unknown_pricing_calls": unknown,
            "input_tokens": inp, "output_tokens": out, "estimated_known_cny": round(total, 6),
            "basis": "北京实时非思考公开原价；缓存未折算；搜索/本地资源不计入"}


def crash(folder, case, detail):
    marker = {"case": case, "at": now(), "monotonic": time.monotonic(),
              "pid": os.getpid(), "signal": "SIGKILL", **detail}
    save(folder / "故障触发.json", marker)
    os.kill(os.getpid(), signal.SIGKILL)


def is_write_checkpoint(checkpoint, versions):
    work = (checkpoint.get("channel_values") or {}).get("research") or {}
    attempts = work.get("node_attempts") or {}
    return bool("research" in versions and work.get("final_report")
                and attempts.get("write") and not attempts.get("review"))


def hook_saver(saver, case, mode, folder):
    original_put, original_writes = saver.aput, saver.aput_writes

    async def aput(self, config, checkpoint, metadata, versions):
        target = mode == "中断前" and is_write_checkpoint(checkpoint, versions)
        detail = {"checkpoint_id": checkpoint["id"], "metadata": metadata,
                  "phase": (checkpoint["channel_values"].get("research") or {}).get("phase")}
        if target and case == "检查点提交前":
            crash(folder, case, {"boundary": "aput之前", **detail})
        result = await original_put(config, checkpoint, metadata, versions)
        append(folder / "节点与提交记录.jsonl", {"kind": "checkpoint_committed",
                    "mode": mode, **detail})
        if target and case == "检查点提交后":
            crash(folder, case, {"boundary": "aput返回之后", **detail})
        return result

    async def writes(self, config, values, task_id, task_path=""):
        result = await original_writes(config, values, task_id, task_path)
        append(folder / "节点与提交记录.jsonl", {"kind": "pending_writes_committed",
                    "mode": mode, "checkpoint_id": config["configurable"].get("checkpoint_id"),
                    "task_id": task_id, "channels": [x[0] for x in values]})
        return result

    saver.aput = types.MethodType(aput, saver)
    saver.aput_writes = types.MethodType(writes, saver)


def install_agents(module, case, mode, folder):
    factory = module.create_agents

    def create(config):
        agents = factory(config)
        for role, agent in agents.items():
            original = agent.process

            async def process(state, original=original, role=role):
                append(folder / "节点与提交记录.jsonl", {"kind": "node_start", "mode": mode,
                       "role": role, "node": state.get("_node"), "phase": state.get("phase"),
                       "attempt": state.get("_attempt")})
                value = await original(state)
                append(folder / "节点与提交记录.jsonl", {"kind": "node_return", "mode": mode,
                       "role": role, "node": state.get("_node")})
                return value
            agent.process = process

        scout = agents["scout"]
        search = scout._execute_search_once
        async def searched(query, count=10, *, state=None):
            value = await search(query, count, state=state)
            if mode == "中断前" and case == "部分搜索" and value:
                crash(folder, case, {"boundary": "一次搜索成功返回后",
                                     "node": (state or {}).get("_node"),
                                     "results_count": len(value),
                                     "queries_used": ((state or {}).get("search_budget") or {}).get("used")})
            return value
        scout._execute_search_once = searched

        writer = agents["writer"]
        write = writer._write_section
        completed = []
        async def written(state, section):
            result = await write(state, section)
            if section.get("status") == "drafted" and section["id"] not in completed:
                completed.append(section["id"])
            if mode == "中断前" and case == "第三章完成" and len(completed) == 3:
                crash(folder, case, {"boundary": "第三个新章节应用后、第四章之前",
                       "node": state.get("_node"), "section_ids": completed,
                       "draft_sections": state.get("draft_sections"),
                       "section_claims": state.get("section_claims")})
            return result
        writer._write_section = written
        # 当前部署可能包含章节级 task。本轮只测节点级恢复，评测实例
        # 沿用同一章节生成方法，并绕过章节产物 task，不修改生产实现。
        writer._write_checkpointed_section = written
        return agents
    module.create_agents = create

    commit_factory = module.make_commit_result
    def commit_factory_hook(**kwargs):
        commit = commit_factory(**kwargs)
        def invoke(snapshot):
            if mode == "中断前" and case == "终态交付提交前":
                crash(folder, case, {"boundary": "终态检查点已保存、commit_result调用之前",
                      "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
                      "pending_nodes": list(snapshot.next)})
            append(folder / "节点与提交记录.jsonl",
                   {"kind": "delivery_start", "mode": mode})
            result = commit(snapshot)
            append(folder / "节点与提交记录.jsonl",
                   {"kind": "delivery_committed", "mode": mode})
            return result
        return invoke
    module.make_commit_result = commit_factory_hook


async def worker(args):
    folder = args.folder.resolve()
    identity = json.loads((folder / "运行身份.json").read_text())
    mode = args.mode
    os.environ["EVAL_TRACE_ENABLED"] = "true"
    os.environ["EVAL_TRACE_PAYLOAD_CHARS"] = "10000000"
    os.environ["EVAL_TRACE_LOCAL_FILE"] = str(folder / (mode + "完整过程埋点.jsonl"))
    from eval.scripts.run_deepresearch_bench import benchmark_service, decode_service_event
    from service.deep_research_v2 import service as module
    from service.research_events import acquire_research_run_lease, ResearchRunLeaseConflict
    started = time.monotonic()
    status = {"mode": mode, "started_at": now(), "started_monotonic": started,
              "completed": False, "errors": []}
    save(folder / (mode + "执行结果.json"), status)
    install_agents(module, identity["case"], mode, folder)
    async with benchmark_service(identity["max_iterations"]) as service:
        hook_saver(service.checkpointer, identity["case"], mode, folder)
        if mode == "节点恢复":
            before = await service.read_state(identity["run_id"])
            save(folder / "恢复请求状态.json", before)
            # 与生产路由相同的前置拒绝，不能用评测器补交付来隐藏失败。
            if not before["exists"] or not before["pending_nodes"]:
                status.update(http_status=400, error_code="graph_finished_without_delivery",
                              error="图已执行结束，请直接读取研究结果",
                              duration_seconds=time.monotonic() - started)
                save(folder / (mode + "执行结果.json"), status)
                return
        lease_started = time.monotonic()
        while True:
            try:
                lease = await asyncio.to_thread(acquire_research_run_lease,
                    run_id=identity["run_id"], session_id=identity["session_id"],
                    user_id=identity["user_id"])
                break
            except ResearchRunLeaseConflict:
                if time.monotonic() - lease_started > 120:
                    raise
                await asyncio.sleep(2)
        status["lease_wait_seconds"] = time.monotonic() - lease_started
        save(folder / (mode + "执行结果.json"), status)
        async for chunk in service.research(query=identity["query"],
            session_id=identity["session_id"], user_id=identity["user_id"],
            run_id=identity["run_id"], turn_id=identity["turn_uuid"],
            resume=mode == "节点恢复", lease=lease,
            research_context={"original_query": identity["query"],
                              "resolved_query": identity["query"], "is_follow_up": False,
                              "conversation_context": [], "memory_context": []}):
            event = decode_service_event(chunk)
            if event is None:
                continue
            append(folder / (mode + "研究事件.jsonl"), {"event": event})
            if event["type"] == "research_complete":
                data = event.get("data") or event
                status.update(completed=bool(data.get("final_report")), delivery=data)
                (folder / "研究报告.md").write_text(data.get("final_report") or "")
            if event["type"] in {"error", "research_paused", "research_cancelled"}:
                status["errors"].append(event)
        status["duration_seconds"] = time.monotonic() - started
        status["ended_at"] = now()
        save(folder / (mode + "执行结果.json"), status)


def historical(task, folder):
    from core.database import SessionLocal
    from models.research import AgentEvent
    from eval.models import AgentRun
    query = task["turns"][0]["user_query"]
    with SessionLocal() as db:
        starts = db.query(AgentEvent).filter(AgentEvent.event_type == "research_start",
            AgentEvent.payload["query"].astext == query).order_by(AgentEvent.occurred_at.desc()).all()
        chosen, end = None, None
        for row in starts:
            end = db.query(AgentEvent).filter(AgentEvent.run_id == row.run_id,
                AgentEvent.event_type == "research_complete").order_by(AgentEvent.seq).first()
            if end is not None:
                chosen = row
                break
        calls = []
        if chosen is not None:
            calls = db.query(AgentRun).filter(AgentRun.session_id == chosen.session_id,
                AgentRun.query == query, AgentRun.created_at >= chosen.occurred_at,
                AgentRun.created_at <= end.occurred_at).order_by(AgentRun.created_at).all()
            events = db.query(AgentEvent).filter(AgentEvent.run_id == chosen.run_id).order_by(AgentEvent.seq).all()
            save(folder / "历史基线事件.json", [{"seq": r.seq, "type": r.event_type,
                 "at": r.occurred_at, "payload": r.payload} for r in events])
            baseline = {"run_id": chosen.run_id, "session_id": chosen.session_id,
                        "historical_completed": True,
                        "duration_seconds": (end.occurred_at - chosen.occurred_at).total_seconds(),
                        "start": chosen.occurred_at, "end": end.occurred_at,
                        "basis": "已有research_start至research_complete事件区间；不含请求前上下文构建",
                        "historical_models": sorted({r.model for r in calls}),
                        "report_sha256": hashlib.sha256(str((end.payload or {}).get("final_report", "")).encode()).hexdigest()}
        else:
            calls = db.query(AgentRun).filter(AgentRun.query == query).order_by(AgentRun.created_at.desc()).all()
            if calls:
                sid = calls[0].session_id
                calls = sorted([r for r in calls if r.session_id == sid], key=lambda r: r.created_at)
            baseline = {"duration_seconds": None, "historical_completed": False,
                        "basis": "未找到完整开始/完成事件，费用仅为旧失败运行的已知消耗，不能充当完整基线"}
            if starts:
                failed_start = starts[0]
                failed_events = db.query(AgentEvent).filter_by(run_id=failed_start.run_id).order_by(AgentEvent.seq).all()
                save(folder / "历史未完成轨迹事件.json", [{"seq": r.seq, "type": r.event_type,
                     "at": r.occurred_at, "payload": r.payload} for r in failed_events])
                baseline.update(run_id=failed_start.run_id, session_id=failed_start.session_id,
                                terminal_errors=[r.payload for r in failed_events if r.event_type == "error"])
        raw = [r.to_dict(include_payload=True) for r in calls]
        save(folder / "历史基线模型调用.json", raw)
        baseline["cost_estimate"] = costs(raw)
        baseline["llm_calls"] = len(raw)
        save(folder / "历史基线.json", baseline)
        return baseline


async def export_case(folder, suffix):
    from eval.scripts.run_deepresearch_bench import benchmark_service
    from eval.scripts.run_multiturn_deepresearch import collect_records
    from core.database import SessionLocal
    from models.research import AgentEvent, ResearchRun
    from models.chat import ChatMessage
    ident = json.loads((folder / "运行身份.json").read_text())
    async with benchmark_service(ident["max_iterations"]) as service:
        snapshot = await service.read_state(ident["run_id"])
        save(folder / (suffix + "研究状态.json"), snapshot)
        history = []
        async for cp in service.checkpointer.alist({"configurable": {"thread_id": ident["run_id"]}}):
            history.append({"config": cp.config, "checkpoint": cp.checkpoint,
                            "metadata": cp.metadata, "parent_config": cp.parent_config,
                            "pending_writes": cp.pending_writes})
        save(folder / (suffix + "完整检查点历史.json"), history)
    records = await collect_records(ident["run_id"])
    save(folder / (suffix + "工作历史.json"), records)
    with SessionLocal() as db:
        run = db.query(ResearchRun).filter_by(run_id=ident["run_id"]).one()
        events = db.query(AgentEvent).filter_by(run_id=ident["run_id"]).order_by(AgentEvent.seq).all()
        assistants = db.query(ChatMessage).filter_by(run_id=ident["run_id"], role="assistant").all()
        delivery = {"run_status": run.status, "final_checkpoint_id": run.final_checkpoint_id,
                    "delivered_event_id": str(run.delivered_event_id) if run.delivered_event_id else None,
                    "assistant_messages": len(assistants),
                    "complete_events": sum(r.event_type == "research_complete" for r in events),
                    "fencing_token": run.fencing_token}
        save(folder / (suffix + "事件账本.json"), [{"seq": r.seq, "type": r.event_type,
                                                  "at": r.occurred_at, "payload": r.payload} for r in events])
        save(folder / (suffix + "交付状态.json"), delivery)
    return snapshot, records, delivery


def summarize_case(folder, baseline):
    ident = json.loads((folder / "运行身份.json").read_text())
    fault_path = folder / "故障触发.json"
    fault = json.loads(fault_path.read_text()) if fault_path.exists() else None
    resumed_path = folder / "节点恢复执行结果.json"
    resumed = json.loads(resumed_path.read_text()) if resumed_path.exists() else {}
    records = json.loads((folder / "最终工作历史.json").read_text())
    prefix_ids = {r["source_id"] for r in json.loads((folder / "中断后工作历史.json").read_text())} \
        if (folder / "中断后工作历史.json").exists() else set()
    usage = [dict(r["payload"], source_id=r["source_id"]) for r in records if r["kind"] == "llm_usage"]
    post_usage = [r for r in usage if r["source_id"] not in prefix_ids]
    delivery = json.loads((folder / "最终交付状态.json").read_text())
    state = json.loads((folder / "最终研究状态.json").read_text())
    nodes_path = folder / "节点与提交记录.jsonl"
    node_rows = [json.loads(x) for x in nodes_path.read_text().splitlines()] if nodes_path.exists() else []
    pre = Counter(r["node"] for r in node_rows if r["kind"] == "node_start" and r["mode"] == "中断前")
    post = Counter(r["node"] for r in node_rows if r["kind"] == "node_start" and r["mode"] == "节点恢复")
    rtime = resumed.get("duration_seconds")
    btime = baseline.get("duration_seconds")
    ok = bool(fault and resumed.get("completed") and not state["pending_nodes"]
              and delivery["assistant_messages"] == delivery["complete_events"] == 1)
    row = {"source_id": ident["source_id"], "case": ident["case"], "run_id": ident["run_id"],
           "fault_triggered": bool(fault), "recovery_completed": ok,
           "fault": fault, "resume_error": resumed.get("error") or resumed.get("errors"),
           "baseline_seconds": btime, "resume_seconds": rtime,
           "lease_wait_seconds": resumed.get("lease_wait_seconds"),
           "estimated_saved_seconds": btime - rtime if ok and btime is not None and rtime is not None else None,
           "baseline_cost": baseline["cost_estimate"], "whole_attempts_cost": costs(usage),
           "historical_completed": baseline["historical_completed"],
           "resume_cost": costs(post_usage), "prefix_node_calls": dict(pre), "resume_node_calls": dict(post),
           "pending_nodes_final": state["pending_nodes"], "delivery": delivery,
           "cost_unknown_error_records": sum(r["kind"] == "llm_error" for r in records),
           "limits": ["历史轨迹与当前配置/上下文/搜索可能不同，收益为近似估计",
                      "费用按非思考公开原价估计，不是实际账单，不含搜索与本地资源",
                      "每题每场景仅一次，未测总体稳定成功率"]}
    row["estimated_saved_cny"] = round(baseline["cost_estimate"]["estimated_known_cny"]
                                     - row["resume_cost"]["estimated_known_cny"], 6) if ok and baseline["historical_completed"] else None
    save(folder / "恢复评估结果.json", row)
    return row


def write_report(output, report, rows, status):
    lines = ["# 开发集五题节点级硬中断恢复评估结果及分析", "",
             "评估日期：2026-10-07（Asia/Singapore）。状态：" + status + "。",
             "", "## 对象、数据与配置",
             "开发集原题8、25、26、27、28的首轮中文问题；每题独立新会话，各分配一个故障点。"
             "复用数据库已有轨迹作基线，没有新跑正常完整对照，也没有跑30轮。",
             "沿用启动时部署模型、压缩、核验和返工配置；应用源码复制到运行目录供worker使用。"
             "评测实例将Writer章节入口绑定到普通生成方法，关闭章节级task，仅保留节点检查点。"
             "基线来自已有开始/完成事件与模型调用；旧上下文、版本与实时来源不同，差值为近似估计。",
             "", "原始目录：" + str(output),
             "", "## 核心指标与逐题分析",
             "|题号|故障位置|触发|正确交付|历史秒数|恢复秒数|估计省秒|恢复已知LLM费用估算/元|",
             "|---|---|---|---|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.2f}"
    for r in rows:
        lines.append(f"|{r['source_id']}|{r['case']}|{r['fault_triggered']}|{r['recovery_completed']}|"
                     f"{fmt(r['baseline_seconds'])}|{fmt(r['resume_seconds'])}|"
                     f"{fmt(r['estimated_saved_seconds'])}|{fmt(r['resume_cost']['estimated_known_cny'])}|")
    lines += ["", f"已归档{len(rows)}/5题。逐题目录中的恢复评估结果.json、历史基线.json、"
              "历史基线模型调用.json、中断后/最终工作历史.json、完整检查点历史.json、事件账本.json和交付状态.json保留原始依据。",
              "恢复时间自worker发起恢复计时，包含租约接管等待；差值=历史完整时间−实际恢复时间。"
              "失败样例不计算已成功节省，不能把快速拒绝当成节省。",
              "原题27、28的旧轨迹以error结束，没有research_complete，因此它们的完整历史耗时留空；"
              "旧费用只代表失败运行的已知消耗，不计算相对完整基线的费用收益。",
              "", "## 节点调用与失败样例"]
    for r in rows:
        lines.append(f"- 原题{r['source_id']}：中断前调用={r['prefix_node_calls']}；"
                     f"恢复调用={r['resume_node_calls']}；最终待执行={r['pending_nodes_final']}；"
                     f"报告消息/完成事件={r['delivery']['assistant_messages']}/{r['delivery']['complete_events']}；"
                     f"失败信息={r['resume_error']}。")
    lines += ["", "## 证据边界与局限性",
              "已测量：只包括表中实际完成并归档的独立进程SIGKILL、真实PostgreSQL持久化、"
              "原run恢复请求、节点物理调用、事件与交付数量。尚在运行的题目不计入完成率。",
              "间接推断：时间收益复用历史基线估计；货币成本使用真实usage和公开单价估算。"
              "缺少usage/价格的调用不填零，已知费用仅是已知部分；失败、重试和被杀的未知调用单列。",
              "尚未评估：其他样本与中断位置组合、多次故障、负载下可靠性、质量评分、"
              "真实HTTP/浏览器端恢复与模型实际账单。检查点提交前还可能保存pending writes，按实际结果解释。",
              "终态交付提交前定义为finish检查点持久化后、commit_result调用前；"
              "若生产恢复拒绝，保留失败，不通过评测器补交付或修改恢复实现。",
              "", "## 结论与下一步",
              "以逐题实测结果判断节点恢复边界；不扩展细粒度恢复。出现失败时保留原始轨迹，"
              "单独讨论修复后复评，避免覆盖本轮失败。", "",
              "费用假设：北京实时非思考原价，无Batch、缓存或免费额度抵扣；"
              "qwen3-max输入≤32K/≤128K/≤256K为2.5/4/7元每百万，输出10/16/28元每百万；"
              "qwen-plus输入≤128K/≤256K/≤1M为0.8/2.4/4.8元每百万，输出2/20/48元每百万。"
              f"来源：[阿里云官方模型价格]({PRICE_SOURCE})；定价与代码口径见运行目录配置。",
              "⚠️ **费用说明**：以上费用为基于官方公示单价的预估价格，仅供参考。实际费用受Token消耗量、"
              "上下文长度阶梯定价、Batch/缓存折扣及计费策略调整等因素影响，请以QianWen控制台实际账单为准："
              "[按量付费账单](https://platform.qianwenai.com/home/billing/pay-as-you-go) | "
              "[Token Plan订阅](https://platform.qianwenai.com/home/billing/subscription/token-plan) | "
              "[用量分析](https://platform.qianwenai.com/home/analytics)。免费额度须自行确认，切勿假设本次调用免费。"]
    report.write_text("\n".join(lines) + "\n")


def execute_child(script, folder, mode, output, timeout):
    logpath = folder / (mode + "进程日志.txt")
    started = time.monotonic()
    with logpath.open("w") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(script), "--worker",
            "--folder", str(folder), "--mode", mode], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=os.environ.copy())
        save(output / "当前进程.json", {"pid": proc.pid, "folder": str(folder), "mode": mode, "at": now()})
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            save(folder / (mode + "外部超时.json"), {"seconds": timeout, "at": now()})
        save(folder / (mode + "进程计时.json"), {
            "returncode": proc.returncode, "duration_seconds": time.monotonic() - started,
            "at": now()})
        return proc.returncode


def orchestrate(args):
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env", override=False)
    output, report = args.output.resolve(), args.report.resolve()
    if output.exists() or report.exists():
        raise ValueError("仅允许新输出与报告")
    output.mkdir(parents=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    from config.llm_config import get_config
    from service.context_policy import selected_policy
    from eval.scripts.采集真实追问开发集 import create_identity, create_session, save_user
    tasks = [json.loads(x) for x in args.dataset.read_text().splitlines() if x.strip()]
    assert [t["source"]["id"] for t in tasks] == [x[0] for x in CASES]
    assert all(t["split"] == "development" for t in tasks)
    policy = selected_policy().snapshot()
    save(output / "配置快照.json", policy)
    os.environ["CONTEXT_ABLATION_CONFIG"] = str(output / "配置快照.json")
    # 继承已加载环境给worker，避免重新读取变动中的.env。
    os.environ["恢复评估已加载环境"] = "1"
    config = get_config()
    manifest = {"date": "2026-10-07", "status": "running", "started_at": now(),
                "source_ids": [x[0] for x in CASES], "queries": "每题首轮",
                "baseline_rerun": False, "dataset": str(args.dataset.resolve()),
                "recovery_granularity": "node_only", "writer_section_tasks_disabled": True,
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "models": {k: getattr(config.agents, k).model for k in
                           ["architect", "scout", "data_analyst", "writer", "critic"]},
                "context_policy_hash": policy["config_hash"], "prices": RATES,
                "price_source": PRICE_SOURCE, "timeout_seconds_per_attempt": args.timeout}
    save(output / "运行清单.json", manifest)
    save(output / "开发集五题输入.json", [{"source_id": t["source"]["id"],
                   "query": t["turns"][0]["user_query"]} for t in tasks])
    frozen = output / "运行源码快照" / "backend"
    shutil.copytree(BACKEND / "app", frozen / "app", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(BACKEND / "eval", frozen / "eval",
        ignore=shutil.ignore_patterns("data", "__pycache__", "benchmark"))
    frozen_script = frozen / "eval/scripts" / Path(__file__).name
    hashes = {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in frozen.rglob("*") if p.is_file()}
    save(output / "源码哈希.json", hashes)
    owner = create_identity(output.name)
    rows = []
    write_report(output, report, rows, "运行中")
    try:
        for task, (sid, case) in zip(tasks, CASES):
            folder = output / f"题目{sid}-{case}"
            folder.mkdir()
            baseline = historical(task, folder)
            session = create_session(owner, f"节点恢复评估 原题{sid}")
            ident = {"source_id": sid, "case": case, "query": task["turns"][0]["user_query"],
                     "run_id": str(uuid.uuid4()), "turn_uuid": str(uuid.uuid4()),
                     "user_id": owner, "session_id": session,
                     "max_iterations": config.research.max_iterations}
            save_user(owner, session, ident["turn_uuid"], ident["run_id"], ident["query"])
            save(folder / "运行身份.json", ident)
            print(json.dumps({"event": "case_start", "source_id": sid, "case": case}, ensure_ascii=False), flush=True)
            code = execute_child(frozen_script, folder, "中断前", output, args.timeout)
            save(folder / "中断前进程退出.json", {"returncode": code, "at": now()})
            asyncio.run(export_case(folder, "中断后"))
            if (folder / "故障触发.json").exists():
                code = execute_child(frozen_script, folder, "节点恢复", output, args.timeout)
                save(folder / "恢复进程退出.json", {"returncode": code, "at": now()})
            asyncio.run(export_case(folder, "最终"))
            row = summarize_case(folder, baseline)
            rows.append(row)
            save(output / "逐题恢复结果.json", rows)
            write_report(output, report, rows, "运行中")
            print(json.dumps({"event": "case_end", "source_id": sid,
                              "fault_triggered": row["fault_triggered"],
                              "recovery_completed": row["recovery_completed"]}, ensure_ascii=False), flush=True)
        manifest.update(status="finished", finished_at=now(), evaluated_cases=len(rows),
                        fault_triggered=sum(r["fault_triggered"] for r in rows),
                        recovery_completed=sum(r["recovery_completed"] for r in rows))
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save(output / "运行清单.json", manifest)
        write_report(output, report, rows, manifest["status"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--timeout", type=int, default=5400)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--folder", type=Path)
    p.add_argument("--mode", choices=["中断前", "节点恢复"])
    args = p.parse_args()
    if args.worker:
        try:
            asyncio.run(worker(args))
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            save(args.folder / (args.mode + "进程异常.json"),
                 {"type": type(exc).__name__, "error": str(exc), "at": now()})
            status_path = args.folder / (args.mode + "执行结果.json")
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(completed=False, error=str(exc), error_type=type(exc).__name__)
            if status.get("started_monotonic") is not None:
                status["duration_seconds"] = time.monotonic() - status["started_monotonic"]
            save(status_path, status)
            raise
    else:
        if not all([args.dataset, args.output, args.report]):
            p.error("需要dataset、output和report")
        orchestrate(args)


if __name__ == "__main__":
    main()
