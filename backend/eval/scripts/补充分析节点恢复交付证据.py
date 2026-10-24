#!/usr/bin/env python3
"""只读已运行的五题证据，检查正文交付、重复节点与费用完整性。"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
spec = importlib.util.spec_from_file_location("recovery_eval", Path(__file__).with_name("评估开发集节点级恢复.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def digest(value):
    return hashlib.sha256((value or "").encode()).hexdigest()


def analyze(folder, stamp):
    from core.database import SessionLocal
    from models.chat import ChatMessage
    from models.research import AgentEvent
    result = read(folder / "恢复评估结果.json")
    if result is None:
        return None
    ident = read(folder / "运行身份.json")
    before = read(folder / "恢复请求状态.json") or read(folder / "中断后研究状态.json")
    final = read(folder / "最终研究状态.json")
    cps = read(folder / "中断后完整检查点历史.json", [])
    latest = cps[0] if cps else {}
    raw_research = latest.get("checkpoint", {}).get("channel_values", {}).get("research") or {}
    journal = [json.loads(x) for x in (folder / "节点与提交记录.jsonl").read_text().splitlines()]
    post_starts = [r for r in journal if r["kind"] == "node_start" and r["mode"] == "节点恢复"]
    records = read(folder / "最终工作历史.json", [])
    requested = {(r["call_id"], (r["payload"] or {}).get("attempt", 1))
                 for r in records if r["kind"] == "llm_request"}
    accounted = {(r["call_id"], (r["payload"] or {}).get("attempt", 1))
                 for r in records if r["kind"] == "llm_usage"}
    usage_by_node = {}
    for node in sorted({r["node"] for r in records if r["kind"] == "llm_usage"}):
        usage_by_node[node] = runner.costs([r["payload"] for r in records
                                           if r["kind"] == "llm_usage" and r["node"] == node])
    report = final.get("final_report") or ""
    with SessionLocal() as db:
        messages = db.query(ChatMessage).filter_by(run_id=ident["run_id"], role="assistant").all()
        completed = db.query(AgentEvent).filter_by(run_id=ident["run_id"], event_type="research_complete").all()
        message_hashes = [digest(r.content) for r in messages]
        event_hashes = [digest((r.payload or {}).get("final_report")) for r in completed]
    restored_events = []
    for mode in ["中断前", "节点恢复"]:
        path = folder / (mode + "研究事件.jsonl")
        if path.exists():
            restored_events += [json.loads(x)["event"] for x in path.read_text().splitlines()
                                if json.loads(x)["event"].get("restored")
                                or (json.loads(x)["event"].get("data") or {}).get("restored")]
    output = {
        "source_id": ident["source_id"], "case": ident["case"], "run_id": ident["run_id"],
        "analysis_at": runner.now(), "first_resumed_node": post_starts[0]["node"] if post_starts else None,
        "frontier_after_fault": before["pending_nodes"],
        "state_status_after_fault": before.get("status"), "state_phase_after_fault": before.get("phase"),
        "raw_latest_checkpoint_has_report": bool(raw_research.get("final_report")),
        "read_state_has_report_after_fault": bool(before.get("final_report")),
        "raw_latest_checkpoint_attempts": raw_research.get("node_attempts", {}),
        "pending_write_channels_after_fault": [w[1] for w in latest.get("pending_writes", [])],
        "restored_section_events": len(restored_events),
        "child_task_return_writes": sum("__return__" in r.get("channels", []) for r in journal),
        "final_report_chars": len(report), "final_report_sha256": digest(report),
        "message_sha256": message_hashes, "complete_event_sha256": event_hashes,
        "report_delivery_matches_checkpoint": bool(report) and message_hashes == event_hashes == [digest(report)],
        "usage_missing_requests": [{"call_id": cid, "attempt": a} for cid, a in sorted(requested - accounted)],
        "llm_attempt_error_records": sum(r["kind"] == "llm_attempt_error" for r in records),
        "llm_error_records": sum(r["kind"] == "llm_error" for r in records),
        "usage_cost_by_node": usage_by_node,
        "completion_reason": final.get("completion_reason"), "workflow_quality_verdict": final.get("quality_verdict"),
        "independent_quality_scoring": False,
    }
    runner.save(folder / f"恢复交付补充证据-{stamp}.json", output)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env", override=False)
    rows = read(args.output / "逐题恢复结果.json", [])
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
    evidence = [r for folder in sorted(args.output.glob("题目*")) if (r := analyze(folder, stamp)) is not None]
    totals = {
        "analyzed_cases": len(evidence), "recovery_completed": sum(r["recovery_completed"] for r in rows),
        "report_delivery_consistent_cases": sum(r["report_delivery_matches_checkpoint"] for r in evidence),
        "known_whole_experiment_cny": round(sum(r["whole_attempts_cost"]["estimated_known_cny"] for r in rows), 6),
        "known_resume_cny": round(sum(r["resume_cost"]["estimated_known_cny"] for r in rows), 6),
        "known_historical_cny": round(sum(r["baseline_cost"]["estimated_known_cny"] for r in rows), 6),
        "unknown_usage_requests": sum(len(r["usage_missing_requests"]) for r in evidence),
        "child_task_return_writes": sum(r["child_task_return_writes"] for r in evidence),
        "notes": "费用为LLM公开原价估算，历史未完成题不充当完整基线；未评估独立质量、HTTP和浏览器"}
    analysis_path = args.output / f"恢复交付与费用补充分析-{stamp}.json"
    runner.save(analysis_path, {"analysis_at": runner.now(), "totals": totals, "cases": evidence,
                               "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if args.report:
        date = datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
        lines = ["", "## 同一运行补充分析", "", f"分析日期：{date}。没有新增模型运行。补充口径：核对正文SHA-256、首个恢复节点、子任务持久化写入和缺失usage请求。",
                 "", "正确交付仅指：图终止、有正文、报告消息与完成事件各一条，且两份正文与终态检查点完全一致；不代表事实质量已通过独立评估。",
                 "故障后的收益口径：继续恢复R对比从头重跑B，节省=B−R；中断前已花时间属于两种策略共有的沉没成本。评测实际支出另外统计中断前加恢复的全部已知调用。",
                 "", f"共分析{totals['analyzed_cases']}题，完成恢复{totals['recovery_completed']}题，正文一致性通过{totals['report_delivery_consistent_cases']}题。",
                 f"正式评测全部已知LLM消耗估算{totals['known_whole_experiment_cny']:.4f}元，其中恢复阶段{totals['known_resume_cny']:.4f}元；缺失usage请求{totals['unknown_usage_requests']}次，无法按零费用处理。",
                 "", "|题号|首个恢复节点|原始完整检查点有正文|pending writes通道|正文交付一致|流程结束原因|",
                 "|---|---|---|---|---|---|"]
        for r in evidence:
            lines.append(f"|{r['source_id']}|{r['first_resumed_node']}|{r['raw_latest_checkpoint_has_report']}|{r['pending_write_channels_after_fault']}|{r['report_delivery_matches_checkpoint']}|{r['completion_reason']}|")
        lines += ["", "### 模型费用拆分（人民币估算）", "",
                  "|题号|历史已知费用|历史是否完整完成|本次中断前加恢复|恢复阶段|相对完整基线节省|",
                  "|---|---:|---|---:|---:|---:|"]
        for r in rows:
            saving = r.get("estimated_saved_cny")
            saving_text = "—" if saving is None else f"{saving:.4f}"
            lines.append(f"|{r['source_id']}|{r['baseline_cost']['estimated_known_cny']:.4f}|{r['historical_completed']}|"
                         f"{r['whole_attempts_cost']['estimated_known_cny']:.4f}|{r['resume_cost']['estimated_known_cny']:.4f}|{saving_text}|")
        lines += ["", "子任务__return__持久化写入数=" + str(totals["child_task_return_writes"]) + "；章节restored事件和细粒度task已单列核对。",
                  "原始补充证据：" + str(analysis_path.resolve()),
                  "运行器机制验证：" + str((args.output / "运行器机制验证.json").resolve()),
                  "准备运行已停止，不计入五题成功率，已知额外LLM消耗估算0.09元；在途未知费用与搜索费用另行保留为未估算。准备记录：" + str((args.output.parent / "开发集五题节点级恢复评估-20261007-233410/准备运行停止分析.json").resolve())]
        lines += ["", "### 按场景解释与下一步", ""]
        for r in evidence:
            sid = r["source_id"]
            if sid == 8:
                text = "部分搜索的节点尚未提交，恢复从research开始，规划复用。该题含审核返工，历史完整时间差为负；节点复用成立不等于单次端到端必然加速。"
            elif sid == 25:
                text = "故障时三章草稿仅在节点内存，检查点草稿为空；首次恢复节点为write，丢失章节重新生成。之后的补搜/分析/写作属于正常审核返工。"
            elif sid == 26:
                if not r["frontier_after_fault"] and "research" in r["pending_write_channels_after_fault"] and not r["first_resumed_node"]:
                    text = "写作输出已保存为pending writes，原始完整检查点尚无正文；读取状态可应用写作输出，但pending_nodes为空，恢复入口前置判断拒绝。状态尚未completed，不应据此当作图已完成；本轮没有续跑审核，暴露入口对待合并节点输出的误判。"
                else:
                    text = "完整检查点提交前仍须检查pending writes；若research输出已持久化且首次恢复为review，表示已完成写作通过节点输出记录滚动前进，并非章节级恢复。"
            elif sid == 27:
                text = "完整写作检查点提交后应从review续跑；旧轨迹未完成，无法按约定计算完整时间或费用收益。本次恢复时长、调用和单次交付仍可独立测量。"
            else:
                text = "终态图已保存而报告/完成事件未提交时，图已无待执行节点。若恢复返回400且两种交付记录均为0，这是交付补偿入口缺口；快速拒绝不算节省。"
            lines.append(f"- 原题{sid}：{text}实际首个恢复节点={r['first_resumed_node']}，正文交付一致={r['report_delivery_matches_checkpoint']}，流程结束原因={r['completion_reason']}。")
        lines += ["", "本轮各位置仅一个样本，结论限定在这五个实测场景。建议恢复入口区分真正终态和已有pending writes待推进的状态，同时补终态交付的幂等恢复入口，再按同一位置复评；本轮不修改恢复机制或启用章节级恢复。所有质量结论仍仅来自流程内Critic，独立质量保持尚未评估。",
                  "已测量：真实SIGKILL、数据库检查点和工作记录、节点调用、交付正文一致性。间接推断：历史完整耗时差及公开原价费用。尚未评估：缺失完整历史基线的收益、独立质量、HTTP/浏览器重连、重复故障与负载可靠性。"]
        with args.report.open("a") as f:
            f.write("\n".join(lines) + "\n")
    print(json.dumps({"analysis_path": str(analysis_path.resolve()), **totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
