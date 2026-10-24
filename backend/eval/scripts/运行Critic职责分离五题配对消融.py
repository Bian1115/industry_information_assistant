"""当前职责分离版本：真实同首稿分叉、完整模型用量、引用审计与匿名审读材料。"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent
sys.path[:0] = [str(ROOT), str(BACKEND), str(BACKEND / "app")]
from eval.scripts import 运行Critic通用研究五题消融 as base
from eval.scripts.记录模型调用用量 import Recorder, summarize


@contextmanager
def experiment_model_aliases(aliases):
    """只在实验进程映射已下线旧型号；实际请求型号由HTTP记录器保存。"""
    from unittest.mock import patch
    from openai.resources.chat.completions import Completions, AsyncCompletions
    sync_create, async_create = Completions.create, AsyncCompletions.create
    def sync_call(client, *args, **kwargs):
        kwargs["model"] = aliases.get(kwargs.get("model"), kwargs.get("model"))
        if kwargs["model"] == "qwen3.8-max":
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "enable_thinking": False}
        return sync_create(client, *args, **kwargs)
    async def async_call(client, *args, **kwargs):
        kwargs["model"] = aliases.get(kwargs.get("model"), kwargs.get("model"))
        if kwargs["model"] == "qwen3.8-max":
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "enable_thinking": False}
        return await async_create(client, *args, **kwargs)
    with patch.object(Completions, "create", sync_call), patch.object(AsyncCompletions, "create", async_call):
        yield


def citation_metrics(state):
    from collections import Counter
    audit = state.get("citation_verification") or {}
    rows = audit.get("claims") or []
    statuses = Counter(row.get("status", "unknown") for row in rows)
    extraction = state.get("report_claim_extraction") or audit.get("claim_extraction") or {}
    count = audit.get("claims_count")
    completed = audit.get("extracted_claims_checked") is True
    denominator = count if completed and type(count) is int and count > 0 else None
    supported = audit.get("supported_count")
    missing = audit.get("missing_citation_count")
    prediction = audit.get("prediction_as_fact_count")
    return {"status": audit.get("status"), "execution_status": audit.get("execution_status"),
        "scope": audit.get("scope"), "extraction_status": audit.get("extraction_status", extraction.get("status")),
        "units_total": extraction.get("units_total"), "unit_status_counts": dict(Counter(
            row.get("status") for row in extraction.get("coverage") or [])),
        "claims_count": count, "supported_count": supported, "missing_citation_count": missing,
        "prediction_as_fact_count": prediction, "review_count": audit.get("review_count"),
        "claim_status_counts": dict(statuses),
        "supported_rate": supported / denominator if denominator and type(supported) is int else None,
        "missing_citation_rate": missing / denominator if denominator and type(missing) is int else None,
        "prediction_as_fact_rate": prediction / denominator if denominator and type(prediction) is int else None,
        "extracted_claims_checked": audit.get("extracted_claims_checked", False),
        "all_claims_checked": audit.get("all_claims_checked", False),
        "report_sha256": audit.get("report_sha256"),
        "rate_denominator": "已完成核验的独立抽取论断数；执行失败或无论断时不计算比例"}


def clone_stores(a, b, state):
    counts = ORIGINAL_CLONE(a, b, state)
    # 新版跨存储协议需要为新运行建立自己的权威预算基线；公共前缀只计一次实际请求。
    from core.database import SessionLocal
    from models.research_budget import ResearchBudget
    with SessionLocal.begin() as db:
        source = db.get(ResearchBudget, a["run_id"])
        if source:
            baseline = deepcopy(state.get("runtime_budget") or {})
            baseline.pop("ledger_version", None)
            db.add(ResearchBudget(run_id=b["run_id"], policy=deepcopy(source.policy),
                baseline={"runtime_budget": baseline, "node_attempts": deepcopy(state.get("node_attempts") or {})}))
            counts["shared_prefix_budget_baselines"] = 1
    state.pop("storage_manifest", None)
    state.pop("storage_protocol", None)
    return counts


ORIGINAL_CLONE = base.clone_stores
ORIGINAL_BUILD = base.build_graph if hasattr(base, "build_graph") else None


def fork_graph_adapter(original):
    def build(agents, saver, **kwargs):
        graph = original(agents, saver, **kwargs)
        if kwargs.get("history_store") is not None or "interrupt_before" in kwargs:
            return graph
        class ForkGraph:
            async def aupdate_state(self, config, value, **update):
                from service.research_events import acquire_research_run_lease, release_research_run_lease
                from service.deep_research_v2.checkpointer import for_execution
                from service.deep_research_v2.storage_reconciliation import StorageCoordinator
                lease = await asyncio.to_thread(acquire_research_run_lease, run_id=kwargs["run_id"],
                    session_id=kwargs["session_id"], user_id=kwargs["user_id"])
                try:
                    fenced = for_execution(saver, lease)
                    storage = StorageCoordinator(fenced.conn, kwargs["run_id"], lease=lease)
                    fenced.storage_coordinator = storage
                    options = {**kwargs, "storage_coordinator": storage}
                    production = original(agents, fenced, **options)
                    value = {"research": await storage.seal(value["research"], node="write")}
                    return await production.aupdate_state(config, value, **update)
                finally:
                    await asyncio.to_thread(release_research_run_lease, lease, "paused")
        return ForkGraph()
    return build


def blind_materials(output, ident):
    directory = output / f"题目{ident}"
    groups = ["关闭审核", "开启审核"]
    random.SystemRandom().shuffle(groups)
    mapping = {}
    for label, group in zip(("报告甲", "报告乙"), groups):
        source = directory / group
        report = source / "研究报告.md"
        if not report.exists():
            return
        dest = output / "盲评材料" / f"题目{ident}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / (label + ".md")).write_text(report.read_text())
        (dest / (label + "来源.json")).write_text((source / "完整来源正文.json").read_text())
        mapping[label] = {"group": group, "report_sha256": base.digest(report.read_text())}
    base.save(directory / "盲评标签映射.json", mapping)


async def worker(args):
    directory = args.output / f"题目{args.only_id}"
    recorder = Recorder(directory / "全部模型API用量.jsonl")
    original_consume, original_export = base.consume, base.export
    from service.deep_research_v2.graph import build_graph
    async def consume(service, identity, folder, **kw):
        with recorder.group(folder.name):
            return await original_consume(service, identity, folder, **kw)
    async def export(service, identity, folder):
        state, calls, metrics = await original_export(service, identity, folder)
        base.save(folder / "引用核验完整结果.json", state.get("citation_verification") or {})
        base.save(folder / "独立论断提取结果.json", state.get("report_claim_extraction") or {})
        base.save(folder / "引用核验指标.json", citation_metrics(state))
        base.save(folder / "模型API用量汇总.json", recorder.summary(folder.name))
        return state, calls, metrics
    # base.worker在函数内导入build_graph，因此在模块导入源上注入分叉适配器。
    from unittest.mock import patch
    aliases = base.read(args.output / "运行配置.json").get("sdk_model_aliases", {})
    with recorder.installed(), experiment_model_aliases(aliases), patch.object(base, "consume", consume), patch.object(base, "export", export), \
            patch.object(base, "clone_stores", clone_stores), \
            patch("service.deep_research_v2.graph.build_graph", fork_graph_adapter(build_graph)):
        try:
            await base.worker(args)
        finally:
            base.save(directory / "实验实际API用量.json", recorder.summary())
            by_group = {name: recorder.summary(name) for name in ["公共首稿", "关闭审核", "开启审核"]}
            full = {"关闭审核": summarize([r for r in recorder.rows if r["group"] in {"公共首稿", "关闭审核"}]),
                    "开启审核": summarize([r for r in recorder.rows if r["group"] in {"公共首稿", "开启审核"}])}
            base.save(directory / "配对完整用量.json", {"incremental_by_group": by_group,
                "logical_full_by_group": full, "actual_experiment": recorder.summary(),
                "note": "公共首稿和其引用核验实际只执行一次；两组逻辑完整用量不可相加当实际账单"})
            result_path = directory / "配对运行结果.json"
            if result_path.exists():
                result = base.read(result_path)
                result["complete_api_usage_path"] = "配对完整用量.json"
                base.save(result_path, result)
                if result.get("status") == "completed":
                    blind_materials(args.output, args.only_id)


def rubric(output):
    previous = ROOT / "backend/eval/data/Critic通用研究五题消融-20261009-221748/Codex直接质量评估口径.json"
    value = base.read(previous)
    value.update(created_at=base.now(), evaluator="当前Codex主代理全文审读；不调用外部质量裁判",
        blind_review="随机甲乙标签；先独立评分，再揭示组别。单一AI评审，仍可能从修改内容猜到组别",
        supplementary_dimensions={"logical_coherence": "报告整体逻辑与结论校准，0至10分，单独报告",
                                  "viewpoint_balance": "反向证据、局限与观点平衡，0至10分，单独报告"})
    value["limits"] = ["一位AI评审单次评分，不称人类金标或统计显著增益", "同一固定五题曾用于历史分析，不是新留出集",
        "同首稿配对比较审核及其触发返工的整体收益，开启组有额外预算，不能当等token比较",
        "完整阅读最终正文，并抽查保存来源；未获得支持不等同现实虚假", "引用审计独立报告，不拿Critic内部评分当外部质量"]
    base.save(output / "Codex直接质量评估口径.json", value)


async def preflight(output):
    """只验证隔离库的新检查点分叉及真实交付，不调用任何模型。"""
    from eval.scripts.run_deepresearch_bench import benchmark_service
    from eval.scripts.采集真实追问开发集 import create_identity, create_session, save_user
    from service.deep_research_v2.graph import build_graph, create_agents, graph_config, pack_state
    from service.deep_research_v2.state import create_initial_state
    from service.deep_research_v2.stage_runtime import make_execute_stage
    from service.research_events import acquire_research_run_lease, release_research_run_lease
    import uuid
    owner = await asyncio.to_thread(create_identity, "消融分叉预检")
    sid = await asyncio.to_thread(create_session, owner, "消融分叉预检")
    run, turn = str(uuid.uuid4()), str(uuid.uuid4())
    await asyncio.to_thread(save_user, owner, sid, turn, run, "机制预检")
    lease = await asyncio.to_thread(acquire_research_run_lease, run_id=run, session_id=sid, user_id=owner)
    await asyncio.to_thread(release_research_run_lease, lease, "paused")
    state = create_initial_state("机制预检", sid, [], critic_enabled=False)
    state["final_report"] = "机制预检占位报告，不进入任何模型节点。"
    identity = {"id": "预检", "query": "机制预检", "session_id": sid, "run_id": run,
                "turn_uuid": turn, "user_id": owner, "critic_enabled": False}
    folder = output / "机制预检"; folder.mkdir()
    async with benchmark_service(2) as service:
        graph = fork_graph_adapter(build_graph)(create_agents(service.config), service.checkpointer,
            run_id=run, user_id=owner, session_id=sid,
            execute_stage=make_execute_stage(run_id=run, session_id=sid), history_store=None)
        await graph.aupdate_state(graph_config(run), pack_state(state), as_node="write")
        result = await base.consume(service, identity, folder, resume=True)
        base.save(folder / "结果.json", result)
        if not result.get("completed") or result.get("error"):
            raise RuntimeError("隔离库分叉预检未完成，尚未启动付费生成")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-id", type=int)
    parser.add_argument("--timeout", type=int, default=7800)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args(); args.output = args.output.resolve()
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
    if args.only_id:
        asyncio.run(worker(args)); return
    # 新独立论断提取与逐条选句增加耗时，两组使用相同实验预算，不修改部署配置。
    os.environ.update(RESEARCH_BUDGET_LEADWRITER_SECONDS="2400", RESEARCH_TOTAL_BUDGET_SECONDS="7200",
        RESEARCH_WRITER_CITATION_VERIFICATION_ENABLED="true", RESEARCH_ALLOWED_DOMAINS="",
        OPENAI_MODEL="qwen3.8-max", LLM_MODEL_SUMMARY="qwen3.8-max")
    for role in ("ARCHITECT", "DATA_ANALYST", "WRITER", "CRITIC", "WIZARD"):
        os.environ["LLM_MODEL_" + role] = "qwen3.8-max"
    tasks, runtime = base.prepare(args)
    rubric(args.output)
    asyncio.run(preflight(args.output))
    config = base.read(args.output / "运行配置.json")
    config.update(experiment_version="critic-responsibility-v2", writer_phase_limit_seconds=2400,
        complete_usage_scope="全部HTTP模型请求，按公共首稿/审核返工/关闭分支记录，含embedding与rerank",
        actual_quality_evaluator="Codex主代理；匿名甲乙材料先评分再揭示组别",
        sdk_model_aliases={"qwen3-max": "qwen3.8-max"},
        actual_model_mode={"qwen3.8-max": {"enable_thinking": False}},
        model_compatibility_note="qwen3-max实际入口返回404，实验中生成与核验实际请求统一使用qwen3.8-max；旧NLI配置标识保留协议兼容，两组使用相同映射，实际模型见API用量记录")
    base.save(args.output / "运行配置.json", config)
    script = runtime / "backend/eval/scripts/运行Critic职责分离五题配对消融.py"
    pending, running, finished = list(tasks), [], []
    guard = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-m", "-s", "-w", str(os.getpid())])
    try:
        while pending or running:
            while pending and len(running) < args.concurrency:
                task = pending.pop(0)
                folder = args.output / f"题目{task['id']}"; folder.mkdir(exist_ok=True)
                log = (folder / "运行日志.log").open("w")
                proc = subprocess.Popen([sys.executable, "-u", str(script), "--output", str(args.output),
                    "--only-id", str(task["id"]), "--timeout", str(args.timeout)], cwd=runtime,
                    env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                running.append({"id": task["id"], "proc": proc, "log": log, "started": time.monotonic()})
                print(json.dumps({"event": "题目启动", "id": task["id"], "pid": proc.pid}, ensure_ascii=False), flush=True)
            for item in list(running):
                if item["proc"].poll() is None:
                    continue
                item["log"].close(); running.remove(item)
                path = args.output / f"题目{item['id']}" / "配对运行结果.json"
                outcome = base.read(path) if path.exists() else {"id": item["id"], "status": "failed", "error": "进程无结果"}
                outcome["exit_code"] = item["proc"].returncode; finished.append(outcome)
                print(json.dumps({"event": "题目结束", "id": item["id"], "status": outcome["status"]}, ensure_ascii=False), flush=True)
            base.save(args.output / "运行状态.json", {"updated_at": base.now(),
                "phase": "生成中" if pending or running else "生成结束待Codex审读",
                "running": [{"id": x["id"], "pid": x["proc"].pid} for x in running],
                "pending_ids": [t["id"] for t in pending], "finished": finished,
                "reports_completed": sum(g.get("success", False) for r in finished for g in r.get("groups", {}).values())})
            if pending or running:
                time.sleep(5)
        base.save(args.output / "生成结果汇总.json", finished)
    finally:
        guard.terminate(); guard.wait(timeout=5)


if __name__ == "__main__":
    main()
