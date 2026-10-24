"""固定源码和配置，真实生成假设开启/关闭各五篇，保存独立质量评分。"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import zipfile
from zoneinfo import ZoneInfo
from datetime import datetime

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
DIMS = ["depth", "comprehensiveness", "readability", "requirements"]


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


async def worker(args):
    from eval.scripts.run_deepresearch_bench import benchmark_service, run_one
    from eval.scripts.采集通用研究评测集 import export_database
    from eval.runtime import tracer
    from service.deep_research_v2.graph import graph_config

    enabled = args.variant == "开启"
    tasks = read(args.output / "固定五题.json")
    task = next(t for t in tasks if t["id"] == args.only_id)
    folder = args.output / args.variant / f"题目{task['id']}"
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["EVAL_TRACE_LOCAL_FILE"] = str(folder / "完整过程埋点.jsonl")
    os.environ["WEB_READER_CACHE_DIR"] = str(folder / "网页缓存")
    async with benchmark_service(read(args.output / "运行配置.json")["max_iterations"]) as service:
        class FrozenSwitch:
            def research(self, **kwargs):
                return service.research(**kwargs, hypothesis_enabled=enabled)
        row = await run_one(FrozenSwitch(), task, "hyp-on" if enabled else "hyp-off",
                            args.output.name[-15:].replace("-", "_"), args.timeout, False,
                            artifact_dir=folder)
        await asyncio.to_thread(tracer._queue.join)
        counts = await asyncio.to_thread(export_database, row["_meta"], folder)
        checkpoints = []
        async for item in service.checkpointer.alist(graph_config(row["_meta"]["research_run_id"])):
            checkpoints.append({"config": item.config, "checkpoint": item.checkpoint,
                                "metadata": item.metadata, "parent_config": item.parent_config,
                                "pending_writes": item.pending_writes})
        save(folder / "完整检查点历史.json", checkpoints)
        state = checkpoints[0]["checkpoint"]["channel_values"].get("research", {}) if checkpoints else {}
        save(folder / "最终研究状态.json", state)
        if state and state.get("hypothesis_enabled") is not enabled:
            raise ValueError("检查点实际开关与实验组不一致")
        if not enabled and any(state.get(k) for k in ["hypotheses", "hypothesis_evidence", "conclusion_evidence_map"]):
            raise ValueError("关闭组出现假设或关系账本")
        calls = read(folder / "LLM调用埋点.json")
        metrics = {"id": task["id"], "variant": args.variant, "hypothesis_enabled": enabled,
                   "success": bool(row["article"]), "error": row["_meta"]["error"],
                   "duration_s": row["_meta"]["duration_s"], "article_chars": len(row["article"]),
                   "llm_calls": len(calls), "known_tokens": sum(x.get("total_tokens") or 0 for x in calls),
                   "hypotheses": len(state.get("hypotheses", [])),
                   "hypothesis_evidence": len(state.get("hypothesis_evidence", [])),
                   "search_budget": state.get("search_budget"), "archive_counts": counts,
                   "checkpoints": len(checkpoints), "report_sha256": hashlib.sha256(row["article"].encode()).hexdigest()}
        save(folder / "运行指标.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)


def freeze(out):
    hashes = {}
    paths = [p for base in [BACKEND / "app", BACKEND / "eval/scripts", BACKEND / "eval/runtime", BACKEND / "eval/configs"]
             for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.suffix in {".py", ".json", ".txt", ".md", ".yaml", ".yml"}]
    paths += list((BACKEND / "eval").glob("*.py"))
    with zipfile.ZipFile(out / "运行源码快照.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            name = str(path.relative_to(ROOT)); data = path.read_bytes()
            archive.writestr(name, data); hashes[name] = hashlib.sha256(data).hexdigest()
    save(out / "源码与提示词哈希.json", hashes)
    runtime = out / "运行工作副本"
    with zipfile.ZipFile(out / "运行源码快照.zip") as archive:
        archive.extractall(runtime)
    return runtime


def prepare(out, timeout):
    if out.exists():
        raise ValueError("每次对照实验必须新建目录")
    out.mkdir(parents=True)
    old = read(BACKEND / "eval/data/五篇报告fewshot复评与复核一致性-20261006/冻结报告输入.json")
    audit = []
    for task in old:
        checkpoints = read(Path(task["folder"]) / "完整检查点历史.json")
        state = checkpoints[0]["checkpoint"]["channel_values"]["research"]
        audit.append({"id": task["id"], "folder": task["folder"], "hypothesis_enabled": state.get("hypothesis_enabled", True),
                      "hypotheses": len(state.get("hypotheses", [])), "hypothesis_evidence": len(state.get("hypothesis_evidence", []))})
    save(out / "历史五篇开关核对.json", audit)
    save(out / "历史五篇输入冻结.json", old)
    tasks = [json.loads(line) for line in (BACKEND / "eval/data/通用研究中文5题-20260930/中文原题评测集.jsonl").read_text().splitlines()]
    assert [t["id"] for t in tasks] == [35, 31, 46, 16, 49]
    assert all(t["prompt"] == next(x["query"] for x in old if x["id"] == t["id"]) for t in tasks)
    save(out / "固定五题.json", tasks)
    runtime = freeze(out)
    database = "hypothesis_eval_" + datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S")
    import psycopg
    from psycopg import sql
    with psycopg.connect(host=os.getenv("POSTGRES_HOST", "localhost"), port=os.getenv("POSTGRES_PORT", "5432"),
                         user=os.getenv("POSTGRES_USER", "postgres"), password=os.getenv("POSTGRES_PASSWORD", "postgres123"),
                         dbname="postgres", autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    os.environ["POSTGRES_DB"] = database
    os.environ.pop("RESEARCH_CHECKPOINT_DATABASE_URL", None)
    os.environ.update(EVAL_TRACE_ENABLED="true", EVAL_TRACE_PAYLOAD_CHARS="10000000", EVAL_TRACE_QUEUE_SIZE="10000")
    from core.database import Base, engine
    import models
    import eval.models
    Base.metadata.create_all(engine)
    from service.deep_research_v2.checkpointer import prepare_checkpoint_schema
    asyncio.run(prepare_checkpoint_schema())
    from config.llm_config import get_config
    from service.context_policy import selected_policy
    from service.deep_research_v2.runtime_budget import total_budget_seconds, max_llm_tokens, phase_budget_seconds
    c, policy = get_config(), selected_policy()
    save(out / "运行配置.json", {"started_at": now(), "database": database, "source": str(runtime),
         "agents": dataclasses.asdict(c.agents), "research": dataclasses.asdict(c.research), "max_iterations": c.research.max_iterations,
         "context_policy": policy.snapshot(), "total_seconds": total_budget_seconds(), "token_limit": max_llm_tokens(),
         "stage_seconds": {n: phase_budget_seconds(n) for n in policy.stage_budgets}, "external_timeout_seconds": timeout,
         "generation_processes": 2, "hypothesis_enabled": [True, False],
         "other_safe_overrides": {k:v for k,v in os.environ.items() if k.startswith(("RESEARCH_", "AGENT_SUMMARY_", "AGENT_COMPRESSION", "CONTEXT_ABLATION")) and "URL" not in k},
         "design": "同一题开启/关闭进程同时运行，每对按题号顺序执行，独立会话与网页缓存；继承当前配置，仅改变假设开关",
         "limitations": ["每题每组只生成一次，模型随机性与动态检索仍可能影响结果", "相同预算上限，不保证实际消耗相同", "旧五篇均开启，历史对照不能用于开关因果归因"]})
    return runtime


def score_reports(out):
    evaluator = module("hyp_quality", BACKEND / "eval/scripts/评估五篇正文与Scout.py")
    validator = module("hyp_validate", BACKEND / "eval/scripts/复评五篇质量与复核一致性.py")
    evaluator.OUT = out / "质量评估"
    rubric = evaluator.QUALITY.replace("当前评估截止2026-10-03", "当前评估截止2026-10-07")
    (out / "质量评分提示词快照.txt").write_text(rubric)
    tasks, jobs, scores, failures = read(out / "固定五题.json"), [], [], []
    mapping = {("开启", 35):"R01", ("关闭", 35):"R02", ("关闭",31):"R03", ("开启",31):"R04", ("开启",46):"R05", ("关闭",46):"R06", ("关闭",16):"R07", ("开启",16):"R08", ("开启",49):"R09", ("关闭",49):"R10"}
    save(out / "盲评编号映射.json", [{"variant":v,"id":i,"code":k} for (v,i),k in mapping.items()])
    frozen=[]
    for t in tasks:
        for variant in ["开启", "关闭"]:
            row=read(out / variant / f"题目{t['id']}" / "运行结果.json")
            if not row["article"]:
                failures.append({"id":t['id'],"variant":variant,"stage":"generation","error":row['_meta']['error']});continue
            body=re.split(r'(?m)^#{1,6}\s*参考文献\s*$', row["article"])[0].strip()
            item={"id":t['id'],"variant":variant,"code":mapping[variant,t['id']],"query":t['prompt'],"body":body,"report_sha256":hashlib.sha256(row['article'].encode()).hexdigest()}
            frozen.append(item)
            for repeat in [1,2]:jobs.append((item,repeat))
    save(out / "盲评报告输入冻结.json", frozen)
    save(out / "评分配置.json", {"model":"qwen3-max","temperature":0.3,"repeats":2,"dimensions":DIMS,"prompt_sha256":hashlib.sha256(rubric.encode()).hexdigest(),"blinding":"裁判仅看到任务与单篇完整正文，无开关标签，无旧分数；正文可能透露假设结构", "expected_calls":len(jobs)})
    def work(item,repeat):
        prompt=rubric+f"\n本次重复编号{repeat}\n用户任务：{item['query']}\n完整正文：\n{item['body']}"
        result=evaluator.call('qwen',prompt,f"{item['code']}-质量-{repeat}",7000,validator.validate_quality,request_options={"model":"qwen3-max","temperature":0.3})
        return {"id":item['id'],"variant":item['variant'],"repeat":repeat,"code":item['code'],"result":result}
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(work,*job):job for job in jobs}
        for future in cf.as_completed(pending):
            try:scores.append(future.result())
            except Exception as exc:
                item,repeat=pending[future];failures.append({"id":item['id'],"variant":item['variant'],"repeat":repeat,"stage":"quality","error":str(exc)})
            save(out / "独立四维质量评分.json",scores)
            save(out / "评分状态.json",{"updated_at":now(),"completed":len(scores),"expected":len(jobs),"failures":failures})
    pairs=[]
    for task in tasks:
        by={v:[r for r in scores if r['id']==task['id'] and r['variant']==v] for v in ['开启','关闭']}
        if not all(len(x)==2 for x in by.values()):continue
        dims={d:{v:statistics.mean(r['result']['dimensions'][d]['score'] for r in by[v]) for v in by} for d in DIMS}
        for d in dims.values():d['delta_on_minus_off']=d['开启']-d['关闭']
        pairs.append({"id":task['id'],"dimensions":dims,"on_mean":statistics.mean(d['开启'] for d in dims.values()),"off_mean":statistics.mean(d['关闭'] for d in dims.values()),"delta":statistics.mean(d['delta_on_minus_off'] for d in dims.values())})
    save(out / "五题配对质量差值.json",{"pairs":pairs,"paired_tasks":len(pairs),"mean_delta":statistics.mean(x['delta'] for x in pairs) if pairs else None,"wins_ties_losses":{"wins":sum(x['delta']>0 for x in pairs),"ties":sum(x['delta']==0 for x in pairs),"losses":sum(x['delta']<0 for x in pairs)},"failures":failures})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=int,default=2500)
    p.add_argument('--variant',choices=['开启','关闭']);p.add_argument('--only-id',type=int);p.add_argument('--score-only',action='store_true')
    args=p.parse_args();args.output=args.output.resolve()
    from dotenv import load_dotenv
    load_dotenv(BACKEND / '.env')
    if args.variant:
        asyncio.run(worker(args));return
    if args.score_only:
        score_reports(args.output);return
    runtime=prepare(args.output,args.timeout)
    script=runtime / 'backend/eval/scripts/运行假设开关五题对照.py'
    guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-m','-s','-w',str(os.getpid())])
    try:
        for index,t in enumerate(read(args.output / '固定五题.json')):
            variants=['开启','关闭'] if index%2==0 else ['关闭','开启'];children=[]
            for variant in variants:
                folder=args.output / variant / f"题目{t['id']}";folder.mkdir(parents=True,exist_ok=True)
                log=(folder / '运行日志.log').open('w')
                proc=subprocess.Popen([sys.executable,'-u',str(script),'--output',str(args.output),'--variant',variant,'--only-id',str(t['id']),'--timeout',str(args.timeout)],cwd=runtime,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                children.append((proc,log,variant))
            print(f"题{t['id']}开启/关闭已启动",flush=True)
            while any(c[0].poll() is None for c in children):
                save(args.output / '运行状态.json',{'updated_at':now(),'phase':'生成','id':t['id'],'children':[{"variant":v,"pid":c.pid,"exit_code":c.poll()} for c,l,v in children]})
                time.sleep(5)
            for proc,log,variant in children:
                log.close()
                print(f"题{t['id']} {variant}退出{proc.returncode}",flush=True)
                if not (args.output / variant / f"题目{t['id']}" / '运行结果.json').exists():
                    raise RuntimeError(f"题{t['id']} {variant}未保存运行结果，查看日志")
        save(args.output / '生成完成.json',{'finished_at':now()})
        score_reports(args.output)
        save(args.output / '运行状态.json',{'updated_at':now(),'phase':'完成','comparison':read(args.output / '五题配对质量差值.json')})
    finally:
        guard.terminate();guard.wait(timeout=5)


if __name__=='__main__':
    main()
