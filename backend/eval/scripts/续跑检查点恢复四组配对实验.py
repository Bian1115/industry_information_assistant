"""充值后补跑未完成分组；执行上一轮冻结Worker，显式复用已完成基线。"""
from __future__ import annotations
import argparse
import asyncio
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from zoneinfo import ZoneInfo

BACKEND=Path(__file__).resolve().parents[2]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n");temp.replace(path)


def read(path):return json.loads(path.read_text())
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    from dotenv import load_dotenv
    load_dotenv(BACKEND/".env",override=False)
    previous=args.previous.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    prior=read(previous/"运行清单.json")
    manifest={"date":datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(),"timezone":"Asia/Singapore",
        "status":"准备中","started_at":now(),"previous_directory":str(previous),"schema":prior["schema"],
        "groups":prior["groups"],"fault_cases":prior["fault_cases"],"shared_fault_prefix":True,
        "thinking_mode":"disabled","real_models":True,"section_tasks_enabled":True,
        "host_awake_guard":"独立caffeinate-dimsu-43200秒，任务结束后撤销",
        "execution_order":[26,27,28,25,8],"phase_order":["共同故障前缀","D","C","A/B"],
        "concurrency":args.concurrency,"completed_phases":[],"errors":[],"reused_completed_groups":[],
        "baseline_boundary":"B25复用上一轮成功样本；其余失败组各补跑一次。报告单列跨日基线和本轮实际消耗。"}
    save(out/"运行清单.json",manifest)
    original_hashes=read(previous/"源码哈希.json")
    for name,expected in original_hashes.items():
        if sha(previous/"冻结源码/backend"/name)!=expected:raise ValueError("上一轮冻结源码已变化")
    shutil.copytree(previous/"冻结源码",out/"冻结源码",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for name in ["源码哈希.json","冻结五题故障注入输入.json","原始开发集五题.jsonl","冻结上下文配置.json","模型可用性与价格核对.json"]:
        shutil.copyfile(previous/name,out/name)
    shutil.copyfile(Path(__file__),out/"补跑控制器源码.py")
    worker=out/"冻结源码/backend/eval/scripts/运行检查点恢复四组配对实验.py"
    spec=importlib.util.spec_from_file_location("frozen_paired_worker",worker)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    for role,settings in prior["models"].items():os.environ["LLM_MODEL_"+role.upper()]=settings["model"]
    main_model=prior["models"]["writer"]["model"]
    for key in ["OPENAI_MODEL","LLM_MODEL_SUMMARY","LLM_MODEL_WIZARD","LLM_FALLBACK_MODEL"]:os.environ[key]=main_model
    os.environ.update(CHECKPOINT_PAIR_NON_THINKING="1",CONTEXT_ABLATION_CONFIG=str(out/"冻结上下文配置.json"),
        RESEARCH_MAX_ITERATIONS=str(prior["max_iterations"]),RESEARCH_MAX_TOTAL_SEARCHES="20")
    from config.llm_config import get_config
    config=get_config()
    models={r:getattr(config.agents,r).to_dict() for r in prior["models"]}
    if models!=prior["models"]:raise ValueError("角色参数与上一轮不一致")
    manifest.update(models=models,max_iterations=config.research.max_iterations,
        context_policy_hash=prior["context_policy_hash"],dataset_sha256=prior["dataset_sha256"],
        evaluation_model_override=prior["evaluation_model_override"])
    # 真实小请求验证已恢复可用；不以模型目录查询替代余额状态验证。
    from openai import OpenAI
    preflight=[]
    with OpenAI(api_key=config.api_key,base_url=config.base_url,timeout=45,max_retries=0) as client:
        for model in [main_model,prior["models"]["scout"]["model"]]:
            row={"model":model,"started_at":now(),"max_tokens":8,"enable_thinking":False}
            try:
                response=client.chat.completions.create(model=model,messages=[{"role":"user","content":"请只回复OK"}],
                    max_tokens=8,temperature=0,extra_body={"enable_thinking":False})
                row.update(ok=True,response_model=response.model,response_id=response.id,
                    usage=response.usage.model_dump() if response.usage else None,finish_reason=response.choices[0].finish_reason)
            except Exception as exc:
                row.update(ok=False,error_type=type(exc).__name__,http_status=getattr(exc,"status_code",None),
                    error_code=(getattr(exc,"body",{}) or {}).get("code") if isinstance(getattr(exc,"body",None),dict) else None)
            preflight.append(row);save(out/"充值后接口预检.json",preflight)
            print(json.dumps({"event":"preflight","model":model,"ok":row["ok"]},ensure_ascii=False),flush=True)
            if not row["ok"]:
                manifest.update(status="接口预检失败",ended_at=now());save(out/"运行清单.json",manifest);return
    # 上一轮已清理隔离库；从本任务的已校验归档恢复，保留B25原文供附加评审。
    import psycopg
    from psycopg import sql
    schema=prior["schema"]
    if not re.fullmatch(r"checkpoint_pair_[0-9a-f]{32}",schema):raise ValueError("拒绝操作非本实验schema")
    archive=previous/"隔离资源归档";record=read(archive/"归档与清理记录.json")
    dump=archive/"隔离库原始快照.dump"
    if sha(dump)!=record["files"][dump.name]["sha256"]:raise ValueError("隔离库快照哈希不匹配")
    params=dict(host=os.getenv("POSTGRES_HOST","localhost"),port=os.getenv("POSTGRES_PORT","5432"),
        user=os.getenv("POSTGRES_USER","postgres"),password=os.getenv("POSTGRES_PASSWORD","postgres123"),
        dbname=os.getenv("POSTGRES_DB","industry_assistant"),connect_timeout=10)
    restored=False
    try:
        with psycopg.connect(**params,autocommit=True) as admin:
            if admin.execute("SELECT 1 FROM pg_namespace WHERE nspname=%s",(schema,)).fetchone():raise ValueError("目标schema已存在，拒绝覆盖")
            with dump.open("rb") as handle:
                result=subprocess.run(["docker","exec","-i","industry_postgres","pg_restore","--username",params["user"],
                    "--dbname",params["dbname"],"--no-owner","--no-privileges","--exit-on-error"],stdin=handle,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
            if result.returncode:raise RuntimeError("隔离库快照还原失败，保留记录供检查")
            restored=True
            counts={name:admin.execute(sql.SQL("SELECT count(*) FROM {}.{}").format(sql.Identifier(schema),sql.Identifier(name))).fetchone()[0]
                    for name in record["database_table_rows"]}
            if counts!=record["database_table_rows"]:raise ValueError("隔离库还原行数不一致")
        save(out/"隔离库还原核对.json",{"at":now(),"source":str(dump),"sha256":sha(dump),"table_rows":counts,
            "boundary":"只还原原自建schema；旧向量未重插，B25复用只读取原文，所有新分组使用新会话建索引。"})
        db=runner.configure(schema)
        from models.user import User
        owner=uuid.uuid4()
        with db.SessionLocal() as session:
            session.add(User(id=owner,username="eval_pair_"+owner.hex[:20],email=owner.hex+"@eval.invalid",
                hashed_password="!disabled-evaluation",is_active=False));session.commit()
        inputs=read(out/"冻结五题故障注入输入.json")
        by_id={r["source_id"]:r for r in inputs}
        for row in inputs:(out/f"题目{row['source_id']}-{row['fault_case']}").mkdir()
        # 只复用已成功完成的B25，不复用未完成样本以假冒成功。
        source=previous/"题目25-第三章已持久化/B组-无故障开启检查点"
        if not read(source/"执行状态.json").get("completed") or not read(source/"交付一致性.json").get("consistent"):
            raise ValueError("上一轮B25不满足复用条件")
        target=out/"题目25-第三章已持久化/B组-无故障开启检查点"
        shutil.copytree(source,target)
        reuse={"source_directory":str(source),"copied_at":now(),"new_generation_calls":0,
            "boundary":"历史已完成基线；费用不计入本轮实际消耗，跨日A/B、B/D差值需单列。"}
        runner.save(target/"复用来源.json",reuse)
        manifest["reused_completed_groups"].append({"source_id":25,"group":"B",**reuse})
        manifest.update(status="运行中",schema_preserved_for_analysis=True)
        save(out/"运行清单.json",manifest)
        lock=threading.Lock();stop=threading.Event()
        def record_phase(sid,group,folder):
            status=read(folder/"执行状态.json") if (folder/"执行状态.json").exists() else {}
            fatal=next((e for e in status.get("errors",[]) if e.get("error_code") in {"Arrearage","invalid_api_key","model_not_found"}),None)
            with lock:
                manifest["completed_phases"].append({"source_id":sid,"group":group,"technical_stream_completed":bool(status.get("completed"))})
                if fatal:
                    stop.set();manifest["fatal_provider_error"]={"source_id":sid,"group":group,"code":fatal.get("error_code")}
                save(out/"运行清单.json",manifest)
        def execute(sid,group,phase,inherited=None):
            if stop.is_set():return None
            row=by_id[sid];base=out/f"题目{sid}-{row['fault_case']}"
            folder=base/("共同故障前缀" if group=="P" else group+"组-"+runner.GROUPS[group]);folder.mkdir()
            ident=runner.create_identity(folder,owner,row["query"],sid,row["fault_case"],"D" if group=="P" else group,inherited)
            print(json.dumps({"event":"start","source_id":sid,"group":group},ensure_ascii=False),flush=True)
            runner.run_child(worker,folder,phase,schema,args.timeout)
            asyncio.run(runner.export_run(folder,schema));record_phase(sid,group,folder)
            print(json.dumps({"event":"end","source_id":sid,"group":group},ensure_ascii=False),flush=True)
            return ident
        def fault_pair(sid):
            ident=execute(sid,"P","共同故障前缀")
            if ident is None:return
            base=out/f"题目{sid}-{by_id[sid]['fault_case']}"
            if not (base/"共同故障前缀/故障触发.json").exists():
                with lock:manifest["errors"].append({"source_id":sid,"error":"目标故障未触发"});save(out/"运行清单.json",manifest)
                return
            execute(sid,"D","检查点续跑",ident)
            execute(sid,"C","从头重跑")
        def baselines(sid):
            for group in (["A","B"] if sid%2==0 else ["B","A"]):
                if sid==25 and group=="B":continue
                execute(sid,group,"正常运行")
        for label,work in [("故障恢复与重跑",fault_pair),("无故障基线",baselines)]:
            if stop.is_set():break
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures=[executor.submit(work,sid) for sid in manifest["execution_order"]]
                for future in concurrent.futures.as_completed(futures):future.result()
            manifest["last_finished_phase_family"]=label;save(out/"运行清单.json",manifest)
        manifest.update(status="运行结束",ended_at=now(),evaluation_outcome="provider_blocked" if stop.is_set() else "planned_phases_finished")
    except BaseException as exc:
        manifest.update(status="中断",ended_at=now(),error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        manifest["schema_preserved_for_analysis"]=restored;save(out/"运行清单.json",manifest)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--concurrency",type=int,choices=[1,2],default=2);parser.add_argument("--timeout",type=int,default=3000)
    main(parser.parse_args())
