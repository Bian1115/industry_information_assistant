"""输出中文机器结果和原始日志；真实库实验仅创建并清理独立schema。"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

BACKEND=Path(__file__).resolve().parents[2]
ROOT=BACKEND.parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--postgres", action="store_true")
    args=parser.parse_args()
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/"验证汇总.json").exists():
        raise FileExistsError("请使用新的结果目录，禁止覆盖历史评估")
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join(map(str,[BACKEND/"app",BACKEND/"eval/tests",BACKEND/"test"]))
    if args.postgres:
        from dotenv import dotenv_values
        for key,value in dotenv_values(BACKEND/".env").items():
            if value is not None and key not in env: env[key]=value
        env["RUN_STORAGE_RECONCILIATION_PG"]="1"
        env["STORAGE_RECONCILIATION_AUDIT"]=str(out/"隔离库生命周期.jsonl")
        modules=["test_跨存储对账.PostgreSQLReconciliation"]
    else:
        modules=["test_跨存储对账.IndexReconciliation", "test_跨存储对账.VectorInventory", "test_预算持久化与检查点租约隔离", "test_单主链路图",
            "test_Writer章节检查点", "test_恢复入口待合并节点结果", "test_待合并节点结果恢复", "test_终态交付补偿恢复",
            "test_图入口对齐", "test_research_runtime", "test_DRB服务接口", "test_会话临时双层RAG.TemporaryRAGTests",
            "test_会话双层RAG分页精排"]
    proc=subprocess.run([sys.executable,"-m","unittest",*modules,"-v"], cwd=ROOT,env=env,text=True,
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/"原始测试日志.txt").write_text(proc.stdout)
    paths=[ROOT/"backend/app/service/deep_research_v2"/name for name in
        ("storage_reconciliation.py","checkpointer.py","graph.py","service.py","state.py","agents/writer.py")]
    paths += [ROOT/"backend/app/service"/name for name in ("session_evidence_rag.py","session_evidence_store.py","session_evidence_vectors.py")]
    paths += [ROOT/"backend/app/router/research_router.py",Path(__file__).resolve(),BACKEND/"eval/tests/test_跨存储对账.py"]
    result={"评估日期":"2026-10-16","运行UTC":datetime.now(timezone.utc).isoformat(),"真实PostgreSQL":args.postgres,
        "真实外部模型请求":0,"主业务schema修改":False,"退出码":proc.returncode,"测试模块":modules,
        "结果摘要":proc.stdout[-1600:],"源码哈希":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    total=re.search(r"Ran (\d+) tests?",proc.stdout)
    skipped=re.search(r"skipped=(\d+)",proc.stdout)
    result.update(测试数=int(total[1]) if total else None,跳过数=int(skipped[1]) if skipped else 0)
    if args.postgres and (out/"隔离库生命周期.jsonl").exists():
        audit=[json.loads(line) for line in (out/"隔离库生命周期.jsonl").read_text().splitlines()]
        created={r["schema"] for r in audit if r["status"]=="created"}
        cleaned={r["schema"] for r in audit if r["status"]=="cleaned"}
        result.update(隔离库数量=len(created),隔离库全部已清理=created==cleaned)
    (out/"验证汇总.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"退出码":proc.returncode,"结果目录":str(out),"结果摘要":proc.stdout[-1600:]},ensure_ascii=False))
    return proc.returncode


if __name__=="__main__":
    sys.exit(main())
