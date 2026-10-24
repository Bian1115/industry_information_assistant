"""固定原题与源码，保持部署配置，启动受保护的完整轨迹采集。"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("重跑必须使用新的空目录，不能覆盖之前的尝试")
    output.mkdir(parents=True)
    source = BACKEND / "eval/data/通用研究中文5题-20260930"
    for name in ("抽样清单.json", "中文原题评测集.jsonl", "官方题库快照.jsonl"):
        (output / name).write_bytes((source / name).read_bytes())

    spec = importlib.util.spec_from_file_location("collector", Path(__file__).with_name("采集通用研究评测集.py"))
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    development = Path("/Users/bxy/projects/DeepResearch医疗研报助手/评测集/原题来源与划分清单-20260927.json")
    tasks, manifest = collector.prepare(output, development, 20260930)
    assert [task["id"] for task in tasks] == [35, 31, 46, 16, 49]

    overrides = {
        "RESEARCH_ALLOWED_DOMAINS": "", "AGENT_SUMMARY_NLI_ENABLED": "0",
        "AGENT_SUMMARY_CHECKS_ENABLED": "0", "RESEARCH_GATE_CITATION_SUPPORT_ENABLED": "false",
        "RESEARCH_GATE_CITATION_REPAIR_ENABLED": "false", "EVAL_TRACE_ENABLED": "true",
        "EVAL_TRACE_PAYLOAD_CHARS": "10000000", "EVAL_TRACE_QUEUE_SIZE": "10000",
        "EVAL_TRACE_LOCAL_FILE": str(output / "完整过程埋点.jsonl"),
    }
    os.environ.update(overrides)
    from config.llm_config import get_config
    from service.context_policy import selected_policy
    from service.deep_research_v2.runtime_budget import total_budget_seconds, max_llm_tokens, phase_budget_seconds
    from core.database import engine
    from models.agent_history import AgentContext, AgentMessage
    from models.research import AgentEvent, ResearchRun
    from eval.models import AgentRun
    from sqlalchemy import inspect
    config, policy = get_config(), selected_policy()
    assert not policy.semantic_verification and policy.nli is None
    schema = inspect(engine)
    missing = {}
    for model in (AgentContext, AgentMessage, AgentEvent, ResearchRun, AgentRun):
        actual = {column["name"] for column in schema.get_columns(model.__tablename__)}
        absent = set(model.__table__.columns.keys()) - actual
        if absent:
            missing[model.__tablename__] = sorted(absent)
    if missing:
        collector.save(output / "启动检查失败.json", {"missing_database_columns": missing})
        raise RuntimeError("数据库结构尚不兼容当前源码，已保存检查结果")

    collector.save(output / "运行配置.json", {
        "started_at": collector.now(), "agents": dataclasses.asdict(config.agents),
        "research": dataclasses.asdict(config.research), "context_policy": policy.snapshot(),
        "max_iterations": config.research.max_iterations, "timeout_seconds": args.timeout, "concurrency": 1,
        "runtime_budget": {"total_limit_seconds": total_budget_seconds(), "token_limit": max_llm_tokens(),
                           "phase_seconds": {agent: phase_budget_seconds(agent) for agent in policy.stage_budgets}},
        "citation_verification": "关闭报告引用核验、摘要NLI与语义检查；历史摘要沿用当前部署策略",
        "allowed_domains": [], "local_trace_full_payload": True,
        "langfuse_enabled": os.getenv("LANGFUSE_ENABLED"), "sampling_manifest": "抽样清单.json",
        "power_guard": "caffeinate -i -s，覆盖父进程及整个顺序采集期间",
        "run_scope": "固定中文评测集重跑；不执行外部裁判评分",
    })
    collector.freeze_code(output)
    extra_paths = list((BACKEND / "eval").glob("*.py"))
    extra_paths += [Path(__file__), Path(__file__).with_name("采集进程超时保护.py"), Path(__file__).with_name("采集墙钟看门狗.py")]
    extra_paths += [p for p in (BACKEND / "eval/configs").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    hashes = json.loads((output / "源码与提示词哈希.json").read_text())
    with zipfile.ZipFile(output / "运行源码快照.zip", "a", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(extra_paths)):
            relative = str(path.relative_to(REPO))
            if relative in hashes:
                continue
            data = path.read_bytes()
            archive.writestr(relative, data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
    collector.save(output / "源码与提示词哈希.json", hashes)
    runtime = output / "运行工作副本"
    with zipfile.ZipFile(output / "运行源码快照.zip") as archive:
        archive.extractall(runtime)
    collector.save(output / "重跑来源说明.json", {
        "created_at": collector.now(), "previous_run_directory": str(source),
        "question_ids": manifest["evaluation_ids"], "questions_resampled": False, "prompts_rewritten": False,
        "fixed_questions_sha256": hashlib.sha256((output / "中文原题评测集.jsonl").read_bytes()).hexdigest(),
        "execution_source": str(runtime), "source_frozen": True, "environment_file_exported": False,
        "database_schema_preflight": "passed", "original_artifacts_modified": False,
        "configuration_overrides": overrides, "parent_pid": os.getpid(),
    })
    # 凭据由已加载的进程环境继承；不复制.env，不保存密钥。
    command = [sys.executable, "-u", str(runtime / "backend/eval/scripts/采集进程超时保护.py"),
               "--output", str(output), "--development-manifest", str(development), "--timeout", str(args.timeout)]
    print(f"源码已冻结；题目{manifest['evaluation_ids']}；单题时限{args.timeout}s；开始采集", flush=True)
    code = subprocess.call(command, cwd=runtime, env=os.environ.copy())
    collector.save(output / "采集进程结束.json", {"finished_at": collector.now(), "exit_code": code})
    raise SystemExit(code)


if __name__ == "__main__":
    main()
