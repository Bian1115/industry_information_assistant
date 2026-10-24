"""用户指定只用Claude补跑剩余第一题，另外两题不运行；单独记录配置与用量。"""
import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
from eval.scripts import 运行Critic职责分离五题配对消融 as paired
from eval.scripts.Claude消融原生协议适配 import claude_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-id", type=int)
    parser.add_argument("--timeout", type=int, default=7800)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--retrieval-profile", choices=["hybrid", "lexical"], default="hybrid")
    args = parser.parse_args();args.output=args.output.resolve()
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
    if args.only_id:
        with claude_profile(lexical_only=args.retrieval_profile == "lexical"):
            asyncio.run(paired.worker(args))
        return
    model = os.environ["CLAUDE_RELAY_MODEL"]
    os.environ.update(OPENAI_MODEL=model, LLM_MODEL_SUMMARY=model, LLM_FALLBACK_MODEL=model,
        RESEARCH_BUDGET_LEADWRITER_SECONDS="2400", RESEARCH_TOTAL_BUDGET_SECONDS="7200",
        RESEARCH_WRITER_CITATION_VERIFICATION_ENABLED="true", RESEARCH_ALLOWED_DOMAINS="")
    for role in ("ARCHITECT", "SCOUT", "DATA_ANALYST", "WRITER", "CRITIC", "WIZARD"):
        os.environ["LLM_MODEL_" + role] = model
    tasks, runtime = paired.base.prepare(args)
    paired.base.save(args.output / "原计划五题快照.json", tasks)
    selected = [t for t in tasks if t["id"] == 46]
    paired.base.save(args.output / "固定五题.json", selected)
    paired.rubric(args.output)
    rubric=paired.base.read(args.output / "Codex直接质量评估口径.json")
    rubric["requirements_by_task"] = [t for t in rubric["requirements_by_task"] if t["id"] == 46]
    rubric["limits"].append("仅一题Claude同首稿对照，题目不同，不能推断相对Qwen的模型总体优劣，不混合计算总体增益")
    if args.retrieval_profile == "lexical":
        rubric["limits"].append("本题使用既有词法检索降级，和前一Qwen运行检索配置不同")
    paired.base.save(args.output / "Codex直接质量评估口径.json",rubric)
    config=paired.base.read(args.output / "运行配置.json")
    config.update(experiment_version="claude-single-case-critic-v1", planned_task_ids=[46],
        sdk_model_aliases={"qwen3-max":model}, actual_model_mode={model:{"temperature":"不传，服务默认", "thinking":"不显式设置"}},
        actual_llm_protocol="Anthropic Messages，使用已有Claude Relay配置；HTTP请求与响应usage留痕",
        generation_models="所有生成Agent、历史摘要、论断提取、证据选句与NLI均经Claude原生协议适配",
        retrieval_profile=("SessionEvidenceRAG现有词法降级；实验拦截Embedding/Rerank" if args.retrieval_profile == "lexical" else "SessionEvidenceRAG原有混合检索，Embedding=text-embedding-v4，Rerank=qwen3-rerank；2026-10-18用户已充值恢复服务"),
        retrieval_profile_name=args.retrieval_profile,
        usage_accounting="原生input_tokens加cache_creation_input_tokens和cache_read_input_tokens；再加output_tokens。provider total_tokens如有则优先；不重复增加thinking_tokens",
        context_token_estimate="沿用项目本地Qwen分词器作上下文预算估算；实际用量取Claude服务返回usage",
        parent_experiment="Critic职责分离通用研究五题消融-20261017-231745",
        user_scope_change="2026-10-18用户要求剩余三题只选一题用Claude，另两题先不跑；选择原顺序第一题46",
        skipped_task_ids=[16,49], no_external_quality_judge=True)
    paired.base.save(args.output / "运行配置.json",config)
    manifest=paired.base.read(args.output / "运行清单.json")
    manifest.update(source_ids=[46],planned_reports=2,held_by_user=[16,49])
    paired.base.save(args.output / "运行清单.json",manifest)
    with claude_profile(lexical_only=args.retrieval_profile == "lexical"):
        asyncio.run(paired.preflight(args.output))
    folder=args.output/"题目46";folder.mkdir()
    log=(folder/"运行日志.log").open("w")
    script=runtime/"backend/eval/scripts/运行Claude房地产单题Critic消融.py"
    proc=subprocess.Popen([sys.executable,"-u",str(script),"--output",str(args.output),"--only-id","46","--timeout",str(args.timeout),"--retrieval-profile",args.retrieval_profile],cwd=runtime,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
    guard=subprocess.Popen(["/usr/bin/caffeinate","-i","-m","-s","-w",str(os.getpid())])
    try:
        while proc.poll() is None:
            paired.base.save(args.output/"运行状态.json",{"updated_at":paired.base.now(),"phase":"Claude单题生成中","running":[{"id":46,"pid":proc.pid}],"pending_ids":[],"held_by_user":[16,49]})
            time.sleep(5)
        result=paired.base.read(folder/"配对运行结果.json")
        paired.base.save(args.output/"生成结果汇总.json",[result])
        paired.base.save(args.output/"运行状态.json",{"updated_at":paired.base.now(),"phase":"生成结束待Codex审读","running":[],"pending_ids":[],"held_by_user":[16,49],"finished":[{"id":46,"status":result["status"],"exit_code":proc.returncode}]})
    finally:
        log.close();guard.terminate();guard.wait(timeout=5)


if __name__ == "__main__":
    main()
