#!/usr/bin/env python3
"""独立deepagents端到端DRB对照；不启动业务服务、不连接业务数据库。"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
from eval.baselines.deepagents_runner import (
    CONFIG_DIR, Settings, load_leaf, load_tasks, package_versions, run_task, sha256, write_json,
)


def manifest_for(settings, tasks, dataset, prompt):
    source_paths = [Path(__file__), BACKEND / "eval/baselines/deepagents_runner.py",
                    CONFIG_DIR / "研究提示词.txt", BACKEND / "requirements-deepagents对照.txt",
                    BACKEND / "requirements-deepagents锁定.txt"]
    source_paths += [BACKEND / "app/service" / p for p in (
        "web_search_service.py", "web_reader_service.py", "web_document.py", "deep_research_v2/source_scope.py")]
    if settings.citation_verification:
        source_paths += [BACKEND / "eval/scripts/deepagents引用核验工作器.py"]
        source_paths += [BACKEND / "app/service" / p for p in ("nli_verifier.py", "context_policy.py", "qwen_token_counter.py", "llm_client.py")]
        source_paths += [BACKEND / "app/prompts/Qwen引用支持性核验.txt"]
    url = urlsplit(os.getenv("DASHSCOPE_BASE_URL", ""))
    # 端点不记录认证信息或查询参数。
    endpoint = urlunsplit((url.scheme, url.netloc.split("@")[-1], url.path, "", ""))
    reader_env = {key: sha256(value.encode())
                  for key, value in os.environ.items()
                  if key.startswith(("WEB_READER_", "CRAWL4AI_")) and key != "WEB_READER_CACHE_DIR"}
    body = {"schema_version": 1, "settings": settings.model_dump(), "tasks": tasks,
            "dataset_sha256": sha256(dataset.read_bytes()), "system_prompt": prompt,
            "packages": package_versions(), "endpoint": endpoint, "reader_environment": reader_env,
            "source_sha256": {str(p.relative_to(BACKEND)): sha256(p.read_bytes()) for p in source_paths},
            "framework_defaults": "deepagents原生中间件与StateBackend；可选已读原文引用核验工具；无业务图、记忆或Critic",
            "search_mode": "live_bocha_scoped" if settings.allowed_domains else "live_bocha_open_web", "reader_cache": "每个输出目录独立"}
    body["manifest_hash"] = sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode())
    return body


def prepare_output(output, manifest, resume=False):
    if output.exists():
        if not resume:
            raise ValueError("输出目录已存在；续跑请显式使用--resume，或选择新目录")
        previous = json.loads((output / "运行配置.json").read_text())
        if previous != manifest:
            raise ValueError("配置、题目、源码或依赖变化，拒绝混入已有对照运行")
    else:
        if resume:
            raise ValueError("续跑目录不存在")
        output.mkdir(parents=True)
        write_json(output / "运行配置.json", manifest)


def latest_results(output):
    rows = {}
    for path in sorted(output.glob("题目-*/*/结果.json")):
        row = json.loads(path.read_text())
        key = str(row["id"])
        if rows.get(key, {}).get("status") != "completed":
            rows[key] = row
    return rows


def export_results(output, tasks):
    rows = latest_results(output)
    selected = [rows[str(t["id"])] for t in tasks if str(t["id"]) in rows]
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    temp = output / "逐题结果.jsonl.tmp"
    temp.write_text(text, encoding="utf-8")
    temp.replace(output / "逐题结果.jsonl")
    return selected


def write_analysis(output, tasks, *, dry_run=False):
    rows = export_results(output, tasks)
    successful = sum(row["status"] == "completed" for row in rows)
    # 重试开销不能因为最终成功而消失：另汇总所有落盘attempt。
    attempts = [json.loads(p.read_text()) for p in sorted(output.glob("题目-*/*/结果.json"))]
    incomplete = sum(not (p / "结果.json").exists() for p in output.glob("题目-*/尝试-*"))
    metrics = {"planned_tasks": len(tasks), "finished_tasks": len(rows), "completed_tasks": successful,
               "failed_tasks": len(rows) - successful, "attempts": len(attempts),
               "elapsed_seconds_sum": sum(r["elapsed_seconds"] for r in attempts),
               "model_calls": sum(r["metrics"]["model_calls"] for r in attempts),
               "reported_input_tokens": sum(r["metrics"]["reported_input_tokens"] for r in attempts),
               "reported_output_tokens": sum(r["metrics"]["reported_output_tokens"] for r in attempts),
               "incomplete_attempts": incomplete,
               "usage_complete": bool(attempts) and not incomplete and all(r["metrics"]["usage_complete"] for r in attempts),
               "cost_estimate": (sum(r["metrics"]["cost_estimate"] for r in attempts)
                                 if attempts and not incomplete and all(r["metrics"]["cost_estimate"] is not None for r in attempts) else None),
               "quality_evaluation": "not_run", "dry_run": dry_run}
    write_json(output / "汇总.json", metrics)
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S-%f")
    report = BACKEND.parent / "docs" / f"deepagents独立对照运行分析-{stamp}.md"
    details = "\n".join(f"- 题目{r['id']}：{r['status']}；错误类型{r.get('error', '无')}；记录：{r['artifact_dir']}" for r in rows) or "- 尚未执行研究任务。"
    report.write_text(f"""# deepagents独立对照运行分析

评估日期：{datetime.now().astimezone().isoformat()}。评估对象：独立deepagents端到端研究组。

配置与输入以[{output.name}]({output})下的运行配置.json为准，包含模型、预算、来源范围、题目、依赖版本和源码哈希。
原始结果：{output}/逐题结果.jsonl；原始attempt、调用输入输出和网页快照在各题目目录。续跑的所有attempt均纳入开销汇总。

核心指标：
```json
{json.dumps(metrics, ensure_ascii=False, indent=2)}
```

逐题分析与失败样例：
{details}

已测量：{'仅检查配置和输入，未调用模型或网页服务。' if dry_run else '完成/失败状态、耗时、调用数及供应商实际返回的token用量；缺失用量显式标记。'}
间接推断：输出字段兼容现有DRB报告评分入口，不代表已经获得质量分数。
尚未评估：RACE/FACT质量、引用准确率、人工评分、账单核对及与业务组的配对显著性。
局限性：线上检索动态变化；必须用相同题号、模型、来源范围和预算配对比较。配置里的窗口容量需核对供应商。
费用仅按显式配置的单价估算模型和搜索开销；未知单价或用量时为null，不包含本地CPU、浏览器/OCR服务等额外成本。缓存及阶梯价尚未建模，不能当作实际账单。
结论与下一步：这是独立系统对照，不能归因为某个上下文压缩组件的消融收益。完成相同任务后，统一评分器再比较质量与成本。
""", encoding="utf-8")
    return report


async def execute(tasks, settings, output, prompt, concurrency=1):
    rows = latest_results(output)
    semaphore = asyncio.Semaphore(concurrency)
    async def one(task):
        if rows.get(str(task["id"]), {}).get("status") == "completed":
            return
        task_root = output / ("题目-" + sha256(str(task["id"]).encode())[:16])
        task_root.mkdir(exist_ok=True)
        number = len(list(task_root.glob("尝试-*"))) + 1
        folder = task_root / f"尝试-{number:04d}"
        folder.mkdir()
        async with semaphore:
            result = await run_task(task, settings, folder, prompt)
        export_results(output, tasks)
        print(json.dumps({"id": task["id"], "status": result["status"]}, ensure_ascii=False))
    await asyncio.gather(*(one(task) for task in tasks))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "端到端研究.json")
    parser.add_argument("--dataset", type=Path, default=BACKEND / "eval/data/deepresearch_bench/query.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=["zh", "en", "all"], default="zh")
    parser.add_argument("--ids", help="逗号分隔题号")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="冻结配置和输入，不调用模型/搜索")
    args = parser.parse_args(argv)
    if args.limit <= 0 or args.concurrency <= 0:
        parser.error("--limit和--concurrency必须为正数")
    load_dotenv(BACKEND / ".env", override=False)
    # 独立进程内关闭外部追踪，调用快照只写本地对照目录。
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    output = args.output.resolve()
    os.environ["WEB_READER_CACHE_DIR"] = str(output / "网页缓存")
    scope = load_leaf("deep_research_v2/source_scope.py")
    config = json.loads(args.config.read_text())
    config["allowed_domains"] = (scope.normalize_allowed_domains(config.get("allowed_domains"))
                                  or scope.configured_allowed_domains())
    settings = Settings.model_validate(config)
    tasks = load_tasks(args.dataset, args.language, set(args.ids.split(",")) if args.ids else None, args.limit)
    if not tasks:
        parser.error("没有符合条件的题目")
    if not args.dry_run:
        missing = [key for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "BOCHA_API_KEY") if not os.getenv(key)]
        if missing:
            parser.error("缺少配置：" + ", ".join(missing))
    prompt = (CONFIG_DIR / "研究提示词.txt").read_text() + ("\n允许来源域名：" + ", ".join(settings.allowed_domains) if settings.allowed_domains else "\n来源范围：全网；search_web的domain留空。")
    if settings.citation_verification:
        prompt += "\n已提供verify_citations工具。最终答复前，提交报告中关键事实和数字核验，每项包含claim及references:[{url,quote}]，quote逐字取自已读原文。核验不支持时按证据修改或明确不确定性；未核验事实不能声称通过。可以按最多32条分批核验。"
    manifest = manifest_for(settings, tasks, args.dataset, prompt)
    manifest["concurrency"] = args.concurrency
    prepare_output(output, manifest, args.resume)
    if args.dry_run:
        print(write_analysis(output, tasks, dry_run=True))
        return 0
    try:
        asyncio.run(execute(tasks, settings, output, prompt, args.concurrency))
    finally:
        print(write_analysis(output, tasks))
    return 0 if all(r["status"] == "completed" for r in latest_results(output).values()) and len(latest_results(output)) == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
