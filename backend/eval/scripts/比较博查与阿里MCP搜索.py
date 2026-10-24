"""使用 DeepScout 的真实搜索入口进行查询级配对评估，不调用 LLM。

示例（从项目根目录运行）：
  backend/.venv-eval-py311/bin/python backend/eval/scripts/比较博查与阿里MCP搜索.py --discover
  backend/.venv-eval-py311/bin/python backend/eval/scripts/比较博查与阿里MCP搜索.py --query '大语言模型 医学研究 综述' --count 5
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import random
import statistics
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND / "app"))

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env", override=False)

from config.llm_config import LLMConfig
from service.aliyun_search_mcp import AliyunSearchMCP, MCP_CONNECTION_MODE
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.source_scope import normalize_allowed_domains


class RecordingScout(DeepScout):
    """保留每个 site 子查询的诊断，避免只记录最后一次请求。"""
    async def _run_web_search_tool(self, args, runtime_context):
        try:
            return await super()._run_web_search_tool(args, runtime_context)
        finally:
            self.search_calls.append(copy.deepcopy(self.last_search_diagnostics))


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * p) - 1)], 2)


def metrics(rows):
    summary = {}
    for provider in ("bocha", "aliyun_mcp"):
        cases = [r for r in rows if r["provider"] == provider]
        times = [r["duration_ms"] for r in cases]
        summary[provider] = {
            "queries": len(cases), "errors": sum(r["status"] == "error" for r in cases),
            "successful_empty_queries": sum(r["status"] == "ok" and not r["results"] for r in cases),
            "nonempty_queries": sum(bool(r["results"]) for r in cases),
            "unique_urls": len({p["url"] for r in cases for p in r["results"]}),
            "mean_duration_ms": round(statistics.mean(times), 2) if times else None,
            "p50_duration_ms": percentile(times, .5), "p95_duration_ms": percentile(times, .95),
            "cost": None,
        }
    return summary


def save_report(run_dir, info, rows):
    summary = metrics(rows)
    (run_dir / "评估汇总.json").write_text(json.dumps({
        "configuration": info, "metrics": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "docs" / f"博查与阿里MCP搜索评估结果及分析-{run_dir.name}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "逐次搜索结果.jsonl"
    lines = ["# 博查与阿里 MCP 搜索评估结果及分析", "",
             f"评估日期：{info['date']}。执行状态：{info['status']}。", "",
             "## 对象与配置口径", "", "比较现有 Bocha 搜索后端和百炼 EnhancedSearch MCP 后端。",
             "均经过 DeepScout → ToolRuntime → 统一结果契约 → 网站范围过滤；不调用 LLM。",
             "搜索缓存关闭，配对顺序按种子随机。每次 MCP 搜索包括连接、工具发现、工具调用和关闭会话。",
             "Bocha 保留生产重试/限流/熔断策略；MCP 不自动重试。单次耗时包含这些策略。",
             "MCP 当前只接受 query，本地截取前 K 条；Bocha 向服务端请求 K 条。实际返回数保存在原始结果。", "",
             "```json", json.dumps(info, ensure_ascii=False, indent=2), "```", "",
             "## 原始结果", "", f"- [{raw_path.name}]({raw_path})",
             f"- [评估汇总.json]({run_dir / '评估汇总.json'})", "",
             "## 已测量的核心指标", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
             "## 逐题结果及失败样例", "", "| 轮次 | 查询 | 后端 | 状态 | 合格 URL 数 | 耗时 ms | 错误类型 |",
             "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for row in rows:
        query = row["query"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['round']} | {query} | {row['provider']} | {row['status']} | "
                     f"{len(row['results'])} | {row['duration_ms']} | {row.get('error_type', '')} |")
    if not rows:
        lines.append("未执行搜索对比；工具发现结果见原始文件。")
    lines.extend(["", "## 局限性、结论与下一步", "",
                  "已测量：调用是否成功、空结果、过滤后 URL 数量、端到端搜索延迟，以及原始请求/响应。",
                  "间接推断：统一契约允许两种结果复用已有正文阅读和证据抽取流程；这不是报告质量结论。",
                  "尚未评估：Precision@K/nDCG、来源权威性、正文可读率、最终报告引用质量和真实费用。",
                  "费用字段为 null，需用阿里/Bocha 账单对齐真实调用量，不凭延迟或结果数估算。",
                  "不能仅按结果数或单题耗时判定谁更好。建议用相同的 20–50 个医学研究问题，人工标注"
                  "相关性和来源等级，再以相同模型、提示词、网站范围、搜索预算分别生成报告。",
                  "这项比较改变了搜索服务，不能把差异全部归因于 MCP 协议。", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


async def run(args):
    now = datetime.now(ZoneInfo("Asia/Singapore"))
    run_dir = args.output_dir or BACKEND / "eval" / "data" / f"搜索后端对比-{now:%Y%m%d-%H%M%S-%f}"
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = LLMConfig()
    info = {"date": now.isoformat(), "count": args.count, "repeat": args.repeat, "seed": args.seed,
            "allowed_domains": args.allowed_domain,
            "cache_enabled": False, "mcp_sdk_version": None, "status": "running"}
    info["mcp_connection_mode"] = MCP_CONNECTION_MODE
    rows = []
    raw = run_dir / "逐次搜索结果.jsonl"
    raw.touch()
    try:
        info["allowed_domains"] = normalize_allowed_domains(args.allowed_domain)
        info["mcp_sdk_version"] = version("mcp")
        queries = list(args.query or [])
        if args.queries:
            supplied = json.loads(args.queries.read_text(encoding="utf-8"))
            if not isinstance(supplied, list):
                raise ValueError("查询文件必须是 JSON 字符串数组")
            queries.extend(supplied)
        if any(not isinstance(q, str) or not q.strip() or len(q) > 500 for q in queries):
            raise ValueError("每个查询必须是 1–500 字符的非空字符串")
        info["queries"] = queries
        if args.discover:
            client = AliyunSearchMCP(config.aliyun_search_mcp_api_key, url=config.aliyun_search_mcp_url,
                                     tool_name=config.aliyun_search_mcp_tool_name)
            tools = await client.discover_tools()
            with raw.open("a", encoding="utf-8") as out:
                out.write(json.dumps({"action": "list_tools", "tools": tools}, ensure_ascii=False) + "\n")
            print(json.dumps(tools, ensure_ascii=False, indent=2))
        if queries:
            scouts = {provider: RecordingScout(
                config.api_key or "search-only-no-llm", config.base_url, config.search_api_key,
                search_provider=provider, aliyun_search_mcp_api_key=config.aliyun_search_mcp_api_key,
                aliyun_search_mcp_url=config.aliyun_search_mcp_url,
                aliyun_search_mcp_tool_name=config.aliyun_search_mcp_tool_name,
            ) for provider in ("bocha", "aliyun_mcp")}
            for scout in scouts.values():
                scout.search_cache_enabled = False
            rng = random.Random(args.seed)
            for round_number in range(1, args.repeat + 1):
                for query in queries:
                    order = list(scouts)
                    rng.shuffle(order)
                    for provider in order:
                        scout = scouts[provider]
                        scout.search_calls = []
                        started = time.perf_counter()
                        row = {"round": round_number, "query": query, "provider": provider,
                               "status": "ok", "results": []}
                        state = {"messages": [], "allowed_domains": info["allowed_domains"]}
                        try:
                            row["results"] = await scout._execute_search(query, args.count, state=state)
                            if any(call.get("status") != "ok" for call in scout.search_calls):
                                row["status"] = "error"
                                row["error_type"] = "SearchBackendFailure"
                        except Exception as exc:
                            row.update(status="error", error_type=type(exc).__name__)
                        row.update(duration_ms=round((time.perf_counter() - started) * 1000, 2),
                                   search_calls=scout.search_calls, search_budget=state.get("search_budget"))
                        rows.append(row)
                        with raw.open("a", encoding="utf-8") as out:
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(f"{provider}: {row['status']}, {len(row['results'])} URLs, {row['duration_ms']} ms")
        info["status"] = "completed_with_errors" if any(r["status"] == "error" for r in rows) else "completed"
    except BaseException as exc:
        info.update(status="failed_or_interrupted", error_type=type(exc).__name__)
        raise
    finally:
        print(f"报告：{save_report(run_dir, info, rows)}")
    return 1 if any(r["status"] == "error" for r in rows) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true", help="仅发现 MCP 工具，无查询时不执行搜索")
    parser.add_argument("--query", action="append", help="待比较的查询，可重复指定")
    parser.add_argument("--queries", type=Path, help="JSON 字符串数组文件")
    parser.add_argument("--count", type=int, choices=range(1, 21), default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed-domain", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, help="新建的结果目录，必须尚不存在")
    args = parser.parse_args()
    if args.repeat < 1 or len(args.allowed_domain) > 20:
        parser.error("repeat 至少为 1；单题网站范围最多 20 个")
    if not (args.discover or args.query or args.queries):
        parser.error("请指定 --discover、--query 或 --queries")
    logging.basicConfig(level=logging.WARNING)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
