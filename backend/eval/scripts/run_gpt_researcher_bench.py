#!/usr/bin/env python3
"""Run GPT Researcher on a resumable subset of DeepResearch Bench.

Generation uses the project's existing DashScope account, while search uses
Bocha.  Evaluation is intentionally kept in a separate script so generation
artifacts remain reusable with different judges.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = BACKEND_DIR / "eval/data/deepresearch_bench/query.jsonl"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "eval/data/deepresearch_bench/gpt_researcher"


def configure_environment() -> None:
    load_dotenv(BACKEND_DIR / ".env")

    required = ("BOCHA_API_KEY", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"缺少环境变量: {', '.join(missing)}")

    # GPT Researcher supports OpenAI-compatible endpoints.  Reuse DashScope
    # without changing the project's regular environment variables.
    os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = os.environ["DASHSCOPE_BASE_URL"]
    generation_model = os.getenv("GPTR_GENERATION_MODEL", os.getenv("OPENAI_MODEL", "qwen3-max"))

    defaults = {
        "RETRIEVER": "bocha",
        "SCRAPER": "bs",
        "FAST_LLM": f"openai:{generation_model}",
        "SMART_LLM": f"openai:{generation_model}",
        "STRATEGIC_LLM": f"openai:{generation_model}",
        "EMBEDDING": "openai:text-embedding-v4",
        # LangChain otherwise tokenizes first and sends integer arrays, while
        # DashScope's OpenAI-compatible embeddings endpoint accepts strings.
        "EMBEDDING_KWARGS": '{"check_embedding_ctx_length": false, "chunk_size": 10}',
        "LANGUAGE": "chinese",
        "REPORT_FORMAT": "APA",
        "TOTAL_WORDS": "4500",
        "FAST_TOKEN_LIMIT": "8000",
        "SMART_TOKEN_LIMIT": "20000",
        "STRATEGIC_TOKEN_LIMIT": "10000",
        "SUMMARY_TOKEN_LIMIT": "1200",
        "BROWSE_CHUNK_MAX_LENGTH": "12000",
        "MAX_SEARCH_RESULTS_PER_QUERY": "7",
        "MAX_SCRAPER_WORKERS": "8",
        "CURATE_SOURCES": "true",
        "DEEP_RESEARCH_BREADTH": "4",
        "DEEP_RESEARCH_DEPTH": "1",
        "DEEP_RESEARCH_CONCURRENCY": "2",
        "IMAGE_GENERATION_ENABLED": "false",
        "MCP_STRATEGY": "disabled",
        "VERBOSE": "false",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


_bocha_lock = threading.Lock()
_bocha_last_request = 0.0


def _source_score(url: str, original_rank: int) -> float:
    """Prefer primary/authoritative sources without hard-blocking fallbacks."""
    host = (urlparse(url).hostname or "").lower()
    path = urlparse(url).path.lower()
    score = -original_rank * 0.02
    if host.endswith(".gov.cn") or host in {
        "stats.gov.cn", "www.stats.gov.cn", "pbc.gov.cn", "www.pbc.gov.cn"
    }:
        score += 5.0
    if any(token in host for token in ("gov", "edu", "cssn", "worldbank", "imf", "oecd")):
        score += 2.0
    if path.endswith(".pdf") or "annual-report" in path or "annual_report" in path:
        score += 1.5
    if any(token in host for token in (
        "toutiao", "sohu", "163.com", "docin", "doc88", "taodocs",
        "book118", "csdn", "yebaike", "baike", "wanyiwang",
    )):
        score -= 3.0
    return score


def patch_bocha_retriever() -> None:
    """Add pacing, retry and source ranking to GPT Researcher's Bocha adapter."""
    import httpx
    from gpt_researcher.retrievers.bocha.bocha import BoChaSearch

    if getattr(BoChaSearch, "_industry_assistant_patched", False):
        return

    def search(self, max_results: int = 7):
        global _bocha_last_request
        requested = min(max(max_results * 2, 10), 20)
        payload = {
            "query": self.query,
            "freshness": "noLimit",
            "summary": True,
            "count": requested,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = "unknown"
        for attempt in range(1, 6):
            with _bocha_lock:
                delay = 1.25 - (time.monotonic() - _bocha_last_request)
                if delay > 0:
                    time.sleep(delay)
                try:
                    response = httpx.post(
                        "https://api.bochaai.com/v1/web-search",
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )
                    _bocha_last_request = time.monotonic()
                except Exception as exc:
                    _bocha_last_request = time.monotonic()
                    last_error = f"{type(exc).__name__}: {exc}"
                    response = None

            if response is not None and response.status_code == 200:
                body = response.json()
                values = ((((body or {}).get("data") or {}).get("webPages") or {}).get("value") or [])
                normalized = [
                    {
                        "title": item.get("name", ""),
                        "href": item.get("url", ""),
                        "body": item.get("summary") or item.get("snippet", ""),
                    }
                    for item in values
                    if item.get("url")
                ]
                ranked = sorted(
                    enumerate(normalized),
                    key=lambda pair: _source_score(pair[1]["href"], pair[0]),
                    reverse=True,
                )
                return [item for _, item in ranked[:max_results]]

            if response is not None:
                last_error = f"HTTP {response.status_code}: {response.text[:160]}"
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 20)
            else:
                wait = min(2 ** attempt, 20)
            time.sleep(wait)
        print(f"Bocha search failed after retries: {last_error}", file=sys.stderr)
        return []

    BoChaSearch.search = search
    BoChaSearch._industry_assistant_patched = True


def load_tasks(
    path: Path,
    *,
    language: str,
    topic: str | None,
    limit: int,
    task_ids: set[int] | None = None,
) -> list[dict]:
    tasks: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if task_ids and int(item.get("id", -1)) not in task_ids:
                continue
            if language and item.get("language") != language:
                continue
            if topic and topic.lower() not in str(item.get("topic", "")).lower():
                continue
            tasks.append(item)
            if len(tasks) >= limit:
                break
    return tasks


def load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("article") and not item.get("error"):
                completed.add(int(item["id"]))
    return completed


def report_prompt(query: str, target_words: int) -> str:
    return f"""请依据后附的研究上下文，完成下面的中文深度研究任务：

{query}

硬性要求：
1. 直接完整回答任务的所有子问题，不要把问题改写成泛泛综述。
2. 全文不少于 {target_words} 个中文词语或等量信息密度；优先保证事实、数据、比较表和推理质量。
3. 每个可外部核验的关键数字和事实，必须在同一句或紧邻句使用 Markdown 行内链接引用，格式为 [来源标题](https://完整URL)。不得只在文末堆链接。
4. 明确区分已核验事实、计算结果、假设和预测；计算需给出公式、口径、年份与原始数值。
5. 涉及横向比较时必须覆盖题目要求的全部对象和全部维度；缺失值标为“未获得可靠数据”，不得臆造。
6. 优先使用监管机构、政府统计、公司年报、交易所、国际组织和学术论文；说明数据的发布日期及适用期。
7. 使用清晰标题、Markdown 表格、结论、风险与局限。最后附“来源索引”，但正文仍须保留行内引用。
8. 不得出现占位符、未定义变量、伪造引文或无法追溯的精确数字。
"""


def research_prompt(query: str) -> str:
    return f"""研究任务：{query}

检索执行要求（这些要求用于搜集证据，不是要在报告里复述）：
- 当前日期按 2026 年处理，但每项数据必须保留其真实统计期和发布日期，不能把旧数据改写成 2026 年数据。
- 先把任务拆成“对象 × 指标 × 年份 × 所需原始证据”的证据矩阵，再为每个缺口生成具体检索词。
- 优先检索政府统计、监管机构、公司年报/财报、交易所公告、国际组织、大学和同行评议论文；搜索词应包含机构或公司全名，禁止使用“某公司”等占位符。
- 对精确数字至少寻找原始发布页或可下载报告；搜索摘要只能作线索，必须尽量读取落地页正文。
- 降低聚合转载、自媒体、文档预览/付费下载站的优先级；只有找不到一手来源时才作为补充，并显式保留不确定性。
- 横向比较任务必须逐个覆盖全部对象与全部指标，不能只研究排名靠前的少数对象。
"""


def task_research_prompt(task: dict) -> str:
    prompt = research_prompt(task["prompt"])
    task_id = int(task["id"])
    if task_id == 2:
        prompt += """

本题专项矩阵：固定逐家研究 Allianz、Berkshire Hathaway、China Life、Ping An、
Prudential Financial、AXA、MetLife、Legal & General、Manulife、Nippon Life。
每家公司至少检索一份官方年报/财报，并分别寻找：融资或资本结构、信用评级原始页、
过去五年可比增长指标、实际现金分红、在华牌照/实体/战略。任何精确比率必须来自能定位
该数字的具体页面或 PDF，不得把评级机构首页、公司首页当作数字证据；找不到就写缺失，
禁止按照行业常识补数。最终候选推荐必须由表中已核验指标推导。
"""
    elif task_id == 3:
        prompt += """

本题专项矩阵：分别研究投行、PE/VC、固定收益、财富管理/资管、金融科技、绿色金融。
每个领域至少获取市场规模或业务量、近年增速、监管政策、人才需求、盈利/风险约束中的
三类原始证据。建立统一评分模型并公开权重和原始输入；最终只能推荐证据覆盖充分的领域。
重点检索人民银行、金融监管总局、证监会、基金业协会、银行间市场交易商协会、国家统计局
及上市金融机构年报，不能用培训机构文章代替行业规模数据。
"""
    elif task_id == 4:
        prompt += """

本题专项矩阵：历史区间必须覆盖 2010-01-01 至当前日期 2026-08-23，并把实际历史与
2026-08-23 之后的预测严格分开。优先检索 LBMA、World Gold Council、CME、FRED、
各国央行和权威行情数据；关键压力/支撑必须写明所依据的数据截止日和计算方法。
如果只能取得截至2024年的来源，不得声称“分析至今”。思维导图必须与正文情景和价位一致。
"""
    elif task_id == 5:
        prompt += """

本题专项矩阵：覆盖银行间拆借、同业存单/同业负债、回购、债券持仓、资管产品、
非银机构与影子银行六类关系；分别给出网络模型、CoVaR/MES/SRISK等尾部模型及压力测试
的变量、方程、数据源和适用边界。优先使用人民银行、金融监管总局、交易商协会、清算所、
中债登及同行评议论文。
"""
    return prompt


async def run_task(task: dict, report_type: str, target_words: int, verbose: bool) -> dict:
    from gpt_researcher import GPTResearcher

    patch_bocha_retriever()
    started = time.time()
    researcher = GPTResearcher(
        query=task_research_prompt(task),
        report_type=report_type,
        verbose=verbose,
        mcp_strategy="disabled",
    )
    await researcher.conduct_research()
    article = await researcher.write_report(
        custom_prompt=report_prompt(task["prompt"], target_words)
    )
    urls = sorted(str(url) for url in researcher.visited_urls)
    return {
        "id": task["id"],
        "topic": task.get("topic"),
        "language": task.get("language"),
        "prompt": task["prompt"],
        "article": article,
        "generator": "GPT Researcher",
        "generation_model": os.getenv("GPTR_GENERATION_MODEL", os.getenv("OPENAI_MODEL", "qwen3-max")),
        "retriever": "bocha",
        "scraper": "bs",
        "report_type": report_type,
        "visited_urls": urls,
        "elapsed_seconds": round(time.time() - started, 2),
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--ids", help="只跑指定题号，逗号分隔，例如 2,4")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--topic", default="Finance")
    parser.add_argument("--report-type", choices=("deep", "research_report"), default="deep")
    parser.add_argument("--target-words", type=int, default=4500)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_environment()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or DEFAULT_OUTPUT_DIR / f"gpt-researcher-5q-{stamp}.jsonl"
    task_ids = {int(value) for value in args.ids.split(",")} if args.ids else None
    tasks = load_tasks(
        args.dataset,
        language=args.language,
        topic=args.topic,
        limit=args.limit,
        task_ids=task_ids,
    )
    completed = load_completed(output)
    print(f"output={output}")
    print(f"selected={len(tasks)} completed={len(completed)} report_type={args.report_type}")

    for index, task in enumerate(tasks, 1):
        if int(task["id"]) in completed:
            print(f"[{index}/{len(tasks)}] id={task['id']} resume-skip")
            continue
        print(f"[{index}/{len(tasks)}] id={task['id']} start", flush=True)
        try:
            result = asyncio.run(run_task(task, args.report_type, args.target_words, args.verbose))
            append_jsonl(output, result)
            markdown_path = output.parent / f"{output.stem}-q{task['id']}.md"
            markdown_path.write_text(result["article"], encoding="utf-8")
            print(
                f"[{index}/{len(tasks)}] id={task['id']} ok "
                f"chars={len(result['article'])} urls={len(result['visited_urls'])} "
                f"seconds={result['elapsed_seconds']}",
                flush=True,
            )
        except Exception as exc:
            error = {
                "id": task["id"],
                "prompt": task["prompt"],
                "error": f"{type(exc).__name__}: {exc}",
                "generated_at": datetime.now().astimezone().isoformat(),
            }
            append_jsonl(output, error)
            print(f"[{index}/{len(tasks)}] id={task['id']} failed: {error['error']}", file=sys.stderr)

    print(f"done output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
