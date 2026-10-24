#!/usr/bin/env python3
"""Claude-based local RACE + FACT-compatible evaluation for DRB reports.

RACE follows the official reference-relative, adaptive-criteria calculation.
FACT follows the official claim/URL -> scrape -> support judgment structure,
and uses the project's local WebReaderService (static HTML, PDF, Crawl4AI and
cache) instead of Jina Reader.  Therefore results are useful local iteration
scores, not an official leaderboard submission.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from json_repair import repair_json


BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from service.web_reader_service import WebReaderService

DATA_DIR = BACKEND_DIR / "eval/data/deepresearch_bench"
OFFICIAL_DIR = DATA_DIR / "official"
OFFICIAL_BASE = "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/main"
ASSETS = {
    "criteria.jsonl": f"{OFFICIAL_BASE}/data/criteria_data/criteria.jsonl",
    "reference.jsonl": f"{OFFICIAL_BASE}/data/test_data/cleaned_data/reference.jsonl",
}


def ensure_assets() -> None:
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        target = OFFICIAL_DIR / name
        if target.exists() and target.stat().st_size > 1000:
            continue
        response = httpx.get(url, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ClaudeJudge:
    def __init__(self) -> None:
        load_dotenv(BACKEND_DIR / ".env")
        self.api_key = os.getenv("CLAUDE_RELAY_API_KEY", "")
        self.base_url = os.getenv("CLAUDE_RELAY_BASE_URL", "").rstrip("/")
        self.model = os.getenv("CLAUDE_RELAY_MODEL", "claude-opus-5")
        if not self.api_key or not self.base_url:
            raise RuntimeError("缺少 CLAUDE_RELAY_API_KEY / CLAUDE_RELAY_BASE_URL")
        self.url = (
            f"{self.base_url}/messages"
            if self.base_url.endswith("/v1")
            else f"{self.base_url}/v1/messages"
        )

    def json(self, prompt: str, *, max_tokens: int = 20000) -> object:
        last_error = "unknown"
        timeout = httpx.Timeout(connect=20, read=180, write=60, pool=20)
        for attempt in range(1, 4):
            for trust_env in (False, True):
                try:
                    with httpx.Client(trust_env=trust_env, timeout=timeout) as client:
                        payload = {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "stream": True,
                            "messages": [{"role": "user", "content": prompt}],
                        }
                        with client.stream(
                            "POST",
                            self.url,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json",
                            },
                            json=payload,
                        ) as response:
                            if response.status_code != 200:
                                error_body = response.read().decode("utf-8", errors="replace")
                                raise RuntimeError(f"HTTP {response.status_code}: {error_body[:500]}")
                            text_parts = []
                            for line in response.iter_lines():
                                if not line.startswith("data:"):
                                    continue
                                data_text = line[5:].strip()
                                if not data_text or data_text == "[DONE]":
                                    continue
                                event = json.loads(data_text)
                                if event.get("type") == "content_block_delta":
                                    delta = event.get("delta") or {}
                                    if delta.get("type") == "text_delta":
                                        text_parts.append(delta.get("text", ""))
                                elif event.get("type") == "error":
                                    raise RuntimeError(f"stream error: {event}")
                            text = "".join(text_parts)
                            if not text:
                                raise RuntimeError("Claude stream returned no text")
                    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
                    candidate = fenced.group(1) if fenced else text
                    return json.loads(repair_json(candidate))
                except Exception as exc:
                    last_error = f"trust_env={trust_env} {type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 15))
        raise RuntimeError(f"Claude JSON 调用失败: {last_error}")


def clean_for_race(article: str) -> str:
    article = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", article)
    article = re.sub(r"\[(?:\d+|\d+(?:\s*,\s*\d+)+)\]", "", article)
    source_heading = re.search(
        r"(?im)^#{1,6}[^\n]*(?:来源索引|参考文献|references|sources)\s*$", article
    )
    if source_heading:
        article = article[: source_heading.start()]
    return article.strip()


def format_criteria(criteria: dict) -> str:
    output = {}
    for dimension, items in criteria["criterions"].items():
        output[dimension] = [
            {"criterion": item["criterion"], "explanation": item["explanation"]}
            for item in items
        ]
    return json.dumps(output, ensure_ascii=False, indent=2)


def race_prompt(task: str, target: str, reference: str, criteria: dict) -> str:
    return f"""你是一名严格、细致、客观的调研文章评估专家。请比较两篇针对同一任务的文章。

<task>{task}</task>
<article_1>{target}</article_1>
<article_2>{reference}</article_2>
<criteria_list>{format_criteria(criteria)}</criteria_list>

必须逐条覆盖 criteria_list 中的每条标准。每条标准分别给两篇文章 0-10 连续分数：
0-2 很差，2-4 较差，4-6 中等，6-8 较好，8-10 出色。不得因为文章更长而自动给高分，
必须检查事实与数据支持、任务子要求覆盖、推理深度和可读性。

只返回合法 JSON 对象，键必须是 comprehensiveness、insight、instruction_following、readability；
每个键对应数组，每项格式：
{{"criterion":"原标准文本","analysis":"简洁对比理由","article_1_score":0到10,"article_2_score":0到10}}
"""


def calculate_race(raw: dict, criteria: dict) -> dict:
    dimension_weights = criteria["dimension_weight"]
    target_total = reference_total = 0.0
    dimensions = {}
    for dimension in ("comprehensiveness", "insight", "instruction_following", "readability"):
        expected = criteria["criterions"].get(dimension, [])
        returned = raw.get(dimension, [])
        by_text = {str(item.get("criterion", "")).strip(): item for item in returned if isinstance(item, dict)}
        t_sum = r_sum = weight_sum = 0.0
        for expected_item in expected:
            text = expected_item["criterion"].strip()
            match = by_text.get(text)
            if match is None:
                match = next(
                    (value for key, value in by_text.items() if text in key or key in text),
                    None,
                )
            if match is None:
                continue
            weight = float(expected_item["weight"])
            t_sum += float(match["article_1_score"]) * weight
            r_sum += float(match["article_2_score"]) * weight
            weight_sum += weight
        target_avg = t_sum / weight_sum if weight_sum else 0.0
        reference_avg = r_sum / weight_sum if weight_sum else 0.0
        normalized = target_avg / (target_avg + reference_avg) if target_avg + reference_avg else 0.0
        dimensions[dimension] = {
            "target": round(target_avg, 4),
            "reference": round(reference_avg, 4),
            "normalized": round(normalized, 4),
        }
        dimension_weight = float(dimension_weights[dimension])
        target_total += target_avg * dimension_weight
        reference_total += reference_avg * dimension_weight
    overall = target_total / (target_total + reference_total) if target_total + reference_total else 0.0
    return {
        "overall_score": round(overall, 4),
        "target_weighted": round(target_total, 4),
        "reference_weighted": round(reference_total, 4),
        "dimensions": dimensions,
        "raw_judgment": raw,
    }


LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def body_before_sources(article: str) -> str:
    source_heading = re.search(
        r"(?im)^#{1,6}[^\n]*(?:来源索引|参考文献|references|sources)\s*$", article
    )
    return article[: source_heading.start()] if source_heading else article


def sentence_for_link(article: str, start: int, end: int) -> str:
    # A citation usually supports its nearest clause.  Using the entire
    # sentence incorrectly assigns an adjacent claim (and its other URL) to
    # this source, especially in Chinese prose joined by commas/semicolons.
    boundaries = ("。", "！", "？", "；", ";", "，", ",", "\n")
    left = max(article.rfind(mark, 0, start) for mark in boundaries) + 1
    right_candidates = [article.find(mark, end) for mark in boundaries]
    right_candidates = [pos for pos in right_candidates if pos >= 0]
    right = min(right_candidates) + 1 if right_candidates else min(len(article), end + 300)
    sentence = article[left:right].strip()
    return LINK_RE.sub(lambda match: match.group(1), sentence)[:1200]


def extract_citations(article: str) -> list[dict]:
    article = body_before_sources(article)
    seen = set()
    citations = []
    for match in LINK_RE.finditer(article):
        url = match.group(2).rstrip(".,;，。；）)]")
        fact = sentence_for_link(article, match.start(), match.end())
        key = (fact, url)
        if len(fact) < 8 or key in seen:
            continue
        seen.add(key)
        citations.append({"fact": fact, "url": url, "label": match.group(1)})
    return citations


def fetch_source(url: str) -> dict:
    """Read one citation through the same local reader used by DeepResearch."""
    try:
        result = asyncio.run(WebReaderService().read_url(url))
        usable = result.success and result.failure_category is None
        return {
            "url": url,
            "content": result.content if usable else "",
            "extractor": result.extractor or None,
            "error": None if usable else result.error,
            "fetch_status": result.fetch_status,
            "failure_category": result.failure_category,
            "http_status": result.http_status,
            "recovery_action": result.recovery_action,
            "from_cache": result.from_cache,
        }
    except Exception as exc:
        return {
            "url": url,
            "content": "",
            "extractor": None,
            "error": f"local_reader:{type(exc).__name__}:{exc}",
            "fetch_status": "failed",
            "failure_category": "extraction_error",
            "http_status": None,
            "recovery_action": "",
            "from_cache": False,
        }


def fact_validate_prompt(reference: str, facts: list[str]) -> str:
    statements = "\n".join(f"{index + 1}. {fact}" for index, fact in enumerate(facts))
    return f"""你是严格的引用核验员。判断每条 statement 相对于 reference 是 supported、unsupported 或 unknown。
规则：若页面无有效内容则全部 unknown；若 statement 的关键事实或数据可在页面中全部或部分找到（允许合理四舍五入），
则 supported；有效页面中完全找不到则 unsupported。不要用你自己的常识补证。

<reference>{reference}</reference>
<statements>{statements}</statements>

只返回 JSON 数组，长度必须等于 statements 数量：
[{{"idx":1,"result":"supported|unsupported|unknown","reason":"一句话依据"}}]
"""


def relevant_reference(content: str, facts: list[str], max_chars: int = 24000) -> str:
    """Select page chunks that overlap claim terms, keeping FACT prompts small."""
    if len(content) <= max_chars:
        return content
    facts_text = " ".join(facts)
    latin_or_number = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|\d+(?:\.\d+)?%?", facts_text))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", facts_text))
    bigrams = {chinese[index:index + 2] for index in range(max(0, len(chinese) - 1))}
    tokens = latin_or_number | bigrams
    chunks = [content[index:index + 3000] for index in range(0, len(content), 3000)]
    ranked = sorted(
        enumerate(chunks),
        key=lambda pair: (sum(token in pair[1] for token in tokens), -pair[0]),
        reverse=True,
    )
    selected_indices = {0}
    for index, _ in ranked:
        selected_indices.add(index)
        if sum(len(chunks[pos]) for pos in selected_indices) >= max_chars:
            break
    return "\n\n".join(chunks[index] for index in sorted(selected_indices))[:max_chars]


def evaluate_fact(article: str, judge: ClaudeJudge, *, fetch_workers: int) -> dict:
    citations = extract_citations(article)
    by_url: dict[str, list[dict]] = {}
    for citation in citations:
        by_url.setdefault(citation["url"], []).append(citation)
    scraped = {}
    with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
        futures = {pool.submit(fetch_source, url): url for url in by_url}
        for future in as_completed(futures):
            item = future.result()
            scraped[item["url"]] = item

    judgments = []
    for url, items in by_url.items():
        page = scraped[url]
        scrape_meta = {
            "extractor": page["extractor"],
            "fetch_status": page["fetch_status"],
            "failure_category": page["failure_category"],
            "http_status": page["http_status"],
            "recovery_action": page["recovery_action"],
            "from_cache": page["from_cache"],
        }
        if not page["content"]:
            for item in items:
                judgments.append({
                    **item,
                    "result": "unknown",
                    "reason": page["error"] or page["failure_category"] or "local reader failed",
                    **scrape_meta,
                })
            continue
        facts = [item["fact"] for item in items]
        reference = relevant_reference(page["content"], facts)
        try:
            raw = judge.json(fact_validate_prompt(reference, facts), max_tokens=6000)
        except Exception as exc:
            for item in items:
                judgments.append({
                    **item,
                    "result": "unknown",
                    "reason": f"judge_error:{type(exc).__name__}:{exc}",
                    **scrape_meta,
                })
            continue
        by_idx = {int(item.get("idx", 0)) - 1: item for item in raw if isinstance(item, dict)}
        for index, item in enumerate(items):
            verdict = by_idx.get(index, {})
            result = verdict.get("result", "unknown")
            if result not in {"supported", "unsupported", "unknown"}:
                result = "unknown"
            judgments.append({
                **item,
                "result": result,
                "reason": verdict.get("reason", ""),
                **scrape_meta,
            })
    non_unknown = sum(item["result"] != "unknown" for item in judgments)
    supported = sum(item["result"] == "supported" for item in judgments)
    accuracy = supported / non_unknown if non_unknown else 0.0
    extractor_counts = Counter(
        item["extractor"] or "none" for item in scraped.values()
    )
    failure_counts = Counter(
        item["failure_category"] for item in scraped.values()
        if item["failure_category"]
    )
    return {
        "extracted_citations": len(citations),
        "unique_urls": len(by_url),
        "verifiable_citations": non_unknown,
        "supported_citations": supported,
        "citation_accuracy": round(accuracy, 4),
        "scrape_success_urls": sum(bool(item["content"]) for item in scraped.values()),
        "scrape_extractors": dict(sorted(extractor_counts.items())),
        "scrape_failure_categories": dict(sorted(failure_counts.items())),
        "scrape_urls": [scraped[url] | {"content": ""} for url in sorted(scraped)],
        "judgments": judgments,
    }


def write_summary(path: Path, source: Path, results: list[dict], judge_model: str) -> None:
    results = [item for item in results if item.get("race") and item.get("fact")]
    lines = [
        "# 前五题 Claude 兼容 RACE+FACT 评估",
        "",
        f"- 输入：`{source}`",
        f"- 评审模型：`{judge_model}`",
        "- FACT 网页读取：本地 WebReaderService（静态 HTML / PDF / Crawl4AI / 缓存），不使用 Jina Reader。",
        "- 本地通过门槛：RACE ≥ 0.45，FACT 引用准确率 ≥ 0.80，且 supported 引用 ≥ 10。该门槛不是官方榜单规则。",
        "",
        "| 题目 | RACE | FACT准确率 | supported/可核验/提取 | 抓取URL | 本地门禁 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for item in sorted(results, key=lambda value: value["id"]):
        race = item["race"]["overall_score"]
        fact = item["fact"]
        passed = race >= 0.45 and fact["citation_accuracy"] >= 0.8 and fact["supported_citations"] >= 10
        lines.append(
            f"| {item['id']} | {race:.4f} | {fact['citation_accuracy']:.1%} | "
            f"{fact['supported_citations']}/{fact['verifiable_citations']}/{fact['extracted_citations']} | "
            f"{fact['scrape_success_urls']}/{fact['unique_urls']} | {'PASS' if passed else 'FAIL'} |"
        )
    passed_count = sum(
        item["race"]["overall_score"] >= 0.45
        and item["fact"]["citation_accuracy"] >= 0.8
        and item["fact"]["supported_citations"] >= 10
        for item in results
    )
    lines += ["", f"总通过：{passed_count}/{len(results)}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-race-from", type=Path,
                        help="可选：复用另一评测文件里已完成的 RACE，重新跑 FACT")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--fetch-workers", type=int, default=6)
    parser.add_argument(
        "--force-ids", type=int, nargs="*", default=[],
        help="重新评估指定题号；保留已有 RACE，只覆盖对应 FACT",
    )
    args = parser.parse_args()

    ensure_assets()
    judge = ClaudeJudge()
    reports = [item for item in load_jsonl(args.input) if item.get("article")][: args.limit]
    criteria_map = {int(item["id"]): item for item in load_jsonl(OFFICIAL_DIR / "criteria.jsonl")}
    reference_map = {int(item["id"]): item for item in load_jsonl(OFFICIAL_DIR / "reference.jsonl")}
    output = args.output or args.input.with_suffix(".claude-race-fact.json")
    existing = []
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8")).get("results", [])
        except Exception:
            existing = []
    elif args.reuse_race_from and args.reuse_race_from.exists():
        try:
            reused = json.loads(args.reuse_race_from.read_text(encoding="utf-8")).get("results", [])
            existing = [
                {"id": item["id"], "prompt": item.get("prompt", ""), "race": item.get("race"), "fact": None}
                for item in reused
                if item.get("race") is not None
            ]
        except Exception:
            existing = []
    done = {
        int(item["id"])
        for item in existing
        if item.get("race") is not None
        and item.get("fact") is not None
        and int(item["id"]) not in set(args.force_ids)
    }
    results = list(existing)

    for index, report in enumerate(reports, 1):
        task_id = int(report["id"])
        if task_id in done:
            print(f"[{index}/{len(reports)}] id={task_id} resume-skip")
            continue
        partial = next((item for item in results if int(item["id"]) == task_id), None)
        if partial and partial.get("race") is not None:
            race = partial["race"]
            print(f"[{index}/{len(reports)}] id={task_id} RACE resume", flush=True)
        else:
            print(f"[{index}/{len(reports)}] id={task_id} RACE", flush=True)
            criteria = criteria_map[task_id]
            reference = reference_map[task_id]["article"]
            target = clean_for_race(report["article"])
            raw_race = judge.json(race_prompt(report["prompt"], target, reference, criteria))
            race = calculate_race(raw_race, criteria)
            partial = {"id": task_id, "prompt": report["prompt"], "race": race, "fact": None}
            results = [item for item in results if int(item["id"]) != task_id] + [partial]
            output.write_text(
                json.dumps({"judge_model": judge.model, "input": str(args.input), "results": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(f"[{index}/{len(reports)}] id={task_id} FACT", flush=True)
        fact = evaluate_fact(report["article"], judge, fetch_workers=args.fetch_workers)
        completed_item = {"id": task_id, "prompt": report["prompt"], "race": race, "fact": fact}
        results = [item for item in results if int(item["id"]) != task_id] + [completed_item]
        output.write_text(
            json.dumps({"judge_model": judge.model, "input": str(args.input), "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_summary(output.with_suffix(".md"), args.input, results, judge.model)
        print(
            f"[{index}/{len(reports)}] id={task_id} race={race['overall_score']:.4f} "
            f"fact={fact['citation_accuracy']:.1%} supported={fact['supported_citations']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
