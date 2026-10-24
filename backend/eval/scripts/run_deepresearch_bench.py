#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
DeepResearch Bench 评测 Runner

## 执行入口

通过当前 DeepResearchV2Service.research 执行，复用 PostgreSQL 检查点、
事件账本与终态交付。运行前需完成业务库迁移和检查点建表。
RESEARCH_ALLOWED_DOMAINS 默认留空使用全网；配置后来源限制同样适用于 benchmark。
CLI 运行不关联登录用户或聊天 turn，报告保存在研究事件和本地结果文件中。

## 数据格式（来自官方 README）

输入  data/prompt_data/query.jsonl
      {"id": 1, "topic": "Finance & Business", "language": "zh", "prompt": "..."}
      共 100 条，50 中文 + 50 英文，覆盖 22 个领域

输出  data/test_data/raw_data/<model_name>.jsonl
      {"id": "task_id", "prompt": "original_query_text", "article": "..."}

拿到输出后，把文件放进 DRB 仓库对应目录，将 model_name 加进
run_benchmark.sh 的 TARGET_MODELS，执行 `bash run_benchmark.sh` 即可。
RACE 与 FACT 的评测需要 OpenRouter/OpenAI key 和 Jina key（见官方 README）。

## 用法

    cd backend

    # 1) 下载题目
    python eval/scripts/run_deepresearch_bench.py --download

    # 2) 先跑 2 道确认链路通
    python eval/scripts/run_deepresearch_bench.py --lang zh --limit 2

    # 3) 跑全部 50 道中文题（耗时长，支持断点续跑）
    python eval/scripts/run_deepresearch_bench.py --lang zh --model-name my-agent-v1

    # 4) 查看进度 / 导出
    python eval/scripts/run_deepresearch_bench.py --stat --model-name my-agent-v1

中断后重跑同一条命令即可，已完成的题目会自动跳过。
"""

import os
import sys
import json
import math
import re
import time
import asyncio
import argparse
import logging
import uuid
from pathlib import Path
from contextlib import aclosing, asynccontextmanager
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Any, List

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "app"))

from dotenv import load_dotenv

load_dotenv()

QUERY_URL = (
    "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench"
    "/main/data/prompt_data/query.jsonl"
)

BENCH_DIR = os.path.join(BACKEND_DIR, "eval", "data", "deepresearch_bench")
QUERY_FILE = os.path.join(BENCH_DIR, "query.jsonl")
RAW_DIR = os.path.join(BENCH_DIR, "raw_data")
RUN_NAME_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<count>\d+)q-run(?P<number>\d+)-(?P<stamp>\d{8}_\d{6})$"
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DRB")


def valid_quality_score(value: Any):
    """返回合法的 1-10 分数；历史 -1/0/空值统一视为无效评分。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 1 <= score <= 10:
        return None
    return score


def display_quality_score(meta: Dict[str, Any]) -> str:
    """报告表格中的评分显示，显式标出历史非法值和降级评分。"""
    raw = meta.get("quality_score")
    score = valid_quality_score(raw)
    if score is None:
        return "-" if raw is None else f"无效({raw})"
    text = str(int(score)) if score.is_integer() else f"{score:g}"
    status = meta.get("quality_score_status")
    if status == "fallback_invalid_model_output":
        return f"{text}（降级）"
    if status == "valid_after_retry":
        return f"{text}（重试）"
    if status == "derived_from_dimensions":
        return f"{text}（分维度）"
    if status == "derived_from_issue_severity":
        return f"{text}（问题级别换算）"
    return text


# ---------------------------------------------------------------------------
# 题目加载
# ---------------------------------------------------------------------------

def download_queries() -> bool:
    import urllib.request

    os.makedirs(BENCH_DIR, exist_ok=True)
    print(f"下载题目: {QUERY_URL}")
    try:
        with urllib.request.urlopen(QUERY_URL, timeout=60) as resp:
            content = resp.read().decode("utf-8")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print(f"   可手动下载后放到: {QUERY_FILE}")
        return False

    with open(QUERY_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    tasks = load_queries()
    zh = sum(1 for t in tasks if t.get("language") == "zh")
    print(f"✅ 已保存 {len(tasks)} 道题到 {QUERY_FILE}")
    print(f"   中文 {zh} 道 / 英文 {len(tasks) - zh} 道")
    return True


def load_queries() -> List[Dict[str, Any]]:
    if not os.path.exists(QUERY_FILE):
        return []
    tasks = []
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def filter_tasks(tasks, lang, topic, limit, ids) -> List[Dict[str, Any]]:
    if ids:
        wanted = {str(i) for i in ids}
        tasks = [t for t in tasks if str(t["id"]) in wanted]
    if lang and lang != "all":
        tasks = [t for t in tasks if t.get("language") == lang]
    if topic:
        tasks = [t for t in tasks if topic.lower() in (t.get("topic") or "").lower()]
    if limit:
        tasks = tasks[:limit]
    return tasks


# ---------------------------------------------------------------------------
# 结果文件（JSONL，边跑边追加以支持断点续跑）
# ---------------------------------------------------------------------------

def _source_result_files():
    """列出原始运行文件，排除 RACE/FACT/submit 派生文件。"""
    if not os.path.isdir(RAW_DIR):
        return []
    excluded = (".race.jsonl", ".fact.jsonl", ".submit.jsonl")
    return [
        os.path.join(RAW_DIR, name)
        for name in os.listdir(RAW_DIR)
        if name.endswith(".jsonl") and not name.endswith(excluded)
    ]


def _result_question_count(path: str) -> int:
    """按题号去重计算一次历史运行包含的题目数。"""
    ids = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("id") is not None:
                    ids.add(str(obj["id"]))
    except OSError:
        return 0
    return len(ids)


def next_run_number(question_count: int) -> int:
    """按题量独立编号，首次采用新命名时也计入旧格式历史运行。"""
    generated_numbers = []
    legacy_count = 0
    for path in _source_result_files():
        stem = os.path.basename(path)[:-len(".jsonl")]
        match = RUN_NAME_RE.match(stem)
        if match and int(match.group("count")) == question_count:
            generated_numbers.append(int(match.group("number")))
        elif not match and _result_question_count(path) == question_count:
            legacy_count += 1
    return max(generated_numbers + [legacy_count], default=0) + 1


def build_run_name(prefix: str, question_count: int, run_number: int,
                   stamp: str) -> str:
    """生成短且可读的运行名，确保派生 session_id 不超过 64 字符。"""
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", prefix or "drb").strip("-_")
    safe_prefix = (safe_prefix or "drb")[:12]
    return f"{safe_prefix}-{question_count}q-run{run_number:03d}-{stamp}"


def propose_run_name(prefix: str, question_count: int, stamp: str = None) -> str:
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return build_run_name(prefix, question_count, next_run_number(question_count), stamp)


def reserve_run_name(prefix: str, question_count: int, stamp: str = None) -> str:
    """原子创建本轮结果文件，避免并发启动分配到同一个运行名。"""
    os.makedirs(RAW_DIR, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_number = next_run_number(question_count)
    while True:
        name = build_run_name(prefix, question_count, run_number, stamp)
        try:
            fd = os.open(output_path(name), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            os.close(fd)
            return name
        except FileExistsError:
            run_number += 1


def run_name_parts(run_name: str) -> Dict[str, Any]:
    match = RUN_NAME_RE.match(run_name)
    if not match:
        return {}
    return {
        "prefix": match.group("prefix"),
        "question_count": int(match.group("count")),
        "run_number": int(match.group("number")),
        "stamp": match.group("stamp"),
    }

def output_path(model_name: str) -> str:
    return os.path.join(RAW_DIR, f"{model_name}.jsonl")


def is_hard_pass(record: Dict[str, Any]) -> bool:
    """原生评审通过且分数至少 7，才满足本轮 hard-pass 目标。"""
    meta = record.get("_meta") or {}
    score = valid_quality_score(meta.get("quality_score"))
    return bool(
        record.get("article")
        and meta.get("quality_verdict") == "pass"
        and score is not None
        and score >= 7
    )


def load_done(model_name: str, require_pass: bool = False) -> Dict[str, Dict[str, Any]]:
    """读出最新结果；空文章不算完成，require_pass 时只接受 hard pass。"""
    path = output_path(model_name)
    latest = {}
    if not os.path.exists(path):
        return latest
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[str(obj.get("id"))] = obj

    if require_pass:
        return {task_id: obj for task_id, obj in latest.items() if is_hard_pass(obj)}
    return {task_id: obj for task_id, obj in latest.items() if obj.get("article")}


def append_result(model_name: str, record: Dict[str, Any]) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(output_path(model_name), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 报告后处理
# ---------------------------------------------------------------------------

REF_HEADINGS = ("## 参考文献", "## 参考资料", "## References", "## 引用来源")


def strip_references(article: str) -> str:
    """
    剥掉正文末尾的参考文献段，用于导出 RACE 变体。

    为什么要剥：RACE 的 clean 阶段会把整段引用连同所有 URL 删光
    （实测 139 个 URL → 0），留着只是白付输入 token——
    单题约 4,731 token / $0.024，50 题 $1.18。

    为什么不在生成时就不写：FACT 反过来必须要正文里有 URL，
    否则抽不到「论断-URL」对直接 0 分。生成一次两边都要用，
    所以正文照常带引用，导出时再按评测类型分流。
    """
    if not article:
        return article
    # 取"最早出现"的引用标题作为切点：一篇文章可能有两个参考文献段
    # （报告自带一个 + append_references 又追加一个），从第一个开始剥才能剥干净。
    # 用 find 而非 rfind——rfind 只会剥掉最后那个，前一个仍留在正文里。
    cut = min(
        (i for i in (article.find(h) for h in REF_HEADINGS) if i > 0),
        default=-1,
    )
    return article[:cut].rstrip() if cut > 0 else article


def has_references_section(article: str) -> bool:
    return any(h in (article or "") for h in REF_HEADINGS)


def append_references(article: str, references: List[Dict[str, Any]]) -> str:
    """
    把引用的 URL 附到正文末尾。

    这一步对 FACT 打分是必需的：FACT 从文章正文里抽取「论断-URL」对，
    而本系统的报告正文只有 [1][2] 这样的角标，URL 单独放在 references 字段里。
    不附上 URL，FACT 会一条引用都抽不到，得分归零——
    这不是"刷分"，是把本来就存在的引用放到评测能读到的位置。

    注：writer 的程序化拼接已经会自带「## 参考文献」段，
    此时本函数不会重复追加（下面的 URL 数量判断会挡住）。
    """
    if not references:
        return article

    # 正文里已经有足量 URL 就不重复附加
    if article.count("http") >= max(3, len(references) // 2):
        return article

    lines = ["\n\n## 参考文献\n"]
    for i, ref in enumerate(references, 1):
        url = (ref.get("link") or ref.get("url") or "").strip()
        if not url:
            continue
        title = (ref.get("title") or ref.get("source") or f"来源 {i}").strip()
        lines.append(f"[{ref.get('id', i)}] {title}. {url}")

    if len(lines) == 1:
        return article
    return article + "\n".join(lines)


# ---------------------------------------------------------------------------
# 单题执行
# ---------------------------------------------------------------------------

def make_session_id(model_name: str, run_id: str, task_id: str) -> str:
    """
    构造追踪用的 session_id：drb_<模型名>_<运行时间戳>_<题号>

    为什么要带运行时间戳：早期版本用的是 drb_<模型名>_<题号>，
    重跑同一个 model_name 时两轮会共用同一个 session_id，
    agent_runs 表里新旧数据混在一起——实测踩过这个坑，
    查"本轮最慢调用"时查出的是上一轮遗留的 3547 秒记录。

    加上时间戳后每轮独立，可以用
        WHERE session_id LIKE 'drb_<模型名>_<时间戳>_%'
    精确圈出某一轮。断点续跑不受影响：它是靠输出文件里的题号判断的，
    与 session_id 无关。
    """
    return f"drb_{model_name}_{run_id}_{task_id}"


@asynccontextmanager
async def benchmark_service(max_iterations: int):
    """在整轮评测期间持有生产检查点连接，不在评测中自动迁移数据库。"""
    from service.deep_research_v2.checkpointer import checkpointer_lifespan
    from service.deep_research_v2.service import DeepResearchV2Service

    if not 0 <= max_iterations <= 5:
        raise ValueError("max_iterations 必须在 0 到 5 之间")
    app = SimpleNamespace(state=SimpleNamespace())
    async with checkpointer_lifespan(app) as saver:
        service = DeepResearchV2Service(checkpointer=saver)
        # get_config 返回进程级单例；CLI 覆盖只作用于本轮服务。
        service.config = deepcopy(service.config)
        service.config.research.max_iterations = max_iterations
        service.resolve_allowed_domains()  # 非法域名配置在调用模型前失败；留空表示全网
        yield service


def decode_service_event(chunk: str):
    """服务每次 yield 一个完整 SSE 帧；忽略结束标记及无 data 的帧。"""
    data = "\n".join(
        line[5:].lstrip(" ") for line in chunk.splitlines()
        if line.startswith("data:")
    )
    if not data or data == "[DONE]":
        return None
    event = json.loads(data)
    if not isinstance(event, dict):
        raise ValueError("研究 SSE data 必须为 JSON 对象")
    return event


async def run_one(service, task: Dict[str, Any], model_name: str, run_id: str,
                  timeout: int, no_refs: bool, artifact_dir=None) -> Dict[str, Any]:
    """跑一道题，返回 DRB 要求的记录格式。"""
    task_id = str(task["id"])
    prompt = task["prompt"]
    session_id = make_session_id(model_name, run_id, task_id)
    research_run_id = str(uuid.uuid4())  # 每道题/每次尝试独立，不能共用批次时间戳

    started = time.time()
    final_report = ""
    references: List[Dict[str, Any]] = []
    quality_score = None
    quality_score_status = "not_reviewed"
    quality_verdict = "not_reviewed"
    completion_meta: Dict[str, Any] = {}
    error = None
    artifacts = Path(artifact_dir) if artifact_dir else None
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "运行身份.json").write_text(json.dumps({
            "task": task, "session_id": session_id, "research_run_id": research_run_id,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _consume():
        nonlocal final_report, references, quality_score, quality_score_status, quality_verdict
        stream = service.research(
            query=prompt,
            session_id=session_id,
            run_id=research_run_id,
            resume=False,
            # 评测延续无登录用户的运行方式；无 turn_id，不写聊天消息。
            user_id=None,
        )
        completed = False
        # 解析失败或收到错误事件时也关闭生成器，确保服务释放租约。
        async with aclosing(stream):
            async for chunk in stream:
                if artifacts:
                    with (artifacts / "原始SSE轨迹.jsonl").open("a", encoding="utf-8") as saved:
                        saved.write(json.dumps({"received_at": time.time(), "frame": chunk}, ensure_ascii=False) + "\n")
                event = decode_service_event(chunk)
                if event is None:
                    continue
                if artifacts:
                    with (artifacts / "研究事件.jsonl").open("a", encoding="utf-8") as saved:
                        saved.write(json.dumps(event, ensure_ascii=False) + "\n")
                etype = event.get("type")
                payload = event.get("data")
                if not isinstance(payload, dict):
                    payload = event
                if etype == "research_complete":
                    final_report = payload.get("final_report", "") or ""
                    if not final_report.strip():
                        raise RuntimeError("research_complete 未包含非空报告")
                    completed = True
                    references = payload.get("references", []) or []
                    quality_score = payload.get("quality_score")
                    quality_score_status = payload.get("quality_score_status", "not_reviewed")
                    quality_verdict = payload.get("quality_verdict", "not_reviewed")
                    for key in ("completion_reason", "citation_verification", "allowed_domains",
                                "graph_version", "checkpoint_id"):
                        if key in payload:
                            completion_meta[key] = payload[key]
                elif etype in {"error", "research_paused", "research_cancelled"}:
                    detail = payload.get("content") or payload.get("message") or etype
                    raise RuntimeError(f"{etype}: {detail}")
        if not completed:
            raise RuntimeError("研究流已结束，但未收到 research_complete")

    try:
        await asyncio.wait_for(_consume(), timeout=timeout)
    except asyncio.TimeoutError:
        error = f"timeout after {timeout}s"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    article = final_report if error is None else ""
    if article and not no_refs:
        article = append_references(article, references)

    record = {
        "id": task["id"],
        "prompt": prompt,
        "article": article,
        # 以下为本地分析用的附加字段。DRB 只读 id/prompt/article，
        # 多余字段不影响它的评测脚本。
        "_meta": {
            "topic": task.get("topic"),
            "language": task.get("language"),
            "run_id": run_id,
            "research_run_id": research_run_id,
            "session_id": session_id,
            **completion_meta,
            "quality_score": quality_score,
            "quality_score_status": quality_score_status,
            "quality_verdict": quality_verdict,
            "references_count": len(references),
            "article_chars": len(article),
            # 正文含引用段时，导出 RACE 变体会把它剥掉；记录剥离后的净正文长度，
            # 因为 RACE 实际评的是这一部分
            "body_chars": len(strip_references(article)),
            "has_refs_section": has_references_section(article),
            "duration_s": round(time.time() - started, 1),
            "error": error,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    if artifacts:
        (artifacts / "运行结果.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        (artifacts / "研究报告.md").write_text(article, encoding="utf-8")
        if error and final_report:
            (artifacts / "未完成报告.md").write_text(final_report, encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# 批量执行
# ---------------------------------------------------------------------------

async def run_bench(args):
    tasks = load_queries()
    if not tasks:
        print(f"❌ 未找到题目文件: {QUERY_FILE}")
        print("   先执行: python eval/scripts/run_deepresearch_bench.py --download")
        return False

    selected = filter_tasks(tasks, args.lang, args.topic, args.limit, args.ids)
    if not selected:
        print("❌ 筛选后没有题目")
        return False

    done = load_done(args.model_name, require_pass=args.require_pass)
    pending = [t for t in selected if str(t["id"]) not in done]

    # 每轮一个时间戳，用于隔离追踪数据（见 make_session_id）
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("DeepResearch Bench 评测")
    print("=" * 70)
    print(f"独立运行名  : {args.model_name}")
    print(f"本轮 run_id : {run_id}")
    print(f"筛选        : lang={args.lang} topic={args.topic or '全部'}")
    print(f"选中题目    : {len(selected)}   已完成 {len(done)}   待跑 {len(pending)}")
    print(f"并发        : {args.concurrency}   单题超时 {args.timeout}s")
    print(f"完成标准    : {'hard pass（verdict=pass 且评分>=7）' if args.require_pass else '文章非空'}")
    print(f"输出        : {output_path(args.model_name)}")
    print("=" * 70)

    if not pending:
        print("\n✅ 所选题目已全部完成。用 --stat 查看统计。")
        return True

    if args.dry_run:
        print("\n[--dry-run] 将要执行的题目：")
        for t in pending:
            print(f"  #{t['id']:<4}[{t.get('language')}] {t.get('topic','-'):<24} {t['prompt'][:44]}")
        return True

    sem = asyncio.Semaphore(args.concurrency)
    counter = {"done": 0, "failed": 0}
    total = len(pending)
    t_start = time.time()

    async def worker(task):
        async with sem:
            idx = counter["done"] + counter["failed"] + 1
            print(f"\n[{idx}/{total}] #{task['id']} [{task.get('language')}] "
                  f"{task['prompt'][:50]}...")
            record = await run_one(service, task, args.model_name, run_id,
                                   args.timeout, args.no_refs)
            meta = record["_meta"]
            if record["article"]:
                counter["done"] += 1
                print(f"        ✅ {meta['article_chars']} 字符 / "
                      f"{meta['references_count']} 引用 / "
                      f"评分 {display_quality_score(meta)} / "
                      f"{meta.get('quality_verdict', 'not_reviewed')} / {meta['duration_s']}s")
            else:
                counter["failed"] += 1
                print(f"        ❌ 失败: {meta['error']}")
            # 边跑边落盘，中断也不丢已完成的结果
            append_result(args.model_name, record)

    async with benchmark_service(args.max_iterations) as service:
        results = await asyncio.gather(*(worker(t) for t in pending), return_exceptions=True)
        # run_one 会记录研究错误；落盘等执行器错误必须向上传递，不能假报成功。
        for result in results:
            if isinstance(result, BaseException):
                raise result

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"完成 {counter['done']} 道 / 失败 {counter['failed']} 道 / 耗时 {elapsed / 60:.1f} 分钟")
    print("=" * 70)
    print(f"\n下一步：把结果交给 DRB 官方评测脚本")
    print(f"  1) cp {output_path(args.model_name)} \\")
    print(f"       <deep_research_bench>/data/test_data/raw_data/{args.model_name}.jsonl")
    print(f"  2) 编辑 run_benchmark.sh: TARGET_MODELS=(\"{args.model_name}\")")
    print(f"  3) bash run_benchmark.sh   # 需要 OPENROUTER_API_KEY / JINA_API_KEY")
    print(f"\n本轮追踪数据（run_id={run_id}）：")
    print(f"  整轮汇总: python eval/scripts/eval_report.py "
          f"--session-prefix drb_{args.model_name}_{run_id}")
    print(f"  单题详情: python eval/scripts/eval_report.py "
          f"--session drb_{args.model_name}_{run_id}_<题号>")
    return counter["failed"] == 0


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def show_stat(model_name: str):
    path = output_path(model_name)
    if not os.path.exists(path):
        print(f"❌ 尚无结果文件: {path}")
        return

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # 同一题可能因重跑出现多条，保留最后一条
    latest = {str(r.get("id")): r for r in records}
    ok = [r for r in latest.values() if r.get("article")]
    bad = [r for r in latest.values() if not r.get("article")]

    print(f"\n结果文件: {path}")
    print(f"题目数    : {len(latest)}   成功 {len(ok)}   失败 {len(bad)}")

    if ok:
        chars = [r["_meta"]["article_chars"] for r in ok if "_meta" in r]
        refs = [r["_meta"]["references_count"] for r in ok if "_meta" in r]
        durs = [r["_meta"]["duration_s"] for r in ok if "_meta" in r]
        scores = [score for r in ok
                  if (score := valid_quality_score(
                      r.get("_meta", {}).get("quality_score")
                  )) is not None]
        native_scores = [
            score for r in ok
            if r.get("_meta", {}).get("quality_score_status") == "valid"
            and (score := valid_quality_score(
                r.get("_meta", {}).get("quality_score")
            )) is not None
        ]
        derived_scores = [
            score for r in ok
            if r.get("_meta", {}).get("quality_score_status") in {
                "derived_from_dimensions", "derived_from_issue_severity",
                "valid_after_retry", "fallback_invalid_model_output",
            }
            and (score := valid_quality_score(
                r.get("_meta", {}).get("quality_score")
            )) is not None
        ]
        invalid_scores = sum(
            1 for r in ok
            if r.get("_meta", {}).get("quality_score") is not None
            and valid_quality_score(r["_meta"]["quality_score"]) is None
        )
        passed = [
            r for r in ok
            if r.get("_meta", {}).get("quality_verdict") == "pass"
            and (valid_quality_score(
                r.get("_meta", {}).get("quality_score")
            ) or 0) >= 7
        ]
        if chars:
            print(f"篇幅      : 平均 {sum(chars) // len(chars):,} 字符  "
                  f"(最短 {min(chars):,} / 最长 {max(chars):,})")
        if refs:
            print(f"引用数    : 平均 {sum(refs) / len(refs):.1f} 条  "
                  f"(0 引用的有 {sum(1 for r in refs if r == 0)} 题)")
        if durs:
            print(f"单题耗时  : 平均 {sum(durs) / len(durs) / 60:.1f} 分钟")
        if scores:
            print(f"自评分    : 平均 {sum(scores) / len(scores):.2f}  "
                  f"（仅供参考，不能当评测指标）")
            if native_scores:
                print(f"原生评分  : 平均 {sum(native_scores) / len(native_scores):.2f}  "
                      f"（{len(native_scores)} 条）")
            if derived_scores:
                print(f"派生/纠错 : 平均 {sum(derived_scores) / len(derived_scores):.2f}  "
                      f"（{len(derived_scores)} 条，与原生评分分开解读）")
        if invalid_scores:
            print(f"无效评分  : {invalid_scores} 条（未计入平均值）")
        print(f"硬通过    : {len(passed)}/{len(ok)}  "
              f"（verdict=pass 且 score>=7）")

    if bad:
        print(f"\n失败题目：")
        for r in bad:
            print(f"  #{r.get('id')}: {r.get('_meta', {}).get('error')}")

    # 导出 DRB 要求的纯净格式（去掉 _meta），按评测类型分两个变体：
    #   .race.jsonl —— 剥掉参考文献段。RACE 的 clean 阶段本就会删掉它，
    #                  留着只是白付输入 token（单题约 $0.024）
    #   .fact.jsonl —— 保留参考文献段。FACT 要从正文抽「论断-URL」对，
    #                  没有 URL 直接 0 分
    # 一次生成同时服务两种评测，换评测不必重新跑题。
    ordered = sorted(ok, key=lambda x: int(x["id"]))
    variants = {
        "race": lambda a: strip_references(a),
        "fact": lambda a: a,
    }
    print()
    for name, transform in variants.items():
        out = path.replace(".jsonl", f".{name}.jsonl")
        total = 0
        with open(out, "w", encoding="utf-8") as f:
            for r in ordered:
                art = transform(r["article"])
                total += len(art)
                f.write(json.dumps(
                    {"id": r["id"], "prompt": r["prompt"], "article": art},
                    ensure_ascii=False,
                ) + "\n")
        avg = total // len(ordered) if ordered else 0
        print(f"✅ {name.upper():<5} 变体: {out}")
        print(f"   {len(ordered)} 条，平均 {avg:,} 字符/篇")


DOCS_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "docs")


def _fetch_trace_stats(model_name: str):
    """
    从 agent_runs 表捞本模型所有轮次的追踪统计。

    数据库不可用时返回 None——报告仍然能生成，只是少了追踪那一节，
    不能因为评估的附属功能失败就让整份报告产不出来。
    """
    try:
        sys.path.insert(0, os.path.join(BACKEND_DIR, "app"))
        from sqlalchemy import func, case
        from core.database import SessionLocal
        from eval.models import AgentRun

        like = f"drb_{model_name}_%"
        db = SessionLocal()
        try:
            by_agent = db.query(
                AgentRun.agent_name,
                func.count(AgentRun.id),
                func.sum(case((AgentRun.success.is_(False), 1), else_=0)),
                func.avg(AgentRun.duration_ms),
                func.max(AgentRun.duration_ms),
                func.sum(AgentRun.total_tokens),
            ).filter(AgentRun.session_id.like(like)).group_by(
                AgentRun.agent_name
            ).order_by(func.count(AgentRun.id).desc()).all()

            by_parse = db.query(
                AgentRun.parse_strategy, func.count(AgentRun.id),
            ).filter(
                AgentRun.session_id.like(like),
                AgentRun.parse_strategy.isnot(None),
            ).group_by(AgentRun.parse_strategy).all()

            failures = db.query(
                AgentRun.agent_name, AgentRun.error_type,
                func.count(AgentRun.id), func.max(AgentRun.duration_ms),
            ).filter(
                AgentRun.session_id.like(like),
                AgentRun.success.is_(False),
            ).group_by(AgentRun.agent_name, AgentRun.error_type).all()

            return {"by_agent": by_agent, "by_parse": by_parse, "failures": failures}
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"追踪统计读取失败，报告将省略该节: {e}")
        return None


def write_report(model_name: str, generated_at: str) -> str:
    """
    把结果文件渲染成 markdown 报告写进 docs/。

    为什么要单独生成而不是只留 JSONL：
    JSONL 是给 DRB 评测脚本吃的机器格式，人看不了；
    终端统计关掉就没了。评测结论需要一份能提交、能回看、
    能直接放进作品集的文字记录。

    每次调用都从本轮结果文件重新渲染（幂等），只更新本轮报告；
    不同运行有不同文件名，永不覆盖其它轮次。
    """
    path = output_path(model_name)
    if not os.path.exists(path):
        return ""

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    latest = {str(r.get("id")): r for r in records}          # 重跑保留最后一条
    rows = sorted(latest.values(), key=lambda x: int(x["id"]))
    ok = [r for r in rows if r.get("article")]
    bad = [r for r in rows if not r.get("article")]

    run_ids = sorted({r.get("_meta", {}).get("run_id") for r in rows
                      if r.get("_meta", {}).get("run_id")})

    L = []
    A = L.append
    # 追踪那一节在数据库不可用时会整节跳过，所以编号必须动态生成，
    # 否则会出现「二、逐题结果」后面直接跟「四、解读须知」的跳号
    _n = iter("一二三四五六七八九十")
    def SEC(title):
        A(f"## {next(_n)}、{title}\n")
    name_meta = run_name_parts(model_name)
    if name_meta:
        A(
            f"# DeepResearch Bench 评测结果 · {name_meta['question_count']}题"
            f" · 第{name_meta['run_number']}次\n"
        )
        A(f"> 独立运行名：`{model_name}`")
    else:
        A(f"# DeepResearch Bench 评测结果 · {model_name}\n")
    A(f"> 生成时间：{generated_at}　　被测系统：industry_information_assistant deep_research_v2")
    A(f"> 结果文件：`backend/{os.path.relpath(path, BACKEND_DIR)}`")
    if run_ids:
        A(f"> 运行批次：{', '.join(run_ids)}")
    A("")
    A("> ⚠️ 本文件由 `eval/scripts/run_deepresearch_bench.py` 自动生成，请勿手工编辑。")
    A("> 重新生成只会更新当前运行；其它轮次的结果不会被覆盖。\n")
    A("---\n")

    # ---- 汇总 ----
    SEC("汇总")
    A(f"- 题目数：**{len(rows)}**　成功 **{len(ok)}**　失败 **{len(bad)}**")
    if ok:
        chars = [r["_meta"]["article_chars"] for r in ok if "_meta" in r]
        refs = [r["_meta"]["references_count"] for r in ok if "_meta" in r]
        durs = [r["_meta"]["duration_s"] for r in ok if "_meta" in r]
        scores = [score for r in ok
                  if (score := valid_quality_score(
                      r.get("_meta", {}).get("quality_score")
                  )) is not None]
        invalid_scores = sum(
            1 for r in ok
            if r.get("_meta", {}).get("quality_score") is not None
            and valid_quality_score(r["_meta"]["quality_score"]) is None
        )
        if chars:
            A(f"- 报告篇幅：平均 **{sum(chars) // len(chars):,}** 字符"
              f"（{min(chars):,} ~ {max(chars):,}）")
        if refs:
            zero = sum(1 for x in refs if x == 0)
            A(f"- 引用条数：平均 **{sum(refs) / len(refs):.1f}** 条"
              f"（0 引用 {zero} 题）")
        if durs:
            A(f"- 单题耗时：平均 **{sum(durs) / len(durs) / 60:.1f}** 分钟"
              f"，合计 {sum(durs) / 3600:.1f} 小时")
        if scores:
            A(f"- CriticMaster 自评分：平均 **{sum(scores) / len(scores):.2f}**"
              f"（⚠️ 系统自评，不构成评测指标）")
        if invalid_scores:
            A(f"- 无效自评分：**{invalid_scores}** 条（未计入平均值；历史 `-1/0` 不再参与统计）")
    A("")

    # ---- 逐题 ----
    SEC("逐题结果")
    A("| # | 领域 | 状态 | 字符数 | 引用 | 自评分 | 耗时 |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        m = r.get("_meta", {})
        status = "✅" if r.get("article") else f"❌ {m.get('error', '未知')}"
        A(f"| {r['id']} | {m.get('topic', '-')} | {status} | "
          f"{m.get('article_chars', 0):,} | {m.get('references_count', 0)} | "
          f"{display_quality_score(m)} | {m.get('duration_s', 0) / 60:.1f}m |")
    A("")

    # ---- 追踪 ----
    stats = _fetch_trace_stats(model_name)
    if stats and stats["by_agent"]:
        SEC("调用链追踪")
        A("数据来自 `agent_runs` 表，涵盖该模型名下的所有运行批次。\n")
        A("| Agent | 调用 | 失败 | 平均耗时 | 最长耗时 | token |")
        A("|---|---|---|---|---|---|")
        for name, cnt, fail, avg_ms, max_ms, tok in stats["by_agent"]:
            A(f"| {name or '-'} | {cnt} | {int(fail or 0)} | "
              f"{float(avg_ms or 0) / 1000:.1f}s | {float(max_ms or 0) / 1000:.1f}s | "
              f"{int(tok or 0):,} |")
        A("")

        if stats["by_parse"]:
            total = sum(c for _, c in stats["by_parse"])
            direct = next((c for s, c in stats["by_parse"] if s == "direct"), 0)
            A(f"**JSON 解析质量**：共 {total} 次，"
              f"`direct` 直接成功 {direct} 次，"
              f"**降级率 {(total - direct) / total * 100:.1f}%**"
              f"（降级率上升 = 提示词的 JSON 约束在退化）\n")
            A("| 解析路径 | 次数 |")
            A("|---|---|")
            for strategy, cnt in sorted(stats["by_parse"], key=lambda x: -x[1]):
                A(f"| {strategy} | {cnt} |")
            A("")

        if stats["failures"]:
            A("**失败调用**\n")
            A("| Agent | 异常类型 | 次数 | 最长耗时 |")
            A("|---|---|---|---|")
            for name, etype, cnt, max_ms in stats["failures"]:
                A(f"| {name} | {etype or '-'} | {cnt} | "
                  f"{float(max_ms or 0) / 1000:.1f}s |")
            A("")

    # ---- 限制 ----
    SEC("结果解读须知")
    A("1. **本文件的分数不是 DRB 官方分数**。这里只记录报告的生成情况；")
    A("   RACE / FACT 评分需要把 `.race.jsonl` / `.fact.jsonl` 交给 DRB 官方脚本另行运行。")
    A("2. **自评分不能当评测指标**。表中的自评分来自系统内部的 CriticMaster，")
    A("   自己给自己打分不构成裁判——DRB II 论文对一代的批评正是这一点。")
    A("3. **官方评委已于 2026-05-11 换为 GPT-5.5**（FACT 用 GPT-5.4-mini）。")
    A("   报告分数时必须注明评委模型，否则与排行榜历史分数不可比。")
    A("")
    A("**提交官方评测：**\n")
    A("```bash")
    A(f"# RACE 用 .race.jsonl（已剥掉会被清洗步骤删掉的引用段）；FACT 用 .fact.jsonl")
    A(f"cp backend/eval/data/deepresearch_bench/raw_data/{model_name}.race.jsonl \\")
    A(f"   <deep_research_bench>/data/test_data/raw_data/{model_name}.jsonl")
    A(f'# 编辑 run_benchmark.sh: TARGET_MODELS=("{model_name}")')
    A("bash run_benchmark.sh")
    A("```")

    os.makedirs(DOCS_DIR, exist_ok=True)
    if name_meta:
        doc_name = (
            f"评测结果-DRB-{name_meta['question_count']}题-第{name_meta['run_number']}次-"
            f"{name_meta['stamp']}-{name_meta['prefix']}.md"
        )
    else:
        doc_name = f"评测结果-DRB-{model_name}.md"
    doc_path = os.path.join(DOCS_DIR, doc_name)
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return doc_path


def main():
    parser = argparse.ArgumentParser(
        description="DeepResearch Bench 评测 Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--download", action="store_true", help="下载官方题目文件")
    parser.add_argument("--model-name", default="industry-assistant-v1",
                        help="新运行的名称前缀；--stat/--resume 时传完整独立运行名")
    parser.add_argument("--lang", default="zh", choices=["zh", "en", "all"],
                        help="题目语言，默认 zh（本系统面向中文场景）")
    parser.add_argument("--topic", help="按领域过滤，如 'Finance'")
    parser.add_argument("--ids", nargs="+", help="只跑指定题号")
    parser.add_argument("--limit", type=int, default=0, help="最多跑几道，0 表示不限")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发数，默认 1（每题都会大量调用搜索 API）")
    parser.add_argument("--timeout", type=int, default=1800, help="单题超时秒数，默认 1800")
    parser.add_argument("--max-iterations", type=int, default=2, choices=range(6),
                        help="研究最大迭代轮数，默认 2")
    parser.add_argument("--no-refs", action="store_true",
                        help="不把引用 URL 附到正文末尾（会导致 FACT 抽不到引用）")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要跑的题目")
    parser.add_argument("--resume", action="store_true",
                        help="显式续跑 --model-name 指定的已有独立运行；默认始终创建新运行")
    parser.add_argument("--require-pass", action="store_true",
                        help="断点续跑时仅把 verdict=pass 且评分>=7 的题目视为完成")
    parser.add_argument("--stat", action="store_true", help="查看进度并导出提交文件")
    parser.add_argument("--no-report", action="store_true",
                        help="不生成 docs/ 下的 markdown 报告")
    args = parser.parse_args()

    if args.download:
        sys.exit(0 if download_queries() else 1)

    if args.stat:
        show_stat(args.model_name)
        _maybe_report(args)
        return

    tasks = load_queries()
    selected = filter_tasks(tasks, args.lang, args.topic, args.limit, args.ids)
    if not selected:
        print("❌ 筛选后没有题目，未创建运行文件")
        sys.exit(1)

    if args.resume:
        if not os.path.exists(output_path(args.model_name)):
            print(f"❌ --resume 指定的运行不存在: {output_path(args.model_name)}")
            sys.exit(1)
    else:
        prefix = args.model_name
        if args.dry_run:
            args.model_name = propose_run_name(prefix, len(selected))
        else:
            args.model_name = reserve_run_name(prefix, len(selected))

    ok = asyncio.run(run_bench(args))
    _maybe_report(args)
    sys.exit(0 if ok else 1)


def _maybe_report(args):
    """跑完 / 查统计后自动落一份 markdown 报告，除非显式关闭。"""
    if args.no_report:
        return
    try:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        doc = write_report(args.model_name, stamp)
        if doc:
            print(f"\n📄 结果文档已生成: {doc}")
    except Exception as e:
        # 报告生成失败不该影响跑题结果，JSONL 才是权威数据
        print(f"⚠️  报告生成失败（不影响结果文件）: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。重跑同一命令可从断点继续。")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)
