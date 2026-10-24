#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""LongMemEval-S Oracle/Retrieved Raw Session 对照实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_longmemeval_regression as regression


DEFAULT_BASELINE_DIR = (
    regression.DEFAULT_RESULTS_DIR / "20260822_172508_full100_e2e_k5"
)
DEFAULT_RETRIEVAL_SNAPSHOT = DEFAULT_BASELINE_DIR / "questions.jsonl"
FROZEN_RETRIEVAL_SHA256 = (
    "ce5801ea8f6203a3a2a7db1e27a387815d7a004a820bca87408cebaad1c97777"
)
DEFAULT_COMPARISON_ROOT = (
    regression.DEFAULT_DATA_DIR / "comparisons"
)
RAW_CONTEXT_RENDERER_VERSION = "longmemeval-raw-session-v1"
DEFAULT_MAX_CONTEXT_CHARS = 120_000


@dataclass(frozen=True)
class RawSession:
    source_session_id: str
    source_index: int
    date: str
    turns: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class RenderedContext:
    text: str
    session_count: int
    turn_count: int
    context_chars: int
    context_sha256: str


def unique_strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def latest_rows_by_question(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in regression.read_jsonl(path):
        question_id = str(row.get("question_id", ""))
        if question_id:
            latest[question_id] = row
    return latest


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_retrieval_snapshot(path: Path, expected_sha256: Optional[str]) -> str:
    if not path.exists():
        raise FileNotFoundError(f"检索快照不存在：{path}")
    actual = regression.sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(
            f"检索快照 SHA256={actual}，预期为 {expected_sha256}；拒绝混用基线"
        )
    return actual


def select_sessions_by_source_ids(
    row: Mapping[str, Any], source_session_ids: Sequence[Any]
) -> list[RawSession]:
    occurrences: dict[str, list[int]] = {}
    for index, source_id in enumerate(row["haystack_session_ids"]):
        occurrences.setdefault(str(source_id), []).append(index)

    selected = []
    for source_id in unique_strings(source_session_ids):
        indices = occurrences.get(source_id, [])
        if not indices:
            raise ValueError(f"{row['question_id']} 找不到原始会话 {source_id}")
        if len(indices) != 1:
            raise ValueError(
                f"{row['question_id']} 的原始会话 ID {source_id} 出现 {len(indices)} 次，"
                "无法仅凭 source_session_id 唯一映射"
            )
        index = indices[0]
        selected.append(RawSession(
            source_session_id=source_id,
            source_index=index,
            date=str(row["haystack_dates"][index]),
            turns=row["haystack_sessions"][index],
        ))
    return selected


def select_oracle_sessions(row: Mapping[str, Any]) -> list[RawSession]:
    return select_sessions_by_source_ids(row, row.get("answer_session_ids", []))


def select_retrieved_sessions(
    row: Mapping[str, Any], snapshot_row: Mapping[str, Any]
) -> list[RawSession]:
    if str(snapshot_row.get("question_id")) != str(row["question_id"]):
        raise ValueError("检索快照 question_id 与数据集题目不一致")
    return select_sessions_by_source_ids(
        row, snapshot_row.get("retrieved_session_ids", [])
    )


def render_raw_context(
    sessions: Sequence[RawSession], *, max_context_chars: int
) -> RenderedContext:
    blocks = []
    turn_count = 0
    for session_number, session in enumerate(sessions, 1):
        lines = [
            f"[session {session_number}; session_id={session.source_session_id}; "
            f"date={session.date}]"
        ]
        for turn in session.turns:
            role = str(turn.get("role", ""))
            if role not in {"user", "assistant", "system"}:
                raise ValueError(
                    f"{session.source_session_id} 含非法消息角色：{role!r}"
                )
            lines.append(f"{role}: {str(turn.get('content', ''))}")
            turn_count += 1
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks) or "(no raw sessions were selected)"
    context_chars = len(text)
    if context_chars > max_context_chars:
        raise ValueError(
            f"原始上下文 {context_chars} 字符，超过硬限制 {max_context_chars}；"
            "为避免实验偏差，拒绝静默截断"
        )
    return RenderedContext(
        text=text,
        session_count=len(sessions),
        turn_count=turn_count,
        context_chars=context_chars,
        context_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def raw_model_config() -> dict:
    models = dict(regression.full_model_config())
    models["answer_prompt_version"] = RAW_CONTEXT_RENDERER_VERSION
    return models


def build_raw_answer_prompt(
    row: Mapping[str, Any], rendered: RenderedContext
) -> str:
    return f"""Answer the user's question using only the raw conversation sessions below.
The sessions may be incomplete or conflicting. Use their dates to prefer the latest applicable information.
If the sessions do not contain enough information, explicitly say that the question cannot be answered.
Answer directly and concisely. Do not mention the benchmark or any reference answer.

Question date: {row['question_date']}
Raw conversation sessions:
{rendered.text}

Question: {row['question']}"""


def answer_raw_and_judge(
    row: Mapping[str, Any], rendered: RenderedContext
) -> dict:
    if str(regression.APP_DIR) not in sys.path:
        sys.path.insert(0, str(regression.APP_DIR))
    from service.llm_client import create_openai_client

    answer_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not answer_api_key:
        raise RuntimeError("Raw Session 实验需要 DASHSCOPE_API_KEY")
    answer_base_url = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    judge_api_key = os.getenv("CLAUDE_RELAY_API_KEY")
    judge_base_url = os.getenv("CLAUDE_RELAY_BASE_URL")
    if not judge_api_key or not judge_base_url:
        raise RuntimeError(
            "Raw Session 实验需要 CLAUDE_RELAY_API_KEY 和 CLAUDE_RELAY_BASE_URL"
        )

    models = raw_model_config()
    answer_client = create_openai_client(
        api_key=answer_api_key, base_url=answer_base_url
    )
    answer_error = None
    answer_started = time.monotonic()
    try:
        hypothesis = regression.call_with_retries(
            lambda: regression.chat_completion(
                answer_client,
                models["answer_model"],
                build_raw_answer_prompt(row, rendered),
                models["answer_max_tokens"],
            ),
            label="Raw Session 回答模型",
            attempts=models["answer_request_attempts"],
        )
    except Exception as exc:
        if not regression.is_content_filter_error(exc):
            raise
        answer_error = {
            "type": type(exc).__name__,
            "code": "data_inspection_failed",
            "policy": models["answer_content_filter_policy"],
        }
        hypothesis = "[ANSWER_PROVIDER_REFUSAL: data_inspection_failed]"
    answer_seconds = time.monotonic() - answer_started

    judge_started = time.monotonic()
    def request_and_parse_judge() -> tuple[str, bool]:
        response = regression.claude_messages_completion(
            judge_api_key,
            judge_base_url,
            models["judge_model"],
            regression.build_judge_prompt(row, hypothesis),
            models["judge_max_tokens"],
        )
        return response, regression.parse_judge_label(response)

    judge_response, judge_label = regression.call_with_retries(
        request_and_parse_judge,
        label="Raw Session Judge",
        attempts=models["judge_request_attempts"],
    )
    return {
        "hypothesis": hypothesis,
        "answer_status": "provider_refusal" if answer_error else "ok",
        "answer_error": answer_error,
        "answer_model": models["answer_model"],
        "judge_label": judge_label,
        "judge_model": models["judge_model"],
        "judge_raw_response": judge_response,
        "answer_seconds": round(answer_seconds, 3),
        "judge_seconds": round(time.monotonic() - judge_started, 3),
    }


def run_raw_question(
    row: Mapping[str, Any],
    *,
    context_mode: str,
    snapshot_row: Mapping[str, Any],
    max_context_chars: int,
) -> dict:
    started = time.monotonic()
    if context_mode == "oracle-raw":
        sessions = select_oracle_sessions(row)
        selection_source = "dataset-answer-session-ids"
    elif context_mode == "retrieved-raw":
        sessions = select_retrieved_sessions(row, snapshot_row)
        selection_source = "frozen-retrieval-snapshot"
    else:
        raise ValueError(f"未知 context mode：{context_mode}")

    rendered = render_raw_context(
        sessions, max_context_chars=max_context_chars
    )
    rendered_at = time.monotonic()
    completion = answer_raw_and_judge(row, rendered)
    selected_session_ids = [session.source_session_id for session in sessions]
    abstention = str(row["question_id"]).endswith("_abs")
    return {
        "question_id": row["question_id"],
        "question_type": row["question_type"],
        "abstention": abstention,
        "status": "ok",
        "mode": "full",
        "context_mode": context_mode,
        "selection_source": selection_source,
        "selected_session_ids": selected_session_ids,
        "session_count": rendered.session_count,
        "turn_count": rendered.turn_count,
        "context_chars": rendered.context_chars,
        "context_sha256": rendered.context_sha256,
        "retrieved_session_ids": selected_session_ids,
        "retrieval": None if abstention else regression.score_retrieval(
            row["answer_session_ids"], selected_session_ids
        ),
        "hypothesis": completion["hypothesis"],
        "answer_status": completion["answer_status"],
        "answer_error": completion["answer_error"],
        "answer_model": completion["answer_model"],
        "judge_label": completion["judge_label"],
        "judge_model": completion["judge_model"],
        "judge_raw_response": completion["judge_raw_response"],
        "timing_seconds": {
            "select_and_render": round(rendered_at - started, 3),
            "answer": completion["answer_seconds"],
            "judge": completion["judge_seconds"],
            "total": round(time.monotonic() - started, 3),
        },
    }


def average(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    return sum(items) / len(items) if items else None


def aggregate_raw_metrics(results: Sequence[Mapping[str, Any]]) -> dict:
    metrics = regression.aggregate_metrics(results)
    successful = [row for row in results if row.get("status") == "ok"]
    answerable = [row for row in successful if not row.get("abstention")]
    abstention = [row for row in successful if row.get("abstention")]
    context_chars = [int(row["context_chars"]) for row in successful]
    metrics.update({
        "answerable_questions": len(answerable),
        "answerable_qa_accuracy": average(
            float(row["judge_label"]) for row in answerable
        ),
        "abstention_qa_accuracy": average(
            float(row["judge_label"]) for row in abstention
        ),
        "answer_provider_refusals": sum(
            row.get("answer_status") == "provider_refusal" for row in successful
        ),
        "context": {
            "mean_chars": average(context_chars),
            "max_chars": max(context_chars) if context_chars else None,
            "mean_sessions": average(
                float(row["session_count"]) for row in successful
            ),
            "mean_turns": average(float(row["turn_count"]) for row in successful),
        },
    })
    return metrics


def format_percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def raw_markdown_report(metadata: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    retrieval = metrics["retrieval"]
    mean_context_chars = metrics["context"]["mean_chars"]
    mean_context_text = (
        "-" if mean_context_chars is None else f"{mean_context_chars:.1f}"
    )
    lines = [
        "# LongMemEval-S Raw Session 对照实验报告",
        "",
        f"- 运行时间：{metadata['created_at']}",
        f"- Context Mode：{metadata['context_mode']}",
        f"- 回答模型：{metadata['models']['answer_model']}",
        f"- Judge：{metadata['models']['judge_model']}",
        f"- 成功/总数：{metrics['successful_questions']}/{metrics['total_rows']}",
        f"- 错误：{metrics['errors']}",
        "",
        "## 总指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| QA Accuracy | {format_percent(metrics['qa_accuracy'])} |",
        f"| Answerable QA Accuracy | {format_percent(metrics['answerable_qa_accuracy'])} |",
        f"| Abstention QA Accuracy | {format_percent(metrics['abstention_qa_accuracy'])} |",
        f"| Retrieval Any Recall@K | {format_percent(retrieval['any_recall'])} |",
        f"| Retrieval All Recall@K | {format_percent(retrieval['all_recall'])} |",
        f"| MRR | {retrieval['mrr'] if retrieval['mrr'] is not None else '-'} |",
        f"| Provider Refusals | {metrics['answer_provider_refusals']} |",
        f"| Mean Context Chars | {mean_context_text} |",
        f"| Max Context Chars | {metrics['context']['max_chars']} |",
        "",
        "## 分题型",
        "",
        "| 题型 | N | QA Accuracy |",
        "|---|---:|---:|",
    ]
    for question_type, values in metrics["by_question_type"].items():
        lines.append(
            f"| {question_type} | {values['questions']} | "
            f"{format_percent(values['qa_accuracy'])} |"
        )
    return "\n".join(lines) + "\n"


def filter_selected_questions(
    selected: Sequence[Mapping[str, Any]], question_ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    if not question_ids:
        return list(selected)
    wanted = unique_strings(question_ids)
    selected_by_id = {str(row["question_id"]): row for row in selected}
    missing = [question_id for question_id in wanted if question_id not in selected_by_id]
    if missing:
        raise ValueError(f"指定题目不在固定 100 题中：{missing}")
    return [selected_by_id[question_id] for question_id in wanted]


def command_prepare(args: argparse.Namespace, suite: Mapping[str, Any]) -> int:
    selected, _ = regression.prepare_suite(
        suite, args.dataset, args.manifest, allow_download=not args.no_download
    )
    snapshot_sha = validate_retrieval_snapshot(
        args.retrieval_snapshot, args.expected_retrieval_sha256
    )
    snapshot = latest_rows_by_question(args.retrieval_snapshot)
    oracle_chars = []
    retrieved_chars = []
    for row in selected:
        question_id = str(row["question_id"])
        if question_id not in snapshot:
            raise ValueError(f"检索快照缺少题目：{question_id}")
        oracle_chars.append(render_raw_context(
            select_oracle_sessions(row), max_context_chars=args.max_context_chars
        ).context_chars)
        retrieved_chars.append(render_raw_context(
            select_retrieved_sessions(row, snapshot[question_id]),
            max_context_chars=args.max_context_chars,
        ).context_chars)

    def describe(values: Sequence[int]) -> dict:
        ordered = sorted(values)
        return {
            "questions": len(values),
            "mean_chars": round(statistics.mean(values), 1),
            "p95_chars": ordered[int(0.95 * (len(ordered) - 1))],
            "max_chars": max(values),
            "total_chars": sum(values),
        }

    print(json.dumps({
        "dataset_sha256": suite["dataset"]["sha256"],
        "retrieval_snapshot": str(args.retrieval_snapshot),
        "retrieval_snapshot_sha256": snapshot_sha,
        "max_context_chars": args.max_context_chars,
        "oracle_raw": describe(oracle_chars),
        "retrieved_raw": describe(retrieved_chars),
    }, ensure_ascii=False, indent=2))
    return 0


def command_run(args: argparse.Namespace, suite: Mapping[str, Any]) -> int:
    selected, manifest = regression.prepare_suite(
        suite, args.dataset, args.manifest, allow_download=not args.no_download
    )
    selected = filter_selected_questions(selected, args.question_id)
    snapshot_sha = validate_retrieval_snapshot(
        args.retrieval_snapshot, args.expected_retrieval_sha256
    )
    snapshot = latest_rows_by_question(args.retrieval_snapshot)
    for row in selected:
        if str(row["question_id"]) not in snapshot:
            raise ValueError(f"检索快照缺少题目：{row['question_id']}")

    run_name = args.run_name or (
        f"{datetime.now():%Y%m%d_%H%M%S}_{args.context_mode.replace('-', '_')}_k5"
    )
    run_dir = args.results_root / run_name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"评估目录已存在，拒绝覆盖：{run_dir}；续跑请显式添加 --resume"
        )
    if args.resume and not run_dir.exists():
        raise FileNotFoundError(f"--resume 指定目录不存在：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    models = raw_model_config()
    run_config = {
        "run_name": run_name,
        "context_mode": args.context_mode,
        "renderer_version": RAW_CONTEXT_RENDERER_VERSION,
        "max_context_chars": args.max_context_chars,
        "workers": args.workers,
        "question_ids": [str(row["question_id"]) for row in selected],
        "dataset_sha256": suite["dataset"]["sha256"],
        "suite_sha256": regression.suite_digest(suite),
        "retrieval_snapshot": str(args.retrieval_snapshot),
        "retrieval_snapshot_sha256": snapshot_sha,
        "models": models,
    }
    snapshots = {
        "run_config.json": run_config,
        "suite_snapshot.json": dict(suite),
        "fixed_100_manifest_snapshot.json": manifest,
    }
    for filename, value in snapshots.items():
        path = run_dir / filename
        if path.exists():
            if regression.read_json(path) != value:
                raise ValueError(f"断点目录中的 {filename} 与本次配置不一致")
        else:
            regression.write_json(path, value)

    result_path = run_dir / "questions.jsonl"
    prior_rows = regression.read_jsonl(result_path)
    successful_ids = {
        str(row["question_id"])
        for row in prior_rows
        if row.get("status") == "ok"
        and row.get("context_mode") == args.context_mode
    }
    pending = [
        row for row in selected if str(row["question_id"]) not in successful_ids
    ]
    print(
        f"Raw 对照题集 {len(selected)} 题；已完成 {len(successful_ids)}；"
        f"待运行 {len(pending)}；context_mode={args.context_mode}",
        flush=True,
    )

    def evaluate(row: Mapping[str, Any]) -> dict:
        question_id = str(row["question_id"])
        try:
            return run_raw_question(
                row,
                context_mode=args.context_mode,
                snapshot_row=snapshot[question_id],
                max_context_chars=args.max_context_chars,
            )
        except Exception as exc:
            return {
                "question_id": question_id,
                "question_type": row["question_type"],
                "abstention": question_id.endswith("_abs"),
                "status": "error",
                "mode": "full",
                "context_mode": args.context_mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def persist(completed: int, result: Mapping[str, Any]) -> None:
        if result["status"] == "error":
            print(
                f"[{completed}/{len(pending)}] {result['question_id']} ERROR "
                f"{result['error_type']}: {result['error']}",
                flush=True,
            )
        else:
            print(
                f"[{completed}/{len(pending)}] {result['question_id']} OK "
                f"judge={int(bool(result['judge_label']))} "
                f"sessions={result['session_count']} chars={result['context_chars']}",
                flush=True,
            )
        regression.append_jsonl(result_path, result)

    if args.workers == 1:
        for completed, row in enumerate(pending, 1):
            persist(completed, evaluate(row))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate, row) for row in pending]
            for completed, future in enumerate(as_completed(futures), 1):
                persist(completed, future.result())

    latest = latest_rows_by_question(result_path)
    report_rows = [
        latest[str(row["question_id"])]
        for row in selected
        if str(row["question_id"]) in latest
    ]
    metrics = aggregate_raw_metrics(report_rows)
    metadata = {
        "suite_name": suite["suite_name"],
        "suite_sha256": regression.suite_digest(suite),
        "manifest_question_id_list_sha256": manifest["question_id_list_sha256"],
        "dataset_sha256": suite["dataset"]["sha256"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_directory": str(run_dir),
        "mode": "full",
        "context_mode": args.context_mode,
        "top_k": 5,
        "retrieval_snapshot_sha256": snapshot_sha,
        "models": models,
    }
    report = {"metadata": metadata, "metrics": metrics}
    regression.write_json(run_dir / "metrics.json", report)
    (run_dir / "report.md").write_text(
        raw_markdown_report(metadata, metrics), encoding="utf-8"
    )
    selection_rows = [
        {
            "question_id": row["question_id"],
            "context_mode": row.get("context_mode"),
            "selected_session_ids": row.get("selected_session_ids"),
            "session_count": row.get("session_count"),
            "turn_count": row.get("turn_count"),
            "context_chars": row.get("context_chars"),
            "context_sha256": row.get("context_sha256"),
        }
        for row in report_rows
        if row.get("status") == "ok"
    ]
    selection_path = run_dir / "selected_sessions_snapshot.jsonl"
    write_jsonl_atomic(selection_path, selection_rows)

    print(raw_markdown_report(metadata, metrics))
    print(f"结果目录：{run_dir}")
    failures = []
    if metrics["errors"]:
        failures.append(f"有 {metrics['errors']} 道题运行失败")
    if metrics["successful_questions"] != len(selected):
        failures.append(
            f"成功题数 {metrics['successful_questions']}，应为 {len(selected)}"
        )
    if failures:
        print("实验运行失败：")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


def load_run_rows(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "questions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"运行目录缺少 questions.jsonl：{run_dir}")
    return latest_rows_by_question(path)


def arm_accuracy(
    rows: Mapping[str, Mapping[str, Any]], ordered_ids: Sequence[str]
) -> dict:
    selected = [rows[question_id] for question_id in ordered_ids]
    answerable = [row for row in selected if not row.get("abstention")]
    abstention = [row for row in selected if row.get("abstention")]
    return {
        "questions": len(selected),
        "correct": sum(bool(row.get("judge_label")) for row in selected),
        "accuracy": average(float(row.get("judge_label")) for row in selected),
        "answerable_questions": len(answerable),
        "answerable_correct": sum(
            bool(row.get("judge_label")) for row in answerable
        ),
        "answerable_accuracy": average(
            float(row.get("judge_label")) for row in answerable
        ),
        "abstention_questions": len(abstention),
        "abstention_correct": sum(
            bool(row.get("judge_label")) for row in abstention
        ),
        "abstention_accuracy": average(
            float(row.get("judge_label")) for row in abstention
        ),
    }


def paired_bootstrap_ci(
    left: Sequence[bool], right: Sequence[bool], *, samples: int = 10_000
) -> list[float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap 需要等长非空标签")
    rng = random.Random(20260822)
    differences = []
    size = len(left)
    for _ in range(samples):
        total = 0
        for _ in range(size):
            index = rng.randrange(size)
            total += int(right[index]) - int(left[index])
        differences.append(total / size)
    differences.sort()
    return [
        differences[int(0.025 * (samples - 1))],
        differences[int(0.975 * (samples - 1))],
    ]


def mcnemar_exact(left: Sequence[bool], right: Sequence[bool]) -> dict:
    left_only = sum(bool(a) and not bool(b) for a, b in zip(left, right))
    right_only = sum(not bool(a) and bool(b) for a, b in zip(left, right))
    discordant = left_only + right_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def normalize_short_answer(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def command_compare(args: argparse.Namespace, suite: Mapping[str, Any]) -> int:
    baseline = load_run_rows(args.baseline)
    retrieved = load_run_rows(args.retrieved_raw)
    oracle = load_run_rows(args.oracle_raw)
    expected_ids = regression.read_json(args.manifest)["question_ids"]
    expected_ids = [str(value) for value in expected_ids]
    expected_set = set(expected_ids)
    for name, rows in (
        ("baseline", baseline),
        ("retrieved_raw", retrieved),
        ("oracle_raw", oracle),
    ):
        actual = set(rows)
        if actual != expected_set:
            raise ValueError(
                f"{name} question ID 集合不一致：missing={sorted(expected_set-actual)} "
                f"extra={sorted(actual-expected_set)}"
            )
        errors = [qid for qid in expected_ids if rows[qid].get("status") != "ok"]
        if errors:
            raise ValueError(f"{name} 含非成功题目：{errors}")

    arms = {
        "compressed_baseline": arm_accuracy(baseline, expected_ids),
        "retrieved_raw": arm_accuracy(retrieved, expected_ids),
        "oracle_raw": arm_accuracy(oracle, expected_ids),
    }
    answerable_ids = [
        qid for qid in expected_ids if not baseline[qid].get("abstention")
    ]

    def labels(rows: Mapping[str, Mapping[str, Any]]) -> list[bool]:
        return [bool(rows[qid].get("judge_label")) for qid in answerable_ids]

    compressed_labels = labels(baseline)
    retrieved_labels = labels(retrieved)
    oracle_labels = labels(oracle)

    def comparison(left: Sequence[bool], right: Sequence[bool]) -> dict:
        delta = average(float(b) for b in right) - average(float(a) for a in left)
        return {
            "delta": delta,
            "bootstrap_95_ci": paired_bootstrap_ci(left, right),
            "mcnemar_exact": mcnemar_exact(left, right),
        }

    deltas = {
        "compression_context_gain_R_minus_C": comparison(
            compressed_labels, retrieved_labels
        ),
        "retrieval_selection_gain_O_minus_R": comparison(
            retrieved_labels, oracle_labels
        ),
        "oracle_residual_error": 1 - average(float(x) for x in oracle_labels),
    }

    dataset_rows = {
        str(row["question_id"]): row
        for row in regression.read_json(args.dataset)
        if str(row["question_id"]) in expected_set
    }

    def exact_audited_label(
        question_id: str, row: Mapping[str, Any]
    ) -> bool:
        if bool(row.get("judge_label")):
            return True
        return normalize_short_answer(row.get("hypothesis", "")) == normalize_short_answer(
            dataset_rows[question_id].get("answer", "")
        )

    def exact_audited_arm(rows: Mapping[str, Mapping[str, Any]]) -> dict:
        selected = [rows[question_id] for question_id in expected_ids]
        answerable_arm_ids = [
            question_id for question_id in expected_ids
            if not rows[question_id].get("abstention")
        ]
        abstention_arm_ids = [
            question_id for question_id in expected_ids
            if rows[question_id].get("abstention")
        ]
        all_labels = [
            exact_audited_label(question_id, rows[question_id])
            for question_id in expected_ids
        ]
        answerable_arm_labels = [
            exact_audited_label(question_id, rows[question_id])
            for question_id in answerable_arm_ids
        ]
        abstention_arm_labels = [
            exact_audited_label(question_id, rows[question_id])
            for question_id in abstention_arm_ids
        ]
        return {
            "questions": len(selected),
            "correct": sum(all_labels),
            "accuracy": average(float(value) for value in all_labels),
            "answerable_questions": len(answerable_arm_ids),
            "answerable_correct": sum(answerable_arm_labels),
            "answerable_accuracy": average(
                float(value) for value in answerable_arm_labels
            ),
            "abstention_questions": len(abstention_arm_ids),
            "abstention_correct": sum(abstention_arm_labels),
            "abstention_accuracy": average(
                float(value) for value in abstention_arm_labels
            ),
        }

    exact_audited_arms = {
        "compressed_baseline": exact_audited_arm(baseline),
        "retrieved_raw": exact_audited_arm(retrieved),
        "oracle_raw": exact_audited_arm(oracle),
    }
    exact_audited_deltas = {
        "compression_context_gain_R_minus_C": (
            exact_audited_arms["retrieved_raw"]["answerable_accuracy"]
            - exact_audited_arms["compressed_baseline"]["answerable_accuracy"]
        ),
        "retrieval_selection_gain_O_minus_R": (
            exact_audited_arms["oracle_raw"]["answerable_accuracy"]
            - exact_audited_arms["retrieved_raw"]["answerable_accuracy"]
        ),
        "oracle_residual_error": (
            1 - exact_audited_arms["oracle_raw"]["answerable_accuracy"]
        ),
    }
    question_types = sorted({
        str(baseline[qid]["question_type"]) for qid in answerable_ids
    })
    by_type = {}
    for question_type in question_types:
        typed_ids = [
            qid for qid in answerable_ids
            if baseline[qid]["question_type"] == question_type
        ]
        values = {}
        for name, rows in (
            ("C", baseline), ("R", retrieved), ("O", oracle)
        ):
            typed_labels = [bool(rows[qid]["judge_label"]) for qid in typed_ids]
            values[name] = {
                "questions": len(typed_ids),
                "correct": sum(typed_labels),
                "accuracy": average(float(x) for x in typed_labels),
            }
        values["R_minus_C"] = values["R"]["accuracy"] - values["C"]["accuracy"]
        values["O_minus_R"] = values["O"]["accuracy"] - values["R"]["accuracy"]
        by_type[question_type] = values

    paired_rows = []
    exact_match_false_negatives = Counter()
    for question_id in expected_ids:
        source = dataset_rows[question_id]
        arm_rows = {
            "C": baseline[question_id],
            "R": retrieved[question_id],
            "O": oracle[question_id],
        }
        for arm, row in arm_rows.items():
            if not row.get("judge_label") and normalize_short_answer(
                row.get("hypothesis", "")
            ) == normalize_short_answer(source.get("answer", "")):
                exact_match_false_negatives[arm] += 1
        paired_rows.append({
            "question_id": question_id,
            "question_type": source["question_type"],
            "abstention": question_id.endswith("_abs"),
            "question": source["question"],
            "reference_answer": source["answer"],
            "compressed": {
                "label": baseline[question_id].get("judge_label"),
                "hypothesis": baseline[question_id].get("hypothesis"),
            },
            "retrieved_raw": {
                "label": retrieved[question_id].get("judge_label"),
                "hypothesis": retrieved[question_id].get("hypothesis"),
                "selected_session_ids": retrieved[question_id].get(
                    "selected_session_ids"
                ),
            },
            "oracle_raw": {
                "label": oracle[question_id].get("judge_label"),
                "hypothesis": oracle[question_id].get("hypothesis"),
                "selected_session_ids": oracle[question_id].get(
                    "selected_session_ids"
                ),
            },
        })

    transitions = {
        "C_wrong_R_right": [
            qid for qid in answerable_ids
            if not baseline[qid]["judge_label"] and retrieved[qid]["judge_label"]
        ],
        "R_wrong_O_right": [
            qid for qid in answerable_ids
            if not retrieved[qid]["judge_label"] and oracle[qid]["judge_label"]
        ],
        "O_still_wrong": [
            qid for qid in answerable_ids if not oracle[qid]["judge_label"]
        ],
    }

    comparison_name = args.comparison_name or (
        f"{datetime.now():%Y%m%d_%H%M%S}_raw_session_ablation"
    )
    comparison_dir = args.comparison_root / comparison_name
    if comparison_dir.exists():
        raise FileExistsError(f"比较目录已存在，拒绝覆盖：{comparison_dir}")
    comparison_dir.mkdir(parents=True)
    config = {
        "comparison_name": comparison_name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline": str(args.baseline),
        "retrieved_raw": str(args.retrieved_raw),
        "oracle_raw": str(args.oracle_raw),
        "dataset_sha256": suite["dataset"]["sha256"],
        "run_question_sha256": {
            "baseline": regression.sha256_file(args.baseline / "questions.jsonl"),
            "retrieved_raw": regression.sha256_file(
                args.retrieved_raw / "questions.jsonl"
            ),
            "oracle_raw": regression.sha256_file(args.oracle_raw / "questions.jsonl"),
        },
    }
    report = {
        "config": config,
        "arms": arms,
        "exact_match_audited_arms": exact_audited_arms,
        "deltas": deltas,
        "exact_match_audited_deltas": exact_audited_deltas,
        "by_question_type_answerable": by_type,
        "transitions": transitions,
        "exact_match_false_negatives": dict(exact_match_false_negatives),
    }
    regression.write_json(comparison_dir / "comparison_config.json", config)
    regression.write_json(comparison_dir / "comparison.json", report)
    for row in paired_rows:
        regression.append_jsonl(comparison_dir / "paired_questions.jsonl", row)

    lines = [
        "# LongMemEval-S Raw Session 三组配对比较",
        "",
        "## 总体结果",
        "",
        "| 实验组 | 全部正确率 | 可回答题正确率 | Abstention 正确率 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("compressed_baseline", "retrieved_raw", "oracle_raw"):
        values = arms[name]
        lines.append(
            f"| {name} | {format_percent(values['accuracy'])} | "
            f"{format_percent(values['answerable_accuracy'])} | "
            f"{format_percent(values['abstention_accuracy'])} |"
        )
    lines.extend([
        "",
        "## Exact-match 审计修正",
        "",
        "只把回答与 reference 归一化后完全一致、但 LLM Judge 判错的题修正为正确。",
        "",
        "| 实验组 | 全部正确率 | 可回答题正确率 |",
        "|---|---:|---:|",
    ])
    for name in ("compressed_baseline", "retrieved_raw", "oracle_raw"):
        values = exact_audited_arms[name]
        lines.append(
            f"| {name} | {format_percent(values['accuracy'])} | "
            f"{format_percent(values['answerable_accuracy'])} |"
        )
    lines.extend([
        "",
        "## 因果差值（可回答题）",
        "",
        f"- `R-C`：{format_percent(deltas['compression_context_gain_R_minus_C']['delta'])}",
        f"- `O-R`：{format_percent(deltas['retrieval_selection_gain_O_minus_R']['delta'])}",
        f"- Oracle 残余误差：{format_percent(deltas['oracle_residual_error'])}",
        "",
        "## 分题型（可回答题）",
        "",
        "| 题型 | N | C | R | O | R-C | O-R |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for question_type, values in by_type.items():
        lines.append(
            f"| {question_type} | {values['C']['questions']} | "
            f"{format_percent(values['C']['accuracy'])} | "
            f"{format_percent(values['R']['accuracy'])} | "
            f"{format_percent(values['O']['accuracy'])} | "
            f"{format_percent(values['R_minus_C'])} | "
            f"{format_percent(values['O_minus_R'])} |"
        )
    lines.extend([
        "",
        "## 配对翻转",
        "",
        f"- C 错、R 对：{len(transitions['C_wrong_R_right'])} 题",
        f"- R 错、O 对：{len(transitions['R_wrong_O_right'])} 题",
        f"- O 仍错：{len(transitions['O_still_wrong'])} 题",
        f"- 明显 exact-match Judge 假阴性：{dict(exact_match_false_negatives)}",
        "",
    ])
    (comparison_dir / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"比较目录：{comparison_dir}")
    return 0


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=regression.DEFAULT_DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=regression.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--no-download", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    add_dataset_arguments(prepare)
    prepare.add_argument(
        "--retrieval-snapshot", type=Path, default=DEFAULT_RETRIEVAL_SNAPSHOT
    )
    prepare.add_argument(
        "--expected-retrieval-sha256", default=FROZEN_RETRIEVAL_SHA256
    )
    prepare.add_argument(
        "--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS
    )

    run = subparsers.add_parser("run")
    add_dataset_arguments(run)
    run.add_argument(
        "--context-mode", choices=("oracle-raw", "retrieved-raw"), required=True
    )
    run.add_argument(
        "--retrieval-snapshot", type=Path, default=DEFAULT_RETRIEVAL_SNAPSHOT
    )
    run.add_argument(
        "--expected-retrieval-sha256", default=FROZEN_RETRIEVAL_SHA256
    )
    run.add_argument(
        "--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS
    )
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--question-id", action="append", default=[])
    run.add_argument("--run-name")
    run.add_argument("--results-root", type=Path, default=regression.DEFAULT_RESULTS_DIR)
    run.add_argument("--resume", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_DIR)
    compare.add_argument("--retrieved-raw", type=Path, required=True)
    compare.add_argument("--oracle-raw", type=Path, required=True)
    compare.add_argument("--dataset", type=Path, default=regression.DEFAULT_DATASET_PATH)
    compare.add_argument("--manifest", type=Path, default=regression.DEFAULT_MANIFEST_PATH)
    compare.add_argument(
        "--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT
    )
    compare.add_argument("--comparison-name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if hasattr(args, "max_context_chars") and args.max_context_chars < 1:
        raise ValueError("--max-context-chars 必须大于 0")
    if hasattr(args, "workers") and args.workers < 1:
        raise ValueError("--workers 必须大于 0")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    suite = regression.read_json(regression.SUITE_PATH)
    if args.command == "prepare":
        return command_prepare(args, suite)
    if args.command == "run":
        return command_run(args, suite)
    return command_compare(args, suite)


if __name__ == "__main__":
    raise SystemExit(main())
