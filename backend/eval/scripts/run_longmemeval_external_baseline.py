#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""Run the fixed LongMemEval-S 100 questions against an external memory system.

Examples (from ``backend``)::

    .venv/bin/python eval/scripts/run_longmemeval_external_baseline.py run \
        --provider mem0 --mode retrieval --limit 1

    .venv/bin/python eval/scripts/run_longmemeval_external_baseline.py run \
        --provider mem0 --mode full

Every run owns a new timestamped directory.  ``--resume`` is the only way to
reuse a directory, and resumes both session-level ingestion and question-level
evaluation without overwriting earlier JSONL rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


BACKEND_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_longmemeval_regression import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_MANIFEST_PATH,
    SUITE_PATH,
    aggregate_metrics,
    answer_and_judge,
    append_jsonl,
    full_model_config,
    markdown_report,
    prepare_suite,
    read_json,
    read_jsonl,
    score_retrieval,
    suite_digest,
    write_json,
)


DEFAULT_RESULTS_ROOT = (
    BACKEND_DIR / "eval" / "data" / "longmemeval_s" / "external_results"
)
DEFAULT_RUNTIME_ROOT = (
    BACKEND_DIR / "eval" / "data" / "longmemeval_s" / "runtime"
)
WORKER_PATH = SCRIPT_DIR / "longmemeval_external_worker.py"
DEFAULT_LLM_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMS = 1024


def _atomic_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, value)


def _tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _package_version(python: Path, package: str) -> Optional[str]:
    operation = subprocess.run(
        [str(python), "-c", (
            "import importlib.metadata; "
            f"print(importlib.metadata.version({package!r}))"
        )],
        capture_output=True,
        text=True,
        check=False,
    )
    return operation.stdout.strip() if operation.returncode == 0 else None


def default_worker_python(provider: str) -> Path:
    if provider == "mem0":
        return DEFAULT_RUNTIME_ROOT / "venvs" / "mem0" / "bin" / "python"
    if provider == "graphiti":
        return DEFAULT_RUNTIME_ROOT / "venvs" / "graphiti" / "bin" / "python"
    return BACKEND_DIR / ".venv" / "bin" / "python"


def provider_preflight(provider: str, worker_python: Path) -> dict:
    failures = []
    if not WORKER_PATH.exists():
        failures.append(f"缺少 worker: {WORKER_PATH}")
    if not worker_python.exists():
        failures.append(f"缺少 Provider Python: {worker_python}")

    package = {"mem0": "mem0ai", "graphiti": "graphiti-core"}.get(provider)
    version = _package_version(worker_python, package) if package and worker_python.exists() else None
    if package and not version:
        failures.append(f"{worker_python} 未安装 {package}")

    if provider in {"mem0", "graphiti"}:
        for name in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"):
            if not os.getenv(name):
                failures.append(f"缺少 {name}")
    elif provider == "supermemory":
        if not os.getenv("SUPERMEMORY_BASE_URL"):
            failures.append("缺少 SUPERMEMORY_BASE_URL")

    if failures:
        raise RuntimeError("Provider preflight 失败：\n- " + "\n- ".join(failures))
    return {"worker_python": str(worker_python), "package": package, "version": version}


class ProviderProcess:
    def __init__(
        self,
        provider: str,
        worker_python: Path,
        state_dir: Path,
        log_path: Path,
        *,
        llm_model: str,
        embedding_model: str,
        embedding_dims: int,
        timeout_seconds: int,
    ):
        self.provider = provider
        self.log_path = log_path
        self.timeout_seconds = timeout_seconds
        self.unhealthy = False
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = log_path.open("a", encoding="utf-8")
        self.stdout_buffer = b""
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["MEM0_TELEMETRY"] = "false"
        environment["GRAPHITI_TELEMETRY_ENABLED"] = "false"
        if provider in {"mem0", "graphiti"}:
            environment["BASELINE_LLM_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
            environment["BASELINE_LLM_BASE_URL"] = os.environ["DASHSCOPE_BASE_URL"]
        self.process = subprocess.Popen(
            [
                str(worker_python),
                str(WORKER_PATH),
                "--provider", provider,
                "--state-dir", str(state_dir),
                "--llm-model", llm_model,
                "--embedding-model", embedding_model,
                "--embedding-dims", str(embedding_dims),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_handle,
            bufsize=0,
            env=environment,
        )

    def _readline(self, deadline: float) -> bytes:
        assert self.process.stdout is not None
        while b"\n" not in self.stdout_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                return b""
            self.stdout_buffer += chunk
        line, self.stdout_buffer = self.stdout_buffer.split(b"\n", 1)
        return line

    def request(self, operation: str, **payload: Any) -> dict:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Provider worker 已退出（code={self.process.returncode}）：\n{_tail(self.log_path)}"
            )
        request_id = str(uuid.uuid4())
        command = {"request_id": request_id, "operation": operation, **payload}
        assert self.process.stdin is not None
        self.process.stdin.write((json.dumps(command, ensure_ascii=False) + "\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout_seconds
        response = None
        while response is None:
            try:
                raw_line = self._readline(deadline)
            except TimeoutError as exc:
                self.unhealthy = True
                raise TimeoutError(
                    f"Provider {operation} 超过 {self.timeout_seconds}s；日志：\n{_tail(self.log_path)}"
                ) from exc
            if not raw_line:
                self.unhealthy = True
                raise RuntimeError(
                    f"Provider {operation} 未返回结果；日志：\n{_tail(self.log_path)}"
                )
            # Native dependencies may bypass redirect_stdout (for example a
            # model-download progress bar).  Keep the protocol resilient and
            # copy such noise into provider.log rather than treating it as JSON.
            marker = raw_line.find(b'{"request_id"')
            candidate = raw_line[marker:] if marker >= 0 else raw_line.strip()
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                self.log_handle.write(
                    "[worker stdout] " + raw_line.decode(errors="replace") + "\n"
                )
                self.log_handle.flush()
                continue
            if parsed.get("request_id") != request_id:
                self.log_handle.write(
                    "[worker unexpected response] "
                    + json.dumps(parsed, ensure_ascii=False) + "\n"
                )
                self.log_handle.flush()
                continue
            response = parsed
        if not response.get("ok"):
            raise RuntimeError(
                f"{response.get('error_type', 'ProviderError')}: {response.get('error', '')}"
            )
        return dict(response.get("result") or {})

    def close(self) -> None:
        if self.process.poll() is None and not self.unhealthy:
            try:
                self.request("close")
            except Exception:
                self.unhealthy = True
                self.process.terminate()
        elif self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.log_handle.close()

    def __enter__(self) -> "ProviderProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


@dataclass
class ExternalEvidence:
    memory_id: str
    fragment_id: str
    memory_type: str
    content: str
    score: float
    source_session_ids: list[str]
    metadata: dict
    rank: int

    @property
    def source_session_id(self) -> Optional[str]:
        return self.source_session_ids[0] if self.source_session_ids else None


@dataclass
class ExternalIndexOutcome:
    status: str
    total_sessions: int
    ready_sessions: int
    failures: list[dict]
    stored_items: int


class ExternalMemoryAdapter:
    def __init__(
        self,
        provider: ProviderProcess,
        checkpoint_path: Path,
        *,
        retries: int,
        verbose_sessions: bool,
    ):
        self.provider = provider
        self.checkpoint_path = checkpoint_path
        self.retries = retries
        self.verbose_sessions = verbose_sessions
        self.checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {
            "schema_version": 1,
            "provider": provider.provider,
            "questions": {},
        }
        if self.checkpoint.get("provider") != provider.provider:
            raise ValueError("索引断点中的 provider 与本次不一致")

    def _save(self) -> None:
        _atomic_checkpoint(self.checkpoint_path, self.checkpoint)

    def ensure_index(self, row: Mapping[str, Any]) -> ExternalIndexOutcome:
        question_id = str(row["question_id"])
        question_state = self.checkpoint["questions"].setdefault(question_id, {
            "completed_session_ids": [],
            "session_results": {},
            "finalized": False,
        })
        complete = set(str(value) for value in question_state["completed_session_ids"])
        initially_complete = len(complete)
        session_records = []
        source_occurrences: dict[str, int] = {}
        for source_id, session_date, messages in zip(
            row["haystack_session_ids"], row["haystack_dates"], row["haystack_sessions"]
        ):
            source_id = str(source_id)
            occurrence = source_occurrences.get(source_id, 0) + 1
            source_occurrences[source_id] = occurrence
            ingestion_session_id = (
                source_id if occurrence == 1 else f"{source_id}__occurrence_{occurrence}"
            )
            normalized = {
                "source_session_id": source_id,
                "ingestion_session_id": ingestion_session_id,
                "session_date": str(session_date),
                "messages": [
                    {
                        "role": str(item.get("role", "")),
                        "content": str(item.get("content", "")),
                    }
                    for item in messages
                ],
            }
            session_records.append((ingestion_session_id, normalized))

        total_sessions = len(session_records)
        stored_items = sum(
            int(value.get("stored_items") or 0)
            for value in question_state.get("session_results", {}).values()
        )
        failures = []

        for session_number, (ingestion_session_id, session_payload) in enumerate(session_records, 1):
            if ingestion_session_id in complete:
                continue
            if self.verbose_sessions:
                print(
                    f"  [{question_id}] {self.provider.provider} 写入会话 "
                    f"{session_number}/{total_sessions}: {ingestion_session_id}",
                    flush=True,
                )
            error = None
            for attempt in range(1, self.retries + 1):
                try:
                    result = self.provider.request(
                        "add",
                        scope_id=question_id,
                        source_session_id=session_payload["source_session_id"],
                        ingestion_session_id=ingestion_session_id,
                        session_date=session_payload["session_date"],
                        messages=session_payload["messages"],
                    )
                    question_state["completed_session_ids"].append(ingestion_session_id)
                    question_state["session_results"][ingestion_session_id] = result
                    question_state["finalized"] = False
                    complete.add(ingestion_session_id)
                    stored_items += int(result.get("stored_items") or 0)
                    self._save()
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    if attempt < self.retries:
                        time.sleep(min(attempt, 3))
            if error is not None:
                failures.append({
                    "source_session_id": session_payload["source_session_id"],
                    "ingestion_session_id": ingestion_session_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })

        if not failures and len(complete) == total_sessions:
            # Kuzu requires an explicit FTS rebuild; Supermemory uses this hook
            # to let asynchronous document processing settle.
            self.provider.request("finalize_scope", scope_id=question_id)
            question_state["finalized"] = True
            self._save()
        status = "partial" if failures else (
            "reused" if initially_complete == total_sessions else (
                "resumed" if initially_complete else "created"
            )
        )
        return ExternalIndexOutcome(
            status=status,
            total_sessions=total_sessions,
            ready_sessions=len(complete),
            failures=failures,
            stored_items=stored_items,
        )

    def search(self, row: Mapping[str, Any], top_k: int) -> list[ExternalEvidence]:
        response = None
        error = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.provider.request(
                    "search",
                    scope_id=str(row["question_id"]),
                    query=str(row["question"]),
                    top_k=top_k,
                )
                error = None
                break
            except Exception as exc:
                error = exc
                if attempt < self.retries:
                    time.sleep(min(attempt, 3))
        if error is not None:
            raise error
        assert response is not None
        evidence = []
        for item in response.get("evidence") or []:
            evidence.append(ExternalEvidence(
                memory_id=str(item.get("memory_id") or ""),
                fragment_id=str(item.get("fragment_id") or ""),
                memory_type=str(item.get("memory_type") or "external_memory"),
                content=str(item.get("content") or ""),
                score=float(item.get("score") or 0),
                source_session_ids=[str(value) for value in item.get("source_session_ids") or []],
                metadata=dict(item.get("metadata") or {}),
                rank=int(item.get("rank") or len(evidence) + 1),
            ))
        return evidence


def retrieved_session_ids(evidence: Sequence[ExternalEvidence]) -> list[str]:
    seen = set()
    result = []
    for item in evidence:
        for source_id in item.source_session_ids:
            if source_id not in seen:
                seen.add(source_id)
                result.append(source_id)
    return result


def evaluate_one(
    adapter: ExternalMemoryAdapter,
    row: Mapping[str, Any],
    *,
    provider_name: str,
    top_k: int,
    mode: str,
) -> dict:
    started = time.monotonic()
    index_outcome = adapter.ensure_index(row)
    searched_at = time.monotonic()
    evidence = adapter.search(row, top_k)
    answered_at = time.monotonic()
    session_ids = retrieved_session_ids(evidence)
    question_id = str(row["question_id"])
    abstention = question_id.endswith("_abs")
    result = {
        "question_id": question_id,
        "question_type": row["question_type"],
        "abstention": abstention,
        "status": "partial" if index_outcome.failures else "ok",
        "provider": provider_name,
        "mode": mode,
        "top_k": top_k,
        "index_status": index_outcome.status,
        "ingestion": {
            "complete": not index_outcome.failures
            and index_outcome.ready_sessions == index_outcome.total_sessions,
            "ready_sessions": index_outcome.ready_sessions,
            "total_sessions": index_outcome.total_sessions,
            "stored_items": index_outcome.stored_items,
            "failures": index_outcome.failures,
        },
        "retrieved_session_ids": session_ids,
        "retrieval": None if abstention else score_retrieval(
            row["answer_session_ids"], session_ids
        ),
        "evidence": [asdict(item) for item in evidence],
        "hypothesis": None,
        "answer_status": None,
        "answer_error": None,
        "answer_model": None,
        "judge_label": None,
        "judge_model": None,
        "timing_seconds": {
            "index": round(searched_at - started, 3),
            "search": round(answered_at - searched_at, 3),
        },
    }
    if mode == "full":
        answer_started = time.monotonic()
        hypothesis, label, answer_model, judge_model, answer_error = answer_and_judge(
            row, evidence
        )
        result.update(
            hypothesis=hypothesis,
            answer_status="provider_refusal" if answer_error else "ok",
            answer_error=answer_error,
            answer_model=answer_model,
            judge_label=label,
            judge_model=judge_model,
        )
        result["timing_seconds"]["answer_and_judge"] = round(
            time.monotonic() - answer_started, 3
        )
    return result


def external_markdown_report(metadata: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    report = markdown_report(metadata, metrics)
    provider = metadata["provider"]
    insertion = (
        f"- 记忆系统：{provider.get('provider', metadata['provider_key'])}\n"
        f"- Provider 版本：{provider.get('version', '-')}\n"
        f"- 写入粒度：{provider.get('ingestion_unit', '-')}\n"
    )
    first_break = report.find("\n\n")
    return report[: first_break + 2] + insertion + report[first_break + 2 :]


def load_question_subset(path: Path) -> list[str]:
    payload = read_json(path)
    question_ids = payload.get("question_ids") if isinstance(payload, Mapping) else payload
    if not isinstance(question_ids, list) or not question_ids:
        raise ValueError(f"{path} 必须是非空 question_ids 数组或包含该数组的对象")
    normalized = [str(question_id) for question_id in question_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{path} 中包含重复 question_id")
    return normalized


def command_run(args: argparse.Namespace) -> int:
    suite = read_json(SUITE_PATH)
    selected, manifest = prepare_suite(
        suite, args.dataset, args.manifest, allow_download=not args.no_download
    )
    subset_question_ids = None
    if args.question_id_file:
        subset_question_ids = load_question_subset(args.question_id_file)
        rows_by_id = {str(row["question_id"]): row for row in selected}
        missing = [question_id for question_id in subset_question_ids if question_id not in rows_by_id]
        if missing:
            raise ValueError(
                f"{args.question_id_file} 中有题目不属于固定 100 题：{missing}"
            )
        selected = [rows_by_id[question_id] for question_id in subset_question_ids]
    elif args.question_id:
        selected = [row for row in selected if str(row["question_id"]) == args.question_id]
        if not selected:
            raise ValueError(f"固定 100 题中不存在 question_id={args.question_id}")
    if args.limit is not None:
        selected = selected[: args.limit]

    selected_question_ids = [str(row["question_id"]) for row in selected]
    selected_question_id_list_sha256 = hashlib.sha256(
        "\n".join(selected_question_ids).encode()
    ).hexdigest()

    worker_python = args.worker_python or default_worker_python(args.provider)
    preflight = provider_preflight(args.provider, worker_python)
    models = full_model_config() if args.mode == "full" else None
    run_name = args.run_name or (
        f"{datetime.now():%Y%m%d_%H%M%S}_{args.provider}_{args.mode}_k{args.top_k}"
    )
    run_dir = args.results_root / run_name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"评估目录已存在，拒绝覆盖：{run_dir}")
    if args.resume and not run_dir.exists():
        raise FileNotFoundError(f"--resume 指定的评估目录不存在：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "schema_version": 1,
        "run_name": run_name,
        "provider": args.provider,
        "mode": args.mode,
        "top_k": args.top_k,
        "limit": args.limit,
        "question_id": args.question_id,
        "question_id_file": (
            str(args.question_id_file.resolve()) if args.question_id_file else None
        ),
        "selected_question_ids": selected_question_ids,
        "selected_question_id_list_sha256": selected_question_id_list_sha256,
        "dataset_sha256": suite["dataset"]["sha256"],
        "suite_sha256": suite_digest(suite),
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "embedding_dims": args.embedding_dims,
        "ingestion_unit": "session",
        "workers": args.workers,
        "worker_python": str(worker_python),
        "provider_package": preflight,
    }
    if models:
        run_config["models"] = models
    snapshots = {
        "run_config.json": run_config,
        "suite_snapshot.json": suite,
        "fixed_100_manifest_snapshot.json": manifest,
        "question_subset_snapshot.json": {
            "question_count": len(selected_question_ids),
            "question_id_list_sha256": selected_question_id_list_sha256,
            "question_ids": selected_question_ids,
        },
    }
    for filename, value in snapshots.items():
        path = run_dir / filename
        if path.exists() and read_json(path) != value:
            raise ValueError(f"断点目录中的 {filename} 与本次配置不一致")
        if not path.exists():
            write_json(path, value)

    result_path = run_dir / "questions.jsonl"
    prior_rows = read_jsonl(result_path)
    successful_ids = {
        str(row["question_id"])
        for row in prior_rows
        if row.get("status") == "ok"
        and row.get("provider") == args.provider
        and row.get("mode") == args.mode
        and row.get("top_k") == args.top_k
    }
    pending = [row for row in selected if str(row["question_id"]) not in successful_ids]
    print(
        f"Provider={args.provider}；固定子集 {len(selected)} 题；"
        f"已完成 {len(successful_ids)}；待运行 {len(pending)}；"
        f"mode={args.mode} top_k={args.top_k}",
        flush=True,
    )

    provider_state_root = run_dir / "provider_state"
    checkpoint_root = run_dir / "ingestion_checkpoints"
    provider_log_root = run_dir / "provider_logs"

    def evaluate_isolated(row: Mapping[str, Any]) -> tuple[dict, dict]:
        question_id = str(row["question_id"])
        try:
            with ProviderProcess(
                args.provider,
                worker_python,
                provider_state_root / question_id,
                provider_log_root / f"{question_id}.log",
                llm_model=args.llm_model,
                embedding_model=args.embedding_model,
                embedding_dims=args.embedding_dims,
                timeout_seconds=args.provider_timeout,
            ) as provider:
                provider_info = provider.request("info")
                adapter = ExternalMemoryAdapter(
                    provider,
                    checkpoint_root / f"{question_id}.json",
                    retries=args.session_retries,
                    verbose_sessions=args.verbose_sessions,
                )
                result = evaluate_one(
                    adapter,
                    row,
                    provider_name=args.provider,
                    top_k=args.top_k,
                    mode=args.mode,
                )
                provider_info = provider.request("info")
                return result, provider_info
        except Exception as exc:
            return ({
                "question_id": question_id,
                "question_type": row["question_type"],
                "abstention": question_id.endswith("_abs"),
                "status": "error",
                "provider": args.provider,
                "mode": args.mode,
                "top_k": args.top_k,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, {})

    def persist_result(completed: int, result: Mapping[str, Any], provider_info: Mapping[str, Any]) -> None:
        append_jsonl(result_path, result)
        if provider_info:
            write_json(run_dir / "provider_manifest.json", provider_info)
        question_id = str(result["question_id"])
        if result["status"] == "error":
            print(
                f"[{completed}/{len(pending)}] {question_id} ERROR "
                f"{result['error_type']}: {result['error']}",
                flush=True,
            )
        else:
            retrieval = result["retrieval"]
            recall = "abstention" if retrieval is None else (
                f"any={int(retrieval['any_recall'])} all={int(retrieval['all_recall'])}"
            )
            print(
                f"[{completed}/{len(pending)}] {question_id} "
                f"{result['status'].upper()} {recall}",
                flush=True,
            )

    if args.workers == 1:
        for completed, row in enumerate(pending, 1):
            result, provider_info = evaluate_isolated(row)
            persist_result(completed, result, provider_info)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_isolated, row) for row in pending]
            for completed, future in enumerate(as_completed(futures), 1):
                result, provider_info = future.result()
                persist_result(completed, result, provider_info)

    provider_manifest_path = run_dir / "provider_manifest.json"
    if not provider_manifest_path.exists():
        raise RuntimeError("未生成 provider_manifest.json")

    all_rows = read_jsonl(result_path)
    latest_by_id = {}
    selected_ids = [str(row["question_id"]) for row in selected]
    for row in all_rows:
        question_id = str(row.get("question_id"))
        if (
            question_id in selected_ids
            and row.get("provider") == args.provider
            and row.get("mode") == args.mode
            and row.get("top_k") == args.top_k
        ):
            latest_by_id[question_id] = row
    report_rows = [latest_by_id[qid] for qid in selected_ids if qid in latest_by_id]
    metrics = aggregate_metrics(report_rows)
    provider_info = read_json(run_dir / "provider_manifest.json")
    metadata = {
        "suite_name": suite["suite_name"],
        "suite_sha256": suite_digest(suite),
        "manifest_question_id_list_sha256": manifest["question_id_list_sha256"],
        "selected_question_id_list_sha256": selected_question_id_list_sha256,
        "selected_question_ids": selected_question_ids,
        "dataset_sha256": suite["dataset"]["sha256"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_directory": str(run_dir),
        "provider_key": args.provider,
        "provider": provider_info,
        "mode": args.mode,
        "top_k": args.top_k,
        "result_file": str(result_path),
    }
    if models:
        metadata["models"] = models
    report = {"metadata": metadata, "metrics": metrics}
    write_json(run_dir / "metrics.json", report)
    report_text = external_markdown_report(metadata, metrics)
    (run_dir / "report.md").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"评估目录：{run_dir}")
    return 1 if metrics["errors"] or metrics["ingestion"]["partial_questions"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--provider", choices=("mem0", "graphiti", "supermemory"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--mode", choices=("retrieval", "full"), default="full")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id")
    parser.add_argument("--question-id-file", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dims", type=int, default=DEFAULT_EMBEDDING_DIMS)
    parser.add_argument("--session-retries", type=int, default=3)
    parser.add_argument("--provider-timeout", type=int, default=900)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose-sessions", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k 必须大于 0")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须大于 0")
    selectors = sum(
        value is not None
        for value in (args.limit, args.question_id, args.question_id_file)
    )
    if selectors > 1:
        raise ValueError("--limit、--question-id、--question-id-file 只能指定一个")
    if args.embedding_dims < 1:
        raise ValueError("--embedding-dims 必须大于 0")
    if args.session_retries < 1:
        raise ValueError("--session-retries 必须大于 0")
    if args.workers < 1:
        raise ValueError("--workers 必须大于 0")
    return command_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
