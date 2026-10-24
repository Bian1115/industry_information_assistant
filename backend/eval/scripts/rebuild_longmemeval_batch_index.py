#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""用 DashScope Batch 为固定 LongMemEval 子集重建可恢复的原生记忆索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from array import array
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(APP_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

load_dotenv(BACKEND_DIR / ".env")

import run_longmemeval_regression as regression  # noqa: E402
from service.memory_contracts import MemoryTurn  # noqa: E402
from service.memory_service import (  # noqa: E402
    MEMORY_MAX_ATOMIC_FACTS,
    _normalize_atomic_facts,
    _parse_summary_json,
)


TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}
# DashScope 同步接口接受日期快照，但 Batch API 当前只接受其支持列表中的别名。
DEFAULT_MODEL = "qwen-plus"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"


@dataclass
class SessionTask:
    custom_id: str
    question_id: str
    source_session_id: str
    session_index: int
    user_id: str
    session_id: str
    turns: List[MemoryTurn]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_dict(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return dict(value)


def download_content(client: Any, file_id: str) -> bytes:
    response = client.files.content(file_id)
    content = getattr(response, "content", None)
    if content is not None:
        return bytes(content)
    reader = getattr(response, "read", None)
    if callable(reader):
        return bytes(reader())
    raise RuntimeError(f"无法读取 DashScope 文件内容：{file_id}")


class BatchRunner:
    def __init__(self, client: Any, run_dir: Path, poll_seconds: int):
        self.client = client
        self.run_dir = run_dir
        self.poll_seconds = poll_seconds

    def run(
        self,
        name: str,
        endpoint: str,
        requests: Sequence[Mapping[str, Any]],
    ) -> List[dict]:
        input_path = self.run_dir / f"{name}.input.jsonl"
        output_path = self.run_dir / f"{name}.output.jsonl"
        state_path = self.run_dir / f"{name}.batch.json"
        if not state_path.exists():
            write_jsonl(input_path, requests)
        elif not input_path.exists():
            raise FileNotFoundError(f"{name} 的 checkpoint 存在，但输入文件缺失")
        expected_sha = file_sha256(input_path)

        if state_path.exists():
            state = regression.read_json(state_path)
            if state.get("input_sha256") != expected_sha:
                raise ValueError(f"{name} 的 Batch 输入与 checkpoint 不一致")
        else:
            with input_path.open("rb") as handle:
                uploaded = self.client.files.create(file=handle, purpose="batch")
            batch = self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint=endpoint,
                completion_window="24h",
                metadata={"ds_name": name},
            )
            state = {
                "name": name,
                "endpoint": endpoint,
                "input_sha256": expected_sha,
                "input_file_id": uploaded.id,
                "batch_id": batch.id,
                "status": getattr(batch, "status", "validating"),
            }
            regression.write_json(state_path, state)
            print(f"[{name}] 已提交 Batch：{batch.id}，请求数={len(requests)}", flush=True)

        while state.get("status") not in TERMINAL_BATCH_STATES:
            batch = self.client.batches.retrieve(state["batch_id"])
            snapshot = object_dict(batch)
            state.update({
                "status": snapshot.get("status"),
                "output_file_id": snapshot.get("output_file_id"),
                "error_file_id": snapshot.get("error_file_id"),
                "request_counts": snapshot.get("request_counts"),
            })
            regression.write_json(state_path, state)
            counts = state.get("request_counts") or {}
            print(
                f"[{name}] status={state['status']} counts={counts}",
                flush=True,
            )
            if state.get("status") not in TERMINAL_BATCH_STATES:
                time.sleep(self.poll_seconds)

        if state.get("status") != "completed":
            raise RuntimeError(
                f"DashScope Batch {name} 未成功：status={state.get('status')} "
                f"batch_id={state.get('batch_id')}"
            )
        if not output_path.exists():
            output_file_id = state.get("output_file_id")
            if not output_file_id:
                raise RuntimeError(f"DashScope Batch {name} 缺少 output_file_id")
            output_path.write_bytes(download_content(self.client, output_file_id))
        return regression.read_jsonl(output_path)


class EmbeddingCache:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(cache_key TEXT PRIMARY KEY, dimensions INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self.connection.commit()

    def keys(self) -> set[str]:
        return {row[0] for row in self.connection.execute("SELECT cache_key FROM embeddings")}

    def put(self, cache_key: str, vector: Sequence[float]) -> None:
        values = array("f", (float(value) for value in vector))
        self.connection.execute(
            "INSERT OR REPLACE INTO embeddings(cache_key, dimensions, vector) VALUES (?, ?, ?)",
            (cache_key, len(values), values.tobytes()),
        )

    def commit(self) -> None:
        self.connection.commit()

    def get(self, cache_key: str) -> List[float]:
        row = self.connection.execute(
            "SELECT dimensions, vector FROM embeddings WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            raise KeyError(cache_key)
        values = array("f")
        values.frombytes(row[1])
        if len(values) != row[0]:
            raise ValueError(f"embedding cache 损坏：{cache_key}")
        return values.tolist()


def build_tasks(adapter: Any, rows: Sequence[Mapping[str, Any]]) -> List[SessionTask]:
    tasks: List[SessionTask] = []
    for question_number, row in enumerate(rows, 1):
        question_id = str(row["question_id"])
        for session_index, (source_id, date_text, source_turns) in enumerate(zip(
            row["haystack_session_ids"],
            row["haystack_dates"],
            row["haystack_sessions"],
        )):
            created_at = regression.parse_longmemeval_date(str(date_text))
            turns = [
                MemoryTurn(
                    message_id=str(adapter.message_id(
                        question_id, str(source_id), session_index, turn_index
                    )),
                    role=str(source_turn.get("role", "")),
                    content=str(source_turn.get("content", "")),
                    created_at=created_at + timedelta(microseconds=turn_index),
                )
                for turn_index, source_turn in enumerate(source_turns)
            ]
            tasks.append(SessionTask(
                custom_id=f"q{question_number:02d}-s{session_index:03d}",
                question_id=question_id,
                source_session_id=str(source_id),
                session_index=session_index,
                user_id=str(adapter.user_id(question_id)),
                session_id=str(adapter.session_id(question_id, str(source_id), session_index)),
                turns=turns,
            ))
    return tasks


def extraction_prompt(task: SessionTask) -> str:
    conversation = "\n".join(
        f"[message_id={turn.message_id}; time={turn.created_at.isoformat(timespec='seconds')}] "
        f"{turn.role}: {turn.content}"
        for turn in task.turns
    )
    return f"""请从下面一个完整会话中一次性生成可检索记忆。只提取原文明示且能由 exact_quote 验证的事实，禁止推测、补全或修改名称、数字、URL和时间。

会话：
{conversation}

只输出一个合法 JSON object，不要 Markdown：
{{
  "summary": "用2-3句话概括会话主题和结论",
  "key_insights": ["3-5条简洁关键信息"],
  "user_preferences": {{
    "interests": ["明确表达的兴趣"],
    "communication_style": "明确体现的沟通偏好，没有则为空字符串",
    "focus_areas": ["明确关注领域"]
  }},
  "topics": ["简短主题标签"],
  "atomic_facts": [
    {{
      "fact": "可独立理解的单一事实",
      "subject": "主体",
      "predicate": "属性或关系",
      "value": "原文中的精确值",
      "occurred_at": null,
      "valid_from": null,
      "source_message_ids": ["真实 message_id"],
      "exact_quote": "支持事实的短原文片段"
    }}
  ]
}}
atomic_facts 最多 {MEMORY_MAX_ATOMIC_FACTS} 条；没有可靠事实时返回空数组。"""


def extraction_request(task: SessionTask, model: str) -> dict:
    return {
        "custom_id": task.custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是严谨的会话记忆抽取器，只输出合法 json object。",
                },
                {"role": "user", "content": extraction_prompt(task)},
            ],
            "temperature": 0,
            "max_tokens": 6000,
            "response_format": {"type": "json_object"},
        },
    }


def response_body(row: Mapping[str, Any]) -> dict:
    response = row.get("response") or {}
    if int(response.get("status_code") or 0) != 200:
        raise RuntimeError(json.dumps(response, ensure_ascii=False)[:1000])
    body = response.get("body")
    if not isinstance(body, dict):
        raise RuntimeError("Batch 响应缺少 body")
    return body


def extract_all(
    runner: BatchRunner,
    tasks: Sequence[SessionTask],
    model: str,
    run_dir: Path,
    attempts: int,
) -> Dict[str, dict]:
    output_path = run_dir / "extractions.json"
    extracted = regression.read_json(output_path) if output_path.exists() else {}
    task_by_id = {task.custom_id: task for task in tasks}
    errors: Dict[str, str] = {}
    for attempt in range(1, attempts + 1):
        pending = [task for task in tasks if task.custom_id not in extracted]
        if not pending:
            break
        rows = runner.run(
            f"extraction-round-{attempt}",
            "/v1/chat/completions",
            [extraction_request(task, model) for task in pending],
        )
        for row in rows:
            custom_id = str(row.get("custom_id") or "")
            task = task_by_id.get(custom_id)
            if not task:
                continue
            try:
                body = response_body(row)
                text = body["choices"][0]["message"]["content"]
                parsed = _parse_summary_json(str(text))
                parsed["atomic_facts"] = _normalize_atomic_facts(
                    parsed.get("atomic_facts"),
                    task.turns,
                    max_facts=MEMORY_MAX_ATOMIC_FACTS,
                )
                parsed["summary_provider"] = "dashscope-batch"
                parsed["fact_batch_count"] = 1
                extracted[custom_id] = parsed
                errors.pop(custom_id, None)
            except Exception as exc:
                errors[custom_id] = f"{type(exc).__name__}: {exc}"
        regression.write_json(output_path, extracted)
        regression.write_json(run_dir / "extraction_errors.json", errors)
        print(
            f"[抽取] round={attempt} 成功={len(extracted)}/{len(tasks)} "
            f"待重试={len(tasks) - len(extracted)}",
            flush=True,
        )
    missing = [task.custom_id for task in tasks if task.custom_id not in extracted]
    if missing:
        raise RuntimeError(f"仍有 {len(missing)} 个会话抽取失败：{missing[:10]}")
    return extracted


def materialize_database(adapter: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    from models.chat import ChatMessage, ChatSession
    from models.user import User

    for number, row in enumerate(rows, 1):
        question_id = str(row["question_id"])
        adapter._delete_existing(question_id)
        user_id = adapter.user_id(question_id)
        suffix = hashlib.sha256(question_id.encode()).hexdigest()[:12]
        db = adapter.SessionLocal()
        try:
            db.add(User(
                id=user_id,
                username=f"longmemeval_{suffix}",
                email=f"longmemeval_{suffix}@regression.invalid",
                hashed_password="benchmark-account-no-login",
                is_active=False,
            ))
            for session_index, (source_id, date_text, source_turns) in enumerate(zip(
                row["haystack_session_ids"],
                row["haystack_dates"],
                row["haystack_sessions"],
            )):
                session_id = adapter.session_id(question_id, str(source_id), session_index)
                created_at = regression.parse_longmemeval_date(str(date_text))
                db.add(ChatSession(
                    id=session_id,
                    user_id=user_id,
                    title=f"LongMemEval {source_id}",
                    session_type="longmemeval_regression",
                    created_at=created_at,
                    updated_at=created_at,
                ))
                for turn_index, source_turn in enumerate(source_turns):
                    message_id = adapter.message_id(
                        question_id, str(source_id), session_index, turn_index
                    )
                    db.add(ChatMessage(
                        id=message_id,
                        session_id=session_id,
                        turn_id=regression.deterministic_uuid(
                            "longmemeval-turn", str(message_id)
                        ),
                        role=str(source_turn.get("role", "")),
                        content=str(source_turn.get("content", "")),
                        created_at=created_at + timedelta(microseconds=turn_index),
                    ))
            db.commit()
            print(f"[数据库] {number}/{len(rows)} {question_id} 会话已写入", flush=True)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def prepare_documents(
    adapter: Any,
    tasks: Sequence[SessionTask],
    extracted: Mapping[str, dict],
    run_dir: Path,
) -> List[dict]:
    path = run_dir / "memory_documents.jsonl"
    if path.exists():
        return regression.read_jsonl(path)
    documents: List[dict] = []
    service = adapter.memory_service
    for number, task in enumerate(tasks, 1):
        db = adapter.SessionLocal()
        try:
            memory = service.prepare_memory_record(
                db,
                user_id=task.user_id,
                session_id=task.session_id,
                messages=task.turns,
                summary_data=extracted[task.custom_id],
                source_start_message_id=task.turns[0].message_id,
                source_end_message_id=task.turns[-1].message_id,
            )
            memory_id = str(memory.id)
            rows = service.build_memory_vector_documents(
                memory_id=memory_id,
                user_id=task.user_id,
                session_id=task.session_id,
                summary_data=extracted[task.custom_id],
            )
            for row in rows:
                row["memory_id"] = memory_id
                row["embedding_key"] = hashlib.sha256(
                    row["content"].encode("utf-8")
                ).hexdigest()
            documents.extend(rows)
        finally:
            db.close()
        if number % 25 == 0 or number == len(tasks):
            print(f"[记忆文档] {number}/{len(tasks)}", flush=True)
    write_jsonl(path, documents)
    return documents


def embedding_requests(
    keys: Sequence[str],
    content_by_key: Mapping[str, str],
    model: str,
    round_number: int,
) -> tuple[List[dict], Dict[str, List[str]]]:
    requests = []
    groups: Dict[str, List[str]] = {}
    for index in range(0, len(keys), 10):
        group = list(keys[index:index + 10])
        custom_id = f"emb-r{round_number}-{index // 10:05d}"
        groups[custom_id] = group
        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": model,
                "input": [content_by_key[key] for key in group],
                "dimensions": 1024,
                "encoding_format": "float",
            },
        })
    return requests, groups


def embed_all(
    runner: BatchRunner,
    documents: Sequence[Mapping[str, Any]],
    model: str,
    run_dir: Path,
    attempts: int,
) -> EmbeddingCache:
    cache = EmbeddingCache(run_dir / "embedding_cache.sqlite3")
    content_by_key = {
        str(document["embedding_key"]): str(document["content"])
        for document in documents
    }
    for attempt in range(1, attempts + 1):
        existing = cache.keys()
        pending = sorted(set(content_by_key) - existing)
        if not pending:
            break
        requests, groups = embedding_requests(
            pending, content_by_key, model, attempt
        )
        rows = runner.run(
            f"embedding-round-{attempt}",
            "/v1/embeddings",
            requests,
        )
        for row in rows:
            custom_id = str(row.get("custom_id") or "")
            keys = groups.get(custom_id)
            if not keys:
                continue
            try:
                body = response_body(row)
                values = sorted(body.get("data") or [], key=lambda item: item.get("index", 0))
                if len(values) != len(keys):
                    raise ValueError("embedding 返回数量不匹配")
                for key, item in zip(keys, values):
                    vector = item.get("embedding")
                    if not isinstance(vector, list) or len(vector) != 1024:
                        raise ValueError("embedding 维度不是 1024")
                    cache.put(key, vector)
            except Exception as exc:
                print(f"[embedding] {custom_id} 解析失败：{exc}", flush=True)
        cache.commit()
        print(
            f"[embedding] round={attempt} 成功={len(cache.keys())}/{len(content_by_key)}",
            flush=True,
        )
    missing = sorted(set(content_by_key) - cache.keys())
    if missing:
        raise RuntimeError(f"仍有 {len(missing)} 个 embedding 失败")
    return cache


def write_index(adapter: Any, documents: Sequence[dict], cache: EmbeddingCache) -> None:
    from models.chat import LongTermMemory

    grouped: Dict[str, List[dict]] = {}
    for source in documents:
        document = dict(source)
        document["vector"] = cache.get(str(document["embedding_key"]))
        grouped.setdefault(str(document["memory_id"]), []).append(document)

    service = adapter.memory_service
    for number, (memory_id, rows) in enumerate(grouped.items(), 1):
        db = adapter.SessionLocal()
        try:
            memory = db.query(LongTermMemory).filter(LongTermMemory.id == memory_id).first()
            if not memory:
                raise RuntimeError(f"数据库缺少记忆：{memory_id}")
            if memory.index_status == "ready" and memory.milvus_ids:
                continue
            milvus_ids = service.store_preembedded_memory_vectors(memory_id, rows)
            memory.milvus_ids = milvus_ids
            memory.index_status = "ready"
            memory.last_error = None
            db.commit()
        except Exception as exc:
            db.rollback()
            memory = db.query(LongTermMemory).filter(LongTermMemory.id == memory_id).first()
            if memory:
                memory.index_status = "failed"
                memory.last_error = str(exc)[:4000]
                db.commit()
            raise
        finally:
            db.close()
        if number % 25 == 0 or number == len(grouped):
            print(f"[Milvus] {number}/{len(grouped)} 条记忆 ready", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=os.getenv("MEMORY_BATCH_MODEL", DEFAULT_MODEL))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只生成并校验第一轮 Batch 输入，不提交外部任务",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 1 or args.limit > 10:
        raise ValueError("本批处理重建只允许 1-10 题")
    if args.attempts < 1:
        raise ValueError("--attempts 必须大于 0")
    regression.configure_dashscope_direct_connection()
    regression.validate_evaluation_runtime()

    suite = regression.read_json(regression.SUITE_PATH)
    selected, manifest = regression.prepare_suite(
        suite,
        regression.DEFAULT_DATASET_PATH,
        regression.DEFAULT_MANIFEST_PATH,
        allow_download=False,
    )
    selected = selected[:args.limit]
    run_dir = regression.DEFAULT_RESULTS_DIR / args.run_name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"运行目录已存在：{run_dir}；续跑请添加 --resume")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": args.run_name,
        "pipeline": "dashscope-batch-memory-index-v1",
        "limit": args.limit,
        "question_ids": [str(row["question_id"]) for row in selected],
        "dataset_sha256": suite["dataset"]["sha256"],
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "embedding_dimensions": 1024,
        "proxy_bypass": "dashscope.aliyuncs.com",
        "attempts": args.attempts,
    }
    config_path = run_dir / "batch_index_config.json"
    if config_path.exists() and regression.read_json(config_path) != config:
        raise ValueError("续跑配置与 batch_index_config.json 不一致")
    regression.write_json(config_path, config)

    adapter = regression.ProjectMemoryAdapter(str(suite["dataset"]["sha256"]))
    tasks = build_tasks(adapter, selected)
    print(
        f"固定前 {args.limit} 题，共 {len(tasks)} 个会话；"
        f"抽取模型={args.model}，embedding={args.embedding_model}",
        flush=True,
    )
    if args.prepare_only:
        input_path = run_dir / "extraction-round-1.input.jsonl"
        write_jsonl(
            input_path,
            [extraction_request(task, args.model) for task in tasks],
        )
        print(
            f"Batch 输入已准备：{input_path} bytes={input_path.stat().st_size}",
            flush=True,
        )
        return 0

    from service.llm_client import create_openai_client

    client = create_openai_client(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        max_retries=2,
    )
    runner = BatchRunner(client, run_dir, args.poll_seconds)
    extracted = extract_all(
        runner, tasks, args.model, run_dir, args.attempts
    )

    state_path = run_dir / "pipeline_state.json"
    state = regression.read_json(state_path) if state_path.exists() else {}
    if not state.get("database_materialized"):
        materialize_database(adapter, selected)
        state["database_materialized"] = True
        regression.write_json(state_path, state)

    documents = prepare_documents(adapter, tasks, extracted, run_dir)
    cache = embed_all(
        runner, documents, args.embedding_model, run_dir, args.attempts
    )
    write_index(adapter, documents, cache)
    state["index_ready"] = True
    state["ready_sessions"] = len(tasks)
    state["vector_documents"] = len(documents)
    regression.write_json(state_path, state)
    print(
        f"索引重建完成：questions={args.limit} sessions={len(tasks)} "
        f"documents={len(documents)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
