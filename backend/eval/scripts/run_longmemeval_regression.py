#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""LongMemEval-S 固定 100 题日常回归。

准备数据：
    cd backend
    .venv/bin/python eval/scripts/run_longmemeval_regression.py prepare

日常检索回归（复用已建立的记忆，缺失时才写入）：
    .venv/bin/python eval/scripts/run_longmemeval_regression.py run

重建记忆后回归：
    .venv/bin/python eval/scripts/run_longmemeval_regression.py run --rebuild

端到端回答 + LLM judge：
    .venv/bin/python eval/scripts/run_longmemeval_regression.py run --mode full

脚本把大数据、生成的具体题目清单和结果写到 eval/data/（已 gitignore）。
仓库中的 suite_100.json 锁定数据 SHA、选择算法、随机种子和题型配额；
因此不同机器首次 prepare 得到的具体 100 个 question_id 完全一致。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import sys
import time
import urllib.request
import uuid
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
SUITE_PATH = BACKEND_DIR / "eval" / "longmemeval_s" / "suite_100.json"
DEFAULT_DATA_DIR = BACKEND_DIR / "eval" / "data" / "longmemeval_s"
DEFAULT_DATASET_PATH = DEFAULT_DATA_DIR / "longmemeval_s_cleaned.json"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_DIR / "fixed_100_manifest.json"
DEFAULT_RESULTS_DIR = DEFAULT_DATA_DIR / "results"
UUID_NAMESPACE = uuid.UUID("38cdd376-8784-5e94-94fb-d7c8f51a8659")
DEFAULT_LONGMEMEVAL_ANSWER_MODEL = "qwen3.7-plus-2026-05-26"
DEFAULT_LONGMEMEVAL_JUDGE_MODEL = "claude-opus-5"
DEFAULT_ANSWER_MAX_TOKENS = 1000
DEFAULT_JUDGE_MAX_TOKENS = 1024
DEFAULT_LLM_REQUEST_ATTEMPTS = 3
CONTEXT_MEMORY_TYPES = {
    "current": ("summary", "insight", "preference", "topics"),
    "fact-only": ("atomic_fact",),
    "fact+preference": ("atomic_fact", "preference"),
    "hybrid": ("atomic_fact", "preference"),
}

load_dotenv(BACKEND_DIR / ".env")


def configure_dashscope_direct_connection() -> str:
    """让本次评估中的 DashScope 流量绕过系统 HTTP/SOCKS 代理。"""
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    hostname = urlparse(base_url).hostname
    if not hostname:
        raise ValueError(f"DASHSCOPE_BASE_URL 缺少合法主机名：{base_url!r}")

    for variable in ("NO_PROXY", "no_proxy"):
        entries = [
            value.strip()
            for value in os.getenv(variable, "").split(",")
            if value.strip()
        ]
        if hostname not in entries:
            entries.append(hostname)
        os.environ[variable] = ",".join(entries)
    return hostname


def full_model_config() -> dict:
    """返回 full 模式使用的固定回答模型与独立 Judge 配置。"""
    return {
        "answer_provider": "dashscope",
        "answer_model": os.getenv(
            "LONGMEMEVAL_ANSWER_MODEL", DEFAULT_LONGMEMEVAL_ANSWER_MODEL
        ),
        "answer_max_tokens": DEFAULT_ANSWER_MAX_TOKENS,
        "answer_prompt_version": "longmemeval-rag-v1",
        "answer_content_filter_policy": "record-refusal-and-judge",
        "answer_request_attempts": DEFAULT_LLM_REQUEST_ATTEMPTS,
        "judge_provider": "claude-relay",
        "judge_protocol": "anthropic-messages",
        "judge_model": os.getenv(
            "LONGMEMEVAL_JUDGE_MODEL",
            os.getenv("CLAUDE_RELAY_MODEL", DEFAULT_LONGMEMEVAL_JUDGE_MODEL),
        ),
        "judge_max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
        "judge_prompt_version": "longmemeval-adapted-official-v1",
        "judge_request_attempts": DEFAULT_LLM_REQUEST_ATTEMPTS,
    }


def memory_extraction_config() -> dict:
    """记录记忆抽取配置，确保重建结果可以复现和审计。"""
    provider = os.getenv("MEMORY_SUMMARY_PROVIDER", "dashscope").strip().lower()
    if provider in {"claude", "claude-relay"}:
        model = os.getenv(
            "MEMORY_SUMMARY_MODEL",
            os.getenv("CLAUDE_RELAY_MODEL", DEFAULT_LONGMEMEVAL_JUDGE_MODEL),
        )
        provider = "claude-relay"
    else:
        model = os.getenv(
            "MEMORY_SUMMARY_MODEL",
            os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
        )
    return {
        "provider": provider,
        "model": model,
        "fact_batch_max_chars": int(os.getenv("MEMORY_FACT_BATCH_MAX_CHARS", "6000")),
        "fact_batch_max_facts": int(os.getenv("MEMORY_FACT_BATCH_MAX_FACTS", "12")),
        "fact_max_tokens": int(os.getenv("MEMORY_FACT_MAX_TOKENS", "3000")),
        "max_atomic_facts": int(os.getenv("MEMORY_MAX_ATOMIC_FACTS", "40")),
        "claude_request_attempts": int(os.getenv("MEMORY_CLAUDE_REQUEST_ATTEMPTS", "5")),
        "dashscope_request_attempts": int(
            os.getenv("MEMORY_DASHSCOPE_REQUEST_ATTEMPTS", "3")
        ),
        "json_request_attempts": int(os.getenv("MEMORY_JSON_REQUEST_ATTEMPTS", "2")),
        "content_moderation_fallback": os.getenv(
            "MEMORY_CONTENT_MODERATION_FALLBACK", "claude-relay"
        ),
    }


def evaluation_runtime_config() -> dict:
    try:
        pymilvus_version = importlib.metadata.version("pymilvus")
    except importlib.metadata.PackageNotFoundError:
        pymilvus_version = None
    return {
        "python": platform.python_version(),
        "pymilvus": pymilvus_version,
        "dashscope_proxy_bypass": urlparse(
            os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        ).hostname,
    }


def validate_evaluation_runtime() -> None:
    runtime = evaluation_runtime_config()
    if runtime["pymilvus"] != "2.3.7":
        raise RuntimeError(
            "LongMemEval 当前连接 Milvus Server 2.3.3，必须使用 "
            "PyMilvus 2.3.7；当前为 "
            f"{runtime['pymilvus']!r}。请使用 backend/.venv-eval-py311/bin/python。"
        )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSONL") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def suite_digest(suite: Mapping[str, Any]) -> str:
    payload = json.dumps(suite, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def download_dataset(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "industry-information-assistant-longmemeval/1.0"},
    )
    print(f"下载 LongMemEval-S：{url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                if total and downloaded % (20 * 1024 * 1024) < len(block):
                    print(f"  {downloaded / 1024 / 1024:.0f}/{total / 1024 / 1024:.0f} MiB")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def validate_dataset(rows: Any, expected_questions: int) -> List[dict]:
    if not isinstance(rows, list):
        raise ValueError("LongMemEval 数据顶层必须是数组")
    if len(rows) != expected_questions:
        raise ValueError(f"题目数应为 {expected_questions}，实际为 {len(rows)}")

    required = {
        "question_id",
        "question_type",
        "question",
        "answer",
        "question_date",
        "haystack_session_ids",
        "haystack_dates",
        "haystack_sessions",
        "answer_session_ids",
    }
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"第 {index} 题不是对象")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"第 {index} 题缺少字段：{sorted(missing)}")
        question_id = str(row["question_id"])
        if question_id in seen:
            raise ValueError(f"question_id 重复：{question_id}")
        seen.add(question_id)
        session_lengths = (
            len(row["haystack_session_ids"]),
            len(row["haystack_dates"]),
            len(row["haystack_sessions"]),
        )
        if len(set(session_lengths)) != 1:
            raise ValueError(f"{question_id} 的 session id/date/content 数量不一致：{session_lengths}")
    return rows


def _rank_key(seed: str, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def select_fixed_questions(rows: Sequence[dict], selection: Mapping[str, Any]) -> List[dict]:
    """按固定 hash 排序分层抽样，不依赖输入文件的物理顺序。"""
    seed = str(selection["seed"])
    quotas = dict(selection["quotas"])
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        question_id = str(row["question_id"])
        group = "abstention" if question_id.endswith("_abs") else str(row["question_type"])
        groups[group].append(row)

    selected = []
    for group, quota in quotas.items():
        candidates = sorted(
            groups.get(group, []),
            key=lambda row: (_rank_key(seed, str(row["question_id"])), str(row["question_id"])),
        )
        if len(candidates) < int(quota):
            raise ValueError(f"{group} 只有 {len(candidates)} 题，无法抽取 {quota} 题")
        selected.extend(candidates[: int(quota)])

    expected = sum(int(value) for value in quotas.values())
    if len(selected) != expected or len({row["question_id"] for row in selected}) != expected:
        raise ValueError("固定题集选择结果数量或唯一性异常")
    return sorted(selected, key=lambda row: str(row["question_id"]))


def build_manifest(suite: Mapping[str, Any], selected: Sequence[dict]) -> dict:
    quotas = Counter(
        "abstention" if str(row["question_id"]).endswith("_abs") else row["question_type"]
        for row in selected
    )
    question_ids = [str(row["question_id"]) for row in selected]
    return {
        "suite_name": suite["suite_name"],
        "suite_sha256": suite_digest(suite),
        "dataset_sha256": suite["dataset"]["sha256"],
        "selection_algorithm": suite["selection"]["algorithm"],
        "selection_seed": suite["selection"]["seed"],
        "question_count": len(question_ids),
        "question_id_list_sha256": hashlib.sha256("\n".join(question_ids).encode()).hexdigest(),
        "distribution": dict(sorted(quotas.items())),
        "question_ids": question_ids,
    }


def prepare_suite(
    suite: Mapping[str, Any],
    dataset_path: Path,
    manifest_path: Path,
    *,
    allow_download: bool,
) -> tuple[List[dict], dict]:
    if not dataset_path.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"缺少 {dataset_path}；先运行 prepare，或手动放入官方 cleaned S 文件"
            )
        download_dataset(str(suite["dataset"]["url"]), dataset_path)

    actual_sha = sha256_file(dataset_path)
    expected_sha = str(suite["dataset"]["sha256"])
    if actual_sha != expected_sha:
        raise ValueError(
            "LongMemEval-S 文件 SHA256 不匹配，拒绝生成不可比的固定题集：\n"
            f"  expected={expected_sha}\n  actual={actual_sha}"
        )

    rows = validate_dataset(read_json(dataset_path), int(suite["dataset"]["expected_questions"]))
    selected = select_fixed_questions(rows, suite["selection"])
    expected_manifest = build_manifest(suite, selected)
    if manifest_path.exists():
        current_manifest = read_json(manifest_path)
        if current_manifest != expected_manifest:
            raise ValueError(
                f"{manifest_path} 与固定选择规则不一致；请删除该生成文件后重新 prepare"
            )
    else:
        write_json(manifest_path, expected_manifest)
    return selected, expected_manifest


def parse_longmemeval_date(value: str) -> datetime:
    """解析 `2023/01/19 (Thu) 10:47`，不依赖机器 locale。"""
    normalized = re.sub(r"\s*\([^)]*\)\s*", " ", value).strip()
    for pattern in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(normalized, pattern)
        except ValueError:
            pass
    raise ValueError(f"无法解析 LongMemEval 日期：{value!r}")


def deterministic_uuid(*parts: str) -> uuid.UUID:
    return uuid.uuid5(UUID_NAMESPACE, ":".join(parts))


@dataclass
class RetrievedEvidence:
    memory_id: str
    fragment_id: str
    memory_type: str
    content: str
    score: float
    source_session_id: Optional[str]
    app_session_id: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class IndexOutcome:
    status: str
    total_sessions: int
    ready_sessions: int
    failures: List[dict]


class ProjectMemoryAdapter:
    """把 LongMemEval 的一题映射到项目真实 PostgreSQL + MemoryProvider。"""

    def __init__(self, dataset_sha256: str, *, verbose_sessions: bool = False):
        if str(APP_DIR) not in sys.path:
            sys.path.insert(0, str(APP_DIR))
        from core.database import SessionLocal
        from service.memory_service import get_memory_provider, get_memory_service

        self.SessionLocal = SessionLocal
        self.provider = get_memory_provider()
        self.memory_service = get_memory_service()
        self.dataset_sha256 = dataset_sha256
        self.verbose_sessions = verbose_sessions

    def user_id(self, question_id: str) -> uuid.UUID:
        return deterministic_uuid("longmemeval-s", self.dataset_sha256, "user", question_id)

    def session_id(
        self,
        question_id: str,
        source_session_id: str,
        session_index: int,
    ) -> uuid.UUID:
        return deterministic_uuid(
            "longmemeval-s",
            self.dataset_sha256,
            "session",
            question_id,
            str(session_index),
            source_session_id,
        )

    def message_id(
        self,
        question_id: str,
        source_session_id: str,
        session_index: int,
        turn_index: int,
    ) -> uuid.UUID:
        return deterministic_uuid(
            "longmemeval-s",
            self.dataset_sha256,
            "message",
            question_id,
            source_session_id,
            str(session_index),
            str(turn_index),
        )

    def _delete_existing(self, question_id: str) -> None:
        from models.chat import LongTermMemory
        from models.user import User

        user_id = self.user_id(question_id)
        db = self.SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return
            has_memories = db.query(LongTermMemory.id).filter(
                LongTermMemory.user_id == user_id,
                LongTermMemory.index_status != "deleted",
            ).first()
        finally:
            db.close()

        if has_memories:
            self.provider.reset_user(str(user_id), f"longmemeval-reset:{question_id}")

        db = self.SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                db.delete(user)
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _is_complete(self, row: Mapping[str, Any]) -> bool:
        from models.chat import ChatSession, LongTermMemory
        from models.user import User

        user_id = self.user_id(str(row["question_id"]))
        expected_sessions = len(row["haystack_session_ids"])
        db = self.SessionLocal()
        try:
            if not db.query(User.id).filter(User.id == user_id).first():
                return False
            actual_sessions = db.query(ChatSession.id).filter(ChatSession.user_id == user_id).count()
            ready_sessions = db.query(LongTermMemory.session_id).filter(
                LongTermMemory.user_id == user_id,
                LongTermMemory.index_status == "ready",
                LongTermMemory.deleted_at.is_(None),
            ).distinct().count()
            return actual_sessions == expected_sessions and ready_sessions == expected_sessions
        finally:
            db.close()

    def ensure_index(self, row: Mapping[str, Any], *, rebuild: bool) -> str:
        from models.chat import ChatMessage, ChatSession, LongTermMemory
        from models.user import User
        from service.memory_contracts import MemoryScope, MemoryTurn

        question_id = str(row["question_id"])
        if not rebuild and self._is_complete(row):
            total_sessions = len(row["haystack_sessions"])
            return IndexOutcome("reused", total_sessions, total_sessions, [])
        if rebuild:
            self._delete_existing(question_id)

        user_id = self.user_id(question_id)
        suffix = hashlib.sha256(question_id.encode()).hexdigest()[:12]
        db = self.SessionLocal()
        sessions_with_turns = []
        try:
            existing_user = db.query(User).filter(User.id == user_id).first()
            expected_sessions = len(row["haystack_sessions"])
            expected_messages = sum(len(turns) for turns in row["haystack_sessions"])
            if existing_user:
                actual_sessions = db.query(ChatSession.id).filter(
                    ChatSession.user_id == user_id
                ).count()
                actual_messages = db.query(ChatMessage.id).join(
                    ChatSession, ChatMessage.session_id == ChatSession.id
                ).filter(ChatSession.user_id == user_id).count()
                if (
                    actual_sessions != expected_sessions
                    or actual_messages != expected_messages
                ):
                    db.close()
                    self._delete_existing(question_id)
                    db = self.SessionLocal()
                    existing_user = None

            if not existing_user:
                db.add(User(
                    id=user_id,
                    username=f"longmemeval_{suffix}",
                    email=f"longmemeval_{suffix}@regression.invalid",
                    hashed_password="benchmark-account-no-login",
                    is_active=False,
                ))
            for session_index, (source_session_id, date_text, source_turns) in enumerate(zip(
                row["haystack_session_ids"], row["haystack_dates"], row["haystack_sessions"]
            )):
                app_session_id = self.session_id(
                    question_id, str(source_session_id), session_index
                )
                session_date = parse_longmemeval_date(str(date_text))
                if not existing_user:
                    db.add(ChatSession(
                        id=app_session_id,
                        user_id=user_id,
                        title=f"LongMemEval {source_session_id}",
                        session_type="longmemeval_regression",
                        created_at=session_date,
                        updated_at=session_date,
                    ))
                memory_turns = []
                for turn_index, source_turn in enumerate(source_turns):
                    role = str(source_turn.get("role", ""))
                    if role not in {"user", "assistant", "system"}:
                        raise ValueError(f"{question_id}/{source_session_id} 含非法 role：{role!r}")
                    message_id = self.message_id(
                        question_id,
                        str(source_session_id),
                        session_index,
                        turn_index,
                    )
                    created_at = session_date + timedelta(microseconds=turn_index)
                    content = str(source_turn.get("content", ""))
                    if not existing_user:
                        db.add(ChatMessage(
                            id=message_id,
                            session_id=app_session_id,
                            turn_id=deterministic_uuid("longmemeval-turn", str(message_id)),
                            role=role,
                            content=content,
                            created_at=created_at,
                        ))
                    memory_turns.append(MemoryTurn(
                        message_id=str(message_id),
                        role=role,
                        content=content,
                        created_at=created_at,
                    ))
                sessions_with_turns.append((app_session_id, memory_turns, str(source_session_id)))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        db = self.SessionLocal()
        try:
            ready_session_ids = {
                str(session_id)
                for (session_id,) in db.query(LongTermMemory.session_id).filter(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.index_status == "ready",
                    LongTermMemory.deleted_at.is_(None),
                ).all()
            }
        finally:
            db.close()

        failures: List[dict] = []
        total_sessions = len(sessions_with_turns)
        for session_number, (app_session_id, memory_turns, source_session_id) in enumerate(
            sessions_with_turns, 1
        ):
            if str(app_session_id) in ready_session_ids:
                continue
            if self.verbose_sessions:
                print(
                    f"  [{question_id}] 写入会话 {session_number}/{total_sessions}: "
                    f"{source_session_id}",
                    flush=True,
                )
            try:
                self.provider.add_turns(
                    MemoryScope(user_id=str(user_id), session_id=str(app_session_id)),
                    memory_turns,
                    operation_id=f"longmemeval-add:{question_id}:{session_number}:{source_session_id}",
                )
            except Exception as exc:
                if getattr(exc, "abort_run", False):
                    raise
                failures.append({
                    "source_session_id": source_session_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                if self.verbose_sessions:
                    print(
                        f"  [{question_id}] 会话写入失败，继续后续会话："
                        f"{source_session_id} {type(exc).__name__}: {exc}",
                        flush=True,
                    )
        if failures:
            return IndexOutcome(
                status="partial",
                total_sessions=total_sessions,
                ready_sessions=total_sessions - len(failures),
                failures=failures,
            )
        if not self._is_complete(row):
            raise RuntimeError(f"{question_id} 写入结束后索引仍不完整")
        return IndexOutcome(
            status="rebuilt" if rebuild else ("resumed" if existing_user else "created"),
            total_sessions=total_sessions,
            ready_sessions=total_sessions,
            failures=[],
        )

    def search(
        self,
        row: Mapping[str, Any],
        top_k: int,
        context_mode: str = "current",
    ) -> List[RetrievedEvidence]:
        from service.memory_contracts import MemoryScope, MemorySearchOptions

        if context_mode not in CONTEXT_MEMORY_TYPES:
            raise ValueError(f"未知 context mode：{context_mode}")
        question_id = str(row["question_id"])
        user_id = str(self.user_id(question_id))
        source_by_app = {
            str(self.session_id(question_id, str(source_id), session_index)): str(source_id)
            for session_index, source_id in enumerate(row["haystack_session_ids"])
        }
        hits = self.provider.search(
            MemoryScope(user_id=user_id),
            str(row["question"]),
            MemorySearchOptions(
                limit=top_k,
                memory_types=CONTEXT_MEMORY_TYPES[context_mode],
            ),
        )
        raw_by_memory: Dict[str, List[dict]] = defaultdict(list)
        if context_mode == "hybrid" and hits:
            for item in self.memory_service._load_raw_evidence(
                user_id,
                str(row["question"]),
                hits,
            ):
                raw_by_memory[str(item.get("memory_id") or "")].append(item)

        evidence = [
            RetrievedEvidence(
                memory_id=hit.memory_id,
                fragment_id=hit.fragment_id,
                memory_type=hit.memory_type,
                content=self._hybrid_content(
                    hit.content,
                    raw_by_memory.get(hit.memory_id, []),
                ),
                score=hit.score,
                source_session_id=source_by_app.get(str(hit.session_id)),
                app_session_id=hit.session_id,
                metadata=dict(hit.metadata),
            )
            for hit in hits
        ]
        return evidence

    @staticmethod
    def _hybrid_content(content: str, raw_evidence: Sequence[Mapping[str, Any]]) -> str:
        if not raw_evidence:
            return content
        rendered = "\n".join(
            f"- [{item.get('created_at') or 'unknown'}; "
            f"message_id={item.get('message_id') or 'unknown'}; "
            f"role={item.get('role') or 'unknown'}] {item.get('content') or ''}"
            for item in raw_evidence
        )
        return f"{content}\n\n[Raw evidence]\n{rendered}"


def unique_session_ids(evidence: Sequence[RetrievedEvidence]) -> List[str]:
    seen = set()
    values = []
    for item in evidence:
        if item.source_session_id and item.source_session_id not in seen:
            seen.add(item.source_session_id)
            values.append(item.source_session_id)
    return values


def score_retrieval(gold_session_ids: Sequence[str], retrieved_session_ids: Sequence[str]) -> dict:
    gold = list(dict.fromkeys(str(value) for value in gold_session_ids))
    retrieved = list(dict.fromkeys(str(value) for value in retrieved_session_ids))
    gold_set = set(gold)
    recalled = [value for value in gold if value in set(retrieved)]
    first_rank = next((index + 1 for index, value in enumerate(retrieved) if value in gold_set), None)
    return {
        "gold_session_count": len(gold),
        "recalled_session_count": len(recalled),
        "any_recall": bool(recalled),
        "all_recall": bool(gold) and len(recalled) == len(gold),
        "session_recall": len(recalled) / len(gold) if gold else None,
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
    }


def build_answer_prompt(row: Mapping[str, Any], evidence: Sequence[RetrievedEvidence]) -> str:
    rendered_evidence = []
    for index, item in enumerate(evidence, 1):
        metadata = getattr(item, "metadata", {}) or {}
        attributes = [
            f"memory {index}",
            f"session={item.source_session_id or 'unknown'}",
            f"type={item.memory_type}",
            f"score={item.score:.4f}",
        ]
        if metadata.get("occurred_at"):
            attributes.append(f"occurred_at={metadata['occurred_at']}")
        if metadata.get("valid_from"):
            attributes.append(f"valid_from={metadata['valid_from']}")
        source_times = metadata.get("source_times") or []
        if not isinstance(source_times, list):
            source_times = [source_times]
        if source_times:
            attributes.append(f"source_time={source_times[0]}")
        rendered_evidence.append(f"[{' ; '.join(attributes)}]\n{item.content}")
    evidence_text = "\n\n".join(rendered_evidence) or "(no relevant memory was retrieved)"
    return f"""Answer the user's question using only the retrieved long-term memories below.
The memories may be incomplete or conflicting. Prefer the latest applicable information.
If the memories do not contain enough information, explicitly say that the question cannot be answered.
Answer directly and concisely. Do not mention the benchmark or the gold answer.

Question date: {row['question_date']}
Retrieved memories:
{evidence_text}

Question: {row['question']}"""


def build_judge_prompt(row: Mapping[str, Any], hypothesis: str) -> str:
    question = row["question"]
    answer = row["answer"]
    if str(row["question_id"]).endswith("_abs"):
        instruction = (
            "Answer yes only if the model correctly identifies the question as unanswerable or "
            "explains that the necessary information is missing."
        )
    elif row["question_type"] == "single-session-preference":
        instruction = (
            "The reference is a personalization rubric. Answer yes if the response correctly uses "
            "the user's personal information; it need not satisfy every rubric point."
        )
    elif row["question_type"] == "knowledge-update":
        instruction = (
            "Answer yes if the response contains the updated correct answer. Older information may "
            "also appear as long as the updated answer is clear."
        )
    elif row["question_type"] == "temporal-reasoning":
        instruction = (
            "Answer yes if the response is equivalent to the reference. Do not penalize an off-by-one "
            "error when counting days, weeks, or months."
        )
    else:
        instruction = (
            "Answer yes if the response contains the complete correct answer or equivalent reasoning. "
            "A response containing only a required subset is not correct."
        )
    return f"""You are evaluating a model response. {instruction}
Return exactly one word: yes or no.

Question: {question}
Reference answer or rubric: {answer}
Model response: {hypothesis}"""


def chat_completion(client: Any, model: str, prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def claude_messages_completion(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    """评估器对项目 Claude 原生客户端的薄封装。"""
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from service.claude_client import messages_completion

    return messages_completion(api_key, base_url, model, prompt, max_tokens)


def parse_judge_label(response: str) -> bool:
    normalized = response.strip().lower()
    if normalized not in {"yes", "no"}:
        raise RuntimeError(
            f"Claude Judge 应只返回 yes/no，实际返回：{response[:200]!r}"
        )
    return normalized == "yes"


def is_transient_llm_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        any(token in name for token in (
            "timeout",
            "connecterror",
            "connectionerror",
            "ratelimit",
        ))
        or "connection reset" in message
        or "没有 text 内容" in str(exc)
        or "只返回 yes/no" in str(exc)
    )


def call_with_retries(operation: Any, *, label: str, attempts: int) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts or not is_transient_llm_error(exc):
                raise
            delay = 0.5 * attempt
            logging.warning(
                "%s 第 %d/%d 次调用失败，将在 %.1fs 后重试：%s: %s",
                label,
                attempt,
                attempts,
                delay,
                type(exc).__name__,
                exc,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def is_content_filter_error(exc: Exception) -> bool:
    normalized = str(exc).lower()
    return "data_inspection_failed" in normalized or "inappropriate content" in normalized


def answer_and_judge(
    row: Mapping[str, Any], evidence: Sequence[RetrievedEvidence]
) -> tuple[str, bool, str, str, Optional[dict]]:
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from service.llm_client import create_openai_client

    answer_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not answer_api_key:
        raise RuntimeError("--mode full 需要 DASHSCOPE_API_KEY")
    answer_base_url = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    judge_api_key = os.getenv("CLAUDE_RELAY_API_KEY")
    judge_base_url = os.getenv("CLAUDE_RELAY_BASE_URL")
    if not judge_api_key or not judge_base_url:
        raise RuntimeError(
            "--mode full 的 Claude Judge 需要 CLAUDE_RELAY_API_KEY 和 "
            "CLAUDE_RELAY_BASE_URL"
        )

    models = full_model_config()
    answer_client = create_openai_client(
        api_key=answer_api_key, base_url=answer_base_url
    )
    answer_error = None
    try:
        hypothesis = call_with_retries(
            lambda: chat_completion(
                answer_client,
                models["answer_model"],
                build_answer_prompt(row, evidence),
                models["answer_max_tokens"],
            ),
            label="回答模型",
            attempts=models["answer_request_attempts"],
        )
    except Exception as exc:
        if not is_content_filter_error(exc):
            raise
        answer_error = {
            "type": type(exc).__name__,
            "code": "data_inspection_failed",
            "policy": models["answer_content_filter_policy"],
        }
        hypothesis = "[ANSWER_PROVIDER_REFUSAL: data_inspection_failed]"
    judge_response = call_with_retries(
        lambda: claude_messages_completion(
            judge_api_key,
            judge_base_url,
            models["judge_model"],
            build_judge_prompt(row, hypothesis),
            models["judge_max_tokens"],
        ),
        label="Judge 模型",
        attempts=models["judge_request_attempts"],
    )
    label = parse_judge_label(judge_response)
    return (
        hypothesis,
        label,
        models["answer_model"],
        models["judge_model"],
        answer_error,
    )


def aggregate_metrics(results: Sequence[Mapping[str, Any]]) -> dict:
    successful = [row for row in results if row.get("status") in {"ok", "partial"}]
    retrieval_rows = [row for row in successful if not row.get("abstention")]

    def average(values: Iterable[float]) -> Optional[float]:
        items = list(values)
        return sum(items) / len(items) if items else None

    recalled = sum(int(row["retrieval"]["recalled_session_count"]) for row in retrieval_rows)
    gold = sum(int(row["retrieval"]["gold_session_count"]) for row in retrieval_rows)
    retrieval = {
        "evaluated_questions": len(retrieval_rows),
        "any_recall": average(float(row["retrieval"]["any_recall"]) for row in retrieval_rows),
        "all_recall": average(float(row["retrieval"]["all_recall"]) for row in retrieval_rows),
        "macro_session_recall": average(
            float(row["retrieval"]["session_recall"]) for row in retrieval_rows
        ),
        "micro_session_recall": recalled / gold if gold else None,
        "mrr": average(float(row["retrieval"]["reciprocal_rank"]) for row in retrieval_rows),
        "recalled_sessions": recalled,
        "gold_sessions": gold,
    }
    judged = [row for row in successful if row.get("judge_label") is not None]
    by_type = {}
    for question_type in sorted({str(row["question_type"]) for row in successful}):
        typed = [row for row in successful if row["question_type"] == question_type]
        typed_retrieval = [row for row in typed if not row.get("abstention")]
        typed_judged = [row for row in typed if row.get("judge_label") is not None]
        by_type[question_type] = {
            "questions": len(typed),
            "retrieval_any_recall": average(
                float(row["retrieval"]["any_recall"]) for row in typed_retrieval
            ),
            "retrieval_all_recall": average(
                float(row["retrieval"]["all_recall"]) for row in typed_retrieval
            ),
            "qa_accuracy": average(float(row["judge_label"]) for row in typed_judged),
        }
    ingestion_rows = [row["ingestion"] for row in successful if row.get("ingestion")]
    total_sessions = sum(int(row["total_sessions"]) for row in ingestion_rows)
    ready_sessions = sum(int(row["ready_sessions"]) for row in ingestion_rows)
    ingestion = {
        "complete_questions": sum(bool(row.get("complete")) for row in ingestion_rows),
        "partial_questions": sum(not bool(row.get("complete")) for row in ingestion_rows),
        "ready_sessions": ready_sessions,
        "total_sessions": total_sessions,
        "session_success_rate": ready_sessions / total_sessions if total_sessions else None,
    }
    return {
        "total_rows": len(results),
        "successful_questions": len(successful),
        "errors": len(results) - len(successful),
        "ingestion": ingestion,
        "abstention_questions": sum(bool(row.get("abstention")) for row in successful),
        "retrieval": retrieval,
        "qa_evaluated_questions": len(judged),
        "qa_accuracy": average(float(row["judge_label"]) for row in judged),
        "by_question_type": by_type,
    }


def _format_percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def markdown_report(metadata: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    retrieval = metrics["retrieval"]
    ingestion = metrics["ingestion"]
    models = metadata.get("models") or {}
    lines = [
        f"# {metadata['suite_name']} 回归报告",
        "",
        f"- 运行时间：{metadata['created_at']}",
        f"- 模式：{metadata['mode']}",
        f"- Context Mode：{metadata.get('context_mode', 'current')}",
        f"- Top-K：{metadata['top_k']}",
        *(
            [
                f"- 回答模型：{models['answer_model']} ({models['answer_provider']})",
                f"- Judge 模型：{models['judge_model']} ({models['judge_provider']})",
            ]
            if models
            else []
        ),
        f"- 成功/总数：{metrics['successful_questions']}/{metrics['total_rows']}",
        f"- 错误：{metrics['errors']}",
        "",
        "## 总指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| Retrieval Any Recall@K | {_format_percent(retrieval['any_recall'])} |",
        f"| Retrieval All Recall@K | {_format_percent(retrieval['all_recall'])} |",
        f"| Macro Session Recall@K | {_format_percent(retrieval['macro_session_recall'])} |",
        f"| Micro Session Recall@K | {_format_percent(retrieval['micro_session_recall'])} |",
        f"| MRR | {retrieval['mrr'] if retrieval['mrr'] is not None else '-'} |",
        f"| QA Accuracy | {_format_percent(metrics['qa_accuracy'])} |",
        f"| Session Ingestion Success | {_format_percent(ingestion['session_success_rate'])} |",
        f"| Partial Questions | {ingestion['partial_questions']} |",
        "",
        "检索指标按 LongMemEval 官方约定排除 abstention；QA Accuracy 包含 abstention。",
        "",
        "## 分题型",
        "",
        "| 题型 | N | Any Recall | All Recall | QA Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for question_type, values in metrics["by_question_type"].items():
        lines.append(
            f"| {question_type} | {values['questions']} | "
            f"{_format_percent(values['retrieval_any_recall'])} | "
            f"{_format_percent(values['retrieval_all_recall'])} | "
            f"{_format_percent(values['qa_accuracy'])} |"
        )
    return "\n".join(lines) + "\n"


def metric_at(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    value: Any = metrics
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return float(value) if value is not None else None


def check_gates(metrics: Mapping[str, Any], args: argparse.Namespace) -> List[str]:
    failures = []
    absolute = {
        "retrieval.any_recall": args.min_any_recall,
        "retrieval.all_recall": args.min_all_recall,
        "qa_accuracy": args.min_qa_accuracy,
    }
    for key, minimum in absolute.items():
        actual = metric_at(metrics, key)
        if minimum is not None and (actual is None or actual < minimum):
            failures.append(f"{key}={actual} 低于门槛 {minimum}")

    if args.baseline:
        baseline_data = read_json(args.baseline)
        baseline_metadata = baseline_data.get("metadata", {})
        expected_metadata = {
            "dataset_sha256": args.dataset_sha256,
            "mode": args.mode,
            "context_mode": getattr(args, "context_mode", "current"),
            "top_k": args.top_k,
        }
        if args.mode == "full":
            expected_metadata["models"] = args.models
        for key, expected in expected_metadata.items():
            actual = baseline_metadata.get(
                key,
                "current" if key == "context_mode" else None,
            )
            if actual != expected:
                raise ValueError(
                    f"基线 {key}={actual!r}，本次为 {expected!r}；不同配置不能直接比较"
                )
        baseline_metrics = baseline_data.get("metrics", baseline_data)
        for key in ("retrieval.any_recall", "retrieval.all_recall", "qa_accuracy"):
            baseline = metric_at(baseline_metrics, key)
            actual = metric_at(metrics, key)
            if baseline is None or actual is None:
                continue
            if baseline - actual > args.max_regression_drop:
                failures.append(
                    f"{key} 从 {baseline:.4f} 降到 {actual:.4f}，超过允许降幅 "
                    f"{args.max_regression_drop:.4f}"
                )
    return failures


def run_one(
    adapter: ProjectMemoryAdapter,
    row: Mapping[str, Any],
    *,
    top_k: int,
    mode: str,
    context_mode: str,
    rebuild: bool,
) -> dict:
    started = time.monotonic()
    index_outcome = adapter.ensure_index(row, rebuild=rebuild)
    searched_at = time.monotonic()
    evidence = adapter.search(row, top_k, context_mode)
    if not evidence and context_mode == "current":
        raise RuntimeError(
            "记忆已标记为 ready，但检索没有返回任何碎片；请检查 Milvus 和 embedding 服务"
        )
    retrieved_sessions = unique_session_ids(evidence)
    abstention = str(row["question_id"]).endswith("_abs")
    result = {
        "question_id": row["question_id"],
        "question_type": row["question_type"],
        "abstention": abstention,
        "status": "partial" if index_outcome.failures else "ok",
        "mode": mode,
        "context_mode": context_mode,
        "top_k": top_k,
        "index_status": index_outcome.status,
        "ingestion": {
            "complete": not index_outcome.failures,
            "ready_sessions": index_outcome.ready_sessions,
            "total_sessions": index_outcome.total_sessions,
            "failures": index_outcome.failures,
        },
        "retrieved_session_ids": retrieved_sessions,
        "retrieval": None if abstention else score_retrieval(
            row["answer_session_ids"], retrieved_sessions
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
            "search": round(time.monotonic() - searched_at, 3),
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


def command_run(args: argparse.Namespace, suite: Mapping[str, Any]) -> int:
    validate_evaluation_runtime()
    selected, manifest = prepare_suite(
        suite, args.dataset, args.manifest, allow_download=not args.no_download
    )
    models = full_model_config() if args.mode == "full" else None
    args.models = models
    if args.limit is not None:
        selected = selected[: args.limit]
    selected_by_id = {str(row["question_id"]): row for row in selected}
    selected_question_ids = list(selected_by_id)
    selected_question_id_list_sha256 = hashlib.sha256(
        "\n".join(selected_question_ids).encode()
    ).hexdigest()

    run_name = args.run_name or (
        f"{datetime.now():%Y%m%d_%H%M%S}_{args.context_mode}_{args.mode}_k{args.top_k}"
    )
    run_dir = args.results_root / run_name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"评估目录已存在，拒绝覆盖：{run_dir}；如需断点续跑请显式添加 --resume"
        )
    if args.resume and not run_dir.exists():
        raise FileNotFoundError(f"--resume 指定的评估目录不存在：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshots = {
        "suite_snapshot.json": dict(suite),
        "fixed_100_manifest_snapshot.json": manifest,
    }
    for filename, snapshot in snapshots.items():
        snapshot_path = run_dir / filename
        if snapshot_path.exists():
            if read_json(snapshot_path) != snapshot:
                raise ValueError(f"断点目录中的 {filename} 与本次配置不一致")
        else:
            write_json(snapshot_path, snapshot)

    run_config = {
        "run_name": run_name,
        "mode": args.mode,
        "context_mode": args.context_mode,
        "top_k": args.top_k,
        "limit": args.limit,
        "rebuild": args.rebuild,
        "workers": args.workers,
        "dataset_sha256": suite["dataset"]["sha256"],
        "suite_sha256": suite_digest(suite),
        "selected_question_ids": selected_question_ids,
        "selected_question_id_list_sha256": selected_question_id_list_sha256,
        "memory_extraction": memory_extraction_config(),
        "runtime": evaluation_runtime_config(),
    }
    if models is not None:
        run_config["models"] = models
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        if read_json(run_config_path) != run_config:
            raise ValueError("断点目录中的 run_config.json 与本次参数不一致")
    else:
        write_json(run_config_path, run_config)

    result_path = run_dir / "questions.jsonl"
    prior_rows = read_jsonl(result_path)
    successful_ids = {
        str(row["question_id"])
        for row in prior_rows
        if row.get("status") == "ok"
        and row.get("mode") == args.mode
        and row.get("context_mode", "current") == args.context_mode
        and row.get("top_k") == args.top_k
        and str(row.get("question_id")) in selected_by_id
    }
    if args.rebuild and successful_ids and not args.resume:
        raise ValueError(
            f"{result_path} 已有成功结果；--rebuild 请换 --run-name 或删除对应结果文件"
        )

    adapter = ProjectMemoryAdapter(
        str(suite["dataset"]["sha256"]),
        verbose_sessions=args.verbose_sessions,
    )
    if args.workers > 1 and hasattr(adapter.provider, "service"):
        # 避免多个线程同时懒初始化 OpenAI/Milvus 客户端。
        _ = adapter.provider.service.client
        _ = adapter.provider.service.milvus
    pending = [row for row in selected if str(row["question_id"]) not in successful_ids]
    print(
        f"固定题集 {len(selected)} 题；已完成 {len(successful_ids)}；待运行 {len(pending)}；"
        f"mode={args.mode} context={args.context_mode} top_k={args.top_k}"
    )
    def evaluate_row(row: Mapping[str, Any]) -> dict:
        question_id = str(row["question_id"])
        try:
            return run_one(
                adapter,
                row,
                top_k=args.top_k,
                mode=args.mode,
                context_mode=args.context_mode,
                rebuild=args.rebuild,
            )
        except Exception as exc:
            if getattr(exc, "abort_run", False):
                raise
            return {
                "question_id": question_id,
                "question_type": row["question_type"],
                "abstention": question_id.endswith("_abs"),
                "status": "error",
                "mode": args.mode,
                "context_mode": args.context_mode,
                "top_k": args.top_k,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def persist_result(completed: int, result: Mapping[str, Any]) -> None:
        question_id = str(result["question_id"])
        if result["status"] == "error":
            print(
                f"[{completed}/{len(pending)}] {question_id} ERROR "
                f"{result['error_type']}: {result['error']}",
                flush=True,
            )
        else:
            metric = result["retrieval"]
            recall_text = "abstention" if metric is None else (
                f"any={int(metric['any_recall'])} all={int(metric['all_recall'])}"
            )
            status_text = "PARTIAL" if result["status"] == "partial" else "OK"
            print(
                f"[{completed}/{len(pending)}] {question_id} {status_text} {recall_text}",
                flush=True,
            )
        append_jsonl(result_path, result)

    if args.workers == 1:
        for completed, row in enumerate(pending, 1):
            persist_result(completed, evaluate_row(row))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_row, row) for row in pending]
            for completed, future in enumerate(as_completed(futures), 1):
                persist_result(completed, future.result())

    all_rows = read_jsonl(result_path)
    latest_by_id = {}
    for row in all_rows:
        question_id = str(row.get("question_id"))
        if (
            question_id in selected_by_id
            and row.get("mode") == args.mode
            and row.get("context_mode", "current") == args.context_mode
            and row.get("top_k") == args.top_k
        ):
            latest_by_id[question_id] = row
    report_rows = [latest_by_id[qid] for qid in selected_by_id if qid in latest_by_id]
    metrics = aggregate_metrics(report_rows)
    metadata = {
        "suite_name": suite["suite_name"],
        "suite_sha256": suite_digest(suite),
        "manifest_question_id_list_sha256": manifest["question_id_list_sha256"],
        "selected_question_ids": selected_question_ids,
        "selected_question_id_list_sha256": selected_question_id_list_sha256,
        "dataset_sha256": suite["dataset"]["sha256"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_directory": str(run_dir),
        "mode": args.mode,
        "context_mode": args.context_mode,
        "top_k": args.top_k,
        "result_file": str(result_path),
        "memory_extraction": memory_extraction_config(),
        "runtime": evaluation_runtime_config(),
    }
    if models is not None:
        metadata["models"] = models
    report = {"metadata": metadata, "metrics": metrics}
    metrics_path = run_dir / "metrics.json"
    markdown_path = run_dir / "report.md"
    write_json(metrics_path, report)
    markdown_path.write_text(markdown_report(metadata, metrics), encoding="utf-8")

    args.dataset_sha256 = suite["dataset"]["sha256"]
    failures = check_gates(metrics, args)
    print(markdown_report(metadata, metrics))
    print(f"评估目录：{run_dir}\n逐题结果：{result_path}\n指标：{metrics_path}\n报告：{markdown_path}")
    if metrics["errors"]:
        failures.append(f"有 {metrics['errors']} 道题运行失败")
    if metrics["ingestion"]["partial_questions"]:
        failures.append(
            f"有 {metrics['ingestion']['partial_questions']} 道题的历史会话未完整写入"
        )
    if metrics["successful_questions"] != len(selected):
        failures.append(
            f"成功题数 {metrics['successful_questions']}，应为 {len(selected)}"
        )
    if failures:
        print("回归门禁失败：")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    if args.write_baseline:
        write_json(args.write_baseline, report)
        print(f"基线已写入：{args.write_baseline}")
    return 0


def build_parser(suite: Mapping[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--mode", choices=("retrieval", "full"), default=suite["defaults"]["mode"])
    parser.add_argument(
        "--context-mode",
        choices=tuple(CONTEXT_MEMORY_TYPES),
        default="current",
        help=(
            "回答上下文消融：current=摘要/洞察/偏好/主题；"
            "fact-only=原子事实；fact+preference=原子事实+偏好；"
            "hybrid=原子事实+偏好+来源原文回填"
        ),
    )
    parser.add_argument("--top-k", type=int, default=int(suite["defaults"]["top_k"]))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="并行评估题目数，默认 1")
    parser.add_argument(
        "--verbose-sessions",
        action="store_true",
        help="打印每个历史会话的写入进度；完整 100 题通常不需要",
    )
    parser.add_argument("--limit", type=int, help="仅用于 smoke test；正式日常回归不要设置")
    parser.add_argument(
        "--run-name",
        help="独立结果目录名；默认使用精确到秒的时间戳、模式和 Top-K",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="显式续跑已存在的 --run-name；不设置时绝不覆盖同名目录",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--write-baseline", type=Path)
    parser.add_argument(
        "--max-regression-drop",
        type=float,
        default=float(suite["defaults"]["max_regression_drop"]),
    )
    parser.add_argument("--min-any-recall", type=float)
    parser.add_argument("--min-all-recall", type=float)
    parser.add_argument("--min-qa-accuracy", type=float)
    return parser


def main() -> int:
    dashscope_hostname = configure_dashscope_direct_connection()
    suite = read_json(SUITE_PATH)
    args = build_parser(suite).parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k 必须大于 0")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须大于 0")
    if args.workers < 1:
        raise ValueError("--workers 必须大于 0")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("service.memory_service").setLevel(logging.WARNING)
    logging.getLogger("service.milvus_service").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=".*PyMilvusDeprecationWarning.*")
    logging.getLogger(__name__).info(
        "DashScope 直连已启用（NO_PROXY=%s）；保留客户端与任务级重试",
        dashscope_hostname,
    )

    if args.command == "prepare":
        selected, manifest = prepare_suite(
            suite, args.dataset, args.manifest, allow_download=not args.no_download
        )
        print(f"数据校验通过：{args.dataset}")
        print(f"固定清单：{args.manifest}")
        print(f"题数：{len(selected)}；分布：{manifest['distribution']}")
        print(f"question_id list SHA256：{manifest['question_id_list_sha256']}")
        return 0
    return command_run(args, suite)


if __name__ == "__main__":
    raise SystemExit(main())
