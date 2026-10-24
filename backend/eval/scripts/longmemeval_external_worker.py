#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""Provider-specific worker for LongMemEval external memory baselines.

The parent runner talks to this process over newline-delimited JSON.  Keeping
provider imports in a child process lets Mem0 and Graphiti use pinned, isolated
virtual environments without changing the application's dependency set.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _json_response(value: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}")
    return value


def _iso_date(value: str) -> str:
    for pattern in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return value


def _session_text(messages: list[dict], session_date: str) -> str:
    lines = [f"Session date: {session_date}"] if session_date else []
    for message in messages:
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", ""))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


class Mem0Worker:
    def __init__(self, args: argparse.Namespace):
        # mem0's OpenAI adapter otherwise lets a machine-wide OpenRouter key
        # override the explicitly configured compatible endpoint.
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ["MEM0_TELEMETRY"] = "false"
        os.environ["MEM0_DIR"] = str(args.state_dir / "mem0_home")

        from mem0 import Memory

        api_key = _require_env("BASELINE_LLM_API_KEY")
        base_url = _require_env("BASELINE_LLM_BASE_URL")
        qdrant_path = args.state_dir / "qdrant"
        config = {
            "version": "v1.1",
            "history_db_path": str(args.state_dir / "history.db"),
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "longmemeval_mem0",
                    "embedding_model_dims": args.embedding_dims,
                    "path": str(qdrant_path),
                    "on_disk": True,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": args.llm_model,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "temperature": 0,
                    "max_tokens": 2000,
                    "is_reasoning_model": False,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": args.embedding_model,
                    "embedding_dims": args.embedding_dims,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                },
            },
        }
        self.memory = Memory.from_config(config)
        self.args = args
        self.bm25_status = "unavailable"
        try:
            # FastEmbed's BM25 implementation only needs a stopword file, but
            # its default constructor contacts Hugging Face to obtain it.  Build
            # the equivalent local resource from the already pinned spaCy
            # English stopword set so benchmark startup is deterministic and
            # does not silently lose Mem0's keyword retrieval on network errors.
            from fastembed import SparseTextEmbedding
            from spacy.lang.en.stop_words import STOP_WORDS

            bm25_dir = args.state_dir / "bm25_model"
            bm25_dir.mkdir(parents=True, exist_ok=True)
            (bm25_dir / "english.txt").write_text(
                "\n".join(sorted(STOP_WORDS)) + "\n", encoding="utf-8"
            )
            (bm25_dir / "mock.file").touch()
            self.memory.vector_store._bm25_encoder = SparseTextEmbedding(
                model_name="Qdrant/bm25",
                specific_model_path=str(bm25_dir),
            )
            self.bm25_status = "enabled-local-spacy-stopwords"
        except Exception as exc:
            logging.getLogger(__name__).warning("BM25 initialization failed: %s", exc)
            self.memory.vector_store._bm25_encoder = False
            self.bm25_status = f"disabled:{type(exc).__name__}"

    def info(self) -> dict:
        return {
            "provider": "mem0_oss",
            "package": "mem0ai",
            "version": importlib.metadata.version("mem0ai"),
            "repository": "https://github.com/mem0ai/mem0",
            "storage": "embedded-qdrant",
            "ingestion_unit": "one LongMemEval session per add()",
            "retrieval": "Mem0 OSS search, rerank=false, threshold=0",
            "bm25": self.bm25_status,
        }

    def add(self, command: Mapping[str, Any]) -> dict:
        session_date = str(command.get("session_date") or "")
        messages = list(command.get("messages") or [])
        # OSS does not accept Mem0 Platform's timestamp argument.  Supplying the
        # date both as extraction context and created_at keeps temporal facts and
        # storage chronology observable without pretending to use a cloud feature.
        dated_messages = ([{
            "role": "system",
            "content": f"The following conversation happened on {session_date}.",
        }] if session_date else []) + messages
        result = self.memory.add(
            dated_messages,
            user_id=str(command["scope_id"]),
            metadata={
                "source_session_id": str(command["source_session_id"]),
                "ingestion_session_id": str(command.get("ingestion_session_id") or command["source_session_id"]),
                "session_date": session_date,
                "created_at": _iso_date(session_date),
            },
        )
        return {
            "stored_items": len(result.get("results") or []),
            "events": [item.get("event") for item in result.get("results") or []],
        }

    def search(self, command: Mapping[str, Any]) -> dict:
        response = self.memory.search(
            str(command["query"]),
            top_k=int(command["top_k"]),
            filters={"user_id": str(command["scope_id"])},
            threshold=0,
            rerank=False,
            explain=True,
        )
        evidence = []
        for rank, item in enumerate(response.get("results") or [], 1):
            metadata = item.get("metadata") or {}
            source_id = metadata.get("source_session_id")
            evidence.append({
                "memory_id": str(item.get("id") or f"mem0-rank-{rank}"),
                "fragment_id": str(item.get("id") or f"mem0-rank-{rank}"),
                "memory_type": "mem0_fact",
                "content": str(item.get("memory") or ""),
                "score": float(item.get("score") or 0),
                "source_session_ids": [str(source_id)] if source_id else [],
                "metadata": metadata,
                "rank": rank,
            })
        return {"evidence": evidence}

    def finalize_scope(self, command: Mapping[str, Any]) -> dict:
        return {"finalized": True}

    def close(self) -> None:
        clients = []
        for store in (getattr(self.memory, "vector_store", None), getattr(self.memory, "_entity_store", None)):
            client = getattr(store, "client", None)
            if client is not None and client not in clients:
                clients.append(client)
        for client in clients:
            with contextlib.suppress(Exception):
                client.close()
        with contextlib.suppress(Exception):
            self.memory.db.close()


class GraphitiWorker:
    _FTS_INDEXES = (
        ("Entity", "node_name_and_summary", "['name', 'summary']"),
        ("RelatesToNode_", "edge_name_and_fact", "['name', 'fact']"),
        ("Community", "community_name", "['name']"),
        ("Episodic", "episode_content", "['content', 'source', 'source_description']"),
    )

    def __init__(self, args: argparse.Namespace):
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
        os.environ.setdefault("SEMAPHORE_LIMIT", "4")

        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        api_key = _require_env("BASELINE_LLM_API_KEY")
        base_url = _require_env("BASELINE_LLM_BASE_URL")
        llm_config = LLMConfig(
            api_key=api_key,
            model=args.llm_model,
            small_model=args.llm_model,
            base_url=base_url,
            temperature=0,
            max_tokens=8192,
        )
        llm_client = OpenAIGenericClient(
            config=llm_config,
            max_tokens=8192,
            structured_output_mode="json_object",
        )
        embedder = OpenAIEmbedder(OpenAIEmbedderConfig(
            api_key=api_key,
            base_url=base_url,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dims,
        ))
        cross_encoder = OpenAIRerankerClient(
            config=llm_config,
            client=llm_client.client,
        )
        self.driver = KuzuDriver(db=str(args.state_dir / "kuzu"))
        # graphiti-core 0.29.3's deprecated Kuzu driver does not initialize the
        # base class' database marker although add_episode() reads it.
        self.driver._database = ""
        self.graphiti = Graphiti(
            graph_driver=self.driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            store_raw_episode_content=False,
            max_coroutines=4,
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.args = args
        self.episode_map_path = args.state_dir / "episode_sources.json"
        self.episode_sources: dict[str, str] = _read_json(self.episode_map_path, {})
        self._ensure_empty_fts()

    def _run(self, coroutine: Any) -> Any:
        return self.loop.run_until_complete(coroutine)

    def _ensure_empty_fts(self) -> None:
        async def create() -> None:
            for table, index_name, columns in self._FTS_INDEXES:
                try:
                    await self.driver.execute_query(
                        f"CALL CREATE_FTS_INDEX('{table}', '{index_name}', {columns})"
                    )
                except Exception:
                    pass
        self._run(create())

    def _rebuild_fts(self) -> None:
        async def rebuild() -> None:
            for table, index_name, columns in self._FTS_INDEXES:
                with contextlib.suppress(Exception):
                    await self.driver.execute_query(
                        f"CALL DROP_FTS_INDEX('{table}', '{index_name}')"
                    )
                await self.driver.execute_query(
                    f"CALL CREATE_FTS_INDEX('{table}', '{index_name}', {columns})"
                )
        self._run(rebuild())

    def info(self) -> dict:
        return {
            "provider": "graphiti_oss",
            "package": "graphiti-core",
            "version": importlib.metadata.version("graphiti-core"),
            "repository": "https://github.com/getzep/graphiti",
            "storage": "embedded-kuzu (deprecated upstream; isolated benchmark runtime)",
            "ingestion_unit": "one LongMemEval session per add_episode()",
            "retrieval": "Graphiti edge hybrid RRF search",
        }

    def add(self, command: Mapping[str, Any]) -> dict:
        from graphiti_core.nodes import EpisodeType

        scope_id = str(command["scope_id"])
        source_id = str(command["source_session_id"])
        ingestion_id = str(command.get("ingestion_session_id") or source_id)
        session_date = str(command.get("session_date") or "")
        reference_time = datetime.fromisoformat(_iso_date(session_date))
        result = self._run(self.graphiti.add_episode(
            name=ingestion_id,
            episode_body=_session_text(list(command.get("messages") or []), session_date),
            source_description=f"LongMemEval source session {source_id}",
            reference_time=reference_time,
            source=EpisodeType.message,
            group_id=scope_id,
        ))
        episode_uuid = str(result.episode.uuid)
        self.episode_sources[episode_uuid] = source_id
        _write_json(self.episode_map_path, self.episode_sources)
        return {
            "stored_items": len(result.edges),
            "episode_uuid": episode_uuid,
            "entities": len(result.nodes),
            "edges": len(result.edges),
        }

    def finalize_scope(self, command: Mapping[str, Any]) -> dict:
        self._rebuild_fts()
        return {"finalized": True}

    def search(self, command: Mapping[str, Any]) -> dict:
        edges = self._run(self.graphiti.search(
            query=str(command["query"]),
            group_ids=[str(command["scope_id"])],
            num_results=int(command["top_k"]),
        ))
        evidence = []
        for rank, edge in enumerate(edges, 1):
            source_ids = []
            for episode_id in edge.episodes:
                source_id = self.episode_sources.get(str(episode_id))
                if source_id and source_id not in source_ids:
                    source_ids.append(source_id)
            evidence.append({
                "memory_id": str(edge.uuid),
                "fragment_id": str(edge.uuid),
                "memory_type": "graphiti_edge",
                "content": str(edge.fact),
                "score": 1.0 / rank,
                "source_session_ids": source_ids,
                "metadata": {
                    "relation": edge.name,
                    "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                    "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                    "reference_time": edge.reference_time.isoformat() if edge.reference_time else None,
                },
                "rank": rank,
            })
        return {"evidence": evidence}

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._run(self.graphiti.close())
        with contextlib.suppress(Exception):
            self.loop.close()


class SupermemoryWorker:
    """Thin HTTP adapter supporting both hosted and self-hosted Supermemory."""

    def __init__(self, args: argparse.Namespace):
        self.base_url = _require_env("SUPERMEMORY_BASE_URL").rstrip("/")
        self.api_key = os.getenv("SUPERMEMORY_API_KEY", "").strip()
        hostname = (urllib.parse.urlparse(self.base_url).hostname or "").lower()
        if not self.api_key and hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError("非本地 Supermemory 需要 SUPERMEMORY_API_KEY")
        self.args = args
        run_directory = args.state_dir.parent.parent.resolve()
        namespace = hashlib.sha256(str(run_directory).encode()).hexdigest()[:12]
        self.scope_prefix = f"longmemeval_{namespace}"
        self.documents_path = args.state_dir / "supermemory_documents.json"
        self.document_ids: dict[str, list[str]] = _read_json(self.documents_path, {})

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "longmemeval-supermemory-baseline/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"Supermemory HTTP {exc.code}: {body}") from exc

    def info(self) -> dict:
        return {
            "provider": "supermemory",
            "repository": "https://github.com/supermemoryai/supermemory",
            "version": os.getenv("SUPERMEMORY_SERVER_VERSION", "unknown"),
            "base_url": self.base_url,
            "deployment": "self-hosted" if not self.api_key else "authenticated-http",
            "scope_namespace": self.scope_prefix,
            "embedding": os.getenv(
                "SUPERMEMORY_EMBEDDING_MODEL", "Xenova/bge-base-en-v1.5"
            ),
            "ingestion_unit": "one LongMemEval session per document",
            "retrieval": "Supermemory hybrid search",
        }

    def add(self, command: Mapping[str, Any]) -> dict:
        source_id = str(command["source_session_id"])
        ingestion_id = str(command.get("ingestion_session_id") or source_id)
        provider_scope_id = f"{self.scope_prefix}_{command['scope_id']}"
        payload = {
            "content": _session_text(
                list(command.get("messages") or []), str(command.get("session_date") or "")
            ),
            "containerTag": provider_scope_id,
            "customId": ingestion_id,
            "metadata": {
                "type": "conversation",
                "source_session_id": source_id,
                "ingestion_session_id": ingestion_id,
                "session_date": str(command.get("session_date") or ""),
            },
            "dreaming": "instant",
        }
        result = self._request("POST", "/v3/documents", payload)
        document_id = result.get("id") or result.get("documentId")
        if not document_id:
            raise RuntimeError(f"Supermemory add 未返回 document id：{result}")
        scope_documents = self.document_ids.setdefault(str(command["scope_id"]), [])
        if str(document_id) not in scope_documents:
            scope_documents.append(str(document_id))
            _write_json(self.documents_path, self.document_ids)
        return {
            "stored_items": 1,
            "document_id": document_id,
        }

    def finalize_scope(self, command: Mapping[str, Any]) -> dict:
        scope_id = str(command["scope_id"])
        pending = set(self.document_ids.get(scope_id) or [])
        deadline = time.monotonic() + float(command.get("timeout_seconds") or 900)
        while pending and time.monotonic() < deadline:
            completed = set()
            for document_id in pending:
                result = self._request(
                    "GET", f"/v3/documents/{urllib.parse.quote(document_id, safe='')}"
                )
                status = str(result.get("status") or "").lower()
                if status == "failed":
                    raise RuntimeError(f"Supermemory document {document_id} 处理失败")
                if status == "done":
                    completed.add(document_id)
            pending -= completed
            if pending:
                time.sleep(1.5)
        if pending:
            raise TimeoutError(f"Supermemory 等待 {len(pending)} 个文档处理完成超时")
        return {"finalized": True, "documents": len(self.document_ids.get(scope_id) or [])}

    def search(self, command: Mapping[str, Any]) -> dict:
        result = self._request("POST", "/v4/search", {
            "q": str(command["query"]),
            "containerTag": f"{self.scope_prefix}_{command['scope_id']}",
            "limit": int(command["top_k"]),
            "searchMode": "hybrid",
            "include": {"relatedMemories": True},
        })
        rows = result.get("results") or result.get("memories") or []
        evidence = []
        for rank, item in enumerate(rows[: int(command["top_k"])], 1):
            metadata = item.get("metadata") or {}
            source_id = (
                metadata.get("source_session_id")
                or item.get("customId")
                or item.get("custom_id")
            )
            content = (
                item.get("memory")
                or item.get("content")
                or item.get("chunk")
                or item.get("text")
                or ""
            )
            evidence.append({
                "memory_id": str(item.get("id") or f"supermemory-rank-{rank}"),
                "fragment_id": str(item.get("chunkId") or item.get("id") or rank),
                "memory_type": "supermemory_hybrid",
                "content": str(content),
                "score": float(item.get("score") or item.get("similarity") or 0),
                "source_session_ids": [str(source_id)] if source_id else [],
                "metadata": metadata,
                "rank": rank,
            })
        return {"evidence": evidence}

    def close(self) -> None:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mem0", "graphiti", "supermemory"), required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--llm-model", default="qwen3.7-plus-2026-05-26")
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    parser.add_argument("--embedding-dims", type=int, default=1024)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    worker_class = {
        "mem0": Mem0Worker,
        "graphiti": GraphitiWorker,
        "supermemory": SupermemoryWorker,
    }[args.provider]

    # Provider libraries occasionally print notices to stdout.  Redirect those
    # notices so stdout remains a valid NDJSON response channel.
    with contextlib.redirect_stdout(sys.stderr):
        worker = worker_class(args)
    try:
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            request_id = None
            try:
                command = json.loads(raw_line)
                request_id = command.get("request_id")
                operation = str(command["operation"])
                if operation == "close":
                    _json_response({"request_id": request_id, "ok": True, "result": {}})
                    break
                with contextlib.redirect_stdout(sys.stderr):
                    if operation == "info":
                        result = worker.info()
                    elif operation == "add":
                        result = worker.add(command)
                    elif operation == "finalize_scope":
                        result = worker.finalize_scope(command)
                    elif operation == "search":
                        result = worker.search(command)
                    else:
                        raise ValueError(f"未知 operation: {operation}")
                _json_response({"request_id": request_id, "ok": True, "result": result})
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "Provider operation failed: %s", command.get("operation") if 'command' in locals() else "unknown"
                )
                _json_response({
                    "request_id": request_id,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
