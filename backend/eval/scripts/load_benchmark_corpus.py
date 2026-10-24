#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
把 benchmark 语料灌进 profile 指定的检索集合

配合适配层使用：profile 决定"读哪个集合"，本脚本负责"把语料写进那个集合"。
写入侧和读取侧共用 kb_collection_name()，所以只要 profile 一致，两边必然对齐。

同时写 Milvus（向量）和 ES（BM25），因为系统用的是 RRF 混合检索，
只灌一边会让评测结果偏离真实链路。

支持的输入格式（--format）：
    jsonl   每行一个 JSON 对象，取 --text-field 指定的字段作为正文
    json    一个 JSON 数组，同上
    dir     目录下所有 .txt / .md 文件，每个文件一篇

用法：
    cd backend

    # CRUD-RAG 的新闻语料（80000 条，建议先用 --limit 小规模验证）
    python eval/scripts/load_benchmark_corpus.py \
        --profile crud_rag \
        --input eval/data/crud_rag/80000_docs.json \
        --format json --text-field content --limit 500

    # 查看已灌入情况
    python eval/scripts/load_benchmark_corpus.py --profile crud_rag --stat

    # 重灌前先清空
    python eval/scripts/load_benchmark_corpus.py --profile crud_rag --drop
"""

import os
import sys
import json
import uuid
import argparse
from typing import List, Dict, Any, Iterator

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "app"))

from dotenv import load_dotenv

load_dotenv()

from eval.runtime.profile import get_profile, use_profile  # noqa: E402
from service.retrieval_service import kb_collection_name  # noqa: E402


# 与业务侧上传文档保持一致的切片粒度
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_BATCH = 16


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """按字符切片。中文场景下按字符切比按 token 切更直观，且与业务侧一致。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def iter_documents(path: str, fmt: str, text_field: str, title_field: str) -> Iterator[Dict[str, Any]]:
    """统一成 {"title": ..., "content": ...} 的流。"""
    if fmt == "dir":
        if not os.path.isdir(path):
            raise ValueError(f"--format dir 需要目录，收到: {path}")
        for name in sorted(os.listdir(path)):
            if not name.lower().endswith((".txt", ".md")):
                continue
            with open(os.path.join(path, name), "r", encoding="utf-8") as f:
                yield {"title": name, "content": f.read()}
        return

    if not os.path.isfile(path):
        raise ValueError(f"输入文件不存在: {path}")

    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  ⚠️  第 {i + 1} 行不是合法 JSON，跳过")
                    continue
                yield {
                    "title": str(obj.get(title_field) or f"doc_{i}"),
                    "content": str(obj.get(text_field) or ""),
                }
    elif fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 兼容 {"documents": [...]} 这类外层包装
            for key in ("documents", "docs", "data", "corpus"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError("--format json 期望顶层是数组，或含 documents/docs/data/corpus 数组字段")
        for i, obj in enumerate(data):
            if isinstance(obj, str):
                yield {"title": f"doc_{i}", "content": obj}
            else:
                yield {
                    "title": str(obj.get(title_field) or f"doc_{i}"),
                    "content": str(obj.get(text_field) or ""),
                }
    else:
        raise ValueError(f"未知格式: {fmt}")


def build_chunks(doc: Dict[str, Any], kb_id: str) -> List[Dict[str, Any]]:
    """切片并组装成 Milvus / ES 都能吃的结构。"""
    doc_id = str(uuid.uuid4())
    pieces = chunk_text(doc["content"])
    return [
        {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "kb_id": kb_id,
            "filename": doc["title"][:200],
            "content": piece,
            "chunk_index": idx,
        }
        for idx, piece in enumerate(pieces)
    ]


def load(profile_name: str, input_path: str, fmt: str, text_field: str,
         title_field: str, limit: int, batch: int):
    from service.embedding_service import generate_embedding
    from service.milvus_service import get_milvus_service
    from service import es_service

    profile = get_profile(profile_name)
    if not profile.kb_collection:
        print(f"❌ profile '{profile_name}' 未配置 kb_collection，无法确定写入目标")
        print("   请在 backend/eval/profiles.json 里补上该字段")
        return False

    # kb_id 在 benchmark 场景没有业务含义，用 profile 名占位即可
    kb_id = f"bench_{profile_name}"

    with use_profile(profile):
        collection = kb_collection_name(kb_id)

    print(f"\nprofile   : {profile.name}")
    print(f"目标集合  : {collection}   (Milvus 集合 + ES 索引同名)")
    print(f"输入      : {input_path}  [{fmt}]")
    print(f"正文字段  : {text_field}")
    if limit:
        print(f"限制      : 最多 {limit} 篇")
    print("-" * 60)

    milvus = get_milvus_service()
    pending: List[Dict[str, Any]] = []
    n_docs = n_chunks = 0
    n_milvus = n_es = 0

    def flush():
        nonlocal pending, n_milvus, n_es
        if not pending:
            return
        texts = [c["content"] for c in pending]
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            vectors.extend(generate_embedding(texts[i:i + EMBED_BATCH]) or [])

        if len(vectors) != len(pending):
            print(f"  ⚠️  向量数({len(vectors)}) 与切片数({len(pending)}) 不符，跳过本批")
            pending = []
            return

        for chunk, vec in zip(pending, vectors):
            chunk["vector"] = vec

        try:
            n_milvus += milvus.insert_documents(collection, pending)
        except Exception as e:
            print(f"  ❌ Milvus 写入失败: {e}")
        try:
            n_es += es_service.insert_documents(collection, pending)
        except Exception as e:
            print(f"  ❌ ES 写入失败: {e}")

        print(f"  已写入 {n_milvus} 切片(Milvus) / {n_es} 切片(ES)")
        pending = []

    for doc in iter_documents(input_path, fmt, text_field, title_field):
        if limit and n_docs >= limit:
            break
        chunks = build_chunks(doc, kb_id)
        if not chunks:
            continue
        n_docs += 1
        n_chunks += len(chunks)
        pending.extend(chunks)
        if len(pending) >= batch:
            flush()

    flush()

    print("-" * 60)
    print(f"✅ 完成：{n_docs} 篇文档 → {n_chunks} 个切片")
    print(f"   Milvus {n_milvus} 条 / ES {n_es} 条")
    print(f"\n验证检索：")
    print(f"   python eval/scripts/eval_profile.py --verify {profile_name}")
    return True


def stat(profile_name: str):
    from service.milvus_service import get_milvus_service

    profile = get_profile(profile_name)
    if not profile.kb_collection:
        print(f"❌ profile '{profile_name}' 未配置 kb_collection")
        return
    collection = profile.kb_collection

    print(f"\n集合: {collection}")
    try:
        stats = get_milvus_service().get_collection_stats(collection)
        print(f"  Milvus: {stats}")
    except Exception as e:
        print(f"  Milvus: 读取失败 ({e})")

    try:
        from service.es_service import get_es_client
        client = get_es_client()
        if client and client.indices.exists(index=collection):
            cnt = client.count(index=collection)["count"]
            print(f"  ES    : {cnt} 条")
        else:
            print(f"  ES    : 索引不存在")
    except Exception as e:
        print(f"  ES    : 读取失败 ({e})")
    print()


def drop(profile_name: str):
    from service.milvus_service import get_milvus_service
    from service import es_service

    profile = get_profile(profile_name)
    if not profile.kb_collection:
        print(f"❌ profile '{profile_name}' 未配置 kb_collection")
        return

    collection = profile.kb_collection
    # 防呆：只允许删 benchmark 集合，避免手滑把生产知识库删了
    if not collection.startswith("bench_"):
        print(f"❌ 拒绝删除非 benchmark 集合: {collection}")
        print("   benchmark 集合名必须以 bench_ 开头")
        return

    confirm = input(f"确认清空集合 {collection} ? [y/N] ").strip().lower()
    if confirm != "y":
        print("已取消")
        return

    try:
        get_milvus_service().delete_collection(collection)
        print(f"✅ 已删除 Milvus 集合 {collection}")
    except Exception as e:
        print(f"⚠️  Milvus 删除失败: {e}")
    try:
        es_service.delete_index(collection)
        print(f"✅ 已删除 ES 索引 {collection}")
    except Exception as e:
        print(f"⚠️  ES 删除失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="把 benchmark 语料灌进 profile 指定的集合")
    parser.add_argument("--profile", required=True, help="目标 profile 名")
    parser.add_argument("--input", help="语料文件或目录")
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "json", "dir"])
    parser.add_argument("--text-field", default="content", help="正文字段名，默认 content")
    parser.add_argument("--title-field", default="title", help="标题字段名，默认 title")
    parser.add_argument("--limit", type=int, default=0, help="最多灌入多少篇，0 表示不限")
    parser.add_argument("--batch", type=int, default=64, help="每批写入的切片数")
    parser.add_argument("--stat", action="store_true", help="查看集合现状")
    parser.add_argument("--drop", action="store_true", help="清空集合")
    args = parser.parse_args()

    if args.stat:
        stat(args.profile)
    elif args.drop:
        drop(args.profile)
    elif args.input:
        load(args.profile, args.input, args.format, args.text_field,
             args.title_field, args.limit, args.batch)
    else:
        parser.error("需要 --input，或使用 --stat / --drop")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)
