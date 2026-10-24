# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""Elasticsearch 全文检索服务

职责：为知识库切片提供**关键词**检索能力，与 Milvus 的**语义**检索互补。

为什么需要它：向量检索擅长语义相近（问"怎么缓解拥堵"能召回"信号配时优化"），
但对精确字符串很弱——政策文号「交办科技函〔2025〕661号」、设备型号 XN-300、
公司全称这类查询，嵌入向量表达不出差异，经常召不回。BM25 全文检索正好补上这块。

分词器选用内置的 cjk：它把中文切成二元组（bigram），是 ES 核心自带、
无需安装插件的中文可用方案。若后续要更好的中文分词，可在容器里装 IK 插件
并把 analyzer 换成 ik_max_word。

容错策略：ES 不可用时所有方法记日志并返回空/False，绝不抛异常打断主流程——
全文检索是对向量检索的增强，挂了应当降级而不是让整个问答失败。
"""

import logging
import os
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", "1200"))  # docker-compose 把容器 9200 映射到宿主机 1200

_client: Optional[Elasticsearch] = None


def get_es_client() -> Optional[Elasticsearch]:
    """获取 ES 客户端单例；连不上返回 None"""
    global _client
    if _client is not None:
        return _client

    try:
        client = Elasticsearch(
            f"http://{ES_HOST}:{ES_PORT}",
            request_timeout=10,
            max_retries=2,
            retry_on_timeout=True,
        )
        if not client.ping():
            logger.warning(f"Elasticsearch 无法连接: {ES_HOST}:{ES_PORT}，全文检索将不可用")
            return None
        _client = client
        logger.info(f"已连接到 Elasticsearch: {ES_HOST}:{ES_PORT}")
        return _client
    except Exception as e:
        logger.warning(f"Elasticsearch 连接失败: {e}，全文检索将不可用")
        return None


def ensure_index(index_name: str) -> bool:
    """确保索引存在（不存在则按中文友好的映射创建）"""
    client = get_es_client()
    if client is None:
        return False

    try:
        if client.indices.exists(index=index_name):
            return True

        client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "kb_id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    # content 用 cjk 分词做全文匹配，
                    # 子字段 keyword 保留原串，便于将来做精确过滤或聚合
                    "content": {
                        "type": "text",
                        "analyzer": "cjk",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                }
            },
            settings={"number_of_shards": 1, "number_of_replicas": 0},
        )
        logger.info(f"已创建 ES 索引: {index_name}")
        return True
    except Exception as e:
        logger.error(f"创建 ES 索引 {index_name} 失败: {e}")
        return False


def insert_documents(index_name: str, documents: List[Dict[str, Any]]) -> int:
    """批量写入切片

    Args:
        index_name: 索引名（与 Milvus 集合名保持一致，见 kb_collection_name）
        documents: 切片列表，字段与写 Milvus 时相同（vector 字段会被忽略）

    Returns:
        成功写入条数；失败返回 0
    """
    if not documents:
        return 0

    client = get_es_client()
    if client is None or not ensure_index(index_name):
        return 0

    try:
        operations: List[Dict[str, Any]] = []
        for doc in documents:
            operations.append({"index": {"_index": index_name, "_id": doc["id"]}})
            operations.append({
                "doc_id": doc.get("doc_id", ""),
                "kb_id": doc.get("kb_id", ""),
                "filename": doc.get("filename", ""),
                "chunk_index": doc.get("chunk_index", 0),
                "content": doc.get("content", ""),
            })

        resp = client.bulk(operations=operations, refresh=True)
        if resp.get("errors"):
            failed = [
                item["index"].get("error")
                for item in resp.get("items", [])
                if item.get("index", {}).get("error")
            ]
            logger.error(f"ES 批量写入部分失败({len(failed)} 条): {failed[:2]}")

        succeeded = sum(
            1 for item in resp.get("items", []) if not item.get("index", {}).get("error")
        )
        logger.info(f"ES 索引 {index_name} 写入 {succeeded}/{len(documents)} 条")
        return succeeded
    except Exception as e:
        logger.error(f"ES 写入失败: {e}")
        return 0


def search(index_name: str, question: str, top_k: int = 5, *, offset: int = 0) -> List[Dict[str, Any]]:
    """BM25 全文检索

    Returns:
        结果列表，字段与 Milvus 检索结果对齐：
        doc_id / filename / content / score
    """
    client = get_es_client()
    if client is None:
        return []

    try:
        if not client.indices.exists(index=index_name):
            logger.warning(f"ES 索引 {index_name} 不存在")
            return []

        resp = client.search(
            index=index_name,
            query={"match": {"content": {"query": question}}},
            size=top_k,
            from_=offset,
            sort=[{"_score": "desc"}, {"doc_id": "asc"}, {"chunk_index": "asc"}],
        )

        results = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            results.append({
                "id": hit.get("_id"),
                "doc_id": src.get("doc_id", ""),
                "filename": src.get("filename", ""),
                "content": src.get("content", ""),
                "chunk_index": src.get("chunk_index", 0),
                "score": hit.get("_score", 0.0),
            })
        return results
    except Exception as e:
        logger.error(f"ES 检索失败: {e}")
        return []


def delete_by_doc_id(index_name: str, doc_id: str) -> int:
    """删除某个文档的全部切片（与 milvus_service.delete_by_doc_id 对应）

    Args:
        index_name: 索引名
        doc_id: 文档 ID，取值规则见 docmind_service —— md5(文件名)

    Returns:
        实际删除条数；索引不存在或失败返回 0
    """
    client = get_es_client()
    if client is None:
        return 0

    try:
        if not client.indices.exists(index=index_name):
            return 0

        resp = client.delete_by_query(
            index=index_name,
            query={"term": {"doc_id": doc_id}},
            refresh=True,
        )
        deleted = resp.get("deleted", 0)
        logger.info(f"ES 索引 {index_name} 删除文档 {doc_id} 的 {deleted} 个切片")
        return deleted
    except Exception as e:
        logger.error(f"ES 删除文档 {doc_id} 失败: {e}")
        return 0


def delete_index(index_name: str) -> bool:
    """删除索引（知识库删除时调用）"""
    client = get_es_client()
    if client is None:
        return False
    try:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
            logger.info(f"已删除 ES 索引: {index_name}")
        return True
    except Exception as e:
        logger.error(f"删除 ES 索引 {index_name} 失败: {e}")
        return False
