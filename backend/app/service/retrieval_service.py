# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
知识库检索服务 - 基于 Milvus

功能：
1. retrieve_content - 从指定集合检索内容
2. retrieve_from_knowledge_base - 从知识库检索内容
"""

from typing import List, Dict, Any, Optional
import hashlib
import json
from service.milvus_service import get_milvus_service
from service.embedding_service import generate_embedding
import logging

logger = logging.getLogger(__name__)


def knowledge_retrieval_settings():
    """普通检索与会话证据共用分页、RRF、精排及上下文预算配置。"""
    from service.session_evidence_rag import SessionEvidenceSettings
    return SessionEvidenceSettings.from_env()


def kb_collection_name(kb_id: str, *, allow_benchmark: bool = True) -> str:
    """由知识库 ID 生成 Milvus 集合名

    必须用 ID 而不是知识库名称，原因有二：
    1. Milvus 集合名只允许字母、数字、下划线。中文名会直接被服务端拒绝
       （Invalid collection name），而中文产品里知识库几乎都是中文名。
    2. 用 ID 后，知识库改名不会导致已有向量失联。

    写入侧（knowledge_router 上传文档）和读取侧（聊天检索）必须共用本函数，
    否则命名规则一改就会两边失步。

    评测适配：激活了 benchmark profile 时，这里会返回 profile 指定的集合名，
    从而把整条检索链路（Milvus + ES 都用这个名字）指向 benchmark 语料。
    未激活时行为完全不变。
    """
    default_name = f"kb_{str(kb_id).replace('-', '_')}"
    # 用户接口与后台写入固定使用私有集合，评测配置不能把它们重定向到共享语料。
    if not allow_benchmark:
        return default_name
    try:
        from service.eval.profile import resolve_kb_collection
        return resolve_kb_collection(default_name)
    except ImportError:
        return default_name


def retrieve_content(
    indexNames: str,
    question: str,
    top_k: int = 5,
    kb_id: Optional[str] = None,
    *,
    offset: int = 0,
    query_vector: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    检索相关内容

    Args:
        indexNames: 集合名称（知识库索引）
        question: 查询问题
        top_k: 返回结果数量
        kb_id: 知识库ID（可选过滤）

    Returns:
        检索结果列表
    """
    try:
        # 1. 生成查询向量
        if query_vector is None:
            query_vectors = generate_embedding([question])
            if not query_vectors:
                logger.error("生成查询向量失败")
                return []
            query_vector = query_vectors[0]

        # 2. 执行向量搜索
        milvus = get_milvus_service()
        results = milvus.search(
            collection_name=indexNames,
            query_vector=query_vector,
            top_k=top_k,
            kb_id=kb_id,
            offset=offset,
        )

        # 3. 格式化结果
        extracted_data = []
        for i, result in enumerate(results, start=1):
            message = {
                "id": i,
                "chunk_id": result.get("id"),
                "chunk_index": result.get("chunk_index"),
                "document_id": result.get("doc_id", "N/A"),
                "document_name": result.get("filename", "N/A"),
                "content_with_weight": result.get("content", ""),
                "score": result.get("score", 0),
            }
            extracted_data.append(message)

        return extracted_data

    except Exception as e:
        logger.error(f"检索错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def retrieve_from_knowledge_base(
    kb_id: str,
    question: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    从指定知识库检索内容（纯向量语义检索）

    Args:
        kb_id: 知识库 ID（不是名称，见 kb_collection_name 的说明）
        question: 查询问题
        top_k: 返回结果数量

    Returns:
        检索结果列表
    """
    return retrieve_content(kb_collection_name(kb_id), question, top_k)


# RRF 的平滑常数。业界惯例取 60：数值越大，各路排名靠前项之间的分差越平缓，
# 越不容易被某一路的头名结果一家独大。
RRF_K = 60


def knowledge_chunk_key(item):
    """按稳定切片标识归并，避免相同前缀吞掉不同的正文。"""
    if item.get("chunk_id"):
        return ("chunk", str(item["chunk_id"]))
    if item.get("document_id") and item.get("chunk_index") is not None:
        return ("position", str(item["document_id"]), item["chunk_index"])
    content = item.get("content_with_weight") or item.get("content") or ""
    if not content:
        return None
    return ("content", str(item.get("document_id", "")), hashlib.sha256(content.encode()).hexdigest())


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    top_k: int = 5,
    key_fn=None,
    *,
    rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """用 RRF（Reciprocal Rank Fusion）融合多路检索结果

    RRF 只看每条结果在各路中的**排名**，不看原始分数：

        score(doc) = Σ  1 / (rrf_k + rank_in_list)

    之所以不直接加权求和原始分数，是因为两路分数不可比——Milvus 的 COSINE
    相似度与 ES 的 BM25 分数不在同一尺度。
    强行归一化需要按语料调参且不稳定；RRF 用排名规避了这个问题，是多路召回
    融合的常用默认方案。

    Args:
        result_lists: 多路检索结果，每路已按各自相关度降序排列
        top_k: 融合后返回条数
        key_fn: 从结果项提取稳定切片标识的函数

    Returns:
        融合排序后的结果列表，每项附带 rrf_score 字段
    """
    if key_fn is None:
        key_fn = knowledge_chunk_key

    scores: Dict[Any, float] = {}
    items: Dict[Any, Dict[str, Any]] = {}

    for results in result_lists:
        seen = set()
        for rank, item in enumerate(results, start=1):
            key = key_fn(item)
            if not key or key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            # 同一条内容被多路召回时保留首次出现的那份元数据
            items.setdefault(key, item)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))

    fused = []
    for key, score in ranked[:top_k]:
        item = dict(items[key])
        item["rrf_score"] = score
        fused.append(item)
    return fused


def hybrid_retrieve(
    kb_id: str,
    question: str,
    top_k: Optional[int] = None,
    *,
    allow_benchmark: bool = True,
    settings=None,
    audit=None,
) -> List[Dict[str, Any]]:
    """混合检索：向量语义 + BM25 全文，RRF 融合

    两路互补：
      - Milvus 向量检索擅长语义相近（"缓解拥堵" → "信号配时优化"）
      - ES BM25 擅长精确串（文号、型号、公司全称）

    任一路失败都不影响另一路——两者都做了容错，最差退化为单路检索。

    Args:
        kb_id: 知识库 ID
        question: 查询问题
        top_k: 可选的粗排返回上限；默认保留融合后前3页，交给后续精排
        settings: 与会话证据RAG共用的配置，默认每页128条、最多3页

    Returns:
        融合后的结果列表
    """
    collection = kb_collection_name(kb_id, allow_benchmark=allow_benchmark)
    settings = settings or knowledge_retrieval_settings()
    page_size, pages = settings.coarse_page_size, settings.coarse_pages
    candidate_limit = page_size * pages
    vector_hits, text_hits = [], []
    dense_pages, lexical_pages = [], []
    try:
        vectors = generate_embedding([question])
        if vectors:
            for page in range(pages):
                hits = retrieve_content(collection, question, top_k=page_size,
                    offset=page * page_size, query_vector=vectors[0])
                vector_hits.extend(hits)
                dense_pages.append({"page": page + 1, "offset": page * page_size, "count": len(hits)})
                if len(hits) < page_size:
                    break
    except Exception as e:
        logger.warning("知识库向量召回降级: %s", type(e).__name__)

    try:
        from service import es_service
        for page in range(pages):
            hits = es_service.search(collection, question, top_k=page_size, offset=page * page_size)
            text_hits.extend(hits)
            lexical_pages.append({"page": page + 1, "offset": page * page_size, "count": len(hits)})
            if len(hits) < page_size:
                break
    except Exception as e:
        logger.warning("知识库关键词召回降级: %s", type(e).__name__)

    # ES 结果字段名对齐 retrieve_content 的输出，便于下游统一处理
    normalized_text_hits = [
        {
            "id": i,
            "chunk_id": h.get("id"),
            "chunk_index": h.get("chunk_index"),
            "document_id": h.get("doc_id", "N/A"),
            "document_name": h.get("filename", "N/A"),
            "content_with_weight": h.get("content", ""),
            "score": h.get("score", 0),
        }
        for i, h in enumerate(text_hits, start=1)
    ]

    fused = reciprocal_rank_fusion(
        [vector_hits, normalized_text_hits],
        top_k=candidate_limit,
        rrf_k=settings.rrf_k,
    )
    if top_k is not None:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("粗排返回数量必须是正整数")
        fused = fused[:top_k]
    if audit is not None:
        audit.update(coarse_page_size=page_size, coarse_pages_requested=pages,
                     coarse_candidate_limit=candidate_limit, dense_pages=dense_pages,
                     lexical_pages=lexical_pages, dense_count=len(vector_hits),
                     lexical_count=len(text_hits), coarse_count=len(fused), rrf_k=settings.rrf_k,
                     coarse_pages=[{"page": i // page_size + 1, "count": len(fused[i:i + page_size])}
                                   for i in range(0, len(fused), page_size)])

    logger.info(
        f"混合检索 {collection}: 向量 {len(vector_hits)} 条 + 全文 {len(text_hits)} 条 "
        f"→ 融合后 {len(fused)} 条"
    )
    return fused


async def rerank_knowledge_candidates(question, documents, *, settings=None, counter=None, api=None, audit=None):
    """复用会话RAG精排组批；保留原始编号映射，并按上下文预算返回视图。"""
    if not documents:
        return []
    from service.session_evidence_ranking import rerank_candidates, RerankBudgetError
    from service.summary_temporary_retrieval import RetrievalAPI, TemporaryRetrievalSettings
    from service.qwen_token_counter import get_qwen_token_counter
    settings = settings or knowledge_retrieval_settings()
    counter = counter or get_qwen_token_counter()
    audit = audit if audit is not None else {}
    managed_api = api is None
    api = api or RetrievalAPI(TemporaryRetrievalSettings(rerank_model=settings.rerank_model))
    selected = list(documents[:settings.context_top_k])
    try:
        ranking = await rerank_candidates(api, counter, question,
            [doc.get("content") or doc.get("content_with_weight") or "" for doc in documents],
            min(settings.context_top_k, len(documents)), layer="context", settings=settings, audit=audit)
        selected = []
        for item in ranking:
            doc = dict(documents[item["index"]])
            if "relevance_score" in item:
                doc["rerank_score"] = doc["weight"] = item["relevance_score"]
            selected.append(doc)
    except Exception as exc:
        audit.update(rerank_status="failed", rerank_error_type=type(exc).__name__)
        if isinstance(exc, RerankBudgetError):
            audit["rerank_error_reason"] = str(exc)
        logger.warning("知识库精排失败，使用粗排结果: %s", type(exc).__name__)
    finally:
        if managed_api:
            await api.close()
    result = []
    for doc in selected:
        row = dict(doc, id=len(result) + 1)
        if counter.count_text(json.dumps(result + [row], ensure_ascii=False)) <= settings.context_tokens:
            result.append(row)
    audit.update(selected_count=len(result), budget_tokens=settings.context_tokens,
                 selected_tokens=counter.count_text(json.dumps(result, ensure_ascii=False)))
    return result
