"""摘要核验专用内存索引：句子/表格单元、BM25与向量RRF融合、精排。"""
from __future__ import annotations

from collections import Counter
from array import array
import math
import os
import re
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from service.summary_sentence_evidence import sentence_catalog


class TemporaryRetrievalSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    embedding_model: str = 'text-embedding-v4'
    embedding_dimensions: int = Field(default=1024, ge=64, le=2048)
    rerank_model: str = 'qwen3-rerank'
    chunk_tokens: int = Field(default=1024, ge=64, le=2048)
    dense_top_k: int = Field(default=40, ge=1, le=500)
    lexical_top_k: int = Field(default=40, ge=1, le=500)
    coarse_top_k: int = Field(default=60, ge=1, le=500)
    rerank_top_k: int = Field(default=12, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    neighbor_sentences: int = Field(default=1, ge=0, le=2)
    max_chunks: int = Field(default=20000, ge=1)
    batch_size: int = Field(default=10, ge=1, le=10)
    timeout_seconds: float = Field(default=60.0, gt=0, le=120)

    @model_validator(mode='after')
    def validate_counts(self):
        if not self.rerank_top_k <= self.coarse_top_k <= self.dense_top_k + self.lexical_top_k:
            raise ValueError('精排数量须不大于融合候选数量，融合数量须不大于两路召回总量')
        return self


def tokenize(text):
    """中文单字及相邻双字、英文词和数字；不依赖下载分词词典。"""
    result = []
    for part in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*', text.casefold()):
        if re.fullmatch(r'[\u4e00-\u9fff]+', part):
            result.extend(part)
            result.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            result.append(part)
    return result


def unit_key(unit):
    return unit['source_id'], unit['sentence_id']


class RetrievalAPI:
    """可取消的异步API；不复用返回无原始index的旧rerank_similarity。"""
    def __init__(self, settings):
        self.settings = settings
        self.key = os.getenv('SUMMARY_RETRIEVAL_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
        base = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1').rstrip('/')
        self.embedding_url = os.getenv('SUMMARY_RETRIEVAL_EMBEDDING_URL', base + '/embeddings')
        parts = urlsplit(base)
        native_url = urlunsplit((parts.scheme, parts.netloc,
                                '/api/v1/services/rerank/text-rerank/text-rerank', '', ''))
        self.rerank_url = os.getenv('SUMMARY_RETRIEVAL_RERANK_URL') or native_url
        self.rerank_protocol = 'compatible' if urlsplit(self.rerank_url).path.rstrip('/').endswith('/reranks') else 'dashscope'
        self.client = None

    async def request(self, url, payload):
        if not self.key:
            raise ValueError('临时检索缺少SUMMARY_RETRIEVAL_API_KEY或DASHSCOPE_API_KEY')
        if self.client is None:
            import httpx
            self.client = httpx.AsyncClient(timeout=self.settings.timeout_seconds)
        response = await self.client.post(url, headers={'Authorization': 'Bearer ' + self.key}, json=payload)
        if response.is_error:
            # 不输出服务端正文或带鉴权的URL，防止错误日志包含凭据或输入全文。
            raise RuntimeError(f'临时检索API失败：HTTP {response.status_code}；请检查区域、模型和端点配置')
        return response.json()

    async def embed(self, texts):
        vectors = []
        for start in range(0, len(texts), self.settings.batch_size):
            batch = texts[start:start + self.settings.batch_size]
            result = await self.request(self.embedding_url, {
                'model': self.settings.embedding_model, 'input': batch,
                'dimensions': self.settings.embedding_dimensions, 'encoding_format': 'float',
            })
            rows = result.get('data', [])
            if sorted(r.get('index', -1) for r in rows) != list(range(len(batch))):
                raise ValueError('向量API返回数量或原始index不一致')
            vectors.extend(r['embedding'] for r in sorted(rows, key=lambda r: r['index']))
        return vectors

    async def rerank(self, query, documents, top_n, *, instruct=None):
        instruction = instruct or 'Retrieve evidence relevant to verifying the claim, including support, negation, corrections, and necessary conditions. Do not prefer supporting evidence over contrary evidence.'
        # 项目当前的dashscope旧域名使用原生input/parameters；完整/reranks端点使用顶层协议。
        if self.rerank_protocol == 'compatible':
            payload = {'model': self.settings.rerank_model, 'query': query, 'documents': documents,
                       'top_n': top_n, 'instruct': instruction}
        else:
            payload = {'model': self.settings.rerank_model, 'input': {'query': query, 'documents': documents},
                       'parameters': {'top_n': top_n, 'return_documents': False, 'instruct': instruction}}
        result = await self.request(self.rerank_url, payload)
        self.last_rerank_metadata = {'request_id': result.get('request_id'),
                                    'usage': {k: v for k, v in (result.get('usage') or {}).items()
                                              if k in {'total_tokens', 'prompt_tokens'} and type(v) is int and v >= 0}}
        return result.get('results', result.get('output', {}).get('results', []))

    async def close(self):
        client, self.client = self.client, None
        self.key = None
        if client is not None:
            await client.aclose()


class TemporaryEvidenceIndex:
    """一轮摘要共享一个索引；close清空文本、倒排统计及向量，无落盘。"""
    def __init__(self, settings=None, *, backend=None, counter=None):
        self.settings = settings or TemporaryRetrievalSettings()
        self.backend = backend or RetrievalAPI(self.settings)
        self.counter = counter
        self.units = []
        self.vectors = []
        self.terms = []
        self.df = Counter()
        self.average_length = 0.0
        self.closed = False
        self.built = False

    async def build(self, context):
        if self.closed or self.built:
            raise ValueError('临时索引已关闭或已构建，不允许混用上下文快照')
        if self.counter is None:
            from service.qwen_token_counter import get_qwen_token_counter
            self.counter = get_qwen_token_counter()
        fits = lambda text: self.counter.count_text(text) <= self.settings.chunk_tokens
        units = sentence_catalog({'source_ids': list(context['sources'])}, context, fits=fits)
        if len(units) > self.settings.max_chunks:
            raise ValueError('临时索引证据单元超过上限，未静默丢弃')
        if any(not fits(u['text']) for u in units):
            raise ValueError('过长单句超过向量输入预算；需要显式拆分，未截断原文')
        self.units = units
        self.terms = [Counter(tokenize(u['text'])) for u in units]
        for terms in self.terms:
            self.df.update(terms.keys())
        self.average_length = sum(sum(t.values()) for t in self.terms) / max(1, len(units))
        self.vectors = self._normalize(await self.backend.embed([u['text'] for u in units]), len(units)) if units else []
        self.built = True

    def _normalize(self, vectors, expected):
        if len(vectors) != expected:
            raise ValueError('临时检索向量数量不匹配')
        normalized = []
        for vector in vectors:
            if (len(vector) != self.settings.embedding_dimensions
                    or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector)):
                raise ValueError('临时检索向量维度或数值不合法')
            norm = math.hypot(*vector)
            if norm == 0:
                raise ValueError('临时检索收到零向量')
            normalized.append(array('f', (x / norm for x in vector)))
        return normalized

    def lexical_scores(self, query):
        tokens = set(tokenize(query)); count = len(self.units); scores = []
        for terms in self.terms:
            length = sum(terms.values()); score = 0.0
            for token in tokens:
                frequency = terms[token]
                if frequency:
                    idf = math.log(1 + (count - self.df[token] + .5) / (self.df[token] + .5))
                    score += idf * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * length / max(1, self.average_length)))
            scores.append(score)
        return scores

    def expand_neighbors(self, selected):
        seeds = {unit_key(u) for u in selected}; keys = set(seeds); radius = self.settings.neighbor_sentences
        for i, unit in enumerate(self.units):
            if unit_key(unit) not in seeds or unit.get('unit_type') != 'sentence':
                continue
            for direction in (-1, 1):
                previous = unit
                for step in range(1, radius + 1):
                    j = i + direction * step
                    if not 0 <= j < len(self.units): break
                    neighbor = self.units[j]
                    if neighbor['source_id'] != unit['source_id'] or neighbor.get('unit_type') != 'sentence': break
                    left, right = sorted((previous, neighbor), key=lambda u: u['start'])
                    # 只扩充相接句子，绝不跨过未提供的消息区间。
                    if left['end'] > right['start'] or right['start'] - left['end'] > 8: break
                    keys.add(unit_key(neighbor)); previous = neighbor
        return [u for u in self.units if unit_key(u) in keys]

    async def retrieve(self, query):
        if self.closed or not self.built:
            raise ValueError('临时索引尚未构建或已关闭')
        if not self.units:
            return [], {'dense_count': 0, 'lexical_count': 0, 'coarse_count': 0, 'rerank_count': 0}
        if self.counter.count_text(query) > self.settings.chunk_tokens:
            raise ValueError('论断超过临时检索查询预算，未截断')
        vector = self._normalize(await self.backend.embed([query]), 1)[0]
        dense_scores = [sum(a * b for a, b in zip(vector, v)) for v in self.vectors]
        lexical_scores = self.lexical_scores(query)
        dense = sorted(range(len(self.units)), key=lambda i: (-dense_scores[i], i))[:self.settings.dense_top_k]
        lexical = sorted((i for i, s in enumerate(lexical_scores) if s > 0),
                         key=lambda i: (-lexical_scores[i], i))[:self.settings.lexical_top_k]
        fused = {}
        for ranking in (dense, lexical):
            for rank, index in enumerate(ranking, 1):
                fused[index] = fused.get(index, 0.0) + 1 / (self.settings.rrf_k + rank)
        coarse = sorted(fused, key=lambda i: (-fused[i], i))[:self.settings.coarse_top_k]
        top_n = min(self.settings.rerank_top_k, len(coarse))
        ranking = await self.backend.rerank(query, [self.units[i]['text'] for i in coarse], top_n)
        if len(ranking) != top_n:
            raise ValueError('精排返回数量不匹配')
        seen = set()
        for item in ranking:
            index, score = item.get('index'), item.get('relevance_score')
            if (type(index) is not int or not 0 <= index < len(coarse) or index in seen
                    or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('精排原始index重复、越界或分数无效')
            seen.add(index)
        ranking = sorted(ranking, key=lambda r: (-r['relevance_score'], r['index']))
        selected = [self.units[coarse[r['index']]] for r in ranking]
        audit = {'dense_count': len(dense), 'lexical_count': len(lexical), 'coarse_count': len(coarse),
                 'rerank_count': len(selected), 'fusion': 'equal_weight_rrf',
                 'coarse': [{'source_id': self.units[i]['source_id'], 'sentence_id': self.units[i]['sentence_id'],
                             'rrf_score': fused[i]} for i in coarse],
                 'reranked': [{'source_id': u['source_id'], 'sentence_id': u['sentence_id'],
                               'relevance_score': r['relevance_score']} for u, r in zip(selected, ranking)]}
        return self.expand_neighbors(selected), audit

    async def close(self):
        self.closed = True
        self.built = False
        self.units.clear(); self.vectors.clear(); self.terms.clear(); self.df.clear()
        self.average_length = 0.0
        self.counter = None
        await self.backend.close()
