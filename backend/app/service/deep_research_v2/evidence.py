"""原文单份保存、事实关联及按需证据视图；不持久化派生账本。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Dict, Mapping
from urllib.parse import urlsplit, urlunsplit

from service.web_document import resolve_sentences, source_available


PRIMARY_SOURCE_TYPES = {"official", "academic", "report", "filing", "regulator"}
LOW_QUALITY_DOMAINS = {
    "docin.com", "doc88.com", "book118.com", "360doc.com", "360doc.cn",
    "toutiao.com", "m.toutiao.com", "csdn.net", "download.csdn.net",
    "zhihu.com", "sohu.com", "jianhu.com", "baijiahao.baidu.com",
}
TIER_RANK = {"low": 0, "secondary": 1, "primary": 2}


def normalize_url(value: Any) -> str:
    try:
        parts = urlsplit(str(value or "").strip())
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return ""
        host = parts.hostname.lower()
        if parts.port:
            host += f":{parts.port}"
        return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/") or "/", parts.query, ""))
    except ValueError:
        return ""


def source_key(value: Any) -> str:
    raw = str(value or "").strip()
    return raw if raw.startswith("local://") else normalize_url(raw)


def source_tier(source_type: Any, url: Any) -> str:
    kind = str(source_type or "").strip().lower()
    host = (urlsplit(str(url or "")).hostname or "").lower()
    if any(host == domain or host.endswith(f".{domain}") for domain in LOW_QUALITY_DOMAINS):
        return "low"
    if kind in PRIMARY_SOURCE_TYPES or host.endswith(".gov.cn") or host.endswith(".edu.cn"):
        return "primary"
    return "secondary"


def _records(state: Mapping[str, Any], field: str):
    return [item for item in state.get(field, []) or [] if isinstance(item, dict)]


def source_indexes(state: Mapping[str, Any]):
    by_id, by_url = {}, {}
    for source in _records(state, "raw_sources"):
        if source.get("id"):
            by_id[str(source["id"])] = source
        key = source_key(source.get("url") or source.get("source_url"))
        if key:
            by_url[key] = source
    return by_id, by_url


def resolve_source(record: Mapping[str, Any], indexes):
    """显式 source_id 不存在或与 URL 冲突时拒绝回退，防止错绑。"""
    by_id, by_url = indexes
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    source_id = record.get("source_id") or metadata.get("source_id")
    key = source_key(record.get("source_url") or record.get("source"))
    if source_id:
        source = by_id.get(str(source_id))
        if source and key and key != source_key(source.get("url") or source.get("source_url")):
            return None
        return source
    return by_url.get(key)


def _strip_snapshot(record):
    for key in list(record):
        if key == "snapshot" or key.startswith("snapshot_"):
            record.pop(key, None)


def normalize_evidence_state(state: Dict[str, Any]) -> None:
    """原地迁移旧检查点；不从旧账本摘录或模型事实伪造原文。"""
    sources = []
    aliases = {}
    for raw in _records(state, "raw_sources"):
        source = dict(raw)
        url = str(source.get("url") or source.get("source_url") or "").strip()
        content = str(source.get("content") or source.get("full_content") or "")
        ident = str(source.get("id") or source.get("snapshot_id") or (
            "source_" + hashlib.sha256((url + "\n" + content).encode()).hexdigest()[:16]
        ))
        if source.get("snapshot_id"):
            aliases[str(source["snapshot_id"])] = ident
        source.update(id=ident, url=url)
        if source.get("evidence_document_ref"):
            # 来源引用沿用数据库正文标识，不能把缺省空串写回内容哈希。
            source.pop("content", None)
            source.pop("sentence_index", None)
        else:
            source["content"] = content
            source["content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        source.setdefault("fetched_at", "")
        source.pop("source_url", None)
        source.pop("full_content", None)
        _strip_snapshot(source)
        sources.append(source)
    state["raw_sources"] = sources
    indexes = source_indexes(state)
    for fact in _records(state, "facts"):
        # 历史版本可能只在 metadata 中保存来源引用。
        metadata = dict(fact.get("metadata") or {})
        old_id = fact.get("source_id") or metadata.get("source_id") or fact.get("snapshot_id")
        if old_id:
            fact["source_id"] = aliases.get(str(old_id), str(old_id))
        source = resolve_source(fact, indexes)
        if source is not None:
            fact["source_id"] = source["id"]
            if fact.get("source_name") and not source.get("title"):
                source["title"] = fact["source_name"]
            if fact.get("source_type") and not source.get("source_type"):
                source["source_type"] = fact["source_type"]
        elif old_id and source_key(fact.get("source_url")):
            # 丢弃冲突引用，后续绑定门禁会移除此事实。
            fact["source_id"] = ""
        for key in ("source_url", "source_name", "source_type", "source_tier",
                    "verified", "related_hypothesis", "hypothesis_support"):
            fact.pop(key, None)
        for key in ("source_id", "extractor", "evidence_excerpt"):
            metadata.pop(key, None)
        _strip_snapshot(metadata)
        _strip_snapshot(fact)
        fact["metadata"] = metadata

    def clean_relation(raw):
        relation = dict(raw)
        _strip_snapshot(relation)
        for key in ("source_id", "source_url", "source_title", "content_sha256", "fetched_at"):
            relation.pop(key, None)
        return relation

    relations = [clean_relation(raw) for raw in _records(state, "hypothesis_evidence")]
    if "hypothesis_evidence" in state:
        state["hypothesis_evidence"] = relations
    by_relation_id = {str(item.get("id")): item for item in relations}
    for hypothesis in _records(state, "hypotheses"):
        for bucket in ("evidence_for", "evidence_against", "evidence_inconclusive"):
            if bucket in hypothesis:
                hypothesis[bucket] = [
                    by_relation_id.get(str(item.get("id")), clean_relation(item))
                    if isinstance(item, dict) else item
                    for item in hypothesis[bucket] or []
                ]
    for index, point in enumerate(_records(state, "data_points")):
        if not point.get("id"):
            payload = json.dumps(point, sort_keys=True, ensure_ascii=False, default=str)
            point["id"] = "dp_" + hashlib.sha256(f"{index}|{payload}".encode()).hexdigest()[:16]
        if point.get("source") == "deterministic_calculation":
            point["origin"] = "derived"
            point.setdefault("input_point_ids", [])
            point.setdefault("calculation_status", "unknown")
    state.pop("evidence_ledger", None)


def fact_views(state: Mapping[str, Any], *, verified_only: bool = False):
    """关联原文供提示词/UI读取；返回新字典，不把来源信息复制回事实。"""
    indexes = source_indexes(state)
    views = []
    for fact in _records(state, "facts"):
        source = resolve_source(fact, indexes)
        verified = bool(source and source_key(source.get("url") or source.get("source_url"))
                        and source_available(source))
        if verified and source.get("document_version"):
            if source.get("evidence_document_ref"):
                verified = (fact.get("document_id") == source.get("document_id")
                            and bool(fact.get("sentence_ids") or fact.get("evidence_unit_ids")))
            else:
                verified = bool(resolve_sentences(source, fact.get("document_id"), fact.get("sentence_ids")))
        if verified_only and not verified:
            continue
        source = source or {}
        url = source.get("url") or source.get("source_url") or fact.get("source_url", "")
        kind = source.get("source_type") or fact.get("source_type", "")
        views.append({
            **fact,
            "source_id": source.get("id", ""),
            "source_url": url,
            "source_name": source.get("title") or fact.get("source_name", ""),
            "source_type": kind,
            "source_tier": source_tier(kind, url) if verified else "low",
            "reader_verified": verified,
        })
    return views


def build_evidence_view(state: Mapping[str, Any]):
    """按需组装通过来源门禁的事实、正文及假设关系。"""
    indexes = source_indexes(state)
    result = []
    for fact in fact_views(state, verified_only=True):
        source = resolve_source(fact, indexes) or {}
        sentences = resolve_sentences(source, fact.get("document_id"), fact.get("sentence_ids"))
        content = str(source.get("content") or "")
        # 新事实只消费已定位原句；历史短正文保持可读，长正文无绑定不猜片段。
        excerpt = (sentences[0]["text"] if sentences
                   else content if not source.get("document_version") and len(content) <= 1600 else "")
        result.append({
            **fact, "evidence_id": fact.get("id"), "claim_id": fact.get("id"),
            "claim": fact.get("content", ""), "source_title": fact["source_name"],
            "excerpt": excerpt,
            "document_id": source.get("document_id", ""),
            "original_sentences": sentences,
            "hypothesis_evidence_ids": [item.get("id") for item in _records(state, "hypothesis_evidence")
                                        if item.get("fact_id") == fact.get("id")],
        })
    return result


def hypothesis_evidence_views(state: Mapping[str, Any]):
    facts = {str(fact.get("id")): fact for fact in fact_views(state, verified_only=True)}
    return [{**item, "source_id": facts.get(str(item.get("fact_id")), {}).get("source_id", ""),
             "source_url": facts.get(str(item.get("fact_id")), {}).get("source_url", "")}
            for item in _records(state, "hypothesis_evidence")]


def is_derived(point: Mapping[str, Any]) -> bool:
    return point.get("origin") == "derived" or point.get("source") == "deterministic_calculation"


def data_point_views(state: Mapping[str, Any]):
    """递归追溯派生数据输入；缺失、重复 ID、循环或未验证来源均不能升级。"""
    points = _records(state, "data_points")
    counts = Counter(str(p.get("id")) for p in points if p.get("id"))
    by_id = {str(p["id"]): p for p in points if p.get("id") and counts[str(p["id"])] == 1}
    indexes = source_indexes(state)
    # 兼容迁移前来源类型仍存于 facts 的状态。
    verified_facts = fact_views(state, verified_only=True)
    tiers = {source_key(f["source_url"]): f["source_tier"] for f in verified_facts}
    fact_sources = {str(f["id"]): f["source_id"] for f in verified_facts}
    cache = {}

    def resolve(point, visiting):
        ident = str(point.get("id") or "")
        if ident and ident in visiting:
            return {**point, "reader_verified": False, "source_tier": "low"}
        if ident in cache:
            return cache[ident]
        view = dict(point)
        if is_derived(point):
            inputs = point.get("input_point_ids")
            input_ids = [str(i) for i in inputs] if isinstance(inputs, list) else []
            resolved = [resolve(by_id[i], visiting | {ident}) for i in input_ids if i in by_id]
            valid = bool(point.get("calculation_status") == "passed" and point.get("formula")
                         and input_ids and len(resolved) == len(input_ids)
                         and all(p.get("reader_verified") for p in resolved))
            view.update(origin="derived", reader_verified=valid,
                        source_tier=min((p["source_tier"] for p in resolved), key=TIER_RANK.get) if valid else "low")
        else:
            source = resolve_source(point, indexes)
            valid = bool(source and source_key(source.get("url") or source.get("source_url"))
                         and source_available(source))
            if valid and source.get("document_version"):
                valid = fact_sources.get(str(point.get("fact_id") or "")) == source.get("id")
            url = (source or {}).get("url") or (source or {}).get("source_url")
            tier = source_tier((source or {}).get("source_type"), url)
            view.update(reader_verified=valid,
                        source_tier=tiers.get(source_key(url), tier) if valid else "low")
            if valid:
                view["source_id"] = source.get("id")
        if ident and counts[ident] == 1:
            cache[ident] = view
        return view

    return [resolve(point, set()) for point in points]
