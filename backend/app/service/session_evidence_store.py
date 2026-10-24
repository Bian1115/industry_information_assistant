"""临时正文的权威存储；所有读取、建索引提交均验证会话及关闭generation。"""
from copy import deepcopy
from datetime import timedelta
import uuid

from sqlalchemy import select
from core.database import SessionLocal
from core.time_utils import utc_now, as_utc
from models.chat import ChatSession
from models.session_evidence import SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk
from service.session_evidence_chunks import stable_id, chunk_document
from service.web_document import source_usable, sentence_records


class EvidenceExpiredError(ValueError):
    pass


def document_reference(source):
    """正文只存证据库，研究状态仅携带来源元数据。"""
    return {**{key: deepcopy(value) for key, value in source.items()
               if key not in {"content", "full_content", "sentence_index"}},
            "evidence_document_ref": True, "rag_usable": not bool(source.get("rag_chunk_error"))}


class SessionEvidenceStore:
    def __init__(self, session_factory=SessionLocal, *, now=utc_now, ttl_seconds=86400, ttl_mode="idle"):
        self.session_factory = session_factory
        self.now = now
        self.ttl_seconds = ttl_seconds
        self.ttl_mode = ttl_mode

    def _space(self, db, user_id, session_id, *, create=False, generation=None):
        uid, sid = uuid.UUID(str(user_id)), uuid.UUID(str(session_id))
        now = self.now()
        chat = db.execute(select(ChatSession).where(ChatSession.id == sid,
                          ChatSession.user_id == uid).with_for_update()).scalar_one_or_none()
        if chat is None or as_utc(chat.expires_at) <= now:
            raise EvidenceExpiredError("会话不存在或已过期，不能读取临时证据")
        space = db.get(SessionEvidenceSpace, sid)
        if space is None:
            if not create:
                return None
            expires = now + timedelta(seconds=self.ttl_seconds)
            if self.ttl_mode == "idle":
                expires = as_utc(chat.expires_at) - timedelta(hours=24) + timedelta(seconds=self.ttl_seconds)
            space = SessionEvidenceSpace(session_id=sid, user_id=uid, status="active", generation=1,
                                         revision=0, ttl_mode=self.ttl_mode, ttl_seconds=self.ttl_seconds, expires_at=expires)
            db.add(space); db.flush()
        if space.user_id != uid or space.status != "active":
            raise EvidenceExpiredError("临时证据已结束或过期，不允许续期复活")
        if generation is not None and space.generation != generation:
            raise EvidenceExpiredError("临时证据generation已变化，拒绝旧任务写入")
        if space.ttl_mode == "idle":
            # 只同步新消息已提交的期限；检索本身不以当前时间续期。
            message_time = as_utc(chat.expires_at) - timedelta(hours=24)
            expires = message_time + timedelta(seconds=space.ttl_seconds)
            if expires > as_utc(space.expires_at) and message_time < as_utc(space.expires_at):
                space.expires_at = expires
        if as_utc(space.expires_at) <= now:
            raise EvidenceExpiredError("临时证据已过期，不允许旧任务复活")
        return space

    def ensure(self, user_id, session_id):
        with self.session_factory() as db:
            space = self._space(db, user_id, session_id, create=True)
            result = {"generation": space.generation, "revision": space.revision,
                      "expires_at": as_utc(space.expires_at).isoformat(), "active_topic_id": space.active_topic_id}
            db.commit()
            return result

    def activate_topic(self, user_id, session_id, topic_id):
        with self.session_factory() as db:
            space = self._space(db, user_id, session_id, create=True)
            space.active_topic_id = topic_id
            db.commit()

    def ingest(self, user_id, session_id, topic_id, sources, counter, chunk_tokens):
        with self.session_factory() as db:
            space = self._space(db, user_id, session_id, create=True)
            added = 0
            added_documents = 0
            updated_documents = 0
            failures = []
            references = []
            for source in sources:
                if not source.get("document_id"):
                    continue
                key = stable_id(session_id, topic_id, source["document_id"])
                if source.get("evidence_document_ref") and not source.get("content"):
                    # 常规节点只核对来源元数据；避免为未变化的引用加载整个payload正文。
                    fields = ("document_id", "content_sha256", "document_version", "rag_chunk_error",
                              "related_sections", "is_supplementary", "critic_issue_id", "task_origin",
                              "search_depth", "search_type")
                    stored = db.execute(select(*[SessionEvidenceDocument.payload[field].label(field)
                        for field in fields]).where(SessionEvidenceDocument.id == key)).mappings().one_or_none()
                    if stored is None:
                        raise EvidenceExpiredError("来源引用对应的会话证据不存在")
                    if any(stored[field] != source.get(field) for field in
                           ("document_id", "content_sha256", "document_version")):
                        raise ValueError("同一正文编号不能覆盖为不同内容")
                    changed = (any(section not in (stored["related_sections"] or [])
                                   for section in source.get("related_sections") or [])
                               or any(field in source and source[field] != stored[field]
                                      for field in fields[5:]))
                    if not changed:
                        metadata = dict(source)
                        if stored["rag_chunk_error"]:
                            metadata["rag_chunk_error"] = stored["rag_chunk_error"]
                        else:
                            metadata.pop("rag_chunk_error", None)
                        reference = document_reference(metadata)
                        references.append(reference)
                        continue
                old = db.get(SessionEvidenceDocument, key)
                if old is not None:
                    if any(old.payload.get(field) != source.get(field) for field in
                           ("document_id", "content_sha256", "document_version")):
                        raise ValueError("同一正文编号不能覆盖为不同内容")
                    # 原文不可变，报告章节关系和补搜来源信息可以增补。
                    payload = deepcopy(old.payload)
                    sections = list(payload.get("related_sections") or [])
                    for section in source.get("related_sections") or []:
                        if section not in sections:
                            sections.append(section)
                    if sections or "related_sections" in payload:
                        payload["related_sections"] = sections
                    for field in ("is_supplementary", "critic_issue_id", "task_origin", "search_depth", "search_type"):
                        if field in source:
                            payload[field] = deepcopy(source[field])
                    if payload != old.payload:
                        old.payload = payload
                        for chunk in db.execute(select(SessionEvidenceChunk).where(
                                SessionEvidenceChunk.document_key == key)).scalars():
                            chunk.locator = {**deepcopy(chunk.locator), "related_sections": sections}
                        updated_documents += 1
                    references.append(document_reference(payload))
                    continue
                if source.get("evidence_document_ref"):
                    raise EvidenceExpiredError("来源引用对应的会话证据不存在")
                if not source.get("content"):
                    continue
                if not sentence_records(source):
                    raise ValueError("临时证据正文的哈希或编号无效")
                payload = deepcopy(source)
                payload["rag_chunk_profile"] = {"version": "session-evidence-v1", "chunk_tokens": chunk_tokens}
                chunks = []
                if source_usable(source):
                    try:
                        chunks = chunk_document(source, user_id=user_id, session_id=session_id,
                                                topic_id=topic_id, counter=counter, chunk_tokens=chunk_tokens)
                    except ValueError as exc:
                        payload["rag_chunk_error"] = {"type": type(exc).__name__, "reason": str(exc)}
                        failures.append({"document_id": source["document_id"], **payload["rag_chunk_error"]})
                else:
                    payload["rag_chunk_error"] = {"type": "source_not_admitted", "reason": "来源准入未通过"}
                    failures.append({"document_id": source["document_id"], **payload["rag_chunk_error"]})
                references.append(document_reference(payload))
                db.add(SessionEvidenceDocument(id=key, session_id=uuid.UUID(str(session_id)), topic_id=topic_id,
                                                document_id=source["document_id"], payload=payload))
                added_documents += 1
                db.flush()
                for row in chunks:
                    db.add(SessionEvidenceChunk(id=row["id"], document_key=key, unit_type=row["unit_type"],
                                                content=row["content"], locator=row["metadata"], index_status="pending"))
                added += len(chunks)
            if added_documents or updated_documents:
                space.revision += 1
            db.flush()
            pending = db.execute(select(SessionEvidenceChunk).join(SessionEvidenceDocument).where(
                SessionEvidenceDocument.session_id == space.session_id,
                SessionEvidenceDocument.topic_id == topic_id,
                SessionEvidenceChunk.index_status != "indexed")).scalars().all()
            result = {"added_chunks": added, "added_documents": added_documents, "updated_documents": updated_documents,
                      "revision": space.revision, "generation": space.generation,
                      "sources": references,
                      "chunk_errors": failures,
                      "pending": [self._row(r, space) for r in pending]}
            db.commit()
            return result

    def _row(self, row, space):
        return {"id": row.id, "unit_type": row.unit_type, "content": row.content,
                "metadata": {**deepcopy(row.locator), "evidence_space_id": str(space.session_id),
                             "evidence_space_generation": space.generation, "corpus_revision": space.revision,
                             "expires_at": as_utc(space.expires_at).isoformat()}, "index_status": row.index_status}

    def mark_indexed(self, user_id, session_id, generation, ids):
        self._mark_index_status(user_id, session_id, generation, ids, "indexed")

    def mark_pending(self, user_id, session_id, generation, ids):
        self._mark_index_status(user_id, session_id, generation, ids, "pending")

    def _mark_index_status(self, user_id, session_id, generation, ids, status):
        with self.session_factory() as db:
            space = self._space(db, user_id, session_id, generation=generation)
            if space is None:
                raise EvidenceExpiredError("证据空间不存在")
            rows = db.execute(select(SessionEvidenceChunk).join(SessionEvidenceDocument).where(
                SessionEvidenceDocument.session_id == space.session_id,
                SessionEvidenceChunk.id.in_(ids))).scalars().all()
            if len(rows) != len(set(ids)):
                raise ValueError("索引提交包含未知或其他会话chunk")
            for row in rows:
                row.index_status = status
            db.commit()

    def reconcile_chunks(self, user_id, session_id, topic_id, generation, document_ids, counter, chunk_tokens):
        """由不可变正文重建派生数据；逐文档比较完整集合，不只检查是否非空。"""
        from service.deep_research_v2.storage_reconciliation import digest
        with self.session_factory() as db:
            space = self._space(db, user_id, session_id, generation=generation)
            if space is None:
                raise EvidenceExpiredError("临时正文空间不存在")
            documents = db.execute(select(SessionEvidenceDocument).where(
                SessionEvidenceDocument.session_id == space.session_id,
                SessionEvidenceDocument.topic_id == topic_id,
                SessionEvidenceDocument.document_id.in_(set(document_ids)))).scalars().all()
            if len(documents) != len(set(document_ids)):
                raise EvidenceExpiredError("恢复索引所需的原始正文缺失")
            repaired, legacy = 0, 0
            for document in documents:
                payload = document.payload
                if not sentence_records(payload):
                    raise EvidenceExpiredError("原始正文哈希或原句编号不合法，拒绝重建索引")
                profile = payload.get("rag_chunk_profile")
                legacy += int(profile is None)
                if profile and profile.get("version") != "session-evidence-v1":
                    raise ValueError("不支持的临时证据切分协议，不能静默更换索引版本")
                if not payload.get("rag_chunk_error") and not source_usable(payload):
                    raise ValueError("正文准入协议已变化，不能静默丢弃既有索引")
                # 当初未准入/无法切分的正文仍可保留审计，不冒充已完成索引。
                expected = chunk_document(payload, user_id=user_id, session_id=session_id, topic_id=topic_id,
                    counter=counter, chunk_tokens=profile["chunk_tokens"] if profile else chunk_tokens
                    ) if source_usable(payload) and not payload.get("rag_chunk_error") else []
                rows = db.execute(select(SessionEvidenceChunk).where(SessionEvidenceChunk.document_key == document.id)).scalars().all()
                actual = {row.id: row for row in rows}
                wanted = {row["id"] for row in expected}
                for row in rows:
                    if row.id not in wanted:
                        db.delete(row)
                        repaired += 1
                for row in expected:
                    old = actual.get(row["id"])
                    if old is None:
                        db.add(SessionEvidenceChunk(id=row["id"], document_key=document.id, unit_type=row["unit_type"],
                            content=row["content"], locator=row["metadata"], index_status="pending"))
                        repaired += 1
                    elif digest([old.unit_type, old.content, old.locator]) != digest([row["unit_type"], row["content"], row["metadata"]]):
                        old.unit_type, old.content, old.locator = row["unit_type"], row["content"], row["metadata"]
                        old.index_status = "pending"
                        repaired += 1
            db.flush()
            rows = db.execute(select(SessionEvidenceChunk).join(SessionEvidenceDocument).where(
                SessionEvidenceDocument.session_id == space.session_id,
                SessionEvidenceDocument.topic_id == topic_id,
                SessionEvidenceDocument.document_id.in_(set(document_ids)))).scalars().all()
            result = {"topic_id": topic_id, "repaired_chunks": repaired, "legacy_chunk_profiles": legacy,
                      "rows": [self._row(row, space) for row in rows]}
            db.commit()
            return result

    def chunks(self, user_id, session_id, topic_id, *, kinds=None, ids=None, limit=20000):
        with self.session_factory() as db:
            space = self._space(db, user_id, session_id)
            if space is None:
                return []
            query = select(SessionEvidenceChunk).join(SessionEvidenceDocument).where(
                SessionEvidenceDocument.session_id == space.session_id, SessionEvidenceDocument.topic_id == topic_id)
            if kinds is not None:
                query = query.where(SessionEvidenceChunk.unit_type.in_(kinds))
            if ids is not None:
                query = query.where(SessionEvidenceChunk.id.in_(ids))
            rows = db.execute(query.order_by(SessionEvidenceChunk.id).limit(limit + 1)).scalars().all()
            if len(rows) > limit:
                raise ValueError("会话chunk数量超过检索上限，需要分批索引")
            result = [self._row(row, space) for row in rows]
            db.commit()
            return result

    def documents(self, user_id, session_id, topic_id, *, document_ids=None):
        with self.session_factory() as db:
            space = self._space(db, user_id, session_id)
            if space is None:
                return []
            query = select(SessionEvidenceDocument).where(SessionEvidenceDocument.session_id == space.session_id,
                                                          SessionEvidenceDocument.topic_id == topic_id)
            if document_ids is not None:
                query = query.where(SessionEvidenceDocument.document_id.in_(document_ids))
            result = [deepcopy(row.payload) for row in db.execute(query).scalars()]
            db.commit()
            return result

    def adjacent(self, user_id, session_id, topic_id, chunk_id, *, layer="context", radius=1):
        if layer not in {"context", "citation"} or radius not in {0, 1, 2}:
            raise ValueError("邻居召回参数无效")
        rows = self.chunks(user_id, session_id, topic_id, ids=[chunk_id])
        if len(rows) != 1:
            raise ValueError("chunk不属于当前会话或主题")
        seed = rows[0]
        result = []
        seen = {chunk_id}
        for direction in ("prev_chunk_id", "next_chunk_id"):
            current = seed
            for _ in range(radius):
                cid = current["metadata"].get("adjacent_chunk_ids", {}).get(layer, {}).get(direction)
                if cid is None:
                    break
                found = self.chunks(user_id, session_id, topic_id, ids=[cid])
                if len(found) != 1 or found[0]["metadata"]["document_id"] != seed["metadata"]["document_id"]:
                    raise ValueError("相邻chunk不属于同一正文版本")
                current = found[0]
                if cid not in seen:
                    result.append(current); seen.add(cid)
        return {"chunk": seed, "neighbors": result}


def prepare_session_evidence_schema():
    """仅补建本功能的四张表；兼容历史环境未完整登记Alembic版本的情况。"""
    from core.database import engine
    from models.session_evidence import SessionEvidencePurge
    with engine.begin() as db:
        for model in (SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk, SessionEvidencePurge):
            model.__table__.create(db, checkfirst=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="初始化会话临时证据表，不修改现有研究数据")
    parser.add_argument("--prepare-schema", action="store_true", required=True)
    parser.parse_args()
    prepare_session_evidence_schema()
