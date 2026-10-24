"""独立Milvus临时集合；所有操作必须带已验证的用户、会话、主题和generation。"""
import os
import uuid
from service.session_evidence_chunks import stable_id


class SessionEvidenceVectors:
    def __init__(self, dimensions=1024, model="text-embedding-v4"):
        self.dimensions = dimensions
        self.collection_name = "session_evidence_" + stable_id(model, dimensions)[:12]
        self.alias = "session_evidence"

    def _collection(self, *, create=True):
        from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
        if not connections.has_connection(self.alias):
            connections.connect(alias=self.alias, host=os.getenv("MILVUS_HOST", "localhost"),
                                port=os.getenv("MILVUS_PORT", "19530"))
        if not utility.has_collection(self.collection_name, using=self.alias):
            if not create:
                return None
            fields = [FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=64,
                                   is_primary=name == "id") for name in
                      ("id", "user_id", "session_id", "topic_id", "document_id", "unit_type")]
            fields += [FieldSchema(name="generation", dtype=DataType.INT64),
                       FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimensions)]
            collection = Collection(self.collection_name, CollectionSchema(fields), using=self.alias)
            collection.create_index("vector", {"index_type": "FLAT", "metric_type": "COSINE", "params": {}})
        collection = Collection(self.collection_name, using=self.alias)
        collection.load()
        return collection

    def _scope(self, user_id, session_id, topic_id=None, generation=None):
        uid, sid = str(uuid.UUID(str(user_id))), str(uuid.UUID(str(session_id)))
        expr = f'user_id == "{uid}" and session_id == "{sid}"'
        if topic_id is not None:
            if not topic_id or any(c not in "0123456789abcdef-" for c in topic_id):
                raise ValueError("临时索引topic_id必须是UUID或哈希")
            expr += f' and topic_id == "{topic_id}"'
        if generation is not None:
            expr += f" and generation == {int(generation)}"
        return expr

    def upsert(self, rows, vectors):
        if len(rows) != len(vectors):
            raise ValueError("chunk与向量数量不一致")
        collection = self._collection()
        data = []
        for row, vector in zip(rows, vectors):
            meta = row["metadata"]
            self._scope(meta["user_id"], meta["session_id"], meta["topic_id"], meta["evidence_space_generation"])
            data.append({"id": row["id"], "user_id": meta["user_id"], "session_id": meta["session_id"],
                         "topic_id": meta["topic_id"], "document_id": meta["document_id"],
                         "unit_type": row["unit_type"], "generation": meta["evidence_space_generation"], "vector": vector})
        if data:
            collection.upsert(data)
            collection.flush()

    def prepare(self):
        self._collection()

    def existing_ids(self, rows):
        """核对实际索引；scope过滤防止同ID的错误generation/归属被当成成功。"""
        import json
        from collections import defaultdict
        groups = defaultdict(list)
        for row in rows:
            meta = row["metadata"]
            groups[(meta["user_id"], meta["session_id"], meta["topic_id"], meta["evidence_space_generation"])].append(row)
        collection = self._collection(create=False)
        if collection is None:
            return set()
        found = set()
        for scope, scoped in groups.items():
            for offset in range(0, len(scoped), 256):
                batch = {row["id"]: row for row in scoped[offset:offset + 256]}
                expr = self._scope(*scope) + " and id in " + json.dumps(list(batch))
                for hit in collection.query(expr, output_fields=["id", "document_id", "unit_type"],
                                            consistency_level="Strong", timeout=15):
                    row = batch.get(hit["id"])
                    if row and hit["document_id"] == row["metadata"]["document_id"] and hit["unit_type"] == row["unit_type"]:
                        found.add(hit["id"])
        return found

    def search(self, user_id, session_id, topic_id, generation, vector, kinds, limit, *, offset=0):
        if type(offset) is not int or type(limit) is not int or offset < 0 or limit <= 0 or offset + limit >= 16384:
            raise ValueError("临时向量检索分页超出Milvus范围")
        collection = self._collection(create=False)
        if collection is None:
            return []
        allowed = {"paragraph", "section", "sentence", "table", "record"}
        if not set(kinds) <= allowed:
            raise ValueError("非法临时证据类型")
        expr = self._scope(user_id, session_id, topic_id, generation)
        expr += " and unit_type in [" + ",".join('"' + k + '"' for k in sorted(kinds)) + "]"
        hits = collection.search([vector], "vector", {"metric_type": "COSINE", "params": {}, "offset": offset},
                                 limit=limit, expr=expr, consistency_level="Strong")
        return [hit.id for hit in hits[0]] if hits else []

    def delete_session(self, user_id, session_id):
        collection = self._collection(create=False)
        if collection is not None:
            collection.delete(self._scope(user_id, session_id))
            collection.flush()
