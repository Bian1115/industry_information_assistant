"""将依赖版本随产物提交；原始数据缺失时拒绝恢复，索引另行重建。

清单嵌入 LangGraph checkpoint/pending write，不引入另一套先后提交的清单表。
历史与预算允许继续追加或迟到结算，已引用的不可变记录不得删除/改写。
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
import json

from .checkpointer import lease_transaction, shared_connection

PROTOCOL = "research-storage-v1"
_current = ContextVar("research_storage_coordinator", default=None)
# 与SessionEvidenceStore._space相同：只有到期前的新消息能延长idle期限。
# 对账只读计算，不以当前时间续期，也不在run锁内反向锁chat。
_ALIVE = ("(CASE WHEN s.ttl_mode='idle' AND c.expires_at - interval '24 hours' < s.expires_at "
          "THEN greatest(s.expires_at, c.expires_at - interval '24 hours' + s.ttl_seconds * interval '1 second') "
          "ELSE s.expires_at END) > clock_timestamp() AND c.expires_at > clock_timestamp()")


def digest(value):
    # 与 agent_history.content_hash 使用相同的 JSON 规范。
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


class StorageConsistencyError(RuntimeError):
    code = "research_storage_inconsistent"


def require(condition, message):
    if not condition:
        raise StorageConsistencyError(message)


@contextmanager
def bind_storage_coordinator(coordinator):
    token = _current.set(coordinator)
    try:
        yield coordinator
    finally:
        _current.reset(token)


def current_storage_coordinator():
    return _current.get()


def artifacts(value):
    """只识别真实图状态/章节返回值，不把任意历史 payload 当执行结果。"""
    if not isinstance(value, dict):
        return []
    if "research" in value:
        return artifacts(value["research"])
    if "storage_manifest" in value or "storage_protocol" in value:
        return [value]
    if "graph_version" in value or (value.get("version") == 1 and "input_hash" in value and "content" in value):
        return [value]
    return []


def document_refs(research):
    work = research
    while work:
        space = work.get("evidence_space") or {}
        for source in work.get("raw_sources") or []:
            if source.get("evidence_document_ref"):
                require(bool(space), "外置正文缺少证据空间信息")
                yield space, source
        work = work.get("session_parent")


class StorageCoordinator:
    def __init__(self, conn, run_id, *, lease=None):
        require(lease is None or lease.run_id == run_id, "对账运行与执行租约不匹配")
        self.conn, self.run_id, self.lease = conn, run_id, lease

    async def _run(self, cur):
        await cur.execute("SELECT session_id, user_id, next_event_seq, final_checkpoint_id, delivered_event_id "
                          "FROM research_runs WHERE run_id=%s", (self.run_id,))
        run = await cur.fetchone()
        require(run is not None, "运行记录不存在，无法对账")
        return run

    async def _documents(self, cur, research, run):
        from service.web_document import sentence_records
        result = {}
        for space, source in document_refs(research):
            sid, uid = str(space.get("session_id")), str(space.get("user_id"))
            require(sid == str(run["session_id"]) and uid == str(run["user_id"]), "正文引用不属于当前用户和会话")
            await cur.execute(
                "SELECT s.generation, s.revision, s.status, "
                + _ALIVE + " AS alive "
                "FROM session_evidence_spaces s JOIN chat_sessions c ON c.id=s.session_id "
                "WHERE s.session_id=%s AND s.user_id=%s AND c.user_id=%s", (sid, uid, uid))
            current = await cur.fetchone()
            require(current and current["alive"] and current["status"] == "active", "临时正文空间已缺失、关闭或过期")
            require(current["generation"] == space.get("generation"), "临时正文generation已变化")
            require(current["revision"] >= int(space.get("revision", 0)), "临时正文revision发生回退")
            await cur.execute("SELECT id, payload FROM session_evidence_documents "
                              "WHERE session_id=%s AND topic_id=%s AND document_id=%s",
                              (sid, space.get("topic_id"), source.get("document_id")))
            rows = await cur.fetchall()
            require(len(rows) == 1, "检查点引用的临时正文不存在或编号不唯一")
            row, payload = rows[0], rows[0]["payload"]
            require(payload.get("document_id") == source.get("document_id") and
                    payload.get("content_sha256") == source.get("content_sha256") and
                    payload.get("document_version") == source.get("document_version") and
                    bool(sentence_records(payload)), "检查点引用的正文哈希、版本或原句编号不一致")
            result[row["id"]] = {"id": row["id"], "payload_hash": digest(payload),
                                 "generation": current["generation"], "session_id": sid, "user_id": uid,
                                 "topic_id": space.get("topic_id"), "document_id": source["document_id"]}
        return [result[key] for key in sorted(result)]

    async def _history(self, cur):
        await cur.execute("SELECT id, kind, call_id, invocation_id, node, payload, content_hash "
                          "FROM agent_messages WHERE run_id=%s ORDER BY id", (self.run_id,))
        result = []
        for row in await cur.fetchall():
            require(digest(row["payload"]) == row["content_hash"], "Agent历史正文与其哈希不一致")
            result.append({k: str(row[k]) for k in ("id", "kind", "call_id", "invocation_id", "node", "content_hash")})
        return result

    async def _budget(self, cur, required=False):
        await cur.execute("SELECT policy, baseline FROM research_budgets WHERE run_id=%s", (self.run_id,))
        header = await cur.fetchone()
        require(header is not None or not required, "检查点已引用预算账本，但权威预算记录缺失")
        if header is None:
            return None
        await cur.execute("SELECT attempt_id, owner_id, fencing_token, node, phase, agent, started_at, "
                          "elapsed_seconds, fallback_tokens, fallback_calls "
                          "FROM research_budget_attempts WHERE run_id=%s ORDER BY attempt_id", (self.run_id,))
        attempts = await cur.fetchall()
        for row in attempts:
            row["started_at"] = row["started_at"].isoformat()
        await cur.execute("SELECT c.call_id, c.attempt_id, c.model, c.reserved_tokens, c.actual_tokens FROM research_budget_calls c "
                          "JOIN research_budget_attempts a ON a.attempt_id=c.attempt_id "
                          "WHERE a.run_id=%s ORDER BY c.call_id", (self.run_id,))
        return {"header_hash": digest(header), "attempts": attempts, "calls": await cur.fetchall()}

    async def _events(self, cur, run):
        await cur.execute("SELECT count(*) AS n, min(seq) AS first, max(seq) AS last, "
                          "count(DISTINCT seq) AS unique_n, bool_and(session_id=%s AND user_id IS NOT DISTINCT FROM %s::uuid) AS owned "
                          "FROM agent_events WHERE run_id=%s", (run["session_id"], run["user_id"], self.run_id))
        row = await cur.fetchone()
        require((row["n"] == 0 and run["next_event_seq"] == 0) or
                (row["first"] == 1 and row["last"] == row["n"] == row["unique_n"] == run["next_event_seq"]),
                "事件账本序号缺失或计数不一致；进度事件不能代替执行产物")
        require(not row["n"] or row["owned"], "事件账本混入其他用户或会话的记录")
        return int(row["last"] or 0)

    async def seal(self, value, research=None, *, node="input", kind="research"):
        require(self.lease is not None, "只有持有执行租约的运行才能生成产物清单")
        value = deepcopy(value)
        value.pop("storage_manifest", None)
        if kind == "research":
            value["storage_protocol"] = PROTOCOL
        dependencies_source = research if research is not None else value
        async with lease_transaction(self.conn, self.lease) as cur:
            run = await self._run(cur)
            dependencies = {"documents": await self._documents(cur, dependencies_source, run),
                            "history": await self._history(cur),
                            "budget": await self._budget(cur, bool((dependencies_source.get("runtime_budget") or {}).get("ledger_version"))),
                            "event_seq": await self._events(cur, run)}
        value["storage_manifest"] = {"version": PROTOCOL, "run_id": self.run_id,
            "session_id": str(run["session_id"]), "user_id": str(run["user_id"]),
            "node": node, "kind": kind, "artifact_hash": digest(value), "dependencies": dependencies}
        return value

    async def validate(self, cur, value):
        """调用者已持有run行锁；校验与saver写入共用同一事务。"""
        run = await self._run(cur)
        manifest = value.get("storage_manifest")
        require(value.get("storage_protocol") in (None, PROTOCOL), "不支持的跨存储协议版本")
        require(bool(manifest) or not value.get("storage_protocol"), "产物声明跨存储协议但缺少清单")
        if "storage_manifest" in value:
            require(isinstance(manifest, dict) and bool(manifest), "产物依赖清单为空或损坏")
        if manifest:
            require(isinstance(manifest, dict) and manifest.get("kind") in {"research", "writer_section"} and
                    isinstance(manifest.get("dependencies"), dict), "产物依赖清单结构不合法")
            deps = manifest["dependencies"]
            require(all(isinstance(deps.get(key), list) for key in ("documents", "history")) and
                    isinstance(deps.get("event_seq"), int) and "budget" in deps, "产物依赖清单字段不完整")
            require(manifest.get("version") == PROTOCOL and manifest.get("run_id") == self.run_id and
                    manifest.get("session_id") == str(run["session_id"]) and
                    manifest.get("user_id") == str(run["user_id"]), "产物清单版本或归属不一致")
            require(manifest.get("artifact_hash") == digest({k: v for k, v in value.items() if k != "storage_manifest"}),
                    "检查点/章节内容与产物清单哈希不一致")
        docs = await self._documents(cur, value, run)
        history = await self._history(cur)
        budget = await self._budget(cur, bool((value.get("runtime_budget") or {}).get("ledger_version")))
        event_seq = await self._events(cur, run)
        if manifest:
            deps = manifest["dependencies"]
            # 章节的依赖来源是其输入快照，返回值自身没有 raw_sources。
            for expected in deps["documents"]:
                await cur.execute("SELECT d.payload, s.generation, s.status, "
                    + _ALIVE + " AS alive "
                    "FROM session_evidence_documents d JOIN session_evidence_spaces s ON s.session_id=d.session_id "
                    "JOIN chat_sessions c ON c.id=s.session_id WHERE d.id=%s AND d.session_id=%s "
                    "AND d.topic_id=%s AND d.document_id=%s AND s.user_id=%s AND c.user_id=%s",
                    (expected["id"], expected["session_id"], expected["topic_id"], expected["document_id"],
                     expected["user_id"], expected["user_id"]))
                live = await cur.fetchone()
                require(expected["session_id"] == str(run["session_id"]) and expected["user_id"] == str(run["user_id"]) and
                        live and live["alive"] and live["status"] == "active" and
                        live["generation"] == expected["generation"] and digest(live["payload"]) == expected["payload_hash"],
                        "产物依赖的不可变正文已缺失、过期或改写")
            if manifest["kind"] == "research":
                require(docs == deps["documents"], "检查点正文引用与依赖清单不一致")
            self._subset(deps["history"], history, "id", "Agent历史")
            if deps["budget"]:
                require(budget and deps["budget"]["header_hash"] == budget["header_hash"], "预算策略或基线已缺失/改写")
                self._budget_subset(deps["budget"], budget)
            require(event_seq >= deps["event_seq"], "事件账本发生回退")
        return {"legacy": not bool(manifest), "history_rows": len(history),
                "uncommitted_history_rows": len(history) - len(manifest["dependencies"]["history"]) if manifest else None,
                "event_seq": event_seq}

    @staticmethod
    def _subset(expected, actual, key, label):
        current = {row[key]: row for row in actual}
        require(all(current.get(row[key]) == row for row in expected), label + "依赖记录缺失或被改写")

    @staticmethod
    def _budget_subset(expected, actual):
        for group, key, label in (("attempts", "attempt_id", "预算尝试"), ("calls", "call_id", "预算请求")):
            live = {row[key]: row for row in actual[group]}
            for row in expected[group]:
                current = live.get(row[key])
                require(current is not None, label + "依赖记录缺失")
                for field, value in row.items():
                    if field == "actual_tokens" and value is None:
                        # 结果未知的预留可以迟到结算；已知usage不可反向减少或改写。
                        require(current[field] is None or current[field] >= 0, "预算结算值不合法")
                    elif field in {"elapsed_seconds", "fallback_tokens", "fallback_calls"}:
                        require(current[field] >= value, label + "累计消耗发生回退")
                    else:
                        require(current[field] == value, label + "依赖记录被改写")

    async def _delivery(self, cur, snapshot):
        research = (snapshot.values or {}).get("research") or {}
        if research.get("status") != "completed" or snapshot.next or snapshot.tasks:
            return "not_terminal"
        run = await self._run(cur)
        await cur.execute("SELECT event_id, turn_id, payload FROM agent_events WHERE run_id=%s "
                          "AND event_type='research_complete'", (self.run_id,))
        events = await cur.fetchall()
        if not events:
            require(not run["delivered_event_id"] and not run["final_checkpoint_id"], "交付指针存在但完成事件缺失")
            return "pending"
        require(len(events) == 1, "同一运行存在多个完成事件")
        event = events[0]
        checkpoint_id = str((snapshot.config or {}).get("configurable", {}).get("checkpoint_id") or "")
        require(str(run["delivered_event_id"]) == str(event["event_id"]) and
                run["final_checkpoint_id"] == checkpoint_id == event["payload"].get("checkpoint_id") and
                str(event["payload"].get("final_report") or "").strip() == str(research.get("final_report") or "").strip(),
                "终态检查点、完成事件与交付指针不一致")
        if event["turn_id"] is not None:
            await cur.execute("SELECT content FROM chat_messages WHERE run_id=%s AND role='assistant'", (self.run_id,))
            messages = await cur.fetchall()
            require(len(messages) == 1 and messages[0]["content"].strip() == str(research.get("final_report") or "").strip(),
                    "已交付报告与终态检查点不一致")
        return "delivered"

    async def inspect(self, snapshot, checkpoint_tuple=None):
        """只读对账覆盖未合并节点及章节task；不会据历史/事件推导节点成功。"""
        candidates = artifacts((snapshot.values or {}).get("research"))
        if checkpoint_tuple is not None:
            for value in checkpoint_tuple.checkpoint.get("channel_values", {}).values():
                candidates.extend(artifacts(value))
            for _, _, value in checkpoint_tuple.pending_writes or []:
                candidates.extend(artifacts(value))
        # 同一个research通常同时出现在已保存值和合并视图，避免重复扫描。
        candidates = list({digest(value): value for value in candidates}.values())
        try:
            from psycopg.rows import dict_row
            async with shared_connection(self.conn) as connection:
                async with connection.transaction(), connection.cursor(row_factory=dict_row) as cur:
                    # 所有写入先锁run；与并发持久化相隔离，但不锁chat，避免清理锁序反转。
                    await cur.execute("SELECT run_id FROM research_runs WHERE run_id=%s FOR SHARE", (self.run_id,))
                    results = [await self.validate(cur, value) for value in candidates]
                    if not candidates:
                        await self._events(cur, await self._run(cur))
                    delivery = await self._delivery(cur, snapshot)
            return {"status": "legacy_checked" if any(r["legacy"] for r in results) else "consistent",
                    "blocked": False, "checked_artifacts": len(results),
                    "delivery": delivery, "derived_indexes": "checked_on_resume",
                    "legacy_limit": "旧产物未保存依赖清单，只能核对仍存在的数据" if any(r["legacy"] for r in results) else "",
                    "uncommitted_history_rows": max((r["uncommitted_history_rows"] or 0 for r in results), default=0),
                    "events_are_observations": True}
        except StorageConsistencyError as exc:
            return {"status": "blocked", "blocked": True, "error_code": exc.code, "reason": str(exc),
                    "checked_artifacts": len(candidates)}
        except Exception as exc:
            # 不把数据库异常文本（可能含连接信息）透传给API。
            return {"status": "unavailable", "blocked": True, "error_code": "research_storage_unavailable",
                    "reason": "跨存储对账暂不可用：" + type(exc).__name__, "checked_artifacts": len(candidates)}

    async def require_consistent(self, snapshot, checkpoint_tuple=None):
        result = await self.inspect(snapshot, checkpoint_tuple)
        require(not result["blocked"], result.get("reason", "跨存储对账失败"))
        return result
