"""会话任务的类型协议、授权成果索引和确定性执行计划；不读取智能体历史。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import logging
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TASK_VERSION = "session-task-v1"
logger = logging.getLogger(__name__)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TaskOperation(StrictModel):
    kind: Literal["edit", "evidence", "structure", "research"]
    # 溯源字段只供离线评估，异常类型也保留，不参与执行门槛。
    source_sentence_ids: Any = Field(default_factory=list, json_schema_extra={"type": "array", "items": {"type": "string"}})
    quote: Any = None  # 兼容旧任务；模型原文引用不再用于执行校验。


class RequirementChange(StrictModel):
    op: Literal["set", "remove"]
    field: Literal["max_body_chars", "required_entities", "critical_questions", "explicit_requirements", "required_metrics", "required_calculations", "time_range", "geography", "language", "output_format"]
    value: Any = None
    source_sentence_ids: Any = Field(default_factory=list, json_schema_extra={"type": "array", "items": {"type": "string"}})
    quote: Any = None


class SearchTask(StrictModel):
    query: str = Field(min_length=1, max_length=600)
    target_section: str = Field(default="", max_length=300,
        description="所选报告真实大纲的章节ID，不能填写标题；空字符串表示整篇补证")
    evidence_goal: str = Field(min_length=1, max_length=1000)


class SearchSectionAssignment(StrictModel):
    task_index: int = Field(ge=0, le=9)
    target_section: str = Field(max_length=300)


class SearchSectionRepair(StrictModel):
    assignments: list[SearchSectionAssignment] = Field(min_length=1, max_length=10)


class SearchSectionReferenceError(ValueError):
    """报告已确定，但搜索任务引用了该报告之外的章节。"""

    def __init__(self, artifact_ref: str, sections: list[dict], invalid_targets: list[str]):
        self.artifact_ref = artifact_ref
        self.sections = deepcopy(sections)
        self.invalid_targets = list(invalid_targets)
        super().__init__("搜索任务必须关联所选报告的真实章节ID：" + ", ".join(invalid_targets))


def outline_sections(outline: list[dict]) -> list[dict]:
    """目录来自研究outline，不从正文标题猜测或生成ID。"""
    sections, seen = [], set()
    for section in outline:
        identity, title = section.get("id"), section.get("title")
        if not isinstance(identity, str) or not identity.strip() or identity != identity.strip() or "," in identity:
            raise ValueError("研究大纲缺少合法章节ID")
        if identity in seen or not isinstance(title, str) or not title.strip():
            raise ValueError("研究大纲章节ID重复或标题缺失")
        seen.add(identity)
        sections.append({"id": identity, "title": title})
    return sections


def checkpoint_section_catalogs(candidates: list[dict], session_id: str) -> dict[str, list[dict]]:
    """只读旧报告已完成检查点；候选须先经过用户/会话授权筛选。"""
    if not candidates:
        return {}
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.conninfo import make_conninfo
    from service.deep_research_v2.checkpointer import checkpoint_database_url
    result = {}
    try:
        # 不setup、不保存、不恢复图；新报告已随交付保存目录，只有旧报告需要此读取。
        with PostgresSaver.from_conn_string(make_conninfo(checkpoint_database_url(), connect_timeout=3)) as saver:
            for candidate in candidates:
                try:
                    config = {"configurable": {"thread_id": candidate["run_id"], "checkpoint_ns": ""}}
                    item = saver.get_tuple(config)
                    state = (item.checkpoint.get("channel_values") or {}).get("research") if item else None
                    if not state or state.get("session_id") != session_id or state.get("status") != "completed":
                        continue
                    if report_hash(str(state.get("final_report") or "").strip()) != candidate["report_hash"]:
                        continue
                    result[candidate["artifact_ref"]] = outline_sections(state.get("outline") or [])
                except Exception as exc:
                    logger.warning("旧报告大纲读取不可用，run=%s，错误类型=%s", candidate["run_id"], type(exc).__name__)
    except Exception as exc:
        logger.warning("检查点目录读取不可用，错误类型=%s", type(exc).__name__)
    return result


def repair_search_section_ids(value: dict, patch: dict, sections: list[dict]) -> dict:
    """局部修正只能改关联ID，不能改报告选择、关键词、研究要求或补证目标。"""
    repair = SearchSectionRepair.model_validate(patch)
    tasks = value.get("search_tasks") or []
    indexes = [assignment.task_index for assignment in repair.assignments]
    if len(indexes) != len(tasks) or set(indexes) != set(range(len(tasks))):
        raise ValueError("章节ID修正必须逐项覆盖原搜索任务，不能重复或增删")
    known = {section["id"] for section in outline_sections(sections)}
    result = deepcopy(value)
    for assignment in repair.assignments:
        target = assignment.target_section.strip()
        if target not in known and not (target == "" and tasks[assignment.task_index].get("target_section", "") == ""):
            raise ValueError("章节ID修正仍引用所选报告以外的章节")
        result["search_tasks"][assignment.task_index]["target_section"] = target
    return result


class SessionTaskDecision(StrictModel):
    version: Literal["session-task-v1"] = TASK_VERSION
    topic_relation: Literal["continue", "return_to_previous", "new", "uncertain"]
    depends_on_history: bool
    target_artifact_ref: str | None = None
    resolved_query: str = Field(min_length=1, max_length=8000)
    operations: list[TaskOperation] = Field(min_length=1, max_length=8)
    requirement_changes: list[RequirementChange] = Field(default_factory=list, max_length=20)
    revision_instructions: list[str] = Field(default_factory=list, max_length=20)
    search_tasks: list[SearchTask] = Field(default_factory=list, max_length=10)
    used_memory_ids: list[str] = Field(default_factory=list, max_length=10)
    assumptions: list[str] = Field(default_factory=list, max_length=10)
    unresolved_references: list[str] = Field(default_factory=list, max_length=10)
    clarification_question: str = ""

    @model_validator(mode="after")
    def consistent_topic(self):
        if self.topic_relation == "new" and (self.depends_on_history or self.target_artifact_ref):
            raise ValueError("新话题不能依赖历史或选择旧成果")
        if any(not task.query.strip() or not task.evidence_goal.strip() for task in self.search_tasks):
            raise ValueError("搜索任务不能为空")
        if any(not x.strip() for x in self.revision_instructions):
            raise ValueError("修改指令不能为空")
        return self


class SessionTaskBudgetExceeded(ValueError):
    """会话判断和研究共享整任务预算，耗尽不能降级绕过。"""


class SessionTaskClarification(ValueError):
    code = "needs_clarification"

    def __init__(self, question: str):
        self.question = question
        super().__init__(question)


def query_sentence_index(query: str, turn_id: str | None = None) -> dict:
    """确定性分句；位置是Python Unicode字符偏移[start,end)，原文不改写。"""
    protected = set()
    pattern = r"```[\s\S]*?(?:```|$)|`[^`\n]*`|https?://[^\s<>。！？；]+|“[^”]*”|‘[^’]*’"
    for match in re.finditer(pattern, query):
        protected.update(range(match.start(), match.end()))
    sentences = []

    def append(start, end):
        while start < end and query[start].isspace():
            start += 1
        while end > start and query[end - 1].isspace():
            end -= 1
        if start < end:
            sentences.append({"id": f"S{len(sentences) + 1}", "text": query[start:end], "start": start, "end": end})

    start = 0
    for index, char in enumerate(query):
        boundary = char in "。！？；!?;\n\r" or (
            char == "." and (index + 1 == len(query) or query[index + 1].isspace()))
        if boundary and index not in protected:
            append(start, index + 1)
            start = index + 1
    append(start, len(query))
    return {"version": "session-query-sentences-v1", "turn_id": turn_id,
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "offset_unit": "unicode_code_points", "sentences": sentences}


def task_source_trace(value: Any, query: str, turn_id: str | None = None) -> dict:
    """引用可定位性只记诊断；缺失、错号、异常类型均不抛错或改变执行计划。"""
    trace = query_sentence_index(query, turn_id)
    known = {sentence["id"] for sentence in trace["sentences"]}
    entries = []
    for group in ("operations", "requirement_changes"):
        items = value.get(group, []) if isinstance(value, dict) else []
        if not isinstance(items, list):
            continue  # 非溯源部分的结构错误交由任务协议校验。
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            ids = item.get("source_sentence_ids", [])
            resolved, warnings = [], []
            if not isinstance(ids, list):
                warnings.append("invalid_id_list")
            elif not ids:
                warnings.append("missing_sentence_ids")
            else:
                for identity in ids:
                    if not isinstance(identity, str):
                        warnings.append("invalid_id_type")
                    elif identity not in known:
                        warnings.append("unknown_sentence_id")
                    elif identity in resolved:
                        warnings.append("duplicate_sentence_id")
                    else:
                        resolved.append(identity)
            entries.append({"group": group, "index": index, "reported_sentence_ids": deepcopy(ids),
                            "resolved_sentence_ids": resolved, "legacy_quote": deepcopy(item.get("quote")),
                            "warnings": warnings})
    trace.update(entries=entries, status="issues" if any(x["warnings"] for x in entries) else "resolved",
                 item_count=len(entries), issue_item_count=sum(bool(x["warnings"]) for x in entries))
    return trace


def report_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_artifact_candidates(user_id: str, session_id: str, exclude_run_id: str) -> list[dict]:
    """保留最早及最近成果，按时间排序；模型只看到有界索引。"""
    from core.database import SessionLocal
    from models.chat import ChatMessage, ChatSession
    from core.time_utils import utc_now
    with SessionLocal() as db:
        query = (db.query(ChatMessage).join(ChatSession, ChatSession.id == ChatMessage.session_id)
                 .filter(ChatSession.user_id == uuid.UUID(user_id), ChatSession.id == uuid.UUID(session_id),
                         ChatSession.expires_at > utc_now(),
                         ChatMessage.role == "assistant", ChatMessage.execution_mode == "research",
                         ChatMessage.research_status == "completed", ChatMessage.run_id.isnot(None),
                         ChatMessage.run_id != exclude_run_id))
        recent = query.order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(24).all()
        first = query.order_by(ChatMessage.created_at, ChatMessage.id).limit(4).all()
        rows = sorted({str(r.id): r for r in first + recent}.values(), key=lambda r: (r.created_at, str(r.id)))
        turns = [r.turn_id for r in rows if r.turn_id]
        users = {str(r.turn_id): r.content for r in db.query(ChatMessage).filter(
            ChatMessage.session_id == uuid.UUID(session_id), ChatMessage.role == "user",
            ChatMessage.turn_id.in_(turns)).all()} if turns else {}
        result = []
        for row in rows:
            meta = row.references_data or {}
            contract = meta.get("research_contract") or {}
            content = str(row.content or "")
            result.append({"artifact_ref": "report:" + str(row.id), "run_id": row.run_id,
                           "query": users.get(str(row.turn_id), contract.get("query", ""))[:700],
                           "excerpt": content[:240], "report_hash": report_hash(content),
                           "section_titles": re.findall(r"^##\s+(.+)$", content, re.M)[:12],
                           "sections": outline_sections(meta.get("outline_sections") or []),
                           "requirements": {k: contract[k] for k in (
                               "active_constraints", "time_range", "geography", "language", "output_format",
                               "required_entities", "critical_questions", "explicit_requirements", "required_metrics", "required_calculations") if k in contract}})
        missing = [candidate for candidate in result if not candidate["sections"]]
        catalogs = checkpoint_section_catalogs(missing, session_id)
        for candidate in missing:
            candidate["sections"] = catalogs.get(candidate["artifact_ref"], [])
        return result


def authorize_artifact(user_id: str, session_id: str, run_id: str, expected_hash: str) -> None:
    """再次验证输入契约，不以模型生成的run_id作为授权。"""
    from core.database import SessionLocal
    from models.chat import ChatMessage, ChatSession
    from core.time_utils import utc_now
    with SessionLocal() as db:
        row = (db.query(ChatMessage).join(ChatSession, ChatSession.id == ChatMessage.session_id)
               .filter(ChatSession.user_id == uuid.UUID(user_id), ChatSession.id == uuid.UUID(session_id),
                       ChatSession.expires_at > utc_now(),
                       ChatMessage.role == "assistant", ChatMessage.execution_mode == "research",
                       ChatMessage.research_status == "completed", ChatMessage.run_id == run_id).one_or_none())
        if row is None or report_hash(row.content) != expected_hash:
            raise PermissionError("目标成果不属于当前用户会话、未交付或版本已变化")


def validate_decision(value: dict, original_query: str, candidates: list[dict], *, turn_id: str | None = None) -> dict:
    """保留原问题的决定由程序落实；只有服务端候选可成为目标。"""
    decision = SessionTaskDecision.model_validate(value)
    # 共用已有要求类型校验，set的add/replace语义在实际基线下确定。
    from service.deep_research_v2.contract_updates import ContractOperation
    for change in decision.requirement_changes:
        ContractOperation.model_validate({"field": change.field, "value": change.value, "quote": "",
            "op": "add" if change.op == "set" else "remove"}, context={"validate_quotes": False})
    if not decision.depends_on_history:
        decision.resolved_query = original_query
    selected = next((x for x in candidates if x["artifact_ref"] == decision.target_artifact_ref), None)
    if decision.target_artifact_ref and selected is None:
        raise ValueError("任务引用不在授权候选成果中")
    kinds = {x.kind for x in decision.operations}
    clarify = decision.topic_relation == "uncertain" or bool(decision.unresolved_references)
    if (decision.topic_relation == "return_to_previous" or kinds <= {"edit", "evidence"}) and not selected:
        clarify = True
    if decision.depends_on_history and decision.topic_relation == "new":
        raise ValueError("话题与历史依赖不一致")
    if "evidence" in kinds and not decision.search_tasks and "structure" not in kinds and "research" not in kinds:
        raise ValueError("定向补证必须给出搜索任务")
    plan = "replan" if kinds & {"structure", "research"} or not selected else "augment" if "evidence" in kinds else "edit"
    if plan == "augment" and not clarify:
        sections = outline_sections(selected.get("sections") or [])
        if not sections:
            raise SearchSectionReferenceError(selected["artifact_ref"], [], ["目录不可用"])
        known = {section["id"] for section in sections}
        invalid = []
        for search in decision.search_tasks:
            search.target_section = search.target_section.strip()
            if search.target_section and search.target_section not in known:
                invalid.append(search.target_section)
        if invalid:
            raise SearchSectionReferenceError(selected["artifact_ref"], sections, invalid)
    result = decision.model_dump()
    result.update(status="clarify" if clarify else "ready", selected_artifact=deepcopy(selected),
                  execution_plan=plan)
    result["source_trace"] = task_source_trace(value, original_query, turn_id)
    if clarify:
        result["clarification_question"] = decision.clarification_question or "请明确要修改的报告、章节或研究对象。"
    return result


def fallback_task(query: str, reason: str) -> dict:
    # 判断不可用时，明显依赖旧对象的输入不能直接当作新研究。
    dependent = bool(re.search(r"这份|该报告|上一[轮份个]|最开始|第二[个份]|^继续\s*$|^再|^(?:请)?(?:把|将)?(?:报告)?(?:缩短|压缩|精简)|取消.{0,8}限制", query))
    return {"version": TASK_VERSION, "status": "clarify" if dependent else "fallback",
            "topic_relation": "uncertain", "depends_on_history": dependent, "resolved_query": query,
            "selected_artifact": None, "execution_plan": "replan", "reason": reason,
            "clarification_question": "请明确本次研究对象或要修改的报告，当前未能可靠解析历史指代。" if dependent else ""}


def commit_requirements(base: dict, task: dict, original_query: str) -> dict:
    from service.deep_research_v2.contract_updates import apply_contract_patch
    baseline = deepcopy(base)
    operations = []
    for change in task.get("requirement_changes") or []:
        target = baseline.get("active_constraints", {}) if change["field"] == "max_body_chars" else baseline
        exists = change["field"] in target and target[change["field"]] not in (None, "", [])
        if change["op"] == "remove" and not exists:
            continue  # 已取消项的再次取消为幂等；类型仍由契约协议校验。
        operations.append({"field": change["field"], "value": change.get("value"), "quote": "",
            "op": "remove" if change["op"] == "remove" else "replace" if exists else "add"})
    committed = apply_contract_patch(baseline, {"base_revision": baseline.get("contract_revision", 0), "operations": operations},
                                original_query=original_query, resolved_query=task["resolved_query"],
                                instructions=task.get("revision_instructions") or [original_query], reuse_previous=True,
                                validate_quotes=False)
    committed["contract_update"]["source_trace"] = deepcopy(task.get("source_trace") or task_source_trace(task, original_query))
    return committed
