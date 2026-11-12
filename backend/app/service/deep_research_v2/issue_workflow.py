"""当前轮审核意见的按章修订与修改记录；下一轮 Critic 重新审核完整报告。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping


def text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def open_issues(state: Mapping) -> list[dict]:
    """仅返回最近一次完整审核的问题；不把上一轮问题自动合并进来。"""
    return list(state.get("critic_feedback") or [])


def section_issues(state: Mapping, section_id: str) -> list[dict]:
    return [issue for issue in open_issues(state)
            if section_id in issue["resolved_section_ids"]]


def acceptance(issue: Mapping) -> str:
    return issue["modification"]


def issue_query(issues: list[dict]) -> str:
    return "\n".join(" ".join(text(issue.get(key)) for key in
                              ("description", "evidence", "modification", "search_query"))
                     for issue in issues)


def revision_context(state: Mapping, section_id: str) -> str:
    """将当前轮意见及其补搜结果交给对应章节的 Writer，要求具体修改与依据。"""
    rows = []
    for issue in section_issues(state, section_id):
        search = (state.get("issue_search_results") or {}).get(issue["issue_id"], {})
        rows.append({
            "issue_id": issue["issue_id"], "target": issue["target"],
            "requirement": state.get("query", ""),
            "problem": issue["description"], "judgment_basis": issue["evidence"],
            "requested_change": issue["modification"], "acceptance_criteria": acceptance(issue),
            "new_evidence": search or {
                "status": "not_requested" if not issue["requires_new_search"] else "not_recorded",
                "meaning": "没有本问题的补搜结果；按实际原文判断，不得假称已取得支持。"},
        })
    return json.dumps(rows, ensure_ascii=False)


ISSUE_RESPONSE_INSTRUCTIONS = """## 按问题修改与修改记录
上方按章问题清单中的每个 issue_id 都须返回一项 issue_responses，不能遗漏或更换ID。
基于所指原句/用户要求、判断依据、具体修改要求、实际原文及验收条件处理；保留无关的正确内容。
允许认为原批评有误而维持原文，但必须说明依据；无证据时如实保留缺口，不能靠添加链接宣称修复。
每项格式：{"issue_id":"原ID","status":"addressed/unable_to_address/disputed",
"revised_quote":"本次输出中可逐字定位的完整句段或表格行；删除时引用保留的相关上下文",
"change_description":"改了什么，或为何维持原文",
"evidence":[{"document_id":"本次提供的编号","sentence_ids":["本次提供的原句编号"]}],
"evidence_summary":"这些原文具体支持什么；纯逻辑/格式修改可说明无需外部证据",
"remaining_gaps":"仍缺什么，或明确无剩余缺口","acceptance_check":"逐项回应验收条件"}。
若提供的是 evidence_units，evidence 使用 document_id 和 evidence_unit_ids；不得创建未提供的编号。
首尾合稿只提供章节依据时，evidence 改用 {"source_claim_ids":["本次提供的C编号"]}，不伪造来源原句编号。
Writer的addressed只表示已处理、等待复审，不代表审核通过，下一轮由 Critic 根据完整报告重新判断。无问题时 issue_responses 返回 []。
不能用旧addressed_issues数组代替issue_responses的逐项定位、依据和验收回应。
"""


def evidence_lookup(packet: Mapping) -> dict[tuple, dict]:
    """只允许本次真正给 Writer 展示过的原句/单元，不能引用库里未展示的内容。"""
    lookup = {}
    for doc in packet.get("documents") or []:
        for sentence in doc.get("original_sentences") or []:
            lookup[(doc.get("document_id"), "sentence_ids", sentence.get("id"))] = {
                "text": sentence.get("text", ""), "source_url": doc.get("url") or doc.get("source_url", "")}
    for unit in packet.get("evidence_units") or []:
        lookup[(unit.get("document_id"), "evidence_unit_ids", unit.get("id"))] = {
            "text": unit.get("text", ""), "source_url": unit.get("source_url") or unit.get("url", "")}
    for ident, item in (packet.get("source_claim_index") or {}).items():
        lookup[(None, "source_claim_ids", ident)] = {
            "text": item["declaration"]["claim"], "kind": "section_claim_reference",
            "evidence": deepcopy(item["declaration"].get("evidence") or [])}
    return lookup


def record_writer_responses(state: dict, section_id: str, content: str,
                            result: Mapping, packet: Mapping) -> list[dict]:
    """缺失/伪造定位或证据留作未处理记录，不依据旧 addressed_issues 自动关闭。"""
    issues = section_issues(state, section_id)
    if not issues:
        return []
    supplied = result.get("issue_responses")
    supplied = supplied if isinstance(supplied, list) else []
    lookup = evidence_lookup(packet)
    rows = []
    for issue in issues:
        matches = [row for row in supplied if isinstance(row, dict) and row.get("issue_id") == issue["issue_id"]]
        row = deepcopy(matches[0]) if len(matches) == 1 else {}
        errors = []
        if len(matches) != 1:
            errors.append("本问题缺少唯一的Writer逐项回应")
        if row.get("status") not in {"addressed", "unable_to_address", "disputed"}:
            errors.append("Writer处理状态无效")
        quote = text(row.get("revised_quote"))
        if not quote or quote not in content:
            errors.append("修改位置不在本次实际正文中")
        for key in ("change_description", "evidence_summary", "remaining_gaps", "acceptance_check"):
            if not text(row.get(key)):
                errors.append(f"缺少{key}")
        refs = row.get("evidence")
        excerpts = []
        if not isinstance(refs, list):
            errors.append("evidence必须是数组")
            refs = []
        for ref in refs:
            valid = isinstance(ref, dict)
            fields = [key for key in ("sentence_ids", "evidence_unit_ids", "source_claim_ids") if isinstance(ref, dict) and key in ref]
            valid = valid and len(fields) == 1
            valid = valid and (ref.get("document_id") is None if fields[0] == "source_claim_ids"
                               else bool(text(ref.get("document_id"))))
            ids = ref.get(fields[0]) if valid else None
            valid = valid and isinstance(ids, list) and bool(ids) and all(isinstance(i, str) for i in ids)
            valid = valid and len(set(ids)) == len(ids) and all(
                (ref.get("document_id"), fields[0], ident) in lookup for ident in ids)
            if not valid:
                errors.append("引用了本次未展示或无效的证据编号")
                continue
            excerpts.extend({**lookup[(ref.get("document_id"), fields[0], ident)],
                             "document_id": ref.get("document_id"), fields[0]: [ident]} for ident in ids)
        row.update(issue_id=issue["issue_id"], section_id=section_id,
                   iteration=int(state.get("iteration", 0)),
                   content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                   validation_errors=errors, evidence_excerpts=excerpts)
        if errors:
            row["status"] = "not_reported"
        rows.append(row)
    apply_writer_records(state, rows)
    return rows


def apply_writer_records(state: dict, rows: list[dict]) -> None:
    """首次生成和章节检查点重放共用幂等写入；不改变复审状态。"""
    keys = {(r["issue_id"], r["section_id"], r["iteration"]) for r in rows}
    state["writer_revision_records"] = [r for r in state.get("writer_revision_records") or []
        if (r["issue_id"], r["section_id"], r["iteration"]) not in keys] + rows
    for issue in state["critic_feedback"]:
        responses = [r for r in state["writer_revision_records"]
                     if r["issue_id"] == issue["issue_id"] and r["iteration"] == int(state.get("iteration", 0))]
        if responses:
            expected = {section["id"] for section in state.get("outline") or []
                        if any(item["issue_id"] == issue["issue_id"] for item in section_issues(state, section["id"]))}
            if any(item["issue_id"] == issue["issue_id"] for item in section_issues(state, "__synthesis__")):
                expected.add("__synthesis__")
            statuses = {row["status"] for row in responses}
            if "not_reported" in statuses:
                status = "not_reported"
            elif "unable_to_address" in statuses:
                status = "unable_to_address"
            elif not expected.issubset({row["section_id"] for row in responses}) or len(statuses) != 1:
                status = "partial"
            else:
                status = next(iter(statuses))
            issue["writer_status"] = status
