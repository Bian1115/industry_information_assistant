"""按问题返工的输入、修改定位与逐项复审；Writer 无权关闭审核问题。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping


TERMINAL_REVIEWS = {"resolved", "dismissed"}


def text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def signature(issue: Mapping) -> str:
    fields = {key: issue.get(key) for key in
              ("target_section", "issue_type", "location", "description")}
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def issue_ledger(state: Mapping) -> list[dict]:
    """旧检查点的 resolved 可能来自 Writer 自报，须重新经过显式复审。"""
    result, seen = [], set()
    for index, original in enumerate(state.get("critic_feedback") or []):
        issue = deepcopy(original)
        ident = text(issue.get("id")) or f"issue_{signature(issue)}"
        if ident in seen:
            ident = f"issue_{signature(issue)}_{index}"
        issue["id"] = ident
        seen.add(ident)
        if issue.get("review_status") not in TERMINAL_REVIEWS:
            if issue.get("resolved"):
                issue.setdefault("writer_status", "addressed")
            issue["resolved"] = False
            issue.setdefault("review_status", "pending")
        else:
            issue["resolved"] = True  # 兼容旧消费者；review_status 区分修复和撤销误报。
        result.append(issue)
    return result


def open_issues(state: Mapping) -> list[dict]:
    return [issue for issue in issue_ledger(state) if not issue["resolved"]]


def section_issues(state: Mapping, section_id: str) -> list[dict]:
    known = {s["id"] for s in state.get("outline") or []}
    result = []
    for issue in open_issues(state):
        targets = set(re.findall(r"sec_[A-Za-z0-9_]+", str(issue.get("target_section") or ""))) & known
        if (not targets or section_id in targets):
            result.append(issue)
    return result


def acceptance(issue: Mapping) -> Any:
    return issue.get("acceptance_criteria") or issue.get("suggestion") or (
        "回应所指原句或用户要求，说明依据和仍未解决的缺口，由 Critic 逐项复查。")


def issue_query(issues: list[dict]) -> str:
    return "\n".join(" ".join(text(issue.get(key)) for key in
                              ("report_quote", "requirement", "description", "search_query"))
                     for issue in issues)


def revision_context(state: Mapping, section_id: str) -> str:
    """来源已读不代表语义支持；支持范围与缺口须在原文基础上显式回应。"""
    rows = []
    for issue in section_issues(state, section_id):
        search = (state.get("issue_search_results") or {}).get(issue["id"], {})
        rows.append({
            "issue_id": issue["id"], "location": issue.get("location", ""),
            "report_quote": issue.get("report_quote", ""),
            "requirement": issue.get("requirement") or state.get("query", ""),
            "problem": issue.get("description", ""),
            "judgment_basis": issue.get("evidence") or "原审核未提供具体依据；须核对原句和实际材料，允许指出旧批评有误。",
            "requested_change": issue.get("suggestion") or issue.get("description") or "核对问题与原要求，缺证据时保留局限。",
            "acceptance_criteria": acceptance(issue),
            "new_evidence": search or {
                "status": "not_requested" if not issue.get("requires_new_search") else "not_recorded",
                "meaning": "没有本问题的补搜结果；按下方实际原文判断，不得假称已取得支持。"},
            "evidence_instruction": "逐项阅读提供的原文，说明新/已有证据具体支持什么、仍缺什么。来源数量或读取成功不代表问题解决。",
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
Writer的addressed只表示已处理、等待复审，不代表已解决。无问题时 issue_responses 返回 []。
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
        matches = [row for row in supplied if isinstance(row, dict) and row.get("issue_id") == issue["id"]]
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
        row.update(issue_id=issue["id"], section_id=section_id,
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
    state["critic_feedback"] = issue_ledger(state)
    for issue in state["critic_feedback"]:
        responses = [r for r in state["writer_revision_records"]
                     if r["issue_id"] == issue["id"] and r["iteration"] == int(state.get("iteration", 0))]
        if responses:
            expected = {section["id"] for section in state.get("outline") or []
                        if any(item["id"] == issue["id"] for item in section_issues(state, section["id"]))}
            if any(item["id"] == issue["id"] for item in section_issues(state, "__synthesis__")):
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


def review_packet(state: Mapping) -> str:
    rows = []
    for record in state.get("writer_revision_records") or []:
        if record.get("iteration") != int(state.get("iteration", 0)):
            continue
        row = deepcopy(record)
        quote = text(row.get("revised_quote"))
        row["located_in_current_report"] = bool(quote and quote in str(state.get("final_report") or ""))
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False)


def reconcile_reviews(state: Mapping, result: dict) -> tuple[list[dict], list[dict]]:
    """显式逐项复查是唯一关闭入口；遗漏、分数、指纹和来源数量都不关闭问题。"""
    ledger = issue_ledger(state)
    previous = {row["id"]: row for row in ledger if not row["resolved"]}
    reviews = result.get("issue_reviews", [])
    if not isinstance(reviews, list):
        raise ValueError("issue_reviews必须是数组")
    checks = {}
    for review in reviews:
        if not isinstance(review, dict) or not text(review.get("issue_id")) or review["issue_id"] not in previous:
            raise ValueError("复查引用了未知或已关闭的问题ID")
        ident = review["issue_id"]
        if ident in checks or text(review.get("status")) not in {"resolved", "unresolved", "dismissed"}:
            raise ValueError("复查问题ID重复或状态无效")
        checks[ident] = review
    returned = {text(row.get("id")): row for row in result["issues"]}
    returned_ids = [text(row.get("id")) for row in result["issues"] if text(row.get("id"))]
    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("issues包含重复问题ID，不能覆盖不同问题")
    for ident, issue in previous.items():
        review = checks.get(ident)
        if review is None:
            issue.update(review_status="unresolved", resolved=False,
                         resolution_reason="本轮未逐项复查，继续保留问题")
            continue
        status = review["status"]
        quote, reason = text(review.get("report_quote")), text(review.get("reason"))
        issue["last_review"] = deepcopy(review)
        if status in TERMINAL_REVIEWS and (not reason or not quote):
            status = "unresolved"
            reason = "复查未提供非空的报告摘录和判断依据，不能关闭问题"
        if status in TERMINAL_REVIEWS and ident in returned:
            status, reason = "unresolved", "复查结论与待修问题列表冲突，保留问题"
        issue.update(review_status=status, resolved=status in TERMINAL_REVIEWS, resolution_reason=reason)
        issue["last_review"].update(effective_status=status, effective_reason=reason,
                                   iteration=int(state.get("iteration", 0)))
    by_id = {row["id"]: row for row in ledger}
    by_signature = {signature(row): row for row in ledger if not row["resolved"]}
    for original in result["issues"]:
        update = deepcopy(original)
        ident = text(update.get("id"))
        existing = by_id.get(ident) or by_signature.get(signature(update))
        if existing and not existing["resolved"]:
            # 旧问题更新内容但保留身份、Writer状态和复查轨迹。
            update.pop("id", None)
            update.pop("resolved", None)
            update.pop("review_status", None)
            update.pop("writer_status", None)
            update.pop("last_review", None)
            update.pop("resolution_reason", None)
            existing.update(update)
            existing.update(resolved=False, review_status="unresolved")
            continue
        if not ident or ident in by_id:
            ident = f"issue_{signature(update)}"
        while ident in by_id:
            ident += "_new"
        update.update(id=ident, resolved=False, review_status="pending", writer_status="pending")
        ledger.append(update)
        by_id[ident] = update
        by_signature[signature(update)] = update
    return ledger, [row for row in ledger if not row["resolved"]]
