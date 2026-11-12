"""审核 JSON 的当前契约：三维百分制、真实章节定位和新增章节执行。"""

import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

QUALITY_SCORE_BASIS = "report_review_3d_100_v1"
DIMENSIONS = ("readability", "comprehensiveness", "depth")
ISSUE_TYPES = {"missing_source", "logical_error", "bias", "outdated", "content_omission",
               "insufficient_depth", "readability_issue", "instruction_violation"}
SEVERITIES = {"critical", "major", "minor"}
TARGET_FIELDS = {"target_type", "chapter_id", "chapter_title", "specific_location",
                 "insert_after_chapter_id", "proposed_chapter_title"}


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def normalize_quality_score(value: Any):
    """只接受有限的 0—100 数字；0 是合法评分，字符串和布尔值不是评分。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and 0 <= value <= 100 else None


def validate_critic_issues(result: Any) -> None:
    """拒绝旧字段及缺失的路由标记，不从描述文字猜测搜索或修改目标。"""
    if not isinstance(result, dict) or not isinstance(result.get("issues"), list):
        raise ValueError("Critic 审核结果必须包含 issues 数组")
    seen = set()
    for issue in result["issues"]:
        if not isinstance(issue, dict) or not nonempty(issue.get("issue_id")):
            raise ValueError("每条审核问题必须提供 issue_id")
        if issue["issue_id"] in seen:
            raise ValueError("审核问题 issue_id 不能重复")
        seen.add(issue["issue_id"])
        if issue.get("issue_type") not in ISSUE_TYPES or issue.get("severity") not in SEVERITIES:
            raise ValueError("审核问题的 issue_type 或 severity 无效")
        for name in ("description", "evidence", "modification"):
            if not nonempty(issue.get(name)):
                raise ValueError(f"审核问题必须提供非空 {name}")
        if type(issue.get("requires_new_search")) is not bool:
            raise ValueError("requires_new_search 必须是布尔值")
        query = issue.get("search_query")
        if issue["requires_new_search"]:
            if not nonempty(query):
                raise ValueError("要求补搜时必须提供非空 search_query")
        elif query != "":
            raise ValueError("不补搜时 search_query 必须为空字符串")
        target = issue.get("target")
        if not isinstance(target, dict) or set(target) != TARGET_FIELDS:
            raise ValueError("target 必须包含当前契约的六个定位字段")
        for name in ("chapter_id", "insert_after_chapter_id"):
            if target[name] is not None and not nonempty(target[name]):
                raise ValueError(f"{name} 必须是非空字符串或 null")
        if target["target_type"] == "existing_section":
            if not all(nonempty(target[name]) for name in ("chapter_title", "specific_location")):
                raise ValueError("已有章节必须提供标题和具体位置")
            if target["insert_after_chapter_id"] is not None or target["proposed_chapter_title"] is not None:
                raise ValueError("已有章节不得填写新增章节字段")
        elif target["target_type"] == "new_section":
            if not nonempty(target["proposed_chapter_title"]):
                raise ValueError("新增章节必须提供 proposed_chapter_title")
            if any(target[name] is not None for name in ("chapter_id", "chapter_title", "specific_location")):
                raise ValueError("新增章节的原章节字段必须为 null")
        else:
            raise ValueError("target_type 必须是 existing_section 或 new_section")


def normalize_review_result(result: dict) -> None:
    """程序计算三维均分及结论，不按严重程度压分，也不信任模型自报 pass。"""
    assessment = result.get("overall_assessment")
    if not isinstance(assessment, dict) or not isinstance(assessment.get("dimension_scores"), dict):
        raise ValueError("审核缺少三个维度的数字评分")
    dimensions = assessment["dimension_scores"]
    if set(dimensions) != set(DIMENSIONS) or any(
        normalize_quality_score(dimensions[name]) is None for name in DIMENSIONS
    ):
        raise ValueError("审核必须提供三个 0—100 的有限数字评分，不能使用旧评分维度")
    if not isinstance(assessment.get("summary"), str) or not isinstance(result.get("strength_points"), list):
        raise ValueError("审核必须提供 summary 字符串和 strength_points 数组")
    score = float((sum(Decimal(str(dimensions[name])) for name in DIMENSIONS) / Decimal(len(DIMENSIONS)))
                  .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    assessment.setdefault("raw_quality_score", assessment.get("quality_score"))
    assessment.setdefault("raw_verdict", assessment.get("verdict"))
    assessment["quality_score"] = score
    assessment["score_status"] = "derived_from_dimensions"
    severities = {issue["severity"] for issue in result["issues"]}
    if score >= 60 and not severities & {"critical", "major"}:
        assessment["verdict"] = "pass"
    elif score < 60 or "critical" in severities:
        assessment["verdict"] = "major_issues"
    else:
        assessment["verdict"] = "needs_revision"
    result["issues"].sort(key=lambda issue: ("critical", "major", "minor").index(issue["severity"]))


def title_text(title: str) -> str:
    """去掉 Markdown 标记和显示编号，仅用于对照同一个真实标题。"""
    title = re.sub(r"^[#\s]+|[#\s]+$", "", title).replace("**", "")
    return re.sub(r"^(?:\d+(?:\.\d+)*[.、]?\s+|第[一二三四五六七八九十百\d]+[章节]\s*)", "", title).strip()


def report_sections(state: dict) -> list[dict]:
    """从实际报告标题建定位表，将子节关联到 Writer 持有的正文或首尾章节。"""
    outline = state.get("outline") or []
    rows, parent, fence = [], None, None
    for line in state["final_report"].splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker[1][0]
            fence = None if fence == token else (token if fence is None else fence)
            continue
        if fence:
            continue
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        heading = match[2]
        title = title_text(heading)
        matches = [s["id"] for s in outline if title_text(s["title"]) == title]
        number = re.match(r"^(\d+(?:\.\d+)*)[.、]?(?:\s|$)", heading)
        if len(matches) > 1 and number and "." not in number[1]:
            index = int(number[1]) - 1
            matches = [outline[index]["id"]] if 0 <= index < len(outline) and outline[index]["id"] in matches else matches
        if len(matches) == 1:
            parent = matches[0]
        elif title in {"执行摘要", "结论与展望", "参考文献"}:
            parent = "__synthesis__"
        elif len(match[1]) <= 2:
            parent = None
        rows.append({"heading": heading, "title": title,
                     "number": number[1] if number else None, "section_id": parent})
    return rows


def bind_issue_targets(state: dict, issues: list[dict]) -> None:
    """先在未变动的报告上校验所有目标，避免插入章节后旧编号发生漂移。"""
    rows = report_sections(state)
    known = {section["id"] for section in state.get("outline") or []}
    for issue in issues:
        target = issue["target"]
        if issue.get("rule_based") and target["chapter_title"] == "全文":
            issue["resolved_section_ids"] = [s["id"] for s in state.get("outline") or []] + ["__synthesis__"]
            continue
        if target["target_type"] == "new_section":
            anchor = target["insert_after_chapter_id"]
            if anchor is None:
                # 未编号报告无法给出编号时，新增章节放在正文末尾、结论之前。
                ident = (state.get("outline") or [{}])[-1].get("id")
            else:
                candidates = {row["section_id"] for row in rows if
                              row["number"] == anchor or row["section_id"] == anchor}
                if len(candidates) != 1 or not candidates <= known:
                    raise ValueError(f"新增章节的插入位置不存在或不唯一: {anchor}")
                ident = candidates.pop()
            issue["insert_after_section_id"] = ident
            issue["resolved_section_ids"] = []
            continue
        title, chapter_id = title_text(target["chapter_title"]), target["chapter_id"]
        candidates = [row for row in rows if row["title"] == title]
        if chapter_id is not None:
            candidates = [row for row in candidates if chapter_id in {row["number"], row["section_id"]}]
        ids = {row["section_id"] for row in candidates}
        if len(ids) != 1 or None in ids:
            raise ValueError(f"审核章节无法唯一定位到报告: {chapter_id} / {target['chapter_title']}")
        issue["resolved_section_ids"] = sorted(ids)


def materialize_new_sections(state: dict) -> None:
    """程序把新增章节意见加入内存中的研究大纲，供 Scout 绑定证据和 Writer 撰写。"""
    outline = state["outline"]
    anchors = {}
    for index, issue in enumerate(state["critic_feedback"], 1):
        target = issue["target"]
        if target["target_type"] != "new_section":
            continue
        title = target["proposed_chapter_title"].strip()
        matching = [s for s in outline if title_text(s["title"]) == title_text(title)]
        if len(matching) > 1:
            raise ValueError(f"新增章节标题不唯一: {title}")
        if matching:
            section = matching[0]
            # 大纲已规划但尚未写出的章节可直接补写，不能复制已有正文为新章节。
            if (state.get("draft_sections") or {}).get(section["id"]):
                raise ValueError(f"新增章节已存在正文，请使用 existing_section: {title}")
            section["description"] = section.get("description", "") + "\n" + issue["modification"]
            section["status"] = "pending"
        else:
            ident = f"sec_review_{state.get('iteration', 0)}_{index}"
            if any(s["id"] == ident for s in outline):
                raise ValueError(f"新增章节ID重复: {ident}")
            section = {"id": ident, "title": title, "description": issue["modification"],
                       "section_type": "mixed", "status": "pending", "subsections": [],
                       "requires_data": False, "requires_chart": False}
            anchor = issue["insert_after_section_id"]
            after = anchors.get(anchor, anchor)
            position = next((i + 1 for i, s in enumerate(outline) if s["id"] == after), len(outline))
            outline.insert(position, section)
            anchors[anchor] = ident
        issue["resolved_section_ids"] = [section["id"]]
