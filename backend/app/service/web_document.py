"""正文准入、稳定分句索引与原句回读；不把定位成功当成语义支持。"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

BODY_VERSION = "numbered-body-v1"
# 正文准入规则可升级，原句编号协议与已保存的偏移保持独立。
QUALITY_VERSION = "body-quality-v2"
CHALLENGE_HEADING = re.compile(
    r"^(?:Are you a robot|Verify (?:that )?you are human|Human verification|"
    r"Security (?:check|verification)|Checking your browser(?: before accessing)?|"
    r"Just a moment|Attention required(?:\s*\|\s*Cloudflare)?|"
    r"人机验证|安全验证|验证码验证|请完成人机验证|验证您是真人)[?!！。.\s]*$", re.I)
CHALLENGE_INSTRUCTION = re.compile(
    r"^(?:(?:Please\s+)?(?:confirm|verify|prove)\s+(?:that\s+)?you\s+are\s+(?:a\s+)?human\b|"
    r"(?:Please\s+)?complete\s+(?:the\s+)?(?:captcha|security check|human verification)\b|"
    r"To continue[,\s]+(?:please\s+)?verify\b|"
    r"请(?:先)?(?:完成|通过)(?:以下|下方|本次)?(?:的)?(?:验证码|安全验证|人机验证)|"
    r"请(?:确认|验证)(?:您|你)(?:是(?:真人|人类)|不是机器人))", re.I)
JS_FEATURES = (
    re.compile(r"\bfunction(?:\s+[\w$]+)?\s*\([^)]{0,150}\)\s*\{"),
    re.compile(r"\b(?:window|document|navigator)\s*\.\s*[\w$]+"),
    re.compile(r"\b(?:var|let|const)\s+[\w$]+\s*="),
    re.compile(r"\btypeof\s+[\w$]+"),
    re.compile(r"=>\s*\{"),
)
RECOMMENDATION = re.compile(r"^(?:日阅读\s*TOP\s*\d+|热门文档|推荐文档|相关推荐|猜你喜欢|相关阅读|Recommended\s+(?:articles|documents)|Related\s+articles)\s*[:：]?$", re.I)
ERROR_PAGE = re.compile(r"^(?:Access denied|(?:403\s+)?Forbidden|(?:404\s+)?(?:Page\s+)?not found|Just a moment|Verify you are human|Sign in|Log in|访问被拒绝|页面不存在|安全验证|人机验证|请先登录|登录后查看)[.!。！\s]*$", re.I)
PREVIEW = re.compile(r"(?:仅供预览|剩余.{0,12}(?:付费|下载|登录)|登录后(?:阅读|查看)|购买后(?:阅读|查看)|试读结束|preview only|subscribe to (?:read|continue))", re.I)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _abnormal_body_signals(lines: list[str], title: str) -> dict:
    """组合信号识别验证流程及压缩脚本；技术文章的代码块不等于污染。"""
    heading = bool(CHALLENGE_HEADING.fullmatch(title.strip()))
    instruction = False
    script_chars = unfenced_script_chars = article_chars = 0
    script_lines = []
    fence = ""
    for i, line in enumerate(lines):
        marker = re.match(r"^(`{3,}|~{3,})", line)
        if marker:
            if not fence:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = ""
            continue
        plain = re.sub(r"^[#*\s]+|[*\s]+$", "", line)
        if i < 20 and not fence:
            heading = heading or bool(CHALLENGE_HEADING.fullmatch(plain))
            instruction = instruction or bool(CHALLENGE_INSTRUCTION.search(plain))
        # 长度只用于选择检查对象；必须同时命中多种可执行脚本语法。
        counts = [len(pattern.findall(line)) for pattern in JS_FEATURES] if len(line) >= 1500 else []
        punctuation = sum(line.count(char) for char in "{}();=") / max(1, len(line))
        scripted = bool(counts and sum(counts) >= 6 and sum(c > 0 for c in counts) >= 2
                        and punctuation >= 0.04)
        if scripted:
            script_chars += len(line)
            script_lines.append(i + 1)
            if not fence:
                unfenced_script_chars += len(line)
            continue
        if (fence or line.startswith(("#", ">", "[", "!", "*", "|"))
                or CHALLENGE_INSTRUCTION.search(plain) or CHALLENGE_HEADING.fullmatch(plain)
                or re.match(r"^(?:Verification successful|Waiting for|Reference number|IP Address|"
                            r"User Agent|Timestamp|Cookie|Privacy|Terms)\b", plain, re.I)):
            continue
        if len(line) >= 100 and (len(re.findall(r"[A-Za-z]{2,}", line)) >= 12
                                or len(re.findall(r"[\u4e00-\u9fff]", line)) >= 40):
            article_chars += len(line)
    return {"challenge_heading": heading, "challenge_instruction": instruction,
            "article_chars": article_chars, "script_chars": script_chars,
            "unfenced_script_chars": unfenced_script_chars, "script_lines": script_lines}


def assess_body(text: str, title: str = "") -> tuple[str, dict]:
    """明确问题拒绝；混合或不完整正文留给模型确认。单个关键词不拒绝整页。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    signals = _abnormal_body_signals(lines, title)
    script_lines = set(signals["script_lines"])
    plain_lines = [re.sub(r"https?://\S+", "", re.sub(r"!?\[([^\]]*)\]\([^)]+\)", r"\1", line)) for line in lines]
    narrative = sum(len(line) for i, line in enumerate(plain_lines, 1) if i not in script_lines
        and (len(line) >= 60 or re.search(
            r"[。！？!?]|[a-zA-Z]\.(?:\s|$)", re.sub(r"^\d+[.)、]\s*", "", line))))
    metrics = {"chars": len(text), "lines": len(lines),
               "narrative_ratio": round(narrative / max(1, sum(map(len, plain_lines))), 3),
               "listing_metadata_count": sum(bool(re.match(r"^(?:上传人|上传时间|下载次数|阅读次数|页数)\s*[:：]", line)) for line in lines)}
    quality = {"version": BODY_VERSION, "quality_version": QUALITY_VERSION,
               "content_sha256": content_hash(text), "status": "passed",
               "reasons": [], "metrics": metrics, "signals": signals}
    if not text.strip():
        quality.update(status="rejected", reasons=["empty_body"])
        return text, quality
    has_article = signals["article_chars"] >= 600
    if signals["challenge_heading"] or signals["challenge_instruction"]:
        if not has_article:
            quality.update(status="rejected", reasons=["captcha_challenge"])
            return text, quality
        if signals["challenge_instruction"]:
            quality.update(status="uncertain", reasons=["mixed_challenge_content"])
    if signals["script_chars"] >= max(1500, len(text) * 0.5) and not has_article:
        quality.update(status="rejected", reasons=quality["reasons"] + ["script_body"])
        return text, quality
    if signals["unfenced_script_chars"]:
        quality.update(status="uncertain", reasons=quality["reasons"] + ["script_contamination"])
    if ERROR_PAGE.fullmatch(title.strip()) or (
        len(text) < 2000 and any(ERROR_PAGE.fullmatch(line) for line in lines[:5])
        and metrics["narrative_ratio"] < 0.5
    ):
        quality.update(status="rejected", reasons=["error_or_login_page"])
        return text, quality
    for i, line in enumerate(lines):
        if RECOMMENDATION.fullmatch(re.sub(r"^[\s*#\-•]+|[\s*#]+$", "", line)):
            # 能分离的尾部推荐模块剔除；正文没有明确边界时不猜测。
            prefix = "\n".join(lines[:i])
            if len(prefix) >= 200 and re.search(r"[。！？.!?]", prefix):
                quality["reasons"].append("recommendation_tail_removed")
                quality["raw_content_sha256"] = content_hash(text)
                return assess_cleaned(prefix, quality)
            if metrics["narrative_ratio"] < 0.35 or (i < 10 and metrics["listing_metadata_count"] >= 3):
                quality.update(status="rejected", reasons=["recommendation_list"])
            else:
                quality.update(status="uncertain", reasons=["mixed_recommendation_content"])
            return text, quality
    if PREVIEW.search(text):
        quality.update(status="uncertain", reasons=["preview_or_paywall"])
    elif (len(lines) >= 8 and metrics["narrative_ratio"] < 0.2
          and not signals["script_chars"] and not any("|" in line for line in lines)):
        quality.update(status="uncertain", reasons=["title_list_or_navigation"])
    return text, quality


def assess_cleaned(text: str, previous: dict) -> tuple[str, dict]:
    cleaned, quality = assess_body(text)
    quality["reasons"] = previous["reasons"] + quality["reasons"]
    quality["raw_content_sha256"] = previous["raw_content_sha256"]
    return cleaned, quality


def split_sentences(text: str) -> list[dict]:
    """保留原文偏移；表格逐行编号，英文小数、缩写和 URL 不按点号拆碎。"""
    result, offset = [], 0
    boundary = re.compile(r'[。！？!?]+[”’」』"\']*|\.(?=\s+[A-Z0-9“"\'])')
    for block_no, line in enumerate(text.splitlines(keepends=True), 1):
        start = 0
        urls = [(match.start(), match.start() + len(match.group().rstrip("。！？.!?")))
                for match in re.finditer(r'https?://[^\s<>"\'。！？]+', line)]
        ends = [] if "|" in line or line.lstrip().startswith("<table") else [match.end() for match in boundary.finditer(line)
            if not any(left <= match.start() < right for left, right in urls)
            and not (match.group() == "." and re.search(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Fig|No|vs|e\.g|i\.e|U\.S)$", line[:match.start()], re.I))]
        for end in [*ends, len(line)]:
            left, right = start, end
            while left < right and line[left].isspace():
                left += 1
            while right > left and line[right - 1].isspace():
                right -= 1
            if right > left:
                result.append({"id": f"s{len(result) + 1:06d}", "block_id": f"p{block_no:06d}",
                               "start": offset + left, "end": offset + right})
            start = end
        offset += len(line)
    return result


def index_document(text: str, url: str) -> dict:
    return {"document_version": BODY_VERSION,
            "document_id": "doc_" + content_hash(url + "\n" + text)[:24],
            "content_sha256": content_hash(text), "sentence_index": split_sentences(text)}


def sentence_records(source: Mapping[str, Any]) -> list[dict]:
    text = str(source.get("content") or "")
    if source.get("document_version") != BODY_VERSION or source.get("content_sha256") != content_hash(text):
        return []
    url = str(source.get("url") or source.get("source_url") or "")
    if source.get("document_id") != "doc_" + content_hash(url + "\n" + text)[:24]:
        return []
    index = source.get("sentence_index")
    if not isinstance(index, list) or index != split_sentences(text):
        return []
    return [{**item, "text": text[item["start"]:item["end"]]} for item in index]


def source_available(source: Mapping[str, Any]) -> bool:
    """已入库的来源引用沿用准入元数据，正文仅由RAG按需提供。"""
    if source.get("evidence_document_ref"):
        return bool(source.get("document_id") and source.get("content_sha256")
                    and source.get("rag_usable") is True)
    return source_usable(source)


def source_usable(source: Mapping[str, Any]) -> bool:
    quality = source.get("body_quality")
    if source.get("document_version") and not isinstance(quality, dict):
        return False
    if isinstance(quality, dict) and quality.get("status") not in {"passed", "partial"}:
        return False
    text = str(source.get("content") or "")
    if (not isinstance(quality, dict) or quality.get("quality_version") != QUALITY_VERSION
            or quality.get("content_sha256") != content_hash(text)):
        # 旧检查点的passed不能绕过新准入。只校验，不改原件及原句偏移。
        title = (quality.get("page_title") if isinstance(quality, dict) else "") or source.get("title") or ""
        cleaned, fresh = assess_body(text, str(title))
        if cleaned != text or fresh["status"] not in {"passed", "partial"}:
            return False
    return bool(text.strip())


def resolve_sentences(source: Mapping[str, Any], document_id: Any, ids: Any) -> list[dict]:
    if not source_usable(source) or document_id != source.get("document_id"):
        return []
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        return []
    rows = sentence_records(source)
    by_id = {row["id"]: row for row in rows}
    if any(i not in by_id for i in ids):
        return []
    return sorted((by_id[i] for i in ids), key=lambda row: row["start"])


def quote_matches_sentences(source: Mapping[str, Any], document_id: Any, ids: Any, quote: str) -> bool:
    """允许连续支持句中的原文引用，禁止跨未选择原句拼接。"""
    rows = resolve_sentences(source, document_id, ids)
    if not rows or not quote:
        return False
    groups = []
    for row in rows:
        if groups and int(row["id"][1:]) == int(groups[-1][-1]["id"][1:]) + 1:
            groups[-1].append(row)
        else:
            groups.append([row])
    text = str(source.get("content") or "")
    return any(quote in text[group[0]["start"]:group[-1]["end"]] for group in groups)


def format_numbered_body(item: dict, query: str = "", max_chars: int = 12000) -> str:
    """优先相关句及相邻上下文，完整句边界截取；不默默只取文档开头。"""
    rows = sentence_records({**item, "content": item.get("full_content", "")})
    terms = set(re.findall(r"[a-zA-Z0-9]{2,}|[\u4e00-\u9fff]{2}", query.lower()))
    rank = sorted(range(len(rows)), key=lambda i: (-sum(term in rows[i]["text"].lower() for term in terms), i))
    selected, used = set(), 0
    for i in rank:
        window = [j for j in (i - 1, i, i + 1) if 0 <= j < len(rows) and j not in selected]
        # 核心句先选，随后补相邻上下文。
        for j in sorted(window, key=lambda j: (j != i, j)):
            cost = len(rows[j]["text"]) + 12
            if used + cost <= max_chars:
                selected.add(j)
                used += cost
    visible = [rows[i] for i in sorted(selected)]
    item["visible_sentence_ids"] = [row["id"] for row in visible]
    return (f"正文编号: {item.get('document_id', '未编号')}；质量: {item.get('body_quality', {}).get('status', 'unknown')}\n"
            + "\n".join(f"[{row['id']}] {row['text']}" for row in visible)
            + ("\n（其余原句未在本次输入展示，不得引用其编号。）" if len(visible) < len(rows) else ""))
