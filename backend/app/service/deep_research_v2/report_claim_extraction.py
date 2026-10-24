"""独立读取最终报告的论断、正文位置和实际引用；不接收 Writer 声明。"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from service.context_policy import NLISettings
from .evidence import normalize_url

VERSION = "final-report-claims-v1"
LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\((https?://[^\s)]+)(?:\s+[^)]*)?\)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
REFERENCES = re.compile(r"^(?:[\d.、\s]+)?(?:参考文献|参考资料|参考来源|引用来源|references|bibliography|sources)$", re.I)


def report_units(report: str) -> list[dict]:
    """覆盖全文段落、列表和表格行；表头作上下文，参考文献区只作引用目录。"""
    units, paragraph = [], []
    heading, reference_depth, table_header, offset = "", None, "", 0

    def flush():
        if paragraph:
            start, _ = paragraph[0]
            text = "".join(line for _, line in paragraph).rstrip()
            units.append({"start": start, "end": start + len(text), "text": text,
                          "heading": heading, "table_header": ""})
            paragraph.clear()

    lines = report.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        match = HEADING.match(stripped)
        if match:
            flush()
            depth = len(match[1])
            if reference_depth is not None and depth <= reference_depth:
                reference_depth = None
            if REFERENCES.fullmatch(match[2].strip()):
                reference_depth = depth
            heading, table_header = match[2], ""
            if reference_depth is None:
                left = offset + len(line) - len(line.lstrip())
                units.append({"start": left, "end": left + len(stripped), "text": stripped,
                              "heading": heading, "table_header": ""})
        elif reference_depth is not None or re.match(r"^\s*\[[^\]]+\]:\s*https?://", line):
            flush()
        elif not stripped or re.fullmatch(r"[-*_]{3,}", stripped):
            flush()
            table_header = ""
        elif re.match(r"^\|[\s:|\-]+\|$", stripped):
            flush()
        elif stripped.startswith("|"):
            flush()
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if re.match(r"^\|[\s:|\-]+\|$", next_line):
                table_header = stripped
            else:
                left = offset + len(line) - len(line.lstrip())
                units.append({"start": left, "end": left + len(stripped), "text": stripped,
                              "heading": heading, "table_header": table_header})
        elif re.match(r"^\s*(?:[-*+]\s|\d+[.)]\s)", line):
            flush()
            left = offset + len(line) - len(line.lstrip())
            units.append({"start": left, "end": left + len(stripped), "text": stripped,
                          "heading": heading, "table_header": ""})
        else:
            paragraph.append((offset, line))
        offset += len(line)
    flush()
    for index, unit in enumerate(units):
        unit["unit_id"] = f"report_unit_{index + 1}"
    return units


def reference_catalog(report: str) -> dict[str, str]:
    result = {m[1].strip().casefold(): m[2] for m in re.finditer(
        r"(?m)^\s*\[([^\]]+)\]:\s*(https?://\S+)", report)}
    for line in report.splitlines():
        number = re.match(r"\s*(?:\[(\d+)\]|(\d+)[.)])\s*", line)
        link = LINK.search(line)
        if number and link:
            result[number[1] or number[2]] = link[1]
    return result


def citation_urls(quote: str, catalog: dict[str, str]) -> tuple[list[str], list[str]]:
    urls = [m[1] for m in LINK.finditer(quote)]
    remaining = LINK.sub("", quote)
    errors = []
    for match in re.finditer(r"\[[^\]]+\]\[([^\]]+)\]|\[(\d+(?:\s*[,，]\s*\d+)*)\]", remaining):
        keys = [match[1]] if match[1] else re.split(r"\s*[,，]\s*", match[2])
        for key in keys:
            url = catalog.get(key.strip().casefold())
            if url:
                urls.append(url)
            else:
                errors.append("unresolved_report_citation")
    return list(dict.fromkeys(normalize_url(url) for url in urls)), errors


def extraction_messages(units: list[dict]) -> list[dict]:
    prompt = (Path(__file__).parents[2] / "prompts/最终报告独立论断提取.txt").read_text()
    return [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps({
        "current_date": datetime.now().strftime("%Y-%m-%d"), "units": units}, ensure_ascii=False)}]


async def call_report_model(messages, settings: NLISettings, *, component: str) -> tuple[str, dict]:
    """抽取与证据句选择共用无历史调用，并分别登记冻结预算。"""
    from config.llm_config import LLMConfig
    from service.llm_client import create_async_openai_client
    from service.qwen_token_counter import get_qwen_token_counter
    from service.verification_budget import current_budget
    if settings.model_id != "qwen3-max":
        from service.nli_verifier import RetiredNLIBackendError
        raise RetiredNLIBackendError("独立报告抽取要求 qwen3-max，请为旧运行创建新核验配置")
    config = LLMConfig()
    if not config.api_key:
        raise ValueError("报告论断提取缺少DASHSCOPE_API_KEY")
    ledger = current_budget()
    estimated = get_qwen_token_counter().count_messages(messages)
    ticket = ledger.reserve(settings.model_id, estimated, settings.output_tokens) if ledger else None
    if ticket is not None:
        ticket["component"] = component
    async with create_async_openai_client(api_key=config.api_key, base_url=config.base_url,
            read_timeout=settings.timeout_seconds, max_retries=0) as client:
        response = await client.chat.completions.create(model=settings.model_id, messages=messages,
            temperature=0, max_tokens=settings.output_tokens, response_format={"type": "json_object"},
            extra_body={"enable_thinking": False}, timeout=settings.timeout_seconds)
    usage = response.usage.model_dump() if response.usage else {}
    if ledger:
        ledger.complete(ticket, usage)
    if response.choices[0].finish_reason != "stop":
        raise ValueError("报告论断提取输出未完整结束")
    return response.choices[0].message.content, {"model": settings.model_id, "usage": usage,
        "request_id": response.id, "estimated_input_tokens": estimated, "component": component}


async def call_extractor(messages, settings: NLISettings) -> tuple[str, dict]:
    return await call_report_model(messages, settings, component="report_claim_extraction")


def parse_extraction(content: str, units: list[dict], state: dict) -> tuple[list[dict], list[dict]]:
    value = json.loads(content)
    rows = value.get("units") if isinstance(value, dict) else None
    expected = {unit["unit_id"]: unit for unit in units}
    if (not isinstance(rows, list) or len(rows) != len(units) or
            any(not isinstance(row, dict) for row in rows)):
        raise ValueError("论断提取必须逐项覆盖全部报告单元")
    indexed = {}
    for row in rows:
        ident = row.get("unit_id")
        if not isinstance(ident, str) or ident not in expected or ident in indexed:
            raise ValueError("论断提取单元编号未知或重复")
        if row.get("status") not in {"claims", "no_external_claims", "uncertain"}:
            raise ValueError("论断提取单元状态无效")
        if not isinstance(row.get("claims"), list) or not isinstance(row.get("reason"), str):
            raise ValueError("论断提取缺少claims或reason")
        if (row["status"] == "claims") != bool(row["claims"]):
            raise ValueError("论断提取状态与claims不一致")
        if row["status"] != "claims" and not row["reason"].strip():
            raise ValueError("跳过或不确定单元必须说明理由")
        indexed[ident] = row
    claims, audit = [], []
    catalog = reference_catalog(state["final_report"])
    for ident, unit in expected.items():
        row = indexed[ident]
        audit.append({"unit_id": ident, "start": unit["start"], "end": unit["end"],
                      "status": row["status"], "reason": row["reason"], "claims_count": len(row["claims"])})
        for original in row["claims"]:
            if not isinstance(original, dict):
                raise ValueError("提取论断必须为对象")
            claim, quote = original.get("claim"), original.get("report_quote")
            if (not isinstance(claim, str) or not claim.strip() or not isinstance(quote, str) or
                    not quote.strip() or unit["text"].count(quote) != 1 or
                    original.get("assertion_kind") not in {"fact", "forecast"}):
                raise ValueError("提取论断缺少有效文本、原文定位或事实/预测属性")
            start = unit["start"] + unit["text"].index(quote)
            # 句末紧邻的行内引用也属于该句；不借用同段其他句子的链接。
            tail = unit["text"][unit["text"].index(quote) + len(quote):]
            suffix = re.match(r"^\s*((?:(?:\[[^\]\n]+\]\(https?://[^)]+\)|\[[^\]\n]+\]\[[^\]\n]+\]|\[\d+(?:\s*[,，]\s*\d+)*\])\s*)+)", tail)
            if suffix:
                quote += suffix[0]
            urls, errors = citation_urls(quote, catalog)
            if not urls and not errors:
                errors.append("missing_inline_citation")
            sections = [key for key, text in (state.get("draft_sections") or {}).items() if quote in text]
            section = sections[0] if len(sections) == 1 else "__synthesis__"
            claims.append({"claim_id": f"report_claim_{start}_{len(claims)}", "claim": claim.strip(),
                "report_quote": quote, "report_start": start, "report_end": start + len(quote),
                "section_id": section, "unit_id": ident, "assertion_kind": original["assertion_kind"],
                "citation_urls": urls, "binding_errors": errors, "evidence": []})
    return claims, audit


async def extract_report_claims(state: dict, settings: NLISettings, *, call=None, counter=None) -> dict:
    """按完整报告单元分批，缺单元/输出截断直接失败，不回退到 Writer 声明。"""
    from service.qwen_token_counter import get_qwen_token_counter
    report = str(state.get("final_report") or "")
    if not report.strip():
        raise ValueError("独立论断提取要求非空最终报告")
    units = report_units(report)
    counter = counter or get_qwen_token_counter()
    limit = min(16000, settings.max_length, 262144 - settings.output_tokens - settings.safety_tokens)
    batches, batch = [], []
    for unit in units:
        if batch and (len(batch) >= 8 or counter.count_messages(extraction_messages([*batch, unit])) > limit):
            batches.append(batch)
            batch = []
        if counter.count_messages(extraction_messages([unit])) > limit:
            raise ValueError("单个报告单元超过抽取预算，未截断原文")
        batch.append(unit)
    if batch:
        batches.append(batch)
    claims, coverage, calls = [], [], []
    for batch in batches:
        content, metadata = await (call or call_extractor)(extraction_messages(batch), settings)
        extracted, audit = parse_extraction(content, batch, state)
        claims.extend(extracted)
        coverage.extend(audit)
        calls.append(metadata)
    # 跨批次编号也须唯一。
    for index, claim in enumerate(claims):
        claim["claim_id"] = f"report_claim_{index + 1}"
    return {"version": VERSION, "report_sha256": hashlib.sha256(report.encode()).hexdigest(),
            "status": "completed" if units and all(row["status"] != "uncertain" for row in coverage) else "incomplete",
            "scope": "final_report_extracted_claims", "units_total": len(units), "coverage": coverage,
            "claims": claims, "api_calls": calls, "all_claims_extracted": False,
            "limitation": "单元已逐项处理；语义抽取仍可能遗漏论断，不表示召回率为100%"}
