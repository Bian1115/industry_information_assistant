#!/usr/bin/env python3
"""只读分析真实会话实验落盘记录，分开内部失败、SDK错误和整轮终态。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


VERSION = "容量与错误分层-v1.1"
TERMINAL_FAILURES = {"failed", "clarified", "clarification", "cancelled", "paused", "interrupted"}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def numeric(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def maximum(values):
    known = [value for value in values if numeric(value) is not None]
    return max(known) if known else None


def safe_error(value):
    text = str(value or "")
    text = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [已隐去]", text)
    text = re.sub(r"(?i)(api[_-]?key|authorization|password|secret|access[_-]?token)([\"']?\s*[:=]\s*[\"']?)[^\s\"',;}]+", r"\1\2[已隐去]", text)
    return text[:2000]


def error_kind(message, error_type=""):
    text = (str(error_type) + " " + str(message)).lower()
    if "contextbudgetexceeded" in text or "context_budget_exceeded" in text:
        return "local_context_budget"
    if "历史回读次数超限" in text:
        return "history_read_limit"
    if re.search(r"context[_ -]?(?:length|window)|maximum context|range of input length|input (?:length|tokens).*(?:exceed|limit)|exceed.*(?:input|token)|too many tokens|上下文.*(?:超|过长)|输入.*(?:超|过长|硬预算)|max_input", text):
        return "provider_context_limit"
    if any(term in text for term in ("整任务", "整条对话", "调用预算耗尽", "模型token预算耗尽")):
        return "experiment_budget"
    if "阶段超过" in text or "stagebudgetexceeded" in text:
        return "stage_budget"
    if re.search(r"研究任务超过\s*\d+\s*s", text):
        return "research_time_budget"
    if any(term in text for term in ("rate limit", "ratelimiterror", "429", "too many requests", "限流")):
        return "rate_limit"
    if any(term in text for term in ("timeout", "timed out", "超时", "时限耗尽")):
        return "timeout"
    if any(term in text for term in ("finish_reason=length", "finish_reason: length", "输出被截断")):
        return "output_truncated"
    return "other_error"


class Reader:
    """不访问服务和配置环境；并发写入导致的半行/半文件作为观测缺口记录。"""

    def __init__(self, root):
        self.root, self.sources, self.issues = root, {}, []

    def relative(self, path):
        return str(path.relative_to(self.root))

    def issue(self, path, reason, **extra):
        self.issues.append({"path": self.relative(path), "reason": reason, **extra})

    def record(self, path, digest, before, after):
        changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        self.sources[self.relative(path)] = {"sha256_read": digest, "bytes_before": before.st_size,
            "bytes_after": after.st_size, "modified_at_ns": after.st_mtime_ns,
            "changed_during_read": changed}
        if changed:
            self.issue(path, "changed_during_read")

    def json(self, name, default=None):
        path = self.root / name
        try:
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            self.record(path, hashlib.sha256(data).hexdigest(), before, after)
            return json.loads(data)
        except FileNotFoundError:
            return default
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.issue(path, type(exc).__name__)
            return default

    def json_lines(self, name):
        path = self.root / name
        digest = hashlib.sha256()
        try:
            before = path.stat()
            with path.open("rb") as stream:
                for number, line in enumerate(stream, start=1):
                    digest.update(line)
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except (UnicodeError, json.JSONDecodeError):
                        self.issue(path, "invalid_or_incomplete_jsonl_line", line=number)
                        continue
                    if isinstance(value, dict):
                        yield value
            self.record(path, digest.hexdigest(), before, path.stat())
        except FileNotFoundError:
            return
        except OSError as exc:
            self.issue(path, type(exc).__name__)


def dictionaries(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def request_view(row):
    fields = ("request_id", "turn_id", "run_id", "session_id", "model", "started_at", "ended_at",
        "status", "input_tokens_estimated", "output_tokens_reserved", "sdk_max_retries",
        "http_attempts_reserved", "actual_http_attempts", "finish_reasons", "error_type")
    result = {key: row[key] for key in fields if key in row}
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    result["usage"] = {key: numeric(usage.get(key)) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if row.get("error"):
        result["error"] = safe_error(row["error"])
        result["error_kind"] = error_kind(row["error"], row.get("error_type", ""))
    return result


def collect_requests(reader):
    ledger = reader.json("API预算账本.json", {}) or {}
    merged = {}
    for row in dictionaries(ledger.get("requests")):
        identity = row.get("request_id")
        if identity:
            merged[identity] = request_view(row)
    for row in reader.json_lines("模型请求与用量.jsonl"):
        identity = row.get("request_id")
        if not identity:
            continue
        value = request_view(row)
        # JSONL是实时记录；旧request_start不能覆盖账本中已经完成的request_end。
        previous = merged.get(identity, {})
        if row.get("event") == "request_end" or not previous or previous.get("status") == "running":
            merged[identity] = {**previous, **value}
    return sorted(merged.values(), key=lambda row: (row.get("started_at", ""), row.get("request_id", "")))


def usage_stats(requests):
    by_model = {}
    for model in sorted({str(row.get("model", "unknown")) for row in requests}):
        selected = [row for row in requests if str(row.get("model", "unknown")) == model]
        usage = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            values = [row["usage"].get(key) for row in selected]
            known = [value for value in values if numeric(value) is not None]
            usage[key] = sum(known) if len(known) == len(values) else None
            usage["known_reported_" + key] = sum(known) if known else None
            usage["unknown_" + key + "_requests"] = len(values) - len(known)
        by_model[model] = {"requests": len(selected), "statuses": dict(Counter(row.get("status", "unknown") for row in selected)), **usage}
    actual = [row["usage"].get("prompt_tokens") for row in requests]
    return {"sdk_requests": len(requests), "statuses": dict(Counter(row.get("status", "unknown") for row in requests)),
        "max_input_tokens_estimated": maximum(row.get("input_tokens_estimated") for row in requests),
        "max_input_tokens_actual": maximum(actual), "max_actual_scope": "仅供应商已返回prompt_tokens的请求",
        "unknown_actual_input_requests": sum(numeric(value) is None for value in actual),
        "max_actual_over_all_requests": maximum(actual) if requests and all(numeric(value) is not None for value in actual) else None,
        "by_model": by_model}


def call_classification(payload):
    system, content = str(payload.get("system") or ""), str(payload.get("content") or "")
    if system.startswith("你检查网页正文质量。") and content.startswith("判断是否读到了目标文档正文"):
        return "auxiliary_web_body_quality", "system/content均匹配正文质量检查提示词"
    patterns = (
        ("primary_architect_planning", "行业研究规划师", ""),
        ("primary_scout_evidence_extraction", "", "你是一位资深的研究分析师，擅长从搜索结果中提取关键信息"),
        ("primary_scout_source_trace", "专业的信息验证专家", ""),
        ("primary_scout_supplement", "专业的信息提取专家", "正在补充搜索"),
        ("primary_data_extraction", "专业的数据分析师", ""),
        ("primary_graph_extraction", "知识图谱专家", ""),
        ("primary_writer", "擅长撰写专业的研究报告", ""),
        ("primary_writer_integration", "研究报告主编", ""),
        ("primary_critic", "质量审核专家", ""),
    )
    for category, system_marker, content_marker in patterns:
        if (not system_marker or system_marker in system) and (not content_marker or content_marker in content):
            return category, "已知业务提示词标记匹配"
    return "unclassified", "未匹配已确认提示词；不根据Agent名称假定主调用"


def deduplicate_records(records):
    unique, seen = [], set()
    for record in records:
        identity = record.get("source_id") or hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        if identity not in seen:
            seen.add(identity)
            unique.append(record)
    return unique


def call_key(record):
    return str(record.get("run_id") or ""), str(record.get("call_id") or "")


def analyze_history(records, turn_id, history_path):
    linked = defaultdict(list)
    for index, record in enumerate(records):
        if record.get("call_id"):
            linked[call_key(record)].append((index, record))
    local_errors, views = [], []
    for index, record in enumerate(records):
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        group = linked.get(call_key(record), [])
        inputs = [item for position, item in group if position <= index and item.get("kind") == "llm_input"]
        input_payload = (inputs[-1].get("payload") or {}) if inputs else {}
        category, classification_evidence = call_classification(input_payload)
        common = {"turn_id": turn_id, "run_id": record.get("run_id"), "call_id": record.get("call_id"),
            "source_id": record.get("source_id"), "agent_name": record.get("agent_name"), "node": record.get("node"),
            "seq": record.get("seq"), "created_at": record.get("created_at"), "record_index": index,
            "call_category": category, "classification_evidence": classification_evidence,
            "source_path": history_path}
        if record.get("kind") == "context_view":
            selected = {key: payload.get(key) for key in ("budget", "tokens", "tokens_before_compression",
                "tokens_after_input_history", "tokens_after_early_offload", "summary_attempted", "summary_adopted",
                "stages", "compression_enabled", "history_replay", "soft_threshold", "summary_threshold",
                "summary_version_before", "summary_version_after", "covered_seq_before", "covered_seq_after")}
            views.append({**common, **selected})
        if record.get("kind") not in {"llm_error", "llm_attempt_error", "summary_error"}:
            continue
        message = payload.get("message") or payload.get("error") or ""
        kind = error_kind(message, payload.get("error_type", ""))
        contexts = [item for position, item in group if position <= index and item.get("kind") == "context_view"]
        context = (contexts[-1].get("payload") or {}) if contexts else {}
        tokens_match = re.search(r"当前上下文\s*(\d+)\s*tokens.*输入预算\s*(\d+)\s*tokens", str(message))
        observed_input = context.get("tokens")
        budget = context.get("budget")
        if tokens_match:
            observed_input, budget = map(int, tokens_match.groups())
        continuation = any(item.get("kind") in {"llm_input", "llm_request", "llm_output", "tool_dispatch"} for item in records[index + 1:])
        local_errors.append({**common, "kind": record["kind"], "error_kind": kind,
            "error_type": payload.get("error_type"), "message": safe_error(message),
            "input_tokens": numeric(observed_input), "budget_tokens": numeric(budget),
            "has_llm_request_record": any(item.get("kind") == "llm_request" for _, item in group),
            "has_llm_usage_record": any(item.get("kind") == "llm_usage" for _, item in group),
            "subsequent_work_observed": continuation,
            "system_prompt_sha256": hashlib.sha256(str(input_payload.get("system") or "").encode()).hexdigest() if inputs else None})
    reductions = []
    for view in views:
        before, after = numeric(view.get("tokens_before_compression")), numeric(view.get("tokens"))
        if before is not None and after is not None:
            reductions.append({"call_id": view["call_id"], "agent_name": view["agent_name"], "before_tokens": before,
                "after_tokens": after, "tokens_saved": before - after, "ratio": after / before if before else None,
                "stages": view.get("stages"), "summary_attempted": view.get("summary_attempted"), "summary_adopted": view.get("summary_adopted")})
    return {"record_count": len(records), "record_kinds": dict(Counter(row.get("kind", "unknown") for row in records)),
        "local_errors": local_errors, "context_views": views,
        "compression": {"context_view_count": len(views), "summary_attempted_count": sum(view.get("summary_attempted") is True for view in views),
            "summary_adopted_count": sum(view.get("summary_adopted") is True for view in views),
            "input_reduction_count": sum(row["tokens_saved"] > 0 for row in reductions), "per_call_token_changes": reductions,
            "max_context_view_tokens": maximum(view.get("tokens") for view in views),
            "max_tokens_before_compression": maximum(view.get("tokens_before_compression") for view in views)}}


def analyze(input_dir):
    started = timestamp()
    reader = Reader(input_dir)
    manifest = reader.json("运行清单.json", {}) or {}
    raw_rows = dictionaries(reader.json("逐轮结果.json", []))
    turns = {int(row["turn_id"]): dict(row) for row in raw_rows if isinstance(row.get("turn_id"), int)}
    directories = {}
    for path in input_dir.iterdir():
        match = re.fullmatch(r"第(\d+)轮", path.name)
        if path.is_dir() and match:
            number = int(match.group(1))
            directories[number] = path
            row = reader.json(f"{path.name}/轮次结果.json") or reader.json(f"{path.name}/轮次输入.json", {}) or {}
            turns[number] = {**row, **turns.get(number, {})}
    requests = collect_requests(reader)
    request_groups = defaultdict(list)
    for request in requests:
        request_groups[request.get("turn_id")].append(request)
    analyzed, all_local, all_views, all_terminal = [], [], [], []
    for number, row in sorted(turns.items()):
        folder = directories.get(number, input_dir / f"第{number}轮")
        prefix = folder.name + "/"
        context = reader.json(prefix + "会话任务输入.json", {}) or {}
        contract = reader.json(prefix + "契约与路由.json", {}) or {}
        state = reader.json(prefix + "研究状态.json", {}) or {}
        raw_messages = dictionaries(reader.json(prefix + "会话原始消息.json", []))
        history_name = prefix + "工作历史.json"
        records = reader.json(history_name)
        if records is None:
            history_name = prefix + "会话累计工作历史.json"
            records = reader.json(history_name, [])
        records = deduplicate_records(dictionaries(records))
        if row.get("run_id"):
            records = [record for record in records if not record.get("run_id") or record["run_id"] == row["run_id"]]
        history = analyze_history(records, number, history_name)
        selected_requests = request_groups.get(number, [])
        events = list(reader.json_lines(prefix + "研究事件.jsonl"))
        terminal = []
        for event in events:
            if event.get("type") in {"error", "research_cancelled", "research_paused", "session_clarification"}:
                data = event.get("data") if isinstance(event.get("data"), dict) else event
                message = data.get("message") or data.get("content") or data.get("error") or json.dumps(data, ensure_ascii=False, default=str)
                terminal.append({"turn_id": number, "type": event["type"], "error_kind": error_kind(message),
                    "message": safe_error(message), "source_path": prefix + "研究事件.jsonl"})
        if row.get("error"):
            terminal.append({"turn_id": number, "type": "runner_turn_error", "error_kind": error_kind(row["error"], row.get("error_type", "")),
                "runner_error_kind": row.get("error_kind"), "message": safe_error(row["error"]), "source_path": "逐轮结果.json"})
        task = row.get("session_task") or context.get("session_task") or {}
        metrics = row.get("session_task_builder_metrics") or context.get("task_metrics") or {}
        whole_failed = row.get("status") in TERMINAL_FAILURES or bool(terminal) or task.get("status") == "clarify"
        sdk_errors = [request for request in selected_requests if request.get("status") == "failed"]
        value = {"turn_id": number, "run_id": row.get("run_id"), "session_id": row.get("session_id"),
            "status": row.get("status", "unknown"), "completed": row.get("completed") is True,
            "completion_finalized": row.get("completed") is True and bool(row.get("ended_at")),
            "whole_turn_failed": whole_failed, "started_at": row.get("started_at"), "ended_at": row.get("ended_at"),
            "execution_plan": row.get("execution_plan") or (row.get("delivery") or {}).get("execution_plan") or contract.get("execution_plan") or state.get("execution_plan"),
            "routing": {key: task.get(key) for key in ("status", "topic_relation", "depends_on_history", "target_artifact_ref", "execution_plan")},
            "route_fallback_reason": state.get("route_fallback_reason"),
            "parent_run_id": contract.get("parent_run_id"),
            "raw_session_message_count": len(raw_messages),
            "raw_session_message_roles": dict(Counter(message.get("role", "unknown") for message in raw_messages)),
            "active_max_body_chars": (contract.get("active_constraints") or {}).get("max_body_chars"),
            "routing_error": safe_error(metrics.get("error")) if metrics.get("error") else None,
            "history_metrics": row.get("history_metrics") or context.get("history_metrics") or {},
            "session_summary_metrics": row.get("session_summary_metrics") or context.get("session_summary_metrics") or {},
            "api_usage": usage_stats(selected_requests), "sdk_errors": sdk_errors, "terminal_errors": terminal,
            "local_error_count": len(history["local_errors"]),
            "local_context_overflows": sum(item["error_kind"] == "local_context_budget" for item in history["local_errors"]),
            "auxiliary_context_overflows": sum(item["error_kind"] == "local_context_budget" and item["call_category"].startswith("auxiliary_") for item in history["local_errors"]),
            "telemetry_status": row.get("telemetry_status"), **history}
        analyzed.append(value)
        all_local.extend(history["local_errors"])
        all_views.extend(history["context_views"])
        all_terminal.extend(terminal)
    local_overflow = [item for item in all_local if item["error_kind"] == "local_context_budget"]
    sdk_errors = [item for item in requests if item.get("status") == "failed"]
    provider_context = [item for item in sdk_errors if item.get("error_kind") == "provider_context_limit"]
    provider_rejected = [item for item in sdk_errors if item.get("error_type") in {"BadRequestError", "UnprocessableEntityError", "RateLimitError", "AuthenticationError", "PermissionDeniedError"}
        or re.search(r"(?:error code|status(?:_code)?)[\s:]+(?:400|401|403|413|422|429)\b", str(item.get("error", "")), re.I)]
    consecutive = 0
    for row in analyzed:
        if row["turn_id"] != consecutive + 1 or not row["completed"]:
            break
        consecutive += 1
    sessions = {row["session_id"] for row in analyzed if row.get("session_id")}
    planned = manifest.get("planned_turns")
    first_failure = next((row for row in analyzed if row["whole_turn_failed"]), None)
    summary_fields = ("model", "keep_messages", "context_window", "trigger_context_window", "trigger_fraction", "trim_tokens_to_summarize", "max_output_tokens", "request_attempts")
    summary_config = manifest.get("session_summary_config") or {}
    return {"schema_version": 1, "analysis_version": VERSION, "analysis_started_at": started, "analysis_finished_at": timestamp(),
        "input_directory": str(input_dir), "experiment_status": manifest.get("status", "unknown"),
        "source_id": manifest.get("source_id"), "source_snapshot_execution": manifest.get("source_snapshot_execution"),
        "configuration_scope": {"compression_enabled": manifest.get("compression_enabled"),
            "session_compression_enabled": manifest.get("session_compression_enabled"), "semantic_context_flag": manifest.get("semantic_context_flag"),
            "session_task_model": manifest.get("session_task_model"),
            "session_summary": {key: summary_config.get(key) for key in summary_fields}},
        "summary": {"planned_turns": planned, "attempted_turns": len(analyzed), "completed_turns": sum(row["completed"] for row in analyzed),
            "finalized_completed_turns": sum(row["completion_finalized"] for row in analyzed), "consecutive_completed_turns": consecutive,
            "single_session_observed": len(sessions) == 1 if sessions else None, "session_ids": sorted(sessions),
            "unexecuted_turns": [number for number in range(1, planned + 1) if number not in turns] if isinstance(planned, int) else None,
            "first_local_context_overflow": local_overflow[0] if local_overflow else None,
            "first_whole_turn_failure": {key: first_failure.get(key) for key in ("turn_id", "status", "routing", "terminal_errors")} if first_failure else None,
            "first_sdk_error": sdk_errors[0] if sdk_errors else None,
            "first_provider_rejection": provider_rejected[0] if provider_rejected else None,
            "first_provider_context_rejection": provider_context[0] if provider_context else None,
            "local_context_overflow_count": len(local_overflow),
            "local_context_overflow_by_category": dict(Counter(row["call_category"] for row in local_overflow)),
            "max_local_context_overflow_input_tokens": maximum(row.get("input_tokens") for row in local_overflow),
            "sdk_error_count": len(sdk_errors), "provider_context_rejection_count": len(provider_context),
            "terminal_error_records": len(all_terminal), "whole_turn_failure_count": sum(row["whole_turn_failed"] for row in analyzed),
            "summary_attempted_calls": sum(row.get("summary_attempted") is True for row in all_views),
            "summary_adopted_calls": sum(row.get("summary_adopted") is True for row in all_views)},
        "api_usage": usage_stats(requests), "unassigned_sdk_requests": [row for row in requests if row.get("turn_id") not in turns],
        "turns": analyzed, "sdk_request_metadata": requests, "read_issues": reader.issues, "source_files_read": reader.sources,
        "measurement_boundaries": {"measured": "只分析采集时点已落盘记录；SDK完整请求和供应商usage与本地拦截分开计数。",
            "inferred": "提示词分类由已确认标记匹配；subsequent_work_observed仅证明后续记录，不证明整轮成功。",
            "not_evaluated": "事实正确率、未执行轮次、尚未落盘事件、被其他并发任务影响的独立性能和费用。",
            "unknown_usage": "没有usage则为None；known_reported_*只是已知部分之和，不能充当总量。",
            "snapshot_consistency": "多个文件未在一个事务中拍快照；运行期间可能变化，read_issues记录半行/缺口。",
            "overflow_scope": "辅助本地超限不等于整轮失败；SDK无错误也不能推出未出现本地超限。"}}


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="真实运行目录；只读")
    parser.add_argument("--output", type=Path, required=True, help="容量与错误分层统计JSON路径")
    args = parser.parse_args()
    root, output = args.input.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error("输入目录不存在")
    result = analyze(root)
    write_atomic(output, result)
    summary = result["summary"]
    print(json.dumps({"output": str(output), "snapshot_at": result["analysis_finished_at"],
        "attempted_turns": summary["attempted_turns"], "completed_turns": summary["completed_turns"],
        "consecutive_completed_turns": summary["consecutive_completed_turns"],
        "first_local_overflow_turn": (summary["first_local_context_overflow"] or {}).get("turn_id"),
        "local_overflows": summary["local_context_overflow_count"],
        "first_whole_failure_turn": (summary["first_whole_turn_failure"] or {}).get("turn_id"),
        "first_provider_context_rejection_turn": (summary["first_provider_context_rejection"] or {}).get("turn_id"),
        "sdk_errors": summary["sdk_error_count"], "read_issues": len(result["read_issues"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
