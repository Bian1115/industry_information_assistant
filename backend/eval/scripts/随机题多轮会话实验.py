#!/usr/bin/env python3
"""单一真实会话的顺序追问实验；遇到失败或澄清立即停止，不替换部署配置。"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import uuid
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
from eval.scripts.采集真实追问开发集 import create_identity, create_session, export, save

CURRENT_TURN = ContextVar("随机题会话实验轮次", default={})
MAX_SECONDS = 86400
MAX_REQUESTS = 10000
MODEL_TOKEN_LIMIT = 300_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps({"event": event, "time": now(), **fields}, ensure_ascii=False, default=str), flush=True)


def error_kind(value):
    message = str(value).lower()
    if any(term in message for term in ("整任务", "整条对话", "调用预算耗尽", "模型token预算耗尽")):
        return "experiment_budget"
    if any(term in message for term in ("context_budget_exceeded", "contextbudgetexceeded", "max_input", "输入硬预算", "输入超过预算", "输入超预算", "摘要请求超过配置窗口")):
        return "local_context_budget"
    if re.search(r"context[_ -]?(?:length|window)|maximum context|range of input length|input (?:length|tokens).*(?:exceed|limit)|exceed.*(?:input|token)|too many tokens|上下文.*(?:超|过长)", message):
        return "provider_context_limit"
    if any(term in message for term in ("rate limit", "429", "too many requests", "限流")):
        return "rate_limit"
    if re.search(r"超过\s*\d+(?:\.\d+)?\s*(?:s|秒).*预算|phase[_ -](?:timeout|budget)|阶段.*(?:超时|时限)", message):
        return "stage_timeout"
    if any(term in message for term in ("timeout", "timed out", "超时", "时限耗尽")):
        return "timeout"
    if any(term in message for term in ("finish_reason=length", "finish_reason: length", "输出被截断", "摘要输出被截断")):
        return "output_truncated"
    if any(term in message for term in ("search_budget", "revision_budget", "recursion limit", "搜索预算", "返工预算")):
        return "search_or_revision_budget"
    if any(term in message for term in ("jsondecodeerror", "expecting value", "expecting ',' delimiter", "json解析", "json parse")):
        return "output_parse_error"
    if any(term in message for term in ("clarification", "请明确", "需要澄清")):
        return "clarification"
    return "other_error"


def load_dialogue(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("turns"), list):
        raise ValueError("数据集须为含 source 和 turns 的 JSON object")
    turns = value["turns"]
    if not turns or any(not isinstance(turn, dict) or not str(turn.get("user_query") or "").strip() for turn in turns):
        raise ValueError("每轮须提供非空 user_query")
    if [turn.get("turn_id") for turn in turns] != list(range(1, len(turns) + 1)):
        raise ValueError("turn_id 须从 1 连续递增")
    if not isinstance(value.get("source"), dict) or value["source"].get("id") is None:
        raise ValueError("source 须提供原题 id")
    return value


def load_resume(directory, dialogue, dataset_path):
    """只接受同一固定数据集的终止运行；不伪造完成轮次或重新创建用户。"""
    manifest = json.loads((directory / "运行清单.json").read_text())
    rows = json.loads((directory / "逐轮结果.json").read_text())
    if manifest.get("status") not in {"stopped_at_first_failure", "failed", "interrupted"}:
        raise ValueError("只允许恢复已终止的实验，不恢复仍在运行或已完成的实验")
    if manifest.get("dataset_sha256") != hashlib.sha256(dataset_path.read_bytes()).hexdigest():
        raise ValueError("续跑数据集哈希与原实验不一致")
    if not rows or rows[-1].get("completed") or any(not row.get("completed") for row in rows[:-1]):
        raise ValueError("续跑要求已完成前缀及最后一个失败轮次")
    if [row["turn_id"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("原实验轮次不是连续前缀")
    for row in rows:
        if row["query"] != dialogue["turns"][row["turn_id"] - 1]["user_query"]:
            raise ValueError("续跑原话与固定数据集不一致")
        if row["session_id"] != manifest["session_id"] or row["user_id"] != manifest["evaluation_user_id"]:
            raise ValueError("原实验用户或会话归属不一致")
    return {"manifest": manifest, "rows": rows, "failed_row": rows[-1]}


def completed_resume_prefix(resume_data, restart_turn=None):
    """重跑轮次之后的旧结果不作为本次已完成前缀；历史预算另行全量继承。"""
    rows = resume_data["rows"]
    if restart_turn is None:
        return rows[:-1]
    if not 1 <= restart_turn <= len(rows):
        raise ValueError("重跑起点必须在原实验已尝试的轮次内")
    prefix = rows[:restart_turn - 1]
    if not all(row.get("completed") for row in prefix):
        raise ValueError("重跑起点之前必须全部交付")
    return prefix


def verify_restart_prefix(owner, session_id, rows, directory):
    """原会话必须已归档旧后缀并恢复至指定前缀，防止偷看未来消息。"""
    from core.database import SessionLocal
    from models.chat import ChatMessage
    from service.conversation_service import get_conversation_service
    with SessionLocal() as db:
        get_conversation_service().require_session(db, session_id, owner)
        messages = db.query(ChatMessage).filter_by(session_id=uuid.UUID(session_id)).all()
        expected = {(row["turn_uuid"], role) for row in rows for role in ("user", "assistant")}
        actual = {(str(message.turn_id), message.role) for message in messages}
        if actual != expected or len(messages) != len(expected):
            raise ValueError("原会话未恢复指定前缀，包含缺失、重复或未来消息")
        by_turn = {(str(m.turn_id), m.role): m for m in messages}
        for row in rows:
            user = by_turn[row["turn_uuid"], "user"]
            assistant = by_turn[row["turn_uuid"], "assistant"]
            report = (directory / f"第{row['turn_id']}轮/研究报告.md").read_text()
            if user.content != row["query"] or assistant.content != report or assistant.run_id != row["run_id"]:
                raise ValueError("前缀用户原话、报告或研究归属与原实验不一致")
        return {"owner_verified": True, "session_id": session_id, "message_count": len(messages),
                "inherited_turns": len(rows), "future_messages": 0}


class ApiLedger:
    """全会话 SDK 调用账本；内部 HTTP 重试按上限保守预留，不虚构实际用量。"""

    def __init__(self, directory, budget):
        self.directory, self.budget = directory, budget
        self.started = time.monotonic()
        self.rows, self.reserved = [], {}
        self.request_attempts_reserved = 0
        self.lock = threading.Lock()
        from service.qwen_token_counter import get_qwen_token_counter
        self.counter = get_qwen_token_counter()

    def append_event(self, event):
        with (self.directory / "模型请求与用量.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def begin(self, resource, kwargs):
        if kwargs.get("stream"):
            raise ValueError("本实验账本仅支持研究服务的非流式模型调用")
        model = str(kwargs.get("model") or "unknown")
        messages = kwargs.get("messages") or []
        try:
            tokens = self.counter.count_messages(messages)
            counting = "Qwen chat template"
        except Exception:
            tokens = self.counter.count_text(json.dumps(messages, ensure_ascii=False, default=str))
            counting = "Qwen JSON估算；chat template不可用"
        output = int(kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 16000)
        retries = max(0, int(getattr(resource._client, "max_retries", 0) or 0))
        attempts = retries + 1
        with self.lock:
            if time.monotonic() - self.started >= MAX_SECONDS:
                raise TimeoutError("整条对话时间预算耗尽")
            if self.request_attempts_reserved + attempts > MAX_REQUESTS:
                raise ValueError("整条对话调用预算耗尽，含SDK内部重试上限")
            reserved = attempts * (tokens + output)
            if self.reserved.get(model, 0) + reserved > MODEL_TOKEN_LIMIT:
                raise ValueError("整条对话模型token预算耗尽")
            # 已配置模型之外的辅助调用也按同一固定上限纳入实际账本。
            self.budget["max_tokens_by_model"].setdefault(model, MODEL_TOKEN_LIMIT)
            self.reserved[model] = self.reserved.get(model, 0) + reserved
            self.request_attempts_reserved += attempts
            row = {"request_id": str(uuid.uuid4()), **CURRENT_TURN.get(), "model": model,
                   "started_at": now(), "input_tokens_estimated": tokens, "token_count_method": counting,
                   "output_tokens_reserved": output, "sdk_max_retries": retries,
                   "http_attempts_reserved": attempts, "actual_http_attempts": None,
                   "tokens_reserved": reserved, "usage": None, "status": "running"}
            self.rows.append(row)
            self.append_event({"event": "request_start", **row, "request": kwargs})
        return row

    def finish(self, row, response=None, error=None):
        with self.lock:
            row["ended_at"] = now()
            if error is not None:
                row.update(status="failed", error_type=type(error).__name__, error=str(error), error_kind=error_kind(error))
            else:
                usage = getattr(response, "usage", None)
                row.update(status="completed", usage=usage.model_dump() if usage is not None else None,
                           finish_reasons=[getattr(choice, "finish_reason", None) for choice in getattr(response, "choices", [])])
                if row["usage"]:
                    actual = int(row["usage"].get("total_tokens") or 0)
                    self.reserved[row["model"]] += max(0, actual - row["tokens_reserved"])
            self.append_event({"event": "request_end", **row})

    def report(self):
        with self.lock:
            return {"duration_seconds": time.monotonic() - self.started,
                    "sdk_create_calls": len(self.rows), "http_attempts_reserved": self.request_attempts_reserved,
                    "tokens_reserved_by_model": dict(self.reserved), "requests": list(self.rows),
                    "max_input_tokens_estimated": max((row["input_tokens_estimated"] for row in self.rows), default=0),
                    "max_input_tokens_actual": max((int(row["usage"]["prompt_tokens"]) for row in self.rows if row.get("usage") and row["usage"].get("prompt_tokens") is not None), default=None),
                    "api_price": None, "note": "失败用量和实际SDK内部HTTP重试数未知；按上限保守预留，不估算价格"}


@contextmanager
def observe_api(ledger):
    from openai.resources.chat.completions import AsyncCompletions, Completions
    original_sync, original_async = Completions.create, AsyncCompletions.create

    def sync_call(resource, *args, **kwargs):
        ticket = ledger.begin(resource, kwargs)
        try:
            response = original_sync(resource, *args, **kwargs)
        except BaseException as exc:
            ledger.finish(ticket, error=exc)
            raise
        ledger.finish(ticket, response=response)
        return response

    async def async_call(resource, *args, **kwargs):
        ticket = ledger.begin(resource, kwargs)
        try:
            response = await original_async(resource, *args, **kwargs)
        except BaseException as exc:
            ledger.finish(ticket, error=exc)
            raise
        ledger.finish(ticket, response=response)
        return response

    with patch.object(Completions, "create", sync_call), patch.object(AsyncCompletions, "create", async_call):
        yield


def persist_user(owner, session_id, turn_id, run_id, query):
    from core.database import SessionLocal
    from service.conversation_service import get_conversation_service
    with SessionLocal() as db:
        conversations = get_conversation_service()
        session = conversations.require_session(db, session_id, owner)
        conversations.save_user_message(db, session, turn_id, query,
            execution_mode="research", run_id=run_id, research_status="running")


def persist_clarification(owner, session_id, turn_id, task):
    from core.database import SessionLocal
    from models.chat import ChatMessage
    from service.conversation_service import get_conversation_service
    with SessionLocal() as db:
        conversations = get_conversation_service()
        conversations.require_session(db, session_id, owner)
        user = db.query(ChatMessage).filter_by(session_id=uuid.UUID(session_id), turn_id=uuid.UUID(turn_id), role="user").one()
        user.research_status, user.execution_mode, user.run_id = "clarification", "chat", None
        message = conversations.save_assistant_message(db, session_id, owner, turn_id, task["clarification_question"],
            execution_mode="chat", research_status="clarification", references_data={"session_task": task})
        # 与HTTP入口相同的澄清响应，本地记录；不伪造研究完成事件。
        return {"type": "session_clarification", "content": message.content, "question": message.content, "run_id": turn_id}


def validate_task_rebuild_row(row):
    """仅重建会话澄清，或研究建图前发生的澄清失败；图为空仍须另行核验。"""
    if not (row.get("status") == "clarified" or
            row.get("status") == "failed" and row.get("error_kind") == "clarification"):
        raise ValueError("重建会话任务仅适用于未建图的澄清轮或澄清失败轮")


def reuse_clarified_user(owner, session_id, turn_id, run_id, query, *, activate=False, allow_failed_run=False):
    """核验并复用原话；失败run仅在调用方已确认没有研究图时允许复用。"""
    from core.database import SessionLocal
    from models.chat import ChatMessage
    from models.research import ResearchRun
    from service.conversation_service import get_conversation_service
    with SessionLocal() as db:
        get_conversation_service().require_session(db, session_id, owner, for_update=True)
        run = db.query(ResearchRun).filter_by(run_id=run_id).first()
        if run is not None:
            if not allow_failed_run:
                raise ValueError("澄清轮已有研究运行，不能重建初始输入")
            if str(run.user_id) != owner or run.session_id != session_id:
                raise ValueError("失败运行的用户或会话归属与续跑输入不一致")
            if (run.status != "failed" or run.final_checkpoint_id or
                    run.last_projected_checkpoint_id or run.delivered_event_id or
                    run.lease_until and run.lease_until > datetime.utcnow()):
                raise ValueError("已有运行尚未终止或已推进研究，不能重建初始输入")
            assistant = db.query(ChatMessage).filter_by(session_id=uuid.UUID(session_id),
                turn_id=uuid.UUID(turn_id), role="assistant").first()
            if assistant is not None and assistant.research_status != "clarification":
                raise ValueError("本轮已有助手交付，不能重建初始输入")
        message = db.query(ChatMessage).filter_by(
            session_id=uuid.UUID(session_id), turn_id=uuid.UUID(turn_id), role="user").one()
        if message.content != query or message.research_status not in {"clarification", "running", "failed"}:
            raise ValueError("原用户消息或澄清状态与续跑输入不一致")
        if message.run_id not in {None, run_id}:
            raise ValueError("原用户消息已关联其他研究运行")
        record = {"id": str(message.id), "session_id": session_id, "turn_uuid": turn_id,
                  "content": message.content, "previous_execution_mode": message.execution_mode,
                  "previous_run_id": message.run_id, "previous_research_status": message.research_status,
                  "original_created_at": str(message.created_at), "activated": activate}
        if activate:
            message.execution_mode, message.run_id, message.research_status = "research", run_id, "running"
            db.commit()
        return record


def mark_failed(owner, session_id, turn_id):
    from core.database import SessionLocal
    from models.chat import ChatMessage
    with SessionLocal() as db:
        db.query(ChatMessage).filter_by(session_id=uuid.UUID(session_id), turn_id=uuid.UUID(turn_id), role="user").update({"research_status": "failed"})
        db.commit()


def write_report(path, output, manifest, rows):
    completed = sum(row.get("completed", False) for row in rows)
    first = next((row for row in rows if not row.get("completed")), None)
    violations = [
        {"turn_id": row["turn_id"], "stage": "final_response", "kind": row["error_kind"], "error": row.get("error")}
        for row in rows if row.get("error_kind") in {"local_context_budget", "provider_context_limit"}
    ]
    violations += [
        {"turn_id": row["turn_id"], "stage": "provider_request", "kind": request["error_kind"], "error": request.get("error"), "request_id": request["request_id"]}
        for row in rows for request in row.get("requests", [])
        if request.get("error_kind") in {"local_context_budget", "provider_context_limit"}
    ]
    summary = {"status": manifest.get("status"), "planned_turns": manifest.get("planned_turns", 0),
               "attempted_turns": len(rows), "completed_turns": completed,
               "first_failure_turn": first.get("turn_id") if first else None,
               "first_failure_kind": first.get("error_kind") if first else None,
               "first_context_overflow_turn": min((item["turn_id"] for item in violations), default=None),
               "context_violations": violations,
               "all_planned_turns_delivered_without_context_overflow": bool(rows) and len(rows) == manifest.get("planned_turns") and completed == len(rows) and not violations,
               "unexecuted_turns": list(range(len(rows) + 1, manifest.get("planned_turns", 0) + 1)),
               "answer_correctness": "尚未评估；completed只表示交付成功"}
    save(output / "实验统计.json", summary)
    lines = ["# 随机原题多轮会话实验结果及分析", "", f"评估日期：{manifest.get('started_at', now())}。状态：{manifest.get('status')}。",
             f"评估对象：DRB 原题 {manifest.get('source_id')} 的单一真实会话；计划 {summary['planned_turns']} 轮。",
             f"原始结果目录：[{output.name}]({output.resolve()})。", "", "## 数据与配置口径", "",
             "仅本轮user_query进入模型，objective/probes/expected_operation仅供事后分析。每轮独立run，用户与session固定；父成果由生产路由及授权继承，Agent内部历史按run隔离。",
             f"冻结配置：[配置快照.json]({output.resolve() / '配置快照.json'})；数据：[实验对话.json]({output.resolve() / '实验对话.json'})。",
             "使用部署当前开关及模型；不为完成实验降低阈值或强制预期路由。", "", "## 核心指标", "",
             f"已尝试 {len(rows)} 轮，已交付 {completed} 轮；首次失败轮次：{summary['first_failure_turn']}；首次上下文超限：{summary['first_context_overflow_turn']}。",
             f"启动/入口错误：{manifest.get('error', '无')}。", "", "## 逐轮分析", "",
             "| 轮次 | 交付 | 路由 | 会话token | 最大实际输入token | 状态/错误 |", "|---|---|---|---:|---:|---|" ]
    for row in rows:
        lines.append(f"| {row['turn_id']} | {row.get('completed', False)} | {row.get('execution_plan', '')} | {(row.get('history_metrics') or {}).get('router_tokens', '')} | {row.get('max_input_tokens_actual', '')} | {row.get('error_kind', row.get('status', ''))} |")
    lines += ["", "## 失败样例", "", json.dumps(first or {"startup_error": manifest.get("error"), "note": "尚无已记录轮次失败"}, ensure_ascii=False, default=str),
              "", "## 局限性与测量边界", "", "已测量：实际发送请求、用量与错误、生产会话历史/摘要/路由、已交付正文及原始账本（仅限已执行轮次）。",
              "间接推断：预算预留和Qwen token估算用于解释输入增长，不能代替供应商真实usage；SDK内部HTTP重试数未知。",
              "尚未评估：人工事实正确率、要求遵守率、未执行轮次、独立于并发其他实验的性能。不能把成功交付或未观察到超限视为回答正确或无限轮次保证。",
              "实时搜索、外部限流及并发任务可能影响结果。费用不推测；失败调用未知用量不记零。",
              "", "## 结论与下一步", "", "以实际首失败及异常分类定位限制；核对逐轮正文和预设检查项，再决定是否需要压缩或修复路由。",
              "机器可读文件：运行清单.json、逐轮结果.json、实验统计.json、模型请求与用量.jsonl、API预算账本.json、业务预算账本.json、本地完整埋点.jsonl；逐轮目录保存生产上下文、报告、工作历史和会话原文。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args, dialogue, manifest, budget):
    from eval.scripts.run_deepresearch_bench import benchmark_service, decode_service_event
    from service.research_context_service import get_research_context_service
    from service.verification_budget import task_budget
    from config.llm_config import get_config
    rows, owner, session_id = [], None, None
    api = ApiLedger(args.output, budget)
    resume_data = load_resume(args.resume_from, dialogue, args.dataset) if args.resume_from else None
    rebuild_task = bool(getattr(args, "rebuild_session_task", False))
    restart_turn = getattr(args, "restart_from_turn", None)
    if restart_turn is not None and (not resume_data or rebuild_task):
        raise ValueError("从指定轮重跑须提供原实验，且不能同时重建澄清任务")
    if rebuild_task:
        if not resume_data:
            raise ValueError("重建会话任务必须指定原终止实验")
        validate_task_rebuild_row(resume_data["failed_row"])
    if resume_data:
        previous_api = json.loads((args.resume_from / "API预算账本.json").read_text())
        api.rows = previous_api["requests"]
        api.reserved = dict(previous_api["tokens_reserved_by_model"])
        api.request_attempts_reserved = previous_api["http_attempts_reserved"]
        api.started -= previous_api["duration_seconds"]
        # 累计调用日志复制到新分段，新增请求只追加；历史分段保留原样。
        shutil.copyfile(args.resume_from / "模型请求与用量.jsonl", args.output / "模型请求与用量.jsonl")
        rows = [dict(row, inherited_from_segment=str(args.resume_from))
                for row in completed_resume_prefix(resume_data, restart_turn)]
        for row in rows:
            shutil.copytree(args.resume_from / f"第{row['turn_id']}轮", args.output / f"第{row['turn_id']}轮")
    save(args.output / "逐轮结果.json", rows)
    try:
        if resume_data:
            owner, session_id = resume_data["manifest"]["evaluation_user_id"], resume_data["manifest"]["session_id"]
            manifest.update(continuation_of=str(args.resume_from), resumed_turn=restart_turn or resume_data["failed_row"]["turn_id"],
                            restarted_from_turn=restart_turn,
                            cumulative_budget_includes_previous_segment=True)
            if restart_turn is not None:
                prefix_check = await asyncio.to_thread(verify_restart_prefix, owner, session_id, rows, args.resume_from)
                save(args.output / "指定轮重跑前缀核验.json", prefix_check)
        else:
            owner = await asyncio.to_thread(create_identity, args.output.name)
            session_id = await asyncio.to_thread(create_session, owner, f"随机原题{dialogue['source']['id']}多轮实验")
        manifest.update(evaluation_user_id=owner, session_id=session_id)
        save(args.output / "运行清单.json", manifest)
        with task_budget(budget) as ledger, observe_api(api):
            if resume_data:
                previous_business = json.loads((args.resume_from / "业务预算账本.json").read_text())
                ledger.requests = previous_business["requests"]
                ledger.reserved = dict(previous_business["tokens_reserved_by_model"])
                ledger.started -= previous_business["duration_seconds"]
            async with benchmark_service(get_config().research.max_iterations) as service:
                if resume_data and restart_turn is None:
                    pending = await service.read_state(resume_data["failed_row"]["run_id"])
                    if rebuild_task and pending["exists"]:
                        raise ValueError("澄清轮或失败轮已有研究图，不能重建初始输入")
                    resumable = pending.get("resumable", bool(pending.get("exists") and pending.get("pending_nodes")))
                    if not rebuild_task and not resumable:
                        raise ValueError("原运行没有待执行的生产图检查点")
                    save(args.output / "续跑入口状态.json", pending)
                    manifest["resume_mode"] = "rebuild_session_task" if rebuild_task else "graph_checkpoint"
                elif restart_turn is not None:
                    manifest["resume_mode"] = "restart_from_turn"
                    save(args.output / "续跑入口状态.json", {"mode": "restart_from_turn",
                         "restarted_from_turn": restart_turn, "inherited_completed_turns": len(rows),
                         "original_graph_resumed": False})
                deadline = asyncio.get_running_loop().time() + MAX_SECONDS - (time.monotonic() - api.started)
                for turn in dialogue["turns"]:
                    if turn["turn_id"] <= len(rows):
                        continue
                    resumed = bool(resume_data and restart_turn is None and turn["turn_id"] == resume_data["failed_row"]["turn_id"])
                    directory = args.output / f"第{turn['turn_id']}轮"
                    row = {"turn_id": turn["turn_id"], "source_id": dialogue["source"]["id"], "user_id": owner,
                           "session_id": session_id, "run_id": str(uuid.uuid4()), "turn_uuid": str(uuid.uuid4()),
                           "query": turn["user_query"], "started_at": now(), "completed": False, "status": "running"}
                    if resumed:
                        old_row = resume_data["failed_row"]
                        row.update(run_id=old_row["run_id"], turn_uuid=old_row["turn_uuid"],
                                   resumed=True, original_attempt_started_at=old_row["started_at"],
                                   prior_terminal_error=old_row.get("error"),
                                   session_task_rebuilt=rebuild_task, graph_resumed=not rebuild_task)
                    if restart_turn is not None:
                        old_row = next((r for r in resume_data["rows"] if r["turn_id"] == turn["turn_id"]), None)
                        row.update(restarted=True, graph_resumed=False,
                                   prior_run_id=old_row.get("run_id") if old_row else None,
                                   prior_turn_uuid=old_row.get("turn_uuid") if old_row else None)
                    rows.append(row)
                    save(directory / "轮次输入.json", row)
                    offset = len(api.rows)
                    token = CURRENT_TURN.set({"turn_id": row["turn_id"], "run_id": row["run_id"], "session_id": session_id})

                    async def consume():
                        if resumed and not rebuild_task:
                            context = json.loads((args.resume_from / f"第{row['turn_id']}轮" / "会话任务输入.json").read_text())
                        else:
                            if resumed:
                                reused = await asyncio.to_thread(reuse_clarified_user, owner, session_id,
                                    row["turn_uuid"], row["run_id"], row["query"], allow_failed_run=True)
                                save(directory / "原用户消息复用核验.json", reused)
                            else:
                                await asyncio.to_thread(persist_user, owner, session_id, row["turn_uuid"], row["run_id"], row["query"])
                            context = await asyncio.to_thread(get_research_context_service().build,
                                user_id=owner, session_id=session_id, current_turn_id=row["turn_uuid"], current_query=row["query"])
                        save(directory / "会话任务输入.json", context)
                        row.update(history_metrics=context.get("history_metrics", {}), session_summary_metrics=context.get("session_summary_metrics", {}),
                                   session_task_builder_metrics=context.get("task_metrics", {}), session_task=context.get("session_task", {}))
                        fallback_error = row["session_task_builder_metrics"].get("error")
                        if fallback_error and error_kind(fallback_error) in {"local_context_budget", "provider_context_limit"}:
                            raise RuntimeError("会话判断模型上下文超限：" + fallback_error)
                        if row["session_task"].get("status") == "clarify":
                            response = await asyncio.to_thread(persist_clarification, owner, session_id, row["turn_uuid"], row["session_task"])
                            save(directory / "澄清响应.json", response)
                            row.update(status="clarified", clarified=True, error_kind="clarification", error=response["content"])
                            return
                        if resumed and rebuild_task:
                            activated = await asyncio.to_thread(reuse_clarified_user, owner, session_id,
                                row["turn_uuid"], row["run_id"], row["query"], activate=True, allow_failed_run=True)
                            save(directory / "原用户消息重新关联研究.json", activated)
                        async with aclosing(service.research(query=row["query"], session_id=session_id, user_id=owner,
                                run_id=row["run_id"], turn_id=row["turn_uuid"], research_context=context,
                                resume=resumed and not rebuild_task)) as stream:
                            async for chunk in stream:
                                event = decode_service_event(chunk)
                                if event is None:
                                    continue
                                with (directory / "研究事件.jsonl").open("a", encoding="utf-8") as file:
                                    file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                                payload = event.get("data") if isinstance(event.get("data"), dict) else event
                                if event["type"] == "research_complete":
                                    article = payload.get("final_report") or ""
                                    if not article.strip():
                                        raise ValueError("完成事件缺少正文")
                                    (directory / "研究报告.md").write_text(article, encoding="utf-8")
                                    row.update(completed=True, status="completed", delivery=payload)
                                elif event["type"] in {"error", "research_cancelled", "research_paused"}:
                                    raise RuntimeError(json.dumps(payload, ensure_ascii=False))
                        if not row["completed"]:
                            raise RuntimeError("未收到完成事件")

                    async def heartbeat():
                        while True:
                            await asyncio.sleep(30)
                            try:
                                await export(service, row, directory)
                            except Exception as exc:
                                row["live_export_error"] = str(exc)
                            save(args.output / "逐轮结果.json", rows)
                            save(args.output / "API预算账本.json", api.report())
                            emit("heartbeat", turn=row["turn_id"], completed=sum(item.get("completed", False) for item in rows), sdk_calls=len(api.rows))

                    observer = asyncio.create_task(heartbeat())
                    emit("turn_start", turn=row["turn_id"], run_id=row["run_id"], session_id=session_id)
                    try:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise TimeoutError("整条对话时间预算耗尽")
                        await asyncio.wait_for(consume(), timeout=remaining)
                    except BaseException as exc:
                        interrupted = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
                        row.update(status="interrupted" if interrupted else "failed", completed=False,
                                   error_type=type(exc).__name__, error=str(exc),
                                   error_kind="interrupted" if interrupted else error_kind(exc))
                        try:
                            await asyncio.to_thread(mark_failed, owner, session_id, row["turn_uuid"])
                        except Exception as persistence_error:
                            row["failure_status_persistence_error"] = str(persistence_error)
                        if interrupted:
                            raise
                    finally:
                        observer.cancel()
                        await asyncio.gather(observer, return_exceptions=True)
                        try:
                            await export(service, row, directory)
                        except Exception as exc:
                            row.update(telemetry_status="error", telemetry_error=str(exc))
                        requests = api.rows[offset:]
                        row.update(ended_at=now(), sdk_calls=len(requests), requests=requests,
                            max_input_tokens_estimated=max((item["input_tokens_estimated"] for item in requests), default=0),
                            max_input_tokens_actual=max((int(item["usage"]["prompt_tokens"]) for item in requests if item.get("usage") and item["usage"].get("prompt_tokens") is not None), default=None))
                        save(directory / "轮次结果.json", row)
                        save(args.output / "逐轮结果.json", rows)
                        save(args.output / "API预算账本.json", api.report())
                        save(args.output / "业务预算账本.json", ledger.report())
                        write_report(args.report, args.output, manifest, rows)
                        CURRENT_TURN.reset(token)
                    emit("turn_end", turn=row["turn_id"], completed=row["completed"], error_kind=row.get("error_kind"))
                    if not row["completed"]:
                        break
        manifest["status"] = "completed" if len(rows) == len(dialogue["turns"]) and all(row["completed"] for row in rows) else "stopped_at_first_failure"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed", error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
            raise
    finally:
        manifest.update(ended_at=now(), completed_turns=sum(row.get("completed", False) for row in rows))
        save(args.output / "运行清单.json", manifest)
        save(args.output / "API预算账本.json", api.report())
        write_report(args.report, args.output, manifest, rows)
    return rows


def snapshot_source(output):
    hashes = {}
    for folder in (BACKEND / "app", BACKEND / "eval"):
        for file in folder.rglob("*"):
            if not file.is_file() or file.suffix not in {".py", ".txt"} or any(part in {"data", "__pycache__", ".venv"} for part in file.relative_to(BACKEND).parts):
                continue
            relative = file.relative_to(BACKEND)
            target = output / "代码快照" / "backend" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
            hashes[str(relative)] = hashlib.sha256(file.read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=BACKEND / ".env", help="源码副本可显式加载原部署env；不会复制密钥")
    parser.add_argument("--resume-from", type=Path, help="恢复原真实运行目录的最后失败轮，沿用用户/会话/运行及累计预算")
    parser.add_argument("--rebuild-session-task", action="store_true", help="仅恢复未建图的澄清轮或澄清失败轮，复用用户原话并重新调用会话任务构建")
    parser.add_argument("--restart-from-turn", type=int, help="从指定轮重新执行；原会话须先归档旧后缀并恢复此前已交付消息，累计预算仍全量继承")
    args = parser.parse_args()
    args.dataset, args.output, args.report = args.dataset.resolve(), args.output.resolve(), args.report.resolve()
    args.env_file = args.env_file.resolve()
    args.resume_from = args.resume_from.resolve() if args.resume_from else None
    if args.rebuild_session_task and not args.resume_from:
        parser.error("重建会话任务必须指定原终止实验--resume-from")
    if args.restart_from_turn is not None and (not args.resume_from or args.rebuild_session_task):
        parser.error("指定轮重跑须有--resume-from，且不能与--rebuild-session-task组合")
    if args.output.exists() or args.report.exists():
        parser.error("只允许新输出目录与新中文报告，不覆盖历史结果")
    args.output.mkdir(parents=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"started_at": now(), "status": "preparing", "pid": os.getpid(), "dataset": str(args.dataset),
                "execution_entry": "production_session_task_builder_and_service", "history_mode": "single_owned_session_independent_research_runs",
                "agent_history_continues_across_turns": False, "user_memory_outbox": "使用生产ConversationService，不跳过",
                "source_snapshot_execution": os.getenv("EVAL_SOURCE_SNAPSHOT_EXECUTION") == "1", "backend_source_root": str(BACKEND), "planned_turns": 0,
                "environment_file": str(args.env_file), "dotenv_override": False}
    rows = []
    try:
        from dotenv import load_dotenv
        if not args.env_file.is_file():
            raise FileNotFoundError("指定环境配置文件不存在：" + str(args.env_file))
        load_dotenv(args.env_file, override=False)
        os.environ["EVAL_TRACE_LOCAL_FILE"] = str(args.output / "本地完整埋点.jsonl")
        dialogue = load_dialogue(args.dataset)
        if args.resume_from:
            load_resume(args.resume_from, dialogue, args.dataset)
        manifest.update(source_id=dialogue["source"]["id"], planned_turns=len(dialogue["turns"]), dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest())
        save(args.output / "实验对话.json", dialogue)
        from service.context_policy import selected_policy
        from service.session_summarization import SessionSummaryConfig
        from service.deep_research_v2.report_constraints import body_length_hard_check_enabled
        from service.llm_client import default_fallback_model
        from config.llm_config import get_config, get_summary_model
        policy, config = selected_policy(), get_config()
        save(args.output / "配置快照.json", policy.snapshot())
        os.environ["CONTEXT_ABLATION_CONFIG"] = str(args.output / "配置快照.json")
        agent_names = ("architect", "scout", "data_analyst", "writer", "critic", "wizard")
        models = {name: getattr(config.agents, name).to_dict() for name in agent_names}
        session_model = os.getenv("SESSION_TASK_MODEL", os.getenv("OPENAI_MODEL", "qwen3-max"))
        summary_config = SessionSummaryConfig.from_env()
        model_names = {item["model"] for item in models.values()} | {session_model, summary_config.model, get_summary_model(), "qwen-plus", "qwen3-max"}
        if default_fallback_model():
            model_names.add(default_fallback_model())
        model_names.update(value for key, value in os.environ.items() if key.startswith("LLM_MODEL_") and value.strip())
        budget = {"max_seconds": MAX_SECONDS, "max_requests": MAX_REQUESTS, "max_tokens_by_model": {name: MODEL_TOKEN_LIMIT for name in sorted(model_names)}}
        save(args.output / "对话总预算.json", budget)
        manifest.update(context_policy=policy.snapshot(), models=models, session_task_model=session_model,
            report_body_length_hard_check_enabled=body_length_hard_check_enabled(),
            session_summary_config=asdict(summary_config), agent_summary_model=get_summary_model(), fallback_model=default_fallback_model(),
            compression_enabled=policy.compression_enabled, session_compression_enabled=policy.effective_session_compression_enabled,
            semantic_context_flag=os.getenv("SEMANTIC_CONTEXT_ENABLED", "true"), budget=budget,
            source_sha256=snapshot_source(args.output), status="running", max_iterations=config.research.max_iterations,
            token_count_scope="会话指标不含业务模板/报告候选；API账本另记完整SDK请求估算及真实usage",
            limitations=["实时搜索未冻结", "人工质量尚未标注", "并发实验可能影响限流及速度", "SDK内部重试数未知按上限预留"])
        save(args.output / "运行清单.json", manifest)
        write_report(args.report, args.output, manifest, rows)
        rows = asyncio.run(run(args, dialogue, manifest, budget))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__, error=str(exc), ended_at=now())
        emit("experiment_error", error_type=type(exc).__name__, error=str(exc))
    finally:
        if manifest.get("source_sha256"):
            manifest["changed_source_files"] = [name for name, digest in manifest["source_sha256"].items() if not (BACKEND / name).exists() or hashlib.sha256((BACKEND / name).read_bytes()).hexdigest() != digest]
        # run内部已实时保存中断前各轮，避免asyncio.run异常后用空rows覆盖。
        existing = args.output / "逐轮结果.json"
        if existing.exists():
            rows = json.loads(existing.read_text())
        save(args.output / "运行清单.json", manifest)
        write_report(args.report, args.output, manifest, rows)
    emit("experiment_end", status=manifest["status"], completed=sum(row.get("completed", False) for row in rows), planned=manifest["planned_turns"])
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
