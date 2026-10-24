"""评测进程专用HTTP用量旁路记录；不修改请求、响应或生产客户端。"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from datetime import datetime
import inspect
import json
from pathlib import Path
import threading
import time
import uuid
from unittest.mock import patch
from zoneinfo import ZoneInfo

GROUP = ContextVar("eval_api_usage_group", default="初始化")
COMPONENT = ContextVar("eval_api_usage_component", default=None)
NODES = {"plan": "ChiefArchitect", "research": "DeepScout", "supplement": "DeepScout",
         "re_research": "DeepScout", "analyze": "DataAnalyst", "write": "LeadWriter",
         "edit": "LeadWriter", "revise": "LeadWriter", "review": "CriticMaster"}


def summarize(rows):
    def totals(items):
        values = {}
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            known = [row[field] for row in items if type(row.get(field)) is int]
            values[field + "_known"] = sum(known)
            values[field + "_missing_calls"] = len(items) - len(known)
            values[field] = sum(known) if len(known) == len(items) else None
        return {"calls": len(items), "failed_calls": sum(row.get("status") != "completed" for row in items), **values}
    components, models, kinds, agents = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        components[row["component"]].append(row)
        models[row.get("model") or "unknown"].append(row)
        kinds[row["kind"]].append(row)
        agents[row.get("stage_agent") or "无所属Agent"].append(row)
    return {**totals(rows), "by_component": {k: totals(v) for k, v in components.items()},
            "by_model": {k: totals(v) for k, v in models.items()},
            "by_kind": {k: totals(v) for k, v in kinds.items()},
            "by_stage_agent": {k: totals(v) for k, v in agents.items()},
            "scope": "当前实验进程实际HTTP模型请求；包含失败和SDK重试，不含Codex对话自身推理用量"}


def context():
    try:
        from service.agent_history import current_history_scope
        scope = current_history_scope()
    except ImportError:
        scope = None
    result = {"group": GROUP.get(), "node": getattr(scope, "node", None),
              "run_id": getattr(scope, "run_id", None), "component": COMPONENT.get() or "辅助模型",
              "stage_agent": NODES.get(getattr(scope, "node", None))}
    if COMPONENT.get():
        return result
    frame = inspect.currentframe()
    try:
        while frame:
            filename, function = Path(frame.f_code.co_filename).name, frame.f_code.co_name
            if filename == "report_claim_extraction.py" and function == "call_report_model":
                result["component"] = frame.f_locals.get("component", "report_claim_extraction")
                break
            if filename == "nli_verifier.py" and function == "predict":
                result["component"] = "引用支持性NLI"
                break
            if filename == "base.py" and function == "call_llm":
                agent = getattr(frame.f_locals.get("self"), "name", None)
                result["component"] = agent or NODES.get(result["node"], "Agent")
                break
            if filename in {"semantic_context_service.py", "session_summarization.py", "summary_message_verification.py"}:
                result["component"] = "历史摘要及核验"
            if filename in {"web_reader_service.py", "web_document.py"}:
                result["component"] = "网页读取辅助"
            frame = frame.f_back
    finally:
        del frame
    return result


class Recorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows, self.lock = [], threading.Lock()

    @contextmanager
    def group(self, value):
        token = GROUP.set(value)
        try:
            yield
        finally:
            GROUP.reset(token)

    @contextmanager
    def component(self, value):
        token = COMPONENT.set(value)
        try:
            yield
        finally:
            COMPONENT.reset(token)

    def start(self, request):
        try:
            payload = json.loads(request.content)
        except (ValueError, UnicodeError, RuntimeError):
            return None
        if not isinstance(payload, dict) or not payload.get("model"):
            return None
        path = request.url.path.lower()
        if "embed" in path:
            kind, component = "embedding", "检索Embedding"
        elif "rerank" in path:
            kind, component = "rerank", "检索Rerank"
        elif "messages" in payload or "input" in payload:
            kind, component = "language_model", None
        else:
            return None
        meta = context()
        if component:
            meta["component"] = component
        return {**meta, "call_id": uuid.uuid4().hex, "model": payload["model"], "kind": kind,
                "started_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                "endpoint_path": request.url.path, "monotonic_start": time.monotonic()}

    def finish(self, row, response=None, error=None):
        if row is None:
            return
        row["duration_seconds"] = time.monotonic() - row.pop("monotonic_start")
        data = {}
        if response is not None:
            row["http_status"] = response.status_code
            try:
                data = response.json()
            except Exception:
                row["usage_unavailable_reason"] = "响应非JSON或流式正文未读取"
        usage = data.get("usage") or {} if isinstance(data, dict) else {}
        row.update(provider_request_id=(data.get("id") or data.get("request_id")) if isinstance(data, dict) else None,
                   usage=usage, status="completed" if response is not None and response.is_success else "error")
        def number(*keys):
            return next((usage[key] for key in keys if type(usage.get(key)) is int and usage[key] >= 0), None)
        row["input_tokens"] = number("prompt_tokens", "input_tokens")
        row["output_tokens"] = number("completion_tokens", "output_tokens")
        row["total_tokens"] = number("total_tokens")
        # Anthropic原生input_tokens不含缓存读写；OpenAI prompt_tokens通常已包含。
        if row["endpoint_path"].rstrip("/").endswith("/messages") and row["input_tokens"] is not None:
            row["provider_input_tokens_excluding_cache"] = row["input_tokens"]
            cache = [number(key) for key in ("cache_creation_input_tokens", "cache_read_input_tokens")]
            row["input_tokens"] += sum(value for value in cache if value is not None)
            row["input_tokens_method"] = "anthropic_input_plus_cache_creation_plus_cache_read"
        if row["total_tokens"] is None and row["input_tokens"] is not None and row["output_tokens"] is not None:
            row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
            row["total_tokens_method"] = "provider_input_plus_output"
        if error is not None:
            row["error_type"] = type(error).__name__
        with self.lock:
            self.rows.append(row)
            with self.path.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def summary(self, group=None):
        return summarize([r for r in self.rows if group is None or r["group"] == group])

    @contextmanager
    def installed(self):
        import httpx
        original_async, original_sync = httpx.AsyncClient.send, httpx.Client.send
        recorder = self
        async def async_send(client, request, *args, **kwargs):
            row = recorder.start(request)
            try:
                response = await original_async(client, request, *args, **kwargs)
            except BaseException as exc:
                recorder.finish(row, error=exc)
                raise
            recorder.finish(row, response)
            return response
        def sync_send(client, request, *args, **kwargs):
            row = recorder.start(request)
            try:
                response = original_sync(client, request, *args, **kwargs)
            except BaseException as exc:
                recorder.finish(row, error=exc)
                raise
            recorder.finish(row, response)
            return response
        with ExitStack() as stack:
            from service.deep_research_v2.agents.base import BaseAgent
            from service.deep_research_v2 import citation_verification, report_claim_extraction
            original_agent = BaseAgent.call_llm
            original_report = report_claim_extraction.call_report_model
            original_verify = citation_verification.verify_citations
            async def agent_call(agent, *args, **kwargs):
                with recorder.component(agent.name):
                    return await original_agent(agent, *args, **kwargs)
            async def report_call(*args, **kwargs):
                with recorder.component(kwargs.get("component", "report_claim_extraction")):
                    return await original_report(*args, **kwargs)
            async def verify_call(*args, **kwargs):
                with recorder.component("报告引用支持性NLI"):
                    return await original_verify(*args, **kwargs)
            stack.enter_context(patch.object(BaseAgent, "call_llm", agent_call))
            stack.enter_context(patch.object(report_claim_extraction, "call_report_model", report_call))
            stack.enter_context(patch.object(citation_verification, "verify_citations", verify_call))
            stack.enter_context(patch.object(httpx.AsyncClient, "send", async_send))
            stack.enter_context(patch.object(httpx.Client, "send", sync_send))
            yield self
