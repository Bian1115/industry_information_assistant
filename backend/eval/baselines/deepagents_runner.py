"""独立deepagents研究组：只复用搜索/读取叶子模块，不加载业务图或数据库。"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import message_to_dict
from pydantic import BaseModel, ConfigDict, Field, model_validator

BACKEND = Path(__file__).resolve().parents[2]
CONFIG_DIR = BACKEND / "eval/configs/deepagents独立对照"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    group: str = "deepagents独立端到端"
    model: str = Field(default="qwen3-max", min_length=1)
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=6000, gt=0)
    model_context_tokens: int = Field(default=32768, gt=0)
    timeout_seconds: float = Field(default=900, gt=0)
    recursion_limit: int = Field(default=100, gt=0)
    max_model_calls: int = Field(default=40, gt=0)
    max_search_requests: int = Field(default=30, gt=0)
    max_page_reads: int = Field(default=30, gt=0)
    citation_verification: bool = False
    max_verification_requests: int = Field(default=8, gt=0)
    verification_python: str = ""
    allowed_domains: list[str] = Field(default_factory=list)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    search_price_per_request: float | None = Field(default=None, ge=0)
    currency: str = Field(default="CNY", min_length=1)

    @model_validator(mode="after")
    def check_budget(self):
        if self.max_output_tokens >= self.model_context_tokens:
            raise ValueError("输出预留必须小于模型上下文容量")
        return self


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def load_leaf(relative: str):
    # service/__init__.py会加载业务依赖；按文件加载三个无业务依赖的叶子模块。
    # 新版读取器引用 service.web_document；仅建立命名空间，跳过业务 __init__。
    if "service" not in sys.modules:
        package = types.ModuleType("service")
        package.__path__ = [str(BACKEND / "app/service")]
        sys.modules["service"] = package
    path = BACKEND / "app/service" / relative
    name = "_deepagents_shared_" + path.stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class Journal:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def record(self, kind, **payload):
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str) + "\n")
            handle.flush()


class BudgetExceeded(RuntimeError):
    pass


class UsageLedger(BaseCallbackHandler):
    """回调涵盖主模型及框架内部调用；不以最终消息列表估算全部用量。"""
    raise_error = True

    def __init__(self, settings, journal):
        self.settings, self.journal = settings, journal
        self.calls = {}
        self.tool_calls = 0
        self.lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self.lock:
            if len(self.calls) >= self.settings.max_model_calls:
                raise BudgetExceeded("达到整任务模型调用上限")
            self.calls[str(run_id)] = {"status": "started", "usage": None}
        self.journal.record("model_input", run_id=str(run_id),
                            messages=[[message_to_dict(m) for m in batch] for batch in messages])

    def on_llm_end(self, response, *, run_id, **kwargs):
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
        if usage is None:
            raw = (response.llm_output or {}).get("token_usage", {})
            if "prompt_tokens" in raw and "completion_tokens" in raw:
                usage = {"input_tokens": raw["prompt_tokens"], "output_tokens": raw["completion_tokens"]}
        if usage is not None and not all(
            isinstance(usage.get(k), int) and not isinstance(usage[k], bool) and usage[k] >= 0
            for k in ("input_tokens", "output_tokens")
        ):
            usage = None
        with self.lock:
            self.calls[str(run_id)] = {"status": "completed", "usage": usage}
        self.journal.record("model_output", run_id=str(run_id), message=message_to_dict(message), usage=usage)

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self.lock:
            if str(run_id) in self.calls:
                self.calls[str(run_id)]["status"] = "error"
        self.journal.record("model_error", run_id=str(run_id), error_type=type(error).__name__)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        with self.lock:
            self.tool_calls += 1
        self.journal.record("tool_input", run_id=str(run_id), name=(serialized or {}).get("name"), input=input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.journal.record("tool_output", run_id=str(run_id), output=str(output))

    def summary(self, search_requests=0):
        rows = list(self.calls.values())
        usages = [r["usage"] for r in rows if r["usage"] is not None]
        complete = bool(rows) and len(usages) == len(rows) and all(r["status"] == "completed" for r in rows)
        input_tokens = sum(u["input_tokens"] for u in usages)
        output_tokens = sum(u["output_tokens"] for u in usages)
        cfg = self.settings
        cost = None
        if complete and cfg.input_price_per_million is not None and cfg.output_price_per_million is not None:
            cost = (input_tokens * cfg.input_price_per_million + output_tokens * cfg.output_price_per_million) / 1e6
        search_cost = (search_requests * cfg.search_price_per_request
                       if cfg.search_price_per_request is not None else None)
        return {"model_calls": len(rows), "tool_calls": self.tool_calls,
                "reported_input_tokens": input_tokens, "reported_output_tokens": output_tokens,
                "usage_complete": complete, "calls_missing_usage": len(rows) - len(usages),
                "model_cost_estimate": cost, "search_cost_estimate": search_cost,
                "cost_estimate": cost + search_cost if cost is not None and search_cost is not None else None,
                "currency": cfg.currency, "billing_verified": False}


class WebTools:
    def __init__(self, settings, journal, search, reader):
        self.settings, self.journal, self.search, self.reader = settings, journal, search, reader
        self.search_requests = self.page_reads = 0
        self.search_failures = self.read_failures = 0
        self.visited_urls = set()
        self.documents = {}
        self.verification_requests = 0
        self.verification_reports = []
        self.verification_lock = asyncio.Lock()
        self.scope = load_leaf("deep_research_v2/source_scope.py")
        self.reader.allowed_domains = list(settings.allowed_domains)

    async def search_web(self, query: str, domain: str = "") -> str:
        """搜索网络。配置域名范围时domain必须取允许列表；全网模式留空。"""
        if self.settings.allowed_domains and domain not in self.settings.allowed_domains:
            return json.dumps({"error": "域名不在允许列表中"}, ensure_ascii=False)
        if self.search_requests >= self.settings.max_search_requests:
            raise BudgetExceeded("达到搜索请求上限")
        self.search_requests += 1
        query = self.scope.strip_site_filters(query)
        if self.settings.allowed_domains:
            query = self.scope.scoped_query(domain, query)
        result = await asyncio.to_thread(self.search.search, query)
        if result.get("error"):
            self.search_failures += 1
            filtered = {"error": True, "message": "搜索服务失败"}
        else:
            values = self.search.extract_search_results(result)
            filtered = {"results": self.scope.filter_results_in_scope(values, [domain] if self.settings.allowed_domains else [])}
        self.journal.record("search_snapshot", query=query, result=filtered)
        return json.dumps(filtered, ensure_ascii=False)

    async def read_web(self, url: str) -> str:
        """读取来源正文，支持网页和PDF；仅允许研究范围内的URL，跳转也受相同限制。"""
        if not self.scope.url_in_scope(url, self.settings.allowed_domains):
            return json.dumps({"error": "URL不在允许范围"}, ensure_ascii=False)
        if self.page_reads >= self.settings.max_page_reads:
            raise BudgetExceeded("达到网页读取上限")
        self.page_reads += 1
        result = await self.reader.read_url(url)
        if result.success:
            self.visited_urls.add(result.final_url)
        else:
            self.read_failures += 1
        value = result.to_dict()
        if result.success:
            self.documents[url] = self.documents[result.final_url] = value
        self.journal.record("read_snapshot", result=value)
        return json.dumps(value, ensure_ascii=False)

    async def verify_citations(self, claims: list[dict]) -> str:
        """核验已读取原文对论断的支持。claims每项含claim和references数组；每个reference含url和quote（逐字原文）。没有引用仍保留references:[]。工具从网页快照回取quote，拒绝未读取URL或伪造引文；结果只覆盖提交的论断，不代表全报告正确。"""
        if not self.settings.citation_verification:
            return json.dumps({"status": "not_executed"}, ensure_ascii=False)
        async with self.verification_lock:
            if self.verification_requests >= self.settings.max_verification_requests:
                return json.dumps({"status": "incomplete", "reason": "引用核验请求达到上限"}, ensure_ascii=False)
            if not claims or len(claims) > 32:
                return json.dumps({"status": "incomplete", "reason": "每批提交1至32条论断"}, ensure_ascii=False)
            bound = []
            for index, item in enumerate(claims):
                refs, errors = [], []
                if not isinstance(item, dict) or not isinstance(item.get("claim"), str) or not item["claim"].strip():
                    return json.dumps({"status": "incomplete", "reason": "论断格式错误"}, ensure_ascii=False)
                for ref in item.get("references", []):
                    if not isinstance(ref, dict):
                        errors.append("invalid_reference")
                        continue
                    document = self.documents.get(ref.get("url", ""))
                    quote = ref.get("quote")
                    content = (document or {}).get("content", "")
                    if not document or not isinstance(quote, str) or not quote.strip() or quote not in content:
                        errors.append("unread_url_or_quote_not_in_snapshot")
                        continue
                    start = content.index(quote)
                    refs.append({"source_id": document["final_url"], "quote": content[start:start + len(quote)],
                                 "start": start, "end": start + len(quote), "content_sha256": sha256(content.encode())})
                bound.append({"claim_id": f"claim-{index + 1}", "claim": item["claim"], "references": refs,
                              "binding_errors": errors})
            self.verification_requests += 1
            number = self.verification_requests
            folder = self.journal.path.parent
            request = folder / f"引用核验请求-{number:03d}.json"
            response = folder / f"引用核验结果-{number:03d}.json"
            write_json(request, {"claims": bound})
            try:
                report = await self._run_verifier(request, response)
            except Exception as exc:
                report = {"status": "incomplete", "execution_status": "error", "error": type(exc).__name__, "claims": []}
            for claim, row in zip(bound, report.get("claims", [])):
                row["claim_id"] = claim["claim_id"]
                row["binding_errors"] = claim["binding_errors"]
                if claim["binding_errors"]:
                    row.update(status="invalid_citation", needs_review=True)
            report.update(scope="submitted_claims_only", all_claims_checked=False,
                          submitted_claims=len(bound), bound_claims=bound)
            report["supported_count"] = sum(r.get("status") == "supported" for r in report.get("claims", []))
            if any(c["binding_errors"] for c in bound):
                report["status"] = "rejected"
            self.verification_reports.append(report)
            write_json(response, report)
            self.journal.record("citation_verification", result=report)
            # 原文已在请求和网页快照保存，工具输出只回传判定，减少重复上下文。
            return json.dumps({k: v for k, v in report.items() if k != "bound_claims"}, ensure_ascii=False)

    async def _run_verifier(self, request, response):
        executable = self.settings.verification_python or str(BACKEND / ".venv-eval-py311/bin/python")
        with request.with_suffix(".日志.txt").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                executable, str(BACKEND / "eval/scripts/deepagents引用核验工作器.py"),
                str(request), str(response), stdout=log, stderr=log)
            try:
                async with asyncio.timeout(330):
                    code = await process.wait()
            except BaseException:
                process.kill()
                await process.wait()
                raise
        if code or not response.exists():
            raise RuntimeError("引用核验工作器未完成")
        return json.loads(response.read_text())


def build_agent(model, tools, system_prompt):
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    return create_deep_agent(model=model, tools=tools, system_prompt=system_prompt,
                             backend=StateBackend(), name="deepagents独立研究")


def final_article(state):
    messages = state.get("messages", [])
    if not messages or messages[-1].type != "ai" or messages[-1].tool_calls:
        raise ValueError("任务未产出终态报告")
    content = messages[-1].content
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("报告为空")
    return content.strip()


async def run_task(task, settings, folder, system_prompt, *, model=None, search=None, reader=None):
    journal = Journal(folder / "调用记录.jsonl")
    ledger = UsageLedger(settings, journal)
    started = time.monotonic()
    result = {"id": task["id"], "prompt": task["prompt"], "topic": task.get("topic"),
              "language": task.get("language"), "generator": "deepagents", "group": settings.group,
              "generation_model": settings.model, "article": "", "status": "failed"}
    gateway = None
    try:
        if model is None:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=settings.model, api_key=os.environ["DASHSCOPE_API_KEY"],
                               base_url=os.environ["DASHSCOPE_BASE_URL"], temperature=settings.temperature,
                               max_tokens=settings.max_output_tokens, max_retries=0, timeout=120,
                               profile={"max_input_tokens": settings.model_context_tokens,
                                        "max_output_tokens": settings.max_output_tokens, "tool_calling": True})
        search = search or load_leaf("web_search_service.py").WebSearchService(provider="bocha")
        reader = reader or load_leaf("web_reader_service.py").WebReaderService()
        gateway = WebTools(settings, journal, search, reader)
        available_tools = [gateway.search_web, gateway.read_web]
        if settings.citation_verification:
            available_tools.append(gateway.verify_citations)
        agent = build_agent(model, available_tools, system_prompt)
        async with asyncio.timeout(settings.timeout_seconds):
            state = await agent.ainvoke({"messages": [{"role": "user", "content": task["prompt"]}]},
                                       config={"recursion_limit": settings.recursion_limit, "callbacks": [ledger]})
        result["article"] = final_article(state)
        result["status"] = "completed"
        write_json(folder / "最终状态.json", {**state, "messages": [message_to_dict(m) for m in state["messages"]]})
        (folder / "研究报告.md").write_text(result["article"], encoding="utf-8")
    except (Exception, asyncio.CancelledError) as exc:
        # 错误正文可能带供应商请求细节，结果中仅记类型；模型异常不会被当作成功报告。
        status = "interrupted" if isinstance(exc, asyncio.CancelledError) else "timeout" if isinstance(exc, TimeoutError) else "failed"
        result.update(error=type(exc).__name__, status=status, article="")
    result.update(elapsed_seconds=time.monotonic() - started,
                  metrics=ledger.summary(gateway.search_requests if gateway else 0),
                  search_requests=gateway.search_requests if gateway else 0,
                  page_reads=gateway.page_reads if gateway else 0,
                  search_failures=gateway.search_failures if gateway else 0,
                  read_failures=gateway.read_failures if gateway else 0,
                  citation_verification={"enabled": settings.citation_verification,
                     "requests": gateway.verification_requests if gateway else 0,
                     "submitted_claims": sum(r.get("submitted_claims", 0) for r in gateway.verification_reports) if gateway else 0,
                     "supported_count": sum(r.get("supported_count", 0) for r in gateway.verification_reports) if gateway else 0,
                     "api_calls": [c for r in gateway.verification_reports for c in r.get("api_calls", [])] if gateway else [],
                     "all_claims_checked": False},
                  visited_urls=sorted(gateway.visited_urls) if gateway else [], artifact_dir=str(folder))
    write_json(folder / "结果.json", result)
    if result["status"] == "interrupted":
        raise asyncio.CancelledError
    return result


def package_versions():
    return dict(sorted((d.metadata["Name"], d.version) for d in distributions()))


def load_tasks(path, language="zh", ids=None, limit=1):
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for task in tasks:
        if str(task["id"]) in seen or not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ValueError("题目ID重复或prompt为空")
        seen.add(str(task["id"]))
    if ids and not set(ids) <= seen:
        raise ValueError("指定题号不在数据集中")
    return [t for t in tasks if (not ids or str(t["id"]) in ids)
            and (language == "all" or t.get("language") == language)][:limit]
