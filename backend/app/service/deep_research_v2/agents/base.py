# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
DeepResearch V2.0 - Agent 基类

所有专家Agent的基类，提供通用的LLM调用、日志记录等功能。
"""

import json
import logging
import asyncio
import random
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from service.llm_client import (
    create_async_openai_client,
    default_agent_max_attempts,
    default_fallback_model,
    default_max_output_tokens,
)
from service.context_policy import policy_for_scope
from service.research_control import (
    ResearchCancelledError,
    get_current_cancellation_token,
)
from ..state import ResearchState, AgentLog
from ..progress import progress_counts
from service.agent_history import current_history_scope, record_agent_message, content_hash

try:
    from service.eval.tracer import (
        record_llm_call,
        attach_parse_result,
        flush_pending,
    )
except ImportError:  # pragma: no cover - 评估模块缺失时退化为无埋点
    def record_llm_call(*args, **kwargs):
        pass

    def attach_parse_result(*args, **kwargs):
        pass

    def flush_pending(*args, **kwargs):
        pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')


class BaseAgent(ABC):
    """
    Agent 基类

    所有专家Agent继承此类，实现特定的 process 方法。
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max"
    ):
        self.name = name
        self.role = role
        self.model = model
        # 必须带超时：不设的话 SDK 默认 read=600s + retries=2，
        # 单次调用最坏能挂 1800 秒。实测出现过 3547 秒的挂起，
        # 直接把整批评测拖死。详见 service/llm_client.py 的说明。
        # DeepResearch 在应用层显式记录每一次重试，所以关闭 SDK 的隐藏重试。
        self.client = create_async_openai_client(
            api_key=llm_api_key,
            base_url=llm_base_url,
            max_retries=0,
        )
        self.llm_call_count = 0
        self.llm_total_tokens = 0
        self.logger = logging.getLogger(f"Agent.{name}")

    @abstractmethod
    async def process(self, state: ResearchState) -> ResearchState:
        """
        处理状态并返回更新后的状态

        Args:
            state: 当前研究状态

        Returns:
            更新后的状态
        """
        pass

    def _output_limit(self):
        scope = current_history_scope()
        if scope is not None and scope.context_policy:
            return policy_for_scope(scope).output_limits[self.name]
        return default_max_output_tokens(self.name)

    async def _record_history_read_error(self, request, code, message, call_id, **details):
        """模型回读参数错误留痕并反馈，持久化失败仍向上传播。"""
        row = await record_agent_message(self.name, "history_read_error", {
            "error_code": code, "error_type": "ValueError",
            "source_id": request.get("source_id"), "request": request,
            "message": message, "recoverable": True, **details,
        }, call_id)
        self.logger.warning("历史回读未执行，记录后继续当前任务: %s", message)
        return row

    async def call_llm(
        self, system_prompt: str, user_prompt: str, json_mode: bool = True,
        temperature: float = 0.3, max_tokens: int = 16000,
    ) -> str:
        """原始输入与实际发送视图分开记录，摘要不会覆盖原始输入。"""
        scope = current_history_scope()
        call_id = str(uuid.uuid4())
        from service.session_evidence_rag import fit_evidence_prompt
        user_prompt = fit_evidence_prompt(system_prompt, user_prompt, scope, self.name,
                                         min(max_tokens, self._output_limit()))
        if scope is None:
            from service.agent_context import (
                is_evidence_batch_input, protected_agent_messages,
                scoped_input_budget, ContextBudgetExceeded,
            )
            if is_evidence_batch_input():
                from service.qwen_token_counter import get_qwen_token_counter
                actual = get_qwen_token_counter().count_messages(
                    protected_agent_messages(None, system_prompt, user_prompt))
                budget = scoped_input_budget(None, self.name, min(max_tokens, self._output_limit()))
                if actual > budget:
                    raise ContextBudgetExceeded(budget, actual)
            return await self._call_llm_impl(system_prompt, user_prompt, json_mode, temperature, max_tokens)
        policy = policy_for_scope(scope, self.name)
        await record_agent_message(self.name, "llm_input", {
            "system": system_prompt, "content": user_prompt,
            "contract": scope.contract,
        }, call_id)
        try:
            from service.agent_context import AgentCallContext, prepare_agent_messages
            call_context = AgentCallContext()
            read_attempt = 0
            while True:
                # 与工具/模型图中的请求中间件一致：回读后的每一次请求也重新准备。
                messages = await prepare_agent_messages(
                    self, scope, system_prompt, user_prompt, call_id,
                    min(max_tokens, self._output_limit()), call_context=call_context,
                )
                result = await self._call_llm_impl(
                    system_prompt, user_prompt, json_mode, temperature, max_tokens,
                    history_call_id=call_id, prepared_messages=messages,
                )
                try:
                    parsed = json.loads(result)
                except (ValueError, TypeError):
                    parsed = None
                read = parsed.get("_history_read") if isinstance(parsed, dict) else None
                if not isinstance(read, dict):
                    await record_agent_message(self.name, "llm_output", {"content": result}, call_id)
                    return result
                from service.agent_context import is_evidence_batch_input
                if is_evidence_batch_input():
                    raise ValueError("独立正文提取批次不允许追加历史回读；仅引用本批原句")
                intermediate = await record_agent_message(self.name, "llm_intermediate", {"content": result}, call_id)
                source_id = read.get("source_id")
                try:
                    if not isinstance(source_id, str):
                        raise ValueError("历史引用ID必须是UUID字符串")
                    uuid.UUID(source_id)
                except ValueError:
                    failure = await self._record_history_read_error(read, "invalid_history_source_id",
                        "历史引用ID不是合法UUID，本次回读未执行。", call_id)
                    call_context.records.extend([intermediate, failure])
                    continue
                if policy.history_read_max_calls is not None and read_attempt >= policy.history_read_max_calls:
                    raise ValueError("历史回读次数超限")
                start, chars = read.get("start", 0), read.get("chars", min(4000, policy.history_read_max_chars))
                if type(start) is not int or type(chars) is not int or start < 0 or not 1 <= chars <= policy.history_read_max_chars:
                    failure = await self._record_history_read_error(read, "invalid_history_read_range",
                        "历史回读范围不合法，本次回读未执行。", call_id,
                        max_chars=policy.history_read_max_chars)
                    call_context.records.extend([intermediate, failure])
                    continue
                row = await scope.store.read_source(scope, self.name, source_id)
                if row["content_hash"] != content_hash(row["payload"]):
                    raise ValueError("历史回读快照哈希不一致")
                from service.agent_context import source_text
                from service.verification_checks import source_type
                full_text = source_text(row)
                fragment = {"read_id": str(uuid.uuid4()), "source_id": row["source_id"], "start": start,
                            "end": min(start + chars, len(full_text)), "total_chars": len(full_text),
                            "origin_kind": row["kind"], "origin_source_type": source_type(row),
                            "source_text_hash": content_hash(full_text), "text": full_text[start:start + chars]}
                if not fragment["text"]:
                    failure = await self._record_history_read_error(read, "empty_history_read_range",
                        "历史回读范围没有内容，请使用现有资料继续任务。", call_id, total_chars=len(full_text))
                    call_context.records.extend([intermediate, failure])
                    continue
                fragment["text_hash"] = content_hash(fragment["text"])
                read_record = await record_agent_message(self.name, "history_read", fragment, call_id)
                call_context.records.extend([intermediate, read_record])
                read_attempt += 1
        except BaseException as exc:
            await record_agent_message(self.name, "llm_error", {
                "error_type": type(exc).__name__, "message": str(exc),
            }, call_id)
            raise

    async def _call_llm_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 16000,
        *, history_call_id: Optional[str] = None,
        prepared_messages: Optional[List[Dict[str, str]]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """
        调用 LLM

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            json_mode: 是否强制JSON输出
            temperature: 温度参数
            max_tokens: 最大token数
            model_override: 本次调用单独选模，重试保持该模型，不改变 Agent 配置

        Returns:
            LLM 响应文本
        """
        start_time = time.time()
        max_tokens = min(max_tokens, self._output_limit())

        # 上一次调用如果没走 parse_json_response，这里兜底冲刷，避免记录悬挂
        flush_pending()
        request_model = model_override or self.model
        active_model = request_model

        try:
            kwargs = {
                "model": request_model,
                "messages": prepared_messages if prepared_messages is not None else [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            retryable_errors = (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
            max_attempts = default_agent_max_attempts()
            # 摘要等显式选模的调用重试仍使用指定模型，且不修改 Agent 本身。
            fallback_model = request_model if model_override else default_fallback_model()
            response = None
            active_model = request_model
            for attempt in range(max_attempts):
                attempt_started = time.time()
                runtime_ticket = None
                runtime_settled = False
                from service.deep_research_v2.budget_ledger import current_runtime_budget
                runtime_scope = current_runtime_budget()
                try:
                    # 首次使用任务主模型；瞬时故障后切换备用模型，避免对同一
                    # 故障端点/模型重复做无效请求。若两者相同则保持原模型。
                    active_model = (
                        fallback_model
                        if attempt > 0 and fallback_model
                        else request_model
                    )
                    kwargs["model"] = active_model
                    from service.verification_budget import current_budget
                    budget_ledger = current_budget()
                    budget_ticket = None
                    if budget_ledger is not None:
                        from service.qwen_token_counter import get_qwen_token_counter
                        budget_ticket = budget_ledger.reserve(active_model, get_qwen_token_counter().count_messages(kwargs["messages"]), kwargs["max_tokens"])
                        budget_ticket.update(call_id=history_call_id, attempt=attempt + 1)
                    if history_call_id:
                        from service.verification_trace import read_links
                        await record_agent_message(self.name, "llm_request", {
                            "attempt": attempt + 1, "request": kwargs,
                            "request_hash": content_hash(kwargs), "history_read_links": read_links(kwargs["messages"]),
                        }, history_call_id)
                    if runtime_scope is not None:
                        from service.qwen_token_counter import get_qwen_token_counter
                        runtime_store, runtime_attempt = runtime_scope
                        runtime_ticket = await runtime_store.reserve(
                            runtime_attempt, active_model,
                            get_qwen_token_counter().count_messages(kwargs["messages"]) + kwargs["max_tokens"],
                        )
                    self.llm_call_count = getattr(self, "llm_call_count", 0) + 1
                    token = get_current_cancellation_token()
                    if token:
                        token.raise_if_cancelled()
                    request_task = asyncio.create_task(
                        self.client.chat.completions.create(**kwargs)
                    )
                    cancel_task = asyncio.create_task(token.wait()) if token else None
                    try:
                        waiters = {request_task}
                        if cancel_task:
                            waiters.add(cancel_task)
                        done, _ = await asyncio.wait(
                            waiters,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if cancel_task and cancel_task in done:
                            raise ResearchCancelledError("研究已被用户取消")
                        response = await request_task
                        if runtime_ticket is not None:
                            usage = getattr(response, "usage", None)
                            usage_data = ({key: getattr(usage, key, None) for key in
                                           ("prompt_tokens", "completion_tokens", "total_tokens")} if usage else None)
                            await runtime_store.complete(runtime_attempt, runtime_ticket, usage_data)
                            runtime_settled = True
                    finally:
                        # 无论用户取消、浏览器断开还是请求报错，都回收两个子任务。
                        cleanup_tasks = [request_task]
                        if cancel_task:
                            cleanup_tasks.append(cancel_task)
                        for pending_task in cleanup_tasks:
                            if not pending_task.done():
                                pending_task.cancel()
                        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                    break
                except retryable_errors as exc:
                    if history_call_id:
                        await record_agent_message(self.name, "llm_attempt_error", {
                            "attempt": attempt + 1, "model": active_model,
                            "error_type": type(exc).__name__, "message": str(exc),
                        }, history_call_id)
                    if attempt == max_attempts - 1:
                        raise
                    # 失败的中间尝试也进入 AgentRun，便于定位抖动和限流。
                    record_llm_call(
                        agent_name=self.name,
                        model=active_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        output="",
                        temperature=temperature,
                        json_mode=json_mode,
                        duration_ms=int((time.time() - attempt_started) * 1000),
                        success=False,
                        error=exc,
                    )
                    retry_after = None
                    response_obj = getattr(exc, "response", None)
                    if response_obj is not None:
                        try:
                            retry_after = float(response_obj.headers.get("retry-after", ""))
                        except (TypeError, ValueError):
                            retry_after = None
                    delay = retry_after if retry_after is not None else 2 * (2 ** attempt)
                    delay += random.uniform(0, min(1.0, delay * 0.2))
                    error_code = {
                        RateLimitError: "rate_limited",
                        APITimeoutError: "llm_timeout",
                        APIConnectionError: "llm_connection_error",
                        InternalServerError: "provider_internal_error",
                    }.get(type(exc), "llm_transient_error")
                    self.logger.warning(
                        "LLM transient error code=%s，%.2fs 后重试 (%s/%s)，下一模型=%s: %s",
                        error_code,
                        delay,
                        attempt + 1,
                        max_attempts,
                        fallback_model or self.model,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(delay)
                finally:
                    if runtime_ticket is not None and not runtime_settled:
                        # 预留已先落盘；取消、供应商错误及进程中断均不会变成零消耗。
                        await runtime_store.complete(runtime_attempt, runtime_ticket, None)

            if response is None:  # pragma: no cover - 循环只可能返回或抛异常
                raise RuntimeError("LLM 调用未返回响应")

            content = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
            try:
                self.llm_total_tokens = (
                    getattr(self, "llm_total_tokens", 0) + int(total_tokens or 0)
                )
            except (TypeError, ValueError):
                pass
            duration = int((time.time() - start_time) * 1000)

            if history_call_id:
                usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {
                    key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
                if budget_ledger is not None and budget_ticket is not None:
                    budget_ledger.complete(budget_ticket, usage_dict)
                await record_agent_message(self.name, "llm_usage", {
                    "model": active_model, "usage": usage_dict, "attempt": attempt + 1,
                    "duration_ms": duration, "cost_usd": None, "cost_status": "unpriced",
                }, history_call_id)

            self.logger.info(f"LLM call completed in {duration}ms, response length: {len(content)}")

            # json_mode 下预期调用方随后会 parse_json_response，
            # 所以先暂存，等解析结果回填后再落库，保证一次调用只落一行
            record_llm_call(
                agent_name=self.name,
                model=active_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output=content or "",
                temperature=temperature,
                json_mode=json_mode,
                duration_ms=duration,
                usage=usage,
                success=True,
                defer_parse=bool(json_mode),
            )

            return content

        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            self.logger.error(f"LLM call failed: {e}")
            record_llm_call(
                agent_name=self.name,
                model=active_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output="",
                temperature=temperature,
                json_mode=json_mode,
                duration_ms=duration,
                success=False,
                error=e,
            )
            raise

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """安全解析JSON响应，处理markdown代码块和格式问题"""
        import re

        def fix_escaped_newlines(s: str) -> str:
            """修复过度转义的换行符"""
            # 处理多层转义: \\\\n -> \n, \\n -> \n
            s = s.replace('\\\\\\\\n', '\n')
            s = s.replace('\\\\n', '\n')
            # 处理可能的 \\r\\n
            s = s.replace('\\\\\\\\r', '\r')
            s = s.replace('\\\\r', '\r')
            return s

        def try_parse(s: str) -> Optional[Dict]:
            """尝试解析JSON，包含修复逻辑"""
            # 清理常见问题
            s = s.strip()
            # 移除可能的BOM
            if s.startswith('\ufeff'):
                s = s[1:]

            try:
                result = json.loads(s)
                # 成功解析后，修复值中的转义字符
                return self._fix_escaped_values(result)
            except json.JSONDecodeError:
                pass

            # 尝试修复常见JSON问题
            try:
                # 修复无效的JSON转义序列 \[ \] \# 等 (LLM经常产生这种错误)
                # 需要在字符串值中修复，但避免影响已转义的反斜杠
                # \\[ 是有效的(表示 \[ 字面量)，但 \[ 不是有效的JSON转义
                s = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', '', s)
                # 移除注释
                s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
                s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
                # 修复尾随逗号
                s = re.sub(r',(\s*[}\]])', r'\1', s)
                # 修复缺少逗号的情况（在 } 或 ] 后面缺少逗号）
                s = re.sub(r'([}\]])(\s*)([{\[])', r'\1,\2\3', s)
                # 修复没有引号的key
                s = re.sub(r'(\{|\,)\s*(\w+)\s*:', r'\1"\2":', s)
                result = json.loads(s)
                return self._fix_escaped_values(result)
            except json.JSONDecodeError:
                pass

            return None

        # 1. 先尝试直接解析
        result = try_parse(response)
        if result:
            self.logger.debug("Direct JSON parse succeeded")
            attach_parse_result(True, "direct")
            return result

        # 2. 尝试提取markdown代码块
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(code_block_pattern, response)
        if match:
            result = try_parse(match.group(1))
            if result:
                self.logger.debug("Extracted JSON from code block")
                attach_parse_result(True, "code_block")
                return result

        # 3. 尝试找到最外层的 {...}
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            result = try_parse(response[start:end+1])
            if result:
                self.logger.debug("Extracted JSON from braces")
                attach_parse_result(True, "braces")
                return result

        # 4. 最后尝试用更宽松的方式解析
        try:
            # 使用 ast.literal_eval 作为备选
            import ast
            # 将 true/false/null 转换为 Python 格式
            s = response
            s = re.sub(r'\btrue\b', 'True', s)
            s = re.sub(r'\bfalse\b', 'False', s)
            s = re.sub(r'\bnull\b', 'None', s)
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1:
                result = ast.literal_eval(s[start:end+1])
                if isinstance(result, dict):
                    self.logger.debug("Parsed using ast.literal_eval")
                    attach_parse_result(True, "literal_eval")
                    return result
        except Exception:
            # 同上：不能用裸 except，否则会吞掉 CancelledError，
            # 让外层的超时取消无法生效
            pass

        self.logger.error(f"JSON parse error, could not extract valid JSON")
        self.logger.warning(f"Raw response (first 800 chars): {response[:800]}")
        attach_parse_result(False, "failed")
        return {}

    def _fix_escaped_values(self, obj: Any, key: str = None) -> Any:
        """
        递归修复字典和列表中的转义字符

        注意：对于 'code' 字段，不处理转义，因为代码中的 \n 是有意义的转义序列
        """
        if isinstance(obj, dict):
            return {k: self._fix_escaped_values(v, key=k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._fix_escaped_values(item, key=key) for item in obj]
        elif isinstance(obj, str):
            # 对于代码字段，不进行转义处理
            # 因为代码中的 \n 应该保持为 \n（两个字符），而不是真正的换行
            if key in ('code', 'fixed_code', 'revised_content'):
                return obj

            # 对于其他字段，修复过度转义的换行符
            result = obj
            result = result.replace('\\\\n', '\n')
            result = result.replace('\\n', '\n')
            result = result.replace('\\\\r', '\r')
            result = result.replace('\\r', '\r')
            result = result.replace('\\\\t', '\t')
            result = result.replace('\\t', '\t')
            return result
        else:
            return obj

    def add_message(self, state: ResearchState, event_type: str, content: Any) -> None:
        """Agent 统一的进度事件出口。

        进度事件发往 LangGraph custom 流，由服务层落账本再推送 SSE。
        完成通知由节点/章节包装器暂存，随 checkpoint/task 结果同事务提交。
        ``_emit_event`` 是节点包装器为本次运行注入的函数引用，不持久化到
        图状态；调用它只表示"已发出这条进度观察"，既不等于数据库已保存，
        也不代表所在节点的检查点已提交。

        Args:
            state: 研究状态
            event_type: 事件类型
            content: 消息内容
        """
        message = {
            "type": event_type,
            "agent": self.name,
            "node": state.get("_node", ""),
            "iteration": int(state.get("iteration", 0) or 0),
            "attempt": int(state.get("_attempt", 1) or 1),
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "progress": progress_counts(state),
        }
        emit_event = state.get("_emit_event")
        if emit_event is None:
            # 单元测试和工具级定向测试会在没有图运行上下文时调用 Agent。
            self.logger.debug("[事件] 当前运行未注入事件出口: %s", event_type)
            return
        emit_event(message)

    def add_log(
        self,
        state: ResearchState,
        action: str,
        input_summary: str,
        output_summary: str,
        duration_ms: int,
        tokens_used: int = 0
    ) -> None:
        """添加执行日志"""
        log = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "action": action,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "duration_ms": duration_ms,
            "tokens_used": tokens_used
        }
        state["logs"].append(log)


class AgentRegistry:
    """Agent 注册表"""

    _agents: Dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        """注册Agent"""
        cls._agents[agent.name] = agent

    @classmethod
    def get(cls, name: str) -> Optional[BaseAgent]:
        """获取Agent"""
        return cls._agents.get(name)

    @classmethod
    def all(cls) -> Dict[str, BaseAgent]:
        """获取所有Agent"""
        return cls._agents.copy()
