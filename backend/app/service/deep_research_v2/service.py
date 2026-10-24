# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""DeepResearch 服务入口：鉴权后的运行参数、图执行与 SSE 输出。

service 继续承接运行租约、事件账本、取消与超时；执行调度全部交给唯一主图。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from service.research_control import (
    ResearchCancelledError,
    cancellation_context,
    clear_cancel_flag,
    create_cancellation_token,
    release_cancellation_token,
    request_cancel,
)
from service.research_events import (
    DEFAULT_LEASE_SECONDS,
    ResearchEventEnvelope,
    ResearchEventPersistenceError,
    ResearchEventRecorder,
    ResearchRunLease,
    ResearchRunLeaseLost,
    acquire_research_run_lease,
    release_research_run_lease,
    renew_research_run_lease,
)

from .delivery import ResearchDeliveryError, make_commit_result
from .graph import agents_by_phase, build_graph, create_agents, graph_config, pack_state
from .checkpointer import for_execution
from .budget_ledger import PostgresRuntimeBudget, read_runtime_budget
from .storage_reconciliation import StorageCoordinator, StorageConsistencyError, artifacts, document_refs
from service.agent_history import PostgresAgentHistory
from .source_scope import configured_allowed_domains, normalize_allowed_domains
from .stage_runtime import LeaseLostDuringStage, StageBudgetExceeded, make_execute_stage
from .state import GRAPH_VERSION, create_initial_state

try:
    from config.llm_config import get_config
except ImportError:  # pragma: no cover - 兼容以 app 为根的导入
    from app.config.llm_config import get_config

try:
    from service.eval.tracer import flush_pending, trace_context
except ImportError:  # pragma: no cover - 评估模块缺失时退化为无埋点
    from contextlib import nullcontext

    def trace_context(**kwargs):
        return nullcontext()

    def flush_pending():
        pass

logger = logging.getLogger("DeepResearchV2Service")


class ResearchStatePausedError(RuntimeError):
    """图停在中途；上层按暂停/待恢复处理，不得宣布完成。"""


class LegacyRunNotResumableError(ValueError):
    """旧版本运行没有本图可用的检查点，必须由用户明确重新发起。"""


def encode_sse(payload: Dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _snapshot_resume_state(snapshot: Any) -> Dict[str, Any]:
    """区分图续跑与终态交付；空 next 可能仍有待合并的成功结果。"""
    research = (snapshot.values or {}).get("research") or {}
    pending_nodes = list(snapshot.next or ())
    saved_result_nodes = [
        task.name for task in getattr(snapshot, "tasks", ())
        if getattr(task, "result", None) is not None
        and getattr(task, "error", None) is None
        and not getattr(task, "interrupts", ())
    ]
    checkpoint_id = ((getattr(snapshot, "config", None) or {}).get("configurable") or {}).get("checkpoint_id")
    return {
        "pending_nodes": pending_nodes,
        "saved_result_nodes": saved_result_nodes,
        "resumable": bool(snapshot.values)
        and research.get("status") != "completed"
        and bool(pending_nodes or saved_result_nodes),
        # 只交付已提交的终态，不把尚待合并的 finish 输出当作终态检查点。
        "delivery_ready": bool(snapshot.values)
        and research.get("status") == "completed"
        and not pending_nodes
        and not getattr(snapshot, "tasks", ())
        and bool(checkpoint_id)
        and bool(str(research.get("final_report") or "").strip()),
    }


class DeepResearchV2Service:
    """唯一主链路的研究服务。"""

    def __init__(self, checkpointer: Any = None):
        """Args: checkpointer: 应用生命周期持有的 LangGraph saver。"""
        self.config = get_config()
        self.checkpointer = checkpointer

    # ------------------------------------------------------------------
    # 运行参数
    # ------------------------------------------------------------------
    def resolve_allowed_domains(
        self, allowed_domains: Optional[List[str]] = None
    ) -> List[str]:
        """默认全网；部署配置或请求可以指定网站范围。"""
        configured = configured_allowed_domains()
        if not allowed_domains:
            return configured
        requested = normalize_allowed_domains(allowed_domains)
        unknown = [d for d in requested if configured and d not in configured]
        if unknown:
            raise ValueError(f"以下网站不在后端允许的来源范围内: {unknown}")
        return requested

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    async def research(
        self,
        *,
        query: str,
        session_id: str,
        user_id: str,
        run_id: str,
        turn_id: Optional[str] = None,
        resume: bool = False,
        research_context: Optional[Dict[str, Any]] = None,
        allowed_domains: Optional[List[str]] = None,
        citation_verification: Optional[Dict[str, Any]] = None,
        hypothesis_enabled: Optional[bool] = None,
        critic_enabled: Optional[bool] = None,
        agent_compression: Optional[Dict[str, Any]] = None,
        lease: Optional[ResearchRunLease] = None,
        persist_events: bool = True,
    ) -> AsyncGenerator[str, None]:
        """执行或恢复一次研究，按 SSE 输出。

        首次运行传入业务数据；技术中断后的续跑传 ``resume=True``，从已保存的
        执行位置继续，不重新提交初始研究输入。
        """
        if self.checkpointer is None:
            raise RuntimeError("未注入 LangGraph 检查点，拒绝启动研究")
        if critic_enabled is not None and type(critic_enabled) is not bool:
            raise ValueError("critic_enabled 必须是布尔值")

        active_lease = lease
        if persist_events and active_lease is None:
            active_lease = await asyncio.to_thread(
                acquire_research_run_lease,
                run_id=run_id, session_id=session_id, user_id=user_id,
            )
        if active_lease and (
            active_lease.run_id != run_id or active_lease.session_id != session_id
        ):
            raise ValueError("ResearchRunLease 与当前 run/session 不匹配")

        recorder = ResearchEventRecorder(
            session_id=session_id, run_id=run_id, turn_id=turn_id,
            user_id=user_id, persist=persist_events, lease=active_lease,
        )
        commit_result = make_commit_result(
            recorder=recorder, lease=active_lease, run_id=run_id,
            session_id=session_id, user_id=user_id, turn_id=turn_id,
        )

        lease_stop = asyncio.Event()
        lease_lost = asyncio.Event()

        async def maintain_lease() -> None:
            if not active_lease:
                return
            interval = max(1.0, DEFAULT_LEASE_SECONDS / 3)
            while not lease_stop.is_set():
                try:
                    await asyncio.wait_for(lease_stop.wait(), timeout=interval)
                    return
                except asyncio.TimeoutError:
                    if not await asyncio.to_thread(
                        renew_research_run_lease, active_lease
                    ):
                        lease_lost.set()
                        request_cancel(run_id, session_id)
                        return

        async def lease_valid() -> bool:
            return not lease_lost.is_set()

        lease_task = asyncio.create_task(maintain_lease()) if active_lease else None
        terminal_status = "paused"

        clear_cancel_flag(run_id, session_id)
        cancellation_token = create_cancellation_token(run_id, session_id)
        evidence_service = None

        try:
            execution_checkpointer = for_execution(self.checkpointer, active_lease) if persist_events else self.checkpointer
            if persist_events:
                execution_checkpointer.event_turn_id = turn_id
            budget_store = PostgresRuntimeBudget(execution_checkpointer.conn, active_lease) if persist_events else None
            storage = StorageCoordinator(execution_checkpointer.conn, run_id, lease=active_lease) if persist_events else None
            if storage is not None:
                execution_checkpointer.storage_coordinator = storage
            from service.session_evidence_rag import SessionEvidenceRAG, SessionEvidenceSettings
            evidence_settings = SessionEvidenceSettings.from_env()
            if persist_events and user_id and evidence_settings.enabled:
                evidence_service = SessionEvidenceRAG(user_id, session_id, settings=evidence_settings)
            # 每个研究创建自己的 Agent 实例；共享的只有连接池等线程安全依赖。
            agents = create_agents(self.config)
            execute_stage = make_execute_stage(
                run_id=run_id, session_id=session_id,
                check_lease=lease_valid, agents_of=agents_by_phase(agents),
                budget_store=budget_store,
            )
            graph = build_graph(
                agents, execution_checkpointer,
                run_id=run_id, user_id=user_id, session_id=session_id,
                execute_stage=execute_stage,
                history_store=PostgresAgentHistory() if persist_events else None,
                history_fencing_token=active_lease.fencing_token if active_lease else None,
                evidence_service=evidence_service,
                storage_coordinator=storage,
            )
            config = graph_config(run_id)

            delivery_snapshot = await graph.aget_state(config) if resume else None
            if delivery_snapshot is not None and storage is not None:
                raw = await execution_checkpointer.aget_tuple(delivery_snapshot.config or config)
                await storage.require_consistent(delivery_snapshot, raw)
                pending = [artifact for _, _, value in (raw.pending_writes if raw else []) for artifact in artifacts(value)]
                index_reconciliation = []
                if evidence_service is not None and delivery_snapshot.values:
                    index_reconciliation = await evidence_service.reconcile(delivery_snapshot.values["research"], pending)
                elif any(document_refs((delivery_snapshot.values or {}).get("research") or {})) or any(
                    (artifact.get("storage_manifest") or {}).get("dependencies", {}).get("documents") for artifact in pending):
                    raise StorageConsistencyError("检查点依赖外置证据，请启用会话证据服务后恢复")
                # 索引修复不改变原始数据。修复后再校验，防止期间过期/被清理。
                audit = await storage.require_consistent(delivery_snapshot, raw)
                audit["indexes"] = index_reconciliation
                observation = await asyncio.to_thread(recorder.record, {
                    "type": "storage_reconciled", "run_id": run_id,
                    "content": "恢复依赖已核对", "storage_reconciliation": audit,
                })
                yield encode_sse(observation.to_wire())
            if delivery_snapshot is not None and delivery_snapshot.values and critic_enabled is not None:
                saved = delivery_snapshot.values["research"].get("critic_enabled", True)
                if saved != critic_enabled:
                    raise ValueError("恢复运行须沿用已保存的Critic开关；更改开关请新建运行")
            if delivery_snapshot is not None and _snapshot_resume_state(delivery_snapshot)["delivery_ready"]:
                # 图已完成：仅补做幂等交付，不进入 astream 或重新执行 Agent。
                envelope = await asyncio.to_thread(commit_result, delivery_snapshot)
                terminal_status = "completed"
                yield encode_sse(envelope.to_wire())
            else:
                graph_input = await self._graph_input(
                    graph, config,
                    resume=resume, query=query, session_id=session_id,
                    research_context=research_context,
                    allowed_domains=allowed_domains,
                    citation_verification=citation_verification,
                    hypothesis_enabled=hypothesis_enabled,
                    critic_enabled=critic_enabled,
                    agent_compression=agent_compression,
                    user_id=user_id, run_id=run_id,
                    evidence_service=evidence_service,
                )
                if storage is not None and graph_input is not None:
                    graph_input["research"] = await storage.seal(graph_input["research"])
                start_event = await asyncio.to_thread(recorder.record, {
                    "type": "research_resumed" if resume else "research_start",
                    "query": query,
                    "run_id": run_id,
                    "session_id": session_id,
                    "graph_version": GRAPH_VERSION,
                    "allowed_domains": list(
                        (graph_input or {}).get("research", {}).get("allowed_domains")
                        or await self._resumed_domains(graph, config)
                    ),
                })
                yield encode_sse(start_event.to_wire())

                with trace_context(
                    session_id=session_id, user_id=user_id, query=query,
                ), cancellation_context(cancellation_token):
                    try:
                        from .checkpoint_events import committed_stream
                        stream = graph.astream(
                            graph_input,
                            config=config,
                            stream_mode="custom",
                            durability="sync",
                        )
                        async for envelope in committed_stream(
                            stream, recorder, execution_checkpointer, after_seq=start_event.seq,
                        ):
                            if lease_lost.is_set():
                                raise ResearchRunLeaseLost(
                                    f"run {run_id} 的执行租约续租失败"
                                )
                            yield encode_sse(envelope.to_wire())
                    finally:
                        flush_pending()

                snapshot = await graph.aget_state(config)
                if snapshot.next:
                    # 仍有待执行节点，不能宣布正常完成。
                    raise ResearchStatePausedError(
                        f"run {run_id} 仍有待执行节点: {snapshot.next}"
                    )
                envelope = await asyncio.to_thread(commit_result, snapshot)
                terminal_status = "completed"
                yield encode_sse(envelope.to_wire())

        except ResearchCancelledError:
            terminal_status = "cancelled"
            envelope = await self._safe_record(recorder, {
                "type": "research_cancelled",
                "run_id": run_id,
                "message": "研究已暂停，执行位置已保存，可从检查点继续",
            })
            if envelope:
                yield encode_sse(envelope.to_wire())
        except asyncio.CancelledError:
            # 客户端断开 SSE 时也终止同一 run，避免后台 Agent 泄漏。
            request_cancel(run_id, session_id)
            raise
        except (ResearchEventPersistenceError, ResearchRunLeaseLost):
            terminal_status = "failed"
            request_cancel(run_id, session_id)
            logger.exception("研究事件可靠性保护触发，关闭 SSE: run=%s", run_id)
            raise
        except ResearchStatePausedError as exc:
            terminal_status = "paused"
            logger.warning("研究未执行到终态: %s", exc)
            envelope = await self._safe_record(recorder, {
                "type": "research_paused",
                "run_id": run_id,
                "content": str(exc),
            })
            if envelope:
                yield encode_sse(envelope.to_wire())
        except Exception as exc:
            terminal_status = "failed"
            code = (
                "research_budget_exceeded"
                if isinstance(exc, StageBudgetExceeded)
                else "research_lease_lost"
                if isinstance(exc, LeaseLostDuringStage)
                else "research_delivery_failed"
                if isinstance(exc, ResearchDeliveryError)
                else getattr(exc, "code", "research_failed")
            )
            logger.exception("研究执行失败: run=%s", run_id)
            envelope = await self._safe_record(recorder, {
                "type": "error", "run_id": run_id,
                "content": str(exc) or exc.__class__.__name__,
                "error_code": code,
            })
            if envelope:
                yield encode_sse(envelope.to_wire())
        finally:
            if evidence_service is not None:
                await evidence_service.close()
            lease_stop.set()
            if lease_task:
                await asyncio.gather(lease_task, return_exceptions=True)
            release_cancellation_token(run_id)
            if active_lease:
                await asyncio.to_thread(
                    release_research_run_lease, active_lease, terminal_status
                )

        # 事件和终态都已提交后才发送结束标记。
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # 输入构造与状态读取
    # ------------------------------------------------------------------
    async def _graph_input(
        self,
        graph: Any,
        config: Dict[str, Any],
        *,
        resume: bool,
        query: str,
        session_id: str,
        research_context: Optional[Dict[str, Any]],
        allowed_domains: Optional[List[str]],
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        citation_verification: Optional[Dict[str, Any]] = None,
        hypothesis_enabled: Optional[bool] = None,
        critic_enabled: Optional[bool] = None,
        agent_compression: Optional[Dict[str, Any]] = None,
        evidence_service: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """首跑返回业务输入；技术中断后的续跑返回 None 以沿用执行位置。"""
        if not resume:
            from service.context_policy import selected_policy
            policy = selected_policy()
            if agent_compression is not None:
                policy = policy.with_agent_compression(agent_compression)
            context_policy = policy.snapshot()
            initial = create_initial_state(
                query, session_id,
                allowed_domains=self.resolve_allowed_domains(allowed_domains),
                research_context=research_context,
                max_iterations=self.config.research.max_iterations,
                hypothesis_enabled=hypothesis_enabled,
                critic_enabled=True if critic_enabled is None else critic_enabled,
            )
            initial["context_policy"] = context_policy
            from .citation_verification import resolve_citation_verification_options
            options = resolve_citation_verification_options(citation_verification)
            initial["citation_verification_config"] = options.model_dump()
            if options.scout_facts or options.report_claims:
                from service.context_policy import NLISettings, runtime_nli_settings
                # 与摘要开关独立，只共用已冻结的模型参数；缺失时使用固定默认参数。
                initial["citation_verification_settings"] = NLISettings.model_validate(
                    context_policy.get("config", {}).get("nli") or runtime_nli_settings().model_dump()
                ).model_dump()
            from .followup_context import (parent_artifacts, digest,
                SessionInheritanceSettings, planning_context)
            from service.session_task_service import (authorize_artifact, validate_decision,
                SessionTaskDecision, SessionTaskClarification, report_hash)
            from .session_task_execution import prepare_task
            inheritance = SessionInheritanceSettings.from_env()
            initial["session_inheritance_config"] = inheritance.model_dump()
            initial["session_inheritance_config_hash"] = digest(inheritance.model_dump())
            task = (research_context or {}).get("session_task")
            parent = None
            if task:
                if task.get("status") == "clarify":
                    raise SessionTaskClarification(task["clarification_question"])
                if task.get("status") == "ready":
                    selected = task.get("selected_artifact")
                    # 重建合法计划，不接收调用方绕过能力判断的execution_plan。
                    trace = (research_context or {}).get("task_trace") or {}
                    task = validate_decision({k: task[k] for k in SessionTaskDecision.model_fields if k in task},
                                             initial["original_query"], [selected] if selected else [],
                                             turn_id=trace.get("sentence_index", {}).get("turn_id"))
                    selected = task["selected_artifact"]
                    if selected:
                        if not inheritance.enabled:
                            raise SessionTaskClarification("当前关闭了会话成果继承，请开启后再修改旧报告。")
                        await asyncio.to_thread(authorize_artifact, user_id, session_id,
                                                selected["run_id"], selected["report_hash"])
                        parent_id = selected["run_id"]
                        parent = parent_artifacts(await graph.aget_state(graph_config(parent_id)), session_id)
                        if evidence_service is not None:
                            await evidence_service.attach(parent, run_id=parent_id)
                        if report_hash(parent["final_report"]) != selected["report_hash"]:
                            raise ValueError("目标报告的检查点与已交付内容不一致")
                        initial["parent_run_id"] = parent_id
                        initial["parent_report_hash"] = parent["report_hash"]
                        initial["parent_checkpoint_id"] = parent.get("checkpoint_id")
                        if allowed_domains is None and parent.get("allowed_domains"):
                            initial["allowed_domains"] = self.resolve_allowed_domains(parent["allowed_domains"])
                        initial["session_planning_context"] = planning_context(parent, parent_id, inheritance.planning_context_tokens)
                if task.get("topic_relation") == "new" or task.get("status") == "fallback":
                    initial["conversation_context"] = []
                    initial["memory_context"] = []
                prepare_task(initial, task, parent)
                initial["session_task"] = task
                initial["session_task_hash"] = digest(task)
                initial["session_task_metrics"] = (research_context or {}).get("task_metrics") or {}
            elif any((research_context or {}).get(key) for key in ("conversation_context", "memory_context", "is_follow_up")):
                raise SessionTaskClarification("追问须先由会话层构建并提交本轮任务，不能交由Architect解析旧契约。")
            if evidence_service is not None:
                await evidence_service.attach(initial, parent=parent, run_id=run_id)
                await evidence_service.commit_stage(initial)
            return pack_state(initial)

        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            raise LegacyRunNotResumableError(
                "没有可恢复的图状态；该运行属于旧版本，请重新发起研究"
            )
        research = snapshot.values.get("research") or {}
        if "pending_contract_update" in (research.get("research_contract") or {}):
            raise LegacyRunNotResumableError("检查点包含已停用的Architect旧追问契约更新，请先在会话层重新构建本轮任务")
        if agent_compression is not None:
            from service.context_policy import ContextPolicy
            saved = research.get("context_policy")
            if not saved or ContextPolicy.from_snapshot(saved).with_agent_compression(agent_compression).fingerprint != saved["config_hash"]:
                raise ValueError("恢复运行须沿用已保存的逐 Agent 压缩配置；更改配置请新建运行")
        if research.get("session_inheritance_config"):
            from .followup_context import SessionInheritanceSettings, digest
            SessionInheritanceSettings.model_validate(research["session_inheritance_config"])
            if digest(research["session_inheritance_config"]) != research.get("session_inheritance_config_hash"):
                raise ValueError("会话成果继承配置快照哈希不匹配")
        from .hypothesis_policy import hypotheses_enabled, resolve_hypothesis_enabled
        saved_hypothesis_enabled = hypotheses_enabled(research)
        if critic_enabled is not None and (
            type(critic_enabled) is not bool
            or critic_enabled != research.get("critic_enabled", True)
        ):
            raise ValueError("恢复运行须沿用已保存的Critic开关；更改开关请新建运行")
        if hypothesis_enabled is not None and resolve_hypothesis_enabled(hypothesis_enabled) != saved_hypothesis_enabled:
            raise ValueError("恢复运行须沿用已保存的假设机制开关；更改开关请新建运行")
        from .citation_verification import CitationVerificationOptions
        saved_options = CitationVerificationOptions.model_validate(research.get("citation_verification_config") or {})
        if saved_options.scout_facts or saved_options.report_claims:
            from service.context_policy import NLISettings
            NLISettings.model_validate(research.get("citation_verification_settings") or {})
        if citation_verification is not None and CitationVerificationOptions.model_validate(citation_verification) != saved_options:
            raise ValueError("恢复运行须沿用已保存的引用核验开关；更改开关请新建运行")
        if research.get("context_policy"):
            from service.context_policy import ContextPolicy
            ContextPolicy.from_snapshot(research["context_policy"]).require_supported()
        if research.get("session_task"):
            from .followup_context import digest
            if digest(research["session_task"]) != research.get("session_task_hash"):
                raise ValueError("会话任务决策快照哈希不匹配")
        saved_graph_version = str(research.get("graph_version") or "")
        # 完成事件与执行产物的事务协议要求新版状态，不从旧图静默续跑。
        legacy_router = research.get("router_config") or {}
        if legacy_router.get("enabled") or set(snapshot.next) & {"followup_route", "user_revision"}:
            raise LegacyRunNotResumableError("检查点包含已删除的追问路由，请重新发起研究")
        if saved_graph_version != GRAPH_VERSION:
            raise LegacyRunNotResumableError(
                f"检查点图版本 {research.get('graph_version')!r} 与当前 "
                f"{GRAPH_VERSION} 不一致，请重新发起研究"
            )
        resume_state = _snapshot_resume_state(snapshot)
        if not resume_state["resumable"]:
            if research.get("status") == "completed":
                raise ValueError("图已执行结束，应读取报告或补做幂等交付")
            raise ResearchStatePausedError(
                "研究尚未完成，但检查点没有待执行节点或可恢复的节点结果，请核查断点"
            )
        if not snapshot.next:
            # aget_state会应用pending writes并从next排除已有结果的节点，
            # 但尚未推进到下一步；空next不等于业务完成。由LangGraph恢复
            # 已保存的节点结果并调度后继节点，不重建输入或手工修改检查点。
            logger.info("恢复已保存的节点结果并推进后继节点: run=%s, nodes=%s",
                        run_id, resume_state["saved_result_nodes"])
        return None

    async def _resumed_domains(self, graph: Any, config: Dict[str, Any]) -> List[str]:
        """恢复时保持同一次运行的来源范围，不悄悄替换。"""
        snapshot = await graph.aget_state(config)
        return list((snapshot.values.get("research") or {}).get("allowed_domains") or [])

    async def read_state(self, run_id: str) -> Dict[str, Any]:
        """按 thread 读取最新检查点；调用方必须先完成 run 的归属校验。

        thread_id 只负责定位，不证明访问权限。
        """
        if self.checkpointer is None:
            raise RuntimeError("未注入 LangGraph 检查点")
        graph = build_graph(
            create_agents(self.config), self.checkpointer,
            run_id=run_id, user_id=None, session_id="",
            execute_stage=make_execute_stage(run_id=run_id, session_id=""),
        )
        snapshot = await graph.aget_state(graph_config(run_id))
        research = dict((snapshot.values or {}).get("research") or {})
        reconciliation = {"status": "not_persistent", "blocked": False}
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        if isinstance(self.checkpointer, AsyncPostgresSaver):
            reconciliation = await StorageCoordinator(self.checkpointer.conn, run_id).inspect(
                snapshot, await self.checkpointer.aget_tuple(snapshot.config or graph_config(run_id)))
            live_budget = await read_runtime_budget(self.checkpointer.conn, run_id)
            if live_budget is not None:
                research["runtime_budget"] = live_budget
                research["node_attempts"] = dict(live_budget["node_attempts"])
        resume_state = _snapshot_resume_state(snapshot)
        if reconciliation["blocked"]:
            resume_state.update(resumable=False, delivery_ready=False)
        return {
            "run_id": run_id,
            "exists": bool(snapshot.values),
            "graph_version": research.get("graph_version", ""),
            **resume_state,
            "storage_reconciliation": reconciliation,
            "phase": research.get("phase", ""),
            "status": research.get("status", ""),
            "completion_reason": research.get("completion_reason", ""),
            "iteration": research.get("iteration", 0),
            "runtime_budget": research.get("runtime_budget", {}),
            "quality_verdict": research.get("quality_verdict", "not_reviewed"),
            "citation_verification": research.get("citation_verification", {}),
            "scout_citation_verification": research.get("scout_citation_verification", {}),
            "citation_repair": research.get("citation_repair", {}),
            "citation_verification_config": research.get("citation_verification_config", {"scout_facts": False, "report_claims": False}),
            "hypothesis_enabled": research.get("hypothesis_enabled", True),
            "allowed_domains": list(research.get("allowed_domains") or []),
            "final_report": research.get("final_report", ""),
            "outline": research.get("outline", []),
            "research": research,
        }

    @staticmethod
    async def _safe_record(
        recorder: ResearchEventRecorder, payload: Dict[str, Any]
    ) -> Optional[ResearchEventEnvelope]:
        """终态事件本身保存失败时不再抛新异常盖掉原始错误。"""
        try:
            return await asyncio.to_thread(recorder.record, payload)
        except Exception:
            logger.exception("终态事件保存失败: %s", payload.get("type"))
            return None


def create_service(checkpointer: Any = None) -> DeepResearchV2Service:
    """工厂函数：创建服务实例。"""
    return DeepResearchV2Service(checkpointer=checkpointer)
