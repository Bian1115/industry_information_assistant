# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""唯一 LangGraph 主图：规划 → 搜索 → 分析 → 写作 → 审核与返工。

图只提交可序列化的业务状态；运行身份、事件出口与 Agent 属于本次执行。
阶段保护由 execute_stage 提供，终态业务交付由服务层在检查点提交后完成。
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from service.tool_runtime import ToolRuntime
from service.agent_history import HistoryScope, bind_agent_history
from .agents import ChiefArchitect, CriticMaster, DataAnalyst, DeepScout, LeadWriter
from .deterministic_analysis import run_deterministic_analysis
from .state import ResearchPhase, citation_verification_status
from .progress import progress_counts
from .checkpoint_events import EVENTS_CHANNEL, event_intent, requires_commit

# 最长默认路径：首次五步 + 两次补搜/分析/写作/审核 + finish = 14 步。
# 这是异常循环的兜底，不代替 Critic 管理的业务返工次数。
RECURSION_LIMIT = 20
RUNTIME_ONLY_KEYS = frozenset({
    "_run_id", "_user_id", "_session_id", "_node", "_attempt", "_emit_event",
})
PHASE_AGENTS = {
    "init": "architect", "planning": "architect",
    "researching": "scout", "re_researching": "scout",
    "analyzing": "data_analyst", "writing": "writer", "revising": "writer",
    "reviewing": "critic",
}
PHASE_LABELS = {
    "init": "开始规划研究", "researching": "开始深度搜索",
    "analyzing": "开始数据分析", "writing": "开始撰写报告",
    "reviewing": "审核报告", "re_researching": "根据审核反馈补充搜索",
    "revising": "根据审核反馈修订报告",
}


class GraphState(TypedDict):
    """串行节点整体替换 research，不累加旧快照中的列表。"""

    research: Dict[str, Any]
    checkpoint_events: list[dict]


def create_agents(config: Any) -> Dict[str, Any]:
    """每次研究独立创建五个 Agent，复用部署中的各角色模型配置。"""
    api_key, base_url = config.api_key, config.base_url
    settings = config.research
    roles = config.agents
    runtime = ToolRuntime()
    return {
        "architect": ChiefArchitect(
            api_key, base_url, roles.architect.model,
            min_sections=settings.min_sections,
            max_sections=settings.max_sections,
            complex_max_sections=settings.complex_max_sections,
            temperature=roles.architect.temperature,
            max_tokens=roles.architect.max_tokens,
        ),
        "scout": DeepScout(
            api_key, base_url, config.search_api_key, roles.scout.model,
            tool_runtime=runtime, max_total_searches=settings.max_total_searches,
            search_provider=getattr(config, "search_provider", "bocha"),
            aliyun_search_mcp_api_key=getattr(config, "aliyun_search_mcp_api_key", api_key),
            aliyun_search_mcp_url=getattr(config, "aliyun_search_mcp_url", None),
            aliyun_search_mcp_tool_name=getattr(config, "aliyun_search_mcp_tool_name", "search_pro"),
        ),
        "data_analyst": DataAnalyst(
            api_key, base_url, roles.data_analyst.model, generate_charts=False,
        ),
        "writer": LeadWriter(
            api_key, base_url, roles.writer.model,
            target_report_chars=settings.target_report_chars,
            min_section_chars=settings.min_section_chars,
            max_section_chars=settings.max_section_chars,
            summary_chars=settings.summary_chars,
            outlook_chars=settings.outlook_chars,
            max_tokens=roles.writer.max_tokens,
        ),
        "critic": CriticMaster(api_key, base_url, roles.critic.model),
    }


def agents_by_phase(agents: Dict[str, Any]) -> Callable[[str], Any]:
    """运行保护层通过阶段查询本次 Agent 的预算计数器。"""
    return lambda phase: agents.get(PHASE_AGENTS.get(phase, ""))


def pack_state(work: Dict[str, Any]) -> GraphState:
    """剥离显式登记的运行字段，拒绝将其他对象静默写入检查点。"""
    saved = {key: value for key, value in work.items() if key not in RUNTIME_ONLY_KEYS}
    leaked = [key for key in saved if key.startswith("_")]
    if leaked:
        raise RuntimeError(f"未登记的运行期字段进入了图状态: {sorted(leaked)}")
    # 旧检查点可能仍包含已停用的临时消息队列，提交新状态时移除。
    saved.pop("messages", None)
    # 不使用 default=str：把函数等对象转成字符串会掩盖状态契约错误。
    json.dumps(saved, ensure_ascii=False, allow_nan=False)
    return {"research": saved, EVENTS_CHANNEL: []}


def make_node(
    action: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    phase: str,
    node_name: str,
    *,
    run_id: str,
    user_id: Optional[str],
    session_id: str,
    execute_stage: Callable[..., Awaitable[Dict[str, Any]]],
    history_store: Any = None,
    history_fencing_token: Optional[int] = None,
    evidence_service: Any = None,
    storage_coordinator: Any = None,
):
    """复制已提交状态，注入执行上下文，并在阶段保护内执行 Agent。"""
    async def node(graph_state: GraphState) -> GraphState:
        work = deepcopy(graph_state["research"])
        work.pop("messages", None)
        if work.get("context_policy"):
            from service.context_policy import ContextPolicy
            policy = ContextPolicy.from_snapshot(work["context_policy"])
            policy.require_supported()
            if history_store is None and policy.policy_version != "legacy-runtime-v1":
                raise ValueError("上下文消融必须启用独立历史存储，禁止无历史运行冒充消融")
        work["phase"] = phase
        completed_events = []
        stream_writer = get_stream_writer()

        def emit(event):
            if requires_commit(event):
                completed_events.append(event_intent(event))
            else:
                stream_writer(event)
        attempts = work.setdefault("node_attempts", {})
        attempts[node_name] = int(attempts.get(node_name, 0) or 0) + 1
        work.update({
            "_run_id": run_id, "_user_id": user_id, "_session_id": session_id,
            "_node": node_name, "_attempt": attempts[node_name],
            "_emit_event": emit,
        })
        if evidence_service is not None:
            await evidence_service.prepare_stage(work, phase, run_id)

        async def run_stage(state: Dict[str, Any]) -> Dict[str, Any]:
            # 只是进度观察；不得在节点提交前发送 checkpoint_saved 或完成事件。
            state["_emit_event"]({
                "type": "phase", "phase": "planning" if phase == "init" else phase,
                "content": PHASE_LABELS[phase], "node": node_name,
                "iteration": int(state.get("iteration", 0) or 0),
                "attempt": state["_attempt"], "timestamp": datetime.now().isoformat(),
                "progress": progress_counts(state),
            })
            scope = HistoryScope(
                store=history_store, run_id=run_id, session_id=session_id,
                user_id=user_id, node=node_name,
                contract=state.get("research_contract") or {},
                fencing_token=history_fencing_token,
                context_policy=state.get("context_policy"),
            ) if history_store is not None else None
            from service.session_evidence_rag import bind_session_evidence
            from .storage_reconciliation import bind_storage_coordinator
            with bind_agent_history(scope), bind_session_evidence(evidence_service), bind_storage_coordinator(storage_coordinator):
                return await action(state)

        result = await execute_stage(run_stage, work, phase)
        if not isinstance(result, dict):
            raise TypeError(f"{node_name} 节点未返回研究状态")
        if phase in {"writing", "revising"} and not str(
            result.get("final_report") or ""
        ).strip():
            raise RuntimeError(f"{node_name} 节点未产生报告")
        if evidence_service is not None:
            await evidence_service.commit_stage(result)
            from .state import GRAPH_VERSION
            result["graph_version"] = GRAPH_VERSION
        packed = pack_state(result)
        packed[EVENTS_CHANNEL] = completed_events + [event_intent({
            "type": "node_completed", "node": node_name,
            "iteration": int(result.get("iteration", 0) or 0),
            "attempt": work["_attempt"], "progress": progress_counts(result),
        })]
        if storage_coordinator is not None:
            packed["research"] = await storage_coordinator.seal(packed["research"], node=node_name)
        return packed

    return node


def route_after_review(state: GraphState) -> str:
    """Critic 决定质量结论、返工次数与方向；图负责校验并执行该方向。"""
    phase = state["research"]["phase"]
    routes = {
        ResearchPhase.RE_RESEARCHING.value: "re_research",
        ResearchPhase.REVISING.value: "revise",
        ResearchPhase.COMPLETED.value: "finish",
    }
    if phase not in routes:
        raise RuntimeError(f"审核返回未知阶段: {phase}")
    return routes[phase]


def route_entry(state: GraphState) -> str:
    plan = state["research"].get("execution_plan", "replan")
    if plan not in {"replan", "edit", "augment"}:
        raise ValueError("未支持的会话执行计划")
    return plan


def route_after_write(state: GraphState) -> str:
    """关闭审核时直接提交未审核终态；缺省仍沿用生产审核链。"""
    enabled = state["research"].get("critic_enabled", True)
    if type(enabled) is not bool:
        raise ValueError("critic_enabled 必须是布尔值")
    return "review" if enabled else "finish"


def finish(state: GraphState) -> GraphState:
    """只提交执行终态，完成事件由服务层完成幂等交付后发送。"""
    work = deepcopy(state["research"])
    from .report_constraints import check_body_length
    check_body_length(work)
    if not str(work.get("final_report") or "").strip():
        raise RuntimeError("结束节点未产生报告")
    verdict = work.get("quality_verdict")
    if route_after_write(state) == "finish":
        from .quality_pipeline import evaluate_report_gate
        gate = evaluate_report_gate(work)
        if not gate["passed"]:
            raise RuntimeError("未审核报告的基础发布检查失败")
        if verdict != "not_reviewed" or work.get("critic_feedback") or (
            int((work.get("node_attempts") or {}).get("review", 0))
        ):
            raise RuntimeError("关闭审核的运行不能包含已有审核结果")
        work["quality_score_status"] = "not_reviewed"
        work["quality_score_basis"] = "not_reviewed"
        reason = "critic_disabled"
    elif verdict == "pass":
        reason = "passed"
    elif verdict in {"needs_revision", "major_issues"} and (
        int(work.get("iteration", 0)) >= int(work.get("max_iterations", 0))
    ):
        reason = "revision_limit"
    else:
        raise RuntimeError("审核未通过且返工额度未耗尽，拒绝宣布完成")
    work["phase"] = ResearchPhase.COMPLETED.value
    work["status"] = "completed"
    work["completion_reason"] = reason
    work.setdefault("citation_verification", citation_verification_status())
    packed = pack_state(work)
    packed[EVENTS_CHANNEL] = [event_intent({
        "type": "node_completed", "node": "finish", "status": "completed",
        "completion_reason": reason,
    })]
    return packed


def build_graph(
    agents: Dict[str, Any],
    checkpointer: Any,
    *,
    run_id: str,
    user_id: Optional[str],
    session_id: str,
    execute_stage: Callable[..., Awaitable[Dict[str, Any]]],
    history_store: Any = None,
    history_fencing_token: Optional[int] = None,
    evidence_service: Any = None,
    storage_coordinator: Any = None,
    interrupt_before: Optional[list[str]] = None,
):
    """构建可恢复的主图；缺少检查点时拒绝回退到手写流程。"""
    if checkpointer is None:
        raise ValueError("未提供 LangGraph 检查点")

    async def analyze_action(work: Dict[str, Any]) -> Dict[str, Any]:
        work = await agents["data_analyst"].process(work)
        # 数据节点不再生成语义摘要；这里保留已有结构化输入的程序计算。
        run_deterministic_analysis(work)
        return work

    def wrap(action, phase, name):
        return make_node(
            action, phase, name, run_id=run_id, user_id=user_id,
            session_id=session_id, execute_stage=execute_stage,
            history_store=history_store,
            history_fencing_token=history_fencing_token,
            evidence_service=evidence_service,
            storage_coordinator=storage_coordinator,
        )

    async def write_action(work):
        work.pop("writing_mode", None)
        return await agents["writer"].process(work)

    builder = StateGraph(GraphState)
    builder.add_node("plan", wrap(agents["architect"].process, "init", "plan"))
    builder.add_node("research", wrap(agents["scout"].process, "researching", "research"))
    builder.add_node("analyze", wrap(analyze_action, "analyzing", "analyze"))
    builder.add_node("write", wrap(write_action, "writing", "write"))
    builder.add_node("edit", wrap(agents["writer"].process, "writing", "edit"))
    builder.add_node("supplement", wrap(agents["scout"].process, "re_researching", "supplement"))
    builder.add_node("review", wrap(agents["critic"].process, "reviewing", "review"))
    builder.add_node("re_research", wrap(agents["scout"].process, "re_researching", "re_research"))
    builder.add_node("revise", wrap(agents["writer"].process, "revising", "revise"))
    async def sealed_finish(state):
        result = finish(state)
        result["research"] = await storage_coordinator.seal(result["research"], node="finish")
        return result

    builder.add_node("finish", sealed_finish if storage_coordinator is not None else finish)
    builder.add_conditional_edges(START, route_entry, {"replan": "plan", "edit": "edit", "augment": "supplement"})
    builder.add_conditional_edges("edit", route_after_write, {"review": "review", "finish": "finish"})
    builder.add_edge("supplement", "analyze")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "analyze")
    builder.add_edge("analyze", "write")
    builder.add_conditional_edges("write", route_after_write, {"review": "review", "finish": "finish"})
    builder.add_edge("re_research", "analyze")
    builder.add_edge("revise", "review")
    builder.add_conditional_edges("review", route_after_review, {
        "re_research": "re_research", "revise": "revise", "finish": "finish",
    })
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def graph_config(run_id: str) -> Dict[str, Any]:
    """同一次运行复用 thread_id；与业务返工次数独立的执行步数上限。"""
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}
