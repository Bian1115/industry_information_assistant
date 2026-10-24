"""将已校验会话任务编译为研究输入；这里没有模型判断。"""
from copy import deepcopy

from service.session_task_service import commit_requirements, SessionTaskClarification
from .contract_updates import parent_contract, contract_query
from .followup_context import inherit_artifacts
from .quality_pipeline import evaluate_report_gate


def prepare_task(state, task, parent=None):
    if task.get("status") == "clarify":
        raise SessionTaskClarification(task["clarification_question"])
    if task.get("status") != "ready":
        state["execution_plan"] = "replan"
        return
    base = parent_contract(parent) if parent else {"contract_revision": 0, "active_constraints": {}}
    committed = commit_requirements(base, task, state["original_query"])
    state["research_contract"] = committed
    state["active_constraints"] = deepcopy(committed["active_constraints"])
    state["query"] = contract_query(committed)
    state["resolved_query"] = committed["query"]
    state["revision_instructions"] = list(committed["followup_instructions"])
    plan = task["execution_plan"]
    scope_fields = {"required_entities", "time_range", "geography", "critical_questions", "required_metrics", "required_calculations"}
    if any(x["field"] in scope_fields for x in task.get("requirement_changes", [])):
        plan = "replan"
        state["route_fallback_reason"] = "研究范围或计算要求变化，需要重新规划"
    usable = parent and parent.get("outline") and parent.get("draft_sections") and all(
        (parent["draft_sections"].get(x.get("id")) or "").strip() for x in parent["outline"])
    if plan != "replan" and not usable:
        plan = "replan"
        state["route_fallback_reason"] = "旧成果没有完整章节材料，先规划"
    if parent and plan == "replan":
        state["session_parent"] = deepcopy(parent)
        state["parent_research_contract"] = parent_contract(parent)
    elif parent:
        inherit_artifacts(state, parent)
        # 成果白名单含旧契约，继承后必须重装本轮已提交要求。
        state["research_contract"] = committed
        state["active_constraints"] = deepcopy(committed["active_constraints"])
        if not evaluate_report_gate(state)["passed"]:
            raise SessionTaskClarification("目标报告中的来源不符合本次网站范围，请明确是否改为按当前范围重新研究。")
        for section in state["outline"]:
            section["status"] = "final"
        if plan == "edit":
            state["writing_mode"] = "user_edit"
        else:
            tasks = []
            titles = {x["id"]: x.get("title", "") for x in state["outline"]}
            for number, item in enumerate(task["search_tasks"], 1):
                target = item["target_section"].strip()
                if target and target not in titles:
                    raise SessionTaskClarification(f"补证任务引用了当前大纲以外的章节ID：{target}")
                tasks.append({**item, "id": f"user_search_{number}", "issue_id": f"user_search_{number}",
                              "origin": "user_followup", "target_section": target or ",".join(titles)})
            state["pending_research_tasks"] = tasks
    state["execution_plan"] = plan
