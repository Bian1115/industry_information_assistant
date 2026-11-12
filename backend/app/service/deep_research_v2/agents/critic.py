# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""完整报告的三维百分制审核；程序计算总分并决定通过、补搜或修订。"""

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from service.agent_context import isolated_agent_input
from .base import BaseAgent
from ..state import ResearchState, ResearchPhase, citation_verification_status
from ..quality_pipeline import deterministic_critic_issues
from ..review_contract import (
    QUALITY_SCORE_BASIS, normalize_review_result, validate_critic_issues,
    bind_issue_targets, materialize_new_sections,
)


class CriticMaster(BaseAgent):
    """模型审核当前报告，程序只按当前轮的问题和三维均分执行返工。"""

    REVIEW_PROMPT = (Path(__file__).resolve().parents[1] / "prompts" /
                     "对抗式审核提示词.txt").read_text(encoding="utf-8")

    def __init__(self, llm_api_key: str, llm_base_url: str, model: str = "qwen-max"):
        super().__init__(name="CriticMaster", role="研究报告质量审核专家",
                         llm_api_key=llm_api_key, llm_base_url=llm_base_url, model=model)

    async def process(self, state: ResearchState) -> ResearchState:
        if state["phase"] != ResearchPhase.REVIEWING.value:
            return state
        self.add_message(state, "thought", {
            "agent": self.name, "content": "开始按可读性、全面性、深度和指令遵循度审核完整报告...",
        })
        result = await self._review_content(state)
        validate_critic_issues(result)
        # 来源范围等确定性约束仍形成 critical 问题，但不得改变三维均分。
        result["issues"].extend(deterministic_critic_issues(state))
        validate_critic_issues(result)
        normalize_review_result(result)
        bind_issue_targets(state, result["issues"])
        assessment = result["overall_assessment"]

        # 新提示词不含旧问题复查字段；历史留档，当前问题由本轮完整审核替换。
        if state.get("quality_score_basis") == QUALITY_SCORE_BASIS:
            state.setdefault("critic_history", []).append(deepcopy(state.get("critic_feedback") or []))
        state["critic_feedback"] = deepcopy(result["issues"])
        state["issue_search_results"] = {}  # 模型每轮可以重新从 ISSUE-001 编号。
        state["quality_score"] = assessment["quality_score"]
        state["quality_score_status"] = assessment["score_status"]
        state["quality_score_basis"] = QUALITY_SCORE_BASIS
        state["quality_dimension_scores"] = assessment["dimension_scores"]
        state["quality_verdict"] = assessment["verdict"]
        state["quality_review_summary"] = assessment["summary"]
        state["unresolved_issues"] = sum(
            issue["severity"] in {"critical", "major"} for issue in result["issues"])
        state.setdefault("citation_verification", citation_verification_status())
        state["pending_research_tasks"] = []

        self.add_message(state, "review", {
            "agent": self.name, "verdict": state["quality_verdict"],
            "quality_score": state["quality_score"],
            "quality_score_status": state["quality_score_status"],
            "quality_score_basis": QUALITY_SCORE_BASIS,
            "quality_dimension_scores": state["quality_dimension_scores"],
            "citation_verification": state["citation_verification"],
            "issues_count": len(result["issues"]), "issues": deepcopy(result["issues"]),
            "critical_issues": sum(i["severity"] == "critical" for i in result["issues"]),
            "major_issues": sum(i["severity"] == "major" for i in result["issues"]),
            "summary": assessment["summary"], "strength_points": result["strength_points"],
        })

        # 通过判断优先：60 分且无 major/critical 即通过，minor 补搜建议不阻塞。
        if state["quality_verdict"] == "pass":
            state["phase"] = ResearchPhase.COMPLETED.value
        elif state["iteration"] >= state["max_iterations"]:
            state["phase"] = ResearchPhase.COMPLETED.value
            self.add_message(state, "warning", {
                "agent": self.name, "content": "已达最大迭代次数，审核未通过，部分问题可能未解决",
            })
        else:
            state["revision_original_sections"] = {}
            materialize_new_sections(state)
            tasks = self._build_research_tasks({"issues": state["critic_feedback"]})
            if tasks:
                state["pending_research_tasks"] = tasks
                state["phase"] = ResearchPhase.RE_RESEARCHING.value
            else:
                state["phase"] = ResearchPhase.REVISING.value
            state["iteration"] += 1
        return state

    def _build_research_tasks(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """只使用模型的明确补搜标记；任务保留问题ID、程序章节ID和修改目标。"""
        validate_critic_issues(result)
        return [{
            "issue_id": issue["issue_id"],
            "target_section": " ".join(issue["resolved_section_ids"]),
            "query": issue["search_query"].strip(), "acceptance": issue["modification"],
        } for issue in result["issues"] if issue["requires_new_search"]]

    async def _review_content(self, state: ResearchState) -> Dict[str, Any]:
        report = state.get("final_report", "")
        if not report.strip():
            raise ValueError("进入审核阶段时 final_report 为空")
        inputs = {
            "user_question": state["query"],
            "research_outline": json.dumps(state["outline"], ensure_ascii=False),
            "report": report,
        }
        # 一次替换三个占位符，输入中恰好出现同名占位符时也不会被再次替换。
        prompt = re.sub(r"\{\{(user_question|research_outline|report)\}\}",
                        lambda match: inputs[match[1]], self.REVIEW_PROMPT)
        # 底层照常记录调用，但不向审核模型回放历史或追加任务契约等第四类材料。
        with isolated_agent_input():
            response = await self.call_llm(
                system_prompt="你是独立的研究报告质量审核专家。只根据本轮用户问题、研究大纲和报告审核，严格输出指定 JSON。",
                user_prompt=prompt, json_mode=True, temperature=0.2, max_tokens=16000,
            )
        result = self.parse_json_response(response)
        validate_critic_issues(result)
        normalize_review_result(result)
        return result
