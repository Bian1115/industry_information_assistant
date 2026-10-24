# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
DeepResearch V2.0 - 总架构师 Agent (ChiefArchitect)

职责：
1. 意图解码 - 深度理解用户问题
2. 知识图谱初始化 - 识别关键实体和关系
3. 动态大纲生成 - 创建可执行的研究计划
4. 进度监控 - 根据研究进展动态调整大纲
"""

import json
from copy import deepcopy
import uuid
from typing import Dict, Any, List
from datetime import datetime

from .base import BaseAgent
from ..state import ResearchState, ResearchPhase
from service.research_context_service import render_planning_context
from ..quality_pipeline import build_task_context
from ..contract_updates import contract_query, retain_committed_contract


class ChiefArchitect(BaseAgent):
    """
    总架构师 - 研究规划的大脑

    特点：
    - 动态DAG调度
    - 大局观，能看到全貌
    - 根据新发现调整计划
    """

    # 类级默认值兼容通过 __new__ 构造的轻量测试；正常运行时由配置覆盖。
    min_sections = 4
    max_sections = 5
    complex_max_sections = 6
    temperature = 0.3
    max_tokens = 16000

    @staticmethod
    def _render_historical_context(state: ResearchState) -> str:
        resolution = dict(state.get("context_resolution") or {})
        context = render_planning_context({
            "conversation_context": state.get("conversation_context") or [],
            "memory_context": state.get("memory_context") or [],
            "assumptions": resolution.get("assumptions") or [],
            "unresolved_references": resolution.get("unresolved_references") or [],
        })
        if state.get("session_planning_context"):
            context += "\n### 同一会话上一份研究的背景与成果索引\n" + json.dumps(
                state["session_planning_context"], ensure_ascii=False)
        if state.get("research_material_context"):
            context += "\n本轮按问题召回的来源材料（待核验）：\n" + json.dumps(
                state["research_material_context"], ensure_ascii=False)
        if state.get("inherited_evidence_manifest"):
            context += "\n### 从父研究继承的可追溯材料（索引，非新的检索结论）\n" + json.dumps({
                "facts": state.get("facts", [])[:30],
                "parent_requirements": state.get("parent_research_contract") or {},
                "active_constraints": state.get("active_constraints") or {},
                "note": "仅展示前30项；根据新大纲安排补搜，重新确认时间、人群与适用范围。",
            }, ensure_ascii=False)
        return context

    PLANNING_PROMPT = """你负责将当前研究任务规划为可执行的大纲和研究约束。此阶段只制定计划，不编造检索结果或研究结论。

## 用户本轮原话
{original_query}

## 结合历史补全后的当前研究课题
{query}

## 历史上下文与可继承材料
{historical_context}

## 理解任务与约束
1. 直接通读本轮原话、当前课题和上下文，识别研究目标、对象、时间、地域、比较维度、指标、计算、字数、语言、输出格式及禁止事项。未明确的要求不要擅自补成用户硬约束。
2. 本轮明确修改或取消的要求优先于旧要求；未修改的有效约束继续继承。如果后文提供已提交研究契约，以其为本轮有效约束，保持要求ID，不恢复已取消项。
3. 历史对话用于理解背景和指代；模型建议的例子、实体、数据集或假设不等于用户指定对象。存在歧义时说明假设，不虚构用户要求。
4. 所有影响研究结论的事实性内容都必须有可追溯原文及来源。可继承适用的父研究原文快照；对材料缺口、时效性或适用范围不明的内容安排重新检索、交叉验证。历史对话和长期记忆本身不是事实证据。

## 生成计划
1. outline 围绕研究问题组织，每章说明要回答的问题、研究范围和检索方向，合并重复内容。
2. 直接在大纲和章节 description 中落实用户问题。research_contract 仅保留用户明确提出的字数、语言和输出格式，未提出时返回空对象；不生成量化验收要求。
3. 普通时间、数量、字数和格式要求直接落实。只有需要核实的框架定义才在检索计划中安排核验，不预先断言存在或不存在，也不要求输出权威状态和替代预案字段。
4. hypotheses 只输出待验证假设的 id 和 content；状态和证据数组由程序初始化，没有必要的假设时输出空数组。

## 输出协议
仅输出合法 JSON，顶层只允许 outline、research_contract、hypotheses 三个字段，不要代码围栏或解释：
{{
  "outline": [
    {{
      "id": "sec_1",
      "title": "章节标题",
      "description": "本章问题、范围与需要的证据",
      "search_queries": ["具体检索词"]
    }}
  ],
  "research_contract": {{
    "active_constraints": {{}}
  }},
  "hypotheses": [{{"id": "h_1", "content": "待验证的假设"}}]
}}

仅在用户明确提出时保留 language、output_format 字符串和 active_constraints.max_body_chars（整数）；时间与地域要求直接落实到章节描述。
已提交的用户要求需要关联新章节时，可在章节增加 requirement_ids 引用其ID。章节确有数据或图表需求时可增加 requires_data、requires_chart；未提供时由程序使用默认值。
通常生成 {min_sections}-{max_sections} 个章节；仅多个独立对象或维度的复杂问题可扩展到 {complex_max_sections} 个。不要为满足章节数量拆出重复内容。"""

    REVISION_PROMPT = """你是总架构师，需要根据研究进展动态调整大纲。

## 原始问题
{query}

## 当前大纲
{current_outline}

## 新发现的重要信息
{new_findings}

## 当前进度
- 已完成章节: {completed_sections}
- 收集的事实数量: {facts_count}
- 发现的数据点: {data_points_count}

## 任务
评估是否需要调整大纲。可能的调整包括：
1. 新增章节（发现了重要的新方向）
2. 删除章节（发现某方向信息太少）
3. 调整章节顺序或优先级
4. 细化或合并章节

调整时必须重新核对原始问题中的明确框架、数量、排名、时间范围、维度和对象清单要求。
凡承担这些要求的章节不得被静默删除；如新发现表明用户指定框架没有权威定义，应保留该要求，
并把章节改为显式说明“未找到该框架的权威定义/未找到该确切数量的官方分类”，再给出有来源的替代方案、理由和对应关系。

输出JSON格式：
```json
{{
    "needs_revision": true或false,
    "revision_reason": "调整原因",
    "revised_outline": [...],  // 如果needs_revision为true
    "new_search_queries": ["新增的搜索关键词"]  // 如果需要补充搜索
}}
```"""

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max",
        min_sections: int = 4,
        max_sections: int = 5,
        complex_max_sections: int = 6,
        temperature: float = 0.3,
        max_tokens: int = 16000,
    ):
        super().__init__(
            name="ChiefArchitect",
            role="总架构师",
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            model=model
        )
        self.min_sections = max(1, int(min_sections))
        self.max_sections = max(self.min_sections, int(max_sections))
        self.complex_max_sections = max(self.max_sections, int(complex_max_sections))
        self.temperature = max(0.0, min(2.0, float(temperature)))
        self.max_tokens = max(1000, int(max_tokens))

    def _convert_flat_to_outline(self, flat_result: Dict) -> Dict:
        """将扁平JSON格式转换为标准outline格式"""
        outline = []
        for i in range(1, self.complex_max_sections + 1):
            title_key = f"sec_{i}_title"
            desc_key = f"sec_{i}_desc"
            query_key = f"sec_{i}_query"

            if title_key not in flat_result:
                break

            section = {
                "id": f"sec_{i}",
                "title": flat_result.get(title_key, f"章节{i}"),
                "description": flat_result.get(desc_key, ""),
                "section_type": "mixed",
                "requires_data": i <= 2,  # 前两章需要数据
                "requires_chart": i <= 2,
                "search_queries": [flat_result.get(query_key, flat_result.get(title_key, ""))]
            }
            outline.append(section)

        # 处理研究问题
        questions_str = flat_result.get("questions", "")
        if isinstance(questions_str, str):
            research_questions = [q.strip() for q in questions_str.split(";") if q.strip()]
        else:
            research_questions = []

        # 处理研究假设（假设驱动研究）
        hypotheses = []
        for i in range(1, 6):  # 最多5个假设
            h_key = f"hypothesis_{i}"
            if h_key in flat_result and flat_result[h_key]:
                hypotheses.append({
                    "id": f"h_{i}",
                    "content": flat_result[h_key],
                    "status": "unverified",  # unverified, supported, refuted, mixed, uncertain, partially_supported
                    "evidence_for": [],
                    "evidence_against": [],
                    "evidence_inconclusive": []
                })

        return {
            "outline": outline,
            "research_contract": {"critical_questions": research_questions},
            "hypotheses": hypotheses
        }

    def _compact_outline(self, outline: List[Dict]) -> List[Dict]:
        """将超长大纲压到工业上限，同时保留溢出章节的要求和检索词。"""
        valid = [dict(section) for section in outline if isinstance(section, dict)]
        limit = getattr(self, "complex_max_sections", 6)
        if len(valid) <= limit:
            return valid

        kept = valid[: limit - 1]
        merged_sections = valid[limit - 1 :]
        merged = dict(merged_sections[0])
        merged["title"] = "综合分析与补充事项"
        merged["description"] = "；".join(
            str(section.get("description") or section.get("title") or "").strip()
            for section in merged_sections
            if str(section.get("description") or section.get("title") or "").strip()
        )
        merged["requires_data"] = any(section.get("requires_data") for section in merged_sections)
        merged["requires_chart"] = any(section.get("requires_chart") for section in merged_sections)
        merged["requirement_ids"] = list(dict.fromkeys(
            str(req_id)
            for section in merged_sections
            for req_id in (
                [section.get("requirement_ids")]
                if isinstance(section.get("requirement_ids"), str)
                else section.get("requirement_ids", []) or []
            )
            if req_id
        ))
        merged["search_queries"] = list(dict.fromkeys(
            str(query).strip()
            for section in merged_sections
            for query in (
                [section.get("search_queries")]
                if isinstance(section.get("search_queries"), str)
                else section.get("search_queries", []) or []
            )
            if str(query).strip()
        ))
        return kept + [merged]

    @staticmethod
    def _validate_planning_output(result: Any, hypothesis_enabled: bool = True) -> List[str]:
        """只检查执行所需的基础结构，不生成或校验逐条要求响应。"""
        if not isinstance(result, dict):
            return ["规划必须是JSON对象"]
        issues = []
        outline = result.get("outline")
        if not isinstance(outline, list) or len(outline) < 3 or any(
            not isinstance(section, dict) or not str(section.get("title") or "").strip()
            for section in outline
        ):
            issues.append("outline必须至少包含三个有标题的章节对象")
        if not isinstance(result.get("research_contract"), dict):
            issues.append("research_contract必须是对象")
        hypotheses = result.get("hypotheses")
        if hypothesis_enabled and (not isinstance(hypotheses, list) or any(
            not isinstance(h, dict) or not str(h.get("content") or "").strip()
            for h in hypotheses
        )):
            issues.append("hypotheses必须是包含假设content的对象数组，可为空")
        return issues

    async def process(self, state: ResearchState) -> ResearchState:
        """
        处理入口

        根据当前阶段执行不同的规划任务
        """
        if state["phase"] == ResearchPhase.INIT.value:
            return await self._initial_planning(state)
        elif state["phase"] == ResearchPhase.REVIEWING.value:
            return await self._check_revision(state)
        else:
            return state

    async def _initial_planning(self, state: ResearchState) -> ResearchState:
        """初始规划"""
        committed_contract = deepcopy(state.get('research_contract') or {})
        if 'pending_contract_update' in committed_contract:
            raise ValueError('Architect旧追问契约更新已停用，请先在会话层构建并提交本轮任务')
        self.logger.info(f"Starting initial planning for: {state['query'][:50]}...")

        # 发送 research_step 开始事件
        self.add_message(state, "research_step", {
            "step_id": f"step_planning_{uuid.uuid4().hex[:8]}",
            "step_type": "planning",
            "title": "研究计划",
            "subtitle": "分析问题，制定大纲",
            "status": "running",
            "stats": {}
        })

        # 发送状态消息
        self.add_message(state, "thought", {
            "agent": self.name,
            "content": "正在分析研究问题，构建知识图谱和研究大纲..."
        })

        # 模型直接读取原话、课题与上下文；程序只校验基础输出结构。
        from ..hypothesis_policy import hypotheses_enabled
        enabled = hypotheses_enabled(state)
        template = self.PLANNING_PROMPT
        if not enabled:
            template = template.replace(
                "4. hypotheses 只输出待验证假设的 id 和 content；状态和证据数组由程序初始化，没有必要的假设时输出空数组。\n", ""
            ).replace("outline、research_contract、hypotheses 三个字段", "outline、research_contract 两个字段").replace(
                ',\n  "hypotheses": [{{"id": "h_1", "content": "待验证的假设"}}]', ""
            )
        base_prompt = template.format(
            query=state["query"],
            original_query=state.get("original_query", state["query"]),
            historical_context=self._render_historical_context(state),
            min_sections=self.min_sections,
            max_sections=self.max_sections,
            complex_max_sections=self.complex_max_sections,
        )
        if (committed_contract.get('contract_update') or {}).get('status') == 'committed':
            base_prompt += '\n以下是本轮已提交研究契约。用户要求不可删除或改写，不能复活已取消的要求；' \
                           '为显式要求保留ID并在新大纲requirement_ids中覆盖。模型自己选的实体不属于用户硬约束。\n' \
                           + json.dumps(committed_contract, ensure_ascii=False)
        prompt = base_prompt
        result = None
        validation_issues = []
        max_retries = 2

        for attempt in range(max_retries + 1):
            response = await self.call_llm(
                system_prompt="你是一位专业的行业研究规划师。请严格按照要求的JSON格式输出，不要添加任何额外内容。",
                user_prompt=prompt,
                json_mode=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Debug: 记录原始响应
            self.logger.info(f"Raw LLM response length: {len(response)} (attempt {attempt + 1})")
            self.logger.debug(f"Raw LLM response (first 1000 chars): {response[:1000]}")

            result = self.parse_json_response(response)

            # 检查是否是扁平格式，需要转换
            if isinstance(result, dict) and result.get("sec_1_title") and not result.get("outline"):
                result = self._convert_flat_to_outline(result)

            validation_issues = self._validate_planning_output(result, enabled)
            if not validation_issues:
                self.logger.info(f"Successfully parsed outline with {len(result['outline'])} sections")
                break

            self.logger.warning(
                f"Attempt {attempt + 1}: Planning output invalid: {'; '.join(validation_issues)}"
            )

            if attempt < max_retries:
                self.logger.warning(f"Outline generation failed or incomplete, retrying... (attempt {attempt + 1})")
                issue_text = "; ".join(validation_issues) or "JSON 解析失败或章节数量不足"
                prompt = (
                    f"{base_prompt}\n\n"
                    f"上一次输出存在以下问题：{issue_text}。请修正后重新输出完整 JSON；"
                    + ("顶层仍只输出outline、research_contract、hypotheses，保留有效约束。"
                       if enabled else "顶层仅输出outline、research_contract，保留有效约束。")
                )

        if validation_issues:
            state["errors"].append("Failed to generate research plan after retries")
            self.logger.error(f"Raw LLM response: {response[:800]}")
            return state

        if (committed_contract.get('contract_update') or {}).get('status') == 'committed':
            # 先提交本轮有效用户要求给后续规划处理，不能使用模型重建的旧清单。
            state['query'] = contract_query(committed_contract)
            state['resolved_query'] = committed_contract['query']
            state['active_constraints'] = deepcopy(committed_contract['active_constraints'])
            state['revision_instructions'] = list(committed_contract['followup_instructions'])
            parent = state.pop('session_parent', None)
            if parent is not None:
                from ..followup_context import inherit_artifacts
                if committed_contract['contract_update'].get('reuse_previous'):
                    inherit_artifacts(state, parent, replan=True)
                else:
                    state.pop('session_planning_context', None)
        result["outline"] = self._compact_outline(result.get("outline") or [])

        # Debug: log outline count
        self.logger.info(f"Parsed result keys: {list(result.keys())}")
        outline = result.get("outline", [])
        self.logger.info(f"Outline in result: {len(outline)} sections")
        if not outline:
            self.logger.warning(f"No outline found! Full parsed result: {str(result)[:500]}")

        # 更新状态
        # 兼容旧提示词或旧模型输出：三个证据分桶必须始终存在，避免
        # inconclusive 证据只能改变状态、却没有独立可审计的落盘位置。
        normalized_hypotheses = []
        raw_hypotheses = result.get("hypotheses", []) if enabled else []
        if not isinstance(raw_hypotheses, list):
            raw_hypotheses = []
        for index, hypothesis in enumerate(raw_hypotheses[:5], 1):
            if not isinstance(hypothesis, dict):
                continue
            item = dict(hypothesis)
            item.setdefault("id", f"h_{index}")
            item.update(status="unverified", evidence_for=[], evidence_against=[], evidence_inconclusive=[])
            normalized_hypotheses.append(item)
        state["hypotheses"] = normalized_hypotheses  # 假设驱动研究
        if not enabled:
            state["hypothesis_evidence"] = []
            state["conclusion_evidence_map"] = []
        state["knowledge_graph"] = {"nodes": [], "edges": []}  # 知识图谱初始化

        # 处理大纲 - 确保每个章节都有必要字段
        processed_outline = []
        for i, section in enumerate(outline):
            if not isinstance(section, dict):
                continue
            processed_section = {
                "id": section.get("id", f"sec_{i+1}"),
                "title": section.get("title", f"章节{i+1}"),
                "description": section.get("description", ""),
                "section_type": section.get("section_type", "mixed"),
                "requires_data": section.get("requires_data", False),
                "requires_chart": section.get("requires_chart", False),
                "priority": section.get("priority", i+1),
                "search_queries": section.get("search_queries", [section.get("title", "")]),
                "requirement_ids": section.get("requirement_ids", []),
                "status": "pending"
            }
            if isinstance(processed_section["requirement_ids"], str):
                processed_section["requirement_ids"] = [processed_section["requirement_ids"]]
            processed_section["requirement_ids"] = [
                str(req_id) for req_id in processed_section["requirement_ids"] if req_id
            ]
            # 确保 search_queries 是列表且非空
            if not isinstance(processed_section["search_queries"], list):
                processed_section["search_queries"] = [str(processed_section["search_queries"])]
            # 过滤空字符串，如果结果为空则使用章节标题
            processed_section["search_queries"] = [q for q in processed_section["search_queries"] if q and q.strip()]
            if not processed_section["search_queries"]:
                processed_section["search_queries"] = [processed_section["title"]]
            processed_outline.append(processed_section)

        state["outline"] = processed_outline
        planned_contract = build_task_context(result, state["query"])
        state["research_contract"] = retain_committed_contract(planned_contract, committed_contract, processed_outline)
        # 保留本轮原话和已经解析的有效约束；历史检索批次本身不包含完整任务。
        state["research_contract"]["original_query"] = state.get("original_query") or state["query"]
        state["active_constraints"] = deepcopy(state["research_contract"].get("active_constraints") or {})
        # 兼容前端和检查点的展示字段，从最终契约派生，不再由模型重复生成。
        state["research_questions"] = list(state["research_contract"].get("critical_questions") or [state["query"]])
        state["key_entities"] = list(state["research_contract"].get("required_entities") or [])
        state["mind_map"] = {}
        self.logger.info(f"Processed outline: {len(processed_outline)} sections")

        # 发送大纲事件
        self.add_message(state, "outline", {
            "outline": processed_outline,
            "research_questions": state["research_questions"],
        })

        # 更新阶段
        state["phase"] = ResearchPhase.PLANNING.value

        # 发送 research_step 完成事件
        self.add_message(state, "research_step", {
            "step_type": "planning",
            "title": "研究计划",
            "subtitle": "分析问题，制定大纲",
            "status": "completed",
            "stats": {
                "sections_count": len(processed_outline),
                "questions_count": len(state["research_questions"])
            }
        })

        self.logger.info(f"Planning completed. Generated {len(outline)} sections.")

        return state

    async def _check_revision(self, state: ResearchState) -> ResearchState:
        """检查是否需要修订大纲"""
        # 收集新发现
        new_findings = []
        for fact in state["facts"][-10:]:  # 最近10条事实
            new_findings.append(f"- {fact.get('content', '')[:100]}")

        if not new_findings:
            return state

        # 统计进度
        completed = [s for s in state["outline"] if s.get("status") == "final"]

        prompt = self.REVISION_PROMPT.format(
            query=state["query"],
            current_outline=state["outline"],
            new_findings="\n".join(new_findings),
            completed_sections=len(completed),
            facts_count=len(state["facts"]),
            data_points_count=len(state["data_points"])
        )

        response = await self.call_llm(
            system_prompt="你是总架构师，需要判断是否需要调整研究计划。",
            user_prompt=prompt,
            json_mode=True
        )

        result = self.parse_json_response(response)

        if result and result.get("needs_revision") and result.get("revised_outline"):
            revised_outline = result["revised_outline"]

            # 动态改纲不能抹掉初始规划中已经落实的用户显式要求。
            current_by_id = {
                section.get("id"): section
                for section in state["outline"]
                if isinstance(section, dict) and section.get("id")
            }
            required_ids = {
                str(requirement_id)
                for section in state["outline"]
                if isinstance(section, dict)
                for requirement_id in (section.get("requirement_ids") or [])
            }
            revised_required_ids = set()
            for section in revised_outline:
                if not isinstance(section, dict):
                    continue
                previous = current_by_id.get(section.get("id"), {})
                if not section.get("requirement_ids") and previous.get("requirement_ids"):
                    section["requirement_ids"] = previous["requirement_ids"]
                requirement_ids = section.get("requirement_ids") or []
                if isinstance(requirement_ids, str):
                    requirement_ids = [requirement_ids]
                    section["requirement_ids"] = requirement_ids
                revised_required_ids.update(str(req_id) for req_id in requirement_ids if req_id)

            missing_required_ids = required_ids - revised_required_ids
            if missing_required_ids:
                self.logger.warning(
                    "Rejected outline revision because it dropped explicit requirements: "
                    f"{sorted(missing_required_ids)}"
                )
                return state

            state["outline"] = revised_outline
            self.add_message(state, "outline_revision", {
                "reason": result.get("revision_reason"),
                "new_outline": revised_outline
            })
            self.logger.info(f"Outline revised: {result.get('revision_reason')}")

        return state
