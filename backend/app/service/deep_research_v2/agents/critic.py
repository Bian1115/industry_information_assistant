# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
DeepResearch V2.0 - 毒舌评论家 Agent (CriticMaster)

职责：
1. 对抗式质检 - 永远不满意，找出问题
2. 逻辑漏洞检测 - 检查推理链条
3. 要求覆盖与整体时效检查
4. 偏见识别 - 发现观点偏颇
"""

import asyncio
from service.agent_history import AgentHistoryError
from service.agent_context import ContextBudgetExceeded
import json
import math
from copy import deepcopy
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent
from ..state import ResearchState, ResearchPhase, citation_verification_status
from ..quality_pipeline import deterministic_critic_issues, evidence_fingerprint
from ..issue_workflow import acceptance, issue_ledger, open_issues, reconcile_reviews, review_packet
from ..critic_evidence import build_critic_evidence


QUALITY_SCORE_BASIS = "report_review_3d_v2"
REPORT_DIMENSION_WEIGHTS = {
    "logical_coherence": 4,
    "requirement_coverage": 3,
    "freshness": 2,
}


def handoff_legacy_citation_issues(state: Dict[str, Any]) -> None:
    """旧检查点的引用问题保留交接记录，不再由Critic复查，也不假称已修复。"""
    kept = []
    for issue in state.get("critic_feedback") or []:
        if not issue.get("rule_based") and issue.get("issue_type") in {
            "missing_source", "citation_error", "prediction_as_fact", "unsupported_citation",
        }:
            state.setdefault("citation_issue_handoffs", []).append({**deepcopy(issue),
                "owner": "citation_verification", "handoff_status": "awaiting_independent_report_verification"})
        else:
            kept.append(issue)
    state["critic_feedback"] = kept


def validate_critic_issues(result: Any) -> None:
    """严格校验补搜契约；缺失字段或错误类型直接报错，不推断旧格式。"""
    if not isinstance(result, dict) or not isinstance(result.get("issues"), list):
        raise ValueError("Critic 审核结果必须包含 issues 数组，无问题时返回空数组")
    for index, issue in enumerate(result["issues"]):
        if not isinstance(issue, dict):
            raise ValueError(f"Critic issues[{index}] 必须是对象")
        if not isinstance(issue.get("requires_new_search"), bool):
            raise ValueError(f"Critic issues[{index}].requires_new_search 必须是布尔值")
        if issue["requires_new_search"]:
            query = issue.get("search_query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"Critic issues[{index}] 要求补搜时必须提供非空字符串 search_query"
                )


def normalize_quality_score(value: Any) -> Optional[int]:
    """把模型评分规范为 1-10 的整数；非法值返回 None。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not score.is_integer() or not 1 <= score <= 10:
        return None
    return int(score)


def derive_quality_score(result: Any) -> Optional[Dict[str, Any]]:
    """
    当模型分数非法或提供了分维度分数时，用可复现规则计算总分。

    返回 {score, status} 而不是再让 LLM 凭摘要重打一次分，
    避免同一份审核在 -1 / 1 / 4 之间随机波动。
    """
    if not isinstance(result, dict):
        return None
    assessment = result.get("overall_assessment")
    if not isinstance(assessment, dict):
        return None

    dimensions = assessment.get("dimension_scores")
    weights = REPORT_DIMENSION_WEIGHTS
    valid_dimensions = {}
    if isinstance(dimensions, dict):
        for name, weight in weights.items():
            score = normalize_quality_score(dimensions.get(name))
            if score is not None:
                valid_dimensions[name] = (score, weight)

    issues = [issue for issue in result.get("issues", []) if isinstance(issue, dict)]
    counts = {
        severity: sum(issue.get("severity") == severity for issue in issues)
        for severity in ("critical", "major", "minor")
    }

    if len(valid_dimensions) >= 3:
        total_weight = sum(weight for _, weight in valid_dimensions.values())
        weighted = sum(
            score * weight for score, weight in valid_dimensions.values()
        ) / total_weight
        score = int(math.floor(weighted + 0.5))
        status = "derived_from_dimensions"
    elif issues:
        # 无分维度结果的兼容路径。critical 必须有明显惩罚，
        # 但一个问题不会直接把报告打成 1 分。
        raw = (
            10
            - 3.0 * counts["critical"]
            - 1.25 * counts["major"]
            - 0.25 * counts["minor"]
        )
        score = int(math.floor(raw + 0.5))
        status = "derived_from_issue_severity"
    else:
        return None

    verdict = assessment.get("verdict")
    if counts["critical"] >= 3:
        score = min(score, 2)
    elif counts["critical"]:
        score = min(score, 4)
    if counts["major"] >= 3:
        score = min(score, 5)
    if verdict == "major_issues":
        score = min(score, 4)
    elif verdict == "needs_revision":
        score = min(score, 6)
    elif verdict == "pass" and not counts["critical"] and not counts["major"]:
        score = max(score, 7)

    return {
        "score": max(1, min(10, score)),
        "status": status,
        "issue_counts": counts,
    }


class CriticMaster(BaseAgent):
    """
    毒舌评论家 - 质量守门人

    特点：
    - 对抗式思维：假设一切都有问题
    - 严格的证据要求
    - 逻辑一致性检查
    - 有权打回重写
    """

    REVIEW_PROMPT = """你是整篇研究报告的审核人，负责整体逻辑、用户要求覆盖、整体时效和观点平衡。

## 审核基准日期
当前日期是 {current_date}。按用户要求判断整篇报告的时间覆盖，历史分析可以使用对应年份资料。不要仅凭年份新旧判断单条论断是否为预测。

## 审核原则（必须严格执行）
1. **职责边界**：独立引用核验模块从最终报告提取论断，负责遗漏引用、来源支持性、来源中的预测/计划是否被写成事实，以及对象、数字、年份和限定词是否忠实于原文。不要重复执行这些检查，不输出missing_source、citation_error或prediction_as_fact问题，也不据引用核验是否执行或通过给分。没有核验不等于内容有误或核验通过。
2. **整体逻辑**：检查跨章节矛盾、比较口径、汇总结论与前文数据是否一致、是否把相关性擅自推为因果，以及证据冲突是否被隐去。logical_coherence 只衡量这类报告整体推理。
3. **观点平衡**：检查是否只呈现单方观点、忽略重要反对意见或研究局限，以及结论强度是否与报告整体讨论相称；这项归入logical_coherence。
4. **整体时效**：检查用户要求的时间窗口是否覆盖，例如要求近期进展却整体只讨论多年以前的材料。单条引文与原文的时间一致性以及预测/事实混淆由引用核验负责，不在此重复裁定。
5. **要求覆盖**：检查研究对象、范围、指标、分析任务和明确输出要求是否完整。

## 研究问题
{query}

## 研究大纲
{outline}

## 研究契约与已有门禁摘要（仅含非空检查、证据覆盖与来源范围，不含引用核验）
{research_contract}
{quality_gates}

## 待审核内容

### 待审核的最终报告
{final_report}

### 使用的数据点
{data_points}

### 当前报告全部现有论断与引用原文片段
{claim_evidence}

这些材料供理解报告和逐项复查旧问题，结合当前正文、Writer修改记录和验收条件判断是否解决。共用原文按evidence_id关联；原文片段未截断。注意claim_scope、抽取状态、定位、缺失证据和核验状态，不能把有来源或Writer声称已处理等同于问题已解决。引用核验的职责边界保持不变。

## 必须逐条复查的旧问题（保留原ID）
{previous_issues}

## Writer本轮逐问题处理记录（addressed只代表已处理，仍待你复查）
{writer_revisions}

逐项对照当前完整报告、修改位置、具体依据与验收条件：确认修复用resolved；仍未修复或材料不足用unresolved；原批评本身错误、可保留原文用dismissed。
每个旧问题必须恰好返回一条issue_reviews，不能因来源增加、分数提高或本轮未再列出问题而关闭。
resolved/dismissed必须给出非空的report_quote及具体reason，说明验收条件如何满足，或原批评为何不成立。report_quote用于说明涉及的报告内容，不要求与当前报告逐字匹配。证据不足不能写成已验证。
issues只列新问题以及仍需补搜/修改的旧问题；旧问题必须沿用ID。已关闭问题不得重新用同一ID表示不同问题。

## 任务
完成上述报告层面检查。只针对整体逻辑、要求覆盖、整体时效和观点平衡提出问题。需要拓展研究范围、补齐用户要求或补充缺失观点时，可以要求补搜；不能以核对单条引文、补行内引用或判断预测是否被改写为事实为由提出补搜。
只给 logical_coherence、requirement_coverage、freshness 三项分数，不输出 evidence_traceability、factual_reliability 或 fact_check_results。本次审核不包含引用核验，评分不得因为“没有核验失败”而加分。

## 输出格式
```json
{{
    "overall_assessment": {{
        "quality_score": 1,
        "verdict": "pass/needs_revision/major_issues",
        "summary": "整体评估摘要",
        "dimension_scores": {{
            "logical_coherence": 1,
            "requirement_coverage": 1,
            "freshness": 1
        }}
    }},
    "issues": [
        {{
            "id": "issue_1",
            "target_section": "章节ID或'全局'",
            "issue_type": "logic_error/bias/outdated/incomplete",
            "severity": "critical/major/minor",
            "location": "具体位置描述",
            "report_quote": "有问题的报告原句，要求缺失时可留空",
            "requirement": "对应的用户原始要求",
            "description": "问题详细描述",
            "evidence": "为什么这是问题的证据",
            "suggestion": "具体的修改建议",
            "acceptance_criteria": "修订后怎样逐项验收，不以来源数量或总分替代",
            "requires_new_search": true,
            "search_query": "一条可直接执行的补搜查询，包含具体对象、年份与所需证据"
        }}
    ],
    "issue_reviews": [
        {{"issue_id": "必须复查的原问题ID", "status": "resolved/unresolved/dismissed",
          "report_quote": "相关报告内容，非空，不要求逐字匹配", "reason": "逐项验收或撤销原批评的具体依据"}}
    ],
    "strength_points": ["报告中做得好的地方"]
}}
```

## 问题与补搜字段硬性约束
- issues 必须是数组，无问题时返回 []；每项都必须包含布尔值 requires_new_search，不能省略，不能使用字符串或数字。
- requires_new_search=true 表示需要新证据，必须提供非空 search_query，每项只放一条查询；需要多个查询时拆成多条问题。
- requires_new_search=false 表示用已有证据修订，search_query 留空。问题类型与严重程度不替代这一判断。
- 所有遗漏方面也必须写入 issues，使用 issue_type=incomplete，并明确标记是否补搜；不要另行输出 missing_aspects。
- 不得只在 suggestion 或 description 中提出补搜要求。系统只执行 requires_new_search=true 的 search_query，不会猜测或补齐。

## 严重程度说明
- critical: 必须修复，否则报告不可用（如：核心结论自相矛盾、研究对象错误）
- major: 强烈建议修复，影响报告质量（如：遗漏关键要求、逻辑漏洞、观点严重片面）
- minor: 建议修复，提升报告质量（如：表述不够精确）

## 评分标准（1-10分制）
- 9-10分：优秀，几乎无问题，可直接发布
- 7-8分：良好，有小问题但不影响整体质量，审核通过（verdict=pass）
- 5-6分：一般，有明显问题需要修订
- 3-4分：较差，问题较多，需要大幅修改
- 1-2分：很差，存在严重问题或大量错误

## 评分字段硬性约束
- quality_score 必须始终是 1 到 10 之间的整数
- 即使报告严重到“无法发布”或“必须重写”，也必须给 1 分或 2 分
- 严禁返回 -1、0、null、字符串、区间或“无法评分”作为 quality_score
- quality_score >= 7 时才能设置 verdict 为 "pass"
- 先给出上述 3 个 dimension_scores，quality_score 按逻辑、要求覆盖、时效的 4:3:2 权重归一化计算，并与问题严重程度和 verdict 一致
- 单一表述问题不足以给 1-2 分；1-2 分必须对应多个 critical 问题或核心结论整体不可用

开始你的审核："""

    SCORE_REPAIR_PROMPT = """你只负责修复一次审核结果里的评分字段，不要重新审核原报告。

## 原审核摘要
{assessment}

## 原审核问题计数
{issue_counts}

## 输出要求
只输出以下结构的 JSON：
{{
  "overall_assessment": {{
    "quality_score": 1,
    "verdict": "pass/needs_revision/major_issues"
  }}
}}

硬性约束：
1. quality_score 必须是 1 到 10 之间的整数；
2. 原审核认为报告无法评分、不可用或问题极严重时给 1 分，不能给 -1、0 或 null；
3. quality_score 小于 7 时 verdict 不能是 pass。
"""

    def __init__(self, llm_api_key: str, llm_base_url: str, model: str = "qwen-max"):
        super().__init__(
            name="CriticMaster",
            role="毒舌评论家",
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            model=model
        )

    async def process(self, state: ResearchState) -> ResearchState:
        """处理入口"""
        self.logger.info(f"[CriticMaster] ========== process 开始 ==========")
        self.logger.info(f"[CriticMaster] phase: {state['phase']}, final_report 长度: {len(state.get('final_report', ''))}")

        if state["phase"] != ResearchPhase.REVIEWING.value:
            self.logger.info(f"[CriticMaster] phase 不是 REVIEWING，跳过")
            return state

        self.add_message(state, "thought", {
            "agent": self.name,
            "content": "开始严格审核研究报告，准备找出所有问题..."
        })

        # 执行审核
        handoff_legacy_citation_issues(state)
        state["critic_feedback"] = issue_ledger(state)
        self.logger.info(f"[CriticMaster] 开始调用 _review_content...")
        review_result = await self._review_content(state)
        self.logger.info(f"[CriticMaster] 审核完成，结果: {bool(review_result)}")
        validate_critic_issues(review_result)

        if review_result:
            rule_issues = deterministic_critic_issues(state)
            if rule_issues:
                review_result.setdefault("issues", []).extend(rule_issues)
                assessment = review_result.setdefault("overall_assessment", {})
                assessment["quality_score"] = min(
                    normalize_quality_score(assessment.get("quality_score")) or 1,
                    4,
                )
                assessment["verdict"] = "major_issues"

            validate_critic_issues(review_result)

            # 旧问题保留稳定身份；漏复查不关闭，Writer自报也不关闭。
            previous = deepcopy(state.get("critic_feedback") or [])
            previous_open = {issue["id"] for issue in open_issues(state)}
            ledger, active = reconcile_reviews(state, review_result)
            if previous:
                state.setdefault("critic_history", []).append(previous)
            state["critic_feedback"] = ledger
            review_result["issues"] = active
            current_evidence = evidence_fingerprint(state)
            for issue in active:
                issue.setdefault("evidence_fingerprint", current_evidence)

            # 决策和执行共用这一份任务列表；不从类型、建议或遗漏摘要补猜查询。
            research_tasks = self._build_research_tasks(review_result)
            state["pending_research_tasks"] = []

            # 更新质量分数
            assessment = review_result.get("overall_assessment", {})
            blocks_pass = research_tasks or any(
                issue["id"] in previous_open or issue.get("severity") in {"critical", "major"}
                for issue in active
            )
            if blocks_pass and assessment.get("verdict") == "pass":
                assessment["verdict"] = "needs_revision"
                assessment["quality_score"] = min(
                    normalize_quality_score(assessment.get("quality_score")) or 1, 6
                )
            state["quality_score"] = assessment.get("quality_score", 1)
            state["quality_score_status"] = assessment.get("score_status", "valid")
            state["quality_score_basis"] = QUALITY_SCORE_BASIS
            state["quality_dimension_scores"] = {
                name: score
                for name in REPORT_DIMENSION_WEIGHTS
                if (score := normalize_quality_score(
                    (assessment.get("dimension_scores") or {}).get(name)
                )) is not None
            }
            # 保留独立核验器的实际结果，审核评分不覆盖引用核验审计。
            state.setdefault("citation_verification", citation_verification_status())
            state["quality_verdict"] = assessment.get("verdict", "needs_revision")
            state["unresolved_issues"] = len([i for i in review_result.get("issues", []) if i.get("severity") in ["critical", "major"]])

            # 发送审核结果
            self.add_message(state, "review", {
                "agent": self.name,
                "verdict": review_result.get("overall_assessment", {}).get("verdict"),
                "quality_score": state["quality_score"],
                "quality_score_status": state["quality_score_status"],
                "quality_score_basis": state["quality_score_basis"],
                "quality_dimension_scores": state["quality_dimension_scores"],
                "citation_verification": state["citation_verification"],
                "issues_count": len(review_result.get("issues", [])),
                "issue_reviews": review_result.get("issue_reviews", []),
                "issue_ledger": deepcopy(ledger),
                "critical_issues": len([i for i in review_result.get("issues", []) if i.get("severity") == "critical"]),
                "major_issues": len([i for i in review_result.get("issues", []) if i.get("severity") == "major"]),
                "summary": review_result.get("overall_assessment", {}).get("summary", ""),
                "missing_aspects": [
                    issue.get("description", "")
                    for issue in review_result["issues"]
                    if issue.get("issue_type") == "incomplete"
                ],
            })

            # 如果有严重问题，发送具体反馈
            critical_issues = [i for i in review_result.get("issues", []) if i.get("severity") == "critical"]
            for issue in critical_issues[:3]:  # 最多展示3个严重问题
                self.add_message(state, "critic_feedback", {
                    "agent": self.name,
                    "issue_type": issue.get("issue_type"),
                    "severity": issue.get("severity"),
                    "description": issue.get("description"),
                    "suggestion": issue.get("suggestion")
                })

            # 决定下一步 - 智能路由
            verdict = state["quality_verdict"]

            if verdict == "pass":
                state["phase"] = ResearchPhase.COMPLETED.value
            elif state["iteration"] >= state["max_iterations"]:
                # 达到最大迭代次数，强制完成
                state["phase"] = ResearchPhase.COMPLETED.value
                self.add_message(state, "warning", {
                    "agent": self.name,
                    "content": "已达最大迭代次数，部分问题可能未解决"
                })
            else:
                state["revision_original_sections"] = {}
                if research_tasks:
                    # 需要补充搜索 -> 回到研究阶段
                    state["phase"] = ResearchPhase.RE_RESEARCHING.value
                    state["pending_research_tasks"] = research_tasks
                    self.add_message(state, "thought", {
                        "agent": self.name,
                        "content": f"发现信息缺失问题，需要补充搜索: {', '.join(task['query'] for task in research_tasks[:3])}"
                    })
                else:
                    # 仅需要文字修改 -> 修订阶段
                    state["phase"] = ResearchPhase.REVISING.value

                state["iteration"] += 1

        return state

    def _build_research_tasks(self, review_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """仅按必填补搜标记生成任务，每个问题对应一条查询。"""
        validate_critic_issues(review_result)
        return [
            {
                "issue_id": issue.get("id"),
                "target_section": issue.get("target_section", ""),
                "query": issue["search_query"].strip(),
                "acceptance": acceptance(issue),
            }
            for issue in review_result["issues"]
            if issue["requires_new_search"]
        ][:10]

    async def _review_content(self, state: ResearchState) -> Dict[str, Any]:
        """审核内容"""
        self.logger.info(f"[CriticMaster] _review_content 开始")

        # REVIEWING 阶段必须已经由 Writer 生成完整报告。这里不再回退到
        # draft_sections，避免用旧章节掩盖上游组装失败或修订不同步。
        if not state.get("final_report", "").strip():
            raise ValueError("进入审核阶段时 final_report 为空")

        self.logger.info(
            f"[CriticMaster] 待审核内容长度: {len(state['final_report'])}"
        )

        # 准备数据点列表
        data_summary = []
        for dp in state["data_points"][:50]:
            data_summary.append(f"- {dp.get('name')}: {dp.get('value')} {dp.get('unit', '')} (来源: {dp.get('source')})")

        # 复用当前报告的论断和证据编号回读原文，不让标题/URL替代实际依据。
        claim_evidence = await build_critic_evidence(state)

        # 格式化大纲
        outline_summary = []
        for section in state["outline"]:
            outline_summary.append(f"- {section.get('id')}: {section.get('title')} ({section.get('status', 'pending')})")

        # 只传门禁摘要和完整的紧凑判断记录，不重复投喂原文要求模型再验一次。
        gates = state.get("quality_gates") or {}
        gate_summary = {
            name: {key: gate[key] for key in ("enabled", "passed", "blocking_reasons") if key in gate}
            for name, gate in gates.items()
            if isinstance(gate, dict)
        }
        prompt = self.REVIEW_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            query=state["query"],
            outline="\n".join(outline_summary),
            # 不截断报告，确保 Critic 能看到后半部分和参考文献。
            final_report=state["final_report"],
            data_points="\n".join(data_summary) if data_summary else "（暂无数据点）",
            claim_evidence=json.dumps(claim_evidence, ensure_ascii=False, separators=(",", ":")),
            research_contract=json.dumps(
                state.get("research_contract") or {}, ensure_ascii=False
            )[:12000],
            quality_gates=json.dumps(gate_summary, ensure_ascii=False),
            previous_issues=json.dumps(open_issues(state), ensure_ascii=False),
            writer_revisions=review_packet(state),
        )

        self.logger.info(f"[CriticMaster] 调用 LLM 进行审核...")
        response = await self.call_llm(
            system_prompt=(
                "你是一位严谨且校准良好的质量审核专家。"
                "引用覆盖、预测与事实混淆、来源支持性由独立引用核验模块负责，你不重复检查或评分。"
                "只审核整篇报告的逻辑、用户要求覆盖、整体时效与观点平衡；仍按逻辑、要求覆盖、时效三项给分。"
            ),
            user_prompt=prompt,
            json_mode=True,
            temperature=0.2,
            max_tokens=16000  # 拉满到最大值
        )
        self.logger.info(f"[CriticMaster] LLM 响应长度: {len(response)}")

        result = self.parse_json_response(response)
        validate_critic_issues(result)
        if self._normalize_review_result(result):
            self.logger.info(
                f"[CriticMaster] JSON 解析结果: True, "
                f"verdict: {result.get('overall_assessment', {}).get('verdict')}"
            )
            return result

        raw_score = (result.get("overall_assessment") or {}).get("quality_score") \
            if isinstance(result, dict) else None
        self.logger.warning(
            f"[CriticMaster] quality_score 非法: {raw_score!r}，纠错重试一次"
        )

        # 这里只需修复评分字段，不应把数万字报告再发送并完整审核一次。
        # 旧逻辑会因此多耗数分钟，甚至吃满整题超时。复用原审核摘要和问题
        # 严重度计数，限制短输出，并为修复调用设置独立的时间上限。
        retry_result: Dict[str, Any] = {}
        if isinstance(result, dict) and isinstance(result.get("overall_assessment"), dict):
            issue_counts = {"critical": 0, "major": 0, "minor": 0}
            for issue in result.get("issues", []):
                if isinstance(issue, dict) and issue.get("severity") in issue_counts:
                    issue_counts[issue["severity"]] += 1

            repair_prompt = self.SCORE_REPAIR_PROMPT.format(
                assessment=json.dumps(
                    result["overall_assessment"], ensure_ascii=False
                ),
                issue_counts=json.dumps(issue_counts, ensure_ascii=False),
            )
            try:
                retry_response = await asyncio.wait_for(
                    self.call_llm(
                        system_prompt=(
                            "你是评分字段校验器，只修复已有审核的评分，"
                            "不重新审核报告。"
                        ),
                        user_prompt=repair_prompt,
                        json_mode=True,
                        temperature=0.0,
                        max_tokens=256,
                    ),
                    timeout=120,
                )
                retry_result = self.parse_json_response(retry_response)
            except (AgentHistoryError, ContextBudgetExceeded):
                raise
            except asyncio.TimeoutError:
                self.logger.warning(
                    "[CriticMaster] 评分字段纠错超过 120 秒，按最低合法分降级"
                )
            except Exception as exc:
                self.logger.warning(
                    "[CriticMaster] 评分字段纠错失败，按最低合法分降级: %s",
                    exc,
                )

        if self._normalize_review_result(retry_result):
            repaired = retry_result["overall_assessment"]
            assessment = result["overall_assessment"]
            assessment["raw_quality_score"] = raw_score
            assessment["quality_score"] = repaired["quality_score"]
            assessment["verdict"] = repaired["verdict"]
            assessment["score_status"] = "valid_after_retry"
            return result

        # 评分修复只返回评估字段，必须保留已校验的首轮 issues。
        # score_status 会一路透传到 benchmark 元数据，避免把降级分误当模型原始分。
        fallback = result
        if not isinstance(fallback, dict):
            fallback = {}
        assessment = fallback.setdefault("overall_assessment", {})
        if not isinstance(assessment, dict):
            assessment = {}
            fallback["overall_assessment"] = assessment
        assessment["raw_quality_score"] = assessment.get("quality_score", raw_score)
        assessment["quality_score"] = 1
        assessment["score_status"] = "fallback_invalid_model_output"
        if assessment.get("verdict") == "pass" or not assessment.get("verdict"):
            assessment["verdict"] = "major_issues"
        assessment.setdefault(
            "summary",
            "审核模型连续两次返回非法评分，系统按最低合法分 1 分降级处理。",
        )
        return fallback

    @staticmethod
    def _normalize_review_result(result: Any) -> bool:
        """原地规范审核结果；只有评分和 verdict 均可安全使用时返回 True。"""
        if not isinstance(result, dict):
            return False
        assessment = result.get("overall_assessment")
        if not isinstance(assessment, dict):
            return False

        # 引用质量不再由模型打分；只接收报告审核的三个维度。
        result.pop("fact_check_results", None)
        result.pop("citation_audit", None)
        dimensions = assessment.get("dimension_scores")
        assessment["dimension_scores"] = {
            name: score
            for name in REPORT_DIMENSION_WEIGHTS
            if isinstance(dimensions, dict)
            and (score := normalize_quality_score(dimensions.get(name))) is not None
        }
        raw_score = assessment.get("quality_score")
        score = normalize_quality_score(raw_score)
        derived = derive_quality_score(result)
        # 三项齐全时统一计算；评分纠错响应仍可只返回总分与判定。
        valid_dimension_count = len(assessment["dimension_scores"])
        if derived and (score is None or valid_dimension_count >= 3):
            assessment["raw_quality_score"] = raw_score
            score = derived["score"]
            assessment["score_status"] = derived["status"]
            assessment["issue_counts"] = derived["issue_counts"]
        elif score is not None:
            assessment["score_status"] = "valid"
        else:
            return False

        assessment["quality_score"] = score
        if score < 7 and assessment.get("verdict") == "pass":
            assessment["verdict"] = "needs_revision"
        elif not assessment.get("verdict"):
            assessment["verdict"] = "pass" if score >= 7 else "needs_revision"
        return True
