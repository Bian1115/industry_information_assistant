# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
DeepResearch V2.0 - 首席笔杆 Agent (LeadWriter)

职责：
1. 深度写作 - 将零散信息串联成逻辑严密的报告
2. Markdown排版 - 专业的格式排版
3. 图文混排 - 整合文字、图表、数据
4. 参考文献 - 规范的引用格式
"""

import json
import re
import uuid
from copy import deepcopy
from hashlib import sha256
from typing import Dict, Any, List, Mapping
from urllib.parse import urlsplit, urlunsplit

from langgraph.config import get_config
from langgraph.func import task

from ..evidence import fact_views, hypothesis_evidence_views
from ..hypothesis_policy import hypotheses_enabled
from ..writer_claims import (
    CLAIM_INSTRUCTIONS,
    SYNTHESIS_CLAIM_INSTRUCTIONS,
    bind_synthesis_claims,
    build_synthesis_claim_index,
    collect_report_claims,
    format_synthesis_claim_index,
    normalize_claims,
)
from .base import BaseAgent
from service.session_evidence_rag import BEGIN, END, evidence_prompt, revision_evidence_prompt
from service.web_document import source_available
from ..state import ResearchState, ResearchPhase, reopen_sections_for_rewrite
from ..quality_pipeline import (
    evaluate_report_gate,
    MARKDOWN_LINK,
)
from ..issue_workflow import (
    ISSUE_RESPONSE_INSTRUCTIONS, apply_writer_records, issue_query,
    open_issues, record_writer_responses, revision_context, section_issues,
)


class InsufficientEvidenceError(ValueError):
    """没有可用原文时，拒绝生成看似完整的正式报告。"""


class LeadWriter(BaseAgent):
    """
    首席笔杆 - 最终输出的打磨者

    特点：
    - 深度写作能力
    - 专业的行业研究报告风格
    - 逻辑严密的叙述结构
    - 规范的引用和排版
    """

    # 类级默认值兼容通过 __new__ 构造的轻量测试；正常运行时会由配置覆盖。
    target_report_chars = 7500
    min_section_chars = 300
    max_section_chars = 500
    summary_chars = 250
    outlook_chars = 150
    max_tokens = 60000

    SECTION_WRITING_PROMPT = """

## 研究主题
{query}

## 当前章节信息
标题: {section_title}
描述: {section_description}
类型: {section_type}

### 本章节必须响应的用户显式要求
{specific_requirements}

## 可用素材

### 可逐字引用的原文片段（片段以外的内容不可据此声称已引用）
{claim_evidence}

{claim_instructions}

### 假设与三类证据账本
{hypothesis_ledger}

### 数据点
{data_points}

### 相关图表
{charts_info}

### 已完成章节的术语上下文
{terminology_context}

## 写作要求
1. **专业性**：使用行业术语，体现专业深度
2. **逻辑性**：论点清晰，论据充分，层层递进
3. **数据支撑**：关键观点必须有数据或事实支撑
4. **正文行内引用**：在 content 的关键事实、数字和依赖外部来源的判断后，直接插入可点击链接 [来源名称](URL)；表格须在对应数据单元格或同一行的来源列插入链接，明确引用对应的内容。
   URL 必须逐字来自本次提供的原文证据；没有 URL 时不得自行编造链接。
   参考文献由程序从实际行内引用生成；不要另外输出 citations 或 references 数组。
5. **图表整合**：在合适位置插入图表引用 ![图表标题](chart_id)
6. **结构化数据必须用表格**：凡涉及多主体对比、分层/分类结构、时间序列、多个指标罗列、排名或对象清单，
   必须使用 Markdown 表格作为主要呈现方式，不能只用连续段落或项目符号替代表格。表头须明确对象、指标、单位、
   时间/统计口径和来源（适用时）；同一列保持可比口径。表格后补充解读，不要只贴表。素材不足时明确标注“暂无可靠数据”
   或口径差异，严禁为填满表格而编造数值
7. **术语界定与全文一致**：关键术语首次出现时必须给出明确的操作性定义、统计口径或边界，并说明容易混淆概念之间的区别。
   后文及表格必须沿用同一术语和口径；若不同来源定义不一致，先解释差异和映射方法，再比较数据，不能无说明地混用或换名
8. **显式要求不得遗漏**：完整执行上方“本章节必须响应的用户显式要求”。若权威框架或确切数量的官方分类无法证实，
   必须明确写出“未找到该框架的权威定义”或“未找到该确切数量的官方分类”，并给出有来源的替代体系、采用理由及对应关系
9. **篇幅要求**：本章节目标约 **{section_target_chars} 字**，允许范围为 **{min_section_chars}-{max_section_chars} 字**。优先保留结论、关键数据、来源和因果分析，删除重复背景与泛泛表述
10. **不要重复标题**：正文开头不要再写章节标题
11. **深度展开**：每章设置 1-2 个三级标题，只对最重要的 1-2 个核心结论按「结论 → 数据支撑 → 归因分析 → 影响推演」展开；次要内容用表格或简短段落呈现
12. **严禁占位**：绝对不允许出现「此处省略」「内容同上」「见上文」「如前所述」「（略）」
   「待补充」等任何形式的省略或占位表述。每一节都必须写出完整实质内容
13. **缺失值不得参与评分**：若关键指标标注“暂无可靠数据”，不得仍给该对象打分、排名或计算综合分；
   只能停止该项排名，或使用程序已明确提供且权重重新归一化的计算结果，并披露缺失处理方法
14. **正反证据必须同时消费**：若相关假设同时存在支持与反驳证据，正文必须呈现适用边界或冲突，不能只选择支持材料；
   inconclusive 证据应明确写成证据不足或仍有不确定性，不能强行改写为支持结论

## 输出格式
content 必须是已经包含行内引用的完整正文，来源名称和 URL 必须来自提供的素材。
```json
{{
    "content": "章节完整正文（Markdown格式，不包含章节标题；关键事实和数字须在对应位置插入引用，例如：具体事实或数字（[来源名称](已提供的完整URL)）。）",
    "key_points": ["本章节的核心要点"],
    "suggested_improvements": ["如果有更多信息可以改进的地方"],
    "claims": [
        {{"claim": "本章一个原子事实论断", "evidence": [
            {{"document_id": "已提供的正文编号", "sentence_ids": ["已提供的原句编号"]}}
        ]}}
    ]
}}
```
上述示例仅说明字段结构和引用位置；实际输出须填写完整正文，并将来源名称和 URL 替换为本次原文证据中提供的真实值，不得原样输出占位文字。

## 写作风格示例
- 好的开头："2024年，中国AI芯片市场正经历深刻变革。根据[IDC数据](https://www.idc.com)，市场规模达到..."
- 避免的开头："关于AI芯片，首先我们来看..."
- 数据引用示例："市场规模达5000亿元（[艾瑞咨询报告](https://www.iresearch.cn/report)）"

开始撰写："""

    SYNTHESIS_PROMPT = """你是首席笔杆，需要将各章节整合成完整的研究报告。

## 当前问题
{query}

## 当前研究要求
{research_requirements}

## 各章节内容
{sections_content}

## 章节论断索引（编号仅在本次调用有效）
{section_claim_index}

{claim_instructions}
本次 claims 必须覆盖执行摘要、核心结论和未来展望中的事实，不重复收集各章节正文的 claims。

## 任务
基于上面的章节内容，产出报告的**首尾部分**：
1. 撰写执行摘要（Executive Summary）
2. 提炼核心结论，结论正文合计最多 {conclusion_chars} 字，保留适用条件与不确定性
3. 撰写未来展望
参考文献列表由程序从正文、摘要、结论和展望的实际行内引用生成，无需输出。

正文各章节已经写好并由程序拼接，**不在你的任务范围内**。

## 关键要求

### 1. 引用格式规则（确保可点击）
- 行内引用：使用 [来源名称](URL) 格式，如 [艾瑞咨询](https://www.iresearch.cn)
- 数据引用：在数据后标注来源，如"市场规模达5000亿元（[IDC报告](https://www.idc.com)）"
- 只能复用各章节正文中已给出的精确 URL，不得猜测、改写或编造 URL

### 2. 执行摘要的写法
- 约 {summary_chars} 字，独立成篇，读完它就能掌握报告全貌
- 必须包含关键数据与核心判断，不能只是章节标题的罗列
- 不要写"本报告将分析……"这类导语，直接给结论

### 4. 你不需要重写正文（重要）
各章节正文已经写好，**由程序直接拼接**，不需要你重新输出一遍。
你只负责产出下面 JSON 里的执行摘要、结论、展望，以及这些内容对应的 claims。

这样设计的原因：正文若经由你这次调用重新吐出，会受单次输出上限约束，
报告一长就被截断、JSON 破损。拆开之后仍须遵守本次任务的整篇正文上限。

### 5. 严禁任何形式的省略与占位
你产出的每个字段都必须是完整实质内容，不允许出现
「此处省略」「内容同上」「见上文」「如前所述」「（略）」「待补充」「TODO」「同前」。
执行摘要应控制在约 {summary_chars} 字，不能因为"正文里有"就略写。

## 输出格式
只输出下面这个 JSON，**不要输出正文**：

```json
{{
    "executive_summary": "执行摘要全文（约{summary_chars}字，完整内容，不是占位）",
    "conclusions": [
        {{
            "id": "c_1",
            "text": "核心结论1",
            "confidence": "high|medium|low",
            "boundary_conditions": ["该结论成立的条件或不确定性"]
        }}
    ],
    "outlook": "未来展望（约{outlook_chars}字）",
    "claims": [
        {{"claim": "摘要、结论或展望中的一个原子事实论断",
          "report_quote": "本次首尾输出中包含该论断的完整段落，保留行内引用",
          "source_claim_ids": ["C1", "C2"]}}
    ]
}}
```

程序会把你产出的这些字段与已写好的各章节正文组装成最终报告，
结构为：执行摘要 → 各章节正文 → 结论与展望 → 参考文献。"""

    REVISION_PROMPT = """你是首席笔杆，需要根据审核反馈修订报告。

## 原始报告
{original_content}

## 审核反馈
{feedback}

## 补充的新信息
{new_info}

## 任务
根据反馈修订报告，解决指出的问题。

## 修订原则
1. 针对性修改：只修改有问题的部分
2. 补充来源：对缺少来源的观点补充引用
3. 修正错误：纠正事实错误或逻辑漏洞
4. 保持风格：修订后保持报告整体风格一致

输出JSON：
```json
{{
    "revised_content": "修订后的内容",
    "changes_made": ["修改1", "修改2"],
    "addressed_issues": ["已解决的问题ID"],
    "unable_to_address": ["无法解决的问题及原因"]
}}
```"""

    SECTION_REVISION_PROMPT = """你是负责修订研究报告的资深编辑。

## 研究问题
{query}

## 当前章节
章节ID: {section_id}
章节标题: {section_title}

## 原章节正文
{original_content}

## 本章需要处理的审核反馈
{feedback}

## 可用的补充证据
{new_info}

## 可逐字引用的原文片段（包含原章节和补充事实的来源）
{claim_evidence}

{claim_instructions}
claims 必须覆盖修订后的完整章节，包括保留的事实；返回整章新列表，不能只返回变更部分。

## 修订规则
1. 只输出该章节的正文，不要输出报告其他章节，不要重复章节标题。
2. 必须保留原章节中未被指出有问题的内容，不得用摘要代替全文。
3. 只能使用本次提供的事实或原句证据中明确给出的 URL，不得编造链接。
4. 证据不足时应降低结论强度或明确标注局限，不得编造数据、模拟结果或验证结论。
5. 修订后的章节应控制在 {min_section_chars}-{max_section_chars} 字，优先压缩重复背景，不得删掉关键证据与引用。

只输出 JSON：
```json
{{
  "revised_content": "修订后的完整章节正文",
  "addressed_issues": ["已解决的问题ID"],
  "unable_to_address": ["仍无法解决的问题ID及原因"],
  "claims": [
    {{"claim": "修订后的一个原子事实论断", "evidence": [
      {{"document_id": "已提供的正文编号", "sentence_ids": ["已提供的原句编号"]}}
    ]}}
  ]
}}
```"""

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max",
        target_report_chars: int = 7500,
        min_section_chars: int = 300,
        max_section_chars: int = 500,
        summary_chars: int = 250,
        outlook_chars: int = 150,
        max_tokens: int = 60000,
    ):
        super().__init__(
            name="LeadWriter",
            role="首席笔杆",
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            model=model
        )
        self.target_report_chars = max(2000, int(target_report_chars))
        self.min_section_chars = max(300, int(min_section_chars))
        self.max_section_chars = max(
            self.min_section_chars, int(max_section_chars)
        )
        self.summary_chars = max(100, int(summary_chars))
        self.outlook_chars = max(80, int(outlook_chars))
        self.max_tokens = max(1000, int(max_tokens))

    def _writing_budget(self, state: ResearchState) -> Dict[str, Any]:
        """先按当前任务的整篇上限分配，默认章节下限不能覆盖用户上限。"""
        from ..report_constraints import body_character_count
        outline = state.get("outline") or []
        limit = (state.get("active_constraints") or {}).get("max_body_chars")
        total = limit if limit is not None else self.target_report_chars
        headings = "执行摘要核心结论未来展望结论与展望" + "".join(
            f"{i} {section.get('title') or f'章节{i}'}" for i, section in enumerate(outline, 1))
        structure = body_character_count(headings) + 20 + len(outline) * 4
        margin = max(20, total // 20)
        available = total - structure - margin
        if available < len(outline) + 3:
            raise ValueError("正文上限不足以容纳当前章节标题和报告结构，请调整篇幅或大纲")
        summary = min(self.summary_chars, max(1, available * 12 // 100))
        conclusions = min(200, max(1, available * 10 // 100))
        outlook = min(self.outlook_chars, max(1, available * 8 // 100))
        per_section = (available - summary - conclusions - outlook) // max(1, len(outline))
        maximum = min(self.max_section_chars, per_section)
        sections = {s["id"]: {"min_chars": min(self.min_section_chars, maximum),
                                "max_chars": maximum, "target_chars": maximum} for s in outline}
        budget = {"source": "user" if limit is not None else "default", "max_body_chars": limit,
                  "target_body_chars": total, "summary_chars": summary,
                  "conclusion_chars": conclusions, "outlook_chars": outlook,
                  "structure_chars": structure, "safety_chars": margin, "sections": sections}
        state["writing_budget"] = budget
        return budget

    async def process(self, state: ResearchState) -> ResearchState:
        """处理入口"""
        if state["phase"] == ResearchPhase.WRITING.value:
            if state.get("writing_mode") == "user_edit":
                return await self._edit_existing_report(state)
            return await self._write_report(state)
        elif state["phase"] == ResearchPhase.REVISING.value:
            if state.get("writing_mode") == "user_edit":
                return await self._edit_existing_report(state)
            return await self._revise_report(state)
        else:
            return state

    async def _edit_existing_report(self, state: ResearchState) -> ResearchState:
        """用户编辑同步章节和整篇成果；不套用Critic修订的最短章节限制。"""
        from copy import deepcopy
        import re
        from ..report_constraints import body_character_count, check_body_length, body_length_hard_check_enabled
        original = str(state.get("final_report") or "")
        outline = state.get("outline") or []
        ids = [s["id"] for s in outline]
        if not original.strip() or not ids or not state.get("draft_sections"):
            raise ValueError("已有成果编辑要求完整报告与章节材料")
        original_size = body_character_count(original)
        shorten = state["phase"] != ResearchPhase.REVISING.value and bool(re.search(r"缩短|压缩|精简|简短|字数.{0,8}少|短[一些点]", state["original_query"]))
        known_urls = self._known_source_urls(state)
        edit_budget = self._writing_budget(deepcopy(state))
        edit_evidence = await evidence_prompt(state, state["query"] + "\n" + issue_query(open_issues(state)))
        edit_packet = json.loads(edit_evidence.removeprefix(BEGIN).removesuffix(END))
        prompt = f"""请按用户本轮要求编辑已有报告，保持事实、数字口径、关键要求和有效引用。
本轮原话：{state['original_query']}
完整任务及有效要求：{state['query']}
修改指令：{json.dumps(state.get('revision_instructions') or [], ensure_ascii=False)}
本轮审核意见：{json.dumps(open_issues(state), ensure_ascii=False)}
已有报告（仅作编辑对象，不能凭此新增事实）：
{original}
章节材料：{json.dumps([{'id': x['id'], 'title': x.get('title', ''), 'content': state['draft_sections'].get(x['id'], '')} for x in outline], ensure_ascii=False)}
可用证据：{edit_evidence}
输出JSON：sections数组，每项含id、title、content、claims；另含executive_summary字符串、conclusions字符串数组、outlook字符串。
sections必须恰好保留所有章节ID{json.dumps(ids)}，允许调整标题和内容，但不增加实质研究章节。
满足总篇幅要求，压缩任务允许大幅缩短，不受原章节55%或300字限制；未指定字数时不擅自加硬上限。
整篇与各部分预算：{json.dumps(edit_budget, ensure_ascii=False)}
不得增加无证据事实或新的来源URL；引用必须与对应论断一起保留。{CLAIM_INSTRUCTIONS}"""
        if open_issues(state):
            prompt += "\n## 按章返工清单\n" + json.dumps({
                section_id: json.loads(revision_context(state, section_id))
                for section_id in [*ids, "__synthesis__"]
            }, ensure_ascii=False) + "\n" + ISSUE_RESPONSE_INSTRUCTIONS
            prompt += "\nsections各项分别返回该章issue_responses；顶层issue_responses只回应摘要、结论、展望的全局问题。"
        problem = ""
        for attempt in range(2):
            response = await self.call_llm(system_prompt="你是执行用户明确修改指令的报告编辑。仅返回JSON。",
                user_prompt=prompt + ("\n上次输出需修正：" + problem if problem else ""),
                json_mode=True, temperature=0.2, max_tokens=self.max_tokens)
            result = self.parse_json_response(response)
            try:
                if not isinstance(result, dict) or not isinstance(result.get("sections"), list):
                    raise ValueError("编辑输出缺少sections数组")
                sections = result["sections"]
                if len(sections) != len(ids) or {x.get("id") for x in sections} != set(ids):
                    raise ValueError("编辑输出章节ID不完整或重复")
                if any(not isinstance(x.get("content"), str) or not x["content"].strip()
                       or not isinstance(x.get("title"), str) or not x["title"].strip() for x in sections):
                    raise ValueError("编辑章节标题和正文不能为空")
                if any(not isinstance(result.get(k), str) for k in ("executive_summary", "outlook")):
                    raise ValueError("编辑摘要和展望必须是字符串")
                conclusions = result.get("conclusions")
                if not isinstance(conclusions, list) or any(not isinstance(x, str) for x in conclusions):
                    raise ValueError("编辑结论必须为字符串数组")
                proposal = deepcopy(state)
                by_id = {x["id"]: x for x in sections}
                for section in proposal["outline"]:
                    section["title"] = by_id[section["id"]]["title"]
                    section["status"] = "final"
                proposal["draft_sections"] = {x["id"]: self._sanitize_citation_links(x["content"], known_urls) for x in sections}
                proposal["section_claims"] = {x["id"]: self._read_claims(x) for x in sections}
                proposal["synthesis_claims"] = self._read_claims(result)
                proposal["final_report"] = self._assemble_report(proposal, result["executive_summary"], conclusions, result["outlook"])
                proposal["claims"] = collect_report_claims(proposal)
                check_body_length(proposal)
                if shorten and body_character_count(proposal["final_report"]) >= original_size:
                    proposal["body_length_check"]["shorten_requirement_met"] = False
                    proposal["body_length_check"]["shorten_failure_reason"] = "用户要求缩短，但编辑结果没有缩短正文"
                    if body_length_hard_check_enabled():
                        raise ValueError("用户要求缩短，但编辑结果没有缩短正文")
                gate = evaluate_report_gate(proposal)
                if not gate["passed"]:
                    raise ValueError("编辑报告发布检查未通过：" + ";".join(gate["blocking_reasons"]))
                proposal.setdefault("quality_gates", {})["report"] = gate
                state.update(proposal)
                break
            except (ValueError, TypeError, KeyError) as exc:
                problem = str(exc)
                if attempt == 1:
                    raise ValueError("用户编辑未生成有效报告：" + problem) from exc
        verification = await self._verify_and_repair_citations(state)
        for section_result in result["sections"]:
            section_id = section_result["id"]
            record_writer_responses(state, section_id, state["draft_sections"][section_id], section_result, edit_packet)
        record_writer_responses(state, "__synthesis__", state["final_report"], result, edit_packet)
        state["quality_gates"]["report"]["citation_verification"] = verification["status"]
        state["user_edit_result"] = {"before_body_chars": original_size,
            "after_body_chars": body_character_count(state["final_report"]), "attempts": attempt + 1}
        state["phase"] = ResearchPhase.REVIEWING.value
        self.add_message(state, "report_draft", {"agent": self.name, "content": state["final_report"],
            "word_count": len(state["final_report"]), "edit_result": state["user_edit_result"],
            "citation_verification": verification, "citation_repair": state.get("citation_repair", {})})
        return state

    async def _write_report(self, state: ResearchState) -> ResearchState:
        """撰写报告"""
        if not any(source_available(source) for source in state.get("raw_sources") or []):
            message = (
                "检索阶段未获得任何可用原文，已停止生成正式研究报告，"
                "以避免模型在无证据情况下补全数据和引用。"
            )
            if message not in state["errors"]:
                state["errors"].append(message)
            self.add_message(state, "warning", {
                "agent": self.name,
                "content": message,
                "error_code": "insufficient_evidence",
            })
            raise InsufficientEvidenceError(message)

        # 发送 research_step 开始事件
        # 注意: step_type 必须是 "writing" 以匹配 graph.py 发送的 phase 事件
        self.add_message(state, "research_step", {
            "step_id": f"step_writing_{uuid.uuid4().hex[:8]}",
            "step_type": "writing",
            "title": "内容生成",
            "subtitle": "撰写研究报告",
            "status": "running",
            "stats": {"sections_count": len(state["outline"]), "word_count": 0}
        })

        self.add_message(state, "thought", {
            "agent": self.name,
            "content": "开始撰写深度研究报告..."
        })

        # 逐章节撰写
        self._writing_budget(state)
        for section in state["outline"]:
            if section.get("status") not in ["final", "drafted"]:
                await self._write_checkpointed_section(state, section)

        # 整合报告
        await self._synthesize_report(state)

        # 发送 research_step 完成事件
        word_count = len(state.get("final_report", ""))
        self.add_message(state, "research_step", {
            "step_type": "writing",
            "title": "内容生成",
            "subtitle": "撰写研究报告",
            "status": "completed",
            "stats": {
                "sections_count": len(state["outline"]),
                "word_count": word_count,
                "references_count": len(state.get("references", []))
            }
        })

        # 更新阶段
        state["phase"] = ResearchPhase.REVIEWING.value

        return state

    async def _write_checkpointed_section(self, state: ResearchState, section: Dict) -> None:
        """独立保存生成产物；节点重入时重建章节，合稿后才统一核验。

        图内 task 的输入/返回值可序列化，运行对象留在闭包里。直接调用
        Writer 的离线入口沿用普通方法，不冒充具有持久化恢复能力。
        """
        try:
            config = get_config()
        except RuntimeError:
            config = {}
        # get_config 也可能来自普通 Runnable，只有图的任务调度上下文
        # 才能调用 task；此键对应当前支持的 LangGraph 1.x 运行协议。
        if not config.get("configurable", {}).get("__pregel_call"):
            await self._write_section(state, section)
            return

        from ..graph import pack_state, RUNTIME_ONLY_KEYS

        saved = deepcopy(pack_state(state)["research"])
        from service.session_evidence_rag import current_session_evidence
        rag = current_session_evidence()
        if rag is not None:
            rag.externalize(saved)
        stable = {key: value for key, value in saved.items()
                  if key not in ({"node_attempts", "runtime_budget", "storage_manifest", "evidence_rag_metrics"}
                                 if saved.get("storage_protocol") else {"node_attempts", "runtime_budget"})}
        identity = {
            "version": 1, "section_id": section["id"], "research": stable,
            "model": getattr(self, "model", ""), "max_tokens": self.max_tokens,
            "section_prompt": self.SECTION_WRITING_PROMPT,
            "issue_response_prompt": ISSUE_RESPONSE_INSTRUCTIONS,
            "min_section_chars": self.min_section_chars,
            "max_section_chars": self.max_section_chars,
            "target_report_chars": self.target_report_chars,
            "summary_chars": self.summary_chars, "outlook_chars": self.outlook_chars,
        }
        input_hash = sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()
        from ..checkpoint_events import EVENTS_CHANNEL, event_intent

        @task(name="writer_section_draft_v1")
        async def generate(payload: dict) -> dict:
            work = deepcopy(payload["research"])
            work.update({key: state[key] for key in RUNTIME_ONLY_KEYS if key in state})
            events = []
            emit = work.get("_emit_event")

            def capture(event):
                if event.get("type") in {"section_content", "observation"}:
                    events.append(event_intent(event))
                elif emit is not None:
                    emit(event)

            work["_emit_event"] = capture
            target = next(item for item in work["outline"] if item["id"] == payload["section_id"])
            await self._write_section(work, target)
            content = work["draft_sections"].get(target["id"], "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("章节未产生有效正文，拒绝保存章节产物")
            artifact = {
                "version": 1, "section_id": target["id"], "input_hash": payload["input_hash"],
                "content": content, "claims": deepcopy(work.get("section_claims", {}).get(target["id"], [])),
                "citation_verification": "not_evaluated",
                EVENTS_CHANNEL: events,
                "writer_revision_records": deepcopy([
                    row for row in work.get("writer_revision_records") or []
                    if row["section_id"] == target["id"] and row["iteration"] == int(work.get("iteration", 0))
                ]),
            }
            from ..storage_reconciliation import current_storage_coordinator
            coordinator = current_storage_coordinator()
            if coordinator is not None:
                artifact = await coordinator.seal(artifact, saved, node="write:" + str(target["id"]), kind="writer_section")
            return artifact

        artifact = await generate({"research": saved, "section_id": section["id"], "input_hash": input_hash})
        if (not isinstance(artifact, dict) or artifact.get("version") != 1
                or artifact.get("section_id") != section["id"]
                or artifact.get("input_hash") != input_hash
                or not isinstance(artifact.get("content"), str) or not artifact["content"].strip()
                or not isinstance(artifact.get("claims"), list)):
            raise ValueError("已保存章节与当前输入或写作配置不一致，请重新发起研究")
        # 首次执行与恢复重放都应用同一份正文/claims；按章节 ID 替换，
        # 不向列表重复追加。来源目录从当前整份草稿确定性重建。
        state.setdefault("draft_sections", {})[section["id"]] = artifact["content"]
        state.setdefault("section_claims", {})[section["id"]] = deepcopy(artifact["claims"])
        apply_writer_records(state, deepcopy(artifact.get("writer_revision_records") or []))
        section["status"] = "drafted"
        state["references"] = self._references_from_content(state, self._draft_content(state))
        # 恢复复用原产物及原事件。发送器从账本重放，不再创建 restored 新事件。

    async def _write_section(self, state: ResearchState, section: Dict) -> None:
        """撰写单个章节"""
        section_id = section["id"]
        section_budget = self._writing_budget(state)["sections"].get(section_id, {
            "target_chars": self.max_section_chars, "min_chars": self.min_section_chars,
            "max_chars": self.max_section_chars})
        self.logger.info(f"Writing section: {section.get('title')}")

        self.add_message(state, "action", {
            "agent": self.name,
            "tool": "writing_section",
            "section": section.get("title")
        })

        # 仅为历史检查点的原句绑定保留事实 ID；不把事实摘要重新注入写作。
        related_facts = [f for f in fact_views(state, verified_only=True) if section_id in f.get("related_sections", [])]
        if not related_facts:
            related_facts = fact_views(state, verified_only=True)[:10]

        # 格式化数据点。不能再固定截取全局前10条，否则“10家公司×多个指标”
        # 的后半家公司永远进不了写作提示词。
        related_urls = {str(source.get("url") or source.get("source_url") or "")
                        for source in state.get("raw_sources") or []
                        if section_id in (source.get("related_sections") or []) and source_available(source)}
        related_points = [
            point for point in state["data_points"]
            if str(point.get("source") or "") in related_urls
        ]
        data_text = []
        for dp in related_points[:60]:
            data_text.append(
                f"- 对象={dp.get('entity') or '未标注'}；指标={dp.get('name')}；"
                f"数值={dp.get('value')} {dp.get('unit', '')}；"
                f"年份/期间={dp.get('year') or dp.get('period') or 'N/A'}；"
                f"口径={dp.get('accounting_standard') or '未标注'}；"
                f"来源={dp.get('source') or '无'}"
            )

        # 格式化图表信息
        charts_info = []
        for chart in state["charts"]:
            if chart.get("section_id") == section_id:
                charts_info.append(f"- 图表: {chart.get('title')} (ID: {chart.get('id')})")

        # 架构师已将用户的明确框架/数量要求下沉到对应章节；写作时再次显式展示，
        # 避免长报告中只完成主题分析却漏掉用户指定的结构。
        specific_requirements = []
        for requirement in (state.get("research_contract") or {}).get("explicit_requirements", []):
            if not isinstance(requirement, dict):
                continue
            if requirement.get("id") not in (section.get("requirement_ids") or []):
                continue
            line = (
                f"- [{requirement.get('id', 'requirement')}] 原始要求："
                f"{requirement.get('original_requirement', '')}；"
                f"规划响应：{requirement.get('planned_response', '')}；"
                f"权威性状态：{requirement.get('authority_status', 'to_verify')}；"
                f"无权威定义时：{requirement.get('fallback_if_unverified', '')}"
            )
            if requirement.get("alternative"):
                line += (
                    f"；替代体系：{requirement.get('alternative')}；"
                    f"替代理由：{requirement.get('alternative_rationale', '')}"
                )
            specific_requirements.append(line)
        # 自动量化契约已停用；旧检查点的指标范围/年份可能已被拆成字符。
        # 用户要求仍来自原问题及 explicit_requirements，不把这些旧字段再注入写作。

        # 逐章独立调用模型时提供少量既有正文，帮助后续章节沿用已经建立的术语定义和口径。
        terminology_context_parts = []
        for previous_section in state["outline"]:
            previous_id = previous_section.get("id")
            previous_content = (state["draft_sections"].get(previous_id) or "").strip()
            if not previous_content:
                continue
            terminology_context_parts.append(
                f"- {previous_section.get('title', previous_id)}：{previous_content[:1000]}"
            )
        terminology_context = "\n".join(terminology_context_parts)[:4000]

        pending_issues = section_issues(state, section_id)
        claim_evidence = await evidence_prompt(state,
            ((state.get("session_task") or {}).get("resolved_query") or state["query"]) + "\n" + section.get("title", "") + "\n" + section.get("description", "") + "\n" + issue_query(pending_issues),
            [str(f.get("id") or "") for f in related_facts])
        evidence_payload = json.loads(claim_evidence.removeprefix(BEGIN).removesuffix(END))
        if not (evidence_payload.get("evidence_units")
                or any(document.get("original_sentences") for document in evidence_payload.get("documents", []))
                or any(source.get("text") for source in evidence_payload.get("legacy_sources", []))):
            raise InsufficientEvidenceError(f"章节「{section.get('title', section_id)}」未检索到可用原文证据")

        prompt = self._hypothesis_prompt(self.SECTION_WRITING_PROMPT, state).format(
            query=state["query"],
            section_title=section.get("title", ""),
            section_description=section.get("description", ""),
            section_type=section.get("section_type", "mixed"),
            specific_requirements=(
                "\n".join(specific_requirements)
                if specific_requirements else "（本章节无单独映射的显式数量/结构要求，仍须遵守原问题范围）"
            ),
            claim_evidence=claim_evidence,
            claim_instructions=CLAIM_INSTRUCTIONS,
            hypothesis_ledger=self._format_hypothesis_ledger(state),
            data_points="\n".join(data_text) if data_text else "（暂无数据点）",
            charts_info="\n".join(charts_info) if charts_info else "（暂无图表）",
            terminology_context=terminology_context or "（尚无已完成章节；请在本章首次出现时建立清晰术语定义）",
            section_target_chars=section_budget["target_chars"],
            min_section_chars=section_budget["min_chars"],
            max_section_chars=section_budget["max_chars"],
        )
        if pending_issues:
            prompt += ("\n## 本章按问题返工清单\n" + revision_context(state, section_id)
                       + "\n## 返工前章节正文（只作编辑对象）\n"
                       + (state.get("revision_original_sections") or {}).get(section_id, "")
                       + "\n" + ISSUE_RESPONSE_INSTRUCTIONS)
        elif state.get("quality_review_summary") and state.get("quality_verdict") != "pass":
            prompt += ("\n## 本轮审核总评（未列出单章问题，请据此改进）\n"
                       + state["quality_review_summary"]
                       + "\n## 返工前章节正文\n"
                       + (state.get("revision_original_sections") or {}).get(section_id, ""))

        result = {}
        for attempt in range(2):
            retry_prompt = prompt
            if attempt:
                retry_prompt += (
                    "\n\n上一轮输出不是合法JSON或缺少content。"
                    "本次只输出结构正确、content非空的JSON，不要添加代码块外文字。"
                )
            response = await self.call_llm(
                system_prompt="你是顶级的行业研究分析师，擅长撰写专业的研究报告。",
                user_prompt=retry_prompt,
                json_mode=True,
                temperature=0.4 if attempt == 0 else 0.2,
                max_tokens=self.max_tokens,
            )
            result = self.parse_json_response(response)
            if result and str(result.get("content") or "").strip():
                break

        if not result or not str(result.get("content") or "").strip():
            message = f"章节写作失败：{section.get('title')} 未返回有效正文"
            state.setdefault("errors", []).append(message)
            raise ValueError(message)

        if result and result.get("content"):
            section_content = self._sanitize_citation_links(
                result["content"], self._known_source_urls(state)
            )
            state["draft_sections"][section_id] = section_content
            state.setdefault("section_claims", {})[section_id] = self._read_claims(result)
            record_writer_responses(state, section_id, section_content, result, evidence_payload)
            section["status"] = "drafted"

            # 从实际正文链接生成来源目录，不接受模型另报的来源列表。
            state["references"] = self._references_from_content(
                state, self._draft_content(state)
            )

            # 发送章节内容到"过程报告" - 包含完整内容用于流式显示
            self.add_message(state, "section_content", {
                "agent": self.name,
                "section_id": section_id,
                "section_title": section.get("title"),
                "content": section_content,  # 完整章节内容
                "word_count": len(section_content),
                "key_points": result.get("key_points", []),
                "claims": state["section_claims"][section_id],
            })

            # 发送观察消息（显示在左侧步骤流程）
            self.add_message(state, "observation", {
                "agent": self.name,
                "content": f"章节「{section.get('title')}」撰写完成\n字数: {len(section_content)}\n要点: {', '.join(result.get('key_points', [])[:2]) if result.get('key_points') else '无'}"
            })

    def _read_claims(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(result.get("claims"), list):
            self.logger.warning("[LeadWriter] 未返回有效 claims 数组；引用声明缺失，未执行核验")
        return normalize_claims(result.get("claims"))

    @staticmethod
    def _conclusion_text(conclusion: Any) -> str:
        if isinstance(conclusion, Mapping):
            return str(
                conclusion.get("text")
                or conclusion.get("conclusion")
                or ""
            ).strip()
        return str(conclusion or "").strip()

    def _format_hypothesis_ledger(self, state: ResearchState) -> str:
        """给 Writer 提供紧凑、带稳定 ID 的三类假设证据。"""
        if not hypotheses_enabled(state):
            return ""
        evidence_by_hypothesis: Dict[str, List[Dict[str, Any]]] = {}
        for evidence in hypothesis_evidence_views(state):
            if not isinstance(evidence, Mapping):
                continue
            hypothesis_id = str(evidence.get("hypothesis_id") or "")
            if not hypothesis_id:
                continue
            evidence_by_hypothesis.setdefault(hypothesis_id, []).append({
                "evidence_id": evidence.get("id"),
                "fact_id": evidence.get("fact_id"),
                "type": evidence.get("evidence_type"),
            })

        payload = []
        for hypothesis in state.get("hypotheses", []) or []:
            if not isinstance(hypothesis, Mapping):
                continue
            hypothesis_id = str(hypothesis.get("id") or "")
            payload.append({
                "hypothesis_id": hypothesis_id,
                "content": hypothesis.get("content"),
                "status": hypothesis.get("status", "unverified"),
                "evidence": evidence_by_hypothesis.get(hypothesis_id, []),
            })
        if not payload:
            return "（未生成研究假设）"
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _normalize_conclusion_evidence_map(
        self, conclusions: List[Any], state: ResearchState
    ) -> List[Dict[str, Any]]:
        """过滤模型编造的 ID，生成稳定的结论证据映射。"""
        if not hypotheses_enabled(state):
            return []
        known_hypotheses = {
            str(item.get("id"))
            for item in state.get("hypotheses", []) or []
            if isinstance(item, Mapping) and item.get("id")
        }
        evidence_meta_by_id = {
            str(item.get("id")): (
                str(item.get("evidence_type") or ""),
                str(item.get("hypothesis_id") or ""),
            )
            for item in state.get("hypothesis_evidence", []) or []
            if isinstance(item, Mapping) and item.get("id")
        }
        type_for_field = {
            "supporting_evidence_ids": "supports",
            "contradicting_evidence_ids": "refutes",
            "inconclusive_evidence_ids": "inconclusive",
        }
        result = []
        seen_conclusion_ids = set()
        for index, conclusion in enumerate(conclusions, 1):
            raw = dict(conclusion) if isinstance(conclusion, Mapping) else {}
            text = self._conclusion_text(conclusion)
            if not text:
                continue
            proposed_id = str(raw.get("id") or f"c_{index}").strip()
            conclusion_id = proposed_id if proposed_id not in seen_conclusion_ids else f"c_{index}"
            seen_conclusion_ids.add(conclusion_id)
            mapping = {
                "conclusion_id": conclusion_id,
                "conclusion": text,
                "hypothesis_ids": [
                    str(value) for value in raw.get("hypothesis_ids", [])
                    if str(value) in known_hypotheses
                ],
                "confidence": str(raw.get("confidence") or "low"),
                "boundary_conditions": [
                    str(value).strip()
                    for value in raw.get("boundary_conditions", [])
                    if str(value).strip()
                ],
            }
            for field, expected_type in type_for_field.items():
                mapping[field] = [
                    str(value) for value in raw.get(field, [])
                    if evidence_meta_by_id.get(str(value), ("", ""))[0]
                    == expected_type
                    and evidence_meta_by_id.get(str(value), ("", ""))[1]
                    in mapping["hypothesis_ids"]
                ]
            mapping["traceability_status"] = (
                "mapped"
                if mapping["hypothesis_ids"]
                and any(mapping[field] for field in type_for_field)
                else "unmapped"
            )
            result.append(mapping)
        return result

    @staticmethod
    def _synthesis_requirements(state: ResearchState, budget: Dict[str, Any]) -> str:
        """只展示用户要求和当前预算，避免把旧契约的素材带回整合输入。"""
        contract = state.get("research_contract") or {}
        requirements = {
            "本轮用户原话": state.get("original_query") or state["query"],
            "追问及修订要求": state.get("revision_instructions") or contract.get("followup_instructions") or [],
            "正文总字数上限": budget["max_body_chars"],
            "本次首尾预算": {
                "执行摘要": budget["summary_chars"],
                "核心结论合计": budget["conclusion_chars"],
                "未来展望": budget["outlook_chars"],
            },
        }
        for field in ("language", "output_format", "required_entities", "time_range", "geography", "critical_questions"):
            if contract.get(field):
                requirements[field] = contract[field]
        explicit = [
            item["original_requirement"]
            for item in contract.get("explicit_requirements") or []
            if isinstance(item, Mapping) and item.get("original_requirement")
        ]
        if explicit:
            requirements["用户显式要求"] = explicit
        return json.dumps(requirements, ensure_ascii=False, separators=(",", ":"))

    async def _synthesize_report(self, state: ResearchState) -> None:
        """整合完整报告"""
        # 新版本生成/门禁失败时不能沿用上一版本的核验通过状态。
        from ..state import citation_verification_status
        state["citation_verification"] = citation_verification_status()
        state["report_verification_claims"] = []
        state["report_claim_extraction"] = {}
        state["citation_repair"] = {"status": "not_executed", "reason": "新报告尚未执行引用失败处理"}
        if (state.get("citation_verification_config") or {}).get("report_claims"):
            state["citation_verification"]["reason"] = "本版报告尚未执行引用核验"
        self.add_message(state, "thought", {
            "agent": self.name,
            "content": "正在整合各章节，生成完整研究报告..."
        })

        # 准备各章节内容
        sections_content = []
        for section in state["outline"]:
            section_id = section["id"]
            content = state["draft_sections"].get(section_id, "")
            if content:
                sections_content.append(f"## {section.get('title')}（章节编号：{section_id}）\n{content}")

        budget = self._writing_budget(state)
        claim_index = build_synthesis_claim_index(state)
        prompt = self.SYNTHESIS_PROMPT.format(
            query=state["query"],
            research_requirements=self._synthesis_requirements(state, budget),
            sections_content="\n\n".join(sections_content) if sections_content else "（暂无章节内容）",
            section_claim_index=format_synthesis_claim_index(claim_index),
            summary_chars=budget["summary_chars"],
            conclusion_chars=budget["conclusion_chars"],
            outlook_chars=budget["outlook_chars"],
            claim_instructions=SYNTHESIS_CLAIM_INSTRUCTIONS,
        )
        if section_issues(state, "__synthesis__"):
            prompt += ("\n## 摘要、结论与展望的按问题返工清单\n"
                       + revision_context(state, "__synthesis__")
                       + "\n" + ISSUE_RESPONSE_INSTRUCTIONS)

        self.logger.info(f"[LeadWriter] 调用 LLM 生成首尾部分...")
        # 首尾仅归纳章节正文；来源原文留给后续按编号回读的引用核验。
        response = await self.call_llm(
            system_prompt="你是资深的研究报告主编，负责撰写报告的执行摘要、结论与展望。参考文献由程序生成。",
            user_prompt=prompt,
            json_mode=True,
            temperature=0.3,
            max_tokens=self.max_tokens,
        )

        result = self.parse_json_response(response)
        self.logger.info(f"[LeadWriter] JSON 解析结果: {bool(result)}, keys: {result.keys() if result else 'N/A'}")

        executive_summary = (result or {}).get("executive_summary", "")
        conclusions = (result or {}).get("conclusions", []) or []
        if not isinstance(conclusions, list):
            conclusions = []
        outlook = (result or {}).get("outlook", "")
        if not isinstance((result or {}).get("claims"), list):
            self.logger.warning("[LeadWriter] 首尾未返回有效 claims 数组；章节依据缺失")
        state["synthesis_claims"] = bind_synthesis_claims(
            (result or {}).get("claims"), state,
            [str(executive_summary), *[self._conclusion_text(c) for c in conclusions], str(outlook)],
            claim_index=claim_index,
        )
        # 本次未向模型提供假设账本，清除上一版本的假设级映射。
        state["conclusion_evidence_map"] = []

        if not result:
            # 短字段都没解析出来也不致命：正文本来就是程序拼的，
            # 只是少了摘要和结论，报告主体仍然完整
            self.logger.warning("[LeadWriter] ⚠️ 首尾部分解析失败，仅用章节正文组装")

        state["final_report"] = self._assemble_report(
            state, executive_summary, conclusions, outlook
        )
        executive_summary, conclusions, outlook = await self._fit_report_to_budget(
            state, executive_summary, conclusions, outlook)
        state["claims"] = collect_report_claims(state)
        # 发布前的确定性检查只剩两条：报告非空、正文链接不超出来源范围。
        # 可选引用核验在门禁之后单独记录，不能混同为来源范围检查。
        report_gate = evaluate_report_gate(state)
        state.setdefault("quality_gates", {})["report"] = report_gate
        if not report_gate["passed"]:
            out_of_scope = set(report_gate.get("out_of_scope_citations") or [])
            if out_of_scope:
                # 范围外链接先从参考目录剔除再重拼一次；正文里的超范围链接
                # 由下面的复检拦截，不做任何改写正文的自动修复。
                state["references"] = [
                    ref for ref in state.get("references", [])
                    if self._normalize_url(ref.get("url") or ref.get("link"))
                    not in out_of_scope
                ]
                state["final_report"] = self._assemble_report(
                    state, executive_summary, conclusions, outlook
                )
                report_gate = evaluate_report_gate(state)
                state["quality_gates"]["report"] = report_gate
            if not report_gate["passed"]:
                message = "报告发布检查未通过：" + "；".join(
                    report_gate["blocking_reasons"]
                )
                if message not in state["errors"]:
                    state["errors"].append(message)
                self.add_message(state, "warning", {
                    "agent": self.name,
                    "content": message,
                    "error_code": "report_gate_failed",
                    "gate": report_gate,
                })
                # final_report 是对外发布字段；清空它可防止上层在异常分支中
                # 误把未通过发布检查的报告返回前端。
                state["final_report"] = ""
                state["claims"] = []
                raise InsufficientEvidenceError(message)
        verification = await self._verify_and_repair_citations(state)
        record_writer_responses(state, "__synthesis__", state["final_report"], result or {},
                                {"source_claim_index": claim_index})
        state["quality_gates"]["report"]["citation_verification"] = verification["status"]
        if verification["status"] in {"rejected", "incomplete", "error"}:
            self.add_message(state, "warning", {
                "agent": self.name, "content": "可选引用核验发现待复核项，详见核验审计结果",
                "citation_verification": verification,
            })
        self.logger.info(
            f"[LeadWriter] ✅ 报告组装完成，长度: {len(state['final_report'])} 字符"
            f"（章节 {len(state.get('draft_sections') or {})} 节）"
        )

        # 发送报告完成事件 - 包含完整报告内容用于前端流式显示
        self.add_message(state, "report_draft", {
            "agent": self.name,
            "content": state["final_report"],  # 完整报告内容
            "executive_summary": state.get("report_synthesis", {}).get("executive_summary", executive_summary),
            "conclusions": [self._conclusion_text(item) for item in state.get("report_synthesis", {}).get("conclusions", conclusions)],
            "conclusion_evidence_map": state.get("conclusion_evidence_map", []),
            "claims": state.get("claims", []),
            "citation_verification": verification,
            "citation_repair": state.get("citation_repair", {}),
            "word_count": len(state["final_report"]),
            "references_count": len(state["references"])
        })

    async def _verify_and_repair_citations(self, state):
        from ..citation_verification import verify_stage_citations
        from ..citation_repair import repair_report_citations
        verification = await verify_stage_citations(state, "report_claims")
        async def call(prompt):
            return await self.call_llm(system_prompt="你是依据保存原文修订正文的编辑。仅返回JSON。",
                user_prompt=prompt, json_mode=True, temperature=0, max_tokens=self.max_tokens)
        def assemble(proposal):
            parts = proposal["report_synthesis"]
            report = self._assemble_report(proposal, parts["executive_summary"], parts["conclusions"], parts["outlook"])
            proposal["conclusion_evidence_map"] = self._normalize_conclusion_evidence_map(parts["conclusions"], proposal)
            return report
        try:
            return await repair_report_citations(state, call=call, assemble=assemble)
        except Exception as exc:
            # 核验/改写失败是可观测的实验结果，不中断研究交付。
            state.setdefault("citation_repair", {}).update(status="error", reason=str(exc)[:500], error_type=type(exc).__name__)
            return verification

    async def _fit_report_to_budget(self, state, summary, conclusions, outlook):
        """合稿超限时最多重写两次；保留章节结构、证据及引用，不截断原文。"""
        from copy import deepcopy
        from ..report_constraints import body_character_count, check_body_length, body_length_hard_check_enabled
        limit = (state.get("active_constraints") or {}).get("max_body_chars")
        before = body_character_count(state["final_report"])
        if not body_length_hard_check_enabled():
            measured = check_body_length(state)
            state["body_length_repair"] = {
                "before_body_chars": before, "after_body_chars": before,
                "max_body_chars": limit, "attempts": 0,
                "status": "advisory_exceeded" if measured["exceeded"] else "not_needed",
                "reason": "按实验要求只记录篇幅超限，不以篇幅阻断交付或额外缩写"}
            return summary, conclusions, outlook
        if limit is None or before <= limit:
            state["body_length_repair"] = {"before_body_chars": before, "after_body_chars": before,
                                           "max_body_chars": limit, "attempts": 0, "status": "not_needed"}
            return summary, conclusions, outlook
        ids = [section["id"] for section in state["outline"]]
        known_urls = self._known_source_urls(state)
        prompt = (
            f"将以下报告正文缩写到最多{limit}字，字数不计参考文献、空白和链接URL。"
            "保留所有章节ID与章节顺序，保留关键事实、用户要求和对应引用，不增加新事实或来源。"
            "允许降低默认章节字数下限，不可用省略号或占位文字代替实质内容。"
            "返回JSON：sections数组（每项id、content、claims）、executive_summary字符串、"
            "conclusions数组、outlook字符串及摘要结论展望对应的claims数组。\n"
            f"有效任务：{state['query']}\n"
            f"整篇预算：{json.dumps(self._writing_budget(state), ensure_ascii=False)}\n"
            f"原报告：{state['final_report']}\n"
            f"章节ID：{json.dumps(ids, ensure_ascii=False)}\n"
            f"可用原句证据：{await evidence_prompt(state)}\n{CLAIM_INSTRUCTIONS}"
        )
        problem = ""
        for attempt in range(2):
            response = await self.call_llm(
                system_prompt="你是按正文总字数上限缩写报告的编辑，保留事实与引用，仅返回JSON。",
                user_prompt=prompt + ("\n上次输出问题：" + problem if problem else ""),
                json_mode=True, temperature=0.2, max_tokens=self.max_tokens)
            try:
                result = self.parse_json_response(response)
                sections = result.get("sections") if isinstance(result, dict) else None
                if (not isinstance(sections, list) or len(sections) != len(ids)
                        or any(not isinstance(s, dict) for s in sections)
                        or {s.get("id") for s in sections} != set(ids)):
                    raise ValueError("缩写输出章节ID缺失或重复")
                if any(not isinstance(s.get("content"), str) or not s["content"].strip() for s in sections):
                    raise ValueError("缩写章节正文不能为空")
                if (not isinstance(result.get("executive_summary"), str)
                        or not isinstance(result.get("outlook"), str)
                        or not isinstance(result.get("conclusions"), list)):
                    raise ValueError("缩写输出缺少有效摘要、结论或展望")
                proposal = deepcopy(state)
                proposal["draft_sections"] = {
                    s["id"]: self._sanitize_citation_links(s["content"], known_urls) for s in sections}
                proposal["section_claims"] = {s["id"]: self._read_claims(s) for s in sections}
                proposal["synthesis_claims"] = self._read_claims(result)
                summary, conclusions, outlook = result["executive_summary"], result["conclusions"], result["outlook"]
                proposal["conclusion_evidence_map"] = self._normalize_conclusion_evidence_map(conclusions, proposal)
                proposal["final_report"] = self._assemble_report(proposal, summary, conclusions, outlook)
                check_body_length(proposal)
                proposal["body_length_repair"] = {
                    "before_body_chars": before,
                    "after_body_chars": body_character_count(proposal["final_report"]),
                    "max_body_chars": limit, "attempts": attempt + 1, "status": "succeeded"}
                state.update(proposal)
                return summary, conclusions, outlook
            except (ValueError, TypeError, KeyError) as exc:
                problem = str(exc)
        state["body_length_repair"] = {"before_body_chars": before, "max_body_chars": limit,
                                       "attempts": 2, "status": "failed", "error": problem}
        # 超限草稿不能作为对外报告发出。
        state["final_report"] = ""
        state["claims"] = []
        raise ValueError("报告缩写未满足正文上限：" + problem)

    def _assemble_report(
        self,
        state: ResearchState,
        executive_summary: str,
        conclusions: list,
        outlook: str,
    ) -> str:
        """
        程序化拼接最终报告。

        为什么不让 LLM 输出整篇：正文若经由一次 LLM 调用重新吐出，
        长度就被该次调用的 max_tokens 卡死——实测报告要求写长之后，
        输出恰好顶到 16000 上限，JSON 被截断、解析失败。
        改由代码拼接后，正常情况下无需再输出整篇正文；仍按任务总预算检查，
        仅超限时触发有界缩写。

        章节正文来自 draft_sections（各章节单独写作的产物），
        首尾的摘要/结论/展望来自合成调用的短字段。
        """
        from copy import deepcopy
        state["report_synthesis"] = {"executive_summary": executive_summary,
                                     "conclusions": deepcopy(conclusions), "outlook": outlook}
        parts = []

        if executive_summary:
            parts.append(f"## 执行摘要\n\n{executive_summary.strip()}\n")
            parts.append("---\n")

        # 按大纲顺序拼正文，并补上一级编号
        idx = 0
        drafts = state.get("draft_sections") or {}
        for section in state.get("outline") or []:
            content = (drafts.get(section.get("id")) or "").strip()
            if not content:
                continue
            idx += 1
            title = section.get("title") or f"章节{idx}"
            parts.append(f"## {idx} {title}\n\n{content}\n")

        if conclusions or outlook:
            parts.append("---\n")
            parts.append("## 结论与展望\n")
            if conclusions:
                parts.append("### 核心结论\n")
                for i, c in enumerate(conclusions, 1):
                    text = self._conclusion_text(c)
                    if text:
                        parts.append(f"{i}. {text}")
                parts.append("")
            if outlook:
                parts.append(f"### 未来展望\n\n{str(outlook).strip()}\n")

        refs = self._references_from_content(state, "\n".join(parts))
        state["references"] = refs
        if refs:
            parts.append("---\n")
            parts.append("## 参考文献\n")
            for i, ref in enumerate(refs, 1):
                title = ref.get("title") or ref.get("source") or f"来源 {i}"
                url = ref.get("url") or ref.get("link") or ""
                extra = " - ".join(x for x in [ref.get("author"), ref.get("date")] if x)
                line = f"{i}. [{title}]({url})" if url else f"{i}. {title}"
                parts.append(f"{line}{' - ' + extra if extra else ''}")

        return "\n".join(parts).strip()


    async def _revise_report(self, state: ResearchState) -> ResearchState:
        """按章节修订报告，避免局部输出覆盖整篇。"""
        self.add_message(state, "thought", {
            "agent": self.name,
            "content": "根据审核反馈修订报告..."
        })

        unresolved = open_issues(state)
        drafts = state.get("draft_sections") or {}
        if not unresolved or not drafts:
            # 低分但 issues 为空仍须实际写作；把总评与原正文交给逐章写作入口。
            state.setdefault("revision_original_sections", {}).update(drafts)
            reopen_sections_for_rewrite(state, [section["id"] for section in state["outline"]])
            return await self._write_report(state)

        budget = self._writing_budget(state)
        known_urls = self._known_source_urls(state)
        addressed_ids = set()
        unable_to_address = []
        revised_sections = 0

        for section in state.get("outline") or []:
            section_id = section.get("id")
            pending_issues = section_issues(state, section_id)
            if pending_issues and section.get("status") == "pending" and not drafts.get(section_id):
                # Critic 新增的章节没有旧正文，调用章节写作入口生成后再合稿。
                await self._write_checkpointed_section(state, section)
                revised_sections += 1
                continue
            original_content = (drafts.get(section_id) or "").strip()
            if not section_id or not original_content:
                continue

            if not pending_issues:
                continue
            feedback_text = revision_context(state, section_id)
            related_facts = [
                fact for fact in fact_views(state, verified_only=True)
                if section_id in (fact.get("related_sections") or [])
            ]
            new_info = "\n".join(
                f"- {source.get('title') or '原文来源'} (URL: {source.get('url') or source.get('source_url')})"
                for source in state.get("raw_sources") or []
                if section_id in (source.get("related_sections") or []) and source_available(source)
            ) or "无可用补充证据；只能降低结论强度并说明局限。"

            claim_evidence = await revision_evidence_prompt(
                state, section_id, [str(fact.get("id") or "") for fact in related_facts],
                query=issue_query(pending_issues),
            )
            evidence_payload = json.loads(claim_evidence.removeprefix(BEGIN).removesuffix(END))
            prompt = self.SECTION_REVISION_PROMPT.format(
                query=state["query"],
                section_id=section_id,
                section_title=section.get("title", ""),
                original_content=original_content,
                feedback=feedback_text,
                new_info=new_info,
                claim_evidence=claim_evidence,
                claim_instructions=CLAIM_INSTRUCTIONS,
                min_section_chars=budget["sections"][section_id]["min_chars"],
                max_section_chars=budget["sections"][section_id]["max_chars"],
            ) + "\n" + ISSUE_RESPONSE_INSTRUCTIONS
            response = await self.call_llm(
                system_prompt=(
                    "你是负责修订单个报告章节的资深编辑，"
                    "必须保持内容完整且不得编造证据。"
                ),
                user_prompt=prompt,
                json_mode=True,
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
            result = self.parse_json_response(response)
            revised_content = (result or {}).get("revised_content", "").strip()
            if not revised_content:
                unable_to_address.append(f"{section_id}: 修订输出无效")
                continue

            minimum_length = (budget["sections"][section_id]["min_chars"]
                              if budget["max_body_chars"] is not None else max(self.min_section_chars, int(len(original_content) * 0.55)))
            if len(revised_content) < minimum_length:
                unable_to_address.append(
                    f"{section_id}: 修订结果过短({len(revised_content)}<{minimum_length})"
                )
                continue

            drafts[section_id] = self._sanitize_citation_links(
                revised_content, known_urls
            )
            state.setdefault("section_claims", {})[section_id] = self._read_claims(result or {})
            responses = record_writer_responses(state, section_id, drafts[section_id], result, evidence_payload)
            revised_sections += 1
            addressed_ids.update(row["issue_id"] for row in responses if row["status"] == "addressed")
            unable_to_address.extend((result or {}).get("unable_to_address", []))

        if revised_sections or section_issues(state, "__synthesis__"):
            await self._synthesize_report(state)

        self.add_message(state, "revision_complete", {
            "agent": self.name,
            "changes_count": revised_sections,
            "addressed_issues": sorted(addressed_ids),
            "unable_to_address": unable_to_address,
        })

        # 回到审核阶段
        state["phase"] = ResearchPhase.REVIEWING.value

        return state

    @staticmethod
    def _hypothesis_prompt(template: str, state: ResearchState) -> str:
        """在插入用户材料前移除关闭机制的任务、账本和输出字段。"""
        if hypotheses_enabled(state):
            return template
        template = template.replace("\n### 假设与三类证据账本\n{hypothesis_ledger}\n", "").replace(
            "\n## 假设与可审计证据账本\n{hypothesis_ledger}\n", ""
        ).replace(
            "2. 提炼核心结论，并输出“结论 → 假设 → 支持/反驳/不确定证据”的 ID 映射",
            "2. 提炼核心结论，保留适用条件与不确定性",
        )
        omitted = ("14. **正反证据必须同时消费**", '"hypothesis_ids":',
                   '"supporting_evidence_ids":', '"contradicting_evidence_ids":',
                   '"inconclusive_evidence_ids":')
        return "\n".join(line for line in template.split("\n")
                         if not line.lstrip().startswith(omitted))

    @staticmethod
    def _normalize_url(url: Any) -> str:
        raw = str(url or "").strip()
        if not raw:
            return ""
        try:
            parts = urlsplit(raw)
        except ValueError:
            return ""
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            return ""
        host = (parts.hostname or "").lower()
        if host in {"example.com", "www.example.com", "example.org", "example.net"}:
            return ""
        return urlunsplit((
            parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"),
            parts.query, "",
        ))

    def _known_source_urls(self, state: ResearchState) -> set:
        urls = set()
        for item in (
            fact_views(state, verified_only=True)
            + list(state.get("raw_sources") or [])
            + list(state.get("references") or [])
        ):
            normalized = self._normalize_url(
                item.get("source_url") or item.get("url") or item.get("link")
            )
            if normalized:
                urls.add(normalized)
        return urls

    @staticmethod
    def _draft_content(state: ResearchState) -> str:
        drafts = state.get("draft_sections") or {}
        return "\n\n".join(
            str(drafts.get(section.get("id")) or "")
            for section in state.get("outline") or []
        )

    def _references_from_content(
        self, state: ResearchState, content: str
    ) -> List[Dict[str, Any]]:
        """按实际正文链接的首次出现顺序生成参考文献，不执行语义核验。

        已有目录只用于历史来源元数据回退，未再次引用的条目不会保留。
        优先使用读取器保存的标题、作者和日期，不依赖模型另报的目录。
        """
        metadata = {}
        for item in (
            list(state.get("references") or [])
            + fact_views(state, verified_only=True)
            + list(state.get("raw_sources") or [])
        ):
            url = self._normalize_url(
                item.get("source_url") or item.get("url") or item.get("link")
            )
            if not url:
                continue
            record = metadata.setdefault(url, {})
            title = item.get("title") or item.get("source_name") or item.get("source")
            if title:
                record["title"] = title
            for field in ("author", "date"):
                if item.get(field):
                    record[field] = item[field]

        references, seen = [], set()
        for label, raw_url in MARKDOWN_LINK.findall(str(content or "")):
            url = self._normalize_url(raw_url)
            if not url or url in seen or url not in metadata:
                continue
            info = metadata[url]
            title = info.get("title") or label
            references.append({
                "id": len(references) + 1,
                "title": title,
                "source": title,
                "url": url,
                **{field: info[field] for field in ("author", "date") if field in info},
            })
            seen.add(url)
        return references

    def _append_reference(self, state: ResearchState, reference: Dict[str, Any]) -> bool:
        url = self._normalize_url(reference.get("url") or reference.get("link"))
        if not url or url not in self._known_source_urls(state):
            return False
        if any(
            self._normalize_url(ref.get("url") or ref.get("link")) == url
            for ref in state.get("references") or []
        ):
            return False
        item = dict(reference)
        item["id"] = len(state["references"]) + 1
        item["url"] = url
        state["references"].append(item)
        return True

    def _deduplicated_references(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        seen = set()
        for reference in references:
            url = self._normalize_url(reference.get("url") or reference.get("link"))
            if not url or url in seen:
                continue
            seen.add(url)
            item = dict(reference)
            item["id"] = len(result) + 1
            item["url"] = url
            result.append(item)
        return result

    @staticmethod
    def _sanitize_citation_links(content: str, known_urls: set) -> str:
        pattern = re.compile(r"(?<!!)\[([^\]]+)\]\((https?://[^)]+)\)")

        def replace(match):
            normalized = LeadWriter._normalize_url(match.group(2))
            if normalized and normalized in known_urls:
                return f"[{match.group(1)}]({normalized})"
            return f"{match.group(1)}（来源待核验）"

        return pattern.sub(replace, str(content or ""))

    @staticmethod
    def _target_section_ids(target: Any) -> set:
        return set(re.findall(r"sec_\d+", str(target or ""), flags=re.IGNORECASE))
