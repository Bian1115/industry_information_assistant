# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
DeepResearch V2.0 - 状态管理模块

实现全局工作记忆（Global Working Memory），所有Agent共享此状态。
使用 TypedDict 确保类型安全，与 LangGraph 完美兼容。
"""

import uuid
from typing import TypedDict, List, Dict, Any, Optional, Literal, Tuple, NotRequired
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .source_scope import normalize_allowed_domains
from .hypothesis_policy import resolve_hypothesis_enabled

# 图拓扑与状态字段的版本号。恢复时用它判断旧检查点能否直接续跑；
# 改变节点、边或必需字段时必须同时升版本；service显式适配无路由配置的v1恢复。
GRAPH_VERSION = "single-main-v6-atomic-events"


class ResearchPhase(str, Enum):
    """研究阶段状态机"""
    INIT = "init"                    # 初始化
    PLANNING = "planning"            # 规划阶段
    RESEARCHING = "researching"      # 深度探索阶段
    ANALYZING = "analyzing"          # 数据分析阶段
    WRITING = "writing"              # 撰写阶段
    REVIEWING = "reviewing"          # 对抗审核阶段
    RE_RESEARCHING = "re_researching"  # 补充搜索阶段（审核发现缺失信息后）
    REVISING = "revising"            # 修订阶段（仅文字修改）
    COMPLETED = "completed"          # 完成


@dataclass
class Section:
    """报告章节"""
    id: str
    title: str
    description: str
    section_type: Literal["qualitative", "quantitative", "mixed"]  # 定性/定量/混合
    status: Literal["pending", "researching", "drafted", "reviewed", "final"]
    content: str = ""
    sources: List[str] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    requires_data: bool = False
    requires_chart: bool = False


@dataclass
class Fact:
    """结构化事实；假设关联与极性只保存在 HypothesisEvidence 中。"""
    id: str
    content: str
    source_id: str                         # 关联 raw_sources 中唯一原文记录
    credibility_score: float  # 可信度评分 0-1
    extracted_at: datetime
    related_sections: List[str] = field(default_factory=list)  # 关联章节ID
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: Optional[str] = None       # 正文版本编号
    sentence_ids: List[str] = field(default_factory=list)  # 支持句编号，原句按来源索引回读


class HypothesisEvidence(TypedDict):
    """事实—假设关系。一个 fact_id 可对应多条不同 hypothesis_id 的关系。"""
    id: str
    fact_id: str
    hypothesis_id: str
    evidence_type: Literal["supports", "refutes", "inconclusive"]
    summary: str
    evidence_quote: str
    recorded_at: str


@dataclass
class DataPoint:
    """来源数据或程序派生数据；计算通过与输入来源准入是独立维度。"""
    id: str
    name: str
    value: Any
    unit: str
    year: Optional[int]
    source: str
    confidence: float
    source_id: Optional[str] = None
    origin: Literal["observed", "derived"] = "observed"
    input_point_ids: List[str] = field(default_factory=list)
    formula: Optional[str] = None
    calculation_status: Optional[str] = None
    fact_id: Optional[str] = None          # 来源数据关联到具有原句定位的事实


@dataclass
class Chart:
    """图表配置"""
    id: str
    title: str
    chart_type: Literal["line", "bar", "pie", "scatter", "table", "heatmap"]
    data: Dict[str, Any]
    code: str  # 生成图表的Python代码
    image_path: Optional[str] = None
    section_id: Optional[str] = None


@dataclass
class CriticFeedback:
    """评论家反馈"""
    id: str
    target_section: str
    issue_type: Literal["missing_source", "logic_error", "bias", "hallucination", "outdated", "incomplete"]
    severity: Literal["critical", "major", "minor"]
    description: str
    suggestion: str
    resolved: bool = False
    report_quote: str = ""
    requirement: str = ""
    evidence: str = ""
    acceptance_criteria: str = ""
    requires_new_search: bool = False
    search_query: str = ""
    writer_status: str = "pending"     # 处理记录，不是复审结论
    review_status: str = "pending"     # resolved / dismissed 才结束待办


@dataclass
class AgentLog:
    """Agent执行日志"""
    timestamp: datetime
    agent: str
    action: str
    input_summary: str
    output_summary: str
    duration_ms: int
    tokens_used: int = 0


class ResearchState(TypedDict):
    """
    LangGraph 状态定义

    这是整个研究过程的全局状态，所有Agent都在读写这个状态。
    使用 TypedDict 以获得类型提示和 LangGraph 兼容性。
    """
    # 基础信息
    query: str                              # 用户原始问题
    original_query: str                     # 本轮用户原话
    resolved_query: str                     # 结合历史补全后的独立研究问题
    conversation_context: List[Dict[str, Any]]  # 当前会话近期上下文
    memory_context: List[Dict[str, Any]]     # 语义召回的长期记忆
    context_resolution: Dict[str, Any]       # 指代消解与上下文审计信息
    context_policy: Dict[str, Any]           # 创建运行时冻结的压缩配置及哈希；续跑沿用
    parent_run_id: Optional[str]
    parent_checkpoint_id: Optional[str]
    parent_report_hash: str
    session_parent: Dict[str, Any]
    session_planning_context: Dict[str, Any]
    session_inheritance_config: Dict[str, Any]
    session_inheritance_config_hash: str
    revision_instructions: List[str]
    active_constraints: Dict[str, Any]
    writing_budget: Dict[str, Any]         # 当前任务的整篇与分章节篇幅预算
    body_length_repair: Dict[str, Any]     # 合稿超限缩写结果
    body_length_check: Dict[str, Any]      # 正文实测与是否超限，关闭硬检查后仍记录
    inherited_evidence_manifest: Dict[str, Any]
    session_id: str                         # 会话ID
    phase: str                              # 当前阶段
    iteration: int                          # 当前迭代轮次
    max_iterations: int                     # 最大迭代次数
    critic_enabled: bool                    # 每次运行冻结的审核返工开关
    status: str                             # running / completed（执行状态，不代表质量结论）
    completion_reason: str                  # passed / revision_limit / not_completed
    graph_version: str                      # 图拓扑与状态契约版本，用于判断能否续跑

    # 来源范围（后端配置，运行期固定）
    allowed_domains: List[str]              # 本次运行允许检索和读取的网站；空列表表示全网
    search_budget: Dict[str, Any]           # 单篇报告联网搜索预算及消耗明细
    search_backend: Dict[str, Any]          # 首次搜索冻结的后端信息（不含 API Key）

    # 规划输出
    outline: List[Dict[str, Any]]           # 动态大纲 (Section序列化)
    mind_map: Dict[str, Any]                # 知识图谱/思维导图
    key_entities: List[str]                 # 关键实体
    research_questions: List[str]           # 待研究的子问题
    hypotheses: List[Dict[str, Any]]        # 研究假设（假设驱动研究）
    hypothesis_enabled: bool               # 每次运行冻结，默认开启；独立于引用核验
    hypothesis_evidence: List[HypothesisEvidence]  # 事实—假设关系账本，极性的唯一来源
    knowledge_graph: Dict[str, Any]         # 知识图谱 {nodes: [], edges: []}
    research_contract: Dict[str, Any]       # 研究需求契约（实体/指标/年份/来源/验收口径）

    # 知识库
    facts: List[Dict[str, Any]]             # 结构化事实库
    data_points: List[Dict[str, Any]]       # 数据点
    raw_sources: List[Dict[str, Any]]       # 启用会话RAG时仅来源引用；离线兼容入口使用本地来源
    evidence_space: Dict[str, Any]         # 会话临时库定位；正文和证据块保存在库中
    storage_protocol: NotRequired[str]     # 旧检查点缺省；新产物使用独立的依赖协议版本
    storage_manifest: NotRequired[Dict[str, Any]]  # 与节点/章节结果一起持久化的不可变依赖清单
    research_material_context: List[Dict[str, Any]]  # 本轮规划的有界段落/表格视图，仅运行期使用
    evidence_rag_metrics: List[Dict[str, Any]]       # 入库、两层召回与请求预算测量
    retrieved_citation_bindings: List[Dict[str, Any]] # 可回读的引用候选，不代表核验通过
    verified_citation_bindings: List[Dict[str, Any]]  # 实际通过NLI的支持证据定位
    citation_retrieval_status: Dict[str, Any]
    evidence_coverage: Dict[str, Any]       # 章节事实/全文证据覆盖统计
    quality_gates: Dict[str, Any]           # 确定性质量门禁结果
    web_reader_attempts: List[Dict[str, Any]]  # 网页读取尝试（含失败分类）
    web_reader_stats: Dict[str, Any]        # 唯一URL成功率及各失败类别统计

    # 分析输出
    charts: List[Dict[str, Any]]            # 生成的图表（主链路不生成）
    chart_generation: Dict[str, Any]        # 固定为 not_executed 及其原因
    code_executions: List[Dict[str, Any]]   # 代码执行记录
    insights: List[str]                     # 数据洞察

    # 写作输出
    draft_sections: Dict[str, str]          # 章节草稿 {section_id: content}
    section_claims: Dict[str, List[Dict[str, Any]]]  # 各章待核验的论断与原文引用声明
    synthesis_claims: List[Dict[str, Any]]   # 摘要、结论与展望的引用声明
    claims: List[Dict[str, Any]]             # 当前报告汇总，附程序赋值的 section_id
    report_verification_claims: List[Dict[str, Any]]  # 从最终报告独立提取，区别于Writer声明
    report_claim_extraction: Dict[str, Any]   # 独立提取单元覆盖与调用轨迹
    citation_issue_handoffs: List[Dict[str, Any]]  # 旧Critic引用问题交接记录，不代表已解决
    final_report: str                       # 最终报告
    references: List[Dict[str, Any]]        # 参考文献
    conclusion_evidence_map: List[Dict[str, Any]]  # 结论 -> 假设 -> 正反/不确定证据映射

    # 审核反馈
    critic_feedback: List[Dict[str, Any]]   # 评论家反馈
    critic_history: List[List[Dict[str, Any]]]  # 各轮问题账本快照
    writer_revision_records: List[Dict[str, Any]]  # Writer按问题/章节返回的定位与证据，待复审
    revision_original_sections: Dict[str, str]  # 补搜重开前的章节，仅作编辑对象
    issue_search_results: Dict[str, Dict[str, Any]]  # 按问题保存的补搜结果，不代表语义支持
    unresolved_issues: int                  # 未解决问题数
    quality_score: float                    # 质量评分
    quality_score_basis: str                # 报告三维评分口径版本
    quality_dimension_scores: Dict[str, int]  # 整体逻辑、要求覆盖、时效性
    citation_verification: Dict[str, Any]    # 当前报告的可选核验审计
    scout_citation_verification: Dict[str, Any]  # Scout 事实的独立核验审计
    citation_verification_config: Dict[str, Any]  # 每次运行冻结阶段开关与正文失败策略
    citation_verification_settings: Dict[str, Any]  # 开启时冻结 NLI 参数
    report_synthesis: Dict[str, Any]  # 可重组的摘要、结论与展望
    citation_repair: Dict[str, Any]  # 一次改写/删除的前后审计与失败记录
    quality_score_status: str               # not_reviewed/valid/derived_from_dimensions/derived_from_issue_severity/valid_after_retry/fallback_invalid_model_output
    quality_verdict: str                    # not_reviewed/pass/needs_revision/major_issues
    pending_research_tasks: List[Dict[str, Any]]  # 带问题ID/章节/验收条件的定向补证任务

    session_task: Dict[str, Any]          # 已验证会话任务，恢复时固定
    session_task_hash: str
    session_task_metrics: Dict[str, Any]
    execution_plan: str                  # replan/edit/augment；程序编译
    writing_mode: str

    # 运行控制
    runtime_budget: Dict[str, Any]          # 阶段预算、耗时与熔断统计

    # 元数据
    logs: List[Dict[str, Any]]              # 执行日志
    errors: List[str]                       # 错误记录


def citation_verification_status() -> Dict[str, Any]:
    """开关未开启时只如实记录"未执行"。

    这里刻意不返回任何通过率或分数：没有核验就不存在"核验通过"，
    也不能让下游把缺失的核验渲染成满分或绿色通过标记。
    """
    return {
        "status": "not_executed",
        "reason": "引用核验开关关闭",
        "all_claims_checked": False,
        "declared_claims_checked": False,
    }


def create_initial_state(
    query: str,
    session_id: str,
    allowed_domains: List[str],
    research_context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 2,
    hypothesis_enabled: Optional[bool] = None,
    critic_enabled: bool = True,
) -> ResearchState:
    """创建唯一主链路的初始状态

    Args:
        query: 用户查询
        session_id: 会话ID
        allowed_domains: 后端解析并验证过的允许搜索网站；空列表表示全网，运行期固定不变
        research_context: 指代消解与多轮上下文
        max_iterations: 业务返工上限（审核不通过时最多返工几轮）
    """
    if type(critic_enabled) is not bool:
        raise ValueError("critic_enabled 必须是布尔值")
    context = dict(research_context or {})
    original_query = str(context.get("original_query") or query)
    resolved_query = str(context.get("resolved_query") or query)
    # 这里再规范化一次：调用方即使直接传了配置原文，状态里保存的也必须是
    # 已验证的纯域名，恢复时才能安全地按同一范围继续执行。
    domains = normalize_allowed_domains(allowed_domains)
    return ResearchState(
        status="running",
        completion_reason="not_completed",
        graph_version=GRAPH_VERSION,
        allowed_domains=domains,
        query=resolved_query,
        original_query=original_query,
        resolved_query=resolved_query,
        conversation_context=list(context.get("conversation_context") or []),
        memory_context=list(context.get("memory_context") or []),
        context_resolution={
            key: context.get(key)
            for key in (
                "is_follow_up",
                "used_memory_ids",
                "assumptions",
                "unresolved_references",
                "token_counts",
                "working_memory_cache_hit",
                "working_memory_last_message_id",
            )
        },
        session_id=session_id,
        phase=ResearchPhase.INIT.value,
        iteration=0,
        max_iterations=max(0, int(max_iterations)),
        critic_enabled=critic_enabled,
        search_budget={
            "limit": 20,
            "used": 0,
            "remaining": 20,
            "exhausted": False,
            "skipped": 0,
            "queries": [],
        },
        outline=[],
        mind_map={},
        key_entities=[],
        research_questions=[],
        hypotheses=[],  # 假设驱动研究
        hypothesis_enabled=resolve_hypothesis_enabled(hypothesis_enabled),
        hypothesis_evidence=[],
        knowledge_graph={"nodes": [], "edges": []},  # 知识图谱
        research_contract={},
        facts=[],
        data_points=[],
        raw_sources=[],
        evidence_coverage={},
        quality_gates={},
        web_reader_attempts=[],
        web_reader_stats={},
        charts=[],
        # 唯一主链路的分析节点只做数据分析与确定性计算，不跑代码沙箱绘图。
        # 如实标成"未执行"，避免页面把"没画图"显示成"图表生成通过"。
        chart_generation={
            "status": "not_executed",
            "reason": "图表生成不在单主链路的分析节点上",
        },
        code_executions=[],
        insights=[],
        draft_sections={},
        writing_budget={},
        body_length_repair={},
        body_length_check={},
        section_claims={},
        synthesis_claims=[],
        claims=[],
        report_verification_claims=[],
        report_claim_extraction={},
        citation_issue_handoffs=[],
        final_report="",
        references=[],
        conclusion_evidence_map=[],
        critic_feedback=[],
        critic_history=[],
        writer_revision_records=[],
        revision_original_sections={},
        issue_search_results={},
        unresolved_issues=0,
        quality_score=0.0,
        quality_score_basis="not_reviewed",
        quality_dimension_scores={},
        citation_verification=citation_verification_status(),
        scout_citation_verification=citation_verification_status(),
        citation_verification_config={"scout_facts": False, "report_claims": False},
        citation_verification_settings={},
        report_synthesis={},
        citation_repair={},
        quality_score_status="not_reviewed",
        quality_verdict="not_reviewed",
        pending_research_tasks=[],
        runtime_budget={},
        logs=[],
        errors=[],
    )


def reopen_sections_for_rewrite(
    state: "ResearchState",
    section_ids: Optional[List[str]] = None,
) -> List[str]:
    """把章节退回 pending，并清掉对应草稿，让 Writer 结合新证据重写。

    Writer 的 ``_write_report`` 会跳过状态已经是 drafted/final 的章节。补搜
    之后如果不退回状态，新证据只会留在事实库里，正文仍是上一轮的旧草稿。

    Args:
        section_ids: 需要重写的章节 ID；为空表示无法定位具体章节，
            这时退回全部章节，宁可多写一遍也不交付沿用旧草稿的报告。

    Returns:
        实际被退回的章节 ID 列表。
    """
    outline = state.get("outline") or []
    targets = {str(x) for x in (section_ids or []) if str(x)}
    drafts = state.get("draft_sections")
    if not isinstance(drafts, dict):
        drafts = {}
        state["draft_sections"] = drafts

    reopened: List[str] = []
    for section in outline:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "")
        if targets and section_id not in targets:
            continue
        section["status"] = "pending"
        section["content"] = ""
        if state.get("critic_feedback") and drafts.get(section_id):
            state.setdefault("revision_original_sections", {})[section_id] = drafts[section_id]
        drafts.pop(section_id, None)
        (state.get("section_claims") or {}).pop(section_id, None)
        reopened.append(section_id)
    if reopened:
        # 重写期间没有新的完整报告声明；不可继续把旧汇总当作当前结果。
        state["synthesis_claims"] = []
        state["claims"] = []
        state["citation_verification"] = citation_verification_status()
        state["citation_verification"]["reason"] = "章节重写后待重新核验"
        state["citation_repair"] = {"status": "not_executed", "reason": "章节重写后待重新处理"}
    return reopened


def data_point_key(
    dp: Dict[str, Any]
) -> Tuple[str, str, Optional[str], str, str]:
    """
    生成数据点的去重键（canonical key / 规范化键）

    直接比较两个数据点的 dict 是没用的：它们的 id 是随机生成的、confidence 由
    LLM 每次现打、字段数量还可能不一样，所以 a == b 几乎永远是 False。
    这里只挑"决定身份"的五个字段，并把它们捏成统一形状，让写法不同但含义
    相同的数据点得到同一个键。

    - name:  去首尾空格、转小写、去掉所有空格 —— "GDP增长率" == " gdp 增长率 "
    - year:  统一转字符串 —— 2024 == "2024"；None 保持 None（表示"无年份"）
    - value: 能转成数字的按数字比 —— 5000 == 5000.0 == "5000"；
             转不了的（如 "约5000亿"）退回按字符串比。
             LLM 什么脏数据都可能吐，所以这里必须用 try/except 兜住。

    返回元组而不是列表，因为要放进 set 做 O(1) 查找，而 list 不可哈希。
    """
    entity = str(dp.get("entity", "")).strip().lower().replace(" ", "")
    name = str(dp.get("name", "")).strip().lower().replace(" ", "")
    unit = str(dp.get("unit", "")).strip().lower().replace(" ", "")

    year = dp.get("year")
    year = str(year).strip() if year is not None else None

    raw_value = dp.get("value")
    try:
        value = f"{float(raw_value):.6g}"
    except (TypeError, ValueError):
        value = str(raw_value).strip().lower()

    return (entity, name, year, unit, value)


def add_data_points(
    state: "ResearchState",
    new_points: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """
    数据点写入的统一入口：补齐字段 + 重新发号 + 去重 + 追加

    state["data_points"] 有四个生产者，如果各写各的，就会出现三个问题：
    1. 重复：scout 搜索时"搭便车"提取过一批，data_analyst 又拿同一批 facts
       的正文重挖一遍，同一个数字进两次。虚高的数量会骗过
       wizard.py 的 ">= 3" 门槛，还会占满 writer / critic 的切片名额。
    2. id 撞车：data_analyst 的提示词示例写的是 "dp_001"，模型每次都从 001
       开始编。这里一律重新发号，不采信调用方或 LLM 给的 id。
       已有数据点 ID 保持稳定，供派生计算的 input_point_ids 引用。
    3. 形状不一：股票 API 那一路原本缺 year / confidence，下游只能靠 .get()
       兜底，报告里会渲染出 "(N/A)"。这里用 setdefault 统一补齐。

    Args:
        state: 全局研究状态
        new_points: 待写入的数据点列表

    Returns:
        (新增数量, 跳过数量)。跳过包含重复项和结构无效项。
    """
    if not new_points:
        return (0, 0)

    if state.get("data_points") is None:
        state["data_points"] = []

    bucket = state["data_points"]
    existing_keys = {data_point_key(dp) for dp in bucket}

    added = 0
    skipped = 0

    for dp in new_points:
        # LLM 偶尔会在 data_points 里塞字符串而不是对象
        if not isinstance(dp, dict) or not str(dp.get("name", "")).strip():
            skipped += 1
            continue

        key = data_point_key(dp)
        if key in existing_keys:
            skipped += 1
            continue

        # 补齐标准字段（已有值不覆盖）
        dp.setdefault("unit", "")
        dp.setdefault("entity", "")
        dp.setdefault("period", "")
        dp.setdefault("accounting_standard", "")
        dp.setdefault("year", None)
        dp.setdefault("source", "")
        dp.setdefault("confidence", 0.5)
        # id 一律重新发号
        dp["id"] = f"dp_{uuid.uuid4().hex[:8]}"

        bucket.append(dp)
        existing_keys.add(key)
        added += 1

    return (added, skipped)


# ---------------------------------------------------------------------------
# 知识图谱节点
# ---------------------------------------------------------------------------

# 节点 type 的合法取值 —— 必须和前端 knowledge-graph.tsx 的 typeConfig 一字不差。
# 前端靠 typeConfig[node.type] 取颜色和图例文字，取不到就退回默认蓝色 + 直接显示
# 原始英文。scout 的提示词原来写的是 "technology"，而前端只认 "tech"，
# 结果技术类实体和核心概念撞成同一个蓝色，图例上还多出一项 "technology"。
GRAPH_NODE_TYPES = {"core", "tech", "company", "policy", "product", "person"}
GRAPH_NODE_TYPE_DEFAULT = "core"

# LLM 常见的其它写法 -> 合法取值
GRAPH_NODE_TYPE_ALIASES = {
    "technology": "tech",
    "技术": "tech",
    "corporation": "company",
    "organization": "company",
    "org": "company",
    "企业": "company",
    "公司": "company",
    "people": "person",
    "人物": "person",
    "政策": "policy",
    "产品": "product",
    "concept": "core",
    "核心": "core",
}


def normalize_node_type(value: Any) -> str:
    """把 LLM 给的 type 收敛到前端认识的 6 个取值之一"""
    raw = str(value or "").strip().lower()
    if raw in GRAPH_NODE_TYPES:
        return raw
    return GRAPH_NODE_TYPE_ALIASES.get(raw, GRAPH_NODE_TYPE_DEFAULT)


# importance(1-10) -> 圆圈直径(像素) 的映射参数。
# 前端 knowledge-graph.tsx 的兜底公式必须和这里保持一致，否则同一个 importance
# 会因为"这个节点的 size 是谁算的"而渲染成不同大小。
GRAPH_NODE_BASE_SIZE = 20        # 基础直径：保证再不重要的节点也看得见、点得中
GRAPH_NODE_SIZE_STEP = 3         # 每多 1 分重要性，直径多几像素
GRAPH_IMPORTANCE_MIN = 1
GRAPH_IMPORTANCE_MAX = 10
GRAPH_IMPORTANCE_DEFAULT = 5


def normalize_importance(value: Any) -> int:
    """
    把 LLM 给的 importance 收进 1-10

    提示词里写了"importance范围1-10"，但模型不一定听话。没有这道夹逼的话，
    一个 importance=100 会算出 320 像素的圆圈，糊满整个画布。
    非法值（None、"高"、空字符串）一律退回默认值 5。
    """
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return GRAPH_IMPORTANCE_DEFAULT
    return max(GRAPH_IMPORTANCE_MIN, min(GRAPH_IMPORTANCE_MAX, score))


def graph_node_size(importance: Any) -> int:
    """
    把"这个实体有多重要"翻译成"这个圆圈画多大"（像素）

    importance 1 -> 23, 5 -> 35, 10 -> 50
    """
    return GRAPH_NODE_BASE_SIZE + normalize_importance(importance) * GRAPH_NODE_SIZE_STEP


def normalize_graph_nodes(
    nodes: List[Dict[str, Any]],
    known_names: Optional[set] = None,
    known_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    知识图谱节点的归一化 + 去重（不碰 state）

    knowledge_graph["nodes"] 有两个生产者，形状本来对不上：
    - data_analyst 的 LLM 会给 importance，并据此算出 size
    - scout 的实体既没有 importance 也没有 size，到了前端全部按兜底值渲染成
      同样大小，视觉上"谁重要"的信息就丢了

    这里统一补齐 importance / size / type，让两边产出同一形状。

    关于 id：**不做无条件重发**。边(edges)是靠 id / name 关联节点的，
    随意换 id 会把所有边指空。只在 id 缺失或与已有节点撞车时才补一个新的。

    Args:
        nodes: 待归一化的节点列表
        known_names: 已存在的节点名（小写归一化后），用于跨批次去重
        known_ids: 已存在的节点 id，用于检测撞车

    Returns:
        归一化后的节点列表（已剔除重复项和结构非法项）
    """
    seen_names = set(known_names or ())
    seen_ids = set(known_ids or ())
    result: List[Dict[str, Any]] = []

    for node in nodes or []:
        # LLM 偶尔会在 nodes 里塞字符串而不是对象
        if not isinstance(node, dict):
            continue

        name = str(node.get("name", "")).strip()
        if not name or name.lower() in seen_names:
            continue

        importance = normalize_importance(node.get("importance"))
        node["importance"] = importance
        node["size"] = graph_node_size(importance)
        node["type"] = normalize_node_type(node.get("type"))

        node_id = str(node.get("id") or "").strip()
        if not node_id or node_id in seen_ids:
            node_id = f"node_{uuid.uuid4().hex[:8]}"
        node["id"] = node_id

        result.append(node)
        seen_names.add(name.lower())
        seen_ids.add(node_id)

    return result


def add_graph_nodes(
    state: "ResearchState",
    new_nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    往 state 的知识图谱里追加节点（归一化 + 去重 + 追加）

    Args:
        state: 全局研究状态
        new_nodes: 待写入的节点列表

    Returns:
        实际新增的节点列表（重复项和非法项不在其中）。

        这里返回列表而不是像 add_data_points 那样返回计数，是因为调用方通常还需要
        知道"具体哪几个节点是新的"（比如打日志、或者只对新实体做后续处理）。
    """
    graph = state.get("knowledge_graph")
    if not isinstance(graph, dict):
        graph = {"nodes": [], "edges": []}
        state["knowledge_graph"] = graph
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])

    bucket = graph["nodes"]
    known_names = {str(n.get("name", "")).strip().lower() for n in bucket}
    known_ids = {str(n.get("id")) for n in bucket if n.get("id")}

    added = normalize_graph_nodes(new_nodes, known_names, known_ids)
    bucket.extend(added)
    return added


def add_graph_edges(
    state: "ResearchState",
    new_edges: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """
    往 state 的知识图谱里追加边（解析端点 + 去重 + 追加）

    两个生产者写边的方式本来对不上：
    - data_analyst 用节点 id 引用: {"source": "baidu", "target": "ai", "relation": "布局"}
    - scout 用节点名字引用，而且压根没写 target

    而 ECharts 的 graph 系列是用 retrieve(node.id, node.name, index) 给节点建索引的
    （见 echarts/lib/chart/helper/createGraphFromNodeEdge.js）—— 节点带了 id，
    links 里的 source/target 就**必须**是 id。对不上时 addEdge 返回 false，
    这条边被静默丢掉，前端连个报错都没有，只是"画不出来"。

    所以这里统一把两种写法都解析成节点 id：id、旧 id、名字都认，解析不了的丢弃并计数。

    注意：
    1. 本函数只认已经在图里的节点，所以**必须先 add_graph_nodes 再 add_graph_edges**。
    2. 已知的一个边界情况：如果 LLM 给的节点 id 和图里已有节点撞车（同 id 不同名），
       normalize_graph_nodes 会给新节点换号，而这里引用该 id 的边会挂到**原本占着这个
       id 的老节点**上。想彻底避免只能让 LLM 别复用 id，代价不值；
       且这种情况下两端仍是图里真实存在的实体，不会画出断边。

    Args:
        state: 全局研究状态
        new_edges: 待写入的边列表

    Returns:
        (新增数量, 丢弃数量)。丢弃包含三种情况：端点解析不到节点、自环、重复边。
    """
    graph = state.get("knowledge_graph")
    if not isinstance(graph, dict):
        graph = {"nodes": [], "edges": []}
        state["knowledge_graph"] = graph
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])

    # 端点索引：id 和名字(小写) 都能查到 id。
    # 分两趟建，是为了让 id 优先于名字 —— 万一某个节点的名字恰好等于另一个节点的 id，
    # 以 id 为准。
    ref_to_id: Dict[str, str] = {}
    for node in graph["nodes"]:
        if isinstance(node, dict):
            node_id = str(node.get("id") or "").strip()
            if node_id:
                ref_to_id[node_id] = node_id
    for node in graph["nodes"]:
        if isinstance(node, dict):
            node_id = str(node.get("id") or "").strip()
            name = str(node.get("name", "")).strip().lower()
            if node_id and name:
                ref_to_id.setdefault(name, node_id)

    def resolve(ref: Any) -> Optional[str]:
        key = str(ref or "").strip()
        if not key:
            return None
        return ref_to_id.get(key) or ref_to_id.get(key.lower())

    bucket = graph["edges"]
    existing_keys = {
        (e.get("source"), e.get("target"), e.get("relation"))
        for e in bucket
        if isinstance(e, dict)
    }

    added = 0
    dropped = 0

    for edge in new_edges or []:
        if not isinstance(edge, dict):
            dropped += 1
            continue

        source = resolve(edge.get("source"))
        target = resolve(edge.get("target"))
        # 自环(A 指向 A)画出来是个没有信息量的小圈，一并丢掉
        if not source or not target or source == target:
            dropped += 1
            continue

        relation = str(edge.get("relation", "")).strip()
        key = (source, target, relation)
        if key in existing_keys:
            dropped += 1
            continue

        edge["source"] = source
        edge["target"] = target
        edge["relation"] = relation

        bucket.append(edge)
        existing_keys.add(key)
        added += 1

    return (added, dropped)


def section_to_dict(section: Section) -> Dict[str, Any]:
    """Section 序列化"""
    return {
        "id": section.id,
        "title": section.title,
        "description": section.description,
        "section_type": section.section_type,
        "status": section.status,
        "content": section.content,
        "sources": section.sources,
        "subsections": [section_to_dict(s) for s in section.subsections],
        "requires_data": section.requires_data,
        "requires_chart": section.requires_chart
    }


def fact_to_dict(fact: Fact) -> Dict[str, Any]:
    """Fact 序列化"""
    return {
        "id": fact.id,
        "content": fact.content,
        "source_id": fact.source_id,
        "credibility_score": fact.credibility_score,
        "extracted_at": fact.extracted_at.isoformat(),
        "related_sections": fact.related_sections,
        "metadata": fact.metadata
    }
