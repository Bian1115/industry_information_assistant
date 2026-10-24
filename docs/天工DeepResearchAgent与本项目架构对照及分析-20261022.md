# 天工 DeepResearchAgent 与本项目架构对照及分析

## 评估日期、对象与口径

- 日期：2026-10-22（Asia/Singapore）。
- 对象：SkyworkAI/DeepResearchAgent 与本仓库当前 deep_research_v2 主流程。
- 方法：阅读官方仓库、历史 README、当前源码和本地工作树，进行静态架构对照。没有安装运行外部项目，没有调用模型，没有执行基准或对照实验。
- 远程 main 固定提交：`5e3c95d14266f8c4aa6a5deae1fe165c7cd1b87b`（提交日期 2026-05-04）；历史 README 固定提交：`4acd4c704ea109743f9c6390e2116923f4055693`（2025-09-29）。
- 本地工作树存在大量既有修改，本次以读取到的工作树代码为准；原始记录包含文件内容和 SHA-256，不能只用本地 HEAD 复现。
- 原始证据：[源码核查证据.json](../backend/eval/data/天工深度研究项目架构对照-20261022/源码核查证据.json)、[说明变更记录.json](../backend/eval/data/天工深度研究项目架构对照-20261022/说明变更记录.json)。

## 结论

两者属于相近的多智能体研究系统：都使用规划、检索、分析与结果综合等职责分工。但本项目当前以明确的 LangGraph 阶段、独立审核和研究交付为中心；对方更强调通用任务的动态规划、能力组合以及后续扩展的自进化运行框架。不能据此宣称报告质量、成本、速度或可靠性哪一方更强。

## 版本与项目介绍

历史 README 将项目描述为两层多智能体系统：上层 Planning Agent 理解需求、拆分任务、协调执行；下层包括 Deep Researcher、Deep Analyzer、Browser Use、MCP Manager 和 General Tool Calling Agent。研究只是其通用任务解决能力的一部分。

当前 README 已改为自进化协议与运行时，区分 RSPL（资源注册、状态、生命周期、版本）与 SEPL（提出、评估、提交改进，追踪与回滚）。模块包含 agent、tool、environment、memory、optimizer、tracer、version。这里的“自进化”是项目机制定位，不等于本次验证了持续提升，也不能仅凭 GRPO 文件名认定运行了模型权重训练。

当前源码进一步区分“规划”和“调度”：PlanningAgent 生成 PlanDecision 并记录计划文件；AgentBus 负责循环、派发、并发收集结果和再次调用规划器。DeepResearcher 和 DeepAnalyzer 在当前实现中是 workflow tools，不能把历史角色名原封不动地当成当前独立 Agent 清单。

## 逐类分析

| 维度 | 对方项目的源码/文档证据 | 本项目当前证据 | 判断 |
| --- | --- | --- | --- |
| 规划 | PlanningAgent 输出子任务、计划更新、是否完成；AgentBus 再调度 | ChiefArchitect 规划，图支持 replan/edit/augment 入口 | 都有规划；对方强调逐轮动态派发，本项目阶段路径更明确 |
| 检索 | DeepResearcher 多轮生成查询、搜索、合并摘要、判断完整性 | DeepScout 研究节点及补搜节点 | 属于相近的迭代研究机制，不能仅按角色名称判断质量 |
| 分析 | DeepAnalyzer 处理附带文件与多模态输入、分步分析 | DataAnalyst 与确定性计算节点 | 有职责重合，但输入范围和执行方法不同 |
| 写作 | Researcher 调用 Report.complete 汇总研究记录，生成报告与摘要 | LeadWriter 有独立写作和修订节点 | 对方也产出报告，但当前研究工作流不等于固定的独立 Writer 阶段 |
| 质量闭环 | Researcher 判断答案是否完整；规划器分析上一轮结果 | CriticMaster 独立审核，反馈驱动补搜或改写 | 本项目独立审核的职责边界更显式；不能将完整性自评等同于独立事实核验 |
| 证据核验 | 本次未对其做完整的论断—来源核验能力审计 | 有独立 citation_verification 及报告论断处理模块，支持审计/删除/一次改写等配置 | 本项目有相关机制；未测量实际正确率，也不能断言对方完全没有 |
| 状态与记忆 | 当前框架包含会话记忆、优化记忆、轨迹和版本模块 | 图要求 checkpointer，另有 Agent 历史、证据准备/提交、运行保护 | 双方都重视状态；尚未实测恢复与隔离能力 |
| 执行组织 | AgentBus 内使用 asyncio.gather 并发收集一轮派发 | 主图核心阶段串行连接，审核后条件路由 | 顶层调度有差异；不能据此说本项目内部完全没有并发 |

关键源码：

- 对方：`src/agent/planning_agent.py`、`src/interaction/bus.py`、`src/tool/workflow_tools/deep_researcher.py`、`src/tool/workflow_tools/deep_analyzer.py`、`src/tool/workflow_tools/reporter.py`。
- 本地：[主图](../backend/app/service/deep_research_v2/graph.py)、[Critic](../backend/app/service/deep_research_v2/agents/critic.py)、[引用核验](../backend/app/service/deep_research_v2/citation_verification.py)。

配置边界：当前 `configs/bus.py` 默认注册 planning 与 tool_calling；其 tool_names 未列入 deep_researcher/deep_analyzer，虽然导入并更新了对应配置。因此不能承诺默认运行 bus 示例就自动得到完整研究团队。`configs/planning_agent.py` 列有研究与分析工具，但浏览器配置被注释；这些示例的端到端可运行性尚未验证。

## 核心指标与证据边界

- **已测量：**没有运行性能指标；本次研究任务运行数为 0。代码结构与版本信息属于静态核查事实。
- **间接推断：**架构上相似，且调度、审核职责存在上述差异；这些结论来自文档和代码，不能替代执行轨迹。
- **尚未评估：**同题报告质量、引用正确率、覆盖率、耗时、token 成本、并发扩展、恢复成功率、医疗研究可靠性及自进化收益。
- **外部作者报告，未经本次复现：**历史 README 写 GAIA Test 83.39、Validation 82.4；关联论文 arXiv:2506.12508v6（2026-05-28）摘要报告 GAIA Test 89.04%。版本、模型与评测口径不同，不合并比较，也不把论文成绩归到当前下载代码的一次运行。
- GAIA 主要用于通用助手任务；其分数不能直接证明长篇研究报告的覆盖、论证与引用质量。

## 失败样例与局限性

没有模型运行，因此没有生成失败样例，也没有实际错误率。资料核查中，官方项目主页由 web 工具访问返回 404；另一次源码路径 `src/bus/bus.py` 返回 404，依据真实文件树更正为 `src/interaction/bus.py` 后成功读取。这些是访问异常，不是研究任务失败。

核查只覆盖主要模块，未逐文件审计全仓。历史 README、当前代码与论文最新版本并非同一时间快照。本项目 README 存在历史说明，本次架构判断以当前 graph.py 为主要依据，不据旧文档断言默认入口行为。

## 下一步建议

1. 优先借鉴 PlanningAgent 的结构化派发结果、AgentBus 的调度职责分离，以及 Researcher 的每轮完整性判断。
2. 将“本轮已找到什么、仍缺什么、为什么需要继续搜索”明确写入研究状态；是否提升结果需要消融实验。
3. 保留本项目已经明确的独立 Critic、引用核验和检查点边界；没有证据支持整体迁移框架。
4. 如果后续做性能对照，先固定对方可运行提交、实际启用工具、模型和搜索供应商；使用相同问题、预算与评价口径，分别评估内容质量、引用支持性、成本及耗时。本次不执行该实验。

## 官方来源

- [当前仓库](https://github.com/SkyworkAI/DeepResearchAgent)
- [固定版本 README](https://github.com/SkyworkAI/DeepResearchAgent/blob/5e3c95d14266f8c4aa6a5deae1fe165c7cd1b87b/README.md)
- [历史架构说明](https://github.com/SkyworkAI/DeepResearchAgent/blob/4acd4c704ea109743f9c6390e2116923f4055693/README.md)
- [当前规划器](https://github.com/SkyworkAI/DeepResearchAgent/blob/5e3c95d14266f8c4aa6a5deae1fe165c7cd1b87b/src/agent/planning_agent.py)
- [当前总线调度](https://github.com/SkyworkAI/DeepResearchAgent/blob/5e3c95d14266f8c4aa6a5deae1fe165c7cd1b87b/src/interaction/bus.py)
- [当前研究工具](https://github.com/SkyworkAI/DeepResearchAgent/blob/5e3c95d14266f8c4aa6a5deae1fe165c7cd1b87b/src/tool/workflow_tools/deep_researcher.py)
- [关联论文 v6](https://arxiv.org/abs/2506.12508v6)
