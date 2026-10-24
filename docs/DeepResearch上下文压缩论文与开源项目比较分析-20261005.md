# DeepResearch上下文压缩论文与开源项目比较分析

## 日期、对象与检索口径

- 调研及比较日期：2026-10-05，Asia/Singapore。
- 对象：2024—2026年与长链路网页研究、研究轨迹压缩、子任务折叠及可恢复摘要相关的论文和官方开源项目。
- 用户问题：近年是否有讲 Deep Research 上下文压缩的顶会论文或高质量开源项目。
- 核验来源：正式会议论文集、OpenReview官方论文索引、作者arXiv全文、作者/机构GitHub仓库及源码。二手列表只用于发现条目，不作为会议身份或效果的最终依据。
- 本次仅为文献和代码静态比较；没有安装项目、调用模型、复跑基准或修改本项目运行逻辑。开源推荐依据是机制针对性、官方来源和可检查的实现/运行入口，不构成运行可靠性排名。

原始机器可读材料：[文献检索与来源核验-20261005.json](../backend/eval/data/DeepResearch上下文压缩调研-20261005/文献检索与来源核验-20261005.json)。该文件保存候选信息、检索返回记录、访问失败及部分仓库提交SHA。

证据边界：

- **已测量**：本次没有新增性能测量，下面所有数值均为作者实验报告。
- **已核验**：所列正式论文的会议记录/官方PDF索引、官方仓库说明和指定源代码。
- **间接推断**：各方法对本项目的适配优先级、可能的失效方式与建议指标。
- **尚未评估**：在本项目中文研究报告上的遗漏率、引用保真、端到端质量、时延和总成本；完整训练复现程度与线上稳定性。

## 核心结论与逐项比较

直接面向 Deep Research 的代表包括 RE-TRAC、AgentFold、Context Folding 和 ReSum。ACON更偏通用长链路智能体压缩策略优化；ReadAgent适合参考摘要后精确回读。已有五篇正式顶会论文可供选择，ReSum的最终会议身份本次尚未核实。

| 条目 | 会议身份 | 压缩对象和机制 | 主要阅读用途 |
| --- | --- | --- | --- |
| RE-TRAC | ICML 2026正式论文 | 研究轨迹结束后形成证据、结论、来源状态和不确定性，供下一轮探索使用 | 研究摘要怎样保留证据与失败信息 |
| AgentFold | ICLR 2026正式论文 | 最新交互保留完整信息；较早历史形成多尺度摘要，可细粒度压缩或合并子任务 | 避免所有历史反复合并成同一份摘要 |
| Context Folding | ICML 2026正式论文 | branch进入子轨迹，return返回结果并折叠中间过程；FoldGRPO训练该行为 | 按子问题控制主研究上下文 |
| ACON | ICML 2026正式论文 | 同时压缩观察与历史，根据压缩失效优化自然语言指南 | 用任务失败改进压缩规则 |
| ReSum | 2025预印本，2026年v3；主会身份待核实 | 近窗口上限时外部摘要，原问题加摘要重启工作上下文；可用GRPO适应 | 周期性摘要后继续研究 |
| ReadAgent | ICML 2024正式论文 | 分块gist记忆加按需回查原文 | 数字、条件、原句的可恢复保留 |

### 1. RE-TRAC：最贴近研究证据和探索状态

完整题名：**RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents**。[正式论文](https://proceedings.mlr.press/v306/zhu26bd.html)，[官方实现](https://github.com/microsoft/InfoAgent/tree/main/retrac)。

它在研究轮次之间传递结构化状态，使下一轮能够利用已收集证据和失败信息改变搜索方向。本次核验的[配置源码](https://github.com/microsoft/InfoAgent/blob/8a6f17670a84f619505d117d13c6d26a2f3f21cf/retrac/retrac/deep_research.yaml)明确要求事实关联来源与验证状态、推断关联证据、冲突双方都保留；继续研究时允许质疑上一轮摘要。

适配判断：对行业研究中的证据追溯、待核实信息和重复搜索问题最有启发。局限：跨研究轨迹压缩与逐轮Writer历史摘要并不等同；详细摘要也可能增长，仍需预算和回读机制。本次确认推理代码与30B权重链接，未验证完整SFT训练复现。

### 2. AgentFold：选择压缩范围和粒度

会议题名：**AgentFold: Long-Horizon Web Agents with Proactive Context Folding**；[2025初稿](https://arxiv.org/abs/2510.24699)题名使用Context Management。[ICLR正式版PDF](https://openreview.net/pdf?id=IuZoTgsUws)，[官方关联项目](https://github.com/Alibaba-NLP/DeepResearch)。

机制是较早历史的多尺度摘要与最新完整交互共同组成工作上下文；可压缩单步，也可整合一段已完成探索。AgentFold以SFT学习折叠协议。

适配判断：可借鉴“保留当前细节、合并已完成阶段”的组织方式。完整结果不能归因于一段摘要提示词。官方仓库列出论文，但本次没有确认完整AgentFold训练、模型及推理发布范围。正式PDF索引显示ICLR 2026身份；实时打开OpenReview被验证页阻挡。

### 3. Context Folding / FoldAgent：按子任务折叠过程

正式题名：**Scaling Long-Horizon Agent via Context Folding**；[ICML论文](https://proceedings.mlr.press/v306/sun26x.html)，[作者代码](https://github.com/sunnweiwei/FoldAgent)。

agent可进入子问题轨迹，结束后返回摘要，主轨迹不累积该分支的全部中间步骤。[核心源码](https://github.com/sunnweiwei/FoldAgent/blob/58a2d6964ecebe99940529eace50a0558901b8a5/agents/fold_agent.py)可检查branch/return及摘要对照逻辑。

适配判断：适用于按章节、候选对象、局部核验分解研究。跨子任务共享事实和全局限制应在返回结果中保留。README明确开源版本是基于verl的重实现，可能不同于论文训练代码；完整收益包括FoldGRPO，未经训练的协议迁移需要独立验证。

### 4. ACON：把压缩引起的失败用来优化规则

完整题名：**ACON: Optimizing Context Compression for Long-horizon LLM Agents**。[ICML论文](https://proceedings.mlr.press/v306/kang26b.html)，[Microsoft官方代码](https://github.com/microsoft/acon)。

ACON对观察及历史进行压缩，通过失败分析改进自然语言压缩指南，之后可以蒸馏较小的压缩模型。仓库提供AppWorld、OfficeBench及多目标QA的运行和优化说明。

适配判断：适合比较“原历史能完成、压缩历史失败”的样例，查出遗漏后调整指南。主模型不一定需要微调，但优化数据和评估仍有成本。原实验并非中文长报告，峰值token下降也不能直接解释为总费用同比下降。仓库树API本次超时，README和正式论文可访问；未运行框架。

### 5. ReSum：周期性摘要与摘要后的推理适应

完整题名：**ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization**。[论文](https://arxiv.org/abs/2509.13313)，[v3全文](https://arxiv.org/html/2509.13313v3)，[确切官方代码目录](https://github.com/Alibaba-NLP/DeepResearch/tree/f72f75d8c3eb842f2bbbab096a12206ff66e270f/WebAgent/WebResummer)。

ReSum接近上下文上限时用外部模型压缩历史，以原问题和摘要启动后续研究；ReSum-GRPO训练摘要后的继续推理。已核验README、run_resum运行说明及summary_utils.py。

版本注意：v3附录C明确不强制摘要器列出信息缺口及行动计划，避免分散摘要目标或诱发重复核验。因此固定三栏目并非最新版必需协议。目录README仍称ReSumTool-30B待发布；代码开放不等于专门权重和完整训练管线都已确认。本次未取得正式会议录用记录，保留“主会身份待核实”。

### 6. ReadAgent：摘要之外保留回读能力

完整题名：**A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts**。[ICML 2024论文](https://proceedings.mlr.press/v235/lee24c.html)，[作者项目页](https://read-agent.github.io/)。

用短gist代表原文片段，缺少细节时查回原文。对研报的精确数字、限定条件及原句引用有迁移价值。它主要评估长文阅读理解，不能直接推断在线搜索或中文报告质量。建议参考其“摘要负责选择，原文负责恢复细节”的设计。

## 作者报告的核心指标及可比性

| 条目 | 作者报告 | 数据与配置口径 | 本次状态 |
| --- | --- | --- | --- |
| RE-TRAC | BrowseComp相对ReAct提升15—20个百分点 | 官方摘要及仓库，前沿模型的研究框架比较 | 未复现 |
| AgentFold | BrowseComp 36.2%；BrowseComp-ZH 47.3% | 30B-A3B，SFT后的整套agent | 未复现 |
| Context Folding | 活跃上下文最多缩小10倍；项目页BrowseComp-Plus 62.0%、SWE-Bench Verified 58.0% | 项目页分别N=150、N=500；32K活跃上下文和FoldGRPO | 未复现 |
| ACON | 峰值token使用下降26—54% | AppWorld、OfficeBench、Multi-objective QA | 未复现 |
| ReSum | 无需训练时较ReAct平均提升4.5个百分点；GRPO进一步提升8.2个百分点 | 作者网页搜索基准与对应训练设置 | 未复现 |
| ReadAgent | 有效上下文扩展3.5—20倍 | QuALITY、NarrativeQA、QMSum | 未复现 |

数值出处为各条目链接的论文、作者项目页和README。BrowseComp与BrowseComp-Plus、在线搜索与固定语料、不同模型和工具预算均不同，不能据此横向排列优劣。“活跃上下文”与“峰值token”也不是整次任务总token。

## 值得优先读代码的开源项目

1. **Microsoft ACON**：压缩器、指南优化与基准流程，适合构建压缩失败驱动的优化实验。
2. **Microsoft InfoAgent/retrac**：已有LangGraph推理实现；先读deep_research.yaml的状态格式和graph.py的轮次切换。
3. **sunnweiwei/FoldAgent**：包含branch/return及FoldGRPO相关实现，适合研究主动分解与折叠。
4. **Alibaba-NLP/DeepResearch/WebAgent/WebResummer**：本次新核验的ReSum确切目录；适合检查触发、旧摘要与新历史合并及外部摘要调用。
5. **LangChain Open Deep Research**：[仓库](https://github.com/langchain-ai/open_deep_research)、[研究流程源码](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py)、[提示词](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/prompts.py)。子研究完成后经compress_research将发现提交supervisor，另有网页级摘要，并保留raw_notes。研究汇总提示词强调删除无关和重复内容、保留相关事实与来源，适合作为信息清理基线。

Open Deep Research的compress_research并非固定小窗口压缩器，也不是逐轮滚动历史压缩；提示词允许详细输出，不能保证固定压缩比。本次检查main版本，未记录提交SHA。以上推荐是源码阅读与实验参考，未完成运行可靠性评估。

## 失败样例与尚未执行的压力测试

本次无新运行，所以没有新增实测失败样例。以下均为**构造的测试情境/间接推断**，不属于任何论文或本项目新测失败结果：

| 情境 | 应保留的信息 | 压缩可能导致的失败 |
| --- | --- | --- |
| 两个来源都报告增长率，但年份、币种及统计范围不同 | 数值、时期、单位、范围与各自来源 | 数字保留但限定条件消失，最终写成可直接比较 |
| 一轮搜索未能证实某个候选对象 | 已检索方向、反证、尚存不确定性 | 只留下积极结果，下一轮重复失败搜索 |
| 摘要已有暂定判断，随后发现相反证据 | 两方证据、验证状态与修订缘由 | 旧判断被当作已确认事实持续传播 |
| 子章节结束，但其事实被后续章节共同依赖 | 跨章节事实及定位索引 | 折叠了过程，也丢失共享条件 |
| 写作需要精确引用旧来源原句 | 原文位置与可恢复来源 | 摘要是改写文本，难以还原原句 |

这些情境应以原轨迹、压缩状态和原始证据三者对照，检查遗漏与误写。已有引用核验能够检查留下来的内容是否受支持，但不能单独证明重要信息没有被遗漏。

## 局限性、结论和下一步建议

本次不是系统综述，也未穷尽2026年全部相关论文。部分会议实时页面受到访问限制；ReSum录用、AgentFold完整发布及ReSumTool权重状态存在未核实部分。开源代码可检查不代表能够无改动重现论文结果；作者的搜索问答准确率不直接等于长篇研报质量。

建议阅读顺序为 **RE-TRAC → AgentFold → ACON → Context Folding**；研究周期性摘要可加读 **ReSum**，研究原文恢复可加读 **ReadAgent**。若当前目标是改进摘要实现，可先检查RE-TRAC状态提示词、ReSum摘要调用和Open Deep Research信息清理基线。

对本项目的建议均为间接推断：

- 事实保留来源定位、统计口径及验证状态；用户要求与探索判断分别标注出处。
- 压缩已完成子任务的过程，当前阶段和跨阶段依赖保留细节。
- 摘要用于导航，精确原句和关键数字通过原文索引回读。
- 对同一轨迹使用相同模型、任务和预算，比较现有摘要、ReSum式摘要、RE-TRAC式状态及分块折叠。

后续至少测量关键信息保留率、数值与限定条件正确率、来源回读成功率、引用支持率、任务/报告质量，以及包含压缩调用的总token与时延。稳定轨迹回放适合查遗漏；端到端重跑才能评估后续搜索行为改变与最终收益。本次没有实施这些改造或测量。

与同日[智能体历史摘要改进调研](智能体历史摘要改进论文与开源项目调研-20261005.md)相比，本报告新增RE-TRAC、Context Folding、ACON及ReSum确切代码目录核验，聚焦Deep Research。它是新的文献比较记录，保留原文档，未覆盖历史结论。

