# 高质量DeepResearch项目信源评级机制调研及分析

- 调研与评估日期：2026-10-18，Asia/Singapore。
- 评估对象：GPT Researcher、Stanford STORM、Jina node-DeepResearch、LangChain Open Deep Research、Tongyi DeepResearch 的公开来源筛选与评级机制。另核验 RA-RAG 的论文摘要和官方项目说明。
- 数据／配置口径：联网读取官方仓库、GitHub API 与官方论文记录；五个主项目固定提交，保存 README、选定主路径源码与 SHA-256。本轮不安装项目，不运行完整研究，不调用模型。对 STORM 一个独立 URL 准入纯函数进行了五条合成网址探针。
- 原始运行目录：[DeepResearch项目信源评级调研-20261018](../backend/eval/data/DeepResearch项目信源评级调研-20261018/)。入口为[核查索引.json](../backend/eval/data/DeepResearch项目信源评级调研-20261018/核查索引.json)、[比较结果与证据边界.json](../backend/eval/data/DeepResearch项目信源评级调研-20261018/比较结果与证据边界.json)及[斯坦福来源过滤机制探针.json](../backend/eval/data/DeepResearch项目信源评级调研-20261018/斯坦福来源过滤机制探针.json)。每个项目子目录另有仓库元数据、提交信息、文件树、源码快照和含行号片段的“机制核查结果.json”。
- 本次是新的比较分析，不覆盖既有调研；没有修改项目业务代码。以下“值得参考”表示与问题相关的工程或研究参考价值，不是本轮端到端质量排行榜。

## 主要结论

这五个项目中，GPT Researcher 提供了最直接的可选 LLM 来源筛选模块；STORM 提供静态名单和布尔过滤接口；Jina 主要做阅读优先级、域名偏好与回答质量检查；LangChain 和通义的所查主路径没有独立的来源信誉评级模块。

因此，不能因为一个 Deep Research 系统会引用、重排或反思，就把它描述成已经实现“可解释、可校准、基于独立背景证据的信源评级”。上述几种机制解决的问题不同，也没有在本轮被证明能为任意未知来源给出可靠的事实正确概率。

## 核心指标与核查版本

| 项目 | 固定提交 | 本轮重点 |
| --- | --- | --- |
| GPT Researcher | `3582a6841fe7dc801e2a34e34ee408641aea9bce` | 筛选提示、调用入口、默认开关与失败回退 |
| LangChain Open Deep Research | `1b7d2e80db9faa586165c60e09096dbbfd483a64` | 搜索、去重、摘要及研究压缩 |
| Stanford STORM | `fb951af7744dab086e34962e9bc6fe878e145f83` | 来源限制名单、检索器回调、GPT 示例接线 |
| Jina node-DeepResearch | `fd323b521a51264d497bec333bfb997da1bf3210` | URL 排序、正文准入和回答评估 |
| Tongyi DeepResearch | `f72f75d8c3eb842f2bbbab096a12206ff66e270f` | 主推理提示、搜索工具、访问工具与 ReAct 入口 |

五个主项目均取得固定提交及文件树，保存 29 份 README／源码快照。完整 Agent 运行数、模型调用数均为 0；独立纯函数探针为 1 组、5 条网址。来源评级准确率、分数校准、中文覆盖、端到端质量、成本和延迟均尚未评估。

**已测量／直接核验：**固定提交、配置与源码内容、隔离函数对五条合成网址的返回值。**间接推断：**各机制的失效风险、接入价值与优先级。**尚未评估：**真实来源评级质量及完整系统表现。静态核查不能证明模型在运行时完全不做隐式的来源判断。

## 逐项目分析

### 1. GPT Researcher：可选的 LLM 来源筛选，适合学习接入位置

[项目仓库](https://github.com/assafelovic/gpt-researcher)提供 `SourceCurator.curate_sources()`，输入研究问题与已收集材料／研究上下文，调用配置中的 `smart_llm_model`，要求返回筛选后的来源 JSON 列表。源码设置 temperature 为 0.2，默认 `max_results=10`；这只是调用参数与提示要求，不能当作已严格保证十条以内。

[筛选提示](https://github.com/assafelovic/gpt-researcher/blob/3582a6841fe7dc801e2a34e34ee408641aea9bce/gpt_researcher/prompts.py#L319)包含五方面：相关性、可信度、时效性、客观性和定量信息价值。它倾向保留可用材料：优先权威来源，但并不要求一律剔除非权威或有偏向的来源；也允许内容重叠。

关键边界：

- 默认配置 `CURATE_SOURCES=False`，启用后才进入这一步。[默认配置](https://github.com/assafelovic/gpt-researcher/blob/3582a6841fe7dc801e2a34e34ee408641aea9bce/gpt_researcher/config/variables/default.py#L18)
- 当前调用发生在 `conduct_research()` 得到 `research_data` 后；该数据可能已经过压缩或格式化，不能保证评级模型每次都看到完整抓取原文。[调用位置](https://github.com/assafelovic/gpt-researcher/blob/3582a6841fe7dc801e2a34e34ee408641aea9bce/gpt_researcher/skills/researcher.py#L211)
- 输出沿用来源列表格式，没有要求逐源输出标准化的可信度分数、评级依据、背景核验 URL 或未知原因。方法 docstring 提到 reasoning，但实际提示与返回契约不能支持“必然输出逐源理由”的表述。
- 所查 `SourceCurator` 直接调用 LLM，没有另行检索发布者背景或核验发布主体。输出解析异常时返回原输入材料。[筛选实现](https://github.com/assafelovic/gpt-researcher/blob/3582a6841fe7dc801e2a34e34ee408641aea9bce/gpt_researcher/skills/curator.py#L63)

适配判断：值得参考其独立模块和可选接入方式，但若目标是可追溯评级，需要补充身份核验、背景证据、逐维度理由和未知状态。不能只复制提示中的“权威来源优先”，便把筛选结果展示成可信度百分比。

### 2. Stanford STORM：静态来源限制名单与 URL 准入接口

[STORM](https://github.com/stanford-oval/storm)适合参考多视角提问和带引用的长文组织。与本题直接相关的是 `GENERALLY_UNRELIABLE`、`DEPRECATED`、`BLACKLISTED` 三个静态集合，以及 `is_valid_wikipedia_source(url)`：合并名单后，在网址的 `netloc` 中做字符串包含判断，返回通过或拒绝。[名单与函数](https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/knowledge_storm/storm_wiki/modules/retriever.py#L225)

这属于来源准入规则，没有连续信誉分数，也不对当前文章逐篇调查。所查 YouRM、BingSearch 等检索器支持 `is_valid_source` 回调；未传回调时默认 `lambda x: True`。[回调默认行为](https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/knowledge_storm/rm.py#L14) 本轮 GPT 运行示例在构造所列检索器时没有显式传入名单函数，不能看到名单存在就断言默认运行已经使用。[运行示例](https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/examples/storm_examples/run_storm_wiki_gpt.py#L100)

隔离探针还发现名单项有媒体名称、大小写及下划线形式，与真实 hostname 未统一；例如 `InfoWars` 对 `www.infowars.com` 不匹配。另一类问题是 `bestgore.com` 的子串规则也拒绝合成域名 `notbestgore.com`。这些是字符串机制结果，不是对这些网站的本轮信誉评价。

适配判断：可以借鉴“搜索结果进入阅读前有明确来源准入策略”，但需规范域名、做边界匹配、保留名单来源与更新时间，并检查接口是否真的接入。静态英文百科名单不能直接替代中文行业或专业研究评级标准。

### 3. Jina node-DeepResearch：URL 阅读优先级与回答评估

[项目仓库](https://github.com/jina-ai/node-DeepResearch)的 `rankURLs()` 综合 hostname 频率、路径前缀频率、URL 累积出现权重、指定域名加分及已有 Jina rerank 分数，形成 `finalScore`。`boostHostnames` 命中可额外加 2，默认总分截在 0–5；这不是“事实正确概率”。[排序实现](https://github.com/jina-ai/node-DeepResearch/blob/fd323b521a51264d497bec333bfb997da1bf3210/src/utils/url-tools.ts#L250)

主循环还调用 `keepKPerHostname(weightedURLs, 2)` 控制同站点候选数量，帮助分散阅读预算。正文处理对较短内容调用分类器识别拦截提示等不可用页面；判定网页可用与判定发布者可靠仍是两件事。回答评估器检查 definitive、freshness、plurality、completeness 和 strict 等项目，没有因此形成独立发布者信誉档案。[回答评估器](https://github.com/jina-ai/node-DeepResearch/blob/fd323b521a51264d497bec333bfb997da1bf3210/src/tools/evaluator.ts#L633)

实现上的额外边界：`rankURLs()` 启动 rerank 的 `.then()` 后同步继续算排序，没有在函数内等待这次异步结果。静态代码不能保证新返回的 rerank 分数参与当前调用的排序；本轮没有运行时验证这一点。评价其排序机制时应区分设计信号与当前轮实际生效信号。

适配判断：适合借鉴阅读预算、域名偏好和同站点配额。URL 反复出现可能来自转载、重复搜索或同一机构，不能据此认为有多个独立来源证明某个结论。

### 4. LangChain Open Deep Research：研究流程参考，所查主路径没有独立信誉评级

[Tavily 实现](https://github.com/langchain-ai/open_deep_research/blob/1b7d2e80db9faa586165c60e09096dbbfd483a64/src/open_deep_research/utils.py#L44)先搜索，再按 URL 去重，取得网页内容后生成摘要，最后向研究模型提供标题、URL 和材料。研究压缩提示要求保留信息与引用；这些步骤不构成独立的发布者背景核验或来源评级。

结论仅适用于本轮所读主路径、工具及提示，不能排除用户自行接入的 MCP 或其他搜索工具内含信誉数据。第三方搜索服务的排序也不能自动解释为本项目输出的来源可信度。

截至本轮核验，[官方仓库](https://github.com/langchain-ai/open_deep_research)显示已于 2026-08-21 归档，当前只读。它仍可用于学习 supervisor／researcher 组织方式，作为新项目长期依赖需要把归档状态纳入判断。

### 5. Tongyi DeepResearch：主要依靠研究模型和工具交互中的隐式判断

[主系统提示](https://github.com/Alibaba-NLP/DeepResearch/blob/f72f75d8c3eb842f2bbbab096a12206ff66e270f/inference/prompt.py#L1)要求使用可信、多样的来源；[搜索工具](https://github.com/Alibaba-NLP/DeepResearch/blob/f72f75d8c3eb842f2bbbab096a12206ff66e270f/inference/tool_search.py#L81)将搜索结果的标题、链接以及存在时的日期、source 和摘要交给模型；访问工具读取正文并围绕目标提取材料。

在所查 `inference/` 主推理提示、搜索、访问及 ReAct 入口中，没有发现独立的来源评级接口、来源信誉数值输出、外部评级库或专门的发布者背景核验流程。模型可能在推理中比较来源，但本轮没有测量这种隐式判断的能力。

适配判断：值得研究搜索／访问的工具协议和信息搜寻行为；其整体研究能力不能直接证明一个透明评级子系统存在。仓库包含大量其他 WebAgent／Agent 子项目，本轮未逐个审计，不能把本结论推广到整个仓库所有实验系统。

## 相邻方法：要研究“可靠性如何计算”，RA-RAG 更直接

[RA-RAG，EMNLP 2025 主会](https://aclanthology.org/2025.emnlp-main.1738/)先通过跨来源信息核验估计来源可靠性，再选择可靠且相关的来源，并用加权多数投票聚合答案。[官方代码](https://github.com/ml-postech/RA-RAG)明确把来源可靠性用于检索和答案生成。

它更接近“估计来源可靠性并使其影响结果”的目标，但属于多来源 RAG 方法，不能当作已经具备完整开放网页研究流程的即用替代品。本轮只重新核验论文摘要和官方 README，没有重新审计或运行它的代码。交叉共识会如何受到转载、共同上游和系统性偏差影响，仍需在目标数据上验证。

## 失败样例与风险边界

| 类别 | 样例／结果 | 证据性质与影响 |
| --- | --- | --- |
| STORM 名单漏匹配 | `https://www.infowars.com/article` 返回 True；名单包含 `InfoWars` | 已测量的纯函数行为，大小写／名称到域名映射未规范 |
| STORM 名单漏匹配 | `https://arxiv.org/abs/example` 返回 True；名单包含 `arXiv` | 已测量的纯函数行为，不表达本报告认可其名单评级 |
| STORM 子串误匹配 | `https://bestgore.com/example` 和合成 `https://notbestgore.com/example` 均返回 False | 已测量；非域名边界匹配会连带拦截合成不同域名 |
| STORM 未配置接线 | 默认回调放行，所查 GPT 示例未显式传入名单函数 | 静态直接核验，未运行完整示例 |
| GPT 筛选故障回退 | 异常时返回输入材料 | 静态直接核验；未注入运行时异常，不能报告真实故障率 |
| 自称权威的材料 | 网页自称“官方发布”，模型据此偏好 | 机制推断；本轮未测试模型误判 |
| 转载形成虚假共识 | 多 URL 来自一篇上游稿件 | 机制推断；URL 频率／域名分散不证明证据独立 |
| 可靠来源被错误引用 | 高信誉来源的原文不支持具体结论 | 机制推断；需要独立检查引用支持性 |

探针只使用预设网址字符串，没有访问这些网页，也没有针对来源信誉建立人工金标准。因此五条返回值不得换算成来源评级准确率、漏报率或误报率。

抓取过程中遇到 GitHub API／raw 域名 SSL 连接错误和超时，重试后完成所需快照；Jina `url-tools.ts` 最后通过文件树中的 Git blob API 补抓。最初猜测的 `src/tools/search.ts` 在固定文件树不存在，已在索引标注。网络错误与路径猜测不算项目算法失败。

搜索早期读取了分支 URL；发现网页索引中的分支内容与本次固定提交快照存在差异后，主要实现结论统一按固定 SHA 写成。历史搜索摘要不作为当前代码存在或启用状态的证据。

## 对本地项目的建议

以下为设计建议，没有在本轮实现或测量收益。建议将来源评级从事实提炼中独立出来，接在正文保存与元数据可用之后，分别保留：

1. **发布者身份与文档性质**：是谁发布，身份通过什么证据确认，是原始文件还是转载／解读；证据不足允许 unknown。
2. **来源背景与文档质量**：历史记录、作者／机构信息、方法与数据透明度、利益关系，保存支持判断的背景网址和原文。
3. **任务相关性与时效性**：控制阅读与检索顺序，不混写成来源信誉。
4. **原始出处和来源独立性**：归并同一上游，避免多家转载被算成多份独立证明。
5. **论断支持性**：单独判断当前引用是否支持具体句子，保留支持、反驳、证据不足状态。

先输出分项标签、依据及未知原因，再决定是否需要合成排序分数。如果需要展示 0–1 概率，必须明确预测对象并另做校准；将任意加权分数直接显示成“可信度 92%”没有本轮证据支持。

本地历史背景可参见[当前信源评级机制实现核查及分析-20261017.md](当前信源评级机制实现核查及分析-20261017.md)：它记录了旧模型评分未接入当时主链路。本轮没有重新审计本地生产调用链，不把昨日记录表述为今日重新测得的结果。

## 局限性及下一步

本轮是定向代码核查，不是穷尽式项目调查，也不是五套系统的相同任务实测。未全面审查安全性、许可证依赖、线上部署和第三方搜索后台；没有公开足够细节的商业系统不纳入实现对比。原始信誉与文档证据质量依任务而变，通用新闻规则不能直接用于专业文献质量判定。

下一步优先参考 GPT Researcher 的独立模块接入方式，结合可回查的身份／背景证据和 RA-RAG 的可靠性参与检索思想；STORM、Jina 的机制用于准入和阅读预算。随后建立包含原始文件、转载、自称官方、未知来源、过期材料和冲突来源的人工标注集，分别测量身份误认、未知覆盖、独立上游去重、评级依据支持性以及最终引用支持率。只有在相同任务配对运行后，才能回答该模块是否实际提高本地报告质量。
