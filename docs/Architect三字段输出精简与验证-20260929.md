# Architect三字段输出精简与验证

日期：2026-09-29。对象：Architect首次规划、追问重规划及路由失败后的契约更新输出。本次落实用户要求，模型顶层输出收敛为outline、research_contract、hypotheses。

## 实现结果

- 删除模型requirement_responses字段及对应校验、规范化、补要求章节逻辑。不要求模型把这份清单改名后再次输出到契约。
- 同时删除模型understanding、coverage_check、research_questions、key_entities输出。前端和检查点所需的research_questions、key_entities从最终有效契约派生。
- outline默认只要求id、title、description、search_queries；其他已有执行字段程序默认，确有需要可选填。历史契约要求/指标仍可通过requirement_ids关联。
- research_contract保留核心问题、对象、必需指标/计算及明确的范围、语言、输出形式、字数限制；builder实际保存这些范围字段，并把有效字数限制同步至state.active_constraints。
- hypotheses只要求id、content，程序初始化unverified和三个空证据数组，模型不能把未研究的假设预置为已证实。
- 路由失败后的契约更新数据改为research_contract.followup_contract_update，顶层仍只有三个字段；已提交追问契约继续优先，不被本轮规划重建覆盖。
- Writer从既有契约的explicit_requirements读取历史要求，不再依赖逐章requirement_responses副本；路由更新要求时清除历史副本。保留旧契约explicit_requirements的兼容读取，不新增模型生成要求。
- 只保留JSON/章节/契约/假设的基本结构检查和已有有限重试；结构重试耗尽时不提交无效规划。
- build_research_contract不再从顶层requirement_responses构建要求。新契约明确给空required_entities时，不会再被旧key_entities兜底填回。

修改范围为agents/architect.py、agents/writer.py、quality_pipeline.py、followup_router.py及两份定向测试。旧REVISION_PROMPT动态修订判断入口没有新增调用路径，本次三字段协议覆盖当前首次规划/真实追问重规划所用_initial_planning入口。未修改线上配置、未重启服务、未提交Git。

## 数据口径与原始结果

使用确定性模型桩和内存检查点，未调用线上模型或搜索；没有复跑开发集。原始目录：`backend/eval/data/Architect三字段输出验证-20260929/`。

- `运行验证.py`：可复现unittest入口。
- `首轮测试日志.txt`、`首轮验证结果.json`：第一次64项通过。
- `测试日志.txt`、`验证结果.json`：补充异常字段类型边界后复测，64项通过。
- `规划提示词模板.txt`：新提示词模板；`验证代码哈希.json`：被验证的源码哈希。

运行命令：`backend/.venv/bin/python backend/eval/data/Architect三字段输出验证-20260929/运行验证.py`。

## 核心指标与逐类结果

| 类别 | 测试数 | 结果 |
| --- | ---: | --- |
| 三字段规划执行与兼容 | 8 | 全通过 |
| 追问契约状态更新 | 19 | 全通过 |
| 追问智能路由 | 21 | 全通过 |
| 契约/质量管线/搜索与写作相关回归 | 16 | 全通过 |
| 合计 | 64 | 0失败、0错误、0跳过 |

三字段用例验证最简模型输出一次规划成功、字数和范围契约落盘、核心问题前端事件派生、旧多余字段不能覆盖新契约、普通约束不触发权威预案/覆盖重试、错误对象/章节/假设类型能重试、连续错误不提交、假设不能伪装成已证实、系统来源策略不能被模型覆盖。
追问用例验证已提交约束保留、取消不复活、嵌套契约补丁在降级路径生效及缺少合法补丁时失败关闭。

## 失败样例与局限

测试中构造的失败输入包括research_contract为字符串、outline为数字、hypotheses为null或字符串，以及缺失合法追问补丁；这些是预期拒绝/重试样例，不是本轮线上运行失败。两次测试运行均无断言失败。

已测量：离线状态转换、基础解析检查、字段派生、提示词格式化、契约构建及已有相关测试。
间接推断：减少冗余生成字段可缩短输出并减少字段之间的不一致；尚未测量实际token收益。
尚未评估：真实模型指令遵循、约束提取完整性、输出质量、端到端成功率。本次去掉逐条要求覆盖检查，程序不再保证每条用户要求都有单独的结构化响应或章节关联。通过测试不代表这些语义要求已得到满足。

## 结论与下一步

当前规划模型协议已收敛为三字段，不强制生成逐条要求响应。后续可观察真实轨迹的契约内容、要求遗漏和报告质量，再决定是否需要增加检查。之前摘要重复修复/重复压缩问题不在此次修改范围，不能将其视为已修复。

## 当前提示词全文

```text
你负责将当前研究任务规划为可执行的大纲和研究约束。此阶段只制定计划，不编造检索结果或研究结论。

## 用户本轮原话
{original_query}

## 结合历史补全后的当前研究课题
{query}

## 历史上下文与可继承材料
{historical_context}

## 理解任务与约束
1. 直接通读本轮原话、当前课题和上下文，识别研究目标、对象、时间、地域、比较维度、指标、计算、字数、语言、输出格式及禁止事项。未明确的要求不要擅自补成用户硬约束。
2. 本轮明确修改或取消的要求优先于旧要求；未修改的有效约束继续继承。如果后文提供已提交研究契约，以其为本轮有效约束，保持要求ID，不恢复已取消项。若提供契约更新任务，按该任务输出补丁。
3. 历史对话用于理解背景和指代；模型建议的例子、实体、数据集或假设不等于用户指定对象。存在歧义时说明假设，不虚构用户要求。
4. 所有影响研究结论的事实性内容都必须有可追溯原文及来源。可继承适用的父研究原文快照；对材料缺口、时效性或适用范围不明的内容安排重新检索、交叉验证。历史对话和长期记忆本身不是事实证据。

## 生成计划
1. outline 围绕研究问题组织，每章说明要回答的问题、研究范围和检索方向，合并重复内容。
2. research_contract 集中表达有效研究约束：核心问题、用户指定对象、必需指标和计算，以及明确的时间、地域、语言、输出格式和字数限制。未提出的限制不补造；空数组或空对象表示没有对应要求。其他具体要求可写入相关章节 description 或契约 output_format。
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
    "critical_questions": ["必须回答的问题"],
    "required_entities": [],
    "required_metrics": [],
    "required_calculations": [],
    "active_constraints": {{}}
  }},
  "hypotheses": [{{"id": "h_1", "content": "待验证的假设"}}]
}}

契约中仅在用户提出时增加 time_range、geography、language、output_format 字符串；正文限字数使用 active_constraints.max_body_chars（整数）。
required_metrics 有条目时用 id、name、entity_scope、years、unit、section_id 表达指标及适用范围；required_calculations 有条目时提供 id、type 和具体计算参数，不编造数据值。
已有契约的指标或历史要求需要关联新章节时，可在章节增加 requirement_ids 引用其ID；无需重新输出逐条响应清单。章节确有数据或图表需求时可增加 requires_data、requires_chart；未提供时由程序使用默认值。
通常生成 {min_sections}-{max_sections} 个章节；仅多个独立对象或维度的复杂问题可扩展到 {complex_max_sections} 个。不要为满足章节数量拆出重复内容。
```
