# Architect规划提示词重写与线索移除验证

日期：2026-09-29。对象：当前仓库 ChiefArchitect 规划提示词及要求覆盖处理。为本次代码修改的离线机制验证，不是开发集线上复跑，不改变原运行失败结论。

## 修改内容

- 重写 PLANNING_PROMPT：直接读取本轮原话、补全后的课题、历史上下文与继承材料；保留后接的已提交契约/路由降级契约更新协议。
- 删除正则线索提取函数、提示词占位符及注入、线索字符串覆盖校验、基于线索自动合成要求的兜底逻辑，移除无用 re 导入。
- 模型负责识别完整用户要求，输出 requirement_responses、大纲、章节关联和 research_contract；程序仍使用 build_research_contract 构建状态中的契约，并保留追问已提交契约。
- 保留对已有要求的字段检查与章节关联检查；补章节仅针对模型输出或已提交契约中的要求，不从查询文本新增要求。
- 普通时间、数量、字数、格式要求可使用 not_applicable，不强制编写无权威定义的替代方案。需要核验的框架定义仍检查预案；not_found 仍检查替代体系及理由。要求规范化不会将普通要求默认改为 to_verify。
- 提示词要求区分用户指定实体与探索性对象，模型自拟例子放在检索计划，不能写成用户硬性要求。

修改代码：`backend/app/service/deep_research_v2/agents/architect.py`。
新增测试：`backend/test/test_规划要求直接识别.py`。
本次未改全局对话消解、路由逻辑、BaseAgent历史注入、动态大纲修订入口或新增Critic路径，也未自动提交Git。

## 数据、配置与原始结果

原始目录：`backend/eval/data/规划提示词重写与线索移除验证-20260929/`。

- `运行验证.py`：使用 unittest 执行三个定向测试模块的可复现入口。
- `单元测试日志.txt`、`验证结果.json`：最终运行日志及机器可读结果。
- `规划提示词模板.txt`：新提示词模板快照；双花括号用于 Python format 转义。
- `测试日志.txt`：初始 pytest 启动失败记录（环境未安装 pytest）。
- `首次测试加载失败日志.txt`、`首次测试加载失败结果.json`：验证脚本起初使用错误目录层级导致的3个模块加载错误，已修正脚本路径并完整重跑。

运行方式：在仓库根目录执行 `backend/.venv/bin/python backend/eval/data/规划提示词重写与线索移除验证-20260929/运行验证.py`。使用确定性模型桩和内存检查点，无模型、搜索或NLI调用，无服务重启。

## 核心指标与逐类结果

最终执行46项测试，46项通过，0失败、0错误、0跳过。其中新增6项，已有追问契约19项、路由21项。

| 类别 | 已测量结果 |
| --- | --- |
| 普通要求规划 | 时间、字数、禁止表格、普通对象数量四种子场景均只调用模型桩一次，要求进入契约，无预案误拒 |
| 删除线索兜底 | 查询包含日期、维度、排名而模型未输出要求时，不再由规则合成要求或新增要求章节 |
| 框架定义检查 | 未核验框架缺少预案时仍拒绝；已确认无定义但缺少替代及理由时仍拒绝 |
| 已有要求覆盖 | 漏章节关联仍能检查出来，兜底只为已有要求补关联 |
| 普通要求规范化 | 缺省权威状态归为 not_applicable，不注入官方分类预案 |
| 追问契约和路由 | 40项既有测试通过，包含已提交契约保留、取消要求不复活、路由降级补丁处理、三条路由及检查点机制 |

失败样例：本次两次测试启动问题属于验证环境/脚本入口错误，并非业务断言失败；日志均保留。普通要求无预案、框架缺少必要预案、要求缺少章节分配为本次构造的边界样例。

## 局限、结论和下一步

已测量：上述代码控制流程及状态传递；提示词能正常 format 并进入规划调用。
间接推断：去掉规则提取和对普通要求的预案误拒，可减少这一类不必要规划重试；尚未测量线上节省的token。
尚未评估：新提示词在真实模型上的要求识别完整率、契约质量、规划重试率以及开发集最终成功率。模型若漏掉原话中的要求，现在不会再由日期/数量正则补回；章节覆盖校验只能检查已有要求，不能证明用户意图完整识别。
另外，当前 build_research_contract 对空 required_entities 仍有回退到 key_entities 的兼容逻辑，本次靠提示词约束 key_entities 仅含有效用户实体，未把它改成结构性保障。

本次完成提示词与对应移除逻辑。后续若复跑，应比较要求遗漏、无效重试与规划消耗；之前的摘要失败/重复压缩问题未在本次修改，不能据此宣称已修好全链路。

## 新规划提示词全文

以下是代码中的模板，运行时填充输入后，还会按既有逻辑追加本轮已提交契约或路由降级契约更新任务。

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

## 生成大纲和研究约束
1. 将每项有效用户要求写入 requirement_responses，original_requirement 保留用户原话中的具体要求或已提交契约中的要求，planned_response 说明如何执行。通过 requirement_ids 分配给相关章节，全局字数、语言和格式要求应覆盖所有适用章节。不要只复述要求而没有执行安排。
2. 普通时间范围、对象数量、排名、字数和格式要求直接安排落实，authority_status 使用 not_applicable，fallback_if_unverified、alternative、alternative_rationale 可为空。用户要“比较三家公司”不意味着存在“三家公司”的官方分类。
3. 只有用户指定某个权威框架、官方分类或需核实其定义的体系时，才安排权威定义核验。未核实用 to_verify 并说明无法核实时如何透明披露、选择有来源的替代体系及解释对应关系。有可靠材料才能使用 confirmed 或 not_found；not_found 时填写 alternative 和 alternative_rationale。不得把检索尚未开展写成“已确认不存在”。
4. 大纲围绕研究问题组织，每章写明要回答的问题、适用范围、需要的材料和具体搜索方向。合并重复内容，不机械套用通用行业模板。coverage_check 列出每项要求对应的章节。
5. research_contract 输出关键问题、用户明确指定的实体、必需指标与计算。缺省项用空数组，不能用自行想到的例子填满。key_entities 也只列用户明确指定或有效契约已确认的实体；探索性对象可写入章节检索计划，不能充当用户硬约束。
6. hypotheses 仅列需要验证的假设，status 为 unverified，各证据数组为空。requirement_responses 无有效具体要求时输出空数组。

## 输出协议
仅输出合法 JSON，不要 Markdown 代码围栏或额外解释。保留以下字段，示例中的说明文字应替换为实际内容：
{{
  "understanding": {{"research_goal": "研究目标", "scope": "范围、口径及必要假设"}},
  "requirement_responses": [
    {{
      "id": "req_1",
      "original_requirement": "有效用户要求",
      "requirement_type": "framework/count/ranking/time_range/dimensions/object_list/output_structure/length/language/exclusion/other 中的一个",
      "authority_status": "not_applicable/to_verify/confirmed/not_found 中的一个",
      "planned_response": "具体执行方式",
      "fallback_if_unverified": "仅需核验定义的要求填写，否则为空字符串",
      "alternative": "仅已确认无权威定义时填写，否则为空字符串",
      "alternative_rationale": "替代体系的理由，否则为空字符串"
    }}
  ],
  "outline": [
    {{
      "id": "sec_1",
      "title": "章节标题",
      "description": "本章问题、范围、证据需求及关联要求的落实方式",
      "section_type": "mixed",
      "requires_data": true,
      "requires_chart": false,
      "requirement_ids": ["req_1"],
      "search_queries": ["与本章问题相关的具体检索词"]
    }}
  ],
  "coverage_check": [
    {{"requirement_id": "req_1", "covered_by_sections": ["sec_1"], "coverage_note": "覆盖方式"}}
  ],
  "research_contract": {{
    "critical_questions": ["必须回答的问题"],
    "required_entities": [],
    "required_metrics": [],
    "required_calculations": []
  }},
  "research_questions": ["核心问题"],
  "hypotheses": [
    {{"id": "h_1", "content": "待验证的假设", "status": "unverified", "evidence_for": [], "evidence_against": [], "evidence_inconclusive": []}}
  ],
  "key_entities": []
}}

required_metrics 如有条目，每项包含 id、name、entity_scope（字符串数组）、years（字符串数组）、unit、critical（布尔值）、section_id、minimum_source_tier（通常为 primary）。required_calculations 如有条目，必须包含 type 和明确的计算要求，不得编造待采集的数据值。
通常生成 {min_sections}-{max_sections} 个章节；仅多个独立对象或维度的复杂问题可扩展到 {complex_max_sections} 个。不要为满足章节数量拆出重复内容。
```
