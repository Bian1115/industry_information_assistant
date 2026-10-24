# Architect输出字段使用情况与精简建议

审计日期：2026-09-29。对象：当前工作区新规划提示词及DeepResearch V2后续消费链路。本次是静态代码审计，没有执行开发集、模型、搜索或运行时性能测量，也没有修改业务代码。

## 数据口径与原始结果

原始引用索引：`backend/eval/data/Architect输出字段消费审计-20260929/字段引用索引.json`，包含文件哈希、字段匹配和行号。扫描V2服务、Agent上下文组装和聊天前端，另搜索backend/app与frontend/src排除遗漏。引用数量不等于实际执行次数，整体JSON传递也不能由单字段字符串搜索穷尽。

## 核心结论与逐类分析

| 字段 | 当前用途 | 精简建议 |
| --- | --- | --- |
| 顶层coverage_check | Architect将模型输出原样放入outline事件；自身要求覆盖校验只读requirement_responses与outline.requirement_ids。未发现后续按模型coverage_check执行校验的读取；前端outline处理器不渲染它 | 删除模型生成字段，需要覆盖表时从最终大纲关联程序派生 |
| research_questions | 保存state、发送outline事件、questions_count计数、前端核心问题列表；构建契约时仅在raw.critical_questions为空时兜底；普通追问可继承 | 不再让模型重复生成，由最终契约critical_questions派生现有state和前端事件字段 |
| research_contract.critical_questions | 进入正式契约；pinned_contract开启且绑定历史scope时契约加入Agent输入和摘要输入；Critic直接接收契约序列化文本（当前截断至12000字符） | 保留为核心问题的唯一模型输出来源；这是模型上下文约束，当前没有逐题答案完成度硬门禁 |
| understanding | 放入outline事件；未发现V2下游结构化读取；当前前端outline分支不渲染。其他react_controller路径同名字段不是本规划字段 | 无明确展示需求可删除，必要范围信息应进入有效课题、契约或章节描述 |
| key_entities | 转换为state名称列表、发送事件、追问继承；required_entities为空时构建契约用其兜底 | 与required_entities重复，建议保留契约字段，程序派生兼容状态；迁移时必须去掉旧兜底防止重新提升模型建议实体 |
| requirement_responses | 覆盖校验、构建contract.explicit_requirements、下沉章节；Writer逐章读取执行要求 | 保留。后续可统一放入契约，但需同时改builder/章节派生，不能仅删提示词字段 |
| outline | 章节调度、检索词、写作结构、要求关联和关键章节派生 | 保留 |
| hypotheses | Scout读取假设、整理支持/反驳/不确定证据并更新状态；Writer读取假设与证据映射；delivery输出 | 保留假设驱动功能时不能删除。初始status和空证据桶已有程序默认值，可考虑不再由模型生成固定值 |
| research_contract | 指标要求、计算要求、章节门禁、约束注入、Critic评审和追问更新等实际消费 | 保留；应逐子字段区分程序硬检查和模型软约束 |

关键定位：architect.py:294（实际覆盖校验）、:620（构建契约）、:631（outline事件）；quality_pipeline.py:61（关键问题兜底）、:70（实体兜底）、:183（写作前门禁）；frontend/src/pages/chat/index.tsx:760（大纲和核心问题显示）；contract_updates.py:130（重规划契约保留及派生覆盖表）；agent_context.py:224（契约注入）；critic.py:495（契约评审输入）；writer.py:491（逐章要求）、:645（假设）；scout.py:605（补搜分析接收假设）、:1976（假设证据处理）。实际准确行号以原始引用索引为准。

## 两类coverage_check不能混为一谈

1. 模型顶层输出的coverage_check只进入outline事件，未被build_research_contract接入首轮契约，也不参与_validate_requirement_coverage。
2. retain_committed_contract在追问重规划时根据最终outline.requirement_ids重算contract.coverage_check。这是程序派生的另一份字段，可能随整个契约进入模型输入；不等同于读取或信任模型顶层coverage_check。
3. verification_checks.py中名为coverage_check的调用阶段属于摘要遗漏检查，不是Architect字段消费者。

因此“生成一份覆盖表”不等于“程序完成了覆盖验收”；实际结构校验确认要求至少关联某章节，也不能保证报告语义上完成了要求。

## 失败/风险样例与边界

以下是代码层面可推断的风险，未进行新的模型实测：模型research_questions与contract.critical_questions不同时，前端可展示前者，后续契约使用后者；模型coverage_check即使声称全部覆盖，校验仍只依赖真实章节关联。删除research_questions却不更新事件派生，会让前端核心问题列表变空。required_entities空值触发key_entities兜底，还可能把模型探索对象变成硬约束。

已核查：字段保存、显式读取、整体契约注入、前端展示及别名兜底关系。
间接推断：减少重复模型输出可降低不一致风险并减少输出token；收益大小未测量。
尚未评估：新提示词线上质量、字段删除后的端到端行为、token节省与任务成功率。本次没有运行失败样本、测试或基准，不将静态引用数量当作业务覆盖指标。

## 结论与下一步

上次提示词重写保留了兼容旧链路的字段，尚未完成输出最小化。建议新模型输出保留outline、requirement_responses、research_contract和hypotheses；程序派生coverage_check、research_questions兼容字段和key_entities兼容字段，移除无消费者的understanding。这样仍可保留当前前端与追问接口，避免相同信息由模型生成两份。hypotheses内部固定默认值可进一步交给程序。此处是建议，本次未执行删除。
