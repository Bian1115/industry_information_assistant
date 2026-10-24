# LangChain会话摘要提示词核查与项目适配建议

- 核查日期：2026-10-05（Asia/Singapore）。
- 对象：LangChain 经典会话摘要提示词和现代摘要中间件。
- 数据与配置口径：查阅官方源码与 API 文档；官方 master 分支可能后续变化，本记录以核查日内容为准。没有运行摘要生成、测试或多轮实验，也没有修改业务代码。
- 原始记录：[官方提示词核查记录.json](../backend/eval/data/LangChain会话摘要提示词核查-20261005/官方提示词核查记录.json)。

## 用户可能记得的经典提示词

经典 `ConversationSummaryMemory` 和 `ConversationSummaryBufferMemory` 使用 `SUMMARY_PROMPT`，有两个输入变量：`summary` 和 `new_lines`。其任务是把新增对话融合进已有摘要，返回更新后的全文摘要。提示词还附一个单示例：旧摘要已有助手的看法，新增对话解释原因，新摘要把原因融合进原来的内容。下面是省略该示例的中文意译，不是英文完整原文：

```text
综合之前的摘要和本次新增对话，生成更新后的摘要。

之前的摘要：
{summary}

本次新增对话：
{new_lines}

更新后的摘要：
```

来源：[经典提示词源码](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/memory/prompt.py)。

`ConversationSummaryMemory.save_context` 默认把最近两条消息与已有摘要交给模型，再用模型结果替换旧摘要；`ConversationSummaryBufferMemory.prune` 则在原始缓冲区 token 超阈值时移出旧消息，把被移出的消息与旧摘要一起生成新摘要。保留多少原文由代码决定，提示词本身不承担窗口选择。

来源：[摘要调用源码](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/memory/summary.py)、[摘要缓冲区源码](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/memory/summary_buffer.py)。注意后者实际检查的是 raw buffer 的 token，不能将其描述为摘要与原文的严格联合 token 上限。

## 当前中间件的提示词

当前 `SummarizationMiddleware.DEFAULT_SUMMARY_PROMPT` 更强调压缩后的信息必须足以继续任务，要求输出四个栏目：

| 栏目 | 保存的信息 |
|---|---|
| SESSION INTENT | 用户的主要目标与本次会话目的 |
| SUMMARY | 重要上下文、选择、结论、策略与决策原因 |
| ARTIFACTS | 创建、修改或访问过的成果、文件和资源 |
| NEXT STEPS | 未完成任务和下一步 |

没有相关信息的栏目要显式说明为空；只输出整理后的上下文。输入模板使用 `{messages}`。它与经典的双变量模板不同，不能把两版的变量接口混用。

来源：[现代中间件源码](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py)。

现代 API 通过 `trigger` 指定触发条件，通过 `keep` 指定保留消息数、token 或模型窗口比例；默认保留近期 20 条消息。`keep=("messages", 5)` 的单位是消息，不等同于 5 个完整用户问答轮；含工具调用时边界还需要保持调用与结果配对。

来源：[SummarizationMiddleware API](https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware)。经典 memory 类已经标注弃用，新接入可参考中间件机制，无需仅为借鉴提示词迁移整个框架。

同日补充本地依赖核查：`backend/requirements-deepagents锁定.txt:30` 锁定 `langchain==1.4.2`，本地 uv 缓存的这一版本使用上述四个栏目。该版本默认待摘要输入还会裁到约 4000 token，并优先保留输入尾部；因此旧摘要作为历史消息参与下一次压缩，并不保证每次完整进入摘要模型。此补充仍为源码核查，没有新增运行评估。

## 与本项目的关系及建议

当前项目是“旧有效条目＋新增消息 → 抽取变化 → 程序合并”；经典 LangChain 则是“旧摘要＋新增消息 → 完整新摘要 → 替换旧摘要”。这只是机制比较，不是效果优劣评估。

建议借鉴它的递归更新方式，同时保留本项目已有的用户权威要求、来源校验和修正关系。可以使用以下原创适配草案，尚未写入业务代码：

```text
你负责维护能够支持后续追问的会话摘要。
结合旧摘要和本次需要压缩的较早消息，输出完整的更新摘要。

保留仍有效的用户目标、研究对象、时间与地域范围、字数和格式要求、排除项。
完整保留用户明确的修正；仅在明确替代时让旧要求失效。
区分用户要求、助手历史结论和未验证陈述，不把历史结论视为本轮事实证据。
区分不同主题及对应的目标报告，记录成果引用和仍待确认的问题。
合并重复信息，删除已失效要求及冗余过程；不得引入来源未支持的信息。
已有仍有效内容不能因本次消息未提及就被遗漏。
只输出更新摘要，并保留必要的来源消息标识。

旧摘要：
{summary}

本次需要压缩的较早消息：
{new_lines}

更新摘要：
```

窗口外哪些消息参加压缩、窗口内原文保留多少、触发阈值、覆盖游标、摘要预算和压缩率记录，都必须由程序实现。当前应沿用用户已决定的会话层与研究内部历史分离，不把最初内部“5个对话”的口径直接改成会话层“5个用户问答轮”。

## 指标、失败样例与局限性

已核对：经典模板有两个输入变量；现代模板有四个输出栏目；保留策略由程序参数控制。本次模型调用和测试执行均为 0，未得到压缩率、信息保留率、追问准确率、时延或成本测量。

没有本次实测失败样例。源码分析推断的风险包括：只用简短自然语言摘要可能漏掉旧约束；将 messages 数量等同完整对话轮会截错窗口；只换提示词无法保证异步摘要覆盖原文裁剪范围。这些风险尚需真实多轮评估。

下一步若实施，应先定义会话原文窗口、按主题组织摘要，再将旧摘要与新移出的消息重新摘要，并分别记录原文/旧摘要/新摘要 token、生成调用用量、时延、失败与重试、来源拒绝数和信息保留指标。本次不据源码核查宣称 LangChain 提示词比现有提示词效果更好。
