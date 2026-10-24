# Claude Code SSE 与检查点恢复机制核查及分析

核查日期：2026-10-22（Asia/Singapore）。

## 对象、口径与证据边界

本次回答“Claude Code 源码里的 SSE 和检查点恢复如何实现”。当前项目及相邻项目目录未定位到用户指定的 Claude Code 核心源码版本，因此以 Anthropic 官方公开 SDK 源码和官方产品文档为依据。未把 SDK 包装层当成 CLI 内部全部实现，未审阅第三方还原源码。

- **已测量**：没有运行性能、断网、杀进程或文件恢复实验；没有实测成功率、延迟、丢失率。
- **静态确认**：SSE 解码、消息增量聚合、SDK 子进程 JSON 通道、resume 参数传递、rewind 控制请求。
- **文档确认**：会话持久化位置、文件编辑前备份、按用户消息 UUID 恢复、会话恢复与文件恢复的职责。
- **间接推断**：这些机制需要在不同层处理展示、上下文和文件副作用，不能将一种恢复的保证迁移给另一种。
- **尚未评估**：CLI 内部落盘原子性、文件备份实际数据结构、崩溃窗口、工具幂等、UI 断线回放、核心重试策略及各版本差异。

## 原始材料与版本

原始目录：[Claude代码流式与恢复机制核查-20261022](../backend/eval/data/Claude代码流式与恢复机制核查-20261022/)。

[来源清单.json](../backend/eval/data/Claude代码流式与恢复机制核查-20261022/来源清单.json)保存固定 commit、原始 URL 和 SHA-256。

| 来源 | 固定版本 | 本地材料 |
| --- | --- | --- |
| anthropics/anthropic-sdk-typescript | `5d489aae1470959806e7c555d2396e36075cbf35` | `SSE解析源码.ts`、`消息流聚合源码.ts` |
| anthropics/claude-agent-sdk-python | `dd27b703647a4f8989ab656f00c85f98090a60fd` | `子进程传输源码.py`、`控制协议源码.py`、`客户端源码.py` |

官方文档是核查时在线页面，未固定发布版本，可能与 SDK main 的接口细节存在差异。下文 SDK 调用链以归档源码为准，产品行为以文档为准。

## SSE：从网络字节到内容增量

公开 TypeScript API SDK 的调用链：

```text
HTTP Response.body
  → ReadableStreamToAsyncIterable
  → LineDecoder：网络块拆成行
  → SSEDecoder：event/data 行组装成事件
  → Stream.fromSSEResponse：JSON 解析和异常处理
  → MessageStream：更新消息内容并发出增量回调
```

具体代码位置：

1. `SSE解析源码.ts:223` 的 `_iterSSEMessages()`读取 body。网络 chunk 不等于完整 SSE 事件，所以先做行解码，再做事件解码。
2. `SSE解析源码.ts:257` 的 `SSEDecoder`维护 event、data 数组和原始行。遇到空行才交付事件；多条 data 用换行拼接；冒号开头的注释行忽略。
3. `SSE解析源码.ts:52` 的 `Stream.fromSSEResponse()`忽略 ping，对 error 构造 APIError，其余有数据的命名事件 JSON.parse 后 yield。提前退出迭代会中止请求。
4. `消息流聚合源码.ts:677` 起按 content block 的 index 更新快照。text_delta 拼接文本；input_json_delta 累积 partial_json，再交给工具输入解析逻辑。这是接收端内存消息快照，不是持久化任务检查点。

事件通常经过 message_start、content_block_start、多次 content_block_delta、content_block_stop、message_delta、message_stop。工具参数可能跨多个 JSON 字符串片段，不能把单个片段当成已经可以执行的完整工具参数。

所查 SSEDecoder 只处理 event 和 data 字段，没有实现 id/retry 游标及 Last-Event-ID 续传。此结论限定在这段解析代码，不能据此断言 CLI 其他模块绝无任何恢复机制。

官方 [Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming) 的错误恢复说明使用“保存已收到内容并发起新请求”的方式；具体续写格式随模型版本变化。这与恢复旧 HTTP 流、从固定 token 位置重放不是同一个保证。

## Agent SDK：CLI 输出采用 NDJSON

`子进程传输源码.py:574` 创建 CLI 命令并设置 `--output-format stream-json`，第 793 行设置 `--input-format stream-json`。第 695 行按选项添加 `--include-partial-messages`。

第 1101 行 `_read_messages_impl()`从 stdout 读取块，经 `_LineFramer`切成行，解析每行 JSON 并 yield。它还限制单条消息缓冲区大小；进程结束时若尾部 JSON 已截断，会丢弃无法解析的尾部。这说明流式输出本身不构成可靠持久化日志。

官方 [流式输出说明](https://code.claude.com/docs/en/agent-sdk/streaming-output)规定：开启 include_partial_messages 后，API 增量事件包装为 StreamEvent，与 AssistantMessage、ResultMessage 一起返回。应用要区分增量展示和完整消息，不能把两者都直接追加到同一正文，否则可能重复显示。

## 会话 resume：加载持久化对话再继续执行

`子进程传输源码.py:639` 附近将 continue_conversation 转成 `--continue`；第 649 行将 resume 转成 `--resume=<session-id>`。SDK 在这里传递参数，并未自行恢复 CLI 的执行栈。

官方 [会话文档](https://code.claude.com/docs/en/agent-sdk/sessions)说明对话历史写入 `~/.claude/projects/<encoded-cwd>/*.jsonl`，可受配置目录设置影响。对话包括用户消息、工具调用、工具结果和模型回复；resume 加载对应会话继续工作。

因此可确认的是恢复已持久化的对话上下文，不能据此声称恢复进程内存、模型 KV cache、正在生成的某个 token 或任意工具执行到一半的状态。CLI 如何修复不完整工具记录、如何选择重试边界，本次未获得核心源码证据。

## 文件 checkpoint/rewind：编辑前备份与按消息恢复

官方 [文件检查点文档](https://code.claude.com/docs/en/agent-sdk/file-checkpointing)说明：开启后，在受支持文件编辑工具修改文件前保存备份；用户消息 UUID 用作恢复锚点；恢复时还原已修改文件，并删除恢复点之后由其创建且受跟踪的文件。Bash 直接改文件不在此机制覆盖范围内。`rewind_files()`本身只恢复文件，不回退会话上下文。

公开 Python SDK 的调用链：

```text
ClaudeAgentOptions(enable_file_checkpointing=True)
  → 设置子进程环境变量 CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING=true

extra_args={"replay-user-messages": None}
  → 在响应中取得 UserMessage.uuid

ClaudeSDKClient.rewind_files(user_message_id)
  → Query.rewind_files(user_message_id)
  → _send_control_request(...)
  → stdin 写入一行 JSON
  → CLI 内部执行恢复
  → SDK 按 request_id 等待对应响应
```

对应代码为 `子进程传输源码.py:864`、`客户端源码.py:364`、`控制协议源码.py:814`、`控制协议源码.py:706`。控制消息的关键结构为：

```json
{
  "type": "control_request",
  "request_id": "示意请求编号",
  "request": {
    "subtype": "rewind_files",
    "user_message_id": "目标用户消息UUID"
  }
}
```

SDK 为 request_id 建立等待事件，收到对应结果后唤醒，错误和超时走异常路径。这里确认的是控制协议；备份索引结构、复制算法、恢复事务及跨文件原子性没有在这些 SDK 文件中实现，不能补写成已确认源码事实。

交互式 CLI 的 [/rewind](https://code.claude.com/docs/en/checkpointing)另有恢复代码、恢复对话、同时恢复二者的选项。因此“CLI 可以恢复对话”和“SDK rewind_files 仅恢复文件”并不矛盾，它们指不同操作。

## 核心指标与失败场景

本次无实验样本，恢复成功率、恢复耗时、丢失事件数、重复事件数及重复工具执行数均为**未测量**，不记为零。

以下是静态机制推导的失败场景，非本次实测失败样本：

| 场景 | 可以确认或推导的边界 |
| --- | --- |
| 收到半个工具参数 JSON 后断线 | 参数尚不完整；SSE 解码器不提供按事件 ID 继续取余下字节的实现 |
| CLI 在输出 JSON 一半时退出 | 所查传输层会丢弃无法解析的尾部记录，不能用 stdout 当可靠存档 |
| 工具外部操作完成，但结果尚未可靠写入历史就崩溃 | 可能出现外部状态与历史不同步；没有证据证明 exactly-once，实际 CLI 处理待验证 |
| Bash 修改文件后执行 rewind | 官方文档明确该修改不属于文件编辑工具检查点覆盖范围 |
| 备份已清理、缺失或不可读取 | 文件恢复能力受备份可用性约束；文档说明存在恢复失败或跳过路径的情况 |

## 结论与下一步

SSE 解决增量传输，session resume 解决持久化对话的继续使用，文件 checkpoint/rewind 解决受跟踪文件修改的回退。三者不是一套统一的断线恢复协议。本次也没有证据支持将它们描述为通用工作流节点检查点、事件游标补发及工具副作用恰好一次执行。

下一步若要回答某份 Claude Code 核心源码的精确调用链，需要用户给出源码目录、链接或版本，再追踪模型请求重试、会话写盘与读取、文件历史管理及 rewind handler。若要验证可靠性，需单独设计断网、工具完成后杀进程、部分 JSON 输出以及文件恢复失败实验，并按新的评估运行归档。


## 2026-10-22 补充复核：用户指定的官方仓库

复核对象为 https://github.com/anthropics/claude-code 。变更口径：用户已明确仓库，追加归属、目录和许可证检查，不覆盖前文 SDK 来源及证据边界。

固定提交为 `d014fc0fa64280540728772fc84063918ed185cc`。GitHub 递归目录接口未截断，共返回 2188 个条目（包含目录，不能称为文件数）。原始材料为[官方仓库目录树.json](../backend/eval/data/Claude代码流式与恢复机制核查-20261022/官方仓库目录树.json)及[官方仓库补充核查.json](../backend/eval/data/Claude代码流式与恢复机制核查-20261022/官方仓库补充核查.json)。

该仓库属于 Anthropic 官方组织，是官方公开仓库；当前内容以插件、示例、脚本、开发配置、更新日志及问题反馈入口为主，未提供 Claude Code CLI 核心源码工程。没有根 src 目录只是辅助观察，结论结合目录整体、README 的安装与插件定位及根许可证判断，不能把缺少某个目录名单独当作证明。

根 LICENSE.md 标注 All rights reserved，并指向 Anthropic 商业服务条款，没有以 MIT、Apache 等开源许可证发布 Claude Code 核心产品。未逐个核查插件独立许可证，不将这一结论扩展为所有附属内容都无开源许可。

对原问题的影响：无法从该仓库直接追踪 CLI 内部的 SSE 重试、检查点备份及恢复 handler。前文分析来自另外两个官方 SDK 仓库与产品文档，并非从此仓库读到的核心实现。

本次为静态核查，无运行实验与失败样本；性能及恢复成功率仍为未测量。下一步可继续阅读已归档的 SDK 源码，或在获得明确核心源码版本后另行核查。
