# LongMemEval-S 固定 100 题日常回归

这套回归直接调用项目当前的 `MemoryProvider`。每道题映射为一个隔离用户，
每段历史映射为该用户的一次聊天会话，因此测到的是跨会话长期记忆，而不是
把完整历史直接塞给回答模型。

## 固定题集

`suite_100.json` 锁定了以下内容：

- 官方 `longmemeval_s_cleaned.json` 的 SHA256；
- 固定的 hash 排序算法和 seed；
- 100 题的题型配额，其中 6 题为 abstention。

`prepare` 会下载约 277 MiB 的官方数据，校验 SHA256，并把具体 100 个
`question_id` 写到 `eval/data/longmemeval_s/fixed_100_manifest.json`。该目录
不入 Git，但相同数据和 suite 配置必然生成相同 ID 清单；脚本每次运行都会
重新计算并校验，不能静默漂移。

```bash
cd backend
.venv/bin/python eval/scripts/run_longmemeval_regression.py prepare
```

如果运行环境不能访问 Hugging Face，可手动把官方文件放到：

```text
backend/eval/data/longmemeval_s/longmemeval_s_cleaned.json
```

再加 `--no-download` 运行。SHA256 不一致时脚本会拒绝执行。

## 首次建立记忆与日常回归

运行前需要 PostgreSQL、Milvus 和 `DASHSCOPE_API_KEY`。首次执行会为缺失题目
建立记忆；之后默认复用完整索引，只执行查询和评分。首次建立 100 题的成本
明显高于日常查询，建议在稳定环境单独完成。

```bash
# 首次运行或日常检索回归；每次自动创建一个时间戳目录，历史结果不会覆盖
.venv/bin/python eval/scripts/run_longmemeval_regression.py run

# 首次建立固定 100 题索引时可并行；并发数会写入 run_config.json
.venv/bin/python eval/scripts/run_longmemeval_regression.py run --workers 8

# 记忆抽取逻辑发生变化时，用新的 run-name 重建
.venv/bin/python eval/scripts/run_longmemeval_regression.py run \
  --rebuild --run-name rebuild_$(date +%Y%m%d)

# 先用 1 题检查完整基础设施；该结果不能作为正式基线
.venv/bin/python eval/scripts/run_longmemeval_regression.py run \
  --limit 1 --run-name smoke
```

如某次运行中断，使用原目录名显式续跑：

```bash
.venv/bin/python eval/scripts/run_longmemeval_regression.py run \
  --run-name 20260820_153000_retrieval_k5 --resume
```

默认只测检索，稳定且成本较低。`--mode full` 会先通过 DashScope 回答模型生成
答案，再通过 Claude 原生 Messages API 调用独立的 Relay Judge 判分。回答模型由
`LONGMEMEVAL_ANSWER_MODEL` 指定，默认固定为
`qwen3.7-plus-2026-05-26`；Claude Judge 使用
`CLAUDE_RELAY_API_KEY`、`CLAUDE_RELAY_BASE_URL`，模型由
`LONGMEMEVAL_JUDGE_MODEL` 指定并默认回退到 `CLAUDE_RELAY_MODEL` /
`claude-opus-5`。Opus 5 请求不传已废弃的 `temperature` 参数。每次 full 运行
都会把两个模型、provider、提示版本、输出 token 上限、重试次数和 Judge 协议写入
结果配置。Judge 输出上限固定为 1024，避免推理模型在较长评分提示上耗尽额度、尚未
生成最终 `yes/no`。连接超时、限流、连接重置和 Judge 空文本最多尝试 3 次。
回答 provider 的内容审核拒绝会作为一次真实的模型拒答写入逐题结果并交给 Judge
判分，不会被误记为评估基础设施故障。

```bash
.venv/bin/python eval/scripts/run_longmemeval_regression.py run \
  --mode full --run-name full_$(date +%Y%m%d)
```

## 指标与回归门禁

检索报告提供 Any Recall@K、All Recall@K、Macro/Micro Session Recall@K 和
MRR。遵循 LongMemEval 官方约定，30 道 abstention 没有可定位的正确证据，
所以固定套件中的 6 道 abstention 不进入检索指标；端到端 QA Accuracy 仍然
包含它们。

报告同时给出 Session Ingestion Success 和 Partial Questions。某些公开历史
可能被上游模型内容审核拒绝；评估器会继续写入其余会话并计算诊断性检索指标，
但该题会标记为 `partial`，回归门禁仍然失败，不能被当作完整有效样本掩盖。

第一次稳定运行后显式晋升基线：

```bash
.venv/bin/python eval/scripts/run_longmemeval_regression.py run \
  --run-name baseline_candidate \
  --write-baseline eval/longmemeval_s/baseline.local.json
```

日常任务用基线门禁，默认允许每个可比总指标最多下降 3 个百分点：

```bash
.venv/bin/python eval/scripts/run_longmemeval_regression.py run \
  --baseline eval/longmemeval_s/baseline.local.json
```

也可以设置绝对门槛：

```bash
.venv/bin/python eval/scripts/run_longmemeval_regression.py run \
  --min-any-recall 0.70 --min-all-recall 0.40
```

任何题目报错、没有跑满固定题集或指标越过门槛，进程都会以非零状态退出。
每次运行会在 `backend/eval/data/longmemeval_s/results/<run-name>/` 创建独立目录，
保存 `questions.jsonl`、`metrics.json`、`report.md`、`run_config.json`，以及套件和
固定 100 题清单的快照。默认不允许覆盖同名目录；只有显式 `--resume` 才会更新
该次未完成的运行。

## Raw Session 对照实验

`run_longmemeval_raw_ablation.py` 用同一固定 100 题运行两种诊断模式：

- `oracle-raw`：使用数据集标注的答案相关原始会话。
- `retrieved-raw`：使用冻结的当前检索会话 ID，再回填这些会话的原始 turns。

先进行纯离线映射、SHA 和上下文长度检查：

```bash
.venv/bin/python eval/scripts/run_longmemeval_raw_ablation.py prepare --no-download
```

正式运行和比较命令见
[`第一次_结果分析_对照试验.md`](./第一次_结果分析_对照试验.md)。Raw 运行与日常回归
一样创建独立结果目录、拒绝覆盖同名目录，并支持显式 `--resume`。当前已完成的正式
三组比较位于
`eval/data/longmemeval_s/comparisons/20260822_212333_raw_session_ablation_v2/`。

## 解释边界

- 默认检索 Top-K 是记忆碎片 Top-K；同一会话可能占多个碎片，因此报告同时
  保留原始 evidence 和去重后的 session ID。
- 默认日常运行复用索引，适合发现检索行为回退；记忆抽取或总结提示词变化后
  必须显式 `--rebuild`，否则不会测到新写入逻辑。
- `--mode full` 的结果受回答模型、judge 模型和服务端模型版本影响，基线中应
  固定模型环境变量，并把检索指标和 QA Accuracy 分开判断。
