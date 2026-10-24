# LongMemEval-S 外部记忆系统基线

统一 runner 将 Mem0 OSS、Graphiti OSS 和 Supermemory 的检索结果转成项目现有的 LongMemEval-S 结果格式。每次运行都会创建新目录，只有显式传入 `--resume --run-name ...` 才会续跑旧目录。

## 可比性约束

- 题集：同一份 `fixed_100_manifest.json`。
- 隔离：每道题使用独立 Provider 存储，题目间不会泄漏记忆。
- 写入粒度：每个 LongMemEval 原始 session 调用一次 Provider 写入，保留 `source_session_id`。
- 检索：默认 Top-5，与项目现有回归对齐。
- 抽取模型：默认 `qwen-plus`，向量模型为 `text-embedding-v4` 1024 维。
- 端到端模式：回答模型与 Judge 直接复用 `run_longmemeval_regression.py` 的固定配置。
- 诉求边界：OSS/自托管结果不冒充各厂商的托管版成绩。

## 隔离依赖

运行时依赖位于 `eval/data/longmemeval_s/runtime/`（已 gitignore），不会改变生产后端依赖。当前验证版本：

- Mem0：`mem0ai==2.0.18`、`fastembed==0.8.0`、`spacy==3.8.15`、`en-core-web-sm==3.8.0`。
- Graphiti：`graphiti-core==0.29.3`，源码 commit `993e081a6d7948a0d8851c12a5fbdbeb49fed862`，隔离评估使用 embedded Kuzu。Kuzu 已被 Graphiti 上游标记废弃，因此正式工业迁移应另用 Neo4j 复测。
- Supermemory：官方 self-hosted server `0.0.8`；对应仓库 commit `34876664810a43a55954a0a83571662a3bd333b8`。

Mem0 环境：

```bash
python3 -m venv eval/data/longmemeval_s/runtime/venvs/mem0
eval/data/longmemeval_s/runtime/venvs/mem0/bin/python -m pip install \
  'mem0ai==2.0.18' 'fastembed==0.8.0' 'spacy==3.8.15' \
  python-dotenv socksio
eval/data/longmemeval_s/runtime/venvs/mem0/bin/python -m spacy download en_core_web_sm
```

Graphiti 环境：

```bash
git clone https://github.com/getzep/graphiti \
  eval/data/longmemeval_s/runtime/graphiti
git -C eval/data/longmemeval_s/runtime/graphiti checkout \
  993e081a6d7948a0d8851c12a5fbdbeb49fed862
python3 -m venv eval/data/longmemeval_s/runtime/venvs/graphiti
eval/data/longmemeval_s/runtime/venvs/graphiti/bin/python -m pip install \
  -e 'eval/data/longmemeval_s/runtime/graphiti[kuzu]' 'httpx[socks]'
```

Supermemory 可连托管 API，也可连官方本地 server：

```bash
npx --yes supermemory@latest local install
npx --yes supermemory@latest local --port 6767
export SUPERMEMORY_BASE_URL=http://127.0.0.1:6767
export SUPERMEMORY_API_KEY='<server 首次启动打印的 sm_...>'
```

## 运行

先做单题端到端 smoke：

```bash
cd backend
.venv/bin/python eval/scripts/run_longmemeval_external_baseline.py run \
  --provider mem0 --mode full --top-k 5 --limit 1 --verbose-sessions
```

固定 100 题正式基线：

```bash
.venv/bin/python eval/scripts/run_longmemeval_external_baseline.py run \
  --provider mem0 --mode full --top-k 5 --workers 4

.venv/bin/python eval/scripts/run_longmemeval_external_baseline.py run \
  --provider graphiti --mode full --top-k 5 --workers 4

.venv/bin/python eval/scripts/run_longmemeval_external_baseline.py run \
  --provider supermemory --mode full --top-k 5 --workers 4
```

结果位于 `eval/data/longmemeval_s/external_results/<timestamp>_<provider>_<mode>_k5/`：

- `questions.jsonl`：逐题结果，续跑时只追加。
- `metrics.json` 和 `report.md`：汇总指标。
- `ingestion_checkpoints/`：逐题、逐会话断点。
- `provider_state/`：Provider 原生索引/图数据。
- `provider_logs/`：每题独立 Provider 日志。
- `provider_manifest.json`：Provider 版本、存储和检索配置。

固定 100 题共有 4,786 个原始会话。完整运行会产生数千次抽取请求；中断后必须使用原 `--run-name` 加 `--resume`，不要新建目录重复写入。
