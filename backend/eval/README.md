# 评估基础设施

项目的评估实现统一放在本目录。业务目录和 `backend/scripts/` 只保留薄兼容入口，
避免历史导入和命令立即失效；新增评估代码应直接写到这里。

## 目录结构

```text
eval/
├── runtime/                  # 在线旁路：LLM trace、benchmark 数据源 profile
├── models.py                # agent_runs SQLAlchemy 模型
├── scripts/                 # DRB、LongMemEval、报表、迁移和语料导入 CLI
├── tests/                   # 评估基础设施的纯离线回归测试
├── data/                    # benchmark 数据与运行结果（大文件默认 gitignore）
├── longmemeval_s/           # LongMemEval-S 固定套件与说明
└── profiles.json            # CRUD-RAG、FinEval、DuSQL、Falcon 等数据源配置
```

## 常用命令

从 `backend/` 目录执行：

```bash
# 追踪表迁移与报表
.venv/bin/python eval/scripts/migrate_eval_schema.py --check
.venv/bin/python eval/scripts/eval_report.py --days 7

# DeepResearch Bench
.venv/bin/python eval/scripts/run_deepresearch_bench.py --lang zh --limit 2

# 每次默认创建独立运行名：<前缀>-<题数>q-run<轮次>-<时间戳>
# 例如：industry-ass-5q-run002-20260820_154548

# 仅在中断后续跑同一轮时，显式传入完整运行名
.venv/bin/python eval/scripts/run_deepresearch_bench.py \
  --resume --model-name industry-ass-5q-run002-20260820_154548 \
  --lang zh --limit 5

# 查看某一轮统计并导出该轮的 RACE/FACT 输入与 Markdown 报告
.venv/bin/python eval/scripts/run_deepresearch_bench.py \
  --stat --model-name industry-ass-5q-run002-20260820_154548

# benchmark profile
.venv/bin/python eval/scripts/eval_profile.py
.venv/bin/python eval/scripts/eval_profile.py --verify crud_rag

# LongMemEval-S
.venv/bin/python eval/scripts/run_longmemeval_regression.py prepare
.venv/bin/python eval/scripts/run_longmemeval_regression.py run

# 离线测试
.venv/bin/python -m unittest discover -s eval/tests -p 'test_*.py' -v
```

DRB Runner 使用当前 `DeepResearchV2Service.research()`，需要业务库已迁移、
LangGraph 检查点表已创建。`RESEARCH_ALLOWED_DOMAINS` 默认留空使用全网搜索，
需要限制网站时再配置逗号分隔的域名。检查点建表沿用
部署命令 `PYTHONPATH=app .venv/bin/python -m service.deep_research_v2.checkpointer`；
Runner 只管理连接生命周期，不自动建表。应在评测专用数据库运行。
CLI 评测保持无登录用户、无聊天 turn 的方式，事件与运行终态正常持久化。
结果中的 `_meta.run_id` 保留批次时间戳，`_meta.research_run_id` 是该题实际运行 ID，
可用于查询检查点与事件；`--resume` 仍是按题跳过已有结果，不是恢复某题的图执行位置。

旧的 `python scripts/<name>.py` 命令仍可用，但仅用于兼容。

## 正文引用核验收益消融

使用已独立评分的中文五题（35、31、46、16、49）固定初稿与原始证据，比较原稿/仅审计、失败单元删除和一次改写。运行当前生产核验器，相同论断–证据输入复用首次结果；改写后复核仍失败只记录并保留。删除按段落/列表项/表格行执行，必须同时查看原已支持及重要事实保留率。

```bash
PYTHONPATH=app .venv/bin/python eval/scripts/运行正文引用核验消融.py

# 恢复同一运行；已完成API响应按请求指纹复用。
PYTHONPATH=app .venv/bin/python eval/scripts/运行正文引用核验消融.py \
  --out eval/data/正文引用核验消融-20261006-170312 --resume
```

默认创建中文名称的独立结果目录，并写入 `docs/` 中文分析报告。保留冻结初稿、配置、逐论断证据、初次/复核输出、改写响应、处理后报告、独立重抽取与支持裁判、保留判定及真实usage/耗时。本实验测固定轨迹后处理；来源支持性与现实真实性分开。该历史实验采用Writer声明论断口径。2026-10-17后的生产入口已独立从最终报告抽取论断、选取引用证据句并核验，仍不宣称穷尽全文事实；见[实现与验证记录](../../docs/最终报告独立引用核验与Critic职责分离实现及验证-20261017.md)。

## Critic职责分离配对消融

`eval/scripts/运行Critic职责分离五题配对消融.py --output <新的中文结果目录>`使用既有通用中文五题，在首次审核前冻结公共首稿，无Critic分支正常交付，开启分支继续真实审核与最多两轮返工。两组均保留独立报告引用核验。运行使用隔离数据库和冻结源码，不调用外部质量裁判；报告质量由Codex主代理全文审读后另行落盘。

全部模型HTTP调用写入`题目*/全部模型API用量.jsonl`，区分公共首稿、审核返工和关闭分支的索引维护；无usage的调用不填零。`汇总职责分离消融过程指标.py --directory <结果目录>`生成引用与用量汇总，`汇总消融累计实际用量.py`可合并失败运行和模型探测账本。匿名评分完整落盘后，才运行`汇总Critic配对匿名质量评估.py`揭示组别并计算增益。正式运行的模型兼容映射、非思考模式和预算配置以每次运行的`运行配置.json`为准。
