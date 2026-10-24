# 医学文献研究后端

默认 `/research/stream` 使用 `medical_literature` 多 Agent 工作流：ChiefArchitect 规划 → DeepScout 检索、筛选与读取 → DataAnalyst 提取证据 → LeadWriter 综合 → CriticMaster 独立审核 → LeadWriter 交付报告。复用现有 Agent 类、各角色模型配置、统一工具运行层、SSE 和细粒度检查点。实施及验证见 [医疗文献多智能体复用改造与回归验证结果](../docs/医疗文献多智能体复用改造与回归验证结果-20260916.md)；接口与范围设置见 [医学文献研究使用与配置说明](../docs/医学文献研究使用与配置说明-20260914.md)。旧通用模式仅供兼容；下文历史示例不代表当前医学默认行为。

## 项目启动

## 会话与记忆存储约定

- PostgreSQL 是会话、用户/助手消息和记忆任务的唯一事实来源。
- Redis 保存研究取消标志及可重建的会话工作上下文；原始聊天记录始终保存在 PostgreSQL。
- Milvus 保存长期记忆向量；`long_term_memories` 与 `memory_jobs` 保存其元数据和可靠任务状态。
- 客户端每轮请求生成一个 UUID `turn_id`。服务端负责同时保存用户与助手消息，前端不要再单独写消息。

会话摘要复现 LangChain 1.4.2 `SummarizationMiddleware`，默认使用 `qwen3.5-plus`、
工作消息达到配置窗口的 85% 时触发，触发后保留最近 10 条消息（包含本轮问题），
并逐字复用默认摘要提示词。未触发时保留全部未覆盖原文。配置、指标及机制验证见
[会话滚动摘要配置与验证结果](../docs/会话滚动摘要配置与验证结果-20261005.md)。
摘要模型切换记录见[会话摘要模型切换验证](../docs/会话摘要模型切换验证结果及分析-20261005.md)。
`SEMANTIC_CONTEXT_*` 设置会话摘要，`CONTEXT_COMPRESSION_ENABLED=0` 仍关闭全部压缩。

评估基础设施统一位于 [`eval/README.md`](eval/README.md)，LongMemEval-S 的详细说明见
[`eval/longmemeval_s/README.md`](eval/longmemeval_s/README.md)。

## 启动中间件

> ⚠️ **本项目有两套中间件编排，二选一，不要同时启动**（都占用 6379 / 19530 / 9000 / 9001，后启动的一套会因端口冲突失败；且两套用的是不同的数据卷，混用会出现"数据莫名消失"）。
>
> | | 根目录 `docker-compose.yml`（推荐） | `backend/docker-compose-base.yml` |
> |---|---|---|
> | 启动方式 | `./start-services.sh start` | 见下方命令 |
> | PostgreSQL | ✅ 内置 | ❌ 需自备外部 PostgreSQL |
> | Elasticsearch | ✅ 内置 | ❌ |
> | Redis 容器名 | `industry_redis` | `document-redis` |
>
> 默认推荐用根目录那套（`./start-services.sh start`），组件齐全、无需额外准备数据库。
> 只有在你已有外部 PostgreSQL、想省资源时，才用下面的精简版。

cd backend
docker compose -f docker-compose-base.yml up -d

### 查看中间件状态
```sh
$ docker ps
CONTAINER ID   IMAGE                                    COMMAND                  CREATED          STATUS                             PORTS                              NAMES
...            redis:7-alpine                           "docker-entrypoint.s…"   ...              Up                                 0.0.0.0:6379->6379/tcp             document-redis
...            milvusdb/milvus:v2.3.3                   "milvus run standalo…"   ...              Up                                 0.0.0.0:19530->19530/tcp           milvus-standalone
...            minio/minio:RELEASE.2023-03-20T20-16-18Z "minio server /minio…"   ...              Up                                 0.0.0.0:9000-9001->9000-9001/tcp   milvus-minio
...            quay.io/coreos/etcd:v3.5.5               "etcd -advertise-cli…"   ...              Up                                                                    milvus-etcd
```

## 安装python依赖包
pip install -r requirements.txt

## 修改env文件
填写模型配置 DASHSCOPE_API_KEY。医学文献检索无需 SERPER_API_KEY 或 BOCHA_API_KEY；这些配置只供旧通用研究使用。

网页深度研究默认使用全网搜索，不再只检索 WHO 和 NIH。
`RESEARCH_ALLOWED_DOMAINS=` 留空或未配置表示全网；需要限定来源时填写逗号分隔的域名
（子域自动包含）。请求中的 `allowed_domains` 可进一步指定范围；部署已限定网站时，
请求只能收窄范围。搜索、正文读取、追问资料继承和报告来源检查使用同一范围。
修改配置后需重启后端，新研究使用新配置；已保存的研究恢复时保留原运行的网站范围。

配置 Milvus 连接（可选，默认 localhost:19530）：
```
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

配置 DocMind 文档解析服务：
```
DOCMIND_ACCESS_KEY_ID=your_access_key_id
DOCMIND_ACCESS_KEY_SECRET=your_access_key_secret
```

# 临时添加环境变量
# 用您的百炼API Key代替YOUR_DASHSCOPE_API_KEY
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"

## 启动后端服务
首次使用 Alembic 管理已有数据库时，先执行：

```sh
alembic stamp 20260819_0001
alembic upgrade head
```

全新数据库以及之后的版本只需执行：

```sh
alembic upgrade head
```

再启动服务：

```sh
python app/app_main.py
```

PostgreSQL 会话 TTL 固定为 24 小时：创建会话及每次写入新用户/助手消息时刷新
`chat_sessions.expires_at`，查看历史、重试同一轮、改标题及后台摘要更新不续期。
到期后会话列表、消息读写及附件读取立即不可用；后台每分钟分批删除会话、消息、
附件及关联记忆任务，并清除 Redis 缓存和附件文件，启动时也补清理一次。
每批最多处理 100 个会话，大量积压会分多轮清理。长期记忆沿用既有会话删除语义，
仅解除会话关联；独立研究运行、事件和图检查点不属于这项会话 TTL 的删除范围。
启用前执行 `alembic upgrade head`（迁移 `20261006_0012`）；已有会话按最后一条消息时间
加 24 小时回填，无消息会话按创建时间回填，已过期的历史会话会在启动清理时删除。


# 接口测试
### 上传文档,用于本地知识库的查询
```sh
cd backend
curl -X POST "http://localhost:8000/documents/upload"   -H "Content-Type: multipart/form-data"   -F "file=@./test/test_doc.pdf"

{"status":"success","message":"成功处理 25 个切片","document_count":25}
```

### 创建会话（需要登录 Token）
```sh
curl -s -X POST http://localhost:8000/sessions \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"新对话"}'
```

### 问答
```sh
curl -N -X POST http://localhost:8000/chat/completion \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "{
    \"session_id\": \"02c32f19-b7f0-42ea-b3c1-7d2bc148c21b\",
    \"turn_id\": \"4f1c61a2-7808-4dbd-adcf-6508814db5e1\",
    \"question\": \"如何解决dify_setups表不存在的问题？\"
  }"

```


### deepseach
```sh
curl -N -X POST "http://localhost:8000/research/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "query": "安责险在矿山行业的应用现状、面临的主要挑战以及改进建议有哪些？",
    "session_id": "02c32f19-b7f0-42ea-b3c1-7d2bc148c21b",
    "turn_id": "d61e9491-e261-4b59-9a5b-e13dc0d235da",
    "max_iterations": 2
  }'

```

### 手动生成长期记忆（异步）

```sh
curl -X POST http://localhost:8000/memories/create \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"02c32f19-b7f0-42ea-b3c1-7d2bc148c21b"}'

# 使用上一步返回的 operation_id 查询任务状态
curl http://localhost:8000/memories/jobs/$OPERATION_ID \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

### 上下文压缩总开关

`CONTEXT_COMPRESSION_ENABLED` 默认 `1`。设为 `0` 后，智能体保留其配置范围内的原始历史回放，跳过资料引用化、工具及旧输出外置、历史摘要生成与旧摘要注入；会话上下文从 PostgreSQL 全量原文读取，跳过压缩缓存、语义摘要注入、消息数及 token 裁剪。会话语义摘要入队也会关闭。单次输入硬预算、当前任务保护和 Scout 原文提取分批仍生效，超限会记录 `context_budget_exceeded`，不会自动切回压缩。

该总开关与逐智能体的 `enabled` 含义不同：后者决定是否注入该智能体的研究内历史，总开关只决定压缩方法。部署参数在新研究的 `context_policy.compression_enabled` 中冻结，技术续跑沿用快照；缺少该字段的旧快照继续原行为与原哈希。显式 `CONTEXT_ABLATION_CONFIG` 文件优先，其 `compression_enabled` 必须显式设为 `false` 才能关闭。真实追问采集器会同步冻结策略与会话层开关。修改 `.env` 后新启动进程生效。

`SESSION_CONTEXT_COMPRESSION_ENABLED` 默认 `1`，可独立关闭会话层压缩。设置总开关 `1`、会话开关 `0` 时，会话从 PostgreSQL 读取全部原文，不使用旧摘要或裁剪缓存；智能体仍按逐 Agent 配置使用摘要、资料引用化和外置。总开关 `0` 仍优先关闭全部方法。显式策略冻结为 `session_compression_enabled`，未包含此字段的旧快照沿用总开关且保持原哈希。会话话题过滤及研究输入硬预算仍生效，因此这一设置测的是当前系统的连续交付轮数，不能直接当作模型原生最大窗口或最大轮数。

### Writer 引用支持性核验

新研究默认开启 Writer 报告论断核验、关闭 Scout 事实核验。部署开关为 `RESEARCH_WRITER_CITATION_VERIFICATION_ENABLED=true`，仅在首跑且请求未提供 `citation_verification` 时读取；设为 `false` 可关闭默认核验。显式请求对象优先，例如 `{"report_claims":false}` 或 `{}` 关闭两个阶段，`{"scout_facts":true}` 只开启 Scout。同次运行续跑沿用冻结配置，缺少配置的历史检查点仍按关闭处理。

Writer 组装、修订或编辑报告后，引用核验模块直接从最终 `final_report` 独立提取原子论断，逐项处理标题、摘要、正文段落、列表、表格行、结论与展望，不使用 Writer 的 `claims` 作为核验清单。参考文献区只用于解析引用目录。每个报告单元须明确返回论断、无外部事实或不确定状态；缺单元、输出截断、单元超预算和 API 失败均不作为通过。提取结果及位置保存在 `report_verification_claims`，覆盖记录、版本和调用用量保存在 `report_claim_extraction`。

程序从论断所在原句解析实际行内引用，支持 Markdown URL 链接、数字引用及引用式链接，在已保存原文中定位被引来源，并用 `qwen3-max`（非思考模式）检查支持性和预测/计划是否被写成事实。缺行内引用记录 `missing_citation`，未知或无法回读的被引来源记录引用错误；其他未被引用的材料不能替代被引来源通过核验，多个被引来源也分别要求支持。核验范围标记 `final_report_extracted_claims`，独立提取完成且核验执行完整时 `extracted_claims_checked=true`；语义抽取仍可能遗漏，始终保留 `all_claims_checked=false`，不表示现实真实性或计算结果已验证。

默认 `report_failure_action=audit` 只记录审计；请求可选 `delete` 删除失败所在单元，或 `rewrite_once` 改写一次后从新报告再次独立提取并复核，复核仍失败只记录并保留，不中断交付。修复使用独立抽取的正文位置，轨迹保存在 `citation_repair`。Critic 聚焦整体逻辑、用户要求覆盖、整体时效和观点平衡，不重复核验遗漏引用或单条来源的预测限定；仍按逻辑、要求覆盖、时效三项评分，口径版本为 `report_review_3d_v2`。旧 Critic 引用类问题交接记录保存在 `citation_issue_handoffs`，交接不代表问题已修复。模型分数尚未校准，真实抽取召回率与核验准确率需另外测量。本次实现与验证见 [最终报告独立引用核验与Critic职责分离实现及验证](../docs/最终报告独立引用核验与Critic职责分离实现及验证-20261017.md)。

### 自然语言摘要与 Claimify

智能体工作历史按 `run_id + agent + context_key` 隔离；同一次研究的续跑继续读取自己的原文、摘要和游标。同一会话的新研究从独立工作历史开始，不能回读上一研究的Agent内部原文。旧会话共享记录仅导入其中属于当前run的消息，跨研究混合摘要保留供审计，不注入模型。会话上下文和显式继承的报告、事实、来源仍通过业务输入传递。

新启动的研究默认仅对 `LeadWriter` 生成和注入本次研究内部的历史摘要。源码部署默认 `AGENT_HISTORY_REPLAY=LeadWriter`；本地和示例配置通过 `AGENT_COMPRESSION_CONFIG` 为 `DeepScout` 保留本run原始历史回放、资料引用化和外置，同时设置 `semantic_summary=false`。其他智能体默认不注入内部历史。请求中的显式逐智能体配置和消融配置文件仍可覆盖新运行默认值。

会话层用一次结构化调用判断话题关系、历史依赖与组合操作。新话题和完整问题保持原话；旧成果只能从同用户同会话已交付候选中选择并再次验证版本。任务准备先提交本轮有效要求，再确定执行路径：纯编辑进入Writer→Critic；定向补证进入Scout→分析→Writer→Critic；结构/范围变化和新研究进入Architect完整路径。目标不明时保存普通会话澄清答复，以`session_clarification`事件显示，不启动研究。`SESSION_TASK_MODEL`可覆盖会话判断模型，未设置时沿用`OPENAI_MODEL`。

执行图只根据已验证计划选择入口，已删除的内部LLM追问路由和`FOLLOWUP_ROUTER_*`配置不恢复。决策、目标成果和配置冻结到本次run，技术恢复不重新路由。技术续跑仅接受当前图协议版本，旧版本或包含已删除路由的检查点明确拒绝恢复。当前会话摘要仍使用已有语义摘要与预算裁剪机制，完整的未摘要增量保护不包含在此次路由改造中。真实追问采集已改为调用生产会话builder，并落盘会话任务输入、计划和判断成本。

研究主图的累计 Agent 模型预算由 `RESEARCH_MAX_LLM_TOKENS` 控制，源码默认 `0`（不限制）；部署环境可设为正整数。单次上下文窗口仍由 `AGENT_CONTEXT_WINDOW` 等参数控制。预算独立保存到 `research_budgets/research_budget_attempts/research_budget_calls`，首个节点初始化时冻结上限，续跑保留失败尝试和已复用章节的消耗。每次 BaseAgent 供应商请求先预留估算输入加输出上限；返回 usage 后结算，未知 usage 保留预留量。已知用量与约束使用的扣账量分别返回，不能把预留称为实际费用。旧运行首次启用时仅导入检查点里已知的历史消耗；升级前未保存的失败消耗无法补造。摘要独立额度尚未实施。

启用预算持久化与检查点租约隔离前，执行 `alembic upgrade head`（新增迁移 `20261014_0014`），并按既有部署步骤运行 `PYTHONPATH=app python -m service.deep_research_v2.checkpointer` 准备 LangGraph 表。检查点、pending writes 与 `research_runs` 租约行必须在同一 PostgreSQL 数据库和 search_path；`RESEARCH_CHECKPOINT_DATABASE_URL` 可换账号，但不能指向另一实例或数据库。写事务锁定运行行并在提交前再次校验 owner、fencing token 和数据库时钟下的期限，失效即回滚完整检查点、blob及pending writes。硬崩溃的精确结束时刻不可知，未结束物理尝试按旧租约截止时间为上界计时，单独标记 `lease_upper_bound`。隔离故障验证运行器：`eval/scripts/验证预算持久化与检查点租约隔离.py --output-dir eval/data/中文新目录`，只操作自建schema和子进程。

图协议 `single-main-v6-atomic-events` 将节点完成通知与完整图检查点同事务保存，将章节正文及完成观察与章节 task 返回值同事务保存。事件账本直接承担事务发件箱，提交后唤醒 SSE 并按 seq 读取已提交事件；重试、重放沿用保存的 event_id/seq，不重新生成章节 restored 事件。实时进度仍即时落库发送。最终 `research_complete` 属于报告交付事务，继续由现有幂等交付与恢复补偿负责，不能把执行检查点完成等同于报告已交付。此次不新增数据库表；旧版本未完成运行需重新发起。验证及边界见[检查点与章节事件原子提交实现及回归分析](../docs/检查点与章节事件原子提交实现及回归分析-20261021.md)。

跨存储恢复采用 `research-storage-v1`：输入、节点、终态和Writer章节的依赖清单嵌入原有LangGraph产物，与checkpoint/pending write一起提交，无新增业务表。清单记录正文版本/归属/generation、历史记录哈希、预算尝试及请求和事件水位；saver在租约事务内复核，恢复入口再次检查原始checkpoint、待合并节点结果和章节task。正文、历史或预算缺失/改写、事件缺号、终态交付冲突会拒绝恢复，状态接口返回 `storage_reconciliation`，HTTP恢复入口返回409。只有额外历史/进度没有图结果时仍按观察保留，不据此跳过节点。

恢复还按冻结的切分参数从已验证正文补建缺失/损坏chunk，核对Milvus中实际存在的ID和作用域，重新索引丢失项。向量服务故障明确记录 `lexical_only`，不补造原文；修复摘要写入 `storage_reconciled` 观察事件。沿用会话TTL，只有到期前已提交的新消息能延长idle期限。旧checkpoint返回 `legacy_checked`，只能核对仍存在的数据，不能追认升级前缺失的依赖。验证命令：`python eval/scripts/验证跨存储对账.py --output-dir eval/data/中文新目录`；加 `--postgres` 使用自建并清理的隔离schema，不调用外部模型。

旧输入仅做资料引用化：将旧提示词中能精确对应已保存工具原文的正文替换为来源编号及回读位置，完整原文保留。引用化与工具记录外置属于同一种原文替换机制，分别处理旧提示词中的正文副本和工具记录本身。`AGENT_INPUT_HISTORY_ENABLED` 控制旧输入引用化，`AGENT_INPUT_HISTORY_REFERENCE_TOKENS` 设置最小替换长度，默认128 token。单条旧输入的模型摘要生成、缓存复用和相关环境开关已删除，整体历史摘要仍由总上下文阈值触发。新配置使用 `input_history.version=input-view-v2`，只含引用化参数；旧版字段仅用于读取历史快照和校验哈希，即使旧开关为true也不在当前源码中执行单条摘要。复现旧评测行为须使用当时冻结的源码和配置。

整体摘要读取本轮已经完成引用替换、工具原文外置后的历史视图。发生变化的内容保存为 `summary_view_source` 独立快照，附原始来源编号、哈希和序号，引用定位基于实际提供的视图；完整原文继续保存，摘要覆盖游标仍按原始历史推进。`summary_input_prepared` 及摘要元数据的 `summary_input` 保存处理前后token数和来源映射（不含提示词封装及旧摘要开销）。旧提示词引用匹配可回查同一run/agent/subtask及配置上下文中已被摘要覆盖的工具结果，仅在本地匹配并校验权限，不重新把整份旧工具正文加入模型输入。未找到精确来源的正文仍保留。

新运行默认 `AGENT_SUMMARY_STRATEGY=claimify-v1`：摘要模型输出 `SESSION INTENT`、`SUMMARY`、`ARTIFACTS` 三栏目自然语言段落，已有未完成事项保存在 `SUMMARY`。格式和长度检查通过后立即保存正文并推进已完成历史的覆盖游标，标记为未核验工作记忆。`AGENT_SUMMARY_CLAIMIFY_ENABLED=0` 默认关闭后续Claimify链路；设为1后，正文提交完成才执行分句、筛选、消歧、拆解、引用定位及按配置启用的NLI。Claimify抽取本身不判定事实真伪；来源定位和NLI仍是不同检查。

正文生成与Claimify审计分别使用 `AGENT_SUMMARY_GENERATION_TIMEOUT_SECONDS` 和 `AGENT_SUMMARY_CLAIMIFY_TIMEOUT_SECONDS`，默认分别为120秒和300秒。`summary_phase_end` 按 `phase=generation/claimify` 分别保存实际 `duration_seconds`，并用 `summary_attempt_id` 关联；保存的摘要元数据包含 `generation_duration_seconds` 与 `claimify_duration_seconds`。Claimify关闭时记录跳过状态，其耗时为null。审计超时或拒绝只记录错误和核验状态，不回退正文或游标，也不自动改写已经提交的正文。原文与来源编号继续保留，可受控回读。`summary_error`、`input_history_error`、阶段及Claimify错误均保存异常类型；阶段开始、结束及正文提交也有独立埋点。既有冻结快照没有 `summary_lifecycle` 字段时，继续使用原来的提交协议和哈希；显式配置文件优先于环境变量，应在新配置中加入该字段才能使用此协议。

提示词来源及输出适配见 [Claimify 提示词来源与接入说明](../docs/Claimify提示词来源与接入说明-20260929.md)。

离线机制回归：`.venv/bin/python eval/scripts/验证Claimify摘要.py`。逐项结果自动保存至 `eval/data/Claimify摘要机制验证-时间戳/`；这些测试不代表真实模型抽取准确率。
