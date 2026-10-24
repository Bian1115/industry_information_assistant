# DeepResearch 网页正文读取

DeepResearch 现在采用“搜索发现 URL、阅读器提取正文、事实与原文逐条绑定、生成后核验引用支持性、发布门禁阻止不可审计报告”的流程。

## 默认链路

1. 博查负责搜索并返回候选 URL。
2. `httpx + trafilatura` 下载、清洗静态 HTML。
3. PDF 使用本地 `PyPDF2` 提取文本。
4. 静态正文不足、403或部分服务端错误时，可选调用 Crawl4AI 处理 JavaScript 页面。
5. 正文和失败结果写入本机临时缓存，避免反复抓取。
6. 搜索摘要只作为线索；未成功读取全文的 URL 不会进入正式事实、数据点或引用。
7. 报告组装后抽取全部“附近论断–URL”对，使用研究阶段已保存的原文快照，核对关键主体、年份、数值、比较和关系。
8. 任一论断被判为 `unsupported` 或 `unknown`，或评审返回缺失/无法解析时，清空对外发布字段并拒绝发布。

本地阅读器不会新增模型调用，也不需要单独充值。

## 失败分类统计

每次读取都会记录 `failure_category` 和可选的 `http_status`。研究状态中的
`web_reader_stats` 按唯一 URL 的最新结果汇总，包含成功数、无效数、无效比例、
分类计数和失败明细；同一 URL 重试不会被重复计算。

核心分类如下：

- `http_403`、`http_404`、`http_429`：对应 HTTP 状态码。
- `timeout`：连接、读取或请求总时限超时。
- `dynamic_page`：页面只有 JavaScript 应用壳，浏览器兜底也未取得足够正文。
- `content_too_short`：网页可读取，但正文低于最低字符数。
- `pdf_no_text_layer`：PDF 没有可提取文本，通常是扫描件。
- `network_error`：DNS、连接、TLS 或代理等网络错误。
- `http_other`、`invalid_url`、`response_too_large`、`redirect_error`、
  `extraction_error`：其余可诊断错误。

研究完成事件的 `stats.reader_failure_categories` 也会携带分类计数，可供前端
或评测脚本直接展示。

## 配置

配置示例已加入 `.env.example`。常用项：

```dotenv
WEB_READER_ENABLED=true
WEB_READER_TIMEOUT_SECONDS=15
WEB_READER_MAX_BYTES=10485760
WEB_READER_MAX_CHARS=30000
WEB_READER_MIN_CHARS=500
WEB_READER_CACHE_TTL_SECONDS=604800
WEB_READER_RETRY_ATTEMPTS=2
WEB_READER_REQUIRE_FULL_TEXT=true
RESEARCH_GATE_REQUIRE_DECLARED_METRICS=true
RESEARCH_GATE_CITATION_SUPPORT_ENABLED=true
RESEARCH_GATE_CITATION_SUPPORT_BATCH_SIZE=6
RESEARCH_GATE_CITATION_SUPPORT_SOURCE_CHARS=4000
```

`RESEARCH_GATE_CITATION_SUPPORT_ENABLED` 默认开启。支持性核验使用已保存的原文快照，不在发布时重新访问网页，以避免页面更新或二次抓取失败改变判定口径。

如果宿主机访问某些网站需要代理，可设置：

```dotenv
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
WEB_READER_BROWSER_PROXY=http://host.docker.internal:7890
```

`httpx` 自动读取 `HTTP_PROXY/HTTPS_PROXY`；Crawl4AI 浏览器使用
`WEB_READER_BROWSER_PROXY`。代理只影响正文抓取，博查搜索结果由博查服务端生成。

## 可选启用 Crawl4AI

Crawl4AI 不是默认依赖。Docker 边车限制为最多 2 GB 内存，并绑定本机端口：

```bash
cd /Users/bxy/projects/industry_information_assistant/backend
docker compose -f compose.crawl4ai.yml up -d
```

随后在 `.env` 设置：

```dotenv
CRAWL4AI_ENDPOINT=http://127.0.0.1:11235
```

查看和停止：

```bash
docker compose -f compose.crawl4ai.yml ps
docker compose -f compose.crawl4ai.yml down
```

也可以把 `requirements-crawl4ai.txt` 安装到独立 Python 环境，但不建议与主后端依赖混装。Crawl4AI 镜像包含 Chromium 与浏览器依赖，镜像层较大；首次拉取在带宽较低时会显著慢于普通 Python 包。

Docker Hub 不稳定时可使用独立 Python sidecar：

```bash
cd /Users/bxy/projects/industry_information_assistant/backend
python3 -m venv .venv-crawl4ai
.venv-crawl4ai/bin/pip install -r requirements-crawl4ai.txt
.venv-crawl4ai/bin/playwright install chromium
.venv-crawl4ai/bin/uvicorn crawl4ai_sidecar:app --host 127.0.0.1 --port 11235
```

主后端仍只需设置 `CRAWL4AI_ENDPOINT=http://127.0.0.1:11235`，不关心兜底来自Docker还是独立进程。

### 本机曾出现的下载故障

Docker Hub 拉取 `unclecode/crawl4ai:0.8.0` 时曾两次报 `short read ... unexpected EOF`：
一个约 190 MB 的层只收到约 159 MB，重试时另一个约 170 MB 的层只收到约 17 MB。
这表示 Docker Hub/桌面代理链路在大文件传输中断，并非包版本不兼容。宿主机经代理访问
PyPI 和 Playwright CDN 正常，因此项目保留 Docker 与独立 Python sidecar 两条部署路径。

2026-08-25 继续拉取时，Docker 复用了此前已完成的镜像层并成功补齐剩余层；
`unclecode/crawl4ai:0.8.0` 已完整安装。当前 `industry-crawl4ai` 容器健康运行，
本地 Python sidecar 仅作为备用方案。

## 分类后的恢复策略

- `content_too_short` / `dynamic_page`：转入 Crawl4AI 浏览器渲染；仍不足则由搜索层换权威来源。
- `http_403` / 部分 `http_other`：浏览器渲染一次，不无限重试。
- `timeout` / `network_error` / `http_429` / 5xx：指数退避，默认最多 2 次。
- `http_404` / `invalid_url`：立即淘汰，由搜索层换源。
- `pdf_no_text_layer`：配置 `WEB_READER_OCR_ENDPOINT` 时转入 OCR，否则明确失败。

## 费用边界

- 必需：阿里云百炼/DashScope 模型额度、博查搜索额度。
- 按需：Jina Reader、OpenRouter、阿里云文档智能、聚合数据。
- 免费本地组件：Trafilatura、BeautifulSoup、PyPDF2、Crawl4AI 自托管及本地缓存。

自托管 Crawl4AI 本身不收费，但会消耗本机 CPU、内存和网络；若另行配置其
LLM 抽取策略，则对应模型供应商仍会计费。本项目当前集成未启用 LLM 抽取。

## 验证

从后端目录运行：

```bash
PYTHONPATH=app .venv/bin/python -m unittest discover -s eval/tests -p 'test_*.py' -v
```
