"""将配对结果、局限和可追溯样例写入新的中文报告。"""
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'backend/eval/data/网页正文与表格优化对照评估-20261018-首轮'
REPORT = REPO / 'docs/网页正文遗漏与表格丢失优化前后评估及分析-20261018.md'


def pct(x):
    return f'{100 * x:.2f}%'


def main():
    comparison = json.loads((ROOT / '优化前后对比汇总.json').read_text())
    before = comparison['versions']['优化前']['summary']
    after = comparison['versions']['优化后']['summary']
    b_rows = {r['id']: r for r in json.loads((ROOT / '优化前/逐页指标汇总.json').read_text())}
    a_rows = {r['id']: r for r in json.loads((ROOT / '优化后/逐页指标汇总.json').read_text())}
    path = '../backend/eval/data/' + ROOT.name
    gt = after['datasets']['WebMainBench545']['modes']['production_pre_gate']['grid_tables']
    bt = before['datasets']['WebMainBench545']['modes']['production_pre_gate']['grid_tables']
    noisy_before = before['datasets']['WCXB分层200']['modes']['production_pre_gate']['snippets']['without']
    noisy_after = after['datasets']['WCXB分层200']['modes']['production_pre_gate']['snippets']['without']
    cases = [
        ('WCXB-0556', '论坛正文遗漏', '帖子被包在批量操作表单内。只排除控件、保留明确帖子节点后补回讨论内容；没有获取新网页。'),
        ('WCXB-4317', '论坛多帖遗漏', '原结果主要保留少量帖子；明确选择多帖正文后，显著补足讨论内容。'),
        ('WCXB-2913', '列表遗漏与新增噪声', '原结果几乎只有页面标题；主区卡片被补回，但筛选信息和趋势入口仍混入。'),
        ('WCXB-0614', '商品信息遗漏', '从明确商品主区补回产品信息、价格等字段；仍有探索推荐入口残留。'),
        ('WebMainBench-8ec3baea-f43f-4f57-bde4-b67507bd56c1', '论文表格丢失', '从页面主区和原始源表恢复结构，表格完整匹配改善；正文整体仍只覆盖部分参考。'),
        ('WebMainBench-670bfc19-ef03-4389-8a76-5881d9476f49', '下载规格表丢失', '恢复架构、包大小和安装大小表；基线只有散文/列表部分。'),
        ('WebMainBench-c96c623a-12ed-439e-9042-6e51c7d61c1f', '主内容边界扩大导致退化', '补回真实提交元信息表，而参考主要包含提交说明和代码。词元 Precision 降低，不能将补回的源内容称为幻觉。'),
        ('WCXB-0592', '表格与金标范围不一致', '补回真实营养成分表，但参考正文未包含该表，造成指标上的额外内容和 F1 退化。'),
        ('WCXB-0637', '集合主区误扩', '集合页主区包含更多商品卡片，参考聚焦介绍文字；范围扩大导致 F1 下降。'),
        ('WebMainBench-dcf914e5-639a-40a8-8b44-ce46934905c2', '尚未解决的表格遗漏', '参考文献类表未通过本轮源表筛选，诊断候选数为零，仍需细分表格类型。'),
        ('WCXB-4147', '尚未解决的正文遗漏', '未匹配到可用的明确帖子节点，正文仍为空；本轮未测试浏览器渲染。'),
        ('WebMainBench-93898d00-0d6c-451d-9f99-4c386c6c2918', '尚未解决的目录遗漏', '小说章节目录仍未被正文选择规则覆盖；不能根据原始页面很长就认定已取得正确正文。')]
    records = []
    for sid, category, note in cases:
        a, b = a_rows[sid], b_rows[sid]
        records.append({'id': sid, '类别': category, '对照结论': note, 'URL': a['url'],
                        '复核方式': 'Codex对照冻结参考与预测；不是独立人工金标或外部LLM裁判',
                        '优化前指标': b['scores']['production_pre_gate'], '优化后指标': a['scores']['production_pre_gate'],
                        '选择诊断': a.get('extraction_diagnostics', {}), '优化前状态': b['production_result']['fetch_status'],
                        '优化后状态': a['production_result']['fetch_status']})
    (ROOT / '失败与改善样例对照核查.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    parts = [f'''# 网页正文遗漏与表格丢失优化前后评估及分析

评估日期：2026-10-18，Asia/Singapore。对象：优化前的 `trafilatura-native-markdown-v1` 与最终 `semantic-body-source-tables-v5`。性质：固定 HTML 的本地开发与配对回归；新增付费 API 调用为 **0**。

正文 Recall、F1 和表格保留得到改善，代价是部分主区扩张带来额外内容。表格完全丢失从 {bt['pages_all_tables_missing']}/57 降到 {gt['pages_all_tables_missing']}/57；WCXB 标注噪声命中页从 {noisy_before['pages_with_hit']}/194 增到 {noisy_after['pages_with_hit']}/194。不能把本次结果描述为所有质量维度均提升。

## 数据、配置及比较边界

- 沿用上次全部 WebMainBench 545 页和 WCXB dev 固定分层 200 页，没有重新选取有利样本。WCXB 的 6 页空正文金标仍保留读取状态，但不参加质量均值，正式比较有效页为 739。
- 每个版本都重新读取同一批冻结 HTML；正文参考和 HTML 的指纹逐页配对。没有把金标、页面类型标签或问题答案传给业务提取函数。
- 最短门槛保持 500 字符，字符预算保持 30000，既有跨表尾延长规则保留；没有普遍提高输出预算换取 Recall。两版均有 1 页因整表延长超过预算，最大 228132 字符。
- 下载由冻结字节替代，socket 连接被阻止，浏览器和 OCR 不执行。每版 745 页，5 个已存档版本共 3725 页次；这不是线上 URL 可达率或浏览器恢复成功率评测。
- 评测环境为 Python 3.11、Trafilatura 2.2.0，新增本地 `markdownify` 1.2.3；应用环境的完整读取回归使用项目现有 Python 3.13 环境。版本及源码指纹见核验文件。
- 本批数据被用于失败分析和规则开发，再作配对复测，因此是开发集回归结果，不能据此宣称对未见网站的泛化提升。WCXB test 未使用。

## 失败定位与实现变更

1. **帖子/卡片被文章规则删掉**：基础提取仍使用 Trafilatura 原生 Markdown；仅在明确帖子正文、多卡片主区或商品结构证据充分时扩大主体范围，剔除控件和菜单。保护单篇文章、单帖页面及已有丰富正文的商品页。帖子正文允许位于批量操作表单内部，表单控件仍排除。
2. **源表存在但输出缺表或损坏**：识别可见、非导航的多行多列表，替换可定位的原生管道表，或补回缺失的源表。净化后的结构 HTML 仅保留表格节点、文本和有限跨度属性，不保留脚本、事件、样式或任意源属性。重复表避免叠加，隐藏弹窗和附件面板排除。
3. **新表格被下游切碎**：预算边界保护完整源表及紧随注释；证据分句把单行源表视为整体，保留原文偏移。
4. **缓存隔离**：更新提取策略版本，使旧正文缓存不会直接冒充优化后的结果。

没有恢复旧 XML 再渲染处理路径，也没有为旧格式断言新增兼容层。主要变更文件：[正文选择与源表恢复](../backend/app/service/web_content_selection.py)、[读取服务](../backend/app/service/web_reader_service.py)、[预算边界](../backend/app/service/web_document_structure.py)、[证据分句](../backend/app/service/web_document.py)、[依赖](../backend/requirements.txt)。

## 正文核心指标

按可见文本计算逐页词元 Precision/Recall/F1，再做宏平均，汉字按字、其他词语按 Unicode 词元处理；另存连续 5-gram 与编辑相似度。Recall 不是“完整页面比例”，1−Precision 也不是人工标注的真实噪声率。新增结构 HTML 的标签不算正文词元；WCXB 原始词级与字面片段检查另存，不能把格式符号造成的变化混成内容变化。

| 数据集/输出口径 | Precision 前→后 | Recall 前→后 | F1 前→后 |
| --- | --- | --- | --- |
''']
    for ds in ['WebMainBench545', 'WCXB分层200']:
        for mode, label in [('production_pre_gate', '门禁前'), ('accepted_output', '离线接纳输出，未接纳计空')]:
            a = before['datasets'][ds]['modes'][mode]['text']; b = after['datasets'][ds]['modes'][mode]['text']
            parts.append(f"| {ds}，{label} | {pct(a['precision'])} → {pct(b['precision'])} | {pct(a['recall'])} → {pct(b['recall'])} | {pct(a['f1'])} → {pct(b['f1'])} |\n")
    parts.append('\n以正文 F1 变化超过 0.5 个百分点计，')
    changes = []
    for ds in ['WebMainBench545', 'WCXB分层200']:
        outcome = comparison['paired_outcomes'][ds]
        changes.append(f"{ds} 改善 {outcome['improved_f1_over_0.5pp']} 页、退化 {outcome['regressed_f1_over_0.5pp']} 页，其余 {outcome['other']} 页")
    parts.append('；'.join(changes) + '。程序接纳页数分别从 472/545、162/200 增为 476/545、173/200；WCXB 接纳统计含 1 页空金标，不进入质量均值。\n')
    parts.append('''
## 按页面类型分析

WCXB 的 200 页近似等额覆盖 7 类，实际有效页数如下；均衡样本的平均分不能当作 dev 原始分布的总体分。

| 类别 | 有效页数 | Recall 前→后 | F1 前→后 |
| --- | --- | --- | --- |
''')
    for name, a in before['WCXB页面类型'].items():
        b = after['WCXB页面类型'][name]
        x, y = a['modes']['production_pre_gate']['text'], b['modes']['production_pre_gate']['text']
        parts.append(f"| {name} | {a['n_scored']} | {pct(x['recall'])} → {pct(y['recall'])} | {pct(x['f1'])} → {pct(y['f1'])} |\n")
    parts.append(f'''
完整的语言、难度及权重校正结果保留在各版本汇总。文章与文档以保持原有质量为主；论坛和列表受益于主体范围扩大，商品及集合页仍存在介绍文字、商品卡片和推荐模块的边界取舍。

## 表格保留

沿用相同金标解析规则：参考叶子表 80 页、199 张；主比较子集为至少 2 行、2 列且 4 个非空单元格的 **57 页、102 张表**。它包含数值、词典、参数、元信息和代码行号表，不等于行业数值表专用测试。WCXB 参考主要是纯文本，不为它推造结构金标。

| 同一57页，生产预算、门禁前 | 优化前 | 优化后 |
| --- | --- | --- |
| 严格整格匹配 Recall | {pct(bt['cells']['recall'])} | {pct(gt['cells']['recall'])} |
| 严格整格匹配 F1 | {pct(bt['cells']['f1'])} | {pct(gt['cells']['f1'])} |
| 横向相邻单元格关系 F1 | {pct(bt['horizontal_relations']['f1'])} | {pct(gt['horizontal_relations']['f1'])} |
| 纵向相邻单元格关系 F1 | {pct(bt['vertical_relations']['f1'])} | {pct(gt['vertical_relations']['f1'])} |
| 完全没有可识别表格的页面 | {bt['pages_all_tables_missing']}/57 | {gt['pages_all_tables_missing']}/57 |

共同适用的 {comparison['paired_table_metrics']['paired_teds_pages']} 页上，TEDS 风格内容/结构相似度为 {pct(comparison['paired_table_metrics']['paired_teds_before'])} → {pct(comparison['paired_table_metrics']['paired_teds_after'])}。这是本项目本地 APTED 诊断指标，不是官方榜单复现；250 节点、单项 3 秒的上限及不适用分母保持一致。严格整格完全相等也会惩罚格式/内容搬移，不能把它解释为只有该比例的数据值被保存。

## 噪声、退化和候选记录

| 版本 | WMB 门禁前 F1 | WCXB 门禁前 F1 | WCXB 噪声命中片段/756 | 噪声命中页/194 |
| --- | --- | --- | --- | --- |
''')
    for name, version in comparison['versions'].items():
        data = version['summary']['datasets']; noise = data['WCXB分层200']['modes']['production_pre_gate']['snippets']['without']
        parts.append(f"| {name} | {pct(data['WebMainBench545']['modes']['production_pre_gate']['text']['f1'])} | {pct(data['WCXB分层200']['modes']['production_pre_gate']['text']['f1'])} | {noise['found']}/756 | {noise['pages_with_hit']}/194 |\n")
    normalized_a = comparison['versions']['优化前']['normalized_snippets']; normalized_b = comparison['versions']['优化后']['normalized_snippets']
    parts.append(f'''
规范化可见文本后，必含片段命中 {normalized_a['with']['hits']}/891 → {normalized_b['with']['hits']}/891；噪声仍为 {normalized_a['without']['hits']} → {normalized_b['without']['hits']} 个片段。最终选择收紧了首轮候选的过度扩张，但噪声尚未回到基线：商品指南、探索入口、论坛展开控件与设备签名等仍有残留。

源表是真实页面内容，也可能超出参考范围。例如提交元信息和营养成分表会使 Precision 降低，不能一概称为幻觉；另一些筛选菜单和推荐入口则确实是正文污染。报告同时保留这两类退化，不以字数增多代替质量。

## 逐页失败与改善对照

样例解释由 Codex 对照冻结参考和预测完成，不是独立人工金标或新付费裁判。[机器记录]({path}/失败与改善样例对照核查.json)包含完整指标、诊断及 URL。

| 类别 | 样例与测量 | 原始配对 |
| --- | --- | --- |
''')
    for sid, category, note in cases:
        a, b = a_rows[sid]['scores']['production_pre_gate']['text'], b_rows[sid]['scores']['production_pre_gate']['text']
        table_note = ''
        old_table = b_rows[sid]['scores']['production_pre_gate'].get('table', {}).get('grid_tables', {})
        new_table = a_rows[sid]['scores']['production_pre_gate'].get('table', {}).get('grid_tables', {})
        if old_table.get('eligible') and new_table.get('eligible'):
            table_note = f" 严格单元格 Recall {pct(old_table['cells']['recall'])} → {pct(new_table['cells']['recall'])}。"
        parts.append(f"| {category} | {note} 正文 Recall {pct(b['recall'])} → {pct(a['recall'])}，F1 {pct(b['f1'])} → {pct(a['f1'])}。{table_note} | [前]({path}/优化前/逐页结果/{sid}/生产门禁前正文.txt)、[后]({path}/优化后/逐页结果/{sid}/生产门禁前正文.txt)、[参考]({path}/优化后/逐页结果/{sid}/参考正文.txt) |\n")
    parts.append(f'''
补充校正：旧报告对词典页“表格消失”的描述不作为本次新增成功证据。本轮该页基线已经能按原生管道表解析，优化只是替换为保留源单元格的结构输出；本轮样例与改善计数均以重新运行的实际配对为准。

## 验证、费用与原始结果

- [全部版本与对比指标]({path}/优化前后对比汇总.json)、[739页逐页配对]({path}/逐页配对变化.json)、[改善与退化候选]({path}/改善与退化候选.json)、[计分器及样例独立重算]({path}/计分器与样例重算核验.json)。
- 每个版本目录包含 `冻结代码/`、`冻结评估脚本.py`、`冻结样本清单.json`、`评估配置.json`、`逐页指标汇总.json`、`核心指标与分层汇总.json`、冻结 HTML 和参考/预测文本，未覆盖历史评估。
- [最终核验]({path}/最终输入源码与费用核验.json)保存全部输入/参考校验、源码指纹、包版本和费用审计；各版 socket 连接尝试和付费 API 调用均为 0。没有调用 MinerU、LLM、OCR、浏览器或其他托管服务。
- 新增外部服务费用为 0；本地电费、存储、既有订阅和网络接入费用未核算。
- [最终应用环境回归]({path}/论坛表单修正后最终回归日志.txt)为 35 项通过，包含新增主体选择、表格跨度、去重、隐藏控件、预算、证据索引，以及既有读取和编码测试。评测虚拟环境首次全量测试因缺少 `psycopg` 导入失败，日志保留在[首次测试记录]({path}/最终机制与读取回归日志.txt)；随后使用依赖完整的应用环境完成验证，没有新增旧接口/格式兼容层来迁就断言。

重算配对统计与报告：

```sh
backend/.venv-eval-py311/bin/python backend/eval/scripts/比较正文与表格优化结果.py
backend/.venv-eval-py311/bin/python backend/eval/scripts/生成正文与表格优化评估报告.py
```

重新提取时应另建目录并复制对应版本的冻结代码、公开参数和原始数据入口，再执行离线评估脚本；不要清空本文运行目录。

## 局限性与下一步

**已测量**：固定批次的正文/结构指标、候选退化、程序接纳状态、同输入配对，以及本地机制和读取回归。新增接纳只是程序状态，不等于经独立语义审核的可用正文；结构 HTML 标签也会影响原有原始字符门槛。

**间接推断**：多帖/卡片选择以及源表补足是主要改善来源；对金标范围外表格的区分仍需要更细致的正文边界规则。本轮没有逐组件消融，不能精确拆分每条规则的独立贡献。

**尚未评估**：线上抓取、反爬、真实浏览器恢复、PDF/Office/OCR、图像数值与未见网站泛化。35 项本地测试也不等于完整 DeepResearch 端到端验证。

后续优先收紧主区内筛选/推荐模块和帖子签名，细分真正的正文表、元信息表与附件清单，再用未参与规则开发的 WCXB test 或新业务页面验证。短页 500 字符门槛、超长目录和长文截断仍是独立问题；本轮没有通过降低门槛或扩大普遍预算掩盖它们。
''')
    mode = 'w' if REPORT.exists() else 'x'
    with REPORT.open(mode, encoding='utf-8') as stream:
        stream.write(''.join(parts))
    print(REPORT)


if __name__ == '__main__':
    main()
