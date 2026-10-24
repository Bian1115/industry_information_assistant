# Trafilatura正文原始定位机制验证及方案分析

评估日期：2026-10-21，Asia/Singapore。

结论：当前本地后端环境的 Trafilatura 2.2.0 可以返回重建后的结构化正文树，但公开 API 没有提供完整的原始正文节点、原始 XPath 或源码区间映射。不能将 bare_extraction 返回的 body 当成原始网页的 article 节点，然后声称其中包含原始正文的全部表格。可以把其输出作为正文识别信号，再映射回原 DOM；这一映射需要额外实现。

## 对象与配置口径

本次检查 backend/.venv 中安装的 Trafilatura 2.2.0，并用 Python 3.13.0 执行一个中文合成 HTML 页面。输入含导航表、main/article 中的正文表、正文段落、caption、跨行跨列表头，以及 id 和 data-source-node 属性。

调用 bare_extraction，参数为 include_comments=False、include_tables=True、favor_precision=True、no_fallback=False、output_format='python'。前四个选项与当前网页读取器的抽取设置一致；业务读取器实际输出格式是 markdown。本测试直接检查序列化前的内容树，不运行整个业务链路，不代表生产线上环境版本已经核实。

原始文件保存在 [Trafilatura正文定位机制验证-20261021](../backend/eval/data/Trafilatura正文定位机制验证-20261021/)：

- [验证脚本.py](../backend/eval/data/Trafilatura正文定位机制验证-20261021/验证脚本.py)
- [原始页面.html](../backend/eval/data/Trafilatura正文定位机制验证-20261021/原始页面.html)
- [提取内容树.xml](../backend/eval/data/Trafilatura正文定位机制验证-20261021/提取内容树.xml)
- [机制测试结果.json](../backend/eval/data/Trafilatura正文定位机制验证-20261021/机制测试结果.json)
- [相关源码摘录.txt](../backend/eval/data/Trafilatura正文定位机制验证-20261021/相关源码摘录.txt)

结果 JSON 记录 Python、Trafilatura 版本、参数、源码路径与哈希。

## 核心结果与逐项分析

| 检查项 | 已测量结果 | 含义 |
| --- | --- | --- |
| 原始表格数 | 2 | 一张导航表，一张正文表 |
| 输出表格数 | 1 | 本例保留正文表，未保留导航表 |
| 正文表原始 XPath | /html/body/main/article/table | 属于输入 DOM |
| 输出表格 XPath | /body/table | 属于重建内容树，不能用于定位原 HTML |
| 源表 id 与自定义节点标识 | 均未保留 | 不能依靠自动透传属性建立来源映射 |
| 输出表格行格标签 | row / cell | 已从原 HTML 的 tr / th / td 转换 |
| 输出 rowspan / colspan 属性数 | 均为 0 | 本例以空占位格展开，没有保留原始合并属性；不等于全部列位置信息消失 |

失败样例是“在返回 body 中按原 id 定位 sales-table，或获取原 rowspan/colspan”的预期：本例均无法实现。表格文字仍存在，因此它不是一次表格内容全丢失的案例。

## 源码与官方接口依据

本地 core._prepare_tree 复制并清洗 DOM，执行标签转换；main_extractor._extract 新建 Element('body')，按正文候选规则处理并追加元素；handle_table 新建 table，将行列转换为 row/cell；core 最终把 postbody 赋给 document.body。因此返回 body 是抽取结果，而不是未经修改的源节点。

[官方 Python 用法](https://trafilatura.readthedocs.io/en/latest/usage-python.html)说明 bare_extraction 返回带正文 LXML 元素的 Document；[核心 API](https://trafilatura.readthedocs.io/en/latest/corefunctions.html)没有定义原始节点来源映射字段。在线文档可能随版本变化，本地行为以本次源码哈希与测试为准。公开输出格式中的 HTML/XML 同样不能自动等同于原 HTML 片段。prune_xpath 用于排除节点，不是返回已识别正文 XPath 的接口。

## 正文区域定位建议

正文未必是单一连续子树。文章可能拆成多个相邻容器，同一 article 内又可能含广告和评论。因此应把目标建模为“正文内容块集合”，再评估其局部容器，而不是强制得到一个包住所有内容的大 div。

推荐当前阶段保持原 DOM 不变，建立节点索引和表格候选清单；在独立副本上运行 Trafilatura。使用抽取正文中的较长、较独特段落匹配源 DOM 的文本块。匹配时处理空白、实体、内联标签和 Unicode 差异，记录一对多歧义；先精确匹配，再使用字符序列或片段相似度。中文不能只依赖空格分词，不能只靠无序词袋。

以匹配锚点支持候选容器评分，综合正文覆盖、正文密度、标题关联、内容顺序、链接/导航/评论等噪声。同时比较父子候选：父节点若只多出广告和菜单，应选更小的内容范围；若两组正文分散在不同区域，可保留多个范围。不要直接取所有命中节点的最近公共祖先，该节点可能退化为 body。没有高置信度匹配时标记不确定，不能默认为整页正文。

源表归属单独判断：确认正文范围内且不在明确噪声子树的表格默认保留；与正文锚点位于同一局部章节、上下文连续的相邻表格列为候选。表格文本是否出现在 Trafilatura 输出中仅作辅助证据，不能设为必要条件，否则提取器漏掉的表永远无法补回。短文、纯表页面缺少段落锚点，需要单独的页面类型与数据表判断。

表格从原 DOM 读取，保留源节点、caption、rowspan/colspan、表头属性和邻近脚注；之后再决定是否用于当前任务。正文身份、数据表身份与任务相关性应分别记录。

## 可选的精确来源追踪

若未来需要每个抽取块都具有可靠来源，可维护受控的 Trafilatura 适配版本：在清洗前给节点分配来源标识，在删除、复制、合并和新建输出节点时维护输出块到源节点集合的映射，并覆盖回退提取分支。只在输入标签上添加 data 属性不足以保证保留，本例已经验证。

内部 BODY_XPATH 可以作为候选生成的参考，但它描述清洗树上的选择逻辑，命中候选不等于最终保留范围。仅记录首个匹配 XPath 无法覆盖子树修剪、多个来源块及回退路径。内部函数并非稳定的公开来源定位接口，采用此路线需要版本固定和相应验证。

## 局限性与下一步

- 已测量：一个合成页面的输出结构、标识与合并属性保留情况。
- 间接推断：正文锚点映射能够辅助原 DOM 定位，保留源表可减少格式转换损失；本次未验证这套方案的准确率。
- 尚未评估：真实网页正文定位、表格归属、多正文容器、短文/纯表页、动态 DOM、来源追踪适配和性能成本。

下一步先用现有冻结页面标注正文块和表格归属，对比当前全页源表补回方式与正文锚点辅助方式，分别统计正文块 Precision/Recall、表格归属 Precision/Recall、错误匹配和无法定位比例。保持部分未知结果，不通过扩成整个 body 掩盖失败。此次只新增机制验证与报告，未修改业务提取器或添加旧机制兼容层。
