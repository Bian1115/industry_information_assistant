# DeepResearchBench参考报告30题评分一致性与分歧分析

评估日期：2026-10-07；最近刷新：2026-10-08T09:38:52.800499+08:00。

状态：**Qwen30题各两次评分完成；Claude及双裁判仍按已完成子集统计**。

固定种子20261007，从官方100题总体均匀随机抽30题，保留首次抽样。ID按抽样顺序：30, 46, 10, 38, 78, 80, 91, 73, 13, 36, 17, 65, 72, 14, 87, 61, 1, 81, 25, 16, 70, 22, 23, 15, 41, 26, 47, 62, 31, 84。

样本：30份清洗后参考报告（18中文、12英文），758条官方标准。来源为[官方DeepResearch Bench](https://github.com/Ayanami0730/deep_research_bench)，提交469cce54ea7f6a63c163d3d9fec879cf289ec484；原始数据指纹见随机抽样清单。

助手第一轮由当前对话模型独立给出逐条分数，先于裁判结果冻结。阅读采用结构逐段浏览与按标准定位补读，较长段落有截取，未做全文逐字或外部事实核验；它是模型评分参照，并非人类金标。双裁判接收完整官方标准和完整正文，不能看到助手评分。

## 数据、模型与汇总口径

Qwen3.8-max：temperature=0.6，enable_thinking=true，reasoning_effort=xhigh，seed=1234，未设置输出token上限；初始调度并发2，本次续跑调度并发6（详见续跑记录）。Claude-opus-5：服务默认采样，输出上限12000，并发1。每模型每题两次，目标120份有效响应。保留当前系统虚构评分锚点，改为完整官方逐题细则及原生JSON完整性校验。

格式补充（同一运行）：Claude出现额外空元数据或JSON引号未转义。沿用当前系统的JSON语法修复，同时要求原始criterion_id与数字score逐项齐全且与解析结果完全一致；完整流结束、全部官方细则和理由必须通过校验。额外null元数据可删除，数字分数不得补造或改变。每个重复编号取首个通过此规则的完整响应，原始失败包及格式核查记录另存。Qwen仍要求严格原生JSON。

所有0–10分按官方细则和维度权重计算。双裁判先分别均两次，再等权平均；不将报告与其自身比较求0.5相对分。重复编号不同，稳定性包含编号提示变化。

已保存有效响应：114/120；助手：30/30；完整Qwen题：30/30；完整Claude题：26/30；完整双裁判题：26/30。

## 已测量一致性

MAE/RMSE衡量0–10分距离；偏差=裁判−助手；±1为绝对差≤1。Pearson/Spearman衡量线性/排序关联，CCC同时考虑尺度和位置一致性。总分以题为样本，维度和细则不可作为额外独立题目。总分MAE/偏差95%区间按题bootstrap4000次，种子20261007。

| 比较 | 完整题数 | MAE | RMSE | 偏差 | ±1比例 | Pearson | Spearman | CCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-max vs 助手 | 30 | 0.341 | 0.470 | 0.282 | 93.3% | 0.843 | 0.790 | 0.744 |
| Claude-opus-5 vs 助手 | 26 | 0.590 | 0.687 | -0.566 | 80.8% | 0.884 | 0.790 | 0.688 |
| 双裁判均分 vs 助手 | 26 | 0.286 | 0.346 | -0.123 | 100.0% | 0.890 | 0.842 | 0.874 |

## 当前完整双裁判子集的分歧维度（N=26，不可代表30题）

| 维度 | MAE | 偏差 | ±1比例 |
|---|---:|---:|---:|
| 洞察力 | 0.369 | -0.145 | 100.0% |
| 全面性 | 0.362 | -0.215 | 100.0% |
| 指令遵循 | 0.270 | +0.022 | 100.0% |
| 可读性 | 0.258 | -0.153 | 100.0% |

## 当前双裁判子集的最大逐条分歧

| 题号 | 细则 | 助手 | 双裁判 | 差值 | 对总分贡献 |
|---|---|---:|---:|---:|---:|
| 14 | 突破潜力评估逻辑的深度与合理性 | 2.00 | 3.88 | +1.88 | +0.225 |
| 14 | 未来理论/技术预测的 обоснованность (Groundedness) 与前瞻性 | 1.00 | 3.12 | +2.12 | +0.213 |
| 10 | 临界点分析与解读的洞察力 | 4.50 | 7.38 | +2.88 | +0.177 |
| 10 | 商业化临界点量化方法的严谨性与合理性 | 4.00 | 6.12 | +2.12 | +0.174 |
| 80 | Logical Coherence and Evidentiary Support for the Manga-Language-Identity Nexus | 4.50 | 6.62 | +2.12 | +0.149 |
| 13 | 对AI提升机制与效果的深度论证 | 6.00 | 7.88 | +1.88 | +0.146 |
| 16 | 输入信号类型的详尽描述 | 9.00 | 7.25 | -1.75 | -0.131 |
| 73 | Derivation of Novel Pedagogical Applications from Frontline Experience | 6.50 | 5.12 | -1.38 | -0.124 |
| 46 | 国家干预措施（政策、资金、导向）有效性与内在逻辑的前瞻性分析 | 6.50 | 7.75 | +1.25 | +0.122 |
| 14 | 跨维度信息的综合运用与模式识别 | 6.00 | 4.50 | -1.50 | -0.120 |
| 16 | 算法策略-信号-准确率关联性深度分析 | 8.00 | 6.50 | -1.50 | -0.112 |
| 80 | In-depth Coverage of the Investigated Link between Manga Consumption and Language Use | 4.50 | 6.12 | +1.62 | +0.110 |
| 16 | 明确呈现所识别策略的输入信号 | 9.00 | 7.25 | -1.75 | -0.108 |
| 10 | 评估体系针对“量化商业化临界点”目标的聚焦程度 | 5.00 | 7.25 | +2.25 | +0.107 |
| 14 | 关键成功要素的识别与阐释 | 7.00 | 5.25 | -1.75 | -0.105 |
| 73 | Depth of Critical Reflection on Practical Implementation (Challenges & Solutions) | 7.00 | 4.75 | -2.25 | -0.101 |
| 80 | Plausibility and Theoretical Grounding of Influence Mechanisms | 7.00 | 5.88 | -1.12 | -0.098 |
| 16 | 明确识别并呈现“当前最好”的算法策略 | 5.00 | 6.25 | +1.25 | +0.097 |
| 81 | Depth of Analysis of Reinterpretation Mechanisms | 8.50 | 7.50 | -1.00 | -0.095 |
| 46 | 未来10年国家政策措施覆盖面与具体性 | 6.50 | 7.88 | +1.38 | +0.093 |

## Qwen与助手的分歧维度

| 维度 | 完整题数 | MAE | 偏差 | ±1比例 |
|---|---:|---:|---:|---:|
| 洞察力 | 30 | 0.455 | +0.430 | 86.7% |
| 指令遵循 | 30 | 0.400 | +0.372 | 90.0% |
| 全面性 | 30 | 0.365 | +0.125 | 90.0% |
| 可读性 | 30 | 0.239 | +0.023 | 100.0% |

## Qwen细则级一致性

已完成题目的758条细则：MAE=0.517，偏差=+0.200；完全同分27.8%，差值≤0.5分72.2%，差值≤1分90.8%，差值≥2分4.0%。各细则等权统计，采用Qwen两次均分；758条不视为758份独立报告，也不能代替官方权重总分。

Qwen总分MAE的95%题目bootstrap区间：[0.231, 0.467]；偏差区间：[+0.152, +0.425]。

## Qwen最大逐条分歧（按对总分贡献绝对值排序）

| 题号/页面 | 细则 | 助手 | Qwen均分 | 差值 | 对总分贡献 |
|---|---|---:|---:|---:|---:|
| [题14](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题14.html) | 突破潜力评估逻辑的深度与合理性 | 2.00 | 6.00 | +4.00 | +0.480 |
| [题14](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题14.html) | 未来理论/技术预测的 обоснованность (Groundedness) 与前瞻性 | 1.00 | 5.00 | +4.00 | +0.400 |
| [题10](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题10.html) | 商业化临界点量化方法的严谨性与合理性 | 4.00 | 7.00 | +3.00 | +0.246 |
| [题10](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题10.html) | 临界点分析与解读的洞察力 | 4.50 | 8.00 | +3.50 | +0.215 |
| [题13](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题13.html) | 对AI提升机制与效果的深度论证 | 6.00 | 8.50 | +2.50 | +0.195 |
| [题46](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题46.html) | 国家干预措施（政策、资金、导向）有效性与内在逻辑的前瞻性分析 | 6.50 | 8.50 | +2.00 | +0.195 |
| [题80](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题80.html) | Logical Coherence and Evidentiary Support for the Manga-Language-Identity Nexus | 4.50 | 7.25 | +2.75 | +0.192 |
| [题61](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题61.html) | Novelty, Strength, and Justification of Established Inter-Domain Correlations | 4.50 | 6.25 | +1.75 | +0.189 |
| [题16](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题16.html) | 明确识别并呈现“当前最好”的算法策略 | 5.00 | 7.25 | +2.25 | +0.174 |
| [题80](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题80.html) | In-depth Coverage of the Investigated Link between Manga Consumption and Language Use | 4.50 | 7.00 | +2.50 | +0.169 |
| [题23](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题23.html) | 获奖课程“资料”类型与获取途径明确性 | 5.00 | 7.00 | +2.00 | +0.165 |
| [题16](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题16.html) | 准确率指标的呈现与说明 | 4.50 | 6.50 | +2.00 | +0.150 |
| [题10](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题10.html) | 评估体系针对“量化商业化临界点”目标的聚焦程度 | 5.00 | 8.00 | +3.00 | +0.143 |
| [题46](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题46.html) | 未来10年国家政策措施覆盖面与具体性 | 6.50 | 8.50 | +2.00 | +0.135 |
| [题46](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题46.html) | 分析逻辑的严谨性与推演合理性 | 5.50 | 7.75 | +2.25 | +0.132 |
| [题80](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题80.html) | Directly Addressing the Link Between Shoujo Manga Consumption and Specified Linguistic Changes | 5.50 | 8.00 | +2.50 | +0.130 |
| [题16](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题16.html) | 明确呈现所识别策略的准确率/性能 | 4.50 | 6.50 | +2.00 | +0.124 |
| [题61](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题61.html) | Addressing the Establishment of Specified Direct Correlations | 4.50 | 7.25 | +2.75 | +0.110 |
| [题14](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题14.html) | 关键数学理论或应用技术预测的执行 | 1.00 | 4.25 | +3.25 | +0.107 |
| [题41](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题41.html) | 预测逻辑的严谨性与论证质量 (Predictive Logic & Justification Rigor) | 6.50 | 7.50 | +1.00 | +0.098 |

## Qwen重复评分稳定性

每題两次总分之间MAE=0.203，CCC=0.886；重复编号提示不同，不能将这两次视作独立的60题。

## Qwen分歧归因复核

Qwen30题已完整完成，以下数字采用每题两次均分与此前冻结的助手首轮评分比较；助手不是人类金标。

**已测量：**总分MAE=0.341，RMSE=0.470，偏差=+0.282，Pearson=0.843，Spearman=0.790，CCC=0.744。28/30题总分差≤1分，23/30题差≤0.5分；24题Qwen更高，6题更低。Qwen均分8.180，助手7.897。

差异主要集中在洞察力和指令遵循：维度MAE分别0.455、0.400，平均偏差分别+0.430、+0.372；全面性MAE0.365，可读性MAE0.239且偏差仅+0.023。实际定量结果、指定预测、可操作资料和实证证据链，是较大分歧的共同位置。

**复核解释：**下面依据官方细则、两次Qwen理由以及助手已保存的维度说明归因。助手未保存每条独立理由，因此涉及助手扣分动机的解释是复核推断；不是第三位独立裁判，也未外部核查事实。

**1. 框架和方向能否替代实际产物。** 主要分歧是方法框架与实际量化结果的边界。商业化临界点量化方法项助手4分、Qwen均分7分；临界点分析项4.5分对8分。Qwen两次均明确承认缺少公式、参数、示例计算或量化结果，却仍对概念框架、触发条件给较高部分分数。官方“临界点分析”解释明确包含量化结果，此处应重点校准实际产物的扣分尺度。 正文未完成且缺少明确未来5–10年团队评估及预测。突破潜力逻辑项助手2分、Qwen6分；预测洞察项1分对5分。两条各差4分，对本题总分贡献分别为+0.480、+0.400。Qwen认可已有前沿方向与评估要素，助手更重视明确预测与评估的交付。双方都看到缺失，主要分歧是给多少部分分数。 获奖课程资料获取明确性项助手5分、Qwen7分；资料响应项5.5分对7.25分。Qwen承认缺少具体链接、文件名、检索路径或可立即下载的资料，但认可资料类型与潜在渠道。分歧在于可操作资料是否已交付，以及渠道概述应拿多少部分分数。

**2. 解释连贯性与证据强度的权衡。** 漫画—语言—身份关联的论证项助手4.5分、Qwen7.25分；两次Qwen理由都承认没有直接消费数据。Qwen更重视推理连贯性和谨慎的“影响有限”结论，助手更强调消费变量、目标年龄群体和语言结果的实证链。相关性是否实际建立，是主要分歧。 AI改善时幅修正的机制与效果论证项助手6分、Qwen8.5分。Qwen强调CMS-PPS、MRPC、PET改善案例以及非线性特征机制；助手维度说明强调混入非AI方法、输入条件推测和可比性不足。此处是证据强度判断差异，不能在没有外部核验时判定谁更准确。 跨域相关强度项助手4.5分、Qwen6.25分；直接相关目标响应项4.5分对7.25分。两次Qwen均承认没有充分统计验证或价格—规格配对数据，但认可环境—生物—市场框架。这里是逻辑联系与实证直接相关的边界。对渔业经济及高质量资源覆盖，Qwen反而比助手低1.75分，显示差异会随细则方向变化。

**3. 标准粒度的不同理解，双方都有偏严或偏宽的例子。** 定量建模“视角的体现”项助手5分、Qwen8.25分。官方解释明确允许通过讨论模型、建模思路、挑战或潜力体现视角，未要求该项必须交付实际求解模型。复核推断：助手可能把实际建模缺口过多投射到仅考核方法视角的细则。此处不能简单归为Qwen过松；第一轮冻结分数仍保留用于独立比较。 图表有效性项助手6分、Qwen两次都2分。官方解释列举柱状图、饼图、趋势图；Qwen严格扣除图形缺失，助手维度理由强调表格与层次易读。复核推断：助手可能把表格呈现也算进了该项部分分数，评分粒度需要校准。此外，官方query与参考报告/细则的“9阶层”要求不完全一致，原始差异已保留。 新手教师可用进一步资源项助手6分、Qwen3.75分。Qwen指出理论人物或组织名称不等于可访问的书目、实践指南、工具和社群清单。助手也承认资源定位较薄，但扣分较少。这是Qwen更严格的反例。 高质量期刊来源限定项助手3分、Qwen3.5分，两者接近。该一致案例说明Qwen能够对明确来源限制扣分；不能把其整体正偏差概括成所有标准都宽松。来源平台与期刊身份需另作外部核验，本轮未开展事实审计。

**不能把全部差异归为稳定偏差。**Qwen自身两次总分MAE=0.203，CCC=0.886，波动不可忽略；这两次编号提示不同且共享固定seed，不是60份独立报告。均分比较仍有正偏差，按题bootstrap95%区间[+0.152,+0.425]。

**相关性敏感性。**事后去掉最大总分分歧第14题，N=29时Pearson从0.843降为0.738、CCC从0.744降为0.642；主要结果仍以完整30题为准。这说明少数低分报告影响相关性，不能只用Pearson宣称尺度完全一致。

**引文定位核查。**1,516段Qwen引文中1,307段在正文连续匹配，207段不匹配、2段为空。207段中73段在仅去掉Markdown加粗/代码标记及空白后匹配，余134段尚待人工复核；有引文来自评分细则名称、混用文字或改写的样例。页面已逐条标记未匹配，不据此认定事实错误，也不改动原始评分。

**双裁判边界。**本次已补齐Qwen全部60份；截至Qwen核查时（2026-10-07 21:04），Claude15份有效响应、7题完整双裁判。双裁判表只代表这7题子集，不能作为30题双裁判结论。Claude仍在运行，综合进度会随有效结果更新。

**后续校准建议。**增加“有框架但无量化结果”“只有方向描述但没有指定预测”“视角讨论已满足该细则”“表格与指定图形的区别”等评分锚点，并在新一轮以逐条独立理由及人工抽检验证尺度。

## 2026-10-07 Qwen续跑记录

本轮续跑开始：2026-10-07T20:16:59.735113+08:00；最近更新：2026-10-07T21:03:40.996235+08:00；状态：完成。仅续跑Qwen，不重抽题、不调整助手评分，不向裁判提供助手结果。模型、提示词、输入、评分尺度和聚合方式均沿用已冻结配置。

执行调度调整：冻结配置原并发2；本轮续跑并发6。首批7份有效评分复用；并发提高时尚未完成的流不计入评分，按原请求重新生成。这是吞吐调度变更，采样参数和评分口径不变；续跑历史与执行配置另存。

[续跑状态](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/Qwen续跑状态.json) · [续跑历史](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/Qwen续跑历史.json) · [调度与冻结参数](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/Qwen续跑执行配置.json) · [续跑日志](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/Qwen续跑日志.txt) · [本次有效评分](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/Qwen续跑评分.json)

## 逐题第一轮结果

| 题号/页面 | 语言 | 助手总分 | 全面性 | 洞察力 | 指令遵循 | 可读性 |
|---|---|---:|---:|---:|---:|---:|
| [题30](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题30.html) | zh | 8.15 | 8.10 | 7.78 | 8.57 | 8.35 |
| [题46](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题46.html) | zh | 7.39 | 7.40 | 6.97 | 7.80 | 7.95 |
| [题10](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题10.html) | zh | 7.17 | 8.15 | 6.45 | 7.40 | 7.10 |
| [题38](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题38.html) | zh | 8.02 | 8.22 | 7.70 | 8.03 | 8.40 |
| [题78](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题78.html) | en | 8.56 | 8.92 | 8.05 | 8.80 | 8.47 |
| [题80](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题80.html) | en | 6.91 | 5.95 | 6.72 | 7.50 | 8.33 |
| [题91](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题91.html) | en | 8.17 | 7.97 | 7.65 | 8.55 | 8.22 |
| [题73](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题73.html) | en | 7.84 | 8.10 | 6.83 | 8.45 | 8.30 |
| [题13](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题13.html) | zh | 7.74 | 8.07 | 7.33 | 7.88 | 8.03 |
| [题36](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题36.html) | zh | 8.53 | 8.62 | 8.35 | 8.88 | 8.45 |
| [题17](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题17.html) | zh | 8.44 | 8.57 | 8.18 | 8.78 | 8.45 |
| [题65](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题65.html) | en | 7.79 | 8.20 | 7.08 | 8.47 | 7.78 |
| [题72](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题72.html) | en | 8.13 | 8.50 | 8.03 | 7.75 | 8.25 |
| [题14](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题14.html) | zh | 5.16 | 7.50 | 3.65 | 4.33 | 6.60 |
| [题87](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题87.html) | en | 8.23 | 8.43 | 7.80 | 8.55 | 8.68 |
| [题61](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题61.html) | en | 7.21 | 7.62 | 6.65 | 6.95 | 8.12 |
| [题1](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题1.html) | zh | 7.52 | 7.58 | 7.80 | 6.80 | 7.72 |
| [题81](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题81.html) | en | 8.43 | 8.53 | 8.18 | 8.74 | 8.40 |
| [题25](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题25.html) | zh | 8.27 | 8.47 | 7.95 | 8.38 | 8.62 |
| [题16](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题16.html) | zh | 7.11 | 7.08 | 7.70 | 6.38 | 7.78 |
| [题70](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题70.html) | en | 8.65 | 8.75 | 8.45 | 8.80 | 8.62 |
| [题22](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题22.html) | zh | 8.30 | 8.52 | 7.88 | 8.70 | 8.30 |
| [题23](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题23.html) | zh | 7.54 | 6.92 | 7.72 | 7.75 | 8.28 |
| [题15](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题15.html) | zh | 8.03 | 8.07 | 7.53 | 8.70 | 8.38 |
| [题41](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题41.html) | zh | 7.99 | 8.30 | 7.20 | 8.65 | 8.70 |
| [题26](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题26.html) | zh | 7.88 | 7.85 | 7.53 | 8.28 | 8.22 |
| [题47](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题47.html) | zh | 8.20 | 8.68 | 7.42 | 9.00 | 8.45 |
| [题62](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题62.html) | en | 8.46 | 8.85 | 8.07 | 8.68 | 8.45 |
| [题31](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题31.html) | zh | 8.57 | 8.78 | 7.88 | 9.00 | 8.58 |
| [题84](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/题84.html) | en | 8.48 | 8.75 | 8.10 | 8.80 | 8.40 |

## 失败样例与第一轮观察

第14题的参考正文保留摘要待撰写占位，末尾句中停止，尚未完成未来5–10年突破团队和技术预测。第72题期刊限定与MPRA、Diva Portal等未证明期刊身份来源有冲突。第10题的临界点及第26题的定量模型主要为框架，实际求解不足。第16题具体算法与准确率的可比对应不足。详情以逐题官方标准及页面第一轮理由为准。

题1存在原始数据内部不一致：query是“所有阶层”，细则与参考报告保留“9阶层”要求。页面同时展示两份提示词，保留官方细则评分，指令遵循分数需结合该冲突阅读。

## 局限性与边界

助手评分不是独立多人金标；助手与双裁判阅读流程不同。没有外部检索核实报告事实；所谓证据不足和待核查风险不等于已证明事实错误。历史参考报告按题目及报告内时间锚点评估，不因2026年评估日期机械扣旧数据分。30题来自全语言总体，语言/主题分组属于探索，不应以小子集下强结论。缺失、超时与账户错误不作为低分；引文自动连续匹配只核查引文定位，不证明论断真实。

## 原始记录与可复现文件

- [30题HTML总览](/Users/bxy/projects/DeepResearch医疗研报助手/多agent评估/DeepResearchBench参考报告30题评分页面-20261007/总览.html)

- [冻结输入](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/冻结输入.json)

- [随机抽样与指纹](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/随机抽样清单.json)

- [配置及助手冻结指纹](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/评估配置.json)

- [双裁判原始响应目录](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/裁判原始记录)

- [机器可读一致性汇总](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/一致性汇总.json)

- [Qwen最终格式与权重复算核查](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/Qwen最终核查.json)

- [30页完整性核查](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/Qwen页面最终核查.json)

- [Qwen分歧归因与来源指纹](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/Qwen分歧归因复核.json)

- [综合运行状态](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/双裁判综合状态.json)

- [原始Claude进程状态（不含独立Qwen续跑）](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/运行状态.json)

## 结论与下一步

Qwen已完成30题各两次评分，可报告Qwen与助手的一致性；完整双裁判结论仍待Claude补齐。

单独续跑Qwen（复用匹配缓存，不并行启动第二个Qwen续跑）：
```sh
cd "/Users/bxy/projects/industry_information_assistant"
backend/.venv/bin/python backend/eval/scripts/续跑DeepResearchBench的Qwen评分.py
backend/.venv/bin/python backend/eval/scripts/生成DeepResearchBench评分页面与分析.py
```