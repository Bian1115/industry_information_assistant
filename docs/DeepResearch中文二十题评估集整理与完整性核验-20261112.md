# DeepResearch中文二十题评估集整理与完整性核验

日期：2026-11-12，Asia/Singapore。

本次从同一版官方题库中排除现有30道中文验证题，将剩余全部20道中文原题整理为“DeepResearch评估集”，并保存到桌面。本次未生成研究报告或调用模型裁判。

## 数据位置与配置口径

- [桌面评估集](/Users/bxy/Desktop/DeepResearch评估集/)
- [题目一览](/Users/bxy/Desktop/DeepResearch评估集/题目一览.md)
- [仓库归档](../backend/eval/data/DeepResearch中文评估集-20261112/)
- [机器可读原题](../backend/eval/data/DeepResearch中文评估集-20261112/评估题目.jsonl)
- [选题清单与文件指纹](../backend/eval/data/DeepResearch中文评估集-20261112/选题清单.json)
- [原始完整性核验结果](../backend/eval/data/DeepResearch中文评估集-20261112/完整性核验.json)

官方来源版本：`469cce54ea7f6a63c163d3d9fec879cf289ec484`。原始来源目录：`/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007`。排除清单来自`/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearch中文验证集-20261108/抽样清单.json`，并已核对仓库与桌面验证集的原题、细则和参考报告均符合冻结文件指纹。

选题方法是集合差集，即程序从50道官方中文题中去掉现有验证集的30道题，保留剩下的全部20道题；不再随机抽样，不翻译或改写题目。

## 核心指标与逐类核对

| 项目 | 已核对结果 |
| --- | --- |
| 官方中文题数 | 50 |
| 现有中文验证集 | 30 |
| 本次中文评估集 | 20 |
| 评估集唯一题号 | 20 |
| 与验证集重叠题号 | 0 |
| 原题与对应评分细则 | 20题全部一致 |
| 参考报告 | 20份 |
| 参考报告任务文本与原题差异 | 0处，详情见选题清单 |
| 桌面与仓库归档 | 逐文件一致 |

官方题号：3、5、6、7、8、11、18、19、27、28、32、33、37、39、40、43、44、48、49、50。

| 官方领域 | 题数 |
| --- | --- |
| Art & Design | 1 |
| Crime & Law | 1 |
| Fashion & Beauty | 1 |
| Finance & Business | 4 |
| Food & Dining | 1 |
| Games | 1 |
| Hardware | 1 |
| Health | 2 |
| History | 1 |
| Science & Technology | 2 |
| Social Life | 1 |
| Software | 1 |
| Software Development | 2 |
| Transportation | 1 |

评分细则与参考报告放在`评分专用材料`目录。研究服务只应读取评估原题，不得把评分材料作为生成输入。现有评分器的默认输入没有变更，后续运行应明确指定本评估集。

## 失败样例与局限

本次数据完整性核验没有失败项。没有报告生成或评分结果，因此没有新测得的研究失败案例。

已测量：题数、语言、唯一性、题号交集、文件指纹、细则对应关系和副本一致性。间接推断：这20题可作为与当前验证集不重叠的评估输入。尚未评估：完整历史实验暴露、研究完成率、报告质量、模型裁判一致性和优化收益。

“不与当前验证集重叠”不等于“从未参与历史调优”。例如既有历史五题实验包含第49题，因此不能仅凭这次划分把整组称为严格未见测试集。本次按用户要求保留全部剩余中文题，没有因历史使用而擅自删除题目。

## 结论与下一步

20题中文评估集已整理到桌面，并在仓库保留可追溯副本。后续可从这份固定清单启动独立评估；若需要严格未见测试，应先审查历史使用记录，再决定是否补充新的题源。
