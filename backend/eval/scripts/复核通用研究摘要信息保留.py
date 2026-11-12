"""Codex逐项复核固定144项：修正裁判把清单原句当摘要、隐含补全等问题。

本脚本的判定表由Codex阅读五份摘要与冻结清单后填写，不是独立人工金标。
不更改任何压缩前信息点；引句由程序从摘要真实段落提取并保存字符位置。
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

# 每行：编号|判定|逐项理由|用于选择摘要真实段落的原文片段（多个以;;分隔）。
REVIEWS = {
16: """
K01|preserved|研究目标及领先算法要求保留。|当前用户目标
K02|preserved|输入信号与准确率两个评估维度保留。|当前用户目标
K03|preserved|静态下肢任务、融合方法、精确率/召回率及0.92阈值保留。|电容-惯性传感器融合方法
K04|preserved|同一融合方法的动作转换任务与0.86阈值保留。|电容-惯性传感器融合方法
K05|preserved|CNN超分辨率、原方法基准和两项提升比例保留。|47.86%
K06|preserved|超分辨率后的进一步提升60.38%保留。|60.38%
K07|partial|保留三数据集及显著优于先前方法，遗漏离散与连续两种任务范围。|RobHAR系统
K08|preserved|宏观有效、微观打字受限的能力边界保留；锻炼只是非必要举例。|商用传感器对微尺度动作
K09|preserved|数据集、自适应多普勒分辨率和宏微观活动能力保留。|mmDoppler数据集
K10|preserved|三台TI雷达、三轴部署、室内实时多人跟踪与跌倒检测保留。|三台德州仪器
K13|partial|保留视觉测量及性能数字，遗漏精密加工零件形位公差这一适用任务，不能由视觉测量泛称补全。|47.86%
K17|partial|保留工业近场分类及器件名称，遗漏反射/遮挡原理及与远场波动的区别。|在工业近场传感方面
K18|preserved|西安交大域名和可识别的non-periodic-lower-limb论文URL保留，可定位该融合方法。|non-periodic-lower-limb
K19|partial|保留EMT URL及视觉结果，但没有明确把该链接绑定到视觉测量论文标题或数值。|emt.etmchina.com
K20|partial|RobHAR与两个arXiv链接都存在，但没有明确2405.01882对应RobHAR，不能用清单补关联。|arxiv.org/abs/2405.01882
K21|partial|mmDoppler与两个arXiv链接都存在，但没有明确2407.21125对应mmDoppler。|arxiv.org/abs/2407.21125
K22|partial|保留X-MOL URL和三雷达描述，未明确绑定来源与该项结果。|x-mol.com/paper
K27|preserved|霍尔传感器线性度、可选值及温度范围完整保留。|霍尔效应磁性传感器
K28|partial|保留25–256 PPR，遗漏光学编码器自身的−40°C至+75°C范围；不能套用霍尔温度。|Bourns非接触式光学编码器
K29|preserved|电感传感器、温度范围、4 mm距离与±0.2 mm偏差保留。|非线性多项式补偿算法
K30|partial|保留红外模块，遗漏安装于触摸面板上方的部署条件。|NEXTY Electronics提供
K33|preserved|Bourns PDF的完整URL保留，域名和文件名可明确对应传感器来源。|Bourns_Position_Sensors_Rotary_Controls_SF.pdf
K34|preserved|保留西安交大温度补偿论文可识别URL，与正文方法能够对应。|a-non-linear-temperature-compensation
K44|preserved|三条物理媒介路径及已完成章节身份保留。|当前领先算法按物理媒介
""",
31: """
K01|preserved|国内外分别总结、代表性国家要求保留，未把选择国家变成用户硬性清单。|用户要求完成选题
K02|preserved|分类自成体系及可选的分类例子保留，未强制要求双维分类。|用户要求完成选题
K03|preserved|科技、教育理念、未来趋势及中国详写要求保留。|用户要求完成选题
K04|partial|2018年5354家保留，但遗漏2019年市场分析报告这一特定来源。|2018年中国博物馆总数
K05|preserved|2017年、国家一级专业等级及130家保留。|截至2017年共有130家
K06|preserved|2018年和国有占比88.6%保留。|2018年中国博物馆总数
K07|partial|近5000家保留，遗漏综合性与专业性范围；现有一词也未提示统计年份未知。|美国现有博物馆近5000家
K08|partial|伦敦团体及250家保留，公共机构身份遗漏。|代表首都250家博物馆
K09|preserved|拨款方、项目及10万英镑完整保留；人民币近似换算为非必要重复数值。|拨款10万英镑
K10|preserved|法德日缺乏直接材料、行业共识和未经验证身份明确保留。|属于上下文未经验证的补充信息
K12|partial|保留条例公平对待，遗漏现实中的补贴、职称、鉴定和借展障碍。|国家在设立条件
K14|partial|福州市政府域名保留，但遗漏教育、研究、欣赏和非营利组织的定义。|fuzhou.gov.cn
K15|preserved|国有与非国有的两种登记路径保留。|依事业单位登记管理法规
K16|partial|仅保留国家文物局域名，遗漏名录印发日期、各省提供数据及备案来源关系。|ncha.gov.cn
K17|partial|查字典域名存在，但未明确美国实践来自该观察性报道，也未保留观察材料身份。|chazidian.com
K19|preserved|双维分类被放在已完成章节描述内，区别于用户可选要求。|并按行政级别
K20|preserved|美英材料与法德日未验证补充信息的区别保留。|对于法国、德国、日本的描述
K21|partial|保留档案和研究资源体系，遗漏具体索引以及观察者报道的范围限制。|严格统一的藏品档案管理
K22|partial|9周至一学年、大学生研究生保留，遗漏艺术史与保护方向的留居研究员职位。|九周至一学年的实习
K24|missing|建川沉浸场景与观复新媒体导览两个案例均未出现在摘要。|
K25|missing|国家级馆专职教育部及K12至成人课程概述未保留。|
K26|missing|东部与中西部部分市级馆差异及地域限定未保留。|
K27|partial|新技术、志愿者和实习生改革保留，遗漏通过公共项目向公众传递创新研究理念。|引进新技术
K28|partial|知识双向流动保留，但双方具体提供的专业知识、藏品、社区及教学资源遗漏。|知识双向流动
K31|missing|没有保留美国馆数缺统计年份的明确限制，不能以未写年份视为保留该警示。|
K32|preserved|资产来源、登记路径及国有内部两类管理归属完整保留。|可细分为文物行政部门
K33|partial|88.6%和观研域名存在，但2019年报告及该比例的明确归属遗漏。|其中国有博物馆占88.6%;;chinabaogao.com
K34|missing|补充县级馆实地调研数据的未完成建议遗漏。|
K35|partial|保留三国直接素材不足，遗漏补充实施与评估数据的明确后续建议。|对于法国、德国、日本的描述
""",
35: """
K01|preserved|政府操作方案、全环节成本及雨污协同要求保留。|用户目标是探讨
K02|partial|2018年100+2规模保留，未明确该规模尚未充分发挥环境效益这一限定。|形成“100+2”基础规模
K03|preserved|80克和120克两种COD假设、2018年及67%和45%保留，未混成同一口径。|按人均COD当量80克计
K04|partial|63%、80%、24%、120%及溢流保留，但遗漏全国470多个厂的调查样本范围。|调研显示63%
K05|partial|南方多雨与CSOs严重保留，遗漏河水倒灌及I/I加剧的具体因果联系。|合流制溢流（CSOs）;;控制管道入流/入渗
K06|preserved|投资、土地、电耗和药剂四项规模数值与单位保留。|污水处理已投资约6000亿元
K07|preserved|生活区和工矿、雨污管道及总投资约70%保留。|生活区与工矿企业
K08|partial|152座、53.8万吨/日及90%覆盖率保留，遗漏小规模导致单位造价上升的规律。|东莞市已建成
K09|partial|154.6万元和97%保留，遗漏73万元年度预算及合同价与总投资口径区别。|湘潭高新区2021年
K10|partial|仅在待办中提到湘潭工地总代表制度，遗漏牵头单位、责任人及实际全过程监督做法。|“工地总代表”制度
K11|partial|渣砂与过曝气问题保留，遗漏达标前提及曝气优化对外碳源和氮磷的具体关系。|预处理渣砂去除效果差
K12|partial|内控、责任和精细化管理问题保留，遗漏材料费占比最大、采购时点及价格波动管理。|成本管理失效的核心症结
K13|partial|机制整合降本的总体方向保留，遗漏分流减负荷、延长寿命与维护成本关系。|机制性整合是结构性降本增效路径
K14|partial|清污分流和智能调蓄保留，遗漏晴天/雨天及初期/中后期雨水具体去向。|“清污分流”作为新型系统
K15|preserved|两类优先序与各项措施完整保留。|“雷厉风行”类措施
K16|partial|错混接和两种体制限制保留，遗漏分流减小水质水量波动的优点及错接流向。|现有体制（合流/分流）
K17|partial|保留超负荷和溢流风险，遗漏截流式合流制晴雨分工与截流井的运行条件。|导致管网频繁溢流
K18|partial|保留分散设施名称与规模，遗漏应急用途、稳定达标难及不宜永久化的核心限制。|东莞市已建成
K19|partial|保留SWMM及英加名字，遗漏STORM和英加已应用于协同模拟规划的陈述关系。|细化雨污协同模型
K20|partial|王洪臣姓名与基础规模都在，遗漏E20、2020年第十八届论坛的来源身份。|王洪臣提出的提质增效优先序
K21|partial|湘潭地方政府网站及2021年项目存在，未保留项目支出绩效评价报告身份。|湘潭高新区2021年;;湘潭高新技术产业开发区网站
K22|partial|东莞生态环境局名称存在，遗漏暂缓建站建议及政协提案办理结果的来源背景。|东莞市生态环境局网站
K23|preserved|前两章完成状态和首章主题保留；再次传递方式不是核心任务状态。|当前任务处于分章节撰写阶段
K25|preserved|统一目标、分摊规则、应急联动作为已有分析策略呈现，未列为新增用户要求。|缺乏跨雨污系统
K26|missing|动态规划、遗传算法及基建费用降低10%以上的结果全部遗漏。|
K27|partial|保留总体降本方向，遗漏分流能耗物耗路径及雨水资源利用缓解用水压力。|机制性整合是结构性降本增效路径
K30|missing|分流制流速防沉淀、改造灵活及广泛应用的这组信息遗漏。|
K33|partial|保留分流技术名称及共同治理目标，但未明确雨污分开收集处理、减冲击和保护水资源的定义链。|现有体制（合流/分流）
K35|preserved|雷厉风行与久久为功的两类措施分别列出，短长期行动区分保留。|“雷厉风行”类措施
""",
46: """
K01|preserved|房地产可持续发展动力的目标保留。|用户目标是研究
K02|preserved|未来10年及政策、资金、导向三个维度保留。|用户目标是研究
K03|missing|基础产业、国民经济增长点及对经济社会可持续发展的影响未保留。|
K04|preserved|代际利益、资源有效利用、长期增长和结构优化定义保留。|房地产业可持续发展被定义为
K05|preserved|技术进步及绿色、智能、装配式建筑例子保留。|技术进步被视为核心动力之一
K06|partial|文件、日期和发文机关保留，遗漏十四五和2035年纲要这一制定依据。|2022年1月19日印发
K07|preserved|300吨/立方米作为历史输入值保留，并显式标注单位存疑需核实。|单位表述存疑
K08|partial|常州、2023年8月和征求意见稿状态保留，遗漏该地方文件具体目标与节能降碳要求。|常州市于2023年8月
K09|preserved|时间、项目、首个身份与两项技术体系保留。|2023年10月
K10|preserved|绿色金融鼓励及相关产业支持保留，上文明确列出智能、绿色及装配式建筑。|金融机构被鼓励
K11|preserved|会议年份、高质量发展和市场+保障双轨体系保留。|2025年中央经济工作会议
K12|preserved|稳住楼市年度任务及政策基调转变保留。|政策基调从
K13|partial|三类需求政策保留，遗漏一线郊区、公积金额度等具体操作对象与手段。|需求端政策包括
K14|preserved|供给侧增存量、收储保障房和好房子标准保留。|供给端政策包括
K15|preserved|2025年30%到2030年60%的目标数字保留；分母另由K22单独计分。|绿色建筑占比目标
K16|preserved|预测身份、2026–2030年、新建商品住宅及年均7亿–8亿平方米保留。|预计“十五五”期间
K17|preserved|REITs扩容领域、权益融资和存量盘活方向保留。|REITs正向商业地产
K18|preserved|增量向城市更新、盘活、好房子和租购并举的生态转变保留。|房地产行业已超越
K19|partial|城市层级和核心区分化保留，遗漏优质项目、改善性需求支撑与低线以价换量机制。|当前市场呈现显著分化格局
K20|preserved|原单位、作者按惯例解释及需核实原PDF均保留，没有声称本轮已验证。|写作过程中发现住建部规划中;;需进一步核实原始文件PDF
K21|partial|短期局部试点和时间城市不明保留，遗漏宏观承受力与地方财政替代方案的具体条件。|房地产税试点虽被列为
K22|missing|绿色建筑比例按新建住宅项目统计的分母限定遗漏。|
K23|missing|销售面积预测为全国总量的明确范围遗漏，不能仅靠中国行业标题补全。|
K24|partial|保留中国政府网发布页和PDF存在，但未保留附件PDF本身的URL。|gov.cn/zhengce/zhengceku
K25|preserved|云南转载来源与建市〔2022〕11号完整保留。|yn.gov.cn/ztgg
K26|preserved|北京转载页面及其现行有效状态保留。|beijing.gov.cn/zhengce
K27|partial|市场全景报告名、来源及产业演进内容保留，遗漏城市高质量发展和区域协调战略的直接作用关系。|《2026-2030年中国房地产行业市场全景调研与发展前景预测报告》
K28|partial|中研来源和主题文章URL保留，但没有完整主题报告名称及0.85可信度。|20251219/162100244.shtml
K29|preserved|三重协同框架及既有章节结论身份保留。|制度约束、技术进步与责任内化三重协同
K32|preserved|五类机制及首章组织身份保留，表格排版不作为必要语义。|竞争机制、创新机制
K34|partial|好房子标准及中国行业研究网来源保留，遗漏健康、耐久等四维具体内容。|推动“好房子”建设标准落地;;来源：中国行业研究网
""",
49: """
K01|preserved|全球、20–30岁女性及两项服务共同需求比重要求保留。|用户要求调研全球范围
K02|preserved|未来服务结合的可能性问题保留。|用户要求调研全球范围
K03|preserved|中国三年龄组和比例完整保留，另有个体重叠率缺口说明，未把73%直接写成全球共同需求比重。|19–23岁;;目前尚无任何一手数据
K04|preserved|BOS、2019年、26–55岁和两项80%均保留。|BOS）2019年调查
K05|preserved|2022年、女性约61%保留。|女性整体占比约61%
K06|preserved|成人正畸、美观目的与86.72%保留。|86.72%
K07|partial|三个分量表与显著相关保留，但明确统计阈值P<0.05遗漏。|三个分量表得分
K08|preserved|16岁以上、自尊/自信及五大因素之一保留。|在16岁以上人群
K09|partial|性别比较与青春期媒体敏感性保留，遗漏14–19岁匈牙利学生的样本范围。|青春期女性因媒体
K10|preserved|照顾者饮食信息经羞耻成分间接影响医美接受度的路径保留。|照顾者的饮食信息
K11|preserved|同一个体实际重叠率一手数据缺失及关键缺口身份保留。|目前尚无任何一手数据
K12|preserved|两假设缺乏实证的状态明确保留；同摘要还有概括性强断言，需另作摘要内部冲突风险而非抹去此限定。|两大假设缺乏实证支撑
K14|preserved|本批素材没有全球医美年龄分布具体值的否定保留。|本批素材未提供全球医美年龄分布
K15|missing|行行查2020人口图表没有可用年龄细分的检索缺口遗漏。|
K16|missing|组卷网题库缺原始图表与20–30岁细分的排除记录遗漏。|
K17|preserved|三个年龄比例及三个皮匠来源、对应URL完整保留。|19–23岁;;9e0bc4be9605ef330c2dc5e1f6b40c20.html
K18|preserved|BOS调查数据与BDJ链接对应关系保留。|BOS）2019年调查;;s41415-019-0777-6
K19|preserved|86.72%及成人错颌研究名称、URL对应关系保留。|86.72%;;13d3e7f5-248a-4389-8d51-56feeff4d1d1
K20|partial|万方归属和URL保留，但此固定条目还包含P<0.05，摘要没有保留阈值。|三个分量表得分;;PeriodicalPaper_zhyxmr201401013
K22|partial|第二章驱动因素主题保留，心理社会文化三类与研究任务身份未完整表述，部分机制被写成结论。|第二章“口腔正畸与医美需求的共同驱动因素”;;两者共享
K24|partial|重叠率、整合效果和满意度证据不足保留，遗漏依从性这一效果指标。|两大假设缺乏实证支撑
K25|missing|20–40岁人口4.60亿、74%患病率及3.40亿推算全部遗漏。|
K26|preserved|2019年BOS调查及75%医生认为私人成年业务增长保留。|75%受访正畸医生
K27|partial|只保留青春期女性与研究主题，国籍、14–19岁、均值和标准差遗漏。|青春期女性因媒体
K28|preserved|成人正畸样本与76.51%受限型治疗需求保留。|76.51%
K29|partial|保留青春期女性身份，遗漏14–19岁匈牙利学生这一具体适用边界。|青春期女性因媒体
K30|partial|16岁以上和16–24岁两种对象保留，遗漏NHS优先服务18岁以下人群的政策对象。|16–24岁年轻参与者
K31|preserved|Current Psychology及Springer URL对应医美接受度研究保留。|s12144-022-02863-z
K36|preserved|61%、2022年、小牛行研及对应URL完整保留。|女性整体占比约61%;;3466182973936633659
K38|partial|社交媒体假设缺乏实证及后续建模建议保留，遗漏缺20–30岁纵向追踪和采纳关联研究的具体界限。|两大假设缺乏实证支撑;;结构方程模型
K42|preserved|家庭对医美的间接路径与对正畸尚未系统考察两者均明确保留。|照顾者的饮食信息;;家庭环境对正畸决策
""",
}

def load(p):
    return json.loads(p.read_text())

def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def main(out):
    total = Counter()
    for topic, spec in REVIEWS.items():
        folder = out / f"题目{topic}"
        summary = (folder / "压缩后历史.md").read_text()
        gold = folder / "冻结关键信息标注.json"
        points = load(gold)["checkpoints"]
        annotation_hash = hashlib.sha256(gold.read_bytes()).hexdigest()
        assert annotation_hash == load(folder / "压缩统计.json")["annotation_sha256"]
        paragraphs = summary.split("\n\n")
        rows = []
        for line in spec.strip().splitlines():
            ident, status, reason, hints = line.split("|", 3)
            quotes = []
            for hint in hints.split(";;") if hints else []:
                matches = [p for p in paragraphs if hint in p]
                assert matches, (topic, ident, hint)
                # 多个段落匹配时保留最短真实段落，避免无关内容过多。
                quote = min(matches, key=len)
                if quote not in quotes:
                    quotes.append(quote)
            assert (status == "missing") == (not quotes)
            rows.append({"id": ident, "status": status, "quotes": quotes, "reason": reason,
                         "summary_spans": [{"quote": q, "start": summary.index(q), "end": summary.index(q)+len(q), "mapping": "exact"} for q in quotes]})
        assert len(rows) == len(points) and {r['id'] for r in rows} == {p['id'] for p in points}
        raw_attempts = []
        for path in sorted((folder / "模型调用/摘要保留评审").glob("尝试[0-9][0-9].json")):
            r = load(path)
            text = r.get("raw_response", "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            try:
                old = {x['id']: x for x in json.loads(text)['coverage']}
            except (ValueError, KeyError):
                continue
            raw_attempts.append({"path": str(path.relative_to(out)), "status": r['status'],
                "counts": dict(Counter(x['status'] for x in old.values())),
                "status_changes": [{"id": x['id'], "before": old[x['id']]['status'], "after": x['status'], "reason": x['reason']} for x in rows if old[x['id']]['status'] != x['status']]})
        meta = {"reviewed_at": datetime.now(ZoneInfo('Asia/Singapore')).isoformat(), "reviewer": "Codex，非独立人工标注员",
                "method": "在未读取匿名质量评分前逐项对照冻结statement与摘要全文；不使用why_needed推导额外要求；所有引句重新定位至摘要真实段落。保留既定144项及其分类，不合并或重分配分母。",
                "annotation_sha256": annotation_hash, "summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
                "is_human_gold": False, "semantic_review_is_model_judgment": True,
                "raw_judge_attempts": raw_attempts, "coverage": rows}
        save(folder / "摘要保留复核结果.json", meta)
        save(folder / "信息保留逐项结果.json", meta)
        total.update(x['status'] for x in rows)
    save(out / "摘要保留复核说明.json", {"reviewer": "Codex", "is_human_gold": False, "counts": dict(total),
        "reason": "qwen-plus两轮输出出现把清单原句当摘要引句、隐含补全年龄和任务口径、用罗列URL代替来源关联等错误；保留原始失败结果，使用透明逐项复核判定作为主统计。",
        "boundary": "同一Codex参与压缩前清单复核与压缩后判定，存在评审者偏差；没有独立人类一致性测量。多要素固定点任一必要要素缺失记partial，非必要举例和重复单位换算可省略。"})
    print(json.dumps(dict(total), ensure_ascii=False))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
