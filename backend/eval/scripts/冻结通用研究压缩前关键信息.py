"""Codex在看摘要之前对首轮模型初标作证据复核，冻结可追溯检查点。"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

spec=importlib.util.spec_from_file_location('evaluation',Path(__file__).with_name('评估通用研究历史压缩与后续写作.py'))
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)

KEEP={
16:list(range(1,11))+[13,17,18,19,20,21,22,27,28,29,30,33,34,44],
31:list(range(1,11))+[12,14,15,16,17,19,20,21,22,24,25,26,27,28,31,32,33,34,35],
35:list(range(1,24))+[25,26,27,30,33,35],
46:list(range(1,30))+[32,34],
49:list(range(1,13))+[14,15,16,17,18,19,20,22,24,25,26,27,28,29,30,31,36,38,42],
}
EDIT={
16:{
'K13':'CNN超分辨率视觉测量材料针对精密加工行业零件形位公差检测；性能数据不能脱离这一任务口径。',
'K18':'电容-惯性融合的输入材料来源为西安交通大学论文页面，保留对应Non-periodic lower-limb motion recognition with noncontact capacitive sensing来源及URL。',
'K19':'CNN超分辨率视觉测量的输入材料来源为EMT期刊Research on Optimization of non-contact measurement accuracy based on vision technology页面，保留来源及URL。',
'K20':'RobHAR的输入材料关联arXiv论文2405.01882及对应URL。',
'K21':'mmDoppler数据集的输入材料关联arXiv论文2407.21125及对应URL。',
'K22':'三台毫米波雷达跟踪与跌倒检测材料关联X-MOL页面1767630501405495296及对应URL。',
'K34':'非线性温度补偿IPS材料关联西安交通大学论文A non-linear temperature compensation model for improving the measurement accuracy of an inductive proximity sensor及对应URL。',
'K44':'既有首章的术语上下文将非接触式感知算法按射频、光学及电容/电场感应三条路径组织，供后续章节保持术语一致。'},
31:{
'K01':'用户要求分别总结国内外博物馆教育现状与特点，其中国外部分按代表性国家分别归纳；未在用户原问题中指定国家清单。',
'K02':'用户要求对博物馆作自成体系的分类总结，例如按行政级别或按国有和私人分类，并未强制同时采用两套分类。',
'K04':'输入材料称2018年中国博物馆总数为5354家，来源为2019年中国博物馆市场分析报告。',
'K05':'输入材料列出截至2017年的中国国家一级博物馆名单，数量为130家；需保留年份与专业等级口径。',
'K06':'输入材料称2018年国有博物馆占比为88.6%。',
'K07':'输入材料称美国现有博物馆近5000家，涵盖综合性与专业性博物馆；这条数量陈述未给出统计年份。',
'K08':'输入材料称伦敦博物馆团体代表英国首都250家博物馆，属公共机构。',
'K09':'输入材料称英国艺术委员会基金组织拨款10万英镑（约100万元人民币）支持博物馆—大学分享与合作项目。',
'K14':'输入材料引用福州市政府页面对博物馆的定义：以教育、研究和欣赏为目的的非营利组织。',
'K15':'输入材料区分国有博物馆按事业单位登记规定设立、非国有博物馆在民政部门登记设立的法人登记路径。',
'K17':'美国博物馆教育实践材料来自查字典教育资讯网的观察性报道，保留该来源归属。',
'K19':'已完成的中国博物馆章节采用行政层级与所有制双维分类；这是已有正文的组织方式，而非用户强制双维分类的要求。',
'K20':'已完成的国外博物馆章节对美英使用给定材料，对法德日的概述明确标注直接素材不足和行业共识性质。',
'K21':'输入材料的观察者称美国博物馆为藏品提供来源、鉴定和研究论文索引等资料；这是该报道的观察，不能扩展为所有博物馆的已核验通则。',
'K24':'既有中国博物馆章节以建川博物馆的沉浸场景和观复博物馆的新媒体导览说明非国有馆教育形式创新，这是历史正文中的案例归纳。',
'K25':'既有中国博物馆章节称国家级馆普遍设专职教育部并开发K12至成人课程，这是历史正文中的概述。',
'K26':'既有中国博物馆章节概括东部市级馆与中西部部分市级馆在教育活动上的差异，需保留地域和范围限定。',
'K31':'美国博物馆近5000家的数量条目没有统计年份，不能当作已核实的最新同口径数据。',
'K32':'输入材料将我国博物馆按资产来源和登记路径划分为国有、非国有，国有又包含文物行政部门管理与其他行业性馆。',
'K33':'2018年国有博物馆88.6%的比例来自2019年中国博物馆市场分析报告，保留这项来源归属。',
'K34':'历史助手把补充县级博物馆教育现状实地调研数据列为改进建议，尚不是已经完成的调研。',
'K35':'历史助手建议补充法国文化通行证、德国教育框架计划和日本社会教育设施的具体实施与评估数据，尚不是已完成补证。'},
35:{
'K09':'湘潭高新区雨污分流改造项目材料称总投资及合同价154.6万元、2021年预算73万元、支付进度97%，并称资金合规未超预算；应保留这些数值的不同口径。',
'K20':'输入材料把王洪臣关于污水处理基础规模与提质增效的发言归属于E20 2020年第十八届水业战略论坛。',
'K21':'湘潭高新区项目材料来自2021年老旧小区雨污分流改造工程项目支出绩效评价报告及地方政府网站。',
'K22':'东莞分散式污水设施材料来自东莞市生态环境局关于暂缓建设分散式污水处理站建议的政协提案办理结果。',
'K23':'首章核拨制运行现状与效率瓶颈分析已经完成，并在第二章请求中作为已完成章节的术语上下文再次提供。',
'K25':'历史洞察将问题归因为缺乏跨雨污系统的统一绩效目标、成本分摊规则和应急联动流程；这是已有分析观点，不是新的用户硬要求。',
'K35':'已有材料使用雷厉风行与久久为功区分短期提效措施和长期建设措施，后续应保持两类措施的时间与行动范围区别。'},
46:{
'K07':'历史输入转述新建建筑垃圾排放量目标为不超300吨/立方米；这是原材料中的可疑单位，不能当成已核实政策值。',
'K20':'已完成首章指出300吨/立方米的单位存疑，并按行业惯例解释为吨/万平方米；应保留这是历史作者的修正解释而非本轮核验结论。',
'K21':'已完成政策章节将房地产税全面推行表述为仍需审慎评估宏观承受力与地方财政替代方案，短期更可能局部试点，属于条件性展望。',
'K24':'输入中十四五建筑业发展规划附件关联中国政府网2022年1月27日发布路径及对应PDF URL。',
'K25':'输入中十四五建筑业发展规划通知还有云南省政府门户转载来源，保留来源归属和建市〔2022〕11号文件号。',
'K26':'输入中十四五建筑业发展规划通知还有北京市政府门户转载来源，页面标注现行有效，保留该状态的来源归属。',
'K29':'已完成首章把核心动力概括为制度约束、技术进步与责任内化三重协同，这是历史正文的分析框架。',
'K32':'已完成首章用竞争、创新、约束、自我调节和宏观调控五类机制组织表格，属于已有章节框架。',
'K34':'输入材料对好房子建设标准的概述包含绿色建筑、智能建造、健康性能、耐久品质四个维度，需保留来源归属。'}
,
49:{
'K03':'输入材料给出2023年中国正畸人群中19–23岁女性16%、24–28岁37%、29–33岁20%，并把合计73%写作20–33岁；应保留原年龄分组，不能直接视为20–30岁全球女性的共同需求比例。',
'K11':'当前输入明确记录尚无一手数据给出正畸与医美在同一个体层面的实际重叠率，这是本题关键证据缺口。',
'K12':'历史材料把社交媒体驱动和整合提升满意度两项假设列为当前批次缺乏实证支撑，不能写成已验证结论。',
'K14':'已完成首章明确写出本批素材未提供全球医美年龄分布的具体数值。',
'K22':'第二章的既定任务是研究正畸与医美需求的心理、社会和文化共同驱动因素；该任务描述本身不是这些机制已经被证明。',
'K24':'既有正文明确尚缺正畸与医美共同需求比重的权威量化研究，也缺整合服务是否提升满意度或依从性的实证数据。',
'K25':'输入材料以中国2020年20–40岁人口4.60亿及74%错颌畸形患病率估算正畸需求3.40亿，需保留这是一项特定年龄人口的推算。',
'K28':'输入中成人正畸患者研究条目给出76.51%存在受限型正畸治疗需求，保留该研究样本身份。',
'K29':'女性医美偏好材料同时涉及14–19岁匈牙利女性学生，不能把该样本自动改写成20–30岁全球女性。',
'K30':'自尊自信的决策因素条目针对16岁以上人群，另有16–24岁参与者支持NHS免费服务优先18岁以下人群的政策偏好，两者年龄和问题口径不同。',
'K31':'女性医美偏好的输入材料关联Current Psychology论文及Springer URL 10.1007/s12144-022-02863-z。',
'K38':'已完成第二章认为目前无法量化社交媒体对共同需求的具体贡献，并指出缺乏20–30岁女性纵向追踪或媒介使用与服务采纳关联分析。',
'K42':'已完成第二章将家庭环境对医美接受度的可能间接影响与其对正畸决策尚未系统考察区分开。'}
}

QUOTE_EDITS={
35:{'K20':'中国人民大学环境学院副院长、中国人民大学低碳水环境技术研究中心主任王洪臣在E20“2020（第十八届）水业战略论坛”上表示，全国已形成“100＋2”的污水处理基础规模，十年翻了一番，排水管道已超过100万公里，全国污水处理能力超过2亿立方米/日。',
    'K23':'### 已完成章节的术语上下文\n- 核拨制在市政污水系统中的运行现状与效率瓶颈分析：当前中国市政污水系统普遍采用的核拨制模式，在缺乏绩效约束与成本激励机制的背景下，已显现出系统性效率瓶颈。'},
49:{'K19':'在成人正畸治疗中，女性患者数量多于男性；86.72%的成人正畸患者以牙齿美观为目的接受治疗；76.51%的患者存在受限型正畸治疗需求。 (来源: Psychological characteristic and nursing of adult malocclusion, URL: https://xuebao.301hospital.com.cn/en/article/id/13d3e7f5-248a-4389-8d51-56feeff4d1d1, 可信度: 0.9)',
    'K24':'目前尚无权威研究量化“正畸+医美”共需的实际比重，也缺乏关于整合服务是否提升用户满意度或依从性的实证数据。'}
}


def views(record):
    yield record['text'],'raw_payload_content'
    try:
        data=ev.parse_json(record['text'])
        if isinstance(data,dict):
            for key,value in data.items():
                if isinstance(value,str):yield value,'$.'+key
                elif isinstance(value,list):
                    for i,item in enumerate(value):
                        if isinstance(item,str):yield item,f'$.{key}[{i}]'
    except (ValueError,TypeError):pass


def resolve(point,records):
    quote=point['quote'].replace('\\n','\n')
    order=sorted(records,key=lambda r:r['source_id']!=point['source_id'])
    for row in order:
        for text,path in views(row):
            match=ev.locate(quote,text,decode_json=False)
            if match:return row,path,text,match
    # 原初标用省略号跳过中段时，恢复单一来源中真实存在的连续范围，绝不拼造引句。
    parts=[s.strip() for s in re.split(r'\.{3,}|…+',quote) if s.strip()]
    if len(parts)>1:
        for row in order:
            for text,path in views(row):
                pos=[];cursor=0
                for part in parts:
                    hit=text.find(part,cursor)
                    if hit<0:break
                    pos.append((hit,hit+len(part)));cursor=hit+len(part)
                if len(pos)==len(parts):
                    left,right=pos[0][0],pos[-1][1]
                    return row,path,text,{'quote':text[left:right],'start':left,'end':right,'mapping':'expanded_contiguous_ellipsis'}
    return None


def main(out):
    for topic in ev.SOURCES:
        folder=out/f'题目{topic}'
        assert not (folder/'压缩后历史.md').exists(),'必须在摘要生成前冻结'
        initial=ev.parse_json(ev.load(folder/'模型调用/关键信息初标/尝试01.json')['raw_response'])['checkpoints']
        sample=ev.load(folder/'冻结样本.json')
        keep=KEEP[topic]
        accepted=[];audit=[]
        for original in initial:
            point=deepcopy(original);n=int(point['id'][1:])
            if n not in keep:
                audit.append({'id':point['id'],'decision':'excluded_before_summary','reason':'Codex复核：重复、引句不足以支持附加推断/强制要求、或对后续写作价值较低；不依据摘要结果筛选','original':original});continue
            point['statement']=EDIT.get(topic,{}).get(point['id'],point['statement'])
            point['quote']=QUOTE_EDITS.get(topic,{}).get(point['id'],point['quote'])
            result=resolve(point,sample['records'])
            if result is None:
                audit.append({'id':point['id'],'decision':'unresolved_before_summary','reason':'无法在原历史定位足以支持该点的连续引句，留作标注缺口，不计入分母','original':original});continue
            row,path,text,span=result
            # 来源归属检查点保留所在原始事实行，以便检验标题、URL、日期是否真实提供。
            if point['category']=='来源与证据定位' and path=='raw_payload_content':
                left=text.rfind('\n',0,span['start'])+1;right=text.find('\n',span['end'])
                if right<0:right=len(text)
                if right-left<2200:span={'quote':text[left:right],'start':left,'end':right,'mapping':'original_source_line'}
            point.update(source_id=row['source_id'],quote=span['quote'],source_span={**span,'text_view':path},source_kind=row['kind'],
                         statement_origin='Codex_review_of_initial_annotation',is_external_fact_verified=False)
            accepted.append(point)
            audit.append({'id':point['id'],'decision':'accepted_before_summary','statement_changed':point['statement']!=original['statement'],
                          'source_id_changed':point['source_id']!=original['source_id'],'original':original,'accepted':point})
        assert len(accepted)>=20,(topic,len(accepted))
        frozen={'checkpoints':accepted,'label_origin':'llm_first_draft_then_codex_source_review','is_human_gold':False,
                'frozen_before_summary':True,'frozen_at':ev.now(),'source_history_sha256':sample['source_history_sha256'],
                'scope':'固定原始历史中的关键点样本，非全文穷尽金标；原始作者观点和资料陈述保留来源身份',
                'initial_count':len(initial),'accepted_count':len(accepted)}
        ev.save(folder/'冻结关键信息标注.json',frozen);ev.save(folder/'压缩前标注复核记录.json',audit)
        lines=[f"# 题目{topic}压缩前冻结关键信息",'',sample['query'],'','模型初标后由Codex在看摘要前核对来源和陈述边界；不是独立人工金标。','']
        for p in accepted:
            lines += [f"## {p['id']} · {p['category']} · {p['importance']}",'',p['statement'],'',f"来源：{p['source_id']}，视图{p['source_span']['text_view']}，字符{p['source_span']['start']}至{p['source_span']['end']}。",'', '> '+p['quote'].replace('\n','\n> '),'']
        (folder/'关键信息标注.md').write_text('\n'.join(lines))
        print(topic,len(initial),len(accepted),'unresolved',[a['id'] for a in audit if a['decision']=='unresolved_before_summary'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
