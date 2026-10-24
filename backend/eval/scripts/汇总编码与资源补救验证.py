#!/usr/bin/env python3
"""汇总固定100页编码修复、2份附件与1张图表验证，保留所有初评与复核。"""
import collections
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'backend/eval/data/通用研究289网址同HTML与MinerU对照-20261017-首轮'
PRIOR = ROOT/'backend/eval/data/通用研究47个不可用网页根因分析-20261017'
OUT = ROOT/'backend/eval/data/网页编码修复与三份资源补救验证-20261017'
REPORT = ROOT/'docs/网页编码修复与政府附件图表补救评估结果及分析-20261017.md'


def read(path):
    return json.loads(path.read_text())


def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pairs = read(SOURCE/'逐页配对结果.json')
    replay = {x['page_id']:x for x in read(OUT/'100页编码修复离线回放结果.json')}
    recovery_map = read(PRIOR/'编码恢复正文复评/编号映射.json')
    recovery = {recovery_map[x['id']]:x for x in read(PRIOR/'编码恢复正文复评/复评结果.json')['items']}
    recovery_text = {x['page_id']:x for x in read(PRIOR/'五页编码恢复机制实验.json')}
    resource_map = read(OUT/'三份资源正文复评/编号映射.json')
    initial = {resource_map[x['id']]['page_id']:x for x in read(OUT/'三份资源正文复评/复评结果.json')['items']}
    chart_review = read(OUT/'图表口径一致性复核/复核结果.json')
    downloads = read(OUT/'三份资源下载结果.json')
    assert all(x['downloaded'] and digest(Path(x['resource_path']))==x['sha256'] for x in downloads)
    assert chart_review['source_image_sha256']==digest(OUT/'补充资源/49-page-007/原始下载资源.png')
    assert chart_review['content_sha256']==digest(OUT/'补充资源/49-page-007/待评正文.txt')
    rows=[]
    for pair in pairs:
        page=pair['page_id']
        raw=replay[page]
        assessment=pair['baseline']['assessment']
        origin=str(SOURCE/'逐页配对结果.json')
        stage='保持原判定'
        content=raw['after']['content_path']
        if raw['content_changed']:
            assert raw['after']['content_sha256']==recovery_text[page]['recovered_content_sha256']
            assessment=recovery[page]
            origin=str(PRIOR/'编码恢复正文复评/复评结果.json')
            stage='编码修复'
        after_encoding=assessment['label'] in ['usable','partial']
        if page in initial:
            content=str(OUT/'补充资源'/page/'待评正文.txt')
            assessment=initial[page]
            origin=str(OUT/'三份资源正文复评/复评结果.json')
            stage='政府附件读取' if page!='49-page-007' else '目标图表读取与估读'
            if page=='49-page-007':
                assessment=chart_review['result']
                origin=str(OUT/'图表口径一致性复核/复核结果.json')
        rows.append({'page_id':page,'topic':pair['topic'],'title':pair['title'],'original_url':pair['url'],
                     'original_usable':pair['baseline']['content_usable'],'original_label':pair['baseline']['assessment']['label'],
                     'usable_after_encoding':after_encoding,'final_usable':assessment['label'] in ['usable','partial'],
                     'final_assessment':assessment,'assessment_source':origin,'recovery_stage':stage,
                     'content_path':content,'content_sha256':digest(Path(content)),
                     'initial_resource_assessment':initial.get(page),
                     'estimated_chart_data':page=='49-page-007','new_mineru_calls':0})
    count=len(rows)
    before=sum(x['original_usable'] for x in rows)
    enc=sum(x['usable_after_encoding'] for x in rows)
    without_chart=sum(x['final_usable'] for x in rows if x['page_id']!='49-page-007')
    final=sum(x['final_usable'] for x in rows)
    improvements=[x for x in rows if x['final_usable'] and not x['original_usable']]
    regression=[x for x in rows if not x['final_usable'] and x['original_usable']]
    assert count==100 and before==53 and enc==58 and without_chart==60 and final==61
    assert len(improvements)==8 and not regression
    budget=read(SOURCE/'MinerU提交预算.json')
    old_budget_hash=read(PRIOR/'交付完整性核对.json')['mineru_budget_sha256']
    assert budget['hard_cap']==budget['reserved_file_submissions']==100 and budget['remaining']==0
    assert digest(SOURCE/'MinerU提交预算.json')==old_budget_hash
    dump(OUT/'补救后100页逐页结果.json',rows)
    dump(OUT/'新增8页可用正文结果.json',improvements)
    summary={'date':'2026-10-17','denominator':100,'original_usable':before,'after_encoding':enc,
             'after_two_government_attachments':without_chart,'after_chart_content_review':final,
             'final_usable_rate':final/count,'gain_percentage_points':final-before,'recovered_original_failures':len(improvements),
             'remaining_unusable':count-final,'regressions':len(regression),'literal_numbers_only_chart_sensitivity':without_chart,
             'chart_initial_label':initial['49-page-007']['label'],'chart_final_label':chart_review['result']['label'],
             'final_labels':dict(collections.Counter(x['final_assessment']['label'] for x in rows)),
             'mineru_new_calls':0,'mineru_total_file_submissions':100,'budget_file_unchanged':True,
             'new_resource_downloads':3,'new_judge_requests':2,'new_judge_outputs':4,
             'encoding_judgments_reused_with_exact_sha256':5,'encoding_phase_unchanged_judgments_reused':95,
             'final_unchanged_original_judgments_reused':92,'final_new_resource_judgments':3,
             'scope':'原100页身份与分母固定；HTML用原字节/浏览器档案离线回放，3个子资源本次真实下载；不是100个URL在线重抓或同HTML模型新比较',
             'implementation_boundary':'编码修复已进入生产读取器源码；附件/图表为验证流程，未自动接入通用读取链路。'}
    dump(OUT/'最终指标汇总.json',summary)
    dump(OUT/'三份资源最终验证结果.json',[{'page_id':x['page_id'],'resource':next(d for d in downloads if d['page_id']==x['page_id']),
          'extraction':read(OUT/'补充资源'/x['page_id']/'提取记录.json'),'final_assessment':x['final_assessment'],
          'initial_assessment':x['initial_resource_assessment'],'content_path':x['content_path']} for x in rows if x['page_id'] in initial])
    dump(OUT/'实验执行记录.json',{'date':'2026-10-17','initial_replay_interruption':{
        'reason':'本地web_document.py新增读取器未使用的source_available函数，整文件相等断言中止；没有开始100页回放或网络请求',
        'resolution':'保留现有用户代码，仅对照读取器依赖AST并单列未使用的新增函数；重跑后原100份待评正文全部逐字复现',
        'log_path':str(OUT/'首次回放中断日志.txt')},
        'chart_boundary_review':{'initial_label':'unusable','review_label':chart_review['result']['label'],
            'reason':'初评将“图片URL不算已提取图片里的数字”误读成“图片中的数字不计入已提取内容”；追加原PNG和原提示词复核，保留所有初评。未把估读数据改说成精确文字。'},
        'network_scope':{'source_html_refetches':0,'resource_downloads':3,'claude_requests':2,'mineru_requests':0}})
    dump(OUT/'评估配置.json',{'date':'2026-10-17','source_run':str(SOURCE),'prior_analysis':str(PRIOR),
        'max_chars':30000,'min_chars':500,'semantic_usable_labels':['usable','partial'],'judge_model':'claude-opus-5',
        'judge_max_tokens':4000,'temperature':'沿用原默认值','fixed_denominator':100,
        'replay':'复现原下载错误、静态字节和已有浏览器正文；网络方法只返回归档内容',
        'resource_methods':{'doc':'macOS textutil转UTF8文本','docx':'python-docx段落和表格','image':'本地Apple Vision OCR＋目标PNG柱高估读＋原图视觉核对'},
        'mineru_forbidden':True,'production_attachment_automation_added':False})

    lines=[
        '# 网页编码修复与政府附件图表补救评估结果及分析', '',
        '评估日期：2026-10-17（Asia/Singapore）。本次新建运行与报告，承接同日47页失败归因，不覆盖此前评估。', '',
        '## 结果', '',
        '**固定原100页，按“真实目标全文、实质部分正文/摘要或目标内容可用”的原口径，最终为61/100（61%），较53/100增加8个百分点，追回原47个失败中的8个，剩余39个不可用。**', '',
        '**其中1页为已取回原图后的图表内容转写与近似估读。若不将该估读图计入分子，结果为60/100，较原来增加7个百分点。** 图中没有柱顶精确数值标签，估读值不能当作原始统计数字精确引用或计算。', '',
        '| 阶段 | 可用页数/固定分母 | 相对原53页增量 |',
        '| --- | ---: | ---: |',
        '| 原正式配对复核结果 | 53/100 | — |',
        '| 修复编码后 | 58/100 | +5个百分点 |',
        '| 再读取2个政府附件 | 60/100 | +7个百分点 |',
        '| 再取得目标图表，计入图表内容转写/估读 | 61/100 | +8个百分点 |', '',
        '编码修复已写入`backend/app/service/web_reader_service.py`。新增11项编码回归、原有8项读取器单元测试均通过。100页离线回放中，修复前实现100/100逐字复现原待评输出；修复后只改变5个乱码样本，其余95份输出SHA-256不变，5个恢复样本均通过当前程序质量门禁。', '',
        '**没有调用MinerU。** 本轮仅真实下载2份政府附件与1张PNG，另调用Claude两次做3份输出初评及1份图表边界复核。MinerU原提交预算文件哈希未变，累计仍为100份、剩余0。', '',
        '## 评估对象、数据及配置口径', '',
        '- 固定上轮从231份HTML分层抽出的100个原URL身份。没有删掉失败页面、减少分母，或把附件另算新页面。历史289页的51.56%不直接参与本次计算。',
        '- 编码阶段不重新抓100个网页：使用原始响应字节、已有浏览器DOM和同次正文、原下载失败记录，回放修复前/后读取器；浏览器调用只返回档案，不访问浏览器或外部网站。仍按原共享HTML协议统一UTF-8与meta声明。',
        '- 编码阶段95份未变文本复用原最终配对复核标签；5份编码修复输出与上一轮已复评的恢复文本逐字SHA-256一致，复用其Claude标签。资源补救后最终保留92份原判定、5份编码恢复判定、3份资源判定。没有仅凭程序success就判为语义可用。',
        '- 3个资源均从失败HTML已存在的同题资源入口下载，保留入口、最终重定向URL、MIME、文件魔数、字节数与SHA-256；提取后使用原题名及同一正文判定提示词。',
        '- 正文上限30000字符，usable与partial均计入分子。DOC为9514待评字符，DOCX为20746字符，均未因上限截断正文。图表输出为目标图形内容的文字表示，数值明确标为估读。',
        '- 裁判请求/实际返回模型均claude-opus-5，max_tokens=4000、默认温度。附件/图表初评使用原提示词、打乱匿名编号；图表复核额外提供原PNG及规则澄清，属于一致性复核，不是另一轮独立盲评。',
        '- 这是“编码修复＋子资源补救”的组合验证，3个资源输入比原共享HTML更丰富；不是修复后现有方案与MinerU的新同输入比较，不能把61%与原MinerU的53%相减当作抽取模型优势。', '',
        '## 编码修复内容与回归', '',
        '旧实现只读取HTTP头中的charset，缺失时直接UTF-8且errors="replace"，忽略HTML里的GB2312声明。修复按BOM、可严格解码的HTTP声明、有效UTF-8、HTML meta候选尝试严格解码；GB2312/GBK按兼容编码GB18030处理。只有严格候选均失败后才使用现有BeautifulSoup的编码探测，最后才以替换字符兜底。', '',
        '先验证完整UTF-8可避免缺少HTTP声明时被过期legacy meta再次解坏；HTTP明确且可严格解码的声明优先保留。支持引号、大小写、HTTP-equiv meta，忽略注释中的假meta，UTF-32 BOM在UTF-16之前匹配。没有增加付费API或新增运行依赖。', '',
        '| 回归检查 | 已测量结果 |',
        '| --- | --- |',
        '| 新增编码协议回归 | 11/11通过，含GB2312、GBK扩展、无/错误/冲突声明、BOM、UTF-8与空响应 |',
        '| 原有读取器单元回归 | 8/8通过，含静态正文、浏览器回退、状态码、超时、PDF无文本层与URL安全 |',
        '| 修复前实现回放 | 100/100原输出逐字复现 |',
        '| 修复后实现回放 | 仅5份乱码输出改变，其余95份未变；0条选中静态/浏览器来源切换 |',
        '| 5份语义复评复用 | 待评文本SHA-256均匹配既有复评输出，5份usable，当前机器质量门禁5/5 passed |', '',
        '工作区`web_document.py`另有未被读取器导入的source_available新增函数。本轮保留该用户改动，核对其他依赖AST与冻结版一致；首次整文件相等断言因此中断，未开始回放或网络请求。调整审计方式后完成100页回放，相关日志全部保留。', '',
        '## 新增8页逐页结果', '',
        '| 页码 | 目标 | 补救方式 | 最终判定 |',
        '| --- | --- | --- | --- |',
    ]
    for x in improvements:
        suffix='（图形估读，非精确原值）' if x['estimated_chart_data'] else ''
        lines.append(f'| {x["page_id"]} | {x["title"].replace("|","／")} | {x["recovery_stage"]} | {x["final_assessment"]["label"]}{suffix} |')
    lines += [
        '', '### 温州政府DOC附件：新增1页可用', '',
        '35-page-043原HTML只有通知题名与DOC链接。本次入口重定向至浙江政务OSS资源，返回application/msword，87040字节，魔数为OLE Word。通过macOS textutil取得《温州市人民政府关于下达2016年温州市国民经济和社会发展计划的通知》及所附年度计划正文，包含具体发展目标、任务与指标。', '',
        '待评正文9514字符；Claude判usable，引句可逐字定位：“《2016年温州市国民经济和社会发展计划》已经市第十二届人大六次会议审议批准，现印发给你们。” 这说明原失败主因是没有跟随附件，HTML提取器本身无法生成未下载的DOC正文。', '',
        '### 湖南政务DOCX附件：新增1页可用', '',
        '46-page-041返回真实DOCX，42174字节，ZIP/OOXML结构。python-docx取得《中共中央关于制定国民经济和社会发展第十五个五年规划的建议》题名、通过日期及分节条文，待评20746字符；Claude判usable。', '',
        '原HTML只有标题、日期和附件入口；补救增加的是附件正文，不能说仅调整HTML选择器就已解决。', '',
        '### 人口结构PNG图表：图形内容可读，数值只能估读', '',
        '49-page-007下载到66918字节、1178×565像素的真实PNG，题名确为“2020年中国人口结构（分性别、年龄）”。原图含男女图例、17个年龄组、单位“万”、纵轴0–7000刻度、20–39岁红框及来源信息；没有柱顶精确数字标签。', '',
        '本地Apple Vision识别28个文本块，原始框坐标和置信度已保留。旋转年龄标签的0-4与5-9有OCR丢失横线/数字，已对照原图修正。另针对这张图按图例颜色定位34根柱，用纵轴刻度线性换算，取整至100万人，保存柱顶像素、估读数据及脚本。它是目标图形内容转写，不是OCR读出原始数值，也不是通用图表准确率测试。', '',
        '例如图形中30–34岁组男性约6500万人、女性约6100万人；这些量仅用于年龄结构和相对高低判断，不用于精确引用或合计。缺少数值标签、图像像素和取整误差均会影响结果，本次没有独立金标误差验证。', '',
        '图表初次文字盲评为unusable，其理由把原规则“图片URL不算已提取图片里的数字”误读成“图片中的数字不计入已提取内容”。本轮保留初评，并追加原PNG和同一规则复核。Claude确认目标实质图表内容已读到，改判usable，同时明确近似值不是图中逐字标注的数据。最终61%计入这1页；如任务只接受可直接提取的精确文字/数字，则排除此页，报告60%。', '',
        '## 失败、残留问题与下一步', '',
        '- 原47个共同失败样本追回8个，剩余39个仍保持失败。本次未对这些剩余页重新换源或渲染，因此不能宣称验证/登录、错页、空模板和其他文库预览已解决。逐页最终标签保留在100页机器结果中。',
        '- 图表初评失败及改判原因完整落盘，不能只留有利的最终标签；精确数据提取这一子目标仍未完成。后续若需要确切人口数值，应另找同图原始报告数据或可下载的数据表。',
        '- 建议下一步将政府附件跟随、文件类型识别与DOC/DOCX文本路由接入通用读取链路，先以这2份资源做集成回归。本轮仅验证了2个真实资源，不代表所有政府附件或Word文件都能成功。',
        '- 图像补救先区分“图中文字OCR”与“图形估读”，把估读状态和精度写入证据元信息；需要精确数值的任务不把估算当作原始证据。',
        '- 对其余失败继续优先做错页/验证识别后的公开换源；拒绝失败页本身不会增加固定分母中的可用页数。', '',
        '## 原始结果路径与可追溯性', '',
        f'- 本轮运行目录：`{OUT.relative_to(ROOT)}`。',
        '- [最终指标汇总](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/最终指标汇总.json)、[补救后100页逐页结果](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/补救后100页逐页结果.json)、[新增8页结果](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/新增8页可用正文结果.json)。',
        '- [100页编码回放核对汇总](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/100页编码回放核对汇总.json)、[100页回放明细](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/100页编码修复离线回放结果.json)、[编码语义标签复用核对](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/编码修复语义判定复用核对.json)。每页修复前后正文、选中HTML及机器门禁结果在`100页离线回放/<page_id>/`。',
        '- [3份资源下载记录](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/三份资源下载结果.json)、[3份资源最终验证](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/三份资源最终验证结果.json)。`补充资源/<page_id>/`保留原文件、全文/待评输出、提取记录与哈希。',
        '- [图表原PNG](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/补充资源/49-page-007/原始下载资源.png)、[本地OCR原始输出](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/补充资源/49-page-007/本地OCR原始结果.json)、[柱形高度测量](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/补充资源/49-page-007/柱形高度测量.json)、[估读数据](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/补充资源/49-page-007/图表估读数据.json)。',
        '- [3份输出初评](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/三份资源正文复评/复评结果.json)、[图表口径一致性复核](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/图表口径一致性复核/复核结果.json)。对应目录保留匿名材料、提示词、编号映射、完整原始服务响应与用量；复核目录另有含原图的完整请求。',
        '- [新增11项编码测试](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/编码回归单元测试结果.json)、[原有8项单元回归](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/原有读取器单元回归结果.json)，同名输出txt保留逐项日志。',
        '- [依赖差异审计](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/读取器依赖差异审计.json)、[代码文件指纹](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/本轮代码文件指纹.json)、[实验执行记录](../backend/eval/data/网页编码修复与三份资源补救验证-20261017/实验执行记录.json)。',
        '- 生产编码修复：`backend/app/service/web_reader_service.py`；回归测试：`backend/eval/tests/test_web_reader_encoding.py`；回放脚本：`backend/eval/scripts/回放100页验证编码修复.py`；图表估读脚本：`backend/eval/scripts/估读人口结构图表.py`。本轮保留修复前/后读取器源码、测试与脚本快照。',
        '- [此前47页根因报告](通用研究47个不可用网页根因分析与可用率改进建议-20261017.md)、[原100页同HTML正式报告](通用研究289网址同HTML与MinerU对照-20261017-首轮结果及分析.md)。', '',
        '## 局限性与证据边界', '',
        '**已测量**：生产解码函数修复、19项单元回归、原100页冻结响应的修复前后回放、5个文本字节与既有语义复评一致、3份子资源真实下载与识别、2个附件正文可用、图表实质内容转写及近似估读、固定100页下最终61%及排除估读图的60%敏感性、0新增MinerU调用。', '',
        '**间接推断**：自动跟随政府附件、按内容类型分流、区分图表识别与估读会改善线上覆盖的方向；2个附件和1张图表是按已知失败机制挑出的样本，不能估计所有附件/图像的普遍成功率。', '',
        '**尚未评估**：100个URL全量在线重抓、部署后的端到端可用率和延迟、旧乱码缓存失效后的实际覆盖、通用附件/图像自动路由、图表原始精确数据与估读误差金标、289页整体收益、修复输入上的MinerU表现、人工双评一致性。附件/图表补救尚未接入生产通用读取链路，不能把本次61%当作线上系统已经稳定达到的指标。', '',
        '61%沿用上轮对滚动会议主页的原判定。若只对上轮已识别的31-page-019改用严格原年份，其分子需减1，最终为60/100；这不是对100页年份做了全新审核。', '',
        '结论：编码问题已修复且在冻结样本中没有观察到正文回退。2个政府附件确认可以救回，图表可读但精确数值不可直接抽取；本次目标内容可用率为61%，若排除估读图则为60%。下一步工程投入优先把已验证的附件路由接入读取器。', '',
    ]
    REPORT.write_text('\n'.join(lines))
    shutil.copy2(__file__,OUT/'冻结指标汇总脚本.py')
    print(json.dumps({'report':str(REPORT),**summary},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
