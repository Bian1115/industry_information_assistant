"""汇总固定断点压缩、保留、再暴露和匿名质量评审；保留逐题配对口径。"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

TOPICS = [16, 31, 35, 46, 49]
ROOT = Path(__file__).resolve().parents[3]


def read(p):
    return json.loads(p.read_text())


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "不适用"


def main(out):
    cfg=read(out / "运行配置.json")
    quality_policy=read(out / "质量评分采用口径.json")
    results, omissions, categories, usage = [], [], defaultdict(Counter), defaultdict(lambda: Counter())
    judges = []
    for topic in TOPICS:
        d = out / f"题目{topic}"
        sample = read(d / "冻结样本.json")
        points = read(d / "冻结关键信息标注.json")["checkpoints"]
        retention = read(d / "信息保留逐项结果.json")["coverage"]
        direct_exposure = read(d / "共同输入原标注引句核对.json")["items"]
        exposure = {r["id"]: {"id": r["id"], "status": "source_quote_reappears" if r['source_quote_reappears'] else "unconfirmed",
            "reason": "程序在后续共同输入中定位到完整原标注引句。" if r['source_quote_reappears'] else "未精确匹配原标注引句；可能存在改写、分散表述或部分信息，不解释为不存在。",
            "common_input_span": r['common_input_span']} for r in direct_exposure}
        lookup = {p["id"]: p for p in points}
        counts = Counter(r["status"] for r in retention)
        critical = Counter(r["status"] for r in retention if lookup[r["id"]]["importance"] == "critical")
        for r in retention:
            point = lookup[r["id"]]
            categories[point["category"]][r["status"]] += 1
            if r["status"] != "preserved":
                omissions.append({"topic": topic, **point, "retention": r, "reexposure": exposure[r["id"]]})
        comp = read(d / "压缩统计.json")
        original = read(d / "原文组/续写结果.json")
        compressed = read(d / "压缩组/续写结果.json")
        assert original["common_material_hashes"] == compressed["common_material_hashes"], "两组固定任务与证据不相同"
        assert original["fixed_chapter_hashes"] == compressed["fixed_chapter_hashes"], "两组断点前报告不同"
        group_judges = []
        request=read(d/'模型调用/匿名质量评审2/请求.json')
        first=read(d/'模型调用/匿名质量评审2/尝试01.json')
        first_text=re.sub(r'^```(?:json)?\s*|\s*```$', '', first['raw_response'].strip())
        candidate_data=json.loads(request['messages'][1]['content'])['candidates']
        first_mapping={c['id']:next(g for g in ['原文组','压缩组'] if c['report']==(d/g/'完整报告.md').read_text()) for c in candidate_data}
        first_raw={'model':request['model'],'mapping':first_mapping,'result':json.loads(first_text),
            'order':'原顺序','quote_validation_passed':first['status']=='accepted','validation_error':first.get('error')}
        reverse_raw=read(d/'质量顺序复评结果.json')
        reverse_raw.update(order='交换顺序',quote_validation_passed=None,validation_error='本调用仅验证数值结构，引句另行检查')
        for raw in [first_raw,reverse_raw]:
            row = {"topic": topic, "model": raw["model"], "order":raw['order'],"quote_validation_passed":raw['quote_validation_passed'], "included_in_primary": True, "winner": raw["mapping"].get(raw["result"]["winner"], "tie"), "winner_id":raw['result']['winner'], "mapping":raw['mapping'], "comparison": raw["result"]["comparison"], "groups": {}}
            for candidate in raw["result"]["candidates"]:
                group = raw["mapping"][candidate["id"]]
                row["groups"][group] = {"overall": statistics.mean(candidate["overall"].values()), "continued": statistics.mean(candidate["continued"].values()), "dimensions": {k: candidate[k] for k in ["overall", "continued"]}, "rationale": candidate["rationale"], "issues": candidate.get("issues", [])}
            row["continued_delta"] = row["groups"]["压缩组"]["continued"] - row["groups"]["原文组"]["continued"]
            row["overall_delta"] = row["groups"]["压缩组"]["overall"] - row["groups"]["原文组"]["overall"]
            judges.append(row)
            group_judges.append(row)
        assert len(group_judges)==2, "每题必须有同一裁判两种候选顺序的首次完整分数"
        unpreserved = [r for r in retention if r["status"] != "preserved"]
        exposure_counts = Counter(exposure[r["id"]]["status"] for r in unpreserved)
        result = {"topic": topic, "points": len(points), "counts": dict(counts), "critical_counts": dict(critical),
            "full_retention_rate": counts["preserved"] / len(points), "compression": comp,
            "history_reduction": 1-comp["summary_message_tokens"]/comp["history_tokens"],
            "unpreserved_reexposure": dict(exposure_counts), "original": original, "compressed": compressed,
            "quality": group_judges,
            "quality_order_averaged_delta": statistics.mean(r['continued_delta'] for r in group_judges),
            "same_winner_across_orders":group_judges[0]['winner']==group_judges[1]['winner']}
        results.append(result)
    for p in out.glob("题目*/**/模型调用/*/尝试[0-9][0-9].json"):
        r = read(p)
        rel = p.relative_to(out)
        group = "原文续写" if "原文组" in rel.parts else "压缩续写" if "压缩组" in rel.parts else "摘要生成" if any(x.startswith("历史摘要生成") for x in rel.parts) else "离线标注与评审"
        u = r.get("usage") or {}
        usage[group]["attempts"] += 1
        usage[group]["accepted"] += int(r.get("status") == "accepted")
        usage[group]["failed"] += int(r.get("status") != "accepted")
        usage[group]["requests_missing_usage"] += int(not bool(u))
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            usage[group][key] += u.get(key) or 0
        usage[group]["request_elapsed_seconds"] += r.get("elapsed_seconds", 0)
    probe=out/'替代模型实际可用性检查.json'
    if probe.exists():
        r=read(probe);u=r.get('usage') or {};usage['模型可用性检查']['attempts']=1
        usage['模型可用性检查']['accepted']=int(r.get('status')=='available')
        for key in ['prompt_tokens','completion_tokens','total_tokens']:usage['模型可用性检查'][key]=u.get(key) or 0
    totals = Counter()
    for r in results:
        totals.update(r["counts"])
    report = {"date": "2026-11-01", "updated_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(), "topics": results,
        "retention_total": dict(totals), "total_points": sum(totals.values()), "strict_full_retention_rate": totals["preserved"]/sum(totals.values()),
        "per_category": {k: dict(v) for k, v in categories.items()}, "per_judge": judges,
        "usage_by_stage": {k: dict(v) for k,v in usage.items()}, "unpreserved_count": len(omissions),
        "annotation_and_judges_are_human_gold": False, "independent_task_count": 5, "runs_per_group_per_task": 1,
        "uncertainty": "未做多随机重复，不把检查点或两位裁判当独立任务样本；不作总体显著性或无损结论。"}
    report["retention_review_method"] = read(out / "摘要保留复核说明.json")
    report["quality_score_policy"] = quality_policy
    report['quality_scope_scores_identical'] = all(g['dimensions']['overall']==g['dimensions']['continued'] for r in judges for g in r['groups'].values())
    report["macro_full_retention_rate"] = statistics.mean(r['full_retention_rate'] for r in results)
    original_usage = usage['原文续写']['total_tokens']
    compression_usage = usage['压缩续写']['total_tokens'] + usage['摘要生成']['total_tokens']
    report["continuation_plus_summary_usage"] = {"original_tokens": original_usage, "compressed_tokens_including_summary": compression_usage,
        "token_reduction": 1-compression_usage/original_usage,
        "boundary": "仅API返回usage的断点后调用，压缩侧包含摘要生成及其重试；不包含离线标注评审，不代表金额节省比例。"}
    save(out / "汇总指标.json", report)
    save(out / "未完整保留信息清单.json", omissions)
    lines = ["# 未完整保留信息逐项分析", "", "完整保留以外的partial、missing、contradicted均列出，部分保留不等于完全丢失。", ""]
    for r in omissions:
        lines += [f"## 题目{r['topic']} / {r['id']} / {r['category']}", "", f"原信息：{r['statement']}", "", f"摘要状态：{r['retention']['status']}。{r['retention']['reason']}", "", f"后续共同输入：{r['reexposure']['status']}。{r['reexposure']['reason']}", "", f"原文定位：{r['source_id']}，字符{r['source_span']['start']}–{r['source_span']['end']}。", ""]
    (out / "未完整保留信息逐项分析.md").write_text("\n".join(lines))
    lines = ["# 通用研究五题历史压缩信息保留与后续报告质量评估", "", "评估日期：2026-11-01，Asia/Singapore。", "",
        "本轮重新调用真实模型，完成压缩前关键信息初标、摘要生成、保留检查和五题两组续写。质量阶段尝试双模型匿名评审，并对qwen3.8-max-0902交换候选顺序复评；由于引句及语义判定存在错误，评分作为探索性结果报告，不当作已验证的质量真值。下述为固定断点、一次摘要的Writer继续写作实验，不是完整研究图重跑。", "",
        "## 核心结果", "", f"共标注{sum(totals.values())}个关键点。摘要完整保留{totals['preserved']}项（{pct(totals['preserved'],sum(totals.values()))}），部分保留{totals['partial']}项，完全遗漏{totals['missing']}项，明确改错{totals['contradicted']}项。", "",
        "|题号|关键点|完整|部分|遗漏|改错|完整保留率|历史输入减少率|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        c=Counter(r["counts"])
        lines.append(f"|{r['topic']}|{r['points']}|{c['preserved']}|{c['partial']}|{c['missing']}|{c['contradicted']}|{100*r['full_retention_rate']:.1f}%|{100*r['history_reduction']:.1f}%|")
    lines += ["", "本次最明确的结果是：研究目标保留较好，适用条件、证据归属和具体操作细节损失较多。10个用户要求全部完整保留，而23个适用条件与不确定性点只有7个完整保留。", "",
        "报告质量没有统一升降方向：同一裁判两种顺序均偏好题31原文组、题35与49压缩组；题16从偏好压缩组变为持平，题46胜者反转。评分本身存在引用与推理错误，不能据此宣布压缩提高了质量或实现无损。后续原有证据和前两章正文会重新提供部分信息，也是不能用最终质量倒推摘要保真度的原因。"]
    lines += ["", "历史输入减少率使用同一本地Qwen tokenizer估算，只比较断点前消息与摘要消息，不是完整请求减少率或费用节省比例。该tokenizer估算与实际服务端prompt_tokens存在差异，运行消耗另用API返回的usage统计。保留率分母是预先标注的关键点，不是原文全部事实。原文组保有全部标注原句，但这不保证执行模型一定正确使用。", "", "## 数据、配置与执行口径", "",
        "五题为16非接触式感知、31中外博物馆教育、35市政污水治理、46房地产可持续发展、49女性正畸与医美需求。各题从已有真实轨迹的首轮前两章完成后分组；此前所有Writer请求和回答均保留到冻结样本。原始档案指纹在样本清单，具体来源批次可能不同。", "",
        f"先冻结全部关键点，再使用项目现有摘要正文提示词和{cfg['summary_model']}压缩历史，摘要输出上限3000 tokens，摘要模型看不到评测清单。原文组与压缩组使用相同{cfg['writer_model']}、温度0、后续第三和第四章任务与证据、写作输出上限16000及统一输入实验上限96000 tokens。", "",
        "初始配置qwen3-max返回model_not_found，未产生可用摘要。实际API验证qwen3.8-max-0902可调用后，本轮将摘要、两组写作及第二裁判改为该固定型号；保留初始配置与失败请求，未修改生产配置。因此本结果不能描述为qwen3-max当前效果。CLI会话已过期，未取得CLI规格或价格；实际执行依据为API可用性检查。", "",
        "194项模型初标经Codex在摘要生成前对照原文复核，剔除重复、低价值和证据不足的附加推断，修正要求、状态及来源边界后冻结144项。完整初标、修正及排除记录均保留。并非对全部历史事实的穷尽标注，也不是独立人工金标。", "",
        "保留检查的qwen-plus初评和重试均出现引句来源混淆：把清单中的原文抄成摘要引句；另有把缺失年龄、场景、数值分母视为隐含保留的语义错误。原始失败响应全部保留。主统计使用Codex在读取报告质量评分前逐项复核的结果，复核脚本固定144项判定与理由，程序验证每条引句确实来自摘要。未更改压缩前清单或分母。同一Codex参与清单和保留复核，因此不是独立人工双盲评价。", "",
        "完整保留要求固定statement中的必要要素均在摘要中；例子或重复换算可省略。某些预先冻结条目仍含多个要素，任一必要要素缺失即记部分保留；相近事实可能分别作为事实项、适用范围项和来源项计分。因此此保留率不可解释为全文相同比例的信息丢失，也不等于模型执行任务的能力比例。", "",
        "前两章正文保持相同。后续两章重新生成，每次写作请求中的已完成章节上下文替换为本组真实产物；报告首尾请求中的章节也来自本组，然后程序组装报告。原运行后续章节正文不作为答案注入。不开放额外原文回读，不执行联网、Critic返工或新的研究规划。本轮只压缩一次，后续新写作消息按原文累积。", "",
        "当前任务输入仍带有其原有证据材料；首尾生成还会再次读到完整章节。因此分别评估摘要本身保留率、后续输入再暴露情况、整篇报告质量和新增部分质量。统一输入实验上限用于运行原文对照，不等于生产默认输入预算。", "",
        "## 未完整保留的内容类型", "", "|类别|总数|完整|部分|遗漏|改错|完整保留率|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,c in categories.items():
        lines.append(f"|{name}|{sum(c.values())}|{c['preserved']}|{c['partial']}|{c['missing']}|{c['contradicted']}|{pct(c['preserved'],sum(c.values()))}|")
    lines += ["", "下面展示各题未完整保留的信息；详细原句、来源、摘要引句、判定理由和全部遗漏项保存在逐项结果中。", ""]
    for topic in TOPICS:
        rows=[r for r in omissions if r["topic"]==topic]
        rows.sort(key=lambda r: ({"contradicted":0,"missing":1,"partial":2}[r["retention"]["status"]],r["importance"]!="critical"))
        lines += [f"### 题目{topic}", ""]
        for r in rows[:4]:
            lines += [f"- {r['id']}（{r['retention']['status']}）：{r['statement']} 摘要检查：{r['retention']['reason']}"]
        if not rows:lines.append("本次已标注清单中未发现未完整保留项，不代表全文无遗漏。")
        lines += [""]
    lines += ["## 遗漏信息是否由后续输入补回", "", "qwen-plus对共同长输入的语义核查存在漏读。例如题31将非国有馆制度障碍、建川/观复案例判为不存在，但程序能精确定位其原标注引句。因此不把该裁判的full/partial/absent比例当成可靠补回率，原始评审保留供复查。下面只报告程序可验证的原标注引句再现情况，是保守的文本核查，不是全部语义补回率。", "", "|题号|未完整保留数|完整原标注引句再现|尚未证实|", "|---|---:|---:|---:|"]
    for r in results:
        e=Counter(r["unpreserved_reexposure"])
        lines.append(f"|{r['topic']}|{sum(e.values())}|{e['source_quote_reappears']}|{e['unconfirmed']}|")
    lines += ["", "该表检查第三、第四章两份共同输入的并集，不保证某条信息在模型首次需要它之前就已出现，也不代表模型实际使用。尚未证实不能解读为未提供。首尾阶段完整章节的再次提供属于另一个补回渠道，不能把报告质量稳定全部归因于摘要保真。", "", "## 报告质量配对评审", "",
        f"计划裁判分别为{'和'.join(cfg['judge_models'])}，均另起上下文、不提供组名。qwen-plus四题两次未通过引句校验，唯一通过校验的题31又把两组相同旧章节误作差异，故不采用其分数作质量结论。对qwen3.8-max-0902另外交换候选顺序：模型、提示词、资料、报告都不变，仅交换候选编号与顺序。两种顺序一律取第一次完整数值响应，避免一边取修复后评分、一边取初评分。", "",
        "六维度为需求覆盖、事实与证据一致性、条件与不确定性、分析深度、跨章节一致性、表达与组织，各1至10分。下表为六维均分，差值=压缩组减原文组。qwen3.8原顺序五次初评中只有题31通过引句校验，其他初评含改写、拼接或虚构证据引句；交换顺序调用仅验证数值结构。因此这些分数可展示裁判倾向及顺序敏感性，不能直接视为已验证的报告质量差异。原始分数与错误均保留，没有人工改分。", "",
        "|题号|候选顺序|原文新增部分|压缩新增部分|新增部分差值|整篇差值|偏好|", "|---|---|---:|---:|---:|---:|---|"]
    for r in judges:
        if not r['included_in_primary']:continue
        lines.append(f"|{r['topic']}|{r['order']}|{r['groups']['原文组']['continued']:.2f}|{r['groups']['压缩组']['continued']:.2f}|{r['continued_delta']:+.2f}|{r['overall_delta']:+.2f}|{r['winner']}|")
    for model in dict.fromkeys(r["model"] for r in judges if r['included_in_primary']):
        rows=[r for r in judges if r["model"]==model and r['included_in_primary']]
        lines += ["", f"{model}：五题、每题两种顺序的新增部分平均差值{statistics.mean(r['continued_delta'] for r in rows):+.3f}分，整篇平均差值{statistics.mean(r['overall_delta'] for r in rows):+.3f}分。两种顺序是同一报告的重复评分，不增加独立任务数。该均值仅是原始裁判输出的描述统计，不是质量提升的可靠估计。"]
    lines += ["", "|题号|原顺序差值|交换顺序差值|两顺序均值|胜者是否一致|", "|---|---:|---:|---:|---|"]
    for r in results:
        a,b=r['quality'];lines.append(f"|{r['topic']}|{a['continued_delta']:+.2f}|{b['continued_delta']:+.2f}|{r['quality_order_averaged_delta']:+.2f}|{'是' if r['same_winner_across_orders'] else '否'}|")
    if report['quality_scope_scores_identical']:
        lines += ["", "另一个评分局限：裁判对所有候选给出的整篇与新增部分六维分数完全相同。虽然输入和评分字段按两种范围分开了，裁判实际没有形成可用的范围区分，不能认为已经可靠排除了共同前两章对分数的影响。"]
    lines += ["", "### 对具体报告差异的文本复核", "", "以下由Codex直接核对生成文本与冻结材料，不是外部事实核验，也未据此人工修改模型分数。"]
    for case in read(out/'报告质量差异实质复核.json')['cases']:
        lines += ["",f"- 题目{case['topic']}：{case['finding']}"]
    lines += ["", "### 逐题质量变化与失败样例", ""]
    for topic in TOPICS:
        lines += [f"#### 题目{topic}", ""]
        for r in [r for r in judges if r["topic"]==topic and r['included_in_primary']]:
            lines += [f"- {r['model']}，{r['order']}（模型原始意见，未经全部语义核验）：{r['comparison']}"]
        lines += [""]
    lines += ["## 资源消耗与执行失败", "", "|阶段|请求尝试|已接受|失败|输入token|输出token|缺少usage|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,u in usage.items():
        lines.append(f"|{name}|{u['attempts']}|{u['accepted']}|{u['failed']}|{u['prompt_tokens']}|{u['completion_tokens']}|{u['requests_missing_usage']}|")
    lines += ["", f"按返回usage的实际请求合计：原文组续写{original_usage:,} tokens；压缩组续写加摘要生成及其重试{compression_usage:,} tokens，减少{100*(1-compression_usage/original_usage):.1f}%。这里同时计入输入和输出token，模型单价不同且没有冻结价格，不能换算为相同比例的费用节省。"]
    lines += ["", "表中失败包含模型不可用、格式不符和引句校验失败，不全是API故障；两组30次写作请求全部完成，10份报告均已保存。质量初评即使有完整分数，未通过引句校验仍记失败，其分数只用于明确标注局限的探索性比较。"]
    lines += ["", "以上仅统计断点之后的新增请求，不包括归档任务此前的搜索和前两章生成。压缩组运行开销应将摘要生成与压缩续写相加；离线标注和质量评审另列。没有冻结货币单价，不报告金额或费用下降百分比。无usage的失败响应不推断费用为零。并发请求耗时相加不是用户等待时间；失败尝试和修复记录保存在各阶段模型调用目录。", "",
        "## 局限性、结论和下一步", "", "已测量：上述五个真实断点上的摘要长度、标注关键点保留、共同输入中原标注引句再现、两组新生成报告和匿名模型评分的顺序敏感性。保留复核引用已逐条定位，质量裁判引句并非全部通过校验，语义评价更不能由字符串校验保证。再暴露模型评审存在可直接证明的漏读，未将其比例作为结论。", "",
        "间接推断：结合遗漏清单与评分理由，可以提出哪些损失可能影响写作，但未对单一遗漏做定向恢复的因果实验。报告质量保持或改善也可能来自去除冗余、当前输入补回信息及生成随机波动。", "",
        "尚未评估：全部原文关键信息的穷尽覆盖、外部世界事实真值、独立人工评审一致性、多次摘要误差累积、原文回读补救、完整研究流程和生产默认预算下的交付率。五题每组仅一次，主质量统计只有一名模型裁判，且与写作模型同型号，不作总体显著性或无损结论。", "",
        "结论：本策略能显著减少历史长度，但在本次固定清单上只完整保留一半信息点；多数未完整项属于部分保留，不能说原文一半事实消失。实际报告出现局部优劣变化，尚不能认定稳定质量提升、下降或无损。", "",
        "下一步建议：摘要优先固定保留数字的分母/年龄/场景、否定与证据不足、结论与来源URL对应关系、未完成事项；先让独立人工复核本次72项未完整保留点及五对报告，再在同一断点多次生成。若要判断遗漏的因果影响，可只补回一类条件信息重跑，比较相同题目的质量变化。不要继续仅依赖自动裁判均分决定压缩策略是否上线。", "",
        "## 可追溯原始产物", ""]
    rel = "../" + str(out.relative_to(ROOT))
    for name in ["运行配置.json","样本清单.json","汇总指标.json","摘要保留复核说明.json","保留复核判定表脚本.py","续写完整性与资源复核.json","共同输入原标注引句核对汇总.json","质量评分采用口径.json","再暴露裁判漏读证据.json","质量评审共同章节问题核查.json","报告质量差异实质复核.json","报告质量差异文本复核.json","未完整保留信息清单.json","未完整保留信息逐项分析.md"]:
        lines += [f"- [{name}]({rel}/{name})"]
    for topic in TOPICS:
        lines += [f"- 题目{topic}：[关键信息标注]({rel}/题目{topic}/关键信息标注.md)、[摘要]({rel}/题目{topic}/压缩后历史.md)、[保留逐项结果]({rel}/题目{topic}/信息保留逐项结果.json)、[原文组报告]({rel}/题目{topic}/原文组/完整报告.md)、[压缩组报告]({rel}/题目{topic}/压缩组/完整报告.md)。"]
    target=ROOT/"docs/通用研究五题历史压缩信息保留与后续报告质量评估-20261101.md"
    target.write_text("\n".join(lines)+"\n")
    save(out / "完整性检查.json", {"paired_current_task_hashes_equal": True, "fixed_chapters_equal": True,
        "source_archives_unchanged": all(hashlib.sha256((Path(r['source_folder'])/'智能体原始历史.json').read_bytes()).hexdigest()==r['source_history_sha256'] for r in read(out/'样本清单.json')),
        "completed_topics": len(results), "completed_reports": len(results)*2, "quality_judgments": len(judges)})
    print(json.dumps({"report":str(target),"points":sum(totals.values()),"retention":dict(totals),"quality_delta_by_judge":{m:statistics.mean(r['continued_delta'] for r in judges if r['model']==m) for m in set(r['model'] for r in judges)}},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
