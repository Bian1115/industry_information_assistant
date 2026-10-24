#!/usr/bin/env python3
"""复算已有五题报告指标；不调用模型，不重标历史材料，不覆盖历史评估。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-10-12"
OLD = ROOT / "backend/eval/data/五篇报告Qwen3-maxNLI复评-20261005-132900"
NEW = ROOT / "backend/eval/data/通用研究五题引用事实独立复核与模块一致性-20261007"
CRITIC = ROOT / "backend/eval/data/Critic五十六条意见正确率复核-20261011-201238"
OUT = ROOT / f"backend/eval/data/通用研究已有报告七项指标复算-{DATE}"
DOC = ROOT / f"docs/通用研究已有报告七项指标结果及分析-{DATE}.md"
TITLES = {16: "非接触式感知算法", 31: "中外博物馆教育", 35: "市政污水与雨污协同", 46: "房地产可持续发展", 49: "正畸与医美共同需求"}


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "percent": 100 * numerator / denominator if denominator else None}


def fmt(value):
    n, d, p = value["numerator"], value["denominator"], value["percent"]
    return f"{p:.2f}%（{n}/{d}）" if p is not None else "不适用（分母为0）"


def link(path, title=None):
    return f"[{title or path.name}]({path})"


def calculate():
    sources = [NEW / "逐条独立复核及模块对齐.json", NEW / "独立复核与模块一致性指标.json",
               NEW / "冻结论断与原始证据.json", NEW / "外部事实重点核查.json",
               OLD / "逐条初评与复核.json", OLD / "指标汇总.json", OLD / "冻结输入.json",
               OLD / "外部核查与评审复核.json", OLD / "审核问题初评.json",
               CRITIC / "逐条判定.json", CRITIC / "正确率汇总.json"]
    rows = read(sources[0])
    stored = read(sources[1])
    material = read(sources[2])
    external_new = read(sources[3])["rows"]
    old_rows = read(sources[4])
    old_summary = read(sources[5])
    tasks = read(sources[6])
    external_old = read(sources[7])["cases"]
    audits = read(sources[8])
    critic_rows = read(sources[9])
    critic_stored = read(sources[10])
    body_old = [r for r in old_rows if r["kind"] == "body" and r["type"] == "factual"]
    by_id = {r["id"]: r for r in rows}
    old_by_id = {r["id"]: r for r in body_old}
    material_by_id = {r["id"]: r for r in material}
    assert len(rows) == len(by_id) == len(body_old) == len(material) == 511
    assert set(by_id) == set(old_by_id) == set(material_by_id)
    for r in rows:
        assert r["claim"] == old_by_id[r["id"]]["claim"] == material_by_id[r["id"]]["claim"]
        assert len(r["ref_support"]) == len(r["references"]) == len(old_by_id[r["id"]]["refs"])
        assert {s["key"] for s in r["ref_support"]} == set(r["references"])
        assert not r["references"] or material_by_id[r["id"]]["citations_explicit"]
    reports = []
    for t in tasks:
        path = Path(t["folder"]) / "研究报告.md"
        frozen_hash = hashlib.sha256(t["report"].encode()).hexdigest()
        assert frozen_hash == t["report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(path)
        reports.append({"topic": t["id"], "path": str(path), "sha256": frozen_hash})

    # 合并同一正文的两轮定向事实核查，以论断ID去重，不传播至未核查的重复句。
    errors = {}
    for case in external_old:
        if not case["status"].startswith("confirmed"):
            continue
        for cid in case.get("claim_ids", []) + case.get("additional_error_ids", []):
            if cid in by_id:
                errors.setdefault(cid, []).append({"source": str(sources[7]), "case_id": case["id"],
                    "reason": case["summary"], "urls": case.get("urls", []),
                    "local_source_id": case.get("local_source_id")})
    assert set(errors) == {r["id"] for r in body_old if r["confirmed_error"]}
    old_error_ids = set(errors)
    for record in external_new:
        if record["reality_status"] == "incorrect":
            assert record["id"] in by_id
            errors.setdefault(record["id"], []).append({"source": str(sources[3]),
                "cluster": record["cluster"], "reason": record["reason"], "sources": record["sources"]})
    contradiction_ids = {r["id"] for r in rows if r["joint_support"] == "contradicted"}
    error_or_contradiction_ids = set(errors) | contradiction_ids

    def body_metrics(rr):
        labels = Counter(r["joint_support"] for r in rr)
        rel_labels = Counter(s["label"] for r in rr for s in r["ref_support"])
        n = len(rr)
        cited = sum(bool(r["references"]) for r in rr)
        pair_n = sum(rel_labels.values())
        definite = [r for r in rr if r["references"] and r["joint_support"] != "unknown"]
        confusion = Counter()
        for r in definite:
            ref = r["joint_support"] == "full"
            pred = r["module_status"] == "supported"
            confusion[(ref, pred)] += 1
        tp, fn, fp, tn = [confusion[k] for k in [(True, True), (True, False), (False, True), (False, False)]]
        ids = {r["id"] for r in rr}
        return {"facts": n, "cited_facts": cited, "citation_pairs": pair_n,
                "labels": dict(labels), "reference_labels": dict(rel_labels),
                "citation_support": rate(rel_labels["full"], pair_n),
                "cited_claim_support": rate(labels["full"], cited),
                "citation_support_determinable_pairs": rate(rel_labels["full"], pair_n - rel_labels["unknown"]),
                "effective_evidence_coverage": rate(labels["full"], n),
                "strict_unsupported": rate(n - labels["full"], n),
                "unsupported_label_only": rate(labels["unsupported"], n),
                "unsupported_or_contradicted": rate(labels["unsupported"] + labels["contradicted"], n),
                "confirmed_error_lower_bound": rate(len(ids & errors.keys()), n),
                "source_contradiction": rate(labels["contradicted"], n),
                "confirmed_error_or_source_contradiction_lower_bound": rate(len(ids & error_or_contradiction_ids), n),
                "hallucination_rate": {"value": None, "status": "尚未评估总体现实事实幻觉率",
                    "reason": "511条论断未穷尽现实事实核查；无引用、partial、unknown不自动代表虚假。"},
                "citation_verifier": {"assessable_cited_n": len(definite),
                    "confusion_support_positive": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
                    "accuracy_against_independent_review": rate(tp + tn, len(definite)),
                    "insufficient_support_detection_recall": rate(tn, tn + fp)}}

    total = body_metrics(rows)
    per_topic = {str(t): body_metrics([r for r in rows if r["topic"] == t]) for t in TITLES}
    assert total["labels"] == stored["total"]["labels"]
    assert total["reference_labels"] == stored["reference_relation_support"]
    assert total["facts"] == stored["total"]["facts"]
    assert total["citation_pairs"] == stored["total"]["citation_pairs"]
    assert total["citation_verifier"]["accuracy_against_independent_review"]["numerator"] == 180
    assert len(old_error_ids) == 8 and len(errors) == 11 and len(error_or_contradiction_ids) == 12

    pre = [r for r in old_rows if r["kind"] == "pre"]
    confirmed_pre = [r for r in pre if r["confirmed_error"]]
    audit_counts = Counter((r.get("model_review") or {}).get("validity", "unknown") for r in audits)
    same_run_audit = {"pre_review_sample_n": len(pre), "confirmed_pre_error_n": len(confirmed_pre),
        "confirmed_pre_error_rows": [{k: r[k] for k in ["id", "topic", "claim", "verified_first_detected", "verified_any_detected"]} for r in confirmed_pre],
        "first_error_detection_recall": rate(sum(r["verified_first_detected"] for r in confirmed_pre), len(confirmed_pre)),
        "any_round_error_detection_recall": rate(sum(r["verified_any_detected"] for r in confirmed_pre), len(confirmed_pre)),
        "issue_n": len(audits), "issue_labels": dict(audit_counts),
        "issue_validity_proxy": rate(audit_counts["valid"], len(audits) - audit_counts["uncertain"] - audit_counts["unknown"]),
        "issue_validity_unknown_as_not_valid": rate(audit_counts["valid"], len(audits)),
        "interpretation": "审核初评模型的意见有效率代理，不是人工金标Accuracy；4条已确认预审错误的检出率不外推全报告。"}
    assert len(pre) == 198 and len(confirmed_pre) == 4
    assert len(audits) == 52 and audit_counts["valid"] == 42 and audit_counts["uncertain"] == 1
    for m in old_summary["metrics"]:
        topic_pre = [r for r in confirmed_pre if r["topic"] == m["id"]]
        assert sum(r["verified_first_detected"] for r in topic_pre) == m["audit_first_confirmed_recall"]["numerator"]
        assert sum(r["verified_any_detected"] for r in topic_pre) == m["audit_any_confirmed_recall"]["numerator"]

    def critic_metrics(rr):
        c = Counter(r["finding_correctness"] for r in rr)
        counts = {label: c[label] for label in ["correct", "partial", "incorrect", "unjudgeable"]}
        return {"n": len(rr), "counts": counts, "strict_issue_correctness": rate(c["correct"], len(rr)),
                "at_least_partly_valid": rate(c["correct"] + c["partial"], len(rr))}
    other_critic = {"run": critic_stored["previous_run"], "evaluation_date": "2026-10-11",
        "total": critic_metrics(critic_rows),
        "per_topic": {str(t): critic_metrics([r for r in critic_rows if r["task_id"] == t]) for t in TITLES},
        "interpretation": "另一批2026-10-09重新生成报告的14次审核，独立复核56条意见；不能与2026-10-03正文指标或原4条错误检出率拼为同一运行。"}
    assert other_critic["total"]["counts"] == critic_stored["total"]["counts"]
    assert len(critic_rows) == len({r["key"] for r in critic_rows}) == 56

    history = {k: rate(sum(m[k]["numerator"] for m in old_summary["metrics"]),
                           sum(m[k]["denominator"] for m in old_summary["metrics"]))
               for k in ["citation_support", "effective_evidence_coverage", "unsupported_strict", "confirmed_error_lower_bound"]}
    details = [{"id": r["id"], "topic": r["topic"], "claim": r["claim"],
                "joint_support": r["joint_support"], "reason": r["reason"], "ref_support": r["ref_support"],
                "module_status": r["module_status"], "confirmed_error": r["id"] in errors,
                "confirmed_error_evidence": errors.get(r["id"], []),
                "confirmed_error_or_source_contradiction": r["id"] in error_or_contradiction_ids,
                "historical_joint_support": old_by_id[r["id"]]["judgment"]["joint_support"]} for r in rows]
    changed = Counter((r["historical_joint_support"], r["joint_support"]) for r in details
                      if r["historical_joint_support"] != r["joint_support"])
    fingerprints = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "bytes": p.stat().st_size} for p in sources]
    fingerprints.append({"path": str(Path(__file__).resolve()),
                         "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    result = {"evaluation_date": DATE, "timezone": "Asia/Singapore", "scope": list(TITLES),
        "method": "复算保存标签，合并同一正文已确认错误ID；不调用模型、不新外查、不改变原标签。",
        "support_annotation_date": "2026-10-07", "original_report_date": "2026-10-03",
        "review_origin": "独立Codex助手复核，非人工金标、非严格盲评",
        "total": total, "per_topic": per_topic, "same_report_critic": same_run_audit,
        "separate_run_critic": other_critic, "historical_20261004_05": history,
        "support_label_changes": [{"old": a, "new": b, "n": n} for (a, b), n in sorted(changed.items())],
        "confirmed_error_ids": sorted(errors), "confirmed_error_added_ids": sorted(set(errors) - old_error_ids),
        "source_contradiction_ids": sorted(contradiction_ids),
        "confirmed_error_or_source_contradiction_ids": sorted(error_or_contradiction_ids),
        "reports": reports}
    checks = {"status": "通过", "checks": ["511条论断ID、文本、引用关系及显式引用对齐",
        "五份报告SHA-256与冻结正文一致", "最新标签分布与历史机器指标一致",
        "8条旧错误与原confirmed_error一致，合并后11条，错误或来源矛盾并集12条",
        "4条预审错误检出计数与原指标一致", "52条历史审核及1条未知保留",
        "另批56条审核ID唯一，分布与原正确率指标一致"], "failures": [],
        "prior_validation_attempts": [{"attempt": 1, "status": "中断",
            "cause": "Counter省略零值unjudgeable键，字典完整性比较失败；补全零值后重新核对，未改标签或指标口径。"}]}
    return result, details, fingerprints, checks


def render(m):
    total = m["total"]
    same = m["same_report_critic"]
    separate = m["separate_run_critic"]
    body_table = "\n".join(
        f"| {t} {title} | {v['facts']} | {fmt(v['citation_support'])} | {fmt(v['effective_evidence_coverage'])} | {fmt(v['strict_unsupported'])} | {fmt(v['confirmed_error_lower_bound'])} | {fmt(v['source_contradiction'])} |"
        for t, title in TITLES.items() for v in [m["per_topic"][str(t)]])
    critic_table = "\n".join(
        f"| {t} | {v['counts'].get('correct', 0)} | {v['counts'].get('partial', 0)} | {v['counts'].get('incorrect', 0)} | {fmt(v['strict_issue_correctness'])} |"
        for t in TITLES for v in [separate["per_topic"][str(t)]])
    errors_table = "\n".join(f"| {t} | {', '.join(cid for cid in m['confirmed_error_ids'] if cid.startswith(str(t)+'-')) or '本次没有确认'} |" for t in TITLES)
    history = m["historical_20261004_05"]
    report_paths = "\n".join(f"- 第{r['topic']}题：{link(Path(r['path']), '原始研究报告')}。" for r in m["reports"])
    return f"""# 通用研究已有报告七项指标结果及分析

评估日期：{DATE}，Asia/Singapore。本轮为新建复算分析，不覆盖历史记录。

评估对象：通用研究第16、31、35、46、49题于2026-10-03完成的五篇报告，共511条正文事实论断、293条带引用论断、296组论断–来源关系。正文支持指标以2026-10-07逐条独立复核为主，保留未知。旧版2026-10-04/05支持裁判结果只作口径对照。评审者为Codex助手，非人工金标、非预注册盲评。

本轮只复算已有材料，未重生成报告、未调用裁判模型、未新做网络事实核查。明确错误计数合并两轮对同一正文的既有定向核查，并按论断ID去重；没有把一处错误扩散至未核查的同义句。审核意见正确率另有2026-10-09运行、2026-10-11复核的数据，单独呈现，不能与原五篇报告合成同一批结果。

## 一、七项指标与公式

| 用户所问指标 | 本轮结果 | 分子、分母及边界 |
| --- | --- | --- |
| 引用支持率 | **{fmt(total['citation_support'])}** | 单个来源完整支持的论断–来源关系 / 全部296组关系；多来源逐个计，64组unknown留在分母。 |
| 有效证据覆盖率 | **{fmt(total['effective_evidence_coverage'])}** | 有显式引用且联合证据完整支持的论断 / 全部511条正文事实论断。 |
| 无支持论断率（严格口径） | **{fmt(total['strict_unsupported'])}** | 1−有效证据覆盖率；包含部分支持、无绑定引用、来源不支持、来源矛盾和无法判断。不是现实假事实率。 |
| 明确错误／矛盾率 | **已确认现实错误下限 {fmt(total['confirmed_error_lower_bound'])}**；若同时包含来源直接矛盾，则并集下限 **{fmt(total['confirmed_error_or_source_contradiction_lower_bound'])}** | 11条既有核查确认错误 / 511；5条所引来源直接矛盾中4条与已确认错误重合，并集为12条，不能将11+5相加。 |
| 审核错误检出率（Critic） | **首轮 {fmt(same['first_error_detection_recall'])}；全轨迹累计 {fmt(same['any_round_error_detection_recall'])}** | 分母仅4条已确认预审错误，非全部198条预审样本，也非11条最终残留错误。 |
| 幻觉率（现实事实口径） | **尚不能可靠估计总体值**；已发现事实错误下限为 {fmt(total['confirmed_error_lower_bound'])} | 缺少全511条现实事实真值。若将“幻觉”操作性定义成任何未充分支持论断，数值即72.02%，但只能称严格无支持率；不能改写为72.02%事实为假。 |
| 审核准确率 | **原报告Critic意见有效率代理 {fmt(same['issue_validity_proxy'])}**；另批Critic严格意见正确率 **{fmt(separate['total']['strict_issue_correctness'])}**；引用核验分类准确率 **{fmt(total['citation_verifier']['accuracy_against_independent_review'])}** | 三种不同口径：原报告意见有效率是模型初评且排除1条未知；另批27/56是Codex独立复核意见正确比例；180/230是引用核验模块相对独立复核的二分类一致率。前两项按意见计算，性质更接近Precision，不能冒充带TN的分类Accuracy。 |

全局均按论断、引用关系或意见条数加权，不平均五个百分比。“引用支持率”144/296不同于“带引用论断完整支持率”143/293＝48.81%，也不同于引用是否存在的293/511＝57.34%。排除未知引用关系后的支持率为{fmt(total['citation_support_determinable_pairs'])}，是敏感性分析，不能替代保留未知的主口径。

## 二、正文逐题结果

| 题目 | 正文论断 | 引用支持率 | 有效证据覆盖率 | 严格无支持率 | 已确认现实错误下限 | 所引来源直接矛盾率 |
| --- | ---: | --- | --- | --- | --- | --- |
{body_table}

- 第16题：27条完整支持，21条部分支持，6条未知；算法任务对象和预处理限定容易被改写。已确认近红外字母图案识别被写成静态手势识别。其余35条无绑定引用，另有2条有引用但不支持。
- 第31题：33条完整支持，26条未知，72条无引用。88.6%免费开放比例被当成国有馆比例，16.4亿人次的统计总体混用；后续还确认大运河博物馆城市错误。证据支持与现实真假必须分别判断。
- 第35题：21条完整支持，14条未知；二进制DOC和残缺内容影响取证。旧报批稿不能证明标准当前尚未实施。并未将缺成本测算或政策建议风险直接计为事实错误。
- 第46题：47条完整支持，为五题最高；18条部分支持主要涉及计划、状态与范围。没有确认现实错误不代表整篇正确；本次对应已确认预审错误分母为0，检出率须记不适用。
- 第49题：15条完整支持、15条部分支持、4条来源矛盾、12条未知。年龄范围、女性内部占比和总消费者分母混用，且独立市场人群相似不能证明个体共同需求比例。严格无支持率最高。

### 支持标签拆分

511条论断＝143条full＋76条partial＋224条unsupported＋5条contradicted＋63条unknown。224条unsupported中218条没有绑定行内引用、6条有引用但不支持。因此368条严格未充分支持＝218无引用＋76部分支持＋6有引用但不支持＋5来源矛盾＋63无法判断。

若只统计字面unsupported标签，为{fmt(total['unsupported_label_only'])}；若统计unsupported与contradicted，为{fmt(total['unsupported_or_contradicted'])}。二者均不能代替72.02%的严格口径，也不能直接作为现实幻觉率。218条无引用尚未逐条判断引用必要性；段落末引用未自动扩展到段内全部事实。

### 明确错误的可追溯计数

旧版定向核查确认8条，2026-10-07新增3条且有1条重复确认国标状态；按ID去重后11条。新增为16-body-047、31-body-115、49-body-033。支持标签unknown不撤销另轨已确认的现实错误，支持标签full也不保证来源自身正确。

| 题号 | 11条已确认错误ID |
| --- | --- |
{errors_table}

所引来源矛盾ID：{', '.join(m['source_contradiction_ids'])}。其中49-body-052尚无独立外部现实核查，故只加入“明确错误或来源矛盾”并集，未加入11条现实错误。每条错误的来源记录、既有理由和原始核查路径见本轮逐条机器结果。建议、效益缺证、统计分母待核及窗口损坏保留各自边界，不升级成确认现实错误。

## 三、审核检出与审核准确率

### 原五篇报告的Critic

198条结构化预审样本中，已确认错误仅4条：35-pre-012、31-pre-008、31-pre-049、49-pre-029。首轮全部漏检；后续只检出49-pre-029的年龄/分母问题，因此首轮0/4、累计1/4。重复的88.6%错误按两次论断出现计，去重错误家族只有3类；本轮保留历史按论断出现次数的分母。新增的最终正文错误没有已冻结的对应预审核查与检出标注，不能擅自加入召回分母。

原52条Critic意见，模型初评42条valid、9条invalid、1条uncertain。排除未知的意见有效率为42/51＝82.35%；未知按未证有效保留时为42/52＝80.77%。这属于历史模型代理，不是独立人工金标或严格Accuracy，且不能证明找全了报告错误。报告中的resolved标记也不作为修正成功证据。

### 另一批Critic五题消融的独立复核

这批在2026-10-09重新生成报告，2026-10-11按每条意见实际看到的当轮正文直接审读；14次审核共56条意见，正确27、部分正确15、错误14、无法判断0。严格意见正确率27/56＝48.21%，至少部分有效42/56＝75.00%。部分正确没有折算半分。

| 题号 | 正确 | 部分正确 | 错误 | 严格意见正确率 |
| --- | ---: | ---: | ---: | --- |
{critic_table}

14条错误意见中10条漏读当轮已经存在的引用；部分正确意见常混有错误位置、分母或建议。该批未建立全部待检错误金标，**尚未评估其检出召回率**。48.21%与82.35%来自不同报告、不同判定者和不同有效性口径，不据此推断版本下降34.14个百分点。

### 引用核验模块的分类准确率

若“审核准确率”指引用核验模块，在230条有引用、独立复核可判断的论断上，完整支持为正类：TP131、FN12、FP38、TN49。Accuracy＝(131+49)/230＝78.26%。模块unknown并入未通过/待复核，复核unknown63条排除主分类指标；218条无引用不纳入语义主指标。

对87条未完整支持论断，核验模块挡下49条，检出召回49/87＝56.32%。它检测的是证据不足，包括partial，不能称现实错误或幻觉检出率。原五条轨迹在线引用核验全部关闭；上述为2026-10-06生产模块离线冻结回放，非当时实际在线表现。

## 四、旧版结果对照与本次变更口径

| 指标 | 2026-10-04/05历史裁判与复核 | 本轮采用2026-10-07独立复核/错误合并 |
| --- | --- | --- |
| 引用支持率 | {fmt(history['citation_support'])} | {fmt(total['citation_support'])} |
| 有效证据覆盖率 | {fmt(history['effective_evidence_coverage'])} | {fmt(total['effective_evidence_coverage'])} |
| 严格无支持率 | {fmt(history['unsupported_strict'])} | {fmt(total['strict_unsupported'])} |
| 已确认现实错误下限 | {fmt(history['confirmed_error_lower_bound'])} | {fmt(total['confirmed_error_lower_bound'])} |

五份报告的SHA-256与历史冻结报告全部一致，差异来自支持性判定和事实核查覆盖更新，**不是重新生成后的报告质量变化**。本次支持率不取两个评估者的平均值；逐条标签迁移写入指标汇总，原标签保持不变。错误下限增加表示多发现了3条已存在错误，不表示报告产生了3条新错误。

## 五、原始路径与本轮产物

原始报告：

{report_paths}

原始机器结果：

- {link(NEW / '逐条独立复核及模块对齐.json')}、{link(NEW / '冻结论断与原始证据.json')}、{link(NEW / '独立复核与模块一致性指标.json')}：最新511条支持复核及引用核验模块对齐。
- {link(NEW / '外部事实重点核查.json')}、{link(OLD / '外部核查与评审复核.json')}：两轮定向现实核查。
- {link(OLD / '逐条初评与复核.json')}、{link(OLD / '审核问题初评.json')}、{link(OLD / '指标汇总.json')}：历史正文支持与原Critic检出标注。
- {link(CRITIC / '逐条判定.json')}、{link(CRITIC / '正确率汇总.json')}：另批56条Critic意见复核。

本轮新增：

- {link(OUT / '指标汇总.json')}：每项分子/分母、逐题指标、批次边界、旧版对照与标签迁移。
- {link(OUT / '逐条统计与错误溯源.json')}：511条论断、引用关系标签与11条错误的核查来源。
- {link(OUT / '来源文件校验.json')}：输入、报告及复算脚本SHA-256。
- {link(OUT / '完整性核对.json')}：对齐、来源指纹、计数和历史指标一致性检查。
- {link(Path(__file__).resolve(), '复算脚本')}：只使用Python标准库；可用 `python3 backend/eval/scripts/复算通用研究已有报告七项指标.py --check-only` 重新核对，默认运行拒绝覆盖历史产物。

## 六、测量边界、结论与下一步

**已测量：** 已有标签下的七项指标可用部分、511条正文及296组引用关系的计数、历史错误合并、原4条预审错误检出、原52条意见模型代理、另批56条意见的独立复核正确率、离线核验分类一致性；输入对齐与SHA-256核对全部通过。

运行核对记录：首次计算在完整性比较处中断，因为Counter省略了0条unjudgeable的键，历史汇总保留该零值键。补齐零值后重新核对通过；未修改历史判定、原始来源或统计口径。中断原因保存在本轮完整性核对JSON中。

**间接推断：** 第49题统计对象与年龄条件、第31题统计总体、第35题取证质量是优先改进方向；Critic漏读已有引用及核验模块放过partial可能削弱审核价值。已有样例支持这些机制解释，未测量改造的因果收益。

**尚未评估：** 全511条现实事实幻觉率、穷尽现实错误率、人工金标Accuracy、全报告Critic召回、另批56条意见所在报告的同口径全文支持指标、在线核验收益与优化前后提升。五题非随机，论断和意见存在重复且题内相关；不将511条或56条视为独立任务，不报告总体显著性或随机抽样置信区间。引用关联有歧义、取证窗口不全、论断抽取可能遗漏，均限制可外推范围。

结论：按最新可追溯支持标签，引用支持48.65%、有效证据覆盖27.98%、严格无支持72.02%；合并定向核查发现11条现实错误，下限2.15%，含来源直接矛盾的并集为12条、2.35%。Critic对原4条已确认预审错误首轮未检出、累计检出1条。审核意见正确率与引用分类准确率须分别报告；真实总体幻觉率不能由证据不足率反推。

下一步建议先按任务冻结同一报告版本与标注口径，优先复核统计分母、时间状态和未知来源；对全部应核查原子事实做现实真值双标与仲裁，才能估计总体事实幻觉率。对同一待审稿建立全部待检错误和正确难例，再同时测Critic意见Precision、错误Recall及修复落实率，避免把不同批次的结果拼成单次实验。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只复算核对，不创建或覆盖产物")
    args = parser.parse_args()
    result, details, fingerprints, checks = calculate()
    if args.check_only:
        if (OUT / "指标汇总.json").exists():
            assert read(OUT / "指标汇总.json") == result
            assert read(OUT / "逐条统计与错误溯源.json") == details
            assert read(OUT / "来源文件校验.json") == fingerprints
            assert read(OUT / "完整性核对.json") == checks
            assert DOC.read_text() == render(result)
        print(json.dumps({"status": "通过", "total": result["total"]}, ensure_ascii=False))
        return
    assert not OUT.exists() and not DOC.exists(), "为保留历史，拒绝覆盖已有复算目录或报告"
    OUT.mkdir(parents=True)
    dump(OUT / "指标汇总.json", result)
    dump(OUT / "逐条统计与错误溯源.json", details)
    dump(OUT / "来源文件校验.json", fingerprints)
    dump(OUT / "完整性核对.json", checks)
    DOC.write_text(render(result))
    print(json.dumps({"status": "通过", "report": str(DOC), "results": str(OUT), "total": result["total"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
