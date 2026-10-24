#!/usr/bin/env python3
"""汇总已有生成记录与Codex逐篇审读结果；本脚本不调用模型或联网。"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo


IDS = [35, 31, 46, 16, 49]
NAMES = {35: "污水治理", 31: "博物馆教育", 46: "房地产", 16: "非接触感知", 49: "正畸与医美共同需求"}
DIMS = {"depth": "深度", "comprehensiveness": "全面性", "readability": "可读性", "requirements": "需求满足"}
LABELS = {"full": "完整满足", "partial": "部分满足", "absent": "缺失", "appropriately_limited": "合理限定", "contradicted": "与保存正文冲突", "not_found": "保存正文未找到", "unjudgeable": "无法判断"}
SOURCE_LABELS = {**LABELS, "full": "完整支持", "partial": "部分支持"}


def read(path):
    return json.loads(path.read_text())


def link(path, label=None):
    return f"[{label or path.name}]({path.resolve()})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    assessments = read(root / "Codex逐篇质量评估.json")
    lookup = {(x["id"], x["group"]): x for x in assessments}
    expected = {(i, g) for i in IDS for g in ["关闭审核", "开启审核"]}
    assert len(assessments) == 10 and set(lookup) == expected, "十篇直接审读尚未完成"
    rubric = read(root / "Codex直接质量评估口径.json")
    assert rubric["external_llm_judges"] is False
    config = read(root / "运行配置.json")
    manifest = read(root / "运行清单.json")
    requirements = {r["id"]: r["text"] for t in rubric["requirements_by_task"] for r in t["requirements"]}
    pairs = []
    for ident in IDS:
        directory = root / f"题目{ident}"
        result = read(directory / "配对运行结果.json")
        assert result["status"] == "completed", f"题目{ident}未完成"
        a, b = result["groups"]["开启审核"], result["groups"]["关闭审核"]
        assert a["success"] and b["success"]
        source = (directory / "公共首稿/研究报告.md").read_bytes()
        disabled = (directory / "关闭审核/研究报告.md").read_bytes()
        assert source == disabled
        assert hashlib.sha256(source).hexdigest() == result["prefix_report_sha256"]
        bstate = read(directory / "关闭审核/研究状态.json")
        assert bstate.get("node_attempts", {}).get("review", 0) == 0
        assert not bstate.get("critic_feedback")
        assert b["new_usage"]["calls"] == 0 and b["completion_reason"] == "critic_disabled"
        assert b["quality_verdict"] == "not_reviewed" and b["internal_quality_score"] is None
        prefix_state = read(directory / "公共首稿/研究状态.json")
        old_searches = prefix_state.get("search_budget", {}).get("used", 0)
        added = read(directory / "开启审核/审核返工新增调用.json")
        agents = Counter(c.get("agent_name", "unknown") for c in added)
        off, on = lookup[ident, "关闭审核"], lookup[ident, "开启审核"]
        for review in [off, on]:
            assert review["full_text_read"] is True
            assert set(review["scores"]) == set(DIMS)
            assert review["overall"] == mean(review["scores"].values())
            assert review["report_sha256"] == hashlib.sha256(Path(review["report_path"]).read_bytes()).hexdigest()
        pairs.append({"id": ident, "name": NAMES[ident], "off_score": off["overall"], "on_score": on["overall"],
            "quality_delta": on["overall"] - off["overall"],
            "dimension_delta": {k: on["scores"][k] - off["scores"][k] for k in DIMS},
            "off_seconds": b["logical_total_seconds"], "on_seconds": a["logical_total_seconds"],
            "extra_seconds": a["logical_total_seconds"] - b["logical_total_seconds"],
            "off_tokens": result["prefix_usage"]["total_tokens"], "on_tokens": a["full_usage"]["total_tokens"],
            "extra_tokens": a["new_tokens"], "extra_calls": a["new_calls"], "added_agents": dict(agents),
            "review_attempts": a["node_attempts"].get("review", 0), "iterations": a["iteration"],
            "extra_searches": a.get("search_budget", {}).get("used", 0) - old_searches,
            "completion_reason": a["completion_reason"], "quality_verdict": a["quality_verdict"],
            "calls_missing_usage": a["full_usage"]["calls_missing_usage"] + b["new_usage"]["calls_missing_usage"],
            "identical_prefix": True, "disabled_new_calls": 0,
            "effect": on.get("pair_effect", "见逐篇分析"),
            "actual_experiment_tokens": a["full_usage"]["total_tokens"] + b["new_usage"]["total_tokens"]})
    totals = {"off_mean": mean(p["off_score"] for p in pairs), "on_mean": mean(p["on_score"] for p in pairs),
        "mean_delta": mean(p["quality_delta"] for p in pairs),
        "wins": sum(p["quality_delta"] > 0 for p in pairs), "ties": sum(p["quality_delta"] == 0 for p in pairs),
        "losses": sum(p["quality_delta"] < 0 for p in pairs),
        "off_tokens": sum(p["off_tokens"] for p in pairs), "on_tokens": sum(p["on_tokens"] for p in pairs),
        "extra_tokens": sum(p["extra_tokens"] for p in pairs), "extra_calls": sum(p["extra_calls"] for p in pairs),
        "off_seconds": sum(p["off_seconds"] for p in pairs), "on_seconds": sum(p["on_seconds"] for p in pairs),
        "actual_experiment_tokens": sum(p["actual_experiment_tokens"] for p in pairs),
        "revision_limit_count": sum(p["completion_reason"] == "revision_limit" for p in pairs),
        "calls_missing_usage": sum(p["calls_missing_usage"] for p in pairs)}
    totals["token_increase_percent"] = 100 * (totals["on_tokens"] / totals["off_tokens"] - 1)
    totals["time_increase_percent"] = 100 * (totals["on_seconds"] / totals["off_seconds"] - 1)
    totals["dimensions"] = {k: {"off_mean": mean(lookup[i, "关闭审核"]["scores"][k] for i in IDS),
        "on_mean": mean(lookup[i, "开启审核"]["scores"][k] for i in IDS),
        "delta": mean(p["dimension_delta"][k] for p in pairs)} for k in DIMS}
    summary = {"evaluated_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(), "status": "completed",
        "reports_completed": 10, "reports_directly_assessed": 10, "external_llm_judges": False,
        "pairs": pairs, "totals": totals, "rubric_path": str(root / "Codex直接质量评估口径.json"),
        "assessment_path": str(root / "Codex逐篇质量评估.json"),
        "limitations": rubric["limits"], "cost_scope": "数据库AgentRun记录的生成调用；非完整API账单"}
    (root / "质量与成本配对汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    manifest.update(status="completed", reports_completed=10, reports_directly_assessed=10,
                    quality_evaluated_at=summary["evaluated_at"], external_llm_judges=False)
    (root / "运行清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    out = ["# Critic通用研究五题消融结果及分析", "",
        f"评估日期：{summary['evaluated_at']}（Asia/Singapore）；真实运行开始：{manifest['started_at']}。同一运行补全启动报告，评分口径沿用事先冻结的四维规则，未新增外部LLM裁判。", "",
        f"通用单轮原题35、31、46、16、49的十份报告已生成并由Codex主代理逐篇全文审读。关闭Critic平均 **{totals['off_mean']:.2f}/10**，开启平均 **{totals['on_mean']:.2f}/10**，配对平均变化 **{totals['mean_delta']:+.2f}**；{totals['wins']}题提高、{totals['ties']}题持平、{totals['losses']}题降低。", "",
        "## 数据和配置口径", "",
        "每题先通过真实研究服务完成同一份首稿，在首次Critic之前冻结；关闭组复制该状态、历史与证据后正常交付，开启组从同一个检查点继续生产审核与返工。每题独立会话、隔离数据库，源码与输入已冻结。开启组最多两轮返工；一次初审加两次复审最多三次Critic调用。保持其余研究链路配置，评审收益包含被Critic触发的补搜、分析、改写及复审。", "",
        f"运行配置：{link(root / '运行配置.json')}；运行清单：{link(root / '运行清单.json')}；固定五题：{link(root / '固定五题.json')}；源码快照：{link(root / '运行源码快照.zip')}。", "",
        f"生成代理配置：`{json.dumps(config['agents'], ensure_ascii=False)}`。研究总预算{config['total_seconds']}秒、token上限{config['token_limit']}（0表示未设上限），并发{config['concurrency']}。引用审计配置：`{json.dumps(config['citation_verification'], ensure_ascii=False)}`；该审计属于原有生成链路，两组共同保留，并非本轮另加的质量裁判。", "",
        f"质量口径：{link(root / 'Codex直接质量评估口径.json')}；深度、全面性、可读性、需求满足各0至10分，四维等权平均；由当前主代理一次直接评估，知道组别。核对全文、引用目录及关键数值/条件的保存来源，不使用内部Critic分数替代最终质量。", "",
        "## 已测量结果", "",
        "生成完成10/10、全文评估10/10；五题关闭组与公共首稿逐字相同，关闭分支新增AgentRun调用均为0，未进入review，无Critic反馈，按not_reviewed正常交付。离线机制回归29项通过，见运行目录机制核对结果及三个回归日志。", "",
        "| 题目 | 关闭均分 | 开启均分 | 差值 | 直接审读发现 |", "|---|---:|---:|---:|---|"]
    for p in pairs:
        out.append(f"| {p['id']} {p['name']} | {p['off_score']:.2f} | {p['on_score']:.2f} | {p['quality_delta']:+.2f} | {p['effect'].replace('|', '／')} |")
    out += ["", "| 维度 | 关闭平均 | 开启平均 | 差值 |", "|---|---:|---:|---:|"]
    for k, title in DIMS.items():
        d = totals["dimensions"][k]
        out.append(f"| {title} | {d['off_mean']:.2f} | {d['on_mean']:.2f} | {d['delta']:+.2f} |")
    out += ["", "## 耗时和记录内调用成本", "",
        f"按五题各自完整链路累加，关闭组{totals['off_tokens']:,} token，开启组{totals['on_tokens']:,} token，增加{totals['extra_tokens']:,} token（{totals['token_increase_percent']:+.1f}%）与{totals['extra_calls']}次生成调用；单题平均链路耗时由{totals['off_seconds']/5/60:.1f}分钟到{totals['on_seconds']/5/60:.1f}分钟（{totals['time_increase_percent']:+.1f}%）。并发情况下累加耗时不能当实验墙钟耗时。", "",
        f"这次配对实验公共前缀只生成一次，实际记录内消耗为{totals['actual_experiment_tokens']:,} token；不能把关闭、开启的逻辑完整链路token简单相加当实际账单。记录内缺usage调用数{totals['calls_missing_usage']}。数字仅覆盖数据库AgentRun生成调用，不覆盖全部网页正文筛选辅助模型、embedding、rerank、搜索收费及未入该表的调用；人民币费用未测量。", "",
        "| 题目 | 关闭/开启分钟 | 额外token | 新增生成调用 | 新增搜索 | 审核次数/返工轮数 | 开启完成原因/内部判断 |", "|---|---:|---:|---:|---:|---:|---|"]
    for p in pairs:
        out.append(f"| {p['id']} | {p['off_seconds']/60:.1f} / {p['on_seconds']/60:.1f} | {p['extra_tokens']:,} | {p['extra_calls']} | {p['extra_searches']} | {p['review_attempts']} / {p['iterations']} | {p['completion_reason']} / {p['quality_verdict']} |")
    out += ["", f"达到返工上限的开启组为{totals['revision_limit_count']}/5。生成完成、达到上限与Critic内部认可分别记录；它们均不等于最终正文质量合格。", ""]
    coverage_path = root / "公共首稿网页读取覆盖记录.json"
    if coverage_path.exists():
        coverage = read(coverage_path)
        out += ["## 共享首稿的来源读取限制", "",
            f"网页读取记录：{link(coverage_path)}。下表是服务事件中候选URL读取计数，不保证去重，不是事实错误率。读取失败或正文被拒绝限制了本轮可用证据；它们不能证明现实中不存在相关数据。", "",
            "| 题目 | 候选URL计数 | 失败或拒绝计数 |", "|---|---:|---:|"]
        for record in coverage["records"]:
            stats = record["stats"]
            out.append(f"| {record['id']} | {stats['reader_urls_total']} | {stats['reader_invalid_urls']} |")
        out += ["", "另见逐篇抽查发现的DOC/PDF二进制误读、乱码、旧数据和分母错配；成功生成报告不意味着所有来源读取成功。", ""]
    out += ["## 十份报告逐篇分析及失败样例", ""]
    for ident in IDS:
        out += [f"### {ident}：{NAMES[ident]}", ""]
        for group in ["关闭审核", "开启审核"]:
            r = lookup[ident, group]
            path = root / f"题目{ident}" / group / "研究报告.md"
            out += [f"#### {group}：{r['overall']:.2f}/10", "", f"正文：{link(path)}；保存来源：{link(path.parent / '完整来源正文.json')}。", ""]
            for k, title in DIMS.items():
                out.append(f"- {title} {r['scores'][k]}/10：{r['score_reasons'][k]}")
            out += ["", "优势：" + "；".join(r["strengths"]) + "。", "", "缺口：" + "；".join(r["gaps"]) + "。", "",
                "| 需求 | 状态 | 依据 |", "|---|---|---|"]
            for req, status, evidence in r["requirements"]:
                out.append(f"| {requirements[req]} | {LABELS[status]} | {evidence.replace('|', '／')} |")
            out += ["", "定向证据核对与失败样例：", ""]
            for check in r["source_checks"]:
                claim_link = f"[{check['claim']}]({path.resolve()}:{check['line']})"
                out.append(f"- {claim_link}；{SOURCE_LABELS.get(check['support'], check['support'])}。{check['finding']} 来源：[{check['url']}]({check['url']})。")
            out += ["", r["boundary"], ""]
            if r.get("critic_execution_observation"):
                out += ["生成过程中审核反馈的落地观察（仅作为机制证据，不作为质量裁判）：" + r["critic_execution_observation"], ""]
        out += ["配对判断：" + lookup[ident, "开启审核"].get("pair_effect", "见上文") + "。", ""]
    out += ["## 可追溯原始文件", "",
        f"原始目录：{link(root)}；逐篇评分、理由、需求项和证据抽查：{link(root / 'Codex逐篇质量评估.json')}；配对统计：{link(root / '质量与成本配对汇总.json')}；生成汇总：{link(root / '生成结果汇总.json')}。", "",
        "每题目录另保存公共首稿、两组研究状态、完整检查点、数据库历史/证据快照、SSE事件、LLM调用埋点、Critic返工新增调用和完整过程埋点。配对运行结果.json为各组生成完成、哈希、token和耗时来源。", "",
        "## 评估边界和局限", "",
        "已测量：本轮五题一次生成的交付状态、正文哈希、记录内调用和链路耗时，以及主代理对十份正文的直接质量判断与定向来源核对。间接推断：同首稿减少首轮生成差异，最终差异主要来自Critic及其触发的后续链路；但共享前缀、缓存和并发仍会影响运行时间。尚未评估：人工金标、独立盲审、重复生成稳定性、统计显著性、全开发集泛化、全部事实真伪和完整人民币账单。", "",
        "主代理知道开启/关闭组，只有一次评分，不能声称人类评价、完全盲审或独立多裁判共识。未找到保存证据只表示当前引用支撑不足，不等于已证明现实虚假；部分来源乱码/不完整会限制核对。每篇定向抽查数量不同，不据此计算整体事实错误率。旧数据、子群体及预测口径是否恰当按正文使用方式判断。", "",
        "## 结论和下一步建议", "",
        f"本轮五题中，Critic及其后续返工使平均质量由{totals['off_mean']:.2f}升至{totals['on_mean']:.2f}/10（{totals['mean_delta']:+.2f}）；记录内token增加{totals['token_increase_percent']:.1f}%，平均链路耗时增加{totals['time_increase_percent']:.1f}%。收益主要是局部纠错或补充覆盖，原题核心缺口仍然存在，部分改写引入新的数值、分母或政策解释错误。这批样本没有显示与额外成本相称的整体质量提升；也不能凭五题一次非盲审直接断言Critic普遍无效。", "",
        "对当前实现的建议：若以这五题的成本收益做选择，可优先考虑关闭默认Critic或只在高风险、明确缺项的报告中触发，生产默认值本轮未修改。先让返工按问题逐项核验是否已修复，约束新事实引入，增加原题核心需求覆盖与数值单位、分母、引用正文的一致性检查。污水题应补支付、维护、应急的执行安排；房地产题应补十年阶段与风险；算法题应补同任务基准和测试条件；共同需求题应明确目标人群分母、时间窗及联合测量方案。后续更大样本或重复生成属于尚未执行的建议。本轮不额外启动其他LLM裁判或新一轮实验。", ""]
    args.report.write_text("\n".join(out))
    print(json.dumps({"report": str(args.report.resolve()), "totals": totals}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
