"""离线提取冻结审核状态，并按独立路由标注复算指标；不调用模型或搜索。"""
import hashlib
import json
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "backend/eval/data/已有轨迹动态路由核验-20261101"
BATCHES = {
    "20261009": "Critic通用研究五题消融-20261009-221748",
    "20261017": "Critic职责分离通用研究五题消融-20261017-231745",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract():
    rows, manifest = [], []
    for batch, name in BATCHES.items():
        base = ROOT / "backend/eval/data" / name
        for cpfile in sorted(base.glob("题目*/开启审核/完整检查点历史.json")):
            folder = cpfile.parent
            q = int(folder.parent.name.removeprefix("题目"))
            histfile = folder / "智能体原始历史.json"
            hist = read(histfile)
            reviews = sorted([h for h in hist if h.get("node") == "review" and h["kind"] == "llm_output"],
                             key=lambda h: datetime.fromisoformat(h["created_at"]))
            if not reviews:
                continue
            cps = sorted(read(cpfile), key=lambda c: c["metadata"]["step"])
            pres = [(i, c) for i, c in enumerate(cps)
                    if c["checkpoint"]["channel_values"].get("research", {}).get("phase") == "reviewing"]
            assert len(pres) == len(reviews), (batch, q, len(pres), len(reviews))
            fullfile = folder / "完整来源正文.json"
            full = {x["id"]: x for x in read(fullfile)}
            for source_file in [cpfile, histfile, fullfile, base / "运行源码快照.zip"]:
                manifest.append({"path": relative(source_file), "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest()})
            for n, ((idx, cp), output) in enumerate(zip(pres, reviews), 1):
                state = cp["checkpoint"]["channel_values"]["research"]
                after = cps[idx + 1]["checkpoint"]["channel_values"]["research"]
                report = state["final_report"].strip()
                cid = output["call_id"]
                inp = next(h for h in hist if h.get("call_id") == cid and h["kind"] == "llm_input")
                req = next(h for h in hist if h.get("call_id") == cid and h["kind"] == "llm_request")
                assert report in inp["payload"]["content"]
                assert any(report in m.get("content", "") for m in req["payload"]["request"]["messages"] if isinstance(m.get("content"), str))
                assert after["final_report"].strip() == report
                result = json.loads(output["payload"]["content"])
                key = f"{batch}-题目{q}-第{n}轮"
                dest = OUT / key
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "待审报告.md").write_text(report + "\n", encoding="utf-8")
                available = []
                for src in state["raw_sources"]:
                    candidate = full.get(src["id"], {})
                    content = src.get("content") or candidate.get("content") or ""
                    expected = src.get("content_sha256")
                    matched = bool(content) and (not expected or sha(content) == expected)
                    # 只恢复当前检查点已存在、且与当时哈希一致的正文，防止倒灌后来补搜资料。
                    available.append({"id": src["id"], "url": src.get("url"), "title": src.get("title"),
                                      "related_sections": src.get("related_sections"), "snippet": src.get("snippet"),
                                      "content": content if matched else "", "content_sha256": sha(content) if matched else None,
                                      "checkpoint_sha256": expected, "content_recovered": matched,
                                      "hash_verified": bool(expected) and matched})
                save(dest / "当时可用来源.json", available)
                save(dest / "模型审核输出.json", result)
                save(dest / "审核输入与状态.json", {
                    "query": state["query"], "outline": state["outline"],
                    "iteration": state["iteration"], "max_iterations": state["max_iterations"],
                    "model_input": inp["payload"], "quality_gates": state.get("quality_gates"),
                    "before_issues": state.get("critic_feedback"), "after_issues": after.get("critic_feedback"),
                    "pending_research_tasks": after.get("pending_research_tasks"),
                    "issue_search_results": state.get("issue_search_results"),
                })
                phase = after["phase"]
                if phase == "completed":
                    action = "通过" if after["quality_verdict"] == "pass" else "额度停止"
                    if action == "额度停止":
                        assert state["iteration"] >= state["max_iterations"]
                else:
                    action = {"re_researching": "补搜", "revising": "修订"}[phase]
                # 以检查点实际路由为主；原始模型建议仅作诊断，采用pass优先的显式定义。
                raw_verdict = result["overall_assessment"]["verdict"]
                model_action = "通过" if raw_verdict == "pass" else ("补搜" if any(i["requires_new_search"] for i in result["issues"]) else "修订")
                start = datetime.fromisoformat(output["created_at"])
                stop = datetime.fromisoformat(cps[idx+2]["checkpoint"]["ts"]) if idx+2 < len(cps) else None
                # 原始历史按智能体上下文分组，并非全局时间排序，不能按数组位置截取后续事件。
                following_tools = [h for h in hist if h.get("node") == "re_research" and h["kind"] == "tool_input"
                                   and datetime.fromisoformat(h["created_at"]) >= start
                                   and (stop is None or datetime.fromisoformat(h["created_at"]) <= stop)]
                executed_queries = [h["payload"]["arguments"]["query"] for h in following_tools
                                    if h["payload"].get("tool_name") == "v2_web_search"]
                if action == "补搜":
                    assert len(executed_queries) == len(after["pending_research_tasks"]), (key, executed_queries)
                save(dest / "路由后补搜调用.json", following_tools)
                row = {
                    "key": key, "batch": batch, "question": q, "round": n, "call_id": cid,
                    "model": req["payload"]["request"]["model"],
                    "temperature": req["payload"]["request"].get("temperature"),
                    "query": state["query"], "report_path": relative(dest / "待审报告.md"),
                    "report_sha256": sha(report), "sources_path": relative(dest / "当时可用来源.json"),
                    "source_count": len(available), "source_content_recovered": sum(x["content_recovered"] for x in available),
                    "source_hash_verified": sum(x["hash_verified"] for x in available),
                    "input_path": relative(dest / "审核输入与状态.json"),
                    "output_path": relative(dest / "模型审核输出.json"),
                    "checkpoint_file": relative(cpfile), "history_file": relative(histfile),
                    "pre_checkpoint_id": cp["checkpoint"]["id"],
                    "post_checkpoint_id": cps[idx+1]["checkpoint"]["id"],
                    "pre_step": cp["metadata"]["step"], "post_step": cps[idx+1]["metadata"]["step"],
                    "phase": phase, "program_verdict": after["quality_verdict"],
                    "iteration_before": state["iteration"], "max_iterations": state["max_iterations"],
                    "actual_action": action, "raw_model_verdict": raw_verdict, "raw_model_action": model_action,
                    "pending_queries": after.get("pending_research_tasks", []),
                    "following_search_tool_inputs": len(following_tools),
                    "executed_search_queries": executed_queries,
                    "query_strings_exact_match": set(executed_queries) == {x["query"] for x in after.get("pending_research_tasks", [])},
                    "next_checkpoint_phase": cps[idx+2]["checkpoint"]["channel_values"]["research"]["phase"] if idx+2 < len(cps) else None,
                }
                rows.append(row)
    save(OUT / "逐轮轨迹提取.json", rows)
    save(OUT / "原始文件索引.json", list({x["path"]: x for x in manifest}.values()))
    print(json.dumps({"rounds": len(rows), "actions": dict(Counter(r["actual_action"] for r in rows)),
                      "sources": sum(r["source_count"] for r in rows),
                      "source_content_recovered": sum(r["source_content_recovered"] for r in rows)}, ensure_ascii=False))
    return rows


def fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def metrics(rows, field="actual_action"):
    matrix = {gold: {pred: 0 for pred in ["通过", "补搜", "修订"]} for gold in ["通过", "补搜", "修订"]}
    decided = [r for r in rows if len(r["acceptable_actions"]) == 1]
    for r in decided:
        matrix[r["acceptable_actions"][0]][r[field]] += 1
    result = {"n": len(rows), "unique_label_n": len(decided), "confusion_matrix": matrix,
              "correct_route": fraction(sum(r[field] in r["acceptable_actions"] for r in rows), len(rows))}
    passed = [r for r in rows if r[field] == "通过"]
    result["false_release"] = fraction(sum("通过" not in r["acceptable_actions"] for r in passed), len(passed))
    for action, name in [("补搜", "search"), ("修订", "revision")]:
        predicted = [r for r in rows if r[field] == action]
        required = [r for r in rows if r["acceptable_actions"] == [action]]
        result[name + "_precision"] = fraction(sum(action in r["acceptable_actions"] for r in predicted), len(predicted))
        result[name + "_recall"] = fraction(sum(r[field] == action for r in required), len(required))
    return result


def summarize(rows):
    labelpath = OUT / "逐轮独立路由标注.json"
    if not labelpath.exists():
        return
    annotations = read(labelpath)
    by_key = {a["key"]: a for a in annotations}
    assert len(by_key) == len(annotations) == len(rows)
    assert set(by_key) == {r["key"] for r in rows}
    for row in rows:
        label = by_key[row["key"]]
        assert label["acceptable_actions"] and set(label["acceptable_actions"]) <= {"通过", "补搜", "修订"}
        report = (ROOT / row["report_path"]).read_text()
        for quote in label["report_quotes"]:
            assert quote in report, (row["key"], quote)
        available = {s["id"]: s for s in read(ROOT / row["sources_path"])}
        for ref in label["source_evidence"]:
            assert ref["source_id"] in available
            assert ref["quote"] in available[ref["source_id"]]["content"], (row["key"], ref)
        row.update(label)
    eligible = [r for r in rows if r["actual_action"] != "额度停止"]
    unique = [r for r in eligible if len(r["acceptable_actions"]) == 1]
    ambiguous = [r for r in eligible if len(r["acceptable_actions"]) > 1]
    sensitivity = []
    for choices in product(*[r["acceptable_actions"] for r in ambiguous]):
        scenario = {r["key"]: choice for r, choice in zip(ambiguous, choices)}
        variant = [dict(r, acceptable_actions=[scenario[r["key"]]]) if r["key"] in scenario else r for r in eligible]
        sensitivity.append({"label_choices": scenario, "metrics": metrics(variant)})
    save(OUT / "非唯一路由口径敏感性.json", sensitivity)
    summary = {
        "assessment_date": "2026-11-01", "reviewer": "Codex单一AI审读者；非人类金标，非盲评",
        "total_rounds": len(rows), "unique_questions": len({r["question"] for r in rows}),
        "task_runs": len({(r["batch"], r["question"]) for r in rows}),
        "actual_action_counts": dict(Counter(r["actual_action"] for r in rows)),
        "forced_stop_excluded": len(rows) - len(eligible),
        "eligible_rounds": len(eligible), "unique_label_rounds": len(unique),
        "non_unique_label_rounds": len(ambiguous),
        "strict_unique_label_descriptive_only": metrics(unique),
        "acceptable_action_set_descriptive_only": metrics(eligible),
        "by_batch": {batch: metrics([r for r in unique if r["batch"] == batch]) for batch in BATCHES},
        "acceptable_action_by_batch": {batch: metrics([r for r in eligible if r["batch"] == batch]) for batch in BATCHES},
        "raw_model_action_all_rounds_diagnostic": metrics(rows, "raw_model_action"),
        "raw_model_by_batch": {batch: metrics([r for r in rows if r["batch"] == batch], "raw_model_action") for batch in BATCHES},
        "forced_stops": [{k: r[k] for k in ["key", "acceptable_actions", "reason"]} for r in rows if r["actual_action"] == "额度停止"],
    }
    save(OUT / "逐轮核验结果.json", rows)
    save(OUT / "指标汇总.json", summary)
    save(OUT / "完整性检查.json", {
        "rounds": len(rows), "all_reports_in_actual_requests": True,
        "all_pre_post_reports_matched": True, "all_labels_covered": True,
        "all_report_quotes_matched": True, "all_source_quotes_matched": True,
        "source_entries": sum(r["source_count"] for r in rows),
        "source_entries_recovered": sum(r["source_content_recovered"] for r in rows),
        "source_entries_hash_verified": sum(r["source_hash_verified"] for r in rows),
        "search_routes_with_verified_query_execution": sum(r["actual_action"] == "补搜" and bool(r["executed_search_queries"]) for r in rows),
        "forced_stop_excluded": len(rows)-len(eligible),
        "raw_file_hashes_verified": all(hashlib.sha256((ROOT / f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in read(OUT / "原始文件索引.json")),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    summarize(extract())
