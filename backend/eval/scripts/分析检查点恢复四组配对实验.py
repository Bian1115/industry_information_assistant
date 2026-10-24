"""从原始文件计算配对指标；续跑显式标记已完成基线，新增消耗剔除复用样本。"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

GROUP_NAMES = {"A": "无故障关闭检查点", "B": "无故障开启检查点",
               "C": "故障后从头重跑", "D": "同位置检查点续跑"}


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


def stats(values):
    values = [x for x in values if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def ratio(new, baseline):
    return 1 - new / baseline if isinstance(new, (int, float)) and isinstance(baseline, (int, float)) and baseline > 0 else None


def timestamp_delta(end, start):
    return (datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds() if end and start else None


def token_cost(history, prices):
    requests = {(r["call_id"], r["payload"].get("attempt", 1)) for r in history if r["kind"] == "llm_request"}
    usage_rows = [r for r in history if r["kind"] == "llm_usage"]
    accounted = {(r["call_id"], r["payload"].get("attempt", 1)) for r in usage_rows}
    usage_by_model = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0})
    known_price, unknown_prices = 0.0, 0
    request_options = {(r["call_id"], r["payload"].get("attempt", 1)): r["payload"].get("request", {}) for r in history if r["kind"] == "llm_request"}
    details = []
    invalid_usage=[]
    for row in usage_rows:
        payload = row["payload"]; model = payload.get("model", "")
        usage = payload.get("usage") or {}
        inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in [inp,out]):
            invalid_usage.append({"source_message_id":row["id"],"call_id":row["call_id"],"attempt":payload.get("attempt",1)})
        item = usage_by_model[model]; item["calls"] += 1
        item["input_tokens"] += inp or 0; item["output_tokens"] += out or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        item["cached_input_tokens"] += cached
        options = request_options.get((row["call_id"], payload.get("attempt", 1)), {})
        thinking = options.get("enable_thinking") or (options.get("extra_body") or {}).get("enable_thinking")
        tier = next((t for t in prices.get("models", {}).get(model, {}).get("tiers", []) if inp is not None and inp <= t["input_up_to"]), None)
        # 费用指数统一不折缓存；不对未知思考模式费用推定非思考单价。
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        priced = tier is not None and inp is not None and out is not None and not (model.startswith("qwen-plus") and (thinking or reasoning))
        value = (inp*tier["input_per_million"] + out*tier["output_per_million"]) / 1_000_000 if priced else None
        if value is None: unknown_prices += 1
        else: known_price += value
        details.append({"source_message_id": row["id"], "call_id": row["call_id"], "attempt": payload.get("attempt", 1),
            "model": model, "usage": usage, "published_price_index_cny": value})
    return {"physical_requests": len(requests), "usage_records": len(usage_rows),
            "missing_usage_requests": [{"call_id": c, "attempt": a} for c,a in sorted(requests-accounted)],
            "invalid_or_missing_usage_records":invalid_usage,
            "unknown_usage_count": len(requests-accounted)+len(invalid_usage), "unknown_price_calls": unknown_prices,
            "input_tokens": sum(r["input_tokens"] for r in usage_by_model.values()),
            "output_tokens": sum(r["output_tokens"] for r in usage_by_model.values()),
            "known_total_tokens": sum(r["input_tokens"]+r["output_tokens"] for r in usage_by_model.values()),
            "by_model": dict(usage_by_model), "known_published_price_index_cny": known_price,
            "published_price_index_complete": not (requests-accounted) and not invalid_usage and unknown_prices == 0,
            "calls": details}


def rag_call_key(call):
    return call.get("api_request_id") or hashlib.sha256(json.dumps(call,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def read_group(folder, group, prices, prefix_ids=None, prefix_rag_keys=None, prefix_budget_ids=None):
    reused=load(folder/"复用来源.json")
    identity = load(folder / "运行身份.json", {})
    status = load(folder / "执行状态.json", {})
    execution = load(folder / "进程计时.json", {})
    process_start = load(folder / "进程启动.json", {})
    state = load(folder / "数据库与图状态.json", {})
    research = state.get("research") or {}
    rag_calls={rag_call_key(c):c for m in research.get("evidence_rag_metrics",[]) for c in m.get("rerank_calls",[])}
    rag_calls={k:v for k,v in rag_calls.items() if k not in (prefix_rag_keys or set())}
    rag_tokens=[c.get("api_usage",{}).get("total_tokens",c.get("api_usage",{}).get("prompt_tokens")) for c in rag_calls.values()]
    delivery = load(folder / "交付一致性.json", {})
    history = load(folder / "工作历史.json", [])
    if prefix_ids:
        history = [r for r in history if r["id"] not in prefix_ids]
    journal = lines(folder / "执行轨迹.jsonl")
    nodes = [r for r in journal if r["kind"] == "node_start"]
    commits = [r for r in journal if r["kind"] == "delivery_committed"]
    sections = [r for r in journal if r["kind"] == "section_generate_start"]
    writes = [r for r in journal if r["kind"] in {"checkpoint_committed", "pending_writes_committed"}]
    restored = {}
    def visit(value):
        if isinstance(value,dict):
            if value.get("restored") is True and value.get("section_id"):
                restored[value["section_id"]] = value
            for child in value.values():
                if isinstance(child,(dict,list)): visit(child)
        elif isinstance(value,list):
            for child in value: visit(child)
    for event in lines(folder/"研究事件.jsonl"): visit(event)
    storage = load(folder / "检查点存储统计.json", {})
    budget_calls=[r for r in load(folder/"预算调用.json",[]) if r["call_id"] not in (prefix_budget_ids or set())]
    wire=lines(folder/"供应商请求实参.jsonl")
    ready = nodes[0]["monotonic"] - status["started_monotonic"] if nodes and status.get("started_monotonic") is not None else None
    wall = ((datetime.fromisoformat(status["ended_at"])-datetime.fromisoformat(status["started_at"])).total_seconds()
            if status.get("started_at") and status.get("ended_at") else None)
    process_wall = ((datetime.fromisoformat(execution["ended_at"])-datetime.fromisoformat(process_start["started_at"])).total_seconds()
                    if execution.get("ended_at") and process_start.get("started_at") else None)
    success = bool(status.get("completed") and delivery.get("consistent") and research.get("status") == "completed")
    score = research.get("quality_score")
    if not isinstance(score, (int,float)) or isinstance(score,bool) or not 1 <= score <= 10: score = None
    return {"group": group, "folder": str(folder), "attempted": bool(identity), "finished": bool(execution),
        "reused_completed_baseline":reused,"incurred_in_current_run":bool(identity) and not reused,
        "identity": identity, "status": {k:v for k,v in status.items() if k != "traceback"},
        "technical_success": success, "delivery": delivery,
        "timing": {"service_continuation_seconds": status.get("duration_seconds"),
            "process_wall_seconds": process_wall,
            "process_monotonic_seconds_including_startup": execution.get("wall_seconds_including_process_startup"),
            "delivery_committed_at":commits[0]["at"] if commits else None,
            "process_start_to_delivery_commit_seconds":timestamp_delta(commits[0]["at"],process_start.get("started_at")) if commits else None,
            "service_start_to_delivery_commit_seconds":timestamp_delta(commits[0]["at"],status.get("started_at")) if commits else None,
            "lease_wait_seconds": status.get("lease_wait_seconds"), "first_new_node_seconds": ready,
            "first_new_node": nodes[0]["node"] if nodes else None,
            "wall_minus_monotonic_seconds": wall-status["duration_seconds"] if wall is not None and status.get("duration_seconds") is not None else None},
        "work": {"node_calls": dict(Counter(r["node"] for r in nodes)),
            "generated_sections": [{"section_id":r["section_id"], "iteration":r.get("iteration")} for r in sections],
            "section_generation_calls": len(sections), "search_calls": sum(r["kind"] == "search_start" for r in journal),
            "search_queries":[{"query":r["query"],"iteration":r.get("iteration")} for r in journal if r["kind"]=="search_start"],
            "restored_section_events": list(restored.values()),
            "successful_search_calls": sum(r["kind"] == "search_return" and bool(r.get("result_count")) for r in journal),
            "checkpoint_write_calls": len(writes), "checkpoint_write_seconds": stats([r["seconds"] for r in writes])},
        "cost": {**token_cost(history, prices), "exported":(folder / "工作历史.json").exists(),
            "transport_request_count":len(wire),
            "request_records_beyond_observed_sdk_invocations":max(0,sum(r["kind"]=="llm_request" for r in history)-len(wire)),
            "observed_sdk_invocations_without_usage_count":max(0,len(wire)-sum(r["kind"]=="llm_usage" for r in history)),
            "transport_all_non_thinking_explicit":all(r.get("extra_body",{}).get("enable_thinking") is False for r in wire) if wire else None},
        "checkpoint_storage": storage,
        "independent_budget_ledger":{"call_records":len(budget_calls),
            "known_actual_tokens":sum(r["actual_tokens"] for r in budget_calls if r.get("actual_tokens") is not None),
            "calls_without_actual_tokens":sum(r.get("actual_tokens") is None for r in budget_calls),
            "reservation_tokens_without_actual_usage":sum(r["reserved_tokens"] for r in budget_calls if r.get("actual_tokens") is None),
            "call_status_counts":dict(Counter(r["status"] for r in budget_calls)),
            "boundary":"预算预留不等于实际消耗；与工作历史usage独立核对，不重复累加。"},
        "resume_frontier":{"pending_nodes":state.get("pending_nodes"),"saved_result_nodes":state.get("saved_result_nodes"),"graph_version":state.get("graph_version")},
        "quality": {"internal_score": score, "internal_verdict": research.get("quality_verdict"),
            "completion_reason": research.get("completion_reason"), "unresolved_issues": research.get("unresolved_issues")},
        "source_fact_count": len(research.get("facts") or []),
        "evidence_index_states": dict(Counter(r.get("index_status") for r in research.get("evidence_rag_metrics",[]) if r.get("event") == "ingest")),
        "observable_rag_usage":{"persisted_rerank_records":len(rag_calls),
            "known_rerank_tokens_lower_bound":sum(v for v in rag_tokens if type(v) is int),
            "records_missing_usage":sum(type(v) is not int for v in rag_tokens),
            "boundary":"只有成功提交业务状态中的精排审计；崩溃节点内未提交调用及embedding usage未完整采集，不能作为整条链路账单。"},
    }


def index_facts(research):
    facts = research.get("facts") or []
    return {r.get("id") or r.get("fact_id"):r for r in facts if r.get("id") or r.get("fact_id")}


def analyze(out):
    manifest = load(out/"运行清单.json")
    prices = load(out/"模型可用性与价格核对.json", {})
    cases = []
    for source_id, fault_case in manifest["fault_cases"]:
        base = out/f"题目{source_id}-{fault_case}"
        prefix_folder = base/"共同故障前缀"
        prefix_history = load(prefix_folder/"工作历史.json", [])
        prefix_ids = {r["id"] for r in prefix_history}
        prefix_budget_ids={r["call_id"] for r in load(prefix_folder/"预算调用.json",[])}
        prefix_state = (load(prefix_folder/"数据库与图状态.json", {}).get("research") or {})
        prefix_rag_keys={rag_call_key(c) for m in prefix_state.get("evidence_rag_metrics",[]) for c in m.get("rerank_calls",[])}
        prefix = read_group(prefix_folder,"P",prices)
        prefix["fault"] = load(prefix_folder/"故障触发.json")
        groups = {g:read_group(base/(g+"组-"+name),g,prices,prefix_ids if g=="D" else None,prefix_rag_keys if g=="D" else None,prefix_budget_ids if g=="D" else None) for g,name in GROUP_NAMES.items()}
        A,B,C,D = (groups[g] for g in "ABCD")
        ab_valid = A["technical_success"] and B["technical_success"]
        cd_valid = bool(prefix["fault"] and C["technical_success"] and D["technical_success"])
        bd_valid = B["technical_success"] and D["technical_success"]
        price_valid = cd_valid and C["cost"]["published_price_index_complete"] and D["cost"]["published_price_index_complete"]
        token_valid = cd_valid and C["cost"]["exported"] and D["cost"]["exported"] and C["cost"]["unknown_usage_count"]==D["cost"]["unknown_usage_count"]==0
        full_price_valid = price_valid and prefix["cost"]["published_price_index_complete"]
        # 消耗为0且没有模型请求的终态补交付是可观测结果，不虚构未知请求消耗。
        facts_before = index_facts(prefix_state)
        final_state = (load(Path(D["folder"])/"数据库与图状态.json",{}).get("research") or {})
        facts_after = index_facts(final_state)
        business_fields=["query","original_query","resolved_query","research_contract","context_policy",
            "session_inheritance_config_hash","allowed_domains","hypothesis_enabled","search_backend"]
        field_checks={k:{"present_before":k in prefix_state,"present_after":k in final_state,
            "equal":prefix_state[k]==final_state[k] if k in prefix_state and k in final_state else False}
            for k in business_fields if k in prefix_state}
        shared = set(facts_before)&set(facts_after)
        before_journal = lines(prefix_folder/"执行轨迹.jsonl")
        artifacts = {}
        for record in before_journal:
            for artifact in record.get("section_artifacts",[]): artifacts[artifact["section_id"]] = artifact
        duplicate_sections = [r for r in D["work"]["generated_sections"] if r["section_id"] in artifacts and r.get("iteration",0)==0]
        prefix_searches=Counter(r["query"].strip() for r in before_journal if r["kind"]=="search_return" and r.get("result_count"))
        resume_searches=Counter(r["query"].strip() for r in D["work"]["search_queries"] if r.get("iteration",0)==0)
        restored_sections = {r["section_id"]:r for r in D["work"]["restored_section_events"]}
        def delta(x,y): return x-y if x is not None and y is not None else None
        Ctime,Dtime = C["timing"]["process_wall_seconds"],D["timing"]["process_wall_seconds"]
        Ptime = prefix["timing"]["process_wall_seconds"]
        pair = {"ab_completed_pair": ab_valid, "cd_completed_pair": cd_valid,
            "ab_includes_historical_completed_baseline":bool(A["reused_completed_baseline"] or B["reused_completed_baseline"]),
            "ab_end_to_end_time_difference_seconds": delta(B["timing"]["process_wall_seconds"],A["timing"]["process_wall_seconds"]) if ab_valid else None,
            "ab_end_to_end_time_overhead_rate": -ratio(B["timing"]["process_wall_seconds"],A["timing"]["process_wall_seconds"]) if ab_valid else None,
            "ab_same_observed_node_call_counts": A["work"]["node_calls"]==B["work"]["node_calls"] if ab_valid else None,
            "post_fault_time_savings_seconds": delta(Ctime,Dtime) if cd_valid else None,
            "post_fault_time_savings_rate": ratio(Dtime,Ctime) if cd_valid else None,
            "D_observed_fault_to_delivery_commit_seconds":timestamp_delta(D["timing"]["delivery_committed_at"],(prefix["fault"] or {}).get("at")),
            "node_call_savings_rate":ratio(sum(D["work"]["node_calls"].values()),sum(C["work"]["node_calls"].values())) if cd_valid else None,
            "search_call_savings_rate":ratio(D["work"]["search_calls"],C["work"]["search_calls"]) if cd_valid else None,
            "transport_model_call_savings_rate":ratio(D["cost"]["transport_request_count"],C["cost"]["transport_request_count"]) if cd_valid else None,
            "known_token_savings_rate": ratio(D["cost"]["known_total_tokens"],C["cost"]["known_total_tokens"]) if token_valid else None,
            "published_price_index_savings_rate": ratio(D["cost"]["known_published_price_index_cny"],C["cost"]["known_published_price_index_cny"]) if price_valid else None,
            "whole_lifecycle_price_index_savings_rate":ratio(prefix["cost"]["known_published_price_index_cny"]+D["cost"]["known_published_price_index_cny"],prefix["cost"]["known_published_price_index_cny"]+C["cost"]["known_published_price_index_cny"]) if full_price_valid else None,
            "whole_lifecycle_time_C_seconds": Ptime+Ctime if cd_valid and Ptime is not None else None,
            "whole_lifecycle_time_D_seconds": Ptime+Dtime if cd_valid and Ptime is not None else None,
            "whole_lifecycle_time_savings_rate": ratio(Ptime+Dtime,Ptime+Ctime) if cd_valid and Ptime is not None else None,
            "internal_quality_delta_D_minus_B": delta(D["quality"]["internal_score"],B["quality"]["internal_score"]) if bd_valid else None,
            "internal_quality_delta_D_minus_C": delta(D["quality"]["internal_score"],C["quality"]["internal_score"]) if cd_valid else None,
            "C_and_D_both_internal_pass": bool(cd_valid and C["quality"]["internal_verdict"]==D["quality"]["internal_verdict"]=="pass"),
            "durable_fact_records_before": len(facts_before), "durable_fact_ids_retained": len(shared),
            "durable_fact_records_unchanged": sum(facts_before[k]==facts_after[k] for k in shared),
            "missing_durable_fact_ids": sorted(set(facts_before)-set(facts_after)),
            "selected_business_input_field_checks":field_checks if D["finished"] else None,
            "saved_chapter_artifacts": len(artifacts),
            "saved_chapter_artifacts_replayed_with_identical_body": sum(k in restored_sections and artifacts[k]["content"]==restored_sections[k].get("content") for k in artifacts),
            "saved_initial_chapter_generation_repeated_after_resume": len(duplicate_sections),
            "initial_successful_prefix_searches_repeated_after_resume":sum((prefix_searches&resume_searches).values()),
        }
        cases.append({"source_id":source_id,"fault_case":fault_case,"prefix":prefix,"groups":groups,"paired_metrics":pair})
    ab = [c for c in cases if c["paired_metrics"]["ab_completed_pair"]]
    ab_new=[c for c in ab if not c["paired_metrics"]["ab_includes_historical_completed_baseline"]]
    cd = [c for c in cases if c["paired_metrics"]["cd_completed_pair"]]
    matched_cd_attempts=[c for c in cases if c["prefix"]["fault"] and c["groups"]["C"]["finished"] and c["groups"]["D"]["finished"]]
    gated = [c for c in cd if c["paired_metrics"]["C_and_D_both_internal_pass"]]
    priced = [c for c in cd if c["paired_metrics"]["published_price_index_savings_rate"] is not None]
    triggered = [c for c in cases if c["prefix"]["fault"]]
    recovery_attempted=[c for c in triggered if c["groups"]["D"]["attempted"]]
    recovery_finished=[c for c in recovery_attempted if c["groups"]["D"]["finished"]]
    nonterminal_cd=[c for c in cd if c["fault_case"]!="终态交付提交前"]
    groups_summary = {}
    for g in "ABCD":
        samples = [c["groups"][g] for c in cases]
        groups_summary[g] = {"planned":len(samples),"attempted":sum(r["attempted"] for r in samples),
            "new_attempted":sum(r["incurred_in_current_run"] for r in samples),
            "reused_completed_baselines":sum(bool(r["reused_completed_baseline"]) for r in samples),
            "finished":sum(r["finished"] for r in samples),"technical_successes":sum(r["technical_success"] for r in samples),
            "completed_process_wall_seconds":stats([r["timing"]["process_wall_seconds"] for r in samples if r["technical_success"]]),
            "internal_passes":sum(r["technical_success"] and r["quality"]["internal_verdict"]=="pass" for r in samples),
            "internal_scores":stats([r["quality"]["internal_score"] for r in samples if r["technical_success"]]),
            "known_tokens_all_attempts":sum(r["cost"]["known_total_tokens"] for r in samples if r["incurred_in_current_run"]),
            "unknown_usage_all_attempts":sum(r["cost"]["unknown_usage_count"] for r in samples if r["incurred_in_current_run"])}
    summary = {"planned_tasks":len(cases),"faults_triggered":len(triggered),
        "recovery_started":sum(c["groups"]["D"]["attempted"] for c in triggered),
        "recovery_finished":len(recovery_finished),
        "recovery_completed":sum(c["groups"]["D"]["technical_success"] for c in triggered),
        "observed_recovery_completion_rate":sum(c["groups"]["D"]["technical_success"] for c in recovery_finished)/len(recovery_attempted) if recovery_attempted and len(recovery_finished)==len(recovery_attempted) else None,
        "groups":groups_summary,"completed_AB_pairs":len(ab),"completed_CD_pairs":len(cd),
        "completed_all_new_AB_pairs":len(ab_new),
        "matched_finished_CD_attempts":len(matched_cd_attempts),
        "matched_CD_delivery_outcomes":dict(Counter(
            "both_success" if c["groups"]["C"]["technical_success"] and c["groups"]["D"]["technical_success"] else
            "D_only_success" if c["groups"]["D"]["technical_success"] else
            "C_only_success" if c["groups"]["C"]["technical_success"] else "both_failed" for c in matched_cd_attempts)),
        "matched_CD_delivery_rate_difference_D_minus_C":statistics.mean(int(c["groups"]["D"]["technical_success"])-int(c["groups"]["C"]["technical_success"]) for c in matched_cd_attempts) if matched_cd_attempts else None,
        "both_internal_pass_CD_pairs":len(gated),"priced_CD_pairs":len(priced),
        "AB_end_to_end_time_overhead_rates":stats([c["paired_metrics"]["ab_end_to_end_time_overhead_rate"] for c in ab]),
        "CD_post_fault_time_savings_rates":stats([c["paired_metrics"]["post_fault_time_savings_rate"] for c in cd]),
        "CD_whole_lifecycle_time_savings_rates":stats([c["paired_metrics"]["whole_lifecycle_time_savings_rate"] for c in cd]),
        "CD_whole_lifecycle_price_index_savings_rates":stats([c["paired_metrics"]["whole_lifecycle_price_index_savings_rate"] for c in cd]),
        "CD_nonterminal_post_fault_time_savings_rates":stats([c["paired_metrics"]["post_fault_time_savings_rate"] for c in nonterminal_cd]),
        "CD_nonterminal_price_index_savings_rates":stats([c["paired_metrics"]["published_price_index_savings_rate"] for c in nonterminal_cd]),
        "CD_known_token_savings_rates":stats([c["paired_metrics"]["known_token_savings_rate"] for c in cd]),
        "CD_published_price_index_savings_rates":stats([c["paired_metrics"]["published_price_index_savings_rate"] for c in priced]),
        "CD_internal_pass_gated_time_savings_rates":stats([c["paired_metrics"]["post_fault_time_savings_rate"] for c in gated]),
        "CD_internal_pass_gated_price_savings_rates":stats([c["paired_metrics"]["published_price_index_savings_rate"] for c in gated]),
        "internal_quality_delta_D_minus_B":stats([c["paired_metrics"]["internal_quality_delta_D_minus_B"] for c in cases]),
        "extra_D_report_records":sum(max(0,c["groups"]["D"]["delivery"].get("report_records",0)-1) for c in cases),
        "extra_D_completion_events":sum(max(0,c["groups"]["D"]["delivery"].get("completion_events",0)-1) for c in cases),
        "D_saved_initial_chapter_repeated_generation_calls":sum(c["paired_metrics"]["saved_initial_chapter_generation_repeated_after_resume"] for c in cases),
        "actual_known_model_tokens_without_double_counting_prefix":sum(c["prefix"]["cost"]["known_total_tokens"]+sum(c["groups"][g]["cost"]["known_total_tokens"] for g in "ABCD" if c["groups"][g]["incurred_in_current_run"]) for c in cases),
    }
    summary["CD_weighted_post_fault_time_savings_rate"] = ratio(sum(c["groups"]["D"]["timing"]["process_wall_seconds"] for c in cd),sum(c["groups"]["C"]["timing"]["process_wall_seconds"] for c in cd))
    summary["CD_nonterminal_weighted_post_fault_time_savings_rate"] = ratio(sum(c["groups"]["D"]["timing"]["process_wall_seconds"] for c in nonterminal_cd),sum(c["groups"]["C"]["timing"]["process_wall_seconds"] for c in nonterminal_cd))
    summary["CD_weighted_published_price_index_savings_rate"] = ratio(sum(c["groups"]["D"]["cost"]["known_published_price_index_cny"] for c in priced),sum(c["groups"]["C"]["cost"]["known_published_price_index_cny"] for c in priced))
    summary["AB_weighted_end_to_end_time_overhead_rate"] = -ratio(sum(c["groups"]["B"]["timing"]["process_wall_seconds"] for c in ab),sum(c["groups"]["A"]["timing"]["process_wall_seconds"] for c in ab)) if ab else None
    summary["AB_all_new_end_to_end_time_overhead_rates"]=stats([c["paired_metrics"]["ab_end_to_end_time_overhead_rate"] for c in ab_new])
    summary["AB_all_new_weighted_end_to_end_time_overhead_rate"]=-ratio(sum(c["groups"]["B"]["timing"]["process_wall_seconds"] for c in ab_new),sum(c["groups"]["A"]["timing"]["process_wall_seconds"] for c in ab_new)) if ab_new else None
    started = [r for c in cases for r in [c["prefix"],*c["groups"].values()] if r["attempted"]]
    summary["all_started_phase_costs_exported"] = all(r["cost"]["exported"] for r in started)
    if not summary["all_started_phase_costs_exported"]:
        summary["exported_known_model_tokens_lower_bound"] = summary["actual_known_model_tokens_without_double_counting_prefix"]
        summary["actual_known_model_tokens_without_double_counting_prefix"] = None
    if not any(c["groups"]["D"]["finished"] for c in cases):
        for key in ["extra_D_report_records","extra_D_completion_events","D_saved_initial_chapter_repeated_generation_calls"]:
            summary[key] = None
    review = load(out/"附加盲审/附加盲审汇总.json")
    if review:
        by_run={r["run_id"]:r for r in review["rows"]}
        reviewed_pairs=[]
        for case in cases:
            for g in "ABCD":
                sample=case["groups"][g]
                sample["additional_blind_quality"]=by_run.get(sample["identity"].get("run_id"))
            B,C,D=(case["groups"][g].get("additional_blind_quality") for g in "BCD")
            metrics=case["paired_metrics"]
            if B and D:
                metrics["blind_quality_delta_D_minus_B"]=D["mean_score"]-B["mean_score"]
            if C and D:
                metrics["blind_quality_delta_D_minus_C"]=D["mean_score"]-C["mean_score"]
                metrics["C_and_D_both_blind_quality_gate_pass"]=C["quality_gate_passed"] and D["quality_gate_passed"]
                if metrics["C_and_D_both_blind_quality_gate_pass"] and metrics["cd_completed_pair"]:reviewed_pairs.append(case)
        summary["blind_review_completed_reports"]=review["completed_reports"]
        summary["blind_quality_delta_D_minus_B"]=stats([c["paired_metrics"].get("blind_quality_delta_D_minus_B") for c in cases])
        summary["blind_quality_delta_D_minus_C"]=stats([c["paired_metrics"].get("blind_quality_delta_D_minus_C") for c in cases])
        summary["blind_quality_gate_completed_CD_pairs"]=len(reviewed_pairs)
        summary["blind_quality_gate_post_fault_time_savings_rates"]=stats([c["paired_metrics"]["post_fault_time_savings_rate"] for c in reviewed_pairs])
        summary["blind_quality_gate_price_index_savings_rates"]=stats([c["paired_metrics"]["published_price_index_savings_rate"] for c in reviewed_pairs])
    result = {"analysis_date":datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(),"analysis_updated_at":datetime.now(timezone.utc).isoformat(),
        "run_directory":str(out),"run_status":manifest["status"],"summary":summary,"cases":cases,
        "boundaries":["费用为主Agent已知usage的公开原价指数，不含搜索/RAG/本地资源与折扣；未知usage不填0",
            "A/B端到端差值含模型、联网、缓存、返工波动；不同调用路径不能纯归因检查点",
            "只有五题各一次，不报告代表生产的P95；技术完成和内部质量门槛单列",
            "内部Critic的配对分差不是独立事实质量评测",
            "共享前缀分别加回C/D逻辑生命周期，真实总调用只计一次"]}
    if manifest.get("reused_completed_groups"):
        result["boundaries"].append("续跑明确复用B25历史已完成基线；跨日配对与全新A/B单列，复用样本费用不计本轮新增消耗。")
    write(out/"四组配对指标.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True)
    args=p.parse_args();result=analyze(args.input.resolve())
    print(json.dumps(result["summary"],ensure_ascii=False,indent=2))


if __name__=="__main__": main()
