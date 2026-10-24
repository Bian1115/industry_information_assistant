"""沿用冻结双裁判基线，同口径评估deepagents五题；允许生成与评分交叠。"""
from pathlib import Path
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import statistics
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "backend/eval/data/五篇报告fewshot复评与复核一致性-20261006"
DIMS = ["depth", "comprehensiveness", "readability", "requirements"]
NAMES = dict(zip(DIMS, ["深度", "全面性", "可读性", "需求覆盖"]))

def read(path):
    return json.loads(path.read_text())

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()

def summarize(scores, baseline):
    pairs = []
    for topic in sorted({r["topic"] for r in scores}):
        for judge in ["qwen", "claude"]:
            current = [r for r in scores if r["topic"] == topic and r["model"] == judge]
            previous = [r for r in baseline if r["topic"] == topic and r["model"] == judge]
            if len(current) != 2 or len(previous) != 2:
                continue
            values = {dim: {"multiagent": statistics.mean(r["result"]["dimensions"][dim]["score"] for r in previous),
                            "deepagents": statistics.mean(r["result"]["dimensions"][dim]["score"] for r in current)} for dim in DIMS}
            for value in values.values():
                value["delta_deepagents_minus_multiagent"] = value["deepagents"] - value["multiagent"]
            pairs.append({"id": topic, "judge": judge, "dimensions": values,
                          "delta_mean": statistics.mean(v["delta_deepagents_minus_multiagent"] for v in values.values())})
    aggregates = {}
    for judge in ["qwen", "claude"]:
        rows = [r for r in pairs if r["judge"] == judge]
        aggregates[judge] = {"tasks": len(rows), "dimensions": {
            dim: {key: statistics.mean(r["dimensions"][dim][key] for r in rows) if rows else None
                  for key in ["multiagent", "deepagents", "delta_deepagents_minus_multiagent"]} for dim in DIMS},
            "delta_mean": statistics.mean(r["delta_mean"] for r in rows) if rows else None}
    return {"paired_scores": pairs, "by_judge": aggregates, "notice": "差值为deepagents减多智能体；分数为模型裁判判断，不是人工金标，五题无显著性结论。"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    evaluator = module("quality_eval", ROOT / "backend/eval/scripts/评估五篇正文与Scout.py")
    validator = module("quality_validate", ROOT / "backend/eval/scripts/复评五篇质量与复核一致性.py")
    evaluator.OUT = out
    os.environ["CLAUDE_RELAY_MODEL"] = "claude-opus-5"
    os.environ["OPENAI_MODEL"] = "qwen3-max"
    baseline = read(BASE / "新质量评分.json")
    baseline_tasks = {r["id"]: r for r in read(BASE / "冻结报告输入.json")}
    baseline_config = read(BASE / "评估配置.json")
    if baseline_config["quality_prompt"]["sha256"] != evaluator.quality_prompt_metadata()["sha256"]:
        raise ValueError("基线与本轮质量提示词版本不同")
    dump(out / "基线双裁判评分冻结.json", baseline)
    dump(out / "基线报告输入冻结.json", list(baseline_tasks.values()))
    (out / "质量评分提示词快照.txt").write_text(evaluator.QUALITY)
    config = {"date": now(), "generation": str(args.generation.resolve()), "baseline": str(BASE),
              "baseline_generation_dates": "2026-10-03", "baseline_scoring_date": "2026-10-06",
              "models": {"qwen": "qwen3-max", "claude": "claude-opus-5"},
              "repeats": 2, "quality_prompt": evaluator.quality_prompt_metadata(),
              "qwen_request_options": {"model": "qwen3-max", "temperature": 0.3},
              "claude_temperature": "沿用服务默认；请求不显式指定", "dimension_scale": "0到10；四维等权",
              "evidence_cutoff_in_frozen_prompt": "2026-10-03；为保持与既有评分完全同口径而沿用，实际联网生成于2026-10-07",
              "comparison_limitations": ["系统整体对照，预算、提示词、规划与工具调度不同", "四天后的动态联网来源并未固定回放", "历史基线评分复用，不重新生成多智能体报告", "引用工具只核验提交的论断；不是全篇事实准确率"]}
    if (out / "评估配置.json").exists():
        old = read(out / "评估配置.json")
        config["date"] = old["date"]
        if config != old:
            raise ValueError("续评分配置发生变化")
    dump(out / "评估配置.json", config)
    scores, failures, submitted, futures, frozen = [], [], set(), {}, {}
    deadline = time.monotonic() + 4200
    def score(task, judge, repeat):
        key = f"题{task['id']}-质量-{judge}-{repeat}"
        prompt = evaluator.QUALITY + "\n本次重复编号" + str(repeat) + "\n用户任务：" + task["query"] + "\n完整正文：\n" + task["body"]
        result = evaluator.call(judge, prompt, key, 7000, validator.validate_quality,
                                request_options=config["qwen_request_options"] if judge == "qwen" else None)
        return {"topic": task["id"], "model": judge, "repeat": repeat, "key": key, "result": result}
    with cf.ThreadPoolExecutor(max_workers=2) as qwen_pool, cf.ThreadPoolExecutor(max_workers=1) as claude_pool:
        while True:
            for path in args.generation.glob("题目-*/尝试-*/结果.json"):
                row = read(path)
                if row["id"] in submitted:
                    continue
                submitted.add(row["id"])
                if row["status"] != "completed":
                    failures.append({"id": row["id"], "stage": "generation", "status": row["status"], "error": row.get("error")})
                    continue
                if row["prompt"] != baseline_tasks[row["id"]]["query"]:
                    raise ValueError("两组用户任务不一致")
                body = row["article"].split("## 参考文献", 1)[0].strip()
                task = {"id": row["id"], "query": row["prompt"], "body": body, "report": row["article"],
                        "article_sha256": hashlib.sha256(row["article"].encode()).hexdigest(), "generation_result": str(path)}
                frozen[row["id"]] = task
                dump(out / "deepagents报告输入冻结.json", list(frozen.values()))
                for judge in ["qwen", "claude"]:
                    pool = qwen_pool if judge == "qwen" else claude_pool
                    for repeat in [1, 2]:
                        future = pool.submit(score, task, judge, repeat)
                        futures[future] = {"id": row["id"], "judge": judge, "repeat": repeat}
            for future in list(futures):
                if not future.done():
                    continue
                job = futures.pop(future)
                try:
                    scores.append(future.result())
                except Exception as exc:
                    failures.append({**job, "stage": "quality", "error": str(exc)[:1000]})
                dump(out / "deepagents双裁判质量评分.json", sorted(scores, key=lambda r: (r["topic"], r["model"], r["repeat"])))
                dump(out / "五题质量配对对比.json", summarize(scores, baseline))
                print("评分完成", len(scores), "/20；失败", len(failures), flush=True)
            dump(out / "运行状态.json", {"date": now(), "generated_tasks_seen": len(submitted), "scores_completed": len(scores),
                                         "failures": failures, "pending_scores": len(futures), "status": "运行中"})
            if len(submitted) == 5 and not futures:
                break
            if time.monotonic() > deadline:
                failures.append({"stage": "orchestrator", "error": "等待生成或评分超过4200秒"})
                break
            time.sleep(3)
    dump(out / "运行状态.json", {"date": now(), "generated_tasks_seen": len(submitted), "scores_completed": len(scores),
                                 "failures": failures, "status": "完成" if len(scores) == 20 and not failures else "未完成"})
    dump(out / "五题质量配对对比.json", summarize(scores, baseline))
    if len(scores) != 20 or failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
