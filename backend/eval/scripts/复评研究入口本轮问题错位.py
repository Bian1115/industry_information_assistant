"""冻结真实入口材料，配对回放修复前后判断器；不运行研究图、不读写业务数据库。"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND / "app"))
from dotenv import load_dotenv

load_dotenv(BACKEND / ".env", override=False)
from service.llm_client import create_openai_client
from service.qwen_token_counter import get_qwen_token_counter
from service.session_task_service import fallback_task, query_sentence_index
from service.verification_budget import task_budget


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=BACKEND / "eval/data/第20题32轮待合并结果恢复续跑-20261008-184009/真实运行")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="qwen3-max")
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("未配置DASHSCOPE_API_KEY，不能执行真实模型回放")
    # 两组必须使用相同型号；历史型号不可用时显式另选，不隐藏模型变化。
    model = args.model
    os.environ["SESSION_TASK_MODEL"] = model
    probe = create_openai_client(api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        max_retries=0, read_timeout=30)
    catalog = [item.id for item in probe.models.list().data]
    save(out / "本次回放模型预检.json", {"model": model, "catalog_contains_model": model in catalog, "checked_at": now()})
    if model not in catalog:
        raise RuntimeError("模型目录不存在指定型号，停止整批回放")
    preflight_request = {"model": model, "messages": [{"role": "user", "content": '只返回JSON：{"ok":true}'}],
                         "temperature": 0, "max_tokens": 32}
    if args.disable_thinking:
        preflight_request["extra_body"] = {"enable_thinking": False}
    try:
        preflight = probe.chat.completions.create(**preflight_request)
        save(out / "本次回放最小请求预检.json", {"status": "completed", "request": preflight_request, "response": preflight.model_dump()})
    except Exception as exc:
        save(out / "本次回放最小请求预检.json", {"status": "failed", "request": preflight_request,
                                           "error_type": type(exc).__name__, "error": str(exc), "checked_at": now()})
        raise RuntimeError("最小模型请求预检失败，停止整批回放") from exc
    modules = {arm: load_module(f"service.入口错位复评_{arm}", out / f"{arm}源码/research_context_service.py")
               for arm in ("修复前", "修复后")}
    local = threading.local()

    def factory(**kwargs):
        client = create_openai_client(**kwargs, read_timeout=120)

        def create(**request):
            if args.disable_thinking:
                request["extra_body"] = {"enable_thinking": False}
            case = local.case
            index = len(case["calls"]) + 1
            entry = {"request_index": index, "started_at": now(), "request": copy.deepcopy(request)}
            path = case["directory"] / f"模型调用-{index}.json"
            save(path, entry)
            start = time.monotonic()
            try:
                response = client.chat.completions.create(**request)
                entry.update(status="completed", response=response.model_dump(), ended_at=now(), elapsed_seconds=time.monotonic() - start)
                return response
            except Exception as exc:
                entry.update(status="failed", error_type=type(exc).__name__, error=str(exc), ended_at=now(), elapsed_seconds=time.monotonic() - start)
                raise
            finally:
                save(path, entry)
                case["calls"].append({"path": str(path.relative_to(out)), "status": entry["status"],
                                      "elapsed_seconds": entry["elapsed_seconds"],
                                      "usage": entry.get("response", {}).get("usage")})

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    for module in modules.values():
        module.create_openai_client = factory
    inputs = {i: json.loads((args.source / f"第{i}轮/会话任务输入.json").read_text()) for i in range(1, 25)}
    gold_path = BACKEND / "eval/data/研究入口继承判断准确率标注-20261017/逐轮标注.jsonl"
    source_list = [args.source / f"第{i}轮/会话任务输入.json" for i in inputs]
    config = {"started_at": now(), "source": str(args.source), "model": model, "temperature": 0,
              "max_tokens": 3000, "workers": args.workers, "planned_cases": 48,
              "read_timeout_seconds": 120, "sdk_max_retries": 0,
              "annotation_sha256": sha(gold_path),
              "disable_thinking": args.disable_thinking,
              "scope": "24轮固定入口上下文，修复前后各一次；首轮无历史不调用模型，后续46个初始模型请求；按生产逻辑允许有界修正。没有向判断器传入期望标签。",
              "source_files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in source_list],
              "code_files": [{"path": str(out / f"{arm}源码/research_context_service.py"), "sha256": sha(out / f"{arm}源码/research_context_service.py")} for arm in modules]}
    save(out / "配对回放运行清单.json", config)
    counter = get_qwen_token_counter()
    # 单次预热，随后各请求复用同一只读tokenizer。
    counter.count_text("本轮任务")

    def run_case(arm, i):
        module = modules[arm]
        original = inputs[i]
        directory = out / "真实模型回放" / arm / f"第{i}轮"
        directory.mkdir(parents=True, exist_ok=True)
        local.case = {"directory": directory, "calls": []}
        names = {f.name for f in fields(module.ResearchContext)}
        values = {k: copy.deepcopy(v) for k, v in original.items() if k in names}
        values.update(resolved_query=original["original_query"], is_follow_up=False, used_memory_ids=[], assumptions=[],
                      unresolved_references=[], session_task={}, task_metrics={},
                      task_trace={"sentence_index": query_sentence_index(original["original_query"], f"回放第{i}轮")})
        context = module.ResearchContext(**values)
        save(directory / "冻结入口输入.json", context.to_dict())
        resolver = module.ResearchContextService.__new__(module.ResearchContextService)
        resolver.token_counter = counter
        started = time.monotonic()
        metadata = {"arm": arm, "turn_id": i, "started_at": now(), "case_status": "completed"}
        budget_config = {"max_seconds": 400, "max_requests": 4, "max_tokens_by_model": {model: 300000}}
        with task_budget(budget_config) as ledger:
            try:
                if not (context.conversation_context or context.memory_context or context.artifact_candidates):
                    # 与生产build无历史分支一致，保持首轮口径，不为提高准确率改动它。
                    context.session_task = fallback_task(context.original_query, "没有历史或关闭会话判断")
                else:
                    resolver._resolve_query(context)
            except Exception as exc:
                metadata.update(case_status="error", error_type=type(exc).__name__, error=str(exc))
            save(directory / "共享预算账本.json", ledger.report())
        save(directory / "本轮判断结果.json", context.to_dict())
        metadata.update(ended_at=now(), elapsed_seconds=time.monotonic() - started,
                        calls=local.case["calls"], task_status=context.session_task.get("status"),
                        model_output_available=any(call["status"] == "completed" for call in local.case["calls"]) if i > 1 else None,
                        path=str(directory.relative_to(out)))
        if i > 1 and not metadata["model_output_available"]:
            metadata["case_status"] = "infrastructure_failed"
        save(directory / "回放状态.json", metadata)
        return metadata

    existing, jobs = [], []
    for i in inputs:
        for arm in modules:
            result_path = out / "真实模型回放" / arm / f"第{i}轮/回放状态.json"
            if result_path.exists():
                existing.append(json.loads(result_path.read_text()))
            else:
                jobs.append((arm, i))
    rows = existing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, arm, i): (arm, i) for arm, i in jobs}
        for future in as_completed(futures):
            arm, i = futures[future]
            row = future.result()
            rows.append(row)
            save(out / "配对回放进度.json", {"updated_at": now(), "completed_cases": len(rows), "planned_cases": 48,
                                         "cases": sorted(rows, key=lambda r: (r["turn_id"], r["arm"]))})
            print(json.dumps({"arm": arm, "turn_id": i, "completed_cases": len(rows), "task_status": row["task_status"],
                              "model_calls": len(row["calls"]), "seconds": round(row["elapsed_seconds"], 2)}, ensure_ascii=False), flush=True)
    config.update(ended_at=now(), status="infrastructure_failed" if any(r["case_status"] == "infrastructure_failed" for r in rows) else "completed",
                  completed_cases=len(rows), infrastructure_failed_cases=sum(r["case_status"] == "infrastructure_failed" for r in rows),
                  source_unchanged=all(sha(ROOT / x["path"]) == x["sha256"] for x in config["source_files"]))
    save(out / "配对回放运行清单.json", config)


if __name__ == "__main__":
    main()
