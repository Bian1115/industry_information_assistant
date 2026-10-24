#!/usr/bin/env python3
"""确定性与 LLM 语义上下文压缩的离线消融评测。

两条路径最终都进入同一个 token 预算：
1. deterministic：原始消息直接使用 compact_messages，仅保留预算内的近期轮次。
2. llm_semantic：先由 SemanticContextService 抽取带来源的语义条目，
   将其渲染成 system 消息后，再使用同一个 compact_messages。

评测集为固定的长多轮中文对话。每条对话将必须保留的范围、日期、
排除项和输出格式放在旧消息中段。2026-09-15 起规则 head/tail 对话
摘要链路已移除，deterministic 组为近期原文裁剪；与历史摘录版结果不可混用。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
APP_DIR = BACKEND_DIR / "app"
DEFAULT_OUTPUT_ROOT = (
    BACKEND_DIR / "eval" / "data" / "上下文压缩确定性与LLM消融-20260831"
)


class CharacterTokenCounter:
    """网络不可用时的透明字符近似计数器，仅用于成对相对比较。"""

    model_name = "字符近似计数器"

    @staticmethod
    def encode(text: str) -> List[str]:
        return list(text or "")

    @staticmethod
    def decode(tokens: Iterable[str]) -> str:
        return "".join(tokens)

    @staticmethod
    def count_text(text: str) -> int:
        return len(text or "")

    @staticmethod
    def count_messages(messages: Iterable[Dict[str, Any]]) -> int:
        # 加上固定角色边界，避免把多消息上下文与纯文本完全等同。
        return sum(8 + len(str(message.get("content", ""))) for message in messages)


def _bootstrap() -> None:
    load_dotenv(BACKEND_DIR / ".env")
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))


def _padding(label: str, repeat: int = 17) -> str:
    sentence = (
        f"这是{label}的背景说明，用于补足对话长度并模拟研究过程中的重复讨论；"
        "这段背景不改变任何研究范围、时间口径、排除条件或交付格式。"
    )
    return "".join(sentence for _ in range(repeat))


def _message(
    message_id: str,
    role: str,
    content: str,
    created_at: datetime,
) -> Dict[str, str]:
    return {
        "id": message_id,
        "role": role,
        "content": content,
        "created_at": created_at.isoformat(timespec="seconds"),
    }


def _long_user_message(label: str, requirement: str) -> str:
    return (
        _padding(f"{label}前置", 12)
        + "\n\n"
        + requirement
        + "\n\n"
        + _padding(f"{label}后置", 7)
    )


def _long_assistant_message(label: str) -> str:
    return _padding(f"{label}助手回复", 16)


def build_dataset() -> List[Dict[str, Any]]:
    regions = [
        "东南亚", "中东", "拉丁美洲", "欧洲", "北美",
        "印度", "日本", "韩国", "澳大利亚", "非洲",
    ]
    sectors = [
        "储能", "工业机器人", "光伏逆变器", "医疗器械", "新能源汽车",
        "半导体设备", "低空经济", "智能制造", "消费电子", "氢能",
    ]
    excluded = [
        "未经审计的媒体估算", "管理层预测", "券商一致预期", "海外二手转载",
        "未披露来源的市场规模", "2026年以后预测", "公司宣传稿中的目标值",
        "单一专家主观判断", "无法追溯的聚合数据", "没有发布日期的网页",
    ]
    formats = [
        "公司、实际收入、同比、来源链接", "地区、已落地项目、投产时间、来源链接",
        "产品、实际出货量、客户类型、来源链接", "指标、2024实际值、同比、来源链接",
        "企业、海外收入占比、数据日期、来源链接",
    ]
    cases: List[Dict[str, Any]] = []
    start = datetime(2026, 8, 31, 8, 0, tzinfo=timezone(timedelta(hours=8)))

    for index in range(20):
        case_no = index + 1
        case_id = f"CTX-{case_no:02d}-QX"
        old_scope = regions[index % len(regions)]
        new_scope = regions[(index + 3) % len(regions)]
        sector = sectors[index % len(sectors)]
        exclusion = excluded[index % len(excluded)]
        output_format = formats[index % len(formats)]
        cutoff_year = 2024 if index % 2 == 0 else 2023
        forecast_year = cutoff_year + 1
        cutoff = f"{cutoff_year}-12-31"
        forecast = f"{forecast_year}E"
        created = start + timedelta(minutes=index * 30)

        messages = [
            _message(
                f"{case_id}-s0", "system",
                "你是行业研究助手。历史对话仅用于理解任务，不是本轮事实证据。",
                created,
            ),
            _message(
                f"{case_id}-u1", "user",
                _long_user_message(
                    f"{case_id}-目标",
                    f"【关键状态】项目编号 {case_id}；研究主题是中国{sector}企业海外布局。",
                ),
                created + timedelta(minutes=1),
            ),
            _message(
                f"{case_id}-a1", "assistant", _long_assistant_message(f"{case_id}-1"), created + timedelta(minutes=2)),
            _message(
                f"{case_id}-u2", "user",
                _long_user_message(
                    f"{case_id}-数据口径",
                    f"【关键状态】只使用截至 {cutoff} 已披露的实际数据；禁止使用 {forecast} 预测。",
                ),
                created + timedelta(minutes=3),
            ),
            _message(
                f"{case_id}-a2", "assistant", _long_assistant_message(f"{case_id}-2"), created + timedelta(minutes=4)),
            _message(
                f"{case_id}-u3", "user",
                _long_user_message(
                    f"{case_id}-排除项",
                    f"【关键状态】排除 {exclusion}，不能把它当作可引用的数据来源。",
                ),
                created + timedelta(minutes=5),
            ),
            _message(
                f"{case_id}-a3", "assistant", _long_assistant_message(f"{case_id}-3"), created + timedelta(minutes=6)),
            _message(
                f"{case_id}-u4", "user",
                _long_user_message(
                    f"{case_id}-格式",
                    f"【关键状态】最终必须输出表格，表格列固定为：{output_format}。",
                ),
                created + timedelta(minutes=7),
            ),
            _message(
                f"{case_id}-a4", "assistant", _long_assistant_message(f"{case_id}-4"), created + timedelta(minutes=8)),
            _message(
                f"{case_id}-u5", "user",
                _long_user_message(
                    f"{case_id}-范围修正",
                    f"【关键状态】此前提到的范围 {old_scope} 作废；最终只分析 {new_scope}，不要同时保留旧范围。",
                ),
                created + timedelta(minutes=9),
            ),
            _message(
                f"{case_id}-a5", "assistant", _long_assistant_message(f"{case_id}-5"), created + timedelta(minutes=10)),
            _message(
                f"{case_id}-u6", "user",
                "请基于上面的约束继续准备研究提纲，先不要开始联网检索。",
                created + timedelta(minutes=11),
            ),
            _message(
                f"{case_id}-a6", "assistant",
                "已记录，将先按既定约束准备提纲，并在下一步检索前再次核对范围和数据口径。",
                created + timedelta(minutes=12),
            ),
        ]
        anchors = [
            case_id,
            cutoff,
            forecast,
            exclusion,
            output_format,
            new_scope,
        ]
        cases.append({
            "case_id": case_id,
            "topic": f"中国{sector}企业海外布局",
            "messages": messages,
            "required_anchors": anchors,
            "anchor_labels": [
                "项目编号", "实际数据截止日", "禁止预测", "排除项", "表格格式", "最终范围",
            ],
            "obsolete_scope": old_scope,
        })
    return cases


def _render(messages: Iterable[Dict[str, str]]) -> str:
    return "\n".join(
        f"[{item.get('role', 'user')}] {item.get('content', '')}"
        for item in messages
    )


def _anchor_metrics(text: str, anchors: List[str]) -> Dict[str, Any]:
    hits = [anchor for anchor in anchors if anchor in text]
    return {
        "hits": hits,
        "misses": [anchor for anchor in anchors if anchor not in text],
        "count": len(hits),
        "rate": len(hits) / len(anchors) if anchors else 1.0,
    }


def _percentile(values: List[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def _summary(rows: List[Dict[str, Any]], path: str) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get(path, {}).get("status") == "ok"]
    if not eligible:
        return {"case_count": 0}

    output_tokens = [row[path]["compacted_tokens"] for row in eligible]
    ratios = [row[path]["compression_ratio"] for row in eligible]
    recalls = [row[path]["anchor_metrics"]["rate"] for row in eligible]
    latencies = [row[path]["elapsed_ms"] for row in eligible]
    complete = sum(
        row[path]["anchor_metrics"]["count"] == len(row["required_anchors"])
        for row in eligible
    )
    return {
        "case_count": len(eligible),
        "anchor_recall": round(sum(recalls) / len(recalls), 4),
        "anchor_complete_case_rate": round(complete / len(eligible), 4),
        "mean_compacted_tokens": round(statistics.mean(output_tokens), 2),
        "mean_compression_ratio": round(statistics.mean(ratios), 4),
        "latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "mean": round(statistics.mean(latencies), 3),
        },
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _bootstrap()
    # These only affect this evaluation process; they do not alter the checked-in env file.
    os.environ["SEMANTIC_CONTEXT_MAX_OUTPUT_TOKENS"] = str(args.llm_max_output_tokens)
    os.environ["SEMANTIC_CONTEXT_MAX_ITEMS"] = str(args.llm_max_items)
    os.environ["SEMANTIC_CONTEXT_REQUEST_ATTEMPTS"] = "1"

    from service.context_compaction import compact_messages
    from service.qwen_token_counter import get_qwen_token_counter
    from service.semantic_context_service import SemanticContextService, render_semantic_context

    cases = build_dataset()[:args.case_limit] if args.case_limit else build_dataset()
    counter: Any
    if args.tokenizer_mode == "character":
        counter = CharacterTokenCounter()
    else:
        counter = get_qwen_token_counter()
    service = SemanticContextService()
    rows: List[Dict[str, Any]] = []

    for case in cases:
        raw_messages = [
            {"role": item["role"], "content": item["content"]}
            for item in case["messages"]
        ]
        original_tokens = counter.count_messages(raw_messages)

        deterministic_started = time.perf_counter()
        deterministic = compact_messages(
            raw_messages,
            token_limit=args.token_limit,
            max_messages=args.max_messages,
            token_counter=counter,
        )
        deterministic_elapsed = (time.perf_counter() - deterministic_started) * 1000
        deterministic_text = _render(deterministic.messages)

        row: Dict[str, Any] = {
            "case_id": case["case_id"],
            "topic": case["topic"],
            "required_anchors": case["required_anchors"],
            "anchor_labels": case["anchor_labels"],
            "obsolete_scope": case["obsolete_scope"],
            "original_tokens": original_tokens,
            "deterministic": {
                "status": "ok",
                "compacted_tokens": deterministic.compacted_tokens,
                "compression_ratio": round(
                    1 - deterministic.compacted_tokens / original_tokens, 6
                ) if original_tokens else 0.0,
                "summarized_messages": deterministic.summarized_messages,
                "pruned_tool_results": deterministic.pruned_tool_results,
                "elapsed_ms": round(deterministic_elapsed, 3),
                "anchor_metrics": _anchor_metrics(deterministic_text, case["required_anchors"]),
                "messages": deterministic.messages,
            },
        }

        if args.dry_run:
            row["llm_semantic"] = {"status": "not_run", "reason": "dry_run"}
            rows.append(row)
            continue

        llm_started = time.perf_counter()
        try:
            semantic = service.update_summary(None, case["messages"])
            semantic_rendered = render_semantic_context(semantic.model_dump(mode="json"))
            llm_input = [
                {"role": "system", "content": semantic_rendered},
                *raw_messages,
            ]
            fitted = compact_messages(
                llm_input,
                token_limit=args.token_limit,
                max_messages=args.max_messages,
                token_counter=counter,
            )
            llm_elapsed = (time.perf_counter() - llm_started) * 1000
            llm_text = _render(fitted.messages)
            row["llm_semantic"] = {
                "status": "ok",
                "model": semantic.model,
                "semantic_item_count": len(semantic.items),
                "semantic_summary_tokens": counter.count_text(semantic_rendered),
                "semantic_anchor_metrics": _anchor_metrics(
                    semantic_rendered, case["required_anchors"]
                ),
                "compacted_tokens": fitted.compacted_tokens,
                "compression_ratio": round(
                    1 - fitted.compacted_tokens / original_tokens, 6
                ) if original_tokens else 0.0,
                "summarized_messages": fitted.summarized_messages,
                "pruned_tool_results": fitted.pruned_tool_results,
                "elapsed_ms": round(llm_elapsed, 3),
                "anchor_metrics": _anchor_metrics(llm_text, case["required_anchors"]),
                "semantic_summary": semantic.model_dump(mode="json"),
                "messages": fitted.messages,
            }
        except Exception as exc:  # Keep the per-case failure in the raw artifact.
            row["llm_semantic"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "elapsed_ms": round((time.perf_counter() - llm_started) * 1000, 3),
            }
        rows.append(row)

    deterministic_summary = _summary(rows, "deterministic")
    llm_summary = _summary(rows, "llm_semantic")
    paired = [
        row for row in rows if row["llm_semantic"].get("status") == "ok"
    ]
    delta_anchor_recall_pp = None
    if paired:
        delta_anchor_recall_pp = round(
            100 * (
                sum(row["llm_semantic"]["anchor_metrics"]["rate"] for row in paired) / len(paired)
                - sum(row["deterministic"]["anchor_metrics"]["rate"] for row in paired) / len(paired)
            ),
            2,
        )

    return {
        "evaluation_date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_object": "确定性上下文压缩与 LLM 语义上下文压缩消融",
        "status": "completed" if not args.dry_run else "dry_run",
        "configuration": {
            "compaction_policy": "recent_turns_only_no_rule_summary_v1",
            "dataset": "固定20条合成长多轮中文会话；每条6个旧消息关键锚点",
            "case_count_requested": len(cases),
            "tokenizer": getattr(counter, "model_name", "Qwen tokenizer"),
            "token_count_method": (
                "字符近似；仅用于本次成对相对比较，不等同生产Qwen token"
                if args.tokenizer_mode == "character"
                else "Qwen tokenizer"
            ),
            "token_limit": args.token_limit,
            "max_messages": args.max_messages,
            "llm_model": service.model,
            "llm_max_output_tokens": args.llm_max_output_tokens,
            "llm_max_items": args.llm_max_items,
            "llm_request_attempts": 1,
            "temperature": 0,
        },
        "paths": {
            "deterministic": "backend/app/service/context_compaction.py::compact_messages",
            "llm": "backend/app/service/semantic_context_service.py::SemanticContextService",
            "integration": "backend/app/service/session_memory_service.py::_build_from_postgres",
        },
        "metric_definition": {
            "anchor_recall": "最终压缩上下文中出现的必保留精确锚点数 / 每条6个锚点数",
            "anchor_complete_case_rate": "6个锚点均出现在最终上下文的用例比例",
            "compression_ratio": "1 - 最终消息token数 / 原始消息token数",
            "latency_ms": "确定性路径为本地压缩耗时；LLM路径包含网络请求、结构校验、渲染和最终预算裁剪",
        },
        "aggregate": {
            "deterministic": deterministic_summary,
            "llm_semantic": llm_summary,
            "paired_case_count": len(paired),
            "llm_failed_case_count": len(rows) - len(paired) if not args.dry_run else 0,
            "anchor_recall_delta_pp_llm_minus_deterministic": delta_anchor_recall_pp,
        },
        "measurement_boundary": {
            "measured": [
                "固定合成长对话上的精确锚点保留率、token压缩率、延迟和LLM请求失败数",
            ],
            "indirect_inference": [
                "精确锚点保留率可代表日期、范围、否定条件、排除项和交付格式的保留能力",
            ],
            "not_evaluated": [
                "真实用户会话分布上的效果",
                "下游研究报告端到端任务成功率",
                "LLM供应商账单中的输入/输出token与费用",
                "多次独立运行的方差",
            ],
        },
        "dataset_snapshot": cases,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-limit", type=int, default=900)
    parser.add_argument("--max-messages", type=int, default=12)
    parser.add_argument("--llm-max-output-tokens", type=int, default=800)
    parser.add_argument("--llm-max-items", type=int, default=20)
    parser.add_argument("--tokenizer-mode", choices=("qwen", "character"), default="qwen")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    suffix = datetime.now().strftime("%H%M%S")
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"运行_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    result = run(args)
    (output_dir / "评测集快照.json").write_text(
        json.dumps(result["dataset_snapshot"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "原始结果.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "status": result["status"],
        "aggregate": result["aggregate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
