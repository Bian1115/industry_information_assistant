#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""Create a non-overwriting comparison of LongMemEval result directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional


BACKEND_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_longmemeval_regression import (  # noqa: E402
    aggregate_metrics,
    read_jsonl as read_result_rows,
)

DEFAULT_OUTPUT_ROOT = (
    BACKEND_DIR / "eval" / "data" / "longmemeval_s" / "comparisons"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def metric_file(path: Path) -> Path:
    return path / "metrics.json" if path.is_dir() else path


def parse_run(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        if label.strip():
            return label.strip(), Path(raw_path)
    path = Path(value)
    return path.parent.name if path.name == "metrics.json" else path.name, path


def load_question_subset(path: Path) -> list[str]:
    payload = read_json(path)
    question_ids = payload.get("question_ids") if isinstance(payload, Mapping) else payload
    if not isinstance(question_ids, list) or not question_ids:
        raise ValueError(f"{path} 必须是非空 question_ids 数组或包含该数组的对象")
    normalized = [str(question_id) for question_id in question_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{path} 中包含重复 question_id")
    return normalized


def percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def number(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def provider_description(metadata: Mapping[str, Any]) -> str:
    provider = metadata.get("provider")
    if isinstance(provider, Mapping):
        name = provider.get("provider") or metadata.get("provider_key")
        version = provider.get("version")
        return f"{name} {version}".strip() if version else str(name)
    if metadata.get("context_mode"):
        return str(metadata["context_mode"])
    return "project-compressed-memory"


def validate_compatibility(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("至少需要一个 --run")
    keys = (
        "dataset_sha256",
        "manifest_question_id_list_sha256",
        "selected_question_id_list_sha256",
        "mode",
        "top_k",
    )
    expected = {key: runs[0]["metadata"].get(key) for key in keys}
    expected_total_rows = runs[0]["metrics"].get("total_rows")
    if not isinstance(expected_total_rows, int) or expected_total_rows < 1:
        raise ValueError(f"{runs[0]['label']} 的 total_rows 无效：{expected_total_rows!r}")
    if expected_total_rows != 100 and not expected["selected_question_id_list_sha256"]:
        raise ValueError("部分题集对比必须记录 selected_question_id_list_sha256")
    for run in runs[1:]:
        for key, value in expected.items():
            actual = run["metadata"].get(key)
            if actual != value:
                raise ValueError(
                    f"{run['label']} 的 {key}={actual!r}，基准为 {value!r}"
                )
        actual_total_rows = run["metrics"].get("total_rows")
        if actual_total_rows != expected_total_rows:
            raise ValueError(
                f"{run['label']} 的 total_rows={actual_total_rows!r}，"
                f"基准为 {expected_total_rows!r}"
            )
    expected["total_rows"] = expected_total_rows
    return expected


def build_comparison(
    entries: list[tuple[str, Path]],
    selected_question_ids: Optional[list[str]] = None,
) -> dict:
    runs = []
    labels = set()
    selected_hash = None
    if selected_question_ids:
        selected_hash = hashlib.sha256(
            "\n".join(selected_question_ids).encode()
        ).hexdigest()
    for label, raw_path in entries:
        if label in labels:
            raise ValueError(f"重复标签：{label}")
        labels.add(label)
        path = metric_file(raw_path)
        payload = read_json(path)
        metadata = dict(payload.get("metadata") or {})
        metrics = payload.get("metrics") or payload
        if selected_question_ids:
            result_path = raw_path / "questions.jsonl" if raw_path.is_dir() else path.parent / "questions.jsonl"
            if not result_path.exists() and metadata.get("result_file"):
                result_path = Path(str(metadata["result_file"]))
            if not result_path.exists():
                raise FileNotFoundError(f"{label} 缺少逐题结果：{result_path}")
            latest_by_id = {}
            selected_set = set(selected_question_ids)
            for row in read_result_rows(result_path):
                question_id = str(row.get("question_id"))
                if question_id in selected_set:
                    latest_by_id[question_id] = row
            missing = [
                question_id
                for question_id in selected_question_ids
                if question_id not in latest_by_id
            ]
            if missing:
                raise ValueError(f"{label} 缺少所选逐题结果：{missing}")
            metrics = aggregate_metrics(
                [latest_by_id[question_id] for question_id in selected_question_ids]
            )
            metadata["selected_question_ids"] = selected_question_ids
            metadata["selected_question_id_list_sha256"] = selected_hash
        runs.append({
            "label": label,
            "metrics_path": str(path.resolve()),
            "provider": provider_description(metadata),
            "metadata": metadata,
            "metrics": metrics,
        })
    compatibility = validate_compatibility(runs)
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection": {
            "question_count": len(selected_question_ids),
            "question_id_list_sha256": selected_hash,
            "question_ids": selected_question_ids,
        } if selected_question_ids else None,
        "compatibility": compatibility,
        "runs": runs,
    }


def markdown(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# LongMemEval-S 记忆系统基线对比",
        "",
        f"- 生成时间：{comparison['created_at']}",
        f"- 模式：{comparison['compatibility']['mode']}",
        f"- Top-K：{comparison['compatibility']['top_k']}",
        "",
        "## 总指标",
        "",
        "| 标签 | 记忆/上下文系统 | 成功题 | Any Recall | All Recall | Macro Recall | Micro Recall | MRR | QA Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in comparison["runs"]:
        metrics = run["metrics"]
        retrieval = metrics.get("retrieval") or {}
        lines.append(
            f"| {run['label']} | {run['provider']} | "
            f"{metrics.get('successful_questions', 0)}/{metrics.get('total_rows', 0)} | "
            f"{percent(retrieval.get('any_recall'))} | "
            f"{percent(retrieval.get('all_recall'))} | "
            f"{percent(retrieval.get('macro_session_recall'))} | "
            f"{percent(retrieval.get('micro_session_recall'))} | "
            f"{number(retrieval.get('mrr'))} | "
            f"{percent(metrics.get('qa_accuracy'))} |"
        )

    question_types = sorted({
        question_type
        for run in comparison["runs"]
        for question_type in (run["metrics"].get("by_question_type") or {})
    })
    lines.extend([
        "",
        "## 分题型 QA Accuracy",
        "",
        "| 题型 | " + " | ".join(run["label"] for run in comparison["runs"]) + " |",
        "|---|" + "---:|" * len(comparison["runs"]),
    ])
    for question_type in question_types:
        values = []
        for run in comparison["runs"]:
            typed = (run["metrics"].get("by_question_type") or {}).get(question_type) or {}
            values.append(percent(typed.get("qa_accuracy")))
        lines.append(f"| {question_type} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## 解读限制",
        "",
        "- Oracle Raw 是回答能力上界，不是可部署记忆系统。",
        "- Retrieved Raw 用原始会话回填，用于隔离“检索丢失”和“压缩丢失”。",
        "- 外部 OSS/自托管 Provider 的成绩只代表 provider_manifest.json 中的当次版本与模型组合。",
        "- 只有数据 SHA、固定题清单、模式和 Top-K 均一致的运行才会生成本报告。",
    ])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="可重复；格式 LABEL=/path/to/run 或 /path/to/metrics.json",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name")
    parser.add_argument(
        "--question-id-file",
        type=Path,
        help="可选；按同一固定题单从每个运行的 questions.jsonl 重新聚合",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selected_question_ids = (
        load_question_subset(args.question_id_file) if args.question_id_file else None
    )
    comparison = build_comparison(
        [parse_run(value) for value in args.run], selected_question_ids
    )
    name = args.name or f"{datetime.now():%Y%m%d_%H%M%S}_memory_baselines"
    output_dir = args.output_root / name
    if output_dir.exists():
        raise FileExistsError(f"对比目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "comparison.json", comparison)
    report = markdown(comparison)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"对比目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
