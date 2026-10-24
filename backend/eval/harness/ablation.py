"""Paired H0/H1 ablation runner for the Agent Harness.

H0 reproduces the historical direct-dispatch semantics from commit f9a9eda6:
resolve a handler and await it with the raw argument dictionary. H1 uses the
current ToolRuntime with validation, risk checks, timeout, cancellation,
result validation, concurrency policy, and middleware terminal events.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from service.research_control import ResearchCancellationToken
from service.tool_runtime import (
    ToolCallContext,
    ToolDefinition,
    ToolExecutionResult,
    ToolRuntime,
)

from .fake_tools import HarnessFakeTools


BASELINE_COMMIT = "f9a9eda6"
BASELINE_FILE = "backend/app/service/tool_executor.py"
BASELINE_SEMANTICS = "handler = self.handlers.get(tool_name); return await handler(params, context)"
FIXED_SEED = 20260830


@dataclass(frozen=True)
class AblationCase:
    case_id: str
    category: str
    tool_name: str
    raw_args: Dict[str, Any]
    context: Dict[str, Any]


class LegacyDirectRuntime:
    """Faithful direct-call policy used by the pre-Harness ToolExecutor."""

    async def execute(
        self,
        definition: ToolDefinition,
        raw_args: Dict[str, Any],
        context: ToolCallContext,
    ) -> Any:
        value = definition.handler(raw_args, context)
        if inspect.isawaitable(value):
            value = await value
        return value


class CaptureTerminalMiddleware:
    """Capture terminal runtime events without flooding benchmark stdout."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    async def before(
        self, definition: ToolDefinition, args: Any, context: ToolCallContext
    ) -> None:
        return None

    async def after(
        self, definition: ToolDefinition, result: Any, context: ToolCallContext
    ) -> None:
        self.events.append({
            "case_id": context.metadata.get("case_id"),
            "tool": definition.name,
            "outcome": "success",
        })

    async def on_error(
        self, definition: ToolDefinition, error: BaseException, context: ToolCallContext
    ) -> None:
        self.events.append({
            "case_id": context.metadata.get("case_id"),
            "tool": definition.name,
            "outcome": "error",
            "error_type": type(error).__name__,
            "error_code": getattr(error, "code", None),
        })


def generate_manifest() -> List[AblationCase]:
    cases: List[AblationCase] = []

    def add(category: str, tool_name: str, raw_args: Dict[str, Any]) -> None:
        index = len(cases) + 1
        cases.append(AblationCase(
            case_id=f"HARNESS-{index:04d}",
            category=category,
            tool_name=tool_name,
            raw_args=raw_args,
            context={
                "session_id": f"session-{index:04d}",
                "run_id": f"run-{index:04d}",
                "user_id": f"user-{index % 7:02d}",
                "approved": False,
            },
        ))

    for index in range(30):
        add("valid", "fake_echo_success", {
            "text": f"echo-{index:02d}",
            "repeat": index % 5 + 1,
        })
    for index in range(30):
        add("valid", "fake_search_success", {
            "query": f"industry-query-{index:02d}",
            "count": index % 5 + 1,
        })
    for index in range(20):
        add("valid", "fake_context_probe", {"label": f"probe-{index:02d}"})
    for index in range(20):
        add("valid", "fake_large_result", {"item_count": 10 + index * 5})

    for index in range(20):
        add("invalid_arguments", "fake_echo_success", {"repeat": index % 5 + 1})
    for _ in range(20):
        add("invalid_arguments", "fake_echo_success", {"text": "", "repeat": 1})
    for index in range(20):
        add("invalid_arguments", "fake_search_success", {
            "query": f"invalid-low-{index:02d}", "count": 0,
        })
    for index in range(20):
        add("invalid_arguments", "fake_search_success", {
            "query": f"invalid-high-{index:02d}", "count": 21,
        })

    for index in range(60):
        add("invalid_result", "fake_invalid_result", {"text": f"bad-{index:02d}"})

    for index in range(60):
        add("unapproved_write", "fake_write_side_effect", {
            "record_id": index + 1,
            "value": f"write-{index:02d}",
        })
    for index in range(40):
        add("unapproved_dangerous", "fake_dangerous_side_effect", {
            "record_id": index + 1,
            "value": f"danger-{index:02d}",
        })

    for index in range(60):
        add("timeout", "fake_slow_timeout", {
            "delay_ms": 30 + index % 5 * 5,
            "value": f"slow-{index:02d}",
        })
    for index in range(60):
        add("cancellation", "fake_cancellable", {
            "delay_ms": 30 + index % 5 * 5,
            "value": f"cancel-{index:02d}",
        })

    for index in range(20):
        add("dependency_failure", "fake_dependency_failure", {
            "message": f"provider-503-{index:02d}",
        })
    for index in range(20):
        add("serial_concurrency", "fake_serial_counter", {
            "delay_ms": 3,
            "sequence": index,
        })

    if len(cases) != 500:
        raise AssertionError(f"expected 500 cases, generated {len(cases)}")
    return cases


def _state_snapshot(fake_tools: HarnessFakeTools, tool_name: str) -> Dict[str, int]:
    state = fake_tools.state
    return {
        "calls": state.call_counts[tool_name],
        "side_effects": state.side_effect_counts[tool_name],
        "cancellations": state.cancellation_counts[tool_name],
    }


def _context_for(case: AblationCase) -> ToolCallContext:
    return ToolCallContext(
        session_id=case.context["session_id"],
        run_id=case.context["run_id"],
        user_id=case.context["user_id"],
        approved=case.context["approved"],
        metadata={"case_id": case.case_id},
    )


async def _await_handler_start(
    fake_tools: HarnessFakeTools,
    tool_name: str,
    previous_calls: int,
    timeout_seconds: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while fake_tools.state.call_counts[tool_name] <= previous_calls:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"handler did not start: {tool_name}")
        await asyncio.sleep(0)


def _value_from_runtime(value: Any) -> Any:
    if isinstance(value, ToolExecutionResult):
        return value.value
    return value


async def _observe_case(
    variant: str,
    runtime: Any,
    definition: ToolDefinition,
    case: AblationCase,
    fake_tools: HarnessFakeTools,
    *,
    concurrent_group: bool = False,
) -> Dict[str, Any]:
    before = _state_snapshot(fake_tools, case.tool_name)
    context = _context_for(case)
    started_ns = time.perf_counter_ns()
    status = "success"
    error_type = None
    error_code = None
    error_message = None
    value = None
    try:
        if case.category == "cancellation":
            token = ResearchCancellationToken(
                run_id=context.run_id or "",
                session_id=context.session_id or "",
            )
            context.cancellation_token = token
            execution = asyncio.create_task(
                runtime.execute(definition, case.raw_args, context)
            )
            await _await_handler_start(
                fake_tools, case.tool_name, before["calls"]
            )
            token.cancel()
            value = _value_from_runtime(await execution)
        else:
            value = _value_from_runtime(
                await runtime.execute(definition, case.raw_args, context)
            )
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        error_code = getattr(exc, "code", None)
        error_message = str(exc)[:500]
    duration_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    after = _state_snapshot(fake_tools, case.tool_name)
    observation = {
        "variant": variant,
        "status": status,
        "error_type": error_type,
        "error_code": error_code,
        "error_message": error_message,
        "duration_ms": round(duration_ms, 6),
        "handler_reached": after["calls"] > before["calls"],
        "side_effect_delta": after["side_effects"] - before["side_effects"],
        "cancellation_delta": after["cancellations"] - before["cancellations"],
        "value": value,
    }
    if concurrent_group:
        observation["handler_reached"] = status == "success"
        observation["side_effect_delta"] = 0
        observation["cancellation_delta"] = 0
    return observation


async def _run_variant(
    variant: str,
    manifest: Sequence[AblationCase],
) -> Dict[str, Any]:
    fake_tools = HarnessFakeTools(
        timeout_seconds=0.01,
        cancellation_timeout_seconds=1.0,
    )
    definitions = fake_tools.definitions()
    capture = CaptureTerminalMiddleware()
    if variant == "H0_legacy_direct":
        runtime: Any = LegacyDirectRuntime()
    elif variant == "H1_full_harness":
        runtime = ToolRuntime(middleware=[capture])
    else:
        raise ValueError(f"unknown variant: {variant}")

    observations: Dict[str, Dict[str, Any]] = {}
    serial_cases: List[AblationCase] = []
    started = time.perf_counter()
    for case in manifest:
        if case.category == "serial_concurrency":
            serial_cases.append(case)
            continue
        observations[case.case_id] = await _observe_case(
            variant,
            runtime,
            definitions[case.tool_name],
            case,
            fake_tools,
        )

    serial_observations = await asyncio.gather(*[
        _observe_case(
            variant,
            runtime,
            definitions[case.tool_name],
            case,
            fake_tools,
            concurrent_group=True,
        )
        for case in serial_cases
    ])
    observations.update({
        case.case_id: observation
        for case, observation in zip(serial_cases, serial_observations)
    })
    elapsed = time.perf_counter() - started
    terminal_case_ids = {
        event["case_id"] for event in capture.events if event.get("case_id")
    }
    return {
        "observations": observations,
        "elapsed_seconds": round(elapsed, 6),
        "max_serial_concurrency": fake_tools.state.max_active_counts[
            "fake_serial_counter"
        ],
        "terminal_event_count": len(capture.events),
        "terminal_event_case_coverage": len(terminal_case_ids),
    }


def _policy_pass(
    case: AblationCase,
    observation: Dict[str, Any],
    max_serial_concurrency: int,
) -> bool:
    if case.category == "valid":
        return observation["status"] == "success"
    if case.category == "invalid_arguments":
        return (
            observation["status"] == "error"
            and observation["error_code"] == "invalid_arguments"
            and not observation["handler_reached"]
        )
    if case.category == "invalid_result":
        return (
            observation["status"] == "error"
            and observation["error_code"] == "invalid_result"
        )
    if case.category in {"unapproved_write", "unapproved_dangerous"}:
        return (
            observation["status"] == "error"
            and observation["error_code"] == "approval_required"
            and observation["side_effect_delta"] == 0
            and not observation["handler_reached"]
        )
    if case.category == "timeout":
        return (
            observation["status"] == "error"
            and observation["error_code"] == "timeout"
            and observation["cancellation_delta"] == 1
        )
    if case.category == "cancellation":
        return (
            observation["status"] == "error"
            and observation["error_code"] == "cancelled"
            and observation["cancellation_delta"] == 1
        )
    if case.category == "dependency_failure":
        return (
            observation["status"] == "error"
            and observation["error_type"] == "FakeDependencyError"
        )
    if case.category == "serial_concurrency":
        return observation["status"] == "success" and max_serial_concurrency == 1
    raise ValueError(f"unknown category: {case.category}")


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[lower], 6)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _mcnemar_exact(left_only: int, right_only: int) -> Dict[str, Any]:
    discordant = left_only + right_only
    if discordant == 0:
        return {"discordant": 0, "p_value": 1.0}
    tail = min(left_only, right_only)
    probability = sum(
        math.comb(discordant, index) / (2 ** discordant)
        for index in range(tail + 1)
    )
    return {
        "discordant": discordant,
        "p_value": min(1.0, 2 * probability),
    }


def summarize(
    manifest: Sequence[AblationCase],
    h0: Dict[str, Any],
    h1: Dict[str, Any],
) -> Dict[str, Any]:
    category_counts = Counter(case.category for case in manifest)
    paired_rows = []
    for case in manifest:
        h0_observation = h0["observations"][case.case_id]
        h1_observation = h1["observations"][case.case_id]
        h0_pass = _policy_pass(case, h0_observation, h0["max_serial_concurrency"])
        h1_pass = _policy_pass(case, h1_observation, h1["max_serial_concurrency"])
        paired_rows.append({
            "case": asdict(case),
            "H0_legacy_direct": h0_observation,
            "H1_full_harness": h1_observation,
            "policy_pass": {
                "H0_legacy_direct": h0_pass,
                "H1_full_harness": h1_pass,
            },
        })

    by_variant: Dict[str, Any] = {}
    for variant, run in (
        ("H0_legacy_direct", h0),
        ("H1_full_harness", h1),
    ):
        category_metrics: Dict[str, Any] = {}
        for category in category_counts:
            rows = [row for row in paired_rows if row["case"]["category"] == category]
            observations = [row[variant] for row in rows]
            passes = [row["policy_pass"][variant] for row in rows]
            category_metrics[category] = {
                "total": len(rows),
                "policy_passed": sum(passes),
                "policy_pass_rate": sum(passes) / len(rows),
                "success_count": sum(item["status"] == "success" for item in observations),
                "handler_reached_count": sum(item["handler_reached"] for item in observations),
                "side_effect_count": sum(item["side_effect_delta"] for item in observations),
                "handler_cancellation_count": sum(
                    item["cancellation_delta"] for item in observations
                ),
                "latency_p50_ms": _percentile(
                    [item["duration_ms"] for item in observations], 0.50
                ),
                "latency_p95_ms": _percentile(
                    [item["duration_ms"] for item in observations], 0.95
                ),
            }
        all_passes = [row["policy_pass"][variant] for row in paired_rows]
        by_variant[variant] = {
            "overall": {
                "total": len(paired_rows),
                "policy_passed": sum(all_passes),
                "policy_pass_rate": sum(all_passes) / len(all_passes),
                "elapsed_seconds": run["elapsed_seconds"],
                "max_serial_concurrency": run["max_serial_concurrency"],
                "terminal_event_count": run["terminal_event_count"],
                "terminal_event_case_coverage": run[
                    "terminal_event_case_coverage"
                ],
            },
            "by_category": category_metrics,
        }

    h0_only = sum(
        row["policy_pass"]["H0_legacy_direct"]
        and not row["policy_pass"]["H1_full_harness"]
        for row in paired_rows
    )
    h1_only = sum(
        row["policy_pass"]["H1_full_harness"]
        and not row["policy_pass"]["H0_legacy_direct"]
        for row in paired_rows
    )
    both_pass = sum(
        row["policy_pass"]["H0_legacy_direct"]
        and row["policy_pass"]["H1_full_harness"]
        for row in paired_rows
    )
    both_fail = len(paired_rows) - h0_only - h1_only - both_pass

    valid_rows = [row for row in paired_rows if row["case"]["category"] == "valid"]
    valid_output_parity = sum(
        row["H0_legacy_direct"]["value"] == row["H1_full_harness"]["value"]
        for row in valid_rows
    )
    critical_categories = {
        "invalid_arguments",
        "invalid_result",
        "unapproved_write",
        "unapproved_dangerous",
        "timeout",
        "cancellation",
    }
    critical_rows = [
        row for row in paired_rows
        if row["case"]["category"] in critical_categories
    ]
    h1_critical_failures = sum(
        not row["policy_pass"]["H1_full_harness"] for row in critical_rows
    )

    metrics = {
        "evaluation_date": "2026-08-30",
        "evaluation_object": "500-case paired H0 legacy direct vs H1 full Agent Harness ablation",
        "baseline": {
            "commit": BASELINE_COMMIT,
            "file": BASELINE_FILE,
            "semantics": BASELINE_SEMANTICS,
        },
        "manifest": {
            "total": len(manifest),
            "seed": FIXED_SEED,
            "category_counts": dict(category_counts),
        },
        "variants": by_variant,
        "paired_comparison": {
            "both_pass": both_pass,
            "H0_only_pass": h0_only,
            "H1_only_pass": h1_only,
            "both_fail": both_fail,
            "mcnemar_exact": _mcnemar_exact(h0_only, h1_only),
        },
        "headline_metrics": {
            "valid_output_parity": f"{valid_output_parity}/{len(valid_rows)}",
            "H0_policy_pass_rate": by_variant["H0_legacy_direct"]["overall"][
                "policy_pass_rate"
            ],
            "H1_policy_pass_rate": by_variant["H1_full_harness"]["overall"][
                "policy_pass_rate"
            ],
            "H0_invalid_argument_escape_rate": (
                by_variant["H0_legacy_direct"]["by_category"]["invalid_arguments"][
                    "handler_reached_count"
                ] / category_counts["invalid_arguments"]
            ),
            "H1_invalid_argument_escape_rate": (
                by_variant["H1_full_harness"]["by_category"]["invalid_arguments"][
                    "handler_reached_count"
                ] / category_counts["invalid_arguments"]
            ),
            "H0_invalid_result_escape_rate": (
                by_variant["H0_legacy_direct"]["by_category"]["invalid_result"][
                    "success_count"
                ] / category_counts["invalid_result"]
            ),
            "H1_invalid_result_escape_rate": (
                by_variant["H1_full_harness"]["by_category"]["invalid_result"][
                    "success_count"
                ] / category_counts["invalid_result"]
            ),
            "H0_unsafe_action_allowance_rate": (
                sum(
                    by_variant["H0_legacy_direct"]["by_category"][category][
                        "side_effect_count"
                    ]
                    for category in ("unapproved_write", "unapproved_dangerous")
                ) / (
                    category_counts["unapproved_write"]
                    + category_counts["unapproved_dangerous"]
                )
            ),
            "H1_unsafe_action_allowance_rate": (
                sum(
                    by_variant["H1_full_harness"]["by_category"][category][
                        "side_effect_count"
                    ]
                    for category in ("unapproved_write", "unapproved_dangerous")
                ) / (
                    category_counts["unapproved_write"]
                    + category_counts["unapproved_dangerous"]
                )
            ),
            "H0_timeout_containment_rate": by_variant["H0_legacy_direct"][
                "by_category"
            ]["timeout"]["policy_pass_rate"],
            "H1_timeout_containment_rate": by_variant["H1_full_harness"][
                "by_category"
            ]["timeout"]["policy_pass_rate"],
            "H0_cancellation_containment_rate": by_variant["H0_legacy_direct"][
                "by_category"
            ]["cancellation"]["policy_pass_rate"],
            "H1_cancellation_containment_rate": by_variant["H1_full_harness"][
                "by_category"
            ]["cancellation"]["policy_pass_rate"],
            "H0_max_serial_concurrency": h0["max_serial_concurrency"],
            "H1_max_serial_concurrency": h1["max_serial_concurrency"],
            "H0_terminal_event_coverage": (
                h0["terminal_event_case_coverage"] / len(manifest)
            ),
            "H1_terminal_event_coverage": (
                h1["terminal_event_case_coverage"] / len(manifest)
            ),
            "H1_critical_guard_failures": h1_critical_failures,
            "H1_critical_guard_cases": len(critical_rows),
        },
        "measurement_boundary": {
            "measured": [
                "500 deterministic paired synthetic cases",
                "input/output validation",
                "unapproved WRITE/DANGEROUS side effects",
                "timeout and in-flight cancellation",
                "non-concurrency-safe serialization",
                "uniform terminal middleware event coverage",
            ],
            "not_evaluated": [
                "real network/database/tool providers",
                "cross-process or multi-replica cancellation",
                "call_id-level structured trace completeness",
                "production traffic failure rate",
                "context compaction quality",
                "true no-op runtime overhead without fixture-interface differences",
            ],
        },
    }
    return {"metrics": metrics, "paired_rows": paired_rows}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


async def run_ablation(output_dir: Path) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refuse to overwrite non-empty run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_manifest()
    h0 = await _run_variant("H0_legacy_direct", manifest)
    h1 = await _run_variant("H1_full_harness", manifest)
    summary = summarize(manifest, h0, h1)

    backend_dir = Path(__file__).resolve().parents[2]
    configuration = {
        "evaluation_date": "2026-08-30",
        "python": sys.version,
        "platform": platform.platform(),
        "seed": FIXED_SEED,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_file": BASELINE_FILE,
        "baseline_semantics": BASELINE_SEMANTICS,
        "network": False,
        "database": False,
        "filesystem_side_effects": False,
        "fake_tool_timeout_seconds": 0.01,
        "fake_tool_cancellation_timeout_seconds": 1.0,
        "source_sha256": {
            "fake_tools.py": _sha256(backend_dir / "eval/harness/fake_tools.py"),
            "ablation.py": _sha256(backend_dir / "eval/harness/ablation.py"),
            "tool_runtime.py": _sha256(backend_dir / "app/service/tool_runtime.py"),
        },
    }
    _write_json_atomic(
        output_dir / "500条配对用例清单.json",
        [asdict(case) for case in manifest],
    )
    _write_jsonl_atomic(
        output_dir / "500条逐题配对结果.jsonl",
        summary["paired_rows"],
    )
    _write_json_atomic(output_dir / "汇总指标.json", summary["metrics"])
    _write_json_atomic(output_dir / "运行配置.json", configuration)
    return summary["metrics"]
