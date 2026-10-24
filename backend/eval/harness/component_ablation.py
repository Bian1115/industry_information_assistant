"""Single-component ablations for the Agent Harness.

The production ToolRuntime is not modified. Evaluation-only wrappers transform
ToolDefinition or ToolCallContext immediately before delegating to the real
runtime, which keeps all non-ablated policies identical to H1.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from pydantic import BaseModel, ConfigDict

from service.tool_runtime import (
    ToolCallContext,
    ToolDefinition,
    ToolRisk,
    ToolRuntime,
)

from .ablation import (
    AblationCase,
    CaptureTerminalMiddleware,
    FIXED_SEED,
    _mcnemar_exact,
    _observe_case,
    _policy_pass,
    _sha256,
    _write_json_atomic,
    _write_jsonl_atomic,
    generate_manifest,
)
from .fake_tools import HarnessFakeTools


class PermissiveArgs(BaseModel):
    """Evaluation-only model that accepts every raw input without constraints."""

    model_config = ConfigDict(extra="allow")


class NoTraceMiddleware:
    """Keep middleware invocation valid while emitting no trace events."""

    async def before(
        self, definition: ToolDefinition, args: Any, context: ToolCallContext
    ) -> None:
        return None

    async def after(
        self, definition: ToolDefinition, result: Any, context: ToolCallContext
    ) -> None:
        return None

    async def on_error(
        self, definition: ToolDefinition, error: BaseException, context: ToolCallContext
    ) -> None:
        return None


class ComponentAblationRuntime:
    """Delegate to ToolRuntime after removing exactly one evaluated policy."""

    def __init__(self, variant: str, capture: CaptureTerminalMiddleware) -> None:
        self.variant = variant
        middleware: Any
        if variant == "H5_without_trace":
            middleware = NoTraceMiddleware()
        else:
            middleware = capture
        self.runtime = ToolRuntime(middleware=[middleware])

    async def execute(
        self,
        definition: ToolDefinition,
        raw_args: Dict[str, Any],
        context: ToolCallContext,
    ) -> Any:
        evaluated_definition = definition
        evaluated_context = context
        if self.variant == "H2_without_schema":
            evaluated_definition = replace(
                definition,
                args_model=PermissiveArgs,
                result_model=None,
            )
        elif self.variant == "H3_without_risk_check":
            evaluated_definition = replace(definition, risk=ToolRisk.READ_ONLY)
        elif self.variant == "H4_without_timeout_cancel":
            evaluated_definition = replace(definition, timeout_seconds=1.0)
            evaluated_context = replace(context, cancellation_token=None)
        elif self.variant in {"H1_full_harness", "H5_without_trace"}:
            pass
        else:
            raise ValueError(f"unknown component ablation variant: {self.variant}")
        return await self.runtime.execute(
            evaluated_definition,
            raw_args,
            evaluated_context,
        )


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "关闭Schema校验": {
        "variant": "H2_without_schema",
        "categories": {"valid", "invalid_arguments", "invalid_result"},
        "expected_cases": 240,
        "removed_policy": "input and output schema validation",
    },
    "关闭风险检查": {
        "variant": "H3_without_risk_check",
        "categories": {"valid", "unapproved_write", "unapproved_dangerous"},
        "expected_cases": 200,
        "removed_policy": "WRITE and DANGEROUS approval check",
    },
    "关闭超时取消": {
        "variant": "H4_without_timeout_cancel",
        "categories": {"valid", "timeout", "cancellation"},
        "expected_cases": 220,
        "removed_policy": "runtime timeout and cancellation token propagation",
    },
    "关闭Trace": {
        "variant": "H5_without_trace",
        "categories": None,
        "expected_cases": 500,
        "removed_policy": "uniform terminal middleware events",
    },
}


def experiment_slice(
    manifest: Sequence[AblationCase], experiment_name: str
) -> List[AblationCase]:
    configuration = EXPERIMENTS[experiment_name]
    categories = configuration["categories"]
    if categories is None:
        selected = list(manifest)
    else:
        selected = [case for case in manifest if case.category in categories]
    if len(selected) != configuration["expected_cases"]:
        raise AssertionError(
            f"{experiment_name}: expected {configuration['expected_cases']}, "
            f"selected {len(selected)}"
        )
    return selected


async def run_component_variant(
    variant: str,
    manifest: Sequence[AblationCase],
) -> Dict[str, Any]:
    fake_tools = HarnessFakeTools(
        timeout_seconds=0.01,
        cancellation_timeout_seconds=1.0,
    )
    definitions = fake_tools.definitions()
    capture = CaptureTerminalMiddleware()
    runtime = ComponentAblationRuntime(variant, capture)
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
        "terminal_events": capture.events,
        "terminal_event_count": len(capture.events),
        "terminal_event_case_coverage": len(terminal_case_ids),
    }


def _behavior_signature(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": observation["status"],
        "error_type": observation["error_type"],
        "error_code": observation["error_code"],
        "handler_reached": observation["handler_reached"],
        "side_effect_delta": observation["side_effect_delta"],
        "cancellation_delta": observation["cancellation_delta"],
        "value": observation["value"],
    }


def summarize_component_experiment(
    experiment_name: str,
    cases: Sequence[AblationCase],
    h1: Dict[str, Any],
    ablated: Dict[str, Any],
) -> Dict[str, Any]:
    configuration = EXPERIMENTS[experiment_name]
    variant = configuration["variant"]
    rows = []
    for case in cases:
        h1_observation = h1["observations"][case.case_id]
        ablated_observation = ablated["observations"][case.case_id]
        h1_pass = _policy_pass(
            case, h1_observation, h1["max_serial_concurrency"]
        )
        ablated_pass = _policy_pass(
            case,
            ablated_observation,
            ablated["max_serial_concurrency"],
        )
        rows.append({
            "experiment": experiment_name,
            "removed_policy": configuration["removed_policy"],
            "case": asdict(case),
            "H1_full_harness": h1_observation,
            variant: ablated_observation,
            "policy_pass": {
                "H1_full_harness": h1_pass,
                variant: ablated_pass,
            },
            "behavior_equal": (
                _behavior_signature(h1_observation)
                == _behavior_signature(ablated_observation)
            ),
        })

    h1_passed = sum(row["policy_pass"]["H1_full_harness"] for row in rows)
    ablated_passed = sum(row["policy_pass"][variant] for row in rows)
    h1_only = sum(
        row["policy_pass"]["H1_full_harness"]
        and not row["policy_pass"][variant]
        for row in rows
    )
    ablated_only = sum(
        not row["policy_pass"]["H1_full_harness"]
        and row["policy_pass"][variant]
        for row in rows
    )
    both_pass = sum(
        row["policy_pass"]["H1_full_harness"]
        and row["policy_pass"][variant]
        for row in rows
    )
    both_fail = len(rows) - h1_only - ablated_only - both_pass
    valid_rows = [row for row in rows if row["case"]["category"] == "valid"]

    metrics: Dict[str, Any] = {
        "experiment": experiment_name,
        "variant": variant,
        "removed_policy": configuration["removed_policy"],
        "total_cases": len(rows),
        "H1_policy_passed": h1_passed,
        "H1_policy_pass_rate": h1_passed / len(rows),
        "ablated_policy_passed": ablated_passed,
        "ablated_policy_pass_rate": ablated_passed / len(rows),
        "normal_behavior_parity": (
            sum(row["behavior_equal"] for row in valid_rows)
            / len(valid_rows)
        ),
        "all_behavior_parity": sum(row["behavior_equal"] for row in rows) / len(rows),
        "H1_terminal_event_coverage": (
            len({
                event["case_id"] for event in h1["terminal_events"]
                if event.get("case_id") in {case.case_id for case in cases}
            }) / len(rows)
        ),
        "ablated_terminal_event_coverage": (
            ablated["terminal_event_case_coverage"] / len(rows)
        ),
        "paired_comparison": {
            "both_pass": both_pass,
            "H1_only_pass": h1_only,
            "ablated_only_pass": ablated_only,
            "both_fail": both_fail,
            "mcnemar_exact": _mcnemar_exact(ablated_only, h1_only),
        },
    }

    if experiment_name == "关闭Schema校验":
        invalid_arguments = [
            row for row in rows if row["case"]["category"] == "invalid_arguments"
        ]
        invalid_results = [
            row for row in rows if row["case"]["category"] == "invalid_result"
        ]
        metrics.update({
            "H1_invalid_argument_escape_rate": sum(
                row["H1_full_harness"]["handler_reached"]
                for row in invalid_arguments
            ) / len(invalid_arguments),
            "ablated_invalid_argument_escape_rate": sum(
                row[variant]["handler_reached"] for row in invalid_arguments
            ) / len(invalid_arguments),
            "H1_invalid_result_escape_rate": sum(
                row["H1_full_harness"]["status"] == "success"
                for row in invalid_results
            ) / len(invalid_results),
            "ablated_invalid_result_escape_rate": sum(
                row[variant]["status"] == "success" for row in invalid_results
            ) / len(invalid_results),
        })
    elif experiment_name == "关闭风险检查":
        unsafe = [
            row for row in rows
            if row["case"]["category"] in {
                "unapproved_write", "unapproved_dangerous"
            }
        ]
        metrics.update({
            "H1_unsafe_action_allowance_rate": sum(
                row["H1_full_harness"]["side_effect_delta"] > 0
                for row in unsafe
            ) / len(unsafe),
            "ablated_unsafe_action_allowance_rate": sum(
                row[variant]["side_effect_delta"] > 0 for row in unsafe
            ) / len(unsafe),
        })
    elif experiment_name == "关闭超时取消":
        timeout_rows = [
            row for row in rows if row["case"]["category"] == "timeout"
        ]
        cancellation_rows = [
            row for row in rows if row["case"]["category"] == "cancellation"
        ]
        metrics.update({
            "H1_timeout_containment_rate": sum(
                row["policy_pass"]["H1_full_harness"] for row in timeout_rows
            ) / len(timeout_rows),
            "ablated_timeout_containment_rate": sum(
                row["policy_pass"][variant] for row in timeout_rows
            ) / len(timeout_rows),
            "H1_cancellation_containment_rate": sum(
                row["policy_pass"]["H1_full_harness"]
                for row in cancellation_rows
            ) / len(cancellation_rows),
            "ablated_cancellation_containment_rate": sum(
                row["policy_pass"][variant] for row in cancellation_rows
            ) / len(cancellation_rows),
        })
    elif experiment_name == "关闭Trace":
        terminal_accuracy = sum(
            event["outcome"]
            == h1["observations"][event["case_id"]]["status"]
            for event in h1["terminal_events"]
            if event.get("case_id") in {case.case_id for case in cases}
        )
        metrics.update({
            "H1_terminal_outcome_accuracy": terminal_accuracy / len(rows),
            "ablated_terminal_outcome_accuracy": None,
        })
    return {"metrics": metrics, "rows": rows}


async def run_component_ablations(output_dir: Path) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refuse to overwrite non-empty run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_manifest()
    h1 = await run_component_variant("H1_full_harness", manifest)
    summaries = []
    all_rows: List[Dict[str, Any]] = []
    experiment_manifests: Dict[str, Any] = {}
    for experiment_name, configuration in EXPERIMENTS.items():
        cases = experiment_slice(manifest, experiment_name)
        ablated = await run_component_variant(configuration["variant"], cases)
        summary = summarize_component_experiment(
            experiment_name, cases, h1, ablated
        )
        summaries.append(summary["metrics"])
        all_rows.extend(summary["rows"])
        experiment_manifests[experiment_name] = {
            "variant": configuration["variant"],
            "removed_policy": configuration["removed_policy"],
            "case_ids": [case.case_id for case in cases],
            "case_count": len(cases),
        }

    aggregate = {
        "evaluation_date": "2026-08-30",
        "evaluation_object": "Agent Harness single-component ablations",
        "shared_H1_control_cases": len(manifest),
        "experiment_count": len(summaries),
        "experiment_case_executions": sum(
            item["total_cases"] for item in summaries
        ),
        "results": summaries,
        "measurement_boundary": {
            "measured": [
                "schema contribution on 240 paired cases",
                "risk-check contribution on 200 paired cases",
                "timeout/cancellation contribution on 220 paired cases",
                "terminal middleware trace contribution on 500 paired cases",
            ],
            "not_evaluated": [
                "real provider and production traffic distributions",
                "call_id-level structured trace completeness",
                "persistent PostgreSQL trace durability",
                "context compaction",
                "multi-process cancellation",
            ],
        },
    }
    backend_dir = Path(__file__).resolve().parents[2]
    configuration = {
        "evaluation_date": "2026-08-30",
        "python": sys.version,
        "platform": platform.platform(),
        "seed": FIXED_SEED,
        "network": False,
        "database": False,
        "filesystem_side_effects": False,
        "shared_H1_control": True,
        "experiments": {
            name: {
                **item,
                "categories": (
                    sorted(item["categories"])
                    if item["categories"] is not None
                    else None
                ),
            }
            for name, item in EXPERIMENTS.items()
        },
        "source_sha256": {
            "fake_tools.py": _sha256(backend_dir / "eval/harness/fake_tools.py"),
            "ablation.py": _sha256(backend_dir / "eval/harness/ablation.py"),
            "component_ablation.py": _sha256(
                backend_dir / "eval/harness/component_ablation.py"
            ),
            "tool_runtime.py": _sha256(backend_dir / "app/service/tool_runtime.py"),
        },
    }
    _write_json_atomic(output_dir / "四组消融用例清单.json", experiment_manifests)
    _write_jsonl_atomic(output_dir / "四组消融逐条配对结果.jsonl", all_rows)
    _write_json_atomic(output_dir / "四组消融汇总指标.json", aggregate)
    _write_json_atomic(output_dir / "运行配置.json", configuration)
    return aggregate
