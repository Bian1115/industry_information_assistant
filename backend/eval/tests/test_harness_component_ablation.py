"""Offline tests for single-component Agent Harness ablations."""

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(APP_DIR))

from eval.harness.ablation import generate_manifest  # noqa: E402
from eval.harness.component_ablation import (  # noqa: E402
    EXPERIMENTS,
    experiment_slice,
    run_component_variant,
)


class ComponentAblationManifestTests(unittest.TestCase):
    def test_four_slices_have_expected_sizes(self):
        manifest = generate_manifest()
        self.assertEqual(
            {
                name: len(experiment_slice(manifest, name))
                for name in EXPERIMENTS
            },
            {
                "关闭Schema校验": 240,
                "关闭风险检查": 200,
                "关闭超时取消": 220,
                "关闭Trace": 500,
            },
        )


class ComponentAblationBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_off_reaches_handler_for_invalid_arguments(self):
        case = next(
            item for item in generate_manifest()
            if item.category == "invalid_arguments"
        )
        result = await run_component_variant("H2_without_schema", [case])
        observation = result["observations"][case.case_id]
        self.assertTrue(observation["handler_reached"])
        self.assertNotEqual(observation["error_code"], "invalid_arguments")

    async def test_risk_off_allows_unapproved_side_effect(self):
        case = next(
            item for item in generate_manifest()
            if item.category == "unapproved_write"
        )
        result = await run_component_variant("H3_without_risk_check", [case])
        observation = result["observations"][case.case_id]
        self.assertEqual(observation["status"], "success")
        self.assertEqual(observation["side_effect_delta"], 1)

    async def test_timeout_cancel_off_allows_slow_tool_to_complete(self):
        case = next(
            item for item in generate_manifest()
            if item.category == "timeout"
        )
        result = await run_component_variant(
            "H4_without_timeout_cancel", [case]
        )
        observation = result["observations"][case.case_id]
        self.assertEqual(observation["status"], "success")
        self.assertEqual(observation["cancellation_delta"], 0)

    async def test_trace_off_keeps_behavior_but_emits_no_terminal_event(self):
        case = next(item for item in generate_manifest() if item.category == "valid")
        h1 = await run_component_variant("H1_full_harness", [case])
        h5 = await run_component_variant("H5_without_trace", [case])
        h1_observation = h1["observations"][case.case_id]
        h5_observation = h5["observations"][case.case_id]
        for key in (
            "status",
            "error_type",
            "error_code",
            "handler_reached",
            "side_effect_delta",
            "cancellation_delta",
            "value",
        ):
            self.assertEqual(h1_observation[key], h5_observation[key])
        self.assertEqual(h1["terminal_event_case_coverage"], 1)
        self.assertEqual(h5["terminal_event_case_coverage"], 0)


if __name__ == "__main__":
    unittest.main()
