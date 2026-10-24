"""Offline tests for the Agent Harness H0/H1 ablation runner."""

import sys
import unittest
from collections import Counter
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(APP_DIR))

from eval.harness.ablation import (  # noqa: E402
    LegacyDirectRuntime,
    _policy_pass,
    generate_manifest,
)
from eval.harness.fake_tools import HarnessFakeTools  # noqa: E402
from service.tool_runtime import ToolCallContext, ToolRuntime, ToolRuntimeError  # noqa: E402


class HarnessAblationManifestTests(unittest.TestCase):
    def test_manifest_has_fixed_500_case_distribution(self):
        manifest = generate_manifest()
        self.assertEqual(len(manifest), 500)
        self.assertEqual(len({case.case_id for case in manifest}), 500)
        self.assertEqual(
            Counter(case.category for case in manifest),
            {
                "valid": 100,
                "invalid_arguments": 80,
                "invalid_result": 60,
                "unapproved_write": 60,
                "unapproved_dangerous": 40,
                "timeout": 60,
                "cancellation": 60,
                "dependency_failure": 20,
                "serial_concurrency": 20,
            },
        )


class LegacyDirectBaselineTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_baseline_passes_raw_args_without_validation(self):
        fake_tools = HarnessFakeTools()
        definition = fake_tools.definitions()["fake_search_success"]
        value = await LegacyDirectRuntime().execute(
            definition,
            {"query": "raw", "count": 0},
            ToolCallContext(),
        )
        self.assertEqual(value, {"query": "raw", "results": []})
        self.assertEqual(fake_tools.state.call_counts[definition.name], 1)

    async def test_full_harness_rejects_same_invalid_args(self):
        fake_tools = HarnessFakeTools()
        definition = fake_tools.definitions()["fake_search_success"]
        with self.assertRaises(ToolRuntimeError) as caught:
            await ToolRuntime().execute(
                definition,
                {"query": "raw", "count": 0},
            )
        self.assertEqual(caught.exception.code, "invalid_arguments")
        self.assertEqual(fake_tools.state.call_counts[definition.name], 0)

    def test_policy_requires_serialization_for_serial_cases(self):
        case = next(
            item for item in generate_manifest()
            if item.category == "serial_concurrency"
        )
        observation = {"status": "success"}
        self.assertFalse(_policy_pass(case, observation, 20))
        self.assertTrue(_policy_pass(case, observation, 1))


if __name__ == "__main__":
    unittest.main()
