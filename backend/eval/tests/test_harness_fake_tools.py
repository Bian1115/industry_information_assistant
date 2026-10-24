"""Offline behavior tests for deterministic Agent Harness fake tools."""

import asyncio
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(APP_DIR))

from eval.harness.fake_tools import (  # noqa: E402
    FakeDependencyError,
    HarnessFakeTools,
)
from service.research_control import ResearchCancellationToken  # noqa: E402
from service.tool_runtime import (  # noqa: E402
    ToolCallContext,
    ToolErrorCode,
    ToolRisk,
    ToolRuntime,
    ToolRuntimeError,
)


class HarnessFakeToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_tools = HarnessFakeTools(
            timeout_seconds=0.02,
            cancellation_timeout_seconds=2.0,
        )
        self.definitions = self.fake_tools.definitions()
        self.runtime = ToolRuntime()

    def test_registry_exposes_eleven_scenarios(self):
        self.assertEqual(len(self.definitions), 11)
        self.assertEqual(
            self.definitions["fake_write_side_effect"].risk,
            ToolRisk.WRITE,
        )
        self.assertEqual(
            self.definitions["fake_dangerous_side_effect"].risk,
            ToolRisk.DANGEROUS,
        )
        self.assertFalse(self.definitions["fake_serial_counter"].concurrency_safe)

    async def test_success_tools_are_deterministic(self):
        first = await self.runtime.execute(
            self.definitions["fake_search_success"],
            {"query": "新能源", "count": 2},
        )
        second = await self.runtime.execute(
            self.definitions["fake_search_success"],
            {"query": "新能源", "count": 2},
        )
        self.assertEqual(first.value, second.value)
        self.assertEqual(first.value["results"][0]["rank"], 1)
        self.assertEqual(
            self.fake_tools.state.call_counts["fake_search_success"], 2
        )

    async def test_invalid_arguments_never_reach_handler(self):
        with self.assertRaises(ToolRuntimeError) as caught:
            await self.runtime.execute(
                self.definitions["fake_echo_success"],
                {"repeat": 2},
            )
        self.assertEqual(caught.exception.code, ToolErrorCode.INVALID_ARGUMENTS.value)
        self.assertEqual(
            self.fake_tools.state.call_counts["fake_echo_success"], 0
        )

    async def test_context_probe_echoes_runtime_identifiers(self):
        result = await self.runtime.execute(
            self.definitions["fake_context_probe"],
            {"label": "trace"},
            ToolCallContext(
                session_id="session-1",
                run_id="run-1",
                user_id="user-1",
            ),
        )
        self.assertEqual(result.value["session_id"], "session-1")
        self.assertEqual(result.value["run_id"], "run-1")
        self.assertEqual(result.value["user_id"], "user-1")

    async def test_invalid_result_is_rejected(self):
        with self.assertRaises(ToolRuntimeError) as caught:
            await self.runtime.execute(
                self.definitions["fake_invalid_result"],
                {"text": "bad"},
            )
        self.assertEqual(caught.exception.code, ToolErrorCode.INVALID_RESULT.value)
        self.assertEqual(
            self.fake_tools.state.call_counts["fake_invalid_result"], 1
        )

    async def test_unapproved_write_and_dangerous_tools_have_no_side_effect(self):
        for name in ("fake_write_side_effect", "fake_dangerous_side_effect"):
            with self.subTest(name=name):
                with self.assertRaises(ToolRuntimeError) as caught:
                    await self.runtime.execute(
                        self.definitions[name],
                        {"record_id": 1, "value": "blocked"},
                    )
                self.assertEqual(
                    caught.exception.code,
                    ToolErrorCode.APPROVAL_REQUIRED.value,
                )
                self.assertEqual(self.fake_tools.state.call_counts[name], 0)
                self.assertEqual(self.fake_tools.state.side_effect_counts[name], 0)

    async def test_approved_write_has_one_observable_side_effect(self):
        result = await self.runtime.execute(
            self.definitions["fake_write_side_effect"],
            {"record_id": 7, "value": "approved"},
            ToolCallContext(approved=True),
        )
        self.assertEqual(result.value, {"success": True, "record_id": 7})
        self.assertEqual(
            self.fake_tools.state.side_effect_counts["fake_write_side_effect"], 1
        )

    async def test_timeout_cancels_slow_handler(self):
        with self.assertRaises(ToolRuntimeError) as caught:
            await self.runtime.execute(
                self.definitions["fake_slow_timeout"],
                {"delay_ms": 200},
            )
        self.assertEqual(caught.exception.code, ToolErrorCode.TIMEOUT.value)
        self.assertEqual(
            self.fake_tools.state.cancellation_counts["fake_slow_timeout"], 1
        )

    async def test_in_flight_cancellation_reaps_handler(self):
        token = ResearchCancellationToken("run-cancel", "session-cancel")
        execution = asyncio.create_task(
            self.runtime.execute(
                self.definitions["fake_cancellable"],
                {"delay_ms": 1_000},
                ToolCallContext(cancellation_token=token),
            )
        )
        for _ in range(100):
            if self.fake_tools.state.call_counts["fake_cancellable"]:
                break
            await asyncio.sleep(0)
        token.cancel()
        with self.assertRaises(ToolRuntimeError) as caught:
            await execution
        self.assertEqual(caught.exception.code, ToolErrorCode.CANCELLED.value)
        self.assertEqual(
            self.fake_tools.state.cancellation_counts["fake_cancellable"], 1
        )

    async def test_dependency_failure_is_stable(self):
        with self.assertRaisesRegex(FakeDependencyError, "provider-503"):
            await self.runtime.execute(
                self.definitions["fake_dependency_failure"],
                {"message": "provider-503"},
            )
        self.assertEqual(
            self.fake_tools.state.call_counts["fake_dependency_failure"], 1
        )

    async def test_non_concurrency_safe_tool_is_serialized(self):
        results = await asyncio.gather(*[
            self.runtime.execute(
                self.definitions["fake_serial_counter"],
                {"sequence": index, "delay_ms": 5},
            )
            for index in range(10)
        ])
        self.assertEqual(
            self.fake_tools.state.max_active_counts["fake_serial_counter"], 1
        )
        self.assertEqual(
            sorted(item.value["sequence"] for item in results),
            list(range(10)),
        )

    async def test_large_result_is_deterministic(self):
        result = await self.runtime.execute(
            self.definitions["fake_large_result"],
            {"item_count": 500},
        )
        self.assertEqual(len(result.value["items"]), 500)
        self.assertEqual(result.value["items"][:3], [0, 1, 2])
        self.assertEqual(result.value["items"][-1], 499)


if __name__ == "__main__":
    unittest.main()
