"""唯一 LangGraph 主链路的拓扑、路由与运行保护测试。

不调用真实模型、搜索或数据库：Agent 用可编程的假实现，检查点用内存 saver。
覆盖计划中列出的必须验证情况：首次通过、只修订、补搜后再分析写作、
两轮返工后终止但不改质量结论、空报告不误报完成、步数兜底、取消与预算。
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parents[2] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

os.environ.setdefault("RESEARCH_ALLOWED_DOMAINS", "who.int,nih.gov")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from service.deep_research_v2.graph import (  # noqa: E402
    RECURSION_LIMIT,
    build_graph,
    graph_config,
    pack_state,
)
from service.deep_research_v2.state import (  # noqa: E402
    ResearchPhase,
    create_initial_state,
)
from service.research_control import (  # noqa: E402
    ResearchCancelledError,
    clear_cancel_flag,
    request_cancel,
)
from service.deep_research_v2.stage_runtime import (  # noqa: E402
    StageBudgetExceeded,
    make_execute_stage,
)


class FakeAgent:
    """记录调用顺序的假 Agent；行为由注入的函数决定。"""

    def __init__(self, name, behaviour=None):
        self.name = name
        self.calls = []
        self.llm_call_count = 0
        self.llm_total_tokens = 0
        self._behaviour = behaviour

    async def process(self, state):
        self.calls.append(state["phase"])
        if self._behaviour is not None:
            self._behaviour(state)
        return state


def critic_script(*decisions):
    """按调用次序返回审核结论；Critic 自己负责 iteration 计数。"""
    calls = {"n": 0}

    def behaviour(state):
        index = min(calls["n"], len(decisions) - 1)
        verdict, phase = decisions[index]
        calls["n"] += 1
        state["quality_verdict"] = verdict
        state["quality_score"] = 8 if verdict == "pass" else 5
        if verdict == "pass":
            state["phase"] = ResearchPhase.COMPLETED.value
        elif state["iteration"] >= state["max_iterations"]:
            # 额度用尽只结束执行，不把质量结论改成通过。
            state["phase"] = ResearchPhase.COMPLETED.value
        else:
            state["phase"] = phase
            state["iteration"] += 1

    return behaviour


def writes_report(state):
    state["final_report"] = "# 报告\n见 [来源](https://who.int/a)"


def make_agents(**overrides):
    agents = {
        "architect": FakeAgent("ChiefArchitect"),
        "scout": FakeAgent("DeepScout"),
        "data_analyst": FakeAgent("DataAnalyst"),
        "writer": FakeAgent("LeadWriter", writes_report),
        "critic": FakeAgent("CriticMaster", critic_script(("pass", ""))),
    }
    agents.update(overrides)
    return agents


async def passthrough_stage(action, work, phase):
    return await action(work)


def run_graph(agents, *, execute_stage=passthrough_stage, run_id="run-1", state=None):
    graph = build_graph(
        agents, InMemorySaver(),
        run_id=run_id, user_id="user-1", session_id="sess-1",
        execute_stage=execute_stage,
    )
    initial = state or create_initial_state("研究问题", "sess-1", ["who.int"])

    async def main():
        events = []
        async for event in graph.astream(
            pack_state(initial), config=graph_config(run_id), stream_mode="custom"
        ):
            events.append(event)
        snapshot = await graph.aget_state(graph_config(run_id))
        return events, snapshot

    return asyncio.run(main())


class MainGraphTopology(unittest.TestCase):
    def test_first_pass_runs_plan_research_analyze_write_review_finish(self):
        agents = make_agents()
        _, snapshot = run_graph(agents)
        research = snapshot.values["research"]
        self.assertEqual(agents["architect"].calls, ["init"])
        self.assertEqual(agents["scout"].calls, ["researching"])
        self.assertEqual(agents["data_analyst"].calls, ["analyzing"])
        self.assertEqual(agents["writer"].calls, ["writing"])
        self.assertEqual(agents["critic"].calls, ["reviewing"])
        self.assertEqual(snapshot.next, ())
        self.assertEqual(research["status"], "completed")
        self.assertEqual(research["completion_reason"], "passed")
        self.assertEqual(research["quality_verdict"], "pass")

    def test_revision_only_goes_back_to_review_without_new_search(self):
        agents = make_agents(critic=FakeAgent("CriticMaster", critic_script(
            ("needs_revision", ResearchPhase.REVISING.value), ("pass", ""),
        )))
        _, snapshot = run_graph(agents)
        self.assertEqual(agents["writer"].calls, ["writing", "revising"])
        self.assertEqual(agents["scout"].calls, ["researching"])
        self.assertEqual(agents["data_analyst"].calls, ["analyzing"])
        self.assertEqual(snapshot.values["research"]["completion_reason"], "passed")

    def test_re_research_reanalyzes_and_rewrites_before_review(self):
        agents = make_agents(critic=FakeAgent("CriticMaster", critic_script(
            ("needs_revision", ResearchPhase.RE_RESEARCHING.value), ("pass", ""),
        )))
        _, snapshot = run_graph(agents)
        # 补搜后必须重新分析并重新写作，而不是直接回审核。
        self.assertEqual(agents["scout"].calls, ["researching", "re_researching"])
        self.assertEqual(agents["data_analyst"].calls, ["analyzing", "analyzing"])
        self.assertEqual(agents["writer"].calls, ["writing", "writing"])
        self.assertEqual(agents["critic"].calls, ["reviewing", "reviewing"])
        self.assertEqual(snapshot.values["research"]["iteration"], 1)

    def test_revision_limit_stops_execution_without_changing_verdict(self):
        agents = make_agents(critic=FakeAgent("CriticMaster", critic_script(
            ("needs_revision", ResearchPhase.REVISING.value),
            ("needs_revision", ResearchPhase.REVISING.value),
            ("major_issues", ResearchPhase.REVISING.value),
        )))
        _, snapshot = run_graph(agents)
        research = snapshot.values["research"]
        self.assertEqual(research["iteration"], 2)
        self.assertEqual(research["max_iterations"], 2)
        self.assertEqual(research["status"], "completed")
        # 执行结束不等于质量通过。
        self.assertEqual(research["completion_reason"], "revision_limit")
        self.assertEqual(research["quality_verdict"], "major_issues")

    def test_unknown_review_phase_aborts_instead_of_passing_silently(self):
        def bad_review(state):
            state["phase"] = "surprise"

        agents = make_agents(critic=FakeAgent("CriticMaster", bad_review))
        with self.assertRaises(RuntimeError) as ctx:
            run_graph(agents)
        self.assertIn("审核返回未知阶段", str(ctx.exception))

    def test_empty_report_never_reaches_review(self):
        agents = make_agents(writer=FakeAgent("LeadWriter", lambda state: None))
        with self.assertRaises(RuntimeError) as ctx:
            run_graph(agents)
        self.assertIn("未产生报告", str(ctx.exception))
        self.assertEqual(agents["critic"].calls, [])

    def test_recursion_limit_covers_longest_business_path(self):
        # plan/research/analyze/write/review + 两轮 (re_research/analyze/write/review) + finish
        self.assertGreaterEqual(RECURSION_LIMIT, 5 + 4 + 4 + 1)

    def test_runtime_only_fields_never_enter_the_checkpoint(self):
        agents = make_agents()
        _, snapshot = run_graph(agents)
        research = snapshot.values["research"]
        self.assertEqual([k for k in research if k.startswith("_")], [])
        self.assertNotIn("messages", research)
        self.assertEqual(research["node_attempts"]["review"], 1)

    def test_unregistered_runtime_field_is_rejected(self):
        with self.assertRaises(RuntimeError):
            pack_state({"_surprise": object()})

    def test_non_serializable_business_field_is_rejected(self):
        with self.assertRaises(TypeError):
            pack_state({"facts": [object()]})


class StageRuntimeProtection(unittest.TestCase):
    def setUp(self):
        clear_cancel_flag("run-protect", "sess-1")

    def test_unlimited_tokens_allows_millions_and_preserves_usage(self):
        from service.deep_research_v2.stage_runtime import consumed_tokens

        agent = FakeAgent("DeepScout")
        execute_stage = make_execute_stage(
            run_id="run-unlimited", session_id="sess-1", agents_of=lambda phase: agent,
        )
        work = {"runtime_budget": {"phases": {
            "DeepScout": {"runs": [{"llm_tokens": 2000000}]},
        }}}

        async def action(state):
            agent.llm_total_tokens += 3000000
            agent.llm_call_count += 1
            await asyncio.sleep(0.02)
            return state

        with patch.dict(os.environ, {"RESEARCH_MAX_LLM_TOKENS": "0"}), patch(
            "service.deep_research_v2.stage_runtime.POLL_SECONDS", 0.005,
        ):
            result = asyncio.run(execute_stage(action, work, "researching"))
        self.assertEqual(result["runtime_budget"]["token_limit"], 0)
        self.assertEqual(consumed_tokens(result), 5000000)

    def test_explicit_positive_token_limit_still_stops_execution(self):
        agent = FakeAgent("DeepScout")
        execute_stage = make_execute_stage(
            run_id="run-token-limit", session_id="sess-1", agents_of=lambda phase: agent,
        )

        async def action(state):
            agent.llm_total_tokens += 2000
            await asyncio.sleep(1)
            return state

        with patch.dict(os.environ, {"RESEARCH_MAX_LLM_TOKENS": "1000"}), patch(
            "service.deep_research_v2.stage_runtime.POLL_SECONDS", 0.005,
        ):
            with self.assertRaisesRegex(StageBudgetExceeded, "1000 token"):
                asyncio.run(execute_stage(action, {}, "researching"))

    def test_cancel_before_stage_stops_execution(self):
        request_cancel("run-protect", "sess-1")
        execute_stage = make_execute_stage(run_id="run-protect", session_id="sess-1")

        async def never_runs(work):
            raise AssertionError("已取消的研究不应执行节点")

        with self.assertRaises(ResearchCancelledError):
            asyncio.run(execute_stage(never_runs, {"phase": "init"}, "init"))
        clear_cancel_flag("run-protect", "sess-1")

    def test_phase_budget_exhaustion_raises(self):
        os.environ["RESEARCH_BUDGET_CHIEFARCHITECT_SECONDS"] = "1"
        try:
            agent = FakeAgent("ChiefArchitect")
            execute_stage = make_execute_stage(
                run_id="run-budget", session_id="sess-1",
                agents_of=lambda phase: agent,
            )

            async def slow(work):
                await asyncio.sleep(5)
                return work

            with self.assertRaises(StageBudgetExceeded):
                asyncio.run(execute_stage(slow, {"phase": "init"}, "init"))
        finally:
            os.environ.pop("RESEARCH_BUDGET_CHIEFARCHITECT_SECONDS", None)

    def test_lease_loss_stops_the_stage(self):
        async def lost():
            return False

        execute_stage = make_execute_stage(
            run_id="run-lease", session_id="sess-1", check_lease=lost,
        )

        async def action(work):
            return work

        from service.deep_research_v2.stage_runtime import LeaseLostDuringStage

        with self.assertRaises(LeaseLostDuringStage):
            asyncio.run(execute_stage(action, {"phase": "init"}, "init"))

    def test_budget_accumulates_across_nodes_for_resume(self):
        agent = FakeAgent("DeepScout")
        execute_stage = make_execute_stage(
            run_id="run-acc", session_id="sess-1", agents_of=lambda phase: agent,
        )
        work = {"phase": "researching", "runtime_budget": {
            "phases": {"DeepScout": {"runs": [{"llm_tokens": 120}]}},
            "elapsed_seconds": 42.0,
        }}

        async def action(state):
            return state

        result = asyncio.run(execute_stage(action, work, "researching"))
        budget = result["runtime_budget"]
        # 恢复后继续累计，不从零重新计时。
        self.assertGreaterEqual(budget["elapsed_seconds"], 42.0)
        self.assertEqual(len(budget["phases"]["DeepScout"]["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
