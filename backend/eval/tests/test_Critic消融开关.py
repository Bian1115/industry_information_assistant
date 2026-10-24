"""真实图上的审核旁路与同首稿分叉；只用假Agent和内存检查点。"""
import asyncio
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "app"), str(Path(__file__).parent)]
from test_单主链路图 import FakeAgent, critic_script, make_agents, passthrough_stage, run_graph
from langgraph.checkpoint.memory import InMemorySaver
from service.deep_research_v2.graph import build_graph, finish, graph_config, pack_state
from service.deep_research_v2.state import create_initial_state


class CriticAblationTests(unittest.TestCase):
    def state(self, **kwargs):
        return create_initial_state("研究问题", "sess-1", ["who.int"], **kwargs)

    def test_disabled_reaches_real_finish_without_review(self):
        agents = make_agents()
        _, s = run_graph(agents, state=self.state(critic_enabled=False))
        state = s.values["research"]
        self.assertEqual(agents["critic"].calls, [])
        self.assertEqual(state["completion_reason"], "critic_disabled")
        self.assertEqual(state["quality_verdict"], "not_reviewed")
        self.assertEqual(state["iteration"], 0)
        self.assertEqual(s.next, ())

    def test_edit_entry_also_skips_review(self):
        state = self.state(critic_enabled=False)
        state["execution_plan"] = "edit"
        agents = make_agents()
        _, s = run_graph(agents, state=state)
        self.assertEqual(agents["architect"].calls, [])
        self.assertEqual(agents["critic"].calls, [])
        self.assertEqual(s.values["research"]["completion_reason"], "critic_disabled")

    def test_zero_iterations_still_reviews(self):
        agents = make_agents(critic=FakeAgent("CriticMaster", critic_script(("needs_revision", "revising"))))
        _, s = run_graph(agents, state=self.state(max_iterations=0))
        self.assertEqual(agents["critic"].calls, ["reviewing"])
        self.assertEqual(s.values["research"]["completion_reason"], "revision_limit")

    def test_finish_preserves_real_citation_audit(self):
        state = self.state(critic_enabled=False)
        state["final_report"] = "报告"
        state["citation_verification"] = {"status": "completed", "audit_id": "existing"}
        result = finish(pack_state(state))["research"]
        self.assertEqual(result["citation_verification"], state["citation_verification"])

    def test_disabled_does_not_release_empty_or_out_of_scope_report(self):
        for report in ["", "错误 [来源](https://other.example/a)"]:
            state = self.state(critic_enabled=False)
            state["final_report"] = report
            with self.subTest(report=report), self.assertRaises(RuntimeError):
                finish(pack_state(state))

    def test_disabled_rejects_prior_review_results(self):
        state = self.state(critic_enabled=False)
        state.update(final_report="报告", quality_verdict="pass")
        with self.assertRaises(RuntimeError):
            finish(pack_state(state))

    def test_switch_requires_bool(self):
        for value in [0, 1, "false", None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.state(critic_enabled=value)

    def test_fork_at_committed_first_write_preserves_exact_report(self):
        async def run():
            saver = InMemorySaver()
            a_agents = make_agents()
            args = dict(run_id="A", user_id="user-1", session_id="sess-1", execute_stage=passthrough_stage)
            prefix = build_graph(a_agents, saver, **args, interrupt_before=["review"])
            await prefix.ainvoke(pack_state(self.state()), graph_config("A"))
            initial = await prefix.aget_state(graph_config("A"))
            self.assertEqual(initial.next, ("review",))
            self.assertEqual(a_agents["critic"].calls, [])
            b = deepcopy(initial.values["research"])
            b.update(critic_enabled=False, session_id="sess-B")
            b_agents = make_agents()
            b_graph = build_graph(b_agents, saver, run_id="B", user_id="user-1", session_id="sess-B", execute_stage=passthrough_stage)
            await b_graph.aupdate_state(graph_config("B"), pack_state(b), as_node="write")
            self.assertEqual((await b_graph.aget_state(graph_config("B"))).next, ("finish",))
            await b_graph.ainvoke(None, graph_config("B"))
            final_b = (await b_graph.aget_state(graph_config("B"))).values["research"]
            self.assertEqual(final_b["final_report"], initial.values["research"]["final_report"])
            self.assertTrue(all(not agent.calls for agent in b_agents.values()))
            self.assertTrue(initial.values["research"]["critic_enabled"])
            a_graph = build_graph(a_agents, saver, **args)
            await a_graph.ainvoke(None, graph_config("A"))
            self.assertEqual((await a_graph.aget_state(graph_config("A"))).values["research"]["completion_reason"], "passed")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
