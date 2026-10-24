"""真实图与服务入口的集成回归；仅替换模型、数据库交付及 Redis。"""
import json
import logging
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "app"), str(Path(__file__).parent)]
from test_单主链路图 import FakeAgent, make_agents, writes_report
from langgraph.checkpoint.memory import InMemorySaver
from config.llm_config import get_config
from service.deep_research_v2 import DeepResearchV2Service
from service.deep_research_v2 import graph as graph_module, service as service_module
from service.deep_research_v2.agents.base import BaseAgent
from service.deep_research_v2.delivery import build_complete_event, checkpoint_id_of
from service.deep_research_v2.state import create_initial_state
from service.research_events import ResearchRunLease
from service import research_control


class GraphServiceIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_service_runs_actual_graph_streams_events_and_resumes_failed_write(self):
        """真实 astream/custom/检查点贯穿服务入口，失败后只重跑写作。"""
        writes = {"n": 0}
        def write(state):
            writes["n"] += 1
            if writes["n"] == 1:
                state["facts"].append({"content": "失败节点的未提交数据"})
                raise RuntimeError("注入写作失败")
            self.assertEqual(state["facts"], [])
            writes_report(state)
        architect = FakeAgent("ChiefArchitect")
        architect.logger = logging.getLogger("test.graph_service")
        architect._behaviour = lambda state: BaseAgent.add_message(
            architect, state, "thought", {"content": "实际 Agent 事件出口"}
        )
        agents = make_agents(architect=architect, writer=FakeAgent("LeadWriter", write))
        service = DeepResearchV2Service(checkpointer=InMemorySaver())
        delivered = []
        def make_commit(**kwargs):
            def commit(snapshot):
                self.assertEqual(snapshot.next, ())
                research = snapshot.values["research"]
                self.assertEqual(research["status"], "completed")
                self.assertIn("calculation_checks", research)
                self.assertEqual([k for k in research if k.startswith("_")], [])
                delivered.append(snapshot)
                return kwargs["recorder"].record(build_complete_event(
                    research, run_id=kwargs["run_id"], checkpoint_id=checkpoint_id_of(snapshot),
                ))
            return commit
        async def collect(resume):
            return [chunk async for chunk in service.research(
                query="研究问题", session_id="session-integration", user_id="user-integration",
                run_id="run-integration", resume=resume, persist_events=False,
                allowed_domains=["who.int"],
            )]
        with patch.object(service_module, "create_agents", return_value=agents), patch.object(
            service_module, "make_commit_result", side_effect=make_commit
        ), patch("service.deep_research_v2.followup_context.latest_parent_run", return_value=None):
            first = await collect(False)
            failed = [json.loads(c[6:]) for c in first if c != "data: [DONE]\n\n"]
            self.assertEqual(failed[-1]["type"], "error")
            self.assertNotIn("research_complete", [e["type"] for e in failed])
            state = await service.read_state("run-integration")
            self.assertEqual(state["pending_nodes"], ["write"])
            self.assertEqual(state["research"]["facts"], [])
            self.assertEqual(delivered, [])
            resumed = await collect(True)
        events = [json.loads(c[6:]) for c in resumed if c != "data: [DONE]\n\n"]
        self.assertEqual(events[0]["type"], "research_resumed")
        self.assertEqual([e["phase"] for e in events if e["type"] == "phase"], ["writing", "reviewing"])
        self.assertEqual(events[-1]["type"], "research_complete")
        self.assertEqual(events[-1]["quality_verdict"], "pass")
        self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
        thought = next(e for e in failed if e["type"] == "thought")
        self.assertEqual(thought["node"], "plan")
        self.assertEqual(thought["content"]["content"], "实际 Agent 事件出口")
        self.assertEqual(len(delivered), 1)
        for name in ("architect", "scout", "data_analyst", "critic"):
            self.assertEqual(len(agents[name].calls), 1)
        self.assertEqual(len(agents["writer"].calls), 2)
        self.assertEqual(resumed[-1], "data: [DONE]\n\n")

    async def test_agent_setup_failure_releases_lease_and_cancellation_token(self):
        lease = ResearchRunLease(
            run_id="run-setup-failure", session_id="session-setup-failure",
            owner_id="owner", fencing_token=1,
            lease_until=datetime.utcnow() + timedelta(minutes=1),
        )
        service = DeepResearchV2Service(checkpointer=InMemorySaver())
        with patch.object(service_module, "create_agents", side_effect=RuntimeError("注入初始化失败")), patch.object(
            service_module, "release_research_run_lease", return_value=True
        ) as release:
            chunks = [c async for c in service.research(
                query="问题", session_id=lease.session_id, user_id="user",
                run_id=lease.run_id, lease=lease, persist_events=False,
                allowed_domains=["who.int"],
            )]
        self.assertEqual(json.loads(chunks[0][6:])["error_code"], "research_failed")
        release.assert_called_once_with(lease, "failed")
        self.assertNotIn(lease.run_id, research_control._tokens)

    async def test_main_graph_analyst_does_not_call_chart_generator(self):
        analyst = graph_module.create_agents(get_config())["data_analyst"]
        self.assertFalse(analyst.generate_charts)
        analyst._extract_data = AsyncMock(return_value={"data_points": [{"value": 1}]})
        analyst._build_knowledge_graph = AsyncMock(return_value={"nodes": [], "edges": []})
        analyst._generate_charts = AsyncMock(side_effect=AssertionError("不应绘图"))
        analyst.add_message = Mock()
        state = create_initial_state("问题", "session", ["who.int"])
        state["phase"] = "analyzing"
        result = await analyst.process(state)
        analyst._generate_charts.assert_not_awaited()
        self.assertEqual(result["charts"], [])
        self.assertEqual(result["chart_generation"]["status"], "not_executed")


class GraphContract(unittest.TestCase):
    def test_agent_factory_uses_role_models_and_run_isolation(self):
        config = get_config()
        first, second = graph_module.create_agents(config), graph_module.create_agents(config)
        self.assertEqual(set(first), {"architect", "scout", "data_analyst", "writer", "critic"})
        for role, agent in first.items():
            self.assertEqual(agent.model, getattr(config.agents, role).model)
            self.assertIsNot(agent, second[role])
        self.assertIsNot(first["scout"].tool_runtime, second["scout"].tool_runtime)
        self.assertIs(graph_module.agents_by_phase(first)("re_researching"), first["scout"])
        self.assertIs(graph_module.agents_by_phase(first)("revising"), first["writer"])
        self.assertIsNone(graph_module.agents_by_phase(first)("routing"))

    def test_premature_completion_and_missing_checkpointer_are_rejected(self):
        state = create_initial_state("问题", "session", ["who.int"])
        state.update(final_report="报告", phase="completed", quality_verdict="needs_revision")
        with self.assertRaisesRegex(RuntimeError, "拒绝宣布完成"):
            graph_module.finish({"research": state})
        with self.assertRaisesRegex(ValueError, "检查点"):
            graph_module.build_graph({}, None, run_id="run", user_id="user", session_id="session", execute_stage=None)
