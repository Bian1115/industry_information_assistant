"""无网络验证请求记账接入、完成后保护和部署拒绝边界。"""
import asyncio
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "app"), str(Path(__file__).parent)]
from openai import APIConnectionError
from service.deep_research_v2.agents import base as base_module
from service.deep_research_v2.agents.base import BaseAgent
from service.deep_research_v2.budget_ledger import bind_runtime_budget
from service.deep_research_v2.checkpointer import checkpoint_database_url, for_execution
from service.deep_research_v2.stage_runtime import LeaseLostDuringStage, StageBudgetExceeded, make_execute_stage
from test_单主链路图 import FakeAgent


class Agent(BaseAgent):
    async def process(self, state):
        return state


def fake_agent(responses):
    agent = Agent.__new__(Agent)
    agent.name, agent.model = "测试模型调用", "固定模型"
    agent.logger = logging.getLogger("test.runtime_ledger")
    agent.llm_call_count = agent.llm_total_tokens = 0
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=responses))))
    return agent


def response(usage=True):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"result":1}'))],
                           usage=SimpleNamespace(total_tokens=40, prompt_tokens=30, completion_tokens=10) if usage else None)


class RuntimeBudgetProtection(unittest.IsolatedAsyncioTestCase):
    async def test_lease_loss_when_action_finishes_is_rejected(self):
        valid = True
        async def check():
            return valid
        async def action(state):
            nonlocal valid
            valid = False
            return state
        execute = make_execute_stage(run_id="离线租约", session_id="离线会话", check_lease=check)
        with patch("service.deep_research_v2.stage_runtime.is_research_cancelled", return_value=False):
            with self.assertRaises(LeaseLostDuringStage):
                await execute(action, {}, "init")

    async def test_fast_action_cannot_skip_token_limit_and_failure_is_not_completed(self):
        agent, work = FakeAgent("DeepScout"), {}
        async def action(state):
            agent.llm_total_tokens += 20
            return state
        execute = make_execute_stage(run_id="离线预算", session_id="离线会话", agents_of=lambda p: agent)
        with patch("service.deep_research_v2.stage_runtime.is_research_cancelled", return_value=False), patch.dict(
            os.environ, {"RESEARCH_MAX_LLM_TOKENS": "10"}
        ):
            with self.assertRaises(StageBudgetExceeded):
                await execute(action, work, "researching")
        self.assertFalse(work["runtime_budget"]["phases"]["DeepScout"]["runs"][0]["completed"])


class RequestAccounting(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, agent, store):
        with bind_runtime_budget(store, "物理尝试"), patch.object(base_module, "record_llm_call"), patch.object(
            base_module, "flush_pending"
        ), patch.object(base_module, "get_current_cancellation_token", return_value=None), patch(
            "service.qwen_token_counter.get_qwen_token_counter", return_value=Mock(count_messages=Mock(return_value=20))
        ), patch.object(base_module, "default_agent_max_attempts", return_value=2), patch.object(
            base_module, "default_fallback_model", return_value="备用模型"
        ):
            return await agent._call_llm_impl("system", "user", max_tokens=100)

    async def test_reservation_precedes_request_and_usage_is_settled(self):
        order = []
        store = SimpleNamespace(reserve=AsyncMock(side_effect=lambda *a: order.append("reserve") or "请求"), complete=AsyncMock())
        agent = fake_agent([response()])
        original = agent.client.chat.completions.create
        async def create(**kwargs):
            order.append("request")
            return await original(**kwargs)
        agent.client.chat.completions.create = create
        await self.invoke(agent, store)
        self.assertEqual(order, ["reserve", "request"])
        store.reserve.assert_awaited_once_with("物理尝试", "固定模型", 120)
        store.complete.assert_awaited_once_with("物理尝试", "请求", {"total_tokens": 40, "prompt_tokens": 30, "completion_tokens": 10})

    async def test_reservation_rejection_makes_no_provider_request(self):
        store = SimpleNamespace(reserve=AsyncMock(side_effect=StageBudgetExceeded("累计预算不足")), complete=AsyncMock())
        agent = fake_agent([response()])
        with self.assertRaises(StageBudgetExceeded):
            await self.invoke(agent, store)
        agent.client.chat.completions.create.assert_not_awaited()
        store.complete.assert_not_awaited()

    async def test_missing_usage_retains_unknown_reservation(self):
        store = SimpleNamespace(reserve=AsyncMock(return_value="请求"), complete=AsyncMock())
        await self.invoke(fake_agent([response(False)]), store)
        store.complete.assert_awaited_once_with("物理尝试", "请求", None)

    async def test_each_provider_retry_has_its_own_reservation(self):
        store = SimpleNamespace(reserve=AsyncMock(side_effect=["失败请求", "成功请求"]), complete=AsyncMock())
        agent = fake_agent([APIConnectionError(request=Mock()), response()])
        # 只压缩本用例的供应商退避，不改变阶段运行器的轮询。
        with patch.object(base_module.asyncio, "sleep", new=AsyncMock()):
            await self.invoke(agent, store)
        self.assertEqual(store.reserve.await_count, 2)
        self.assertEqual(store.complete.await_args_list[0].args, ("物理尝试", "失败请求", None))
        self.assertEqual(store.complete.await_args_list[1].args[1], "成功请求")


class DeploymentBoundary(unittest.TestCase):
    def test_separate_database_cannot_silently_disable_atomic_fencing(self):
        with patch.dict(os.environ, {"POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "业务库",
            "RESEARCH_CHECKPOINT_DATABASE_URL": "postgresql://test:test@localhost:5432/其他库"}):
            with self.assertRaisesRegex(ValueError, "检查点必须与运行租约同库"):
                checkpoint_database_url()

    def test_unsupported_execution_saver_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "PostgreSQL saver"):
            for_execution(object(), object())


class LiveBudgetProjection(unittest.IsolatedAsyncioTestCase):
    async def test_state_uses_durable_budget_instead_of_failed_nodes_old_checkpoint(self):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from service.deep_research_v2 import service as service_module
        old = {"runtime_budget": {"known_tokens": 40}, "status": "running"}
        snapshot = SimpleNamespace(values={"research": old}, next=("research",), tasks=(), config={})
        durable = {"known_tokens": 140, "charged_tokens": 140, "node_attempts": {"research": 2}}
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=snapshot))
        service = service_module.DeepResearchV2Service(checkpointer=AsyncPostgresSaver(object()))
        with patch.object(service_module, "build_graph", return_value=graph), patch.object(
            service_module, "create_agents", return_value={}
        ), patch.object(service_module, "read_runtime_budget", new=AsyncMock(return_value=durable)):
            with patch.object(service.checkpointer, "aget_tuple", new=AsyncMock(return_value=None)), patch.object(
                service_module, "StorageCoordinator", return_value=SimpleNamespace(inspect=AsyncMock(
                    return_value={"status":"consistent", "blocked":False}))) :
                state = await service.read_state("预算投影测试")
        self.assertEqual(state["runtime_budget"]["known_tokens"], 140)
        self.assertEqual(state["research"]["node_attempts"]["research"], 2)
        self.assertEqual(old["runtime_budget"]["known_tokens"], 40)


if __name__ == "__main__":
    unittest.main()
