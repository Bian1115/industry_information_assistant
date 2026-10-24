"""独立历史与模型视图的机制回归；模型输出使用桩，不请求付费 API。"""

import asyncio
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import unittest
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "app"))

from service.agent_history import HistoryScope, PostgresAgentHistory, bind_agent_history, current_history_scope, agent_subtask, content_hash, research_history_scope_id
from service.agent_context import prepare_agent_messages, parse_summary, ContextBudgetExceeded, summarize_history, summary_reference_quotes
from service.deep_research_v2.agents.base import BaseAgent
from service.tool_runtime import ToolRuntime, ToolDefinition, ToolCallContext, ToolRuntimeError
from pydantic import BaseModel


class MemoryStore:
    """仅用于确定性测试，生产代码不存在内存降级。"""
    def __init__(self):
        self.data = {}

    def key(self, scope, agent):
        return research_history_scope_id(scope), agent, scope.context_key

    async def load(self, scope, agent, *, replay_only=False):
        result = copy.deepcopy(self.data.setdefault(self.key(scope, agent), {
            "messages": [], "summary": "", "summary_version": 0,
            "covered_seq": 0, "summary_metadata": {},
        }))
        if replay_only:
            result["messages"] = [r for r in result["messages"] if r["seq"] > result["covered_seq"]]
        return result

    async def append(self, scope, agent, kind, payload, call_id, **kwargs):
        await self.load(scope, agent)
        rows = self.data[self.key(scope, agent)]["messages"]
        row = {"source_id": str(uuid.uuid4()), "seq": len(rows)+1, "kind": kind,
               "payload": copy.deepcopy(payload), "call_id": call_id,
               "content_hash": content_hash(payload), "node": scope.node, "invocation_id": scope.invocation_id,
               "run_id": scope.run_id}
        rows.append(row)
        return copy.deepcopy(row)

    async def save_summary(self, scope, agent, *, expected_version, covered_seq, summary, metadata):
        target = self.data[self.key(scope, agent)]
        if target["summary_version"] != expected_version:
            return False
        target.update(summary=summary, summary_version=expected_version+1, covered_seq=covered_seq, summary_metadata=metadata)
        return True

    async def read_source(self, scope, agent, source_id):
        for row in (await self.load(scope, agent))["messages"]:
            if row["source_id"] == source_id:
                return row
        raise PermissionError("scope")

    async def load_tool_sources(self, scope, agent, *, through_seq):
        return [r for r in (await self.load(scope, agent))["messages"]
                if r["kind"] == "tool_output" and r["seq"] <= through_seq]

    async def mark_offloaded(self, scope, agent, source_ids):
        self.data[self.key(scope, agent)]["offloaded_source_ids"] = sorted(source_ids)


class CharCounter:
    def count_text(self, text):
        return len(text) // 4 + 1
    def count_messages(self, messages):
        return sum(self.count_text(m["content"]) + 4 for m in messages)


class Agent(BaseAgent):
    async def process(self, state):
        return state


def make_agent(name="DeepScout", response='{"ok":true}'):
    agent = Agent.__new__(Agent)
    agent.name = name
    agent.model = "test-model"
    agent.logger = logging.getLogger("history-tests")
    agent.llm_call_count = agent.llm_total_tokens = 0
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))], usage=None),
    ))))
    return agent


def summary_text(source_id, claim="保留这个结论。"):
    return "## SESSION INTENT\nNone\n\n## SUMMARY\n" + claim + "\n\n## ARTIFACTS\nNone\n\n## NEXT STEPS\nNone\n\n## CLAIMS\n" + json.dumps([
        {"claim": claim, "references": [{"source_id": source_id, "quote": claim}]}
    ], ensure_ascii=False)


def combined_summary(pairs):
    return "## SESSION INTENT\nNone\n\n## SUMMARY\n" + "\n".join(claim for _, claim in pairs) + "\n\n## ARTIFACTS\nNone\n\n## NEXT STEPS\nNone\n\n## CLAIMS\n" + json.dumps([
        {"claim": claim, "references": [{"source_id": source, "quote": claim}]} for source, claim in pairs
    ], ensure_ascii=False)


class HistoryMechanismTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 固定旧策略作为 v3 回归基线；v4 两阶段流程在独立测试文件验证。
        strategy = patch.dict(os.environ, {"AGENT_SUMMARY_STRATEGY": "incremental-v3", "AGENT_HISTORY_REPLAY": "DeepScout"})
        strategy.start()
        self.addCleanup(strategy.stop)
        self.store = MemoryStore()
        self.scope = HistoryScope(self.store, "run", "session", None, node="research")
        self.counter = patch("service.agent_context.get_qwen_token_counter", return_value=CharCounter())
        self.counter.start()
        self.addCleanup(self.counter.stop)
        for name in ("record_llm_call", "flush_pending"):
            p = patch("service.deep_research_v2.agents.base." + name)
            p.start()
            self.addCleanup(p.stop)

    async def test_all_five_agents_record_original_and_dispatched_messages(self):
        for name in ["ChiefArchitect", "DeepScout", "DataAnalyst", "LeadWriter", "CriticMaster"]:
            agent = make_agent(name)
            with bind_agent_history(self.scope):
                await agent.call_llm("system", "当前任务")
            rows = (await self.store.load(self.scope, name))["messages"]
            self.assertEqual(rows[0]["kind"], "llm_input")
            self.assertEqual(rows[-1]["kind"], "llm_output")
            sent = next(r for r in rows if r["kind"] == "llm_request")
            self.assertEqual(sent["payload"]["request"]["messages"][-1]["content"], "当前任务")

    async def test_scope_isolation_for_parallel_agents_and_subtasks(self):
        async def run(key):
            with bind_agent_history(self.scope), agent_subtask(key):
                await asyncio.sleep(0)
                await make_agent().call_llm("system", key)
                self.assertEqual(current_history_scope().context_key, key)
        await asyncio.gather(run("sec_1"), run("sec_2"))
        self.assertIsNone(current_history_scope())
        for key in ["sec_1", "sec_2"]:
            data = await self.store.load(replace(self.scope, context_key=key), "DeepScout")
            self.assertEqual(data["messages"][0]["payload"]["content"], key)

    async def test_tool_full_snapshot_and_invalid_arguments(self):
        class Args(BaseModel):
            query: str
        text = "长正文完整保存。" * 15000
        handler = AsyncMock(return_value={"content": text})
        definition = ToolDefinition("reader", "reader", Args, handler)
        context = ToolCallContext(metadata={"agent_name": "DeepScout"})
        with bind_agent_history(self.scope):
            result = await ToolRuntime().execute(definition, {"query": "q"}, context)
            with self.assertRaises(ToolRuntimeError):
                await ToolRuntime().execute(definition, {}, context)
        rows = (await self.store.load(self.scope, "DeepScout"))["messages"]
        output = next(r for r in rows if r["kind"] == "tool_output")
        self.assertEqual(output["call_id"], result.call_id)
        self.assertEqual(output["payload"]["value"]["content"], text)
        self.assertEqual(rows[-1]["kind"], "tool_error")
        self.assertEqual(handler.await_count, 1)

    async def test_cancelled_call_keeps_terminal_record(self):
        agent = make_agent()
        agent.client.chat.completions.create.side_effect = asyncio.CancelledError()
        with bind_agent_history(self.scope), self.assertRaises(asyncio.CancelledError):
            await agent.call_llm("sys", "task")
        self.assertEqual((await self.store.load(self.scope, agent.name))["messages"][-1]["kind"], "llm_error")

    async def test_hard_budget_stops_before_provider_request(self):
        agent = make_agent()
        with patch("service.agent_context.input_budget", return_value=100), bind_agent_history(self.scope):
            with self.assertRaises(ContextBudgetExceeded):
                await agent.call_llm("sys", "不能丢弃的用户要求" * 1000)
        agent.client.chat.completions.create.assert_not_awaited()

    async def test_scout_replays_own_previous_work_after_reload(self):
        with bind_agent_history(self.scope):
            await make_agent().call_llm("sys", "先研究甲问题")
            second = make_agent()
            await second.call_llm("sys", "继续乙问题")
        messages = second.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("先研究甲问题", messages[1]["content"])
        self.assertEqual(messages[-1]["content"], "继续乙问题")

    async def test_summary_is_adopted_and_original_is_unchanged(self):
        call = str(uuid.uuid4())
        original = "保留这个结论。" + "背景" * 2500
        row = await self.store.append(self.scope, "DeepScout", "llm_input", {"content": original}, call)
        await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "完成"}, call)
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(return_value=summary_text(row["source_id"]))
        # 主模型预算小，摘要器有独立预算，避免混用。
        with patch("service.agent_context.input_budget", side_effect=lambda out: 2000 if out == 3000 else 900), bind_agent_history(self.scope):
            messages = await prepare_agent_messages(agent, self.scope, "sys", "继续", "current", 6000)
        saved = await self.store.load(self.scope, "DeepScout")
        self.assertEqual(saved["summary_version"], 1)
        self.assertIn('"kind": "history_summary"', messages[1]["content"])
        self.assertNotIn("validated_history_summary", messages[1]["content"])
        self.assertEqual(saved["messages"][0]["payload"]["content"], original)
        # 已覆盖原文不在后续输入中重复出现。
        self.assertNotIn("背景" * 30, str(messages))

    async def test_bad_summary_never_overwrites_original_history(self):
        call = str(uuid.uuid4())
        await self.store.append(self.scope, "DeepScout", "llm_input", {"content": "原文" * 2500}, call)
        await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "完成"}, call)
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(return_value="无效摘要")
        with patch("service.agent_context.input_budget", side_effect=lambda out: 2000 if out == 3000 else 900), bind_agent_history(self.scope):
            with self.assertRaises(ContextBudgetExceeded):
                await prepare_agent_messages(agent, self.scope, "sys", "继续", "current", 6000)
        self.assertEqual((await self.store.load(self.scope, "DeepScout"))["summary_version"], 0)

    def test_claim_validation(self):
        parse_summary(summary_text("s1"), {"s1": "资料：保留这个结论。"})
        for sources in [{"s2": "保留这个结论。"}, {"s1": "这句话没有出现"}]:
            with self.assertRaises(ValueError):
                parse_summary(summary_text("s1"), sources)
        with self.assertRaises(ValueError):
            parse_summary(summary_text("s1").replace("## ARTIFACTS\nNone", "## ARTIFACTS\n遗漏的分句。"), {"s1": "保留这个结论。"})

    async def test_all_completed_rounds_offload_is_persistent_and_current_is_preserved(self):
        await self.seed_previous_summary()
        ids = []
        for i in range(4):
            call = str(uuid.uuid4())
            await self.store.append(self.scope, "DeepScout", "tool_input", {"query": str(i)}, call)
            row = await self.store.append(self.scope, "DeepScout", "tool_output", {"content": str(i) * 5000}, call)
            ids.append(row["source_id"])
        with patch("service.agent_context.input_budget", return_value=6000), bind_agent_history(self.scope):
            messages = await prepare_agent_messages(make_agent(), self.scope, "sys", "当前指令", "current", 6000)
        saved = await self.store.load(self.scope, "DeepScout")
        self.assertEqual(set(saved["offloaded_source_ids"]), set(ids))
        self.assertNotIn("3" * 5000, str(messages))
        self.assertEqual(messages[-1]["content"], "当前指令")
        self.assertIn("0" * 5000, str(saved["messages"]))

    async def test_model_can_read_back_saved_source_without_crossing_budget(self):
        row = await self.store.append(self.scope, "DeepScout", "tool_output", {"content": "回读的证据"}, str(uuid.uuid4()))
        agent = make_agent()
        agent.client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"_history_read": {"source_id": row["source_id"], "chars": 100}})))], usage=None),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"done":true}'))], usage=None),
        ]
        with bind_agent_history(self.scope), patch("service.qwen_token_counter.get_qwen_token_counter", return_value=CharCounter()):
            result = await agent.call_llm("system", "分析证据")
        self.assertEqual(result, '{"done":true}')
        self.assertEqual(agent.client.chat.completions.create.await_count, 2)
        rows = (await self.store.load(self.scope, "DeepScout"))["messages"]
        self.assertTrue(any(r["kind"] == "history_read" for r in rows))

    async def test_storage_failure_prevents_provider_call(self):
        from service.agent_history import AgentHistoryError
        agent = make_agent()
        self.store.append = AsyncMock(side_effect=RuntimeError("数据库断开"))
        with bind_agent_history(self.scope), self.assertRaises(AgentHistoryError):
            await agent.call_llm("sys", "task")
        agent.client.chat.completions.create.assert_not_awaited()

    async def seed_previous_summary(self, *, legacy=False):
        call = str(uuid.uuid4())
        row = await self.store.append(self.scope, "DeepScout", "llm_input", {"content": "旧结论。" + "旧原文不应再次进入模型" * 3000}, call)
        end = await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "完成"}, call)
        summary = summary_text(row["source_id"], "旧结论。")
        metadata = {} if legacy else {
            "summary_hash": content_hash(summary), "reference_quotes": summary_reference_quotes(summary),
        }
        await self.store.save_summary(self.scope, "DeepScout", expected_version=0, covered_seq=end["seq"], summary=summary, metadata=metadata)
        return row, summary

    async def test_incremental_summary_keeps_old_references_but_only_sends_new_originals(self):
        old, previous = await self.seed_previous_summary()
        call = str(uuid.uuid4())
        new = await self.store.append(self.scope, "DeepScout", "llm_input", {"content": "新结论。" + "新增完整原文" * 1000}, call)
        end = await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "完成"}, call)
        result = combined_summary([(old["source_id"], "旧结论。"), (new["source_id"], "新结论。")])
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(return_value=result)
        self.store.read_source = AsyncMock(wraps=self.store.read_source)
        with patch("service.agent_context.input_budget", side_effect=lambda out: 4000 if out == 3000 else 1400), bind_agent_history(self.scope):
            messages = await prepare_agent_messages(agent, self.scope, "sys", "继续", "current", 6000)
            await prepare_agent_messages(agent, self.scope, "sys", "继续", "next", 6000)
        sent = json.loads(agent._call_llm_impl.call_args.args[1])
        self.assertEqual(sent["previous_summary"], previous)
        self.assertEqual({r["source_id"] for r in sent["new_records"]}, {new["source_id"], end["source_id"]})
        self.assertIn("新增完整原文" * 1000, sent["new_records"][0]["text"])
        self.assertNotIn("旧原文不应再次进入模型", str(sent))
        self.store.read_source.assert_not_awaited()
        agent._call_llm_impl.assert_awaited_once()
        saved = await self.store.load(self.scope, agent.name)
        self.assertEqual(saved["summary_version"], 2)
        self.assertEqual(saved["covered_seq"], end["seq"])
        self.assertEqual(saved["summary_metadata"]["parent_covered_seq"], 2)
        self.assertEqual(saved["summary_metadata"]["reference_quotes"][old["source_id"]], ["旧结论。"])
        self.assertEqual(messages[-1]["content"], "继续")

    async def test_legacy_summary_only_rechecks_its_referenced_source(self):
        old, previous = await self.seed_previous_summary(legacy=True)
        from service.agent_context import previous_summary_sources
        self.store.read_source = AsyncMock(wraps=self.store.read_source)
        saved = await self.store.load(self.scope, "DeepScout", replay_only=True)
        self.assertEqual(saved["messages"], [])
        sources = await previous_summary_sources(self.scope, "DeepScout", saved)
        self.assertEqual(sources, {old["source_id"]: ["旧结论。"]})
        self.store.read_source.assert_awaited_once_with(self.scope, "DeepScout", old["source_id"])

    async def test_incremental_bad_quote_does_not_advance_cursor(self):
        old, previous = await self.seed_previous_summary()
        call = str(uuid.uuid4())
        await self.store.append(self.scope, "DeepScout", "llm_input", {"content": "新增" * 3000}, call)
        await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "完成"}, call)
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(return_value=summary_text(old["source_id"], "未提供的旧原文。"))
        with patch("service.agent_context.input_budget", side_effect=lambda out: 4000 if out == 3000 else 1400), bind_agent_history(self.scope), self.assertRaises(ContextBudgetExceeded):
            await prepare_agent_messages(agent, self.scope, "sys", "继续", "current", 6000)
        saved = await self.store.load(self.scope, "DeepScout")
        self.assertEqual(saved["summary"], previous)
        self.assertEqual(saved["covered_seq"], 2)
        self.assertEqual(saved["summary_version"], 1)

    async def test_incremental_chunks_pass_previous_result_and_never_replay_previous_raw_chunk(self):
        old, previous = await self.seed_previous_summary()
        records = []
        for claim in ("新增甲。", "新增乙。"):
            records.append(await self.store.append(self.scope, "DeepScout", "llm_output", {"content": claim + "背景" * 2000}, str(uuid.uuid4())))
        first = combined_summary([(old["source_id"], "旧结论。"), (records[0]["source_id"], "新增甲。")])
        final = combined_summary([(old["source_id"], "旧结论。"), (records[0]["source_id"], "新增甲。"), (records[1]["source_id"], "新增乙。")])
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=[first, final])
        with patch("service.agent_context.input_budget", return_value=2300), bind_agent_history(self.scope):
            result = await summarize_history(agent, self.scope, records, CharCounter(), previous_summary=previous, previous_sources=summary_reference_quotes(previous))
        self.assertEqual(result, final)
        self.assertEqual(agent._call_llm_impl.await_count, 2)
        second = json.loads(agent._call_llm_impl.call_args_list[1].args[1])
        self.assertEqual(second["previous_summary"], first)
        self.assertEqual([r["source_id"] for r in second["new_records"]], [records[1]["source_id"]])
        for call in agent._call_llm_impl.call_args_list:
            self.assertLessEqual(CharCounter().count_messages(call.kwargs["prepared_messages"]), 2300)

    async def test_later_chunk_failure_keeps_previous_summary_and_cursor(self):
        old, previous = await self.seed_previous_summary()
        new = []
        for claim in ("新增甲。", "新增乙。"):
            new.append(await self.store.append(self.scope, "DeepScout", "llm_output", {"content": claim + "背景" * 2000}, str(uuid.uuid4())))
        first = combined_summary([(old["source_id"], "旧结论。"), (new[0]["source_id"], "新增甲。")])
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=[first, "第二块无效输出"])
        with patch("service.agent_context.input_budget", side_effect=lambda out: 2300 if out == 3000 else 1400), bind_agent_history(self.scope), self.assertRaises(ContextBudgetExceeded):
            await prepare_agent_messages(agent, self.scope, "sys", "继续", "current", 6000)
        self.assertEqual(agent._call_llm_impl.await_count, 2)
        saved = await self.store.load(self.scope, agent.name)
        self.assertEqual(saved["summary"], previous)
        self.assertEqual(saved["covered_seq"], 2)
        self.assertEqual(saved["summary_version"], 1)

    def test_noncontiguous_old_quotes_cannot_be_joined_into_a_fake_quote(self):
        with self.assertRaises(ValueError):
            parse_summary(summary_text("old", "第一句。第二句。"), {"old": ["第一句。", "第二句。"]})

    async def test_interleaved_calls_are_not_partially_covered(self):
        await self.seed_previous_summary()
        first, pending = str(uuid.uuid4()), str(uuid.uuid4())
        await self.store.append(self.scope, "DeepScout", "llm_input", {"content": "交错调用原文" * 1200}, first)
        await self.store.append(self.scope, "DeepScout", "llm_input", {"content": "尚未返回"}, pending)
        await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "完成第一项"}, first)
        agent = make_agent()
        agent._call_llm_impl = AsyncMock()
        with patch("service.agent_context.input_budget", return_value=1400), bind_agent_history(self.scope), self.assertRaises(ContextBudgetExceeded):
            await prepare_agent_messages(agent, self.scope, "sys", "继续", "current", 6000)
        agent._call_llm_impl.assert_not_awaited()
        self.assertEqual((await self.store.load(self.scope, agent.name))["covered_seq"], 2)


@unittest.skipUnless(os.getenv("TEST_AGENT_HISTORY_POSTGRES") == "1", "显式启用隔离 PostgreSQL 集成测试")
class PostgresHistoryTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import URL
        from sqlalchemy.orm import sessionmaker
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        from models.research import ResearchRun
        load_dotenv(BACKEND / ".env", override=True)
        url = URL.create("postgresql+psycopg2", username=os.getenv("POSTGRES_USER", "postgres"), password=os.getenv("POSTGRES_PASSWORD", "postgres123"), host=os.getenv("POSTGRES_HOST", "localhost"), port=int(os.getenv("POSTGRES_PORT", "5432")), database=os.getenv("POSTGRES_DB", "industry_assistant"))
        cls.schema = "test_agent_history_" + uuid.uuid4().hex
        cls.admin = create_engine(url)
        with cls.admin.begin() as db:
            db.execute(text(f'CREATE SCHEMA "{cls.schema}"'))
        cls.engine = create_engine(url, connect_args={"options": f"-csearch_path={cls.schema}"})
        cls.sessions = sessionmaker(bind=cls.engine)
        with cls.engine.begin() as db:
            db.execute(text("CREATE TABLE users (id UUID PRIMARY KEY)"))
            ResearchRun.__table__.create(db)
            path = BACKEND / "alembic/versions/20260927_0010_智能体独立工作历史.py"
            spec = importlib.util.spec_from_file_location("history_migration", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.op = Operations(MigrationContext.configure(db))
            module.upgrade()
            module.upgrade()  # 预建表后 Alembic 正式迁移可重复校验，不能重复创建。
            cls.migration_run_id=str(uuid.uuid4())
            cls.migration_source_id=uuid.uuid4()
            cls.migration_session='migration_' + uuid.uuid4().hex
            db.execute(ResearchRun.__table__.insert().values(run_id=cls.migration_run_id, session_id=cls.migration_session))
            old_context=uuid.uuid4()
            db.execute(text("INSERT INTO agent_contexts(id,run_id,agent_name,context_key,next_seq,summary,summary_version,covered_seq) "
                "VALUES(:id,:run,'DeepScout','main',1,'迁移前摘要',2,1)"), {'id':old_context,'run':cls.migration_run_id})
            payload={'content':'迁移前原文'}
            db.execute(text("INSERT INTO agent_messages(id,context_id,seq,kind,call_id,invocation_id,node,payload,content_hash) "
                "VALUES(:id,:ctx,1,'llm_output',:call,:inv,'',CAST(:payload AS JSONB),:hash)"),
                {'id':cls.migration_source_id,'ctx':old_context,'call':uuid.uuid4(),'inv':uuid.uuid4(),
                 'payload':json.dumps(payload),'hash':content_hash(payload)})
            path = BACKEND / "alembic/versions/20261003_0011_智能体会话历史接续.py"
            spec = importlib.util.spec_from_file_location("session_history_migration", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.op = Operations(MigrationContext.configure(db))
            module.upgrade()
        cls.store = PostgresAgentHistory(cls.sessions)

    @classmethod
    def tearDownClass(cls):
        from sqlalchemy import text
        cls.engine.dispose()
        with cls.admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{cls.schema}" CASCADE'))
        cls.admin.dispose()

    async def asyncSetUp(self):
        from models.research import ResearchRun
        self.scope = HistoryScope(self.store, str(uuid.uuid4()), "session_" + uuid.uuid4().hex, None)
        with self.sessions.begin() as db:
            db.add(ResearchRun(run_id=self.scope.run_id, session_id=self.scope.session_id))

    async def test_concurrent_sequence_and_database_reload(self):
        await asyncio.gather(*[self.store.append(self.scope, "DeepScout", "tool_input", {"index": i}, str(uuid.uuid4())) for i in range(24)])
        rows = (await PostgresAgentHistory(self.sessions).load(self.scope, "DeepScout"))["messages"]
        self.assertEqual([r["seq"] for r in rows], list(range(1, 25)))
        self.assertEqual({r["payload"]["index"] for r in rows}, set(range(24)))

    async def test_populated_old_schema_migrates_and_retains_source_and_summary(self):
        from models.research import ResearchRun
        scope=replace(self.scope, run_id=self.migration_run_id, session_id=self.migration_session)
        with self.sessions.begin() as db:
            db.get(ResearchRun, scope.run_id).session_id=scope.session_id
        saved=await self.store.load(scope,'DeepScout')
        self.assertEqual(saved['summary'],'迁移前摘要')
        self.assertEqual(saved['summary_version'],2)
        self.assertEqual(saved['covered_seq'],1)
        row=await self.store.read_source(scope,'DeepScout',str(self.migration_source_id))
        self.assertEqual(row['run_id'],self.migration_run_id)
        self.assertEqual(row['payload'],{'content':'迁移前原文'})

    async def test_owner_agent_subtask_isolation_and_idempotency(self):
        call, event = str(uuid.uuid4()), str(uuid.uuid4())
        row = await self.store.append(self.scope, "DeepScout", "tool_output", {"text": "完整原文" * 20000}, call, event_id=event)
        again = await self.store.append(self.scope, "DeepScout", "tool_output", {"text": "完整原文" * 20000}, call, event_id=event)
        self.assertEqual(row, again)
        self.assertEqual(len((await self.store.load(self.scope, "DeepScout"))["messages"]), 1)
        for other in [replace(self.scope, user_id=str(uuid.uuid4())), replace(self.scope, session_id="other")]:
            with self.assertRaises(PermissionError):
                await self.store.load(other, "DeepScout")
        for scope, agent in [(self.scope, "LeadWriter"), (replace(self.scope, context_key="another"), "DeepScout")]:
            with self.assertRaises(PermissionError):
                await self.store.read_source(scope, agent, row["source_id"])

    async def test_stale_summary_cannot_overwrite_new_version(self):
        await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "原文"}, str(uuid.uuid4()))
        first = await self.store.save_summary(self.scope, "DeepScout", expected_version=0, covered_seq=1, summary="新摘要", metadata={})
        old = await self.store.save_summary(self.scope, "DeepScout", expected_version=0, covered_seq=1, summary="迟到摘要", metadata={})
        self.assertTrue(first)
        self.assertFalse(old)
        self.assertEqual((await self.store.load(self.scope, "DeepScout"))["summary"], "新摘要")

    async def test_replay_query_only_loads_rows_after_summary_cursor(self):
        first = await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "已经覆盖的旧原文"}, str(uuid.uuid4()))
        await self.store.save_summary(self.scope, "DeepScout", expected_version=0, covered_seq=first["seq"], summary="旧摘要", metadata={})
        new = await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "新增完整原文"}, str(uuid.uuid4()))
        restored = PostgresAgentHistory(self.sessions)
        replay = await restored.load(self.scope, "DeepScout", replay_only=True)
        self.assertEqual([r["source_id"] for r in replay["messages"]], [new["source_id"]])
        self.assertEqual(replay["summary"], "旧摘要")
        self.assertEqual(len((await restored.load(self.scope, "DeepScout"))["messages"]), 2)

    async def test_archived_tool_sources_are_scoped_and_limited_by_cursor(self):
        old = await self.store.append(self.scope, "DeepScout", "tool_output", {"content": "旧网页"}, str(uuid.uuid4()))
        await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "旧陈述"}, str(uuid.uuid4()))
        await self.store.append(self.scope, "DeepScout", "tool_output", {"content": "新网页"}, str(uuid.uuid4()))
        await self.store.append(self.scope, "LeadWriter", "tool_output", {"content": "其他智能体"}, str(uuid.uuid4()))
        await self.store.append(replace(self.scope, context_key="other"), "DeepScout", "tool_output", {"content": "其他子任务"}, str(uuid.uuid4()))
        await self.store.save_summary(self.scope, "DeepScout", expected_version=0,
            covered_seq=old["seq"], summary="旧摘要", metadata={})
        rows = await self.store.load_tool_sources(self.scope, "DeepScout", through_seq=old["seq"])
        self.assertEqual(rows, [old])
        with self.assertRaises(PermissionError):
            await self.store.load_tool_sources(replace(self.scope, session_id="other"), "DeepScout", through_seq=old["seq"])

    async def test_previous_lease_cannot_write(self):
        from models.research import ResearchRun
        from service.agent_history import AgentHistoryError
        with self.sessions.begin() as db:
            db.get(ResearchRun, self.scope.run_id).fencing_token = 2
        with self.assertRaises(AgentHistoryError):
            await self.store.append(replace(self.scope, fencing_token=1), "DeepScout", "llm_input", {}, str(uuid.uuid4()))

    async def test_same_session_runs_isolate_summary_cursor_and_source_reads(self):
        from models.research import ResearchRun
        first = await self.store.append(self.scope, 'LeadWriter', 'llm_output', {'content':'旧结论。'}, str(uuid.uuid4()))
        await self.store.save_summary(self.scope, 'LeadWriter', expected_version=0, covered_seq=first['seq'], summary='本run摘要', metadata={})
        restored = replace(self.scope, invocation_id=str(uuid.uuid4()))
        self.assertEqual((await self.store.load(restored, 'LeadWriter'))['summary'], '本run摘要')
        self.assertEqual(await self.store.read_source(restored, 'LeadWriter', first['source_id']), first)
        other = replace(self.scope, run_id=str(uuid.uuid4()))
        with self.sessions.begin() as db:
            db.add(ResearchRun(run_id=other.run_id, session_id=other.session_id))
        fresh = await self.store.load(other, 'LeadWriter')
        self.assertEqual((fresh['summary'], fresh['summary_version'], fresh['covered_seq'], fresh['messages']), ('',0,0,[]))
        with self.assertRaises(PermissionError):
            await self.store.read_source(other, 'LeadWriter', first['source_id'])
        second = await self.store.append(other, 'LeadWriter', 'llm_output', {'content':'另一研究。'}, str(uuid.uuid4()))
        self.assertEqual(second['seq'], 1)
        self.assertTrue(await self.store.save_summary(other, 'LeadWriter', expected_version=0, covered_seq=1, summary='另一个run摘要', metadata={}))
        self.assertEqual((await self.store.load(self.scope, 'LeadWriter'))['summary'], '本run摘要')

    async def test_legacy_context_only_imports_current_run_without_changing_sources(self):
        from models.research import ResearchRun
        from models.agent_history import AgentContext, AgentMessage
        run_ids=[self.scope.run_id, str(uuid.uuid4())]; sources=[]
        with self.sessions.begin() as db:
            db.add(ResearchRun(run_id=run_ids[1], session_id=self.scope.session_id)); db.flush()
            for index,run_id in enumerate(run_ids):
                ctx=AgentContext(id=uuid.uuid4(),run_id=run_id,history_scope_id='run:'+run_id,
                    agent_name='DeepScout',context_key='main',next_seq=1,covered_seq=1,
                    summary='旧摘要'+str(index),summary_version=index+1,summary_metadata={})
                db.add(ctx);db.flush()
                source_id=uuid.uuid4();payload={'content':'完整旧原文'+str(index)}
                db.add(AgentMessage(id=source_id,context_id=ctx.id,run_id=run_id,seq=1,
                    kind='llm_output',call_id=uuid.uuid4(),invocation_id=uuid.uuid4(),node='',
                    payload=payload,content_hash=content_hash(payload)))
                sources.append((str(source_id),payload,run_id))
        saved=await self.store.load(self.scope,'DeepScout')
        self.assertEqual((saved['summary'],saved['covered_seq'],saved['summary_version']),('旧摘要0',1,1))
        self.assertEqual([r['source_id'] for r in saved['messages']],[sources[0][0]])
        self.assertEqual((await self.store.read_source(self.scope,'DeepScout',sources[0][0]))['payload'],sources[0][1])
        with self.assertRaises(PermissionError):
            await self.store.read_source(self.scope,'DeepScout',sources[1][0])
        other=replace(self.scope,run_id=run_ids[1])
        second=await self.store.load(other,'DeepScout')
        self.assertEqual(second['summary'],'旧摘要1')
        self.assertEqual(second['messages'][0]['source_id'],sources[1][0])
        self.assertEqual((await self.store.load(self.scope,'DeepScout'))['messages'],saved['messages'])

    async def test_concurrent_runs_use_independent_sequences(self):
        from models.research import ResearchRun
        other=replace(self.scope,run_id=str(uuid.uuid4()))
        with self.sessions.begin() as db:
            db.add(ResearchRun(run_id=other.run_id,session_id=other.session_id))
        await asyncio.gather(*[self.store.append(self.scope if i%2 else other,'ChiefArchitect',
            'llm_output',{'index':i},str(uuid.uuid4())) for i in range(20)])
        for scope in (self.scope,other):
            rows=(await self.store.load(scope,'ChiefArchitect'))['messages']
            self.assertEqual([r['seq'] for r in rows],list(range(1,11)))
            self.assertEqual({r['run_id'] for r in rows},{scope.run_id})

    async def test_shared_legacy_context_imports_only_current_run_and_ignores_mixed_summary(self):
        from models.research import ResearchRun
        from models.agent_history import AgentContext, AgentMessage
        from service.agent_history import session_history_scope_id
        other = replace(self.scope, run_id=str(uuid.uuid4()))
        ids = [uuid.uuid4(), uuid.uuid4()]
        old_id = uuid.uuid4()
        with self.sessions.begin() as db:
            db.add(ResearchRun(run_id=other.run_id, session_id=other.session_id)); db.flush()
            db.add(AgentContext(id=old_id, run_id=self.scope.run_id,
                history_scope_id=session_history_scope_id(self.scope), agent_name='LeadWriter',
                context_key='main', next_seq=2, covered_seq=2, summary_version=4,
                summary='混合多个研究的旧摘要', summary_metadata={}, offloaded_source_ids=[str(x) for x in ids]))
            db.flush()
            for index, scope in enumerate((other, self.scope)):
                payload = {'content': '原文' + str(index)}
                db.add(AgentMessage(id=ids[index], context_id=old_id, run_id=scope.run_id,
                    seq=index+1, kind='llm_output', call_id=uuid.uuid4(), invocation_id=uuid.uuid4(),
                    node='', payload=payload, content_hash=content_hash(payload)))
        current = await self.store.load(self.scope, 'LeadWriter')
        self.assertEqual((current['summary'], current['summary_version'], current['covered_seq']), ('', 0, 0))
        self.assertEqual([(r['source_id'], r['seq'], r['payload']) for r in current['messages']],
                         [(str(ids[1]), 1, {'content': '原文1'})])
        self.assertEqual(current['offloaded_source_ids'], [str(ids[1])])
        with self.assertRaises(PermissionError):
            await self.store.read_source(self.scope, 'LeadWriter', str(ids[0]))
        with self.sessions() as db:
            self.assertEqual(db.get(AgentContext, old_id).summary, '混合多个研究的旧摘要')
            self.assertEqual(db.get(AgentMessage, ids[0]).context_id, old_id)
        second = await self.store.load(other, 'LeadWriter')
        self.assertEqual([r['source_id'] for r in second['messages']], [str(ids[0])])
        self.assertEqual(second['summary'], '')

    async def test_deleting_run_preserves_audit_without_granting_other_run_access(self):
        from models.research import ResearchRun
        from models.agent_history import AgentMessage
        row=await self.store.append(self.scope,'LeadWriter','llm_output',{'content':'旧原文'},str(uuid.uuid4()))
        other=replace(self.scope,run_id=str(uuid.uuid4()))
        with self.sessions.begin() as db:
            db.add(ResearchRun(run_id=other.run_id,session_id=other.session_id));db.flush()
            db.delete(db.get(ResearchRun,self.scope.run_id))
        with self.sessions() as db:
            self.assertEqual(db.get(AgentMessage,uuid.UUID(row['source_id'])).payload,row['payload'])
        with self.assertRaises(PermissionError):
            await self.store.read_source(other,'LeadWriter',row['source_id'])

    async def test_same_session_string_different_authorized_owners_cannot_share_sources(self):
        from sqlalchemy import text
        from models.research import ResearchRun
        row=await self.store.append(self.scope, 'LeadWriter', 'llm_output', {'content':'原用户资料'}, str(uuid.uuid4()))
        other=replace(self.scope, run_id=str(uuid.uuid4()), user_id=str(uuid.uuid4()))
        with self.sessions.begin() as db:
            db.execute(text('INSERT INTO users(id) VALUES(:id)'), {'id':uuid.UUID(other.user_id)})
            db.add(ResearchRun(run_id=other.run_id, session_id=other.session_id, user_id=uuid.UUID(other.user_id)))
        with self.assertRaises(PermissionError):
            await self.store.read_source(other, 'LeadWriter', row['source_id'])

    async def test_split_citations_summary_persists_and_reloads(self):
        from service.summary_citations import render_sections
        row = await self.store.append(self.scope, "DeepScout", "llm_output", {"content": "结论。" + "完整原文" * 1500}, str(uuid.uuid4()))
        item = {"section": "SUMMARY", "claim": "结论。", "source_ids": [row["source_id"]]}
        response = {"section": "SUMMARY", "claim": "结论。", "references": [{"source_id": row["source_id"], "quote": "结论。"}]}
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=[render_sections([item], markers=True), json.dumps(response, ensure_ascii=False)])
        with patch.dict(os.environ, {"AGENT_HISTORY_REPLAY": "DeepScout", "AGENT_SUMMARY_STRATEGY": "split-citations-v4", "AGENT_SUMMARY_NLI_ENABLED": "0"}), bind_agent_history(self.scope), patch("service.agent_context.get_qwen_token_counter", return_value=CharCounter()), patch("service.agent_context.input_budget", side_effect=lambda out: 4000 if out == 3000 else 1400):
            await prepare_agent_messages(agent, self.scope, "sys", "继续", str(uuid.uuid4()), 6000)
        restored = PostgresAgentHistory(self.sessions)
        saved = await restored.load(self.scope, "DeepScout")
        self.assertEqual(saved["summary_version"], 1)
        self.assertEqual(saved["covered_seq"], row["seq"])
        self.assertEqual(saved["summary_metadata"]["mode"], "incremental_split_citations")
        self.assertEqual(saved["messages"][0]["payload"], row["payload"])
        replay = await restored.load(self.scope, "DeepScout", replay_only=True)
        self.assertEqual(replay["messages"], [])
        self.assertEqual([c["claim"] for c in parse_summary(saved["summary"], {row["source_id"]: "结论。"})[1]], ["结论。"])

    async def test_real_graph_persists_each_agent_history(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from service.deep_research_v2.graph import build_graph, graph_config
        from service.deep_research_v2.state import create_initial_state
        class GraphAgent(Agent):
            async def process(self, state):
                await self.call_llm("sys", "阶段：" + self.name)
                if self.name == "LeadWriter":
                    state["final_report"] = "报告"
                if self.name == "CriticMaster":
                    state.update(phase="completed", quality_verdict="pass")
                return state
        agents = {}
        for key, name in [("architect", "ChiefArchitect"), ("scout", "DeepScout"), ("data_analyst", "DataAnalyst"), ("writer", "LeadWriter"), ("critic", "CriticMaster")]:
            agent = make_agent(name)
            agent.__class__ = GraphAgent
            agents[key] = agent
        async def execute(action, state, phase):
            return await action(state)
        graph = build_graph(agents, InMemorySaver(), run_id=self.scope.run_id, user_id=None, session_id=self.scope.session_id, execute_stage=execute, history_store=self.store)
        with patch("service.agent_context.get_qwen_token_counter", return_value=CharCounter()), patch("service.deep_research_v2.agents.base.record_llm_call"), patch("service.deep_research_v2.agents.base.flush_pending"):
            result = await graph.ainvoke({"research": create_initial_state("问题", self.scope.session_id, ["who.int"])}, graph_config(self.scope.run_id))
        self.assertEqual(result["research"]["status"], "completed")
        for agent in agents.values():
            rows = (await self.store.load(self.scope, agent.name))["messages"]
            self.assertTrue(any(r["kind"] == "llm_output" for r in rows), agent.name)
            self.assertTrue(all(r["node"] for r in rows))


if __name__ == "__main__":
    unittest.main()
