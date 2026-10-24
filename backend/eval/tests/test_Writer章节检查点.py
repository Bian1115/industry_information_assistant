"""章节 task 的恢复与合稿后统一核验；不调用真实模型或搜索。"""

import copy
import json
import logging
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "app"), str(Path(__file__).parent)]

from langgraph.checkpoint.memory import InMemorySaver
from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.graph import build_graph, graph_config, pack_state
from test_单主链路图 import make_agents, passthrough_stage
from test_Writer论断引用结构 import make_state, claim, URL, synthesis


def initial_state():
    state = make_state()
    state["outline"] = [
        {"id": f"sec_{i}", "title": f"章节{i}", "status": "pending"}
        for i in range(1, 4)
    ]
    state["citation_verification_config"] = {"report_claims": True}
    return state


class Scenario:
    def __init__(self, *, fail_section=None, fail_synthesis=False):
        self.calls = Counter()
        self.fail_section = fail_section
        self.fail_synthesis = fail_synthesis
        self.failed = False
        self.verifications = []

    def writer(self, model="测试模型"):
        writer = LeadWriter.__new__(LeadWriter)
        writer.name = "LeadWriter"
        writer.model = model
        writer.logger = logging.getLogger("test.writer.checkpoints")
        writer.llm_call_count = writer.llm_total_tokens = 0
        writer.parse_json_response = json.loads

        async def call_llm(**kwargs):
            if "首尾" in kwargs["system_prompt"] or "主编" in kwargs["system_prompt"]:
                key = "synthesis"
            else:
                prompt = kwargs["user_prompt"]
                key = next(f"sec_{i}" for i in range(1, 4) if f"标题: 章节{i}\n" in prompt)
            self.calls[key] += 1
            if not self.failed and (key == self.fail_section or key == "synthesis" and self.fail_synthesis):
                self.failed = True
                raise RuntimeError(f"注入{key}失败")
            if key == "synthesis":
                result = synthesis()
            else:
                result = {"content": f"{key}正文：公司2025年营业收入为100亿元。[年度报告]({URL})",
                          "claims": [claim()], "key_points": [key]}
            return json.dumps(result, ensure_ascii=False)

        async def verify(state):
            self.verifications.append(copy.deepcopy(state))
            assert set(state["draft_sections"]) == {"sec_1", "sec_2", "sec_3"}
            assert state["final_report"]
            assert {c["section_id"] for c in state["claims"]} >= {"sec_1", "sec_2", "sec_3"}
            return {"status": "passed"}

        writer.call_llm = call_llm
        writer._verify_and_repair_citations = verify
        return writer


def graph_for(saver, writer, run_id="章节恢复测试"):
    return build_graph(make_agents(writer=writer), saver, run_id=run_id,
                       user_id="测试用户", session_id="测试会话", execute_stage=passthrough_stage)


async def collect(graph, value, run_id="章节恢复测试"):
    return [event async for event in graph.astream(value, graph_config(run_id),
                                                  stream_mode="custom", durability="sync")]


class WriterSectionCheckpoints(unittest.IsolatedAsyncioTestCase):
    async def test_failed_later_section_restores_drafts_and_claims_with_new_agents(self):
        saver = InMemorySaver()
        scenario = Scenario(fail_section="sec_3")
        first = graph_for(saver, scenario.writer())
        with self.assertRaisesRegex(RuntimeError, "sec_3"):
            await collect(first, pack_state(initial_state()))
        snapshot = await first.aget_state(graph_config("章节恢复测试"))
        self.assertEqual(snapshot.next, ("write",))
        self.assertEqual(snapshot.values["research"]["draft_sections"], {})
        self.assertEqual(scenario.verifications, [])
        stored = await saver.aget_tuple(graph_config("章节恢复测试"))
        artifacts = [value for _, channel, value in stored.pending_writes
                     if channel == "__return__" and isinstance(value, dict) and value.get("version") == 1]
        self.assertEqual({v["section_id"] for v in artifacts}, {"sec_1", "sec_2"})
        self.assertTrue(all(v["citation_verification"] == "not_evaluated" for v in artifacts))
        self.assertTrue(all(sum(e["event"]["type"] == "section_content" for e in v["checkpoint_events"]) == 1
                            for v in artifacts))

        resumed = graph_for(saver, scenario.writer())
        events = await collect(resumed, None)
        result = (await resumed.aget_state(graph_config("章节恢复测试"))).values["research"]
        self.assertEqual(scenario.calls, {"sec_1": 1, "sec_2": 1, "sec_3": 2, "synthesis": 1})
        self.assertEqual(len(scenario.verifications), 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(set(result["draft_sections"]), {"sec_1", "sec_2", "sec_3"})
        self.assertTrue(all(result["section_claims"][key] == [claim()] for key in result["draft_sections"]))
        self.assertEqual([ref["url"] for ref in result["references"]], [URL])
        restored = [e["content"]["section_id"] for e in events
                    if e["type"] == "section_content" and e["content"].get("restored")]
        # 章节完成事件随 task 返回值持久化，恢复不能再生成新编号的 restored 事件。
        self.assertEqual(restored, [])
        self.assertFalse(any(e["type"] == "section_content" for e in events))

    async def test_failed_synthesis_reuses_all_generated_sections_before_one_verification(self):
        saver = InMemorySaver()
        scenario = Scenario(fail_synthesis=True)
        with self.assertRaisesRegex(RuntimeError, "synthesis"):
            await collect(graph_for(saver, scenario.writer()), pack_state(initial_state()))
        self.assertEqual(scenario.verifications, [])
        resumed = graph_for(saver, scenario.writer())
        await collect(resumed, None)
        self.assertEqual(scenario.calls, {"sec_1": 1, "sec_2": 1, "sec_3": 1, "synthesis": 2})
        self.assertEqual(len(scenario.verifications), 1)

    async def test_multiple_interruptions_keep_successful_section_results(self):
        saver = InMemorySaver()
        scenario = Scenario(fail_section="sec_2")
        with self.assertRaisesRegex(RuntimeError, "sec_2"):
            await collect(graph_for(saver, scenario.writer()), pack_state(initial_state()))
        scenario.fail_section, scenario.failed = "sec_3", False
        with self.assertRaisesRegex(RuntimeError, "sec_3"):
            await collect(graph_for(saver, scenario.writer()), None)
        await collect(graph_for(saver, scenario.writer()), None)
        self.assertEqual(scenario.calls, {"sec_1": 1, "sec_2": 2, "sec_3": 2, "synthesis": 1})
        self.assertEqual(len(scenario.verifications), 1)

    async def test_changed_model_does_not_silently_reuse_saved_chapter(self):
        saver = InMemorySaver()
        scenario = Scenario(fail_section="sec_2")
        with self.assertRaisesRegex(RuntimeError, "sec_2"):
            await collect(graph_for(saver, scenario.writer()), pack_state(initial_state()))
        with self.assertRaisesRegex(ValueError, "当前输入或写作配置不一致"):
            await collect(graph_for(saver, scenario.writer("另一个模型")), None)
        self.assertEqual(scenario.calls, {"sec_1": 1, "sec_2": 1})
        self.assertEqual(scenario.verifications, [])

    async def test_new_run_does_not_reuse_another_runs_chapters(self):
        saver = InMemorySaver()
        scenario = Scenario()
        for run in ["研究甲", "研究乙"]:
            await collect(graph_for(saver, scenario.writer(), run), pack_state(initial_state()), run)
        self.assertEqual(scenario.calls, {"sec_1": 2, "sec_2": 2, "sec_3": 2, "synthesis": 2})
        self.assertEqual(len(scenario.verifications), 2)


if __name__ == "__main__":
    unittest.main()
