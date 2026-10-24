"""单章修订材料隔离；模型使用替身，原文和事实保留不变。"""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "app"), str(Path(__file__).resolve().parent)]

from service.deep_research_v2.writer_claims import format_section_revision_evidence
from service.web_document import index_document
from test_Writer证据去重与程序参考文献 import state_fixture, writer_fixture


OBSERVATIONS = []


def fixture():
    state = state_fixture()
    state["outline"].append({"id": "sec_2", "title": "其他项目", "status": "drafted"})
    state["facts"][0]["related_sections"] = ["sec_1"]
    state["facts"][1]["related_sections"] = ["sec_2"]
    state["facts"][1]["is_supplementary"] = True
    state["facts"][1]["critic_issue_id"] = "issue_1"
    state["draft_sections"] = {"sec_1": "本章正文。" * 150, "sec_2": "另一章正文。" * 150}
    state["critic_feedback"] = [{"id": "issue_1", "target_section": "sec_1", "resolved": False}]
    return state


def add_source(state, name, content, **quality):
    url = f"https://official.example.cn/{name}"
    source = {"id": name, "url": url, "title": name, "content": content,
              "body_quality": {"status": "passed", **quality}, **index_document(content, url)}
    state["raw_sources"].append(source)
    return source


def add_fact(state, source, name, sections=(), **metadata):
    fact = {"id": name, "source_id": source["id"], "content": source["content"],
            "document_id": source["document_id"], "sentence_ids": ["s000001"],
            "related_sections": list(sections), **metadata}
    state["facts"].append(fact)
    return fact


def packet(prompt):
    start = "## 可逐字引用的原文片段（包含原章节和补充事实的来源）\n"
    return json.loads(prompt.split(start, 1)[1].split("\n\n## 论断与引用证据声明", 1)[0])


class RevisionInputTests(unittest.IsolatedAsyncioTestCase):
    async def revise(self, state):
        writer = writer_fixture()
        writer.parse_json_response = json.loads
        writer.call_llm = AsyncMock(return_value=json.dumps({
            "revised_content": "保留本章内容与证据范围。" * 100,
            "claims": [], "addressed_issues": [], "unable_to_address": [],
        }, ensure_ascii=False))
        writer._synthesize_report = AsyncMock()
        before = copy.deepcopy({key: state[key] for key in ("facts", "raw_sources")})
        await writer._revise_report(state)
        self.assertEqual({key: state[key] for key in before}, before)
        prompts = [call.kwargs["user_prompt"] for call in writer.call_llm.call_args_list]
        OBSERVATIONS.append({"case": self._testMethodName, "revision_calls": len(prompts),
                             "inputs": [{"facts": [f["evidence_id"] for f in packet(p)["facts"]],
                                         "documents": [d["document_id"] for d in packet(p)["documents"]],
                                         "chars": len(p)} for p in prompts]})
        return prompts

    async def test_same_document_other_chapter_sentence_is_excluded(self):
        state = fixture()
        prompt, = await self.revise(state)
        evidence = packet(prompt)
        self.assertEqual([fact["evidence_id"] for fact in evidence["facts"]], ["f1"])
        self.assertEqual([row["id"] for row in evidence["documents"][0]["original_sentences"]],
                         ["s000001", "s000003"])
        self.assertNotIn("服务人口为6000人", prompt)

    async def test_supplements_require_chapter_binding_even_when_issue_id_matches(self):
        state = fixture()
        current = add_source(state, "当前章补搜", "本章新增边界证据。")
        add_fact(state, current, "current", ["sec_1"], is_supplementary=True)
        unassigned = add_source(state, "未关联补搜", "未关联到本章的材料不得展示。")
        add_fact(state, unassigned, "unassigned", is_supplementary=True, critic_issue_id="issue_1")
        prompt, = await self.revise(state)
        self.assertEqual([f["evidence_id"] for f in packet(prompt)["facts"]], ["f1", "current"])
        self.assertIn(current["content"], prompt)
        self.assertNotIn(unassigned["content"], prompt)
        self.assertNotIn(unassigned["url"], prompt)

    async def test_missing_chapter_evidence_does_not_fall_back_to_global_facts(self):
        state = fixture()
        state["facts"][0]["related_sections"] = []
        prompt, = await self.revise(state)
        self.assertEqual(packet(prompt), {"facts": [], "documents": []})
        self.assertIn("无可用补充证据", prompt)
        self.assertNotIn("项目于2023年投运", prompt)
        self.assertNotIn("服务人口为6000人", prompt)

    async def test_existing_chapter_quote_is_kept_without_expanding_to_other_facts(self):
        state = fixture()
        source = add_source(state, "原章来源", "本章历史原句。\n同源其他章原句。")
        add_fact(state, source, "other", ["sec_2"])
        paragraph = "本章原有论断。"
        state["draft_sections"]["sec_1"] = paragraph + state["draft_sections"]["sec_1"]
        state["section_claims"] = {"sec_1": [{"claim": "本章原有论断", "report_quote": paragraph,
            "evidence": [{"document_id": source["document_id"], "sentence_ids": ["s000001"]}]}]}
        prompt, = await self.revise(state)
        evidence = packet(prompt)
        self.assertEqual([f["evidence_id"] for f in evidence["facts"]], ["f1"])
        original = next(d for d in evidence["documents"] if d["document_id"] == source["document_id"])
        self.assertEqual(original["original_sentences"], [{"id": "s000001", "text": "本章历史原句。"}])
        self.assertNotIn("同源其他章原句", prompt)
        self.assertIn(source["url"], prompt)

    async def test_global_feedback_still_keeps_each_chapters_materials_separate(self):
        state = fixture()
        state["critic_feedback"][0]["target_section"] = "全局"
        first, second = await self.revise(state)
        self.assertEqual([f["evidence_id"] for f in packet(first)["facts"]], ["f1"])
        self.assertEqual([f["evidence_id"] for f in packet(second)["facts"]], ["f2"])
        self.assertEqual([s["id"] for s in packet(second)["documents"][0]["original_sentences"]],
                         ["s000002", "s000003"])
        self.assertNotIn("项目于2023年投运", second)


class OriginalQuoteTests(unittest.TestCase):
    def payload(self, state, fact_ids=()):
        before = copy.deepcopy(state)
        result = json.loads(format_section_revision_evidence(state, "sec_1", fact_ids))
        self.assertEqual(state, before)
        return result

    def test_original_quote_without_fact_is_retained_and_deduplicated(self):
        state = fixture()
        paragraph = state["draft_sections"]["sec_1"][:6]
        reference = {"document_id": state["raw_sources"][0]["document_id"], "sentence_ids": ["s000001"]}
        declaration = {"claim": "原章论断", "report_quote": paragraph, "evidence": [reference, reference]}
        state["section_claims"] = {"sec_1": [declaration]}
        state["facts"] = []
        payload = self.payload(state)
        self.assertEqual(payload["facts"], [])
        self.assertEqual(payload["documents"][0]["original_sentences"],
                         [{"id": "s000001", "text": "项目于2023年投运。"}])

    def test_stale_and_other_chapter_declarations_do_not_expand_scope(self):
        state = fixture()
        ref = {"document_id": state["raw_sources"][0]["document_id"], "sentence_ids": ["s000002"]}
        state["section_claims"] = {
            "sec_1": [{"claim": "过期论断", "report_quote": "已经删除的旧正文。", "evidence": [ref]}],
            "sec_2": [{"claim": "其他章论断", "report_quote": "另一章正文。", "evidence": [ref]}],
        }
        self.assertEqual(self.payload(state), {"facts": [], "documents": []})

    def test_rejected_sources_and_invalid_sentence_ids_are_not_reintroduced(self):
        for case in ("rejected", "missing_sentence", "malformed_document"):
            with self.subTest(case=case):
                state = fixture()
                source = state["raw_sources"][0]
                ref = {"document_id": source["document_id"], "sentence_ids": ["s000001"]}
                if case == "rejected":
                    source["body_quality"]["status"] = "rejected"
                elif case == "missing_sentence":
                    ref["sentence_ids"] = ["s999999"]
                else:
                    ref["document_id"] = []
                state["section_claims"] = {"sec_1": [{"claim": "原章论断", "report_quote": "本章正文。",
                                                      "evidence": [ref]}]}
                self.assertEqual(self.payload(state), {"facts": [], "documents": []})

    def test_fact_and_original_claim_share_sentences_without_duplication(self):
        state = fixture()
        ref = {"document_id": state["raw_sources"][0]["document_id"], "sentence_ids": ["s000003"]}
        state["section_claims"] = {"sec_1": [{"claim": "原章论断", "report_quote": "本章正文。",
                                              "evidence": [ref]}]}
        payload = self.payload(state, ["f1"])
        self.assertEqual([r["id"] for r in payload["documents"][0]["original_sentences"]],
                         ["s000001", "s000003"])


if __name__ == "__main__":
    unittest.main()
