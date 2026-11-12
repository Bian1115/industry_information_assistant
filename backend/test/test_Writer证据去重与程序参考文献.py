"""Writer 输入去重及实际引用目录回归；离线运行，不调用模型。"""

import copy
import json
import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.state import create_initial_state
from service.deep_research_v2.writer_claims import format_claim_evidence
from service.web_document import index_document


URL = "https://official.example.cn/report"
OTHER = URL + "2"
BODY = "项目于2023年投运。\n服务人口为6000人。\n仅覆盖城区。"


def state_fixture():
    state = create_initial_state("分析项目", "test", ["official.example.cn"])
    state["outline"] = [{"id": "sec_1", "title": "项目", "status": "drafted"}]
    state["raw_sources"] = [{
        "id": "source_1", "url": URL, "title": "官方项目报告", "content": BODY,
        "author": "发布机构", "date": "2025-01-01",
        "body_quality": {"status": "passed"}, **index_document(BODY, URL),
    }]
    doc = state["raw_sources"][0]["document_id"]
    state["facts"] = [
        {"id": "f1", "content": "2023年投运", "source_id": "source_1",
         "document_id": doc, "sentence_ids": ["s000001", "s000003"]},
        {"id": "f2", "content": "6000人且仅覆盖城区", "source_id": "source_1",
         "document_id": doc, "sentence_ids": ["s000002", "s000003"]},
    ]
    return state


def writer_fixture():
    writer = LeadWriter.__new__(LeadWriter)
    writer.name = "LeadWriter"
    writer.logger = logging.getLogger("test.writer.compact")
    return writer


class InputTests(unittest.TestCase):
    def test_overlapping_supports_are_deduplicated_without_expanding_fact_binding(self):
        state = state_fixture()
        before = copy.deepcopy(state)
        payload = json.loads(format_claim_evidence(state))
        self.assertEqual([f["sentence_ids"] for f in payload["facts"]],
                         [["s000001", "s000003"], ["s000002", "s000003"]])
        originals = payload["documents"][0]["original_sentences"]
        self.assertEqual(len(originals), 3)
        self.assertEqual(sum(s["id"] == "s000003" for s in originals), 1)
        self.assertEqual(set(originals[0]), {"id", "text"})
        self.assertNotIn("original_excerpt", json.dumps(payload))
        self.assertEqual(state, before)

    def test_same_sentence_number_in_different_documents_is_not_merged(self):
        state = state_fixture()
        other = {"id": "source_2", "url": OTHER, "content": "另一项目于2024年投运。",
                 "body_quality": {"status": "passed"},
                 **index_document("另一项目于2024年投运。", OTHER)}
        state["raw_sources"].append(other)
        state["facts"].append({"id": "f3", "content": "另一项目投运", "source_id": "source_2",
                               "document_id": other["document_id"], "sentence_ids": ["s000001"]})
        payload = json.loads(format_claim_evidence(state))
        self.assertEqual(len(payload["facts"]), 3)
        self.assertEqual(len(payload["documents"]), 2)
        self.assertEqual(payload["documents"][1]["original_sentences"],
                         [{"id": "s000001", "text": "另一项目于2024年投运。"}])

    def test_selected_facts_only_expose_their_own_sentences(self):
        payload = json.loads(format_claim_evidence(state_fixture(), ["f2"]))
        self.assertEqual([f["evidence_id"] for f in payload["facts"]], ["f2"])
        self.assertEqual([s["id"] for s in payload["documents"][0]["original_sentences"]],
                         ["s000002", "s000003"])

    def test_unindexed_legacy_source_does_not_get_invented_sentence_ids(self):
        state = {"raw_sources": [{"id": "old", "url": URL, "content": BODY,
                                  "body_quality": {"status": "passed"}}],
                 "facts": [{"id": "f1", "source_id": "old", "content": "历史事实"}]}
        payload = json.loads(format_claim_evidence(state))
        self.assertEqual(payload["documents"], [])
        self.assertEqual(payload["facts"][0]["sentence_ids"], [])
        self.assertEqual(payload["legacy_sources"][0]["text"], BODY)
        self.assertNotIn("original_excerpt", json.dumps(payload))

    def test_compact_ledger_keeps_all_support_refute_and_inconclusive_ids(self):
        state = state_fixture()
        state["hypotheses"] = [{"id": "h1", "content": "项目可推广", "status": "partial"}]
        state["hypothesis_evidence"] = [
            {"id": f"ev{i}", "hypothesis_id": "h1", "fact_id": "f1",
             "evidence_type": kind, "summary": "冗长解释" * 100}
            for i, kind in enumerate(["supports"] * 5 + ["refutes", "inconclusive"])
        ]
        before = copy.deepcopy(state)
        payload = json.loads(writer_fixture()._format_hypothesis_ledger(state))
        self.assertEqual(len(payload[0]["evidence"]), 7)
        self.assertEqual(payload[0]["evidence"][-2]["type"], "refutes")
        self.assertEqual(set(payload[0]["evidence"][0]), {"evidence_id", "fact_id", "type"})
        self.assertNotIn("冗长解释", json.dumps(payload, ensure_ascii=False))
        self.assertEqual(state, before)


class ReferenceTests(unittest.TestCase):
    def test_actual_inline_links_order_dedup_metadata_and_remove_unused(self):
        state = state_fixture()
        state["raw_sources"].append({"id": "s2", "url": OTHER, "title": "第二报告"})
        state["references"] = [{"url": URL, "title": "旧模型标题"}]
        content = (f"[第二来源]({OTHER}) [首个来源]({URL}#page1) [重复]({URL}) "
                   "[未知](https://unknown.example.cn/) ![图片](https://image.example.cn/) ")
        refs = writer_fixture()._references_from_content(state, content)
        self.assertEqual([r["url"] for r in refs], [OTHER, URL])
        self.assertEqual(refs[1]["title"], "官方项目报告")
        self.assertEqual(refs[1]["author"], "发布机构")
        self.assertEqual(refs[1]["date"], "2025-01-01")
        self.assertEqual(writer_fixture()._references_from_content(state, f"[引用]({OTHER})"),
                         [{"id": 1, "title": "第二报告", "source": "第二报告", "url": OTHER}])

    def test_assembly_includes_summary_conclusion_and_outlook_references(self):
        state = state_fixture()
        state["raw_sources"].append({"id": "s2", "url": OTHER, "title": "第二报告"})
        state["draft_sections"] = {"sec_1": "章节不含外部事实。"}
        report = writer_fixture()._assemble_report(
            state, f"摘要[引用]({OTHER})", [{"text": f"结论[引用]({URL})"}],
            f"展望[重复]({OTHER})",
        )
        self.assertEqual([r["url"] for r in state["references"]], [OTHER, URL])
        self.assertIn("## 参考文献", report)
        self.assertEqual(len(state["references"]), 2)


class SynthesisTests(unittest.IsolatedAsyncioTestCase):
    async def test_section_uses_actual_link_not_model_citation_directory(self):
        state = state_fixture()
        state["raw_sources"].append({"id": "s2", "url": OTHER, "title": "未引用报告"})
        writer = writer_fixture()
        writer.call_llm = AsyncMock(return_value=json.dumps({
            "content": f"项目事实[引用]({URL})", "key_points": [], "claims": [],
            "citations": [{"source": "模型误报来源", "url": OTHER}],
        }))
        writer.parse_json_response = json.loads
        await writer._write_section(state, state["outline"][0])
        self.assertEqual([r["url"] for r in state["references"]], [URL])

    async def test_revision_removes_old_bibliography_entry(self):
        state = state_fixture()
        state["raw_sources"].append({"id": "s2", "url": OTHER, "title": "新版来源"})
        state["draft_sections"] = {"sec_1": f"旧引用[报告]({URL})" + "原文" * 300}
        state["references"] = [{"url": URL, "title": "旧来源"}]
        state["critic_feedback"] = [{"issue_id": "issue1", "resolved_section_ids": ["sec_1"], "target": {"chapter_id": "sec_1"}, "description": "修订本章", "evidence": "核对正文", "modification": "按已有材料修订本章", "requires_new_search": False}]
        writer = writer_fixture()
        writer.call_llm = AsyncMock(side_effect=[json.dumps({
            "revised_content": f"新引用[报告]({OTHER})" + "修订后的章节。" * 100,
            "claims": [], "addressed_issues": ["issue1"], "unable_to_address": [],
        }), json.dumps({"executive_summary": "摘要", "conclusions": [], "outlook": "展望", "claims": []})])
        writer.parse_json_response = json.loads
        await writer._revise_report(state)
        self.assertEqual([r["url"] for r in state["references"]], [OTHER])
        self.assertNotIn(f"]({URL})", state["final_report"])

    async def test_model_reference_array_is_ignored_and_stale_reference_disappears(self):
        state = state_fixture()
        state["raw_sources"].append({"id": "s2", "url": OTHER, "title": "未引用报告"})
        state["references"] = [{"title": "旧引用", "url": OTHER}]
        state["draft_sections"] = {"sec_1": f"事实[引用]({URL})"}
        writer = writer_fixture()
        writer.call_llm = AsyncMock(return_value=json.dumps({
            "executive_summary": "摘要", "conclusions": [], "outlook": "展望", "claims": [],
            "references": [{"title": "模型伪造标题", "url": OTHER}],
        }))
        writer.parse_json_response = json.loads
        await writer._synthesize_report(state)
        self.assertEqual([r["url"] for r in state["references"]], [URL])
        prompt = writer.call_llm.call_args.kwargs["user_prompt"]
        self.assertNotIn('"references":', prompt)
        self.assertNotIn("original_excerpt", prompt)
        self.assertNotIn("收集的所有引用来源", prompt)
        self.assertEqual(state["citation_verification"]["status"], "not_executed")


if __name__ == "__main__":
    unittest.main()
