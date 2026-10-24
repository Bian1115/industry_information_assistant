"""限定网站检索范围与研究终态幂等交付的测试。

不发真实搜索请求、不连数据库：搜索结果与数据库会话都用假实现。
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

APP_DIR = Path(__file__).resolve().parents[2] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from service.deep_research_v2.agents.scout import DeepScout  # noqa: E402
from service.deep_research_v2.delivery import (  # noqa: E402
    ResearchDeliveryError,
    build_complete_event,
    checkpoint_id_of,
)
from service.deep_research_v2.quality_pipeline import evaluate_report_gate  # noqa: E402
from service.deep_research_v2.service import DeepResearchV2Service  # noqa: E402
from service.deep_research_v2.state import create_initial_state  # noqa: E402
from service.deep_research_v2.followup_context import inherit_artifacts  # noqa: E402
from service.deep_research_v2.source_scope import (  # noqa: E402
    SourceScopeError,
    belongs_to,
    configured_allowed_domains,
    normalize_allowed_domains,
    normalize_domain,
    strip_site_filters,
    url_in_scope,
    filter_results_in_scope,
)
from service.web_reader_service import UnsafeURL, WebReaderService  # noqa: E402


def make_scout(recorded):
    """构造只替换搜索实现的 Scout，不初始化模型客户端。"""
    scout = DeepScout.__new__(DeepScout)
    scout.name = "DeepScout"
    import logging

    scout.logger = logging.getLogger("test.scout")

    async def fake_once(query, count=10, *, state=None):
        recorded.append((query, count))
        return FAKE_RESULTS.get(query, [])

    scout._execute_search_once = fake_once
    return scout


FAKE_RESULTS = {
    "阿尔茨海默 治疗": [
        {"url": "https://www.who.int/a", "title": "WHO A"},
        {"url": "https://nih.gov/c", "title": "NIH C"},
        {"url": "https://www.nhs.uk/a", "title": "NHS A"},
        {"url": "https://www.nhs.uk/a", "title": "重复 NHS"},
        {"url": "ftp://www.nhs.uk/a", "title": "非网页协议"},
    ],
    "site:who.int 阿尔茨海默 治疗": [
        {"url": "https://www.who.int/a", "title": "WHO A"},
        {"url": "https://evil.com/who.int", "title": "域外仿冒"},
        {"url": "https://apps.who.int/b", "title": "WHO 子域"},
    ],
    "site:nih.gov 阿尔茨海默 治疗": [
        {"url": "https://nih.gov/c", "title": "NIH C"},
        {"url": "https://www.who.int/a", "title": "跨站重复"},
    ],
}


class SourceScopeRules(unittest.TestCase):
    def test_domain_normalization_rejects_unusable_values(self):
        self.assertEqual(normalize_domain("https://WHO.int/path?q=1"), "who.int")
        for bad in ["", "*.who.int", "localhost", "who", "http://10.0.0.1/x", "www.gov.cn"]:
            with self.assertRaises(SourceScopeError, msg=bad):
                normalize_domain(bad)

    def test_subdomains_match_but_lookalikes_do_not(self):
        self.assertTrue(belongs_to("https://apps.who.int/x", "who.int"))
        self.assertFalse(belongs_to("https://evilwho.int/x", "who.int"))
        self.assertFalse(belongs_to("https://who.int.evil.com/x", "who.int"))
        self.assertFalse(belongs_to("ftp://who.int/x", "who.int"))

    def test_model_supplied_site_filters_are_stripped(self):
        self.assertEqual(
            strip_site_filters("site:a.com -site:b.com 阿尔茨海默 治疗"),
            "阿尔茨海默 治疗",
        )

    def test_empty_or_unset_configuration_means_whole_web(self):
        with patch.dict(os.environ, {"RESEARCH_ALLOWED_DOMAINS": ""}, clear=False):
            self.assertEqual(configured_allowed_domains(), [])
            os.environ.pop("RESEARCH_ALLOWED_DOMAINS")
            self.assertEqual(configured_allowed_domains(), [])

    def test_whole_web_scope_accepts_other_sites_but_rejects_invalid_urls(self):
        self.assertTrue(url_in_scope("https://www.nhs.uk/a", []))
        for url in ["", "ftp://www.nhs.uk/a", "javascript:alert(1)", "https://", "https://[bad"]:
            self.assertFalse(url_in_scope(url, []), url)
        filtered = filter_results_in_scope(FAKE_RESULTS["阿尔茨海默 治疗"], [])
        self.assertEqual(len(filtered), 3)

    def test_initial_state_records_whole_web_scope(self):
        state = create_initial_state("问题", "session", [])
        self.assertEqual(state["allowed_domains"], [])

    def test_service_allows_request_scope_when_deployment_is_unrestricted(self):
        service = DeepResearchV2Service.__new__(DeepResearchV2Service)
        with patch.dict(os.environ, {"RESEARCH_ALLOWED_DOMAINS": ""}, clear=False):
            self.assertEqual(service.resolve_allowed_domains(), [])
            self.assertEqual(service.resolve_allowed_domains(["nhs.uk"]), ["nhs.uk"])
        with patch.dict(os.environ, {"RESEARCH_ALLOWED_DOMAINS": "who.int,nih.gov"}, clear=False):
            self.assertEqual(service.resolve_allowed_domains(), ["who.int", "nih.gov"])
            self.assertEqual(service.resolve_allowed_domains(["nih.gov"]), ["nih.gov"])
            with self.assertRaises(ValueError):
                service.resolve_allowed_domains(["nhs.uk"])

    def test_followup_keeps_other_sites_in_whole_web_mode(self):
        state = create_initial_state("追问", "session", [])
        inherit_artifacts(state, {
            "report_hash": "parent",
            "raw_sources": [{"id": "s1", "url": "https://www.nhs.uk/a", "content": "原文"}],
            "facts": [{"id": "f1", "source_id": "s1"}],
            "references": [{"url": "https://www.nhs.uk/a"}],
        })
        self.assertEqual(len(state["raw_sources"]), 1)
        self.assertEqual(len(state["facts"]), 1)
        self.assertEqual(len(state["references"]), 1)


class ScopedSearchEntry(unittest.IsolatedAsyncioTestCase):
    async def test_one_request_per_domain_with_site_prefix(self):
        recorded = []
        scout = make_scout(recorded)
        state = {"allowed_domains": ["who.int", "nih.gov"], "messages": []}
        results = await scout._execute_search("阿尔茨海默 治疗", count=10, state=state)
        self.assertEqual(
            [query for query, _ in recorded],
            ["site:who.int 阿尔茨海默 治疗", "site:nih.gov 阿尔茨海默 治疗"],
        )
        # count 是每个网站的请求条数，不是合并后的总条数。
        self.assertEqual({count for _, count in recorded}, {10})
        urls = sorted(item["url"] for item in results)
        self.assertEqual(
            urls, ["https://apps.who.int/b", "https://nih.gov/c", "https://www.who.int/a"]
        )

    async def test_out_of_scope_results_are_dropped_not_backfilled(self):
        scout = make_scout([])
        state = {"allowed_domains": ["who.int", "nih.gov"], "messages": []}
        results = await scout._execute_search("阿尔茨海默 治疗", state=state)
        self.assertNotIn(
            "https://evil.com/who.int", [item["url"] for item in results]
        )

    async def test_model_site_filter_does_not_widen_the_scope(self):
        recorded = []
        scout = make_scout(recorded)
        state = {"allowed_domains": ["who.int"], "messages": []}
        await scout._execute_search("site:evil.com 阿尔茨海默 治疗", state=state)
        self.assertEqual([q for q, _ in recorded], ["site:who.int 阿尔茨海默 治疗"])

    async def test_empty_scope_searches_whole_web_without_site_prefix(self):
        recorded = []
        scout = make_scout(recorded)
        results = await scout._execute_search(
            "site:who.int 阿尔茨海默 治疗", state={"allowed_domains": [], "messages": []}
        )
        self.assertEqual(recorded, [("阿尔茨海默 治疗", 10)])
        self.assertEqual(len(results), 3)
        self.assertIn("https://www.nhs.uk/a", [r["url"] for r in results])

    async def test_unrecorded_scope_searches_whole_web(self):
        recorded = []
        results = await make_scout(recorded)._execute_search("阿尔茨海默 治疗", state={})
        self.assertEqual(recorded, [("阿尔茨海默 治疗", 10)])
        self.assertEqual(len(results), 3)

    async def test_deep_read_empty_scope_overrides_previous_reader_restriction(self):
        scout = make_scout([])
        scout.web_reader = SimpleNamespace(
            allowed_domains=["who.int"],
            read_url=AsyncMock(return_value=SimpleNamespace(
                success=True, title="NHS", content="来源正文"
            )),
        )
        scout.call_llm = AsyncMock(return_value='{"facts": []}')
        result = await scout.deep_read_url(
            "https://www.nhs.uk/a", "NHS", "问题", state={"allowed_domains": []}
        )
        self.assertEqual(result, {"facts": []})
        self.assertEqual(scout.web_reader.allowed_domains, [])
        scout.web_reader.read_url.assert_awaited_once_with("https://www.nhs.uk/a", "NHS")

    async def test_query_without_keywords_is_rejected(self):
        scout = make_scout([])
        with self.assertRaises(ValueError):
            await scout._execute_search(
                "site:who.int", state={"allowed_domains": ["who.int"], "messages": []}
            )


class ReaderScopeGuard(unittest.IsolatedAsyncioTestCase):
    async def test_every_hop_is_checked_including_redirect_targets(self):
        reader = WebReaderService.__new__(WebReaderService)
        reader.allowed_domains = ["who.int"]
        self.assertEqual(
            await WebReaderService.validate_url(reader, "https://who.int/a"),
            "https://who.int/a",
        )
        with self.assertRaises(UnsafeURL):
            await WebReaderService.validate_url(reader, "https://elsewhere.com/a")

    async def test_no_scope_means_no_extra_restriction(self):
        reader = WebReaderService.__new__(WebReaderService)
        reader.allowed_domains = None
        self.assertTrue(reader._in_allowed_scope("https://anything.example/a"))

    async def test_empty_scope_keeps_private_address_guard(self):
        reader = WebReaderService.__new__(WebReaderService)
        reader.allowed_domains = []
        self.assertTrue(reader._in_allowed_scope("https://www.nhs.uk/a"))
        for url in ["http://127.0.0.1/a", "http://10.0.0.1/a", "http://localhost/a"]:
            with self.assertRaises(UnsafeURL, msg=url):
                await reader.validate_url(url)


class ReportSourceScopeGate(unittest.TestCase):
    def test_whole_web_report_can_cite_other_sites(self):
        gate = evaluate_report_gate({
            "final_report": "见 [NHS](https://www.nhs.uk/a) 与 [WHO](https://who.int/a)",
            "allowed_domains": [],
        })
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["citations_in_scope"], 2)
        self.assertEqual(gate["citation_verification"], "not_executed")
        empty = evaluate_report_gate({"final_report": "", "allowed_domains": []})
        self.assertFalse(empty["passed"])
        self.assertEqual(empty["blocking_reasons"], ["报告为空"])

    def test_report_gate_blocks_out_of_scope_links_and_empty_report(self):
        gate = evaluate_report_gate({
            "final_report": "见 [来源](https://who.int/a) 与 [外站](https://evil.com/b)",
            "allowed_domains": ["who.int"],
        })
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["out_of_scope_citations"], ["https://evil.com/b"])
        # 门禁只回答非空与来源范围，不声称做过引用核验。
        self.assertEqual(gate["citation_verification"], "not_executed")

        empty = evaluate_report_gate({"final_report": "", "allowed_domains": ["who.int"]})
        self.assertFalse(empty["passed"])
        self.assertIn("报告为空", empty["blocking_reasons"])

    def test_report_gate_passes_when_all_links_are_in_scope(self):
        gate = evaluate_report_gate({
            "final_report": "见 [来源](https://apps.who.int/a)",
            "allowed_domains": ["who.int"],
        })
        self.assertTrue(gate["passed"])


class TerminalDeliveryPayload(unittest.TestCase):
    def test_checkpoint_id_is_required(self):
        class Snapshot:
            config = {"configurable": {"thread_id": "run-1"}}

        with self.assertRaises(ResearchDeliveryError):
            checkpoint_id_of(Snapshot())

    def test_complete_event_separates_execution_quality_and_verification(self):
        payload = build_complete_event(
            {
                "final_report": "正文",
                "status": "completed",
                "completion_reason": "revision_limit",
                "quality_verdict": "major_issues",
                "references": [{"url": "https://who.int/a", "source": "WHO"}],
                "facts": [{"source_url": "https://who.int/a", "content": "内容"}],
            },
            run_id="run-1",
            checkpoint_id="cp-1",
        )
        self.assertEqual(payload["completion_reason"], "revision_limit")
        self.assertEqual(payload["quality_verdict"], "major_issues")
        self.assertEqual(payload["citation_verification"]["status"], "not_executed")
        self.assertFalse(payload["citation_verification"]["all_claims_checked"])
        self.assertEqual(payload["references"][0]["link"], "https://who.int/a")


if __name__ == "__main__":
    unittest.main()
