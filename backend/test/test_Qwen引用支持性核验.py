"""Qwen核验的请求、预算、协议及生产入口回归；API返回使用桩。"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from service.context_policy import ContextPolicy, NLISettings, runtime_nli_settings
from service.nli_verifier import (QwenNLI, NLIResponseError, RetiredNLIBackendError,
                                  get_nli, verify_claims, verify_claims_sync, execution_error, reference_windows)
from service.summary_message_verification import verify_message_claims_sync
from service.deep_research_v2.citation_verification import verify_citations


class Counter:
    def count_messages(self, messages):
        return sum(len(m["content"]) for m in messages) + 10


def response(items, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
        content=json.dumps({"items": items}, ensure_ascii=False)))], model="qwen3-max", id="test-call",
        usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}))


def judgment(i, label="entailment"):
    scores = dict.fromkeys(("entailment", "neutral", "contradiction"), .01)
    scores[label] = .98
    return {"id": i, "scores": scores, "reason": "证据对应完整陈述。"}


class QwenNLITests(unittest.TestCase):
    def setUp(self):
        self.counter_patch = patch("service.qwen_token_counter.get_qwen_token_counter", return_value=Counter())
        self.counter_patch.start()
        self.addCleanup(self.counter_patch.stop)
        self.env_patch = patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only", "LLM_BASE_URL": "https://example.test/v1"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.client = Mock()
        self.client.__enter__ = Mock(return_value=self.client)
        self.client.__exit__ = Mock(return_value=False)
        self.create = self.client.chat.completions.create
        self.create.side_effect = lambda **kw: response([
            judgment(i) for i, _ in enumerate(json.loads(kw["messages"][1]["content"])["items"])])
        self.factory_patch = patch("service.llm_client.create_openai_client", return_value=self.client)
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)

    def test_request_uses_qwen_json_and_disables_thinking_without_local_weights(self):
        with patch("service.nli_verifier.get_local_nli") as local:
            report = verify_claims_sync([{"claim": "完整陈述。", "references": [{"source_id": "s", "quote": "证据" * 1200}]}], NLISettings())
        local.assert_not_called()
        request = self.create.call_args.kwargs
        self.assertEqual(request["model"], "qwen3-max")
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(len(request["messages"]), 2)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["window_count"], 1)
        self.assertGreater(report["claims"][0]["evidence"][0]["input_tokens"], 512)
        self.assertEqual(report["device"], "remote_api")
        self.assertEqual(report["api_calls"][0]["usage"]["total_tokens"], 130)
        self.assertEqual(report["confidence_method"], "min_pair_max_llm_self_reported_v1")
        self.assertEqual(report["confidence_calibration"], "not_calibrated")

    def test_batch_limit_applies_to_whole_request_including_prompt(self):
        model = QwenNLI()
        pairs = [("资料" * 100, "论断") for _ in range(3)]
        one = model.count_pairs(pairs[:1])
        two = model.count_pairs(pairs[:2])
        settings = NLISettings(max_length=(one + two) // 2)
        results = model.predict(pairs, settings)
        self.assertEqual(len(results), 3)
        self.assertEqual(self.create.call_count, 3)
        for call in self.create.call_args_list:
            self.assertLessEqual(Counter().count_messages(call.kwargs["messages"]), settings.max_length)

    def test_oversize_pair_rejected_without_api_or_truncation(self):
        model = QwenNLI()
        result = model.predict([("长文" * 500, "论断")], NLISettings(max_length=800))
        self.assertEqual(result[0]["status"], "input_too_long")
        self.create.assert_not_called()
        self.factory.assert_not_called()

    def test_qualifiers_in_tail_are_sent_intact(self):
        text = "背景。" * 1500 + "仅男性亚组有效，其他人群无效。"
        report = verify_claims_sync([{"claim": "对所有人有效。", "references": [{"source_id": "s", "quote": text}]}], NLISettings())
        payload = json.loads(self.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["items"][0]["premise"], text)
        self.assertEqual(report["window_count"], 1)

    def test_response_ids_are_reordered_and_checked(self):
        rows = QwenNLI.parse_response(json.dumps({"items": [judgment(1, "neutral"), judgment(0)]}), 2)
        self.assertEqual(rows[0]["scores"]["entailment"], .98)
        self.assertEqual(rows[1]["scores"]["neutral"], .98)
        for items in ([judgment(0), judgment(0)], [judgment(0)], [judgment(0), judgment(2)]):
            with self.subTest(items=items), self.assertRaises(NLIResponseError):
                QwenNLI.parse_response(json.dumps({"items": items}), 2)

    def test_malformed_scores_and_missing_reason_are_rejected(self):
        for scores in ({"entailment": True, "neutral": 0, "contradiction": 0},
                       {"entailment": .4, "neutral": .4, "contradiction": .4},
                       {"entailment": float("nan"), "neutral": 0, "contradiction": 0}):
            with self.subTest(scores=scores), self.assertRaises(NLIResponseError):
                QwenNLI.parse_response(json.dumps({"items": [{**judgment(0), "scores": scores}]}), 1)
        with self.assertRaises(NLIResponseError):
            QwenNLI.parse_response('{"items":[{"id":0,"scores":{}}]}', 1)

    def test_incomplete_output_fails_instead_of_accepting_valid_json(self):
        self.create.side_effect = None
        self.create.return_value = response([judgment(0)], finish="length")
        with self.assertRaises(NLIResponseError):
            QwenNLI().predict([("证据", "论断")], NLISettings())

    def test_missing_evidence_skips_api_and_is_not_supported(self):
        report = verify_claims_sync([{"claim": "事实。", "references": []}], NLISettings())
        self.assertEqual(report["claims"][0]["status"], "missing_evidence")
        self.create.assert_not_called()

    def test_summary_message_entry_uses_qwen_and_preserves_offsets(self):
        text = "完整原文。" * 150
        context = {"sources": {"m": [{"start": 2, "text": text}]},
                   "rows": {"m": {"kind": "llm_output", "payload": {"content": "前文" + text + "后文"}}}}
        report = verify_message_claims_sync([{"section": "SUMMARY", "claim": "原文结论。", "source_ids": ["m"]}], context, NLISettings())
        ref = report["claims_with_program_quotes"][0]["references"][0]
        self.assertEqual(ref["start"], 2)
        self.assertEqual(ref["quote"], text)
        self.assertEqual(report["model_id"], "qwen3-max")
        self.assertEqual(report["window_count"], 1)

    def test_retired_model_is_never_loaded(self):
        settings = NLISettings(model_id="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", max_length=512)
        with patch("service.nli_verifier.get_local_nli") as local, self.assertRaises(RetiredNLIBackendError):
            get_nli(settings)
        local.assert_not_called()
        self.assertEqual(execution_error(RetiredNLIBackendError()), "model_retired")

    def test_real_old_preset_snapshot_keeps_its_hash(self):
        folder = Path(__file__).resolve().parents[1] / "eval/configs/上下文压缩消融"
        old = next(json.loads(p.read_text()) for p in folder.glob("*.json")
                   if "mDeBERTa" in p.read_text())
        config = old.get("config", old)
        digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        snapshot = {"config": config, "config_hash": digest}
        restored = ContextPolicy.from_snapshot(snapshot)
        self.assertEqual(restored.snapshot(), snapshot)
        self.assertEqual(restored.nli.max_length, 512)

    def test_current_runtime_budgets_are_frozen(self):
        with patch.dict(os.environ, {"AGENT_NLI_MAX_INPUT_TOKENS": "80000", "AGENT_NLI_MAX_OUTPUT_TOKENS": "6000"}):
            settings = runtime_nli_settings()
        self.assertEqual(settings.model_id, "qwen3-max")
        self.assertEqual(settings.max_length, 80000)
        self.assertEqual(settings.output_tokens, 6000)
        policy = ContextPolicy(group="Qwen配置核查", nli=settings)
        self.assertEqual(ContextPolicy.from_snapshot(policy.snapshot()).nli, settings)

    def test_context_reserves_output_and_safety(self):
        settings = NLISettings(max_length=258048, output_tokens=60000)
        self.assertEqual(QwenNLI.input_limit(settings), 262144 - 60000 - 1024)

    def test_api_timeout_is_not_a_successful_verification(self):
        self.create.side_effect = TimeoutError("模拟超时")
        result = asyncio.run(verify_citations([{"claim": "陈述。", "references": [{"source_id": "s", "quote": "原文。"}]}], enabled=True))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["execution_status"], "timeout")
        self.assertFalse(result["declared_claims_checked"])

    def test_writer_invalid_binding_stays_unverified(self):
        result = asyncio.run(verify_citations([{"claim": "陈述。", "references": [],
                                               "binding_errors": ["unknown_or_ambiguous_document_id"]}], enabled=True))
        self.assertEqual(result["claims"][0]["status"], "invalid_citation")
        self.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
