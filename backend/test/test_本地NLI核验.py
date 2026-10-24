"""NLI门槛、预算和保存链路回归；此文件不加载真实模型。"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid
from unittest.mock import AsyncMock, Mock, patch

from test_智能体独立历史 import MemoryStore, CharCounter, HistoryScope, make_agent, bind_agent_history
from test_摘要合并与引用分离 import clause, draft, citation
from service.agent_context import prepare_agent_messages, ContextBudgetExceeded
from service.context_policy import ContextPolicy, NLISettings, legacy_policy, policy_for_scope
from service.nli_verifier import aggregate_claims, LocalNLI, verify_claims, verify_claims_sync


def scored(e, n, c):
    return {"status": "scored", "input_tokens": 20, "scores": {"entailment": e, "neutral": n, "contradiction": c}}


def claim(text="结论。", count=1):
    return {"section": "SUMMARY", "claim": text, "references": [{"source_id": f"s{i}", "quote": "原句。"} for i in range(count)]}


class NLIUnitTests(unittest.TestCase):
    def test_confidence_is_label_probability_not_entailment(self):
        row = aggregate_claims([claim()], [scored(.03, .95, .02)], NLISettings())["claims"][0]
        self.assertEqual(row["confidence"], .95)
        self.assertEqual(row["evidence"][0]["predicted_label"], "neutral")
        self.assertEqual(row["status"], "unknown")
        self.assertTrue(row["needs_review"])
        self.assertNotIn("low_confidence", row["review_reasons"])

    def test_weak_reference_is_not_hidden_by_strong_support(self):
        report = aggregate_claims([claim(count=2)], [scored(.99,.005,.005), scored(.4,.35,.25)], NLISettings())
        row = report["claims"][0]
        self.assertEqual(report["status"], "passed")  # 复核标记不改变原有放行策略。
        self.assertEqual(row["confidence"], .4)
        self.assertEqual(report["review_claim_indexes"], [0])
        self.assertEqual(report["review_count"], 1)
        self.assertIn("low_confidence", row["review_reasons"])

    def test_review_threshold_boundary_and_gate_independence(self):
        for threshold, review in [(.9, False), (.91, True)]:
            report = aggregate_claims([claim()], [scored(.9,.08,.02)], NLISettings(review_confidence_threshold=threshold))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["needs_review"], review)

    def test_incomplete_and_missing_evidence_have_null_confidence(self):
        for item, predictions in [(claim(count=2), [scored(.99,.005,.005), {"status":"input_too_long"}]),
                                  (claim(count=0), []), (claim("建议：继续检索。", 0), [])]:
            row = aggregate_claims([item], predictions, NLISettings())["claims"][0]
            self.assertIsNone(row["confidence"])
            self.assertTrue(row["needs_review"])

    def test_invalid_scores_cannot_be_used_as_confidence(self):
        for pred in [scored(float("nan"), .1, .1), scored(float("inf"), 0, 0),
                     scored(1.1, -.1, 0), scored(.4, .4, .4)]:
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                aggregate_claims([claim()], [pred], NLISettings())

    def test_old_nli_snapshot_keeps_hash_and_ignores_new_environment(self):
        config = ContextPolicy(group="旧NLI", summary_strategy="split-citations-v4", nli=NLISettings()).config_payload()
        config["nli"].pop("review_confidence_threshold")
        digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with patch.dict(os.environ, {"AGENT_SUMMARY_NLI_REVIEW_CONFIDENCE":"0.99"}):
            restored = ContextPolicy.from_snapshot({"config":config, "config_hash":digest})
        self.assertEqual(restored.snapshot(), {"config":config, "config_hash":digest})
        self.assertEqual(restored.nli.review_confidence_threshold, .8)

    def test_review_setting_is_frozen_validated_and_hashed(self):
        with patch.dict(os.environ, {"AGENT_SUMMARY_NLI_ENABLED":"1", "AGENT_HISTORY_SUMMARY_ENABLED":"1",
                                    "AGENT_SUMMARY_STRATEGY":"split-citations-v4", "AGENT_SUMMARY_NLI_REVIEW_CONFIDENCE":"0.92"}):
            policy = legacy_policy()
        restored = ContextPolicy.from_snapshot(policy.snapshot())
        self.assertEqual(restored.nli.review_confidence_threshold, .92)
        changed = policy.model_copy(update={"nli": policy.nli.model_copy(update={"review_confidence_threshold":.8})})
        self.assertNotEqual(changed.fingerprint, policy.fingerprint)
        for invalid in [-.1, 1.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                NLISettings(review_confidence_threshold=invalid)

    def test_three_classes_and_thresholds(self):
        for pred, expected in [(scored(.9,.08,.02), "supported"), (scored(.1,.1,.8), "contradiction"),
                               (scored(.2,.7,.1), "unknown"), (scored(.84,.15,.01), "unknown"),
                               (scored(.85,.14,.01), "supported")]:
            with self.subTest(expected=expected):
                result = aggregate_claims([claim()], [pred], NLISettings())
                self.assertEqual(result["claims"][0]["status"], expected)
                self.assertEqual(result["status"] == "passed", expected == "supported")

    def test_contradiction_overrides_support_from_another_quote(self):
        result = aggregate_claims([claim(count=2)], [scored(.99,.005,.005), scored(.1,.1,.8)], NLISettings())
        self.assertEqual(result["claims"][0]["status"], "contradiction")

    def test_long_unchecked_reference_cannot_be_hidden_by_supported_one(self):
        result = aggregate_claims([claim(count=2)], [scored(.99,.005,.005), {"status":"input_too_long","input_tokens":800}], NLISettings())
        self.assertEqual(result["status"], "rejected")

    def test_no_evidence_only_explicit_nonfact_is_permitted(self):
        result = aggregate_claims([claim("建议：继续检索。", 0)], [], NLISettings())
        self.assertEqual(result["unverified_count"], 1)
        self.assertEqual(result["supported_count"], 0)
        self.assertEqual(aggregate_claims([claim("已证实结论。", 0)], [], NLISettings())["status"], "rejected")

    def test_too_many_pairs_fails_before_loading_model(self):
        with patch("service.nli_verifier.get_nli") as load, self.assertRaisesRegex(ValueError, "数量"):
            verify_claims_sync([claim(count=2)], NLISettings(max_pairs=1))
        load.assert_not_called()

    def test_missing_snapshot_is_not_downloaded_implicitly(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(FileNotFoundError):
            LocalNLI(folder, NLISettings().revision)

    def test_snapshot_wrong_revision_fails_before_model_import(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"下载清单.json").write_text(json.dumps({"model_id":NLISettings().model_id,"revision":"wrong"}))
            with self.assertRaisesRegex(ValueError,"版本"):
                LocalNLI(folder,NLISettings().revision)

    def test_oversize_pair_is_not_silently_truncated_or_sent_to_model(self):
        class Tokenizer:
            def __call__(self, premise, hypothesis, **kwargs):
                assert kwargs == {"truncation": False}
                return {"input_ids": list(range(513))}
        obj=LocalNLI.__new__(LocalNLI)
        obj.tokenizer=Tokenizer()
        obj.model=Mock()
        obj.model.config.max_position_embeddings=512
        result=obj.predict([("长原文","分句")],NLISettings())
        self.assertEqual(result[0]["status"],"input_too_long")
        obj.model.assert_not_called()

    def test_old_snapshot_hash_survives_new_optional_settings(self):
        policy=ContextPolicy(group="旧运行", semantic_verification=False, summary_strategy="split-citations-v4")
        old_config=policy.model_dump()
        old_config.pop("nli")
        old_config.pop("verification", None)
        old_hash=hashlib.sha256(json.dumps(old_config,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        restored=ContextPolicy.from_snapshot({"config":old_config,"config_hash":old_hash})
        self.assertEqual(restored.fingerprint,old_hash)
        self.assertIsNone(restored.nli)

    def test_thresholds_are_frozen_and_affect_config_hash(self):
        policy=ContextPolicy(group="NLI", summary_strategy="split-citations-v4", nli=NLISettings())
        policy.require_supported()
        scope=HistoryScope(None,"r","s",None,context_policy=policy.snapshot())
        with patch.dict(os.environ,{"AGENT_SUMMARY_NLI_ENABLED":"0","AGENT_SUMMARY_NLI_ENTAILMENT":"0.95"}):
            restored=policy_for_scope(scope)
        self.assertEqual(restored.nli.entailment_threshold,.85)
        changed=policy.model_copy(update={"nli":NLISettings(entailment_threshold=.95)})
        self.assertNotEqual(changed.fingerprint,policy.fingerprint)

    def test_new_default_enables_nli_without_changing_explicit_legacy_strategy(self):
        with patch.dict(os.environ,{"AGENT_SUMMARY_NLI_ENABLED":"1","AGENT_HISTORY_SUMMARY_ENABLED":"1","AGENT_SUMMARY_STRATEGY":"split-citations-v4"}):
            self.assertTrue(legacy_policy().semantic_verification)
        with patch.dict(os.environ,{"AGENT_SUMMARY_NLI_ENABLED":"1","AGENT_SUMMARY_STRATEGY":"incremental-v3"}):
            self.assertFalse(legacy_policy().semantic_verification)


class NLIGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store=MemoryStore()
        self.scope=HistoryScope(self.store,"run","session",None)
        p=patch.dict(os.environ,{"AGENT_SUMMARY_NLI_ENABLED":"1","AGENT_HISTORY_SUMMARY_ENABLED":"1",
                                "AGENT_SUMMARY_STRATEGY":"split-citations-v4", "AGENT_HISTORY_REPLAY":"DeepScout",
                                "AGENT_INPUT_HISTORY_ENABLED":"0", "CONTEXT_COMPRESSION_ENABLED":"1"})
        p.start();self.addCleanup(p.stop)

    async def execute(self, status="passed", *, error=None):
        row=await self.store.append(self.scope,"DeepScout","llm_output",{"content":"原句。"+"原文"*3000},str(uuid.uuid4()))
        item=clause("结论。",row["source_id"])
        agent=make_agent()
        agent._call_llm_impl=AsyncMock(side_effect=[draft(item),citation(item,[(row["source_id"],"原句。")])])
        report = aggregate_claims([claim()], [scored(.9,.08,.02) if status=="passed" else scored(.4,.35,.25)], NLISettings())
        report.update(model_id=NLISettings().model_id, revision=NLISettings().revision)
        with patch("service.nli_verifier.verify_claims",new=AsyncMock(return_value=report,side_effect=error)) as verify, bind_agent_history(self.scope), patch("service.agent_context.get_qwen_token_counter",return_value=CharCounter()), patch("service.agent_context.input_budget",side_effect=lambda out:4000 if out==3000 else 1400):
            if status=="passed" and error is None:
                await prepare_agent_messages(agent,self.scope,"sys","继续","current",6000)
            else:
                with self.assertRaises(ContextBudgetExceeded):
                    await prepare_agent_messages(agent,self.scope,"sys","继续","current",6000)
            verify.assert_awaited_once()
            self.assertEqual(verify.call_args.args[0][0]["references"][0]["quote"],"原句。")
            self.assertEqual(agent._call_llm_impl.await_count,2)
        return await self.store.load(self.scope,"DeepScout")

    async def test_success_is_saved_with_real_gate_metadata(self):
        saved=await self.execute()
        self.assertEqual(saved["summary_version"],1)
        self.assertEqual(saved["summary_metadata"]["validation"]["semantic_support"],"passed")
        self.assertEqual(saved["summary_metadata"]["validation"]["omission_check"],"not_run")
        self.assertTrue(any(r["kind"]=="summary_verification" for r in saved["messages"]))
        self.assertEqual(saved["summary_metadata"]["validation"]["nli"]["claims"][0]["confidence"], .9)

    async def test_neutral_or_contradiction_does_not_advance_cursor(self):
        saved=await self.execute("rejected")
        self.assertEqual((saved["summary_version"],saved["covered_seq"]),(0,0))
        self.assertEqual(next(r["payload"]["status"] for r in saved["messages"] if r["kind"]=="summary_verification"),"rejected")
        report = next(r["payload"] for r in saved["messages"] if r["kind"]=="summary_verification")
        self.assertEqual(report["review_claim_indexes"], [0])
        self.assertIn("low_confidence", report["claims"][0]["review_reasons"])

    async def test_missing_model_does_not_pass_as_verified(self):
        saved=await self.execute(error=FileNotFoundError("缺少NLI模型"))
        self.assertEqual(saved["summary_version"],0)
        self.assertEqual(next(r["payload"]["status"] for r in saved["messages"] if r["kind"]=="summary_verification"),"error")
        report = next(r["payload"] for r in saved["messages"] if r["kind"]=="summary_verification")
        self.assertTrue(report["needs_review"])
        self.assertIsNone(report["claims"][0]["confidence"])
        self.assertEqual(report["claims"][0]["evidence"][0]["quote"], "原句。")

    async def test_inference_timeout_returns_no_success(self):
        def slow(*args):
            time.sleep(.03)
            return {"status":"passed"}
        with patch("service.nli_verifier.verify_claims_sync",side_effect=slow),self.assertRaises(TimeoutError):
            await verify_claims([claim()],NLISettings(timeout_seconds=.005))


if __name__=="__main__":
    unittest.main()
