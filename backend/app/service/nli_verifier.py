"""Qwen三分类NLI：只核验引用对最终分句的支持，不判断医学真实性或遗漏。"""

from __future__ import annotations

import asyncio
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from service.context_policy import NLISettings

WEIGHTS_SHA256 = "65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e"
_INFERENCE_LOCK = threading.Lock()
LEGACY_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


class RetiredNLIBackendError(ValueError):
    """旧冻结配置可读取，但不能再次调用已停用的本地模型。"""


class NLIResponseError(ValueError):
    pass


class NLIVerificationError(ValueError):
    code = "summary_nli_rejected"


class NLIBusyError(RuntimeError):
    pass


def execution_error(exc):
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, NLIBusyError):
        return "resource_busy"
    if isinstance(exc, FileNotFoundError):
        return "model_missing"
    if isinstance(exc, RetiredNLIBackendError):
        return "model_retired"
    if isinstance(exc, NLIResponseError):
        return "invalid_response"
    from openai import APITimeoutError
    if isinstance(exc, APITimeoutError):
        return "timeout"
    return "error"


def pair_fits(model, premise, claim, settings):
    if isinstance(model, QwenNLI):
        return model.count_pairs([(premise, claim)]) <= model.input_limit(settings)
    # 历史离线工具和机制测试仍可使用原模型的分词接口。
    limit = min(settings.max_length, model.model.config.max_position_embeddings)
    return len(model.tokenizer(premise, claim, truncation=False)["input_ids"]) <= limit


def reference_windows(ref, claim, model, settings):
    """完整句段形成有界窗口；过长句子/超窗口上限不截断。旧引句策略保持兼容。"""
    text = ref.get("context", ref["quote"])
    if not ref.get("context_complete", True):
        return [], "context_incomplete"
    fits = lambda s: pair_fits(model, s, claim, settings)
    if fits(text):
        return [(text, 0, len(text))], None
    if "context" not in ref:
        return [], "input_too_long"
    spans = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[^。！？.!?\n]+[。！？.!?\n]*|[。！？.!?\n]+", text)]
    windows = []
    for sentence, start, end in spans:
        if not fits(sentence):
            return [], "input_too_long"
        if windows and fits(windows[-1][0] + sentence):
            previous, first, _ = windows[-1]
            windows[-1] = (previous + sentence, first, end)
        else:
            windows.append((sentence, start, end))
        if len(windows) > ref.get("max_windows", 8):
            return [], "window_limit"
    return windows, None


class LocalNLI:
    """历史离线工具兼容类；生产入口get_nli不会加载此模型。"""
    def __init__(self, directory, revision):
        # 只允许已完整下载并验证的safetensors快照；不在线取权重或执行仓库代码。
        directory = Path(directory)
        manifest = json.loads((directory / "下载清单.json").read_text())
        if manifest.get("model_id") != LEGACY_MODEL_ID or manifest.get("revision") != revision:
            raise ValueError("NLI下载清单版本不匹配")
        if not manifest.get("files"):
            raise ValueError("NLI下载清单缺少文件")
        for entry in manifest["files"]:
            filename = entry["file"]
            if Path(filename).name != filename:
                raise ValueError("NLI下载清单路径不合法")
            path = directory / filename
            with path.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if digest != entry["sha256"] or (filename == "model.safetensors" and digest != WEIGHTS_SHA256):
                raise ValueError("NLI本地文件哈希不匹配")
        if "model.safetensors" not in {e["file"] for e in manifest["files"]}:
            raise ValueError("NLI清单缺少模型权重")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32,
        ).to("cpu").eval()
        self.labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if set(self.labels.values()) != {"entailment", "neutral", "contradiction"}:
            raise ValueError("NLI标签映射异常")

    def predict(self, pairs, settings):
        results = [None] * len(pairs)
        encoded = []
        for i, (premise, hypothesis) in enumerate(pairs):
            # 包含两段文本及特殊token；不允许静默截断尾部条件。
            tokens = self.tokenizer(premise, hypothesis, truncation=False)
            length = len(tokens["input_ids"])
            if length > min(settings.max_length, self.model.config.max_position_embeddings):
                results[i] = {"status": "input_too_long", "input_tokens": length}
            else:
                encoded.append((i, tokens))
        for offset in range(0, len(encoded), settings.batch_size):
            batch = encoded[offset:offset + settings.batch_size]
            tensors = self.tokenizer.pad([t for _, t in batch], padding=True, return_tensors="pt")
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(**tensors).logits.float(), dim=-1).cpu().tolist()
            for (i, tokens), scores in zip(batch, probabilities):
                if not all(math.isfinite(x) for x in scores):
                    raise ValueError("NLI返回非有限分数")
                results[i] = {"status": "scored", "input_tokens": len(tokens["input_ids"]),
                              "scores": {self.labels[k]: float(value) for k, value in enumerate(scores)}}
        return results


@lru_cache(maxsize=1)
def get_local_nli(directory, revision):
    """仅保留显式历史离线复现接口；生产核验不调用此函数。"""
    return LocalNLI(directory, revision)


def get_nli(settings):
    if settings.model_id != "qwen3-max":
        raise RetiredNLIBackendError("mDeBERTa核验后端已停用；请用qwen3-max配置创建新运行")
    return QwenNLI()


class QwenNLI:
    """每次只发送当前证据/论断；按实际消息预算组批，不注入智能体历史。"""

    def __init__(self):
        from service.qwen_token_counter import get_qwen_token_counter
        self.counter = get_qwen_token_counter()
        self.calls = []

    @staticmethod
    def messages(pairs):
        prompt = (Path(__file__).parents[1] / "prompts/Qwen引用支持性核验.txt").read_text()
        return [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(
            {"items": [{"id": i, "premise": premise, "hypothesis": claim}
                       for i, (premise, claim) in enumerate(pairs)]}, ensure_ascii=False)}]

    def count_pairs(self, pairs):
        return self.counter.count_messages(self.messages(pairs))

    @staticmethod
    def input_limit(settings):
        return min(settings.max_length, 258048, 262144 - settings.output_tokens - settings.safety_tokens)

    @staticmethod
    def parse_response(content, count):
        try:
            data = json.loads(content)
            items = data["items"]
            if not isinstance(data, dict) or set(data) != {"items"} or not isinstance(items, list) or len(items) != count:
                raise ValueError("核验条目数量不匹配")
            indexed = {}
            for item in items:
                if (not isinstance(item, dict) or set(item) - {"id", "scores", "reason", "prediction_as_fact"}
                        or not {"id", "scores", "reason"} <= set(item)
                        or type(item["id"]) is not int or not 0 <= item["id"] < count
                        or item["id"] in indexed or not isinstance(item["reason"], str)
                        or not item["reason"].strip() or not isinstance(item["scores"], dict)):
                    raise ValueError("核验编号或字段不合法")
                if "prediction_as_fact" in item and type(item["prediction_as_fact"]) is not bool:
                    raise ValueError("prediction_as_fact必须是布尔值")
                scores = item["scores"]
                if (set(scores) != {"entailment", "neutral", "contradiction"}
                        or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values())
                        or not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-5)):
                    raise ValueError("核验三类分数未归一化或非有限数值")
                indexed[item["id"]] = {"status": "scored", "scores": scores, "reason": item["reason"],
                                       "prediction_as_fact": item.get("prediction_as_fact", False),
                                       "score_method": "llm_self_reported_v1"}
            return [indexed[i] for i in range(count)]
        except (ValueError, TypeError, KeyError) as exc:
            raise NLIResponseError("Qwen NLI返回格式不合法：" + str(exc)) from exc

    def predict(self, pairs, settings):
        results = [None] * len(pairs)
        eligible = []
        for i, pair in enumerate(pairs):
            length = self.count_pairs([pair])
            if length > self.input_limit(settings):
                results[i] = {"status": "input_too_long", "input_tokens": length}
            else:
                eligible.append((i, pair, length))
        if not eligible:
            return results
        from config.llm_config import LLMConfig
        from service.llm_client import create_openai_client
        config = LLMConfig()
        if not config.api_key:
            raise ValueError("Qwen NLI缺少DASHSCOPE_API_KEY")
        deadline = time.monotonic() + settings.timeout_seconds
        with create_openai_client(api_key=config.api_key, base_url=config.base_url,
                                  read_timeout=settings.timeout_seconds, max_retries=0) as client:
            cursor = 0
            while cursor < len(eligible):
                batch = []
                while cursor < len(eligible) and len(batch) < settings.batch_size:
                    candidate = eligible[cursor]
                    if batch and self.count_pairs([e[1] for e in batch] + [candidate[1]]) > self.input_limit(settings):
                        break
                    batch.append(candidate)
                    cursor += 1
                messages = self.messages([e[1] for e in batch])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Qwen NLI核验超过总时间预算")
                response = client.chat.completions.create(
                    model="qwen3-max", messages=messages, temperature=0,
                    max_tokens=settings.output_tokens, response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False}, timeout=remaining,
                )
                choice = response.choices[0]
                if choice.finish_reason != "stop":
                    raise NLIResponseError("Qwen NLI输出未完整结束：" + str(choice.finish_reason))
                predictions = self.parse_response(choice.message.content, len(batch))
                call_index = len(self.calls)
                self.calls.append({"model": "qwen3-max", "response_model": response.model,
                                   "request_id": response.id, "input_sha256": hashlib.sha256(
                                       json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                                   "estimated_input_tokens": self.counter.count_messages(messages),
                                   "usage": response.usage.model_dump() if response.usage else None,
                                   "finish_reason": choice.finish_reason, "pair_count": len(batch)})
                for (i, _, length), prediction in zip(batch, predictions):
                    results[i] = {**prediction, "input_tokens": length, "call_index": call_index}
        return results


def backend_metadata(model, settings):
    return {"model_id": settings.model_id, "revision": settings.revision,
            "settings": settings.model_dump(), "device": "remote_api" if settings.model_id == "qwen3-max" else "cpu",
            "dtype": None if settings.model_id == "qwen3-max" else "float32",
            "api_calls": model.calls if isinstance(model, QwenNLI) else []}


def evidence_confidence(result, settings):
    """最大分类分数未经校准；Qwen自评分数与本地softmax明确区分。"""
    if result["status"] != "scored":
        return {"predicted_label": None, "confidence": None, "needs_review": True,
                "review_reasons": ["not_scored"]}
    scores = result["scores"]
    if (set(scores) != {"entailment", "neutral", "contradiction"}
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values())
            or not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-5)):
        raise ValueError("NLI三类分数必须是有限且归一化的概率")
    label = max(scores, key=scores.get)
    confidence = scores[label]
    reasons = []
    if confidence < settings.review_confidence_threshold:
        reasons.append("low_confidence")
    if label == "neutral":
        reasons.append("insufficient_evidence")
    elif label == "contradiction":
        reasons.append("contradiction")
    elif scores["entailment"] < settings.entailment_threshold:
        reasons.append("below_support_threshold")
    return {"predicted_label": label, "confidence": confidence,
            "needs_review": bool(reasons), "review_reasons": reasons}


def completed_execution(evidence):
    return all(e.get("execution_status", "completed" if e["status"] == "scored" else "incomplete") == "completed" for e in evidence)


def aggregate_claims(claims, predictions, settings):
    """矛盾优先；至少一条引用高置信支持，且没有超长未核验引用才通过。"""
    if sum(len(item["references"]) for item in claims) != len(predictions):
        raise ValueError("NLI预测数量与引用不匹配")
    results = []
    offset = 0
    for item in claims:
        refs = item["references"]
        evidence = []
        for ref in refs:
            result = predictions[offset]
            offset += 1
            evidence.append({"source_id": ref["source_id"],
                             "quote": ref["quote"],
                             "quote_sha256": hashlib.sha256(ref["quote"].encode()).hexdigest(),
                             **result, **evidence_confidence(result, settings)})
        scored = [e["scores"] for e in evidence if e["status"] == "scored"]
        if not refs:
            status = "unverified" if re.match(r"^(推断|建议|待确认|待办)[：:]", item["claim"]) else "missing_evidence"
        elif any(e.get("prediction_as_fact") for e in evidence):
            status = "prediction_as_fact"
        elif any(s["contradiction"] >= settings.contradiction_threshold for s in scored):
            status = "contradiction"
        elif any(e["status"] != "scored" for e in evidence):
            status = "unknown"
        elif any(s["entailment"] >= settings.entailment_threshold for s in scored):
            status = "supported"
        else:
            status = "unknown"
        # 最弱引用的分类分数用于保守筛选，不解释为整个claim状态的概率。
        confidence = (min(e["confidence"] for e in evidence)
                      if evidence and all(e["confidence"] is not None for e in evidence) else None)
        reasons = list(dict.fromkeys(reason for e in evidence for reason in e["review_reasons"]))
        if status != "supported" and status not in reasons:
            reasons.append(status)
        results.append({"section": item.get("section"), "claim": item["claim"], "status": status,
                        "execution_status": "completed" if completed_execution(evidence) else "incomplete",
                        "content_status": {"supported": "supported", "contradiction": "contradiction",
                                           "unknown": "insufficient_support" if completed_execution(evidence) else "unverified"}.get(status, "unverified"),
                        "confidence": confidence, "needs_review": bool(reasons),
                        "review_reasons": reasons, "evidence": evidence})
    accepted = all(item["status"] in {"supported", "unverified"} for item in results)
    return {"status": "passed" if accepted else "rejected", "claims": results,
            "execution_status": "completed" if completed_execution([e for c in results for e in c["evidence"]]) else "incomplete",
            "confidence_method": "min_pair_max_llm_self_reported_v1" if settings.model_id == "qwen3-max" else "min_pair_max_softmax_v1",
            "confidence_calibration": "not_calibrated",
            "review_confidence_threshold": settings.review_confidence_threshold,
            "needs_review": any(item["needs_review"] for item in results),
            "review_count": sum(item["needs_review"] for item in results),
            "review_claim_indexes": [i for i, item in enumerate(results) if item["needs_review"]],
            "supported_count": sum(item["status"] == "supported" for item in results),
            "unverified_count": sum(item["status"] == "unverified" for item in results)}


def verify_claims_sync(claims, settings):
    pairs = [(ref["quote"], item["claim"]) for item in claims for ref in item["references"]]
    if len(pairs) > settings.max_pairs:
        raise ValueError("NLI引用数量超过受控上限")
    if not _INFERENCE_LOCK.acquire(blocking=False):
        raise NLIBusyError("NLI正在处理其他请求，本次不提交摘要")
    try:
        started = time.monotonic()
        # 即使没有引用，也拒绝再次执行旧后端配置；空证据不发API请求。
        model = get_nli(settings) if pairs else None
        if settings.model_id != "qwen3-max":
            raise RetiredNLIBackendError("mDeBERTa核验后端已停用；请用qwen3-max配置创建新运行")
        predictions = []
        expanded, plans = [], []
        for item in claims:
            for ref in item["references"]:
                if (ref.get("verification_scope") != "summary_input_context"
                        and ref.get("source_type") in {"assistant_history", "task_input", "tool_record", "summary_input"}
                        and item.get("section") == "SUMMARY" and not re.match(r"^(助手曾|用户要求|操作记录|待确认[：:])", item["claim"])):
                    plans.append((ref, [], "source_not_research_evidence"))
                    continue
                windows, error = reference_windows(ref, item["claim"], model, settings)
                plan = []
                for text, start, end in windows:
                    plan.append((len(expanded), start + ref.get("context_start", 0), end + ref.get("context_start", 0)))
                    expanded.append((text, item["claim"]))
                plans.append((ref, plan, error))
        if len(expanded) > settings.max_pairs:
            raise ValueError("NLI窗口数量超过受控上限")
        window_predictions = model.predict(expanded, settings) if expanded else []
        for ref, plan, error in plans:
            metadata = {k: ref[k] for k in ("start", "end", "source_type", "source_text_hash", "context_start", "context_end", "verification_scope") if k in ref}
            if error:
                predictions.append({"status": error, "execution_status": error, **metadata})
                continue
            windows = [{**window_predictions[i], "start": start, "end": end} for i, start, end in plan]
            if len(windows) == 1:
                predictions.append({**windows[0], **metadata, "windows": windows})
            else:
                # 不能只选最高分窗口；多窗口均支持才可放行，有矛盾优先，否则待复核。
                scored_windows = [w for w in windows if w["status"] == "scored"]
                contradictions = [w for w in scored_windows if w["scores"]["contradiction"] >= settings.contradiction_threshold]
                if contradictions:
                    selected = max(contradictions, key=lambda w: w["scores"]["contradiction"])
                elif len(scored_windows) == len(windows) and all(w["scores"]["entailment"] >= settings.entailment_threshold for w in scored_windows):
                    selected = min(scored_windows, key=lambda w: w["scores"]["entailment"])
                else:
                    selected = {"status": "joint_evidence_uncertain", "execution_status": "completed"}
                predictions.append({**selected, **metadata, "windows": windows})
        report = aggregate_claims(claims, predictions, settings)
        report.update(**backend_metadata(model, settings),
                      pair_count=len(pairs), window_count=len(expanded), duration_seconds=time.monotonic() - started,
                      omission_check="not_run", threshold_calibration="not_calibrated")
        return report
    finally:
        _INFERENCE_LOCK.release()


async def verify_claims(claims, settings):
    # 同步API请求不阻塞事件循环；超时后丢弃结果，底层请求结束前保持锁。
    return await asyncio.wait_for(asyncio.to_thread(verify_claims_sync, claims, settings), timeout=settings.timeout_seconds)
