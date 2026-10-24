"""正文生成、立即提交、可选Claimify审计三个独立步骤。"""
from __future__ import annotations

import asyncio
import time
import uuid

from service.agent_history import AgentHistoryError, canonical_json, content_hash, exception_details, record_agent_message
from service.research_control import ResearchCancelledError

PROSE_VERSION = "summary-prose-v1"
PROMPT_VERSION = "context-summary-prose-first-v2"


def pack_prose_summary(prose):
    from service.claimify import parse_prose
    parse_prose(prose, allow_legacy=False)
    return prose.strip() + "\n\n## CLAIMS\n" + canonical_json({
        "version": PROSE_VERSION, "claims": [], "verification_status": "unverified",
    })


async def summarize_prose_first(agent, scope, records, counter, *, previous_summary="", previous_sources=None,
                                verification_out=None, previous_source_ids=None, previous_source_hashes=None,
                                on_generated=None, on_audited=None):
    from config.llm_config import get_summary_model
    from service.agent_context import scoped_input_budget, source_text
    from service.claimify import generate_prose, prose_messages, saved_prose
    from service.context_policy import policy_for_scope
    from service.summary_hierarchy import BATCH_POLICY, fragments_for_records, hierarchical_summary

    policy = policy_for_scope(scope, agent.name)
    settings = policy.summary_lifecycle
    output_tokens = policy.summary_output_tokens
    budget = scoped_input_budget(scope, agent.name, output_tokens, summary=True)
    draft_budget = budget - min(output_tokens + 1024, budget // 4)
    contract = scope.contract if policy.pinned_contract else {}
    source_ids = list(dict.fromkeys([*(previous_source_ids or []), *(r["source_id"] for r in records)]))
    source_hashes = {**(previous_source_hashes or {}), **{r["source_id"]: r["content_hash"] for r in records}}
    sources = dict(previous_sources or {})
    sources.update({r["source_id"]: source_text(r) for r in records})
    summary_model = get_summary_model()
    stage_totals = []
    attempt_id = str(uuid.uuid4())
    input_context = None
    source_rows = {r["source_id"]: r for r in records}

    async def call_stage(messages, stage):
        nonlocal input_context
        call_id = str(uuid.uuid4())
        context = None
        if settings.claimify_enabled:
            from service.summary_input_context import number_input_context
            context = await number_input_context(messages, scope, agent.name, call_id, source_rows)
        input_tokens = counter.count_messages(messages)
        if input_tokens > budget:
            raise ValueError(f"{stage} 输入超出摘要器预算")
        if len(stage_totals) >= 64:
            raise ValueError("摘要正文生成调用数量超过本次受控上限")
        started = time.monotonic()
        await record_agent_message(agent.name, "summary_stage", {
            "stage": stage, "phase": "generation", "summary_attempt_id": attempt_id,
            "model": summary_model, "input_tokens": input_tokens, "input_budget": budget,
            "output_budget": output_tokens, "max_stage_calls": 64,
            "timeout_seconds": settings.generation_timeout_seconds, "batch_policy": BATCH_POLICY,
            "prompt_version": PROMPT_VERSION, "input_hash": content_hash(messages),
        }, call_id)
        error = {}
        try:
            output = await agent._call_llm_impl(
                messages[0]["content"], messages[1]["content"], False, 0.0, output_tokens,
                history_call_id=call_id, prepared_messages=messages, model_override=summary_model,
            )
            await record_agent_message(agent.name, "summary_output", {
                "content": output, "stage": stage, "phase": "generation", "summary_attempt_id": attempt_id,
                "output_tokens": counter.count_text(output), "prompt_version": PROMPT_VERSION,
            }, call_id)
            if counter.count_text(output) > output_tokens:
                raise ValueError(f"{stage} 输出超出预算")
            if context is not None:
                input_context = context
            return output
        except BaseException as exc:
            error = exception_details(exc)
            await record_agent_message(agent.name, "summary_stage_error", {
                **error, "stage": stage, "phase": "generation", "summary_attempt_id": attempt_id,
            }, call_id)
            raise
        finally:
            total = {"stage": stage, "call_id": call_id, "phase": "generation", "summary_attempt_id": attempt_id,
                     "duration_seconds": time.monotonic() - started, "input_tokens": input_tokens, **error}
            stage_totals.append(total)
            await record_agent_message(agent.name, "summary_stage_end", total, call_id)

    generation_started = time.monotonic()
    await record_agent_message(agent.name, "summary_phase_start", {
        "phase": "generation", "summary_attempt_id": attempt_id, "timeout_seconds": settings.generation_timeout_seconds,
    }, attempt_id)
    error = {}
    try:
        async with asyncio.timeout(settings.generation_timeout_seconds):
            for row in records:
                if row["content_hash"] != content_hash(row["payload"]):
                    raise ValueError("新增摘要来源快照哈希不一致")
            previous = saved_prose(previous_summary, previous_sources or {})
            def largest_request(fragment):
                return prose_messages({
                    "new_records": [fragment], "previous_summary": "", "latest_contract": contract,
                })
            fragments = list(fragments_for_records(records, counter, draft_budget, largest_request))
            prose = await hierarchical_summary(
                fragments, previous, contract, call_stage, counter, draft_budget, policy, sources,
                build_messages=prose_messages, generate=generate_prose,
            )
            final = pack_prose_summary(prose)
            if counter.count_text(final) > output_tokens:
                raise ValueError("摘要正文与未核验状态标记合计超过输出预算")
    except BaseException as exc:
        exc.summary_phase = "generation"
        error = exception_details(exc)
        raise
    finally:
        generation_duration = time.monotonic() - generation_started
        await record_agent_message(agent.name, "summary_phase_end", {
            "phase": "generation", "summary_attempt_id": attempt_id,
            "duration_seconds": generation_duration,
            "status": "error" if error else "completed", **error,
        }, attempt_id)

    report = {
        "status": "not_run", "execution_status": "not_run", "content_status": "unverified",
        "generation_status": "completed", "generation_duration_seconds": generation_duration,
        "claimify_status": "pending" if settings.claimify_enabled else "disabled",
        "claimify_duration_seconds": None,
        "source_ids": source_ids, "source_hashes": source_hashes, "summary_attempt_id": attempt_id,
        "stage_totals": stage_totals, "omission_check": "not_run",
        "generation_stage_totals": stage_totals,
    }
    if verification_out is not None:
        verification_out.update(report)
    await record_agent_message(agent.name, "summary_generated", {
        "summary": final, "summary_hash": content_hash(final), **report,
    }, attempt_id)
    # 生成时钟已经结束；持久化失败必须传播，不能把未提交视为成功。
    if on_generated is not None and not await on_generated(final):
        report["claimify_status"] = "not_adopted"
        if verification_out is not None:
            verification_out.update(report)
        return final
    if not settings.claimify_enabled:
        await record_agent_message(agent.name, "summary_claimify_skipped", {
            "summary_attempt_id": attempt_id, "summary_hash": content_hash(final), "reason": "disabled",
            "phase": "claimify", "status": "disabled", "duration_seconds": None,
        }, attempt_id)
        return final

    audit_started = time.monotonic()
    await record_agent_message(agent.name, "summary_phase_start", {
        "phase": "claimify", "summary_attempt_id": attempt_id, "timeout_seconds": settings.claimify_timeout_seconds,
    }, attempt_id)
    fatal = None
    try:
        async with asyncio.timeout(settings.claimify_timeout_seconds):
            if settings.atomic_citations:
                from service.summary_atomic_citations import audit_prose_atomic
                audit = await audit_prose_atomic(agent, scope, prose, input_context, counter, attempt_id)
                claims = audit['mapped_claims']
                audited = final
            else:
                from service.summary_citations import summarize_with_citations
                audit = {}
                audited = await summarize_with_citations(
                    agent, scope, records, counter, verification_out=audit,
                    prepared_prose=prose, timeout_seconds=settings.claimify_timeout_seconds,
                    phase="claimify", summary_attempt_id=attempt_id, input_context=input_context,
                )
                from service.agent_context import parse_summary
                _, claims = parse_summary(audited, {sid: source_text(row) for sid, row in input_context["rows"].items()})
            report.update(audit, claimify_status="completed", citation_claims=claims,
                          claimify_stage_totals=audit.get("stage_totals", []),
                          stage_totals=stage_totals + audit.get("stage_totals", []),
                          audited_summary_hash=content_hash(audited), summary_hash=content_hash(final))
    except BaseException as exc:
        report.update(exception_details(exc), status="error", execution_status="error", content_status="unverified",
                      claimify_status="cancelled" if isinstance(exc, asyncio.CancelledError) else "error", phase="claimify")
        await record_agent_message(agent.name, "summary_claimify_error", {
            **report, "summary_hash": content_hash(final),
        }, attempt_id)
        if isinstance(exc, (AgentHistoryError, PermissionError, ResearchCancelledError, asyncio.CancelledError)) or not isinstance(exc, Exception):
            fatal = exc
    finally:
        report["claimify_duration_seconds"] = time.monotonic() - audit_started
        # 只更新状态，不用核验结果替换已经提交的正文或回退游标。
        if verification_out is not None:
            verification_out.update(report)
        if on_audited is not None:
            await on_audited(final)
        await record_agent_message(agent.name, "summary_phase_end", {
            "phase": "claimify", "summary_attempt_id": attempt_id,
            "duration_seconds": report["claimify_duration_seconds"], "status": report["claimify_status"],
            **({key: report[key] for key in ("error_type", "error_module", "message") if key in report}),
        }, attempt_id)
    if fatal is not None:
        raise fatal
    return final
