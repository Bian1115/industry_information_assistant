"""真实本地NLI＋隔离PostgreSQL验证；摘要生成用桩，不调用付费大模型。"""

import asyncio
import json
import os
from pathlib import Path
import sys
import traceback
import uuid
from unittest.mock import AsyncMock, patch

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND / "test"))
os.environ["TEST_AGENT_HISTORY_POSTGRES"] = "1"
from test_智能体独立历史 import PostgresHistoryTests, make_agent, CharCounter, bind_agent_history
from test_摘要合并与引用分离 import clause, draft, citation
from service.agent_context import prepare_agent_messages, ContextBudgetExceeded
from service.agent_history import PostgresAgentHistory


async def run():
    cases = [
        ("中文支持应提交", "本研究纳入了800名受试者。", "研究共有800名参与者。", True),
        ("中文矛盾应拒绝", "本研究未发现治疗能够降低死亡率。", "本研究发现治疗能够降低死亡率。", False),
        ("中文信息不足应拒绝", "本研究纳入了800名受试者。", "本研究纳入了800名男性受试者。", False),
    ]
    results = []
    fixture = PostgresHistoryTests()
    for name, quote, claim, expected in cases:
        await fixture.asyncSetUp()
        original = quote + "其他背景。" * 1200
        row = await fixture.store.append(fixture.scope, "DeepScout", "llm_output", {"content": original}, str(uuid.uuid4()))
        item = clause(claim, row["source_id"])
        agent = make_agent()
        agent._call_llm_impl = AsyncMock(side_effect=[draft(item), citation(item, [(row["source_id"], quote)])])
        error = None
        with patch.dict(os.environ, {"AGENT_SUMMARY_NLI_ENABLED": "1", "AGENT_SUMMARY_STRATEGY": "split-citations-v4", "AGENT_HISTORY_SUMMARY_ENABLED": "1"}), bind_agent_history(fixture.scope), patch("service.agent_context.get_qwen_token_counter", return_value=CharCounter()), patch("service.agent_context.input_budget", side_effect=lambda out: 4000 if out == 3000 else 1400):
            try:
                await prepare_agent_messages(agent, fixture.scope, "sys", "继续研究", str(uuid.uuid4()), 6000)
            except ContextBudgetExceeded as exc:
                error = exc.code
        saved = await PostgresAgentHistory(fixture.sessions).load(fixture.scope, agent.name)
        verification = [r["payload"] for r in saved["messages"] if r["kind"] == "summary_verification"]
        ok = saved["summary_version"] == (1 if expected else 0) and saved["covered_seq"] == (row["seq"] if expected else 0)
        ok = ok and len(verification) == 1 and verification[0]["status"] == ("passed" if expected else "rejected")
        ok = ok and saved["messages"][0]["payload"]["content"] == original
        results.append({"name": name, "expected_adopted": expected, "passed": ok,
                        "summary_version": saved["summary_version"], "covered_seq": saved["covered_seq"],
                        "verification": verification, "summary_metadata": saved["summary_metadata"], "error": error})
    return results


if __name__ == "__main__":
    output = BACKEND / "eval/data/本地NLI接入验证-20260927/真实核验与数据库链路结果.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"date": "2026-09-27", "method": "真实NLI、真实隔离PostgreSQL；摘要与引用补齐输出为桩；主模型预算使用字符计数桩"}
    setup_done = False
    try:
        PostgresHistoryTests.setUpClass()
        setup_done = True
        payload["cases"] = asyncio.run(run())
        payload["passed"] = all(case["passed"] for case in payload["cases"])
    except Exception as exc:
        payload.update(passed=False, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if setup_done:
            PostgresHistoryTests.tearDownClass()
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.exit(0 if payload["passed"] else 1)
