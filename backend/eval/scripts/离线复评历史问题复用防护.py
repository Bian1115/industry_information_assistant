"""用真实历史原始响应检查错位识别及失败闭锁；不伪造修正后的模型预测。"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend/app"))
from service.research_context_service import ResearchContext, ResearchContextService, _historical_query_source
from service.qwen_token_counter import get_qwen_token_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "backend/eval/data/第20题32轮待合并结果恢复续跑-20261008-184009/真实运行"
    names = {f.name for f in fields(ResearchContext)}
    counter = get_qwen_token_counter()
    rows = []
    for i in range(1, 25):
        original = json.loads((source / f"第{i}轮/会话任务输入.json").read_text())
        trace = original.get("task_trace") or original.get("task_metrics", {}).get("task_trace", {})
        raw = trace.get("raw_response")
        if not raw:
            rows.append({"轮次": i, "原始模型响应可用": False, "结果边界": "未测：原始响应缺失，不能用最终任务假装原始响应"})
            continue
        context = ResearchContext(**{k: v for k, v in original.items() if k in names})
        context.resolved_query = context.original_query
        parsed = trace["parsed_response"]
        detected = _historical_query_source(context, parsed.get("resolved_query"))
        previous = json.loads((source / f"第{i-1}轮/会话任务输入.json").read_text())["original_query"] if i > 1 else None
        expected_stale = parsed.get("resolved_query") == previous
        row = {"轮次": i, "原始模型响应可用": True, "原始resolved_query": parsed.get("resolved_query"),
               "真实已知错位": expected_stale, "防护识别错位": detected is not None, "命中来源": detected,
               "识别符合历史标注": (detected is not None) == expected_stale}
        if expected_stale:
            # 故障注入：供应商两次都返回同一历史坏响应；验证不会静默改字段后继续研究。
            client_call = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]))
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=client_call)))
            context.session_task = {}; context.task_metrics = {}; context.task_trace = {}
            resolver = ResearchContextService.__new__(ResearchContextService); resolver.token_counter = counter
            with patch("service.research_context_service.create_openai_client", return_value=client):
                resolver._resolve_query(context)
            row.update(故障注入说明="模拟供应商连续两次返回同一真实旧坏响应，不生成正确答案",
                       重判调用次数=client_call.call_count, 最终状态=context.session_task["status"],
                       历史依赖保留=context.session_task["depends_on_history"],
                       最终问题未被旧问题覆盖=context.resolved_query == context.original_query,
                       错位任务未执行=context.session_task["status"] == "clarify" and context.session_task.get("selected_artifact") is None)
            directory = args.output / "原始响应故障回放" / f"第{i}轮"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "防护结果.json").write_text(json.dumps(context.to_dict(), ensure_ascii=False, indent=2) + "\n")
        rows.append(row)
    available = [x for x in rows if x["原始模型响应可用"]]
    stale = [x for x in available if x["真实已知错位"]]
    normal = [x for x in available if not x["真实已知错位"]]
    summary = {"评估日期": "2026-10-17", "评估性质": "真实历史原始响应的离线识别与故障注入，未调用生成模型",
               "原始响应可用轮数": len(available), "原始响应缺失轮数": 24 - len(available),
               "已知错位识别": {"命中": sum(x["防护识别错位"] for x in stale), "总数": len(stale)},
               "未错位响应误报": {"误报": sum(x["防护识别错位"] for x in normal), "总数": len(normal)},
               "持续错位任务安全终止": {"正确终止": sum(x["错位任务未执行"] for x in stale), "总数": len(stale)},
               "局限": "只验证精确复用防护，不度量新模型重判后的话题、操作、目标、要求或语义准确率；非精确复用的语义漂移尚未覆盖。",
               "逐轮结果": rows}
    (args.output / "历史原始响应防护复评.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    assert len(stale) == 10 and len(normal) == 4
    assert all(x["防护识别错位"] and x["错位任务未执行"] and x["重判调用次数"] == 2 for x in stale)
    assert not any(x["防护识别错位"] for x in normal)
    print(json.dumps({k: v for k, v in summary.items() if k != "逐轮结果"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
