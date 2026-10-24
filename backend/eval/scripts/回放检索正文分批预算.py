"""只回放归档正文和冻结输入预算；不启动服务、不访问网页、不调用模型。"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND / "app"))

from service.agent_history import HistoryScope, bind_agent_history
from service.agent_context import protected_agent_messages
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.evidence_batches import plan_evidence_batches
from service.qwen_token_counter import QwenTokenCounter
from service.web_document import index_document


def replay(run_dir, question, counter):
    folder = run_dir / f"题目{question}"
    rows = json.loads((folder / "智能体原始历史.json").read_text())
    failed = next(row for row in reversed(rows) if row["kind"] == "context_view"
                  and row["payload"]["tokens"] > row["payload"]["budget"])
    original = next(row for row in rows if row["kind"] == "llm_input" and row["call_id"] == failed["call_id"])
    payload = original["payload"]
    read = next(row for row in reversed(rows) if row["seq"] < original["seq"] and row["kind"] == "tool_output"
                and row["payload"]["tool_name"] == "v2_web_fulltext_read")
    before, remainder = payload["content"].split("## 搜索结果\n", 1)
    material, after = remainder.split("## 任务", 1)
    sources = []
    search_metadata = {}
    for row in rows:
        if row["seq"] < original["seq"] and row["kind"] == "tool_output" and row["payload"]["tool_name"] == "v2_web_search":
            search_metadata.update((item["url"], item) for item in row["payload"]["value"])
    for url, expected_id in re.findall(r"URL: ([^\n]+)[\s\S]*?正文编号: ([^；\n]+)", material):
        url = url.strip()
        raw = next(item for item in read["payload"]["value"] if item["url"] == url)
        indexed = index_document(raw["content"], url)
        if indexed["document_id"] != expected_id:
            raise ValueError("归档正文与原运行正文编号不一致，不能冒充相同材料回放")
        sources.append({**search_metadata.get(url, {}), **raw, **indexed,
                        "reader_verified": True, "full_content": raw["content"]})
    agent = DeepScout.__new__(DeepScout)
    agent.name, agent.logger = "DeepScout", logging.getLogger("offline-batch-replay")
    scope = HistoryScope(None, "offline", "offline", None, contract=payload["contract"],
                         context_policy=failed["payload"]["context_policy"])
    prompt_builder = lambda text: before + "## 搜索结果\n" + text + "\n\n## 任务" + after
    with bind_agent_history(scope):
        batches, budget = plan_evidence_batches(agent, sources, prompt_builder,
                                                payload["system"], counter=counter)
    expected = {(source["url"], source["document_id"], sentence["id"])
                for source in sources for sentence in source["sentence_index"]}
    actual = {(url, doc, ident) for batch in batches for (url, doc), ids in batch.bindings.items() for ident in ids}
    for batch in batches:
        sent_count = counter.count_messages(protected_agent_messages(scope, payload["system"], batch.prompt))
        if sent_count != batch.tokens or sent_count > budget:
            raise ValueError("规划 token 与最终完整消息 token 不一致或超限")
    if actual != expected:
        raise ValueError("分批遗漏原句或生成了不存在的原句编号")
    return {
        "题号": question, "原始数据目录": str(folder.resolve()),
        "输入预算": budget, "原运行最终输入tokens": failed["payload"]["tokens"],
        "原运行当前prompt_tokens": counter.count_text(payload["content"]),
        "原运行展示原句次数": len(re.findall(r"\[s\d+\]", material)),
        "来源数": len(sources), "合格正文全部原句数": len(expected),
        "批次数": len(batches), "各批完整消息tokens": [batch.tokens for batch in batches],
        "最大批输入tokens": max((batch.tokens for batch in batches), default=0),
        "覆盖全部原句": actual == expected,
        "原句展示次数含相邻句重叠": sum(len(ids) for batch in batches for ids in batch.bindings.values()),
        "全部批次累计输入tokens": sum(batch.tokens for batch in batches),
        "未删除任务契约": all("最新有效任务约束" in protected_agent_messages(scope, payload["system"], batch.prompt)[0]["content"] for batch in batches),
        "局限": "只验证 tokenizer 预算和正文覆盖；未执行模型提取，未评估报告质量和实际费用。完整正文范围大于旧运行每页12000字符展示范围，累计token不可直接作为费用对比。",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("回放结果必须新建，不能覆盖历史文件")
    counter = QwenTokenCounter(model_name=str(BACKEND / ".cache/qwen-tokenizer"))
    results = {"评估日期": "2026-10-02", "模型调用次数": 0, "在线网页请求次数": 0,
               "口径": "原始失败输入的任务/假设/契约 + 同编号完整归档正文 + 新分批算法",
               "题目": [replay(args.run_dir, question, counter) for question in (46, 49)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
