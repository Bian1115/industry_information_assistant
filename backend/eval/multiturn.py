"""多轮开发集校验与顺序执行；人工检查项永不进入模型输入。"""
from __future__ import annotations

import asyncio
from contextlib import aclosing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Source(StrictModel):
    benchmark: Literal["DeepResearch Bench"]
    id: int
    topic: str
    language: Literal["zh", "en"]
    prompt: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    url: str
    file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def verify_hash(self):
        if hashlib.sha256(self.prompt.encode()).hexdigest() != self.prompt_sha256:
            raise ValueError("原题文本哈希不匹配")
        return self


class Turn(StrictModel):
    turn_id: int = Field(ge=1)
    user_query: str = Field(min_length=1)
    depends_on: list[int]
    objective: str = Field(min_length=1)


class Probe(StrictModel):
    probe_id: str
    introduced_turn: int
    check_turns: list[int]
    kind: Literal["constraint", "evidence", "conflict", "revision"]
    criterion: str
    annotation_status: Literal["user_constraint", "pending_source_annotation"]


class Task(StrictModel):
    schema_version: Literal[1]
    task_id: str = Field(pattern=r"^drb-mt-\d+$")
    split: Literal["development", "selection", "test"]
    source: Source
    as_of: str
    rewrite_notes: str
    turns: list[Turn] = Field(min_length=2)
    probes: list[Probe] = Field(min_length=1)

    @model_validator(mode="after")
    def check_turns(self):
        ids = [t.turn_id for t in self.turns]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError("轮次必须从1连续递增")
        for turn in self.turns:
            if len(set(turn.depends_on)) != len(turn.depends_on) or any(i < 1 or i >= turn.turn_id for i in turn.depends_on):
                raise ValueError("依赖只能指向前面的轮次")
        if len({p.probe_id for p in self.probes}) != len(self.probes):
            raise ValueError("检查项ID重复")
        for probe in self.probes:
            if probe.introduced_turn not in ids or not probe.check_turns or any(i not in ids or i < probe.introduced_turn for i in probe.check_turns):
                raise ValueError("检查项轮次不合法")
        return self


def load_tasks(paths):
    tasks = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                tasks.append(Task.model_validate_json(line))
    if not tasks:
        raise ValueError("评测集不能为空")
    if len({t.task_id for t in tasks}) != len(tasks) or len({t.source.id for t in tasks}) != len(tasks):
        raise ValueError("原始任务或任务ID重复；同一原题不能跨集合")
    for split, maximum in [("development", 5), ("selection", 5), ("test", 20)]:
        if sum(t.split == split for t in tasks) > maximum:
            raise ValueError(f"{split} 超过预定划分数量")
    return tasks


def research_context(history, query):
    # 不把original_prompt、probes、objective或尚未发送的用户轮次暴露给生成器。
    return {"original_query": query, "resolved_query": query,
            "is_follow_up": bool(history), "conversation_context": deepcopy(history),
            "memory_context": []}


def compression_points(records):
    """只有明确遥测字段才能认定摘要尝试/采用；旧事件缺失时保持unknown。"""
    views = []
    for row in records:
        if row["kind"] != "context_view":
            continue
        p = row["payload"]
        requests = [r for r in records if r["kind"] == "llm_request" and r["call_id"] == row["call_id"]
                    and r["agent_name"] == row["agent_name"] and r["context_key"] == row["context_key"]]
        views.append({"agent_name": row["agent_name"], "context_key": row["context_key"],
                      "call_id": row["call_id"], "seq": row["seq"],
                      "provider_request_count":len(requests),
                      "summary_used_in_request":bool(requests) and p.get("summary_adopted") is True,
                      **{k:p.get(k) for k in (
                          "tokens_before_compression", "tokens_after_early_offload", "tokens", "budget",
                          "soft_threshold", "summary_threshold", "summary_attempted", "summary_adopted",
                          "summary_version_before", "summary_version_after", "covered_seq_before", "covered_seq_after",
                          "history_replay", "stages", "shortened_source_ids", "compression_enabled")}})
    return views


def first_summary_replay(records, task_id):
    """导出首个摘要点之前的全量原始历史，供各候选从同一未压缩材料重建视图。"""
    from service.agent_context import REPLAY_KINDS
    for view in records:
        if view["kind"] != "context_view" or not view["payload"].get("summary_attempted") or view["payload"].get("summary_version_before") != 0:
            continue
        same = [r for r in records if r["agent_name"] == view["agent_name"] and r["context_key"] == view["context_key"]]
        expected_runs = set(view["payload"].get("history_run_ids") or [])
        if expected_runs - {r.get("run_id") for r in same}:
            continue  # 单run导出缺少继承原文时，不生成伪完整的跨run回放快照。
        inputs = [r for r in same if r["kind"] == "llm_input" and r["call_id"] == view["call_id"]]
        requests = [r for r in same if r["kind"] == "llm_request" and r["call_id"] == view["call_id"]]
        if not inputs or not requests:
            continue  # 无实际模型名不能猜测模型配置。
        current = inputs[0]
        prior = [r for r in same if r["seq"] < current["seq"] and r["kind"] in REPLAY_KINDS]
        return {"tasks":[{"task_id":task_id, "agent_name":view["agent_name"],
                          "model":requests[0]["payload"]["request"]["model"],
                          "records":prior, "contract":current["payload"].get("contract", {}),
                          "system_prompt":current["payload"]["system"], "next_task":current["payload"]["content"]}]}
    return None


async def run_dialogue(service, task, directory, *, timeout_seconds, collect_records):
    """同session多研究运行；会话成果显式传递，智能体工作历史和摘要按run隔离。"""
    from eval.scripts.run_deepresearch_bench import decode_service_event, append_references
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    session_id = "mt_" + uuid.uuid4().hex
    history = []
    results = []
    for turn in task.turns:
        run_id = str(uuid.uuid4())
        query = turn.user_query
        row = {"task_id": task.task_id, "source_id": task.source.id, "turn_id": turn.turn_id,
               "session_id": session_id, "run_id": run_id, "query": query,
               "history_mode": "full_prior_dialogue_new_run", "agent_history_continues_across_turns": False,
               "completed": False, "article": "", "compression_points": []}
        row["conversation_input_sha256"] = hashlib.sha256(json.dumps(history, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        async def consume():
            async with aclosing(service.research(query=query, session_id=session_id, run_id=run_id,
                    user_id=None, resume=False, research_context=research_context(history, query))) as stream:
                async for chunk in stream:
                    event = decode_service_event(chunk)
                    if event is None:
                        continue
                    with (directory / "研究事件.jsonl").open("a") as f:
                        f.write(json.dumps({"turn_id":turn.turn_id,"run_id":run_id,"event":event}, ensure_ascii=False)+"\n")
                    p = event.get("data") if isinstance(event.get("data"), dict) else event
                    if event.get("type") == "research_complete":
                        body = p.get("final_report", "")
                        if not body.strip():
                            raise ValueError("完成事件缺少正文")
                        row["article"] = append_references(body, p.get("references") or [])
                        row["completed"] = True
                    elif event.get("type") in {"error", "research_cancelled", "research_paused"}:
                        raise RuntimeError(str(p))
                if not row["completed"]:
                    raise RuntimeError("没有收到完成事件")
        try:
            await asyncio.wait_for(consume(), timeout=timeout_seconds)
        except Exception as exc:
            row.update(completed=False, article="", error_type=type(exc).__name__, error=str(exc))
        finally:
            try:
                records = await collect_records(run_id)
                (directory / f"第{turn.turn_id}轮工作历史.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
                row["compression_points"] = compression_points(records)
                replay = first_summary_replay(records, f"{task.task_id}-turn-{turn.turn_id}")
                if replay:
                    (directory / f"第{turn.turn_id}轮首次摘要前回放快照.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2))
                row["telemetry_status"] = "collected" if records else "no_records"
            except Exception as exc:
                row.update(telemetry_status="error", telemetry_error=str(exc))
            results.append(row)
            (directory / "逐轮结果.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        if not row["completed"]:
            break  # 不把失败结果当作已成功的前序回答继续测试。
        history.extend([{"role":"user", "content":query}, {"role":"assistant", "content":row["article"]}])
    return {"task_id":task.task_id, "source_id":task.source.id,
            "planned_turns":len(task.turns), "completed_turns":sum(r["completed"] for r in results),
            "completed":len(results)==len(task.turns) and all(r["completed"] for r in results),
            "observation_complete":len(results)==len(task.turns) and all(
                r.get("telemetry_status")=="collected" and r["compression_points"] for r in results),
            "turns":results, "unexecuted_turns":[t.turn_id for t in task.turns[len(results):]]}
