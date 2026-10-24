"""读取采集文件的紧凑进度，不改变研究运行。"""
import argparse
import collections
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--compact", action="store_true")
args = parser.parse_args()
usage = collections.defaultdict(collections.Counter)
trace = args.output / "完整过程埋点.jsonl"
if trace.exists():
    with trace.open() as source:
        for line in source:
            if not line.endswith("\n"):
                continue
            record = json.loads(line)
            if record.get("event") == "start" and record.get("as_type") == "generation":
                totals = usage[record.get("session_id")]
                totals["recorded_llm_calls"] += 1
                totals["recorded_tokens"] += (record.get("usage_details") or {}).get("total", 0)
rows = []
for task_id in [35, 31, 46, 16, 49]:
    folder = args.output / f"题目{task_id}"
    path = folder / "研究事件.jsonl"
    events = [json.loads(line) for line in path.open() if line.endswith("\n")] if path.exists() else []
    row = {"id": task_id, "events": len(events)}
    identity = folder / "运行身份.json"
    if identity.exists():
        session_id = json.loads(identity.read_text())["session_id"]
        row.update(usage.get(session_id, {"recorded_llm_calls": 0, "recorded_tokens": 0}))
    if events:
        last = events[-1]
        row.update(type=last["type"], agent=last.get("data", {}).get("agent"), node=last.get("data", {}).get("node"))
        progress = [item.get("data", {}).get("progress") for item in events if item["type"] == "research_progress"]
        row["progress"] = progress[-1] if progress else None
        raw = folder / "原始SSE轨迹.jsonl"
        if raw.exists():
            with raw.open() as source:
                first = json.loads(next(source))
            row["wall_seconds"] = round(time.time() - first["received_at"], 1)
    if (folder / "运行结果.json").exists():
        result = json.loads((folder / "运行结果.json").read_text())
        row.update(report_chars=len(result["article"]), error=result["_meta"].get("error"),
                   recorded_duration_s=result["_meta"].get("duration_s"), archived=(folder / "归档统计.json").exists())
        row["wall_seconds"] = result["_meta"].get("duration_s")
    rows.append(row)
status = {"updated_at_epoch": time.time(), "tasks": rows}
(args.output / "采集监测进度.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
display = {"archived": [row["id"] for row in rows if row.get("archived")],
           "active": [row for row in rows if row.get("events") and not row.get("archived")],
           "pending": [row["id"] for row in rows if not row.get("events")]} if args.compact else rows
print(json.dumps(display, ensure_ascii=False))
