"""合并本次请求涉及的正式、失败与模型探测API账本，避免遗漏中断消耗。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "app")]
from eval.scripts.记录模型调用用量 import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_rows, runs, seen = [], [], set()
    for directory in args.run:
        directory = directory.resolve()
        paths = sorted(directory.glob("题目*/全部模型API用量.jsonl"))
        paths += sorted(directory.glob("模型*用量.jsonl"))
        rows, files = [], []
        for path in paths:
            count = 0
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                assert row["call_id"] not in seen, "重复API记录，不应累加"
                seen.add(row["call_id"]); rows.append(row); count += 1
            files.append({"path": str(path), "calls": count})
        runs.append({"directory": str(directory), "files": files, "usage": summarize(rows)})
        all_rows.extend(rows)
    result = {"updated_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "runs": runs, "actual_total_usage": summarize(all_rows),
        "note": "各API调用只计一次；包含失败/中断/兼容探测。未知usage不填零，不包含Codex主代理自身对话推理和未返回可记录usage的服务成本。不同模型token分别展示，未折算货币费用。"}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    usage = result["actual_total_usage"]
    print(json.dumps({k: usage[k] for k in ("calls", "failed_calls", "total_tokens_known", "total_tokens_missing_calls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
