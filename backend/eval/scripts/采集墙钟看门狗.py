"""已有采集进程的墙钟看门狗；时限包括机器暂停期间的墙钟变化。"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--task-dir", type=Path, required=True)
parser.add_argument("--timeout", type=int, default=1800)
args = parser.parse_args()
identity = json.loads((args.task_dir / "运行身份.json").read_text())
first = json.loads((args.task_dir / "原始SSE轨迹.jsonl").read_text().splitlines()[0])
deadline = first["received_at"] + args.timeout


def alive():
    row = subprocess.run(["ps", "-p", str(args.pid), "-o", "command="], capture_output=True, text=True)
    return "采集通用研究评测集.py" in row.stdout and f"--only-id {identity['task']['id']}" in row.stdout


while alive() and not (args.task_dir / "运行结果.json").exists():
    if time.time() >= deadline:
        reason = {"reason": "ExternalWallClockTimeout", "configured_seconds": args.timeout,
                  "actual_wall_seconds": round(time.time() - first["received_at"], 1),
                  "stopped_at_epoch": time.time(), "pid": args.pid}
        (args.task_dir / "外部停止原因.json").write_text(json.dumps(reason, ensure_ascii=False, indent=2))
        os.killpg(args.pid, signal.SIGINT)
        shutdown_deadline = time.time() + 10
        while alive() and time.time() < shutdown_deadline:
            time.sleep(1)
        if alive():
            os.killpg(args.pid, signal.SIGKILL)
        print("已停止超时采集进程", flush=True)
        break
    time.sleep(1)
