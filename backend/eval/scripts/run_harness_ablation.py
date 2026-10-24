#!/usr/bin/env python3
"""Run the deterministic 500-case Agent Harness H0/H1 ablation."""

import argparse
import asyncio
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(APP_DIR))

from eval.harness.ablation import run_ablation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = asyncio.run(run_ablation(args.output_dir))
    print(json.dumps(metrics["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
