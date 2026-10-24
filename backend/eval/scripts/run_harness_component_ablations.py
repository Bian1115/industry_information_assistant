#!/usr/bin/env python3
"""Run four single-component Agent Harness ablations."""

import argparse
import asyncio
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(APP_DIR))

from eval.harness.component_ablation import run_component_ablations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run_component_ablations(args.output_dir))
    print(json.dumps(result["results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
