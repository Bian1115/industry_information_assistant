#!/usr/bin/env python3
"""列出、预检并冻结上下文消融配置；不调用模型、不启动研究、不覆盖旧结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from datetime import datetime, timezone

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent
APP = BACKEND / "app"
sys.path.insert(0, str(APP))
# 配置工具不加载 service/__init__.py 中与本任务无关的搜索/向量库连接。
package = types.ModuleType("service")
package.__path__ = [str(APP / "service")]
sys.modules.setdefault("service", package)

from service.context_policy import PRESET_DIR, read_policy


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="单组JSON；相对路径以backend为根")
    parser.add_argument("--list", action="store_true", help="检查9组预设，不发起研究")
    parser.add_argument("--output", type=Path, help="新增中文结果目录；拒绝覆盖已有目录")
    parser.add_argument("--allow-dirty", action="store_true", help="允许脏工作区的机制预实验，不标为正式实验")
    parser.add_argument("--annotations", type=Path, help="冻结的独立人工标注文件")
    parser.add_argument("--formal", action="store_true", help="正式前置条件检查；缺源码、人工标注或旧基线时拒绝")
    parser.add_argument("--baseline-manifest", type=Path, help="改造前源码与配置冻结清单")
    parser.add_argument("--experiment-manifest", type=Path, help="模型、材料、上游记忆及整任务预算冻结清单")
    args = parser.parse_args()
    if bool(args.config) == bool(args.list):
        parser.error("--config 与 --list 必须二选一")

    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env", override=False)
    paths = sorted(PRESET_DIR.glob("*.json")) if args.list else [args.config]
    rows = []
    for path in paths:
        try:
            policy = read_policy(path)
            blockers = policy.blockers()
            if policy.semantic_verification and policy.nli:
                directory = Path(policy.nli.model_dir)
                if not directory.is_absolute():
                    directory = BACKEND / directory
                download = directory / "下载清单.json"
                if not download.is_file():
                    blockers.append("缺少冻结NLI下载清单")
            rows.append({"配置文件": str(path), "组别": policy.group,
                         "配置有效": True, "组件可执行": not blockers,
                         "阻塞原因": blockers, "配置快照": policy.snapshot()})
        except (ValueError, OSError) as exc:
            rows.append({"配置文件": str(path), "配置有效": False,
                         "组件可执行": False, "阻塞原因": [str(exc)]})

    dirty = bool(git("status", "--porcelain"))
    sources = [
        "backend/app/service/context_policy.py", "backend/app/service/agent_context.py",
        "backend/app/service/agent_history.py", "backend/app/service/deep_research_v2/agents/base.py",
        "backend/app/service/deep_research_v2/service.py", "backend/app/service/deep_research_v2/graph.py",
        "backend/app/prompts/智能体历史摘要.txt",
        "backend/app/service/nli_verifier.py", "backend/app/service/verification_checks.py",
        "backend/app/service/summary_hierarchy.py", "backend/app/service/verification_trace.py",
        "backend/app/service/verification_budget.py", "backend/eval/verification/scoring.py",
    ]
    # 同期新增的正文/引句分离策略也纳入版本清单；保留兼容旧分支。
    for path in [ROOT / "backend/app/service/summary_citations.py", *sorted((APP / "prompts").glob("摘要*.txt"))]:
        if path.is_file():
            sources.append(str(path.relative_to(ROOT)))
    manifest = {
        "生成时间": datetime.now(timezone.utc).isoformat(), "源码提交": git("rev-parse", "HEAD"),
        "工作区有未提交内容": dirty,
        "阶段": "配置预检，未运行质量消融", "允许脏工作区预实验": args.allow_dirty,
        "源码与提示词SHA256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "分词器": {"model": os.getenv("QWEN_TOKENIZER_MODEL", "Qwen/Qwen3-8B"),
                    "revision": os.getenv("QWEN_TOKENIZER_REVISION", "未固定")},
        "结果": rows,
        "正式消融还需固定": ["干净源码提交", "数据及工具快照清单与哈希", "研究/摘要模型实际版本", "提示词和分词器版本", "上游会话摘要与长期记忆输入", "独立人工标注和按轨迹划分", "整任务预算与费用单价", "改造前源码及实际配置"],
    }
    formal_blockers = []
    if dirty:
        formal_blockers.append("源码尚未冻结到干净提交")
    if args.annotations:
        manifest["人工标注文件SHA256"] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
        sys.path.insert(0, str(BACKEND))
        from eval.verification.scoring import validate_gold
        annotated = json.loads(args.annotations.read_text())
        validate_gold(annotated)
        manifest["人工标注版本"] = annotated["annotation_version"]
        if annotated["annotation_origin"] == "synthetic_mechanism":
            formal_blockers.append("合成机制标注不能作为正式人工真值")
    else:
        formal_blockers.append("未提供独立人工标注")
    if args.baseline_manifest:
        baseline = json.loads(args.baseline_manifest.read_text())
        if not baseline.get("source_commit") or not baseline.get("config_hash"):
            formal_blockers.append("改造前基线缺少源码提交或配置哈希")
        manifest["改造前基线清单SHA256"] = hashlib.sha256(args.baseline_manifest.read_bytes()).hexdigest()
    else:
        formal_blockers.append("捕获当前环境不能替代改造前基线")
    if manifest["分词器"]["revision"] == "未固定":
        formal_blockers.append("分词器版本未固定")
    if args.experiment_manifest:
        experiment = json.loads(args.experiment_manifest.read_text())
        required = {"snapshot_sha256", "models", "tokenizer", "task_budget", "upstream_memory_sha256", "prompts_sha256"}
        if not required <= set(experiment):
            formal_blockers.append("实验冻结清单字段不齐")
        else:
            if (not experiment["models"] or any(not m.get("revision") for m in experiment["models"].values())
                    or not experiment["tokenizer"].get("revision")
                    or not experiment["task_budget"].get("max_requests")
                    or not experiment["task_budget"].get("max_seconds")
                    or not experiment["task_budget"].get("max_tokens_by_model")
                    or not experiment["task_budget"].get("prices_per_million")):
                formal_blockers.append("模型/分词器版本、任务总预算或单价未冻结")
        manifest["实验冻结清单SHA256"] = hashlib.sha256(args.experiment_manifest.read_bytes()).hexdigest()
    else:
        formal_blockers.append("未提供材料、模型、上游记忆及任务总预算清单")
    downloads = {}
    for row in rows:
        config = row.get("配置快照", {}).get("config", {})
        nli = config.get("nli")
        if not nli:
            continue
        directory = Path(nli["model_dir"])
        directory = directory if directory.is_absolute() else BACKEND / directory
        path = directory / "下载清单.json"
        if path.is_file() and str(path) not in downloads:
            data = json.loads(path.read_text())
            checks = []
            for item in data.get("files", []):
                if Path(item["file"]).name != item["file"]:
                    raise ValueError("模型清单路径无效")
                file = directory / item["file"]
                actual = None
                if file.is_file():
                    with file.open('rb') as stream:
                        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                checks.append({"file":item["file"], "sha256":actual, "matched":actual == item["sha256"]})
            downloads[str(path)] = {"manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "files":checks}
            if not checks or not all(c["matched"] for c in checks) or data.get("revision") != nli["revision"]:
                formal_blockers.append("NLI模型/分词器清单或文件哈希不匹配")
    manifest["NLI模型与分词器清单"] = downloads
    manifest["正式前置阻塞"] = formal_blockers
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "配置预检清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if len(rows) == 1 and rows[0]["配置有效"]:
            (args.output / "配置快照.json").write_text(json.dumps(rows[0]["配置快照"], ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "配置快照"}, ensure_ascii=False))
    if dirty:
        print("工作区未提交：只能作为预实验；哈希清单不能代替源码提交。", file=sys.stderr)
    if args.formal and (formal_blockers or any(not r["组件可执行"] for r in rows)):
        return 2
    if not args.list and (not rows[0]["组件可执行"] or (dirty and not args.allow_dirty)):
        return 2
    # --list 用于完整报告能力，不把预期的未实现组件当成JSON加载失败。
    return 0 if all(row["配置有效"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
