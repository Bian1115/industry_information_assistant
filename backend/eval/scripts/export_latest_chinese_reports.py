#!/usr/bin/env python3
"""把本次 DeepResearch Bench 每题最新结果导出为中文 Markdown。"""

import json
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
SOURCE_FILE = (
    BACKEND_DIR
    / "eval/data/deepresearch_bench/raw_data"
    / "quality-pass-5q-run005-20260821_153121.jsonl"
)
OUTPUT_DIR = (
    BACKEND_DIR
    / "eval/data/deepresearch_bench/中文结果"
    / "金融五题第5次评测结果-2026年08月22日"
)

TITLES = {
    1: "中国九阶层收入财务状况与中产研究",
    2: "国际前十保险公司横向比较与在华发展潜力",
    3: "中国金融未来趋势与细分领域上升空间",
    4: "二〇一〇年至今黄金走势与未来趋势",
    5: "国内金融机构投资借贷关系与系统性风险建模",
}


def score_text(meta):
    score = meta.get("quality_score")
    return "未评分" if score is None else f"{score}分"


def status_text(meta):
    verdict = meta.get("quality_verdict")
    return {
        "pass": "通过",
        "needs_revision": "待修订",
        "major_issues": "存在重大问题",
        "not_reviewed": "未审核",
    }.get(verdict, verdict or "未知")


def main():
    latest = {}
    with SOURCE_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                latest[int(record["id"])] = record

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for question_id in sorted(latest):
        record = latest[question_id]
        meta = record.get("_meta") or {}
        title = TITLES.get(question_id, f"第{question_id}题")
        status = status_text(meta)
        score = score_text(meta)
        filename = f"{question_id:02d}-{title}-{status}-{score}.md"
        target = OUTPUT_DIR / filename

        lines = [
            f"# {title}",
            "",
            f"- 题号：{question_id}",
            f"- 审核状态：{status}",
            f"- 质量评分：{score}",
            f"- 正文字符数：{meta.get('article_chars', 0)}",
            f"- 引用数量：{meta.get('references_count', 0)}",
            f"- 运行耗时：{meta.get('duration_s', 0)} 秒",
            f"- 完成时间：{meta.get('finished_at') or '无'}",
            f"- 运行编号：{meta.get('run_id') or '无'}",
            "",
            "## 原始问题",
            "",
            record.get("prompt") or "无",
            "",
        ]

        article = record.get("article") or ""
        if article:
            lines.extend(["## 报告正文", "", article.rstrip(), ""])
        else:
            lines.extend([
                "## 运行结果",
                "",
                "本题未生成正式报告。",
                "",
                f"失败原因：{meta.get('error') or '未知错误'}",
                "",
            ])

        target.write_text("\n".join(lines), encoding="utf-8")
        summary_rows.append(
            f"| {question_id} | {title} | {status} | {score} | "
            f"{meta.get('article_chars', 0)} | {meta.get('references_count', 0)} | "
            f"[{filename}]({filename.replace(' ', '%20')}) |"
        )

    summary = [
        "# 金融五题第5次评测结果汇总",
        "",
        f"原始结果文件：`{SOURCE_FILE}`",
        "",
        "本目录采用每个题号在 JSONL 中最后出现的一条记录。",
        "",
        "| 题号 | 主题 | 审核状态 | 评分 | 字符数 | 引用数 | 文件 |",
        "|---:|---|---|---:|---:|---:|---|",
        *summary_rows,
        "",
        "通过标准：`quality_verdict=pass` 且 `quality_score>=7`。",
        "",
    ]
    (OUTPUT_DIR / "00-评测结果汇总.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
