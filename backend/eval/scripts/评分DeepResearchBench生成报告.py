"""用当前GPT与Claude双裁判评分冻结的生成报告；每位裁判两次，原始结果留存。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from eval.报告双裁判 import (PROVIDERS, aggregate, common_prompt, configuration,
    digest, grade_once, load_tasks, safe_error, save_json)


def write_report(output, config, rows, status, error=None):
    completed = [r for r in rows if r.get('ensemble') is not None]
    summary = {'updated_at': datetime.now(ZoneInfo('Asia/Singapore')).isoformat(), 'status': status,
               'tasks': len(config['frozen_tasks']), 'visited_tasks': len(rows), 'scored_reports': len(completed),
               'score_mean': sum(r['ensemble']['total'] for r in completed) / len(completed) if completed else None,
               'error': error, 'results': rows}
    save_json(output / '评分汇总.json', summary)
    report_path = Path(config['analysis_report'])
    lines = ['# DeepResearch生成报告双裁判评分结果及分析', '',
             '评估日期：' + summary['updated_at'], '', '运行状态：' + status,
             '', f"原始结果：`{output}`。输入报告：`{config['reports_path']}`。", '',
             '配置：GPT与Claude各两次，官方逐细则加权，0—10分；在线Critic分数单独保留。', '',
             f"核心指标：计划{summary['tasks']}题；完整双裁判{len(completed)}份；有效总分均值{summary['score_mean']}。", '',
             '| 题号 | 状态 | 双裁判总分 |', '| --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['id']} | {row['status']} | {row['ensemble']['total'] if row['ensemble'] else '未评分'} |")
    lines += ['', '失败与未完成案例：见评分汇总中的状态，以及裁判原始记录中的error、完整原始响应和请求指纹。', '',
              '已测量：表中已完成响应的加权分。间接推断：低分细则可以作为复核线索。尚未评估：失败根因的人工确认、现实事实真实性、优化前后收益。', '',
              '局限：缺失评分不填零，未完成双裁判不计算融合分；模型裁判不等于人类金标。本运行不自动调用Langfuse发布。', '',
              '结论与下一步：先补齐失败评分，再结合报告原文和研究轨迹复核低分细则；完整理由和摘录在裁判原始记录中。']
    if error:
        lines += ['', '中断原因：' + error]
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BACKEND / 'eval/data/DeepResearch中文验证集-20261108')
    parser.add_argument('--reports', type=Path, required=True, help='每题唯一的id/prompt/article JSONL')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='仅复用相同指纹的有效评分，补跑未完成评分')
    parser.add_argument('--prepare-only', action='store_true', help='只校验和冻结输入，不调用模型')
    args = parser.parse_args()
    config = configuration()
    tasks = load_tasks(args.dataset, args.reports)
    common = common_prompt()
    config.update(frozen_tasks=tasks, common_prompt=common, reports_path=str(args.reports.resolve()),
                  dataset_path=str(args.dataset.resolve()))
    fingerprint = digest(config)
    config_path = args.output / '评估配置.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if not args.resume or old['config_sha256'] != fingerprint:
            raise ValueError('目录已存在且未指定相同配置续跑，请选择新输出目录')
        config = old
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('输出目录非空且缺少冻结配置')
        stamp = datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d-%H%M%S-%f')
        config.update(config_sha256=fingerprint, analysis_report=str(BACKEND.parent / 'docs' / f'DeepResearch生成报告双裁判评分结果及分析-{stamp}.md'))
        save_json(config_path, config)
    rows = []
    status, error = ('已准备未调用模型' if args.prepare_only else '运行中'), None
    try:
        if not args.prepare_only:
            for task in tasks:
                packets = []
                if task['generation_status'] == 'completed':
                    for provider in PROVIDERS:
                        for repeat in (1, 2):
                            packets.append(grade_once(task, provider, repeat, common, config, args.output))
                rows.append(aggregate(task, packets))
                write_report(args.output, config, rows, status)
            status = '完成' if all(r['status'] == 'completed' for r in rows) else '存在未完成报告或评分'
    except BaseException as exc:
        status, error = '中断', safe_error(exc)
        raise
    finally:
        write_report(args.output, config, rows, status, error)
    print(json.dumps({'status': status, 'analysis_report': config['analysis_report']}, ensure_ascii=False))
    return 0 if status in {'完成', '已准备未调用模型'} else 1


if __name__ == '__main__':
    sys.exit(main())
