#!/usr/bin/env python3
"""独立评分第7节产物；只读冻结材料和人审判定，不把答案发送给模型。"""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND/'app')]
from eval.verification.scoring import validate_gold, score_task, aggregate, paired_difference, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--judgments', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True, help='运行前冻结的全部任务/组/重复计划，防止中断样本消失')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True, help='新增中文分析报告路径')
    parser.add_argument('--base')
    parser.add_argument('--treatment')
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error('结果或报告已存在，不能覆盖历史评估')
    dataset, runs, judgments = [json.loads(p.read_text()) for p in (args.dataset,args.runs,args.judgments)]
    validate_gold(dataset)
    tasks = {t['task_id']:t for t in dataset['tasks']}
    plan = json.loads(args.plan.read_text())
    if len({r['run_id'] for r in plan}) != len(plan):
        raise ValueError('计划运行ID重复')
    planned = {r['run_id']:r for r in plan}
    if any(r['run_id'] not in planned for r in runs):
        raise ValueError('存在计划外产物')
    existing = {r['run_id'] for r in runs}
    for scheduled in plan:
        if scheduled['run_id'] in existing:
            continue
        checkpoint = tasks[scheduled['task_id']].get('checkpoint_annotations', {}).get(scheduled['group'])
        if checkpoint is None:
            raise ValueError('未执行计划项缺少冻结检查点标注，不能将摘要/回读分母补为空')
        missing = {**scheduled, 'completed':False, 'final_body':'', 'summary_triggered':False,
                   'summary_delivered':False, 'missing_execution':True}
        missing.update(summary_triggered=checkpoint['summary_expected'],
                       summary_enabled=scheduled.get('summary_enabled', checkpoint['summary_expected']))
        runs.append(missing)
        judgments[scheduled['run_id']] = {'artifact_hash':digest(missing),'origin':'missing_execution',
                                          'summary_input_visible_fact_ids':checkpoint['summary_input_visible_fact_ids'],
                                          'summary_input_constraint_keys':checkpoint.get('summary_input_constraint_keys', []),
                                          'offloaded_required_fact_ids':checkpoint['offloaded_required_fact_ids']}
    rows, seen = [], set()
    for run in runs:
        key = run['run_id']
        if key in seen:
            raise ValueError('运行ID重复')
        seen.add(key)
        if any(run[k] != planned[key][k] for k in ('task_id','group','repeat')):
            raise ValueError('产物与运行计划不匹配')
        rows.append(score_task(tasks[run['task_id']],run,judgments[key]))
    result = {'date':datetime.now().isoformat(),'dataset_hash':digest(dataset),
              'annotation_version':dataset['annotation_version'],'origin':dataset['annotation_origin'],
              'tasks':rows, 'aggregate':aggregate(rows)}
    if args.base and args.treatment:
        result['paired'] = paired_difference(rows,args.base,args.treatment)
    args.output.mkdir(parents=True,exist_ok=False)
    raw = args.output/'核验链路评分结果.json'
    raw.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    failures = [{'task_id':r['task_id'],'group':r['group'],'failed':r['failed'],'read_failures':r['read_failures']} for r in rows if r['failed'] or r['read_failures']]
    args.report.write_text('# 核验链路独立评分结果与分析\n\n'
        +f"评估日期：{result['date']}。对象：摘要、回读与最终任务。标注版本：{dataset['annotation_version']}，来源：{dataset['annotation_origin']}。\n\n"
        +f"原始结果：[{raw.name}]({raw.resolve()})。输入：{args.dataset.resolve()}、{args.runs.resolve()}、{args.judgments.resolve()}。\n\n"
        +'## 核心指标与逐任务分析\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n'
        +'## 失败样例\n\n```json\n'+json.dumps(failures,ensure_ascii=False,indent=2)+'\n```\n\n'
        +'## 局限、结论与下一步\n\n已测量：按冻结人工判定及实际请求记录计算的上述指标；缺失费用保留null，不按零计。'
        +'间接推断：指标可定位摘要、回读及最终使用的不同损失。尚未评估：未提供的线上任务、真实成本和置信度校准；合成标注不能代表医疗准确率。'
        +'结论需结合逐任务失败和标注来源，不以NLI自评分证明准确率。下一步补齐盲审、费用及至少三次配对重复。\n')
    print(raw)


if __name__ == '__main__':
    main()
