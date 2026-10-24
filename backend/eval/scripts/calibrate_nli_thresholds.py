#!/usr/bin/env python3
"""用独立人工标签在选择集挑选NLI阈值；测试集仅在选择后报告，不反向调参。"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
BACKEND=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BACKEND))
from eval.verification.scoring import nli_metrics,digest


def select_thresholds(data, max_false_accept=.05):
    if data.get('annotation_origin') not in {'human','human_reviewed','synthetic_mechanism'}:
        raise ValueError('需要独立人工标注，不能使用NLI自己的标签')
    if not data.get('annotation_version') or not data.get('model_id') or not data.get('revision'):
        raise ValueError('缺少标注或模型冻结版本')
    trajectories={}
    for row in data['pairs']:
        if row['split'] not in {'development','selection','test'}:
            raise ValueError('未知划分')
        previous=trajectories.setdefault(row['trajectory_id'],row['split'])
        if previous != row['split']:
            raise ValueError('同一轨迹跨选择/测试集泄漏')
    for row in data['pairs']:
        if not row.get('premise') or not row.get('claim') or row.get('pair_hash') != digest({'premise':row['premise'],'claim':row['claim']}):
            raise ValueError('引用对需保留冻结的完整输入及哈希')
        if row['execution_status'] == 'completed':
            scores=row.get('scores',{})
            if set(scores) != {'entailment','neutral','contradiction'} or any(not math.isfinite(v) or not 0<=v<=1 for v in scores.values()) or not math.isclose(sum(scores.values()),1,abs_tol=1e-5):
                raise ValueError('三类预测分数无效')
    selection=[r for r in data['pairs'] if r['split']=='selection']
    if not any(r['label']=='supported' for r in selection) or not any(r['label']!='supported' for r in selection):
        raise ValueError('选择集须同时包含支持与不支持标签')
    candidates=[]
    for entail in [.7,.75,.8,.85,.9,.95]:
        for contra in [.5,.6,.7]:
            metrics=nli_metrics(selection,entail,contra)
            candidates.append({'entailment_threshold':entail,'contradiction_threshold':contra,'metrics':metrics})
    feasible=[c for c in candidates if c['metrics']['false_accept']['value']<=max_false_accept]
    best=min(feasible,key=lambda c:(c['metrics']['false_reject']['value'],c['metrics']['false_accept']['value'],-c['entailment_threshold'],c['contradiction_threshold'])) if feasible else None
    test=[r for r in data['pairs'] if r['split']=='test']
    bins=[]
    label_map={'supported':'entailment','contradiction':'contradiction','insufficient_support':'neutral'}
    for low,high in zip([0,.5,.6,.7,.8,.9],[.5,.6,.7,.8,.9,1.000001]):
        items=[r for r in test if r['execution_status']=='completed' and low<=max(r['scores'].values())<high]
        bins.append({'lower':low,'upper':min(high,1),'count':len(items),
                     'correct':sum(max(r['scores'],key=r['scores'].get)==label_map[r['label']] for r in items)})
    return {'selected':best,'selection_candidates':candidates,
            'heldout_test':nli_metrics(test,best['entailment_threshold'],best['contradiction_threshold']) if best and test else None,
            'threshold_version':'selection-'+digest(data)[:16], 'test_confidence_bins':bins,
            'calibration_kind':'decision_threshold_selection_not_probability_calibration'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.report.exists():parser.error('不得覆盖历史结果')
    data=json.loads(args.labels.read_text());result=select_thresholds(data)
    result.update(date=datetime.now().isoformat(),input_hash=digest(data),annotation_version=data['annotation_version'],model_id=data['model_id'],revision=data['revision'])
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'NLI阈值选择结果.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text('# NLI阈值选择结果及分析\n\n'+f"评估日期：{result['date']}。对象：冻结NLI引用对。标注来源：{data['annotation_origin']}，版本{data['annotation_version']}。\n\n"
        +f'原始标注：{args.labels.resolve()}。原始结果：[{path.name}]({path.resolve()})。\n\n'
        +'## 核心指标与逐类分析\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n'
        +'支持类看误拒率，不支持类看误放率，执行异常仍计入可用率及相应分母；逐对失败样例可按原始标签与选定门槛重查。\n\n'
        +'## 局限与下一步\n\n已测量：本标注集上的阈值选择及锁定测试结果（如有）。间接推断：仅在相同分布可能适用。'
        +'尚未评估：跨领域泛化和概率校准；小样本/合成标签不能代表医学可靠性。没有合格候选时不推荐阈值。'
        +'结论：结果是候选决策阈值，不自动改运行配置。下一步盲审失败样例并冻结阈值版本。\n')
    print(path)


if __name__=='__main__':main()
