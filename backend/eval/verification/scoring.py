"""第7节独立评分器：人工判定绑定产物哈希，固定事实分母，不调用生成模型。"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def ratio(correct, required):
    return {'correct': correct, 'required': required, 'value': correct / required if required else None}


def validate_gold(dataset):
    if dataset.get('annotation_origin') not in {'human', 'human_reviewed', 'synthetic_mechanism'}:
        raise ValueError('人工标注来源未声明；在线抽取清单不能作为真值')
    if not dataset.get('annotation_version'):
        raise ValueError('缺少冻结标注版本')
    seen = set()
    for task in dataset['tasks']:
        if task['task_id'] in seen:
            raise ValueError('任务ID重复')
        seen.add(task['task_id'])
        ids = [f['fact_id'] for f in task['facts']]
        if len(ids) != len(set(ids)):
            raise ValueError('固定事实清单ID重复')
        for f in task['facts']:
            text = task['sources'][f['source_id']]['text']
            if task['sources'][f['source_id']]['text_hash'] != digest(text):
                raise ValueError('人工标注的来源快照哈希不符')
            if type(f['start']) is not int or type(f['end']) is not int or not 0 <= f['start'] < f['end'] <= len(text) or text[f['start']:f['end']] != f['quote']:
                raise ValueError('人工标注来源区间无效')
        if task.get('split') not in {'development', 'selection', 'test'} or not task.get('trajectory_id'):
            raise ValueError('任务需记录轨迹分组和数据划分')
    groups = {}
    for task in dataset['tasks']:
        previous = groups.setdefault(task['trajectory_id'], task['split'])
        if previous != task['split']:
            raise ValueError('同一原始轨迹不能跨调参集和测试集')


def count_body(text):
    # 预先固定：传入正文，不含外置参考文献；所有非空白Unicode字符都计1，标点/数字/英文逐字符计。
    return sum(not ch.isspace() for ch in text)


def _judged(row, stage, key, allowed):
    values = row.get(stage, {}).get(key, [])
    if len(values) != len(set(values)) or not set(values) <= allowed:
        raise ValueError('评分条目重复或超出固定分母')
    return set(values)


def score_task(task, run, judgment):
    if judgment.get('artifact_hash') != digest(run):
        raise ValueError('人审判定与当前产物哈希不匹配')
    if judgment.get('origin') not in {'human', 'human_reviewed', 'synthetic_mechanism'} and not (
            judgment.get('origin') == 'missing_execution' and run.get('missing_execution') is True
            and run.get('completed') is False and not run.get('summary_delivered')):
        raise ValueError('不接受核验器自评分作为真值')
    facts = {f['fact_id']: f for f in task['facts']}
    # 实际输入范围必须来源于记录；这里由预先冻结的检查点/输入关联确定，不从输出反推。
    visible = set(judgment.get('summary_input_visible_fact_ids', []))
    if not visible <= facts.keys():
        raise ValueError('摘要可见事实ID无效')
    summary_applicable = run.get('summary_triggered', False) and run.get('summary_enabled', False)
    summary_ids = visible if summary_applicable else set()
    summary_ok = _judged(judgment, 'summary', 'correct_fact_ids', summary_ids)
    final_ok = _judged(judgment, 'final', 'correct_fact_ids', set(facts))
    final_sources = _judged(judgment, 'final', 'correct_source_fact_ids', set(facts)) & final_ok
    # 没有实际可用摘要，保留率计0；真实回退正文/旧摘要可独立绑定产物接受人审。
    if not run.get('summary_delivered', False):
        summary_ok = set()
    completed = run.get('completed') is True
    if not completed:
        final_ok, final_sources = set(), set()
    summary_fields = {f'{fid}:{i}' for fid in summary_ids for i, _ in enumerate(facts[fid].get('fields', []))}
    final_fields = {f'{fid}:{i}' for fid in facts for i, _ in enumerate(facts[fid].get('fields', []))}
    sf = _judged(judgment, 'summary', 'correct_field_ids', summary_fields)
    ff = _judged(judgment, 'final', 'correct_field_ids', final_fields)
    if not run.get('summary_delivered', False):
        sf = set()
    if not completed:
        ff = set()
    targets = set(judgment.get('offloaded_required_fact_ids', []))
    if not targets <= facts.keys():
        raise ValueError('回读目标不在固定任务清单')
    retrieved, sent = set(), set()
    for fid in targets:
        fact = facts[fid]
        expected_hash = task['sources'][fact['source_id']]['text_hash']
        for read in run.get('reads', []):
            if read['source_id'] != fact['source_id'] or read.get('source_text_hash') != expected_hash:
                continue
            if not (read['start'] <= fact['start'] and read['end'] >= fact['end']):
                continue
            source_text = task['sources'][fact['source_id']]['text']
            if read.get('text_hash') != digest(source_text[read['start']:read['end']]):
                raise ValueError('回读内容哈希与原文区间不符')
            retrieved.add(fid)
            for request in run.get('requests', []):
                if request.get('execution_status') != 'completed' or request['sequence'] > fact.get('needed_before_sequence', math.inf):
                    continue
                if request.get('request_hash') != digest(request['request']):
                    raise ValueError('实际请求哈希不符')
                from service.verification_trace import read_links
                for link in read_links(request['request']['messages']):
                    if all(link.get(k) == read.get(k) for k in ('read_id','source_id','start','end','source_text_hash','text_hash')):
                        sent.add(fid)
    constraints = {c['key'] for c in task.get('constraints', [])}
    respected = _judged(judgment, 'final', 'satisfied_constraint_keys', constraints) if completed else set()
    length = count_body(run.get('final_body', ''))
    for c in task.get('constraints', []):
        if c.get('type') == 'length':
            respected.discard(c['key'])
            valid = length <= c['value'] if c['mode'] == 'max' else abs(length - c['value']) <= c.get('tolerance', 0)
            if completed and valid:
                respected.add(c['key'])
    output_fields = judgment.get('summary', {}).get('output_fields', []) if run.get('summary_delivered') else []
    if len({f['id'] for f in output_fields}) != len(output_fields):
        raise ValueError('输出字段标注重复')
    summary_sources = _judged(judgment, 'summary', 'correct_source_fact_ids', summary_ids) & summary_ok
    visible_constraints = set(judgment.get('summary_input_constraint_keys', []))
    if not visible_constraints <= constraints:
        raise ValueError('摘要输入约束标注不在有效任务要求中')
    summary_constraints = _judged(judgment, 'summary', 'satisfied_constraint_keys', visible_constraints)
    if not run.get('summary_delivered', False):
        summary_constraints = set()
    claim_labels = judgment.get('summary', {}).get('output_claims', []) if run.get('summary_delivered') else []
    if len({c['id'] for c in claim_labels}) != len(claim_labels) or any(
            c.get('support') not in {'supported','contradiction','unsupported','unknown'} for c in claim_labels):
        raise ValueError('输出claim标注重复或支持状态无效')
    judgeable = [c for c in claim_labels if c['support'] != 'unknown']
    topics = [c for c in claim_labels if c.get('topic') in {'correct','mismatch'}]
    conflicts = set(judgment.get('summary_input_conflict_ids', []))
    if not conflicts <= {c['id'] for c in task.get('conflicts', [])}:
        raise ValueError('摘要冲突标注不在任务清单中')
    preserved_conflicts = _judged(judgment,'summary','preserved_conflict_ids',conflicts) if run.get('summary_delivered') else set()
    metrics = {
        'summary_retention': ratio(len(summary_ok), len(summary_ids)),
        'summary_field_retention': ratio(len(sf), len(summary_fields)),
        'summary_output_field_accuracy': ratio(sum(f['status']=='correct' for f in output_fields), len(output_fields)),
        'summary_output_field_unknown': ratio(sum(f['status']=='unknown' for f in output_fields),len(output_fields)),
        'summary_source_retention': ratio(len(summary_sources),len(summary_ids)),
        'summary_constraint_retention': ratio(len(summary_constraints),len(visible_constraints) if summary_applicable else 0),
        'summary_unsupported_claims': ratio(sum(c['support'] in {'contradiction','unsupported'} for c in judgeable),len(judgeable)),
        'summary_unknown_claims': ratio(len(claim_labels)-len(judgeable),len(claim_labels)),
        'summary_topic_mismatch': ratio(sum(c['topic']=='mismatch' for c in topics),len(topics)),
        'summary_conflict_retention': ratio(len(preserved_conflicts),len(conflicts) if summary_applicable else 0),
        'read_target_retrieval': ratio(len(retrieved), len(targets)),
        'read_target_visible': ratio(len(sent), len(targets)),
        'read_then_correct_use': ratio(len(sent & final_ok), len(sent)),
        'final_fact_use': ratio(len(final_ok), len(facts)),
        'final_field_use': ratio(len(ff), len(final_fields)),
        'final_source_traceability': ratio(len(final_sources), len(facts)),
        'constraint_compliance': ratio(len(respected), len(constraints)),
        'all_constraints_passed': ratio(int(completed and respected == constraints), 1),
        'task_success': ratio(int(completed and judgment.get('task_accepted') is True and respected == constraints), 1),
    }
    return {'task_id':task['task_id'], 'group':run['group'], 'repeat':run['repeat'],
            'failed':not completed, 'summary_applicable':summary_applicable, 'metrics':metrics,
            'annotation_incomplete':[key for key in ('output_fields','output_claims')
                                     if run.get('summary_delivered') and key not in judgment.get('summary',{})],
            'body_characters':length, 'artifact_hash':digest(run), 'cost':run.get('cost', {}),
            'read_failures':[{'fact_id':fid,'reason':'not_retrieved' if fid not in retrieved else 'not_sent_before_needed'} for fid in sorted(targets - sent)]}


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['group'], []).append(row)
    result = {}
    for group, items in groups.items():
        metrics = {}
        for key in items[0]['metrics']:
            values = [r['metrics'][key] for r in items]
            per_task = {}
            for r in items:
                value = r['metrics'][key]['value']
                if value is not None:
                    per_task.setdefault(r['task_id'], []).append(value)
            metrics[key] = {'micro':ratio(sum(v['correct'] for v in values), sum(v['required'] for v in values)),
                            'macro':statistics.mean(statistics.mean(v) for v in per_task.values()) if per_task else None,
                            'not_applicable':sum(v['required']==0 for v in values)}
        durations = sorted(r['cost']['duration_seconds'] for r in items if r['cost'].get('duration_seconds') is not None)
        costs = [r['cost'].get('usd') for r in items]
        result[group] = {'metrics':metrics, 'failed':sum(r['failed'] for r in items), 'runs':len(items),
                         'cost_usd':sum(costs) if all(c is not None for c in costs) else None,
                         'unpriced_runs':sum(c is None for c in costs),
                         'p50_seconds':statistics.median(durations) if durations else None,
                         'p95_seconds':durations[max(0, math.ceil(.95*len(durations))-1)] if durations else None}
    return result


def paired_difference(rows, base, treatment, metric='final_fact_use', seed=42):
    groups = {}
    for r in rows:
        key = (r['task_id'], r['repeat'])
        if (r['group'], key) in groups:
            raise ValueError('重复运行标识')
        groups[r['group'], key] = r['metrics'][metric]['value']
    tasks, missing = {}, []
    keys = {key for g, key in groups if g in {base, treatment}}
    for key in sorted(keys):
        a, b = groups.get((base,key)), groups.get((treatment,key))
        if a is None or b is None:
            missing.append(key)
        else:
            tasks.setdefault(key[0], []).append(b-a)
    means = [statistics.mean(v) for v in tasks.values()]
    rng = random.Random(seed)
    bootstrap = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(2000)) if len(means)>1 else []
    return {'mean_task_difference':statistics.mean(means) if means else None,
            'task_cluster_ci95':[bootstrap[49],bootstrap[1949]] if bootstrap else None,
            'tasks':len(tasks),'missing_pairs':missing,'repeats_per_task':{k:len(v) for k,v in tasks.items()}}


def nli_metrics(rows, entailment_threshold=.85, contradiction_threshold=.5):
    # 标签来自人工；执行失败单列但仍保留在总分母，并计入支持样本误拒。
    if any(r.get('label') not in {'supported','contradiction','insufficient_support'} for r in rows):
        raise ValueError('NLI人工标签缺失')
    def accept(row):
        scores = row.get('scores') or {}
        return row.get('execution_status') == 'completed' and scores.get('entailment',0)>=entailment_threshold and scores.get('contradiction',1)<contradiction_threshold
    unsupported = [r for r in rows if r['label']!='supported']
    supported = [r for r in rows if r['label']=='supported']
    return {'false_accept':ratio(sum(accept(r) for r in unsupported),len(unsupported)),
            'false_reject':ratio(sum(not accept(r) for r in supported),len(supported)),
            'availability':ratio(sum(r.get('execution_status')=='completed' for r in rows),len(rows)),
            'unknown':ratio(sum(r.get('execution_status')=='completed' and not accept(r) and (r.get('scores') or {}).get('contradiction',0)<contradiction_threshold for r in rows),len(rows))}
