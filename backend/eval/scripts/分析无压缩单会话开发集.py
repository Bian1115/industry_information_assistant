#!/usr/bin/env python3
"""从真实采集原始事件汇总用户轮数、模型调用及上下文边界，不调用模型。"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def analyze(run):
    manifest = read(run / '运行清单.json', {})
    groups = read(run / '采集进度.json', [])
    turns = [row for group in groups for row in group['turns']]
    records, observations, failures, agent_stats = [], [], [], defaultdict(list)
    for row in turns:
        directory = run / '合并开发集30轮' / f"第{row['turn_id']}轮"
        history = read(directory / '工作历史.json', [])
        context = read(directory / '会话任务输入.json', {})
        views = [r for r in history if r['kind'] == 'context_view']
        for r in views:
            p = r['payload']
            v = {'turn_id': row['turn_id'], 'agent': r['agent_name'], 'call_id': r['call_id'],
                 **{k: p.get(k) for k in ['tokens', 'budget', 'tokens_before_compression',
                    'tokens_after_input_history', 'history_replay', 'compression_enabled',
                    'stages', 'summary_attempted', 'summary_adopted', 'shortened_source_ids']}}
            observations.append(v)
            agent_stats[r['agent_name']].append(v)
        errors = [r for r in history if r['kind'] in {'llm_error', 'tool_error', 'summary_error'}]
        if row.get('error') or not row.get('completed'):
            failures.append({'turn_id': row['turn_id'], 'source_id': row['source_id'],
                'run_id': row['run_id'], 'error_type': row.get('error_type'), 'error': row.get('error'),
                'phase': row.get('phase'), 'completion_reason': row.get('completion_reason'),
                'over_budget_views': [v for v in observations if v['turn_id'] == row['turn_id']
                                      and v['tokens'] > v['budget']],
                'record_errors': [{'kind': r['kind'], 'agent': r['agent_name'], 'payload': r['payload']}
                                  for r in errors], 'directory': str(directory)})
        row['analysis'] = {'records': dict(Counter(r['kind'] for r in history)),
            'conversation_messages': len(context.get('conversation_context', [])),
            'conversation_tokens': context.get('token_counts', {}).get('conversation'),
            'session_topic_relation': context.get('session_task', {}).get('topic_relation'),
            'max_input_tokens': max((v['payload']['tokens'] for v in views), default=0),
            'context_views': len(views), 'provider_requests': sum(r['kind'] == 'llm_request' for r in history),
            'model_outputs': sum(r['kind'] == 'llm_output' for r in history)}
        records.extend(history)
    counts = Counter(r['kind'] for r in records)
    writer_outputs = [r for r in records if r['kind'] == 'llm_output' and r['agent_name'] == 'LeadWriter']
    used = [request for group in groups for request in group.get('cost', {}).get('requests', [])]
    usage = defaultdict(lambda: {'requests': 0, 'usage_available': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
    for request in used:
        m = usage[request['model']]
        m['requests'] += 1
        u = request.get('usage') or {}
        if u.get('prompt_tokens') is not None and u.get('completion_tokens') is not None:
            m['usage_available'] += 1
            m['prompt_tokens'] += u['prompt_tokens']
            m['completion_tokens'] += u['completion_tokens']
    by_source = []
    for source_id in manifest.get('source_ids', [8, 25, 26, 27, 28]):
        selected = [r for r in turns if r['source_id'] == source_id]
        by_source.append({'source_id': source_id, 'planned_turns': 6, 'attempted_turns': len(selected),
            'completed_turns': sum(r['completed'] for r in selected),
            'provider_requests': sum(r['analysis']['provider_requests'] for r in selected),
            'max_input_tokens': max((r['analysis']['max_input_tokens'] for r in selected), default=0)})
    stats = {a: {'views': len(v), 'history_replay_views': sum(x['history_replay'] is True for x in v),
                 'max_input_tokens': max(x['tokens'] for x in v),
                 'over_budget_views': sum(x['tokens'] > x['budget'] for x in v)} for a, v in agent_stats.items()}
    consecutive = 0
    for row in turns:
        if not row['completed']:
            break
        consecutive += 1
    return {'analysis_date': datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
        'status': manifest.get('status'), 'policy_hash': manifest.get('context_policy', {}).get('config_hash'),
        'compression_enabled': manifest.get('compression_enabled'),
        'session_count': len({g['session_id'] for g in groups}), 'planned_turns': 30,
        'attempted_turns': len(turns), 'completed_turns': sum(r['completed'] for r in turns),
        'consecutive_completed_turns': consecutive, 'by_source': by_source, 'by_agent': stats,
        'record_counts': dict(counts), 'provider_requests': counts['llm_request'],
        'model_outputs': counts['llm_output'], 'context_views': len(observations),
        'writer_completed_calls': len(writer_outputs),
        'writer_input_tokens': [v['tokens'] for v in observations if v['agent'] == 'LeadWriter'],
        'compression_stages': sum(bool(v['stages']) for v in observations),
        'summary_attempts': sum(v['summary_attempted'] is True for v in observations),
        'summary_adoptions': sum(v['summary_adopted'] is True for v in observations),
        'max_input_tokens': max((v['tokens'] for v in observations), default=0),
        'over_budget_views': [v for v in observations if v['tokens'] > v['budget']],
        'compression_disabled_observed': bool(observations) and all(v['compression_enabled'] is False for v in observations),
        'uncompressed_input_observed': bool(observations) and all(v['tokens'] == v['tokens_before_compression'] for v in observations),
        'usage_by_model': dict(usage), 'ledger_requests': len(used), 'turns': turns,
        'failures': failures, 'changed_source_files': manifest.get('changed_source_files', [])}


def report(run, stats, destination):
    config = read(run / '配置快照.json')['config']
    topics = {8: '机器学习优化材料元素配比', 25: '中性粒细胞与脑缺血',
              26: '代谢与表观遗传互作', 27: 'AI与人类心理咨询协作', 28: '大模型与药物影响评估'}
    s = stats
    terminal_context_error = next((f for f in s['failures']
                                   if 'context_budget_exceeded' in (f.get('error') or '')), None)
    if s['consecutive_completed_turns'] == 30:
        conclusion = '本次同一会话无上下文压缩连续交付30轮；只能认定至少30轮，尚未测出最大轮数。'
    elif terminal_context_error and terminal_context_error['over_budget_views']:
        first = terminal_context_error['over_budget_views'][-1]
        conclusion = (f"本次同一会话连续交付{s['consecutive_completed_turns']}轮，第{first['turn_id']}轮"
            f"的{first['agent']}组装输入达到{first['tokens']:,} token，超过{first['budget']:,} token预算。"
            '这是当前固定配置与实际轨迹的观测边界。')
    else:
        conclusion = (f"本次连续交付{s['consecutive_completed_turns']}轮，尝试{s['attempted_turns']}轮后停止。"
            '没有观察到上下文超预算遥测，不能把停止位置认定为无压缩的最大轮数。')
    lines = ['# 无上下文压缩单会话开发集5题复跑结果及分析', '', conclusion, '',
        f"评估日期：{s['analysis_date']}（Asia/Singapore）。状态：{s['status']}。", '',
        '## 评估对象与配置口径', '',
        '固定20260927开发集原题8→25→26→27→28，每题6轮，串行放在同一新会话中执行；每个用户轮次创建独立研究run。'
        '轮数指成功收到非空research_complete正文的用户轮次，研究内部模型调用另计。'
        '使用真实生产会话任务构建器、联网搜索和数据库检查点，首个未交付轮次停止整个连跑。', '',
        f"总开关`CONTEXT_COMPRESSION_ENABLED=0`，冻结策略`compression_enabled={str(config['compression_enabled']).lower()}`；策略哈希`{s['policy_hash']}`。",
        f"单次输入预算：{config['stage_budgets']['LeadWriter']['max_input']:,} token；配置窗口{config['stage_budgets']['LeadWriter']['context_window']:,} token，"
        f"安全余量{config['stage_budgets']['LeadWriter']['safety_tokens']} token；输出预留{json.dumps(config['output_limits'],ensure_ascii=False)}。",
        '总开关关闭资料引用化、工具与旧输出外置、历史摘要生成/注入、会话token及消息数裁剪，并绕过压缩缓存全量读取PostgreSQL。'
        '当前任务契约保护、Scout提取分批、内容证据结构化、会话话题判断及成果选择仍按原生产逻辑执行。'
        '话题判断为new时执行输入会清除旧主题材料，这属于原有任务边界；本次没有改变为全主题累计研究。',
        '智能体工作历史仍按研究run隔离，不跨用户轮次注入；同一会话连续增长的是会话原始对话。'
        '智能体历史注入范围沿用原部署的DeepScout和LeadWriter，关闭总开关不会关闭其历史注入。',
        '真实采集整条对话保护预算为86400秒、10000次模型请求、每个冻结模型300000000 token；'
        '单次研究时间预算沿用部署2400秒、研究累计token上限0（关闭），业务返工上限2。'
        '上述均为运行保护或项目配置，不是供应商最大窗口声明。', '',
        '## 核心指标', '',
        '| 指标 | 结果 |', '| --- | ---: |',
        f"| 实际会话数 | {s['session_count']} |", f"| 连续交付用户轮数 | {s['consecutive_completed_turns']}/30 |",
        f"| 已尝试用户轮数 | {s['attempted_turns']} |", f"| 研究provider请求事件 | {s['provider_requests']} |",
        f"| 研究llm_output事件 | {s['model_outputs']} |", f"| 账本请求（含会话判断） | {s['ledger_requests']} |",
        f"| Writer已返回章节调用 | {s['writer_completed_calls']} |",
        f"| 输入视图数 | {s['context_views']} |", f"| 最大组装输入token | {s['max_input_tokens']:,} |",
        f"| 超预算输入视图 | {len(s['over_budget_views'])} |", f"| 压缩处理阶段 | {s['compression_stages']} |",
        f"| 摘要尝试/采用 | {s['summary_attempts']}/{s['summary_adoptions']} |", '',
        f"实际视图均记录压缩关闭：{s['compression_disabled_observed']}；处理前后token一致：{s['uncompressed_input_observed']}。"
        '未发出provider请求的超预算组装点也计入输入视图。'
        '正文质量检查中的局部超限会被捕获并继续，因此首次超限与整轮停止位置必须区分。', '',
        f"Writer按时间输入token为{s['writer_input_tokens']}；5个章节调用已返回，最终整合调用在发送前被本地预算拒绝。"
        '这些章节原始输出留在工作历史.json；write节点整体失败，最新研究检查点没有提交章节或最终正文，不能视为完整报告交付。', '',
        '## 逐题与逐轮分析', '', '| 原题 | 主题 | 已尝试 | 已交付 | 模型请求 | 最大输入token |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for item in s['by_source']:
        lines.append(f"| {item['source_id']} | {topics[item['source_id']]} | {item['attempted_turns']}/6 | "
            f"{item['completed_turns']}/6 | {item['provider_requests']} | {item['max_input_tokens']:,} |")
    lines += ['', '| 会话轮 | 原题/题内轮 | 交付 | 进入会话任务的消息/token | 最大研究输入token | 研究请求 |',
              '| --- | --- | --- | --- | ---: | ---: |']
    for row in s['turns']:
        a = row['analysis']
        lines.append(f"| {row['turn_id']} | {row['source_id']}/{row['source_turn_id']} | {row['completed']} | "
            f"{a['conversation_messages']}/{a['conversation_tokens']} | {a['max_input_tokens']:,} | {a['provider_requests']} |")
    lines += ['', '## 逐智能体输入分析', '', '| 智能体 | 输入视图 | 注入研究内历史 | 最大输入token | 超预算 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for agent, a in s['by_agent'].items():
        lines.append(f"| {agent} | {a['views']} | {a['history_replay_views']} | {a['max_input_tokens']:,} | {a['over_budget_views']} |")
    if s['over_budget_views']:
        lines += ['', '局部超限输入原始样例：', '']
        for v in s['over_budget_views'][:10]:
            lines.append(f"- 用户第{v['turn_id']}轮，{v['agent']}调用`{v['call_id']}`："
                         f"{v['tokens']:,}/{v['budget']:,} token；stages={v['stages']}。")
    lines += ['', '## 失败样例', '']
    if not s['failures']:
        lines.append('本次采集没有未交付轮次；未开展报告内容错误人工标注。')
    for failure in s['failures']:
        lines += [f"- 会话第{failure['turn_id']}轮，原题{failure['source_id']}；run_id=`{failure['run_id']}`。"
                  f"错误类型`{failure['error_type']}`；错误原文：{failure['error']}。",
                  f"  最近已提交检查点阶段：{failure['phase']}，不代表实际失败节点；实际Writer整合异常位置见运行日志。"
                  f"原始目录：[{Path(failure['directory']).name}]({failure['directory']})。"]
        for view in failure['over_budget_views']:
            lines.append(f"  {view['agent']}调用`{view['call_id']}`：{view['tokens']:,}/{view['budget']:,} token，"
                         f"history_replay={view['history_replay']}，stages={view['stages']}。")
    lines += ['', '## 用量与验证', '',
        '账本usage按模型原样汇总；未返回usage的失败调用不能当成零token。未冻结价格，未计算费用。', '',
        '```json', json.dumps(s['usage_by_model'], ensure_ascii=False, indent=2), '```', '',
        '最终回归：94项，0失败0错误，2项按既有条件跳过；前次机制回归80项与空长期记忆修复13项另存原始日志。'
        '首次机制样例错误和Python3.11依赖缺失已保留，修正后采用项目正式Python3.13环境。',
        '第一次真实尝试在无长期记忆的新身份上仍访问Milvus并持续等待，尚未进入研究模型；已单独落盘中断分析。'
        '复跑只增加无可用记忆时从PostgreSQL确认空候选并跳过向量服务的检查，未删减任何实际记忆内容。',
        '本地历史表应用已有0011迁移，旧上下文109条、消息8596条数量保持。仅独立应用必要迁移，'
        'alembic版本标记保持20260919_0008；将来统一迁移须先核对这些已独立应用的历史表结构。',
        f"采集期间快照源码变更：{s['changed_source_files']}。", '',
        '采集结束后补强会话层显式冻结文件的开关优先级，并完成上述最终回归。'
        '此次真实轨迹使用采集时保存的代码快照；采集时和补强后的有效开关值均为false，未更改实际运行结果。', '',
        '## 局限性与边界', '',
        '已测量：本次真实交付轮数、调用及输入token、压缩开关遥测、具体失败位置，以及离线机制回归。'
        '输入token由本地Qwen tokenizer测量，实际用量仅以供应商返回usage为准。',
        '间接推断：若失败视图超过硬输入预算，可以解释本次停止机制；不能据此推断其他模型、主题、预算或随机采样下也在同一轮停止。',
        '尚未评估：压缩开启同源对照、输出内容质量、事实保留率、引用质量、正式测试集和稳定最大轮数。'
        '实时搜索、模型非确定性、五题目的性抽样与现有run历史隔离限制泛化。若第一题内即停止，其他题未执行。', '',
        '## 结论与下一步建议', '', conclusion,
        '当前部署压缩总开关保持关闭。需要恢复时在backend/.env设`CONTEXT_COMPRESSION_ENABLED=1`并启动新进程；'
        '已有冻结研究继续原配置。下一步使用本次同源历史回放打开压缩，比较超限调用是否可发送及内容保留；'
        '再完整复跑，才能判断压缩对可持续轮数与质量的贡献。', '', '## 原始结果路径', '']
    for name in ['无压缩轮数统计.json', '运行清单.json', '配置快照.json', '开发集5题.jsonl', '采集进度.json', '合并开发集30轮/逐轮结果.json']:
        p = run / name
        lines.append(f"- [{name}]({p})")
    for name in ['复跑运行日志.txt', '机制回归结果.json', '机制回归日志.txt', '空记忆修复回归结果.json',
                 '最终回归结果.json', '最终回归日志.txt', '首轮中断分析.json', '历史表迁移结果.json', '首次验证问题.json']:
        p = run.parent / name
        lines.append(f"- [{name}]({p})")
    destination.write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    stats = analyze(args.run)
    if args.report and stats['status'] == 'running':
        p.error('尚在运行，仅可输出机器统计；终态后才写最终报告')
    (args.run / '无压缩轮数统计.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.report:
        report(args.run, stats, args.report)
    print(json.dumps({k: stats[k] for k in ['status', 'consecutive_completed_turns', 'attempted_turns',
          'provider_requests', 'max_input_tokens', 'over_budget_views']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
