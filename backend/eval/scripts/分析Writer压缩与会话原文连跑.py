#!/usr/bin/env python3
"""汇总Writer压缩、会话原文的真实连跑；不调用模型。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from eval.scripts.分析无压缩单会话开发集 import analyze, read


def build_report(run, destination):
    stats = analyze(run)
    config = read(run / '配置快照.json')['config']
    manifest = read(run / '运行清单.json')
    stats['session_compression_enabled'] = manifest.get('session_compression_enabled')
    stats['writer_methods'] = config['agent_compression']['LeadWriter']
    session_checks = []
    for turn in stats['turns']:
        directory = run / '合并开发集30轮' / f"第{turn['turn_id']}轮"
        context = read(directory / '会话任务输入.json', {})
        session = read(directory / '会话摘要状态.json', {})
        session_checks.append({
            'turn_id': turn['turn_id'], 'source_id': turn['source_id'],
            'history_metrics': context.get('history_metrics', {}),
            'task_metrics': context.get('task_metrics', {}),
            'session_summary_metrics': context.get('session_summary_metrics', {}),
            'persisted_summary_version': session.get('semantic_context_version'),
            'persisted_summary_present': bool(session.get('semantic_context_summary')),
            'persisted_message_count': len(read(directory / '会话原始消息.json', [])),
        })
    stats['session_checks'] = session_checks
    stats['session_raw_history_observed'] = bool(session_checks) and all(
        check['history_metrics'].get('original_tokens') == check['history_metrics'].get('prepared_tokens')
        and check['history_metrics'].get('semantic_context_version') == 0
        and check['history_metrics'].get('summarized_messages') == 0
        and check['persisted_summary_version'] == 0
        and not check['persisted_summary_present'] for check in session_checks)
    writer_views = [view for turn in stats['turns'] for view in turn.get('compression_points', [])
                    if view.get('agent_name') == 'LeadWriter']
    stats['writer_compression_stage_counts'] = dict(Counter(
        stage for view in writer_views for stage in (view.get('stages') or [])))
    raw_path = run / 'Writer压缩与会话原文轮数统计.json'
    raw_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    completed = stats['consecutive_completed_turns']
    conclusion = f'本次在同一会话连续交付 {completed}/30 轮，尝试 {stats["attempted_turns"]} 轮。'
    if completed == 30:
        conclusion += '已测得至少30轮；本开发集未测出最大轮数。'
    elif stats['failures']:
        failure = stats['failures'][0]
        conclusion += f'第{failure["turn_id"]}轮未交付，原因见失败样例。停止位置不能直接认定为模型原生容量上限。'
    else:
        conclusion += '采集尚未完成，不能给出最终边界。'
    lines = ['# Writer三项压缩与会话原文单会话开发集5题结果及分析', '', conclusion, '',
        f'评估日期：{stats["analysis_date"]}（Asia/Singapore）。运行状态：{stats["status"]}。', '',
        '## 评估对象与数据配置口径', '',
        '固定20260927开发集原题8→25→26→27→28，每题6轮，共30轮，放在一个新的隔离评测会话中串行运行。'
        '每个用户轮次独立研究run，智能体工作历史按run隔离。首个未收到非空research_complete正文的轮次停止连跑。',
        f'总开关开启；冻结会话层开关为{stats["session_compression_enabled"]}；Writer方法为`{json.dumps(stats["writer_methods"], ensure_ascii=False)}`。',
        '本轮仅恢复Writer的资料引用化、旧输出/工具外置及语义摘要；DeepScout保留历史注入，三种压缩方法关闭，其他智能体沿用未注入研究历史。'
        'Writer语义核验关闭。会话层从PG保留全部原始对话，不使用旧摘要、裁剪缓存或消息/token裁剪。'
        '摘要开关开启不等于每次调用都生成摘要：引用化及外置处理后，达到摘要阈值才触发。',
        f'研究调用输入硬上限{config["stage_budgets"]["LeadWriter"]["max_input"]:,} token；配置窗口'
        f'{config["stage_budgets"]["LeadWriter"]["context_window"]:,}，输出预留{config["output_limits"]["LeadWriter"]}，'
        f'安全余量{config["stage_budgets"]["LeadWriter"]["safety_tokens"]}。这些是项目配置，并非供应商原生窗口测量。',
        f'整条对话保护预算：`{json.dumps(manifest["budget"], ensure_ascii=False)}`；业务返工上限2；单篇研究2400秒，累计token限额0。',
        f'冻结策略哈希：`{stats["policy_hash"]}`。真实搜索、证据抽取、任务路由、审稿和引用核验按生产逻辑运行。', '',
        '## 核心指标', '', '| 指标 | 结果 |', '| --- | ---: |',
        f'| 实际会话数 | {stats["session_count"]} |',
        f'| 连续交付用户轮数 | {completed}/30 |', f'| 已尝试轮数 | {stats["attempted_turns"]} |',
        f'| 研究provider请求事件 | {stats["provider_requests"]} |',
        f'| 账本请求（包括会话任务判断） | {stats["ledger_requests"]} |',
        f'| Writer成功调用 | {stats["writer_completed_calls"]} |',
        f'| 压缩处理输入视图 | {stats["compression_stages"]} |',
        f'| 智能体摘要尝试/采用 | {stats["summary_attempts"]}/{stats["summary_adoptions"]} |',
        f'| 研究本地超预算视图 | {len(stats["over_budget_views"])} |',
        f'| 会话全量原文且无摘要观测 | {stats["session_raw_history_observed"]} |', '',
        f'Writer实际输入token：`{stats["writer_input_tokens"]}`。',
        f'账本模型使用：`{json.dumps(stats["usage_by_model"], ensure_ascii=False)}`。', '',
        '## 逐题分析', '', '| 原题 | 计划轮数 | 尝试 | 交付 | 研究请求 | 最大组装输入token |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for item in stats['by_source']:
        lines.append(f'| {item["source_id"]} | 6 | {item["attempted_turns"]} | {item["completed_turns"]} | {item["provider_requests"]} | {item["max_input_tokens"]:,} |')
    lines.extend(['', '## 逐轮历史与路由', '',
        '| 轮次/原题 | 路由读取消息数 | 路由历史token | 执行历史消息数 | 执行历史token | 路由状态 |',
        '| --- | ---: | ---: | ---: | ---: | --- |'])
    for check in session_checks:
        h = check['history_metrics']
        lines.append(f'| {check["turn_id"]}/{check["source_id"]} | {h.get("router_messages")} | {h.get("router_tokens")} | {h.get("execution_messages")} | {h.get("execution_tokens")} | {check["task_metrics"].get("status", "首轮未调用")} |')
    lines.extend(['', '路由可读取全部会话历史；判定新主题后，研究执行仍过滤旧主题。编辑及定向补证路径也可能主要消费选中报告与契约，不能假定每个研究模型都累积全部30轮原文。', '', '## 失败样例', ''])
    if stats['failures']:
        for failure in stats['failures']:
            lines.extend([f'第{failure["turn_id"]}轮，原题{failure["source_id"]}，run_id `{failure["run_id"]}`。',
                          f'错误类型：{failure["error_type"]}；错误：{failure["error"]}。',
                          f'超限视图：`{json.dumps(failure["over_budget_views"], ensure_ascii=False)}`。',
                          f'原始轮次目录：`{failure["directory"]}`。', ''])
    else:
        lines.append('当前已采集范围内没有未交付轮次；若运行尚未结束，这不代表剩余轮次通过。')
    lines.extend(['', '## 原始结果及可追溯性', '',
        f'- 运行清单：`{run / "运行清单.json"}`', f'- 冻结配置：`{run / "配置快照.json"}`',
        f'- 逐轮进度：`{run / "采集进度.json"}`', f'- 机器统计：`{raw_path}`',
        f'- 每轮会话任务输入、会话原始消息、会话摘要状态、工作历史、研究状态、研究事件：`{run / "合并开发集30轮"}`',
        '保存的代码快照与运行前后源文件hash用于识别并行改动。', '',
        '## 局限性、结论与下一步', '',
        '**已测量**：上述实际交付轮数、模型请求/usage、输入视图、摘要尝试与采用、会话读取及路由后历史量。'
        '**间接推断**：Writer压缩有助于降低重复资料输入，必须结合处理前后token和实际阶段判断；单次运行不能归因三种方法各自收益。'
        '**尚未评估**：模型原生最大窗口/最大轮数、报告事实正确性与遗忘率、纯无语义摘要条件下的模型容量。',
        '本实验是“无会话摘要、Writer内部允许摘要”的系统连续交付测试，不是“所有模型均无摘要”的实验。'
        '若停止由检索、网络、输出解析、质量返工或运行预算引起，不能以停止轮数作为上下文容量上限；即使是本地输入超限，也仅说明项目硬预算。'
        '后续若测纯模型原生容量，需要独立固定单模型、每轮注入完整历史、关闭所有摘要与裁剪，记录真实发送token和provider上下文拒绝；'
        '若所有30轮通过，本开发集只能给出至少30轮。',
        f'运行期间源文件变化：`{json.dumps(stats["changed_source_files"], ensure_ascii=False)}`。', ''])
    destination.write_text('\n'.join(lines))
    return {key: stats[key] for key in ('status', 'attempted_turns', 'consecutive_completed_turns',
                                      'provider_requests', 'summary_attempts', 'summary_adoptions',
                                      'session_raw_history_observed')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.run.resolve(), args.report.resolve()), ensure_ascii=False))


if __name__ == '__main__':
    main()
