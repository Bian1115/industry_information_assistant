"""报告交付的篇幅约束；与执行入口及追问分类无关。"""
import re
import os


def body_length_hard_check_enabled():
    """可关闭交付阻断，保留用户预算与超限测量。"""
    return os.getenv('REPORT_BODY_LENGTH_HARD_CHECK_ENABLED', 'true').strip().lower() not in {'0', 'false', 'no', 'off'}


def body_character_count(report):
    """沿用现有正文口径，不计参考文献、空白及Markdown链接URL。"""
    body = report.split('## 参考文献', 1)[0]
    body = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', body)
    return len(re.sub(r'\s|[#*`>]', '', body))


def check_body_length(state):
    limit = (state.get('active_constraints') or {}).get('max_body_chars')
    actual = body_character_count(state.get('final_report') or '')
    exceeded = limit is not None and actual > limit
    result = {'body_chars': actual, 'max_body_chars': limit, 'exceeded': exceeded,
              'excess_chars': max(0, actual - limit) if limit is not None else 0,
              'hard_check_enabled': body_length_hard_check_enabled(),
              'reason': '模型生成的正文合计超过有效用户篇幅预算' if exceeded else ''}
    state['body_length_check'] = result
    if exceeded and result['hard_check_enabled']:
        raise ValueError(f'报告正文超过用户要求的{limit}字，本次不交付')
    return result
