"""当前生成报告的GPT与Claude逐细则裁判；历史Qwen实验不作为新运行入口。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from eval.scripts.DeepResearchBench参考报告双裁判一致性 import BASE, SCHEMA, validate

BACKEND = Path(__file__).resolve().parents[1]
VERSION = '逐细则GPT与Claude双裁判-20261108-v1'
DIMENSIONS = ('comprehensiveness', 'insight', 'instruction_following', 'readability')
PROVIDERS = ('gpt', 'claude')


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def configuration():
    load_dotenv(BACKEND / '.env')
    load_dotenv(BACKEND / '.env.eval-judges')
    return {
        'evaluator_version': VERSION, 'repeats': 2,
        'gpt': {'model': os.getenv('EVAL_GPT_MODEL', 'gpt-6-astra'),
                'base_url': os.getenv('EVAL_GPT_BASE_URL', 'https://api.openai.com/v1').rstrip('/'),
                'reasoning_effort': os.getenv('EVAL_GPT_REASONING_EFFORT', 'xhigh'),
                'max_output_tokens': int(os.getenv('EVAL_GPT_MAX_OUTPUT_TOKENS', '32000'))},
        'claude': {'model': os.getenv('CLAUDE_RELAY_MODEL', 'claude-opus-5'),
                   'base_url': os.getenv('CLAUDE_RELAY_BASE_URL', '').rstrip('/'), 'max_tokens': 12000},
        'score_scale': [0, 10], 'dimensions': list(DIMENSIONS),
        'aggregation': '维度内细则权重归一化，官方维度权重加权；先均每位裁判两次，再等权均两裁判',
    }


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def unique_rows(rows):
    indexed = {}
    for row in rows:
        key = str(row['id'])
        if key in indexed:
            raise ValueError('题号重复，必须先明确本轮需要评分的执行尝试：' + key)
        indexed[key] = row
    return indexed


def load_tasks(dataset, reports_path):
    manifest = json.loads((dataset / '抽样清单.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        if hashlib.sha256((dataset / name).read_bytes()).hexdigest() != expected:
            raise ValueError('验证集文件已改变：' + name)
    questions = unique_rows(read_rows(dataset / '验证题目.jsonl'))
    rules = unique_rows(read_rows(dataset / '评分专用材料/官方评分细则.jsonl'))
    reports = unique_rows(read_rows(reports_path))
    if not reports.keys() <= questions.keys():
        raise ValueError('报告包含验证集以外的题号')
    tasks = []
    for ident, question in questions.items():
        raw = rules[ident]
        if raw['prompt'] != question['prompt'] or set(raw['dimension_weight']) != set(DIMENSIONS):
            raise ValueError('细则与原题不对应：' + ident)
        criteria = []
        for dimension in DIMENSIONS:
            rows = raw['criterions'][dimension]
            if not rows or any(type(c['weight']) not in (int, float) or not math.isfinite(c['weight']) or c['weight'] < 0 for c in rows) or sum(c['weight'] for c in rows) <= 0:
                raise ValueError('细则权重无效：' + ident)
            criteria.extend({**c, 'dimension': dimension, 'criterion_id': f'{dimension}:{i:02d}'} for i, c in enumerate(rows, 1))
        weights = raw['dimension_weight']
        if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-6):
            raise ValueError('维度权重必须合计1：' + ident)
        report = reports.get(ident, {})
        if report and report.get('prompt') != question['prompt']:
            raise ValueError('报告原题已改变：' + ident)
        article = report.get('article', '')
        if not isinstance(article, str):
            raise ValueError('报告正文必须是字符串：' + ident)
        meta = report.get('_meta') or {}
        task = {**question, 'article': article, 'criteria': criteria, 'dimension_weight': weights,
                'report_sha256': hashlib.sha256(article.encode()).hexdigest(),
                'generation_status': 'completed' if article.strip() and not meta.get('error') else 'no_completed_report',
                'online_critic': {k: meta.get(k) for k in ('quality_score', 'quality_score_basis', 'quality_dimension_scores', 'quality_verdict')},
                'research_run_id': meta.get('research_run_id'), 'trace_id': meta.get('trace_id')}
        tasks.append(task)
    return tasks


def common_prompt():
    examples = json.loads((BACKEND / 'eval/configs/双裁判质量评分虚构样例.json').read_text())
    if examples.get('synthetic_only') is not True:
        raise ValueError('评分锚点必须是独立虚构样例')
    return BASE + '<虚构锚点>\n' + json.dumps(examples['examples'], ensure_ascii=False, indent=2) + '\n</虚构锚点>\n正式输出格式：' + json.dumps(SCHEMA, ensure_ascii=False)


def make_prompt(task, repeat, common):
    # 评分器只发送原题、细则和待评正文，不发送在线Critic结果、参考报告或另一裁判意见。
    return common + '\n重复编号：' + str(repeat) + '\n<任务>\n' + task['prompt'] + '\n</任务>\n<官方细则>\n' + json.dumps(task['criteria'], ensure_ascii=False, indent=2) + '\n</官方细则>\n<待评报告>\n' + task['article'] + '\n</待评报告>\n只输出完整JSON，全部细则各一次。'


def validate_result(task, result):
    validate(task, result)
    for row in result['scores']:
        quote = row['evidence_quote']
        if quote and quote not in task['article']:
            raise ValueError('评分摘录不是报告中的连续原文：' + row['criterion_id'])


def weighted(task, result):
    scores = {r['criterion_id']: r['score'] for r in result['scores']}
    dimensions = {}
    for dimension in DIMENSIONS:
        rules = [r for r in task['criteria'] if r['dimension'] == dimension]
        dimensions[dimension] = sum(scores[c['criterion_id']] * c['weight'] for c in rules) / sum(c['weight'] for c in rules)
    return {**dimensions, 'total': sum(dimensions[d] * task['dimension_weight'][d] for d in DIMENSIONS)}


def request_parameters(provider, prompt, config):
    settings = config[provider]
    if provider == 'gpt':
        return {'model': settings['model'], 'input': [{'role': 'user', 'content': prompt}],
                'reasoning': {'effort': settings['reasoning_effort']}, 'store': False,
                'max_output_tokens': settings['max_output_tokens'],
                'text': {'format': {'type': 'json_schema', 'name': 'bench_criterion_scores', 'strict': True, 'schema': SCHEMA}}}
    return {'model': settings['model'], 'max_tokens': settings['max_tokens'], 'stream': True,
            'messages': [{'role': 'user', 'content': prompt}]}


def parse_gpt(body):
    if body.get('status') != 'completed':
        raise ValueError('GPT响应未完整完成：' + str(body.get('status')))
    content = [part for item in body.get('output', []) if item.get('type') == 'message' for part in item.get('content', [])]
    if any(part.get('type') == 'refusal' for part in content):
        raise ValueError('GPT拒绝评分')
    text = ''.join(part['text'] for part in content if part.get('type') == 'output_text')
    return json.loads(text)


def safe_error(exc):
    message = str(exc)
    for key in ('EVAL_GPT_API_KEY', 'CLAUDE_RELAY_API_KEY'):
        secret = os.getenv(key)
        if secret:
            message = message.replace(secret, '[已隐藏密钥]')
    return re.sub(r'sk-[A-Za-z0-9_-]+', '[已隐藏密钥]', message)[:1000]


def fetch_judgment(provider, payload, config, packet):
    """请求对象只在内存持有认证头；归档对象packet不包含认证信息。"""
    key_name = 'EVAL_GPT_API_KEY' if provider == 'gpt' else 'CLAUDE_RELAY_API_KEY'
    key = os.getenv(key_name)
    base = config[provider]['base_url']
    if not key or not base:
        raise ValueError('缺少裁判配置：' + key_name + '或服务地址')
    headers = {'Authorization': 'Bearer ' + key}
    with httpx.Client(timeout=httpx.Timeout(240, connect=25), trust_env=False, follow_redirects=False) as client:
        if provider == 'gpt':
            response = client.post(base + '/responses', headers=headers, json=payload)
            if response.status_code != 200:
                raise RuntimeError('GPT HTTP ' + str(response.status_code))
            body = response.json()
            packet.update(raw_response=body, response_model=body.get('model'), usage=body.get('usage'))
            return parse_gpt(body)
        headers['anthropic-version'] = '2023-06-01'
        url = base + '/messages' if base.endswith('/v1') else base + '/v1/messages'
        packet.update(raw_response=[], usage={}, completion_event=None, stop_reason=None)
        text = []
        with client.stream('POST', url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                raise RuntimeError('Claude HTTP ' + str(response.status_code))
            for line in response.iter_lines():
                if time.monotonic() - packet['_started'] > 600:
                    raise TimeoutError('Claude流接收超过600秒')
                if not line.startswith('data:'):
                    continue
                event = json.loads(line[5:].strip())
                packet['raw_response'].append(event)
                kind = event.get('type')
                if kind == 'message_start':
                    message = event.get('message', {})
                    packet['response_model'] = message.get('model')
                    packet['usage'].update(message.get('usage', {}))
                elif kind == 'content_block_delta':
                    text.append(event.get('delta', {}).get('text', ''))
                elif kind == 'message_delta':
                    packet['usage'].update(event.get('usage', {}))
                    packet['stop_reason'] = event.get('delta', {}).get('stop_reason')
                elif kind == 'error':
                    raise RuntimeError('Claude流返回错误事件')
                elif kind == 'message_stop':
                    packet['completion_event'] = 'message_stop'
                    break
        if packet['completion_event'] != 'message_stop' or packet['stop_reason'] != 'end_turn':
            raise ValueError('Claude评分响应未完整结束')
        # 本轮只接受原生JSON，不新增旧格式或不完整响应恢复分支。
        return json.loads(''.join(text))


def grade_once(task, provider, repeat, common, config, output):
    prompt = make_prompt(task, repeat, common)
    payload = request_parameters(provider, prompt, config)
    identity = {'evaluator_version': VERSION, 'provider': provider, 'repeat': repeat,
                'task_id': task['id'], 'report_sha256': task['report_sha256'],
                'dimension_weight': task['dimension_weight'], 'base_url': config[provider]['base_url'], 'request': payload}
    fingerprint = digest(identity)
    directory = output / '裁判原始记录' / ('题' + str(task['id']))
    target = directory / f'{provider}-第{repeat}次.json'
    if target.exists():
        old = json.loads(target.read_text())
        if old['request_sha256'] != fingerprint:
            raise ValueError('裁判缓存与本轮输入或参数不同，请创建新运行目录')
        if old.get('status') == 'completed':
            validate_result(task, old['result'])
            return old
    packet = {**identity, 'request_sha256': fingerprint, 'status': 'failed', 'result': None,
              '_started': time.monotonic()}
    try:
        result = fetch_judgment(provider, payload, config, packet)
        validate_result(task, result)
        packet.update(status='completed', result=result, scores=weighted(task, result))
    except Exception as exc:
        packet['error'] = safe_error(exc)
    finally:
        packet['elapsed_s'] = round(time.monotonic() - packet.pop('_started'), 3)
        save_json(directory / f'{provider}-第{repeat}次-尝试{time.time_ns()}.json', packet)
        save_json(target, packet)
    return packet


def aggregate(task, packets):
    row = {'id': task['id'], 'generation_status': task['generation_status'], 'report_sha256': task['report_sha256'],
           'online_critic': task['online_critic'], 'judges': {}, 'ensemble': None,
           'status': 'no_completed_report' if task['generation_status'] != 'completed' else 'scoring_incomplete'}
    for provider in PROVIDERS:
        selected = [p for p in packets if p['provider'] == provider and p['status'] == 'completed']
        if len(selected) == 2 and {p['repeat'] for p in selected} == {1, 2}:
            row['judges'][provider] = {d: sum(p['scores'][d] for p in selected) / 2 for d in (*DIMENSIONS, 'total')}
    if set(row['judges']) == set(PROVIDERS):
        row['ensemble'] = {d: sum(row['judges'][p][d] for p in PROVIDERS) / 2 for d in (*DIMENSIONS, 'total')}
        row['status'] = 'completed'
    return row
