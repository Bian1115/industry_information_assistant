"""同源回放任务总预算；主任务、摘要、修复和供应商重试共享一个账本。"""
from contextlib import contextmanager
from contextvars import ContextVar
import time
import math

_current = ContextVar('verification_task_budget', default=None)


class TaskBudget:
    def __init__(self, config):
        if (type(config.get('max_requests')) is not int or config['max_requests'] < 1
                or not isinstance(config.get('max_seconds'), (int, float))
                or not math.isfinite(config['max_seconds']) or config['max_seconds'] <= 0
                or not config.get('max_tokens_by_model')
                or any(type(v) is not int or v < 1 for v in config['max_tokens_by_model'].values())):
            raise ValueError('整任务预算必须是有限正数，模型token上限须为整数')
        for prices in config.get('prices_per_million', {}).values():
            if any(not isinstance(prices.get(k), (int, float)) or not math.isfinite(prices[k]) or prices[k] < 0 for k in ('input', 'output')):
                raise ValueError('冻结单价必须是非负有限数值')
        self.config = config
        self.started = time.monotonic()
        self.requests = []
        self.reserved = {}

    def reserve(self, model, input_tokens, output_tokens):
        if time.monotonic() - self.started >= self.config['max_seconds']:
            raise TimeoutError('整任务时限耗尽')
        if len(self.requests) >= self.config['max_requests']:
            raise ValueError('整任务调用预算耗尽，含失败及重试')
        limit = self.config['max_tokens_by_model'].get(model)
        if limit is None:
            raise ValueError('实际模型不在冻结整任务预算清单中')
        tokens = input_tokens + output_tokens
        if self.reserved.get(model,0) + tokens > limit:
            raise ValueError('整任务模型token预算耗尽')
        # 失败也保留上限预留，不用未知用量伪造零成本。
        self.reserved[model] = self.reserved.get(model,0) + tokens
        row = {'model':model,'input_reserved':input_tokens,'output_reserved':output_tokens,'usage':None,'cost_usd':None,
               'execution_status':'not_completed'}
        self.requests.append(row)
        return row

    def complete(self, ticket, usage):
        ticket['usage'] = usage
        ticket['execution_status'] = 'completed'
        if usage.get('prompt_tokens') is None or usage.get('completion_tokens') is None:
            return
        rate = self.config.get('prices_per_million',{}).get(ticket['model'])
        if rate:
            ticket['cost_usd'] = (usage['prompt_tokens']*rate['input']+usage['completion_tokens']*rate['output'])/1_000_000
        # 计数器估算只用于预留；超出预留时记录实际量并阻止后续超预算调用。
        actual = usage['prompt_tokens'] + usage['completion_tokens']
        reserved = ticket['input_reserved'] + ticket['output_reserved']
        self.reserved[ticket['model']] += max(0,actual-reserved)

    def report(self):
        costs=[r['cost_usd'] for r in self.requests]
        return {'duration_seconds':time.monotonic()-self.started,'requests':self.requests,
                'tokens_reserved_by_model':self.reserved,
                'api_usd':sum(costs) if all(c is not None for c in costs) else None, 'usd':None,
                'local_nli_resource_cost':'not_priced',
                'note':'API费用按冻结单价与真实usage计；本地NLI资源成本未折算，不宣称总成本为零'}


@contextmanager
def task_budget(config):
    budget=TaskBudget(config)
    token=_current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)


def current_budget():
    return _current.get()
