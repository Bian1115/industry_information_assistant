"""追问契约的类型化变更与提交；不依赖模型或数据库，不修改父运行。"""
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class ContractOperation(StrictModel):
    op: Literal['add', 'replace', 'remove']
    field: Literal['max_body_chars', 'required_entities', 'critical_questions',
                   'explicit_requirements', 'required_metrics', 'required_calculations',
                   'time_range', 'geography', 'language', 'output_format']
    value: Any = None
    quote: str = Field(default='', max_length=4000)

    @model_validator(mode='after')
    def validate_value(self, info: ValidationInfo):
        if (info.context or {}).get('validate_quotes', True) and not self.quote.strip():
            raise ValueError('契约变更必须附本轮用户原话')
        if self.op == 'remove':
            if self.value is not None:
                raise ValueError('remove的value必须为null')
            return self
        value = self.value
        if self.field == 'max_body_chars':
            if type(value) is not int or not 100 <= value <= 100000:
                raise ValueError('字数上限必须是100到100000的整数')
        elif self.field in {'required_entities', 'critical_questions'}:
            if not isinstance(value, list) or not value or any(
                    not isinstance(x, str) or not x.strip() for x in value):
                raise ValueError('对象/问题必须是非空字符串数组；清空请用remove')
            if len(set(value)) != len(value):
                raise ValueError('对象/问题不能重复')
        elif self.field in {'explicit_requirements', 'required_metrics', 'required_calculations'}:
            required = {'explicit_requirements': ('id', 'original_requirement'),
                        'required_metrics': ('id', 'name'),
                        'required_calculations': ('id', 'type')}[self.field]
            if not isinstance(value, list) or not value or any(
                    not isinstance(x, dict) or any(not isinstance(x.get(k), str) or not x[k].strip()
                                                   for k in required) for x in value):
                raise ValueError('要求/指标/计算数组缺少有效ID或内容')
            if len({x['id'] for x in value}) != len(value):
                raise ValueError('要求/指标/计算ID重复')
            for item in value:
                for key in ('years', 'entity_scope'):
                    if key in item and (not isinstance(item[key], list) or any(
                            not isinstance(x, str) or not x.strip() for x in item[key])):
                        raise ValueError(f'{key}必须是字符串数组')
                if 'critical' in item and type(item['critical']) is not bool:
                    raise ValueError('critical必须为布尔值')
        elif not isinstance(value, str) or not value.strip() or len(value) > 4000:
            raise ValueError('范围/语言/格式必须是非空字符串')
        return self


class ContractPatch(StrictModel):
    base_revision: int = Field(ge=0)
    operations: list[ContractOperation] = Field(max_length=20)


def parent_contract(parent):
    contract = deepcopy(parent.get('research_contract') or {})
    contract.setdefault('contract_revision', 0)
    contract.setdefault('active_constraints', deepcopy(parent.get('active_constraints') or {}))
    return contract


def apply_contract_patch(base, patch, *, original_query, resolved_query, instructions,
                         reuse_previous=True, validate_quotes=True):
    """校验完整补丁后返回新契约。失败不修改任何输入；revision与schema version分开。"""
    patch = ContractPatch.model_validate(patch, context={'validate_quotes': validate_quotes})
    if patch.base_revision != base.get('contract_revision', 0):
        raise ValueError('契约版本已变化，不能应用旧补丁')
    if not resolved_query.strip() or any(not x.strip() for x in instructions):
        raise ValueError('消解问题和修改指令不能为空')
    contract = deepcopy(base) if reuse_previous else {}
    active = contract.setdefault('active_constraints', {})
    seen = set()
    for change in patch.operations:
        field = change.field
        if field in seen:
            raise ValueError('同一字段一次只能变更一次')
        seen.add(field)
        if validate_quotes and (not change.quote.strip() or change.quote not in original_query):
            raise ValueError('契约变更依据不在本轮用户原话中')
        target = active if field == 'max_body_chars' else contract
        exists = field in target and target[field] not in (None, '', [])
        if change.op == 'add' and exists:
            raise ValueError(f'{field}已存在，应使用replace')
        if change.op in {'replace', 'remove'} and not exists:
            raise ValueError(f'{field}不存在，不能替换或删除')
        if change.op == 'remove':
            target.pop(field, None)
        else:
            target[field] = deepcopy(change.value)
    # 防止对象/时间变化后，旧指标中的范围继续成为硬性检查条件。
    if seen & {'required_entities', 'time_range'}:
        for field in ('required_metrics', 'required_calculations'):
            if base.get(field) and reuse_previous and field not in seen:
                raise ValueError(f'对象/时间变化时须同时核对并更新或删除{field}')
    contract.update(query=resolved_query, original_query=original_query,
                    followup_instructions=list(instructions),
                    contract_revision=patch.base_revision + 1,
                    contract_update={'status': 'committed', 'reuse_previous': reuse_previous,
                                     'source_quote_check': validate_quotes,
                                     'patch': patch.model_dump()})
    # 防止本轮约束删除后规划器/旧字段重新补回它。
    contract.pop('pending_contract_update', None)
    return contract


def contract_query(contract):
    """兼容仍以query接收任务背景的下游；契约自身query始终保持干净。"""
    import json
    visible = {k: v for k, v in contract.items()
               if k not in {'contract_update', 'pending_contract_update', 'critical_sections'}}
    return contract['query'] + '\n本轮有效研究契约（历史草稿不能覆盖）：' + json.dumps(visible, ensure_ascii=False)


def retain_committed_contract(planned, committed, outline):
    """重规划只刷新章节派生关系，不能重建已提交用户约束或复活已删除字段。"""
    if (committed.get('contract_update') or {}).get('status') != 'committed':
        return planned
    result = deepcopy(committed)
    result.setdefault('version', planned.get('version', 1))
    # 旧运行可携带历史验收元数据；重规划不再继承这些系统验收项。
    for field in ('source_policy', 'acceptance', 'critical_sections', 'coverage_check'):
        result.pop(field, None)
    # 旧指标的章节编号失效，按新大纲的真实声明重新关联；不得沿用同名旧ID。
    for metric in result.get('required_metrics', []):
        matches = [s['id'] for s in outline if metric['id'] in s.get('requirement_ids', [])]
        metric['section_id'] = matches[0] if len(matches) == 1 else ''
    return result
