"""研究报告中的可复现计算和数据一致性检查。"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional

from .evidence import normalize_evidence_state, data_point_views, is_derived, TIER_RANK


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = str(value).replace(",", "").replace("，", "").strip()
    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    return float(text) if match else None


def calculate_cagr(start_value: Any, end_value: Any, years: Any) -> float:
    start = _number(start_value)
    end = _number(end_value)
    period = _number(years)
    if start is None or end is None or period is None or start <= 0 or end < 0 or period <= 0:
        raise ValueError("CAGR 输入必须是有限数值，且起始值和年数大于0")
    return (end / start) ** (1.0 / period) - 1.0


def run_deterministic_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """校验单位/重复值/权重，并执行契约声明的确定性计算。"""
    normalize_evidence_state(state)
    # 每次从原始数据重算，避免补充搜索/复审循环重复追加派生值。
    points = [
        dict(point) for point in data_point_views(state)
        if not is_derived(point)
    ]
    blocking: List[str] = []
    warnings: List[str] = []

    units_by_metric: Dict[str, set] = defaultdict(set)
    values_by_key: Dict[tuple, set] = defaultdict(set)
    for point in points:
        name = re.sub(r"\s+", "", str(point.get("name") or "").lower())
        unit = str(point.get("unit") or "").strip().lower()
        if name and unit:
            units_by_metric[name].add(unit)
        number = _number(point.get("value"))
        if name and number is not None:
            values_by_key[(
                str(point.get("entity") or "").strip(),
                name,
                str(point.get("year") or point.get("period") or "").strip(),
                str(point.get("accounting_standard") or "").strip(),
            )].add(number)

    inconsistent_units = sorted(name for name, units in units_by_metric.items() if len(units) > 1)
    conflicting_values = sorted(
        "/".join(value for value in (entity, name, year, standard) if value)
        for (entity, name, year, standard), values in values_by_key.items()
        if len(values) > 1
    )
    if inconsistent_units:
        blocking.append("同名指标单位不一致: " + ", ".join(inconsistent_units[:10]))
    if conflicting_values:
        warnings.append("同指标同年份存在冲突值，写作时必须说明口径: " + ", ".join(conflicting_values[:10]))

    # 只有显式标成评分权重的数据才组成一个需要归一化到 1/100 的权重集合。
    # “网络边权重”“机构权重”等领域指标没有总和必须为 100 的语义；仅凭名称
    # 包含“权重”就全局求和，会把金融网络模型数据误判成评分表。
    weight_points = [point for point in points if point.get("role") == "weight"]
    numeric_weights = [_number(point.get("value")) for point in weight_points]
    numeric_weights = [value for value in numeric_weights if value is not None]
    if numeric_weights:
        total = sum(numeric_weights)
        expected = 1.0 if max(numeric_weights) <= 1.0 else 100.0
        if not math.isclose(total, expected, rel_tol=0, abs_tol=0.01):
            blocking.append(f"评分权重合计为 {total:g}，应为 {expected:g}")

    def provenance(input_points):
        verified = all(point.get("reader_verified") for point in input_points)
        return {
            "origin": "derived",
            "input_point_ids": [str(point["id"]) for point in input_points],
            "calculation_status": "passed",
            "reader_verified": bool(input_points) and verified,
            "source_tier": min((point.get("source_tier", "low") for point in input_points),
                               key=TIER_RANK.get) if input_points and verified else "low",
        }

    derived = []
    contract = state.get("research_contract") or {}
    for spec in contract.get("required_calculations", []) or []:
        calculation_type = str(spec.get("type") or "").lower()
        inputs = [str(value) for value in spec.get("inputs", []) if str(value)]
        entities = [
            str(value) for value in spec.get("entity_scope", []) if str(value)
        ] or [
            str(value) for value in contract.get("required_entities", []) if str(value)
        ]

        def find_point(input_name: str, entity: str = "") -> Optional[Dict[str, Any]]:
            return next((
                point for point in points
                if input_name.lower() in str(point.get("name") or "").lower()
                and (
                    not entity
                    or entity == str(point.get("entity") or "")
                    or entity in str(point.get("entity") or "")
                )
                and _number(point.get("value")) is not None
            ), None)

        if calculation_type == "cagr":
            target_entities = entities or [""]
            for entity in target_entities:
                input_points = [find_point(name, entity) for name in inputs[:2]]
                if any(point is None for point in input_points) or len(input_points) != 2:
                    if spec.get("critical", True):
                        blocking.append(
                            f"CAGR计算缺少输入: {entity or '全局'}/{spec.get('output') or spec.get('id')}"
                        )
                    continue
                years = spec.get("years")
                if years is None:
                    try:
                        years = abs(int(input_points[1].get("year")) - int(input_points[0].get("year")))
                    except (TypeError, ValueError):
                        years = None
                try:
                    value = calculate_cagr(input_points[0].get("value"), input_points[1].get("value"), years)
                except ValueError as exc:
                    blocking.append(f"CAGR计算失败 {spec.get('id')}: {exc}")
                    continue
                point = {
                    "id": f"derived_{spec.get('id')}_{len(derived)}",
                    "entity": entity,
                    "name": spec.get("output") or f"{spec.get('id')} CAGR",
                    "value": round(value * 100, 4),
                    "unit": "%",
                    "year": input_points[1].get("year"),
                    "source": "deterministic_calculation",
                    "confidence": min(float(p.get("confidence", 1.0)) for p in input_points),
                    "formula": "(end/start)^(1/years)-1",
                    **provenance(input_points),
                }
                points.append(point)
                derived.append(point)
            continue

        if calculation_type == "weighted_score":
            raw_weights = spec.get("weights")
            if not isinstance(raw_weights, Mapping) or not raw_weights:
                if spec.get("critical", True):
                    blocking.append(f"加权评分缺少显式权重: {spec.get('id')}")
                continue
            weights = {str(name): _number(value) for name, value in raw_weights.items()}
            if any(value is None or value < 0 for value in weights.values()):
                blocking.append(f"加权评分权重非法: {spec.get('id')}")
                continue
            total_weight = sum(weights.values())
            expected = 1.0 if max(weights.values()) <= 1 else 100.0
            if not math.isclose(total_weight, expected, abs_tol=0.01):
                blocking.append(f"加权评分权重合计为 {total_weight:g}，应为 {expected:g}")
                continue
            for entity in entities:
                entity_points = {name: find_point(name, entity) for name in weights}
                missing_inputs = [name for name, point in entity_points.items() if point is None]
                if missing_inputs:
                    blocking.append(
                        f"{entity} 缺少评分输入: {', '.join(missing_inputs)}；不得评分"
                    )
                    continue
                score = sum(
                    _number(entity_points[name]["value"]) * weight / expected
                    for name, weight in weights.items()
                )
                point = {
                    "id": f"derived_{spec.get('id')}_{len(derived)}",
                    "entity": entity,
                    "name": spec.get("output") or "综合得分",
                    "value": round(score, 6),
                    "unit": "分",
                    "source": "deterministic_calculation",
                    "confidence": min(float(p.get("confidence", 1.0)) for p in entity_points.values()),
                    "formula": str(weights),
                    **provenance(list(entity_points.values())),
                }
                points.append(point)
                derived.append(point)
            continue

        if calculation_type == "ranking":
            input_name = inputs[0] if inputs else ""
            values = []
            for entity in entities:
                source_point = find_point(input_name, entity)
                if source_point is None:
                    blocking.append(f"{entity} 缺少排名输入 {input_name}；不得参与排名")
                    continue
                values.append((entity, _number(source_point.get("value")), source_point))
            if len(values) != len(entities):
                continue
            descending = str(spec.get("order") or "desc").lower() != "asc"
            for rank, (entity, _, source_point) in enumerate(
                sorted(values, key=lambda item: item[1], reverse=descending), 1
            ):
                point = {
                    "id": f"derived_{spec.get('id')}_{rank}",
                    "entity": entity,
                    "name": spec.get("output") or f"{input_name}排名",
                    "value": rank,
                    "unit": "名",
                    "source": "deterministic_calculation",
                    "confidence": source_point.get("confidence", 1.0),
                    "formula": f"rank({input_name}, order={'desc' if descending else 'asc'})",
                    **provenance([entry[2] for entry in values]),
                }
                points.append(point)
                derived.append(point)
            continue

        warnings.append(f"未实现的计算类型已跳过: {calculation_type}")

    state["data_points"] = points
    result = {
        "passed": not blocking,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "derived_points": derived,
        "inconsistent_units": inconsistent_units,
        "conflicting_values": conflicting_values,
    }
    state["calculation_checks"] = result
    return result
