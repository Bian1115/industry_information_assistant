"""沿用Claude语法修复，但禁止补造或改变细则ID与数字分数。"""
import json,re
from json_repair import repair_json

def parse_response(raw):
    m=re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)```\s*',raw);body=m.group(1) if m else raw
    try:result=json.loads(body);mode='原生JSON'
    except json.JSONDecodeError:
        result=json.loads(repair_json(body));mode='沿用系统JSON语法修复'
        pairs=re.findall(r'"criterion_id"\s*:\s*"([^"\\]+)"\s*,\s*"score"\s*:\s*(\d+(?:\.\d+)?)',body)
        rows=result.get('scores',[])
        if len(pairs)!=len(rows) or len({i for i,s in pairs})!=len(pairs):raise ValueError('原始ID/分数无法完整逐项确认，不恢复')
        original={i:float(s) for i,s in pairs}
        if original!={r['criterion_id']:r['score'] for r in rows}:raise ValueError('语法修复改变ID或数字分数，不恢复')
    removed=[]
    for row in result.get('scores',[]):
        extra=set(row)-{'criterion_id','score','reason','evidence_quote'}
        if extra and all(row[k] is None for k in extra):
            removed.append({'criterion_id':row['criterion_id'],'fields':sorted(extra)})
            for k in extra:del row[k]
    return result,{'mode':mode,'removed_null_metadata':removed,'scores_not_imputed_or_modified':True}
