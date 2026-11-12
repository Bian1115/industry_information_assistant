"""同一裁判、同一提示词，仅交换候选顺序；保存首次完整分数与原始引句。

用于检查位置敏感性，不把输出引句自动当成有效证据。首次评分不因引用重试而替换。
"""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('evaluation',Path(__file__).with_name('评估通用研究历史压缩与后续写作.py'))
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)

async def main(out):
    from dotenv import load_dotenv
    load_dotenv(ev.BACKEND/'.env',override=False)
    runner=ev.Runner(out)
    async def one(topic):
        folder=out/f'题目{topic}'
        original=ev.load(folder/'模型调用/匿名质量评审2/请求.json')
        messages=original['messages']
        payload=json.loads(messages[1]['content'])
        candidates=payload['candidates']
        reversed_candidates=[dict(candidates[1],id='候选1'),dict(candidates[0],id='候选2')]
        payload['candidates']=reversed_candidates
        new_messages=[messages[0],{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        def validate(value):
            assert len(value['candidates'])==2 and {c['id'] for c in value['candidates']}=={'候选1','候选2'}
            assert value['winner'] in {'候选1','候选2','tie'}
            for candidate in value['candidates']:
                for scope in ['overall','continued']:
                    assert set(candidate[scope])==set(ev.DIMS)
                    assert all(isinstance(x,(int,float)) and not isinstance(x,bool) and 1<=x<=10 for x in candidate[scope].values())
        value=await runner.call(folder,'匿名质量顺序复评',new_messages,original['model'],original['max_tokens'],validator=validate)
        groups=['原文组','压缩组']
        mapping={c['id']:next(g for g in groups if c['report']==(folder/g/'完整报告.md').read_text()) for c in reversed_candidates}
        ev.save(folder/'质量顺序复评结果.json',{'model':original['model'],'mapping':mapping,'result':value,
            'scope':'同一裁判同一提示词，只交换候选顺序；数值结构验证通过，引用与语义质量另行审计。','is_human_gold':False})
    results=await asyncio.gather(*(one(t) for t in ev.SOURCES),return_exceptions=True)
    await runner.client.close()
    failures=[{'topic':t,'error':str(r)} for t,r in zip(ev.SOURCES,results) if isinstance(r,BaseException)]
    ev.save(out/'顺序复评运行状态.json',{'date':ev.now(),'failures':failures,'completed':len(ev.SOURCES)-len(failures)})
    if failures:raise RuntimeError(failures)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main(parser.parse_args().output))
