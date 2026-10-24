#!/usr/bin/env python3
"""复用冻结输入和传输配置，独立补齐Qwen评分；不干扰Claude运行。"""
import concurrent.futures as cf
import importlib.util
import os
import sys
import threading
import argparse
from pathlib import Path

spec = importlib.util.spec_from_file_location('bench_run', Path(__file__).with_name('DeepResearchBench参考报告双裁判一致性.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();assert 1<=args.workers<=8
    cfg = b.read(b.OUT/'评估配置.json')
    tasks = b.read(b.OUT/'冻结输入.json')
    m = b.evaluator()
    assert b.sha(b.OUT/'冻结输入.json') == cfg['input_sha256']
    assert b.sha(Path(m.__file__)) == cfg['transport_sha256']
    assert b.sha(b.OUT/'双裁判提示词快照.txt') == cfg['prompt_sha256']
    for name, digest in cfg['assistant_file_sha256'].items():
        assert b.sha(b.OUT/'助手独立评分'/name) == digest, '冻结助手分数被修改'
    opts = cfg['qwen_request_options']
    done = []
    pending = []
    failed = []
    blocked = threading.Event()
    started = b.now()
    previous=b.read(b.OUT/'Qwen续跑状态.json') if (b.OUT/'Qwen续跑状态.json').exists() else {}
    history=b.read(b.OUT/'Qwen续跑历史.json') if (b.OUT/'Qwen续跑历史.json').exists() else []
    if previous:history.append(previous);b.dump(b.OUT/'Qwen续跑历史.json',history)
    b.dump(b.OUT/'Qwen续跑执行配置.json',{'started_at':started,'pid':os.getpid(),'workers':args.workers,
           'frozen_request_options':opts,'configuration_sha256':b.sha(b.OUT/'评估配置.json'),
           'change':'仅改变请求调度并发；评分输入、模型采样和汇总规则不变。'})
    def save(status):
        b.dump(b.OUT/'Qwen续跑评分.json', sorted(done, key=lambda x:(x['task_id'],x['repeat'])))
        b.dump(b.OUT/'Qwen续跑状态.json', {'status':status,'started_at':started,'updated_at':b.now(),
               'pid':os.getpid(),'workers':args.workers,'model':cfg['models']['qwen'],'expected':60,'completed':len(done),
               'failed':failed,'resumption':'原样使用冻结输入、模型和参数；先实题探测，账户异常时停止剩余新请求。'})
    for t in tasks:
        for n in [1,2]:
            key = f"题{t['id']}-细则-qwen-{n}"
            old = m.read_matching_cache(b.OUT/'裁判原始记录'/f'{key}.json', b.prompt(t,n), opts)
            if old and old.get('result') is not None:
                b.validate(t, old['result'])
                assert old.get('completion_event') == '[DONE]' and old.get('finish_reason') == 'stop'
                done.append({'task_id':t['id'],'model':'qwen','repeat':n,'key':key,'result':old['result']})
            else:
                pending.append((t,n))
    def work(job):
        t,n = job
        if blocked.is_set(): return None
        key = f"题{t['id']}-细则-qwen-{n}"
        try:
            result = m.call('qwen', b.prompt(t,n), key, max_tokens=cfg['claude_max_tokens'],
                            validator=lambda x:b.validate(t,x), request_options=opts)
            return {'task_id':t['id'],'model':'qwen','repeat':n,'key':key,'result':result}
        except Exception as ex:
            if 'Arrearage' in str(ex): blocked.set()
            raise
    def collect(job, future=None):
        try:
            result = future.result() if future is not None else work(job)
            if result is not None: done.append(result)
        except Exception as ex:
            failed.append({'task_id':job[0]['id'],'repeat':job[1],'error':str(ex)})
        save('账户异常阻塞' if blocked.is_set() else '运行中')
        print('Qwen有效评分',len(done),'/60；本次失败',len(failed),flush=True)
    save('检查缓存与接口')
    if pending and not done:
        collect(pending.pop(0))
    if pending and not blocked.is_set():
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            fs = {pool.submit(work,job):job for job in pending}
            for f in cf.as_completed(fs):collect(fs[f],f)
    save('完成' if len(done)==60 else ('账户异常阻塞' if blocked.is_set() else '未完成'))
    if len(done)!=60: sys.exit(2)

if __name__ == '__main__': run()
