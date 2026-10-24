#!/usr/bin/env python3
"""保留原测量文件，追加检查点交付复核、跨浏览器指标汇总及中文报告。"""
import asyncio
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import zipfile

BACKEND = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('SSE_EVAL_OUTPUT') or Path('/tmp/SSE浏览器评测目录.txt').read_text().strip())
sys.path[:0] = [str(BACKEND), str(ROOT / '复评后端源码快照')]
from dotenv import load_dotenv
load_dotenv(BACKEND / '.env')
from core.database import SessionLocal
from models.chat import ChatMessage
from models.research import ResearchRun
from service.deep_research_v2.checkpointer import checkpointer_lifespan
from service.deep_research_v2.graph import graph_config

def read(p): return json.loads(Path(p).read_text())
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def q(values,p):
    if not values: return None
    a=sorted(values);i=(len(a)-1)*p
    return a[int(i)]+(a[min(int(i)+1,len(a)-1)]-a[int(i)])*(i%1)

async def main():
    groups={'主评测':read(ROOT/'评测汇总.json'),'首跑补充':read(ROOT/'首跑补充汇总.json'),
            '连接清理复评':read(ROOT/'连接清理复评汇总.json'),
            '断网补充复跑':read(ROOT/'断网补充复跑汇总.json')}
    rows=[{**x,'evaluation_group':g} for g,items in groups.items() for x in items]
    app=SimpleNamespace(state=SimpleNamespace())
    async with checkpointer_lifespan(app) as saver:
        for row in rows:
            folder=ROOT/row['label'];rid=row['run_id']
            history=[]
            async for item in saver.alist(graph_config(rid)):
                cp=item.checkpoint
                history.append({'config':item.config,'parent_config':item.parent_config,'metadata':item.metadata,
                    'checkpoint_id':cp['id'],'timestamp':cp.get('ts'),'research':cp.get('channel_values',{}).get('research'),
                    'pending_writes':item.pending_writes})
            save(folder/'检查点历史复核.json',history)
            raw=read(folder/'浏览器测量日志.json');server=read(folder/'服务端最终结果.json')
            saved=server['state'];report=saved.get('final_report')or''
            with SessionLocal() as db:
                run=db.query(ResearchRun).filter_by(run_id=rid).one()
                messages=db.query(ChatMessage).filter_by(run_id=rid,role='assistant').all()
                message_hashes=[sha(m.content) for m in messages]
                checkpoint_id=run.final_checkpoint_id
                run_status=run.status
            completion=[x for x in server['ledger'] if x['type']=='research_complete']
            checkpoint_body=next((x['research'].get('final_report','') for x in history if x['checkpoint_id']==checkpoint_id and isinstance(x['research'],dict)),None)
            consistency={'rechecked_at':datetime.now().isoformat(),'measured_original_delivery':server['delivery'],
                'current_run_status':run_status,'checkpoint_id':checkpoint_id,'checkpoint_body_sha256':sha(checkpoint_body.strip())if checkpoint_body else None,
                'message_hashes':message_hashes,'measured_projection_sha256':row['final_projection_sha256'],
                'checkpoint_message_projection_match':bool(checkpoint_body and message_hashes==[sha(checkpoint_body.strip())] and row['final_projection_sha256']==message_hashes[0]),
                'event_checkpoint_match':bool(len(completion)==1 and completion[0]['payload'].get('checkpoint_id')==checkpoint_id)}
            save(folder/'检查点与交付一致性复核.json',consistency)
            row['consistency']=consistency
            logs=[json.loads(l) for l in (folder/'服务端测量日志.jsonl').read_text().splitlines()]
            row['nodes_started']=dict(Counter(x['node']for x in logs if x['kind']=='node_start'))
            sends=[x for x in logs if x['kind']=='frame_send'];row['wire_bytes']=sum(x['bytes']for x in sends)
            row['dropped_sequence']=next((x['seq']for x in logs if x['kind']=='frame_dropped'),None)
            row['duplicated_sequence']=next((x['seq']for x in logs if x['kind']=='duplicate_injected'),None)
            row['gap_repair_to_projection_ms']=[]
            if row['dropped_sequence']:
                for page in raw['pages']:
                    for gap in page['rejected_events']:
                        if gap['reason']!='gap':continue
                        applied=next((e for e in page['applied'] if e['seq']==row['dropped_sequence'] and e['epoch_ms']>=gap['epoch_ms']),None)
                        if applied:row['gap_repair_to_projection_ms'].append(applied['epoch_ms']-gap['epoch_ms'])
            # 单个投影代次内、收到正文事件之后的DOM批次延迟；不把没有DOM变化的事件当渲染样本。
            dom_delays=[]
            for page in raw['pages']:
                for d in page['dom']:
                    if not d['report_text_length']: continue
                    candidates=[e for e in page['received'] if e['seq']==d['seq'] and e['epoch_ms']<=d['epoch_ms']]
                    if candidates: dom_delays.append(d['epoch_ms']-candidates[-1]['epoch_ms'])
            row['receive_to_body_dom_batch_ms']={'count':len(dom_delays),'p50':q(dom_delays,.5),'p95':q(dom_delays,.95)}
            # 真实Chrome合成帧截图作为首显佐证。DOM/rAF时刻仍不等于精确像素呈现时刻。
            first=next((p['first_body']for p in raw['pages'] if p['first_body']),None)
            row['first_body_visible_before_complete']=bool(first and first['status']!='completed')
            if not row['delivery_success']:row['catchup_to_completed_dom_ms']=None
            if first:
                with zipfile.ZipFile(folder/'浏览器轨迹.zip') as z:
                    trace=[json.loads(l) for l in z.read('trace.trace').decode().splitlines()]
                    frames=[x for x in trace if x.get('type')=='screencast-frame' and x.get('frameSwapWallTime',0)>=first['epoch_ms']]
                    if frames:
                        frame=frames[0];name='resources/'+frame['sha1']
                        if name in z.namelist(): (folder/'首显后浏览器合成帧.jpeg').write_bytes(z.read(name))
                        row['first_compositor_frame_after_dom_ms']=frame['frameSwapWallTime']-first['epoch_ms']
                        save(folder/'首显帧证据.json',{'dom_epoch_ms':first['epoch_ms'],'frame':frame,'boundary':'DOM首显之后的第一个已采集合成帧，只是佐证，尚未自动识别全部帧中的首个正文像素'})
                    duplicate=next((s for p in raw['pages'] for s in p['dom'] if s['status']!='completed' and s['body_marker_count']>1),None)
                    if duplicate:
                        frames=[x for x in trace if x.get('type')=='screencast-frame' and x.get('frameSwapWallTime',0)>=duplicate['epoch_ms']]
                        if frames:
                            name='resources/'+frames[0]['sha1']
                            if name in z.namelist(): (folder/'重复章节浏览器合成帧.jpeg').write_bytes(z.read(name))
                            save(folder/'重复章节帧证据.json',{'dom_sample':duplicate,'frame':frames[0]})
    summaries={}
    for group,items in groups.items():
        for scene in sorted(set(x['scenario']for x in items)):
            selected=[x for x in rows if x['evaluation_group']==group and x['scenario']==scene]
            wires=[];apply=[];clock_bounds=[]
            for x in selected:
                raw=read(ROOT/x['label']/'浏览器测量日志.json');offset=raw['clock']['offset_ms'];clock_bounds.append(raw['clock']['error_bound_ms'])
                for page in raw['pages']:
                    wires.extend(e['epoch_ms']+offset-e['wire']['sent_epoch_ms']for e in page['received']if e['wire'])
            summaries[group+':'+scene]={'cases':len(selected),'delivery_success':sum(x['delivery_success']for x in selected),
                'strict_success':sum(x['success']for x in selected),'transient_section_duplicates':sum(x['transient_section_duplicate']for x in selected),
                'missing_sequences':sum(len(x['missing_sequences'])for x in selected),
                'same_generation_duplicate_applications':sum(len(x['duplicate_applied_within_page'])for x in selected),
                'wire_latency_ms':{'count':len(wires),'p50':q(wires,.5),'p95':q(wires,.95)},
                'first_progress_dom_ms':{'count':sum(x['first_progress_dom_from_click_ms']is not None for x in selected),'p50':q([x['first_progress_dom_from_click_ms']for x in selected if x['first_progress_dom_from_click_ms']is not None],.5)},
                'first_body_dom_ms':{'p50':q([x['first_body_dom_from_click_ms']for x in selected if x['first_body_dom_from_click_ms']is not None],.5),'p95':q([x['first_body_dom_from_click_ms']for x in selected if x['first_body_dom_from_click_ms']is not None],.95)},
                'clock_error_bound_ms':{'min':min(clock_bounds),'max':max(clock_bounds)},
                'catchup_completed_dom_ms':{'p50':q([x['catchup_to_completed_dom_ms']for x in selected if x['catchup_to_completed_dom_ms']is not None and x['delivery_success']],.5)},
                'body_visible_before_completion':sum(x.get('first_body_visible_before_complete',False)for x in selected)}
    save(ROOT/'跨场景指标与一致性分析.json',{'analyzed_at':datetime.now().isoformat(),'summaries':summaries,'cases':rows})
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__':asyncio.run(main())
