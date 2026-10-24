#!/usr/bin/env python3
"""真实HTTP/账本/检查点，回放或确定性Agent；仅绑定本机评测端口。"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), os.environ.get('SSE_EVAL_FROZEN_APP', str(BACKEND / 'app'))]
from dotenv import load_dotenv
load_dotenv(BACKEND / '.env')
OUTPUT = Path(os.environ['SSE_EVAL_OUTPUT']).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
for key in ['DASHSCOPE_API_KEY', 'BOCHA_API_KEY', 'OPENAI_API_KEY']:
    os.environ[key] = ''
os.environ['RESEARCH_ALLOWED_DOMAINS'] = ''
os.environ['EVAL_TRACE_ENABLED'] = 'false'
os.environ['SESSION_EVIDENCE_ENABLED'] = 'false'

from fastapi import FastAPI, HTTPException, Request
from core.database import SessionLocal, get_db
from models.user import User
from models.chat import ChatSession, ChatMessage
from models.research import AgentEvent, ResearchRun
from service.deep_research_v2.checkpointer import checkpointer_lifespan
from service.deep_research_v2 import service as service_module
from service.deep_research_v2.graph import build_graph, graph_config, pack_state
from service.deep_research_v2.state import create_initial_state, GRAPH_VERSION
from service.deep_research_v2.delivery import make_commit_result
from service.research_events import (ResearchEventRecorder, acquire_research_run_lease,
                                    release_research_run_lease)
import importlib
routes = importlib.import_module('router.research_router')
# 首跑补充测试使用确定性上下文，真实API仍负责权限、消息、租约和图输入。
routes.get_research_context_service = lambda: SimpleNamespace(build=lambda **kw: {
    'original_query': kw['current_query'], 'resolved_query': kw['current_query'],
    'conversation_context': [], 'memory_context': [],
})
from eval.scripts.采集真实追问开发集 import create_identity, create_session, save_user


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')
    tmp.replace(path)


def log(run_id, kind, **fields):
    folder = JOBS[run_id]['folder']
    with (folder / '服务端测量日志.jsonl').open('a') as f:
        f.write(json.dumps({'kind': kind, 'epoch_ms': time.time() * 1000,
                            'monotonic': time.monotonic(), **fields}, ensure_ascii=False, default=str) + '\n')


JOBS = {}
OWNER = None
BASE = None
EXTERNAL = {'attempts': 0}
# 所有模型/搜索调用都禁止，防止夹具意外退回在线推理。
import httpx
_http_send = httpx.AsyncClient.send
async def local_send(self, request, *args, **kwargs):
    if request.url.host not in {'localhost', '127.0.0.1', '::1'}:
        EXTERNAL['attempts'] += 1
        raise RuntimeError('浏览器机制评测禁止外部HTTP调用')
    return await _http_send(self, request, *args, **kwargs)
httpx.AsyncClient.send = local_send

ROLE_NAMES = {'architect': 'ChiefArchitect', 'scout': 'DeepScout',
              'data_analyst': 'DataAnalyst', 'writer': 'LeadWriter', 'critic': 'CriticMaster'}


def sections(source_id):
    return [f'## 章节{i}\n\nSSE正文标记-题目{source_id}-章节{i}。'
            + '本段是确定性浏览器机制夹具，用于检验流传输、页面投影和检查点恢复，不代表真实调研结论。' * 14
            for i in range(1, 5)]


class Agent:
    def __init__(self, role):
        self.role, self.name = role, ROLE_NAMES[role]
        self.llm_call_count = self.llm_total_tokens = 0
    async def process(self, state):
        rid = state['_run_id']; job = JOBS[rid]; sid = job['source_id']
        log(rid, 'node_start', role=self.role, node=state['_node'], attempt=state['_attempt'])
        emit = state['_emit_event']
        if self.role == 'architect':
            emit({'type': 'thought', 'content': {'content': job['progress_marker']}})
            state['outline'] = [{'id': f'sec_{i}', 'title': f'章节{i}', 'status': 'pending'} for i in range(1, 5)]
            await asyncio.sleep(.3)
        elif self.role == 'scout':
            state['facts'] = [{'id': '机制事实', 'content': '确定性占位事实，非检索结果', 'source_url': 'https://example.invalid/fixture'}]
            emit({'type': 'search_results', 'content': {'results': [{'id': 'fixture', 'title': '机制夹具', 'url': 'https://example.invalid/fixture'}]}})
            await asyncio.sleep(.3)
        elif self.role == 'data_analyst':
            await asyncio.sleep(.2)
        elif self.role == 'writer':
            texts = sections(sid)
            for i, text in enumerate(texts, 1):
                emit({'type': 'section_content', 'content': {'section_id': f'sec_{i}',
                     'section_title': f'章节{i}', 'content': text}})
                await asyncio.sleep(1.25)
            # 断线场景在首章发出后中断，当前节点产物尚未提交。
            state['final_report'] = '\n\n'.join(texts)
            state['draft_sections'] = {f'sec_{i}': s for i, s in enumerate(texts, 1)}
        elif self.role == 'critic':
            state.update(quality_verdict='pass', quality_score=8, phase='completed')
            emit({'type': 'review', 'content': {'verdict': 'pass', 'overall_score': 8, 'issues': []}})
            await asyncio.sleep(.15)
        log(rid, 'node_return', role=self.role, node=state['_node'])
        return state


def agents(config):
    return {role: Agent(role) for role in ROLE_NAMES}
service_module.create_agents = agents

# 记录真实record/交付提交返回时刻，不用历史event.timestamp替代。
_record = ResearchEventRecorder.record
def record(self, event):
    started = time.time() * 1000
    result = _record(self, event)
    if self.run_id in JOBS:
        log(self.run_id, 'event_committed', seq=result.seq, event_type=result.type,
            record_started_epoch_ms=started)
    return result
ResearchEventRecorder.record = record
_commit_factory = service_module.make_commit_result
def commit_factory(**kwargs):
    original = _commit_factory(**kwargs)
    def commit(snapshot):
        result = original(snapshot)
        if kwargs['run_id'] in JOBS:
            log(kwargs['run_id'], 'delivery_committed', seq=result.seq)
        return result
    return commit
service_module.make_commit_result = commit_factory


async def graph_for(rid):
    return build_graph(agents(None), BASE.checkpointer, run_id=rid, user_id=str(OWNER.id),
                       session_id=JOBS[rid]['session_id'],
                       execute_stage=service_module.make_execute_stage(run_id=rid, session_id=JOBS[rid]['session_id']))


class ReplayService:
    async def read_state(self, rid):
        return await BASE.read_state(rid)
    async def research(self, *, run_id, session_id, user_id, turn_id, lease, **kwargs):
        job = JOBS[run_id]; recorder = ResearchEventRecorder(session_id=session_id, run_id=run_id,
            user_id=user_id, turn_id=turn_id, lease=lease)
        status = 'paused'
        try:
            for item in job['events']:
                if item['type'] == 'research_complete':
                    graph = await graph_for(run_id)
                    state = (await graph.aget_state(graph_config(run_id))).values['research']
                    state.update(status='completed', phase='completed', quality_verdict='pass',
                                 completion_reason='passed', final_report=job['expected_report'])
                    await graph.aupdate_state(graph_config(run_id), pack_state(state), as_node='finish')
                    snapshot = await graph.aget_state(graph_config(run_id))
                    envelope = await asyncio.to_thread(commit_factory(recorder=recorder, lease=lease,
                        run_id=run_id, session_id=session_id, user_id=user_id, turn_id=turn_id), snapshot)
                    status = 'completed'
                else:
                    envelope = await asyncio.to_thread(recorder.record, {'type': item['type'], **item['payload']})
                yield service_module.encode_sse(envelope.to_wire())
                await asyncio.sleep(.025)
            yield 'data: [DONE]\n\n'
        finally:
            await asyncio.to_thread(release_research_run_lease, lease, status)


def service(request):
    rid = request.path_params.get('run_id')
    return ReplayService() if rid in JOBS and JOBS[rid]['mode'] == 'replay' else BASE
routes.get_research_service = service


class MeasureWire:
    """对本次真实HTTP帧计时，并注入重复/缺口；事件账本不造重复或缺口。"""
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        first = scope['type'] == 'http' and scope['path'] == '/api/research/stream'
        if first:
            messages = []
            while True:
                message = await receive(); messages.append(message)
                if not message.get('more_body'): break
            body = json.loads(b''.join(m.get('body', b'') for m in messages))
            job = next(j for j in JOBS.values() if j['session_id'] == body['session_id'])
            rid = body['turn_id']; job['run_id'] = rid; JOBS[rid] = job
            saved = json.loads((job['folder'] / '运行身份与配置.json').read_text())
            saved['run_id'] = rid; save(job['folder'] / '运行身份与配置.json', saved)
            original_receive = receive
            async def replay_receive():
                return messages.pop(0) if messages else await original_receive()
            receive = replay_receive
        if scope['type'] != 'http' or ('/resume/' not in scope['path'] and not first):
            return await self.app(scope, receive, send)
        if not first: rid = scope['path'].rsplit('/', 1)[-1]
        if rid not in JOBS: return await self.app(scope, receive, send)
        job = JOBS[rid]
        log(rid, 'http_resume_start', query_string=scope['query_string'].decode())
        async def transmit(message):
            if message['type'] != 'http.response.body' or not message.get('body'):
                return await send(message)
            body = message['body'].decode()
            if not body.startswith('data: ') or '[DONE]' in body:
                return await send(message)
            event = json.loads(body[6:].strip())
            scenario = job['scenario'].removeprefix('fresh_')
            if scenario == 'gap' and event['type'] == 'section_content' and not job.get('injected'):
                job['injected'] = True
                log(rid, 'frame_dropped', seq=event['seq']); return
            copies = 2 if scenario == 'duplicate' and event['type'] == 'section_content' and not job.get('injected') else 1
            if copies == 2: job['injected'] = True; log(rid, 'duplicate_injected', seq=event['seq'])
            for copy in range(copies):
                sent = time.time() * 1000
                wire = {**event, 'measurement': {'sent_epoch_ms': sent, 'copy': copy,
                         'attempt_id': uuid.uuid4().hex}}
                payload = ('data: ' + json.dumps(wire, ensure_ascii=False) + '\n\n').encode()
                log(rid, 'frame_send', seq=event['seq'], event_type=event['type'],
                    sent_epoch_ms=sent, copy=copy, bytes=len(payload))
                await send({**message, 'body': payload})
        await self.app(scope, receive, transmit)


@asynccontextmanager
async def lifespan(app):
    global OWNER, BASE
    owner = create_identity(OUTPUT.name)
    with SessionLocal() as db: OWNER = db.query(User).filter_by(id=uuid.UUID(owner)).one(); db.expunge(OWNER)
    async with checkpointer_lifespan(app) as saver:
        BASE = service_module.DeepResearchV2Service(saver)
        save(OUTPUT / '服务身份.json', {'user_id': owner, 'authentication': '隔离服务依赖注入，未测试真实登录', 'external_http_attempts': EXTERNAL['attempts']})
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(MeasureWire)
app.include_router(routes.router, prefix='/api')
app.dependency_overrides[routes.get_current_user_required] = lambda: OWNER


@app.get('/evaluation/clock')
def clock(): return {'epoch_ms': time.time() * 1000}


@app.get('/evaluation/health')
def health(): return {'ready': BASE is not None, 'external_attempts': EXTERNAL['attempts']}


@app.post('/evaluation/create')
async def create(request: Request):
    data = await request.json(); sid = int(data['source_id']); scenario = data['scenario']
    tasks = [json.loads(x) for x in Path(os.environ['SSE_EVAL_DATASET']).read_text().splitlines()]
    task = next(t for t in tasks if t['source']['id'] == sid)
    query = task['turns'][0]['user_query']; rid = str(uuid.uuid4())
    session_id = create_session(str(OWNER.id), f'SSE浏览器评测-题目{sid}-{scenario}')
    fresh = scenario.startswith('fresh')
    if not fresh: save_user(str(OWNER.id), session_id, rid, rid, query)
    mode = 'graph' if fresh or scenario in {'disconnect', 'reload', 'graph_control'} else 'replay'
    folder = OUTPUT / data['label']; folder.mkdir()
    job = {'run_id': rid, 'session_id': session_id, 'source_id': sid, 'scenario': scenario,
           'query': query, 'mode': mode, 'folder': folder,
           'progress_marker': f'SSE进度标记-题目{sid}', 'body_marker': f'SSE正文标记-题目{sid}-章节1'}
    JOBS[rid] = job
    if not fresh:
        initial = create_initial_state(query, session_id, [])
        graph = await graph_for(rid)
        await graph.aupdate_state(graph_config(rid), pack_state(initial), as_node='__start__' if mode == 'graph' else 'analyze')
        lease = acquire_research_run_lease(run_id=rid, session_id=session_id, user_id=str(OWNER.id))
        release_research_run_lease(lease, 'paused')
        with SessionLocal() as db:
            run = db.query(ResearchRun).filter_by(run_id=rid).one(); run.graph_version = GRAPH_VERSION; db.commit()
    if mode == 'replay':
        source = next(Path(os.environ['SSE_EVAL_SOURCE']).glob(f'题目{sid}-*'))
        records = json.loads((source / '最终事件账本.json').read_text())
        # 26题的最新完整轨迹存放于重试目录。
        if sid == 26:
            source = Path(os.environ['SSE_EVAL_RETRY']) / '题目26-检查点提交前'
            records = json.loads((source / '最终事件账本.json').read_text())
        final = next((e for e in reversed(records) if e['type'] == 'research_complete'), None)
        if final is None:
            saved_state = json.loads((source / '最终研究状态.json').read_text())
            report = saved_state.get('final_report') or (saved_state.get('research') or {}).get('final_report') or (saved_state.get('draft') or {}).get('final_report')
            if not report: raise ValueError('回放来源缺少报告正文')
            final = {'type': 'research_complete', 'payload': {'final_report': report}, 'fixture_synthetic': True}
            records.append(final)
            job['fixture_completion'] = '历史终态检查点正文补充的合成交付，不代表原运行交付成功'
        job['source_ledger'] = str(source / '最终事件账本.json')
        job['expected_report'] = job['body_marker'] + '\n\n' + final['payload']['final_report']
        marked_phase = marked_body = False
        for e in records:
            payload = e['payload']
            if e['type'] == 'phase' and not marked_phase:
                payload['content'] = str(payload.get('content') or '') + ' · ' + job['progress_marker']; marked_phase = True
            if e['type'] == 'section_content' and not marked_body:
                payload['content']['content'] = job['body_marker'] + '\n\n' + payload['content']['content']; marked_body = True
        job['events'] = records
        save(folder / '回放事件夹具.json', records)
    else: job['expected_report'] = '\n\n'.join(sections(sid))
    view = {k: v for k, v in job.items() if k not in {'events', 'folder'}}
    view.update(user_id=str(OWNER.id), auth_token='sse-local-evaluation', folder=str(folder))
    save(folder / '运行身份与配置.json', view)
    return view


@app.get('/api/sessions/{sid}')
def session(sid: str):
    with SessionLocal() as db:
        s = db.query(ChatSession).filter_by(id=uuid.UUID(sid), user_id=OWNER.id).one()
        messages = db.query(ChatMessage).filter_by(session_id=s.id).order_by(ChatMessage.created_at, ChatMessage.id).all()
        return {'id': sid, 'title': s.title, 'session_type': s.session_type,
            'messages': [{c.name: getattr(m, c.name) for c in m.__table__.columns} for m in messages]}


@app.get('/api/knowledge-bases')
def knowledge(): return []


@app.get('/api/sessions')
def sessions(): return []


@app.get('/evaluation/result/{rid}')
async def result(rid: str):
    job = JOBS[rid]
    state = await BASE.read_state(rid)
    with SessionLocal() as db:
        events = db.query(AgentEvent).filter_by(run_id=rid).order_by(AgentEvent.seq).all()
        messages = db.query(ChatMessage).filter_by(run_id=rid, role='assistant').all()
        run = db.query(ResearchRun).filter_by(run_id=rid).one()
        ledger = [{'seq': e.seq, 'type': e.event_type, 'payload': e.payload} for e in events]
        delivery = {'status': run.status, 'assistant_messages': len(messages),
            'complete_events': sum(e.event_type=='research_complete' for e in events),
            'final_checkpoint_id': run.final_checkpoint_id, 'fencing_token': run.fencing_token,
            'report_hashes': [hashlib.sha256(m.content.encode()).hexdigest() for m in messages]}
    save(job['folder'] / '最终研究状态.json', state)
    save(job['folder'] / '事件账本.json', ledger)
    save(job['folder'] / '交付状态.json', delivery)
    summary = {'state': state, 'ledger': ledger, 'delivery': delivery,
               'expected_report': job['expected_report'], 'external_http_attempts': EXTERNAL['attempts']}
    save(job['folder'] / '服务端最终结果.json', summary)
    return summary


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('SSE_EVAL_PORT', '18765')),
                log_level='warning')
