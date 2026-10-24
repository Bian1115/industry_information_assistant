#!/usr/bin/env python3
"""真实服务/父研究定位/交付的开发集采集；创建隔离评测身份，不伪造父检查点。"""
import argparse
import asyncio
from contextlib import aclosing
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / 'app')]
SINGLE_SESSION_DATASET_SHA256 = '969bd91063c5fbf3c127803cc5636e7a6427598939676f825e9b85142a21c647'


def validate_single_session_dataset(path, tasks):
    """一次开发实验的显式入口，不改变其他数据集的默认会话边界。"""
    if (hashlib.sha256(Path(path).read_bytes()).hexdigest() != SINGLE_SESSION_DATASET_SHA256
            or [t.source.id for t in tasks] != [8, 25, 26, 27, 28]
            or any(t.split != 'development' or len(t.turns) != 6 for t in tasks)):
        raise ValueError('--single-session-development仅支持已冻结的20260927五题开发集')


def dialogue_groups(tasks, single_session=False):
    groups = [tasks] if single_session else [[task] for task in tasks]
    return [[(task, turn, number) for number, (task, turn) in enumerate(
        ((task, turn) for task in group for turn in task.turns), start=1)] for group in groups]


def load_resume(args, tasks):
    """仅恢复固定单会话开发集的首个失败轮，保留原始运行目录。"""
    path = getattr(args, 'resume_from', None)
    if path is None:
        return None
    if not getattr(args, 'single_session_development', False):
        raise ValueError('恢复连跑仅支持固定单会话开发集')
    validate_single_session_dataset(args.dataset, tasks)
    old = json.loads((path / '运行清单.json').read_text())
    groups = json.loads((path / '采集进度.json').read_text())
    if old.get('status') == 'running' or len(groups) != 1:
        raise ValueError('原采集必须已停止且只有一个会话')
    rows = groups[0].get('turns', [])
    if (not rows or rows[-1].get('completed') or not all(r.get('completed') for r in rows[:-1])
            or [r['turn_id'] for r in rows] != list(range(1, len(rows) + 1))):
        raise ValueError('仅允许从连续交付之后的首个失败轮恢复')
    items = dialogue_groups(tasks, True)[0]
    if len(rows) > len(items) or any(row['query'] != turn.user_query or row['source_id'] != task.source.id
                                  for row, (task, turn, _) in zip(rows, items)):
        raise ValueError('原运行轮次与冻结开发集不匹配')
    return old, groups[0]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temp.replace(path)


def create_identity(label):
    from core.database import SessionLocal
    from models.user import User
    owner = uuid.uuid4()
    # 禁用登录，仅用于隔离评测会话与运行权限；不读取或复用用户的私人会话。
    with SessionLocal() as db:
        db.add(User(id=owner, username='eval_' + owner.hex[:24],
                    email=owner.hex + '@eval.invalid', hashed_password='!disabled-evaluation',
                    is_active=False, is_superuser=False))
        db.commit()
    return str(owner)


def create_session(owner, title):
    from core.database import SessionLocal
    from models.chat import ChatSession
    sid = uuid.uuid4()
    with SessionLocal() as db:
        db.add(ChatSession(id=sid, user_id=uuid.UUID(owner), title=title, session_type='deepsearch'))
        db.commit()
    return str(sid)


def save_user(owner, sid, turn_id, run_id, query):
    from core.database import SessionLocal
    from models.chat import ChatSession, ChatMessage
    with SessionLocal() as db:
        assert db.query(ChatSession).filter_by(id=uuid.UUID(sid), user_id=uuid.UUID(owner)).one()
        db.add(ChatMessage(session_id=uuid.UUID(sid), turn_id=uuid.UUID(turn_id), role='user',
                           content=query, execution_mode='research', run_id=run_id, research_status='running'))
        db.commit()


async def export(service, row, directory):
    from eval.scripts.run_multiturn_deepresearch import collect_records
    from eval.multiturn import compression_points
    records = await collect_records(row['run_id'])
    save(directory / '工作历史.json', records)
    # 逐run事件与累计会话原文分别保存；后者支持旧引用回查及摘要回放。
    def session_records():
        from core.database import SessionLocal
        from models.agent_history import AgentContext, AgentMessage
        from models.chat import ChatSession, ChatMessage
        with SessionLocal() as db:
            ids = db.query(AgentMessage.context_id).filter(AgentMessage.run_id == row['run_id'])
            contexts = db.query(AgentContext).filter(AgentContext.id.in_(ids)).all()
            records = db.query(AgentMessage).filter(AgentMessage.context_id.in_(ids)).order_by(
                AgentMessage.context_id, AgentMessage.seq).all()
            sid = uuid.UUID(row['session_id'])
            session = db.query(ChatSession).filter_by(id=sid, user_id=uuid.UUID(row['user_id'])).one()
            chats = db.query(ChatMessage).filter_by(session_id=sid).order_by(
                ChatMessage.created_at, ChatMessage.id).all()
            return ([{column.name: getattr(message, column.name) for column in message.__table__.columns}
                     for message in records],
                    [{column.name: getattr(context, column.name) for column in context.__table__.columns}
                     for context in contexts],
                    [{column.name: getattr(message, column.name) for column in message.__table__.columns}
                     for message in chats],
                    {'session_id': str(session.id),
                     'semantic_context_version': session.semantic_context_version,
                     'semantic_context_summary': session.semantic_context_summary})
    cumulative, contexts, chats, session_summary = await asyncio.to_thread(session_records)
    save(directory / '会话累计工作历史.json', cumulative)
    save(directory / '会话智能体摘要与游标.json', contexts)
    save(directory / '会话原始消息.json', chats)
    save(directory / '会话摘要状态.json', session_summary)
    row['record_counts'] = dict(Counter(r['kind'] for r in records))
    row['compression_points'] = compression_points(records)
    checkpoint = await service.checkpointer.aget_tuple({'configurable': {'thread_id': row['run_id']}})
    if checkpoint:
        state = checkpoint.checkpoint.get('channel_values', {}).get('research') or {}
        save(directory / '研究状态.json', state)
        row.update(parent_run_id=state.get('parent_run_id'),
                   body_length_check=state.get('body_length_check'),
                   body_length_repair=state.get('body_length_repair'),
                   contract=state.get('research_contract'), active_constraints=state.get('active_constraints'),
                   execution_plan=state.get('execution_plan'), session_task=state.get('session_task'),
                   session_task_metrics=state.get('session_task_metrics'), phase=state.get('phase'),
                   quality_verdict=state.get('quality_verdict'), completion_reason=state.get('completion_reason'))
        save(directory / '契约与路由.json', {k: row.get(k) for k in (
            'run_id', 'parent_run_id', 'contract', 'active_constraints', 'execution_plan', 'session_task', 'session_task_metrics')})
    row['telemetry_status'] = 'collected' if records else 'no_records'


async def execute(args, tasks, budget, manifest):
    from eval.scripts.run_deepresearch_bench import benchmark_service, decode_service_event
    from service.research_context_service import get_research_context_service
    from service.verification_budget import task_budget
    single_session = getattr(args, 'single_session_development', False)
    if single_session:
        validate_single_session_dataset(args.dataset, tasks)
    resume = load_resume(args, tasks)
    owner = resume[0]['evaluation_user_id'] if resume else await asyncio.to_thread(create_identity, args.output.name)
    manifest['evaluation_user_id'] = owner
    save(args.output / '运行清单.json', manifest)
    results = []
    async with benchmark_service(2) as service:
        manifest['allowed_domains'] = service.resolve_allowed_domains()
        save(args.output / '运行清单.json', manifest)
        for items in dialogue_groups(tasks, single_session):
            source_ids = list(dict.fromkeys(task.source.id for task, _, _ in items))
            sid = resume[1]['session_id'] if resume else await asyncio.to_thread(create_session, owner, f'{args.output.name} 原题{source_ids}')
            task_dir = args.output / ('合并开发集30轮' if single_session else f'任务-{source_ids[0]}')
            task_row = {'source_id': source_ids[0] if len(source_ids) == 1 else None,
                        'source_ids': source_ids, 'session_id': sid, 'planned_turns': len(items), 'turns': []}
            results.append(task_row)
            history = []
            resume_turn = resume[1]['turns'][-1]['turn_id'] if resume else None
            if resume:
                task_row['turns'] = deepcopy(resume[1]['turns'][:-1])
                for old_row in resume[1]['turns']:
                    old_dir = args.resume_from / '合并开发集30轮' / f"第{old_row['turn_id']}轮"
                    shutil.copytree(old_dir, task_dir / old_dir.name)
                    if old_row['completed']:
                        history.extend([{'role': 'user', 'content': old_row['query']},
                                        {'role': 'assistant', 'content': (old_dir / '研究报告.md').read_text()}])
                save(task_dir / f'第{resume_turn}轮' / '恢复前轮次结果.json', resume[1]['turns'][-1])
            with task_budget(budget) as ledger:
                if resume:
                    old_cost = resume[1]['cost']
                    ledger.requests = deepcopy(old_cost['requests'])
                    ledger.reserved = deepcopy(old_cost['tokens_reserved_by_model'])
                    original_start = (resume[0].get('recovery') or {}).get('original_started_at') or resume[0]['date']
                    elapsed = max(0, (datetime.now() - datetime.fromisoformat(original_start)).total_seconds())
                    ledger.started = time.monotonic() - elapsed
                deadline = asyncio.get_running_loop().time() + budget['max_seconds']
                if resume:
                    deadline -= elapsed
                for task, turn, global_turn_id in items:
                    if resume and global_turn_id < resume_turn:
                        continue
                    recovering = resume and global_turn_id == resume_turn
                    row = {'turn_id': global_turn_id, 'source_id': task.source.id, 'source_turn_id': turn.turn_id,
                           'session_id': sid, 'user_id': owner,
                           'task_id': task.task_id, 'run_id': str(uuid.uuid4()), 'turn_uuid': str(uuid.uuid4()),
                           'query': turn.user_query, 'completed': False, 'started_at': datetime.now().isoformat()}
                    if recovering:
                        row = deepcopy(resume[1]['turns'][-1])
                        row['recovery'] = {'resumed_at': datetime.now().isoformat(),
                                           'previous_error': row.get('error'), 'source': str(args.resume_from)}
                        for key in ('error', 'error_type', 'ended_at', 'telemetry_error', 'live_export_error'):
                            row.pop(key, None)
                    task_row['turns'].append(row)
                    directory = task_dir / f'第{global_turn_id}轮'
                    save(directory / '轮次输入.json', {'row': row, 'conversation': history})
                    async def consume():
                        if recovering:
                            context = json.loads((directory / '会话任务输入.json').read_text())
                        else:
                            await asyncio.to_thread(save_user, owner, sid, row['turn_uuid'], row['run_id'], turn.user_query)
                            context = await asyncio.to_thread(get_research_context_service().build,
                                user_id=owner, session_id=sid, current_turn_id=row['turn_uuid'], current_query=turn.user_query)
                        save(directory / '会话任务输入.json', context)
                        row['history_metrics'] = context.get('history_metrics', {})
                        row['session_summary_metrics'] = context.get('session_summary_metrics', {})
                        row['session_task_builder_metrics'] = context.get('task_metrics', {})
                        async with aclosing(service.research(
                            query=turn.user_query, session_id=sid, user_id=owner, run_id=row['run_id'],
                            turn_id=row['turn_uuid'], research_context=context, resume=bool(recovering))) as stream:
                            async for chunk in stream:
                                event = decode_service_event(chunk)
                                if event is None:
                                    continue
                                with (directory / '研究事件.jsonl').open('a') as f:
                                    f.write(json.dumps(event, ensure_ascii=False) + '\n')
                                data = event.get('data') if isinstance(event.get('data'), dict) else event
                                if event['type'] == 'research_complete':
                                    article = data.get('final_report') or ''
                                    if not article.strip():
                                        raise ValueError('完成事件缺少正文')
                                    (directory / '研究报告.md').write_text(article)
                                    row.update(completed=True, delivery=data)
                                    history.extend([{'role':'user','content':turn.user_query},
                                                    {'role':'assistant','content':article}])
                                elif event['type'] in {'error', 'research_cancelled', 'research_paused'}:
                                    raise RuntimeError(str(data))
                        if not row['completed']:
                            raise RuntimeError('未收到完成事件')
                    async def observe():
                        while True:
                            await asyncio.sleep(30)
                            try:
                                await export(service, row, directory)
                                save(args.output / '采集进度.json', results)
                            except Exception as exc:
                                row['live_export_error'] = str(exc)
                    observer = asyncio.create_task(observe())
                    print(json.dumps({'event':'turn_start','source_id':task.source.id,'turn':global_turn_id,'source_turn':turn.turn_id,'run_id':row['run_id']},ensure_ascii=False),flush=True)
                    try:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise TimeoutError('整条对话时间预算耗尽')
                        await asyncio.wait_for(consume(), timeout=remaining)
                    except Exception as exc:
                        row.update(completed=False, error_type=type(exc).__name__, error=str(exc))
                    finally:
                        observer.cancel()
                        await asyncio.gather(observer, return_exceptions=True)
                        try:
                            await export(service, row, directory)
                        except Exception as exc:
                            row.update(telemetry_status='error', telemetry_error=str(exc))
                        row['ended_at'] = datetime.now().isoformat()
                        task_row['cost'] = ledger.report()
                        save(directory / '轮次结果.json', row)
                        save(task_dir / '逐轮结果.json', task_row)
                        save(args.output / '采集进度.json', results)
                    print(json.dumps({'event':'turn_end','source_id':task.source.id,'turn':global_turn_id,
                                      'completed':row['completed'],'error':row.get('error')},ensure_ascii=False),flush=True)
                    if not row['completed']:
                        break
            task_row['completed_turns'] = sum(r['completed'] for r in task_row['turns'])
            task_row['unexecuted_turns'] = [number for _, _, number in items[len(task_row['turns']):]]
            save(args.output / '采集进度.json', results)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--budget', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--single-session-development', action='store_true',
                   help='仅将固定20260927开发集按8→25→26→27→28串成同一会话30轮')
    p.add_argument('--resume-from', type=Path,
                   help='恢复已停止的单会话采集；新输出目录保留原失败与既有交付')
    args = p.parse_args()
    if args.output.exists() or args.report.exists():
        p.error('只允许新目录和新报告，不覆盖历史结果')
    from dotenv import load_dotenv
    load_dotenv(BACKEND / '.env', override=False)
    from eval.multiturn import load_tasks
    from service.context_policy import read_policy
    from service.verification_budget import TaskBudget
    from config.llm_config import get_config, get_summary_model
    tasks = load_tasks([args.dataset])
    assert len(tasks) == 5 and all(t.split == 'development' for t in tasks)
    if args.single_session_development:
        validate_single_session_dataset(args.dataset, tasks)
    policy = read_policy(args.policy)
    policy.require_supported()
    budget = json.loads(args.budget.read_text())
    TaskBudget(budget)
    resumed = load_resume(args, tasks)
    if resumed:
        config = get_config()
        models = {k:getattr(config.agents,k).model for k in ('architect','scout','data_analyst','writer','critic')}
        if (resumed[0]['context_policy']['config_hash'] != policy.fingerprint
                or resumed[0]['models'] != models or resumed[0]['budget'] != budget
                or resumed[0].get('summary_model') != get_summary_model()):
            raise ValueError('恢复连跑必须沿用原模型、压缩配置与对话预算')
    args.output.mkdir(parents=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save(args.output / '配置快照.json', policy.snapshot())
    os.environ['CONTEXT_ABLATION_CONFIG'] = str((args.output / '配置快照.json').resolve())
    # 会话上下文在研究run创建前构建，同样读取完整冻结策略。
    os.environ['CONTEXT_COMPRESSION_ENABLED'] = '1' if policy.compression_enabled else '0'
    os.environ['SESSION_CONTEXT_COMPRESSION_ENABLED'] = '1' if policy.effective_session_compression_enabled else '0'
    shutil.copyfile(args.dataset, args.output / '开发集5题.jsonl')
    save(args.output / '对话总预算.json', budget)
    hashes = {}
    for folder in (BACKEND / 'app', BACKEND / 'eval'):
        for file in folder.rglob('*'):
            if not file.is_file() or file.suffix not in {'.py', '.txt'} or any(
                    x in file.parts for x in ('data', '__pycache__', '.venv')):
                continue
            relative = file.relative_to(BACKEND)
            target = args.output / '代码快照' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
            hashes[str(relative)] = hashlib.sha256(file.read_bytes()).hexdigest()
    config = get_config()
    manifest = {'date':datetime.now().isoformat(), 'status':'running', 'planned_tasks':5,
                'planned_turns':sum(len(t.turns) for t in tasks), 'source_sha256':hashes,
                'source_ids':[t.source.id for t in tasks], 'budget':budget, 'context_policy':policy.snapshot(),
                'history_mode':'production_session_task_builder', 'execution_entry':'validated_session_execution_plan',
                'single_session_development':args.single_session_development,
                'compression_enabled':policy.compression_enabled,
                'session_compression_enabled':policy.effective_session_compression_enabled,
                'planned_sessions':1 if args.single_session_development else len(tasks),
                'agent_history_continues_across_turns':False,
                'budget_scope':'whole_30_turn_dialogue' if args.single_session_development else 'per_task_dialogue',
                'models':{k:getattr(config.agents,k).model for k in ('architect','scout','data_analyst','writer','critic')},
                'summary_model':get_summary_model(), 'pid':os.getpid(), 'max_iterations':2,
                'limitations':['实时搜索未冻结','未进行人工质量标注','与旧无路由采集口径不同']}
    if resumed:
        manifest['recovery'] = {'original_output':str(args.resume_from.resolve()),
            'original_started_at':(resumed[0].get('recovery') or {}).get('original_started_at') or resumed[0]['date'],
            'session_id':resumed[1]['session_id'],
            'from_turn':resumed[1]['turns'][-1]['turn_id'],
            'run_id':resumed[1]['turns'][-1]['run_id'], 'turn_uuid':resumed[1]['turns'][-1]['turn_uuid'],
            'previous_error':resumed[1]['turns'][-1].get('error'),
            'inherited_delivered_turns':len(resumed[1]['turns'])-1,
            'inherited_ledger_requests':len(resumed[1]['cost']['requests']),
            'code_repaired':True, 'strict_uninterrupted_run':False}
    save(args.output / '运行清单.json', manifest)
    args.report.write_text(f'# 真实追问开发集轨迹结果与分析\n\n日期：{manifest["date"]}。状态：运行中。\n\n原始目录：{args.output.resolve()}\n\n原题8、25、26、27、28，共30轮；计划会话数{manifest["planned_sessions"]}；预算范围{manifest["budget_scope"]}。使用生产会话任务构建器，按验证后的编辑、补证或规划路径执行，智能体内部历史按研究run隔离。\n\n已测量：数据及配置校验通过。尚未评估：数据库可用性、完整运行结果、模型质量与契约遵守率。\n')
    try:
        results = asyncio.run(execute(args, tasks, budget, manifest))
        manifest.update(status='finished', completed_turns=sum(r['completed'] for t in results for r in t['turns']))
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                        error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest['ended_at'] = datetime.now().isoformat()
        manifest['changed_source_files'] = [f for f,h in hashes.items() if not (BACKEND/f).exists()
                                            or hashlib.sha256((BACKEND/f).read_bytes()).hexdigest()!=h]
        save(args.output / '运行清单.json', manifest)
        with args.report.open('a') as f:
            f.write('\n运行器终态：' + manifest['status'] + '。具体计数及失败原因见采集进度.json；需补充逐题分析。\n')


if __name__ == '__main__':
    main()
