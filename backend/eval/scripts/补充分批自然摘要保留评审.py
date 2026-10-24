"""对重要信息目录超预算的自然摘要分批补评，独立保存账本，不覆盖原失败。"""
import argparse
import asyncio
import importlib.util
import json
import time
import uuid
from pathlib import Path

spec = importlib.util.spec_from_file_location('natural_audit', Path(__file__).with_name('核验最新轨迹自然摘要.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


async def evaluate(directory, retry_failed=False):
    result = json.loads((directory / '运行结果.json').read_text())
    if result['status'] != 'completed' or result['quality_status'] != 'error': return
    previous = directory / '内容保留分批补评'
    out = previous
    if out.exists():
        if not retry_failed or json.loads((out / '补评结果.json').read_text()).get('status') != 'error': return
        out = directory / '内容保留分批补评-协议修复复评'
        if out.exists(): return
    out.mkdir()
    freeze = json.loads((directory / '冻结摘要及生成输入.json').read_text())
    catalog = json.loads((directory / '原文证据单元目录.json').read_text())
    rows = json.loads((directory / '摘要分句目录.json').read_text())
    units = {u['unit_id']: u for u in catalog}
    sids = {r['sentence_id'] for r in rows}
    counter, config, model = m.get_qwen_token_counter(), m.LLMConfig(), m.get_summary_model()
    agent = m.OfflineAgent(freeze['agent'], '自然摘要内容保留分批补评', config.api_key, config.base_url, model)
    store = m.ReplayStore(out / '独立回放历史', [])
    scope = m.HistoryScope(store, str(uuid.uuid4()), str(uuid.uuid4()), None)
    entries, tick = [], time.monotonic()

    async def call(prompt, payload, stage, limit=6000):
        req = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': m.canonical_json(payload)}]
        tokens = counter.count_messages(req)
        if tokens > 65536: raise ValueError('补评输入超预算，未截断')
        i, start = len(entries) + 1, time.monotonic()
        e = {'stage': stage, 'input_tokens': tokens}; entries.append(e)
        m.save(out / f'请求-{i:03d}-{stage}.json', req)
        m.say(sample=freeze['id'], stage=stage, call=i, supplement=True)
        try:
            async with asyncio.timeout(120):
                raw = await agent._call_llm_impl(prompt, req[1]['content'], True, 0.0, limit,
                    history_call_id=str(uuid.uuid4()), prepared_messages=req, model_override=model)
            (out / f'返回-{i:03d}-{stage}.txt').write_text(raw)
            e['status'] = 'completed'
            return json.loads(raw)
        except Exception as exc:
            e.update(status='error', error=m.exception_details(exc)); raise
        finally:
            e['duration_seconds'] = time.monotonic() - start
            m.save(out / '阶段调用明细.json', entries)

    status = {'status': 'running', 'original_quality_error': result.get('quality_error'), 'scope': 'unchanged_actual_final_generation_input'}
    with m.task_budget({'max_requests': 25, 'max_seconds': 1200, 'max_tokens_by_model': {model: 2_000_000}}) as ledger:
        try:
            with m.bind_agent_history(scope):
                batches, batch, tokens = [], [], 0
                for u in catalog:
                    compact = {'unit_id': u['unit_id'], 'text': u['text']}
                    n = counter.count_text(m.canonical_json(compact)) + 10
                    if batch and tokens + n > 22000:
                        batches.append(batch); batch, tokens = [], 0
                    batch.append(compact); tokens += n
                if batch: batches.append(batch)
                candidates = []
                for i, batch in enumerate(batches, 1):
                    available = {u['unit_id'] for u in batch}
                    old_file = previous / f'返回-{i:03d}-partial_important_checklist.txt'
                    if out != previous and old_file.exists():
                        ref = json.loads(old_file.read_text())
                        if any(not c.get('unit_ids') or not set(c['unit_ids']) <= available for c in ref['checkpoints']):
                            ref = await call(m.REFERENCE.replace('最多16项', '最多12项') + '\n修复上次输出的越界证据编号，所有编号仅从本批evidence_units选择；不要猜编号。',
                                {'evidence_units': batch, 'invalid_previous_output': ref}, 'checklist_id_protocol_repair')
                    else:
                        ref = await call(m.REFERENCE.replace('最多16项', '最多12项'), {'evidence_units': batch}, 'partial_important_checklist')
                    for j, c in enumerate(ref['checkpoints'], 1):
                        assert c['unit_ids'] and all(u in available for u in c['unit_ids'])
                        candidates.append({**c, 'id': f'b{i:02d}k{j:02d}'})
                m.save(out / '分批重要信息候选.json', candidates)
                merge = '''仅依据候选清单选择最多16项最重要且互不重复的信息，兼顾数字/条件/来源边界、操作结果、缺口、资源与用户目标。不新增内容，不看摘要。仅返回JSON {"selected_ids":["候选id"]}。'''
                ids = (await call(merge, {'candidates': candidates}, 'important_checklist_selection'))['selected_ids']
                assert 1 <= len(ids) <= 16 and len(set(ids)) == len(ids) and all(i in {c['id'] for c in candidates} for i in ids)
                cmap = {c['id']: c for c in candidates}
                checks = [{**cmap[i], 'id': f'k{n:02d}', 'candidate_id': i,
                           'references': [units[u] for u in cmap[i]['unit_ids']]} for n, i in enumerate(ids, 1)]
                m.save(out / '重要信息参考清单.json', {'checkpoints': checks})
                quality = await call(m.QUALITY, {'checkpoints': checks, 'summary_sentences': rows}, 'summary_retention_review', 8000)
                assert len(quality['coverage']) == len(checks) and {r['id'] for r in quality['coverage']} == {c['id'] for c in checks}
                for r in quality['coverage']:
                    assert r['status'] in {'preserved', 'partial', 'missing', 'contradicted'}
                    assert all(i in sids for i in r['summary_sentence_ids'])
                    assert r['status'] == 'missing' or r['summary_sentence_ids']
                for r in quality.get('issues', []): assert r['summary_sentence_ids'] and all(i in sids for i in r['summary_sentence_ids'])
                m.save(out / '摘要内容保留模型初评.json', quality)
                status.update(status='completed', coverage_counts=dict(m.Counter(r['status'] for r in quality['coverage'])),
                              batch_count=len(batches), candidate_count=len(candidates))
        except Exception as exc:
            status.update(status='error', error=m.exception_details(exc))
        finally:
            status.update(duration_seconds=time.monotonic()-tick, cost=ledger.report())
            m.save(out / '补评结果.json', status)
            await agent.client.close()
    m.say(sample=freeze['id'], supplement=True, status=status['status'])


async def main(args):
    semaphore = asyncio.Semaphore(2)
    async def run(d):
        async with semaphore: await evaluate(d, retry_failed=args.retry_failed)
    await asyncio.gather(*(run(d) for d in args.directory.iterdir() if (d / '运行结果.json').exists()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--retry-failed', action='store_true', help='复用已合法批次，另存一轮编号协议修复复评')
    asyncio.run(main(p.parse_args()))
