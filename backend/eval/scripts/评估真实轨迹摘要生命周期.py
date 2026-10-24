"""在真实历史的最后研究输入处，独立回放当前外置、摘要提交及再利用机制。"""
import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / 'app')]
from dotenv import load_dotenv
load_dotenv(BACKEND / '.env')
from config.llm_config import LLMConfig, get_summary_model
from eval.verification.replay import ReplayStore
from service.agent_history import HistoryScope, bind_agent_history, content_hash, exception_details
from service.agent_context import prepare_agent_messages
from service.context_policy import legacy_policy
from service.deep_research_v2.agents.base import BaseAgent
from service.qwen_token_counter import get_qwen_token_counter
from service.verification_budget import task_budget


def save(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


class Agent(BaseAgent):
    def _output_limit(self): return 20000
    async def process(self, state): raise NotImplementedError('不执行研究主模型调用')


async def main(output):
    output.mkdir(exist_ok=False)
    config, policy, counter = LLMConfig(), legacy_policy(), get_qwen_token_counter()
    # 使用实际新运行环境配置；审计默认关闭，核验由另一独立回放执行。
    save(output / '冻结策略.json', policy.snapshot())
    results = []
    for topic in [16, 31, 35, 46, 49]:
        source = BACKEND / f'eval/data/通用研究中文5题重跑-20261001-231000/题目{topic}'
        if not (source / '智能体原始历史.json').exists():
            source = BACKEND / f'eval/data/通用研究中文5题重跑-20261001-201500/题目{topic}'
        history_file = source / '智能体原始历史.json'
        history = json.loads(history_file.read_text())
        agents = {r['id']: r['agent_name'] for r in json.loads((source / '智能体上下文.json').read_text())}
        target = next(r for r in reversed(history) if r['kind'] == 'llm_input' and agents[r['context_id']] == 'DeepScout')
        rows = [{**r, 'source_id': r['id'], 'agent_name': 'DeepScout'} for r in history
                if r['context_id'] == target['context_id'] and r['seq'] <= target['seq']]
        rows.sort(key=lambda r: r['seq'])
        d = output / f'题目{topic}'; d.mkdir()
        store = ReplayStore(d / '独立工作历史', rows)
        scope = HistoryScope(store, str(uuid.uuid4()), str(uuid.uuid4()), None,
            node=target.get('node'), invocation_id=target['invocation_id'], contract=target['payload'].get('contract', {}), context_policy=policy.snapshot())
        agent = Agent('DeepScout', '真实历史上下文处理与摘要提交回放', config.api_key, config.base_url, get_summary_model())
        result = {'topic': topic, 'input_seq': target['seq'], 'source': str(history_file),
                  'source_hash': hashlib.sha256(history_file.read_bytes()).hexdigest(), 'production_database_writes': False,
                  'initial_summary_version': 0, 'initial_covered_seq': 0, 'scope': 'fresh_rebuild_of_observed_history_at_last_research_input'}
        started = time.monotonic()
        with task_budget({'max_requests': 24, 'max_seconds': 300, 'max_tokens_by_model': {get_summary_model(): 600000}}) as ledger:
            try:
                with bind_agent_history(scope):
                    prepared = await prepare_agent_messages(agent, scope, target['payload']['system'], target['payload']['content'],
                                                            target['call_id'], policy.output_limits['DeepScout'])
                    save(d / '处理后主模型输入.json', prepared)
                    result.update(status='completed', prepared_tokens=counter.count_messages(prepared))
                    saved_before = store.data['summary_version']
                    calls_before = len(ledger.requests)
                    again = await prepare_agent_messages(agent, scope, target['payload']['system'], target['payload']['content'],
                                                         target['call_id'], policy.output_limits['DeepScout'])
                    result.update(reuse_additional_requests=len(ledger.requests)-calls_before,
                                  reuse_summary_version_before=saved_before, reuse_summary_version_after=store.data['summary_version'],
                                  repeated_prepared_tokens=counter.count_messages(again))
            except Exception as exc:
                result.update(status='error', error=exception_details(exc))
            finally:
                result.update(cost=ledger.report(), duration_seconds=time.monotonic()-started,
                    summary_version=store.data['summary_version'], covered_seq=store.data['covered_seq'],
                    original_rows_unchanged=all(next(x for x in store.data['messages'] if x['source_id']==r['source_id'])==r for r in rows),
                    context_views=[r['payload'] for r in store.data['messages'][len(rows):] if r['kind']=='context_view'],
                    summary_errors=[r['payload'] for r in store.data['messages'][len(rows):] if r['kind']=='summary_error'])
                if store.data['summary']:
                    (d / '实际提交摘要.md').write_text(store.data['summary'])
                save(d / '运行结果.json', result)
                await agent.client.close()
        results.append(result);save(output / '汇总结果.json', results)
        print(json.dumps({k:result.get(k) for k in ['topic','status','prepared_tokens','summary_version','covered_seq','reuse_additional_requests','error']}, ensure_ascii=False),flush=True)
    save(output / '原始档案复核.json', {'unchanged': all(hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==r['source_hash'] for r in results)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main(parser.parse_args().output))
