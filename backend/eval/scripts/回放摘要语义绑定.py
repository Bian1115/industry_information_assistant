"""复用真实摘要与编号输出，验证一次语义判断及定向修复；可选执行逐条NLI。"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import uuid

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / 'app')]
from dotenv import load_dotenv
load_dotenv(BACKEND / '.env')
from config.llm_config import LLMConfig, get_summary_model
from eval.verification.replay import ReplayStore
from service.agent_context import scoped_input_budget
from service.agent_history import HistoryScope, bind_agent_history, record_agent_message, content_hash
from service.context_policy import legacy_policy, NLISettings
from service.deep_research_v2.agents.base import BaseAgent
from service.qwen_token_counter import get_qwen_token_counter
from service.summary_claim_binding import bind_summary_claims, verify_bound_claims
from service.summary_message_verification import CITATION_OUTPUT_TOKENS
from service.verification_budget import task_budget


class JudgeAgent(BaseAgent):
    def _output_limit(self): return CITATION_OUTPUT_TOKENS
    async def process(self, state): raise NotImplementedError('只执行冻结输入回放')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


async def execute(args):
    args.output.mkdir(exist_ok=False)
    prose = (args.input / '生成摘要.md').read_text()
    merged = args.input / '引用补齐合并输出.json'
    raw = (merged if merged.exists() else args.input / '模型原始编号输出-1.txt').read_text()
    context = json.loads((args.input / '上下文编号映射.json').read_text())
    rows = sorted(context['rows'].values(), key=lambda r: r['seq'])
    store = ReplayStore(args.output / '独立工作历史', rows)
    policy = legacy_policy()
    scope = HistoryScope(store, str(uuid.uuid4()), str(uuid.uuid4()), None, context_policy=policy.snapshot())
    config = LLMConfig()
    agent = JudgeAgent('DeepScout', '摘要绑定回放', config.api_key, config.base_url, get_summary_model())
    counter = get_qwen_token_counter()
    save(args.output / '冻结输入.json', {'input': str(args.input), 'prose': prose, 'model_output': raw,
                                      'context': context, 'summary_hash': content_hash(prose)})

    async def call_stage(messages, stage):
        call_id = str(uuid.uuid4())
        tokens = counter.count_messages(messages)
        if tokens > scoped_input_budget(scope, agent.name, CITATION_OUTPUT_TOKENS, summary=True):
            raise ValueError(f'{stage}超出输入预算')
        await record_agent_message(agent.name, 'summary_stage', {'stage': stage, 'input_tokens': tokens}, call_id)
        answer = await agent._call_llm_impl(messages[0]['content'], messages[1]['content'], True, 0.0,
            CITATION_OUTPUT_TOKENS, history_call_id=call_id, prepared_messages=messages, model_override=get_summary_model())
        await record_agent_message(agent.name, 'summary_output', {'stage': stage, 'content': answer}, call_id)
        return answer

    result = {'status': 'not_run', 'nli_status': 'not_run'}
    with task_budget({'max_requests': 32, 'max_seconds': 600, 'max_tokens_by_model': {get_summary_model(): 2_000_000}}) as ledger:
        try:
            with bind_agent_history(scope):
                binding = await bind_summary_claims(raw, prose, context, call_stage)
                save(args.output / '逐条绑定与修复结果.json', binding)
                result.update(binding_status=binding['status'], bound_claims=len(binding['claims']),
                    unresolved_claims=len(binding['unresolved_claims']), semantic_calls=binding['semantic_calls'],
                    repair_calls=binding['repair_calls'], status='binding_completed')
                if args.run_nli:
                    nli = NLISettings(model_dir=str((BACKEND / NLISettings().model_dir).resolve()), timeout_seconds=120.0)
                    audit = await verify_bound_claims(binding, context, nli, call_stage=call_stage)
                    save(args.output / '逐条证据核验结果.json', audit)
                    result.update(status=audit['status'], status_label=audit['status_label'], nli_status='completed')
        finally:
            result['cost'] = ledger.report()
            save(args.output / '运行结果.json', result)
            await agent.client.close()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-nli', action='store_true')
    asyncio.run(execute(parser.parse_args()))
