"""离线复用五题失败输入；使用真实 tokenizer、模拟模型，不访问研究数据库。"""
import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from unittest.mock import Mock

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND/'app'), str(BACKEND/'test')]
os.environ['LANGFUSE_ENABLED'] = 'false'
from service.agent_context import prepare_agent_messages, protected_agent_messages, scoped_input_budget
from service.agent_history import HistoryScope, bind_agent_history
from service.qwen_token_counter import QwenTokenCounter
from service.deep_research_v2.agents.writer import LeadWriter
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.evidence_batches import analyze_evidence_batches
from service.deep_research_v2.quality_pipeline import build_task_context
from service.web_document import index_document, sentence_records
from test_智能体独立历史 import MemoryStore

def read(path):
    return json.loads(path.read_text())

async def execute(root):
    policy = read(root/'运行配置.json')['context_policy']
    counter = QwenTokenCounter(model_name=str(BACKEND/'.cache/qwen-tokenizer'))
    # 所有发送前计数共用本地 tokenizer，避免下载或远程依赖。
    import service.agent_context as context
    import service.deep_research_v2.evidence_batches as batches
    context.get_qwen_token_counter = lambda: counter
    batches.get_qwen_token_counter = lambda: counter
    rows = []
    for tid in [35,31,46,16,49]:
        folder = root/f'题目{tid}'
        history = read(folder/'智能体原始历史.json')
        checkpoints = read(folder/'完整检查点历史.json')
        state = copy.deepcopy(next(x['checkpoint']['channel_values']['research'] for x in checkpoints
                                   if x['checkpoint']['channel_values'].get('research',{}).get('outline')))
        scope = HistoryScope(MemoryStore(), str(uuid.uuid4()), str(uuid.uuid4()), None,
                             contract=copy.deepcopy(state['research_contract']), context_policy=policy)
        if tid in [35,31]:
            events = [json.loads(line) for line in (folder/'研究事件.jsonl').open()]
            drafts = {x['data']['content']['section_id']:x['data']['content']['content']
                      for x in events if x['type']=='section_content'}
            state['draft_sections'] = copy.deepcopy(drafts)
            agent = LeadWriter.__new__(LeadWriter)
            agent.name, agent.logger = 'LeadWriter', logging.getLogger('offline-writer')
            agent.add_message, agent.parse_json_response = Mock(), json.loads
            sent = []
            async def fake_writer(**kwargs):
                messages = await prepare_agent_messages(agent, scope, kwargs['system_prompt'],
                    kwargs['user_prompt'], str(uuid.uuid4()), 6000)
                sent.append(counter.count_messages(messages))
                return json.dumps({'executive_summary':'离线回归桩摘要', 'conclusions':[], 'outlook':'', 'claims':[]})
            agent.call_llm = fake_writer
            with bind_agent_history(scope):
                await agent._synthesize_report(state)
            assert all(content in state['final_report'] for content in drafts.values())
            assert state['draft_sections'] == drafts
            assert sent and max(sent) <= scoped_input_budget(scope, agent.name, 6000)
            rows.append({'id':tid, 'path':'writer_synthesis', 'passed':True, 'input_tokens':sent,
                         'budget':25744, 'draft_sections_preserved':len(drafts), 'model_output':'stub'})
        elif tid in [46,49]:
            failed = next(x for x in reversed(history) if x['kind']=='llm_error')
            original = next(x['payload'] for x in history if x['kind']=='llm_input' and x['call_id']==failed['call_id'])
            prefix, rest = original['content'].split('## 搜索结果\n',1)
            material, suffix = rest.split('\n## 任务',1)
            urls = set(re.findall(r'^URL: (.*)$', material, re.M))
            tool_rows = next(x['payload']['value'] for x in reversed(history)
                            if x['kind']=='tool_output' and x['payload'].get('tool_name')=='v2_web_fulltext_read')
            search_rows = next(x['payload']['value'] for x in reversed(history)
                              if x['kind']=='tool_output' and x['payload'].get('tool_name')=='v2_web_search')
            metadata = {x['url']:x for x in search_rows}
            docs = []
            for value in tool_rows:
                if value['url'] not in urls:
                    continue
                body = value['content']
                docs.append({**metadata.get(value['url'],{}), **copy.deepcopy(value), 'full_content':body,
                             'reader_verified':True, **index_document(body,value['url'])})
            before = copy.deepcopy(docs)
            expected = {(doc['url'],doc['document_id'],row['id']) for doc in docs for row in sentence_records(doc)}
            agent = DeepScout.__new__(DeepScout)
            agent.name, agent.logger, agent.parse_json_response = 'DeepScout', logging.getLogger('offline-scout'), json.loads
            sent, seen = [], set()
            async def fake_scout(**kwargs):
                messages = await prepare_agent_messages(agent,scope,kwargs['system_prompt'],kwargs['user_prompt'],
                                                        str(uuid.uuid4()),6000)
                sent.append(counter.count_messages(messages))
                text = kwargs['user_prompt'].split('## 搜索结果\n',1)[1].split('\n## 任务',1)[0]
                for chunk in re.split(r'(?m)^\[\d+\] ',text)[1:]:
                    url = re.search(r'^URL: (.*)$',chunk,re.M).group(1)
                    doc = re.search(r'正文编号: ([^；\n]+)',chunk).group(1)
                    seen.update((url,doc,ident) for ident in re.findall(r'^\[(s\d+)\]',chunk,re.M))
                return '{"extracted_facts":[]}'
            agent.call_llm = fake_scout
            with bind_agent_history(scope):
                await analyze_evidence_batches(agent,docs,lambda text:prefix+'## 搜索结果\n'+text+'\n## 任务'+suffix,
                                               system_prompt=original['system'])
            assert expected and seen == expected, (len(seen),len(expected))
            assert len(sent)>1 and max(sent)<=scoped_input_budget(scope,agent.name,6000)
            assert all(d['content']==old['content'] and d['document_id']==old['document_id']
                       and d['sentence_index']==old['sentence_index'] for d,old in zip(docs,before))
            rows.append({'id':tid,'path':'scout_evidence_batches','passed':True,'batch_count':len(sent),
                         'input_tokens':sent,'budget':25744,'documents':len(docs),
                         'unique_sentences':len(expected),'all_sentences_preserved':True,'model_output':'stub'})
        else:
            query = state['query']
            task = build_task_context({'research_contract':state['research_contract']},query)
            assert task['query']==query and 'required_metrics' not in task
            agent = LeadWriter.__new__(LeadWriter)
            agent.name, agent.add_message = 'LeadWriter', Mock()
            reached = []
            async def stub_section(state, section): reached.append(section['id'])
            async def stub_synthesis(state): reached.append('synthesis')
            agent._write_section, agent._synthesize_report = stub_section, stub_synthesis
            await agent._write_report(state)
            assert reached[-1]=='synthesis' and 'prewrite' not in state.get('quality_gates',{})
            rows.append({'id':tid,'path':'legacy_metric_gate','passed':True,
                         'old_contract_does_not_block':True,'new_planning_ignores_old_metrics':True,
                         'reached_steps':reached,'model_output':'stub'})
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest = read(args.source/'文件校验清单.json')
    def hashes(): return {name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in manifest}
    before = hashes()
    assert before == manifest
    result = {'source':str(args.source),'mode':'offline_tokenizer_and_stub_models','real_model_calls':0}
    try:
        result['tasks'] = asyncio.run(execute(args.source))
        result['passed'] = all(x['passed'] for x in result['tasks'])
    except BaseException as exc:
        result.update(passed=False,error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        result['source_files_checked'] = len(manifest)
        result['source_files_unchanged'] = hashes()==before
        (args.output/'真实失败输入回归结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
