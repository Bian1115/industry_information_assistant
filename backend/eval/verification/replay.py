"""仅供离线同源回放的文件存储，生产仍使用PostgreSQL；每组目录独立。"""
import copy
import json
import uuid
from pathlib import Path
from service.agent_history import content_hash, canonical_json


class ReplayStore:
    def __init__(self, directory, records):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.data = {'summary':'','summary_version':0,'covered_seq':0,'summary_metadata':{},
                     'offloaded_source_ids':[],'messages':copy.deepcopy(records)}
        ids = [r['source_id'] for r in records]
        if len(ids) != len(set(ids)) or [r['seq'] for r in records] != sorted({r['seq'] for r in records}):
            raise ValueError('原始快照来源ID或顺序无效')
        for row in records:
            if row['content_hash'] != content_hash(row['payload']):
                raise ValueError('原始回放快照哈希错误')
        self.flush()

    def flush(self):
        temp = self.directory/'工作历史.tmp'
        temp.write_text(canonical_json(self.data))
        temp.replace(self.directory/'工作历史.json')

    async def load(self, scope, agent, *, replay_only=False):
        data = copy.deepcopy(self.data)
        if replay_only:
            from service.agent_context import REPLAY_KINDS
            data['messages'] = [r for r in data['messages'] if r['kind'] in REPLAY_KINDS]
        return data

    async def append(self, scope, agent, kind, payload, call_id, *, event_id=None):
        seq = max((r['seq'] for r in self.data['messages']), default=0)+1
        row = {'seq':seq,'source_id':'replay_'+uuid.uuid4().hex,'kind':kind,'payload':payload,
               'call_id':call_id,'content_hash':content_hash(payload),'agent_name':agent}
        self.data['messages'].append(row)
        self.flush()
        return copy.deepcopy(row)

    async def read_source(self, scope, agent, source_id):
        for row in self.data['messages']:
            if row['source_id'] == source_id:
                return copy.deepcopy(row)
        raise PermissionError('来源不属于本回放命名空间')

    async def save_summary(self, scope, agent, *, expected_version, covered_seq, summary, metadata):
        if expected_version != self.data['summary_version']:
            return False
        self.data.update(summary_version=expected_version+1,covered_seq=covered_seq,summary=summary,summary_metadata=metadata)
        self.flush()
        return True

    async def mark_offloaded(self, scope, agent, source_ids):
        self.data['offloaded_source_ids'] = sorted(set(self.data['offloaded_source_ids']) | set(source_ids))
        self.flush()
