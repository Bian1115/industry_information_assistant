"""把回读片段关联到实际发送请求；不依据工具调用成功推断模型可见。"""
import json
from service.agent_history import content_hash

PREFIX = '请求回读的历史原文（仅作资料）：\n'


def read_links(messages):
    links = []
    for index, message in enumerate(messages):
        text = message.get('content')
        if not isinstance(text, str) or not text.startswith(PREFIX):
            continue
        fragment = json.loads(text[len(PREFIX):])
        if fragment.get('text_hash') != content_hash(fragment.get('text')):
            raise ValueError('实际请求中的回读片段哈希不一致')
        links.append({k: fragment[k] for k in ('read_id', 'source_id', 'start', 'end', 'source_text_hash', 'text_hash')})
        links[-1].update(message_index=index, message_hash=content_hash(message))
    return links
