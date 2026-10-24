"""按完整请求预算组批；摘要最多额外分批合并一次，不递归扩大层数。"""
from service.agent_history import content_hash
from service.verification_checks import source_type

BATCH_POLICY = 'token-packed-single-regroup-repair-reserve-v2'


def pack_batches(items, counter, budget, build_messages):
    """保持来源顺序；短记录共用一批，单项放不下时显式报错。"""
    batches, batch = [], []
    for item in items:
        if counter.count_messages(build_messages(batch + [item])) <= budget:
            batch.append(item)
            continue
        if batch:
            batches.append(batch)
        if counter.count_messages(build_messages([item])) > budget:
            raise ValueError('单个片段或中间摘要超出完整请求预算')
        batch = [item]
    if batch:
        batches.append(batch)
    return batches


def fragments_for_records(records, counter, budget, build_messages):
    from service.agent_context import source_text
    for row in records:
        if row['content_hash'] != content_hash(row['payload']):
            raise ValueError('新增摘要来源快照哈希不一致')
        text = source_text(row)
        text_hash = content_hash(text)
        start = 0
        while start < len(text):
            base = {'source_id': row['source_id'], 'kind': row['kind'], 'source_type': source_type(row),
                    'call_id': row['call_id'], 'source_text_hash': text_hash, 'start': start}
            whole = {**base, 'end': len(text), 'text': text[start:]}
            # 能完整放入空批次的记录不拆开，也不按当前批次的剩余空间切碎。
            if counter.count_messages(build_messages(whole)) <= budget:
                yield whole
                break
            low, high = 0, len(text) - start
            while low < high:
                mid = (low + high + 1) // 2
                candidate = {**base, 'end': start + mid, 'text': text[start:start + mid]}
                if counter.count_messages(build_messages(candidate)) <= budget:
                    low = mid
                else:
                    high = mid - 1
            if low == 0:
                raise ValueError('固定提示与必要约束超出摘要器预算')
            yield {**base, 'end': start + low, 'text': text[start:start + low]}
            start += low  # 输入去重区间完整覆盖原文；单个片段不是可独立提交的覆盖游标。


async def hierarchical_summary(fragments, previous, contract, call_stage, counter, budget, policy, sources, *,
                               build_messages=None, generate=None):
    from service.summary_citations import draft_messages, generate_draft
    draft_messages = build_messages or draft_messages
    generate_draft = generate or generate_draft
    build = lambda pieces, old='': draft_messages({'new_records': pieces, 'previous_summary': old, 'latest_contract': contract})
    # 旧快照的块数/层数保留用于反序列化；新算法以请求预算和共享调用/时间上限为准。
    if counter.count_messages(build(fragments, previous)) <= budget:
        return await generate_draft(build(fragments, previous), 'summary_merge', call_stage, sources)
    summaries = [previous] if previous else []
    for batch in pack_batches(fragments, counter, budget, build):
        result = await generate_draft(build(batch), 'summary_chunk', call_stage, sources)
        summaries.append(result)
    if len(summaries) == 1:
        return summaries[0]

    def merge_input(items, level=1):
        return draft_messages({'summary_parts': items, 'latest_contract': contract,
                                 'merge_level': level, 'note': '合并摘要正文；不得添加引句'})

    final_messages = merge_input(summaries)
    if counter.count_messages(final_messages) > budget:
        # 只允许这一次分批合并；不得因结果仍长而继续增加合并层级。
        merged = []
        for batch in pack_batches(summaries, counter, budget, merge_input):
            result = await generate_draft(merge_input(batch), 'summary_hierarchical_merge', call_stage, sources)
            merged.append(result)
        final_messages = merge_input(merged, level=2)
        if counter.count_messages(final_messages) > budget:
            raise ValueError('分批合并一次后仍超出摘要器输入预算，保留原摘要')
    return await generate_draft(final_messages, 'summary_hierarchical_merge', call_stage, sources)
