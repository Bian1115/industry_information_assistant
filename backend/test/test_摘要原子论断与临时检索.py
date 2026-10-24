import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from service.agent_history import content_hash
from service.summary_atomic_citations import verify_atomic_output
from service.summary_message_verification import verify_message_output
from service.summary_temporary_retrieval import RetrievalAPI, TemporaryEvidenceIndex, TemporaryRetrievalSettings


class Counter:
    def count_text(self, text): return len(text)
    def count_messages(self, messages): return sum(len(m['content']) for m in messages)


def context(texts, pieces=None):
    return {'sources': {sid: (pieces[sid] if pieces and sid in pieces else [{'start': 0, 'text': text}])
                        for sid, text in texts.items()},
            'rows': {sid: {'kind': 'llm_output', 'payload': {'content': text},
                           'content_hash': content_hash({'content': text})} for sid, text in texts.items()}}


def prose(sentence):
    return '## SESSION INTENT\nNone\n\n## SUMMARY\n' + sentence + '\n\n## ARTIFACTS\nNone'


def output(sentence, sources):
    return json.dumps({'claims': [{'section': 'SUMMARY', 'claim': sentence, 'source_ids': sources}]}, ensure_ascii=False)


def nli_report(status):
    return {'status': 'passed' if status == 'supported' else 'rejected', 'execution_status': 'completed',
            'claims': [{'status': status}], 'window_count': 1}


def stage_call(children, predicate=lambda claim, unit: True):
    async def call(messages, stage):
        if stage.startswith('claimify_'):
            sentence = messages[1]['content'].split('Sentence:\n\n')[-1].strip()
            if stage == 'claimify_selection':
                return json.dumps({'status': 'selected', 'sentence': sentence, 'reason': '含事实'})
            if stage == 'claimify_disambiguation':
                return json.dumps({'status': 'resolved', 'sentence': sentence, 'reason': '对象明确'})
            if stage == 'claimify_decomposition': return json.dumps({'claims': children}, ensure_ascii=False)
            raise AssertionError(stage)
        data = json.loads(messages[1]['content'])
        evidence = [{'source_id': m['source_id'], 'sentence_ids': [u['sentence_id'] for u in m['sentences']
                      if predicate(data['claim'], u)]} for m in data['messages']]
        return json.dumps({'claim_id': data['claim_id'], 'evidence': evidence})
    return call


class Backend:
    def __init__(self): self.closed = False; self.queries = []; self.documents = []
    async def embed(self, texts):
        self.documents.extend(texts)
        return [[1.0, .1] + [0.0] * 62 for _ in texts]
    async def rerank(self, query, documents, top_n):
        self.queries.append(query)
        return [{'index': i, 'relevance_score': .9 - i * .01} for i in range(top_n)]
    async def close(self): self.closed = True


def small_settings(**options):
    return TemporaryRetrievalSettings(embedding_dimensions=64, dense_top_k=4, lexical_top_k=4,
        coarse_top_k=4, rerank_top_k=2, **options)


class AtomicCitationTests(unittest.IsolatedAsyncioTestCase):
    async def test_compound_parent_becomes_separate_claims_with_different_exact_citations(self):
        parent = '东莞建立集中处理机制，并开展季度巡查。'
        children = ['东莞建立集中处理机制。', '东莞开展季度巡查。']
        ctx = context({'m1': children[0], 'm2': children[1]})
        call = stage_call(children, lambda claim, unit: ('集中' in claim) == ('集中' in unit['text']))
        verify = AsyncMock(return_value=nli_report('supported'))
        factory = AsyncMock()
        report = await verify_message_output(output(parent, ['m1', 'm2']), prose(parent), ctx, None, call,
            claimify_atomic=True, verify=verify, counter=Counter(), index_factory=factory)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual([c['claim'] for c in report['mapped_claims']], children)
        self.assertEqual([[r['source_id'] for r in c['references']] for c in report['mapped_claims']], [['m1'], ['m2']])
        self.assertEqual(verify.await_count, 2)
        self.assertEqual(len({c['parent_claim_id'] for c in report['mapped_claims']}), 1)
        self.assertTrue(all(c['origin']['summary_original'] == parent for c in report['mapped_claims']))
        for child in report['mapped_claims']:
            for ref in child['references']:
                full = ctx['rows'][ref['source_id']]['payload']['content']
                self.assertEqual(full[ref['start']:ref['end']], ref['quote'])
        factory.assert_not_called()

    async def test_missing_candidates_retrieve_only_input_snapshot_and_release_index(self):
        parent = '研究纳入200人。'
        ctx = context({'wrong': '准备回读论文。', 'right': parent + '隐藏尾部。'},
                      {'right': [{'start': 0, 'text': parent}]})
        backend = Backend(); index = TemporaryEvidenceIndex(small_settings(), backend=backend, counter=Counter())
        verify = AsyncMock(side_effect=[nli_report('missing_evidence'), nli_report('supported')])
        report = await verify_atomic_output(output(parent, ['wrong']), prose(parent), ctx, None,
            stage_call([parent], lambda claim, unit: '200' in unit['text']), verify=verify, counter=Counter(),
            retrieval_settings=small_settings(), index_factory=lambda *args, **kwargs: index)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['mapped_claims'][0]['references'][0]['source_id'], 'right')
        self.assertTrue(report['temporary_retrieval']['released'])
        self.assertTrue(backend.closed); self.assertTrue(index.closed)
        self.assertEqual(index.units, []); self.assertEqual(index.vectors, [])
        self.assertFalse(any('隐藏尾部' in text for text in backend.documents))

    async def test_retrieval_adds_evidence_without_discarding_first_selected_condition(self):
        parent = '研究纳入200人。'
        ctx = context({'m1': '研究限定成年人。', 'm2': parent})
        index = TemporaryEvidenceIndex(small_settings(), backend=Backend(), counter=Counter())
        verify = AsyncMock(side_effect=[nli_report('unknown'), nli_report('supported')])
        report = await verify_atomic_output(output(parent, ['m1']), prose(parent), ctx, None, stage_call([parent]),
            verify=verify, counter=Counter(), retrieval_settings=small_settings(), index_factory=lambda *a, **k: index)
        self.assertEqual({r['source_id'] for r in report['mapped_claims'][0]['references']}, {'m1', 'm2'})
        self.assertEqual(set(verify.await_args.args[1]['sources']), {'m1', 'm2'})

    async def test_contradiction_is_not_replaced_by_retrieval_support(self):
        parent = '研究纳入200人。'; factory = AsyncMock()
        report = await verify_atomic_output(output(parent, ['m']), prose(parent), context({'m': '研究纳入100人。'}),
            None, stage_call([parent]), verify=AsyncMock(return_value=nli_report('contradiction')),
            counter=Counter(), index_factory=factory)
        self.assertEqual(report['status'], 'rejected'); factory.assert_not_called()

    async def test_run_error_is_not_treated_as_insufficient_evidence(self):
        parent = '研究纳入200人。'; factory = AsyncMock()
        verify = AsyncMock(return_value={'status': 'error', 'execution_status': 'error', 'claims': [{'status': 'unknown'}]})
        report = await verify_atomic_output(output(parent, ['m']), prose(parent), context({'m': parent}),
            None, stage_call([parent]), verify=verify, counter=Counter(), index_factory=factory)
        self.assertEqual(report['status'], 'incomplete'); factory.assert_not_called()

    async def test_unknown_reference_id_does_not_reach_nli(self):
        parent = '研究纳入200人。'; verify = AsyncMock()
        report = await verify_atomic_output(output(parent, ['outside']), prose(parent), context({'m': parent}),
            None, AsyncMock(), verify=verify, counter=Counter())
        self.assertEqual(report['status'], 'incomplete'); verify.assert_not_awaited()

    async def test_bad_selected_sentence_id_does_not_reach_nli(self):
        parent = '研究纳入200人。'; verify = AsyncMock(); normal = stage_call([parent])
        async def call(messages, stage):
            if stage.startswith('atomic_claim_'):
                data = json.loads(messages[1]['content'])
                return json.dumps({'claim_id': data['claim_id'], 'evidence': [{'source_id': 'm', 'sentence_ids': ['fake']}]})
            return await normal(messages, stage)
        report = await verify_atomic_output(output(parent, ['m']), prose(parent), context({'m': parent}),
            None, call, verify=verify, counter=Counter())
        self.assertEqual(report['status'], 'incomplete'); verify.assert_not_awaited()

    async def test_no_fact_is_recorded_without_dropping_summary_or_running_nli(self):
        parent = '准备检索论文。'; verify = AsyncMock()
        async def call(messages, stage):
            return json.dumps({'status': 'no_verifiable_claims', 'sentence': '', 'reason': '计划不是已发生事实'})
        report = await verify_atomic_output(output(parent, ['m']), prose(parent), context({'m': parent}),
            None, call, verify=verify, counter=Counter())
        self.assertEqual(report['status'], 'no_verifiable_claims')
        self.assertEqual(report['claimify']['excluded'][0]['sentence'], parent); verify.assert_not_awaited()

    async def test_failed_build_still_clears_index(self):
        parent = '研究纳入200人。'; backend = Backend()
        backend.embed = AsyncMock(side_effect=RuntimeError('API unavailable'))
        index = TemporaryEvidenceIndex(small_settings(), backend=backend, counter=Counter())
        report = await verify_atomic_output(output(parent, []), prose(parent), context({'m': parent}), None,
            stage_call([parent]), verify=AsyncMock(return_value=nli_report('missing_evidence')), counter=Counter(),
            retrieval_settings=small_settings(), index_factory=lambda *a, **k: index)
        self.assertEqual(report['status'], 'incomplete'); self.assertTrue(backend.closed)
        self.assertEqual(index.units, []); self.assertTrue(report['temporary_retrieval']['released'])

    async def test_cancellation_clears_index_and_propagates(self):
        parent = '研究纳入200人。'; backend = Backend()
        backend.embed = AsyncMock(side_effect=asyncio.CancelledError())
        index = TemporaryEvidenceIndex(small_settings(), backend=backend, counter=Counter())
        with self.assertRaises(asyncio.CancelledError):
            await verify_atomic_output(output(parent, []), prose(parent), context({'m': parent}), None,
                stage_call([parent]), verify=AsyncMock(return_value=nli_report('missing_evidence')), counter=Counter(),
                retrieval_settings=small_settings(), index_factory=lambda *a, **k: index)
        self.assertTrue(backend.closed); self.assertEqual(index.units, [])


class TemporaryIndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_table_whole_long_table_repeats_header_and_keeps_all_rows(self):
        header = '| 编号 | 数值 |\n| --- | --- |'
        short = header + '\n| A | 90% |'
        table = header + '\n' + '\n'.join(f'| A{i} | {i}% |' for i in range(15))
        index = TemporaryEvidenceIndex(small_settings(chunk_tokens=64), backend=Backend(), counter=Counter())
        await index.build(context({'short': short, 'long': table}))
        self.assertEqual(len([u for u in index.units if u['source_id'] == 'short']), 1)
        groups = [u for u in index.units if u['source_id'] == 'long']
        self.assertGreater(len(groups), 1)
        self.assertTrue(all(u['text'].startswith(header + '\n') for u in groups))
        rows = [table[r['start']:r['end']] for u in groups for r in u['body_rows']]
        self.assertEqual(rows, table.splitlines()[2:]); await index.close()

    async def test_neighbors_do_not_expand_transitively(self):
        index = TemporaryEvidenceIndex(small_settings(), backend=Backend(), counter=Counter())
        await index.build(context({'m': '甲。乙。丙。丁。戊。'}))
        got = index.expand_neighbors([index.units[1]])
        self.assertEqual([u['text'] for u in got], ['甲。', '乙。', '丙。']); await index.close()

    async def test_neighbors_do_not_cross_sources(self):
        index = TemporaryEvidenceIndex(small_settings(), backend=Backend(), counter=Counter())
        await index.build(context({'a': '甲。', 'b': '乙。'}))
        self.assertEqual(index.expand_neighbors([index.units[0]]), [index.units[0]]); await index.close()

    async def test_rerank_uses_original_index_instead_of_zipping_sorted_scores(self):
        backend = Backend()
        backend.rerank = AsyncMock(return_value=[{'index': 2, 'relevance_score': .95}, {'index': 0, 'relevance_score': .8}])
        index = TemporaryEvidenceIndex(small_settings(neighbor_sentences=0), backend=backend, counter=Counter())
        await index.build(context({'m': '甲。乙。丙。'}))
        selected, audit = await index.retrieve('无共同关键词')
        # 选择输入恢复原文顺序，排名顺序保留在审计中。
        self.assertEqual({u['text'] for u in selected}, {'丙。', '甲。'})
        self.assertEqual([r['sentence_id'] for r in audit['reranked']], ['s0003', 's0001'])
        self.assertEqual(audit['fusion'], 'equal_weight_rrf'); await index.close()

    async def test_duplicate_or_out_of_range_rerank_index_is_rejected(self):
        for indexes in ((0, 0), (0, 999)):
            backend = Backend()
            backend.rerank = AsyncMock(return_value=[{'index': i, 'relevance_score': .9} for i in indexes])
            index = TemporaryEvidenceIndex(small_settings(), backend=backend, counter=Counter())
            await index.build(context({'m': '甲。乙。'}))
            with self.assertRaises(ValueError): await index.retrieve('甲')
            await index.close()

    async def test_invalid_vector_dimension_fails_before_search(self):
        backend = Backend(); backend.embed = AsyncMock(return_value=[[1.0]])
        index = TemporaryEvidenceIndex(small_settings(), backend=backend, counter=Counter())
        with self.assertRaises(ValueError): await index.build(context({'m': '甲。'}))
        await index.close()

    async def test_chunk_limit_and_long_sentence_are_explicit_errors(self):
        for settings, text in ((small_settings(max_chunks=1), '甲。乙。'), (small_settings(chunk_tokens=64), '甲' * 65)):
            index = TemporaryEvidenceIndex(settings, backend=Backend(), counter=Counter())
            with self.assertRaises(ValueError): await index.build(context({'m': text}))
            await index.close()


class RetrievalAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_rerank_usage_metadata_is_separate_from_original_index(self):
        api = RetrievalAPI(TemporaryRetrievalSettings())
        api.request = AsyncMock(return_value={'output': {'results': [{'index': 0, 'relevance_score': .9}]},
            'request_id': 'test-request', 'usage': {'total_tokens': 63, 'prompt_tokens': 30, 'other': '未记录'}})
        self.assertEqual(await api.rerank('论断', ['原句'], 1), [{'index': 0, 'relevance_score': .9}])
        self.assertEqual(api.last_rerank_metadata, {'request_id': 'test-request',
            'usage': {'total_tokens': 63, 'prompt_tokens': 30}})
        await api.close()

    async def test_native_and_compatible_rerank_protocols_preserve_index(self):
        for url, native, model in [
                ('https://example.invalid/api/v1/services/rerank/text-rerank/text-rerank', True, 'qwen3-rerank'),
                ('https://example.invalid/compatible-api/v1/reranks', False, 'qwen3-rerank'),
                ('https://example.invalid/api/v1/services/rerank/text-rerank/text-rerank', True, 'gte-rerank-v2')]:
            with patch.dict('os.environ', {'SUMMARY_RETRIEVAL_RERANK_URL': url}):
                api = RetrievalAPI(TemporaryRetrievalSettings(rerank_model=model))
            ranking = [{'index': 1, 'relevance_score': .9}, {'index': 0, 'relevance_score': .8}]
            api.request = AsyncMock(return_value={'output': {'results': ranking}} if native else {'results': ranking})
            got = await api.rerank('论断', ['原句一', '原句二'], 2, instruct='自定义研究材料排序')
            payload = api.request.await_args.args[1]
            self.assertEqual(got, ranking)
            self.assertEqual(payload['input']['documents'] if native else payload['documents'], ['原句一', '原句二'])
            self.assertEqual(payload['parameters']['instruct'] if native else payload['instruct'], '自定义研究材料排序')
            await api.close()

    async def test_embedding_response_order_is_restored_from_index(self):
        api = RetrievalAPI(TemporaryRetrievalSettings())
        api.request = AsyncMock(return_value={'data': [{'index': 1, 'embedding': [2]}, {'index': 0, 'embedding': [1]}]})
        self.assertEqual(await api.embed(['甲', '乙']), [[1], [2]])
        await api.close()

    async def test_duplicate_embedding_index_is_rejected(self):
        api = RetrievalAPI(TemporaryRetrievalSettings())
        api.request = AsyncMock(return_value={'data': [{'index': 0, 'embedding': [1]}, {'index': 0, 'embedding': [2]}]})
        with self.assertRaises(ValueError): await api.embed(['甲', '乙'])
        await api.close()


if __name__ == '__main__': unittest.main()
