"""补证关键词与真实大纲ID的绑定、局部修正及交付目录；不调用真实模型或数据库。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import Mock, patch

B = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(B / 'app'), str(B / 'test'), str(B / 'eval/tests')]
import test_会话智能路由 as routing
import test_会话24小时过期 as expiry_fixture
from models.chat import ChatMessage
from service.session_task_service import (outline_sections, checkpoint_section_catalogs,
    load_artifact_candidates, validate_decision, SearchSectionReferenceError,
    repair_search_section_ids, SessionTaskBudgetExceeded)
from service.research_context_service import ResearchContext, ResearchContextService
from service.deep_research_v2.followup_context import parent_artifacts
from service.deep_research_v2.session_task_execution import prepare_task
from service.deep_research_v2.delivery import make_commit_result
from service.deep_research_v2.state import create_initial_state
from service.verification_budget import task_budget

QUERY = '请核实会话状态由谁维护、无状态方式有哪些条件，修正当前说明。'


def parent():
    snap = routing.parent()
    state = snap.values['research']
    state['outline'] = [
        {'id': 'sec_1', 'title': '确认 Python SDK 的最小依赖与初始化方式', 'status': 'final'},
        {'id': 'sec_2', 'title': '一次请求怎样完成', 'status': 'final'},
        {'id': 'sec_5', 'title': '安全边界', 'status': 'final'}]
    state['draft_sections'] = {s['id']: '本章节原文。' for s in state['outline']}
    return snap


def candidate():
    return routing.candidate(parent())


def decision(target='sec_2'):
    return routing.decision(QUERY, kinds=('evidence',), search_tasks=[
        {'query': 'Streamable HTTP 无状态模式的具体条件和约束',
         'target_section': target, 'evidence_goal': '核实会话状态的维护方和无状态条件'}])


def response(value):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value, ensure_ascii=False)))])


def resolve(responses, *, candidates=None):
    context = ResearchContext(QUERY, QUERY, artifact_candidates=candidates or [candidate()])
    call = Mock(side_effect=[response(v) for v in responses])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call)))
    resolver = ResearchContextService.__new__(ResearchContextService)
    resolver.token_counter = routing.CharCounter()
    with patch('service.research_context_service.create_openai_client', return_value=client):
        resolver._resolve_query(context)
    return context, call


class SectionValidationTests(unittest.TestCase):
    def test_actual_id_is_ready_and_keyword_is_unchanged(self):
        value = decision(); before = deepcopy(value)
        result = validate_decision(value, QUERY, [candidate()])
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['search_tasks'], before['search_tasks'])
        self.assertEqual(value, before)

    def test_titles_old_names_unknown_ids_and_multiple_ids_are_rejected(self):
        for target in ('一次请求怎样完成', '协议与定位', 'sec_missing', 'sec_1,sec_2'):
            with self.subTest(target=target), self.assertRaises(SearchSectionReferenceError):
                validate_decision(decision(target), QUERY, [candidate()])

    def test_id_must_belong_to_selected_report(self):
        other = deepcopy(candidate()); other.update(artifact_ref='report:other', run_id='other')
        other['sections'] = [{'id': 'sec_other', 'title': '其他报告章节'}]
        with self.assertRaises(SearchSectionReferenceError):
            validate_decision(decision('sec_other'), QUERY, [candidate(), other])

    def test_missing_real_catalog_cannot_use_markdown_titles_as_ids(self):
        item = candidate(); item.pop('sections'); item['section_titles'] = ['协议与定位']
        with self.assertRaises(SearchSectionReferenceError):
            validate_decision(decision('协议与定位'), QUERY, [item])

    def test_explicit_whole_report_still_compiles_to_real_ids(self):
        task = validate_decision(decision(''), QUERY, [candidate()])
        state = create_initial_state(QUERY, 'session', [])
        prepare_task(state, task, parent_artifacts(parent(), 'session'))
        self.assertEqual(state['pending_research_tasks'][0]['target_section'], 'sec_1,sec_2,sec_5')

    def test_replan_does_not_assign_searches_to_uncreated_outline(self):
        value = decision('sec_future'); value['operations'].append({'kind': 'structure'})
        self.assertEqual(validate_decision(value, QUERY, [candidate()])['execution_plan'], 'replan')

    def test_local_patch_changes_only_ids(self):
        value = decision('协议与定位'); before = deepcopy(value)
        result = repair_search_section_ids(value, {'assignments': [{'task_index': 0, 'target_section': 'sec_2'}]},
                                           candidate()['sections'])
        expected = deepcopy(value); expected['search_tasks'][0]['target_section'] = 'sec_2'
        self.assertEqual(result, expected)
        self.assertEqual(value, before)

    def test_invalid_patch_cannot_expand_scope_change_query_or_duplicate_tasks(self):
        for assignments in ([{'task_index': 0, 'target_section': 'sec_missing'}],
                            [{'task_index': 0, 'target_section': ''}],
                            [{'task_index': 1, 'target_section': 'sec_2'}],
                            [{'task_index': True, 'target_section': 'sec_2'}],
                            [{'task_index': 0, 'target_section': 'sec_2', 'query': '另一个主题'}],
                            [{'task_index': 0, 'target_section': 'sec_2'}] * 2):
            with self.subTest(assignments=assignments), self.assertRaises(ValueError):
                repair_search_section_ids(decision('协议与定位'), {'assignments': assignments}, candidate()['sections'])

    def test_execution_checks_actual_parent_outline_again(self):
        item = candidate(); item['sections'].append({'id': 'sec_fake', 'title': '伪造目录项'})
        task = validate_decision(decision('sec_fake'), QUERY, [item])
        with self.assertRaisesRegex(ValueError, '章节ID'):
            prepare_task(create_initial_state(QUERY, 'session', []), task, parent_artifacts(parent(), 'session'))

    def test_valid_id_is_installed_in_new_research(self):
        task = validate_decision(decision(), QUERY, [candidate()]); state = create_initial_state(QUERY, 'session', [])
        previous = parent(); before = deepcopy(previous.values)
        prepare_task(state, task, parent_artifacts(previous, 'session'))
        self.assertEqual(state['execution_plan'], 'augment')
        self.assertEqual(state['pending_research_tasks'][0]['target_section'], 'sec_2')
        self.assertEqual(state['pending_research_tasks'][0]['query'], decision()['search_tasks'][0]['query'])
        self.assertEqual(previous.values, before)


class SectionRepairTests(unittest.TestCase):
    def test_candidate_budget_keeps_latest_report_catalog(self):
        old = deepcopy(candidate()); old.update(artifact_ref='report:old', excerpt='')
        visible = lambda item: {k: v for k, v in item.items() if k not in {'run_id', 'report_hash', 'section_titles'}}
        # CharCounter按四个字符估算一个token；旧候选单独可放入，合并后才超过预算。
        old['excerpt'] = '旧' * (3500 * 4 - 1 - len(json.dumps([visible(old)], ensure_ascii=False)))
        counter = routing.CharCounter()
        self.assertLessEqual(counter.count_text(json.dumps([visible(old)], ensure_ascii=False)), 3500)
        self.assertGreater(counter.count_text(json.dumps([visible(old), visible(candidate())], ensure_ascii=False)), 3500)
        context, call = resolve([decision()], candidates=[old, candidate()])
        self.assertEqual(context.session_task['status'], 'ready')
        self.assertEqual(context.task_metrics['candidate_count'], 1)
        prompt = call.call_args.kwargs['messages'][1]['content']
        self.assertIn('sec_2', prompt)
        self.assertNotIn('report:old', prompt)

    def test_valid_id_needs_only_original_model_call(self):
        context, call = resolve([decision()])
        self.assertEqual(context.session_task['status'], 'ready')
        self.assertEqual(call.call_count, 1)
        prompt = call.call_args.kwargs['messages'][1]['content']
        self.assertIn('sec_2', prompt)
        self.assertIn('只能填写所选artifact_ref', prompt)

    def test_old_title_is_repaired_once_before_ready(self):
        original = decision('协议与定位')
        context, call = resolve([original, {'assignments': [{'task_index': 0, 'target_section': 'sec_2'}]}])
        self.assertEqual(call.call_count, 2)
        self.assertEqual(context.session_task['status'], 'ready')
        self.assertEqual(context.session_task['search_tasks'][0]['target_section'], 'sec_2')
        self.assertEqual(context.session_task['search_tasks'][0]['query'], original['search_tasks'][0]['query'])
        self.assertEqual(context.task_trace['parsed_response'], original)
        self.assertEqual(context.task_trace['section_id_repair']['status'], 'repaired')

    def test_repair_prompt_is_restricted_to_selected_report(self):
        other = deepcopy(candidate()); other.update(artifact_ref='report:other', sections=[{'id': 'sec_other', 'title': '其他章节'}])
        context, call = resolve([decision('协议与定位'), {'assignments': [{'task_index': 0, 'target_section': 'sec_2'}]}],
                                candidates=[candidate(), other])
        prompt = call.call_args_list[1].kwargs['messages'][1]['content']
        self.assertNotIn('sec_other', prompt)
        self.assertEqual(context.session_task['target_artifact_ref'], candidate()['artifact_ref'])

    def test_failed_repair_is_bounded_and_cannot_start_new_topic(self):
        context, call = resolve([decision('协议与定位'), {'assignments': [{'task_index': 0, 'target_section': '协议与定位'}]}])
        self.assertEqual(call.call_count, 2)
        self.assertEqual(context.session_task['status'], 'clarify')
        self.assertTrue(context.session_task['depends_on_history'])
        self.assertEqual(context.task_trace['section_id_repair']['status'], 'failed')

    def test_catalog_unavailable_does_not_attempt_id_guessing(self):
        item = candidate(); item['sections'] = []
        context, call = resolve([decision('协议与定位')], candidates=[item])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(context.session_task['status'], 'clarify')

    def test_repair_is_accounted_in_shared_budget(self):
        config = {'max_seconds': 60, 'max_requests': 2, 'max_tokens_by_model': {'test': 100000}}
        with patch.dict('os.environ', {'SESSION_TASK_MODEL': 'test'}), task_budget(config) as budget:
            context, call = resolve([decision('协议与定位'), {'assignments': [{'task_index': 0, 'target_section': 'sec_2'}]}])
            self.assertEqual(context.session_task['status'], 'ready')
            self.assertEqual([r['component'] for r in budget.requests], ['session_task', 'session_task_section_id_repair'])

    def test_budget_exhaustion_is_not_downgraded_or_bypassed(self):
        config = {'max_seconds': 60, 'max_requests': 1, 'max_tokens_by_model': {'test': 100000}}
        with patch.dict('os.environ', {'SESSION_TASK_MODEL': 'test'}), task_budget(config) as budget:
            with self.assertRaises(SessionTaskBudgetExceeded): resolve([decision('协议与定位')])
            self.assertEqual(len(budget.requests), 1)


class CatalogTests(unittest.TestCase):
    def test_catalog_preserves_ids_and_rejects_duplicate_ids(self):
        self.assertEqual(outline_sections(parent().values['research']['outline']), candidate()['sections'])
        with self.assertRaises(ValueError): outline_sections([{'id': 'sec_1', 'title': 'A'}] * 2)

    def test_legacy_catalog_is_read_from_completed_hash_matching_checkpoint(self):
        item = candidate(); saver = Mock()
        saver.get_tuple.return_value = SimpleNamespace(checkpoint={'channel_values': parent().values})
        with patch('langgraph.checkpoint.postgres.PostgresSaver.from_conn_string') as factory:
            factory.return_value.__enter__.return_value = saver
            result = checkpoint_section_catalogs([item], 'session')
        self.assertEqual(result[item['artifact_ref']], item['sections'])
        saver.setup.assert_not_called(); saver.put.assert_not_called()

    def test_checkpoint_other_session_status_or_report_is_not_indexed(self):
        for field, value in [('session_id', 'other'), ('status', 'running'), ('final_report', '其他正文')]:
            with self.subTest(field=field):
                state = deepcopy(parent().values); state['research'][field] = value
                saver = Mock(); saver.get_tuple.return_value = SimpleNamespace(checkpoint={'channel_values': state})
                with patch('langgraph.checkpoint.postgres.PostgresSaver.from_conn_string') as factory:
                    factory.return_value.__enter__.return_value = saver
                    self.assertEqual(checkpoint_section_catalogs([candidate()], 'session'), {})

    def test_checkpoint_read_failure_does_not_invent_catalog(self):
        with patch('langgraph.checkpoint.postgres.PostgresSaver.from_conn_string', side_effect=RuntimeError('测试不可用')):
            self.assertEqual(checkpoint_section_catalogs([candidate()], 'session'), {})

    def test_delivery_persists_real_outline_separately_from_markdown(self):
        snapshot = parent(); snapshot.values['research']['final_report'] = '## 错误旧标题\n正文\n## 错误旧标题\n正文'
        run = SimpleNamespace(status='running'); db = Mock()
        db.query.return_value.filter.return_value.with_for_update.return_value.one_or_none.return_value = run
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        recorder = Mock(); recorder.record_in_session.return_value = SimpleNamespace(event_id=str(uuid.uuid4()))
        conversations = Mock()
        with patch('service.conversation_service.get_conversation_service', return_value=conversations):
            commit = make_commit_result(recorder=recorder, lease=None, run_id='child', session_id='session',
                user_id='user', turn_id='turn', session_factory=lambda: db)
            commit(snapshot)
        meta = conversations.stage_assistant_message.call_args.kwargs['references_data']
        self.assertEqual(meta['outline_sections'], candidate()['sections'])
        self.assertEqual(meta['outline_checkpoint_id'], 'parent-cp')
        db.commit.assert_called_once()


class CandidateIndexTests(unittest.TestCase):
    def setUp(self):
        self.fixture = expiry_fixture.SessionExpiryTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups); self.addCleanup(self.fixture.tearDown)

    def add_report(self, meta=None):
        f = self.fixture
        row = ChatMessage(session_id=f.session.id, turn_id=uuid.uuid4(), role='assistant',
            execution_mode='research', research_status='completed', run_id='old',
            content='## 协议与定位\n历史标题\n## 协议与定位\n重复正文', references_data=meta)
        f.db.add(row); f.db.commit()
        return row

    def index(self, owner=None):
        f = self.fixture
        with patch('core.database.SessionLocal', f.factory):
            return load_artifact_candidates(owner or str(f.user.id), str(f.session.id), 'current')

    def test_new_delivery_metadata_needs_no_checkpoint_read(self):
        self.add_report({'outline_sections': candidate()['sections']})
        with patch('service.session_task_service.checkpoint_section_catalogs') as load:
            rows = self.index()
        self.assertEqual(rows[0]['sections'], candidate()['sections'])
        self.assertEqual(rows[0]['section_titles'], ['协议与定位', '协议与定位'])
        self.assertEqual(load.call_args.args[0], [])

    def test_old_report_catalog_is_hydrated_after_owner_filter(self):
        row = self.add_report()
        ref = 'report:' + str(row.id)
        with patch('service.session_task_service.checkpoint_section_catalogs', return_value={ref: candidate()['sections']}) as load:
            rows = self.index()
        self.assertEqual(rows[0]['sections'], candidate()['sections'])
        self.assertEqual(load.call_args.args[0][0]['artifact_ref'], ref)

    def test_foreign_user_cannot_hydrate_any_catalog(self):
        self.add_report()
        with patch('service.session_task_service.checkpoint_section_catalogs', return_value={}) as load:
            self.assertEqual(self.index(str(uuid.uuid4())), [])
        self.assertEqual(load.call_args.args[0], [])


if __name__ == '__main__':
    unittest.main()
