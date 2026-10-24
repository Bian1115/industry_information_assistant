"""真实 JWT + 双用户 SQLite ORM + HTTP 路由的隔离回归；外部 AI/索引服务使用替身。"""

import asyncio
import hashlib
import importlib
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import JSON, MetaData, create_engine
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app_main
from core import security
from core.database import Base, get_db
from core.time_utils import utc_now
from models.chat import ChatAttachment, ChatMessage, ChatSession, LongTermMemory
from models.knowledge import Document, KnowledgeBase
from models.research import AgentEvent, ResearchRun
from models.user import User
from service.conversation_service import ConversationService
from service.retrieval_service import kb_collection_name

attachments = importlib.import_module('router.attachment_router')
documents = importlib.import_module('router.document_router')
knowledge = importlib.import_module('router.knowledge_router')
chat = importlib.import_module('router.chat_router')
research = importlib.import_module('router.research_router')
memories = importlib.import_module('router.memory_router')


class UserIsolationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        metadata = MetaData()
        for table in Base.metadata.sorted_tables:
            copied = table.to_metadata(metadata)
            for column in copied.columns:
                if isinstance(column.type, (JSONB, ARRAY)):
                    column.type = JSON()
            for index in copied.indexes:
                where = index.dialect_options['postgresql'].get('where')
                if where is not None:
                    index.dialect_options['sqlite']['where'] = where
        metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db = self.factory()
        self.temporary = tempfile.TemporaryDirectory(prefix='用户隔离测试-')
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.db.close)
        self.a = User(id=uuid4(), username='隔离甲', email='a@example.invalid', hashed_password='test')
        self.b = User(id=uuid4(), username='隔离乙', email='b@example.invalid', hashed_password='test')
        self.db.add_all([self.a, self.b])
        self.db.flush()
        self.sa = ChatSession(id=uuid4(), user_id=self.a.id, title='甲私有会话')
        self.sb = ChatSession(id=uuid4(), user_id=self.b.id, title='乙私有会话')
        self.ka = KnowledgeBase(id=uuid4(), user_id=self.a.id, name='甲私有库', document_count=1)
        self.kb = KnowledgeBase(id=uuid4(), user_id=self.b.id, name='乙私有库', document_count=1)
        self.db.add_all([self.sa, self.sb, self.ka, self.kb])
        self.db.flush()
        self.aa = self.new_attachment(self.a, self.sa, '甲私有文本')
        self.ab = self.new_attachment(self.b, self.sb, '乙私有文本')
        self.da = self.new_document(self.a, self.ka, '甲私有文档.pdf')
        self.dob = self.new_document(self.b, self.kb, '乙私有文档.pdf')
        self.db.add_all([
            ChatMessage(session_id=self.sa.id, role='user', content='甲私有消息'),
            ChatMessage(session_id=self.sb.id, role='user', content='乙私有消息'),
            LongTermMemory(user_id=self.a.id, summary='甲私有记忆', index_status='ready'),
            LongTermMemory(user_id=self.b.id, summary='乙私有记忆', index_status='ready'),
        ])
        self.run_a = str(uuid4())
        self.run_b = str(uuid4())
        self.db.add_all([
            ResearchRun(run_id=self.run_a, session_id=str(self.sa.id), user_id=self.a.id, status='failed'),
            ResearchRun(run_id=self.run_b, session_id=str(self.sb.id), user_id=self.b.id, status='failed'),
            AgentEvent(run_id=self.run_a, session_id=str(self.sa.id), user_id=self.a.id, seq=1,
                       event_type='test_event', payload={'private': '甲私有事件'}),
            AgentEvent(run_id=self.run_b, session_id=str(self.sb.id), user_id=self.b.id, seq=1,
                       event_type='test_event', payload={'private': '乙私有事件'}),
        ])
        self.db.commit()
        self.start_patch(patch.object(security, 'SECRET_KEY', 'isolation-test-only-' + 'a' * 48))
        self.token_a = security.create_access_token({'sub': str(self.a.id)})
        self.token_b = security.create_access_token({'sub': str(self.b.id)})
        self.client = TestClient(app_main.app)
        self.addCleanup(self.client.close)
        app_main.app.dependency_overrides[get_db] = lambda: self.db
        self.addCleanup(app_main.app.dependency_overrides.clear)
        self.start_patch(patch('core.database.SessionLocal', self.factory))
        self.start_patch(patch.object(attachments, 'UPLOAD_DIR', self.temporary.name))
        self.start_patch(patch.object(knowledge, 'UPLOAD_DIR', self.temporary.name))
        self.docmind = self.start_patch(patch('service.docmind_service.process_document_with_docmind',
                                            return_value={'success': True, 'document_count': 1, 'message': 'test'}))
        self.milvus = self.start_patch(patch.object(knowledge, 'get_milvus_service'))
        self.es_delete = self.start_patch(patch.object(knowledge.es_service, 'delete_by_doc_id'))
        self.start_patch(patch.object(knowledge.es_service, 'delete_index'))
        self.start_patch(patch('service.memory_service.enqueue_memory_if_due'))
        self.start_patch(patch('service.session_memory_service.get_session_memory_service', return_value=Mock()))
        conversation = ConversationService.__new__(ConversationService)
        conversation.get_prompt_history = Mock(return_value=[])
        self.start_patch(patch.object(chat, 'get_conversation_service', return_value=conversation))
        self.start_patch(patch.object(research, 'get_conversation_service', return_value=conversation))
        app_main.app.dependency_overrides[chat.get_services] = lambda: {'chat_service': SimpleNamespace(
            rerank_documents=lambda query, docs: docs, retrieve_from_web=lambda query: [])}
        self.model_response = self.start_patch(patch.object(chat, '_completion_response',
            side_effect=lambda **values: JSONResponse({'question': values['question_for_model'], 'documents': values['documents']})))
        self.hybrid = self.start_patch(patch.object(documents, 'hybrid_retrieve', return_value=[]))
        self.document_rerank = self.start_patch(patch.object(documents, 'rerank_knowledge_candidates',
            new=AsyncMock(side_effect=lambda query, rows: rows)))
        self.chat_hybrid = self.start_patch(patch.object(chat, 'hybrid_retrieve', return_value=[]))

    def start_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def new_attachment(self, user, session, text):
        path = Path(self.temporary.name) / (uuid4().hex + '.txt')
        path.write_text(text)
        row = ChatAttachment(id=uuid4(), user_id=user.id if user else None, session_id=session.id,
            filename='私有.txt', file_type='txt', file_size=len(text.encode()), file_path=str(path),
            status='completed', content_text=text)
        self.db.add(row)
        return row

    def new_document(self, user, kb, filename):
        path = Path(self.temporary.name) / (uuid4().hex + '.pdf')
        path.write_text(filename)
        row = Document(id=uuid4(), user_id=user.id, knowledge_base_id=kb.id, filename=filename,
            file_type='pdf', file_size=10, file_path=str(path), status='completed', chunk_count=1)
        self.db.add(row)
        return row

    def request(self, method, path, *, user='a', **kwargs):
        headers = kwargs.pop('headers', {})
        if user:
            headers['Authorization'] = 'Bearer ' + (self.token_a if user == 'a' else self.token_b)
        return self.client.request(method, path, headers=headers, **kwargs)

    def chat_body(self, **changes):
        return {'session_id': str(self.sa.id), 'turn_id': str(uuid4()), 'question': '测试',
                'search_web': False, 'search_knowledge': False, **changes}

    def test_every_private_route_requires_real_authentication(self):
        public = {'/auth/register', '/auth/login', '/auth/token', '/hello'}
        checked = 0
        # 使用 OpenAPI 展开的实际路由，兼容 FastAPI 延迟 include_router 的版本。
        for path, operations in app_main.app.openapi()['paths'].items():
            if path in public:
                continue
            import re
            url = re.sub(r'\{[^}]+\}', str(uuid4()), path)
            for method in operations:
                if method not in {'get', 'post', 'put', 'delete', 'patch'}:
                    continue
                checked += 1
                with self.subTest(method=method, path=path):
                    response = self.client.request(method, url)
                    self.assertEqual(response.status_code, 401, response.text)
        self.assertGreater(checked, 30)

    def test_foreign_attachment_upload_has_no_file_or_db_side_effects(self):
        before = set(Path(self.temporary.name).iterdir())
        count = self.db.query(ChatAttachment).count()
        response = self.request('POST', '/attachments', data={'session_id': str(self.sb.id)}, files={'file': ('越权.txt', b'forbidden')})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(set(Path(self.temporary.name).iterdir()), before)
        self.assertEqual(self.db.query(ChatAttachment).count(), count)

    def test_foreign_attachment_read_list_and_delete_are_rejected(self):
        for method, path in [('GET', f'/attachments/{self.ab.id}'), ('GET', f'/attachments/session/{self.sb.id}'), ('DELETE', f'/attachments/{self.ab.id}')]:
            with self.subTest(method=method):
                self.assertEqual(self.request(method, path).status_code, 404)
        self.assertTrue(Path(self.ab.file_path).exists())
        self.assertIsNotNone(self.db.get(ChatAttachment, self.ab.id))

    def test_owner_attachment_upload_read_list_delete_still_work(self):
        response = self.request('POST', '/attachments', data={'session_id': str(self.sa.id), 'user_id': str(self.b.id)}, files={'file': ('报告.txt', '本人上传内容'.encode())})
        self.assertEqual(response.status_code, 201, response.text)
        aid = response.json()['id']
        row = self.db.query(ChatAttachment).filter(ChatAttachment.id == __import__('uuid').UUID(aid)).one()
        self.assertEqual(row.user_id, self.a.id)
        self.db.refresh(row)
        self.assertEqual(row.content_text, '本人上传内容')
        self.assertEqual(self.request('GET', f'/attachments/{aid}').status_code, 200)
        ids = [item['id'] for item in self.request('GET', f'/attachments/session/{self.sa.id}').json()['attachments']]
        self.assertIn(aid, ids)
        self.assertNotIn(str(self.ab.id), ids)
        self.assertEqual(self.request('DELETE', f'/attachments/{aid}').status_code, 204)
        self.assertFalse(Path(row.file_path).exists())

    def test_anonymous_and_mismatched_historical_attachments_are_quarantined(self):
        anonymous = self.new_attachment(None, self.sa, '匿名历史文本')
        mismatch = self.new_attachment(self.b, self.sa, '错误归属文本')
        self.db.commit()
        for row in (anonymous, mismatch):
            self.assertEqual(self.request('GET', f'/attachments/{row.id}').status_code, 404)
            self.assertEqual(self.request('DELETE', f'/attachments/{row.id}').status_code, 404)
        listed = self.request('GET', f'/attachments/session/{self.sa.id}').json()['attachments']
        self.assertEqual([x['id'] for x in listed], [str(self.aa.id)])

    def test_expired_session_cannot_upload_read_list_or_delete_attachment(self):
        self.sa.expires_at = utc_now() - timedelta(seconds=1)
        self.db.commit()
        self.assertEqual(self.request('POST', '/attachments', data={'session_id': str(self.sa.id)}, files={'file': ('过期.txt', b'test')}).status_code, 404)
        for method, path in [('GET', f'/attachments/{self.aa.id}'), ('GET', f'/attachments/session/{self.sa.id}'), ('DELETE', f'/attachments/{self.aa.id}')]:
            self.assertEqual(self.request(method, path).status_code, 404)
        self.assertTrue(Path(self.aa.file_path).exists())

    def test_invalid_attachment_ids_return_400(self):
        for path in ['/attachments/bad', '/attachments/session/bad']:
            self.assertEqual(self.request('GET', path).status_code, 400)

    def test_attachment_filename_never_controls_storage_path(self):
        response = self.request('POST', '/attachments', data={'session_id': str(self.sa.id)}, files={'file': ('../../乙/内容.txt', b'safe')})
        self.assertEqual(response.status_code, 201)
        row = self.db.get(ChatAttachment, __import__('uuid').UUID(response.json()['id']))
        self.assertEqual(row.filename, '内容.txt')
        self.assertEqual(Path(row.file_path).parent, Path(self.temporary.name))
        self.assertIn(self.a.id.hex, Path(row.file_path).name)
        self.assertEqual(Path(row.file_path).stat().st_mode & 0o777, 0o600)

    def test_attachment_worker_rechecks_owner_before_reading_file(self):
        with patch('builtins.open') as reader:
            asyncio.run(attachments.process_attachment(str(self.ab.id), self.ab.file_path, self.factory, user_id=str(self.a.id)))
            reader.assert_not_called()

    def test_foreign_and_cross_session_chat_attachments_never_reach_model(self):
        other = ChatSession(id=uuid4(), user_id=self.a.id)
        self.db.add(other)
        self.db.flush()
        own_other = self.new_attachment(self.a, other, '同用户另一会话文本')
        self.db.commit()
        count = self.db.query(ChatMessage).count()
        for row in (self.ab, own_other):
            response = self.request('POST', '/chat/completion/v3', json=self.chat_body(attachment_ids=[str(row.id)]))
            self.assertEqual(response.status_code, 404, response.text)
        self.model_response.assert_not_called()
        self.assertEqual(self.db.query(ChatMessage).count(), count)

    def test_owner_chat_attachment_reaches_model_and_plain_chat_still_works(self):
        response = self.request('POST', '/chat/completion/v3', json=self.chat_body(attachment_ids=[str(self.aa.id)]))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('甲私有文本', response.json()['question'])
        self.assertNotIn('乙私有文本', response.json()['question'])
        self.assertEqual(self.request('POST', '/chat/completion', json=self.chat_body()).status_code, 200)

    def test_foreign_session_chat_is_rejected_before_model(self):
        response = self.request('POST', '/chat/completion', json=self.chat_body(session_id=str(self.sb.id)))
        self.assertEqual(response.status_code, 404)
        self.model_response.assert_not_called()

    def test_document_apis_require_kb_id_and_reject_arbitrary_index(self):
        cases = [('GET', '/documents/list', {}), ('POST', '/documents/delete', {'json': {'document_ids': [str(self.da.id)]}}),
                 ('POST', '/documents/retrieve', {'json': {'question': 'test'}}),
                 ('POST', '/documents/upload', {'files': {'file': ('私有.pdf', b'fake')}})]
        for method, path, kwargs in cases:
            self.assertEqual(self.request(method, path, **kwargs).status_code, 422)
        response = self.request('POST', f'/documents/upload?kb_id={self.ka.id}&index_name=policy_documents', files={'file': ('私有.pdf', b'fake')})
        self.assertEqual(response.status_code, 400)
        self.docmind.assert_not_called()

    def test_foreign_kb_is_rejected_by_all_document_apis(self):
        before = set(Path(self.temporary.name).iterdir())
        for method, path, kwargs in [
            ('GET', '/documents/list', {}),
            ('POST', '/documents/upload', {'files': {'file': ('越权.pdf', b'fake')}}),
            ('POST', '/documents/delete', {'json': {'document_ids': [str(self.dob.id)]}}),
            ('POST', '/documents/retrieve', {'json': {'question': 'test'}}),
        ]:
            self.assertEqual(self.request(method, f'{path}?kb_id={self.kb.id}', **kwargs).status_code, 404)
        self.docmind.assert_not_called()
        self.hybrid.assert_not_called()
        self.assertEqual(set(Path(self.temporary.name).iterdir()), before)
        self.assertIsNotNone(self.db.get(Document, self.dob.id))

    def test_document_list_only_contains_current_users_documents(self):
        response = self.request('GET', f'/documents/list?kb_id={self.ka.id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['data']['total'], 1)
        self.assertEqual(response.json()['data']['docs'][0]['id'], str(self.da.id))
        filtered = self.request('GET', f'/documents/list?kb_id={self.ka.id}&document_id={self.dob.id}')
        self.assertEqual(filtered.status_code, 404)
        self.assertEqual(self.request('GET', f'/documents/list?kb_id={self.ka.id}&page=0').status_code, 422)

    def test_mixed_document_delete_is_authorized_before_any_side_effect(self):
        response = self.request('POST', f'/documents/delete?kb_id={self.ka.id}', json={'document_ids': [str(self.da.id), str(self.dob.id)]})
        self.assertEqual(response.status_code, 404)
        self.assertTrue(Path(self.da.file_path).exists())
        self.assertIsNotNone(self.db.get(Document, self.da.id))
        self.milvus.assert_not_called()
        self.es_delete.assert_not_called()

    def test_owner_document_delete_cleans_only_owner_collection(self):
        response = self.request('POST', f'/documents/delete?kb_id={self.ka.id}', json={'document_ids': [str(self.da.id)]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(self.db.get(Document, self.da.id))
        self.assertIsNotNone(self.db.get(Document, self.dob.id))
        self.assertTrue(Path(self.dob.file_path).exists())
        self.milvus.return_value.delete_by_doc_id.assert_called_once_with(kb_collection_name(str(self.ka.id), allow_benchmark=False), hashlib.md5(self.da.filename.encode()).hexdigest())

    def test_owner_document_upload_derives_collection_and_paths_on_server(self):
        response = self.request('POST', f'/documents/upload?kb_id={self.ka.id}', files={'file': ('../../乙/同名.pdf', b'fake')})
        self.assertEqual(response.status_code, 200, response.text)
        row = self.db.get(Document, __import__('uuid').UUID(response.json()['document_id']))
        self.assertEqual(row.user_id, self.a.id)
        self.assertEqual(row.knowledge_base_id, self.ka.id)
        self.assertEqual(row.filename, '同名.pdf')
        self.assertEqual(self.docmind.call_args.kwargs['index_name'], kb_collection_name(str(self.ka.id), allow_benchmark=False))
        self.assertEqual(Path(row.file_path).parent, Path(self.temporary.name))
        self.assertIn(self.a.id.hex, Path(row.file_path).name)

    def test_document_retrieval_is_scoped_and_foreign_ids_fail_before_search(self):
        own_hit = {'document_id': hashlib.md5(self.da.filename.encode()).hexdigest(), 'content_with_weight': '甲文档正文'}
        foreign_hit = {'document_id': hashlib.md5(self.dob.filename.encode()).hexdigest(), 'content_with_weight': '乙文档正文'}
        self.hybrid.return_value = [own_hit, foreign_hit]
        response = self.request('POST', f'/documents/retrieve?kb_id={self.ka.id}', json={'question': 'test', 'document_ids': [str(self.da.id)]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['data']['chunks'], [own_hit])
        self.hybrid.assert_called_once_with(str(self.ka.id), 'test', allow_benchmark=False)
        self.document_rerank.assert_awaited_once_with('test', [own_hit])
        self.hybrid.reset_mock()
        denied = self.request('POST', f'/documents/retrieve?kb_id={self.ka.id}', json={'question': 'test', 'document_ids': [str(self.dob.id)]})
        self.assertEqual(denied.status_code, 404)
        self.hybrid.assert_not_called()
        empty = self.request('POST', f'/documents/retrieve?kb_id={self.ka.id}', json={'question': 'test', 'document_ids': []})
        self.assertEqual(empty.json()['data']['chunks'], [])
        self.hybrid.assert_not_called()

    def test_knowledge_worker_wrong_owner_or_kb_cannot_read_write_or_delete_file(self):
        for owner, kb in [(self.b.id, self.ka.id), (self.a.id, self.kb.id)]:
            asyncio.run(knowledge.process_document(str(self.da.id), self.da.file_path, str(kb), self.factory, user_id=str(owner)))
        self.docmind.assert_not_called()
        self.assertTrue(Path(self.da.file_path).exists())

    def test_native_knowledge_routes_deny_foreign_documents(self):
        for method, path in [('GET', f'/knowledge-bases/{self.kb.id}'), ('GET', f'/knowledge-bases/{self.kb.id}/documents'),
            ('GET', f'/knowledge-bases/{self.ka.id}/documents/{self.dob.id}/chunks'),
            ('DELETE', f'/knowledge-bases/{self.ka.id}/documents/{self.dob.id}')]:
            self.assertEqual(self.request(method, path).status_code, 404)
        self.milvus.assert_not_called()
        self.assertTrue(Path(self.dob.file_path).exists())

    def test_mismatched_document_owner_is_hidden_even_in_owned_kb(self):
        bad = self.new_document(self.b, self.ka, '错误归属.pdf')
        self.db.commit()
        listing = self.request('GET', f'/knowledge-bases/{self.ka.id}/documents')
        self.assertEqual([d['id'] for d in listing.json()], [str(self.da.id)])
        self.assertEqual(self.request('GET', f'/knowledge-bases/{self.ka.id}/documents/{bad.id}/chunks').status_code, 404)
        self.assertEqual(self.request('DELETE', f'/knowledge-bases/{self.ka.id}/documents/{bad.id}').status_code, 404)

    def test_chat_knowledge_selection_cannot_search_foreign_kb(self):
        response = self.request('POST', '/chat/completion', json=self.chat_body(search_knowledge=True, kb_ids=[str(self.kb.id)]))
        self.assertEqual(response.status_code, 200)
        self.chat_hybrid.assert_not_called()
        self.request('POST', '/chat/completion', json=self.chat_body(search_knowledge=True))
        self.chat_hybrid.assert_called_once_with(kb_id=str(self.ka.id), question='测试', top_k=None, allow_benchmark=False)

    def test_private_collection_cannot_be_redirected_by_benchmark_profile(self):
        with patch('service.eval.profile.resolve_kb_collection', return_value='shared_eval'):
            self.assertEqual(kb_collection_name(str(self.ka.id)), 'shared_eval')
            private_a = kb_collection_name(str(self.ka.id), allow_benchmark=False)
            private_b = kb_collection_name(str(self.kb.id), allow_benchmark=False)
            self.assertNotEqual(private_a, private_b)
            self.assertNotEqual(private_a, 'shared_eval')

    def test_chat_passes_third_page_candidates_to_reranker_in_both_routes(self):
        own_id = hashlib.md5(self.da.filename.encode()).hexdigest()
        self.chat_hybrid.return_value = [
            {'document_id': own_id, 'chunk_id': f'片段-{i}', 'content_with_weight': f'证据-{i}'}
            for i in range(384)]
        rerank = Mock(side_effect=lambda query, rows: rows[-8:])
        app_main.app.dependency_overrides[chat.get_services] = lambda: {'chat_service': SimpleNamespace(
            rerank_documents=rerank, retrieve_from_web=lambda query: [])}
        for path in ['/chat/completion', '/chat/completion/v3']:
            with self.subTest(path=path):
                response = self.request('POST', path, json=self.chat_body(search_knowledge=True))
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(rerank.call_args.args[1]), 384)
                self.assertEqual(response.json()['documents'][0]['content'], '证据-376')

    def test_chat_knowledge_filters_unowned_or_orphan_index_hits(self):
        bad = self.new_document(self.b, self.ka, '错误归属正文.pdf')
        self.db.commit()
        self.chat_hybrid.return_value = [
            {'document_id': hashlib.md5(self.da.filename.encode()).hexdigest(), 'content_with_weight': '甲已授权正文'},
            {'document_id': hashlib.md5(bad.filename.encode()).hexdigest(), 'content_with_weight': '错误归属正文'},
            {'document_id': 'orphan', 'content_with_weight': '无归属残留正文'},
        ]
        response = self.request('POST', '/chat/completion', json=self.chat_body(search_knowledge=True))
        self.assertEqual(response.status_code, 200)
        self.assertEqual([row['content'] for row in response.json()['documents']], ['甲已授权正文'])

    def test_session_and_memory_lists_are_owned_and_foreign_details_denied(self):
        self.assertEqual([x['id'] for x in self.request('GET', '/sessions').json()], [str(self.sa.id)])
        self.assertEqual(self.request('GET', f'/sessions/{self.sb.id}').status_code, 404)
        self.assertEqual(self.request('GET', f'/sessions/{self.sb.id}/messages').status_code, 404)
        self.assertEqual(self.request('DELETE', f'/sessions/{self.sb.id}').status_code, 404)
        response = self.request('GET', '/memories')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('甲私有记忆', response.text)
        self.assertNotIn('乙私有记忆', response.text)

    def test_research_state_resume_and_events_do_not_read_foreign_run(self):
        fake = SimpleNamespace(read_state=AsyncMock())
        with patch.object(research, 'get_research_service', return_value=fake):
            self.assertEqual(self.request('GET', f'/research/runs/{self.run_b}/state').status_code, 404)
            self.assertEqual(self.request('POST', f'/research/resume/{self.run_b}').status_code, 404)
            fake.read_state.assert_not_called()
        response = self.request('GET', f'/research/runs/{self.run_b}/events')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['events'], [])
        owner_events = self.request('GET', f'/research/runs/{self.run_a}/events')
        self.assertIn('甲私有事件', owner_events.text)
        self.assertNotIn('乙私有事件', owner_events.text)

    def test_invalid_expired_disabled_and_default_key_forged_tokens_are_rejected(self):
        tokens = [
            'invalid-token', security.create_access_token({'sub': str(self.a.id)}, expires_delta=timedelta(seconds=-1)),
            security.create_access_token({'sub': 'not-a-uuid'}),
            jwt.encode({'sub': str(self.a.id)}, 'your-super-secret-key-change-in-production', algorithm='HS256'),
        ]
        for token in tokens:
            response = self.client.get('/sessions', headers={'Authorization': 'Bearer ' + token})
            self.assertEqual(response.status_code, 401)
        self.a.is_active = False
        self.db.commit()
        self.assertEqual(self.request('GET', '/sessions').status_code, 403)

    def test_missing_short_and_public_default_secrets_fail_closed(self):
        for secret in ['', 'short', 'your-super-secret-key-change-in-production']:
            with patch.object(security, 'SECRET_KEY', secret):
                with self.assertRaises(RuntimeError):
                    security.validate_security_configuration()
                with self.assertRaises(RuntimeError):
                    security.create_access_token({'sub': str(self.a.id)})
                self.assertIsNone(security.decode_token(self.token_a))

    def test_second_user_has_own_resources_and_cannot_access_first_user(self):
        self.assertEqual([x['id'] for x in self.request('GET', '/sessions', user='b').json()], [str(self.sb.id)])
        self.assertEqual(self.request('GET', f'/attachments/{self.ab.id}', user='b').status_code, 200)
        self.assertEqual(self.request('GET', f'/attachments/{self.aa.id}', user='b').status_code, 404)
        self.assertEqual(self.request('DELETE', f'/attachments/{self.aa.id}', user='b').status_code, 404)
        self.assertEqual(self.request('GET', f'/documents/list?kb_id={self.ka.id}', user='b').status_code, 404)
        own_docs = self.request('GET', f'/documents/list?kb_id={self.kb.id}', user='b').json()['data']['docs']
        self.assertEqual([doc['id'] for doc in own_docs], [str(self.dob.id)])
        own_memories = self.request('GET', '/memories', user='b')
        self.assertIn('乙私有记忆', own_memories.text)
        self.assertNotIn('甲私有记忆', own_memories.text)


if __name__ == '__main__':
    unittest.main()
