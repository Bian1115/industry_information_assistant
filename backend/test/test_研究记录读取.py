"""研究结果接口：归属校验、图业务底稿与历史兼容。"""
import copy
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
routes = importlib.import_module('router.research_router')


class ResearchResultTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(routes.router)
        self.user = SimpleNamespace(id=uuid4())
        self.db = Mock()
        self.db.query.return_value.filter.return_value.order_by.return_value.first.return_value = SimpleNamespace(payload={'content': '已记录的执行错误'})
        self.app.dependency_overrides[routes.get_db] = lambda: self.db
        self.app.dependency_overrides[routes.get_current_user_required] = lambda: self.user
        self.client = TestClient(self.app)
        self.run_id = uuid4()
        self.state = {'exists': True, 'pending_nodes': ['scout'], 'saved_result_nodes': [], 'resumable': True,
            'graph_version': 'single-main-v1', 'status': 'running', 'research': {
            'query': '研究主题', 'raw_sources': [{'id': 's1', 'content': '来源正文'}],
            'facts': [{'id': 'f1', 'source_id': 's1', 'content': '事实'}],
            'final_report': '已保存报告', 'search_budget': {'queries': [{'query': '检索词'}]},
        }}

    def fetch(self, status='failed', graph_version='single-main-v1', legacy=None):
        service = SimpleNamespace(read_state=AsyncMock(return_value=copy.deepcopy(self.state)))
        with patch.object(routes, '_owned_run', return_value=SimpleNamespace(status=status, graph_version=graph_version)), patch.object(routes, 'get_research_service', return_value=service), patch('service.medical_research.storage.owned_medical_checkpoint', return_value=legacy):
            return self.client.get(f'/research/runs/{self.run_id}/state')

    def test_failed_run_returns_saved_sources_and_authoritative_status(self):
        response = self.fetch()
        self.assertEqual(response.status_code, 200)
        state = response.json()['state']
        self.assertEqual(state['run_status'], 'failed')
        self.assertEqual(state['error'], '已记录的执行错误')
        self.assertEqual(state['draft']['raw_sources'][0]['content'], '来源正文')
        self.assertEqual(state['draft']['final_report'], '已保存报告')
        self.assertTrue(state['resumable'])
        self.assertNotIn('research', state)

    def test_missing_checkpoint_is_readable_but_not_resumable(self):
        self.state.update(exists=False, pending_nodes=[], resumable=False, graph_version='', research={})
        state = self.fetch(graph_version=None).json()['state']
        self.assertEqual(state['pipeline'], 'graph')
        self.assertFalse(state['resumable'])

    def test_legacy_run_is_explicitly_marked(self):
        self.state.update(exists=False, pending_nodes=[], resumable=False, graph_version='', research={})
        self.assertEqual(self.fetch(graph_version=None, legacy=object()).json()['state']['pipeline'], 'legacy')

    def test_unowned_run_never_reads_checkpoint(self):
        with patch.object(routes, '_owned_run', side_effect=HTTPException(404, '未找到本人的研究运行')), patch.object(routes, 'get_research_service') as service:
            self.assertEqual(self.client.get(f'/research/runs/{self.run_id}/state').status_code, 404)
            service.assert_not_called()

    def test_missing_authentication_is_rejected(self):
        self.app.dependency_overrides.pop(routes.get_current_user_required)
        self.assertIn(self.client.get(f'/research/runs/{self.run_id}/state').status_code, (401, 403))


if __name__ == '__main__':
    unittest.main()
