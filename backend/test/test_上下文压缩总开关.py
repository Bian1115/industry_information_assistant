"""总开关关闭后的原文读取及压缩缓存隔离。"""
import os
import uuid
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_session_memory import (FakeDb, FakeRedis, FakeTokenCounter, make_message)
from service.context_compaction import compact_messages
from service.context_policy import (context_compression_enabled, session_context_compression_enabled,
                                   ContextPolicy, policy_for_scope)
from service.agent_history import HistoryScope
from service.semantic_context_service import semantic_context_enabled
from service.session_memory_service import RedisSessionMemoryService


class CompressionSwitchTests(unittest.TestCase):
    def test_writer_methods_enabled_while_session_keeps_full_original_history(self):
        redis = FakeRedis()
        service = RedisSessionMemoryService(redis_client=redis, token_counter=FakeTokenCounter(),
                                            enabled=True, storage_token_limit=256, storage_max_messages=2)
        rows = [make_message(i, "user" if i % 2 == 0 else "assistant", f"第{i}条" * 100)
                for i in range(30)]
        db = FakeDb(rows)
        db.session.semantic_context_summary = {"overview": "不能复用旧摘要"}
        root = ContextPolicy(group="会话原文Writer压缩", compression_enabled=True,
                             session_compression_enabled=False, semantic_verification=False).with_agent_compression({
            "LeadWriter": {"enabled": True, "reference_materials": True,
                           "artifact_offload": True, "semantic_summary": True}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "冻结配置.json"
            path.write_text(json.dumps(root.snapshot()))
            with patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": str(path),
                                         "CONTEXT_COMPRESSION_ENABLED": "0",
                                         "SESSION_CONTEXT_COMPRESSION_ENABLED": "1",
                                         "SEMANTIC_CONTEXT_ENABLED": "true"}), \
                 patch.object(service, "_read_payload", side_effect=AssertionError("不能用裁剪缓存")):
                self.assertTrue(context_compression_enabled())
                self.assertFalse(session_context_compression_enabled())
                self.assertFalse(semantic_context_enabled())
                result = service.get_context(db, user_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()),
                                             token_limit=10, max_messages=1)
                self.assertEqual(result.messages, [{"role": row.role, "content": row.content} for row in rows])
                self.assertEqual(result.original_tokens, result.compacted_tokens)
                self.assertEqual(result.semantic_context_version, 0)
                fitted = compact_messages(result.messages, token_limit=1, max_messages=1,
                                          token_counter=FakeTokenCounter())
                self.assertEqual(fitted.messages, result.messages)
        view = policy_for_scope(HistoryScope(None, "r", "s", None, context_policy=root.snapshot()), "LeadWriter")
        self.assertTrue(view.semantic_summary)
        self.assertTrue(view.artifact_offload)
        self.assertTrue(view.input_history.reference_materials)

    def test_master_switch_wins_and_old_snapshot_hash_is_preserved(self):
        root = ContextPolicy(group="旧快照", semantic_verification=False)
        self.assertNotIn("session_compression_enabled", root.snapshot()["config"])
        self.assertEqual(ContextPolicy.from_snapshot(root.snapshot()).fingerprint, root.fingerprint)
        self.assertTrue(root.effective_session_compression_enabled)
        disabled = root.model_copy(update={"compression_enabled": False, "session_compression_enabled": True})
        self.assertFalse(disabled.effective_session_compression_enabled)
        with patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": "", "CONTEXT_COMPRESSION_ENABLED": "1",
                                     "SESSION_CONTEXT_COMPRESSION_ENABLED": "invalid"}):
            with self.assertRaisesRegex(ValueError, "SESSION_CONTEXT_COMPRESSION_ENABLED"):
                session_context_compression_enabled()

    def test_explicit_policy_controls_session_switch_and_old_files_keep_original_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "冻结策略.json"
            root = ContextPolicy(group="测试", semantic_verification=False, compression_enabled=False)
            path.write_text(json.dumps(root.snapshot()))
            with patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": str(path), "CONTEXT_COMPRESSION_ENABLED": "1"}):
                self.assertFalse(context_compression_enabled())
            path.write_text(json.dumps(ContextPolicy(group="旧配置", semantic_verification=False).snapshot()))
            with patch.dict(os.environ, {"CONTEXT_ABLATION_CONFIG": str(path), "CONTEXT_COMPRESSION_ENABLED": "0"}):
                self.assertTrue(context_compression_enabled())

    def test_off_keeps_all_messages_and_full_tool_result(self):
        messages = [{"role": "user", "content": "第一轮" * 100},
                    {"role": "tool", "content": "工具原文" * 2000},
                    {"role": "assistant", "content": "回答" * 100}]
        with patch.dict(os.environ, {"CONTEXT_COMPRESSION_ENABLED": "0", "SEMANTIC_CONTEXT_ENABLED": "true"}):
            result = compact_messages(messages, token_limit=10, max_messages=1,
                                      token_counter=FakeTokenCounter())
            self.assertEqual(result.messages, messages)
            self.assertEqual(result.original_tokens, result.compacted_tokens)
            self.assertEqual(result.pruned_tool_results, 0)
            self.assertFalse(semantic_context_enabled())

    def test_off_reads_postgres_full_history_without_using_old_cache_or_summary(self):
        redis = FakeRedis()
        service = RedisSessionMemoryService(redis_client=redis, token_counter=FakeTokenCounter(),
                                            enabled=True, storage_token_limit=256, storage_max_messages=2)
        rows = [make_message(i, "user" if i % 2 == 0 else "assistant", f"第{i}条" * 100)
                for i in range(30)]
        db = FakeDb(rows)
        db.session.semantic_context_summary = {"overview": "不应注入旧摘要"}
        with patch.dict(os.environ, {"CONTEXT_COMPRESSION_ENABLED": "0"}), \
             patch.object(service, "_read_payload", side_effect=AssertionError("不应读取压缩缓存")), \
             patch.object(service, "_write_payload", side_effect=AssertionError("不应写入全量原文缓存")):
            result = service.get_context(db, user_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()),
                                         token_limit=10, max_messages=1)
        self.assertEqual(len(result.messages), 30)
        self.assertEqual(result.messages[0]["content"], rows[0].content)
        self.assertEqual(result.messages[-1]["content"], rows[-1].content)
        self.assertEqual(result.original_tokens, result.compacted_tokens)
        self.assertFalse(result.cache_hit)
        self.assertEqual(result.semantic_context_version, 0)

    def test_invalid_switch_fails_instead_of_silently_enabling_compression(self):
        with patch.dict(os.environ, {"CONTEXT_COMPRESSION_ENABLED": "invalid"}):
            with self.assertRaises(ValueError):
                context_compression_enabled()


if __name__ == "__main__":
    unittest.main()
