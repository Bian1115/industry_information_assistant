"""网页读取缓存到期删除与后台调度回归；所有文件仅写入临时目录。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from service.web_reader_service import WebReaderService, WebReadResult


class CacheExpiryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.environment = patch.dict(os.environ, {
            "WEB_READER_CACHE_DIR": self.temporary.name,
            "WEB_READER_CACHE_TTL_SECONDS": "100",
            "WEB_READER_FAILURE_TTL_SECONDS": "10",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.reader = WebReaderService()

    def store(self, url, fetched_at=1000, success=True):
        key = self.reader._cache_key(url)
        result = WebReadResult(
            url=url, final_url=url, fetched_at=fetched_at,
            fetch_status="success" if success else "failed",
            content="保存的网页正文" if success else "",
        )
        self.reader._save_cache(key, result)
        return key, self.reader._cache_path(key)

    def test_expired_memory_entry_and_file_are_deleted_on_read(self):
        key, path = self.store("https://example.org/expired")
        with patch("service.web_reader_service.time.time", return_value=1100):
            self.assertIsNone(self.reader._load_cache(key))
        self.assertNotIn(key, self.reader._memory_cache)
        self.assertFalse(path.exists())

    def test_disk_only_expired_entry_is_deleted_on_read(self):
        key, path = self.store("https://example.org/disk")
        self.reader._memory_cache.clear()
        with patch("service.web_reader_service.time.time", return_value=1100):
            self.assertIsNone(self.reader._load_cache(key))
        self.assertFalse(path.exists())

    def test_sweep_removes_unvisited_expired_successes_and_failures(self):
        old_key, old_path = self.store("https://example.org/old")
        failed_key, failed_path = self.store("https://example.org/failure", success=False)
        fresh_key, fresh_path = self.store("https://example.org/fresh", fetched_at=1095)
        _, fresh_failed_path = self.store("https://example.org/fresh-failure", fetched_at=1095, success=False)
        unrelated = Path(self.temporary.name) / "归档.json"
        unrelated.write_text('{"content":"应保留的研究来源"}', encoding="utf-8")
        with patch("service.web_reader_service.time.time", return_value=1100):
            self.assertEqual(self.reader.cleanup_expired_cache(), 2)
        self.assertFalse(old_path.exists())
        self.assertFalse(failed_path.exists())
        self.assertNotIn(old_key, self.reader._memory_cache)
        self.assertNotIn(failed_key, self.reader._memory_cache)
        self.assertTrue(fresh_path.exists())
        self.assertTrue(fresh_failed_path.exists())
        self.assertIn(fresh_key, self.reader._memory_cache)
        self.assertTrue(unrelated.exists())

    def test_expired_memory_does_not_delete_another_readers_fresh_file(self):
        key, path = self.store("https://example.org/shared")
        other_reader = WebReaderService()
        other_reader._save_cache(key, WebReadResult(
            url="https://example.org/shared", final_url="https://example.org/shared",
            fetched_at=1095, fetch_status="success", content="刷新后的正文",
        ))
        with patch("service.web_reader_service.time.time", return_value=1100):
            self.assertIsNone(self.reader._load_cache(key))
            self.assertEqual(self.reader._load_cache(key).content, "刷新后的正文")
        self.assertTrue(path.exists())

    def test_delete_error_does_not_prevent_other_expired_files_cleanup(self):
        _, blocked_path = self.store("https://example.org/blocked")
        _, removable_path = self.store("https://example.org/removable")
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == blocked_path:
                raise PermissionError("测试文件无法删除")
            return original_unlink(path, *args, **kwargs)

        with patch("service.web_reader_service.time.time", return_value=1100), patch.object(Path, "unlink", unlink):
            self.assertEqual(self.reader.cleanup_expired_cache(), 1)
        self.assertTrue(blocked_path.exists())
        self.assertFalse(removable_path.exists())

    def test_sweep_skips_malformed_files_and_temporary_writes(self):
        malformed = Path(self.temporary.name) / ("a" * 64 + ".json")
        malformed.write_text("{", encoding="utf-8")
        temporary = malformed.with_suffix(".tmp")
        temporary.write_text("写入中", encoding="utf-8")
        with patch("service.web_reader_service.time.time", return_value=1100):
            self.assertEqual(self.reader.cleanup_expired_cache(), 0)
        self.assertTrue(malformed.exists())
        self.assertTrue(temporary.exists())


class CacheSchedulerTests(unittest.IsolatedAsyncioTestCase):
    def test_cleanup_job_is_registered_every_ten_minutes(self):
        from service.scheduler_service import SchedulerService

        scheduler = object.__new__(SchedulerService)
        scheduler._scheduler = MagicMock()
        scheduler._scheduler.running = False
        scheduler.start()
        cleanup_jobs = [call for call in scheduler._scheduler.add_job.call_args_list
                        if call.kwargs.get("id") == "web_reader_cache_cleanup"]
        self.assertEqual(len(cleanup_jobs), 1)
        self.assertEqual(cleanup_jobs[0].args[1].interval.total_seconds(), 600)
        self.assertEqual(cleanup_jobs[0].kwargs["max_instances"], 1)

    async def test_startup_cleans_expired_files_before_starting_scheduler(self):
        from service.scheduler_service import init_scheduler

        scheduler = MagicMock()
        scheduler._cleanup_web_reader_cache = AsyncMock()
        scheduler._cleanup_expired_sessions = AsyncMock()
        with patch("service.scheduler_service.get_scheduler_service", return_value=scheduler):
            await init_scheduler()
        scheduler._cleanup_web_reader_cache.assert_awaited_once()
        scheduler.start.assert_called_once()
        self.assertEqual(scheduler.mock_calls[0][0], "_cleanup_web_reader_cache")


if __name__ == "__main__":
    unittest.main()
