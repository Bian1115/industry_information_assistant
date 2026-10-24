# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""会话过期、记忆、语义上下文与网页缓存后台维护调度。"""
import asyncio
import logging
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class SchedulerService:
    """定时任务调度服务"""

    _instance: Optional['SchedulerService'] = None
    _scheduler: Optional[AsyncIOScheduler] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._scheduler is None:
            self._scheduler = AsyncIOScheduler()

    def start(self):
        """启动调度器"""
        if not self._scheduler.running:
            self._scheduler.add_job(
                self._cleanup_expired_sessions,
                IntervalTrigger(minutes=1),
                id="expired_sessions_cleanup",
                name="清理超过24小时无新消息的会话",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            self._scheduler.add_job(
                self._process_memory_jobs,
                IntervalTrigger(seconds=5),
                id="memory_outbox_worker",
                name="记忆与语义上下文 Outbox Worker",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            self._scheduler.add_job(
                self._reconcile_memory_state,
                IntervalTrigger(minutes=10),
                id="memory_state_reconciliation",
                name="长期记忆一致性检查",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            self._scheduler.add_job(
                self._cleanup_web_reader_cache,
                IntervalTrigger(minutes=10),
                id="web_reader_cache_cleanup",
                name="清理过期网页读取缓存",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )

            self._scheduler.start()
            logger.info("定时任务调度器已启动")
            logger.info("已添加每5秒执行的记忆与语义上下文 Outbox Worker")

    def stop(self):
        """停止调度器"""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown()
            logger.info("定时任务调度器已停止")

    async def _process_memory_jobs(self):
        """在工作线程处理记忆、语义摘要、Embedding 和 Milvus 调用。"""
        from service.memory_service import process_memory_jobs

        processed = await asyncio.to_thread(process_memory_jobs, 10)
        if processed:
            logger.info("本轮处理长期记忆任务 %s 个", processed)

    async def _reconcile_memory_state(self):
        from service.memory_service import reconcile_memory_state

        result = await asyncio.to_thread(reconcile_memory_state)
        if result.get("stale_memories"):
            logger.warning("长期记忆一致性修复: %s", result)

    async def _cleanup_web_reader_cache(self):
        from service.web_reader_service import WebReaderService

        def cleanup():
            return WebReaderService().cleanup_expired_cache()

        removed = await asyncio.to_thread(cleanup)
        if removed:
            logger.info("已清理 %s 个过期网页缓存文件", removed)

    async def _cleanup_expired_sessions(self):
        from service.session_retention_service import cleanup_expired_sessions
        from service.session_evidence_lifecycle import cleanup_expired_evidence, process_evidence_purges

        await asyncio.to_thread(cleanup_expired_evidence)

        removed = await asyncio.to_thread(cleanup_expired_sessions)
        await asyncio.to_thread(process_evidence_purges)
        if removed:
            logger.info("已清理 %s 个过期会话及其消息", removed)

    def get_jobs_info(self) -> list:
        """获取所有任务信息"""
        jobs = []
        if self._scheduler:
            for job in self._scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": str(job.next_run_time) if job.next_run_time else None,
                    "trigger": str(job.trigger)
                })
        return jobs


# 全局调度器实例
_scheduler_service: Optional[SchedulerService] = None


def get_scheduler_service() -> SchedulerService:
    """获取调度器服务单例"""
    global _scheduler_service
    if _scheduler_service is None:
        _scheduler_service = SchedulerService()
    return _scheduler_service


async def init_scheduler():
    """补清理停机期间过期的数据，再启动后台维护任务。"""
    scheduler = get_scheduler_service()
    await scheduler._cleanup_web_reader_cache()
    await scheduler._cleanup_expired_sessions()
    scheduler.start()
