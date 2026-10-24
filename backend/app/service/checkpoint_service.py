# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""历史研究快照的只读读取。

LangGraph 接管执行恢复后，旧的完整业务快照退出新运行的读写路径：
这里只保留按本人归属读取历史记录的查询，不再提供保存、改状态或删除的
写入方法。新研究的执行位置与业务状态由 LangGraph 检查点负责。
"""
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID
from sqlalchemy import false, func
from sqlalchemy.orm import Session

from models.research import AgentEvent, ResearchCheckpoint
from core.database import SessionLocal

logger = logging.getLogger(__name__)


class CheckpointService:
    """检查点服务"""

    def __init__(self):
        pass

    def _get_db(self) -> Session:
        """获取数据库会话"""
        return SessionLocal()

    def _owned_query(
        self,
        db: Session,
        session_id: str,
        user_id: Optional[str],
        run_id: Optional[str] = None,
    ):
        """
        构造一个带归属过滤的查询（归属校验下沉到这一层）

        把 user_id 条件直接写进 SQL 的 WHERE，别人的检查点在数据库层面就查不出来。
        这样上层路由不需要（也无法）自己写校验逻辑，从根上避免"某个接口忘了校验"。

        Args:
            db: 数据库会话
            session_id: 研究会话 ID
            user_id: 归属校验用的用户 ID。
                     传 None 表示"内部可信调用"，不做归属过滤；
                     对外的 HTTP 接口必须传入真实的 user_id。

        Returns:
            已经加好过滤条件的 Query 对象
        """
        query = db.query(ResearchCheckpoint).filter(
            ResearchCheckpoint.session_id == session_id
        )
        if run_id:
            query = query.filter(ResearchCheckpoint.run_id == run_id)

        if user_id is None:
            return query

        try:
            owner_id = UUID(str(user_id))
        except (ValueError, AttributeError, TypeError):
            # user_id 格式非法时，宁可什么都查不到，也不能退化成"不过滤"
            logger.warning(f"[CheckpointService] 非法的 user_id，归属过滤拒绝所有结果: {user_id!r}")
            return query.filter(false())

        return query.filter(ResearchCheckpoint.user_id == owner_id)

    @staticmethod
    def _latest_event_seq(db: Session, run_id: Optional[str]) -> int:
        """返回 Checkpoint 对应事件流已经持久化的最大序号。"""
        if not run_id:
            return 0
        latest = db.query(func.max(AgentEvent.seq)).filter(
            AgentEvent.run_id == run_id
        ).scalar()
        return int(latest or 0)

    def load_checkpoint(
        self,
        session_id: str,
        user_id: Optional[str],
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        加载最新的检查点（仅后端状态）

        Args:
            session_id: 研究会话 ID
            user_id: 归属校验用的用户 ID；内部可信调用可显式传 None。
                     该参数故意不给默认值，强制每个调用方明确表态。

        Returns:
            ResearchState 字典，未找到或不属于该用户返回 None
        """
        db = self._get_db()
        try:
            checkpoint = self._owned_query(db, session_id, user_id, run_id).order_by(
                ResearchCheckpoint.updated_at.desc()
            ).first()

            if not checkpoint:
                return None

            return checkpoint.state_json

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
        finally:
            db.close()

    def load_full_checkpoint(
        self,
        session_id: str,
        user_id: Optional[str],
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        加载完整的检查点（包含后端状态、UI状态和报告）

        Args:
            session_id: 研究会话 ID
            user_id: 归属校验用的用户 ID；内部可信调用可显式传 None。

        Returns:
            完整检查点数据，包含 state_json, ui_state_json, final_report 等；
            未找到或不属于该用户返回 None
        """
        db = self._get_db()
        try:
            checkpoint = self._owned_query(db, session_id, user_id, run_id).order_by(
                ResearchCheckpoint.updated_at.desc()
            ).first()

            if not checkpoint:
                logger.info(f"[CheckpointService] 未找到检查点（或不属于当前用户）: session={session_id}")
                return None

            result = checkpoint.to_dict(include_state=True)
            # 详细日志
            ui_state = result.get("ui_state_json", {})
            if ui_state:
                logger.info(f"[CheckpointService] 加载成功: session={session_id}, phase={result.get('phase')}, "
                           f"ui_state=[steps={len(ui_state.get('research_steps', []))}, "
                           f"search_results={len(ui_state.get('search_results', []))}, "
                           f"charts={len(ui_state.get('charts', []))}, "
                           f"kg_nodes={len((ui_state.get('knowledge_graph') or {}).get('nodes', []))}]")
            else:
                logger.info(f"[CheckpointService] 加载成功但无ui_state: session={session_id}, phase={result.get('phase')}")
            return result

        except Exception as e:
            logger.error(f"Failed to load full checkpoint: {e}")
            return None
        finally:
            db.close()

    def get_checkpoint_info(
        self,
        session_id: str,
        user_id: Optional[str],
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        获取检查点信息（不包含完整状态）

        Args:
            session_id: 研究会话 ID
            user_id: 归属校验用的用户 ID；内部可信调用可显式传 None。

        Returns:
            检查点元信息；未找到或不属于该用户返回 None
        """
        db = self._get_db()
        try:
            checkpoint = self._owned_query(db, session_id, user_id, run_id).order_by(
                ResearchCheckpoint.updated_at.desc()
            ).first()

            if not checkpoint:
                return None

            return checkpoint.to_dict()

        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            return None
        finally:
            db.close()

    def list_checkpoints(
        self,
        user_id: Optional[str],
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        列出检查点

        Args:
            user_id: 用户 ID。传 None 表示"内部可信调用，列出全部"，
                     对外接口必须传入真实用户 ID，否则会返回全站数据。
                     该参数故意不给默认值，强制每个调用方明确表态。
            status: 状态过滤
            limit: 限制数量

        Returns:
            检查点列表
        """
        db = self._get_db()
        try:
            query = db.query(ResearchCheckpoint)

            if user_id is not None:
                try:
                    query = query.filter(ResearchCheckpoint.user_id == UUID(str(user_id)))
                except (ValueError, AttributeError, TypeError):
                    logger.warning(f"[CheckpointService] 非法的 user_id，返回空列表: {user_id!r}")
                    return []
            if status:
                query = query.filter(ResearchCheckpoint.status == status)

            checkpoints = query.order_by(
                ResearchCheckpoint.updated_at.desc()
            ).limit(limit).all()

            return [cp.to_dict() for cp in checkpoints]

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
        finally:
            db.close()


# 全局只读服务实例
_checkpoint_service: Optional[CheckpointService] = None


def get_checkpoint_service() -> CheckpointService:
    """获取检查点服务实例"""
    global _checkpoint_service
    if _checkpoint_service is None:
        _checkpoint_service = CheckpointService()
    return _checkpoint_service
