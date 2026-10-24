# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
评估追踪模型 - 记录每一次 LLM 调用的原始事实。

实现集中在 backend/eval；app/models/eval_trace.py 仅保留兼容导入。

为什么需要它：评估体系的前提是"有数据可查"。系统跑完一次研究，
6 个 Agent 前后调用十几次 LLM，但这些调用的耗时、token、是否解析成功
全部只进了 logging，进程一退就没了。没有落库的调用记录，
就无法回答"这个 badcase 到底坏在哪个 Agent"这类问题。

与 research_checkpoints 的分工：
- research_checkpoints 存"一次研究的最终结果"（一行）
- agent_runs 存"这次研究里每一跳 LLM 调用"（十几行）
两者用 session_id 关联。
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, Boolean, Index
)
from sqlalchemy.dialects.postgresql import UUID

from core.database import Base


class AgentRun(Base):
    """
    单次 LLM 调用记录

    刻意不对 users 表建外键：用户被删除时评估数据仍需保留，
    否则历史 badcase 会随用户注销一起消失。
    """
    __tablename__ = "agent_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # ---- 关联信息 ----
    session_id = Column(String(64), index=True, nullable=False, comment="研究会话 ID")
    user_id = Column(UUID(as_uuid=True), nullable=True, comment="用户 ID（不建外键，保留历史数据）")
    query = Column(Text, comment="原始查询（冗余存储，便于直接翻 badcase）")

    # ---- 调用主体 ----
    agent_name = Column(String(64), index=True, nullable=False, comment="Agent 名称")
    model = Column(String(64), comment="使用的模型")
    phase = Column(String(32), comment="所处阶段 planning/researching/...")
    iteration = Column(Integer, default=0, comment="第几轮迭代")

    # ---- 调用参数 ----
    temperature = Column(Float, comment="温度参数")
    json_mode = Column(Boolean, comment="是否强制 JSON 输出")

    # ---- 消耗与耗时 ----
    prompt_tokens = Column(Integer, comment="输入 token")
    completion_tokens = Column(Integer, comment="输出 token")
    total_tokens = Column(Integer, comment="总 token")
    duration_ms = Column(Integer, comment="耗时（毫秒）")

    # ---- 结果 ----
    success = Column(Boolean, default=True, index=True, comment="调用是否成功")
    error_type = Column(String(128), comment="异常类名")
    error_message = Column(Text, comment="异常信息")

    # ---- JSON 解析质量（本项目最脆弱的一环）----
    # parse_ok=None 表示这次调用不需要解析 JSON
    parse_ok = Column(Boolean, comment="JSON 是否解析成功")
    parse_strategy = Column(
        String(32),
        comment="命中哪层兜底: direct/code_block/braces/literal_eval/failed"
    )

    # ---- 载荷快照（默认截断，避免表膨胀）----
    input_preview = Column(Text, comment="输入截断快照")
    output_preview = Column(Text, comment="输出截断快照")

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        # 翻某次研究的完整调用链：按 session 查 + 按时间排
        Index("ix_agent_runs_session_created", "session_id", "created_at"),
        # 按 Agent 统计失败率
        Index("ix_agent_runs_agent_success", "agent_name", "success"),
    )

    def to_dict(self, include_payload: bool = False):
        result = {
            "id": str(self.id),
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "phase": self.phase,
            "iteration": self.iteration,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_type": self.error_type,
            "parse_ok": self.parse_ok,
            "parse_strategy": self.parse_strategy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_payload:
            result["query"] = self.query
            result["input_preview"] = self.input_preview
            result["output_preview"] = self.output_preview
            result["error_message"] = self.error_message
        return result
