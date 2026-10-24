# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""评估运行时模块：追踪与 benchmark 数据源适配。"""

from .tracer import (
    trace_context,
    record_llm_call,
    attach_parse_result,
    flush_pending,
    is_trace_enabled,
    shutdown_tracer,
)

from .profile import (
    DataProfile,
    PRODUCTION,
    use_profile,
    active_profile,
    get_profile,
    list_profiles,
)

__all__ = [
    # 追踪
    "trace_context",
    "record_llm_call",
    "attach_parse_result",
    "flush_pending",
    "is_trace_enabled",
    "shutdown_tracer",
    # 数据源适配
    "DataProfile",
    "PRODUCTION",
    "use_profile",
    "active_profile",
    "get_profile",
    "list_profiles",
]
