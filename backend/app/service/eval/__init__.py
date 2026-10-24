"""兼容入口：评估实现已集中到 backend/eval/runtime。"""

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
    "trace_context",
    "record_llm_call",
    "attach_parse_result",
    "flush_pending",
    "is_trace_enabled",
    "shutdown_tracer",
    "DataProfile",
    "PRODUCTION",
    "use_profile",
    "active_profile",
    "get_profile",
    "list_profiles",
]

