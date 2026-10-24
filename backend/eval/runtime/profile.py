# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
数据源 Profile —— benchmark harness 适配层

## 解决什么问题

公开 benchmark 是一个固定三元组：(语料库, 问题集, 标准答案)。
它能直接跑的前提是被测系统读的是"它的语料"。
而本系统读的是自己的研报知识库和 industry_stats 表，两边对不上。

硬改数据源去迁就 benchmark，会把业务代码搞脏；
为 benchmark 单独复制一套代码，测的又不是真实系统。

Profile 的做法：**让数据源可切换，但被测代码一行不改**。
业务代码照常调 kb_collection_name() / Text2SQLService，
只是这些函数在读取"当前激活的 Profile"来决定指向哪个集合、哪个库。

## 三种数据源的适配情况

| 部件            | 是否需要适配 | 原因                                  |
|-----------------|--------------|---------------------------------------|
| deep research   | ❌ 不需要    | 靠 web_search 驱动，本身就是领域无关的 |
| RAG 检索        | ✅ 需要      | 绑定私有研报 Milvus 集合 / ES 索引     |
| Text2SQL        | ✅ 需要      | 绑定私有 industry_stats 表结构         |

## 安全性

默认永远是 production（全部字段为 None，等价于完全不介入）。
只有显式调用 use_profile() 或设置 EVAL_PROFILE 环境变量才会切换，
且 use_profile() 是上下文管理器，退出即自动还原。
"""

import os
import json
import logging
import contextvars
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile 定义
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    """
    一套数据源配置。

    所有字段为 None 时表示"不介入"，走生产环境原有逻辑。
    """

    name: str
    description: str = ""

    # ---- 检索侧 ----
    # 覆盖 Milvus 集合名 / ES 索引名。两者共用同一个名字（见 kb_collection_name）
    kb_collection: Optional[str] = None

    # ---- Text2SQL 侧 ----
    sql_dsn: Optional[str] = None          # 数据库连接串
    sql_schema: Optional[str] = None       # 喂给 LLM 的 schema 描述（整段替换）
    sql_schema_file: Optional[str] = None  # 或者从文件读取 schema 描述

    # ---- 元信息（记录用，不参与逻辑）----
    benchmark: Optional[str] = None        # 对应哪个 benchmark
    meta: Dict[str, Any] = field(default_factory=dict)

    def resolved_sql_schema(self) -> Optional[str]:
        """sql_schema 优先；否则尝试读 sql_schema_file。"""
        if self.sql_schema:
            return self.sql_schema
        if self.sql_schema_file:
            try:
                with open(self.sql_schema_file, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError as e:
                logger.warning(f"[Profile:{self.name}] 读取 schema 文件失败: {e}")
        return None

    def is_passthrough(self) -> bool:
        """是否完全不介入（生产模式）。"""
        return not any([self.kb_collection, self.sql_dsn,
                        self.sql_schema, self.sql_schema_file])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 生产环境：全部字段为 None，等价于适配层不存在
PRODUCTION = DataProfile(
    name="production",
    description="生产数据源：私有研报知识库 + industry_stats 表",
)


# ---------------------------------------------------------------------------
# 当前激活的 Profile
# ---------------------------------------------------------------------------

# 用 contextvars 而不是全局变量：
# 评测脚本可能并发跑多个 profile，全局变量会互相覆盖。
# 与 tracer.py 保持一致的做法。
_active: contextvars.ContextVar[Optional[DataProfile]] = contextvars.ContextVar(
    "eval_active_profile", default=None
)


def active_profile() -> DataProfile:
    """
    读取当前激活的 Profile。

    优先级：use_profile() 上下文 > EVAL_PROFILE 环境变量 > production
    任何异常都退化成 production，绝不影响业务流程。
    """
    try:
        current = _active.get()
        if current is not None:
            return current

        env_name = os.getenv("EVAL_PROFILE")
        if env_name and env_name != "production":
            return get_profile(env_name)
    except Exception as e:
        logger.debug(f"[Profile] 读取失败，退回 production: {e}")

    return PRODUCTION


class use_profile:
    """
    临时切换数据源。

    用法：
        with use_profile("crud_rag"):
            hits = hybrid_retrieve(kb_id="任意值", question="...")
        # 退出即自动还原为原来的 profile

    也可以直接传 DataProfile 对象，便于脚本里动态构造。
    """

    def __init__(self, profile):
        if isinstance(profile, str):
            self._profile = get_profile(profile)
        elif isinstance(profile, DataProfile):
            self._profile = profile
        else:
            raise TypeError(f"profile 必须是 str 或 DataProfile，收到 {type(profile)}")
        self._token = None

    def __enter__(self) -> DataProfile:
        self._token = _active.set(self._profile)
        logger.info(f"[Profile] 已切换到: {self._profile.name}")
        return self._profile

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            try:
                _active.reset(self._token)
                logger.info(f"[Profile] 已还原（离开 {self._profile.name}）")
            except ValueError:
                # 跨任务 reset 会抛 ValueError，忽略
                pass
        return False


# ---------------------------------------------------------------------------
# Profile 注册表
# ---------------------------------------------------------------------------

# 配置文件位置：backend/eval/profiles.json
# 本文件在 backend/eval/runtime/profile.py，向上两层到 backend/。
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_BACKEND_DIR, "eval", "profiles.json")

_registry: Optional[Dict[str, DataProfile]] = None


def _config_path() -> str:
    return os.getenv("EVAL_PROFILES_CONFIG", _DEFAULT_CONFIG)


def load_registry(force: bool = False) -> Dict[str, DataProfile]:
    """
    加载 Profile 注册表。配置文件缺失时只保留 production，不报错。
    """
    global _registry
    if _registry is not None and not force:
        return _registry

    registry: Dict[str, DataProfile] = {"production": PRODUCTION}

    path = _config_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for name, cfg in (raw.get("profiles") or {}).items():
                if name == "production":
                    logger.warning("[Profile] 配置文件试图覆盖 production，已忽略")
                    continue
                cfg.pop("name", None)
                # schema 文件写的是相对路径时，按配置文件所在目录解析
                sf = cfg.get("sql_schema_file")
                if sf and not os.path.isabs(sf):
                    cfg["sql_schema_file"] = os.path.join(os.path.dirname(path), sf)
                registry[name] = DataProfile(name=name, **cfg)
            logger.info(f"[Profile] 已加载 {len(registry) - 1} 个 benchmark profile")
        except Exception as e:
            logger.error(f"[Profile] 配置文件解析失败，仅保留 production: {e}")
    else:
        logger.debug(f"[Profile] 未找到配置文件 {path}，仅有 production")

    _registry = registry
    return registry


def get_profile(name: str) -> DataProfile:
    """按名字取 Profile。不存在时退回 production 并告警。"""
    registry = load_registry()
    profile = registry.get(name)
    if profile is None:
        logger.warning(
            f"[Profile] 未找到 profile '{name}'，可用: {list(registry)}；退回 production"
        )
        return PRODUCTION
    return profile


def list_profiles() -> List[DataProfile]:
    return list(load_registry().values())


# ---------------------------------------------------------------------------
# 给业务代码用的解析函数（这三个是适配层的全部对外接口）
# ---------------------------------------------------------------------------

def resolve_kb_collection(default_name: str) -> str:
    """
    检索侧收口：Profile 指定了集合名就用它，否则用业务原本算出来的名字。

    被 retrieval_service.kb_collection_name() 调用。
    Milvus 集合和 ES 索引共用这个名字，所以一处切换两边都生效。
    """
    profile = active_profile()
    if profile.kb_collection:
        return profile.kb_collection
    return default_name


def resolve_sql_dsn(default_dsn: Optional[str]) -> Optional[str]:
    """Text2SQL 连接串收口。"""
    profile = active_profile()
    return profile.sql_dsn or default_dsn


def resolve_sql_schema(default_schema: str) -> str:
    """Text2SQL schema 描述收口。"""
    profile = active_profile()
    return profile.resolved_sql_schema() or default_schema
