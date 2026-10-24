# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
LLM 和 Agent 配置文件

集中管理所有 LLM 相关配置，包括：
- API 配置（密钥、基础 URL）
- 每个 Agent 节点的模型配置
- 研究流程参数

使用方式:
    from app.config.llm_config import LLMConfig, AgentConfig

    config = LLMConfig()
    print(config.default_model)
    print(config.agents.wizard.model)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def _primary_model() -> str:
    """复杂 Agent 共用的主模型，允许部署环境统一切换。"""
    return os.getenv("OPENAI_MODEL", "qwen3-max").strip() or "qwen3-max"


def _agent_model(agent_name: str, default: str) -> str:
    """Agent 专属配置优先；未配置时沿用给定的分层默认值。"""
    key = f"LLM_MODEL_{agent_name.upper()}"
    return os.getenv(key, default).strip() or default


def get_summary_model() -> str:
    """历史摘要单独选模，不随 Scout 的检索模型或全局备用模型切换。"""
    return _agent_model("summary", "qwen3-max")


@dataclass
class AgentModelConfig:
    """单个 Agent 的模型配置"""
    model: str
    temperature: float = 0.7
    max_tokens: int = 8000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


@dataclass
class AgentsConfig:
    """所有 Agent 的配置"""
    # 规划师 - 分析问题，生成研究大纲
    architect: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(
        model=_agent_model("architect", _primary_model()),
        temperature=0.3,
        # 大纲只有 4~6 章；实测 5000 会触发 DashScope 长连接断开，
        # 3000 可完整返回约 1700 输出 tokens 的同一份规划。
        max_tokens=3000
    ))

    # 侦察员 - 深度搜索（使用较快的模型）
    scout: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(
        model=_agent_model("scout", "qwen-plus"),  # 搜索阶段用快速模型
        temperature=0.5,
        max_tokens=3000
    ))

    # 数据分析师 - 数据提取和分析
    data_analyst: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(
        model=_agent_model("data_analyst", _primary_model()),
        temperature=0.3,
        max_tokens=3000
    ))

    # 代码极客 - 代码生成和图表绘制。
    # 不在唯一主链路的分析节点上；保留配置供单独的沙箱绘图能力使用。
    wizard: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(
        model=_agent_model("wizard", _primary_model()),
        temperature=0.3,
        max_tokens=8000
    ))

    # 审核大师 - 对抗式审核
    critic: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(
        model=_agent_model("critic", _primary_model()),
        temperature=0.5,
        max_tokens=3000
    ))

    # 首席写手 - 报告撰写
    writer: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(
        model=_agent_model("writer", _primary_model()),
        temperature=0.7,
        # 单章 JSON 还包含 claims 与原文引句，预留空间避免末尾截断。
        max_tokens=60000
    ))


@dataclass
class ResearchConfig:
    """研究流程配置"""
    # 业务返工上限：审核不通过时最多再走几轮补搜/修订。
    # 与图的 recursion_limit 是两件事：改这里必须重新核算 graph.RECURSION_LIMIT。
    max_iterations: int = field(default_factory=lambda: max(
        0, min(5, int(os.getenv("RESEARCH_MAX_ITERATIONS", "2")))
    ))

    # 单篇报告的联网搜索硬预算；初始检索、换源、深搜和审核补搜共享。
    max_total_searches: int = field(default_factory=lambda: max(
        1, min(20, int(os.getenv("RESEARCH_MAX_TOTAL_SEARCHES", "20")))
    ))

    # 每个章节最多两个初始查询；换源与来源追溯使用各自已有规则。
    max_searches_per_section: int = 2

    # 最大图表数量
    max_charts: int = 5

    # 是否启用代码执行
    enable_code_execution: bool = True

    # 质量评分阈值（1-10分制，低于此分数需要修订）
    quality_threshold: float = 6.0

    # 报告篇幅配置：统一供 Architect 和 Writer 使用，避免散落在 Prompt 中。
    min_sections: int = 4
    max_sections: int = 5
    complex_max_sections: int = 6
    target_report_chars: int = 7500
    min_section_chars: int = 300
    max_section_chars: int = 500
    summary_chars: int = 250
    outlook_chars: int = 150


@dataclass
class LLMConfig:
    """
    LLM 配置主类

    集中管理所有配置，支持从环境变量读取
    """
    # API 配置
    api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ))

    # 搜索 API
    search_api_key: str = field(default_factory=lambda: os.getenv("BOCHA_API_KEY", ""))
    search_provider: str = field(default_factory=lambda: os.getenv("RESEARCH_SEARCH_PROVIDER", "bocha"))
    aliyun_search_mcp_api_key: str = field(default_factory=lambda: (
        os.getenv("ALIYUN_SEARCH_MCP_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    ), repr=False)
    aliyun_search_mcp_url: str = field(default_factory=lambda: os.getenv(
        "ALIYUN_SEARCH_MCP_URL", "https://dashscope.aliyuncs.com/api/v1/mcps/EnhancedSearch/mcp"
    ))
    aliyun_search_mcp_tool_name: str = field(default_factory=lambda: os.getenv(
        "ALIYUN_SEARCH_MCP_TOOL_NAME", "search_pro"
    ))

    # 默认模型（用于未单独配置的场景）
    default_model: str = "deepseek-v3.2"

    # Agent 配置
    agents: AgentsConfig = field(default_factory=AgentsConfig)

    # 研究流程配置
    research: ResearchConfig = field(default_factory=ResearchConfig)

    def get_agent_config(self, agent_name: str) -> AgentModelConfig:
        """获取指定 Agent 的配置"""
        agent_configs = {
            "architect": self.agents.architect,
            "scout": self.agents.scout,
            "data_analyst": self.agents.data_analyst,
            "wizard": self.agents.wizard,
            "critic": self.agents.critic,
            "writer": self.agents.writer,
        }
        return agent_configs.get(agent_name, AgentModelConfig(model=self.default_model))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "api_key": self.api_key[:8] + "..." if self.api_key else "",
            "base_url": self.base_url,
            "search_api_key": self.search_api_key[:8] + "..." if self.search_api_key else "",
            "search_provider": self.search_provider,
            "aliyun_search_mcp_url": self.aliyun_search_mcp_url,
            "aliyun_search_mcp_tool_name": self.aliyun_search_mcp_tool_name,
            "aliyun_search_mcp_key_configured": bool(self.aliyun_search_mcp_api_key),
            "default_model": self.default_model,
            "agents": {
                "architect": self.agents.architect.to_dict(),
                "scout": self.agents.scout.to_dict(),
                "data_analyst": self.agents.data_analyst.to_dict(),
                "wizard": self.agents.wizard.to_dict(),
                "critic": self.agents.critic.to_dict(),
                "writer": self.agents.writer.to_dict(),
            },
            "research": {
                "max_iterations": self.research.max_iterations,
                "max_total_searches": self.research.max_total_searches,
                "max_searches_per_section": self.research.max_searches_per_section,
                "max_charts": self.research.max_charts,
                "enable_code_execution": self.research.enable_code_execution,
                "quality_threshold": self.research.quality_threshold,
                "min_sections": self.research.min_sections,
                "max_sections": self.research.max_sections,
                "complex_max_sections": self.research.complex_max_sections,
                "target_report_chars": self.research.target_report_chars,
                "min_section_chars": self.research.min_section_chars,
                "max_section_chars": self.research.max_section_chars,
                "summary_chars": self.research.summary_chars,
                "outlook_chars": self.research.outlook_chars,
            }
        }


# 全局配置实例（单例模式）
_config_instance: Optional[LLMConfig] = None


def get_config() -> LLMConfig:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = LLMConfig()
    return _config_instance


def reload_config() -> LLMConfig:
    """重新加载配置"""
    global _config_instance
    _config_instance = LLMConfig()
    return _config_instance


# 便捷访问
def get_agent_model(agent_name: str) -> str:
    """快速获取指定 Agent 的模型名称"""
    return get_config().get_agent_config(agent_name).model


def get_default_model() -> str:
    """快速获取默认模型"""
    return get_config().default_model


# 用于打印配置信息
def print_config():
    """打印当前配置（用于调试）"""
    import json
    config = get_config()
    logger.info("=" * 60)
    logger.info("LLM Configuration")
    logger.info("=" * 60)
    logger.info(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    logger.info("=" * 60)


if __name__ == "__main__":
    # 测试配置
    print_config()
