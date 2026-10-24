# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""DeepResearch Agents 模块。

唯一主链路使用 Architect / Scout / DataAnalyst / Writer / Critic 五个 Agent。
CodeWizard 仍作为独立的沙箱绘图能力保留在包内，但已不在主链路的分析节点上；
大纲审计员随"规划节点只调用 Architect"一并下线。
"""

from .base import BaseAgent, AgentRegistry
from .architect import ChiefArchitect
from .scout import DeepScout
from .wizard import CodeWizard
from .critic import CriticMaster
from .writer import LeadWriter
from .data_analyst import DataAnalyst

__all__ = [
    'BaseAgent',
    'AgentRegistry',
    'ChiefArchitect',
    'DeepScout',
    'CodeWizard',
    'CriticMaster',
    'LeadWriter',
    'DataAnalyst',
]
