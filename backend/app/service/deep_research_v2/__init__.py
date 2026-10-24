# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""DeepResearch：单一 LangGraph 主链路的多智能体研究。

只有一条执行链路：规划（Architect）→ 搜索 → 分析 → 写作 → 审核，
审核结果决定补搜、修订还是收尾。没有医疗专属编排，也没有并行保留的
旧通用执行引擎；本链路不执行自动引用核验。

使用方式：

```python
from service.deep_research_v2 import DeepResearchV2Service

service = DeepResearchV2Service()
async for chunk in service.research(query="...", session_id="...", user_id="..."):
    print(chunk)
```
"""

from .state import (
    GRAPH_VERSION,
    ResearchState,
    ResearchPhase,
    Section,
    Fact,
    DataPoint,
    Chart,
    CriticFeedback,
    citation_verification_status,
    create_initial_state,
)

from .graph import GraphState, build_graph, create_agents, pack_state

from .agents import (
    ChiefArchitect,
    DeepScout,
    CriticMaster,
    LeadWriter,
    DataAnalyst,
)

from .service import DeepResearchV2Service

__all__ = [
    'DeepResearchV2Service',
    # State
    'GRAPH_VERSION',
    'ResearchState',
    'ResearchPhase',
    'Section',
    'Fact',
    'DataPoint',
    'Chart',
    'CriticFeedback',
    'citation_verification_status',
    'create_initial_state',

    # Graph
    'GraphState',
    'build_graph',
    'create_agents',
    'pack_state',

    # Agents
    'ChiefArchitect',
    'DeepScout',
    'CriticMaster',
    'LeadWriter',
    'DataAnalyst',
]
