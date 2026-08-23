"""编排层（M4）：主 Agent、路由器、任务规划、专家配置。

组件清单：

- ``router``:          二值路由（规则前置 + LLM 兜底 + 误判二次路由）
- ``experts_config``:  YAML 专家声明式配置加载器
- ``planner``:         任务清单规划器（路由裁决 → 委派序列）
- ``agent``:           HarnessOrchestrator（langgraph StateGraph 编排）
- ``state``:            编排产出类型（无应答权类型级落地）
- ``wiring``:          配置 → 主 Agent 装配工厂
"""

from harness_agent.orchestrator.agent import HarnessOrchestrator, OrchestrationState
from harness_agent.orchestrator.experts_config import (
    DEFAULT_EXPERTS_CONFIG,
    ExpertKind,
    ExpertRegistry,
    ExpertSpec,
    load_experts,
)
from harness_agent.orchestrator.planner import TaskPlanner
from harness_agent.orchestrator.router import (
    BinaryRouter,
    LLMRouter,
    RouteRuleSet,
    RuleRouter,
    default_rule_set,
)
from harness_agent.orchestrator.state import EscalationRequest, OrchestrationResult, TaskOutcome
from harness_agent.orchestrator.wiring import build_orchestrator

__all__ = [
    "BinaryRouter",
    "DEFAULT_EXPERTS_CONFIG",
    "EscalationRequest",
    "ExpertKind",
    "ExpertRegistry",
    "ExpertSpec",
    "HarnessOrchestrator",
    "LLMRouter",
    "OrchestrationResult",
    "OrchestrationState",
    "RouteRuleSet",
    "RuleRouter",
    "TaskOutcome",
    "TaskPlanner",
    "build_orchestrator",
    "default_rule_set",
    "load_experts",
]
