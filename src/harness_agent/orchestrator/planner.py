"""任务清单规划器（M4）：路由裁决 → 委派任务序列。

规划规则（声明式专家注册表驱动）：

- ``need_reasoning``：检索供给（M3 门面）→ 推理专家（唯一合法
  临床结论产出方）；
- ``no_reasoning``：记忆专家装配上下文（复诊免重复问询主路径）；
- ``escalate``：无委派任务，直接产出升级请求。

任务清单是主 Agent 的"计划"外显：审计可回放整轮编排意图。
"""

from __future__ import annotations

from harness_agent.contracts.experts import ExpertTask
from harness_agent.models.session import RouteDecision, RouteRecord, SessionContext
from harness_agent.orchestrator.experts_config import ExpertRegistry

__all__ = ["TaskPlanner"]

#: 各专家在任务清单中的指令占位符填充来源键
_QUESTION_KEY = "question"


class TaskPlanner:
    """按路由裁决从专家注册表规划委派任务清单。"""

    def __init__(self, registry: ExpertRegistry) -> None:
        self._registry = registry

    def plan(
        self,
        user_input: str,
        route: RouteRecord,
        context: SessionContext,
    ) -> list[ExpertTask]:
        """路由裁决 → 任务清单（escalate 时空清单）。"""
        if route.decision == "escalate":
            return []

        experts = self._select_experts(route.decision)
        tasks: list[ExpertTask] = []
        for spec in experts:
            instruction = spec.instruction.format(question=user_input, utterance=user_input)
            tasks.append(
                ExpertTask(
                    expert=spec.name,
                    instruction=instruction,
                    inputs={_QUESTION_KEY: user_input},
                )
            )
        return tasks

    def _select_experts(self, decision: RouteDecision):
        """裁决 → 专家序列（顺序即执行序）。"""
        if decision == "need_reasoning":
            # 检索先行（M3 供给层门面，agent 节点内调用），推理专家殿后
            reasoning = self._registry.by_kind("reasoning")
            if not reasoning:
                raise KeyError("注册表中无 reasoning 专家（临床结论唯一来源缺失）")
            return reasoning
        if decision == "no_reasoning":
            memory = self._registry.by_kind("memory")
            if not memory:
                raise KeyError("注册表中无 memory 专家（上下文装配缺失）")
            return memory
        return []
