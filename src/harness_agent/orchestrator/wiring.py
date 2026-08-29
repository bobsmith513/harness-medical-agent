"""编排层装配工厂（M4）：配置 → 主 Agent。

零依赖默认：Mock 路由 LLM（脚本化应答）+ 内置规则表 + 仓库根
``configs/experts.yaml`` + M3 检索栈，无外部端点即可跑通委派链。

专家实现由调用方注入（``experts`` 参数）：M4 demo / 测试注入桩实现，
M5 接入真实推理专家后同一工厂零改动（接口注入，业务逻辑不分叉）。
"""

from __future__ import annotations

from typing import Any

from harness_agent.config.settings import Settings, get_settings
from harness_agent.contracts.retrieval import RetrievalService
from harness_agent.gates.pipeline import GatePipeline
from harness_agent.gates.quality_judge import LLMJudgeGate
from harness_agent.llm.mock import MockLLMClient
from harness_agent.orchestrator.agent import HarnessOrchestrator
from harness_agent.orchestrator.experts_config import (
    DEFAULT_EXPERTS_CONFIG,
    ExpertRegistry,
    load_experts,
)
from harness_agent.orchestrator.planner import TaskPlanner
from harness_agent.orchestrator.router import BinaryRouter, LLMRouter, RuleRouter
from harness_agent.retrieval.wiring import build_retrieval_stack
from harness_agent.safety import SafetyStack, build_safety_stack

__all__ = ["build_orchestrator"]


def build_orchestrator(
    experts: dict[str, Any],
    *,
    settings: Settings | None = None,
    router_llm: Any | None = None,
    judge_llm: Any | None = None,
    experts_config_path: str | None = None,
    retrieval: RetrievalService | None = None,
    gate_pipeline: GatePipeline | None = None,
    safety: SafetyStack | None = None,
) -> HarnessOrchestrator:
    """装配主 Agent（路由 + 规划 + 检索 + 专家绑定 + 门禁）。

    参数：
        experts:             name → 专家实现（M4 桩 / M5 真实实现）
        router_llm:          路由兜底 LLM（默认 Mock）
        judge_llm:           质量门禁 judge LLM（默认 Mock）
        experts_config_path: 专家 YAML 路径（默认仓库根 configs/experts.yaml）
        retrieval:           检索供给门面（默认新建 M3 栈；可注入共享实例）
        gate_pipeline:       门禁流水线（默认自动装配质量门禁 + 输出闸门；
                             传 None 跳过门禁——仅 M4 demo 场景）
        safety:              安全栈（None 时新建）。**共享注入点**——应与
                             检索层传入的是同一实例，否则输出闸门与检索
                             闸门各持一份过敏史，"两端阻断口径一致"不成立。
    """
    if settings is None:
        settings = get_settings()

    # 路由器：规则前置 + LLM 兜底（Mock 默认，可插拔）
    router_llm = router_llm if router_llm is not None else MockLLMClient(role="router")
    router = BinaryRouter(
        rule_router=RuleRouter(),
        llm_router=LLMRouter(client=router_llm),
    )

    # 专家注册表（声明式 YAML）
    registry: ExpertRegistry = load_experts(
        experts_config_path or settings.orchestrator.experts_config_path or DEFAULT_EXPERTS_CONFIG
    )

    # 检索供给（M3 门面，含三道闸门；可注入共享实例）
    retrieval = retrieval if retrieval is not None else build_retrieval_stack(settings).service

    # 门禁流水线（M5：质量门禁 + 输出闸门；默认自动装配）
    if gate_pipeline is None:
        # 与检索层共用同一份安全栈：两端阻断口径必须一致
        safety = safety if safety is not None else build_safety_stack(settings)
        judge_llm = judge_llm if judge_llm is not None else MockLLMClient(role="judge")
        gate_pipeline = GatePipeline(
            quality_gate=LLMJudgeGate(llm=judge_llm),
            output_gate=safety.output_gate,
        )

    return HarnessOrchestrator(
        router=router,
        planner=TaskPlanner(registry),
        retrieval=retrieval,
        registry=registry,
        experts=experts,
        gate_pipeline=gate_pipeline,
    )
