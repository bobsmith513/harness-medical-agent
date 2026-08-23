"""编排层状态与产出类型（M4）。

**主 Agent 无应答权的类型级落地**：

- ``OrchestrationResult.conclusion`` 类型为 ``ClinicalConclusion | None``，
  但本模块（及 planner / agent）不存在任何构造 ``ClinicalConclusion``
  的路径——它只能透传推理专家的返回值（``ClinicalConclusion`` 的
  构造校验要求自检通过的推理链，主 Agent 编排角色无法伪造）；
- 路由失败唯一出口是 ``EscalationRequest``（转澄清/人工），
  没有"主 Agent 直接回答"的结果形态；
- ``OrchestrationResult`` 只能经 ``from_delegation`` 构造（专家产出
  作为参数），保证产出溯源到委派链。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from harness_agent.contracts.experts import ContextBundle, ExpertTask
from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import EvidencePack
from harness_agent.models.reasoning import ClinicalConclusion
from harness_agent.models.session import RouteRecord

__all__ = ["EscalationRequest", "OrchestrationResult", "TaskOutcome"]


class EscalationRequest(BaseModel):
    """路由失败的升级请求（转澄清 / 转人工）。

    ``clarification_question`` 为面向患者的澄清问句
    （追问意图以完成二值裁决）；``to_human`` 为 True 表示
    需要人工介入（两次 LLM 兜底均误判等硬失败场景）。
    """

    reason: str
    clarification_question: str = ""
    to_human: bool = False


class TaskOutcome(BaseModel):
    """单个委派任务的执行结果（审计与 trace 用）。"""

    task_id: str
    expert: str
    ok: bool = True
    detail: str = ""


class OrchestrationResult(BaseModel):
    """主 Agent 一轮编排的最终产出（纯编排视图，无应答权）。

    - ``conclusion``：透传自推理专家（唯一合法临床结论来源）；
    - ``context_bundle``：透传自记忆专家（no_reasoning 路径产出）；
    - ``escalation``：路由失败升级请求（三选一，其余为 None）。
    """

    session_id: str
    patient_id: str
    user_input: str
    route: RouteRecord
    tasks: list[ExpertTask] = Field(default_factory=list)
    evidence_pack: EvidencePack | None = None
    conclusion: ClinicalConclusion | None = None
    context_bundle: ContextBundle | None = None
    escalation: EscalationRequest | None = None
    gate_verdicts: list[GateVerdict] = Field(default_factory=list)

    @classmethod
    def from_delegation(
        cls,
        *,
        session_id: str,
        patient_id: str,
        user_input: str,
        route: RouteRecord,
        tasks: list[ExpertTask],
        evidence_pack: EvidencePack | None = None,
        conclusion: ClinicalConclusion | None = None,
        context_bundle: ContextBundle | None = None,
        escalation: EscalationRequest | None = None,
        gate_verdicts: list[GateVerdict] | None = None,
    ) -> OrchestrationResult:
        """唯一构造入口：产出必须溯源到委派链（专家产出或升级请求）。"""
        return cls(
            session_id=session_id,
            patient_id=patient_id,
            user_input=user_input,
            route=route,
            tasks=tasks,
            evidence_pack=evidence_pack,
            conclusion=conclusion,
            context_bundle=context_bundle,
            escalation=escalation,
            gate_verdicts=gate_verdicts or [],
        )
