"""推理链与临床结论模型（M1）。

"主 Agent 纯编排无应答权"的类型级落地——核心约束全部固化在
``ClinicalConclusion`` 校验器中：

1. ``reasoning_chain`` 必填：结论不可能脱离推理链凭空构造；
2. 推理链必须 **自检通过**（``self_check_passed=True``）才能产出结论；
3. 结论引用的证据必须 ⊆ 推理链引用集合（结论与依据不可分离）。

推理链固定"证据引用 -> 逐步推断 -> 结论"结构（首步证据、末步结论、
必有推断步），专治"结论正确但依据缺失、因果倒置"。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from harness_agent.models.common import new_id, now_utc

__all__ = ["ReasoningChain", "ReasoningStep", "ClinicalConclusion", "StepKind"]

#: 推理步类型：证据引用 -> 逐步推断 -> 结论
StepKind = Literal["evidence", "inference", "conclusion"]

_KIND_ORDER = ("evidence", "inference", "conclusion")


class ReasoningStep(BaseModel):
    """单步推理：类型 + 文本 + 证据引用（evidence_id 列表）。"""

    step_id: str = Field(default_factory=lambda: new_id("step"))
    kind: StepKind
    text: str
    #: 引用的 evidence_id（evidence 步必须至少一条；inference/conclusion 可空）
    citations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _evidence_step_must_cite(self) -> ReasoningStep:
        """证据引用步必须真正引用证据，否则依据充分性无从校验。"""
        if self.kind == "evidence" and not self.citations:
            raise ValueError("evidence 步必须至少引用一条 evidence_id")
        return self


class ReasoningChain(BaseModel):
    """推理链：证据引用 -> 逐步推断 -> 结论 的三段式结构。"""

    chain_id: str = Field(default_factory=lambda: new_id("chain"))
    steps: list[ReasoningStep] = Field(min_length=1)
    #: 推理专家自检结果：结构完整、引用真实、因果正向
    self_check_passed: bool = False
    self_check_notes: str = ""

    @property
    def cited_evidence_ids(self) -> list[str]:
        """整条链引用过的全部 evidence_id（去重保序）。"""
        seen: list[str] = []
        for step in self.steps:
            for citation in step.citations:
                if citation not in seen:
                    seen.append(citation)
        return seen

    @model_validator(mode="after")
    def _chain_shape_must_be_valid(self) -> ReasoningChain:
        kinds = [s.kind for s in self.steps]
        if "evidence" not in kinds:
            raise ValueError("推理链必须包含证据引用步（依据充分性）")
        if "inference" not in kinds:
            raise ValueError("推理链必须包含推断步（逐步推断）")
        if kinds[-1] != "conclusion":
            raise ValueError("推理链末步必须是结论步")
        if kinds[0] != "evidence":
            raise ValueError("推理链首步必须是证据引用步")
        if kinds != sorted(kinds, key=_KIND_ORDER.index):
            raise ValueError("推理步顺序必须满足 证据引用 -> 逐步推断 -> 结论")
        return self


class ClinicalConclusion(BaseModel):
    """临床结论：全系统唯一合法的临床输出形态。

    类型层面锁死"带核查的推理管线"：
    - 只有推理专家（SFT+DPO 对齐基座）能构造：必须携带自检通过的推理链；
    - 主 Agent（编排角色）拿到的只是本对象的引用，无法绕过校验器构造。
    """

    conclusion_id: str = Field(default_factory=lambda: new_id("cc"))
    statement: str
    reasoning_chain: ReasoningChain
    #: 结论层显式引用的证据（必须出现在推理链引用集合内）
    cited_evidence_ids: list[str] = Field(default_factory=list)
    #: 产出者标识（审计用；正常流程恒为 reasoning_expert）
    produced_by: str = "reasoning_expert"
    created_at: datetime = Field(default_factory=now_utc)

    @model_validator(mode="after")
    def _must_come_from_self_checked_chain(self) -> ClinicalConclusion:
        if not self.reasoning_chain.self_check_passed:
            raise ValueError("临床结论必须出自自检通过的推理链（主 Agent 纯编排无应答权）")
        chain_cited = set(self.reasoning_chain.cited_evidence_ids)
        extra = set(self.cited_evidence_ids) - chain_cited
        if extra:
            raise ValueError(f"结论引用了推理链未引用的证据: {sorted(extra)}")
        return self
