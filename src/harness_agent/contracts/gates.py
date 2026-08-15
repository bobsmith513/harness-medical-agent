"""闸门契约（M1）。

五道闸门统一为一个泛型 ``Gate`` 接口（check 进、裁决出）：

供给三道（M2 实现）：
- input:     查询构造前拦截（过敏硬规则命中即改写/拒绝个性化查询）
- assembly:  证据包交付前复核（过滤含过敏药物的药物实体证据）
- output:    推理输出校验 API（供下游 Agent 输出前调用）

质量两道（M5 实现）：
- quality_judge: LLM-as-judge 忠实度（专查依据缺失与因果倒置）
- drug_safety:   药物安全 API 全量校验

fail-closed 统一语义：``allowed=False`` 即拦截，调用方必须转澄清/人工，
实现方不允许出现"拦截失败时放行"的代码路径。
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import EvidencePack
from harness_agent.models.reasoning import ClinicalConclusion
from harness_agent.models.session import SessionContext

__all__ = [
    "AssemblyGate",
    "ClinicalQuery",
    "Gate",
    "OutputGate",
    "QualityGate",
    "InputGate",
]

T = TypeVar("T")


@runtime_checkable
class ClinicalQuery(Protocol):
    """输入闸门的载荷形态：患者 + 查询文本。

    用 Protocol 而非具体模型，是为了兼容路由前的半结构化上下文
    （M2 实现时自然满足，无需显式实现）。
    """

    @property
    def patient_id(self) -> str: ...

    @property
    def text(self) -> str: ...


@runtime_checkable
class Gate(Protocol[T]):
    """闸门统一接口：``check`` 进、``GateVerdict`` 出。

    泛型参数 T 为载荷类型（查询 / 证据包 / 临床结论 / 会话）；
    实现方在 ``name`` 中声明自己的 GateName。
    """

    name: str

    def check(self, payload: T) -> GateVerdict: ...


@runtime_checkable
class InputGate(Protocol):
    """输入闸门：个性化查询构造前拦截（查询 + 患者过敏史 -> 裁决）。"""

    name: str

    def check(self, query: ClinicalQuery, context: SessionContext) -> GateVerdict: ...


@runtime_checkable
class AssemblyGate(Protocol):
    """装配闸门：证据包交付推理管线前复核。"""

    name: str

    def check(self, pack: EvidencePack) -> GateVerdict: ...


@runtime_checkable
class OutputGate(Protocol):
    """输出闸门：临床结论输出前的药物安全校验 API。

    药物安全裁决依赖患者过敏史，故签名携带会话上下文
    （M2 实现时对 M1 契约做的唯一细化）。
    """

    name: str

    def check(self, conclusion: ClinicalConclusion, context: SessionContext) -> GateVerdict: ...


@runtime_checkable
class QualityGate(Protocol):
    """质量门禁：LLM-as-judge 忠实度校验（引用与证据一致性）。"""

    name: str
    #: 忠实度阈值（0-1），低于阈值即拦截转人工
    threshold: float

    def evaluate(self, conclusion: ClinicalConclusion, evidence: EvidencePack) -> GateVerdict: ...
