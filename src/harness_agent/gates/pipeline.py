"""门禁流水线（M5）：推理结论 → 质量门禁 → 输出闸门 → 放行/拦截。

流水线顺序（fail-closed 串联，任一拦截即终态）::

    临床结论
      → quality_judge（LLM-as-judge：引用一致性 + 因果倒置 + 忠实度阈值）
      → drug_safety（M2 输出闸门：结论+推理链全文扫描过敏药物）
      → 放行（GatePipelineResult.allowed=True）
      └─ 任一拦截 → GatePipelineResult.allowed=False（调用方 interrupt 转人工）

两道门禁不可交换顺序：质量门禁先做（引用/因果问题直接拦截，
不必触发药物安全扫描浪费开销）；但即使质量通过，药物安全
仍必须执行——它是硬规则，不因质量门禁通过而旁路。
"""

from __future__ import annotations

from dataclasses import dataclass

from harness_agent.contracts.gates import OutputGate, QualityGate
from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import EvidencePack
from harness_agent.models.reasoning import ClinicalConclusion
from harness_agent.models.session import SessionContext

__all__ = ["GatePipeline", "GatePipelineResult"]


@dataclass(frozen=True)
class GatePipelineResult:
    """门禁流水线终态：放行或拦截（含被拦门禁名与裁决）。"""

    allowed: bool
    verdicts: list[GateVerdict]  # 按执行顺序，拦截时最后一个为拦截裁决
    blocking_gate: str = ""  # 被拦门禁名（放行时空）

    @property
    def final_verdict(self) -> GateVerdict | None:
        """最终裁决（拦截时为拦截裁决，放行时为最后通过的裁决）。"""
        return self.verdicts[-1] if self.verdicts else None


class GatePipeline:
    """质量门禁 → 输出闸门 串联流水线。

    M4 主 Agent 的 ``reason`` 节点产出结论后调用本流水线；
    ``allowed=False`` 时主 Agent interrupt（转人工 / 让模型重写），
    绝不截断违规片段后放行。
    """

    def __init__(self, quality_gate: QualityGate, output_gate: OutputGate) -> None:
        self._quality_gate = quality_gate
        self._output_gate = output_gate

    def run(
        self,
        conclusion: ClinicalConclusion,
        evidence: EvidencePack,
        context: SessionContext,
    ) -> GatePipelineResult:
        """执行完整门禁流水线（fail-closed 串联）。"""
        verdicts: list[GateVerdict] = []

        # 1. 质量门禁（LLM-judge：引用一致性 + 因果倒置 + 忠实度）
        quality_verdict = self._quality_gate.evaluate(conclusion, evidence)
        verdicts.append(quality_verdict)
        if not quality_verdict.allowed:
            return GatePipelineResult(
                allowed=False,
                verdicts=verdicts,
                blocking_gate=self._quality_gate.name,
            )

        # 2. 输出闸门（M2 药物安全：结论+推理链全文扫描）
        output_verdict = self._output_gate.check(conclusion, context)
        verdicts.append(output_verdict)
        if not output_verdict.allowed:
            return GatePipelineResult(
                allowed=False,
                verdicts=verdicts,
                blocking_gate=self._output_gate.name,
            )

        return GatePipelineResult(allowed=True, verdicts=verdicts)
