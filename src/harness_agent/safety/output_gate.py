"""输出闸门（M2 第三道）：临床结论输出前的药物安全复核。

扫描范围是**结论陈述 + 推理链全文**（任何一步提及过敏相关药物都算
命中）：推理步里"建议使用阿莫西林"而结论陈述避而不谈，同样拦截。
``allowed=False`` 时调用方必须 interrupt（转人工或让模型重写），
绝不截掉违规片段后放行——截断会破坏结论与推理链的一致性。
"""

from __future__ import annotations

from harness_agent.models.audit import GateVerdict
from harness_agent.models.reasoning import ClinicalConclusion
from harness_agent.models.session import SessionContext
from harness_agent.safety.normalization import DrugNormalizer
from harness_agent.safety.resolver import AllergyConflictResolver

__all__ = ["DrugSafetyOutputGate"]


class DrugSafetyOutputGate:
    """输出闸门：临床结论 + 患者过敏史 -> 裁决。"""

    name = "gate:output"

    def __init__(
        self,
        normalizer: DrugNormalizer,
        resolver: AllergyConflictResolver,
    ) -> None:
        self._normalizer = normalizer
        self._resolver = resolver

    def check(self, conclusion: ClinicalConclusion, context: SessionContext) -> GateVerdict:
        """结论与推理链全文扫描，命中阻断集合即拦截。"""
        blocked = self._resolver.blocked_drugs(context.patient_id)
        texts = [conclusion.statement]
        texts.extend(step.text for step in conclusion.reasoning_chain.steps)
        mentioned: set[str] = set()
        for text in texts:
            mentioned.update(
                mention.normalized_name for mention in self._normalizer.find_mentions(text)
            )
        hits = sorted(mentioned & blocked)
        if hits:
            return GateVerdict(
                gate="output",
                allowed=False,
                reason=f"临床结论/推理链提及患者过敏/交叉反应药物: {', '.join(hits)}",
                blocked_drugs=hits,
            )
        return GateVerdict(gate="output", allowed=True, reason="输出校验通过：未涉及过敏药物")
