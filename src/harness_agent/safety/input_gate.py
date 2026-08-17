"""输入闸门（M2 第一道）：个性化查询构造前的过敏硬规则拦截。

裁决语义：查询文本中检出"患者直接过敏或交叉反应"的药物即拦截，
``allowed=False`` 时调用方必须走澄清 / 升级，不得构造个性化查询
（fail-closed，绝不静默降级）。未检出过敏相关药物则放行——
包括查询提及了其他药物（患者不过敏）的情形。
"""

from __future__ import annotations

from harness_agent.contracts.gates import ClinicalQuery
from harness_agent.models.audit import GateVerdict
from harness_agent.models.session import SessionContext
from harness_agent.safety.normalization import DrugNormalizer
from harness_agent.safety.resolver import AllergyConflictResolver

__all__ = ["DrugSafetyInputGate"]


class DrugSafetyInputGate:
    """输入闸门：查询 + 患者过敏史 -> 裁决。"""

    name = "gate:input"

    def __init__(
        self,
        normalizer: DrugNormalizer,
        resolver: AllergyConflictResolver,
    ) -> None:
        self._normalizer = normalizer
        self._resolver = resolver

    def check(self, query: ClinicalQuery, context: SessionContext) -> GateVerdict:
        """扫描查询文本，命中阻断集合即拦截。"""
        blocked = self._resolver.blocked_drugs(context.patient_id)
        mentioned = {
            mention.normalized_name for mention in self._normalizer.find_mentions(query.text)
        }
        hits = sorted(mentioned & blocked)
        if hits:
            return GateVerdict(
                gate="input",
                allowed=False,
                reason=f"查询提及患者过敏/交叉反应药物: {', '.join(hits)}",
                blocked_drugs=hits,
            )
        if mentioned:
            return GateVerdict(
                gate="input",
                allowed=True,
                reason="输入闸门通过：提及药物均不在患者过敏阻断集合内",
            )
        return GateVerdict(gate="input", allowed=True, reason="输入闸门通过：未检出药物提及")
