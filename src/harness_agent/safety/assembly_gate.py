"""装配闸门（M2 第二道）：证据包交付推理管线前的复核与过滤。

两段式语义：

- ``check``：纯裁决——证据包含过敏药物实体即 ``allowed=False``
  （意味着该包**不得**以当前形态进入推理管线）；
- ``apply``：过滤并放行——移除含过敏药物实体的证据，附加复核通过
  裁决后交付（M3 检索服务调用）。过滤后证据为空时 fail-closed
  拒绝交付：宁可让推理链因无证据而中止，也不交付空证据包诱发
  无依据结论。
"""

from __future__ import annotations

from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import Evidence, EvidencePack
from harness_agent.safety.normalization import DrugNormalizer

__all__ = ["DrugSafetyAssemblyGate"]


class DrugSafetyAssemblyGate:
    """装配闸门：证据包 -> 裁决（check）/ 过滤后的证据包（apply）。"""

    name = "gate:assembly"

    def __init__(self, normalizer: DrugNormalizer) -> None:
        self._normalizer = normalizer

    def _partition(self, pack: EvidencePack) -> tuple[list[Evidence], list[Evidence], list[str]]:
        """按阻断集合切分证据：(含过敏实体的, 干净的, 命中的药名)。"""
        blocked = set(pack.blocked_drugs)
        offending: list[Evidence] = []
        kept: list[Evidence] = []
        drugs: set[str] = set()
        for evidence in pack.evidence:
            hits = {
                mention.normalized_name
                for mention in self._normalizer.find_mentions(evidence.content)
            } & blocked
            if hits:
                offending.append(evidence)
                drugs.update(hits)
            else:
                kept.append(evidence)
        return offending, kept, sorted(drugs)

    def check(self, pack: EvidencePack) -> GateVerdict:
        """纯裁决：含过敏药物实体的证据包不得原样交付。"""
        offending, _, drugs = self._partition(pack)
        if offending:
            return GateVerdict(
                gate="assembly",
                allowed=False,
                reason=f"{len(offending)} 条证据含过敏药物实体，须过滤后再交付",
                blocked_drugs=drugs,
            )
        return GateVerdict(gate="assembly", allowed=True, reason="装配复核通过")

    def apply(self, pack: EvidencePack) -> EvidencePack:
        """过滤含过敏药物实体的证据并附加裁决（返回新证据包）。"""
        offending, kept, drugs = self._partition(pack)
        if offending and not kept:
            verdict = GateVerdict(
                gate="assembly",
                allowed=False,
                reason=(
                    f"全部 {len(offending)} 条证据均含过敏药物实体，拒绝交付空证据包（fail-closed）"
                ),
                blocked_drugs=drugs,
            )
        elif offending:
            verdict = GateVerdict(
                gate="assembly",
                allowed=True,
                reason=f"已过滤 {len(offending)} 条含过敏药物实体证据: {', '.join(drugs)}",
                blocked_drugs=drugs,
            )
        else:
            verdict = GateVerdict(gate="assembly", allowed=True, reason="装配复核通过：无需过滤")
        return pack.model_copy(update={"evidence": kept, "assembly_gate": verdict})
