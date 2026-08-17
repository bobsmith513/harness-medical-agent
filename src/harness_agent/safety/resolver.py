"""过敏冲突解析器（M2）。

患者过敏记录 -> 被阻断药物全集：直接过敏药 + 记录携带的交叉反应 +
ATC 交叉反应组的实时扩展（词典升级后自动生效）。
输入闸门与输出闸门共用本解析器，保证两端阻断口径一致。
"""

from __future__ import annotations

from harness_agent.safety.allergy_store import AllergyStore
from harness_agent.safety.atc import ATCService

__all__ = ["AllergyConflictResolver"]


class AllergyConflictResolver:
    """过敏冲突解析：patient_id -> 阻断药物全集（frozenset）。"""

    def __init__(self, atc: ATCService, allergy_store: AllergyStore) -> None:
        self._atc = atc
        self._allergy_store = allergy_store

    def blocked_drugs(self, patient_id: str) -> frozenset[str]:
        """患者被阻断的归一化药名全集（直接过敏 + 全部交叉反应）。"""
        blocked: set[str] = set()
        for record in self._allergy_store.get(patient_id):
            blocked.add(record.normalized_drug)
            blocked.update(record.cross_reactants)
            blocked.update(self._atc.cross_reactants(record.normalized_drug))
        return frozenset(blocked)
