"""患者过敏史供给（M2 硬规则数据源）。

``AllergyStore`` 是供给接口：demo / 测试用内存实现内置合成种子数据；
生产环境对接 HIS / EMR 过敏史接口（连接与协议由部署方实现，此处留空，
实现本接口并注入三道闸门即可，逻辑不分叉）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, runtime_checkable

from harness_agent.models.memory import AllergyRecord
from harness_agent.safety.atc import ATCService
from harness_agent.safety.normalization import DrugNormalizer

__all__ = [
    "AllergyStore",
    "InMemoryAllergyStore",
    "SEED_ALLERGIES",
    "build_allergy_record",
]


@runtime_checkable
class AllergyStore(Protocol):
    """患者过敏史供给接口（硬规则的唯一数据来源）。"""

    def get(self, patient_id: str) -> list[AllergyRecord]: ...


def build_allergy_record(
    patient_id: str,
    raw_drug_name: str,
    normalizer: DrugNormalizer,
    atc: ATCService,
) -> AllergyRecord:
    """由原始过敏记录构建硬规则载体（归一化 + ATC 交叉反应全集）。

    未知药名直接抛错（fail-closed）：词典缺词条的过敏记录无法参与
    精确匹配，静默丢弃等于把风险放行——必须在录入时暴露。
    """
    normalized = normalizer.normalize(raw_drug_name)
    if normalized is None:
        raise ValueError(f"无法归一化的过敏药物（词典缺词条）: {raw_drug_name!r}")
    return AllergyRecord(
        patient_id=patient_id,
        drug_name_raw=raw_drug_name,
        normalized_drug=normalized,
        atc_code=atc.atc_code(normalized) or "",
        cross_reactants=atc.cross_reactants(normalized),
    )


#: 合成种子过敏史（每类 1 条代表，均非真实患者）：
#: - pat-001 青霉素过敏 -> 阻断 beta_lactam 全组（含头孢类）
#: - pat-002 阿司匹林过敏 -> 阻断 nsaid 全组
#: - pat-003 库中无记录（无已知过敏）
SEED_ALLERGIES: list[tuple[str, str]] = [
    ("pat-001", "盘尼西林"),
    ("pat-002", "拜阿司匹灵"),
]


class InMemoryAllergyStore:
    """内存过敏史存储（demo / 测试实现）。"""

    def __init__(self, records: list[AllergyRecord] | None = None) -> None:
        self._by_patient: dict[str, list[AllergyRecord]] = defaultdict(list)
        for record in records or []:
            self.add(record)

    def add(self, record: AllergyRecord) -> None:
        """追加一条过敏记录（按患者分组）。"""
        self._by_patient[record.patient_id].append(record)

    def get(self, patient_id: str) -> list[AllergyRecord]:
        """取患者全部过敏记录（副本；无记录返回空列表）。"""
        return list(self._by_patient.get(patient_id, []))

    @classmethod
    def with_seed_data(cls, normalizer: DrugNormalizer, atc: ATCService) -> InMemoryAllergyStore:
        """以合成种子过敏史构建（见 ``SEED_ALLERGIES``）。"""
        records = [
            build_allergy_record(patient_id, raw, normalizer, atc)
            for patient_id, raw in SEED_ALLERGIES
        ]
        return cls(records)
