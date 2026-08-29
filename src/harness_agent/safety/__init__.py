"""硬规则安全层（M2）：药名归一化、ATC 交叉反应、三道供给闸门。

硬规则不向量化：过敏史走归一化药名 + ATC 交叉反应组精确匹配，
从输入拦截、装配复核到输出校验三道闸门全链路 fail-closed。
"""

from __future__ import annotations

from dataclasses import dataclass

from harness_agent.config.settings import Settings
from harness_agent.safety.allergy_store import (
    SEED_ALLERGIES,
    AllergyStore,
    InMemoryAllergyStore,
    build_allergy_record,
)
from harness_agent.safety.assembly_gate import DrugSafetyAssemblyGate
from harness_agent.safety.atc import ATCService
from harness_agent.safety.dictionary import SEED_DRUG_DICTIONARY, DrugDictionary, DrugEntry
from harness_agent.safety.input_gate import DrugSafetyInputGate
from harness_agent.safety.normalization import DrugMention, DrugNormalizer, fold_text
from harness_agent.safety.output_gate import DrugSafetyOutputGate
from harness_agent.safety.resolver import AllergyConflictResolver

__all__ = [
    "AllergyConflictResolver",
    "AllergyStore",
    "ATCService",
    "DrugDictionary",
    "DrugEntry",
    "DrugMention",
    "DrugNormalizer",
    "DrugSafetyAssemblyGate",
    "DrugSafetyInputGate",
    "DrugSafetyOutputGate",
    "InMemoryAllergyStore",
    "SafetyStack",
    "SEED_ALLERGIES",
    "SEED_DRUG_DICTIONARY",
    "build_allergy_record",
    "build_safety_stack",
    "fold_text",
]


@dataclass(frozen=True)
class SafetyStack:
    """安全层组件全家桶：词典 / 归一化 / ATC / 过敏史 / 解析器 / 三道闸门。

    供给层门面（M3 检索、M4 编排）与质量门禁（M5）共用同一实例，
    保证全链路阻断口径一致。
    """

    dictionary: DrugDictionary
    normalizer: DrugNormalizer
    atc: ATCService
    allergy_store: AllergyStore
    resolver: AllergyConflictResolver
    input_gate: DrugSafetyInputGate
    assembly_gate: DrugSafetyAssemblyGate
    output_gate: DrugSafetyOutputGate


def build_safety_stack(
    settings: Settings | None = None,
    allergy_store: AllergyStore | None = None,
) -> SafetyStack:
    """按配置装配安全层全家桶。

    参数：
        settings:      配置（None 时取全局配置）
        allergy_store: 过敏史供给实现（None 时用合成种子）。
                       **生产接入点**：对接 HIS / EMR 过敏史接口时实现
                       ``AllergyStore`` 协议并在此注入，三道闸门自动换源，
                       业务逻辑零分叉。

    词典路径（``HARNESS_SAFETY__DICTIONARY_PATH``）留空时使用内置
    合成种子词典（32 条）；生产环境填入完整词典 JSON 路径即可。
    """
    if settings is None:
        from harness_agent.config.settings import get_settings

        settings = get_settings()

    if settings.safety.dictionary_path:
        dictionary = DrugDictionary.from_json(settings.safety.dictionary_path)
    else:
        dictionary = DrugDictionary(SEED_DRUG_DICTIONARY)
    normalizer = DrugNormalizer(dictionary)
    atc = ATCService(dictionary)
    if allergy_store is None:
        allergy_store = InMemoryAllergyStore.with_seed_data(normalizer, atc)
    resolver = AllergyConflictResolver(atc, allergy_store)
    return SafetyStack(
        dictionary=dictionary,
        normalizer=normalizer,
        atc=atc,
        allergy_store=allergy_store,
        resolver=resolver,
        input_gate=DrugSafetyInputGate(normalizer, resolver),
        assembly_gate=DrugSafetyAssemblyGate(normalizer),
        output_gate=DrugSafetyOutputGate(normalizer, resolver),
    )
