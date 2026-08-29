"""ATC 交叉反应与过敏冲突解析测试。"""

from __future__ import annotations

import pytest

from factories import make_chain
from harness_agent.safety import (
    SEED_DRUG_DICTIONARY,
    AllergyConflictResolver,
    ATCService,
    DrugDictionary,
    DrugEntry,
    DrugNormalizer,
    InMemoryAllergyStore,
    build_allergy_record,
    build_safety_stack,
)


@pytest.fixture()
def dictionary() -> DrugDictionary:
    return DrugDictionary(SEED_DRUG_DICTIONARY)


@pytest.fixture()
def atc(dictionary: DrugDictionary) -> ATCService:
    return ATCService(dictionary)


class TestATCService:
    def test_atc_code_lookup(self, atc: ATCService):
        assert atc.atc_code("aspirin") == "N02BA01"

    def test_unknown_drug_returns_none(self, atc: ATCService):
        assert atc.atc_code("unknown-drug") is None

    def test_cross_group_membership(self, atc: ATCService):
        assert atc.cross_group("aspirin") == "nsaid"
        assert atc.cross_group("azithromycin") is None

    def test_penicillin_cross_covers_beta_lactam(self, atc: ATCService):
        cross = atc.cross_reactants("penicillin")
        assert "amoxicillin" in cross
        assert "ceftriaxone" in cross
        assert "penicillin" not in cross

    def test_no_cross_group_returns_empty(self, atc: ATCService):
        assert atc.cross_reactants("azithromycin") == []

    def test_unknown_drug_cross_reactants_empty(self, atc: ATCService):
        assert atc.cross_reactants("unknown-drug") == []

    def test_expanded_dictionary_covers_all_cross_groups(self, atc: ATCService):
        """三个交叉反应组均已落地（sulfonamide 曾只存在于文档字符串里）。"""
        assert atc.cross_group("penicillin") == "beta_lactam"
        assert atc.cross_group("aspirin") == "nsaid"
        assert atc.cross_group("sulfamethoxazole") == "sulfonamide"

    def test_penicillin_cross_covers_untagged_group_members(self, atc: ATCService):
        """组内成员即使漏打 cross_group，也必须被 ATC 前缀兜底纳入。"""
        cross = atc.cross_reactants("penicillin")
        for name in ("ampicillin", "piperacillin", "meropenem", "cefepime"):
            assert name in cross


class TestATCPrefixFallback:
    """ATC 前缀兜底：防「新药漏打 cross_group -> 静默 fail-open」。"""

    @staticmethod
    def _dictionary(*entries: DrugEntry) -> DrugDictionary:
        return DrugDictionary(list(entries))

    def test_infer_group_from_prefix(self):
        assert ATCService.infer_group("J01CA12") == "beta_lactam"
        assert ATCService.infer_group("J01DD02") == "beta_lactam"
        assert ATCService.infer_group("M01AE02") == "nsaid"
        assert ATCService.infer_group("N02BA01") == "nsaid"
        assert ATCService.infer_group("J01EC01") == "sulfonamide"

    def test_infer_group_returns_none_for_unmapped(self):
        """刻意排除的前缀：单环 β-内酰胺与对乙酰氨基酚。"""
        assert ATCService.infer_group("J01DF01") is None  # 氨曲南
        assert ATCService.infer_group("N02BE01") is None  # 对乙酰氨基酚
        assert ATCService.infer_group("J01FA10") is None  # 阿奇霉素

    def test_untagged_beta_lactam_still_blocked(self):
        """漏打 cross_group 的新药不得静默放行：ATC 前缀应把它拉回组内。"""
        dictionary = self._dictionary(
            DrugEntry(normalized_name="penicillin", atc_code="J01CE01", cross_group="beta_lactam"),
            # 新药漏打 cross_group —— 修复前这里会静默 fail-open
            DrugEntry(normalized_name="new_ceph", atc_code="J01DD99"),
        )
        atc = ATCService(dictionary)
        assert atc.cross_group("new_ceph") == "beta_lactam"
        assert "new_ceph" in atc.cross_reactants("penicillin")

    def test_monobactam_stays_out_of_beta_lactam(self):
        """例外类别：氨曲南与青霉素交叉反应极低，不得因前缀兜底被误并组。"""
        dictionary = self._dictionary(
            DrugEntry(normalized_name="penicillin", atc_code="J01CE01", cross_group="beta_lactam"),
            DrugEntry(normalized_name="aztreonam", atc_code="J01DF01"),
        )
        atc = ATCService(dictionary)
        assert atc.cross_group("aztreonam") is None
        assert atc.cross_reactants("aztreonam") == []
        assert "aztreonam" not in atc.cross_reactants("penicillin")

    def test_explicit_group_wins_over_inferred(self):
        """显式标注优先于前缀推断（生产词典的权威来源）。"""
        dictionary = self._dictionary(
            # ATC 属 nsaid 前缀范围，但显式声明为独立组
            DrugEntry(
                normalized_name="special_analgesic",
                atc_code="M01AE99",
                cross_group="custom_group",
            ),
        )
        assert ATCService(dictionary).cross_group("special_analgesic") == "custom_group"


class TestAllergyStoreInjection:
    """``AllergyStore`` 注入：生产对接 HIS / EMR 的接入点。

    修复前 ``build_safety_stack`` 硬编码 ``with_seed_data``，注释声称
    "实现本接口并注入三道闸门即可"，但装配代码里根本没有注入口——
    只能改源码。此处锁定注入后三道闸门确实换源。
    """

    @staticmethod
    def _record(patient_id: str, raw_name: str):
        dictionary = DrugDictionary(SEED_DRUG_DICTIONARY)
        normalizer = DrugNormalizer(dictionary)
        atc = ATCService(dictionary)
        return build_allergy_record(patient_id, raw_name, normalizer, atc)

    def test_injected_store_replaces_seed_data(self):
        """注入外部过敏史后，阻断集合按新数据计算（含 ATC 组扩展）。"""
        safety = build_safety_stack(
            allergy_store=InMemoryAllergyStore([self._record("pat-900", "阿莫西林")])
        )
        blocked = safety.resolver.blocked_drugs("pat-900")
        assert {"amoxicillin", "ceftriaxone", "penicillin"} <= blocked

    def test_injected_store_does_not_leak_seed_patients(self):
        """注入后种子患者（pat-001）不应再有任何阻断——口径完全换源。"""
        safety = build_safety_stack(
            allergy_store=InMemoryAllergyStore([self._record("pat-900", "阿莫西林")])
        )
        assert safety.resolver.blocked_drugs("pat-001") == frozenset()

    def test_output_gate_follows_injected_store(self):
        """输出闸门同样换源（注入必须穿透到第三道闸门，而非只换解析器）。"""
        safety = build_safety_stack(
            allergy_store=InMemoryAllergyStore([self._record("pat-900", "阿莫西林")])
        )
        from harness_agent.models.reasoning import ClinicalConclusion
        from harness_agent.models.session import SessionContext

        verdict = safety.output_gate.check(
            ClinicalConclusion(statement="建议改用头孢曲松", reasoning_chain=make_chain()),
            SessionContext(patient_id="pat-900"),
        )
        assert verdict.allowed is False
        assert "ceftriaxone" in verdict.blocked_drugs


class TestConflictResolver:
    @pytest.fixture()
    def resolver(self, atc: ATCService) -> AllergyConflictResolver:
        from harness_agent.safety import DrugNormalizer, InMemoryAllergyStore

        normalizer = DrugNormalizer(DrugDictionary(SEED_DRUG_DICTIONARY))
        store = InMemoryAllergyStore.with_seed_data(normalizer, atc)
        return AllergyConflictResolver(atc, store)

    def test_blocked_set_covers_cross_group(self, resolver: AllergyConflictResolver):
        blocked = resolver.blocked_drugs("pat-001")
        assert {"penicillin", "amoxicillin", "ceftriaxone"} <= blocked

    def test_blocked_set_excludes_unrelated_class(self, resolver: AllergyConflictResolver):
        assert "azithromycin" not in resolver.blocked_drugs("pat-001")

    def test_patient_without_records_has_empty_block_set(self, resolver: AllergyConflictResolver):
        assert resolver.blocked_drugs("pat-003") == frozenset()
