"""ATC 交叉反应与过敏冲突解析测试。"""

from __future__ import annotations

import pytest

from harness_agent.safety import (
    SEED_DRUG_DICTIONARY,
    AllergyConflictResolver,
    ATCService,
    DrugDictionary,
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
