"""三道供给闸门测试（M2）：fail-closed 语义锁定。"""

from __future__ import annotations

import pytest

from factories import make_evidence, make_evidence_pack
from harness_agent.contracts.gates import AssemblyGate, InputGate, OutputGate
from harness_agent.contracts.retrieval import RetrievalQuery
from harness_agent.models.reasoning import ClinicalConclusion, ReasoningChain, ReasoningStep
from harness_agent.models.session import SessionContext
from harness_agent.safety import (
    SEED_DRUG_DICTIONARY,
    AllergyConflictResolver,
    ATCService,
    DrugDictionary,
    DrugNormalizer,
    DrugSafetyAssemblyGate,
    DrugSafetyInputGate,
    DrugSafetyOutputGate,
    InMemoryAllergyStore,
    build_allergy_record,
)


@pytest.fixture()
def normalizer() -> DrugNormalizer:
    return DrugNormalizer(DrugDictionary(SEED_DRUG_DICTIONARY))


@pytest.fixture()
def resolver(normalizer: DrugNormalizer) -> AllergyConflictResolver:
    atc = ATCService(DrugDictionary(SEED_DRUG_DICTIONARY))
    store = InMemoryAllergyStore.with_seed_data(normalizer, atc)
    return AllergyConflictResolver(atc, store)


def _query(text: str, patient_id: str) -> RetrievalQuery:
    return RetrievalQuery(text=text, patient_id=patient_id)


def _context(patient_id: str) -> SessionContext:
    return SessionContext(patient_id=patient_id)


class TestContractCompliance:
    def test_gates_satisfy_m1_contracts(
        self, normalizer: DrugNormalizer, resolver: AllergyConflictResolver
    ):
        assert isinstance(DrugSafetyInputGate(normalizer, resolver), InputGate)
        assert isinstance(DrugSafetyAssemblyGate(normalizer), AssemblyGate)
        assert isinstance(DrugSafetyOutputGate(normalizer, resolver), OutputGate)


class TestInputGate:
    @pytest.fixture()
    def gate(self, normalizer: DrugNormalizer, resolver: AllergyConflictResolver):
        return DrugSafetyInputGate(normalizer, resolver)

    def test_blocks_allergy_drug_mention(self, gate: DrugSafetyInputGate):
        verdict = gate.check(_query("开点阿莫西林", "pat-001"), _context("pat-001"))
        assert verdict.allowed is False
        assert verdict.blocked_drugs == ["amoxicillin"]
        assert "amoxicillin" in verdict.reason

    def test_blocks_cross_reactant(self, gate: DrugSafetyInputGate):
        """青霉素过敏患者查询头孢曲松：交叉反应命中即拦截。"""
        verdict = gate.check(_query("用头孢曲松静滴", "pat-001"), _context("pat-001"))
        assert verdict.allowed is False
        assert verdict.blocked_drugs == ["ceftriaxone"]

    def test_allows_unrelated_drug_class(self, gate: DrugSafetyInputGate):
        verdict = gate.check(_query("阿奇霉素怎么吃", "pat-001"), _context("pat-001"))
        assert verdict.allowed is True
        assert verdict.blocked_drugs == []

    def test_allows_patient_without_allergies(self, gate: DrugSafetyInputGate):
        verdict = gate.check(_query("开点阿莫西林", "pat-003"), _context("pat-003"))
        assert verdict.allowed is True

    def test_allows_query_without_drugs(self, gate: DrugSafetyInputGate):
        verdict = gate.check(_query("患者发热三天", "pat-001"), _context("pat-001"))
        assert verdict.allowed is True
        assert "未检出药物" in verdict.reason


class TestAssemblyGate:
    @pytest.fixture()
    def gate(self, normalizer: DrugNormalizer):
        return DrugSafetyAssemblyGate(normalizer)

    @pytest.fixture()
    def blocked(self, resolver: AllergyConflictResolver) -> list[str]:
        """完整阻断集合（直接过敏 + 交叉反应），即输入闸门传递给证据包的口径。"""
        return sorted(resolver.blocked_drugs("pat-001"))

    def _pack(self, blocked: list[str]) -> object:
        pack = make_evidence_pack(reviewed=False, patient_id="pat-001")
        pack.blocked_drugs = blocked
        pack.evidence.append(
            make_evidence(evidence_id="ev-2", content="阿莫西林胶囊 0.5g，口服每日两次。")
        )
        return pack

    def test_check_flags_offending_evidence(self, gate, blocked):
        verdict = gate.check(self._pack(blocked))
        assert verdict.allowed is False
        assert verdict.blocked_drugs == ["amoxicillin"]

    def test_apply_filters_and_allows(self, gate, blocked):
        """过滤含过敏药物实体的证据后放行，裁决附加到证据包。"""
        pack = gate.apply(self._pack(blocked))
        assert pack.is_reviewed is True
        assert [e.evidence_id for e in pack.evidence] == ["ev-1"]
        assert "已过滤 1 条" in pack.assembly_gate.reason

    def test_apply_fail_closed_when_all_filtered(self, gate, blocked):
        """全部证据含过敏药物实体时拒绝交付空证据包（fail-closed）。"""
        pack = make_evidence_pack(reviewed=False, patient_id="pat-001")
        pack.blocked_drugs = blocked
        pack.evidence = [make_evidence(evidence_id="ev-2", content="阿莫西林胶囊用法说明。")]
        result = gate.apply(pack)
        assert result.assembly_gate is not None
        assert result.assembly_gate.allowed is False
        assert result.evidence == []
        assert result.is_reviewed is False

    def test_apply_clean_pack_passes_without_filter(self, gate):
        pack = make_evidence_pack(reviewed=False, patient_id="pat-003")
        result = gate.apply(pack)
        assert result.is_reviewed is True
        assert "无需过滤" in result.assembly_gate.reason


class TestOutputGate:
    @pytest.fixture()
    def gate(self, normalizer: DrugNormalizer, resolver: AllergyConflictResolver):
        return DrugSafetyOutputGate(normalizer, resolver)

    @staticmethod
    def _conclusion_with(inference_text: str, statement: str) -> ClinicalConclusion:
        chain = ReasoningChain(
            steps=[
                ReasoningStep(kind="evidence", text="引用证据：用药说明。", citations=["ev-1"]),
                ReasoningStep(kind="inference", text=inference_text),
                ReasoningStep(kind="conclusion", text="结论步。"),
            ],
            self_check_passed=True,
        )
        return ClinicalConclusion(statement=statement, reasoning_chain=chain)

    def test_blocks_conclusion_mentioning_allergy_drug(self, gate: DrugSafetyOutputGate):
        conclusion = self._conclusion_with("推断安全。", "建议口服阿莫西林 0.5g。")
        verdict = gate.check(conclusion, _context("pat-001"))
        assert verdict.allowed is False
        assert verdict.blocked_drugs == ["amoxicillin"]

    def test_blocks_drug_hidden_in_reasoning_step(self, gate: DrugSafetyOutputGate):
        """结论陈述干净但推理步提及过敏药物：同样拦截（全文扫描）。"""
        conclusion = self._conclusion_with("建议使用头孢曲松静滴三天。", "建议静脉抗生素治疗。")
        verdict = gate.check(conclusion, _context("pat-001"))
        assert verdict.allowed is False
        assert verdict.blocked_drugs == ["ceftriaxone"]

    def test_allows_clean_conclusion(self, gate: DrugSafetyOutputGate):
        conclusion = self._conclusion_with("推断安全。", "建议口服大环内酯类抗生素。")
        verdict = gate.check(conclusion, _context("pat-001"))
        assert verdict.allowed is True


class TestAllergyStoreAndRecord:
    def test_build_allergy_record_normalizes_and_expands(self, normalizer: DrugNormalizer):
        atc = ATCService(DrugDictionary(SEED_DRUG_DICTIONARY))
        record = build_allergy_record("pat-009", "拜阿司匹灵", normalizer, atc)
        assert record.normalized_drug == "aspirin"
        assert record.atc_code == "N02BA01"
        assert "ibuprofen" in record.cross_reactants

    def test_build_allergy_record_unknown_drug_raises(self, normalizer: DrugNormalizer):
        atc = ATCService(DrugDictionary(SEED_DRUG_DICTIONARY))
        with pytest.raises(ValueError, match="归一化"):
            build_allergy_record("pat-009", "神秘药物", normalizer, atc)

    def test_seed_store_lookup(self, normalizer: DrugNormalizer):
        atc = ATCService(DrugDictionary(SEED_DRUG_DICTIONARY))
        store = InMemoryAllergyStore.with_seed_data(normalizer, atc)
        assert {r.normalized_drug for r in store.get("pat-001")} == {"penicillin"}
        assert store.get("pat-003") == []
