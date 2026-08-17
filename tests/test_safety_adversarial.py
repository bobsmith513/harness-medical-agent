"""M2 验收测试：对抗样本集漏检率必须为 0。

漏检 = ``expect_blocked=True`` 的样本被闸门放行。
任何一条漏检即测试失败（fail-closed 的量化验收）。
"""

from __future__ import annotations

import pytest

from factories import make_chain
from fixtures.adversarial_samples import ADVERSARIAL_SAMPLES
from harness_agent.contracts.retrieval import RetrievalQuery
from harness_agent.models.reasoning import ClinicalConclusion
from harness_agent.models.session import SessionContext
from harness_agent.safety import (
    SEED_DRUG_DICTIONARY,
    AllergyConflictResolver,
    ATCService,
    DrugDictionary,
    DrugNormalizer,
    DrugSafetyInputGate,
    DrugSafetyOutputGate,
    InMemoryAllergyStore,
)


@pytest.fixture(scope="module")
def gates():
    """模块级共享的安全栈（种子数据只读，无状态污染）。"""
    dictionary = DrugDictionary(SEED_DRUG_DICTIONARY)
    normalizer = DrugNormalizer(dictionary)
    atc = ATCService(dictionary)
    store = InMemoryAllergyStore.with_seed_data(normalizer, atc)
    resolver = AllergyConflictResolver(atc, store)
    return DrugSafetyInputGate(normalizer, resolver), DrugSafetyOutputGate(normalizer, resolver)


@pytest.fixture(scope="module")
def input_gate(gates):
    return gates[0]


@pytest.fixture(scope="module")
def output_gate(gates):
    return gates[1]


class TestAdversarialSuiteShape:
    def test_suite_has_minimal_coverage(self):
        """每类 1-2 条代表：总量 >= 7，阳性样本 >= 5。"""
        assert len(ADVERSARIAL_SAMPLES) >= 7
        positives = [s for s in ADVERSARIAL_SAMPLES if s.expect_blocked]
        assert len(positives) >= 5
        assert all(s.expected_drugs for s in positives)


class TestInputGateZeroMissRate:
    @pytest.mark.parametrize("sample", ADVERSARIAL_SAMPLES, ids=lambda s: s.sample_id)
    def test_each_sample(self, sample, input_gate: DrugSafetyInputGate):
        verdict = input_gate.check(
            RetrievalQuery(text=sample.text, patient_id=sample.patient_id),
            SessionContext(patient_id=sample.patient_id),
        )
        if sample.expect_blocked:
            assert verdict.allowed is False, (
                f"漏检: {sample.sample_id} {sample.description}: {sample.text!r}"
            )
            assert set(sample.expected_drugs) <= set(verdict.blocked_drugs)
            assert verdict.reason
        else:
            assert verdict.allowed is True, (
                f"误拦: {sample.sample_id} {sample.description}: {sample.text!r}"
            )

    def test_miss_rate_is_exactly_zero(self, input_gate: DrugSafetyInputGate):
        """漏检率 = 漏检数 / 期望拦截数，必须为 0。"""
        positives = [s for s in ADVERSARIAL_SAMPLES if s.expect_blocked]
        missed = [
            s.sample_id
            for s in positives
            if input_gate.check(
                RetrievalQuery(text=s.text, patient_id=s.patient_id),
                SessionContext(patient_id=s.patient_id),
            ).allowed
        ]
        miss_rate = len(missed) / len(positives)
        assert miss_rate == 0.0, f"漏检率 {miss_rate:.0%}，漏检样本: {missed}"


class TestOutputGateZeroMissRate:
    @pytest.mark.parametrize("sample", ADVERSARIAL_SAMPLES, ids=lambda s: s.sample_id)
    def test_conclusions_with_adversarial_text(self, sample, output_gate):
        """同批对抗文本作为临床结论陈述：输出闸门同样零漏检。"""
        conclusion = ClinicalConclusion(statement=sample.text, reasoning_chain=make_chain())
        verdict = output_gate.check(conclusion, SessionContext(patient_id=sample.patient_id))
        if sample.expect_blocked:
            assert verdict.allowed is False
            assert set(sample.expected_drugs) <= set(verdict.blocked_drugs)
        else:
            assert verdict.allowed is True
