"""推理链与临床结论模型测试：主 Agent 无应答权的类型级锁定。

M1 验收最核心的一组测试——"临床结论均出自带核查的推理管线"
由 ``ClinicalConclusion`` 校验器保证，任何绕过推理链构造结论的
尝试都必须在类型层面失败。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from factories import make_chain, make_conclusion
from harness_agent.models.reasoning import (
    ClinicalConclusion,
    ReasoningChain,
    ReasoningStep,
)


class TestClinicalConclusionPower:
    """临床结论的构造权力只属于自检通过的推理链。"""

    def test_valid_conclusion_constructs(self):
        conclusion = make_conclusion()
        assert conclusion.produced_by == "reasoning_expert"
        assert conclusion.cited_evidence_ids == ["ev-1"]
        assert conclusion.created_at is not None

    def test_conclusion_without_chain_is_rejected(self):
        """结论不可能脱离推理链凭空构造（主 Agent 无应答权）。"""
        with pytest.raises(ValidationError):
            ClinicalConclusion(statement="患者可能是细菌感染。")

    def test_conclusion_rejects_unchecked_chain(self):
        """自检未通过的推理链不得产出临床结论。"""
        unchecked = make_chain(self_check_passed=False)
        with pytest.raises(ValidationError, match="自检通过"):
            ClinicalConclusion(
                statement="结论陈述。",
                reasoning_chain=unchecked,
            )

    def test_conclusion_rejects_citations_beyond_chain(self):
        """结论引用的证据必须出现在推理链引用集合内（结论与依据不可分离）。"""
        with pytest.raises(ValidationError, match="未引用的证据"):
            ClinicalConclusion(
                statement="结论陈述。",
                reasoning_chain=make_chain(),
                cited_evidence_ids=["ev-999"],
            )


class TestReasoningChainShape:
    """推理链固定"证据引用 -> 逐步推断 -> 结论"结构。"""

    def test_chain_must_contain_evidence_step(self):
        with pytest.raises(ValidationError, match="证据引用步"):
            ReasoningChain(
                steps=[
                    ReasoningStep(kind="inference", text="推断。"),
                    ReasoningStep(kind="conclusion", text="结论。"),
                ]
            )

    def test_chain_must_contain_inference_step(self):
        with pytest.raises(ValidationError, match="推断步"):
            ReasoningChain(
                steps=[
                    ReasoningStep(kind="evidence", text="证据。", citations=["ev-1"]),
                    ReasoningStep(kind="conclusion", text="结论。"),
                ]
            )

    def test_chain_last_step_must_be_conclusion(self):
        with pytest.raises(ValidationError, match="末步必须是结论"):
            ReasoningChain(
                steps=[
                    ReasoningStep(kind="evidence", text="证据。", citations=["ev-1"]),
                    ReasoningStep(kind="inference", text="推断。"),
                ]
            )

    def test_chain_first_step_must_be_evidence(self):
        with pytest.raises(ValidationError, match="首步必须是证据"):
            ReasoningChain(
                steps=[
                    ReasoningStep(kind="inference", text="推断。"),
                    ReasoningStep(kind="evidence", text="证据。", citations=["ev-1"]),
                    ReasoningStep(kind="conclusion", text="结论。"),
                ]
            )

    def test_chain_out_of_order_rejected(self):
        """步序必须单调：证据 -> 推断 -> 结论，倒插的证据步直接拒绝。"""
        with pytest.raises(ValidationError, match="顺序"):
            ReasoningChain(
                steps=[
                    ReasoningStep(kind="evidence", text="证据。", citations=["ev-1"]),
                    ReasoningStep(kind="inference", text="推断。"),
                    ReasoningStep(kind="evidence", text="补证据。", citations=["ev-2"]),
                    ReasoningStep(kind="conclusion", text="结论。"),
                ]
            )

    def test_evidence_step_requires_citation(self):
        with pytest.raises(ValidationError, match="引用"):
            ReasoningStep(kind="evidence", text="证据陈述但无引用。")

    def test_cited_evidence_ids_aggregates_deduplicated(self):
        chain = ReasoningChain(
            steps=[
                ReasoningStep(kind="evidence", text="证据 A。", citations=["ev-1", "ev-2"]),
                ReasoningStep(kind="inference", text="推断，引用 A。", citations=["ev-1"]),
                ReasoningStep(kind="conclusion", text="结论。"),
            ]
        )
        assert chain.cited_evidence_ids == ["ev-1", "ev-2"]
