"""M5 推理专家与质量门禁测试。

覆盖：
1. 推理专家：三段式推理链生成 + 自检（引用真实性/因果正向/依据充分性）；
2. LLM-judge 门禁：规则前置拦截 + LLM 兜底忠实度/臆测/因果倒置；
3. 门禁流水线：质量门禁→输出闸门串联，fail-closed；
4. 端到端：取证据→推理→门禁→输出（mock 模型全链路跑通）。
"""

from __future__ import annotations

import json

import pytest

from harness_agent.contracts.experts import ExpertTask
from harness_agent.contracts.gates import QualityGate
from harness_agent.experts.reasoning_expert import ReasoningExpertImpl
from harness_agent.gates.pipeline import GatePipeline
from harness_agent.gates.quality_judge import LLMJudgeGate
from harness_agent.llm.mock import MockLLMClient
from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import Evidence, EvidencePack, SourceRef
from harness_agent.models.reasoning import (
    ClinicalConclusion,
    ReasoningChain,
    ReasoningStep,
)
from harness_agent.models.session import SessionContext

PAT = "pat-003"


def _evidence(eid: str = "ev-1", content: str = "阿奇霉素的适应证说明") -> Evidence:
    return Evidence(
        evidence_id=eid,
        content=content,
        source=SourceRef(source_id="s1", source_type="document", chunk_id="kb-1"),
        confidence="medium",
        provenance="knowledge_base",
    )


def _pack(*evidence: Evidence) -> EvidencePack:

    return EvidencePack(
        session_id="sess-1",
        patient_id=PAT,
        query="测试",
        evidence=list(evidence) or [_evidence()],
        assembly_gate=GateVerdict(gate="assembly", allowed=True, reason="复核通过"),
    )


def _context() -> SessionContext:
    return SessionContext(session_id="sess-1", patient_id=PAT)


def _task(question: str = "阿奇霉素怎么用") -> ExpertTask:
    return ExpertTask(expert="reasoning_expert", instruction=f"分析：{question}")


# ---------------------------------------------------------------------------
# 推理专家
# ---------------------------------------------------------------------------
class TestReasoningExpertContract:
    def test_satisfies_protocol(self):
        llm = MockLLMClient(role="reasoning")
        expert = ReasoningExpertImpl(llm=llm)
        from harness_agent.contracts.experts import ReasoningExpert

        assert isinstance(expert, ReasoningExpert)

    def test_role_must_be_reasoning(self):
        with pytest.raises(ValueError, match="reasoning"):
            ReasoningExpertImpl(llm=MockLLMClient(role="judge"))

    def test_rejects_unreviewed_evidence(self):
        expert = ReasoningExpertImpl(llm=MockLLMClient(role="reasoning"))
        unreviewed = _pack()
        unreviewed.assembly_gate = None
        with pytest.raises(ValueError, match="装配复核"):
            expert.reason(_task(), unreviewed, _context())


class TestReasoningExpertChain:
    def test_parses_llm_output_into_valid_chain(self):
        """LLM 输出合法 JSON → 推理链 + 自检 + 临床结论。"""
        llm_output = json.dumps(
            {
                "steps": [
                    {"kind": "evidence", "text": "引用证据说明", "citations": ["ev-1"]},
                    {"kind": "inference", "text": "基于证据推断"},
                    {"kind": "conclusion", "text": "形成结论", "citations": ["ev-1"]},
                ],
                "statement": "阿奇霉素可用于社区获得性肺炎",
                "self_check_notes": "自检通过",
            }
        )
        llm = MockLLMClient(role="reasoning", script=[llm_output])
        expert = ReasoningExpertImpl(llm=llm)
        conclusion = expert.reason(_task(), _pack(), _context())

        assert isinstance(conclusion, ClinicalConclusion)
        assert conclusion.produced_by == "reasoning_expert"
        assert conclusion.reasoning_chain.self_check_passed is True
        assert "ev-1" in conclusion.cited_evidence_ids

    def test_fallback_chain_on_unparsable_output(self):
        """LLM 输出不可解析 → 兜底链（以首条证据为据）。"""
        llm = MockLLMClient(role="reasoning", script=["这不好说吧"])
        expert = ReasoningExpertImpl(llm=llm)
        conclusion = expert.reason(_task(), _pack(), _context())
        chain = conclusion.reasoning_chain
        assert chain.self_check_passed is True
        assert chain.steps[0].kind == "evidence"
        assert chain.steps[0].citations == ["ev-1"]

    def test_self_check_rejects_fake_citation(self):
        """推理链引用了证据包中不存在的 evidence_id → 自检失败。"""
        llm_output = json.dumps(
            {
                "steps": [
                    {"kind": "evidence", "text": "引用", "citations": ["ghost-id"]},
                    {"kind": "inference", "text": "推断"},
                    {"kind": "conclusion", "text": "结论", "citations": ["ghost-id"]},
                ],
                "statement": "结论",
            }
        )
        llm = MockLLMClient(role="reasoning", script=[llm_output])
        expert = ReasoningExpertImpl(llm=llm)
        with pytest.raises(ValueError, match="不存在"):
            expert.reason(_task(), _pack(), _context())


# ---------------------------------------------------------------------------
# LLM-judge 质量门禁
# ---------------------------------------------------------------------------
class TestLLMJudgeGate:
    def _conclusion(self, ev_id: str = "ev-1") -> ClinicalConclusion:
        chain = ReasoningChain(
            steps=[
                ReasoningStep(kind="evidence", text="引用", citations=[ev_id]),
                ReasoningStep(kind="inference", text="推断"),
                ReasoningStep(kind="conclusion", text="结论", citations=[ev_id]),
            ],
            self_check_passed=True,
        )
        return ClinicalConclusion(
            statement="测试结论", reasoning_chain=chain, cited_evidence_ids=[ev_id]
        )

    def test_satisfies_protocol(self):
        gate = LLMJudgeGate(llm=MockLLMClient(role="judge"))
        assert isinstance(gate, QualityGate)

    def test_role_must_be_judge(self):
        with pytest.raises(ValueError, match="judge"):
            LLMJudgeGate(llm=MockLLMClient(role="reasoning"))

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError, match="阈值"):
            LLMJudgeGate(llm=MockLLMClient(role="judge"), threshold=1.5)

    def test_rule_check_passes_valid_conclusion(self):
        gate = LLMJudgeGate(llm=MockLLMClient(role="judge"))
        verdict = gate.evaluate(self._conclusion(), _pack())
        assert verdict.allowed is True

    def test_rule_check_blocks_fake_citation(self):
        """结论引用了证据包中不存在的 evidence_id → 规则前置拦截。"""
        gate = LLMJudgeGate(llm=MockLLMClient(role="judge"))
        conclusion = self._conclusion(ev_id="ghost")
        verdict = gate.evaluate(conclusion, _pack())
        assert verdict.allowed is False
        assert "不存在" in verdict.reason

    def test_llm_judge_blocks_hallucination(self):
        """judge 检测到臆测 → 拦截。"""
        judge_output = json.dumps(
            {
                "faithfulness": 0.9,
                "has_hallucination": True,
                "causal_inversion": False,
                "reason": "结论引入了证据未提及的推断",
            }
        )
        gate = LLMJudgeGate(llm=MockLLMClient(role="judge", script=[judge_output]))
        verdict = gate.evaluate(self._conclusion(), _pack())
        assert verdict.allowed is False
        assert "臆测" in verdict.reason

    def test_llm_judge_blocks_causal_inversion(self):
        """judge 检测到因果倒置 → 拦截。"""
        judge_output = json.dumps(
            {
                "faithfulness": 0.9,
                "has_hallucination": False,
                "causal_inversion": True,
                "reason": "结论先于证据出现",
            }
        )
        gate = LLMJudgeGate(llm=MockLLMClient(role="judge", script=[judge_output]))
        verdict = gate.evaluate(self._conclusion(), _pack())
        assert verdict.allowed is False
        assert "因果倒置" in verdict.reason

    def test_llm_judge_blocks_low_faithfulness(self):
        """忠实度低于阈值 → 拦截。"""
        judge_output = json.dumps(
            {
                "faithfulness": 0.3,
                "has_hallucination": False,
                "causal_inversion": False,
                "reason": "依据不足",
            }
        )
        gate = LLMJudgeGate(llm=MockLLMClient(role="judge", script=[judge_output]), threshold=0.7)
        verdict = gate.evaluate(self._conclusion(), _pack())
        assert verdict.allowed is False
        assert "忠实度" in verdict.reason

    def test_llm_judge_passes_high_faithfulness(self):
        judge_output = json.dumps(
            {
                "faithfulness": 0.95,
                "has_hallucination": False,
                "causal_inversion": False,
                "reason": "结论有充分证据支撑",
            }
        )
        gate = LLMJudgeGate(llm=MockLLMClient(role="judge", script=[judge_output]))
        verdict = gate.evaluate(self._conclusion(), _pack())
        assert verdict.allowed is True


# ---------------------------------------------------------------------------
# 门禁流水线
# ---------------------------------------------------------------------------
class TestGatePipeline:
    def _conclusion(self, ev_id: str = "ev-1") -> ClinicalConclusion:
        chain = ReasoningChain(
            steps=[
                ReasoningStep(kind="evidence", text="引用", citations=[ev_id]),
                ReasoningStep(kind="inference", text="推断"),
                ReasoningStep(kind="conclusion", text="结论", citations=[ev_id]),
            ],
            self_check_passed=True,
        )
        return ClinicalConclusion(
            statement="安全结论", reasoning_chain=chain, cited_evidence_ids=[ev_id]
        )

    def _pipeline(self, judge_script: list[str] | None = None) -> GatePipeline:
        from harness_agent.safety import build_safety_stack

        safety = build_safety_stack()
        judge_llm = MockLLMClient(role="judge", script=judge_script or [])
        return GatePipeline(
            quality_gate=LLMJudgeGate(llm=judge_llm),
            output_gate=safety.output_gate,
        )

    def test_both_gates_pass(self):
        pipeline = self._pipeline()
        result = pipeline.run(self._conclusion(), _pack(), _context())
        assert result.allowed is True
        assert len(result.verdicts) == 2

    def test_quality_gate_blocks_before_output_gate(self):
        """质量门禁拦截 → 不执行输出闸门（fail-closed 短路）。"""
        bad_output = json.dumps(
            {
                "faithfulness": 0.2,
                "has_hallucination": True,
                "causal_inversion": False,
                "reason": "臆测",
            }
        )
        pipeline = self._pipeline(judge_script=[bad_output])
        result = pipeline.run(self._conclusion(), _pack(), _context())
        assert result.allowed is False
        assert result.blocking_gate == "gate:quality_judge"
        assert len(result.verdicts) == 1  # 质量门禁拦截后不执行输出闸门

    def test_output_gate_blocks_allergy_drug(self):
        """结论提及过敏药物 → 输出闸门拦截（M2 复用）。"""
        from harness_agent.safety import build_safety_stack

        safety = build_safety_stack()
        # pat-001 青霉素过敏：结论陈述提及 penicillin
        chain = ReasoningChain(
            steps=[
                ReasoningStep(kind="evidence", text="引用", citations=["ev-1"]),
                ReasoningStep(kind="inference", text="推断"),
                ReasoningStep(kind="conclusion", text="建议使用 penicillin", citations=["ev-1"]),
            ],
            self_check_passed=True,
        )
        conclusion = ClinicalConclusion(
            statement="建议使用 penicillin 治疗",
            reasoning_chain=chain,
            cited_evidence_ids=["ev-1"],
        )
        context = SessionContext(session_id="sess-1", patient_id="pat-001")
        pipeline = GatePipeline(
            quality_gate=LLMJudgeGate(llm=MockLLMClient(role="judge")),
            output_gate=safety.output_gate,
        )
        result = pipeline.run(conclusion, _pack(), context)
        assert result.allowed is False
        assert result.blocking_gate == "gate:output"
        assert "penicillin" in result.final_verdict.blocked_drugs
