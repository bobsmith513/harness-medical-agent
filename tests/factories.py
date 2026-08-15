"""测试工厂：M1 领域模型的最小合法构造（供各测试文件复用）。"""

from __future__ import annotations

from harness_agent.models.evidence import Evidence, EvidencePack, SourceRef
from harness_agent.models.reasoning import ClinicalConclusion, ReasoningChain, ReasoningStep
from harness_agent.models.session import RouteRecord, SessionContext, TurnRecord

__all__ = [
    "make_chain",
    "make_conclusion",
    "make_evidence",
    "make_evidence_pack",
    "make_turn",
]


def make_evidence(
    evidence_id: str = "ev-1",
    *,
    content: str = "血常规显示 WBC 12.3e9/L，中性粒细胞比例 78%。",
    structural: bool = False,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        content=content,
        source=SourceRef(source_id="src-1", source_type="document", chunk_id="chunk-1"),
        confidence="high",
        provenance="knowledge_base",
        is_structural_completion=structural,
    )


def make_chain(*, self_check_passed: bool = True, citation: str = "ev-1") -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ReasoningStep(
                kind="evidence",
                text="引用证据：检验指标异常。",
                citations=[citation],
            ),
            ReasoningStep(kind="inference", text="逐步推断：感染可能性升高。"),
            ReasoningStep(kind="conclusion", text="结论：建议结合体征进一步排查。"),
        ],
        self_check_passed=self_check_passed,
    )


def make_conclusion(*, chain: ReasoningChain | None = None) -> ClinicalConclusion:
    return ClinicalConclusion(
        statement="结合检验结果，建议进一步排查感染源。",
        reasoning_chain=chain or make_chain(),
        cited_evidence_ids=["ev-1"],
    )


def make_evidence_pack(*, patient_id: str = "pat-001", reviewed: bool = True) -> EvidencePack:
    from harness_agent.models.audit import GateVerdict

    return EvidencePack(
        session_id="sess-1",
        patient_id=patient_id,
        query="患者发热三天，检验结果如何解读？",
        evidence=[make_evidence()],
        assembly_gate=GateVerdict(gate="assembly", allowed=reviewed, reason="复核通过")
        if reviewed
        else None,
    )


def make_turn(index: int = 1, *, decision: str = "need_reasoning") -> TurnRecord:
    return TurnRecord(
        turn_index=index,
        user_input="患者发热三天，如何解读检验结果？",
        route=RouteRecord(decision=decision, by_rule=True, reason="关键词命中"),
    )


def make_session(patient_id: str = "pat-001") -> SessionContext:
    return SessionContext(patient_id=patient_id)
