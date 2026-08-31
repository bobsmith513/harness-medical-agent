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

    def test_unparsable_output_fails_closed(self):
        """LLM 输出不可解析 → 抛异常转升级（fail-closed，绝不交付兜底结论）。"""
        llm = MockLLMClient(role="reasoning", script=["这不好说吧"])
        expert = ReasoningExpertImpl(llm=llm)
        with pytest.raises(ValueError, match="fail-closed"):
            expert.reason(_task(), _pack(), _context())

    def test_empty_output_fails_closed(self):
        """LLM 空输出 → 抛异常转升级（不构造任何"最小合法链"）。"""
        llm = MockLLMClient(role="reasoning", script=[""])
        expert = ReasoningExpertImpl(llm=llm)
        with pytest.raises(ValueError, match="不含 JSON"):
            expert.reason(_task(), _pack(), _context())

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


# ---------------------------------------------------------------------------
# 记忆专家（M5）：上下文装配
# ---------------------------------------------------------------------------
class TestMemoryExpert:
    """provenance 分类 + 装配召回窗口（患者事实不被知识库条目挤占）。"""

    @staticmethod
    def _stack_service():
        from harness_agent.retrieval.bm25 import BM25SparseRetriever
        from harness_agent.retrieval.embeddings import HashingEmbeddingProvider
        from harness_agent.retrieval.fusion import IdentityReranker
        from harness_agent.retrieval.service import HybridRetrievalService
        from harness_agent.retrieval.vector_store import InMemoryVectorStore
        from harness_agent.safety import build_safety_stack

        safety = build_safety_stack()
        return HybridRetrievalService(
            embedding_provider=HashingEmbeddingProvider(),
            vector_store=InMemoryVectorStore(),
            sparse=BM25SparseRetriever(),
            reranker=IdentityReranker(),
            input_gate=safety.input_gate,
            assembly_gate=safety.assembly_gate,
            resolver=safety.resolver,
        )

    def test_low_ranked_patient_memory_still_assembled(self):
        """患者记忆名次低于知识库条目（默认窗口会被挤占）仍进入装配。

        装配窗口下限（12）保证患者分区条目有机会进入分类阶段——
        否则"复诊免重复问询"在知识库条目较多时静默失效。
        """
        from harness_agent.contracts.retrieval import RetrievalQuery, StoredChunk
        from harness_agent.experts.memory_expert import MemoryExpertImpl

        service = self._stack_service()
        # 10 条知识库条目与查询词面强匹配（占据融合头部），患者记忆垫底
        service.index(
            [StoredChunk(chunk_id=f"kb-{i}", content=f"高血压治疗方案说明 {i}") for i in range(10)]
            + [
                StoredChunk(
                    chunk_id="mem-s",
                    patient_id=PAT,
                    content="高血压病史 5 年",
                    metadata={"provenance": "doctor_verified"},
                ),
                StoredChunk(
                    chunk_id="mem-v",
                    patient_id=PAT,
                    content="近期服用氨氯地平 5mg qd",
                    metadata={"provenance": "model_inference"},
                ),
            ]
        )
        expert = MemoryExpertImpl(retrieval=service)
        bundle = expert.assemble(
            RetrievalQuery(text="高血压治疗方案", patient_id=PAT, session_id="sess-1"),
            SessionContext(patient_id=PAT),
        )
        assert "高血压病史 5 年" in bundle.stable_facts
        assert "近期服用氨氯地平 5mg qd" in bundle.volatile_facts

    def test_knowledge_base_evidence_excluded_from_facts(self):
        """知识库条目（provenance=knowledge_base）不进入患者事实。"""
        from harness_agent.contracts.retrieval import RetrievalQuery, StoredChunk
        from harness_agent.experts.memory_expert import MemoryExpertImpl

        service = self._stack_service()
        service.index(
            [
                StoredChunk(chunk_id="kb-1", content="高血压治疗方案标准流程"),
                StoredChunk(
                    chunk_id="mem-s",
                    patient_id=PAT,
                    content="高血压病史 5 年",
                    metadata={"provenance": "doctor_verified"},
                ),
            ]
        )
        expert = MemoryExpertImpl(retrieval=service)
        bundle = expert.assemble(
            RetrievalQuery(text="高血压治疗方案", patient_id=PAT, session_id="sess-1"),
            SessionContext(patient_id=PAT),
        )
        # 知识库条目即使名次更高，也不冒充患者事实
        assert "高血压治疗方案标准流程" not in bundle.stable_facts
        assert "高血压治疗方案标准流程" not in bundle.volatile_facts

    def test_allergies_from_hard_rules_not_retrieval(self):
        """过敏史来自硬规则精确匹配（pat-001 青霉素过敏种子），非向量召回。"""
        from harness_agent.contracts.retrieval import RetrievalQuery
        from harness_agent.experts.memory_expert import MemoryExpertImpl

        service = self._stack_service()
        expert = MemoryExpertImpl(retrieval=service)
        bundle = expert.assemble(
            RetrievalQuery(text="任何查询", patient_id="pat-001", session_id="sess-1"),
            SessionContext(patient_id="pat-001"),
        )
        assert bundle.allergies  # 种子安全栈：pat-001 青霉素过敏
        assert any(r.normalized_drug == "penicillin" for r in bundle.allergies)

    def test_unreviewed_pack_fails_closed(self):
        """证据包未通过装配复核 → 记忆专家拒绝装配并抛错（fail-closed）。

        与推理专家 ``ReasoningExpertImpl.reason`` 对称的强制约束：输入闸门
        拦截时检索门面返回 ``is_reviewed=False`` 的空包，若记忆专家照常遍历
        ``pack.evidence``，将产出空 ``stable_facts`` 的上下文包——表现为
        "无记忆可用"的静默降级。此处必须抛错，由编排层 ``_memory_node``
        的异常兜底转 escalate。
        """
        from harness_agent.contracts.retrieval import RetrievalQuery, StoredChunk
        from harness_agent.experts.memory_expert import MemoryExpertImpl

        service = self._stack_service()
        service.index([StoredChunk(chunk_id="kb-1", content="青霉素的皮试要求与用法")])
        expert = MemoryExpertImpl(retrieval=service)
        with pytest.raises(ValueError, match="装配复核"):
            expert.assemble(
                RetrievalQuery(
                    text="青霉素类抗生素怎么用", patient_id="pat-001", session_id="sess-1"
                ),
                SessionContext(patient_id="pat-001"),
            )
