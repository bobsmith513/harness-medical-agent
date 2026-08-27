"""M5 端到端集成测试：取证据 → 推理 → 门禁 → 输出（全链路 mock）。

验收标准（development-plan.md M5）：
1. 端到端一次"取证据 → 推理 → 门禁 → 输出"跑通（mock 模型）；
2. 门禁拦截 badcase 样例可演示。

覆盖场景：
- 正常路径：质量门禁通过 + 输出闸门通过 → 结论交付；
- Badcase 1：LLM-judge 检测到臆测 → 门禁拦截 → interrupt 转人工；
- Badcase 2：LLM-judge 检测到因果倒置 → 门禁拦截 → interrupt 转人工；
- Badcase 3：忠实度低于阈值 → 门禁拦截 → interrupt 转人工；
- Badcase 4：输出闸门拦截过敏药物 → interrupt 转人工；
- Badcase 5：规则前置拦截虚假引用 → 零 LLM 开销拦截；
- 门禁拦截后结论被撤回（finalize 只看到 escalation）。
"""

from __future__ import annotations

import json

from harness_agent.contracts.retrieval import RetrievalQuery
from harness_agent.experts.reasoning_expert import ReasoningExpertImpl
from harness_agent.gates.pipeline import GatePipeline
from harness_agent.gates.quality_judge import LLMJudgeGate
from harness_agent.llm.mock import MockLLMClient
from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import Evidence, EvidencePack, SourceRef
from harness_agent.models.session import SessionContext
from harness_agent.orchestrator import build_orchestrator
from harness_agent.safety import build_safety_stack

PAT_CLEAN = "pat-003"  # M2 种子：无已知过敏
PAT_PENICILLIN = "pat-001"  # M2 种子：青霉素过敏


def _approved_pack(patient_id: str = PAT_CLEAN) -> EvidencePack:
    """已通过装配复核的证据包（固定 evidence_id 便于推理链引用）。"""
    evidence = Evidence(
        evidence_id="ev-1",
        content="阿奇霉素适用于社区获得性肺炎，成人常规剂量 500mg qd",
        source=SourceRef(source_id="s1", source_type="document", chunk_id="kb-1"),
        confidence="medium",
        provenance="knowledge_base",
    )
    return EvidencePack(
        session_id="sess-1",
        patient_id=patient_id,
        query="测试",
        evidence=[evidence],
        assembly_gate=GateVerdict(gate="assembly", allowed=True, reason="复核通过"),
    )


class _StaticPackRetrieval:
    """检索桩：恒定返回预置证据包。"""

    def __init__(self, pack: EvidencePack) -> None:
        self._pack = pack

    def retrieve(self, query: RetrievalQuery) -> EvidencePack:
        return self._pack


def _reasoning_llm_output(citation: str = "ev-1") -> str:
    """推理专家 LLM 的合法 JSON 输出。"""
    return json.dumps(
        {
            "steps": [
                {
                    "kind": "evidence",
                    "text": "引用证据：阿奇霉素适用于社区获得性肺炎",
                    "citations": [citation],
                },
                {"kind": "inference", "text": "基于适应证与剂量信息推断治疗方案"},
                {
                    "kind": "conclusion",
                    "text": "阿奇霉素 500mg qd 可用于社区获得性肺炎",
                    "citations": [citation],
                },
            ],
            "statement": "阿奇霉素 500mg qd 可用于社区获得性肺炎",
            "self_check_notes": "自检通过（3/3）",
        }
    )


def _judge_pass() -> str:
    """judge 通过裁决。"""
    return json.dumps(
        {
            "faithfulness": 0.95,
            "has_hallucination": False,
            "causal_inversion": False,
            "reason": "结论有充分证据支撑",
        }
    )


def _judge_hallucination() -> str:
    """judge 检测到臆测。"""
    return json.dumps(
        {
            "faithfulness": 0.9,
            "has_hallucination": True,
            "causal_inversion": False,
            "reason": "结论引入了证据未提及的推断",
        }
    )


def _judge_causal_inversion() -> str:
    """judge 检测到因果倒置。"""
    return json.dumps(
        {
            "faithfulness": 0.9,
            "has_hallucination": False,
            "causal_inversion": True,
            "reason": "结论先于证据出现",
        }
    )


def _judge_low_faithfulness() -> str:
    """judge 忠实度低于阈值。"""
    return json.dumps(
        {
            "faithfulness": 0.3,
            "has_hallucination": False,
            "causal_inversion": False,
            "reason": "依据不足",
        }
    )


def _context(patient_id: str = PAT_CLEAN) -> SessionContext:
    return SessionContext(session_id="sess-1", patient_id=patient_id)


def _build_agent(
    *,
    reasoning_script: list[str],
    judge_script: list[str],
    pack: EvidencePack | None = None,
    patient_id: str = PAT_CLEAN,
):
    """装配带门禁流水线的主 Agent（真实推理专家 + LLM-judge 门禁）。"""
    reasoning_llm = MockLLMClient(role="reasoning", script=reasoning_script)
    judge_llm = MockLLMClient(role="judge", script=judge_script)

    experts = {
        "reasoning_expert": ReasoningExpertImpl(llm=reasoning_llm),
        "memory_expert": _StubMemoryExpert(),
    }

    retrieval = _StaticPackRetrieval(pack or _approved_pack(patient_id))

    return build_orchestrator(
        experts=experts,
        retrieval=retrieval,
        judge_llm=judge_llm,
    )


class _StubMemoryExpert:
    """记忆专家桩（no_reasoning 路径用）。"""

    name = "memory_expert"

    def assemble(self, query, context):
        from harness_agent.contracts.experts import ContextBundle

        return ContextBundle(patient_id=context.patient_id, stable_facts=["血型 A 型"])


# ---------------------------------------------------------------------------
# 端到端正常路径：取证据 → 推理 → 门禁 → 输出
# ---------------------------------------------------------------------------
class TestE2EHappyPath:
    """验收：端到端一次"取证据 → 推理 → 门禁 → 输出"跑通（mock 模型）。"""

    def test_full_flow_conclusion_delivered(self):
        """质量门禁通过 + 输出闸门通过 → 结论交付。"""
        agent = _build_agent(
            reasoning_script=[_reasoning_llm_output()],
            judge_script=[_judge_pass()],
        )
        result = agent.handle("阿奇霉素的用药剂量怎么定", _context())

        # 路由 → need_reasoning
        assert result.route.decision == "need_reasoning"
        # 结论透传自推理专家
        assert result.conclusion is not None
        assert result.conclusion.produced_by == "reasoning_expert"
        # 门禁裁决附在结果中
        assert len(result.gate_verdicts) == 2
        assert all(v.allowed for v in result.gate_verdicts)
        # 无升级
        assert result.escalation is None

    def test_gate_verdicts_recorded_for_audit(self):
        """门禁裁决全量记录（审计用）。"""
        agent = _build_agent(
            reasoning_script=[_reasoning_llm_output()],
            judge_script=[_judge_pass()],
        )
        result = agent.handle("帮我看看诊断", _context())

        assert len(result.gate_verdicts) == 2
        assert result.gate_verdicts[0].gate == "quality_judge"
        assert result.gate_verdicts[0].allowed is True
        assert result.gate_verdicts[1].gate == "output"
        assert result.gate_verdicts[1].allowed is True

    def test_conclusion_cited_evidence_traceable(self):
        """结论引用可回溯到证据包。"""
        agent = _build_agent(
            reasoning_script=[_reasoning_llm_output()],
            judge_script=[_judge_pass()],
        )
        result = agent.handle("帮我看看诊断", _context())

        pack_ids = {e.evidence_id for e in result.evidence_pack.evidence}
        assert set(result.conclusion.cited_evidence_ids) <= pack_ids


# ---------------------------------------------------------------------------
# Badcase 演示：门禁拦截
# ---------------------------------------------------------------------------
class TestE2EBadcases:
    """验收：门禁拦截 badcase 样例可演示。"""

    def test_badcase_hallucination_intercepted(self):
        """Badcase 1：judge 检测到臆测 → 门禁拦截 → interrupt 转人工。

        推理专家产出合法结论，但 LLM-judge 判定结论引入了证据
        未提及的推断（hallucination）→ 质量门禁拦截 → 结论被撤回、
        转人工 escalation。
        """
        agent = _build_agent(
            reasoning_script=[_reasoning_llm_output()],
            judge_script=[_judge_hallucination()],
        )
        result = agent.handle("帮我看看诊断", _context())

        # 门禁拦截：结论被撤回
        assert result.conclusion is None
        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "quality_judge" in result.escalation.reason
        assert "臆测" in result.escalation.reason
        # 只执行了质量门禁（短路，未到输出闸门）
        assert len(result.gate_verdicts) == 1
        assert result.gate_verdicts[0].allowed is False

    def test_badcase_causal_inversion_intercepted(self):
        """Badcase 2：judge 检测到因果倒置 → 门禁拦截 → interrupt 转人工。

        结论先于证据出现（先下结论后找证据）→ 质量门禁拦截。
        """
        agent = _build_agent(
            reasoning_script=[_reasoning_llm_output()],
            judge_script=[_judge_causal_inversion()],
        )
        result = agent.handle("帮我看看诊断", _context())

        assert result.conclusion is None
        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "因果倒置" in result.escalation.reason
        assert len(result.gate_verdicts) == 1

    def test_badcase_low_faithfulness_intercepted(self):
        """Badcase 3：忠实度低于阈值 → 门禁拦截 → interrupt 转人工。"""
        agent = _build_agent(
            reasoning_script=[_reasoning_llm_output()],
            judge_script=[_judge_low_faithfulness()],
        )
        result = agent.handle("帮我看看诊断", _context())

        assert result.conclusion is None
        assert result.escalation is not None
        assert "忠实度" in result.escalation.reason
        assert len(result.gate_verdicts) == 1

    def test_badcase_allergy_drug_intercepted(self):
        """Badcase 4：输出闸门拦截过敏药物 → interrupt 转人工。

        结论提及患者过敏药物（青霉素）→ 质量门禁通过但输出闸门
        拦截 → 结论被撤回、转人工 escalation。
        """
        # 推理专家产出提及 penicillin 的结论
        reasoning_output = json.dumps(
            {
                "steps": [
                    {"kind": "evidence", "text": "引用证据", "citations": ["ev-1"]},
                    {"kind": "inference", "text": "推断 penicillin 为首选"},
                    {"kind": "conclusion", "text": "建议使用 penicillin", "citations": ["ev-1"]},
                ],
                "statement": "建议使用 penicillin 治疗",
                "self_check_notes": "自检通过",
            }
        )
        # 青霉素过敏患者
        agent = _build_agent(
            reasoning_script=[reasoning_output],
            judge_script=[_judge_pass()],  # 质量门禁通过
            patient_id=PAT_PENICILLIN,
        )
        result = agent.handle("帮我看看诊断", _context(PAT_PENICILLIN))

        # 输出闸门拦截
        assert result.conclusion is None
        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "output" in result.escalation.reason or "gate:output" in result.escalation.reason
        # 两道门禁都执行了（质量通过、输出拦截）
        assert len(result.gate_verdicts) == 2
        assert result.gate_verdicts[0].allowed is True  # 质量门禁通过
        assert result.gate_verdicts[1].allowed is False  # 输出闸门拦截

    def test_badcase_rule_check_blocks_before_llm(self):
        """Badcase 5：推理专家自检拦截虚假引用（fail-closed）。

        LLM 产出引用 ghost-id 的推理链 → 推理专家自检发现引用
        不存在于证据包 → 抛异常 → 编排层 fail-closed 升级转人工。
        这是质量门禁之前的"第一道防线"——推理专家自检。
        """
        # 推理专家产出引用 ghost-id 的结论
        reasoning_output = json.dumps(
            {
                "steps": [
                    {"kind": "evidence", "text": "引用", "citations": ["ghost-id"]},
                    {"kind": "inference", "text": "推断"},
                    {"kind": "conclusion", "text": "结论", "citations": ["ghost-id"]},
                ],
                "statement": "基于不存在的证据",
                "self_check_notes": "自检通过",
            }
        )
        agent = _build_agent(
            reasoning_script=[reasoning_output],
            judge_script=[],  # 空脚本：judge 不应被调用
        )
        result = agent.handle("帮我看看诊断", _context())

        # 推理专家自检失败 → 编排层 fail-closed 升级
        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "自检失败" in result.escalation.reason or "推理专家" in result.escalation.reason
        assert result.conclusion is None


# ---------------------------------------------------------------------------
# 门禁流水线独立单元（不经过编排图）
# ---------------------------------------------------------------------------
class TestGatePipelineStandalone:
    """门禁流水线直接调用（不经过 langgraph 编排图）。"""

    def _conclusion(self, ev_id: str = "ev-1", statement: str = "安全结论"):
        from harness_agent.models.reasoning import (
            ClinicalConclusion,
            ReasoningChain,
            ReasoningStep,
        )

        chain = ReasoningChain(
            steps=[
                ReasoningStep(kind="evidence", text="引用", citations=[ev_id]),
                ReasoningStep(kind="inference", text="推断"),
                ReasoningStep(kind="conclusion", text=statement, citations=[ev_id]),
            ],
            self_check_passed=True,
        )
        return ClinicalConclusion(
            statement=statement, reasoning_chain=chain, cited_evidence_ids=[ev_id]
        )

    def _pipeline(self, judge_script: list[str] | None = None) -> GatePipeline:
        safety = build_safety_stack()
        judge_llm = MockLLMClient(role="judge", script=judge_script or [])
        return GatePipeline(
            quality_gate=LLMJudgeGate(llm=judge_llm),
            output_gate=safety.output_gate,
        )

    def test_pipeline_pass(self):
        pipeline = self._pipeline()
        result = pipeline.run(self._conclusion(), _approved_pack(), _context())
        assert result.allowed is True
        assert len(result.verdicts) == 2

    def test_pipeline_quality_gate_short_circuits(self):
        """质量门禁拦截 → 不执行输出闸门（短路）。"""
        pipeline = self._pipeline(judge_script=[_judge_hallucination()])
        result = pipeline.run(self._conclusion(), _approved_pack(), _context())
        assert result.allowed is False
        assert result.blocking_gate == "gate:quality_judge"
        assert len(result.verdicts) == 1  # 只有质量门禁的裁决

    def test_pipeline_output_gate_intercepts_allergy(self):
        """输出闸门拦截过敏药物（质量门禁通过后）。"""
        from harness_agent.models.reasoning import (
            ClinicalConclusion,
            ReasoningChain,
            ReasoningStep,
        )

        safety = build_safety_stack()
        chain = ReasoningChain(
            steps=[
                ReasoningStep(kind="evidence", text="引用", citations=["ev-1"]),
                ReasoningStep(kind="inference", text="推断 penicillin 可用"),
                ReasoningStep(kind="conclusion", text="建议 penicillin", citations=["ev-1"]),
            ],
            self_check_passed=True,
        )
        conclusion = ClinicalConclusion(
            statement="建议使用 penicillin",
            reasoning_chain=chain,
            cited_evidence_ids=["ev-1"],
        )
        pipeline = GatePipeline(
            quality_gate=LLMJudgeGate(llm=MockLLMClient(role="judge")),
            output_gate=safety.output_gate,
        )
        result = pipeline.run(conclusion, _approved_pack(), _context(PAT_PENICILLIN))
        assert result.allowed is False
        assert result.blocking_gate == "gate:output"
        assert "penicillin" in result.final_verdict.blocked_drugs
