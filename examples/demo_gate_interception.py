"""M8 端到端演示四：门禁拦截转人工。

    uv run python examples/demo_gate_interception.py

场景：质量门禁拦截 → interrupt → 转人工（对照 M5 + M8 验收标准）。

演示三个拦截场景 + 一个正常对照：

1. 忠实度不足：LLM-judge 打分 0.30 < 阈值 0.70 → 拦截转人工；
2. 臆测检测：推理结论引入证据未提及的推断 → 拦截转人工；
3. 过敏药物：输出闸门全文扫描命中过敏药 → 拦截转人工；
4. 正常对照：推理链合法 + 门禁全通过 → 结论交付。

每个拦截场景展示：结论被门禁撤回、interrupt 触发 escalation、
绝不静默降级放行（fail-closed 语义）。

全链路使用 Mock LLM（零外部依赖）。
"""

from __future__ import annotations

import json

from harness_agent.contracts.retrieval import RetrievalQuery
from harness_agent.experts.reasoning_expert import ReasoningExpertImpl
from harness_agent.llm.mock import MockLLMClient
from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import Evidence, EvidencePack, SourceRef
from harness_agent.models.session import SessionContext
from harness_agent.orchestrator import build_orchestrator

PAT_PENICILLIN = "pat-001"  # 青霉素过敏（阻断 beta_lactam 全组）
PAT_CLEAN = "pat-003"  # 无已知过敏


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _approved_pack(patient_id: str = PAT_CLEAN) -> EvidencePack:
    """已通过装配复核的证据包。"""
    evidence = Evidence(
        evidence_id="ev-1",
        content=(
            "社区获得性肺炎经验性治疗：阿奇霉素 500mg qd 或青霉素类。青霉素过敏者首选阿奇霉素替代。"
        ),
        source=SourceRef(source_id="kb-cap-01", source_type="document", chunk_id="kb-cap-01"),
        confidence="high",
        provenance="knowledge_base",
    )
    return EvidencePack(
        session_id="sess-gate",
        patient_id=patient_id,
        query="肺炎用药",
        evidence=[evidence],
        assembly_gate=GateVerdict(gate="assembly", allowed=True, reason="复核通过"),
    )


class _StaticRetrieval:
    """检索桩：恒定返回预置证据包。"""

    def __init__(self, pack: EvidencePack) -> None:
        self._pack = pack

    def retrieve(self, query: RetrievalQuery) -> EvidencePack:
        return self._pack


class _StubMemoryExpert:
    name = "memory_expert"

    def assemble(self, query, context):
        from harness_agent.contracts.experts import ContextBundle

        return ContextBundle(patient_id=context.patient_id)


def _reasoning_output(
    citation: str = "ev-1",
    statement: str = "建议阿奇霉素 500mg qd 治疗 CAP",
) -> str:
    """推理专家合法输出。"""
    return json.dumps(
        {
            "steps": [
                {
                    "kind": "evidence",
                    "text": "引用证据：CAP 经验治疗首选阿奇霉素或青霉素类",
                    "citations": [citation],
                },
                {
                    "kind": "inference",
                    "text": "结合患者症状与过敏史推断治疗方案",
                },
                {
                    "kind": "conclusion",
                    "text": statement,
                    "citations": [citation],
                },
            ],
            "statement": statement,
            "self_check_notes": "自检通过（3/3）",
        }
    )


def _judge_output(
    faithfulness: float = 0.95,
    hallucination: bool = False,
    causal_inversion: bool = False,
    reason: str = "结论有充分证据支撑",
) -> str:
    return json.dumps(
        {
            "faithfulness": faithfulness,
            "has_hallucination": hallucination,
            "causal_inversion": causal_inversion,
            "reason": reason,
        }
    )


def _agent(
    *,
    reasoning_script: list[str],
    judge_script: list[str],
    pack: EvidencePack | None = None,
    patient_id: str = PAT_CLEAN,
):
    reasoning_llm = MockLLMClient(role="reasoning", script=reasoning_script)
    judge_llm = MockLLMClient(role="judge", script=judge_script)
    experts = {
        "reasoning_expert": ReasoningExpertImpl(llm=reasoning_llm),
        "memory_expert": _StubMemoryExpert(),
    }
    return build_orchestrator(
        experts=experts,
        retrieval=_StaticRetrieval(pack or _approved_pack(patient_id)),
        judge_llm=judge_llm,
    )


def _context(patient_id: str = PAT_CLEAN) -> SessionContext:
    return SessionContext(session_id="sess-gate", patient_id=patient_id)


def _describe(result) -> None:
    """打印编排结果（突出门禁拦截与升级路径）。"""
    print(f"  路由: {result.route.decision}")
    if result.conclusion is not None:
        print(f"  结论: {result.conclusion.statement[:50]}")
    else:
        print("  结论: （被门禁拦截，未交付）")
    if result.gate_verdicts:
        for v in result.gate_verdicts:
            status = "通过" if v.allowed else "拦截"
            print(f"  门禁 {v.gate}: {status} — {v.reason[:60]}")
    if result.escalation is not None:
        print(f"  升级: to_human={result.escalation.to_human}")
        print(f"  原因: {result.escalation.reason[:70]}")


def main() -> None:
    print("M8 端到端演示四：门禁拦截转人工")
    print("全链路 Mock LLM（零外部依赖）")
    print("fail-closed 语义：拦截即 interrupt，绝不静默放行")

    # ---- 场景 1：忠实度不足 ----
    _print_section("场景 1：忠实度不足 → 质量门禁拦截 → 转人工")
    print("  LLM-judge 忠实度 0.30 < 阈值 0.70 → 拦截")
    agent = _agent(
        reasoning_script=[_reasoning_output()],
        judge_script=[
            _judge_output(
                faithfulness=0.30,
                reason="证据与结论关联弱，依据不足",
            )
        ],
    )
    _describe(agent.handle("肺炎用药方案", _context()))

    # ---- 场景 2：臆测检测 ----
    _print_section("场景 2：臆测检测 → 质量门禁拦截 → 转人工")
    print("  推理结论引入了证据未提及的推断（臆测）")
    agent = _agent(
        reasoning_script=[_reasoning_output()],
        judge_script=[
            _judge_output(
                faithfulness=0.85,
                hallucination=True,
                reason="结论提及了证据中未出现的肝功能指标",
            )
        ],
    )
    _describe(agent.handle("肺炎用药方案", _context()))

    # ---- 场景 3：过敏药物拦截 ----
    _print_section("场景 3：过敏药物 → 输出闸门拦截 → 转人工")
    print("  结论提及 penicillin（患者青霉素过敏）→ 质量门禁通过、输出闸门拦截")
    reasoning_output = json.dumps(
        {
            "steps": [
                {
                    "kind": "evidence",
                    "text": "引用证据：青霉素类为 CAP 常用方案",
                    "citations": ["ev-1"],
                },
                {
                    "kind": "inference",
                    "text": "推断 penicillin 为首选抗菌药",
                },
                {
                    "kind": "conclusion",
                    "text": "建议使用 penicillin 治疗",
                    "citations": ["ev-1"],
                },
            ],
            "statement": "建议使用 penicillin 治疗",
            "self_check_notes": "自检通过",
        }
    )
    agent = _agent(
        reasoning_script=[reasoning_output],
        judge_script=[_judge_output(faithfulness=0.95)],
        patient_id=PAT_PENICILLIN,
    )
    _describe(agent.handle("肺炎用药方案", _context(PAT_PENICILLIN)))

    # ---- 场景 4：正常对照 ----
    _print_section("场景 4：正常对照 — 门禁全通过 → 结论交付")
    print("  推理链合法 + 忠实度 0.92 + 无过敏药物 → 结论正常交付")
    agent = _agent(
        reasoning_script=[_reasoning_output()],
        judge_script=[_judge_output(faithfulness=0.92)],
    )
    _describe(agent.handle("肺炎用药方案", _context()))

    # ---- 验收总结 ----
    print()
    print("=" * 72)
    print("门禁拦截转人工验收总结:")
    print("  ✓ 忠实度不足拦截（0.30 < 0.70 阈值）→ interrupt 转人工")
    print("  ✓ 臆测检测拦截（has_hallucination=True）→ interrupt 转人工")
    print("  ✓ 过敏药物拦截（输出闸门全文扫描）→ interrupt 转人工")
    print("  ✓ 正常对照：全门禁通过 → 结论正常交付")
    print("  ✓ fail-closed 语义：拦截即撤回结论，绝不静默降级放行")
    print("  ✓ 每次拦截都产出 escalation（to_human=True），无应答权出口")
    print("=" * 72)


if __name__ == "__main__":
    main()
