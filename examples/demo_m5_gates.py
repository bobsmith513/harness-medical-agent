"""M5 推理专家与质量门禁演示：取证据 → 推理 → 门禁 → 输出（全链路 mock）。

    uv run python examples/demo_m5_gates.py

演示六个场景（对照 development-plan.md M5 验收标准）：

1. 正常路径：推理专家生成合法推理链 → 质量门禁通过 → 输出闸门通过 → 结论交付；
2. Badcase - 臆测拦截：LLM-judge 检测到臆测 → 质量门禁拦截 → interrupt 转人工；
3. Badcase - 因果倒置：LLM-judge 检测到因果倒置 → 质量门禁拦截 → interrupt 转人工；
4. Badcase - 忠实度不足：忠实度低于阈值 → 质量门禁拦截 → interrupt 转人工；
5. Badcase - 过敏药物：输出闸门拦截过敏药物 → interrupt 转人工；
6. Badcase - 自检拦截：推理专家自检发现虚假引用 → fail-closed 升级。

全链路使用 Mock LLM（零外部依赖），端到端可演示。
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

PAT_CLEAN = "pat-003"  # M2 种子：无已知过敏
PAT_PENICILLIN = "pat-001"  # M2 种子：青霉素过敏（阻断 beta_lactam 全组）


# ---------------------------------------------------------------------------
# 共享辅助
# ---------------------------------------------------------------------------
def _approved_pack(patient_id: str = PAT_CLEAN) -> EvidencePack:
    """已通过装配复核的证据包（固定 evidence_id 便于引用）。"""
    evidence = Evidence(
        evidence_id="ev-1",
        content="阿奇霉素适用于社区获得性肺炎，成人常规剂量 500mg qd",
        source=SourceRef(source_id="s1", source_type="document", chunk_id="kb-1"),
        confidence="medium",
        provenance="knowledge_base",
    )
    return EvidencePack(
        session_id="sess-demo",
        patient_id=patient_id,
        query="演示",
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
    """记忆专家桩（no_reasoning 路径用）。"""

    name = "memory_expert"

    def assemble(self, query, context):
        from harness_agent.contracts.experts import ContextBundle

        return ContextBundle(patient_id=context.patient_id, stable_facts=["血型 A 型"])


def _reasoning_output(
    citation: str = "ev-1",
    statement: str = "阿奇霉素 500mg qd 可用于社区获得性肺炎",
) -> str:
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
                {"kind": "conclusion", "text": statement, "citations": [citation]},
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
    """judge LLM 的 JSON 输出。"""
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
    """装配带门禁流水线的主 Agent。"""
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
    return SessionContext(session_id="sess-demo", patient_id=patient_id)


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _describe(result) -> None:
    """打印编排结果。"""
    print(f"  路由: {result.route.decision} (by_rule={result.route.by_rule})")
    print(f"  任务清单: {[t.expert for t in result.tasks] or '（无，升级路径）'}")
    if result.evidence_pack is not None:
        pack = result.evidence_pack
        print(f"  证据包: {len(pack.evidence)} 条, is_reviewed={pack.is_reviewed}")
    if result.conclusion is not None:
        print(f"  结论: {result.conclusion.statement[:60]}")
        chain = result.conclusion.reasoning_chain
        print(f"  推理链: {len(chain.steps)} 步, 自检={chain.self_check_passed}")
        print(
            f"  结论溯源: produced_by={result.conclusion.produced_by}, "
            f"引用={result.conclusion.cited_evidence_ids}"
        )
    else:
        print("  结论: （无——被门禁拦截或推理失败）")
    if result.gate_verdicts:
        for v in result.gate_verdicts:
            status = "通过" if v.allowed else "拦截"
            print(f"  门禁 {v.gate}: {status} — {v.reason[:50]}")
    else:
        print("  门禁: （未执行——推理前失败）")
    if result.escalation is not None:
        print(f"  升级: to_human={result.escalation.to_human}")
        print(f"  原因: {result.escalation.reason[:70]}")


# ---------------------------------------------------------------------------
# 演示场景
# ---------------------------------------------------------------------------
def main() -> None:
    print("M5 推理专家与质量门禁演示")
    print("全链路 Mock LLM（零外部依赖）")

    # 场景 1：正常路径
    _print_section("场景 1：正常路径 — 取证据 → 推理 → 门禁 → 输出")
    print("  推理专家生成合法推理链 → 质量门禁通过 → 输出闸门通过 → 结论交付")
    agent = _agent(
        reasoning_script=[_reasoning_output()],
        judge_script=[_judge_output(faithfulness=0.95)],
    )
    _describe(agent.handle("阿奇霉素的用药剂量怎么定", _context()))

    # 场景 2：臆测拦截
    _print_section("场景 2：Badcase — LLM-judge 检测到臆测")
    print("  推理专家产出合法结论，但 judge 判定结论引入了证据未提及的推断")
    agent = _agent(
        reasoning_script=[_reasoning_output()],
        judge_script=[
            _judge_output(
                faithfulness=0.9,
                hallucination=True,
                reason="结论引入了证据未提及的推断（臆测）",
            )
        ],
    )
    _describe(agent.handle("帮我看看诊断", _context()))

    # 场景 3：因果倒置
    _print_section("场景 3：Badcase — LLM-judge 检测到因果倒置")
    print("  结论先于证据出现（先下结论后找证据）→ 质量门禁拦截")
    agent = _agent(
        reasoning_script=[_reasoning_output()],
        judge_script=[
            _judge_output(
                faithfulness=0.9,
                causal_inversion=True,
                reason="结论先于证据出现",
            )
        ],
    )
    _describe(agent.handle("帮我看看诊断", _context()))

    # 场景 4：忠实度不足
    _print_section("场景 4：Badcase — 忠实度低于阈值")
    print("  忠实度 0.30 < 阈值 0.70 → 质量门禁拦截")
    agent = _agent(
        reasoning_script=[_reasoning_output()],
        judge_script=[
            _judge_output(
                faithfulness=0.30,
                reason="依据不足，证据与结论关联弱",
            )
        ],
    )
    _describe(agent.handle("帮我看看诊断", _context()))

    # 场景 5：过敏药物拦截
    _print_section("场景 5：Badcase — 输出闸门拦截过敏药物")
    print("  结论提及 penicillin（患者青霉素过敏）→ 质量门禁通过、输出闸门拦截")
    reasoning_output = json.dumps(
        {
            "steps": [
                {"kind": "evidence", "text": "引用证据", "citations": ["ev-1"]},
                {"kind": "inference", "text": "推断 penicillin 为首选抗菌药"},
                {"kind": "conclusion", "text": "建议使用 penicillin", "citations": ["ev-1"]},
            ],
            "statement": "建议使用 penicillin 治疗",
            "self_check_notes": "自检通过",
        }
    )
    agent = _agent(
        reasoning_script=[reasoning_output],
        judge_script=[_judge_output(faithfulness=0.95)],  # 质量门禁通过
        patient_id=PAT_PENICILLIN,
    )
    _describe(agent.handle("帮我看看诊断", _context(PAT_PENICILLIN)))

    # 场景 6：自检拦截
    _print_section("场景 6：Badcase — 推理专家自检拦截虚假引用")
    print("  LLM 产出引用 ghost-id 的推理链 → 自检发现引用不存在 → fail-closed 升级")
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
    agent = _agent(
        reasoning_script=[reasoning_output],
        judge_script=[],  # judge 不应被调用（自检先拦）
    )
    _describe(agent.handle("帮我看看诊断", _context()))

    print()
    print("=" * 72)
    print("M5 验收总结：")
    print("  - 端到端'取证据 → 推理 → 门禁 → 输出'跑通（mock 模型）")
    print("  - 门禁拦截 badcase 样例可演示（臆测/因果倒置/忠实度/过敏药物/自检）")
    print("  - 拦截即 interrupt 转人工，结论被门禁撤回，绝不静默放行")
    print("=" * 72)


if __name__ == "__main__":
    main()
