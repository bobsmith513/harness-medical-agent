"""M4 编排层演示：路由 → 规划 → 委派全链路（零依赖 mock 栈）。

    uv run python examples/demo_orchestrator.py

演示五个场景：
1. 规则路由 + 推理委派：诊断类查询 → 检索（M3）→ 推理专家（桩）→ 结论透传；
2. LLM 兜底路由 + 记忆委派：模糊查询 → 记忆专家（桩）装配上下文；
3. 误判二次路由：首次 LLM 输出不可解析 → 纠错重试成功；
4. fail-closed 升级：两次误判 → 转澄清（无应答权出口）；
5. 装配闸门拦截：青霉素过敏患者 → 检索在入口被闸门拦下，转人工。

专家为桩实现（M5 接入真实推理专家后本 demo 零改动复用）。
"""

from __future__ import annotations

from harness_agent.contracts.experts import (
    ContextBundle,
    ExpertTask,
)
from harness_agent.contracts.retrieval import RetrievalQuery, StoredChunk
from harness_agent.llm.mock import MockLLMClient
from harness_agent.models.evidence import EvidencePack
from harness_agent.models.reasoning import (
    ClinicalConclusion,
    ReasoningChain,
    ReasoningStep,
)
from harness_agent.models.session import SessionContext
from harness_agent.orchestrator import OrchestrationResult, build_orchestrator
from harness_agent.retrieval.wiring import build_retrieval_stack
from harness_agent.safety import build_safety_stack

PAT_CLEAN = "pat-003"  # M2 种子：无已知过敏
PAT_PENICILLIN = "pat-001"  # M2 种子：青霉素过敏（阻断 beta_lactam 全组）


# ---------------------------------------------------------------------------
# 桩专家（M5 换真实实现，编排层零改动）
# ---------------------------------------------------------------------------
class DemoReasoningExpert:
    """推理专家桩：以证据包为据构造合法结论（自检通过的三段式推理链）。"""

    def __init__(self) -> None:
        self.name = "reasoning_expert"

    def reason(
        self, task: ExpertTask, evidence: EvidencePack, context: SessionContext
    ) -> ClinicalConclusion:
        citation = evidence.evidence[0].evidence_id if evidence.evidence else ""
        chain = ReasoningChain(
            steps=[
                ReasoningStep(
                    kind="evidence", text=f"引用证据 {citation[:14]}", citations=[citation]
                ),
                ReasoningStep(kind="inference", text="依据证据内容逐步推断（省略展开）"),
                ReasoningStep(kind="conclusion", text="综合形成结论"),
            ],
            self_check_passed=True,
            self_check_notes="demo 桩自检通过",
        )
        return ClinicalConclusion(
            statement="（demo 桩结论）建议结合完整病史与检验结果，由医生确认。",
            reasoning_chain=chain,
            cited_evidence_ids=[citation] if citation else [],
        )


class DemoMemoryExpert:
    """记忆专家桩：装配患者上下文（稳定/易变事实 + 过敏史硬规则）。"""

    def __init__(self) -> None:
        self.name = "memory_expert"

    def assemble(self, query: RetrievalQuery, context: SessionContext) -> ContextBundle:
        safety = build_safety_stack()
        return ContextBundle(
            patient_id=context.patient_id,
            allergies=safety.allergy_store.get(context.patient_id),
            stable_facts=["血型 A 型", "阑尾切除术后（2023-05）"],
            volatile_facts=["近期服用二甲双胍 500mg bid"],
        )


def _describe(result: OrchestrationResult) -> None:
    print(
        f"  路由: {result.route.decision} (by_rule={result.route.by_rule}, "
        f"attempt={result.route.attempt})"
    )
    print(f"  任务清单: {[t.expert for t in result.tasks] or '（无，升级路径）'}")
    if result.evidence_pack is not None:
        pack = result.evidence_pack
        print(f"  证据包: {len(pack.evidence)} 条, is_reviewed={pack.is_reviewed}")
    if result.conclusion is not None:
        print(f"  结论: {result.conclusion.statement[:44]}")
        print(
            f"  结论溯源: produced_by={result.conclusion.produced_by}, "
            f"citations={len(result.conclusion.cited_evidence_ids)} 条"
        )
    if result.context_bundle is not None:
        bundle = result.context_bundle
        print(
            f"  上下文包: 过敏史 {len(bundle.allergies)} 条 / 稳定事实 "
            f"{len(bundle.stable_facts)} 条 / 易变事实 {len(bundle.volatile_facts)} 条"
        )
    if result.escalation is not None:
        print(f"  升级: {result.escalation.reason[:56]}")
        if result.escalation.clarification_question:
            print(f"  澄清问句: {result.escalation.clarification_question}")


def main() -> None:
    # 共享检索栈：先入库知识库条目，供推理路径检索
    stack = build_retrieval_stack()
    stack.service.index(
        [
            StoredChunk(
                chunk_id="kb-demo-1",
                content="阿奇霉素的适应证与常规剂量说明",
                patient_id=None,
            )
        ]
    )

    def _agent(script: list[str] | None = None):
        return build_orchestrator(
            experts={
                "reasoning_expert": DemoReasoningExpert(),
                "memory_expert": DemoMemoryExpert(),
            },
            router_llm=MockLLMClient(role="router", script=script or []),
            retrieval=stack.service,  # 注入共享栈（已入库）
        )

    context = SessionContext(session_id="sess-demo", patient_id=PAT_CLEAN)

    print("=" * 72)
    print("场景 1：规则路由 + 推理委派（诊断类查询，零 LLM 开销）")
    print("=" * 72)
    _describe(_agent().handle("阿奇霉素的用药剂量怎么定", context))

    print()
    print("=" * 72)
    print("场景 2：LLM 兜底路由 + 记忆委派（模糊查询走兜底判无需推理）")
    print("=" * 72)
    _describe(
        _agent(script=['{"decision": "no_reasoning"}']).handle(
            "最近老睡不着白天没精神，想先了解下之前的情况", context
        )
    )

    print()
    print("=" * 72)
    print("场景 3：误判二次路由（首次输出不可解析 → 纠错重试成功）")
    print("=" * 72)
    _describe(
        _agent(script=["这个嘛不好说呢", '{"decision": "need_reasoning"}']).handle(
            "帮我看个事", context
        )
    )

    print()
    print("=" * 72)
    print("场景 4：fail-closed 升级（两次误判 → 转澄清，无应答权出口）")
    print("=" * 72)
    _describe(_agent(script=["嗯嗯嗯", "看不懂"]).handle("帮我看个事", context))

    print()
    print("=" * 72)
    print("场景 5：装配闸门拦截（青霉素过敏患者 → 检索在入口被拦，转人工）")
    print("=" * 72)
    pat_penicillin = SessionContext(session_id="sess-demo-2", patient_id=PAT_PENICILLIN)
    _describe(_agent().handle("青霉素类药物的剂量怎么定", pat_penicillin))


if __name__ == "__main__":
    main()
