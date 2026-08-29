"""M4 主 Agent 编排测试：langgraph 全链路 + 无应答权 + fail-closed。

场景覆盖（委派 demo 的单测锁定）：
1. need_reasoning：路由 → 规划 → 检索（M3 真实栈）→ 委派推理专家 → 结论透传；
2. no_reasoning：路由 → 规划 → 委派记忆专家 → 上下文包透传；
3. escalate：两次 LLM 误判 → 升级请求（无应答权出口）；
4. 证据包未复核 → 推理前置校验 fail-closed；
5. 专家未绑定运行时实现 → fail-closed；
6. 主 Agent 无法构造临床结论（类型级无应答权验证）。
"""

from __future__ import annotations

import pytest

from harness_agent.contracts.experts import (
    ContextBundle,
    ExpertTask,
    MemoryExpert,
    ReasoningExpert,
)
from harness_agent.contracts.retrieval import RetrievalQuery, RetrievalService
from harness_agent.gates.pipeline import GatePipeline, GatePipelineResult
from harness_agent.llm.mock import MockLLMClient
from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import Evidence, EvidencePack, SourceRef
from harness_agent.models.reasoning import (
    ClinicalConclusion,
    ReasoningChain,
    ReasoningStep,
)
from harness_agent.models.session import SessionContext
from harness_agent.orchestrator import (
    HarnessOrchestrator,
    TaskPlanner,
    load_experts,
)
from harness_agent.orchestrator.router import BinaryRouter, LLMRouter, RuleRouter
from harness_agent.retrieval.wiring import build_retrieval_stack

PAT_CLEAN = "pat-003"  # M2 种子：无已知过敏


# ---------------------------------------------------------------------------
# 测试桩
# ---------------------------------------------------------------------------
class StubReasoningExpert:
    """推理专家桩：以证据包首条证据为据构造合法结论（记录委派参数）。"""

    def __init__(self) -> None:
        self.name = "reasoning_expert"
        self.received_tasks: list[ExpertTask] = []
        self.received_packs: list[EvidencePack] = []

    def reason(self, task: ExpertTask, evidence: EvidencePack, context: SessionContext):
        self.received_tasks.append(task)
        self.received_packs.append(evidence)
        citation = evidence.evidence[0].evidence_id if evidence.evidence else ""
        chain = ReasoningChain(
            steps=[
                ReasoningStep(kind="evidence", text="引用证据", citations=[citation]),
                ReasoningStep(kind="inference", text="依据证据推断"),
                ReasoningStep(kind="conclusion", text="得出结论"),
            ],
            self_check_passed=True,
            self_check_notes="桩自检通过",
        )
        return ClinicalConclusion(
            statement="基于证据的桩结论",
            reasoning_chain=chain,
            cited_evidence_ids=[citation] if citation else [],
        )


class StubMemoryExpert:
    """记忆专家桩：装配固定上下文包（记录委派调用）。"""

    def __init__(self) -> None:
        self.name = "memory_expert"
        self.call_count = 0

    def assemble(self, query: RetrievalQuery, context: SessionContext) -> ContextBundle:
        self.call_count += 1
        return ContextBundle(
            patient_id=context.patient_id,
            stable_facts=["血型 A 型"],
            volatile_facts=["近期服用二甲双胍"],
        )


class _StaticPackRetrieval:
    """检索桩：恒定返回预置证据包（测 fail-closed 分支用）。"""

    def __init__(self, pack: EvidencePack) -> None:
        self._pack = pack

    def retrieve(self, query: RetrievalQuery) -> EvidencePack:
        return self._pack


class _PassingGatePipeline:
    """门禁流水线桩：恒定放行（记录调用次数，用于验证 gates 节点真的执行了）。"""

    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        conclusion: ClinicalConclusion,
        evidence: EvidencePack,
        context: SessionContext,
    ) -> GatePipelineResult:
        self.calls += 1
        return GatePipelineResult(
            allowed=True,
            verdicts=[GateVerdict(gate="quality_judge", allowed=True, reason="桩门禁放行")],
        )


class _BlockingGatePipeline:
    """门禁流水线桩：恒定拦截（模拟 quality_judge 忠实度不足）。"""

    def run(
        self,
        conclusion: ClinicalConclusion,
        evidence: EvidencePack,
        context: SessionContext,
    ) -> GatePipelineResult:
        return GatePipelineResult(
            allowed=False,
            verdicts=[
                GateVerdict(gate="quality_judge", allowed=False, reason="桩门禁拦截：忠实度 0.30")
            ],
            blocking_gate="gate:quality_judge",
        )


class _ExplodingGatePipeline:
    """门禁流水线桩：执行即抛异常（模拟 judge 在线端点 502 / 超时）。"""

    def run(
        self,
        conclusion: ClinicalConclusion,
        evidence: EvidencePack,
        context: SessionContext,
    ) -> GatePipelineResult:
        raise RuntimeError("judge 端点不可用")


class _ExplodingRouter:
    """路由桩：route 即抛异常（模拟在线 API 401 / 连接超时）。"""

    def route(self, user_input: str, context: SessionContext):
        raise RuntimeError("路由 LLM 端点 401")


class _EmptyPlanner:
    """规划器桩：恒返回空任务清单（触发"任务清单缺失"fail-closed 分支）。"""

    def plan(self, user_input: str, route, context: SessionContext) -> list[ExpertTask]:
        return []


class _ExplodingMemoryExpert:
    """记忆专家桩：assemble 即抛异常（模拟记忆检索后端故障）。"""

    name = "memory_expert"

    def assemble(self, query: RetrievalQuery, context: SessionContext) -> ContextBundle:
        raise RuntimeError("记忆检索不可用")


def _approved_pack() -> EvidencePack:
    """已通过装配复核的证据包（一条知识库证据）。"""
    evidence = Evidence(
        content="阿奇霉素的适应证说明",
        source=SourceRef(source_id="s1", source_type="document", chunk_id="kb-1"),
        confidence="medium",
        provenance="knowledge_base",
    )
    return EvidencePack(
        session_id="sess-1",
        patient_id=PAT_CLEAN,
        query="测试",
        evidence=[evidence],
        assembly_gate=GateVerdict(gate="assembly", allowed=True, reason="复核通过"),
    )


def _orchestrator(
    *,
    llm_script: list[str] | None = None,
    retrieval: RetrievalService | None = None,
    experts: dict | None = None,
    router: BinaryRouter | None = None,
    planner: TaskPlanner | None = None,
    gate_pipeline: GatePipeline | None = None,
) -> HarnessOrchestrator:
    """装配测试用主 Agent（默认零依赖 mock 栈 + 桩专家）。

    ``gate_pipeline`` 默认为 ``None``——与生产装配的差异：
    ``orchestrator.wiring.build_orchestrator`` 会自动装配真实流水线，
    而这里保留 None 是为了沿用 M4 的「未接门禁」基线用例。
    测 gates 节点时必须显式传入（见 ``TestGatePipelineWiring``）。
    """
    registry = load_experts()
    if router is None:
        llm = MockLLMClient(role="router", script=llm_script or [])
        router = BinaryRouter(rule_router=RuleRouter(), llm_router=LLMRouter(client=llm))
    return HarnessOrchestrator(
        router=router,
        planner=planner if planner is not None else TaskPlanner(registry),
        retrieval=retrieval if retrieval is not None else build_retrieval_stack().service,
        registry=registry,
        experts=experts
        if experts is not None
        else {"reasoning_expert": StubReasoningExpert(), "memory_expert": StubMemoryExpert()},
        gate_pipeline=gate_pipeline,
    )


def _context() -> SessionContext:
    return SessionContext(session_id="sess-1", patient_id=PAT_CLEAN)


# ---------------------------------------------------------------------------
# 契约满足性
# ---------------------------------------------------------------------------
class TestContracts:
    def test_stub_experts_satisfy_protocols(self):
        assert isinstance(StubReasoningExpert(), ReasoningExpert)
        assert isinstance(StubMemoryExpert(), MemoryExpert)


# ---------------------------------------------------------------------------
# need_reasoning 全链路（验收：委派 demo 跑通）
# ---------------------------------------------------------------------------
class TestReasoningPath:
    def test_rule_routed_query_delegates_to_reasoning_expert(self):
        agent = _orchestrator()
        result = agent.handle("阿奇霉素的用药剂量怎么定", _context())

        assert result.route.decision == "need_reasoning"
        assert result.route.by_rule is True
        # 任务清单含 reasoning 委派（指令已填充用户问题）
        assert any(t.expert == "reasoning_expert" for t in result.tasks)
        task = next(t for t in result.tasks if t.expert == "reasoning_expert")
        assert "阿奇霉素" in task.instruction
        # 结论透传自推理专家（唯一合法来源）
        assert result.conclusion is not None
        assert result.conclusion.produced_by == "reasoning_expert"
        assert result.escalation is None
        assert result.context_bundle is None

    def test_conclusion_citations_trace_back_to_evidence(self):
        """结论引用可回溯：cited_evidence_ids ⊆ 证据包 evidence_id 集。"""
        agent = _orchestrator(retrieval=_StaticPackRetrieval(_approved_pack()))
        result = agent.handle("帮我看看诊断", _context())
        pack_ids = {e.evidence_id for e in result.evidence_pack.evidence}
        assert result.conclusion is not None
        assert set(result.conclusion.cited_evidence_ids) <= pack_ids

    def test_result_constructed_via_delegation_only(self):
        """OrchestrationResult 无应答权：结论字段只能透传专家产出。"""
        agent = _orchestrator(retrieval=_StaticPackRetrieval(_approved_pack()))
        result = agent.handle("帮我看看诊断", _context())
        # 结论溯源：推理链自检通过（主 Agent 无法伪造）
        assert result.conclusion.reasoning_chain.self_check_passed is True


# ---------------------------------------------------------------------------
# no_reasoning 链路
# ---------------------------------------------------------------------------
class TestMemoryPath:
    def test_context_query_delegates_to_memory_expert(self):
        agent = _orchestrator()
        result = agent.handle("我上次说过什么", _context())

        assert result.route.decision == "no_reasoning"
        assert result.context_bundle is not None
        assert result.context_bundle.patient_id == PAT_CLEAN
        assert result.conclusion is None
        assert result.escalation is None

    def test_llm_routed_context_query_also_reaches_memory(self):
        """规则未命中 + LLM 兜底判 no_reasoning → 同样委派记忆专家。"""
        agent = _orchestrator(llm_script=['{"decision": "no_reasoning"}'])
        result = agent.handle("最近老睡不着白天没精神想问问之前的情况", _context())
        assert result.route.by_rule is False
        assert result.route.decision == "no_reasoning"
        assert result.context_bundle is not None


# ---------------------------------------------------------------------------
# escalate 链路（无应答权出口）
# ---------------------------------------------------------------------------
class TestEscalationPath:
    def test_double_misjudge_escalates_with_clarification(self):
        agent = _orchestrator(llm_script=["答非所问", "还是不行"])
        result = agent.handle("帮我看个事", _context())

        assert result.route.decision == "escalate"
        assert result.escalation is not None
        assert result.escalation.clarification_question  # 澄清问句非空
        assert result.conclusion is None
        assert result.context_bundle is None
        assert result.tasks == []  # escalate 无委派任务

    def test_unreviewed_evidence_fails_closed_before_reasoning(self):
        """证据包未复核（is_reviewed=False）→ 不进推理，fail-closed 升级。"""
        unreviewed = _approved_pack()
        unreviewed.assembly_gate = None  # 无复核裁决
        agent = _orchestrator(retrieval=_StaticPackRetrieval(unreviewed))
        result = agent.handle("帮我看看诊断", _context())

        assert result.escalation is not None
        assert "装配复核" in result.escalation.reason
        assert result.conclusion is None

    def test_unbound_reasoning_expert_fails_closed(self):
        """注册表有声明但运行时未绑定实现 → fail-closed（结论必须被撤回）。"""
        agent = _orchestrator(experts={"memory_expert": StubMemoryExpert()})
        result = agent.handle("帮我看看诊断", _context())
        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "未绑定" in result.escalation.reason
        assert result.conclusion is None

    def test_unbound_memory_expert_fails_closed(self):
        agent = _orchestrator(experts={"reasoning_expert": StubReasoningExpert()})
        result = agent.handle("我上次说过什么", _context())
        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "未绑定" in result.escalation.reason
        assert result.context_bundle is None

    def test_retrieval_exception_fails_closed(self):
        """检索异常 → 升级人工，绝不含空证据臆造结论。"""

        class _ExplodingRetrieval:
            def retrieve(self, query: RetrievalQuery) -> EvidencePack:
                raise RuntimeError("检索服务不可用")

        agent = _orchestrator(retrieval=_ExplodingRetrieval())  # type: ignore[arg-type]
        result = agent.handle("帮我看看诊断", _context())
        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert result.conclusion is None


# ---------------------------------------------------------------------------
# 门禁节点接线（gates 节点此前在本文件中 0 覆盖：默认装配未传 gate_pipeline）
# ---------------------------------------------------------------------------
class TestGatePipelineWiring:
    def test_gates_node_runs_when_pipeline_bound(self):
        """绑定流水线后 gates 节点必须真的执行（此前 14/15 用例跳过该节点）。"""
        gates = _PassingGatePipeline()
        agent = _orchestrator(retrieval=_StaticPackRetrieval(_approved_pack()), gate_pipeline=gates)
        result = agent.handle("帮我看看诊断", _context())

        assert result.conclusion is not None
        assert gates.calls == 1
        assert result.escalation is None
        assert [v.gate for v in result.gate_verdicts] == ["quality_judge"]

    def test_gate_interception_withdraws_conclusion(self):
        """门禁拦截 → 结论被撤回 + 转人工（fail-closed 核心语义）。"""
        agent = _orchestrator(
            retrieval=_StaticPackRetrieval(_approved_pack()),
            gate_pipeline=_BlockingGatePipeline(),
        )
        result = agent.handle("帮我看看诊断", _context())

        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "门禁拦截" in result.escalation.reason
        assert "gate:quality_judge" in result.escalation.reason
        assert result.conclusion is None  # 拦截即撤回，绝不静默放行
        assert result.gate_verdicts and result.gate_verdicts[-1].allowed is False

    def test_no_pipeline_means_no_gate_verdicts(self):
        """未绑定流水线时门禁节点空转（M4 基线语义，不是放行语义）。"""
        agent = _orchestrator(retrieval=_StaticPackRetrieval(_approved_pack()))
        result = agent.handle("帮我看看诊断", _context())
        assert result.conclusion is not None
        assert not result.gate_verdicts


# ---------------------------------------------------------------------------
# fail-closed 出口补齐（agent.py 中此前零覆盖的 5 条异常/缺失分支）
# ---------------------------------------------------------------------------
class TestFailClosedExits:
    """每条出口都要断言完整语义：to_human=True **且** 结论被撤回。"""

    def test_router_exception_escalates_to_human(self):
        """路由节点异常（在线 API 401/超时）→ 转人工，不裸抛。"""
        agent = _orchestrator(router=_ExplodingRouter())  # type: ignore[arg-type]
        result = agent.handle("帮我看看诊断", _context())

        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "路由器执行异常" in result.escalation.reason
        assert result.conclusion is None

    def test_missing_reasoning_task_escalates_to_human(self):
        """任务清单缺 reasoning 委派（结论唯一来源）→ 转人工。"""
        agent = _orchestrator(planner=_EmptyPlanner())  # type: ignore[arg-type]
        result = agent.handle("帮我看看诊断", _context())

        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "任务清单缺失 reasoning 专家任务" in result.escalation.reason
        assert result.conclusion is None

    def test_missing_memory_task_escalates_to_human(self):
        """任务清单缺 memory 委派 → 转人工，不降级为主 Agent 应答。"""
        agent = _orchestrator(planner=_EmptyPlanner())  # type: ignore[arg-type]
        result = agent.handle("我上次说过什么", _context())

        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "任务清单缺失 memory 专家任务" in result.escalation.reason
        assert result.context_bundle is None

    def test_gate_pipeline_exception_escalates_to_human(self):
        """门禁流水线自身异常（judge 端点故障）→ 转人工，绝不静默放行。"""
        agent = _orchestrator(
            retrieval=_StaticPackRetrieval(_approved_pack()),
            gate_pipeline=_ExplodingGatePipeline(),  # type: ignore[arg-type]
        )
        result = agent.handle("帮我看看诊断", _context())

        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "门禁流水线执行异常" in result.escalation.reason
        assert result.conclusion is None

    def test_memory_expert_exception_escalates_to_human(self):
        """记忆专家执行异常 → 转人工。"""
        agent = _orchestrator(
            experts={
                "reasoning_expert": StubReasoningExpert(),
                "memory_expert": _ExplodingMemoryExpert(),
            }
        )
        result = agent.handle("我上次说过什么", _context())

        assert result.escalation is not None
        assert result.escalation.to_human is True
        assert "记忆专家执行异常" in result.escalation.reason
        assert result.context_bundle is None


# ---------------------------------------------------------------------------
# 任务规划器
# ---------------------------------------------------------------------------
class TestTaskPlanner:
    def test_need_reasoning_plans_reasoning_tasks(self):
        from harness_agent.models.session import RouteRecord

        planner = TaskPlanner(load_experts())
        tasks = planner.plan("怎么治疗", RouteRecord(decision="need_reasoning"), _context())
        assert [t.expert for t in tasks] == ["reasoning_expert"]

    def test_no_reasoning_plans_memory_tasks(self):
        from harness_agent.models.session import RouteRecord

        planner = TaskPlanner(load_experts())
        tasks = planner.plan("上次说了啥", RouteRecord(decision="no_reasoning"), _context())
        assert [t.expert for t in tasks] == ["memory_expert"]

    def test_escalate_plans_empty(self):
        from harness_agent.models.session import RouteRecord

        planner = TaskPlanner(load_experts())
        assert planner.plan("任意", RouteRecord(decision="escalate"), _context()) == []

    def test_missing_reasoning_kind_rejected(self):
        """注册表缺 reasoning 专家（临床结论唯一来源）→ 装配期报错。"""
        from harness_agent.models.session import RouteRecord

        registry = load_experts()
        registry = registry.model_copy(
            update={"experts": [s for s in registry.experts if s.kind != "reasoning"]}
        )
        planner = TaskPlanner(registry)
        with pytest.raises(KeyError, match="reasoning"):
            planner.plan("怎么治疗", RouteRecord(decision="need_reasoning"), _context())
