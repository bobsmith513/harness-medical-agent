"""主 Agent 编排器（M4+M5）：langgraph StateGraph，路由 + 委派 + 门禁。

图结构（每节点单一职责，条件边按路由裁决分流）::

    START → route → plan ─┬─ need_reasoning → retrieve → reason → gates ─┐
                          ├─ no_reasoning ──── memory ────────────────────┤→ finalize → END
                          └─ escalate ─────── escalate_node ───────────────┘

**无应答权的结构级落地**：

- 全图不存在"生成临床结论"的节点——``reason`` 节点只把推理专家的
  返回值透传进 state，``finalize`` 只组装不生产；
- 推理专家委派前置校验 ``evidence_pack.is_reviewed``（M3 输入闸门
  拦截的空包在此 fail-closed 转 escalate，不进推理）；
- **M5 门禁流水线**：``gates`` 节点串联质量门禁 → 输出闸门，
  拦截即 interrupt（转人工），结论被门禁拦截后不得交付；
- 注册表声明的专家若未绑定运行时实现 → fail-closed 升级（配置与
  运行时的一致性在委派前检查）；
- 检索失败（异常）→ escalate，绝不以空证据包臆造结论。
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from harness_agent.contracts.experts import (
    ContextBundle,
    ExpertTask,
    MemoryExpert,
    ReasoningExpert,
)
from harness_agent.contracts.retrieval import RetrievalQuery, RetrievalService
from harness_agent.gates.pipeline import GatePipeline
from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import EvidencePack
from harness_agent.models.reasoning import ClinicalConclusion
from harness_agent.models.session import RouteRecord, SessionContext
from harness_agent.orchestrator.experts_config import ExpertRegistry
from harness_agent.orchestrator.planner import TaskPlanner
from harness_agent.orchestrator.router import BinaryRouter
from harness_agent.orchestrator.state import EscalationRequest, OrchestrationResult

__all__ = ["HarnessOrchestrator", "OrchestrationState"]


class OrchestrationState(TypedDict, total=False):
    """langgraph 编排状态（节点写互斥键，last-write 语义即可）。"""

    session_id: str
    user_input: str
    context: SessionContext
    route_record: RouteRecord
    tasks: list[ExpertTask]
    evidence_pack: EvidencePack
    conclusion: ClinicalConclusion
    context_bundle: ContextBundle
    escalation: EscalationRequest
    gate_verdicts: list[GateVerdict]
    result: OrchestrationResult


class HarnessOrchestrator:
    """主 Agent：路由 + 任务规划 + 专家委派 + 门禁（langgraph 编排）。

    参数：
        router:        二值路由门面（规则前置 + LLM 兜底 + 二次路由）
        planner:       任务清单规划器（路由裁决 → 委派序列）
        retrieval:     检索供给门面（M3，need_reasoning 分支取证据）
        registry:      专家声明式注册表（YAML 加载）
        experts:       name → 运行时专家实现（绑定声明与实现）
        gate_pipeline: 门禁流水线（M5，None 时跳过门禁——仅 M4 demo 场景）

    委派语义（按 kind 分派到对应协议方法）：
        reasoning → ``ReasoningExpert.reason``（唯一合法结论来源）
        memory    → ``MemoryExpert.assemble``
        generic   → ``Expert.run``
    """

    def __init__(
        self,
        *,
        router: BinaryRouter,
        planner: TaskPlanner,
        retrieval: RetrievalService,
        registry: ExpertRegistry,
        experts: dict[str, Any],
        gate_pipeline: GatePipeline | None = None,
    ) -> None:
        self._router = router
        self._planner = planner
        self._retrieval = retrieval
        self._registry = registry
        self._experts = experts
        self._gate_pipeline = gate_pipeline
        self._graph = self._build_graph()

    # ---- 对外入口 ----

    def handle(self, user_input: str, context: SessionContext) -> OrchestrationResult:
        """一轮编排：用户输入进、委派产出出（无应答权）。"""
        final_state = self._graph.invoke(
            {"session_id": context.session_id, "user_input": user_input, "context": context},
            config={"recursion_limit": 10},
        )
        return final_state["result"]

    # ---- 图构建 ----

    def _build_graph(self) -> Any:
        builder: StateGraph = StateGraph(OrchestrationState)
        builder.add_node("route", self._route_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("reason", self._reason_node)
        builder.add_node("gates", self._gates_node)
        builder.add_node("memory", self._memory_node)
        builder.add_node("escalate_node", self._escalate_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "route")
        builder.add_edge("route", "plan")
        builder.add_conditional_edges(
            "plan",
            self._branch_after_plan,
            {
                "need_reasoning": "retrieve",
                "no_reasoning": "memory",
                "escalate": "escalate_node",
            },
        )
        builder.add_edge("retrieve", "reason")
        builder.add_edge("reason", "gates")
        builder.add_edge("gates", "finalize")
        builder.add_edge("memory", "finalize")
        builder.add_edge("escalate_node", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    def _branch_after_plan(self, state: OrchestrationState) -> str:
        if state.get("escalation") is not None:
            return "escalate"
        route = state.get("route_record")
        if route is None:
            return "escalate"
        if route.decision == "need_reasoning":
            return "need_reasoning"
        if route.decision == "no_reasoning":
            return "no_reasoning"
        return "escalate"

    # ---- 节点实现 ----

    def _route_node(self, state: OrchestrationState) -> dict[str, Any]:
        """路由节点异常（在线 API 401/超时等）→ fail-closed 升级，不裸抛。"""
        try:
            record = self._router.route(state["user_input"], state["context"])
        except Exception as exc:  # noqa: BLE001
            return {
                "route_record": RouteRecord(
                    decision="escalate",
                    by_rule=False,
                    attempt=1,
                    reason=f"路由器执行异常: {exc}",
                ),
                "escalation": EscalationRequest(
                    reason=f"路由器执行异常: {exc}", clarification_question="", to_human=True
                ),
            }
        return {"route_record": record}

    def _plan_node(self, state: OrchestrationState) -> dict[str, Any]:
        if state.get("escalation") is not None:
            return {}  # 已升级：保留首个原因，跳过后续节点工作
        tasks = self._planner.plan(state["user_input"], state["route_record"], state["context"])
        return {"tasks": tasks}

    def _retrieve_node(self, state: OrchestrationState) -> dict[str, Any]:
        """检索供给（M3 门面）：查询进、证据包出（含三道闸门裁决）。"""
        if state.get("escalation") is not None:
            return {}  # 已升级：保留首个原因
        context = state["context"]
        try:
            pack = self._retrieval.retrieve(
                RetrievalQuery(
                    text=state["user_input"],
                    patient_id=context.patient_id,
                    session_id=context.session_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "escalation": EscalationRequest(
                    reason=f"检索供给异常: {exc}", clarification_question="", to_human=True
                )
            }
        return {"evidence_pack": pack}

    def _reason_node(self, state: OrchestrationState) -> dict[str, Any]:
        """委派推理专家（唯一合法临床结论来源）。

        前置校验（fail-closed）：
        - 证据包必须通过装配复核（``is_reviewed``）；
        - 任务清单首个 reasoning 专家须已绑定实现；
        - 推理专家自检失败（引用虚假/因果倒置/依据不足）→ 抛异常，
          编排层捕获后 fail-closed 升级（不产出结论）。
        """
        if state.get("escalation") is not None:
            return {}  # 已升级：保留首个原因
        pack: EvidencePack | None = state.get("evidence_pack")
        if pack is None or not pack.is_reviewed:
            reason = "证据包未通过装配复核（is_reviewed=False）"
            if pack is not None and pack.assembly_gate is not None:
                reason = f"{reason}: {pack.assembly_gate.reason}"
            return {
                "escalation": EscalationRequest(
                    reason=reason, clarification_question="", to_human=True
                )
            }

        task = next((t for t in state.get("tasks", []) if self._is_kind(t, "reasoning")), None)
        if task is None:
            return {
                "escalation": EscalationRequest(
                    reason="任务清单缺失 reasoning 专家任务", to_human=True
                )
            }
        expert = self._experts.get(task.expert)
        if not isinstance(expert, ReasoningExpert):
            return {
                "escalation": EscalationRequest(
                    reason=f"专家 {task.expert} 未绑定 ReasoningExpert 实现", to_human=True
                )
            }
        try:
            conclusion = expert.reason(task, pack, state["context"])
        except Exception as exc:  # noqa: BLE001
            return {
                "escalation": EscalationRequest(
                    reason=f"推理专家执行异常（自检失败/LLM 解析失败）: {exc}",
                    clarification_question="",
                    to_human=True,
                )
            }
        return {"conclusion": conclusion}

    def _gates_node(self, state: OrchestrationState) -> dict[str, Any]:
        """M5 门禁流水线：质量门禁 → 输出闸门（拦截即 interrupt 转人工）。

        无门禁流水线时（``gate_pipeline=None``，M4 demo 场景）直接放行。
        拦截时结论被撤回——``finalize`` 只看到 escalation，不看到 conclusion。
        门禁自身异常（在线 API 故障）同样 fail-closed 升级，不裸抛。
        """
        if state.get("escalation") is not None:
            return {}  # 已升级：保留首个原因
        if self._gate_pipeline is None:
            return {}

        conclusion = state.get("conclusion")
        pack = state.get("evidence_pack")
        # 推理失败时（escalation 已在 reason 节点产出）直接跳过门禁
        if conclusion is None or pack is None:
            return {}

        try:
            result = self._gate_pipeline.run(conclusion, pack, state["context"])
        except Exception as exc:  # noqa: BLE001
            return {
                "escalation": EscalationRequest(
                    reason=f"门禁流水线执行异常: {exc}",
                    clarification_question="",
                    to_human=True,
                )
            }
        if not result.allowed:
            # interrupt：拦截即转人工，结论被门禁撤回
            return {
                "escalation": EscalationRequest(
                    reason=f"门禁拦截（{result.blocking_gate}）: "
                    f"{result.final_verdict.reason if result.final_verdict else ''}",
                    clarification_question="",
                    to_human=True,
                ),
                "gate_verdicts": result.verdicts,
            }
        return {"gate_verdicts": result.verdicts}

    def _memory_node(self, state: OrchestrationState) -> dict[str, Any]:
        """委派记忆专家：装配上下文（复诊免重复问询主路径）。"""
        if state.get("escalation") is not None:
            return {}  # 已升级：保留首个原因
        task = next((t for t in state.get("tasks", []) if self._is_kind(t, "memory")), None)
        if task is None:
            return {
                "escalation": EscalationRequest(
                    reason="任务清单缺失 memory 专家任务", to_human=True
                )
            }
        expert = self._experts.get(task.expert)
        if not isinstance(expert, MemoryExpert):
            return {
                "escalation": EscalationRequest(
                    reason=f"专家 {task.expert} 未绑定 MemoryExpert 实现", to_human=True
                )
            }
        context = state["context"]
        try:
            bundle = expert.assemble(
                RetrievalQuery(
                    text=state["user_input"],
                    patient_id=context.patient_id,
                    session_id=context.session_id,
                ),
                context,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "escalation": EscalationRequest(
                    reason=f"记忆专家执行异常: {exc}", clarification_question="", to_human=True
                )
            }
        return {"context_bundle": bundle}

    def _escalate_node(self, state: OrchestrationState) -> dict[str, Any]:
        """路由失败：产出升级请求（转澄清 / 人工），绝不回退为主 Agent 应答。

        已有升级请求（前序节点异常写入）保持 first-writer 语义不覆盖；
        路由二次兜底仍失败（attempt=2）说明 LLM 无法可靠二分，转人工。
        """
        existing = state.get("escalation")
        if existing is not None:
            return {}
        route = state["route_record"]
        return {
            "escalation": EscalationRequest(
                reason=route.reason or "路由无法裁决",
                clarification_question=(
                    "为了准确安排处理，能补充说明您的具体问题吗？"
                    "（例如：是想咨询用药，还是查询既往记录？）"
                ),
                to_human=route.attempt >= 2,
            )
        }

    def _finalize_node(self, state: OrchestrationState) -> dict[str, Any]:
        """组装最终产出（纯编排视图：透传专家产出或升级请求）。"""
        # 门禁拦截时结论被撤回：escalation 存在则 conclusion 不透传
        escalation = state.get("escalation")
        conclusion = state.get("conclusion") if escalation is None else None

        result = OrchestrationResult.from_delegation(
            session_id=state["session_id"],
            patient_id=state["context"].patient_id,
            user_input=state["user_input"],
            route=state["route_record"],
            tasks=state.get("tasks", []),
            evidence_pack=state.get("evidence_pack"),
            conclusion=conclusion,
            context_bundle=state.get("context_bundle"),
            escalation=escalation,
            gate_verdicts=state.get("gate_verdicts"),
        )
        return {"result": result}

    # ---- 辅助 ----

    def _is_kind(self, task: ExpertTask, kind: str) -> bool:
        """任务目标专家的声明 kind 判定（未声明专家名视为不匹配）。"""
        try:
            spec = self._registry.get(task.expert)
        except KeyError:
            return False
        return spec.kind == kind
