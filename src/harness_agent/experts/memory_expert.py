"""记忆专家（M5）：上下文装配（复诊场景免重复问询主路径）。

硬规则走药名归一化 + ATC 交叉反应精确匹配（M2 API）；
软记忆经 BGE+BM25 双路召回、RRF 融合、精排后装配（M3 门面）。
"""

from __future__ import annotations

from harness_agent.contracts.experts import ContextBundle
from harness_agent.contracts.retrieval import RetrievalQuery, RetrievalService
from harness_agent.models.session import SessionContext
from harness_agent.safety import SafetyStack, build_safety_stack

__all__ = ["MemoryExpertImpl"]

#: 记忆装配召回窗口：患者分区条目与共享知识库竞争固定 top_k 名额，但
#: 后者终将被 provenance 分类丢弃——用更大的窗口避免患者事实被知识库
#: 条目挤占（复诊免重复问询的召回质量优先于精排截断的节省）。
_MEMORY_ASSEMBLY_TOP_K = 12


class MemoryExpertImpl:
    """记忆专家实现（M1 ``MemoryExpert`` 契约）。

    装配逻辑：
    - 过敏史走硬规则精确匹配（M2 ``AllergyStore.get``）；
    - 稳定/易变事实从召回的患者记忆中分类（metadata ``volatility`` 标记）；
    - 未审核记忆（``status != approved``）不进入召回（M3 分区隔离 +
      M6 记忆审核闭环双重保障）。
    """

    def __init__(
        self,
        retrieval: RetrievalService,
        safety: SafetyStack | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._safety = safety if safety is not None else build_safety_stack()

    @property
    def name(self) -> str:
        return "memory_expert"

    @property
    def description(self) -> str:
        return "记忆专家：装配患者上下文（稳定/易变事实、过敏史硬规则、已转正记忆）。"

    def assemble(self, query: RetrievalQuery, context: SessionContext) -> ContextBundle:
        """检索患者记忆 + 装配上下文包（证据包未复核则 fail-closed 抛错）。

        fail-closed 前置校验：与推理专家 ``ReasoningExpertImpl.reason`` 对称
        （见 M2「未复核的证据包不得进入推理管线」强制约束）。输入闸门拦截时
        检索门面返回 ``is_reviewed=False`` 的空包，此处必须拒绝装配并抛出，
        由编排层 ``_memory_node`` 的异常兜底转 escalate——若静默放行，将
        得到一个空 ``stable_facts`` 的上下文包，表现为"无记忆可用"的静默降级。
        """
        # 软记忆召回（M3 门面：含三道闸门裁决，分区隔离强制）。
        # 召回窗口取装配专用下限（见 _MEMORY_ASSEMBLY_TOP_K）。
        pack = self._retrieval.retrieve(
            query.model_copy(update={"top_k": max(query.top_k, _MEMORY_ASSEMBLY_TOP_K)})
        )
        if not pack.is_reviewed:
            raise ValueError(
                "证据包未通过装配复核（is_reviewed=False），记忆专家不得基于其装配上下文"
            )

        # 稳定/易变事实分类：按 Evidence.provenance 驱动（M1 契约字段）——
        # - doctor_verified：病历核实事实（血型、既往史）→ 稳定，免重复问询；
        # - model_inference：推断性/待核实内容 → 易变，走确认式追问；
        # - knowledge_base：通用指南条目，非患者事实，不进入患者上下文
        #   （进入 stable 会把"CAP 指南"当成本人病史，进入 volatile 同样失真）。
        stable_facts: list[str] = []
        volatile_facts: list[str] = []
        for evidence in pack.evidence:
            if evidence.provenance == "doctor_verified":
                stable_facts.append(evidence.content)
            elif evidence.provenance == "model_inference":
                volatile_facts.append(evidence.content)

        # 过敏史走硬规则精确匹配（非向量召回）
        allergies = self._safety.allergy_store.get(context.patient_id)

        return ContextBundle(
            patient_id=context.patient_id,
            allergies=allergies,
            stable_facts=stable_facts,
            volatile_facts=volatile_facts,
        )
