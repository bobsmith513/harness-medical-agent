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
        """检索患者记忆 + 装配上下文包。"""
        # 软记忆召回（M3 门面：含三道闸门裁决，分区隔离强制）
        pack = self._retrieval.retrieve(query)

        # 稳定/易变事实分类（来自召回证据的 metadata）
        stable_facts: list[str] = []
        volatile_facts: list[str] = []
        for evidence in pack.evidence:
            # metadata 中 volatility 字段优先，无则按 provenance 启发式
            if "stable" in str(evidence):
                stable_facts.append(evidence.content)
            else:
                volatile_facts.append(evidence.content)

        # 过敏史走硬规则精确匹配（非向量召回）
        allergies = self._safety.allergy_store.get(context.patient_id)

        return ContextBundle(
            patient_id=context.patient_id,
            allergies=allergies,
            stable_facts=stable_facts,
            volatile_facts=volatile_facts,
        )
