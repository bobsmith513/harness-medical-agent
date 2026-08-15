"""专家层契约（M1）。

DeepAgents 子代理委派的接口化：主 Agent 通过 ``Expert.run`` 委派任务，
通过 ``ReasoningExpert`` / ``MemoryExpert`` 两个特化契约获取产出。

类型层面的权力分配（无应答权的落地点）：
- 主 Agent 只能拿到 ``ClinicalConclusion``（由推理专家产出）或
  ``ContextBundle``（由记忆专家装配）；
- ``ClinicalConclusion`` 构造校验要求自检通过的推理链（models/reasoning.py），
  主 Agent 编排角色无法伪造临床结论。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from harness_agent.contracts.retrieval import RetrievalQuery
from harness_agent.models.common import new_id
from harness_agent.models.evidence import EvidencePack
from harness_agent.models.memory import AllergyRecord, Memory
from harness_agent.models.reasoning import ClinicalConclusion
from harness_agent.models.session import SessionContext

__all__ = ["ContextBundle", "Expert", "ExpertTask", "MemoryExpert", "ReasoningExpert"]


class ExpertTask(BaseModel):
    """主 Agent 委派给专家的任务（YAML 声明式配置的运行时形态）。

    ``expert`` 为目标专家名（与 configs/experts.yaml 的声明一致），
    新增专家零改动主流程。
    """

    task_id: str = Field(default_factory=lambda: new_id("task"))
    expert: str
    instruction: str
    inputs: dict[str, str] = Field(default_factory=dict)


class ContextBundle(BaseModel):
    """记忆专家装配的上下文包（复诊场景的响应依据）。

    - 稳定事实（血型、手术史等）命中充分即免重复问询；
    - 易变事实（近期用药、症状变化）确认式追问；
    - 过敏史来自硬规则精确匹配（非向量召回）。
    """

    patient_id: str
    allergies: list[AllergyRecord] = Field(default_factory=list)
    stable_facts: list[str] = Field(default_factory=list)
    volatile_facts: list[str] = Field(default_factory=list)
    #: 召回的已转正记忆（status=approved，未审核记忆不在此出现）
    recalled_memories: list[Memory] = Field(default_factory=list)


@runtime_checkable
class Expert(Protocol):
    """通用专家契约（DeepAgents 子代理的最小接口）。"""

    name: str
    description: str

    def run(self, task: ExpertTask, context: SessionContext) -> dict[str, str]: ...


@runtime_checkable
class ReasoningExpert(Protocol):
    """推理专家：系统内唯一合法的临床结论产出方。

    运行于 SFT+DPO 对齐基座，生成"证据引用 -> 逐步推断 -> 结论"
    推理链并自检；证据包必须已通过装配复核（``is_reviewed``）。
    """

    name: str

    def reason(
        self,
        task: ExpertTask,
        evidence: EvidencePack,
        context: SessionContext,
    ) -> ClinicalConclusion: ...


@runtime_checkable
class MemoryExpert(Protocol):
    """记忆专家：供给编排层（复诊场景主路径）。

    硬规则走药名归一化 + ATC 交叉反应精确匹配（M2 API）；
    软记忆经 BGE+BM25 双路召回、RRF 融合、精排后装配。
    """

    name: str

    def assemble(self, query: RetrievalQuery, context: SessionContext) -> ContextBundle: ...
