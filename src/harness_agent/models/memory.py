"""患者记忆与硬规则模型（M1）。

长期记忆审核闭环与硬性规则的数据载体：

1. **记忆审核转正**：``Memory`` 初始为 ``session_pointer``（仅会话内指针），
   写入 /memories/ 前须进入 ``pending_review`` 队列，抽样人工审核通过才
   ``approved`` 转正为可召回记忆——阻断模型推断固化为检索事实。
2. **硬规则不向量化**：``AllergyRecord`` 是过敏史的精确匹配载体，
   走药名归一化 + ATC 药理类别交叉反应映射（M2 实现词典），
   从不进入向量索引。
3. **患者分区隔离**：``patient_id`` 是记忆库分区键（Milvus partition key
   或本地实现中的分组键），存储与检索全程强制隔离。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from harness_agent.models.common import ConfidenceLevel, Provenance, new_id

__all__ = ["AllergyRecord", "Memory", "MemoryStatus"]

#: 记忆生命周期：
#: - session_pointer: 未审核，仅作会话内指针，不得被召回
#: - pending_review:  已进入审核队列（标注来源与置信度）
#: - approved:        审核通过，转正为可召回记忆（同步向量索引）
#: - rejected:        审核驳回（不得转正）
MemoryStatus = Literal["session_pointer", "pending_review", "approved", "rejected"]


class Memory(BaseModel):
    """患者长期记忆条目（按 patient_id 分区隔离）。"""

    memory_id: str = Field(default_factory=lambda: new_id("mem"))
    patient_id: str
    content: str
    status: MemoryStatus = "session_pointer"
    provenance: Provenance
    confidence: ConfidenceLevel
    #: 产生该记忆的会话轮次（溯源用）
    source_turn: int
    reviewed_at: datetime | None = None
    reviewer: str | None = None

    @model_validator(mode="after")
    def _reviewed_status_requires_review_info(self) -> Memory:
        """审核后的状态必须留下审核时间与审核人（审计追溯底线）。"""
        reviewed = self.status in ("approved", "rejected")
        if reviewed and (self.reviewed_at is None or self.reviewer is None):
            raise ValueError(f"status={self.status} 必须记录 reviewed_at 与 reviewer")
        return self

    def can_be_recalled(self) -> bool:
        """是否可被召回：未审核记忆仅作会话内指针，绝不进入检索结果。"""
        return self.status == "approved"


class AllergyRecord(BaseModel):
    """过敏史硬规则记录：不向量化，精确匹配对象（M2 闸门的数据源）。

    ``normalized_drug`` 与 ``cross_reactants`` 均处于归一化药名空间，
    药名归一化（别名/商品名/中英文）后才能与本记录比对。
    """

    patient_id: str
    #: 原始记录药名（可能是别名 / 商品名 / 英文名）
    drug_name_raw: str
    #: 归一化后标准药名（归一化词典由 M2 提供）
    normalized_drug: str
    #: ATC 药理类别编码（交叉反应映射的分组依据）
    atc_code: str
    #: 同类别交叉反应药（归一化药名空间，ATC 映射产出）
    cross_reactants: list[str] = Field(default_factory=list)
