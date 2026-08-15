"""证据与检索模型（M1）。

多模态 RAG 供给层的领域载体，关键语义：

1. **证据可回溯**：每条 ``Evidence`` 绑定 ``SourceRef``（文档 chunk 或
   原图引用含区域坐标），临床结论引用证据时全程可溯源。
2. **图文统一映射至文本语义空间**：``ImageRef.caption_layer`` 对应
   Qwen3-VL 分层描述（所见 -> 影像特征 -> 临床提示），前两层直接索引，
   临床提示层须审核转正——由 ``provenance`` + ``confidence`` 标注。
3. **结构补全与高置信证据区分**：``is_structural_completion`` 标记
   命中后补齐的同父相邻 chunk，装配与门禁环节据此区分对待。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from harness_agent.models.audit import GateVerdict
from harness_agent.models.common import ConfidenceLevel, Provenance, new_id

__all__ = [
    "CaptionLayer",
    "Evidence",
    "EvidencePack",
    "ImageRef",
    "RetrievalCandidate",
    "SourceRef",
]

#: Qwen3-VL 图像分层描述层级：
#: - observation:  所见（直接索引）
#: - feature:      影像特征（直接索引）
#: - clinical_hint: 临床提示（标注来源与置信度、抽样审核通过后转正）
CaptionLayer = Literal["observation", "feature", "clinical_hint"]


class ImageRef(BaseModel):
    """原始影像引用：证据回溯锚点（绑定原图与区域坐标）。"""

    image_id: str
    doc_id: str
    #: 归一化区域坐标 (x1, y1, x2, y2)，取值 0.0-1.0；整图引用时为 None
    region: tuple[float, float, float, float] | None = None
    caption_layer: CaptionLayer | None = None


class SourceRef(BaseModel):
    """证据来源引用（文档侧或图像侧）。"""

    source_id: str
    source_type: Literal["document", "image"]
    doc_id: str | None = None
    chunk_id: str | None = None
    #: 层级化分块的父 chunk（sibling 补全的结构锚点，parent_id + sibling 链）
    parent_id: str | None = None
    image: ImageRef | None = None


class Evidence(BaseModel):
    """单条证据：内容 + 来源 + 置信度 + 溯源。"""

    evidence_id: str = Field(default_factory=lambda: new_id("ev"))
    content: str
    source: SourceRef
    confidence: ConfidenceLevel
    provenance: Provenance
    #: 结构补全（同父相邻 chunk）标记：与高置信证据区分，计入上下文预算
    is_structural_completion: bool = False
    #: 融合精排分数（RRF 后经 reranker 的最终得分；未精排时为 None）
    score: float | None = None


class RetrievalCandidate(BaseModel):
    """双路召回的候选（RRF 融合前后的中间载体）。"""

    chunk_id: str
    content: str
    #: 稠密路排名（HNSW 召回名次；未进稠密路为 None）
    dense_rank: int | None = None
    #: 稀疏路排名（BM25 召回名次；未进稀疏路为 None）
    sparse_rank: int | None = None
    #: RRF 融合后排名
    fused_rank: int | None = None
    score: float = 0.0


class EvidencePack(BaseModel):
    """证据包：装配闸门复核通过后交付推理专家的证据集合。

    ``blocked_drugs`` 携带输入闸门已拦截的过敏药（归一化药名），
    装配闸门据此复核过滤含过敏药物的药物实体证据；
    ``assembly_gate`` 为装配复核裁决——为 None 表示未复核，
    未复核的证据包不得进入推理管线（M2 强制）。
    """

    pack_id: str = Field(default_factory=lambda: new_id("pack"))
    session_id: str
    patient_id: str
    query: str
    evidence: list[Evidence] = Field(default_factory=list)
    blocked_drugs: list[str] = Field(default_factory=list)
    assembly_gate: GateVerdict | None = None

    @property
    def high_confidence_evidence(self) -> list[Evidence]:
        """高置信证据（非结构补全），推理链引用的优先来源。"""
        return [e for e in self.evidence if not e.is_structural_completion]

    @property
    def is_reviewed(self) -> bool:
        """装配闸门是否已复核放行。"""
        return self.assembly_gate is not None and self.assembly_gate.allowed
