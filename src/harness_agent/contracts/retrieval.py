"""检索供给层契约（M1）。

混合检索全链路的接口切分（M3 实现，mock 与 Milvus 双实现共用）：

    EmbeddingProvider -> VectorStore（稠密 HNSW）
                             \\                RRF 融合
                              SparseRetriever（BM25）
                                        |
                                        v
                                Reranker -> 同父补全 -> RetrievalService

关键设计：
- **分区隔离在接口签名层面强制**：``StoredChunk.patient_id`` 非空即为
  患者记忆（按分区存储/检索），None 为知识库共享条目；
  ``RetrievalService.retrieve`` 永远携带 patient_id，跨患者召回无从谈起。
- **Reranker 可关**：``reranker_enabled=false`` 时注入 identity 实现
  （直接按融合名次截断），接口不变。
- **RetrievalService 是门面**：内部串联三道闸门（输入拦截前置、装配
  复核收尾），对外只暴露"查询进、证据包出"。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from harness_agent.models.common import new_id
from harness_agent.models.evidence import EvidencePack

__all__ = [
    "Embedding",
    "EmbeddingProvider",
    "RetrievedItem",
    "RetrievalQuery",
    "RetrievalService",
    "Reranker",
    "SparseRetriever",
    "StoredChunk",
    "VectorStore",
]

#: 嵌入向量（BGE-large-zh，dim=1024，M0 配置锁定）。
Embedding = list[float]


class StoredChunk(BaseModel):
    """已入索引的 chunk（稠密与稀疏共用的存储单元）。

    ``patient_id`` 为 None 表示知识库条目（全患者共享、已脱敏）；
    非 None 表示患者记忆（Milvus partition key / 本地分组键，强制隔离）。
    """

    chunk_id: str = Field(default_factory=lambda: new_id("chunk"))
    patient_id: str | None = None
    content: str
    #: 同父相邻 chunk（sibling 链：命中后补全的结构依据）
    sibling_ids: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    """检索请求：查询文本 + 患者（分区键）+ 召回参数。"""

    query_id: str = Field(default_factory=lambda: new_id("q"))
    text: str
    patient_id: str
    top_k: int = 5
    session_id: str = ""


class RetrievedItem(BaseModel):
    """召回结果项：chunk + 得分（分数语义由所在阶段决定）。"""

    chunk: StoredChunk
    score: float


@runtime_checkable
class EmbeddingProvider(Protocol):
    """嵌入提供方（BGE-large-zh；CPU/GPU 自动检测，M3 实现）。"""

    def embed(self, texts: list[str]) -> list[Embedding]: ...


@runtime_checkable
class VectorStore(Protocol):
    """稠密向量存储（HNSW）。

    upsert 时 embeddings 与 items 严格等长对位；
    search 按 patient 分区隔离召回（实现方必须保证，接口签名已携带分区语义）。
    """

    def upsert(self, items: list[StoredChunk], embeddings: list[Embedding]) -> None: ...

    def search(
        self, query: RetrievalQuery, embedding: Embedding, top_k: int
    ) -> list[RetrievedItem]: ...


@runtime_checkable
class SparseRetriever(Protocol):
    """稀疏路检索（BM25：Milvus 内置或本地实现，接口统一）。"""

    def search(self, query: RetrievalQuery, top_k: int) -> list[RetrievedItem]: ...


@runtime_checkable
class Reranker(Protocol):
    """精排器（bge-reranker-v2-m3；关闭时注入 identity 实现）。"""

    def rerank(self, query: str, items: list[RetrievedItem], top_k: int) -> list[RetrievedItem]: ...


@runtime_checkable
class RetrievalService(Protocol):
    """供给层门面：查询进 -> 证据包出。

    实现方（M3）内部串联：输入闸门（过敏硬规则拦截）-> 双路召回 ->
    RRF 融合 -> 精排 -> 同父相邻补全 -> 装配闸门复核。
    输出的 ``EvidencePack.assembly_gate.allowed`` 必须为 True，
    否则调用方按 fail-closed 处理。
    """

    def retrieve(self, query: RetrievalQuery) -> EvidencePack: ...
