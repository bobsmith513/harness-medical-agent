"""检索栈装配工厂（M3）：配置 → 全组件接线。

零依赖默认（``store=local`` + ``embedding_provider=hashing``）：
哈希嵌入 + 内存向量存储 + 本地 BM25 + identity 精排 + 种子安全栈，
无任何可选依赖即可全链路跑通。

可插拔升级（配置切换实现，业务逻辑零分叉）：

===============  ==================================  =========================
组件              配置开关                             可选依赖
===============  ==================================  =========================
嵌入              ``RETRIEVAL__EMBEDDING_PROVIDER=bge``  extras=bge
向量存储          ``RETRIEVAL__STORE=milvus``           extras=milvus + URI
精排              ``RETRIEVAL__RERANKER_ENABLED=true``  extras=bge
药名词典          ``SAFETY__DICTIONARY_PATH``           无（JSON 文件）
===============  ==================================  =========================
"""

from __future__ import annotations

from dataclasses import dataclass

from harness_agent.config.settings import RetrievalSettings, Settings, get_settings
from harness_agent.contracts.retrieval import (
    EmbeddingProvider,
    Reranker,
    SparseRetriever,
    VectorStore,
)
from harness_agent.retrieval.bm25 import BM25SparseRetriever
from harness_agent.retrieval.embeddings import BGEEmbeddingProvider, HashingEmbeddingProvider
from harness_agent.retrieval.fusion import BGEReranker, IdentityReranker
from harness_agent.retrieval.service import HybridRetrievalService
from harness_agent.retrieval.vector_store import InMemoryVectorStore, MilvusVectorStore
from harness_agent.safety import build_safety_stack

__all__ = [
    "RetrievalStack",
    "build_retrieval_stack",
]


@dataclass(frozen=True)
class RetrievalStack:
    """检索供给栈：门面 + 各组件引用。

    组件引用保留给入库（``stack.service.index``）、审计与替换单组件
    （如注入真实 BGE 嵌入）使用；日常调用方只依赖 ``service``。
    """

    service: HybridRetrievalService
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    sparse: SparseRetriever
    reranker: Reranker


def _build_embedding_provider(settings: RetrievalSettings) -> EmbeddingProvider:
    if settings.embedding_provider == "bge":
        return BGEEmbeddingProvider(model_name=settings.embedding_model)
    return HashingEmbeddingProvider(dim=settings.embedding_dim)


def _build_vector_store(settings: RetrievalSettings) -> VectorStore:
    if settings.store == "milvus":
        return MilvusVectorStore(
            uri=settings.milvus_uri, dim=settings.embedding_dim, recreate=False
        )
    return InMemoryVectorStore()


def _build_reranker(settings: RetrievalSettings) -> Reranker:
    if settings.reranker_enabled:
        return BGEReranker(model_name=settings.reranker_model)
    return IdentityReranker()


def build_retrieval_stack(settings: Settings | None = None) -> RetrievalStack:
    """按配置装配完整检索供给栈（含 M2 安全闸门）。"""
    if settings is None:
        settings = get_settings()
    retrieval = settings.retrieval

    embedding_provider = _build_embedding_provider(retrieval)
    vector_store = _build_vector_store(retrieval)
    sparse: SparseRetriever = BM25SparseRetriever()
    reranker = _build_reranker(retrieval)
    safety = build_safety_stack(settings)

    service = HybridRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        sparse=sparse,
        reranker=reranker,
        input_gate=safety.input_gate,
        assembly_gate=safety.assembly_gate,
        resolver=safety.resolver,
        dense_top_k=retrieval.dense_top_k,
        sparse_top_k=retrieval.sparse_top_k,
        rrf_k=retrieval.rrf_k,
    )
    return RetrievalStack(
        service=service,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        sparse=sparse,
        reranker=reranker,
    )
