"""混合检索供给层（M3）：双路召回 + RRF 融合 + 精排 + 分区隔离。

组件清单（全部实现 M1 冻结的 ``contracts.retrieval`` 协议）：

- ``tokenizer``:           中文二元组 + ASCII 词元（双路共用词元空间）
- ``embeddings``:          哈希嵌入（默认零依赖）/ BGE-large-zh（可插拔）
- ``vector_store``:        内存分区隔离（默认）/ Milvus HNSW（可插拔）
- ``bm25``:                本地 BM25 稀疏路（统计量同分区隔离）
- ``fusion``:              RRF 融合 + identity / bge-reranker 精排
- ``service``:             HybridRetrievalService 门面（闸门串联）
- ``wiring``:              配置 → 全组件接线（build_retrieval_stack）
"""

from harness_agent.retrieval.bm25 import BM25SparseRetriever
from harness_agent.retrieval.embeddings import BGEEmbeddingProvider, HashingEmbeddingProvider
from harness_agent.retrieval.fusion import (
    BGEReranker,
    IdentityReranker,
    build_fusion_candidates,
    rrf_fuse,
)
from harness_agent.retrieval.service import HybridRetrievalService
from harness_agent.retrieval.vector_store import InMemoryVectorStore, MilvusVectorStore
from harness_agent.retrieval.wiring import RetrievalStack, build_retrieval_stack

__all__ = [
    "BGEEmbeddingProvider",
    "BGEReranker",
    "BM25SparseRetriever",
    "HashingEmbeddingProvider",
    "HybridRetrievalService",
    "IdentityReranker",
    "InMemoryVectorStore",
    "MilvusVectorStore",
    "RetrievalStack",
    "build_fusion_candidates",
    "build_retrieval_stack",
    "rrf_fuse",
]
