"""供给层门面（M3）：HybridRetrievalService——查询进、证据包出。

全链路（闸门永不旁路，fail-closed）::

    输入闸门（过敏硬规则拦截，查询构造前置）
      └─ 拦截 → 空证据包 + gate=input 裁决（is_reviewed=False，调用方转澄清/人工）
    双路召回（稠密 HNSW + 稀疏 BM25，patient_id 分区隔离）
    RRF 融合（k 常数平滑，双路共识上浮）
    精排（identity 默认 / bge-reranker 可插拔）
    同父补全（sibling_ids 取回相邻 chunk，is_structural_completion=True）
    装配闸门复核（过滤含过敏药物实体的证据）
      └─ 全部被过滤 → 拒绝交付（is_reviewed=False）
      └─ 通过 → EvidencePack（is_reviewed=True，进入推理管线）

设计约定（本地栈装配契约外补充能力，构造期校验）：
- ``vector_store`` 须提供 ``get_chunk(chunk_id)``（同父补全依赖）；
- ``sparse`` 须提供 ``upsert(items)``（双路共用一个入库口径）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness_agent.contracts.gates import AssemblyGate, InputGate
from harness_agent.contracts.retrieval import (
    EmbeddingProvider,
    Reranker,
    RetrievalQuery,
    RetrievedItem,
    SparseRetriever,
    StoredChunk,
    VectorStore,
)
from harness_agent.models.evidence import Evidence, EvidencePack, SourceRef
from harness_agent.models.session import SessionContext
from harness_agent.retrieval.fusion import rrf_fuse
from harness_agent.safety.resolver import AllergyConflictResolver

__all__ = ["HybridRetrievalService"]


@runtime_checkable
class _ChunkReadable(Protocol):
    """契约外补充能力：按 chunk_id 取回（同父补全依赖）。"""

    def get_chunk(self, chunk_id: str) -> StoredChunk | None: ...


@runtime_checkable
class _SparseIndexable(Protocol):
    """契约外补充能力：稀疏路入库（本地 BM25 实现，双路共用入库口径）。"""

    def upsert(self, items: list[StoredChunk]) -> None: ...


class HybridRetrievalService:
    """混合检索门面：实现 M1 ``RetrievalService`` 契约。

    患者分区隔离由两路存储实现强制（本门面只透传 patient_id，
    不做、也不能做跨分区拼装）；证据包的 ``blocked_drugs`` 携带
    患者阻断全集（直接过敏 + ATC 交叉反应），装配闸门据此过滤。
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        sparse: SparseRetriever,
        reranker: Reranker,
        input_gate: InputGate,
        assembly_gate: AssemblyGate,
        resolver: AllergyConflictResolver,
        dense_top_k: int = 8,
        sparse_top_k: int = 8,
        rrf_k: int = 60,
    ) -> None:
        if not isinstance(vector_store, _ChunkReadable):
            raise TypeError(
                "vector_store 必须提供 get_chunk(chunk_id)（同父补全依赖，"
                "本地栈装配约定，见 retrieval.wiring）"
            )
        if not isinstance(sparse, _SparseIndexable):
            raise TypeError(
                "sparse 必须提供 upsert(items)（稀疏路入库依赖，"
                "本地栈装配约定，见 retrieval.wiring）"
            )
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._sparse = sparse
        self._reranker = reranker
        self._input_gate = input_gate
        self._assembly_gate = assembly_gate
        self._resolver = resolver
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._rrf_k = rrf_k

    # ---- 入库（双路共用一个口径，MCP 工具 / M6 记忆管线调用） ----

    def index(self, items: list[StoredChunk]) -> None:
        """入库：嵌入 + 稠密 upsert + 稀疏 upsert。

        ``patient_id=None`` 为知识库条目（全患者共享），非 None 为
        患者记忆（分区隔离）；分词语义两路天然一致（共用 tokenizer）。
        """
        if not items:
            return
        embeddings = self._embedding_provider.embed([chunk.content for chunk in items])
        self._vector_store.upsert(items, embeddings)
        self._sparse.upsert(items)

    # ---- 检索门面（M1 RetrievalService 契约） ----

    def retrieve(self, query: RetrievalQuery) -> EvidencePack:
        """查询进 → 证据包出（闸门串联，fail-closed）。"""
        # 1. 输入闸门：查询文本命中患者阻断集合即拦截（不构造检索）
        context = SessionContext(patient_id=query.patient_id)
        blocked = sorted(self._resolver.blocked_drugs(query.patient_id))
        input_verdict = self._input_gate.check(query, context)
        if not input_verdict.allowed:
            # 裁决直接落在 assembly_gate 槽位：is_reviewed=False，
            # 调用方按 fail-closed 处理（转澄清 / 升级人工）
            return EvidencePack(
                session_id=query.session_id,
                patient_id=query.patient_id,
                query=query.text,
                evidence=[],
                blocked_drugs=blocked,
                assembly_gate=input_verdict,
            )

        # 2. 双路召回（分区隔离由两路存储实现强制）
        embedding = self._embedding_provider.embed([query.text])[0]
        dense = self._vector_store.search(query, embedding, self._dense_top_k)
        sparse = self._sparse.search(query, self._sparse_top_k)

        # 3. RRF 融合（窗口 = 召回深度上限，控制精排输入规模）
        fused = rrf_fuse(
            dense, sparse, k=self._rrf_k, top_k=max(self._dense_top_k, self._sparse_top_k)
        )

        # 4. 精排（identity 默认直接截断；分数语义为最终融合分）
        final_k = max(query.top_k, 0)
        reranked = self._reranker.rerank(query.text, fused, top_k=final_k)

        # 5. 同父补全：命中 chunk 的 sibling_ids 取回相邻内容
        evidence: list[Evidence] = []
        seen_ids = {item.chunk.chunk_id for item in reranked}
        for item in reranked:
            evidence.append(self._hit_evidence(item))
            for sibling_id in item.chunk.sibling_ids:
                if sibling_id in seen_ids:
                    continue
                sibling = self._vector_store.get_chunk(sibling_id)
                if sibling is None:
                    continue
                seen_ids.add(sibling_id)
                evidence.append(self._completion_evidence(sibling))

        # 6. 装配闸门复核：过滤含过敏药物实体的证据并附加裁决
        pack = EvidencePack(
            session_id=query.session_id,
            patient_id=query.patient_id,
            query=query.text,
            evidence=evidence,
            blocked_drugs=blocked,
        )
        return self._assembly_gate.apply(pack)

    # ---- 证据构造 ----

    def _hit_evidence(self, item: RetrievedItem) -> Evidence:
        """召回命中 → 证据（分数透传，来源可回溯）。"""
        chunk = item.chunk
        return Evidence(
            content=chunk.content,
            source=self._source_ref(chunk),
            confidence=chunk.metadata.get("confidence", "medium"),
            provenance=chunk.metadata.get("provenance")
            or ("doctor_verified" if chunk.patient_id is not None else "knowledge_base"),
            score=item.score,
        )

    def _completion_evidence(self, chunk: StoredChunk) -> Evidence:
        """同父补全 → 结构性证据（低置信、不计精排分）。"""
        return Evidence(
            content=chunk.content,
            source=self._source_ref(chunk),
            confidence="low",
            provenance=chunk.metadata.get("provenance")
            or ("doctor_verified" if chunk.patient_id is not None else "knowledge_base"),
            is_structural_completion=True,
        )

    @staticmethod
    def _source_ref(chunk: StoredChunk) -> SourceRef:
        return SourceRef(
            source_id=chunk.chunk_id,
            source_type="document",
            doc_id=chunk.metadata.get("doc_id"),
            chunk_id=chunk.chunk_id,
            parent_id=chunk.parent_id,
        )
