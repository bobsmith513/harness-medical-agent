"""BM25 稀疏检索路（M3）：本地零依赖实现。

Okapi BM25（k1=1.5、b=0.75，Lucene 同款 +1 平滑避免负 IDF），
与稠密路共用 ``tokenizer`` 的同一词元空间，保证双路口径一致。

分区隔离与向量存储同语义：索引按 ``patient_id`` 分组登记，
``search`` 只扫「查询分区 + 共享分区」；**IDF/平均长度等统计量
也只来自可见分区**——不向量化、不做跨分区统计聚合，患者语料
统计信息同样不外泄。
"""

from __future__ import annotations

import math
from collections import Counter

from harness_agent.contracts.retrieval import RetrievalQuery, RetrievedItem, StoredChunk
from harness_agent.retrieval.tokenizer import tokenize

__all__ = ["BM25SparseRetriever"]


class _IndexedDoc:
    """已索引文档：chunk + 词频 + 长度。"""

    __slots__ = ("chunk", "term_freq", "length")

    def __init__(self, chunk: StoredChunk) -> None:
        terms = tokenize(chunk.content)
        self.chunk = chunk
        self.term_freq: Counter[str] = Counter(terms)
        self.length = len(terms)


class BM25SparseRetriever:
    """本地 BM25 稀疏检索（零依赖默认实现；Milvus 部署下也可保留本路）。

    ``upsert`` 不在 ``SparseRetriever`` 契约内（契约只冻结检索语义），
    由装配工厂（``build_local_stack``）统一喂数，测试直接调用即可。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError(f"k1 不得为负: {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b 必须在 [0, 1] 区间: {b}")
        self._k1 = k1
        self._b = b
        self._partitions: dict[str | None, list[_IndexedDoc]] = {}
        self._by_chunk_id: dict[str, _IndexedDoc] = {}

    def __len__(self) -> int:
        return len(self._by_chunk_id)

    def upsert(self, items: list[StoredChunk]) -> None:
        """同 chunk_id 覆盖写入（upsert 语义），分区随 chunk.patient_id。"""
        for chunk in items:
            partition = chunk.patient_id
            bucket = self._partitions.setdefault(partition, [])
            bucket[:] = [doc for doc in bucket if doc.chunk.chunk_id != chunk.chunk_id]
            doc = _IndexedDoc(chunk)
            bucket.append(doc)
            self._by_chunk_id[chunk.chunk_id] = doc

    def search(self, query: RetrievalQuery, top_k: int) -> list[RetrievedItem]:
        """BM25 打分只在可见分区（查询分区 + 共享分区）内进行。"""
        limit = max(top_k, 0)
        visible: list[_IndexedDoc] = [
            doc
            for partition in (query.patient_id, None)
            for doc in self._partitions.get(partition, [])
        ]
        if not visible or limit == 0:
            return []

        # 统计量只来自可见分区（跨分区统计不外泄）
        total = len(visible)
        avgdl = sum(doc.length for doc in visible) / total
        if avgdl <= 0:
            avgdl = 1.0
        df: Counter[str] = Counter()
        for doc in visible:
            df.update(doc.term_freq.keys())

        query_terms = set(tokenize(query.text))
        scored: list[tuple[float, _IndexedDoc]] = []
        for doc in visible:
            score = 0.0
            for term in query_terms:
                tf = doc.term_freq.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1.0 + (total - df[term] + 0.5) / (df[term] + 0.5))
                norm = 1.0 - self._b + self._b * (doc.length / avgdl)
                score += idf * (tf * (self._k1 + 1.0)) / (tf + self._k1 * norm)
            scored.append((score, doc))

        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk.chunk_id))
        return [RetrievedItem(chunk=doc.chunk, score=score) for score, doc in scored[:limit]]

    # ---- 同父补全支撑（与 InMemoryVectorStore.get_chunk 同语义） ----

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        doc = self._by_chunk_id.get(chunk_id)
        return doc.chunk.model_copy(deep=True) if doc else None
