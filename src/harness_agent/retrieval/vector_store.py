"""向量存储（M3）：内存分区隔离默认实现 + Milvus 骨架。

分区隔离语义（两实现严格一致，契约签名层面已强制）：

- ``patient_id=None`` → 知识库共享条目（所有患者可召回，已脱敏）；
- ``patient_id="P1"`` → 患者记忆（仅携带 ``patient_id="P1"`` 的查询可召回）；
- ``search`` 只扫「查询分区 + 共享分区」两个分组，其余分区物理上
  不进入扫描集合——跨患者召回不是"过滤后丢弃"而是"从不读起"，
  这是数据面隔离与逻辑过滤的本质区别。

两实现共用 ``VectorStore`` 契约（M1 冻结），另提供 ``get_chunk``
供同父相邻补全（M3 装配阶段按 ``sibling_ids`` 取回完整内容）。
"""

from __future__ import annotations

from harness_agent.contracts.retrieval import (
    Embedding,
    RetrievalQuery,
    RetrievedItem,
    StoredChunk,
)

__all__ = ["InMemoryVectorStore", "MilvusVectorStore"]


def _cosine(a: Embedding, b: Embedding) -> float:
    """点积即余弦相似度（嵌入提供方统一 L2 归一化）。"""
    return sum(x * y for x, y in zip(a, b, strict=True))


class InMemoryVectorStore:
    """本地内存向量存储：按 patient_id 分组隔离（零依赖默认实现）。

    与 Milvus partition key 等价的分组语义，demo / 测试 / 单机部署
    三场景共用；embedding 维度不做强校验（由嵌入提供方自洽）。
    """

    def __init__(self) -> None:
        # 分区键 -> [(chunk, embedding)]；None 键为知识库共享分区
        self._partitions: dict[str | None, list[tuple[StoredChunk, Embedding]]] = {}
        self._by_chunk_id: dict[str, tuple[StoredChunk, Embedding]] = {}

    def __len__(self) -> int:
        return len(self._by_chunk_id)

    # ---- VectorStore 契约 ----

    def upsert(self, items: list[StoredChunk], embeddings: list[Embedding]) -> None:
        """同 chunk_id 覆盖写入（upsert 语义），分区随 chunk.patient_id。"""
        if len(items) != len(embeddings):
            raise ValueError(f"items 与 embeddings 必须严格等长: {len(items)} != {len(embeddings)}")
        for chunk, embedding in zip(items, embeddings, strict=True):
            bucket = self._partitions.setdefault(chunk.patient_id, [])
            bucket[:] = [pair for pair in bucket if pair[0].chunk_id != chunk.chunk_id]
            bucket.append((chunk, list(embedding)))
            self._by_chunk_id[chunk.chunk_id] = (chunk, list(embedding))

    def search(
        self, query: RetrievalQuery, embedding: Embedding, top_k: int
    ) -> list[RetrievedItem]:
        """只在「查询分区 + 共享分区」扫描，按余弦相似度降序取 top_k。"""
        scored: list[tuple[float, StoredChunk]] = []
        for partition in (query.patient_id, None):
            for chunk, stored in self._partitions.get(partition, []):
                scored.append((_cosine(embedding, stored), chunk))
        # 分数降序 + chunk_id 升序（并列时结果确定，测试可锁定）
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        limit = max(top_k, 0)
        return [RetrievedItem(chunk=chunk, score=score) for score, chunk in scored[:limit]]

    # ---- 同父补全支撑（M3 装配阶段使用） ----

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        pair = self._by_chunk_id.get(chunk_id)
        return pair[0].model_copy(deep=True) if pair else None


class MilvusVectorStore:
    """Milvus 稠密向量存储（可插拔真实实现，pymilvus 为可选依赖）。

    集合设计（Milvus 2.4+，与本地实现同一隔离语义）：

    - ``patient_id`` 声明为 partition key（数据库级分区隔离），
      共享知识库条目写入哨兵值 ``""``，检索时 ``filter`` 限定
      ``patient_id == <查询患者> or patient_id == ""`` 触发分区裁剪；
    - ``vector`` 建 HNSW 索引、COSINE 度量（与哈希/BGE 嵌入口径一致）；
    - 完整 chunk 以 JSON payload 字段存取，行结构与领域模型解耦。

    安装：``uv sync --extra milvus``；连接串待填
    ``HARNESS_RETRIEVAL__MILVUS_URI``（如 http://localhost:19530，
    由 docker-compose.milvus.yaml 启动）。稀疏路 BM25 保持本地实现
    （进程内零外部依赖，见 ``bm25.py``）。
    """

    #: 共享知识库条目的分区哨兵值（StoredChunk.patient_id=None 的落库形态）
    SHARED_PARTITION_VALUE = ""

    def __init__(
        self,
        uri: str,
        collection: str = "harness_chunks",
        dim: int = 1024,
        *,
        recreate: bool = False,
    ) -> None:
        if not uri:
            raise ValueError(
                "Milvus URI 未配置：填写 HARNESS_RETRIEVAL__MILVUS_URI"
                "（如 http://localhost:19530）或改用 store=local"
            )
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise ImportError("pymilvus 未安装：uv sync --extra milvus 后重试") from exc

        self._client = MilvusClient(uri=uri)
        self._collection = collection
        self._dim = dim
        if recreate and self._client.has_collection(collection):
            self._client.drop_collection(collection)
        self._ensure_collection()

    def _partition_value(self, patient_id: str | None) -> str:
        return self.SHARED_PARTITION_VALUE if patient_id is None else patient_id

    def _ensure_collection(self) -> None:
        """集合不存在则按 schema + HNSW 索引建集合（幂等）。"""
        from pymilvus import DataType

        if self._client.has_collection(self._collection):
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(
            "patient_id",
            DataType.VARCHAR,
            max_length=128,
            is_partition_key=True,
        )
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("payload", DataType.VARCHAR, max_length=65535)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self._client.create_collection(
            collection_name=self._collection, schema=schema, index_params=index_params
        )

    # ---- VectorStore 契约 ----

    def upsert(self, items: list[StoredChunk], embeddings: list[Embedding]) -> None:
        if len(items) != len(embeddings):
            raise ValueError(f"items 与 embeddings 必须严格等长: {len(items)} != {len(embeddings)}")
        if not items:
            return
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "patient_id": self._partition_value(chunk.patient_id),
                "vector": embedding,
                "payload": chunk.model_dump_json(),
            }
            for chunk, embedding in zip(items, embeddings, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, data=rows)

    def search(
        self, query: RetrievalQuery, embedding: Embedding, top_k: int
    ) -> list[RetrievedItem]:
        """分区裁剪检索：patient 分区 + 共享分区，按 COSINE 距离换算相似度。"""
        limit = max(top_k, 0)
        if limit == 0:
            return []
        results = self._client.search(
            collection_name=self._collection,
            data=[embedding],
            limit=limit,
            output_fields=["payload"],
            filter=f'patient_id == "{query.patient_id}" or patient_id == ""',
        )
        hits = results[0] if results else []
        items: list[RetrievedItem] = []
        for hit in hits:
            chunk = StoredChunk.model_validate_json(hit["entity"]["payload"])
            # Milvus COSINE 返回距离（越小越相似），换算为相似度分数统一口径
            items.append(RetrievedItem(chunk=chunk, score=1.0 - float(hit["distance"])))
        return items

    # ---- 同父补全支撑 ----

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        rows = self._client.query(
            collection_name=self._collection,
            filter=f'chunk_id == "{chunk_id}"',
            output_fields=["payload"],
            limit=1,
        )
        if not rows:
            return None
        return StoredChunk.model_validate_json(rows[0]["payload"])
