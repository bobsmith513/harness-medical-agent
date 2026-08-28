"""M3 检索层测试：分词 / 嵌入 / 向量存储 / BM25 的分区隔离语义。

核心锁定点：
1. **患者分区隔离**：患者记忆只对本患者查询可见，跨患者召回
   无从谈起（稠密路 + 稀疏路双重验证）；
2. **统计量隔离**：BM25 的 IDF/平均长度只来自可见分区，
   其他患者的语料规模不泄露任何信息；
3. **零依赖默认可用**：哈希嵌入 + 内存向量存储 + 本地 BM25
   在无任何可选依赖时全链路可用；
4. **可插拔依赖缺失时的报错友好**：BGE / Milvus 未安装时
   给出明确的 extras 安装指引。
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from harness_agent.contracts.retrieval import (
    EmbeddingProvider,
    RetrievalQuery,
    SparseRetriever,
    StoredChunk,
    VectorStore,
)
from harness_agent.retrieval.bm25 import BM25SparseRetriever
from harness_agent.retrieval.embeddings import (
    BGEEmbeddingProvider,
    HashingEmbeddingProvider,
)
from harness_agent.retrieval.tokenizer import tokenize
from harness_agent.retrieval.vector_store import InMemoryVectorStore, MilvusVectorStore

_PYMILVUS_INSTALLED = importlib.util.find_spec("pymilvus") is not None
_SENTENCE_TRANSFORMERS_INSTALLED = importlib.util.find_spec("sentence_transformers") is not None


def _chunk(
    content: str, patient_id: str | None, chunk_id: str, *, parent: str | None = None
) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        patient_id=patient_id,
        content=content,
        parent_id=parent,
        sibling_ids=[],
    )


def _query(text: str, patient_id: str = "pat-A") -> RetrievalQuery:
    return RetrievalQuery(text=text, patient_id=patient_id)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# 分词器
# ---------------------------------------------------------------------------
class TestTokenizer:
    def test_chinese_bigram_plus_ascii_token(self):
        assert tokenize("患者 penicillin 过敏") == ["患者", "penicillin", "过敏"]

    def test_long_cjk_run_generates_bigrams(self):
        assert tokenize("阿莫西林") == ["阿莫", "莫西", "西林"]

    def test_single_cjk_char_stands_alone(self):
        assert tokenize("热") == ["热"]

    def test_lowercase_folding(self):
        assert tokenize("Penicillin") == ["penicillin"]

    def test_punctuation_splits_runs(self):
        assert tokenize("WBC 12.3e9/L") == ["wbc", "12", "3e9", "l"]

    def test_empty_and_symbol_only(self):
        assert tokenize("") == []
        assert tokenize("，。！ /") == []


# ---------------------------------------------------------------------------
# 嵌入提供方
# ---------------------------------------------------------------------------
class TestHashingEmbedding:
    def test_satisfies_contract(self):
        assert isinstance(HashingEmbeddingProvider(), EmbeddingProvider)

    def test_deterministic(self):
        provider = HashingEmbeddingProvider()
        assert provider.embed(["青霉素 过敏"]) == provider.embed(["青霉素 过敏"])

    def test_l2_normalized(self):
        provider = HashingEmbeddingProvider(dim=128)
        vector = provider.embed(["青霉素 过敏史"])[0]
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)

    def test_similarity_semantics(self):
        provider = HashingEmbeddingProvider()
        base, related, unrelated = provider.embed(
            ["青霉素 过敏", "青霉素 过敏 反应", "糖尿病 饮食"]
        )
        assert _cosine(base, related) > _cosine(base, unrelated)

    def test_invalid_dim_rejected(self):
        with pytest.raises(ValueError, match="维度"):
            HashingEmbeddingProvider(dim=0)

    @pytest.mark.skipif(_SENTENCE_TRANSFORMERS_INSTALLED, reason="sentence-transformers 已安装")
    def test_bge_missing_dependency_hint(self):
        with pytest.raises(ImportError, match="bge"):
            BGEEmbeddingProvider()


# ---------------------------------------------------------------------------
# 稠密路：内存向量存储
# ---------------------------------------------------------------------------
class TestInMemoryVectorStore:
    def test_satisfies_contract(self):
        assert isinstance(InMemoryVectorStore(), VectorStore)

    def test_partition_isolation_patient_memory_invisible_cross_patient(self):
        provider = HashingEmbeddingProvider()
        store = InMemoryVectorStore()
        a_memory = _chunk("患者甲既往青霉素过敏史记录", "pat-A", "chunk-a1")
        shared = _chunk("青霉素类药物用药指导", None, "chunk-s1")
        store.upsert([a_memory, shared], provider.embed([a_memory.content, shared.content]))

        # 患者 B 查询：只见共享知识库，患者 A 的记忆物理上不进入扫描集合
        hits_b = store.search(
            _query("青霉素 过敏", patient_id="pat-B"),
            provider.embed(["青霉素 过敏"])[0],
            top_k=10,
        )
        assert [item.chunk.chunk_id for item in hits_b] == ["chunk-s1"]

        # 患者 A 查询：自己的记忆 + 共享知识库
        hits_a = store.search(
            _query("青霉素 过敏", patient_id="pat-A"),
            provider.embed(["青霉素 过敏"])[0],
            top_k=10,
        )
        assert {item.chunk.chunk_id for item in hits_a} == {"chunk-a1", "chunk-s1"}

    def test_scores_sorted_desc_and_tiebreak_deterministic(self):
        provider = HashingEmbeddingProvider()
        store = InMemoryVectorStore()
        docs = [_chunk(f"共享文档 {i}", None, f"s{i}") for i in range(5)]
        store.upsert(docs, provider.embed([doc.content for doc in docs]))
        hits = store.search(_query("共享 文档"), provider.embed(["共享 文档"])[0], top_k=10)
        scores = [item.score for item in hits]
        assert scores == sorted(scores, reverse=True)
        # 相同文本嵌入相同 → 分数并列 → chunk_id 升序保证确定序
        ids = [item.chunk.chunk_id for item in hits]
        assert ids == sorted(ids)

    def test_upsert_length_mismatch_rejected(self):
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="等长"):
            store.upsert([_chunk("内容", None, "c1")], [[1.0, 0.0], [0.0, 1.0]])

    def test_upsert_overwrites_same_chunk_id(self):
        provider = HashingEmbeddingProvider()
        store = InMemoryVectorStore()
        first = _chunk("原始内容", None, "c1")
        store.upsert([first], provider.embed([first.content]))
        second = _chunk("更新后的内容", None, "c1")
        store.upsert([second], provider.embed([second.content]))

        assert len(store) == 1
        hits = store.search(_query("更新 后"), provider.embed(["更新 后"])[0], top_k=5)
        assert hits[0].chunk.content == "更新后的内容"

    def test_get_chunk_returns_deep_copy(self):
        provider = HashingEmbeddingProvider()
        store = InMemoryVectorStore()
        doc = _chunk("内容", None, "c1")
        store.upsert([doc], provider.embed([doc.content]))

        fetched = store.get_chunk("c1")
        assert fetched is not None
        fetched.content = "被篡改"
        assert store.get_chunk("c1").content == "内容"
        assert store.get_chunk("missing") is None

    def test_empty_store_returns_empty(self):
        store = InMemoryVectorStore()
        assert store.search(_query("任意"), [0.0] * 8, top_k=5) == []

    def test_top_k_zero_returns_empty(self):
        provider = HashingEmbeddingProvider()
        store = InMemoryVectorStore()
        doc = _chunk("共享内容", None, "s1")
        store.upsert([doc], provider.embed([doc.content]))
        assert store.search(_query("共享"), provider.embed(["共享"])[0], top_k=0) == []


# ---------------------------------------------------------------------------
# 稀疏路：BM25
# ---------------------------------------------------------------------------
class TestBM25SparseRetriever:
    def test_satisfies_contract(self):
        assert isinstance(BM25SparseRetriever(), SparseRetriever)

    def test_partition_isolation_same_semantics_as_dense(self):
        sparse = BM25SparseRetriever()
        sparse.upsert(
            [
                _chunk("患者甲青霉素过敏史记录", "pat-A", "a1"),
                _chunk("青霉素类药物用药指导", None, "s1"),
            ]
        )
        hits_b = sparse.search(_query("青霉素 过敏", patient_id="pat-B"), top_k=10)
        assert [item.chunk.chunk_id for item in hits_b] == ["s1"]

        hits_a = sparse.search(_query("青霉素 过敏", patient_id="pat-A"), top_k=10)
        assert {item.chunk.chunk_id for item in hits_a} == {"a1", "s1"}

    def test_bm25_statistics_isolated_across_partitions(self):
        """其他患者分区语料的增删不影响本患者查询的统计量与得分。"""
        sparse = BM25SparseRetriever()
        sparse.upsert(
            [
                _chunk("血糖控制目标与监测频率", None, "s1"),
                _chunk("血糖管理随访记录", "pat-A", "a1"),
            ]
        )
        query = _query("血糖", patient_id="pat-A")
        before = sparse.search(query, top_k=5)

        # 患者分区新增大量语料：IDF/平均长度只来自可见分区，得分必须不变
        sparse.upsert([_chunk(f"其他患者文档 {i}", "pat-B", f"b{i}") for i in range(50)])
        after = sparse.search(query, top_k=5)
        assert [(i.chunk.chunk_id, i.score) for i in before] == [
            (i.chunk.chunk_id, i.score) for i in after
        ]

    def test_exact_term_match_ranks_highest(self):
        sparse = BM25SparseRetriever()
        sparse.upsert(
            [
                _chunk("阿司匹林的抗血小板聚集机制", None, "d1"),
                _chunk("对乙酰氨基酚的解热镇痛作用", None, "d2"),
            ]
        )
        hits = sparse.search(_query("阿司匹林"), top_k=2)
        assert hits[0].chunk.chunk_id == "d1"
        assert hits[0].score > 0
        assert hits[1].score == 0.0

    def test_higher_term_frequency_scores_higher(self):
        sparse = BM25SparseRetriever()
        sparse.upsert(
            [
                _chunk("infection infection infection infection", None, "d1"),
                _chunk("infection recovery care", None, "d2"),
            ]
        )
        hits = sparse.search(_query("infection"), top_k=2)
        assert hits[0].chunk.chunk_id == "d1"
        assert hits[0].score > hits[1].score

    def test_upsert_overwrites_same_chunk_id(self):
        sparse = BM25SparseRetriever()
        sparse.upsert([_chunk("原始内容", None, "c1")])
        sparse.upsert([_chunk("替换内容", None, "c1")])
        assert len(sparse) == 1
        assert sparse.get_chunk("c1").content == "替换内容"

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError, match="k1"):
            BM25SparseRetriever(k1=-1)
        with pytest.raises(ValueError, match="b"):
            BM25SparseRetriever(b=1.5)

    def test_empty_index_returns_empty(self):
        assert BM25SparseRetriever().search(_query("任意"), top_k=5) == []


# ---------------------------------------------------------------------------
# Milvus 骨架：配置与依赖缺失时的 fail-closed 行为
# ---------------------------------------------------------------------------
class TestMilvusSkeleton:
    def test_empty_uri_rejected_with_hint(self):
        with pytest.raises(ValueError, match="MILVUS_URI"):
            MilvusVectorStore(uri="")

    @pytest.mark.skipif(_PYMILVUS_INSTALLED, reason="pymilvus 已安装")
    def test_missing_dependency_hint(self):
        with pytest.raises(ImportError, match="milvus"):
            MilvusVectorStore(uri="http://localhost:19530")


# ---------------------------------------------------------------------------
# 过滤字面量白名单（静态分析整改项）：filter 表达式注入防护
# ---------------------------------------------------------------------------
class TestFilterLiteral:
    """Milvus filter 表达式为字符串拼接构建，值必须经白名单校验。"""

    def test_valid_identifiers_pass(self):
        from harness_agent.retrieval.vector_store import _filter_literal

        assert _filter_literal("pat-001") == "pat-001"
        assert _filter_literal("P1") == "P1"
        assert _filter_literal("ev-3e20167f6dae") == "ev-3e20167f6dae"
        assert _filter_literal("") == ""  # 共享分区哨兵值

    @pytest.mark.parametrize(
        "malicious",
        [
            'pat-001" or patient_id != "',  # 闭合引号改写表达式
            "pat-001; drop collection",  # 分号注入
            "pat\\-001",  # 反斜杠转义逃逸
            "pat 001",  # 空格（fold 后标识符不应含空白）
        ],
    )
    def test_malicious_values_rejected(self, malicious):
        from harness_agent.retrieval.vector_store import _filter_literal

        with pytest.raises(ValueError, match="非法字符"):
            _filter_literal(malicious)
