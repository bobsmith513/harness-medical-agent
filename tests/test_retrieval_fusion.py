"""M3 融合与精排测试：RRF 数学 / 名次迹 / identity 与 BGE 精排。

RRF 数学锁定（k=60）：
- 双路同命中候选融合分 = Σ 1/(k+rank)，天然高于任何单路头部；
- 单路未命中该路贡献为 0；
- 并列分按 chunk_id 升序保证确定序。
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from harness_agent.contracts.retrieval import Reranker, RetrievedItem, StoredChunk
from harness_agent.retrieval.fusion import (
    BGEReranker,
    IdentityReranker,
    build_fusion_candidates,
    rrf_fuse,
)

_SENTENCE_TRANSFORMERS_INSTALLED = importlib.util.find_spec("sentence_transformers") is not None


def _item(chunk_id: str, score: float, *, content: str | None = None) -> RetrievedItem:
    return RetrievedItem(
        chunk=StoredChunk(chunk_id=chunk_id, patient_id=None, content=content or f"内容{chunk_id}"),
        score=score,
    )


# ---------------------------------------------------------------------------
# RRF 融合数学
# ---------------------------------------------------------------------------
class TestRRFFusion:
    def test_dual_path_beats_single_path_head(self):
        """双路第 1 名（2/61）高于任何单路贡献（至多 1/61）。"""
        dense = [_item("c1", 0.9), _item("c2", 0.8)]
        sparse = [_item("c1", 12.0), _item("c3", 5.0)]
        fused = rrf_fuse(dense, sparse, k=60, top_k=5)

        assert fused[0].chunk.chunk_id == "c1"
        assert fused[0].score == pytest.approx(2.0 / 61.0)

    def test_exact_rrf_scores_per_path(self):
        dense = [_item("c1", 0.9), _item("c2", 0.8)]
        sparse = [_item("c3", 7.0), _item("c2", 6.0), _item("c4", 5.0)]
        fused = rrf_fuse(dense, sparse, k=60, top_k=5)
        by_id = {item.chunk.chunk_id: item.score for item in fused}

        # c1: 仅稠密路 rank1；c3: 仅稀疏路 rank1
        assert by_id["c1"] == pytest.approx(1.0 / 61.0)
        assert by_id["c3"] == pytest.approx(1.0 / 61.0)
        # c2: 稠密 rank2 + 稀疏 rank2；c4: 稀疏 rank3
        assert by_id["c2"] == pytest.approx(1.0 / 62.0 + 1.0 / 62.0)
        assert by_id["c4"] == pytest.approx(1.0 / 63.0)
        # c2 融合分最高
        assert fused[0].chunk.chunk_id == "c2"

    def test_tie_break_by_chunk_id_ascending(self):
        """c2/c3 分别在稀疏/稠密路同为 rank1 → 分数并列 → chunk_id 升序。"""
        dense = [_item("c3", 0.9)]
        sparse = [_item("c2", 7.0)]
        fused = rrf_fuse(dense, sparse, k=60, top_k=5)
        assert fused[0].score == pytest.approx(fused[1].score)
        assert [item.chunk.chunk_id for item in fused] == ["c2", "c3"]

    def test_k_smoothing_effect(self):
        """k 增大后名次差异的边际影响收敛（rank1 与 rank2 分差缩小）。"""
        dense = [_item("c1", 0.9), _item("c2", 0.8)]
        gap_k1 = _score_gap(rrf_fuse(dense, [], k=1, top_k=5))
        gap_k60 = _score_gap(rrf_fuse(dense, [], k=60, top_k=5))
        assert gap_k1 > gap_k60

    def test_top_k_truncation_and_zero(self):
        dense = [_item(f"c{i}", 1.0 - i * 0.1) for i in range(10)]
        assert len(rrf_fuse(dense, [], k=60, top_k=3)) == 3
        assert rrf_fuse(dense, [], k=60, top_k=0) == []

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError, match="k"):
            rrf_fuse([], [], k=0)

    def test_empty_inputs(self):
        assert rrf_fuse([], []) == []

    def test_chunk_payload_preserved(self):
        dense = [_item("c1", 0.9, content="完整内容保留")]
        fused = rrf_fuse(dense, [], k=60, top_k=5)
        assert fused[0].chunk.content == "完整内容保留"
        assert fused[0].chunk.patient_id is None


def _score_gap(items: list[RetrievedItem]) -> float:
    return items[0].score - items[1].score


# ---------------------------------------------------------------------------
# 融合名次迹（RetrievalCandidate，供审计）
# ---------------------------------------------------------------------------
class TestFusionCandidates:
    def test_rank_traces_complete(self):
        dense = [_item("c1", 0.9), _item("c2", 0.8)]
        sparse = [_item("c1", 5.0), _item("c3", 4.0)]
        fused = rrf_fuse(dense, sparse, k=60, top_k=5)
        candidates = build_fusion_candidates(dense, sparse, fused)
        by_id = {c.chunk_id: c for c in candidates}

        assert by_id["c1"].dense_rank == 1
        assert by_id["c1"].sparse_rank == 1
        assert by_id["c2"].sparse_rank is None
        assert by_id["c3"].dense_rank is None
        # 融合名次 1..n 连续且与 fused 顺序一致
        assert [c.fused_rank for c in candidates] == list(range(1, len(candidates) + 1))
        assert candidates[0].chunk_id == "c1"
        assert candidates[0].score == pytest.approx(2.0 / 61.0)


# ---------------------------------------------------------------------------
# 精排器
# ---------------------------------------------------------------------------
class TestIdentityReranker:
    def test_satisfies_contract(self):
        assert isinstance(IdentityReranker(), Reranker)

    def test_preserves_fused_order_and_truncates(self):
        items = [_item("c1", 0.03), _item("c2", 0.02), _item("c3", 0.01)]
        reranker = IdentityReranker()
        ranked = reranker.rerank("查询", items, top_k=2)
        assert [item.chunk.chunk_id for item in ranked] == ["c1", "c2"]
        # 分数原样透传（融合分即最终分）
        assert ranked[0].score == pytest.approx(0.03)

    def test_top_k_zero_and_empty(self):
        reranker = IdentityReranker()
        assert reranker.rerank("查询", [_item("c1", 0.1)], top_k=0) == []
        assert reranker.rerank("查询", [], top_k=5) == []


class TestBGEReranker:
    def test_satisfies_contract(self):
        assert isinstance(BGEReranker.__new__(BGEReranker), Reranker)

    @pytest.mark.skipif(_SENTENCE_TRANSFORMERS_INSTALLED, reason="sentence-transformers 已安装")
    def test_missing_dependency_hint(self):
        with pytest.raises(ImportError, match="bge"):
            BGEReranker()

    def test_rerank_orders_by_cross_encoder_score(self, monkeypatch):
        """注入桩 CrossEncoder 验证精排逻辑（分数替换/排序/截断）。"""

        class _StubCrossEncoder:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
                scores = {"最相关内容": 0.9, "次相关内容": 0.5, "无关内容": 0.1}
                return [scores[text] for _, text in pairs]

        stub = types.ModuleType("sentence_transformers")
        stub.CrossEncoder = _StubCrossEncoder
        monkeypatch.setitem(sys.modules, "sentence_transformers", stub)

        reranker = BGEReranker(model_name="stub-model")
        items = [
            _item("c1", 0.03, content="无关内容"),
            _item("c2", 0.02, content="最相关内容"),
            _item("c3", 0.01, content="次相关内容"),
        ]
        ranked = reranker.rerank("查询", items, top_k=2)
        assert [item.chunk.chunk_id for item in ranked] == ["c2", "c3"]
        # 精排分数替换融合分
        assert ranked[0].score == pytest.approx(0.9)
        assert ranked[1].score == pytest.approx(0.5)

    def test_rerank_empty_and_zero_top_k(self, monkeypatch):
        class _StubCrossEncoder:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
                return [0.5 for _ in pairs]

        stub = types.ModuleType("sentence_transformers")
        stub.CrossEncoder = _StubCrossEncoder
        monkeypatch.setitem(sys.modules, "sentence_transformers", stub)

        reranker = BGEReranker(model_name="stub-model")
        assert reranker.rerank("查询", [], top_k=5) == []
        assert reranker.rerank("查询", [_item("c1", 0.1)], top_k=0) == []
