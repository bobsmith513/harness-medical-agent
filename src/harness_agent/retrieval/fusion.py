"""RRF 融合与精排（M3）。

融合数学（Reciprocal Rank Fusion）::

    score(d) = Σ_{路径 p} 1 / (k + rank_p(d))    rank 从 1 计，k 默认 60

- 未进某路的候选该路贡献为 0（等价 rank→∞），双路命中的候选天然
  高于单路头部——稠密路捕捉语义近似、稀疏路捕捉词面精确命中，
  融合让两路共识上浮，这是"混合检索"的意义所在；
- k 越大，名次差异对融合分的边际影响越平缓（60 为文献常用常数，
  配置项 ``HARNESS_RETRIEVAL__RRF_K`` 可调）。

精排两级（共用 ``Reranker`` 契约，注入零逻辑分叉）：

- ``IdentityReranker``（默认）：融合名次即最终名次，直接截断
  （零依赖；``reranker_enabled=false`` 时注入）；
- ``BGEReranker``（可插拔）：bge-reranker-v2-m3 交叉编码器，
  ``uv sync --extra bge``（sentence-transformers）。
"""

from __future__ import annotations

from harness_agent.contracts.retrieval import RetrievedItem
from harness_agent.models.evidence import RetrievalCandidate

__all__ = [
    "BGEReranker",
    "IdentityReranker",
    "build_fusion_candidates",
    "rrf_fuse",
]


def rrf_fuse(
    dense: list[RetrievedItem],
    sparse: list[RetrievedItem],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[RetrievedItem]:
    """双路召回 RRF 融合：按融合分降序（并列按 chunk_id 升序）截断。

    输出的 ``score`` 即 RRF 融合分（进入精排前的统一口径），
    chunk 保留完整领域对象（来自首次出现的召回项）。
    """
    if k <= 0:
        raise ValueError(f"RRF 常数 k 必须为正数: {k}")

    contributions: dict[str, float] = {}
    chunks: dict[str, RetrievedItem] = {}
    for rank, item in enumerate(dense, start=1):
        chunk_id = item.chunk.chunk_id
        contributions[chunk_id] = contributions.get(chunk_id, 0.0) + 1.0 / (k + rank)
        chunks.setdefault(chunk_id, item)
    for rank, item in enumerate(sparse, start=1):
        chunk_id = item.chunk.chunk_id
        contributions[chunk_id] = contributions.get(chunk_id, 0.0) + 1.0 / (k + rank)
        chunks.setdefault(chunk_id, item)

    fused = [
        RetrievedItem(chunk=chunks[chunk_id].chunk, score=score)
        for chunk_id, score in contributions.items()
    ]
    fused.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return fused[: max(top_k, 0)]


def build_fusion_candidates(
    dense: list[RetrievedItem],
    sparse: list[RetrievedItem],
    fused: list[RetrievedItem],
) -> list[RetrievalCandidate]:
    """融合名次迹：稠密/稀疏名次 + 融合后名次的完整记录。

    供审计与可观测（哪路召回、各自名次、融合名次）；
    未进某路的名次为 None（M1 ``RetrievalCandidate`` 语义）。
    """
    dense_ranks = {item.chunk.chunk_id: rank for rank, item in enumerate(dense, start=1)}
    sparse_ranks = {item.chunk.chunk_id: rank for rank, item in enumerate(sparse, start=1)}
    fused_ranks = {item.chunk.chunk_id: rank for rank, item in enumerate(fused, start=1)}
    return [
        RetrievalCandidate(
            chunk_id=item.chunk.chunk_id,
            content=item.chunk.content,
            dense_rank=dense_ranks.get(item.chunk.chunk_id),
            sparse_rank=sparse_ranks.get(item.chunk.chunk_id),
            fused_rank=fused_ranks.get(item.chunk.chunk_id),
            score=item.score,
        )
        for item in fused
    ]


class IdentityReranker:
    """恒等精排（默认）：融合名次即最终名次，直接截断 top_k。"""

    name = "reranker:identity"

    def rerank(self, query: str, items: list[RetrievedItem], top_k: int) -> list[RetrievedItem]:
        return items[: max(top_k, 0)]


class BGEReranker:
    """bge-reranker-v2-m3 交叉编码器精排（可插拔真实实现）。

    安装：``uv sync --extra bge``；模型名来自
    ``HARNESS_RETRIEVAL__RERANKER_MODEL``（默认 BAAI/bge-reranker-v2-m3）。
    精排分数直接替换 RRF 融合分（``Evidence.score`` 语义：融合后经
    reranker 的最终得分）。
    """

    name = "reranker:bge"

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("sentence-transformers 未安装：uv sync --extra bge 后重试") from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, items: list[RetrievedItem], top_k: int) -> list[RetrievedItem]:
        limit = max(top_k, 0)
        if not items or limit == 0:
            return []
        pairs = [(query, item.chunk.content) for item in items]
        scores = self._model.predict(pairs)
        ranked = [
            RetrievedItem(chunk=item.chunk, score=float(score))
            for item, score in zip(items, scores, strict=True)
        ]
        ranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return ranked[:limit]
