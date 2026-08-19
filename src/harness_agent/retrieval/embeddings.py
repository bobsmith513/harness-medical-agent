"""嵌入提供方（M3）：默认零依赖哈希嵌入，BGE 可插拔。

- ``HashingEmbeddingProvider``（demo 默认）：词元稳定哈希投影到固定
  维度并 L2 归一化。相似文本（共享词元/汉字二元组）余弦相似度更高，
  足以驱动双路召回演示；
- ``BGEEmbeddingProvider``（可插拔真实实现）：BGE-large-zh，
  依赖可选包 ``uv sync --extra bge``（sentence-transformers），
  与哈希实现共用 ``EmbeddingProvider`` 契约，替换零逻辑分叉。
"""

from __future__ import annotations

import hashlib
import math

from harness_agent.contracts.retrieval import Embedding
from harness_agent.retrieval.tokenizer import tokenize

__all__ = ["BGEEmbeddingProvider", "HashingEmbeddingProvider"]


class HashingEmbeddingProvider:
    """确定性哈希嵌入（零依赖，demo / 测试默认）。"""

    def __init__(self, dim: int = 1024) -> None:
        if dim <= 0:
            raise ValueError(f"嵌入维度必须为正数: {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[Embedding]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Embedding:
        vector = [0.0] * self._dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self._dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(component * component for component in vector))
        if norm > 0.0:
            vector = [component / norm for component in vector]
        return vector


class BGEEmbeddingProvider:
    """BGE-large-zh 嵌入（可插拔真实实现，CPU/GPU 由 sentence-transformers 自检）。

    安装：``uv sync --extra bge``；模型名来自
    ``HARNESS_RETRIEVAL__EMBEDDING_MODEL``（默认 BAAI/bge-large-zh）。
    """

    def __init__(self, model_name: str = "BAAI/bge-large-zh", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers 未安装：uv sync --extra bge 后重试") from exc
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> list[Embedding]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
