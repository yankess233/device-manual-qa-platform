"""向量编码器：真实 BGE-M3 + 无依赖的 hash 兜底 + 磁盘缓存。

设计要点：
1. **复用生产代码**：真实模式直接调用 ``knowledge.utils.embedding_util``，
   保证评测与线上是同一条编码路径，不是"评测一套、线上一套"。
2. **磁盘缓存**：消融实验会对同一批 query 反复编码 5~6 次；
   缓存把"唯一文本数"控制住，直接把实验成本从 N×K 降到 N。
3. **hash 兜底**：没有 GPU、没下载模型时也能把整条链路跑通（CI / 冒烟测试），
   但它只是字符级词袋，**产出的数字不能作为结论**。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


@dataclass
class Vector:
    """一条文本的双向量。"""

    dense: list[float]
    sparse: dict[int, float]


class Embedder(Protocol):
    name: str
    dim: int

    def encode_queries(self, texts: Sequence[str]) -> list[Vector]: ...


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class BgeM3Embedder:
    """BGE-M3（稠密 + 稀疏），与生产环境同源。"""

    name = "bge-m3"

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 use_fp16: bool | None = None) -> None:
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction

        model_path = model_path or os.getenv("BGE_M3_PATH", "")
        device = device or os.getenv("BGE_DEVICE", "cpu")
        if use_fp16 is None:
            use_fp16 = os.getenv("BGE_FP16", "true").lower() in ("1", "true", "yes")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"BGE_M3_PATH 不存在：{model_path!r}；"
                "请在 knowledge/.env 配置，或改用 --embedder hash 做冒烟测试"
            )
        logger.info("加载 BGE-M3：%s（device=%s, fp16=%s）", model_path, device, use_fp16)
        self._fn = BGEM3EmbeddingFunction(
            model_name=model_path, device=device, use_fp16=use_fp16
        )
        self.dim = 1024

    def encode_queries(self, texts: Sequence[str]) -> list[Vector]:
        from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

        result = generate_bge_m3_hybrid_vectors(self._fn, list(texts), is_query=True)
        return [
            Vector(dense=result["dense"][i], sparse=result["sparse"][i])
            for i in range(len(texts))
        ]

    def encode_documents(self, texts: Sequence[str]) -> list[Vector]:
        from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

        result = generate_bge_m3_hybrid_vectors(self._fn, list(texts), is_query=False)
        return [
            Vector(dense=result["dense"][i], sparse=result["sparse"][i])
            for i in range(len(texts))
        ]


class HashEmbedder:
    """字符级词袋 + 哈希投影。仅用于冒烟测试，数值无语义。"""

    name = "hash"
    dim = 1024
    _TOKEN = re.compile(r"[A-Za-z0-9_.%]+|[\u4e00-\u9fff]")

    def encode_queries(self, texts: Sequence[str]) -> list[Vector]:
        return [self._encode(text) for text in texts]

    encode_documents = encode_queries

    def _encode(self, text: str) -> Vector:
        tokens = self._TOKEN.findall(text.lower())
        if not tokens:
            tokens = [text[:8] or "_empty_"]
        dense = np.zeros(self.dim, dtype=np.float32)
        sparse: dict[int, float] = {}
        for token in tokens:
            bucket = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % self.dim
            dense[bucket] += 1.0
            sparse_id = int(hashlib.md5(token.encode()).hexdigest()[8:14], 16) % 100_000
            sparse[sparse_id] = sparse.get(sparse_id, 0.0) + 1.0
        norm = float(np.linalg.norm(dense))
        if norm > 0:
            dense /= norm
        return Vector(dense=dense.tolist(), sparse=sparse)


class CachedEmbedder:
    """给任意 Embedder 套一层 JSON 缓存（按文本 sha1 命中）。"""

    def __init__(self, inner: Embedder, cache_path: str | None = None) -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.cache_path = cache_path or os.path.join(CACHE_DIR, f"embed_{inner.name}.json")
        self._cache: dict[str, dict] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                self._cache = json.load(handle)
            logger.info("嵌入缓存命中 %d 条：%s", len(self._cache), self.cache_path)

    def encode_queries(self, texts: Sequence[str]) -> list[Vector]:
        missing = [text for text in dict.fromkeys(texts) if text_key(text) not in self._cache]
        if missing:
            logger.info("需要新编码 %d 条文本", len(missing))
            vectors = self.inner.encode_queries(missing)  # type: ignore[attr-defined]
            for text, vector in zip(missing, vectors):
                self._cache[text_key(text)] = {
                    "dense": [float(x) for x in vector.dense],
                    "sparse": {str(k): float(v) for k, v in vector.sparse.items()},
                }
            self._save()
        return [self._lookup(text) for text in texts]

    encode_documents = encode_queries

    def _lookup(self, text: str) -> Vector:
        item = self._cache[text_key(text)]
        return Vector(
            dense=item["dense"],
            sparse={int(k): float(v) for k, v in item["sparse"].items()},
        )

    def _save(self) -> None:
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self._cache, handle)
        os.replace(tmp, self.cache_path)


def build(name: str) -> CachedEmbedder:
    """按名称构造编码器。"""
    key = (name or "bge").lower()
    if key in ("bge", "bge-m3", "real"):
        return CachedEmbedder(BgeM3Embedder())
    if key in ("hash", "mock", "smoke"):
        logger.warning("使用 hash 兜底编码器：链路可跑通，但指标无意义，不可作为结论")
        return CachedEmbedder(HashEmbedder())
    raise ValueError(f"未知 embedder：{name!r}（可选 bge / hash）")