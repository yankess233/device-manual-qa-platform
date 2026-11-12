"""本地检索实现：稠密/稀疏/混合召回、RRF 融合、交叉编码器精排、断崖截断。

为什么要有这一份而不是直接打 :8001：
- 消融实验需要**逐组件开关**，走线上图只能整体开关，改一次配置就要重启服务；
- 精排与融合是纯计算，本地复刻可以秒级跑完 5 组配置 × 50 题；
- 线上服务仍用 ``eval/service_eval.py`` 单独评测，形成"检索指标 + 端到端指标"两套证据。

与生产实现的差异（写进 README，别藏着）：
- Milvus 的 ``WeightedRanker(norm_score=True)`` 是**分数归一化后加权**，
  这里用 min-max 归一化近似复刻，绝对分数不等价，**只用于相对比较**；
- 索引是精确检索而非 HNSW 近似，理论上召回略高于线上（线上有近似误差）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """语料切片。"""

    chunk_id: str
    title: str
    content: str
    item_name: str = ""
    file_title: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "content": self.content,
            "item_name": self.item_name,
            "file_title": self.file_title,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Chunk":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            title=payload.get("title", ""),
            content=payload.get("content", ""),
            item_name=payload.get("item_name", ""),
            file_title=payload.get("file_title", ""),
        )


@dataclass
class Hit:
    """一条检索结果。"""

    chunk_id: str
    score: float


def _minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    low, high = float(scores.min()), float(scores.max())
    if high - low < 1e-12:
        return np.ones_like(scores)
    return (scores - low) / (high - low)


def weighted_rank(
    dense_hits: Sequence[Hit],
    sparse_hits: Sequence[Hit],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    limit: int = 10,
) -> list[Hit]:
    """复刻 Milvus ``WeightedRanker``：各自 min-max 归一化后加权求和。"""
    merged: dict[str, float] = {}
    for hits, weight in ((dense_hits, dense_weight), (sparse_hits, sparse_weight)):
        if not hits or weight <= 0:
            continue
        scores = _minmax(np.array([hit.score for hit in hits], dtype=np.float32))
        for hit, norm in zip(hits, scores):
            merged[hit.chunk_id] = merged.get(hit.chunk_id, 0.0) + weight * float(norm)
    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    return [Hit(chunk_id=cid, score=score) for cid, score in ranked[:limit]]


def rrf_fuse(
    rank_lists: Sequence[Sequence[Hit]],
    k: int = 60,
    max_results: int = 10,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    """Reciprocal Rank Fusion：score = Σ weight / (k + rank)。"""
    weights = list(weights) if weights else [1.0] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError("weights 数量必须与 rank_lists 一致")
    scores: dict[str, float] = {}
    for hits, weight in zip(rank_lists, weights):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [Hit(chunk_id=cid, score=score) for cid, score in ranked[:max_results]]


def cliff_cutoff(
    hits: Sequence[Hit], min_k: int = 3, max_k: int = 10, gap: float = 0.15
) -> list[Hit]:
    """按相邻分数断崖动态截断：遇到第一个大于 gap 的落差就停。"""
    if not hits:
        return []
    upper = min(len(hits), max_k)
    cut = upper
    for index in range(min_k, upper):
        if hits[index - 1].score - hits[index].score >= gap:
            cut = index
            break
    return list(hits[: max(min_k, cut)])


class HybridIndex:
    """内存里的精确混合索引。"""

    def __init__(self, chunks: Sequence[Chunk], dense: np.ndarray,
                 sparse: Sequence[dict[int, float]]) -> None:
        if len(chunks) != len(dense) or len(chunks) != len(sparse):
            raise ValueError("chunks / dense / sparse 长度不一致")
        self.chunks = list(chunks)
        self.ids = [chunk.chunk_id for chunk in self.chunks]
        self._pos = {cid: i for i, cid in enumerate(self.ids)}
        matrix = np.asarray(dense, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense = matrix / norms
        self.sparse = list(sparse)
        self._inverted: dict[int, list[tuple[int, float]]] = {}
        for doc_index, weights in enumerate(self.sparse):
            for token, weight in weights.items():
                self._inverted.setdefault(int(token), []).append((doc_index, float(weight)))

    def __len__(self) -> int:
        return len(self.chunks)

    def by_id(self, chunk_id: str) -> Chunk | None:
        position = self._pos.get(chunk_id)
        return None if position is None else self.chunks[position]

    def dense_search(self, query_dense: Sequence[float], limit: int = 10,
                     allowed: set[str] | None = None) -> list[Hit]:
        vector = np.asarray(query_dense, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        scores = self.dense @ vector
        order = np.argsort(-scores)
        hits: list[Hit] = []
        for index in order:
            cid = self.ids[int(index)]
            if allowed is not None and cid not in allowed:
                continue
            hits.append(Hit(chunk_id=cid, score=float(scores[int(index)])))
            if len(hits) >= limit:
                break
        return hits

    def sparse_search(self, query_sparse: dict[int, float], limit: int = 10,
                      allowed: set[str] | None = None) -> list[Hit]:
        scores: dict[int, float] = {}
        for token, weight in query_sparse.items():
            for doc_index, doc_weight in self._inverted.get(int(token), ()):
                scores[doc_index] = scores.get(doc_index, 0.0) + float(weight) * doc_weight
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        hits: list[Hit] = []
        for doc_index, score in ranked:
            cid = self.ids[doc_index]
            if allowed is not None and cid not in allowed:
                continue
            hits.append(Hit(chunk_id=cid, score=float(score)))
            if len(hits) >= limit:
                break
        return hits

    def search(self, dense: Sequence[float], sparse: dict[int, float], mode: str,
               limit: int = 10, dense_weight: float = 0.5, sparse_weight: float = 0.5,
               allowed: set[str] | None = None) -> list[Hit]:
        if mode == "dense":
            return self.dense_search(dense, limit=limit, allowed=allowed)
        if mode == "sparse":
            return self.sparse_search(sparse, limit=limit, allowed=allowed)
        if mode != "hybrid":
            raise ValueError(f"未知检索模式：{mode!r}")
        return weighted_rank(
            self.dense_search(dense, limit=limit, allowed=allowed),
            self.sparse_search(sparse, limit=limit, allowed=allowed),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            limit=limit,
        )


class CrossEncoderReranker:
    """bge-reranker-large 包装，带磁盘缓存（同一对文本只算一次）。"""

    def __init__(self, cache_path: str | None = None, device: str | None = None) -> None:
        from FlagEmbedding import FlagReranker

        model_path = os.getenv("BGE_RERANKER_LARGE", "")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"BGE_RERANKER_LARGE 不存在：{model_path!r}；可用 --reranker none 跳过精排"
            )
        device = device or os.getenv("BGE_DEVICE", "cpu")
        use_fp16 = os.getenv("BGE_FP16", "true").lower() in ("1", "true", "yes")
        logger.info("加载 reranker：%s（device=%s）", model_path, device)
        self._model = FlagReranker(model_name_or_path=model_path, device=device,
                                  use_fp16=use_fp16)
        default_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     ".cache", "rerank.json")
        self.cache_path = cache_path or default_cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self._cache: dict[str, float] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                self._cache = json.load(handle)
            logger.info("精排缓存命中 %d 条", len(self._cache))

    @staticmethod
    def _key(query: str, content: str) -> str:
        return hashlib.sha1(f"{query}\x00{content}".encode("utf-8")).hexdigest()

    def score(self, query: str, contents: Sequence[str]) -> list[float]:
        missing = [
            content for content in dict.fromkeys(contents)
            if self._key(query, content) not in self._cache
        ]
        if missing:
            pairs = [[query, content] for content in missing]
            raw = self._model.compute_score(pairs, normalize=True)
            if isinstance(raw, (int, float)):
                raw = [raw]
            for content, value in zip(missing, raw):
                self._cache[self._key(query, content)] = float(value)
            self._save()
        return [self._cache[self._key(query, content)] for content in contents]

    def _save(self) -> None:
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self._cache, handle)
        os.replace(tmp, self.cache_path)


def rerank_hits(reindex: HybridIndex, query: str, hits: Iterable[Hit],
                reranker: CrossEncoderReranker) -> list[Hit]:
    """用交叉编码器对候选重新打分，按新分数降序返回。"""
    hit_list = list(hits)
    if not hit_list:
        return []
    contents = []
    for hit in hit_list:
        chunk = reindex.by_id(hit.chunk_id)
        contents.append(chunk.content if chunk else "")
    scores = reranker.score(query, contents)
    rescored = [
        Hit(chunk_id=hit.chunk_id, score=score)
        for hit, score in zip(hit_list, scores)
    ]
    return sorted(rescored, key=lambda h: h.score, reverse=True)