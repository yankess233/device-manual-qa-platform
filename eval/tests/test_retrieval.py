"""检索组件单测：加权融合、RRF、断崖截断、稀疏倒排。"""

from __future__ import annotations

import numpy as np
import pytest

from eval.retrieval import (
    Chunk,
    Hit,
    HybridIndex,
    cliff_cutoff,
    rrf_fuse,
    weighted_rank,
)


def test_weighted_rank_minmax_tie_artifact():
    """min-max 归一化的已知副作用，写成测试固定住行为。

    dense=[a .9, b .5] → a=1.0, b=0.0；sparse=[b 10, c 5] → b=1.0, c=0.0。
    于是 a 和 b 都是 0.5 分并列为第一，顺序取决于谁先进字典（dense 先）。
    这不是 bug，而是"复刻 Milvus WeightedRanker 归一化"的代价：
    两路打分尺度不同时，min-max 会把每路的 top1 抬到 1.0、其余压到 0。
    真实场景里每路返回 10 条，这种极端压扁会明显减弱，
    但也说明"权重"不能只靠拍 0.5/0.5 —— 这正是消融实验要回答的问题。
    """
    dense = [Hit("a", 0.9), Hit("b", 0.5)]
    sparse = [Hit("b", 10.0), Hit("c", 5.0)]
    ranked = weighted_rank(dense, sparse, 0.5, 0.5, limit=3)
    scores = {hit.chunk_id: hit.score for hit in ranked}
    assert scores["a"] == pytest.approx(0.5)
    assert scores["b"] == pytest.approx(0.5)
    assert scores["c"] == pytest.approx(0.0)
    assert [hit.chunk_id for hit in ranked][:2] == ["a", "b"]


def test_weighted_rank_respects_weights():
    dense = [Hit("a", 1.0), Hit("b", 0.0)]
    sparse = [Hit("b", 1.0), Hit("a", 0.0)]
    dense_only = weighted_rank(dense, sparse, 1.0, 0.0, limit=2)
    sparse_only = weighted_rank(dense, sparse, 0.0, 1.0, limit=2)
    assert dense_only[0].chunk_id == "a"
    assert sparse_only[0].chunk_id == "b"


def test_weighted_rank_handles_constant_scores():
    # 所有分数相同：min-max 退化，不能出现 NaN
    ranked = weighted_rank([Hit("a", 0.3), Hit("b", 0.3)], [], 0.5, 0.5, limit=2)
    assert {hit.chunk_id for hit in ranked} == {"a", "b"}
    assert all(not np.isnan(hit.score) for hit in ranked)


def test_rrf_fusion_math():
    first = [Hit("a", 9.0), Hit("b", 8.0)]
    second = [Hit("b", 7.0), Hit("c", 6.0)]
    fused = rrf_fuse([first, second], k=60, max_results=3)
    scores = {hit.chunk_id: hit.score for hit in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert fused[0].chunk_id == "b"


def test_rrf_rejects_weight_length_mismatch():
    with pytest.raises(ValueError):
        rrf_fuse([[Hit("a", 1.0)]], weights=[1.0, 2.0])


def test_cliff_cutoff_cuts_at_gap():
    hits = [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.7), Hit("d", 0.2), Hit("e", 0.1)]
    # min_k=3 保证至少 3 条；3→4 的落差 0.5 ≥ 0.15，在 index=3 截断
    assert [hit.chunk_id for hit in cliff_cutoff(hits, 3, 10, 0.15)] == ["a", "b", "c"]


def test_cliff_cutoff_keeps_all_when_flat():
    hits = [Hit("a", 0.5), Hit("b", 0.49), Hit("c", 0.48)]
    assert len(cliff_cutoff(hits, 3, 10, 0.15)) == 3


def test_cliff_cutoff_caps_max_k():
    hits = [Hit(str(i), 1.0 - i * 0.001) for i in range(20)]
    assert len(cliff_cutoff(hits, 3, 5, 0.15)) == 5


def _tiny_index() -> HybridIndex:
    chunks = [
        Chunk("c0", "## 电阻测量", "将功能转盘置于最高电阻Ω位置"),
        Chunk("c1", "## 电压测量", "将功能转盘置于V DC的位置"),
        Chunk("c2", "## 安全手册", "请勿超过量程"),
    ]
    dense = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    sparse = [{1: 1.0}, {2: 1.0}, {1: 0.5, 2: 0.5}]
    return HybridIndex(chunks, dense, sparse)


def test_dense_search_orders_by_similarity():
    index = _tiny_index()
    hits = index.dense_search([1.0, 0.0], limit=3)
    assert hits[0].chunk_id == "c0"


def test_sparse_search_uses_inverted_index():
    index = _tiny_index()
    hits = index.sparse_search({2: 2.0}, limit=3)
    assert hits[0].chunk_id == "c1"


def test_search_rejects_unknown_mode():
    index = _tiny_index()
    with pytest.raises(ValueError):
        index.search([1.0, 0.0], {}, mode="magic")


def test_index_rejects_length_mismatch():
    with pytest.raises(ValueError):
        HybridIndex([Chunk("c0", "t", "c")], np.zeros((2, 2)), [{1: 1.0}])