"""指标函数的单测：全部用手算值对齐，避免"指标写错但看起来合理"。"""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    summarize,
)

RANKED = ["a", "b", "c", "d"]
GOLD = {"b", "d"}


def test_hit_at_k_boundary():
    assert hit_at_k(RANKED, {"a"}, 1) == 1.0
    assert hit_at_k(RANKED, {"b"}, 1) == 0.0
    assert hit_at_k(RANKED, {"b"}, 2) == 1.0
    assert hit_at_k([], {"a"}, 5) == 0.0


def test_hit_at_k_empty_gold_is_zero():
    assert hit_at_k(RANKED, set(), 5) == 0.0


def test_recall_at_k():
    assert recall_at_k(RANKED, GOLD, 2) == pytest.approx(0.5)  # 只覆盖 b
    assert recall_at_k(RANKED, GOLD, 4) == pytest.approx(1.0)


def test_precision_at_k():
    assert precision_at_k(RANKED, GOLD, 2) == pytest.approx(0.5)
    assert precision_at_k(RANKED, GOLD, 4) == pytest.approx(0.5)


def test_mrr_at_k():
    assert mrr_at_k(RANKED, {"b"}, 5) == pytest.approx(0.5)
    assert mrr_at_k(RANKED, {"c"}, 5) == pytest.approx(1 / 3)
    assert mrr_at_k(RANKED, {"z"}, 5) == 0.0


def test_ndcg_at_k_single_gold():
    # gold 在位置 2：dcg = 1/log2(3)，理想排序也在位置 1 → ideal = 1
    expected = 1 / math.log2(3)
    assert ndcg_at_k(RANKED, {"b"}, 2) == pytest.approx(expected)


def test_ndcg_at_k_two_gold():
    # 位置 2 和 4：dcg = 1/log2(3) + 1/log2(5)；ideal = 1 + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, GOLD, 4) == pytest.approx(dcg / ideal)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["b", "d", "a"], {"b", "d"}, 3) == pytest.approx(1.0)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0) == pytest.approx(1.0)
    assert percentile([1, 2, 3, 4], 100) == pytest.approx(4.0)
    assert percentile([7], 95) == pytest.approx(7.0)
    assert percentile([], 95) == 0.0


def test_percentile_rejects_bad_argument():
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_summarize_skips_unscored_rows():
    rows = [
        {"ranked": ["b", "a"], "gold": ["b"], "latency_ms": 10, "n_docs": 2, "context_chars": 100},
        {"ranked": ["x"], "gold": [], "latency_ms": 30, "n_docs": 1, "context_chars": 50},
    ]
    result = summarize(rows, k=2)
    assert result["n"] == 2
    assert result["n_scored"] == 1
    assert result["hit@2"] == 1.0
    assert result["avg_docs"] == pytest.approx(1.5)
    assert result["avg_context_chars"] == pytest.approx(75.0)