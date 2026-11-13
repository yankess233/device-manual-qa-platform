"""断崖截断与商品名置信度决策的单元测试（P1 #9 点名的两块）。"""

from __future__ import annotations

import pytest

from knowledge.utils.ranking import cliff_cutoff, decide_item_names

# ---------------- 断崖截断 ----------------

def test_cutoff_at_cliff():
    # 0.99 → 0.98 差 0.01；0.98 → 0.01 差 0.97，断崖在 index 1
    assert cliff_cutoff([0.99, 0.98, 0.01, 0.001], min_k=1, max_k=10, gap=0.15) == 2


def test_cutoff_respects_min_k():
    # 断崖在第 1 条之后，但 min_k=3 兜底
    assert cliff_cutoff([0.99, 0.01, 0.001, 0.0001], min_k=3, max_k=10, gap=0.15) == 3


def test_cutoff_respects_max_k():
    scores = [1.0 - i * 0.001 for i in range(20)]
    assert cliff_cutoff(scores, min_k=3, max_k=8, gap=0.15) == 8


def test_cutoff_no_cliff_returns_upper_bound():
    assert cliff_cutoff([0.9, 0.89, 0.88, 0.87], min_k=3, max_k=10, gap=0.15) == 4


def test_cutoff_short_list_returns_all():
    assert cliff_cutoff([0.9, 0.8], min_k=3, max_k=10, gap=0.15) == 2
    assert cliff_cutoff([], min_k=3, max_k=10, gap=0.15) == 0


def test_cutoff_gap_equal_is_not_a_cliff():
    # 严格大于才算断崖：差正好 0.15 不截断
    assert cliff_cutoff([0.95, 0.80, 0.79], min_k=1, max_k=10, gap=0.15) == 3


# ---------------- 置信度决策 ----------------

def test_single_high_confidence_is_confirmed():
    matches = [{"item_name": "RS-12 数字万用表", "score": 0.91}]
    confirmed, options = decide_item_names(matches, extracted="RS-12")
    assert confirmed == ["RS-12 数字万用表"]
    assert options == []


def test_multiple_high_confidence_goes_to_options():
    matches = [
        {"item_name": "A 型", "score": 0.95},
        {"item_name": "B 型", "score": 0.92},
    ]
    confirmed, options = decide_item_names(matches, extracted="型号")
    assert confirmed == []
    assert options == ["A 型", "B 型"]


def test_exact_name_match_confirms_even_with_many_candidates():
    matches = [
        {"item_name": "A 型", "score": 0.95},
        {"item_name": "B 型", "score": 0.91},
    ]
    confirmed, _ = decide_item_names(matches, extracted="B 型")
    assert confirmed == ["B 型"]


def test_mid_confidence_goes_to_options():
    matches = [{"item_name": "C 型", "score": 0.65}]
    confirmed, options = decide_item_names(matches, extracted="型号")
    assert confirmed == []
    assert options == ["C 型"]


def test_low_confidence_is_dropped():
    matches = [{"item_name": "D 型", "score": 0.3}]
    confirmed, options = decide_item_names(matches, extracted="型号")
    assert confirmed == [] and options == []


def test_options_dedupe_and_limit():
    matches = [
        {"item_name": "A", "score": 0.95},
        {"item_name": "A", "score": 0.94},
        {"item_name": "B", "score": 0.93},
        {"item_name": "C", "score": 0.92},
        {"item_name": "D", "score": 0.91},
    ]
    confirmed, options = decide_item_names(matches, extracted="x", max_options=3)
    assert confirmed == []
    assert options == ["A", "B", "C"]


@pytest.mark.parametrize("matches", [[], [{"item_name": "E", "score": 0.0}]])
def test_empty_or_zero_results(matches):
    confirmed, options = decide_item_names(matches, extracted="x")
    assert confirmed == [] and options == []