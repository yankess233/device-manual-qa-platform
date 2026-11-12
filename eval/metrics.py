"""检索评测指标。

全部为纯函数，无第三方依赖，便于单测与复用。

约定：
- ``ranked`` 是按相关性降序排列的 chunk_id 列表；
- ``gold`` 是标注为相关的 chunk_id 集合；
- 当 ``gold`` 为空（例如"手册外"问题）时，检索类指标返回 0，
  由调用方用 ``expect_abstain`` 单独统计，不混入平均值。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def hit_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """命中率：前 k 个里是否至少有一个 gold。"""
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return 1.0 if any(doc in gold_set for doc in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """召回率：前 k 个覆盖了多少比例的 gold。"""
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(ranked[:k]) & gold_set) / len(gold_set)


def precision_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """精确率：前 k 个里 gold 的占比。用于衡量上下文是否被噪声稀释。"""
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(ranked[:k]) & gold_set) / k


def mrr_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """平均倒数排名：第一个 gold 出现位置的倒数。"""
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    for index, doc in enumerate(ranked[:k], start=1):
        if doc in gold_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """二值相关性的 nDCG@k（每个 gold 视作 1 级相关）。"""
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    dcg = 0.0
    for index, doc in enumerate(ranked[:k], start=1):
        if doc in gold_set:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(gold_set), k)
    if ideal_hits == 0:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / ideal


def mean(values: Iterable[float]) -> float:
    """算术平均；空序列返回 0.0，避免除零污染汇总表。"""
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数（与 numpy 默认口径一致），用于 P50/P95 延迟。"""
    if not values:
        return 0.0
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p 必须在 [0, 100] 之间，实际 {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize(rows: Sequence[dict], k: int = 5) -> dict:
    """把逐条结果聚合成一组指标。

    ``rows`` 每项至少包含 ``ranked``、``gold``，可选 ``latency_ms``、
    ``context_chars``、``n_docs``。gold 为空的行不计入检索指标均值。
    """
    scored = [row for row in rows if row.get("gold")]
    return {
        "n": len(rows),
        "n_scored": len(scored),
        f"hit@{k}": mean(hit_at_k(r["ranked"], r["gold"], k) for r in scored),
        f"recall@{k}": mean(recall_at_k(r["ranked"], r["gold"], k) for r in scored),
        f"precision@{k}": mean(precision_at_k(r["ranked"], r["gold"], k) for r in scored),
        f"mrr@{k}": mean(mrr_at_k(r["ranked"], r["gold"], k) for r in scored),
        f"ndcg@{k}": mean(ndcg_at_k(r["ranked"], r["gold"], k) for r in scored),
        "avg_docs": mean(r.get("n_docs", 0) for r in rows),
        "avg_context_chars": mean(r.get("context_chars", 0) for r in rows),
        "p50_latency_ms": percentile([r.get("latency_ms", 0.0) for r in rows], 50),
        "p95_latency_ms": percentile([r.get("latency_ms", 0.0) for r in rows], 95),
    }