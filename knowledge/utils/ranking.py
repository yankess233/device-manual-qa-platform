"""排序后处理：断崖截断与产品名置信度决策。

抽成纯函数的目的：这两段逻辑原本埋在 `rerank.py` / `item_name_confirm.py` 里，
这两个模块 import 就会拉起 FlagEmbedding + torch，导致单测没法在 CI 里跑，
而它们恰恰是最容易改错的业务判断（阈值、边界、并列）。

依赖为零，可直接被 `tests/unit/` 覆盖。
"""

from __future__ import annotations

from typing import Sequence


def cliff_cutoff(
    scores: Sequence[float],
    min_k: int = 3,
    max_k: int = 10,
    gap: float = 0.15,
) -> int:
    """返回断崖截断后的条数。

    规则：在 `max_k` 范围内从前往后找第一个相邻分数差 > `gap` 的位置，
    在断崖处截断；至少保留 `min_k` 条；没有断崖就保留 `max_k`（或全部）。

    例：[0.99, 0.98, 0.01, 0.001] → 第 3 条与第 2 条差 0.97 > 0.15 → 保留 2 条，
    但受 min_k=3 约束 → 3。
    """
    total = len(scores)
    upper = min(max_k, total)
    lower = min(min_k, upper)

    if upper <= lower:
        return upper

    cut = upper
    for index in range(upper - 1):
        if scores[index] - scores[index + 1] > gap:
            cut = index + 1
            break
    return max(cut, lower)


def decide_item_names(
    matches: Sequence[dict],
    extracted: str,
    high: float = 0.7,
    mid: float = 0.6,
    max_options: int = 3,
) -> tuple[list[str], list[str]]:
    """按置信度把候选产品名分成「已确认」与「待用户选择」两组。

    - score >= high：高置信。只有 1 个→直接确认；多个→全部进选项（让用户选）；
      若其中存在与 `extracted` 完全同名的一个→直接确认它。
    - mid <= score < high：中置信，进选项。
    - score < mid：丢弃。

    返回 `(confirmed, options)`，两者都会去重且各自最多 `max_options` 个。
    """
    ranked = sorted(matches, key=lambda item: item.get("score", 0.0), reverse=True)

    confirmed: list[str] = []
    options: list[str] = []

    def _uniq(items: list[str], limit: int) -> list[str]:
        """先去重再截断——顺序反了会出现 ['A','A','B'] 截成 ['A','B'] 少一个。"""
        seen: list[str] = []
        for item in items:
            if item not in seen:
                seen.append(item)
        return seen[:limit]

    high_names = _uniq([m["item_name"] for m in ranked if m.get("score", 0.0) >= high], max_options * 10)
    if high_names:
        same = next((name for name in high_names if name == extracted), None)
        if same:
            confirmed.append(same)
        elif len(high_names) == 1:
            confirmed.append(high_names[0])
        else:
            options.extend(_uniq(high_names, max_options))
    else:
        options.extend(
            _uniq([m["item_name"] for m in ranked if mid <= m.get("score", 0.0) < high], max_options)
        )

    confirmed = _uniq(confirmed, max_options)
    options = [name for name in _uniq(options, max_options) if name not in confirmed]
    return confirmed, options