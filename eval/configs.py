"""消融实验的配置定义。

每条配置对应简历里"消融实验表"的一行，只允许在相邻配置之间增加一个变量，
这样"提升了几个点"才能归因到具体组件。

注意：联网检索（web MCP）无法在本地检索评测里复现，它只在
``eval/service_eval.py``（走真实服务）里体现，因此不在下面的阶梯中。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AblationConfig:
    """一次检索实验的完整参数。"""

    name: str
    mode: str = "hybrid"  # dense | sparse | hybrid
    limit: int = 10  # 每路召回条数
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    use_hyde: bool = False  # 是否加入 HyDE 第二路召回
    use_rrf: bool = False  # 是否用 RRF 融合两路
    use_rerank: bool = False  # 是否用 bge-reranker-large 精排
    dynamic_cutoff: bool = False  # 是否按分数断崖动态截断
    cutoff_min_k: int = 3
    cutoff_max_k: int = 10
    cutoff_gap: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)


ABLATIONS: dict[str, AblationConfig] = {
    cfg.name: cfg
    for cfg in [
        AblationConfig(name="A1_dense", mode="dense"),
        AblationConfig(name="A2_hybrid", mode="hybrid"),
        AblationConfig(name="A3_hyde_rrf", mode="hybrid", use_hyde=True, use_rrf=True),
        AblationConfig(
            name="A4_rerank",
            mode="hybrid",
            use_hyde=True,
            use_rrf=True,
            use_rerank=True,
        ),
        AblationConfig(
            name="A5_cutoff",
            mode="hybrid",
            use_hyde=True,
            use_rrf=True,
            use_rerank=True,
            dynamic_cutoff=True,
        ),
    ]
}

DEFAULT_CONFIG_NAMES: tuple[str, ...] = tuple(ABLATIONS)


def resolve(names: str | None) -> list[AblationConfig]:
    """把 ``"A1_dense,A4_rerank"`` 这样的字符串解析成配置列表。"""
    if not names:
        return [ABLATIONS[name] for name in DEFAULT_CONFIG_NAMES]
    resolved: list[AblationConfig] = []
    for raw in names.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in ABLATIONS:
            raise KeyError(f"未知实验配置 {key!r}，可选：{', '.join(ABLATIONS)}")
        resolved.append(ABLATIONS[key])
    if not resolved:
        raise ValueError("至少需要一个实验配置")
    return resolved