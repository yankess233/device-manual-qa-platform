"""对比多次评测结果（例如换 LLM 前后）。

用法::

    python -m eval.compare_runs eval/results/model-qwen-flash eval/results/model-qwen38-flash
    python -m eval.compare_runs --metric mrr@5 --k 5 A B

设计要点：
- 换 LLM 时，A1（纯稠密）/A2（混合）**不应该变**——它们不调用 LLM。
  如果这两行变了，说明对比不成立（缓存串了 / 语料变了 / 题目变了），先查这个再看别的。
- 表里同时给出绝对值和 delta，delta 用 `+0.000` / `-0.000` 格式，避免"看着像提升"。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

DEFAULT_METRICS = ("hit@5", "recall@5", "mrr@5", "ndcg@5", "avg_docs", "avg_context_chars")


def load_run(path: str) -> dict:
    summary_path = os.path.join(path, "summary.json") if os.path.isdir(path) else path
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_of(block: dict, metric: str, k: int) -> float:
    if metric in block:
        return float(block[metric])
    return float(block.get(f"{metric}@{k}", 0.0))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对比多个评测 run")
    parser.add_argument("runs", nargs="+", help="run 目录或 summary.json 路径")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--markdown", default=None, help="把对照表写到文件")
    args = parser.parse_args(argv)
    if len(args.runs) < 2:
        raise SystemExit("至少给两个 run 才能对比")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    runs = [(os.path.basename(os.path.normpath(p)), load_run(p)) for p in args.runs]

    configs: list[str] = []
    for _, payload in runs:
        for name in payload["configs"]:
            if name not in configs:
                configs.append(name)

    lines = ["| 配置 | 指标 | " + " | ".join(name for name, _ in runs) + " | Δ |",
             "| --- | --- | " + " | ".join("---" for _ in runs) + " | --- |"]
    plain: list[str] = []
    for config in configs:
        for metric in metrics:
            values = []
            for _, payload in runs:
                block = payload["configs"].get(config, {})
                block = block.get(f"k={args.k}", block)
                values.append(metric_of(block, metric, args.k) if block else float("nan"))
            delta = values[-1] - values[0]
            row = "| {} | {} | {} | {:+.3f} |".format(
                config, metric, " | ".join(f"{v:.3f}" for v in values), delta
            )
            lines.append(row)
            plain.append(row)
    table = "\n".join(lines)
    print(table)

    if args.markdown:
        header = f"# 评测对比（k={args.k}）\n\n" + "\n".join(
            f"- `{name}`：LLM=`{payload['meta'].get('llm_model', 'n/a')}`，run_id=`{payload['meta'].get('run_id', '')}`"
            for name, payload in runs
        ) + "\n\n"
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(header + table + "\n")
        print(f"\n已写入 {args.markdown}")

    models = [payload["meta"].get("llm_model", "n/a") for _, payload in runs]
    if len(set(models)) > 1:
        print("\n注意：这几组 run 的 LLM 不同（" + " vs ".join(models) + "）。")
        print("A1/A2 不应有差异；若差异非 0，先排查语料/题目/缓存，再解读 A3 之后的变化。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())