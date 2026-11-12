"""把一次（或多次）评测 run 固化成可提交的 Markdown 报告。

为什么要有这个脚本而不是手抄：
- 手抄的表格一定会和代码漂移（改一次切分，README 里的数字就过期）；
- 报告里必须带**可复现信息**（commit、语料指纹、数据集指纹、模型、思考模式），
  这些从 `summary.json` / `config.json` 里读，不靠记忆；
- 面试官/评审者 clone 下来看不到 `eval/results/`（已 gitignore），
  仓库里必须有一份固化的证据。

用法::

    python -m eval.make_report --runs eval/results/model-qwen38-nothink \
        --compare eval/results/model-qwen-flash,eval/results/model-qwen38-flash \
        --out eval/REPORT-baseline.md
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import median
from typing import Sequence


def load_run(path: str) -> dict:
    with open(os.path.join(path, "summary.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(path: str) -> list[dict]:
    rows_path = os.path.join(path, "per_query.jsonl")
    if not os.path.exists(rows_path):
        return []
    with open(rows_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * p / 100.0
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def main_table(payload: dict, k: int) -> list[str]:
    lines = [
        f"| 配置 | Hit@{k} | Recall@{k} | Precision@{k} | MRR@{k} | nDCG@{k} | 平均文档数 | 平均上下文字数 | P95 检索耗时 | HyDE 额外调用 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, block in payload["configs"].items():
        metrics = block[f"k={k}"]
        hyde = block.get("hyde")
        if not hyde:
            hyde_text = "—"
        elif hyde["cold_calls"]:
            hyde_text = f"{hyde['avg_ms_cold']:.0f} ms/题（冷）"
        else:
            hyde_text = "缓存命中"
        lines.append(
            f"| `{name}` | {metrics[f'hit@{k}']:.3f} | {metrics[f'recall@{k}']:.3f} | "
            f"{metrics[f'precision@{k}']:.3f} | {metrics[f'mrr@{k}']:.3f} | {metrics[f'ndcg@{k}']:.3f} | "
            f"{metrics['avg_docs']:.1f} | {metrics['avg_context_chars']:.0f} | "
            f"{metrics['p95_latency_ms']:.1f} ms | {hyde_text} |"
        )
    return lines


def by_type_table(payload: dict, k: int) -> list[str]:
    best = max(payload["configs"], key=lambda n: payload["configs"][n][f"k={k}"][f"hit@{k}"])
    lines = [
        f"最佳配置：`{best}`",
        "",
        f"| 题型 | 题数 | Hit@{k} | MRR@{k} | nDCG@{k} |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item_type, block in payload["configs"][best]["by_type"].items():
        if block["n_scored"] == 0:
            lines.append(f"| `{item_type}` | {block['n']} | — | — | — |")
            continue
        lines.append(
            f"| `{item_type}` | {block['n_scored']} | {block[f'hit@{k}']:.3f} | "
            f"{block[f'mrr@{k}']:.3f} | {block[f'ndcg@{k}']:.3f} |"
        )
    return lines


def hyde_latency(rows: list[dict]) -> list[str]:
    cold = sorted(row["hyde_ms"] for row in rows if row.get("hyde_cached") is False)
    if not cold:
        return ["（本次 run 没有 HyDE 冷启动调用）"]
    return [
        "| 调用数 | p50 | p95 | 平均 | 最大 |",
        "| --- | --- | --- | --- | --- |",
        f"| {len(cold)} | {median(cold):.0f} ms | {pct(cold, 95):.0f} ms | "
        f"{sum(cold) / len(cold):.0f} ms | {max(cold):.0f} ms |",
    ]


def misses(rows: list[dict], config: str, k: int) -> list[str]:
    picked = [row for row in rows if row["config"] == config and row.get("gold")]
    missed = [row for row in picked if not (set(row["ranked"][:k]) & set(row["gold"]))]
    if not missed:
        return ["本次 run 该配置无未命中题目。"]
    lines = ["| 题号 | 题型 | 问题 | gold | 实际 top-1 |",
             "| --- | --- | --- | --- | --- |"]
    for row in missed:
        lines.append(
            f"| {row['id']} | `{row['type']}` | {row['question']} | "
            f"{','.join(row['gold'])} | {row['top1_title'].strip() or '—'} |"
        )
    return lines


def compare_table(runs: list[tuple[str, dict, list[dict]]], k: int) -> list[str]:
    """多 run 对照。

    注意：思考模式没有写进历史 run 的元信息（只记了 `llm_model`），
    所以这里只按 run_id + 模型对照；思考模式口径见 `eval/model-compare.md`。
    """
    lines = [
        "| run | LLM | A2 Hit@5 | A3 Hit@5 | A4 Hit@5 | A4 MRR@5 | HyDE 冷启动平均 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload, rows in runs:
        meta = payload["meta"]
        configs = payload["configs"]
        cold = [row["hyde_ms"] for row in rows if row.get("hyde_cached") is False]
        cold_text = f"{sum(cold) / len(cold):.0f} ms" if cold else "—"
        lines.append(
            f"| `{meta['run_id']}` | `{meta.get('llm_model') or 'n/a'}` | "
            f"{configs['A2_hybrid'][f'k={k}']['hit@5']:.3f} | "
            f"{configs['A3_hyde_rrf'][f'k={k}']['hit@5']:.3f} | "
            f"{configs['A4_rerank'][f'k={k}']['hit@5']:.3f} | "
            f"{configs['A4_rerank'][f'k={k}']['mrr@5']:.3f} | {cold_text} |"
        )
    return lines


def thinking_text(meta: dict) -> str:
    """思考模式取 run 自己的元信息；老 run 没这个字段就如实写"未记录"，不猜。"""
    value = str(meta.get("llm_thinking") or "").strip()
    return value or "未记录"


def build(primary: str, comparisons: list[str], out_path: str, k: int) -> str:
    payload = load_run(primary)
    meta = payload["meta"]
    rows = load_rows(primary)
    names = list(payload["configs"])
    last_config = names[-1]

    lines = [
        f"# 检索评测报告（{meta['run_id']}）",
        "",
        "> 本文件由 `python -m eval.make_report` 从 `summary.json` / `per_query.jsonl` 直接生成，"
        "不是手抄。改了代码就重跑、重新生成，别手工改数字。",
        "",
        "## 1. 可复现信息",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 运行时间 | {meta['timestamp']} |",
        f"| commit | `{meta['git_sha']}`（评测跑于该提交；后续文档/脚本提交不改变指标口径） |",
        f"| 题目数 | {meta['n_items']}（{meta['type_counts']}） |",
        f"| 语料 | {meta['n_chunks']} 切片，fingerprint `{meta['corpus_fp']}` |",
        f"| 数据集 | fingerprint `{meta['dataset_fp']}` |",
        f"| 编码器 / 精排 | `{meta['embedder']}` / `{meta['reranker']}` |",
        f"| LLM | `{meta.get('llm_model') or 'n/a'}`（思考模式：{thinking_text(meta)}） |",
        f"| 指标 | Hit / Recall / Precision / MRR / nDCG @{k} |",
        "",
        "> 该 run 跑于元信息补丁之前，`llm_thinking` 未落库（run_id 后缀 `nothink` 表示关闭思考）；"
        "口径与三次 run 的完整对照见 `eval/model-compare.md`。",
        "",
        "## 2. 检索消融",
        "",
        "每次只加一个变量：稠密 → +稀疏 → +HyDE&RRF → +精排 → +断崖截断。",
        "",
        *main_table(payload, k),
        "",
        "## 3. 分题型",
        "",
        *by_type_table(payload, k),
        "",
        "> `out_of_scope` / `negative` 没有 gold 切片，检索指标无定义；"
        "它们的正确性只能由答案层的弃答率衡量（`eval/service_eval.py`）。",
        "",
        "## 4. HyDE 调用延迟（冷启动）",
        "",
        *hyde_latency(rows),
        "",
        "## 5. 未命中的题目（最佳配置之外的全量核对）",
        "",
        *misses(rows, last_config, k),
        "",
    ]

    if comparisons:
        runs = [(name, load_run(name), load_rows(name)) for name in [primary, *comparisons]]
        lines += [
            "## 6. 对照 run",
            "",
            *compare_table(runs, k),
            "",
            "> 检索指标与 LLM 选型无关：若表中 A2/A3 出现差异，先怀疑单题翻转（50 题下 1 题 = 2.5 个点），"
            "而不是模型优劣。详见 `eval/model-compare.md`。",
            "",
        ]

    lines += [
        "## 7. 复现",
        "",
        "```bash",
        "python -m eval.link_gold                       # 校验标注能落到当前语料",
        "python -m eval.run_eval --run-id <run_id>      # 重跑消融",
        "python -m eval.make_report --runs eval/results/<run_id> --out eval/REPORT-baseline.md",
        "```",
        "",
        "## 8. 解读边界",
        "",
        "- 本地是精确检索，线上 Milvus 为 HNSW 近似检索，**绝对值有差异，只做相对比较**；",
        "- `WeightedRanker` 用 min-max 归一化近似复刻，与 Milvus 的分数不等价；",
        "- 50 题规模下，单题翻转即 ±2.5 个百分点，**单题差异不解读**；",
        "- 多轮题使用 `rewrite`（理想改写）检索，衡量的是检索上限，改写质量另见 `service_eval.py`；",
        "- 语料从干净 Markdown 重切，不是 Milvus 里那份编码损坏的旧数据。",
        "",
    ]
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成可提交的评测报告")
    parser.add_argument("--runs", required=True, help="主 run 目录（逗号分隔时取第一个为主）")
    parser.add_argument("--compare", default="", help="额外对照 run，逗号分隔")
    parser.add_argument("--out", default="eval/REPORT-baseline.md")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    if not runs:
        raise SystemExit("--runs 不能为空")
    comparisons = [r.strip() for r in args.compare.split(",") if r.strip()]
    path = build(runs[0], comparisons, args.out, args.k)
    print(f"已生成 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())