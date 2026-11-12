"""检索消融实验入口。

用法（在仓库根目录 ``shoper_brain/`` 下执行）::

    # 1) 首次：从干净的 Markdown 重建评测语料
    python -m eval.run_eval --rebuild-corpus

    # 2) 跑全部 5 组消融（需要 GPU / BGE 权重 / DashScope key）
    python -m eval.run_eval

    # 3) 没有模型时只验证链路（数字不可用）
    python -m eval.run_eval --embedder hash --hyde none --configs A1_dense,A2_hybrid

    # 4) 只跑指定题型、只跑前 10 题
    python -m eval.run_eval --types fact,table --limit 10

输出：``eval/results/<run_id>/`` 下的 per_query.jsonl、summary.json、summary.md。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import eval as eval_pkg  # noqa: F401  触发 .env 加载
from eval import configs as ablations
from eval import corpus as corpus_mod
from eval import dataset as dataset_mod
from eval import metrics
from eval.embedder import build as build_embedder
from eval.hyde import HydeGenerator
from eval.retrieval import (
    CrossEncoderReranker,
    Hit,
    HybridIndex,
    cliff_cutoff,
    rerank_hits,
    rrf_fuse,
)

logger = logging.getLogger("eval.run")

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 检索消融评测")
    parser.add_argument("--dataset", default=dataset_mod.dataset_path_default())
    parser.add_argument("--corpus", default=corpus_mod.DEFAULT_CORPUS)
    parser.add_argument("--corpus-source", choices=["md", "milvus"], default="md")
    parser.add_argument("--md", default=None, help="指定 Markdown；默认取最新的 *_new.md")
    parser.add_argument("--rebuild-corpus", action="store_true")
    parser.add_argument("--embedder", default="bge", help="bge | hash")
    parser.add_argument("--reranker", choices=["bge", "none"], default="bge")
    parser.add_argument("--hyde", choices=["llm", "none"], default="llm")
    parser.add_argument("--configs", default=None, help="逗号分隔，如 A1_dense,A4_rerank")
    parser.add_argument("--ks", default="5,10")
    parser.add_argument("--types", default=None, help="逗号分隔的题型过滤")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--write-dataset", action="store_true", help="把解析出的 gold 写回数据集")
    return parser.parse_args(argv)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def select_items(items, types: str | None, limit: int):
    selected = list(items)
    if types:
        wanted = {t.strip() for t in types.split(",") if t.strip()}
        unknown = wanted - set(dataset_mod.TYPES)
        if unknown:
            raise SystemExit(f"未知题型：{unknown}，可选 {dataset_mod.TYPES}")
        selected = [item for item in selected if item.type in wanted]
    if limit and limit > 0:
        selected = selected[:limit]
    if not selected:
        raise SystemExit("过滤后没有任何题目，检查 --types / --limit")
    return selected


def build_index(chunks, embedder) -> HybridIndex:
    started = time.perf_counter()
    vectors = embedder.encode_documents([chunk.content for chunk in chunks])
    index = HybridIndex(
        chunks,
        corpus_mod.dense_matrix(vectors),
        corpus_mod.sparse_list(vectors),
    )
    logger.info("索引就绪：%d 切片，用时 %.1fs", len(index), time.perf_counter() - started)
    return index


def warm_up(index: HybridIndex, embedder, reranker) -> None:
    """预热：首次调用会触发 CUDA 上下文/内核编译，别把这几百毫秒算进 P95。

    不做这一步，A1 的 P95 会是 112ms、A2 变成 0.3ms，看起来像"配置 A1 更慢"，
    实际只是第一组配置替所有配置背了初始化成本。
    """
    probe = index.chunks[0].content[:64]
    vector = embedder.encode_queries([probe])[0]
    index.search(vector.dense, vector.sparse, mode="hybrid", limit=5)
    if reranker is not None:
        reranker.score(probe, [index.chunks[0].content[:64]])
    logger.info("预热完成")


def run_one(item, cfg, index, embedder, reranker, hyde) -> dict:
    """跑一条配置 × 一道题，返回指标行。

    计时口径：``latency_ms`` 只统计「嵌入 + 检索 + 融合 + 精排」，
    HyDE 的 LLM 调用单独记在 ``hyde_ms``。
    混在一起会得出"命中缓存时 HyDE 免费"的错误结论。
    """
    query = item.query_for_retrieval()
    hyde_ms = 0.0
    hyde_cached = None
    hypothetical = ""
    if cfg.use_hyde:
        # HyDE 是一次 LLM 调用，单独计时：它不该混进"检索耗时"里，
        # 否则命中缓存时看起来"HyDE 免费"，冷启动时又看起来"检索很慢"。
        item_names = [item.expect_item] if item.expect_item else []
        hypothetical = hyde.generate(query, item_names)
        hyde_ms = round(hyde.last_ms, 2)
        hyde_cached = hyde.last_cached

    started = time.perf_counter()
    vector = embedder.encode_queries([query])[0]
    hyde_hits: list[Hit] = []
    if cfg.use_hyde:
        hyde_vector = embedder.encode_queries([f"{query}\n{hypothetical}"])[0]
        hyde_hits = index.search(
            hyde_vector.dense,
            hyde_vector.sparse,
            mode=cfg.mode,
            limit=cfg.limit,
            dense_weight=cfg.dense_weight,
            sparse_weight=cfg.sparse_weight,
        )
    hits: list[Hit] = index.search(
        vector.dense,
        vector.sparse,
        mode=cfg.mode,
        limit=cfg.limit,
        dense_weight=cfg.dense_weight,
        sparse_weight=cfg.sparse_weight,
    )
    if cfg.use_hyde:
        hits = rrf_fuse([hits, hyde_hits], k=60, max_results=cfg.limit)
    if cfg.use_rerank:
        hits = rerank_hits(index, query, hits, reranker)
    if cfg.dynamic_cutoff:
        hits = cliff_cutoff(hits, cfg.cutoff_min_k, cfg.cutoff_max_k, cfg.cutoff_gap)
    latency_ms = (time.perf_counter() - started) * 1000

    ranked = [hit.chunk_id for hit in hits]
    context_chars = sum(len(index.by_id(cid).content) for cid in ranked if index.by_id(cid))
    top1 = index.by_id(ranked[0]) if ranked else None
    return {
        "config": cfg.name,
        "id": item.id,
        "type": item.type,
        "question": item.question,
        "query": query,
        "ranked": ranked,
        "gold": item.gold_chunk_ids,
        "n_docs": len(ranked),
        "context_chars": context_chars,
        "latency_ms": round(latency_ms, 2),
        "hyde_ms": hyde_ms,
        "hyde_cached": hyde_cached,
        "top1_title": top1.title if top1 else "",
        "top1_preview": (top1.content[:60] if top1 else ""),
        "expect_abstain": item.expect_abstain,
    }


def summarize_config(config_name: str, rows: list[dict], ks: Sequence[int]) -> dict:
    payload: dict = {"config": config_name, "n": len(rows)}
    for k in ks:
        payload[f"k={k}"] = metrics.summarize(rows, k=k)
    by_type: dict[str, dict] = {}
    for item_type in dataset_mod.TYPES:
        subset = [row for row in rows if row["type"] == item_type]
        if subset:
            by_type[item_type] = metrics.summarize(subset, k=ks[0])
    payload["by_type"] = by_type

    hyde_rows = [row for row in rows if row.get("hyde_cached") is not None]
    if hyde_rows:
        cold = [row["hyde_ms"] for row in hyde_rows if not row["hyde_cached"]]
        payload["hyde"] = {
            "n": len(hyde_rows),
            "cache_hit_rate": metrics.mean(1.0 if row["hyde_cached"] else 0.0 for row in hyde_rows),
            "avg_ms_cold": metrics.mean(cold),
            "cold_calls": len(cold),
        }
    return payload


def render_markdown(summaries: dict[str, dict], ks: Sequence[int], meta: dict) -> str:
    main_k = ks[0]
    lines = [
        f"# 检索消融结果（{meta['run_id']}）",
        "",
        f"- 题目数：{meta['n_items']}（{meta['type_counts']}）",
        f"- 语料：{meta['n_chunks']} 切片，fingerprint `{meta['corpus_fp']}`",
        f"- 编码器：`{meta['embedder']}`；精排：`{meta['reranker']}`；HyDE：`{meta['hyde']}`；LLM：`{meta.get('llm_model') or 'n/a'}`",
        f"- commit：`{meta['git_sha']}`；时间：{meta['timestamp']}",
        "",
        f"## 主表（Hit/Recall/MRR/nDCG @{main_k}）",
        "",
        f"| 配置 | Hit@{main_k} | Recall@{main_k} | Precision@{main_k} | MRR@{main_k} | nDCG@{main_k} | 平均文档数 | 平均上下文字数 | P95 检索耗时 | HyDE 额外 LLM 调用 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in summaries.items():
        block = payload[f"k={main_k}"]
        hyde_block = payload.get("hyde")
        if not hyde_block:
            hyde_text = "—"
        elif hyde_block["cold_calls"]:
            hyde_text = f"冷启动 {hyde_block['avg_ms_cold']:.0f} ms/题"
        else:
            hyde_text = f"全部命中缓存（{hyde_block['n']} 次）"
        lines.append(
            "| {name} | {hit:.3f} | {recall:.3f} | {precision:.3f} | {mrr:.3f} | "
            "{ndcg:.3f} | {docs:.1f} | {chars:.0f} | {p95:.1f} ms | {hyde} |".format(
                name=name,
                hit=block[f"hit@{main_k}"],
                recall=block[f"recall@{main_k}"],
                precision=block[f"precision@{main_k}"],
                mrr=block[f"mrr@{main_k}"],
                ndcg=block[f"ndcg@{main_k}"],
                docs=block["avg_docs"],
                chars=block["avg_context_chars"],
                p95=block["p95_latency_ms"],
                hyde=hyde_text,
            )
        )
    best_name = max(summaries, key=lambda n: summaries[n][f"k={main_k}"][f"hit@{main_k}"])
    lines += ["", f"## 分题型（最佳配置：{best_name}）", "",
              f"| 题型 | 题数 | Hit@{main_k} | MRR@{main_k} | nDCG@{main_k} |",
              "| --- | --- | --- | --- | --- |"]
    for item_type, block in summaries[best_name]["by_type"].items():
        if block["n_scored"] == 0:
            # 弃答题没有 gold，检索指标无定义；写 0 会被误读成"完全答错"
            lines.append(f"| {item_type} | {block['n']} | — | — | — |")
            continue
        lines.append(
            "| {t} | {n} | {hit:.3f} | {mrr:.3f} | {ndcg:.3f} |".format(
                t=item_type,
                n=block["n_scored"] or block["n"],
                hit=block[f"hit@{main_k}"],
                mrr=block[f"mrr@{main_k}"],
                ndcg=block[f"ndcg@{main_k}"],
            )
        )
    lines += [
        "",
        "> `out_of_scope` / `negative` 没有 gold 切片，检索指标无定义（表中为 —）；"
        "它们的正确性只能通过答案层的**弃答率**衡量，见 `eval/service_eval.py`。",
        "",
    ]
    lines += [
        "",
        "## 复现命令",
        "",
        "```bash",
        f"python -m eval.run_eval --configs {','.join(summaries)}",
        "```",
        "",
        "> 说明：本地为精确检索，线上 Milvus 为 HNSW 近似检索，绝对数值会有小幅差异；"
        "WeightedRanker 用 min-max 归一化近似复刻。指标用于**相对比较**，不要直接对外宣称绝对值。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    dfg = dataset_mod.load(args.dataset)
    items_all = dfg
    chunks = corpus_mod.ensure_corpus(
        args.corpus, source=args.corpus_source, md_path=args.md, rebuild=args.rebuild_corpus
    )
    report = dataset_mod.resolve_gold(items_all, chunks)
    if report["problems"]:
        print("\n[gold 解析问题]")
        for problem in report["problems"]:
            print("  -", problem)
        print("提示：先修数据集标注，或用 --write-dataset 落盘后再人工复核\n")
        if args.write_dataset:
            dataset_mod.save(items_all, args.dataset)
        return 2
    if report["ambiguous"]:
        print("\n[gold 解析告警]")
        for warning in report["ambiguous"]:
            print("  -", warning)
    if args.write_dataset:
        dataset_mod.save(items_all, args.dataset)
        print(f"已把 gold_chunk_ids 写回 {args.dataset}")

    items = select_items(items_all, args.types, args.limit)
    selected_configs = ablations.resolve(args.configs)
    if args.hyde == "none":
        dropped = [cfg.name for cfg in selected_configs if cfg.use_hyde]
        selected_configs = [cfg for cfg in selected_configs if not cfg.use_hyde]
        if dropped:
            print(f"[跳过] --hyde none，以下配置被移除：{', '.join(dropped)}")
    needs_rerank = any(cfg.use_rerank for cfg in selected_configs)
    reranker = None
    if needs_rerank:
        if args.reranker == "none":
            raise SystemExit("配置里有精排但 --reranker none，请去掉 A4/A5 或开启精排")
        reranker = CrossEncoderReranker()
    hyde = HydeGenerator() if any(cfg.use_hyde for cfg in selected_configs) else None

    embedder = build_embedder(args.embedder)
    index = build_index(chunks, embedder)
    warm_up(index, embedder, reranker)

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out, run_id)
    os.makedirs(run_dir, exist_ok=True)

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    per_query_path = os.path.join(run_dir, "per_query.jsonl")
    with open(per_query_path, "w", encoding="utf-8") as handle:
        for cfg in selected_configs:
            rows: list[dict] = []
            for item in items:
                row = run_one(item, cfg, index, embedder, reranker, hyde)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            summaries[cfg.name] = summarize_config(cfg.name, rows, ks)
            all_rows.extend(rows)
            block = summaries[cfg.name][f"k={ks[0]}"]
            print(
                f"{cfg.name:<14} Hit@{ks[0]}={block[f'hit@{ks[0]}']:.3f} "
                f"MRR={block[f'mrr@{ks[0]}']:.3f} nDCG={block[f'ndcg@{ks[0]}']:.3f} "
                f"docs={block['avg_docs']:.1f} chars={block['avg_context_chars']:.0f}"
            )

    meta = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha(),
        "llm_model": os.getenv("LLM_DEFAULT_MODEL", ""),
        "llm_thinking": os.getenv("LLM_ENABLE_THINKING", "").strip().lower() or "default",
        "n_items": len(items),
        "type_counts": dataset_mod.type_breakdown(items),
        "n_chunks": len(chunks),
        "corpus_fp": corpus_mod.corpus_fingerprint(chunks),
        "dataset_fp": dataset_mod.dataset_fingerprint(items),
        "embedder": embedder.name,
        "reranker": args.reranker if needs_rerank else "none",
        "hyde": args.hyde if any(c.use_hyde for c in selected_configs) else "none",
        "ks": ks,
        "ablation_params": {cfg.name: cfg.to_dict() for cfg in selected_configs},
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "configs": summaries}, handle, ensure_ascii=False, indent=2)
    markdown = render_markdown(summaries, ks, meta)
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown)

    print(f"\n结果目录：{run_dir}")
    print(f"可直接粘贴的表格：{os.path.join(run_dir, 'summary.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())