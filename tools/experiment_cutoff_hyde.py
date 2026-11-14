"""实验：断崖截断的 max_k 与 HyDE 融合权重，对 Hit@5 的影响。

只读实验，不改流水线代码。复用 eval 层的 embedding / HyDE / rerank 缓存，重复跑很快。

用法：
    python tools/experiment_cutoff_hyde.py
    python tools/experiment_cutoff_hyde.py --dataset eval/dataset-home.jsonl --corpus eval/corpus/home.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import eval as eval_pkg  # noqa: E402,F401
from eval import corpus as corpus_mod  # noqa: E402
from eval import dataset as dataset_mod  # noqa: E402
from eval import metrics  # noqa: E402
from eval.embedder import build  # noqa: E402
from eval.hyde import HydeGenerator  # noqa: E402
from eval.retrieval import (  # noqa: E402
    CrossEncoderReranker,
    Hit,
    HybridIndex,
    cliff_cutoff,
    rrf_fuse,
)

K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/dataset-home.jsonl")
    parser.add_argument("--corpus", default="eval/corpus/home.jsonl")
    parser.add_argument("--dense-limit", type=int, default=10)
    parser.add_argument("--hyde-limit", type=int, default=5)
    parser.add_argument("--fuse-depth", type=int, default=20)
    parser.add_argument("--gap", type=float, default=0.15)
    parser.add_argument("--min-k", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = corpus_mod.load_corpus(args.corpus)
    items = dataset_mod.load(args.dataset)
    dataset_mod.resolve_gold(items, chunks)
    items = [item for item in items if item.gold_chunk_ids]

    embedder = build("bge")
    vectors = embedder.encode_documents([chunk.content for chunk in chunks])
    index = HybridIndex(chunks, corpus_mod.dense_matrix(vectors), corpus_mod.sparse_list(vectors))
    reranker = CrossEncoderReranker()
    hyde = HydeGenerator()

    def build_records(dense_limit: int, hyde_limit: int) -> list[dict]:
        built = []
        for n, item in enumerate(items, 1):
            query = item.query_for_retrieval()
            vector = embedder.encode_queries([query])[0]
            hybrid_hits = index.search(
                vector.dense, vector.sparse, mode="hybrid", limit=dense_limit,
                dense_weight=0.5, sparse_weight=0.5,
            )
            hypothetical = hyde.generate(query, [item.expect_item] if item.expect_item else [])
            hyde_vector = embedder.encode_queries([f"{query}\n{hypothetical}"])[0]
            hyde_hits = index.search(
                hyde_vector.dense, hyde_vector.sparse, mode="hybrid", limit=hyde_limit,
                dense_weight=0.5, sparse_weight=0.5,
            )
            fused = {}
            for weight in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
                lists = [hybrid_hits] if weight == 0.0 else [hybrid_hits, hyde_hits]
                weights = [1.0] if weight == 0.0 else [1.0, weight]
                fused[weight] = rrf_fuse(lists, k=60, max_results=args.fuse_depth, weights=weights)

            pool = {}
            for hits in fused.values():
                for hit in hits:
                    pool.setdefault(hit.chunk_id, hit)
            contents = [index.by_id(cid).content for cid in pool]
            scores = dict(zip(pool, reranker.score(query, contents), strict=False))
            built.append({"item": item, "fused": fused, "scores": scores})
            print(f"\r[{n}/{len(items)}] {item.id}", end="", flush=True)
        print()
        return built

    state = {'records': build_records(args.dense_limit, args.hyde_limit)}

    def evaluate(weight: float, max_k: int, k: int = K) -> dict:
        rows = []
        for record in state['records']:
            ids = [hit.chunk_id for hit in record["fused"][weight]]
            ordered = sorted(ids, key=lambda cid: record["scores"].get(cid, -1e9), reverse=True)
            hits = [Hit(chunk_id=cid, score=record["scores"].get(cid, -1e9)) for cid in ordered]
            kept = cliff_cutoff(hits, min_k=args.min_k, max_k=max_k, gap=args.gap)
            kept_ids = [hit.chunk_id for hit in kept]
            gold = set(record["item"].gold_chunk_ids)
            rows.append(
                {
                    "ranked": kept_ids,
                    "gold": gold,
                    "n_docs": len(kept_ids),
                    "context_chars": sum(
                        len(index.by_id(cid).content) for cid in kept_ids if index.by_id(cid)
                    ),
                }
            )
        return metrics.summarize(rows, k=k)

    def show(title: str, header: str, rows) -> None:
        print(f"\n{title}")
        print(f"{header:<16}{'Hit@5':>8}{'Recall@5':>10}{'MRR@5':>9}{'nDCG@5':>9}{'文档数':>8}{'上下文字数':>10}")
        for label, summary in rows:
            print(
                f"{label:<16}{summary[f'hit@{K}']:>8.3f}{summary[f'recall@{K}']:>10.3f}"
                f"{summary[f'mrr@{K}']:>9.3f}{summary[f'ndcg@{K}']:>9.3f}"
                f"{summary['avg_docs']:>8.1f}{summary['avg_context_chars']:>10.0f}"
            )

    show("表 1：固定 HyDE 权重=1.0，扫描断崖截断 max_k",
         "max_k", [(f"max_k={k}", evaluate(1.0, k)) for k in (5, 10, 15, 20, 30)])
    show("表 2：固定 max_k=10，扫描 HyDE 融合权重",
         "hyde_weight", [(f"w={w}", evaluate(w, 10)) for w in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)])

    print("\n表 3：加深召回（dense/hyde 检索深度），固定 hyde=1.0、max_k=10")
    print(f"{'depth':<16}{'Hit@5':>8}{'Hit@10':>8}{'Recall@5':>10}{'MRR@5':>9}{'文档数':>8}{'上下文字数':>10}")
    for depth in (10, 20, 30):
        saved = state['records']
        state['records'] = build_records(depth, max(5, depth // 2))
        s5 = evaluate(1.0, 10, 5)
        s10 = evaluate(1.0, 10, 10)
        state['records'] = saved
        print(
            f"{depth:<16}{s5[f'hit@5']:>8.3f}{s10[f'hit@10']:>8.3f}"
            f"{s5[f'recall@5']:>10.3f}{s5[f'mrr@5']:>9.3f}"
            f"{s5['avg_docs']:>8.1f}{s5['avg_context_chars']:>10.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())