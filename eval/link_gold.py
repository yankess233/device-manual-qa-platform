"""把 ``gold_sections`` / ``evidence`` 解析成 chunk_id，供人工复核。

用法::

    python -m eval.link_gold            # 只报告，不改文件
    python -m eval.link_gold --write    # 把解析结果写回 dataset.jsonl

设计意图：标注不该手抄 chunk_id（切分参数一变就全废），而应该锚定
"章节名 + 原文证据句"。本脚本负责把锚点翻译成当前语料下的 chunk_id，
并且**解析失败就退出码 1**，防止带着坏标注去跑评测。
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import eval as eval_pkg  # noqa: F401
from eval import corpus as corpus_mod
from eval import dataset as dataset_mod


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测集 gold 解析")
    parser.add_argument("--dataset", default=dataset_mod.dataset_path_default())
    parser.add_argument("--corpus", default=corpus_mod.DEFAULT_CORPUS)
    parser.add_argument("--corpus-source", choices=["md", "milvus"], default="md")
    parser.add_argument("--md", default=None)
    parser.add_argument("--rebuild-corpus", action="store_true")
    parser.add_argument("--write", action="store_true", help="写回 dataset.jsonl")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    items = dataset_mod.load(args.dataset)
    chunks = corpus_mod.ensure_corpus(
        args.corpus, source=args.corpus_source, md_path=args.md, rebuild=args.rebuild_corpus
    )
    report = dataset_mod.resolve_gold(items, chunks)

    print(f"题目 {report['n_items']} 道，语料 {len(chunks)} 切片")
    print(f"题型分布：{dataset_mod.type_breakdown(items)}")
    print(f"解析成功：{report['n_resolved']}/{report['n_items']}")
    print("\n可用章节：")
    for section in report["sections_seen"]:
        print("  -", section)

    if report["ambiguous"]:
        print("\n[告警] gold 过多，建议收紧证据句：")
        for warning in report["ambiguous"]:
            print("  -", warning)

    if report["problems"]:
        print("\n[错误] 以下题目无法解析：")
        for problem in report["problems"]:
            print("  -", problem)
        if args.write:
            dataset_mod.save(items, args.dataset)
        return 1

    print("\n全部题目解析成功。抽样：")
    for item in items[:5]:
        print(f"  {item.id} [{item.type}] gold={item.gold_chunk_ids} ← {item.question[:32]}")
    if args.write:
        dataset_mod.save(items, args.dataset)
        print(f"\n已写回 {args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())