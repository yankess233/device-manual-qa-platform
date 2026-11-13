"""答案层模型 A/B：同一份检索上下文，两个（或多个）模型各答一遍。

为什么单独做这一层：
- 换 LLM **不会**改变检索指标（检索用的是本地 BGE-M3），所以 `run_eval.py` 换模型前后
  A1/A2 必然一模一样——这只能证明"没跑错"，不能说明模型好坏；
- 真正受影响的是两部分：**HyDE 生成质量** 和 **最终答案质量**。前者已在 `run_eval.py` 的
  A3/A4/A5 里体现，后者只能靠这一层看。

做法：固定检索结果（hybrid + 精排，取 top-k），只把 LLM 换掉，用生产同一份 `ANSWER_PROMPT`
调用，输出并排对照 + 耗时/ token 统计，供人工判断。

用法::

    python -m eval.answer_ab --models qwen-flash,qwen3.8-flash --limit 8
    python -m eval.answer_ab --models qwen-flash,qwen3.8-flash --ids F07,T01,P01,O01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Sequence

import eval as eval_pkg  # noqa: F401
from eval import corpus as corpus_mod
from eval import dataset as dataset_mod
from eval.embedder import build as build_embedder
from eval.retrieval import CrossEncoderReranker, HybridIndex, rerank_hits

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_IDS = ["F07", "T01", "T03", "P01", "P02", "M01", "O01", "N02"]
TOP_K = 6
CONTEXT_BUDGET = 6000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="答案层模型 A/B")
    parser.add_argument("--models", default="qwen-flash,qwen3.8-flash")
    parser.add_argument("--dataset", default=dataset_mod.dataset_path_default())
    parser.add_argument("--corpus", default=corpus_mod.DEFAULT_CORPUS)
    parser.add_argument("--ids", default=",".join(DEFAULT_IDS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--rescore", default=None,
                        help="对已跑完的 answer_ab 目录重新判定（不重新调用模型）")
    return parser.parse_args(argv)


def build_context(index: HybridIndex, hits) -> str:
    """按字符预算拼装参考内容，与生产 answer_output 的口径类似（标题 + 正文）。"""
    blocks: list[str] = []
    used = 0
    for position, hit in enumerate(hits, start=1):
        chunk = index.by_id(hit.chunk_id)
        if chunk is None:
            continue
        block = f"【文档{position}】{chunk.title.strip()}\n{chunk.content}"
        if used + len(block) > CONTEXT_BUDGET:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def retrieve(index: HybridIndex, embedder, reranker, item) -> list[str]:
    query = item.query_for_retrieval()
    vector = embedder.encode_queries([query])[0]
    hits = index.search(vector.dense, vector.sparse, mode="hybrid", limit=10)
    hits = rerank_hits(index, query, hits, reranker)
    return [hit.chunk_id for hit in hits]


def parse_model_spec(spec: str) -> tuple[str, str, dict]:
    """解析 ``模型名[:nothink]`` → (展示名, 真实模型名, extra_body)。"""
    raw = spec.strip()
    label, _, suffix = raw.partition(":")
    extra: dict = {}
    if suffix.strip().lower() in ("nothink", "no_think", "off"):
        extra["enable_thinking"] = False
    return (raw if extra else label), label, extra


def ask(client, model: str, prompt: str, extra_body: dict | None = None) -> tuple[str, float, int]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
        extra_body=extra_body or None,
    )
    elapsed = (time.perf_counter() - started) * 1000
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return (response.choices[0].message.content or "").strip(), elapsed, tokens


# 弃答判定词表：宁可宽一点。
# 第一版只写了 6 个词，结果把"未明确说明""没有提及""无法确认"这类**正确弃答**判成了硬答，
# 属于评测脚本自己的假阴性——这也是"答案层必须人工复核"的直接证据。
ABSTAIN_MARKERS = (
    "未提供", "没有提供", "未提及", "没有提及", "未明确", "没有明确", "未说明", "没有说明",
    "未涉及", "没有涉及", "未包含", "没有包含", "未给出", "没有给出", "手册中未", "不在手册",
    "无法确定", "无法确认", "不能确定", "不包含", "没有相关", "无相关", "建议咨询",
)


def verdict(item, answer: str) -> str:
    if item.expect_abstain:
        return "弃答" if any(marker in answer for marker in ABSTAIN_MARKERS) else "硬答*"
    hit = [kw for kw in item.answer_keywords if kw in answer]
    if not item.answer_keywords:
        return "—"
    # 关键词只是辅助信号：同义表述（"断开被测电路电源" vs "断开电源"）会漏判，
    # 所以这里不给 ✅/❌，只给覆盖数，最终以人工 0/1 为准。
    return f"关键词{len(hit)}/{len(item.answer_keywords)}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rescore:
        return rescore(args.rescore, args.dataset)
    from openai import OpenAI

    from knowledge.prompt.query_prompt import ANSWER_PROMPT

    models = [parse_model_spec(m) for m in args.models.split(",") if m.strip()]
    specs = {label: (model, extra) for label, model, extra in models}
    labels = list(specs)
    items = dataset_mod.load(args.dataset)
    chunks = corpus_mod.load_corpus(args.corpus)
    dataset_mod.resolve_gold(items, chunks)
    wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
    items = [item for item in items if item.id in wanted]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("没有匹配到的题目")

    embedder = build_embedder("bge")
    vectors = embedder.encode_documents([chunk.content for chunk in chunks])
    index = HybridIndex(chunks, corpus_mod.dense_matrix(vectors), corpus_mod.sparse_list(vectors))
    reranker = CrossEncoderReranker()

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
    )

    rows: list[dict] = []
    for item in items:
        ranked = retrieve(index, embedder, reranker, item)
        context = build_context(index, [type("H", (), {"chunk_id": cid})() for cid in ranked])
        history_text = "\n".join(
            f"{turn.get('role')}: {turn.get('content')}" for turn in item.history
        ) or "无"
        prompt = ANSWER_PROMPT.format(
            context=context,
            history=history_text,
            item_names=item.expect_item or "无",
            question=item.question,
        )
        entry = {"id": item.id, "type": item.type, "question": item.question, "context_chars": len(context), "answers": {}}
        for label in labels:
            model_name, extra = specs[label]
            try:
                answer, latency, tokens = ask(client, model_name, prompt, extra)
            except Exception as error:  # 单个模型失败不影响其他模型
                answer, latency, tokens = f"__ERROR__ {error}", 0.0, 0
            entry["answers"][label] = {
                "answer": answer,
                "latency_ms": round(latency, 1),
                "tokens": tokens,
                "verdict": verdict(item, answer),
            }
            print(f"{item.id} {label} {latency:.0f}ms tokens={tokens} -> {entry['answers'][label]['verdict']}")
        rows.append(entry)

    run_dir = os.path.join(args.out, f"answer_ab_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "answers.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    write_outputs(rows, labels, run_dir)
    return 0


def build_stats(rows: list[dict], labels: Sequence[str]) -> dict:
    stats: dict[str, dict] = {}
    for model in labels:
        entries = [row["answers"][model] for row in rows if row["answers"][model]["tokens"]]
        abstain = [
            row for row in rows
            if row["answers"][model]["tokens"] and row["answers"][model]["verdict"] in ("弃答", "硬答*")
        ]
        stats[model] = {
            "n": len(entries),
            "avg_latency_ms": round(sum(e["latency_ms"] for e in entries) / len(entries), 1) if entries else 0,
            "avg_tokens": round(sum(e["tokens"] for e in entries) / len(entries), 1) if entries else 0,
            "abstain_ok": sum(1 for row in abstain if row["answers"][model]["verdict"] == "弃答"),
            "abstain_total": len(abstain),
        }
    return stats


def write_outputs(rows: list[dict], labels: Sequence[str], run_dir: str) -> None:
    stats = build_stats(rows, labels)
    with open(os.path.join(run_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    lines = ["# 答案层模型 A/B", "",
             f"- 题目：{len(rows)} 道；模型：{', '.join(labels)}", "",
             "> 判定口径：`弃答/硬答*` 是规则判定（词表命中），`关键词 x/y` 只是辅助信号。",
             "> 同义表述会被漏判（例如\"断开被测电路电源\"不含\"断开电源\"），**最终以人工 0/1 为准**。", "",
             "| 模型 | 平均延迟 | 平均 tokens | 弃答正确 |", "| --- | --- | --- | --- |"]
    for model, block in stats.items():
        lines.append(
            f"| `{model}` | {block['avg_latency_ms']:.0f} ms | {block['avg_tokens']:.0f} "
            f"| {block['abstain_ok']}/{block['abstain_total']} |"
        )
    lines.append("")
    for row in rows:
        lines += [f"## {row['id']} · {row['question']}", "",
                  f"> 上下文 {row['context_chars']} 字 · 题型 `{row['type']}`", ""]
        for model in labels:
            block = row["answers"][model]
            lines += [f"**{model}**（{block['latency_ms']:.0f} ms，{block['tokens']} tokens，{block['verdict']}）", "",
                      "```text", block["answer"].replace("\n", " ")[:900], "```", ""]
    with open(os.path.join(run_dir, "answers.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\n结果：{run_dir}/answers.md")


def rescore(run_dir: str, dataset_path: str) -> int:
    """对已保存的答案重新判定，不重新调用模型（词表/口径改了以后用）。"""
    with open(os.path.join(run_dir, "answers.json"), "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    items = {item.id: item for item in dataset_mod.load(dataset_path)}
    labels: list[str] = []
    for row in rows:
        for label in row["answers"]:
            if label not in labels:
                labels.append(label)
        for block in row["answers"].values():
            block["verdict"] = verdict(items[row["id"]], block["answer"])
    write_outputs(rows, labels, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())