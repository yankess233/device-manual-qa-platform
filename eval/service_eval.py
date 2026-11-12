"""端到端（答案层）评测：直接打真实服务，测"答得对不对、快不快、会不会胡说"。

它和 ``run_eval.py`` 的分工：
- ``run_eval.py``：离线检索指标（Hit/MRR/nDCG）+ 消融，证明"检索组件各自贡献多少"；
- 本脚本：线上服务指标（答案正确率、弃答率、首包延迟、成本），证明"用户最终感受到什么"。

答案正确性**不交给 LLM 自动判分**（自评偏差无法复核），
而是导出 CSV 人工打 0/1，再用 ``--score`` 汇总。人工抽检比例写进 README。

用法::

    # 1) 服务先起起来：python -m knowledge.api.query_router
    python -m eval.service_eval --endpoint http://127.0.0.1:8001
    # 2) 人工把 results/answers_xxx.csv 的 human_score 列填 0/1
    python -m eval.service_eval --score results/answers_xxx.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Sequence

import eval as eval_pkg  # noqa: F401
from eval import dataset as dataset_mod

ABSTAIN_MARKERS = ("未提供", "没有提供", "未提及", "未说明", "手册中未", "不在手册",
                   "无法确定", "不包含", "未涉及", "建议咨询")
DEFAULT_OUT = "eval/results"


def post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(endpoint: str, query: str, session_id: str) -> tuple[str, float]:
    started = time.perf_counter()
    payload = post_json(
        endpoint.rstrip("/") + "/query",
        {"query": query, "session_id": session_id, "is_stream": False},
    )
    elapsed = (time.perf_counter() - started) * 1000
    return payload.get("answer", "") or "", elapsed


def keyword_hit(answer: str, keywords: Sequence[str]) -> int | None:
    if not keywords:
        return None
    return 1 if any(keyword in answer for keyword in keywords) else 0


def abstain_ok(answer: str) -> int:
    return 1 if any(marker in answer for marker in ABSTAIN_MARKERS) else 0


def run(args: argparse.Namespace) -> int:
    items = dataset_mod.load(args.dataset)
    if args.types:
        wanted = {t.strip() for t in args.types.split(",") if t.strip()}
        items = [item for item in items if item.type in wanted]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("没有可跑的题目")

    rows: list[dict] = []
    for index, item in enumerate(items, start=1):
        session_id = f"eval-{item.id}-{uuid.uuid4().hex[:6]}"
        primed = 0
        for turn in item.history:
            question = turn.get("content", "")
            if turn.get("role") == "user" and question:
                ask(args.endpoint, question, session_id)  # 只为了让服务写入历史
                primed += 1
        try:
            answer, latency = ask(args.endpoint, item.question, session_id)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"[失败] {item.id}: {error}")
            rows.append({
                "id": item.id, "type": item.type, "question": item.question,
                "answer": f"__ERROR__ {error}", "latency_ms": 0,
                "keyword_hit": "", "abstain_ok": "", "human_score": "",
                "primed_turns": primed,
            })
            continue
        score = abstain_ok(answer) if item.expect_abstain else keyword_hit(answer, item.answer_keywords)
        rows.append({
            "id": item.id,
            "type": item.type,
            "question": item.question,
            "answer": answer.replace("\n", " ")[:800],
            "latency_ms": round(latency, 1),
            "keyword_hit": "" if score is None else score,
            "abstain_ok": abstain_ok(answer) if item.expect_abstain else "",
            "human_score": "",
            "primed_turns": primed,
        })
        print(f"[{index}/{len(items)}] {item.id} {latency:.0f}ms")

    out_path = args.out if args.out.endswith(".csv") else f"{args.out}/answers_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已导出：{out_path}")
    print("下一步：人工把 human_score 列填 0/1（0=错，1=对），再执行 --score 汇总。")
    return 0


def score(path: str) -> int:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = [row for row in rows if str(row.get("human_score", "")).strip() in ("0", "1")]
    if not scored:
        print("human_score 还没填，先人工打分。")
        return 1
    total = len(scored)
    correct = sum(1 for row in scored if str(row["human_score"]).strip() == "1")
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms")]
    print(f"人工评分样本：{total}/{len(rows)}（覆盖率 {total / len(rows):.0%}）")
    print(f"答案正确率：{correct / total:.1%}（{correct}/{total}）")
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
        print(f"端到端延迟：P50 {statistics.median(ordered):.0f} ms，P95 {p95:.0f} ms")
    by_type: dict[str, list[int]] = {}
    for row in scored:
        by_type.setdefault(row["type"], []).append(int(row["human_score"]))
    for item_type, scores in by_type.items():
        print(f"  {item_type:<12} {sum(scores) / len(scores):.1%}（{sum(scores)}/{len(scores)}）")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端答案评测")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001")
    parser.add_argument("--dataset", default=dataset_mod.dataset_path_default())
    parser.add_argument("--types", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--score", default=None, help="对已人工打分的 CSV 汇总")
    args = parser.parse_args(argv)
    if args.score:
        return score(args.score)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())