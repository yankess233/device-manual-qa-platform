"""评测集加载与 gold 解析。

标注策略（重要，决定评测是否可信）：
1. 题目按**章节 + 原文证据**双锚点标注，而不是手抄 chunk_id。
   chunk_id 会随切分参数变化而漂移，章节名和原文句子不会；
2. 解析时用证据句在切片正文里做子串匹配，**匹配不到就报错**，
   等于用机器反向校验"这道题是否真的能从语料里找到答案"；
3. 解析结果写回 ``gold_chunk_ids``，人工复核后才算定稿。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .retrieval import Chunk

logger = logging.getLogger(__name__)

TYPES = ("fact", "table", "procedure", "multi_turn", "out_of_scope", "negative")
_TYPE_ALIASES = {"事实": "fact", "表格": "table", "步骤": "procedure",
                 "多轮": "multi_turn", "手册外": "out_of_scope", "负样本": "negative"}

_WS = re.compile(r"\s+")
_HEADING_PREFIX = re.compile(r"^#{1,6}\s*")
_PART_SUFFIX = re.compile(r"[（(]\s*\d+\s*/\s*\d+\s*[）)]|\s*-\s*\d+$")


def normalize(text: str) -> str:
    """去掉空白与大小写差异，便于中文子串匹配。"""
    return _WS.sub("", text or "").lower()


def heading_key(title: str) -> str:
    """标题归一化：去 #、去 ``（1/3）``、去 ``-3`` 这类二次切分后缀。"""
    text = _PART_SUFFIX.sub("", _HEADING_PREFIX.sub("", (title or "").strip()))
    return normalize(text)


@dataclass
class EvalItem:
    id: str
    type: str
    question: str
    reference: str = ""
    gold_sections: list[str] = field(default_factory=list)
    evidence: str = ""
    answer_keywords: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    rewrite: str = ""  # 多轮题的"理想改写"，本地检索用；改写质量另在服务模式测
    expect_item: str = ""
    needs_image: bool = False
    difficulty: str = "normal"
    expect_abstain: bool = False
    gold_chunk_ids: list[str] = field(default_factory=list)
    label_status: str = "manual"  # manual | suggested | verified

    def query_for_retrieval(self) -> str:
        """本地检索实际使用的查询串。

        多轮题用 ``rewrite``（理想改写），等价于"检索上限"；
        没有 rewrite 时退化为"历史末轮 + 当前问题"的朴素拼接。
        """
        if self.rewrite:
            return self.rewrite
        if self.history:
            last_user = [turn["content"] for turn in self.history if turn.get("role") == "user"]
            if last_user:
                return f"{last_user[-1]}\n{self.question}"
        return self.question


def load(path: str) -> list[EvalItem]:
    items: list[EvalItem] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON：{error}") from error
            item_type = _TYPE_ALIASES.get(payload.get("type", ""), payload.get("type", ""))
            if item_type not in TYPES:
                raise ValueError(f"{path}:{line_no} 未知题型 {item_type!r}，可选：{TYPES}")
            items.append(
                EvalItem(
                    id=payload["id"],
                    type=item_type,
                    question=payload["question"],
                    reference=payload.get("reference", ""),
                    gold_sections=payload.get("gold_sections", []),
                    evidence=payload.get("evidence", ""),
                    answer_keywords=payload.get("answer_keywords", []),
                    history=payload.get("history", []),
                    rewrite=payload.get("rewrite", ""),
                    expect_item=payload.get("expect_item", ""),
                    needs_image=payload.get("needs_image", False),
                    difficulty=payload.get("difficulty", "normal"),
                    expect_abstain=payload.get("expect_abstain", False),
                    gold_chunk_ids=[str(x) for x in payload.get("gold_chunk_ids", [])],
                    label_status=payload.get("label_status", "manual"),
                )
            )
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("存在重复的题目 id")
    return items


def save(items: Iterable[EvalItem], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            payload = {
                "id": item.id,
                "type": item.type,
                "question": item.question,
                "reference": item.reference,
                "gold_sections": item.gold_sections,
                "evidence": item.evidence,
                "answer_keywords": item.answer_keywords,
                "history": item.history,
                "rewrite": item.rewrite,
                "expect_item": item.expect_item,
                "needs_image": item.needs_image,
                "difficulty": item.difficulty,
                "expect_abstain": item.expect_abstain,
                "gold_chunk_ids": item.gold_chunk_ids,
                "label_status": item.label_status,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_gold(items: Sequence[EvalItem], chunks: Sequence[Chunk],
                 max_gold: int = 4) -> dict:
    """把 ``gold_sections`` + ``evidence`` 解析为 chunk_id，并返回诊断报告。"""
    by_section: dict[str, list[str]] = {}
    for chunk in chunks:
        by_section.setdefault(heading_key(chunk.title), []).append(chunk.chunk_id)
    normalized_content = {chunk.chunk_id: normalize(chunk.content) for chunk in chunks}

    problems: list[str] = []
    ambiguous: list[str] = []
    for item in items:
        if item.expect_abstain:
            item.gold_chunk_ids = []
            continue
        found: list[str] = []
        evidence = normalize(item.evidence)
        hits: list[str] = []
        if evidence:
            hits = [cid for cid, text in normalized_content.items() if evidence in text]
            if not hits:
                problems.append(f"{item.id}: 证据句在语料中找不到：{item.evidence[:40]!r}")

        if hits:
            # 证据优先：证据句能唯一定位答案所在切片时，不再把整个章节都算作 gold，
            # 否则「规格」这种被拆成 2 片的章节会让 Recall 永远上不去。
            found = list(hits)
        else:
            for section in item.gold_sections:
                key = heading_key(section)
                for cid in by_section.get(key, []):
                    if cid not in found:
                        found.append(cid)
                if key not in by_section:
                    # 退化为前缀匹配，兼容被二次切分出的 "规格-1/2"
                    for name, cids in by_section.items():
                        if name.startswith(key) or key.startswith(name):
                            for cid in cids:
                                if cid not in found:
                                    found.append(cid)
        if not item.evidence and not item.gold_sections:
            problems.append(f"{item.id}: 非弃答题必须至少有 gold_sections 或 evidence")
        if len(found) > max_gold:
            ambiguous.append(f"{item.id}: 命中 {len(found)} 个切片（{found}），建议收紧证据句")
        item.gold_chunk_ids = found
        item.label_status = "suggested" if found else item.label_status
        if not found:
            problems.append(f"{item.id}: 没有解析到任何 gold 切片")

    return {
        "n_items": len(items),
        "n_resolved": sum(1 for item in items if item.gold_chunk_ids or item.expect_abstain),
        "problems": problems,
        "ambiguous": ambiguous,
        "sections_seen": sorted(by_section),
    }


def type_breakdown(items: Sequence[EvalItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.type] = counts.get(item.type, 0) + 1
    return counts


def dataset_fingerprint(items: Sequence[EvalItem]) -> str:
    import hashlib

    digest = hashlib.sha1()
    for item in items:
        digest.update(item.id.encode())
        digest.update(item.question.encode("utf-8"))
        digest.update(",".join(item.gold_chunk_ids).encode())
    return digest.hexdigest()[:12]


def dataset_path_default() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.jsonl")