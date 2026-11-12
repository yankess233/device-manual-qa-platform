"""评测语料构建：从干净的 Markdown 重新切分，或从 Milvus 导出。

为什么默认用 Markdown 而不是 Milvus：
实测当前 Milvus 集合 ``kb_chunks_v1`` 里的 ``content`` 是乱码
（UTF-8 被按 GBK 解读后的替换字符），``item_name`` 还存着整段 JSON 字符串。
直接拿它做评测，会得到"看起来很差、但其实是数据坏了"的结论，
既无法归因，也不能写进简历。所以评测语料从 ``*_new.md`` 重新切，
顺带验证当前切分代码是否真的修好了。
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Sequence

import numpy as np

from .retrieval import Chunk

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus", "rs12.jsonl")
_HEADING = re.compile(r"^#{1,6}\s*")


def find_latest_markdown(root: str | None = None) -> str:
    """在知识库临时目录里找最新的 ``*_new.md``。"""
    base = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "knowledge", "temp_data")
    candidates = glob.glob(os.path.join(base, "**", "*_new.md"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(base, "**", "*.md"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"在 {base} 下没找到任何 Markdown，请用 --md 指定")
    return max(candidates, key=os.path.getmtime)


def read_text(path: str) -> str:
    """读文本，UTF-8 失败时回退 GB18030（MinerU 产物偶尔是 GBK）。"""
    raw = open(path, "rb").read()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def mojibake_ratio(text: str) -> float:
    """替换字符（U+FFFD）与拉丁补充区字符的占比，用于识别编码损坏。"""
    if not text:
        return 0.0
    bad = sum(1 for ch in text if ch == "\ufffd" or "\u00a0" <= ch <= "\u00ff")
    return bad / len(text)


def split_with_repo(md_text: str, file_title: str) -> list[Chunk]:
    """调用生产切分节点，保证评测语料与线上同源。"""
    from knowledge.processor.import_process.nodes.ducment_split import DocumentSplitNode

    node = DocumentSplitNode()
    # 不传 file_dir，避免触发 chunks.json 备份的副作用
    state = node.process({"file_title": file_title, "md_content": md_text})
    raw_chunks = state.get("chunks") or []
    if not raw_chunks:
        raise RuntimeError("切分结果为空，检查 Markdown 内容与切分配置")
    chunks: list[Chunk] = []
    for index, item in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                chunk_id=f"c{index:03d}",
                title=item.get("title", ""),
                content=item.get("content", ""),
                item_name=item.get("item_name", ""),
                file_title=item.get("file_title", file_title),
            )
        )
    return chunks


def build_from_markdown(md_path: str | None = None) -> list[Chunk]:
    md_path = md_path or find_latest_markdown()
    text = read_text(md_path)
    ratio = mojibake_ratio(text)
    if ratio > 0.02:
        raise ValueError(
            f"{md_path} 疑似编码损坏（异常字符占比 {ratio:.1%}），请先修编码再评测"
        )
    file_title = os.path.splitext(os.path.basename(md_path))[0]
    chunks = split_with_repo(text, file_title)
    logger.info("从 %s 切出 %d 个切片", os.path.basename(md_path), len(chunks))
    return chunks


def build_from_milvus(uri: str | None = None, collection: str | None = None) -> list[Chunk]:
    """从 Milvus 导出（保真模式，但会检查编码是否损坏）。"""
    from pymilvus import MilvusClient

    uri = uri or os.getenv("MILVUS_URL", "")
    collection = collection or os.getenv("CHUNKS_COLLECTION", "")
    if not uri or not collection:
        raise ValueError("需要 MILVUS_URL 与 CHUNKS_COLLECTION")
    client = MilvusClient(uri=uri)
    rows = client.query(
        collection_name=collection,
        filter="chunk_id >= 0",
        output_fields=["chunk_id", "title", "content", "item_name", "file_title"],
        limit=10000,
    )
    joined = "".join(str(row.get("content", "")) for row in rows)
    ratio = mojibake_ratio(joined)
    if ratio > 0.02:
        raise ValueError(
            f"Milvus 集合 {collection} 的 content 疑似编码损坏（异常字符占比 {ratio:.1%}），"
            "请重新导入或改用 --corpus-source md"
        )
    return [
        Chunk(
            chunk_id=str(row.get("chunk_id")),
            title=row.get("title", ""),
            content=row.get("content", ""),
            item_name=row.get("item_name", ""),
            file_title=row.get("file_title", ""),
        )
        for row in rows
    ]


def save_corpus(chunks: Sequence[Chunk], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    logger.info("语料已写入 %s（%d 条）", path, len(chunks))


def load_corpus(path: str = DEFAULT_CORPUS) -> list[Chunk]:
    with open(path, "r", encoding="utf-8") as handle:
        return [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]


def ensure_corpus(path: str = DEFAULT_CORPUS, source: str = "md",
                  md_path: str | None = None, rebuild: bool = False) -> list[Chunk]:
    if os.path.exists(path) and not rebuild:
        return load_corpus(path)
    chunks = build_from_milvus() if source == "milvus" else build_from_markdown(md_path)
    save_corpus(chunks, path)
    return chunks


def corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    import hashlib

    digest = hashlib.sha1()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode())
        digest.update(chunk.content.encode("utf-8"))
    return digest.hexdigest()[:12]


def dense_matrix(vectors: Sequence[object]) -> np.ndarray:
    return np.asarray([vector.dense for vector in vectors], dtype=np.float32)  # type: ignore[attr-defined]


def sparse_list(vectors: Sequence[object]) -> list[dict[int, float]]:
    return [dict(vector.sparse) for vector in vectors]  # type: ignore[attr-defined]