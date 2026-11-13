"""上传守卫：文件名净化、后缀白名单、流式大小限制。

刻意放在 utils 而不是 api 层：这里全是纯逻辑，不依赖 FastAPI，
可以被单元测试直接覆盖（`tests/unit/test_upload_guard.py`），
也不需要 torch / 向量库。

背景：原实现是

    import_file_path = os.path.join(file_dir, file.filename)

`file.filename` 来自客户端，未做任何处理。上传时把文件名设成 `../../../../tmp/x.md`
就会跳出任务目录写文件；Windows 客户端还能传 `C:\\Windows\\...` 这类绝对路径。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import BinaryIO

# 白名单字符：字母、数字、下划线、点、连字符、中文。其余一律替换为下划线。
SAFE_NAME_PATTERN = re.compile(r"[^0-9A-Za-z._\-\u4e00-\u9fa5]")

DEFAULT_SUFFIXES = ".pdf,.md,.markdown"
DEFAULT_MAX_MB = 50
MAX_NAME_LENGTH = 120


class UploadRejected(Exception):
    """上传被拒绝。`status_code` 直接透传给 HTTP 层。"""

    def __init__(self, detail: str, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def safe_filename(raw: str | None) -> str:
    """只保留最后一段文件名，并过滤危险字符。

    - `../../evil.md`      → `evil.md`
    - `..\\..\\evil.md`    → `evil.md`（Windows 分隔符同样处理）
    - `/etc/passwd`        → `passwd`
    - `..`                 → 拒绝（净化后为空）
    """
    # 统一分隔符：Linux 上 Path 不认反斜杠，先把反斜杠换掉再取 basename
    normalized = (raw or "").replace("\\", "/")
    name = Path(normalized).name
    name = SAFE_NAME_PATTERN.sub("_", name).strip("._ ")
    if not name:
        raise UploadRejected("文件名非法", status_code=400)

    if len(name) > MAX_NAME_LENGTH:
        stem, suffix = os.path.splitext(name)
        keep = max(MAX_NAME_LENGTH - len(suffix), 1)
        name = f"{stem[:keep]}{suffix}"
    return name


def allowed_suffixes() -> set[str]:
    raw = os.getenv("ALLOWED_UPLOAD_SUFFIXES", DEFAULT_SUFFIXES)
    suffixes = set()
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        suffixes.add(item if item.startswith(".") else f".{item}")
    return suffixes


def ensure_allowed_suffix(filename: str) -> str:
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in allowed_suffixes():
        allowed = "、".join(sorted(allowed_suffixes())) or "（未配置）"
        raise UploadRejected(f"不支持的文件类型：{suffix or '无后缀'}，仅支持 {allowed}")
    return suffix


def max_upload_bytes() -> int:
    try:
        mb = int(os.getenv("MAX_UPLOAD_MB", str(DEFAULT_MAX_MB)))
    except (TypeError, ValueError):
        mb = DEFAULT_MAX_MB
    return max(mb, 1) * 1024 * 1024


def save_upload_limited(src: BinaryIO, dest: Path, limit: int | None = None) -> int:
    """边写边计数，超限立刻删除半成品。

    不能"先读完再判断"：`Content-Length` 可以伪造，先读完等于先把磁盘写满。
    """
    limit = limit if limit is not None else max_upload_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise UploadRejected(
                        f"文件超过大小限制（{limit // (1024 * 1024)} MB）", status_code=413
                    )
                out.write(chunk)
    except UploadRejected:
        dest.unlink(missing_ok=True)
        raise
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return total