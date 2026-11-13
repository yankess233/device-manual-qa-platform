"""公共 fixture：只放不依赖 torch / Milvus / 网络的替身。

原则：`tests/unit` 必须能在纯 Python 环境跑完（CI 用），
所以这里刻意不 import `knowledge.utils.client.ai_clients`（它会拉起 FlagEmbedding + torch）。
需要 LLM 替身的用例在 `tests/integration` 里自行 patch。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """默认清空安全/上传相关环境变量，避免本机 .env 影响断言。"""
    for name in ("API_KEYS", "ALLOWED_ORIGINS", "MAX_UPLOAD_MB", "ALLOWED_UPLOAD_SUFFIXES"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def tmp_upload_dir(tmp_path: Path) -> Path:
    return tmp_path / "task-abc123"


class FakeUpload:
    """最小化替身：只实现 upload_guard 用到的 `read`。"""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk, self._data = self._data, b""
            return chunk
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk