"""eval 包：检索与端到端评测。

导入本包时会尝试加载 ``knowledge/.env``——项目里的 ``load_dotenv()`` 是相对当前工作
目录找 .env 的，而密钥放在 ``knowledge/.env``，从仓库根目录跑就找不到。
这里显式指定路径，避免"换个目录跑就报缺 key"。
"""

from __future__ import annotations

from pathlib import Path


def load_env() -> None:
    """加载 knowledge/.env（不覆盖已存在环境变量）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (repo_root / "knowledge" / ".env", repo_root / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


load_env()