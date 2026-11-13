"""在"没有 torch / 向量库 / numpy"的假设下跑测试，模拟 GitHub Actions 的依赖集。

用途：CI 只装 `requirements-ci.txt`，如果某个测试（或它 import 的模块）偷偷用到了
torch / FlagEmbedding / pymilvus / redis / numpy，CI 就会红，而本地因为装了全套依赖
发现不了。这个脚本用 meta_path 钩子把这些包"变没"，让问题在本地就暴露。

用法：
    python tools/check_ci_isolated.py            # 跑 pytest tests eval/tests
    python tools/check_ci_isolated.py --cmd "python -m eval.link_gold"
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

BLOCKED = {
    "torch",
    "torchvision",
    "FlagEmbedding",
    "pymilvus",
    "minio",
    "pymongo",
    "langchain_openai",
    "langgraph",
    "openai",
    "redis",
    "fakeredis",
}


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            # 用 ModuleNotFoundError 而不是裸 ImportError：
            # pytest.importorskip 只对这种"包真的不存在"的异常跳过，
            # 这样带 importorskip 的可选用例（Redis）在 CI 里会正确 skip 而不是红。
            raise ModuleNotFoundError(f"No module named '{fullname}' (ci-isolated)", name=fullname)
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", default="")
    args = parser.parse_args()

    sys.meta_path.insert(0, Blocker())
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    if args.cmd:
        import subprocess

        return subprocess.call(args.cmd, shell=True, cwd=repo)

    import pytest

    return pytest.main(["tests", "eval/tests", "-q"])


if __name__ == "__main__":
    raise SystemExit(main())