"""部署后连通性自检：在虚拟机宿主机上跑，检查端口与两个服务的 HTTP 健康状态。

用法::

    python3 deploy/verify.py                       # 默认 127.0.0.1
    python3 deploy/verify.py --host 192.168.10.150

退出码 0 表示全部通过；有任何一项失败返回 1，便于写进 CI / 部署脚本。
"""

from __future__ import annotations

import argparse
import socket
import sys
import urllib.error
import urllib.request

CHECKS = [
    ("Milvus gRPC", "tcp", 19530),
    ("MinIO S3", "tcp", 9000),
    ("MongoDB", "tcp", 27017),
    ("导入服务 /docs", "http", 8000),
    ("问答服务 /docs", "http", 8001),
]


def check_tcp(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True, "connected"
        except OSError as error:
            return False, f"{type(error).__name__}: {error}"


def check_http(host: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    url = f"http://{host}:{port}/docs"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200, f"HTTP {response.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return False, f"{type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description="部署连通性自检")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    failures = 0
    print(f"{'检查项':<18}{'目标':<26}{'结果'}")
    print("-" * 72)
    for name, kind, port in CHECKS:
        target = f"{args.host}:{port}"
        ok, detail = check_tcp(args.host, port) if kind == "tcp" else check_http(args.host, port)
        if not ok:
            failures += 1
        print(f"{name:<18}{target:<26}{'OK' if ok else 'FAIL':<6}{detail}")

    print("-" * 72)
    if failures:
        print(f"{failures} 项失败。常见原因：容器没起、端口没映射、应用还在加载模型。")
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())