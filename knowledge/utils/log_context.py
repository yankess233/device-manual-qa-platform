"""日志上下文：request_id / task_id 贯穿 + JSON 结构化输出。

为什么要 request_id：
并发上来之后，纯文本日志会糊成一团。一条报错属于哪个请求、哪个任务，
只能靠 request_id 串起来：

    docker compose logs -f query-app | jq 'select(.request_id=="a1b2c3d4e5f6")'

为什么用 contextvars 而不是参数透传：
LangGraph 节点签名是固定的 `process(state)`，透传要改一堆函数；
contextvars 在 asyncio 任务里天然隔离，协程之间不会串味。
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from typing import Any

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
task_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("task_id", default="-")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id.set(rid)
    return rid


def bind_task_id(task_id: str) -> None:
    task_id_ctx.set(task_id or "-")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        record.task_id = task_id_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "task_id": getattr(record, "task_id", "-"),
            "msg": record.getMessage(),
        }
        # 允许业务通过 extra={"node": "...", "duration_ms": 12.3} 附加结构化字段
        for key in ("node", "duration_ms", "stage", "tokens", "path"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_json_logging(level: int = logging.INFO) -> None:
    """幂等：重复调用不会叠加 handler。

    LOG_FORMAT=text 可以退回人类可读格式（本地调试更舒服）。
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_kb_configured", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler._kb_configured = True  # type: ignore[attr-defined]
    handler.addFilter(ContextFilter())

    if os.getenv("LOG_FORMAT", "json").strip().lower() == "text":
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-5s | %(request_id)s | %(task_id)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    else:
        handler.setFormatter(JsonFormatter())

    root.addHandler(handler)
    root.setLevel(level)
    # 第三方库降噪
    for noisy in ("httpx", "urllib3", "pymilvus", "minio", "urllib3.connectionpool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_node_end(logger: logging.Logger, node: str, duration: float, ok: bool = True) -> None:
    logger.info(
        "node_end" if ok else "node_error",
        extra={"node": node, "duration_ms": round(duration * 1000, 1)},
    )


class TimingMiddleware:
    """ASGI 中间件：给每个请求分配 request_id，并在响应头回传。

    用纯 ASGI 而不是 BaseHTTPMiddleware：后者对 SSE 长连接有已知的缓冲问题。
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = new_request_id()
        started = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", rid.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            logging.getLogger("http").info(
                "%s %s -> %s",
                scope.get("method", ""),
                scope.get("path", ""),
                f"{elapsed:.0f}ms",
                extra={"path": scope.get("path", ""), "duration_ms": round(elapsed, 1)},
            )