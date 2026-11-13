"""接口安全：API Key 鉴权与 CORS 白名单。

设计取舍：
- 用依赖注入而不是全局中间件：只保护需要保护的路由（`/docs`、`/import`、`/chat`
  这些页面仍需能打开），也不会把鉴权逻辑撒得到处都是；
- 用 `hmac.compare_digest` 而不是 `==`：避免逐字符比较带来的时序侧信道；
- Status 码统一 401 + `WWW-Authenticate: ApiKey`，前端能据此提示"需要 Key"。

未配置 `API_KEYS` 时**放行**（开发/本地模式），只打一条 warning：
演示时不至于打不开，部署时配了 Key 才真正生效。
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Header, HTTPException, Query

logger = logging.getLogger(__name__)

_warned = False


def configured_api_keys() -> list[str]:
    raw = os.getenv("API_KEYS", "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()]


def require_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
    key: str = Query(default=""),
) -> None:
    """FastAPI 依赖：校验 `X-API-Key` 头或 `?key=` 查询参数。

    之所以同时支持查询参数：`EventSource`（SSE）无法自定义请求头，
    前端订阅 `/stream/{task_id}` 只能走 `?key=`。
    """
    global _warned
    keys = configured_api_keys()
    if not keys:
        if not _warned:
            logger.warning("未配置 API_KEYS，接口处于无鉴权模式（仅建议本地开发使用）")
            _warned = True
        return

    provided = x_api_key or key
    if not provided or not any(hmac.compare_digest(provided, expected) for expected in keys):
        raise HTTPException(
            status_code=401,
            detail="无效或缺失的 API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def allowed_origins() -> list[str]:
    """CORS 白名单。默认只放行本机页面，不再用 `*`。"""
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://127.0.0.1:8001,http://localhost:8000,http://localhost:8001",
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://127.0.0.1:8000", "http://127.0.0.1:8001"]