"""任务状态存储：内存（默认）与 Redis（多 worker / 重启恢复）。

为什么需要这一层：
- 原实现把任务状态放在 `task_util.py` 的模块级字典里，只对**单进程**有效。
  一旦 `uvicorn --workers 2`，上传落在 worker A、`/status` 轮询落到 worker B，
  B 的内存里没有这个 task，前端会一直转圈；
- 服务重启（改代码、OOM、`docker restart`）后所有进行中的任务状态清零。

切换方式：`TASK_BACKEND=redis` + `REDIS_URL=redis://redis:6379/0`。
默认 `memory`，行为与原来完全一致（本地开发、单测不需要 Redis）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Protocol

logger = logging.getLogger(__name__)

TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", str(24 * 3600)))
_KEY_PREFIX = "sb:task"


class TaskStore(Protocol):
    def add_running(self, task_id: str, node: str) -> None: ...
    def add_done(self, task_id: str, node: str) -> None: ...
    def set_status(self, task_id: str, status: str) -> None: ...
    def get_status(self, task_id: str) -> str: ...
    def set_result(self, task_id: str, key: str, value: str) -> None: ...
    def get_result(self, task_id: str, key: str, default: str = "") -> str: ...
    def add_duration(self, task_id: str, node_cn: str, seconds: float) -> None: ...
    def snapshot(self, task_id: str) -> dict: ...


class MemoryTaskStore:
    """进程内实现：语义与旧版 dict 完全一致。"""

    def __init__(self) -> None:
        self._running: dict[str, list[str]] = {}
        self._done: dict[str, list[str]] = {}
        self._status: dict[str, str] = {}
        self._result: dict[str, dict[str, str]] = {}
        self._duration: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    def add_running(self, task_id: str, node: str) -> None:
        with self._lock:
            running = self._running.setdefault(task_id, [])
            if node not in running:
                running.append(node)

    def add_done(self, task_id: str, node: str) -> None:
        with self._lock:
            running = self._running.setdefault(task_id, [])
            if node in running:
                running.remove(node)
            done = self._done.setdefault(task_id, [])
            if node not in done:
                done.append(node)

    def set_status(self, task_id: str, status: str) -> None:
        with self._lock:
            self._status[task_id] = status

    def get_status(self, task_id: str) -> str:
        return self._status.get(task_id, "")

    def set_result(self, task_id: str, key: str, value: str) -> None:
        with self._lock:
            self._result.setdefault(task_id, {})[key] = value

    def get_result(self, task_id: str, key: str, default: str = "") -> str:
        return self._result.get(task_id, {}).get(key, default)

    def add_duration(self, task_id: str, node_cn: str, seconds: float) -> None:
        with self._lock:
            self._duration.setdefault(task_id, {})[node_cn] = round(seconds, 2)

    def snapshot(self, task_id: str) -> dict:
        with self._lock:
            return {
                "status": self._status.get(task_id, ""),
                "running": list(self._running.get(task_id, [])),
                "done": list(self._done.get(task_id, [])),
                "durations": dict(self._duration.get(task_id, {})),
            }


class RedisTaskStore:
    """Redis 实现：多 worker 共享，带 TTL，重启不丢。

    键设计（前缀 `sb:task:`）：
        {id}:meta     Hash   status
        {id}:running  List   运行中节点（RPUSH / LREM）
        {id}:done     List   已完成节点（LPUSH，SQL 式去重由 guard set 保证）
        {id}:dur      Hash   节点中文名 -> 秒
        {id}:res      Hash   结果字段（answer / error ...）
    """

    def __init__(self, url: str | None = None) -> None:
        import redis  # 局部导入：不用 Redis 的部署不需要装这个包

        self._url = url or os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        self._client = redis.Redis.from_url(self._url, decode_responses=True)
        self._client.ping()
        logger.info("TaskStore: Redis 后端 %s", self._url)

    # -- 内部工具 --
    def _key(self, task_id: str, suffix: str) -> str:
        return f"{_KEY_PREFIX}:{task_id}:{suffix}"

    def _touch(self, *keys: str) -> None:
        for key in keys:
            self._client.expire(key, TASK_TTL_SECONDS)

    # -- 接口实现 --
    def add_running(self, task_id: str, node: str) -> None:
        key = self._key(task_id, "running")
        if node not in (self._client.lrange(key, 0, -1) or []):
            self._client.rpush(key, node)
        self._touch(key)

    def add_done(self, task_id: str, node: str) -> None:
        running_key = self._key(task_id, "running")
        done_key = self._key(task_id, "done")
        guard_key = self._key(task_id, "done_guard")
        self._client.lrem(running_key, 0, node)
        if self._client.sadd(guard_key, node):
            self._client.lpush(done_key, node)
        self._touch(running_key, done_key, guard_key)

    def set_status(self, task_id: str, status: str) -> None:
        key = self._key(task_id, "meta")
        self._client.hset(key, "status", status)
        self._touch(key)

    def get_status(self, task_id: str) -> str:
        return self._client.hget(self._key(task_id, "meta"), "status") or ""

    def set_result(self, task_id: str, key: str, value: str) -> None:
        redis_key = self._key(task_id, "res")
        self._client.hset(redis_key, key, value)
        self._touch(redis_key)

    def get_result(self, task_id: str, key: str, default: str = "") -> str:
        value = self._client.hget(self._key(task_id, "res"), key)
        return default if value is None else value

    def add_duration(self, task_id: str, node_cn: str, seconds: float) -> None:
        key = self._key(task_id, "dur")
        self._client.hset(key, node_cn, round(seconds, 2))
        self._touch(key)

    def snapshot(self, task_id: str) -> dict:
        durations = self._client.hgetall(self._key(task_id, "dur")) or {}
        return {
            "status": self.get_status(task_id),
            "running": self._client.lrange(self._key(task_id, "running"), 0, -1) or [],
            # LPUSH 是倒序写入，读出来再反转，保持"完成先后"顺序
            "done": list(reversed(self._client.lrange(self._key(task_id, "done"), 0, -1) or [])),
            "durations": {k: float(v) for k, v in durations.items()},
        }


_store: TaskStore | None = None
_store_lock = threading.Lock()


def get_task_store() -> TaskStore:
    """按 TASK_BACKEND 选择后端；Redis 不可用时降级为内存并告警。"""
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        backend = os.getenv("TASK_BACKEND", "memory").strip().lower()
        if backend == "redis":
            try:
                _store = RedisTaskStore()
            except Exception as error:  # 连不上就别让整个服务起不来
                logger.error("Redis 任务态不可用，降级为内存实现：%s", error)
                _store = MemoryTaskStore()
        else:
            _store = MemoryTaskStore()
        return _store


def reset_task_store() -> None:
    """测试用：清掉单例。"""
    global _store
    _store = None


def dump_result_fields(task_id: str) -> dict:
    """调试辅助：把结果字段整体取出（仅内存后端支持完整导出）。"""
    store = get_task_store()
    if isinstance(store, MemoryTaskStore):
        return dict(store._result.get(task_id, {}))
    return {"note": "redis backend: 请直接读取 sb:task:%s:res" % task_id}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)