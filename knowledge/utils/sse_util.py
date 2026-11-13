import asyncio
import json
import logging
import os
import queue
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request


class SSEEvent:
    PROGRESS = "progress"  # 任务节点进度
    DELTA = "delta"  # LLM 流式输出增量
    FINAL = "final"  # 最终完整答案


# 全局 SSE 任务队列存储
# Key: task_id, Value: queue.Queue
_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的队列"""
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列"""
    # 获取队列对象（队列 阻塞队列(非阻塞队列)）

    q = queue.Queue()
    _task_stream[task_id] = q
    return q


def remove_sse_queue(task_id: str):
    """移除指定任务的队列
    不存在 key 默认返回 None
    """
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"



# ---------------- Redis 模式（TASK_BACKEND=redis）----------------
# 多 worker 部署时，推流（后台任务所在进程）和消费（SSE 请求所在进程）可能不是同一个进程，
# 本地 queue 传不过去，只能走 Redis Pub/Sub。

_redis_sync = None


def _use_redis() -> bool:
    return os.getenv("TASK_BACKEND", "memory").strip().lower() == "redis"


def _redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")


def _channel(task_id: str) -> str:
    return f"sb:stream:{task_id}"


def _get_redis_sync():
    global _redis_sync
    if _redis_sync is None:
        import redis
        _redis_sync = redis.Redis.from_url(_redis_url(), decode_responses=True)
    return _redis_sync


async def _redis_event_stream(task_id: str, request: Request) -> AsyncGenerator:
    """订阅 Redis 频道并把消息转成 SSE 包；空闲时发心跳保活。"""
    from redis.asyncio import Redis as AsyncRedis

    client = AsyncRedis.from_url(_redis_url(), decode_responses=True)
    pubsub = client.pubsub()
    channel = _channel(task_id)
    try:
        await pubsub.subscribe(channel)
        while True:
            if await request.is_disconnected():
                return
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=10.0)
            if message is None:
                yield ": heartbeat\n\n"  # 注释行，前端 EventSource 会忽略
                continue
            try:
                payload = json.loads(message["data"])
            except (TypeError, ValueError):
                continue
            yield _sse_pack(payload.get("event", SSEEvent.DELTA), payload.get("data", {}))
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
        finally:
            await client.close()

def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    """
    通过 task_id 推送事件到 SSE 队列
    """
    # 1. 获取 SSE 队列
    stream_queue = get_sse_queue(task_id)

    # 2. 同进程消费者：直接进本地队列
    if stream_queue:
        stream_queue.put({"event": event, "data": data})

    # 3. 跨进程消费者：走 Redis Pub/Sub（TASK_BACKEND=redis 时）
    if _use_redis():
        try:
            _get_redis_sync().publish(
                _channel(task_id),
                json.dumps({"event": event, "data": data}, ensure_ascii=False),
            )
        except Exception as error:  # 推流失败不能影响主流程
            logging.warning("SSE Redis 发布失败：%s", error)


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator:
    """
    流式输出结果的消费者
    1. 从sse队列中获取结果
    2. 封装队列中的数据以及事件类型为sse协议的数据包格式
    3. 将封装好的数据包yield出去
    Args:
        task_id: 任务id

    Returns:
        AsyncGenerator:异步生成器对象

    """

    # 0. Redis 后端：订阅频道即可，不需要本地队列
    if _use_redis():
        async for chunk in _redis_event_stream(task_id, request):
            yield chunk
        return

    # 1. 根据任务id 获取任务队列对象
    sse_queue = _task_stream.get(task_id)

    # 2. 校验
    if sse_queue is None:
        return

    loop = asyncio.get_event_loop()

    # 3. 让当前线程一直从队列中获取数据【如果队列一旦有数据，就直接获取，如果队列没有数据，等一会，在问一下】
    try:
        while True:

            # 3.1 判断前端sse连接是否关闭（主动探测）--->FastApI:可以感知到：request
            if await request.is_disconnected():
                return
            try:
                # 3.2 从队列中获取(阻塞队列---)为了让事件循环不阻塞，
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                # 3.3 获取事件类型
                event_type = msg.get('event')
                # 3.4 获取事件数据
                event_data = msg.get('data')
                # 3.5 打包返回
                yield _sse_pack(event_type, event_data)  # 打包并且通过yield返回
            except queue.Empty:
                logging.info("队列为空...请稍等")
                continue
    except  (ConnectionResetError, BrokenPipeError):
        # 客户端中断 关闭了窗口或者浏览器
        return

    except asyncio.CancelledError:
        # 服务端中断 协程被取消 重新抛出，让外层知道它被成功取消()
        raise
    finally:
        remove_sse_queue(task_id)
