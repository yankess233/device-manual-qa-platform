"""任务态存储：内存实现的行为契约 + 降级逻辑。

Redis 实测放在集成测试（需要真实 Redis / fakeredis），
这里保证默认路径（内存）与旧版语义完全一致——这是不引入回归的前提。
"""

from __future__ import annotations

import pytest

from knowledge.utils import task_util
from knowledge.utils.task_store import MemoryTaskStore, get_task_store, reset_task_store


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    monkeypatch.delenv("TASK_BACKEND", raising=False)
    reset_task_store()
    yield
    reset_task_store()


def test_running_and_done_lists():
    store = MemoryTaskStore()
    store.add_running("t1", "upload_file")
    store.add_running("t1", "upload_file")          # 去重
    store.add_running("t1", "entry_node")
    assert store.snapshot("t1")["running"] == ["upload_file", "entry_node"]

    store.add_done("t1", "upload_file")
    snap = store.snapshot("t1")
    assert snap["running"] == ["entry_node"]
    assert snap["done"] == ["upload_file"]

    store.add_done("t1", "upload_file")             # 重复完成不重复记录
    assert store.snapshot("t1")["done"] == ["upload_file"]


def test_status_and_result():
    store = MemoryTaskStore()
    assert store.get_status("t") == ""
    store.set_status("t", "processing")
    assert store.get_status("t") == "processing"

    store.set_result("t", "answer", "3 1/2 位")
    assert store.get_result("t", "answer") == "3 1/2 位"
    assert store.get_result("t", "missing", "默认") == "默认"


def test_durations_are_rounded():
    store = MemoryTaskStore()
    store.add_duration("t", "文档切分", 1.23456)
    assert store.snapshot("t")["durations"] == {"文档切分": 1.23}


def test_task_util_keeps_public_behaviour():
    """task_util 的对外行为不变：英文节点名进、中文展示名出。"""
    task_util.add_running_task("t2", "pdf_to_md_node")
    task_util.add_done_task("t2", "pdf_to_md_node")
    task_util.add_done_task("t2", "document_split_node")
    task_util.update_task_status("t2", task_util.TASK_STATUS_COMPLETED)
    task_util.add_node_duration("t2", "document_split_node", 0.5)
    task_util.set_task_result("t2", "answer", "ok")

    info = task_util.get_task_info("t2")
    assert info["status"] == "completed"
    assert info["running_list"] == []
    assert info["done_list"] == ["PDF转Markdown", "文档切分"]
    assert info["durations"] == {"文档切分": 0.5}
    assert task_util.get_task_result("t2", "answer") == "ok"


def test_unknown_task_returns_empty_info():
    info = task_util.get_task_info("not-exist")
    assert info == {"status": "", "running_list": [], "done_list": [], "durations": {}}


def test_redis_backend_falls_back_when_unavailable(monkeypatch):
    """没有 redis 包/连不上时，必须降级为内存而不是让服务起不来。"""
    monkeypatch.setenv("TASK_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")   # 必然连不上
    reset_task_store()

    store = get_task_store()
    assert isinstance(store, MemoryTaskStore)
    store.set_status("t3", "processing")
    assert store.get_status("t3") == "processing"


def test_store_is_singleton():
    assert get_task_store() is get_task_store()
    first = get_task_store()
    reset_task_store()
    assert get_task_store() is not first

# ---------------- Redis 后端（用 fakeredis 验证键设计，不需要真 Redis） ----------------

@pytest.fixture
def fake_redis(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis", reason="未安装 fakeredis 时跳过")
    pytest.importorskip("redis", reason="未安装 redis 客户端时跳过")
    import redis

    client = fakeredis.FakeStrictRedis(decode_responses=True)
    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda *a, **k: client))
    return client


def test_redis_store_contract_matches_memory(fake_redis):
    from knowledge.utils.task_store import RedisTaskStore

    store = RedisTaskStore("redis://fake")
    store.add_running("t", "upload_file")
    store.add_running("t", "upload_file")
    store.add_running("t", "entry_node")
    assert store.snapshot("t")["running"] == ["upload_file", "entry_node"]

    store.add_done("t", "upload_file")
    store.add_done("t", "upload_file")
    snap = store.snapshot("t")
    assert snap["running"] == ["entry_node"]
    assert snap["done"] == ["upload_file"]

    store.set_status("t", "processing")
    store.set_result("t", "answer", "ok")
    store.add_duration("t", "文档切分", 1.239)
    snap = store.snapshot("t")
    assert snap["status"] == "processing"
    assert snap["durations"] == {"文档切分": 1.24}
    assert store.get_result("t", "answer") == "ok"
    assert store.get_result("t", "missing", "默认") == "默认"


def test_redis_store_sets_ttl(fake_redis):
    from knowledge.utils.task_store import RedisTaskStore

    store = RedisTaskStore("redis://fake")
    store.set_status("t", "processing")
    assert fake_redis.ttl("sb:task:t:meta") > 0


def test_redis_done_order_is_chronological(fake_redis):
    """LPUSH 写入是倒序，snapshot 必须还原成「先完成的在前」。"""
    from knowledge.utils.task_store import RedisTaskStore

    store = RedisTaskStore("redis://fake")
    for node in ("a", "b", "c"):
        store.add_done("t", node)
    assert store.snapshot("t")["done"] == ["a", "b", "c"]
