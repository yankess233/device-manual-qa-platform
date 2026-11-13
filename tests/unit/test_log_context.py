"""结构化日志：字段、上下文注入、request_id 响应头。"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from knowledge.utils import log_context


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured():
    logger = logging.getLogger("kb-test")
    logger.setLevel(logging.INFO)
    handler = CaptureHandler()
    handler.addFilter(log_context.ContextFilter())
    logger.addHandler(handler)
    yield logger, handler
    logger.removeHandler(handler)


def test_context_filter_injects_ids(captured):
    logger, handler = captured
    log_context.new_request_id()
    log_context.bind_task_id("abc123")
    logger.info("hello")

    record = handler.records[-1]
    assert len(record.request_id) == 12
    assert record.task_id == "abc123"


def test_json_formatter_shape(captured):
    logger, handler = captured
    log_context.new_request_id()
    log_context.bind_task_id("t-1")
    logger.info("node_end", extra={"node": "document_split_node", "duration_ms": 12.3})

    record = handler.records[-1]
    payload = json.loads(log_context.JsonFormatter().format(record))

    for key in ("ts", "level", "logger", "request_id", "task_id", "msg"):
        assert key in payload
    assert payload["level"] == "INFO"
    assert payload["node"] == "document_split_node"
    assert payload["duration_ms"] == 12.3
    assert payload["msg"] == "node_end"


def test_json_formatter_keeps_exception(captured):
    logger, handler = captured
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")

    payload = json.loads(log_context.JsonFormatter().format(handler.records[-1]))
    assert "boom" in payload["exc"]


def test_setup_json_logging_is_idempotent():
    original = list(logging.getLogger().handlers)
    try:
        log_context.setup_json_logging()
        log_context.setup_json_logging()
        configured = [h for h in logging.getLogger().handlers if getattr(h, "_kb_configured", False)]
        assert len(configured) == 1, "重复调用不应叠加 handler"
    finally:
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_kb_configured", False):
                root.removeHandler(handler)
        for handler in original:
            if handler not in root.handlers:
                root.addHandler(handler)


def test_timing_middleware_sets_request_id_header():
    app = FastAPI()
    app.add_middleware(log_context.TimingMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/ping")

    assert response.status_code == 200
    request_id = response.headers.get("x-request-id", "")
    assert len(request_id) == 12