"""上传守卫与 API Key 鉴权的单元测试（不依赖 torch / FastAPI 之外的框架）。"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from knowledge.api.security import allowed_origins, configured_api_keys, require_api_key
from knowledge.utils import upload_guard
from knowledge.utils.upload_guard import (
    UploadRejected,
    ensure_allowed_suffix,
    max_upload_bytes,
    safe_filename,
    save_upload_limited,
)

# ---------------- 文件名净化 ----------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("万用表RS-12的使用.pdf", "万用表RS-12的使用.pdf"),
        ("../../evil.md", "evil.md"),
        ("..\\..\\evil.md", "evil.md"),
        ("/etc/passwd", "passwd"),
        ("C:\\Windows\\system32\\a.pdf", "a.pdf"),
        ("a b<script>.pdf", "a_b_script_.pdf"),
        ("...hidden.md", "hidden.md"),
    ],
)
def test_safe_filename_strips_path_and_dangerous_chars(raw, expected):
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "..", "/", "///", "\\"])
def test_safe_filename_rejects_empty(raw):
    with pytest.raises(UploadRejected) as exc:
        safe_filename(raw)
    assert exc.value.status_code == 400


def test_safe_filename_truncates_long_names():
    name = safe_filename("文" * 300 + ".pdf")
    assert len(name) <= 120
    assert name.endswith(".pdf")


# ---------------- 后缀白名单 ----------------

def test_allowed_suffix_accepts_pdf_and_md():
    assert ensure_allowed_suffix("a.pdf") == ".pdf"
    assert ensure_allowed_suffix("a.MD") == ".md"


@pytest.mark.parametrize("name", ["a.exe", "a.sh", "a.pdf.sh", "noext"])
def test_allowed_suffix_rejects_others(name):
    with pytest.raises(UploadRejected) as exc:
        ensure_allowed_suffix(name)
    assert exc.value.status_code == 400


def test_allowed_suffix_respects_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_UPLOAD_SUFFIXES", "txt")
    assert ensure_allowed_suffix("a.txt") == ".txt"
    with pytest.raises(UploadRejected):
        ensure_allowed_suffix("a.pdf")


# ---------------- 大小限制 ----------------

def test_save_upload_limited_writes_file(tmp_upload_dir):
    from tests.conftest import FakeUpload

    dest = tmp_upload_dir / "a.md"
    size = save_upload_limited(FakeUpload(b"# hello" * 100), dest, limit=10_000)
    assert dest.read_bytes().startswith(b"# hello")
    assert size == 700


def test_save_upload_limited_rejects_oversize_and_cleans_up(tmp_upload_dir):
    from tests.conftest import FakeUpload

    dest = tmp_upload_dir / "big.md"
    with pytest.raises(UploadRejected) as exc:
        save_upload_limited(FakeUpload(b"x" * 5000), dest, limit=1024)
    assert exc.value.status_code == 413
    assert not dest.exists(), "超限时必须删除半成品文件"


def test_max_upload_bytes_env(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "7")
    assert max_upload_bytes() == 7 * 1024 * 1024

    monkeypatch.setenv("MAX_UPLOAD_MB", "not-a-number")
    assert max_upload_bytes() == upload_guard.DEFAULT_MAX_MB * 1024 * 1024


# ---------------- API Key 依赖 ----------------

def _mini_app() -> FastAPI:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_api_key)])
    def protected():
        return {"ok": True}

    @app.get("/open")
    def open_route():
        return {"ok": True}

    return app


def test_api_key_open_mode_when_not_configured():
    client = TestClient(_mini_app())
    assert client.get("/protected").status_code == 200
    assert configured_api_keys() == []


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k1,k2")
    client = TestClient(_mini_app())

    assert client.get("/protected").status_code == 401
    assert client.get("/protected", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/protected", headers={"X-API-Key": "k2"}).status_code == 200
    # EventSource 场景：查询参数也认
    assert client.get("/protected?key=k1").status_code == 200
    # 未受保护的路由不受影响
    assert client.get("/open").status_code == 200


def test_allowed_origins_defaults_to_local(monkeypatch):
    assert "http://127.0.0.1:8001" in allowed_origins()

    monkeypatch.setenv("ALLOWED_ORIGINS", "https://kb.example.com")
    assert allowed_origins() == ["https://kb.example.com"]