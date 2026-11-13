"""商品名归一化：LLM 返回 JSON 时必须剥成纯文本（BC-01）。"""

from __future__ import annotations

import pytest

from knowledge.utils.text_normalize import normalize_item_name, pick_by_token


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"item_name": "RS PRO RS-12 数字万用表"}', "RS PRO RS-12 数字万用表"),
        ('{\n  "item_name": "RS PRO RS-12 数字万用表"\n}', "RS PRO RS-12 数字万用表"),
        ('{"name": "HAK180 扫描仪"}', "HAK180 扫描仪"),
        ('{"model": "RS-12"}', "RS-12"),
        ('```json\n{"item_name": "RS-12 万用表"}\n```', "RS-12 万用表"),
        ('"RS-12 数字万用表"', "RS-12 数字万用表"),
        ('["RS-12 数字万用表"]', "RS-12 数字万用表"),
        ("RS-12 数字万用表", "RS-12 数字万用表"),
        ("  RS-12\n数字万用表  ", "RS-12 数字万用表"),
        ("", ""),
        (None, ""),
        ("   ", ""),
    ],
)
def test_normalize_item_name(raw, expected):
    assert normalize_item_name(raw) == expected


def test_unknown_json_structure_falls_back_to_first_string():
    assert normalize_item_name('{"foo": "bar"}') == "bar"


# ---------------- 型号精确匹配短路 ----------------

def test_pick_by_token_matches_model_in_long_name():
    assert pick_by_token("RS-12", ["RS PRO RS-12 数字万用表"]) == "RS PRO RS-12 数字万用表"


def test_pick_by_token_ignores_generic_chinese_query():
    """"电池怎么换"没有型号 token，不应短路，交回阈值逻辑。"""
    assert pick_by_token("电池怎么换", ["RS PRO RS-12 数字万用表"]) is None


def test_pick_by_token_respects_different_models():
    assert pick_by_token("RS-12", ["RS-12A 万用表"]) is None


def test_pick_by_token_returns_none_when_ambiguous():
    assert pick_by_token("RS-12", ["RS-12 数字万用表", "RS-12 钳形表"]) is None


def test_pick_by_token_handles_empty():
    assert pick_by_token("", ["a"]) is None
    assert pick_by_token("RS-12", []) is None
