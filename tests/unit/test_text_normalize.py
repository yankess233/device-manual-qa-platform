"""商品名归一化：LLM 返回 JSON 时必须剥成纯文本（BC-01）。"""

from __future__ import annotations

import pytest

from knowledge.utils.text_normalize import normalize_item_name


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