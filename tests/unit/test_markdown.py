"""表格线性化：HTML 表 / MD 表 / 合并单元格。

手册里大量参数是表格形态，表格被切断或结构丢失，答案就会错。
"""

from __future__ import annotations

from knowledge.utils.markdown_util import MarkdownTableLinearizer


def test_html_table_becomes_key_value_lines():
    html = (
        "<table><tr><th>参数</th><th>数值</th></tr>"
        "<tr><td>重量</td><td>350 g</td></tr>"
        "<tr><td>尺寸</td><td>180×90×40 mm</td></tr></table>"
    )
    out = MarkdownTableLinearizer.process(html)

    assert "<table" not in out.lower()
    assert "重量" in out and "350 g" in out
    assert "尺寸" in out and "180×90×40 mm" in out


def test_html_table_without_th_uses_first_row():
    html = (
        "<table><tr><td>量程</td><td>精确度</td></tr>"
        "<tr><td>200Ω</td><td>±0.8%</td></tr></table>"
    )
    out = MarkdownTableLinearizer.process(html)
    assert "200Ω" in out and "±0.8%" in out


def test_markdown_table_becomes_key_value_lines():
    md = (
        "| 参数 | 数值 |\n"
        "| --- | --- |\n"
        "| 显示 | 3 1/2 位 |\n"
        "| 电池 | 9V |\n"
    )
    out = MarkdownTableLinearizer.process(md)
    assert "显示" in out and "3 1/2 位" in out
    assert "电池" in out and "9V" in out
    assert "---" not in out, "分隔行不应残留"


def test_plain_text_is_untouched():
    text = "## 电阻测量\n\n断开电源后测量。"
    assert MarkdownTableLinearizer.process(text) == text


def test_empty_content_returns_as_is():
    assert MarkdownTableLinearizer.process("") == ""