"""切分节点：标题层级、长切、短合、代码围栏保护。

这些逻辑是"答案质量的第一道闸门"——切错了，后面精排再强也救不回来。
之前完全没有测试，改动切分策略只能靠肉眼比对。
"""

from __future__ import annotations

import pytest

from knowledge.processor.import_process.config import ImportConfig
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.nodes.ducment_split import DocumentSplitNode

FILE_TITLE = "万用表RS-12的使用"


def make_node(max_len: int = 1000, min_len: int = 50) -> DocumentSplitNode:
    config = ImportConfig()
    config.max_content_length = max_len
    config.min_content_length = min_len
    return DocumentSplitNode(config=config)


def run_split(md: str, max_len: int = 1000, min_len: int = 50, **extra) -> list[dict]:
    state = {"md_content": md, "file_title": FILE_TITLE, **extra}
    return make_node(max_len, min_len).process(state)["chunks"]


def test_two_headings_produce_two_chunks():
    body_a = "断开电源后进行电阻测量。" * 20
    body_b = "将转盘置于直流电压档位。" * 20
    md = f"# {FILE_TITLE}\n\n## 电阻测量\n\n{body_a}\n\n## 直流电压测量\n\n{body_b}\n"

    chunks = run_split(md, min_len=10)
    titles = [c["title"] for c in chunks]
    assert len(chunks) >= 2
    assert any("电阻测量" in t for t in titles)
    assert any("直流电压测量" in t for t in titles)


def test_long_section_is_split_and_each_part_fits():
    md = f"# {FILE_TITLE}\n\n## 技术指标说明\n\n" + "重量 350克。尺寸 180x90x40 毫米。" * 120
    chunks = run_split(md, max_len=400, min_len=50)

    assert len(chunks) > 1, "超长章节必须二次切分"
    for chunk in chunks:
        # content 含标题前缀，允许标题本身带来的少量超出
        assert len(chunk["content"]) <= 400 + len(chunk["title"]) + 10


def test_short_sections_are_merged():
    md = (
        f"# {FILE_TITLE}\n\n## 一\n\n短内容。\n\n## 二\n\n也很短。\n\n## 三\n\n依然很短。\n"
    )
    chunks = run_split(md, max_len=1000, min_len=500)
    assert len(chunks) < 3, "过短章节应被合并，而不是各成一个切片"


def test_code_fence_heading_is_not_treated_as_section():
    md = (
        f"# {FILE_TITLE}\n\n## 脚本示例\n\n"
        "```python\n# 这是注释，不是标题\nprint('hello')\n```\n\n"
        "## 另一个章节\n\n正文内容。" * 5
    )
    chunks = run_split(md, max_len=1000, min_len=10)
    titles = " ".join(c["title"] for c in chunks)
    assert "这是注释" not in titles, "代码围栏里的 # 不能被当成章节标题"


def test_missing_md_content_raises():
    node = make_node()
    with pytest.raises(ValidationError):
        node.process({"md_content": "", "file_title": FILE_TITLE})


def test_chunk_carries_context_fields():
    md = f"# {FILE_TITLE}\n\n## 电池更换\n\n用螺丝刀拧开电池后盖。\n"
    chunk = run_split(md, min_len=10)[0]
    for key in ("title", "content", "file_title"):
        assert key in chunk, f"切片缺少上下文字段 {key}"
    assert chunk["file_title"] == FILE_TITLE