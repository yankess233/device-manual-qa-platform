"""纯文本归一化工具：把 LLM 输出里夹带的 JSON / 引号 / 代码围栏剥掉。

背景（BC-01 的真实成因）：主体名称识别节点把 LLM 输出直接当商品名用，
而模型有时返回 `{"item_name": "RS PRO RS-12 数字万用表"}` 这种 JSON。
结果这个 JSON 串被写进 Milvus 的 `item_name` 标量字段，并回填到每个 chunk，
前端问答就会原样展示 `{"item_name": ...}`，模糊匹配也匹配不上。
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
# JSON 里可能承载商品名的常见键
_NAME_KEYS = ("item_name", "name", "product_name", "product", "model", "title")


def normalize_item_name(raw: str | None) -> str:
    """把模型输出归一化成一行纯文本商品名。

    - `{"item_name": "RS-12 数字万用表"}` → `RS-12 数字万用表`
    - ```` ```json {...} ``` ```` → 同上
    - `"RS-12"` → `RS-12`
    - 普通文本原样返回（去首尾空白与换行）
    """
    text = (raw or "").strip()
    if not text:
        return ""

    text = _FENCE.sub("", text).strip()

    # 尝试按 JSON 解析（对象 / 数组 / 带引号字符串）
    if text[:1] in "{[\"":
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            for key in _NAME_KEYS:
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            # 兜底：取第一个非空字符串值
            for value in parsed.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
        elif isinstance(parsed, str) and parsed.strip():
            return parsed.strip()

    # 去掉模型偶尔加的前后缀
    text = text.strip("\"'“”‘’ \n\t")
    text = re.sub(r"\s+", " ", text)
    return text.strip()