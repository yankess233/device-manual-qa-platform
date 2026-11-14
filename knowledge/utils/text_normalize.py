"""纯文本归一化工具：把 LLM 输出里夹带的 JSON / 引号 / 代码围栏剥掉。

背景（BC-01 的真实成因）：主体名称识别节点把 LLM 输出直接当产品名用，
而模型有时返回 `{"item_name": "美的 KFR-51LW/N8KS1-1P 变频落地式空调器"}` 这种 JSON。
结果这个 JSON 串被写进 Milvus 的 `item_name` 标量字段，并回填到每个 chunk，
前端问答就会原样展示 `{"item_name": ...}`，模糊匹配也匹配不上。
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
# JSON 里可能承载产品名的常见键
_NAME_KEYS = ("item_name", "name", "product_name", "product", "model", "title")


def normalize_item_name(raw: str | None) -> str:
    """把模型输出归一化成一行纯文本产品名。

    - `{"item_name": "美的空调"}` → `美的空调`
    - ```` ```json {...} ``` ```` → 同上
    - `"KFR-51LW/N8KS1-1P"` → `KFR-51LW/N8KS1-1P`
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


_TOKEN = re.compile(r"[A-Za-z]{1,}[A-Za-z0-9\-]{1,}|\d{2,}[A-Za-z0-9\-]*")


def _tokens(text: str) -> set[str]:
    """抽取"型号型" token：含字母且长度>=3，或纯数字长度>=2。

    刻意不抽中文词——"空调""数字"这类通用词命中率太高，会把别的型号也确认掉。
    """
    tokens = {token.upper() for token in _TOKEN.findall(text or "")}
    return {token for token in tokens if len(token) >= 3 or token.isdigit()}


def pick_by_token(extracted: str | None, candidates: list[str]) -> str | None:
    """型号精确匹配短路。

    当用户问题里的型号 token（如 `KFR-51LW/N8KS1-1P`）出现在候选产品名里时，直接确认，
    不再依赖向量相似度阈值。解决"问 KFR-51LW/N8KS1-1P 怎么测电阻，系统却反问是哪款产品"。
    多个候选同时命中时返回 None（有歧义，交回原有阈值逻辑）。
    """
    query_tokens = _tokens(extracted or "")
    if not query_tokens:
        return None

    hits = [c for c in candidates if query_tokens & _tokens(c)]
    if len(hits) == 1:
        return hits[0]
    return None
