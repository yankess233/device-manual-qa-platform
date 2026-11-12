"""HyDE 假设性文档生成（带磁盘缓存）。

真实 HyDE 要调一次 LLM。50 题 × 多个配置会重复调用同一题，
所以按 (query, item_names) 缓存，把成本压到"每题一次"。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Sequence

logger = logging.getLogger(__name__)

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "hyde.json")


class HydeGenerator:
    """调用生产 Prompt 生成假设性文档，结果落盘复用。"""

    def __init__(self, cache_path: str | None = None) -> None:
        self.cache_path = cache_path or _CACHE
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self._cache: dict[str, str] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                self._cache = json.load(handle)
            logger.info("HyDE 缓存命中 %d 条", len(self._cache))
        self._llm = None
        # 模型名必须进缓存键：否则换了模型还命中旧模型的生成结果，
        # "换模型前后对比"会跑出两组一模一样的数字（等于白跑）。
        self.model = os.getenv("LLM_DEFAULT_MODEL", "")
        # 思考模式会影响生成内容（同一模型开/关思考结果不同），也必须进缓存键
        self.thinking = os.getenv("LLM_ENABLE_THINKING", "").strip().lower()
        # 最近一次 generate 的元信息，供评测记录"这次到底调没调 LLM、花了多久"
        self.last_cached = False
        self.last_ms = 0.0

    def _key(self, query: str, item_names: Sequence[str]) -> str:
        payload = f"{self.model}\x00{self.thinking}\x00{query}\x00{'|'.join(item_names)}"
        return hashlib.sha1(payload.encode()).hexdigest()

    def _client(self):
        if self._llm is None:
            from knowledge.utils.client.ai_clients import AIClients

            self._llm = AIClients.get_llm_openai(False)
        return self._llm

    def generate(self, query: str, item_names: Sequence[str]) -> str:
        started = time.perf_counter()
        key = self._key(query, item_names)
        if key in self._cache:
            self.last_cached = True
            self.last_ms = (time.perf_counter() - started) * 1000
            return self._cache[key]
        self.last_cached = False
        from langchain_core.messages import HumanMessage, SystemMessage

        from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE

        names = list(item_names) or ["设备"]
        system_prompt = f"您是一位{names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=names, rewritten_query=query
        )
        response = self._client().invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        text = (getattr(response, "content", "") or "").strip()
        if not text:
            logger.warning("HyDE 返回空内容，退化为原问题：%s", query[:30])
            text = query
        self._cache[key] = text
        self._save()
        self.last_ms = (time.perf_counter() - started) * 1000
        return text

    def _save(self) -> None:
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self._cache, handle, ensure_ascii=False)
        os.replace(tmp, self.cache_path)