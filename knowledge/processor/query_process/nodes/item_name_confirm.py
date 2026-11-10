import json
import re
from json import JSONDecodeError
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages, update_message_item_names


class ItemNameExtractor:
    """LLM商品名称识别; 清洗LLM返回结果"""

    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name

    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """
        调用LLM进行商品名称识别
        :param original_query: 原始问题   "万用表怎么测量电阻"
        :param history_text: 历史对话内容 10条  最近5轮对话
        :return:
            {
                "item_names": ["RS-12 万用表"],
                "rewritten_query": "RS-12 万用表怎么测量电阻"
            }
        """

        result = {"item_names": [], "rewritten_query": original_query}  # 默认结果

        try:
            llm_client = AIClients.get_llm_openai(True)
        except Exception as e:
            self.logger.warning(f"获取LLM客户端失败，降级为原始问题: {e}")
            return result

        if llm_client is None:
            return result

        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(query=original_query, history_text=history_text)

        try:
            llm_response: AIMessage = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            self.logger.error(f"LLM抽取商品名称失败，降级为原始问题: {e}")
            return result

        llm_content = (getattr(llm_response, "content", "") or "").strip()
        if not llm_content:
            return result

        # 清洗LLM返回结果
        try:
            clean_llm_content_json = self._clean_parse(llm_content)

            result["item_names"] = clean_llm_content_json.get("item_names", [])
            result["rewritten_query"] = clean_llm_content_json.get("rewritten_query", original_query) or original_query
            return result
        except Exception as e:
            self.logger.error(f"LLM返回结果清洗失败：{str(e)}")
            return result

    def _clean_parse(self, llm_content: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        # 1. 清洗 json 代码块围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. 反序列化
        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)
            # 2.1 清洗 item_names
            rwa_item_names = parsed_llm_result.get('item_names')
            if not isinstance(rwa_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in rwa_item_names if isinstance(raw_item, str) and raw_item.strip()]

            # 2.2 清洗 rewritten_query
            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")


class ItemNameAligner:
    """向量化查询、商品名称对齐评分、分数差过滤

    把 LLM 从口语问题里抽出来的商品名（可能是"万用表""RS-12"这类简称），
    对齐到 Milvus 中真实入库的 item_name（如"RS-12 数字万用表"），
    因为下游向量检索是按 item_name 做标量过滤的，名称必须与库中完全一致才能命中。
    """

    def __init__(self, logger, node_name, config):
        self.logger = logger
        self.node_name = node_name
        self.config = config

    def align(self, extracted_names: List[str]) -> List[Dict[str, Any]]:
        """逐条对齐抽取出的商品名

        :param extracted_names: LLM 抽取的商品名列表
        :return: [{"extracted","item_name","file_title","score"}]，按 score 降序
        """
        candidates: List[Dict[str, Any]] = []
        if not extracted_names:
            return candidates

        # 1.获取客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.warning(f"Milvus 客户端不可用，跳过商品名对齐: {e}")
            return candidates

        try:
            embedding_model = AIClients.get_bge_m3_client()
            hybrid_vectors = generate_bge_m3_hybrid_vectors(embedding_model, extracted_names, is_query=True)
        except Exception as e:
            self.logger.warning(f"商品名向量化失败，跳过商品名对齐: {e}")
            return candidates

        # 2.逐条检索 item_name 集合（稠密+稀疏混合）
        for index, extracted in enumerate(extracted_names):
            try:
                search_requests = create_hybrid_search_requests(
                    dense_vector=hybrid_vectors["dense"][index],
                    sparse_vector=hybrid_vectors["sparse"][index],
                    limit=self.config.item_name_max_options,
                )
                hits = execute_hybrid_search_query(
                    milvus_client=milvus_client,
                    collection_name=self.config.item_name_collection,
                    search_requests=search_requests,
                    ranker_weights=(self.config.item_name_dense_weight, self.config.item_name_sparse_weight),
                    norm_score=True,
                    limit=self.config.item_name_max_options,
                    output_fields=["item_name", "file_title"],
                )
            except Exception as e:
                self.logger.warning(f"商品名「{extracted}」对齐检索失败: {e}")
                continue

            if not hits or not hits[0]:
                continue

            hit = hits[0][0]
            entity = hit.get("entity") or {}
            item_name = str(entity.get("item_name") or "").strip()
            if not item_name:
                continue

            candidates.append({
                "extracted": extracted,
                "item_name": item_name,
                "file_title": entity.get("file_title", ""),
                "score": float(hit.get("distance", 0.0)),
            })

        return self._dedupe_and_filter(candidates)

    def _dedupe_and_filter(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """同名商品保留最高分；多商品询问时剔除明显掉队项（仅当命中数大于1）"""
        best_by_name: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            name = candidate["item_name"]
            if name not in best_by_name or candidate["score"] > best_by_name[name]["score"]:
                best_by_name[name] = candidate

        ranked = sorted(best_by_name.values(), key=lambda item: item["score"], reverse=True)
        if len(ranked) <= 1:
            return ranked

        # 分数差异过滤：与最高分差距超过 item_name_gap_abs 的候选视为噪声
        gap_floor = ranked[0]["score"] - self.config.item_name_gap_abs
        filtered = [candidate for candidate in ranked if candidate["score"] >= gap_floor]
        if len(filtered) != len(ranked):
            dropped = [c["item_name"] for c in ranked if c["score"] < gap_floor]
            self.logger.info(f"分数差过滤剔除: {dropped}")
        return filtered


class ItemNameConfirmNode(BaseNode):
    """商品名确认节点

    流程：
        1.获取最近会话（10 条 / 5 轮），拼成指代消解上下文
        2.LLM 抽取 item_names + 改写 rewritten_query
        3.抽取结果向量化，在 item_name 集合中混合检索做对齐
        4.分数差过滤（仅当命中数大于1）
        5.置信度决策：唯一命中→继续检索；多命中歧义→反问；无命中→提示补型号
        6.历史回填（仅当商品名已确认）：把商品名写回此前缺商品名的用户消息
    """

    name = "item_name_confirm"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor(self.logger, self.name)
        self._item_name_aligner = ItemNameAligner(self.logger, self.name, self.config)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id = state.get("session_id")
        original_query = state.get("original_query") or ""

        # 1.获取历史会话（MongoDB 按 ts 倒序返回，这里翻转为时间正序）
        history_messages: List[Dict[str, Any]] = get_recent_messages(session_id)
        state["history"] = list(reversed(history_messages))

        history_text = ""
        for message in history_messages:
            history_text += f"{message.get('role', '')}: {message.get('text', '')}\n"

        # 2.LLM 获取商品名称 + 改写问题
        extract_result: Dict[str, Any] = self._item_name_extractor.extract_item_name(original_query, history_text)
        extracted_names = self._dedupe([
            name.strip() for name in (extract_result.get("item_names") or [])
            if isinstance(name, str) and name.strip()
        ])
        rewritten_query = extract_result.get("rewritten_query") or original_query

        state["extracted_item_names"] = extracted_names
        state["rewritten_query"] = rewritten_query
        self.logger.info(f"LLM 抽取商品名: {extracted_names} | 改写问题: {rewritten_query}")

        # 3.没有识别出商品名，无法做标量过滤，直接反问
        if not extracted_names:
            state["item_names"] = []
            state["answer"] = "请提供具体的设备名称或型号（例如：RS-12 数字万用表），我再为您查阅对应手册。"
            self.logger.info("未识别到商品名称，直接反问用户")
            return state

        # 4.向量化对齐知识库中真实存在的 item_name
        candidates = self._item_name_aligner.align(extracted_names)
        for candidate in candidates:
            self.logger.info(
                f"对齐候选: {candidate['extracted']} -> {candidate['item_name']} "
                f"(score={candidate['score']:.4f}, file={candidate['file_title']})"
            )

        # 5.知识库里一条都没命中（集合为空/服务不可用）时，用抽取结果兜底继续检索
        if not candidates:
            state["item_names"] = extracted_names
            state["answer"] = ""
            self.logger.warning(f"商品名未对齐到知识库，退化为原始抽取结果继续检索: {extracted_names}")
            return state

        # 6.置信度决策
        return self._decide(state, candidates)

    def _decide(self, state: QueryGraphState, candidates: List[Dict[str, Any]]) -> QueryGraphState:
        """根据对齐分数做决策：锁定 / 反问 / 兜底"""
        high_confidence = self.config.item_name_high_confidence
        mid_confidence = self.config.item_name_mid_confidence

        best = candidates[0]
        confident = [candidate for candidate in candidates if candidate["score"] >= mid_confidence]

        # 6.1 全部低于中置信阈值 → 无法确认，提示补充型号
        if not confident:
            state["item_names"] = []
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
            self.logger.info(f"商品名置信度不足（最高 {best['score']:.4f} < {mid_confidence}），返回兜底话术")
            return state

        # 6.2 唯一候选 → 直接锁定
        if len(confident) == 1:
            return self._accept(state, confident)

        # 6.3 多候选：最高分够高且与第二名拉开差距，才敢只取最高分
        gap = best["score"] - confident[1]["score"]
        if best["score"] >= high_confidence and gap >= self.config.item_name_gap_abs:
            self.logger.info(
                f"最高分 {best['score']:.4f} >= {high_confidence} 且领先 {gap:.4f}，锁定 {best['item_name']}"
            )
            return self._accept(state, [best])

        # 6.4 存在歧义 → 反问用户选择，answer 非空会跳过检索直达答案节点
        options = [candidate["item_name"] for candidate in confident[: self.config.item_name_max_options]]
        state["item_names"] = []
        state["answer"] = (
            f"我不确定您指的是哪款产品。您是在询问以下产品吗：{'、'.join(options)}？"
            "请补充具体型号后我再为您查手册。"
        )
        self.logger.info(f"商品名存在歧义，反问候选: {options}")
        return state

    def _accept(self, state: QueryGraphState, chosen: List[Dict[str, Any]]) -> QueryGraphState:
        """锁定商品名，清空 answer 让流程继续走检索"""
        confirmed_names = self._dedupe([candidate["item_name"] for candidate in chosen])
        state["item_names"] = confirmed_names
        state["answer"] = ""
        self.logger.info(f"商品名确认: {confirmed_names}")
        return self._backfill_and_return(state, confirmed_names)

    def _backfill_and_return(self, state: QueryGraphState, confirmed_names: List[str]) -> QueryGraphState:
        """把已确认的商品名回填到此前没有商品名的用户消息上（用于后续轮次指代消解）"""
        history_messages = state.get("history") or []
        ids_to_update = [
            message.get("_id")
            for message in history_messages
            if message.get("role") == "user" and not message.get("item_names")
        ]
        if not ids_to_update:
            return state

        try:
            updated = update_message_item_names(ids_to_update, confirmed_names)
            self.logger.info(f"历史回填 {updated} 条消息 -> {confirmed_names}")
        except Exception as e:
            self.logger.warning(f"历史回填失败（不影响主流程）: {e}")
        return state

    @staticmethod
    def _dedupe(names: List[str]) -> List[str]:
        """去重并保持原顺序"""
        seen = set()
        result = []
        for name in names:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result


if __name__ == '__main__':
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()
    item_name_confirmed_node = ItemNameConfirmNode()

    init_state = {
        "session_id": "sess-f41icdzoh87mr5zeel1",
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
    }
    result_state = item_name_confirmed_node(init_state)
    print("item_names =", result_state.get("item_names"))
    print("rewritten_query =", result_state.get("rewritten_query"))
    print("answer =", result_state.get("answer"))