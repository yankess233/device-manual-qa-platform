import json
import re
from json import JSONDecodeError
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pymilvus import AnnSearchRequest

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import QueryConfig
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

        llm_client = AIClients.get_llm_openai(True)
        if not llm_client:
            return result

        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(query=original_query, history_text=history_text)

        llm_response: AIMessage = llm_client.invoke([
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        llm_content = llm_response.content.strip()
        if not llm_content:
            return result

        # 清洗LLM返回结果
        try:
            clean_llm_content_json = self._clean_parse(llm_content)

            result["item_names"] = clean_llm_content_json.get("item_names", [])
            result["rewritten_query"] = clean_llm_content_json.get("rewritten_query", original_query)
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
                clean_item_names = [raw_item for raw_item in rwa_item_names if raw_item.strip()]

            # 2.2 清洗 rewritten_query
            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}") from e


class ItemNameAligner:
    """向量化查询,商品名称对齐评分,分数差过滤"""

    def __init__(self, logger, node_name, config: QueryConfig):
        self.logger = logger
        self.node_name = node_name
        self.config = config

    def match_vector(self, item_names) -> List[Dict[str, Any]]:
        """向量匹配： 根据LLM识别商品名称，从Milvus中查询相似商品名称（hybrid混合向量检索）
        args: 商品名称列表
        return:  search_results: List[Dict[str, Any]]
            [
                {
                    "extracted_name":extracted_name,
                    "matches": [
                        {"item_name":"商品A1", "score":0.9},
                        {"item_name":"商品A2", "score":0.6},
                    ]
                },
                {
                    "extracted_name":extracted_name,
                    "matches": [
                        {"item_name":"商品B1", "score":0.9},
                        {"item_name":"商品B2", "score":0.6},
                    ]
                }
            ]
        """
        search_results: List[Dict[str, Any]] = []  # 默认结果

        # 1.获取milvus客户端对象
        milvus_client = StorageClients.get_milvus_client()
        if not milvus_client:
            return search_results

        # 2.获取向量转换模型  BGE-M3
        embedding_model = AIClients.get_bge_m3_client()
        if not embedding_model:
            return search_results

        # 3.将商品名称转换为向量,调用工具类，批量转换
        # item_names_embeddings ={
        #    "dense": [[],[]],
        #    "sparse": [{},{}]
        # }
        item_names_embeddings = generate_bge_m3_hybrid_vectors(embedding_model, item_names, True)

        # 4.循环并装填：查询mivlus库中商品名称匹配。混合检索。
        for index, extract_item_name in enumerate(item_names):
            dense = item_names_embeddings["dense"][index]
            sparse = item_names_embeddings["sparse"][index]

            # 4.1 构建查询对象
            hybrid_search_requests: List[AnnSearchRequest] = create_hybrid_search_requests(dense, sparse)

            # 4.2 执行混合检索
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.item_name_collection,  # kb_item_names_v1
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                limit=5,
                output_fields=["item_name"]  # 只查询kb_item_names_v1集合中item_name字段
            )

            # .4.3 封装数据，装填列表
            matches = []
            if hybrid_search_result and hybrid_search_result[0]:
                matches = [
                    {
                        "item_name": hit.get("entity", {}).get("item_name", ""),
                        "score": float(hit.get("distance", 0.0)),
                    }
                    for hit in hybrid_search_result[0]
                    if hit.get("entity", {}).get("item_name")
                ]

            search_result = {
                "extracted_name": extract_item_name,
                "matches": matches,
            }
            search_results.append(search_result)
        return search_results

    def score_align(self, search_results) -> Tuple[List[str], List[str]]:
        """评分对齐
            规则：
                1.   score >=0.7  高分区
                2.   0.6  <= score < 0.7  中分区
                3.   score < 0.6  低分区
            args: search_results   向量检索，根据商品名称，进行向量混合检索得到列表：
                [
                    {
                        "extracted_name":extracted_name,
                        "matches": [
                            {"item_name":"商品A1", "score":0.9},
                            {"item_name":"商品A2", "score":0.8},
                        ]
                    },
                    {
                        "extracted_name":extracted_name,
                        "matches": [
                            {"item_name":"商品A1", "score":0.65},
                            {"item_name":"商品B2", "score":0.6},
                        ]
                    }
                ]
            return :
                confirmed: List[str] 高分区商品名称列表 [:3]  确认商品名称列表，立即三路检索
                options: List[str] 中分区商品名称列表    选项商品名称列表[:3]，需要让用户选择商品名，再次进行问题解答
        """
        confirmed: List[str] = []
        options: List[str] = []

        # 向量匹配的列表
        for search_result in search_results:
            extracted_name = search_result["extracted_name"]

            # 每一个商品名称检索的类别
            matches = sorted(search_result["matches"], key=lambda x: x["score"], reverse=True)  # 分数降序

            # 获取高分区列表
            high = [match["item_name"] for match in matches if match["score"] >= self.config.item_name_high_confidence]

            if high:
                # 找到同名的则退出。没找到同名的返回None
                #samename = next([item_name for item_name in high if item_name == extracted_name], None)
                samename = next((item_name for item_name in high if item_name == extracted_name), None)
                if samename:
                    if samename not in confirmed:
                        confirmed.append(samename)
                elif len(high) == 1:
                    if high[0] not in confirmed:
                        confirmed.append(high[0])
                else:
                    for h in high[:3]:
                        if h not in options and h not in confirmed:
                            options.append(h)
            else:
                mid = [match["item_name"] for match in matches
                       if self.config.item_name_mid_confidence <= match["score"] < self.config.item_name_high_confidence
                       and match["item_name"] not in options
                       and match["item_name"] not in confirmed]
                if mid:
                    for m in mid[:3]:
                        options.append(m)

        return confirmed, options[:3]

    def score_filter(self, confirmed, search_results):
        """分数差过滤
        在高分区中，获取最高分。最高分减去 每一个分数，差值小于 0.15 保留；  否则丢弃
        """
        high_confirmed_item_score = {}  # 所有高置信的名称和分字典
        for search_result in search_results:
            matches = search_result["matches"]
            for match in matches:
                item_name = match["item_name"]
                score = match["score"]
                if item_name in confirmed:
                    high_confirmed_item_score[item_name] = max(high_confirmed_item_score.get(item_name,0), score)

        sorted_high_confirmed_item_score = sorted(high_confirmed_item_score.items(), key=lambda x: x[1], reverse=True)
        high_score = sorted_high_confirmed_item_score[0][1]  # 最高分

        # 保留高置信组中 与最高分差值小于阈值的商品名称
        confirmed = [item_name for item_name, score in high_confirmed_item_score.items()
                     if high_score - score <= self.config.item_name_gap_abs]
        return confirmed


class ItemNameConfirmNode(BaseNode):
    name = "item_name_confirm"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor(self.logger, self.name)
        self._item_name_aligner = ItemNameAligner(self.logger, self.name, self.config)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """负责从用户问题中提取并确认商品名称
            流程步骤：
                1.获取历史会话
                    {
                      "_id": {
                        "$oid": "6a48cc0f82601667381db7a5"
                      },
                      "session_id": "sess-f41icdzoh87mr5zeel1",
                      "role": "user",
                      "text": "万用表怎么测量电阻",
                      "rewritten_query": "万用表怎么测量电阻",
                      "item_names": [],
                      "ts": 1783155727.498414
                    }
                    {
                      "_id": {
                        "$oid": "6a48cc0f82601667381db7a6"
                      },
                      "session_id": "sess-f41icdzoh87mr5zeel1",
                      "role": "assistant",
                      "text": "我不确定您指的是哪款产品。, 您是在询问以下产品吗: RS PRO RS-12 数字万用表、HAK 180 扫描仪？",
                      "rewritten_query": "万用表怎么测量电阻",
                      "item_names": [],
                      "ts": 1783155727.52313
                    }
                2.LLM获取商品名称
                3.向量化匹配
                4.评分对齐
                5.分数差异过滤(仅当confirmed长度大于1)
                6.决策更新状态
                7.历史回填(仅当confirmed有值)
        """
        session_id = state.get("session_id")
        original_query = state.get("original_query") or ""

        # 1.获取历史会话
        history_messages: List[Dict[str, Any]] = get_recent_messages(session_id)

        history_text = ""  # 准备提示词的模板参数
        for message in history_messages:
            history_text += f"{message['role']}: {message['text']}\n"

        # 2.LLM获取商品名称
        extract_item_names: Dict[str, Any] = self._item_name_extractor.extract_item_name(original_query, history_text)
        print("extract_item_names=", extract_item_names)
        # extract_item_names= {'item_names': ['RS-12数字万用表', 'HAK180'],
        # 'rewritten_query': 'RS-12数字万用表如何测量电压，以及HAK180的介质规格有哪些？'}
        item_names = extract_item_names['item_names']
        rewritten_query = extract_item_names['rewritten_query']
        # 去重（LLM 可能把同一商品重复输出），保持原顺序
        item_names = list(dict.fromkeys(item_names or []))

        # 3.向量化匹配
        # LLM识别的商品名称 -> BGE-M3稠密+稀疏向量 -> Milvus item_name 集合混合检索 Top5
        # search_results = [
        #     {"extracted_name": "万用表", "matches": [{"item_name": "RS-12 数字万用表", "score": 0.92}, ...]},
        #     {"extracted_name": "HAK180", "matches": [{"item_name": "HAK180 扫描仪", "score": 0.78}, ...]},
        # ]
        search_results: List[Dict[str, Any]] = []
        align_failed = False  # 对齐链路是否异常（Milvus/嵌入模型不可用）

        if item_names:
            try:
                search_results = self._item_name_aligner.match_vector(item_names)
            except Exception as e:
                align_failed = True
                self.logger.error(f"商品名称向量匹配失败: {e}")

        # 4.评分对齐
        # confirmed 高置信区商品名称（得分>=0.7）：与用户问题中的商品几乎一致，可直接进入三路检索
        # options   中置信区商品名称（0.6<=得分<0.7）或高置信多选：需要用户二次选择
        confirmed, options = self._item_name_aligner.score_align(search_results)

        # 5.分数差异过滤(仅当confirmed长度大于1)
        # 与最高分差值超过 item_name_gap_abs 的商品视为噪声，丢弃
        if len(confirmed) > 1:
            confirmed = self._item_name_aligner.score_filter(confirmed, search_results)

        # 5.1 对齐链路异常时兜底：用LLM识别的名称继续检索，不阻断问答
        if align_failed:
            confirmed, options = list(item_names or []), []

        # state['answer'] 存在   -> 反问用户（options）或兜底道歉，流程直接进答案节点
        # state['answer'] 不存在 -> 商品名称已确认，继续走三路检索
        # 6.决策更新状态
        self._decide(state, confirmed, options, rewritten_query)

        # 7.历史回填(仅当confirmed有值)
        # 已确认的商品名称写回历史会话（MongoDB + 内存），供后续轮次指代消解
        self._backfill_history(history_messages, confirmed)

        # 历史会话按时间正序放入state，给后续节点（答案节点拼提示词）使用
        state["history"] = list(reversed(history_messages))

        return state

    def _backfill_history(self, history_messages, confirmed):
        """把已确认的商品名称回填到历史会话中此前没有商品名的用户消息

        :param history_messages: MongoDB 查出的最近会话（按 ts 倒序）
        :param confirmed: 已确认的商品名称列表，为空时不做回填
        :return: 实际更新到 MongoDB 的消息条数
        """
        # 1.未确认商品名称，不回填，避免把错误名称写进历史
        if not confirmed:
            return 0

        # 2.筛选需要回填的用户消息：此前没有商品名称的
        ids_to_update = []
        for message in history_messages:
            if message.get("role") != "user":
                continue
            if message.get("item_names"):
                continue
            message["item_names"] = confirmed  # 内存中的历史同步更新，后续节点直接用
            if message.get("_id") is not None:
                ids_to_update.append(message.get("_id"))

        if not ids_to_update:
            return 0

        # 3.持久化到 MongoDB
        updated = update_message_item_names(ids_to_update, confirmed)
        self.logger.info(f"历史回填商品名称: {updated} 条 -> {confirmed}")
        return updated

    def _decide(self, state, confirmed, options, rewritten_query):
        """
        如果confirmed存在，直接确认，进行三路检索
        如果options存在,生成state["answer"] 提示用户选择
        否则：生成state["answer"] 抱歉
        :param state:
        :param confirmed: 确认商品名称列表
        :param options: 待选商品列表，需要用户来选择
        :param rewritten_query: 查询问题  被LLM润色后问题【指代消除,润色】
        :return: None
        """
        # 问题改写结果统一落库，选项/抱歉分支也要带上，历史里才是指代消解后的问题
        state["rewritten_query"] = rewritten_query or state.get("original_query") or ""

        if confirmed:
            # 商品名称已确认：清空answer，让流程图继续走三路检索
            state["item_names"] = confirmed
            state["answer"] = ""
            self.logger.info(f"商品名称已确认: {confirmed}")
        elif options:
            # 存在多个候选商品：不检索，直接反问用户选择
            state["item_names"] = []
            state["answer"] = (
                f"我不确定您指的是哪款产品。您是在询问以下产品吗：{'、'.join(options)}？"
                "请补充具体型号后我再为您查手册。"
            )
            self.logger.info(f"商品名称存在歧义，反问候选: {options}")
        else:
            # 既没有确认商品，也没有候选商品：兜底话术
            state["item_names"] = []
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
            self.logger.info("未识别到可用的商品名称，返回兜底话术")


if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmNode()
    init_state = {
        "session_id": "sess-f41icdzoh87mr5zeel1",
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        #"original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
        # "original_query": "苹果手机怎么用?"  # 单个商品询问
        # "original_query": "今天天气怎么样?"  # 单个商品询问
        "original_query": "手表怎么测电压??"  # 单个商品询问
    }
    processed_state = item_name_confirmed_node(init_state)
    print(json.dumps(processed_state, indent=4, ensure_ascii=False))
