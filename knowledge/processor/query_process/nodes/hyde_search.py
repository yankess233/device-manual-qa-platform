from typing import Dict, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query


class HyDeSearchNode(BaseNode):
    name = "search_embedding_hyde"

    def process(self, state: QueryGraphState) -> Dict:
        """
        三路检索之一： HyDE 假设性文档嵌入检索
        :param state: rewritten_query,item_names
        :return:
            {
                "hyde_embedding_chunks": []
            }
        """
        # 1.参数校验
        validated_rewritten_query, validated_item_names = self._validate_input(state)

        # 2.生成假设性文档  根据问题 -> LLM  ->  200-300字的问题答案
        hyde_document = self._generate_hyde_documents(validated_rewritten_query, validated_item_names)

        # 3.拼接  问题 + 假设性文档  =>  向量化
        embedding_query_str = f"{validated_rewritten_query}\n{hyde_document}"

        # 2.获取客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Failed to get milvus client: {e}")
            return {}

        try:
            embedding_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"Failed to get bge_m3_client: {e}")
            return {}
        try:
            # 3.查询向量化
            embedding_hybrid_result = generate_bge_m3_hybrid_vectors(embedding_model, [embedding_query_str])

            # 4.构建查询条件
            expr, expr_params = item_names_filter(validated_item_names)  # 标量查询条件
            # 5.创建混合检索请求
            search_requests = create_hybrid_search_requests(  # 混合查询条件
                dense_vector=embedding_hybrid_result["dense"][0],
                sparse_vector=embedding_hybrid_result["sparse"][0],
                expr=expr,
                expr_params=expr_params,
                limit=self.config.hyde_search_limit,
                mode=self.config.retrieval_mode
            )

            # 6.执行混合检索
            hybrid_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=search_requests,
                ranker_weights=(self.config.embedding_dense_weight,
                                self.config.embedding_sparse_weight),
                norm_score=True,
                limit=self.config.hyde_search_limit,
                output_fields=["chunk_id", "item_name", "content", "title"]
            )

            # 7.回填数据
            return {
                "hyde_embedding_chunks": hybrid_result[0]
            }
        except Exception as e:
            self.logger.error(f"Failed to execute vector search: {e}")
            return {}

    def _validate_input(self, state) -> Tuple[str, list]:
        # 1.参数校验
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"Invalid rewritten_query: {rewritten_query}")
            raise StateFieldError(self.name, "rewritten_query", str)

        item_names = state.get("item_names")
        if not item_names or not isinstance(item_names, list):
            self.logger.error(f"Invalid item_names: {item_names}")
            raise StateFieldError(self.name, "item_names", str)
        return rewritten_query, item_names

    def _generate_hyde_documents(self, validated_rewritten_query, validated_item_names) -> str:
        """
        生成假设性文档  根据问题 -> LLM  ->  200-300字的问题答案
        :param validated_rewritten_query: 用户问题
        :param validated_item_names: 商品名称列表
        :return: str  假设性文档
        """
        # 1.获取LLM客户端
        try:
            # 不用非得返回json格式结果
            llm_client = AIClients.get_llm_openai(False)
        except ConnectionError as e:
            self.logger.error(f"Failed to get LLM client: {e}")
            return ""

        # 2.调用LLM   重点：提示词
        system_prompt = f"您是一位{validated_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(item_names=validated_item_names,
                                                       rewritten_query=validated_rewritten_query)
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        if not llm_response or not llm_response.content.strip():
            self.logger.info("LLM生成假设性文档内容为空")
            return ""

        # 3.解析结果并返回
        llm_content = llm_response.content.strip()

        return llm_content


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    print(chunks)
    # for i, chunk in enumerate(chunks, 1):
    #     entity = chunk.get("entity", {})
    #     print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
    #           f"item_name={entity.get('item_name')} "
    #           f"distance={chunk.get('distance', 'N/A')}")
    #     content = entity.get("content", "")
    #     print(f"      内容: {content[:80]}...")

    print("-" * 60)
    print("测试完成")

"""
[
	{
		'chunk_id': 467775438085177891, 
		'distance': 0.8033741116523743, 
		'entity': 
			{
			'title': '## 直流电压测量', 
			'chunk_id': 467775438085177891, 
			'item_name': 'RS-12 数字万用表', 
			'content': '## 直流电压测量\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n1. 将功能转盘置于V DC的位置。\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。'
			}
	}, 


	{
		'chunk_id': 467775438085177893, 
		'distance': 0.8024689555168152, 
		'entity': 
			{
				'title': '## 直流电流测量- 1', 
				'chunk_id': 467775438085177893, 
				'item_name': 'RS-12 数字万用表', 
				'content': '## 直流电流测量- 1\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n![直流电流测量接线示意图（10A档位）](http://127.0.0.1:9000/knowledge-base-files/万用表RS-12的使用/8eb1e59b1e3f5e200f6d947da47dcd767fe061b91e72b1ba5325869677dcdad2.jpg)\n1. 将黑色表笔插入负极COM端口。\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n6. 接通电源。\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。'
			}
	}
]
"""