import json
from typing import Dict, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    item_names_filter,
)


class VectorSearchNode(BaseNode):
    name = "search_embedding"

    # def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict]:
    # def process(self, state: QueryGraphState) -> QueryGraphState or Dict:
    def process(self, state: QueryGraphState) -> QueryGraphState | Dict:
        """
        三路检索之一： 向量检索
        :param state: rewritten_query   item_names
        :return:
            {
                "embedding_chunks": []
            }
        """
        validated_rewritten_query, validated_item_names = self._validate_input(state)

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
            embedding_hybrid_result = generate_bge_m3_hybrid_vectors(embedding_model, [validated_rewritten_query])

            # 4.构建查询条件
            expr, expr_params = item_names_filter(validated_item_names)  # 标量查询条件
            # 5.创建混合检索请求
            search_requests = create_hybrid_search_requests(  # 混合查询条件
                dense_vector=embedding_hybrid_result["dense"][0],
                sparse_vector=embedding_hybrid_result["sparse"][0],
                expr=expr,
                expr_params=expr_params,
                limit=self.config.embedding_search_limit,
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
                limit=self.config.embedding_search_limit,
                output_fields=["chunk_id", "item_name", "content", "title"]
            )

            # 7.回填数据
            return {
                "embedding_chunks": hybrid_result[0]
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


if __name__ == '__main__':
    state = {
        "rewritten_query": "空调如何测量电阻",
        "item_names": ["美的空调"]
    }

    vector_search = VectorSearchNode()
    result = vector_search(state)

    for r in result.get('embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))
"""
{
  "chunk_id": 467775438085177895,
  "distance": 0.7380014657974243,
  "entity": {
    "chunk_id": 467775438085177895,
    "item_name": "美的空调",
    "content": "## 清洁滤尘网\n\n1. 停止运行并断开电源。\n2. 抓住机身背面的滤尘网扣手位，向外用力拉并取出滤尘网。\n3. 用清水洗净后放置在阴凉处晾干。\n4. 待滤网干燥后装回原位。",
    "title": "## 清洁滤尘网"
  }
}
{
  "chunk_id": 467775438085177884,
  "distance": 0.7050827741622925,
  "entity": {
    "chunk_id": 467775438085177884,
    "item_name": "美的空调",
    "content": "## 清洁滤尘网\n\n1. 停止运行并断开电源。\n2. 抓住机身背面的滤尘网扣手位，向外用力拉并取出滤尘网。\n3. 用清水洗净后放置在阴凉处晾干。\n4. 待滤网干燥后装回原位。",
    "title": "## 规格- 3"
  }
}
{
  "chunk_id": 467775438085177892,
  "distance": 0.42116472125053406,
  "entity": {
    "chunk_id": 467775438085177892,
    "item_name": "美的空调",
    "content": "## 故障核查\n\n显示屏出现 E1 代码时：关机并关闭电源，10 分钟后重新上电开机；仍显示代码则联系检修。",
    "title": "## 交流电压测量"
  }
}
{
  "chunk_id": 467775438085177891,
  "distance": 0.41844987869262695,
  "entity": {
    "chunk_id": 467775438085177891,
    "item_name": "美的空调",
    "content": "## 故障核查\n\n显示屏出现 E1 代码时：关机并关闭电源，10 分钟后重新上电开机；仍显示代码则联系检修。",
    "title": "## 直流电压测量"
  }
}
{
  "chunk_id": 467775438085177893,
  "distance": 0.4167860746383667,
  "entity": {
    "chunk_id": 467775438085177893,
    "item_name": "美的空调",
    "content": "## 故障核查\n\n显示屏出现 E1 代码时：关机并关闭电源，10 分钟后重新上电开机；仍显示代码则联系检修。",
    "title": "## 直流电流测量- 1"
  }
}
"""
