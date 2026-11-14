from typing import Any, Dict, List, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    name = "rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        RRF 倒数融合排序
        :param state:
                embedding_chunks  向量检索结果
                    [
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
                    ]
                hyde_embedding_chunks HyDE（假设性文档）检索结果
                    [
                        {
                            'chunk_id': 467775438085177891,
                            'distance': 0.8033741116523743,
                            'entity':
                                {
                                'chunk_id': 467775438085177891,
                                'title': '## 直流电压测量',
                                'item_name': '美的空调',
                                'content': '## 故障核查\n\n显示屏出现 E1 代码时：关机并关闭电源，10 分钟后重新上电开机；仍显示代码则联系检修。'
                                }
                        },
                    ]
        :return: rrf_chunks  RRF融合后的切片
        """
        # 1.获取各路检索结果
        embedding_chunks: list = state.get("embedding_chunks", [])
        hyde_embedding_chunks: list = state.get("hyde_embedding_chunks", [])

        # 2.格式规整化  只要每路中的entity部分
        embedding_chunks_entities: List[Dict[str, Any]] = self._normalize_input(embedding_chunks)
        hyde_embedding_chunks_entities: List[Dict[str, Any]] = self._normalize_input(hyde_embedding_chunks)

        # 3.设置各路权重
        rrf_inputs = [(embedding_chunks_entities, 1.0), (hyde_embedding_chunks_entities, 1.0)]

        # 4.RRF 计算
        # self.config.rrf_k   平滑参数，默认60
        # self.config.rrf_max_results   最大结果数，默认10   排序后截取前10个文档
        rrf_chunks = self._rrf(rrf_inputs, self.config.rrf_k, self.config.rrf_max_results)


        # 6.更新状态返回
        state["rrf_chunks"] = [entity for entity,_ in rrf_chunks] # 只要文档，不要分

        return state

    def _normalize_input(self, input_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        各路列表数据规整化。只保留entity部分
        :param embedding_chunks:
        :return:
        """
        if not input_chunks:
            return []
        normalized_output = []
        for input_chunk in input_chunks:
            entity = input_chunk.get("entity", {})
            if not entity:
                continue
            normalized_output.append(entity)
        return normalized_output

    def _rrf(self, rrf_inputs: List[Tuple[List[Dict[str, Any]], float]], rrf_k: int = 60, rrf_max_results: int = 5) -> \
    List[Tuple[Dict[str, Any], float]]:
        """
        倒数融合排序RRF
        :param rrf_inputs:  待排序的多路数据，每路设置好权重
            [
                (List[Dict[str,Any]], float),
                (List[Dict[str,Any]], float),
            ]
        :param rrf_k: 平滑参数 默认60
        :return: 排好序的列表
            [
                (entity, score),(entity, score)
            ]
        """
        rrf_scores = {}  # 记录每个chunk_id的score
        rrf_doc = {}  # 记录每个文档的数据
        for entity_list, weight in rrf_inputs:
            for index, entity in enumerate(entity_list, 1):  # 索引序号从1开始。而不是从0开始了。
                chunk_id = entity.get("chunk_id")
                if not chunk_id:
                    continue

                # 公式      score =  weight/ (k + ranker_i)
                score = weight / (rrf_k + index)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
                rrf_doc.setdefault(chunk_id, entity)  # 只在第一次出现会存储。多路都有你，也只会记录一次。

        # 按score排序
        data_result = [(rrf_doc.get(chunk_id), score) for chunk_id, score in rrf_scores.items() or []]
        data_result_sorted = sorted(data_result, key=lambda x: x[1], reverse=True)
        print(data_result_sorted)
        # 取前几名
        rrf_chunks = data_result_sorted[:rrf_max_results] if rrf_max_results else data_result_sorted
        return rrf_chunks

# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)
    print(result)
    # print("\n【融合结果】:")
    # for i, chunk in enumerate(result["rrf_chunks"], 1):
    #     print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")
    #
    # print("-" * 60)
    # print("测试完成")