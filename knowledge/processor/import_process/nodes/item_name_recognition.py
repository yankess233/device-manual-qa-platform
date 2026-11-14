import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.text_normalize import normalize_item_name


class ItemNameRecognitionNode(BaseNode):
    """
        产品名称识别
        利用3个chunk的内容上下文 小于2500字符。通过提示词模板让LLM识别出产品名称及型号。将产品名称信息Bge-m3向量化存Milvus，供后续匹配使用。
    """

    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1.校验参数，获取file_title,chunks,item_name_chunk_k,item_name_chunk_size
        # config对象中属性不合法： ValidationError
        # state对象中属性不合法： StateFieldError
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._valiedate_state(state)

        # 2.构建识别产品名称上下文，取前3个切片，限制字符数2500。
        item_name_recognition_context:str = self._prepare_context(chunks,item_name_chunk_k,item_name_chunk_size)

        # 3.调用LLM,获取客户端AIClients，识别产品名称。 提示词模板；  降级：file_title
        item_name = self._recognition_item_name(file_title,item_name_recognition_context)

        # 4.BGE-M3向量化
        dense_vector,sparse_vector = self._embedding_item_name(item_name)

        # 5.存储到Milvus   StorageClients获取客户端，创建集合，数据保存;  存储失败跳过，并记录日志。
        #标量字段：pk, file_title ,item_name
        #向量字段：dense_vector,sparse_vector
        self.insert_milvus(file_title,item_name,dense_vector,sparse_vector)

        # 6.回填   将item_name回填每一个chunk对象中。也同时回填state对象中。
        self._fill_item_name(item_name,state,chunks)

        return state

    def _valiedate_state(self, state) -> Tuple[str, List[Dict[str, Any]], int, int]:
        """
        1.校验参数，获取file_title,chunks,item_name_chunk_k,item_name_chunk_size
        :param state: ImportGraphState
            必须提供：file_title,chunks
        :return: Dict[str,Any]
        """
        file_title = state['file_title']
        chunks = state['chunks']
        item_name_chunk_k = self.config.item_name_chunk_k  # 3
        item_name_chunk_size = self.config.item_name_chunk_size  # 25000

        if not file_title:
            self.logger.error("file_title不存在")
            raise StateFieldError(self.name, "file_title", expected_type=str)
        if not chunks and not isinstance(chunks, list):
            self.logger.error("chunks不存在")
            raise StateFieldError(self.name, "chunks", expected_type=list)

        if not item_name_chunk_k or item_name_chunk_k <= 0:
            self.logger.error("item_name_chunk_k不存在或小于等于零")
            raise ValidationError("item_name_chunk_k不存在或小于等于零", self.name)
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            self.logger.error("item_name_chunk_size不存在或小于等于零")
            raise ValidationError("item_name_chunk_size不存在或小于等于零", self.name)

        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_context(self, chunks, item_name_chunk_k, item_name_chunk_size) -> str:
        """
        通过3个切片的内容组装LLM提示词模板参数。
        :param chunks: List[Dict[str, Any]] 切片列表
        :param item_name_chunk_k: 提取上下文的切片数量限制
        :param item_name_chunk_size: LLM提示词模板参数长度的限制。
        :return: 用于提取产品名称的上下文参考内容。
        """
        #取前三个chunk
        chunks = chunks[0:item_name_chunk_k]
        final_context = []
        total = 0
        for index,chunk in enumerate(chunks):

            if not chunk and not isinstance(chunk,dict):
                continue

            chunk_content = chunk.get("content","")

            content = f"【切片-{index+1}】\n" + chunk_content
            total += len(content)
            if total < item_name_chunk_size:
                final_context.append(content)

        return "\n\n".join(final_context)

    def _recognition_item_name(self, file_title, item_name_recognition_context) -> str:
        """
        3.调用LLM,获取客户端AIClients，识别产品名称。 提示词模板；  降级：file_title
        :param file_title: 文件名称
        :param item_name_recognition_context: 识别产品名称的上下文参考内容
        :return: 产品名称
        """

        #获取LLM模型客户端对象
        llm_client = AIClients.get_llm_openai(False) # 返回是文本，不是json

        #用户提示词模板
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_recognition_context)

        #调用LLM模型
        llm_response = llm_client.invoke([
            SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        #处理LLM模型返回结果失败，降级处理
        if not llm_response or llm_response.content=="UNKNOWN":
            self.logger.error("LLM模型调用失败,降级处理,用文件名称作为产品名称")
            return file_title

        #获取响应内容：模型可能返回 JSON / 代码围栏 / 带引号字符串，统一剥成纯文本。
        # 不剥的话，{"item_name": "..."} 会被原样写进 Milvus 标量字段（BC-01）。
        item_name = normalize_item_name(llm_response.content)
        if not item_name:
            self.logger.error("LLM 返回空产品名，降级用文件名称")
            return file_title

        return item_name

    def _embedding_item_name(self, item_name) -> Tuple[List[float],Dict[int,float]]:
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error("BGE-M3模型客户端初始化失败", e)
            return None, None #降级处理

        try:
            embeddings = bge_m3_client.encode_documents([item_name])

            dense_vector = embeddings["dense"][0].tolist()  # List[float], 长度 1024
            sparse_matrix = embeddings["sparse"]  # CSR 稀疏矩阵
            start_idx = sparse_matrix.indptr[0] #  indptr = [0,5]
            end_idx = sparse_matrix.indptr[1]
            token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()  # indices = [11,22,33,44,55]
            weights = sparse_matrix.data[start_idx:end_idx].tolist() # indices = [111,222,333,444,555]
            sparse_vector = dict(zip(token_ids, weights, strict=False))  # Dict[int, float]

            # 返回 稠密向量，稀疏向量
            return dense_vector,sparse_vector
        except Exception as e:
            self.logger.error("BGE-M3向量化产品名称失败", e)
            return None,None

    def insert_milvus(self, file_title, item_name, dense_vector, sparse_vector):
        # 5.存储到Milvus   StorageClients获取客户端，创建集合，数据保存;  存储失败跳过，并记录日志。
        # 标量字段：pk, file_title ,item_name
        # 向量字段：dense_vector,sparse_vector

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error("Milvus 客户端初始化失败", e)
            return

        try:
            item_name_collection = self.config.item_name_collection
            if not milvus_client.has_collection(item_name_collection):
                self.logger.info(f"{item_name_collection}不存在,正在创建...")
                self._create_item_name_collection(item_name_collection,milvus_client)
                self.logger.info(f"{item_name_collection}集合创建成功")
        except Exception as e:
            self.logger.error(f"{item_name_collection}集合创建失败", e)

        try:
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            self.logger.info(f"正在保存数据...={data}")
            milvus_client.insert(item_name_collection,[data])
            self.logger.info("保存数据成功...")
        except Exception as e:
            self.logger.error("保存产品名称数据失败", e)

    def _create_item_name_collection(self, collection_name, milvus_client):
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,is_primary=True, auto_id=True, max_length=100) #主键
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)  # 标量字段
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535) # 标量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024) # 稠密向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR) #稀疏向量

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE") # 模糊近似查询   余弦相似度  只考虑方向，夹角，不考虑长度
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")  # IP内积  方向和长度都考虑 (查询更精准)    如果归一化后，与COSINE相似

        milvus_client.create_collection(collection_name=collection_name,
                                        schema=schema, index_params=index_param)
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name, state, chunks):
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name
        state["chunks"] = chunks


if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"E:\project\agent_project\shoper_brain\knowledge\processor\import_process\temp_dir\美的空调\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "美的空调_new",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    processed_state = node(state)

    # 5. 输出结果
    print(f"产品名: {processed_state.get('item_name')}")
    print(f"chunks数量: {len(processed_state.get('chunks', []))}")
    print(f"首5个chunk是否含item_name: {'item_name' in processed_state['chunks'][0]}")