from typing import Dict, List

from knowledge.utils.task_store import get_task_store

"""
任务id: 主要追踪上传文件（任务）的状态流程的
上传一个文件就属于一个任务（唯一的任务id）---未来如果想知道上传的这个文件处理到哪里了就需要用任务id来查询

状态存取委托给 `task_store`：
- 默认内存实现（单进程，行为与旧版一致）
- `TASK_BACKEND=redis` 时切到 Redis，多 worker 共享、重启不丢
原来这里是 4 个模块级 dict，加 `--workers N` 后 /status 会落到另一个进程查不到任务。
"""

TASK_STATUS_PROCESSING = "processing"  # 任务处理中
TASK_STATUS_COMPLETED = "completed"  # 任务完成
TASK_STATUS_FAILED = "failed"  # 任务失败

_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "上传文件",
    "entry_node": "检查文件",
    "pdf_to_md_node": "PDF转Markdown",
    "md_to_img_node": "Markdown图片处理",
    "document_split_node": "文档切分",
    "item_name_recognition_node": "主体名称识别",
    "embedding_chunk_node": "向量生成",
    "import_milvus_node": "导入向量数据库",
    "__end__": "处理完成",
    # --- Query 流程节点（kb/query_process/main_graph.py）---
    "item_name_confirm_node": "确认问题产品",
    "answer_output_node": "生成答案",
    "rerank_node": "重排序",
    "rrf_node": "倒排融合",
    "web_search_mcp_node": "网络搜索",
    "vector_search_node": "切片搜索",
    "hyde_search_node": "切片搜索(假设性文档)"
}


def _to_cn(node_name: str) -> str:
    # 1. 从节点映射字典中获取中文名，若未配置则直接返回原英文名
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    get_task_store().add_running(task_id, node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    get_task_store().add_done(task_id, node_name)


def get_running_task_list(task_id: str) -> List[str]:
    # 1. 获取指定任务运行中的节点列表，并通过列表推导式统一转换为中文展示名返回
    return [_to_cn(n) for n in get_task_store().snapshot(task_id)["running"]]


def get_done_task_list(task_id: str) -> List[str]:
    # 1. 获取指定任务已完成的节点列表，并通过列表推导式统一转换为中文展示名返回
    return [_to_cn(n) for n in get_task_store().snapshot(task_id)["done"]]


def get_task_status(task_id: str) -> str:
    """
    根据任务ID 获取任务状态
    :param task_id:
    :return:
    """
    # 1. 安全获取指定任务的总体运行状态，若不存在则返回空字符串
    return get_task_store().get_status(task_id)


def update_task_status(task_id: str, status_name: str) -> None:
    # 1. 更新指定任务的总体运行状态（如 processing 等）
    get_task_store().set_status(task_id, status_name)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """
    存储任务结果字段（如 answer / error）。
    """
    get_task_store().set_result(task_id, key, value)


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """
    获取任务结果字段（如 answer / error）。
    """
    return get_task_store().get_result(task_id, key, default)



def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    """记录节点耗时（秒）"""
    cn_name = _to_cn(node_name)
    get_task_store().add_duration(task_id, cn_name, duration)

def get_node_durations(task_id: str) -> Dict[str, float]:
    """获取所有节点的耗时"""
    return get_task_store().snapshot(task_id)["durations"]

def get_task_info(task_id: str) -> Dict[str, any]:
    """
    获取任务的全局信息（状态 + 运行中节点 + 已完成节点）
    :param task_id: 任务ID
    :return: 包含 status、running_list、done_list 的字典
    """
    return {
        "status": get_task_status(task_id),
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id),
        "durations": get_node_durations(task_id)
    }

