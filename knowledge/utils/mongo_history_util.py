import logging
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from pymongo.collection import Collection
from pymongo import DESCENDING

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def _get_collection() -> Collection:
    """获取 chat_message 集合"""
    return StorageClients.get_mongo_db()["chat_message"]  # "chat_message"表名


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        message_id: str = None,
) -> str:
    """
    MongoDB的写入操作
    新增(message_id如果为空) or  修改（message_id不为空）
    Args:
        session_id:
        role:
        text:
        rewritten_query:
        item_names:
        message_id:

    Returns:

    """
    ts = datetime.now().timestamp()

    # 1. 构建记录结构
    document = {
        "session_id": session_id,  # 会话id
        "role": role,  # 角色
        "text": text,  # 内容
        "rewritten_query": rewritten_query,  # 重写后问题
        "item_names": item_names or [],  # 商品名列表
        "ts": ts,  # 时间戳
    }

    # 2. 获取集合[客户端 db collection]
    collection = _get_collection()
    if message_id:
        collection.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id
    else:
        result = collection.insert_one(document)
        return str(result.inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        logger.error(f"Error getting recent messages: {e}")
        return []


def clear_history(session_id: str) -> int:
    try:
        result = _get_collection().delete_many({"session_id": session_id})
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error clearing history for session {session_id}: {e}")
        return 0

def update_message_item_names(ids_to_update, confirmed):
    """把已确认的商品名回填到历史消息

    :param ids_to_update: 需要回填的消息 _id 列表（ObjectId 或 16 进制字符串）
    :param confirmed: 已确认的商品名列表
    :return: 实际修改的文档数
    """
    if not ids_to_update or not confirmed:
        return 0

    # 1.统一转成 ObjectId，非法 id 跳过
    object_ids = []
    for _id in ids_to_update:
        try:
            object_ids.append(ObjectId(str(_id)))
        except Exception:
            logger.warning(f"历史回填跳过非法消息 id: {_id}")
            continue

    if not object_ids:
        return 0

    # 2.批量更新
    try:
        result = _get_collection().update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"item_names": list(confirmed)}},
        )
        logger.info(f"历史回填商品名: {result.modified_count} 条 -> {confirmed}")
        return result.modified_count
    except Exception as e:
        logger.error(f"Error updating item_names: {e}")
        return 0