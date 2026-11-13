import logging
import uuid
from typing import Any, Dict, List

from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.log_context import bind_task_id
from knowledge.utils.task_util import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_task_result,
    update_task_status,
)

logger = logging.getLogger(__name__)


class QueryService:

    def run_query_graph(self, task_id, session_id, query, is_stream):

        bind_task_id(task_id)   # 后续这个任务里所有日志都会带上 task_id
        try:
            update_task_status(task_id=task_id, status_name=TASK_STATUS_PROCESSING)
            state = {
                "original_query": query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
            }
            # 只能看到最后结果
            query_app.invoke(state)

            update_task_status(task_id=task_id, status_name=TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"启动查询流程失败：{e}")
            update_task_status(task_id=task_id, status_name=TASK_STATUS_FAILED)

    def create_task_id(self):
        return str(uuid.uuid4())

    def create_session_id(self):
        return str(uuid.uuid4())

    def get_answer(self, task_id: str) -> str:
        return get_task_result(task_id, "answer", "")

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        from knowledge.utils.mongo_history_util import get_recent_messages
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        from knowledge.utils.mongo_history_util import clear_history
        return clear_history(session_id)