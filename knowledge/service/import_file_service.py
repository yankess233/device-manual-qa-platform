import datetime
import logging
import os.path
import time
import uuid
from pathlib import Path

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.exceptions import FileProcessingError, MinioError
from knowledge.processor.import_process.main_graph import kb_import_graph
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.log_context import bind_task_id
from knowledge.utils.task_util import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_node_duration,
    add_running_task,
    update_task_status,
)
from knowledge.utils.upload_guard import (
    UploadRejected,
    ensure_allowed_suffix,
    safe_filename,
    save_upload_limited,
)

logger = logging.getLogger(__name__)


class ImportFileService:
    """业务层
        1.处理文件上传  双写（上传文件保存本地，上传文件保存minio）
        2.开启langgraph流程，完成文件导入milvus [7个步骤]
    """

    def run_langgraph_import(self,task_id, file_dir, import_file_path):
        """开启langgraph流程，完成文件导入milvus [7个步骤]"""
        bind_task_id(task_id)   # 后续这个任务里所有日志都会带上 task_id
        update_task_status(task_id,TASK_STATUS_PROCESSING)

        try:
            state = {
                "task_id": task_id,
                "import_file_path": import_file_path,
                "file_dir": file_dir
            }

            for event in kb_import_graph.stream(state):
                for name_node, _state in event.items():
                    print(f"{task_id} | 执行节点名称 -  {name_node}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception:
            update_task_status(task_id, TASK_STATUS_FAILED)

    def process_upload_file(self, file):
        """处理文件上传  双写
            1.创建任务id
            2.设置任务状态："upload_file"任务开始了
            3.文件写到本地磁盘
            4.文件上传至minio服务器
            5.设置任务状态："upload_file"任务结束了
            6.返回数据：task_id,file_dir,import_file_path
        """
        # 0.上传校验：文件名净化和后缀白名单在落盘前完成，避免路径穿越/任意文件写盘
        filename = safe_filename(file.filename)
        ensure_allowed_suffix(filename)

        # 1.创建任务id
        task_id = self._get_task_id()
        # D:\workspace\workspaceAI260528\shopkeeper_brain\knowledge\temp_data\20260720\ABCABCAB
        file_dir = os.path.join(self._get_path_with_date(), task_id)
        start = time.time()
        # 2.设置任务状态："upload_file"任务开始了
        add_running_task(task_id,"upload_file")
        # 3.文件写到本地磁盘
        import_file_path = self._save_upload_file_to_local(file, file_dir, filename)
        # 4.文件上传至minio服务器
        self._save_upload_file_to_minio(import_file_path, filename)
        # 5.设置任务状态："upload_file"任务结束了

        add_done_task(task_id,"upload_file")
        duration = time.time() -start
        add_node_duration(task_id, "upload_file",duration)

        # 6.返回数据：task_id,file_dir,import_file_path
        return task_id, file_dir, import_file_path

    def _get_task_id(self):
        return uuid.uuid4().hex[:8]

    def _get_path_with_date(self):
        # %Y   四位  "2026"
        # %y   两位  "26"
        # D:\workspace\workspaceAI260528\shopkeeper_brain\knowledge\temp_data\20260720
        return os.path.join(get_local_base_dir(), datetime.datetime.now().strftime("%Y%m%d"))

    def _save_upload_file_to_local(self, file: UploadFile, file_dir, filename: str):
        """写本地:保存到本地（边写边校验大小，超限删除半成品）"""
        # D:\workspace\workspaceAI260528\shopkeeper_brain\knowledge\temp_data\20260720\ABCABCAB\美的空调.pdf
        os.makedirs(file_dir, exist_ok=True)
        import_file_path = os.path.join(file_dir, filename)  # 文件名已净化，不含目录部分
        try:
            save_upload_limited(file.file, Path(import_file_path))
        except UploadRejected:
            raise
        except Exception as e:
            logger.error(f"上传文件到本地失败: {e}")
            raise FileProcessingError("上传文件到本地失败", e) from e

        return import_file_path

    def _save_upload_file_to_minio(self, import_file_path, filename):
        """写远程:上传minio"""
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"连接Minio失败: {e}")
            raise MinioError("连接Minio失败", e) from e
        try:
            MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "")
            object_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{filename}"
            minio_client.fput_object(MINIO_BUCKET_NAME, object_name, import_file_path)
        except Exception as e:
            logger.error(f"上传文件到Minio失败: {e}")
            raise MinioError("上传文件到Minio失败", e) from e
