import datetime
import logging
import os.path
import shutil
import time
import uuid

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.exceptions import FileProcessingError, MinioError
from knowledge.processor.import_process.main_graph import kb_import_graph
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import update_task_status, add_running_task, add_done_task, TASK_STATUS_PROCESSING, \
    TASK_STATUS_FAILED, TASK_STATUS_COMPLETED, add_node_duration

logger = logging.getLogger(__name__)


class ImportFileService:
    """业务层
        1.处理文件上传  双写（上传文件保存本地，上传文件保存minio）
        2.开启langgraph流程，完成文件导入milvus [7个步骤]
    """

    def run_langgraph_import(self,task_id, file_dir, import_file_path):
        """开启langgraph流程，完成文件导入milvus [7个步骤]"""
        update_task_status(task_id,TASK_STATUS_PROCESSING)

        try:
            state = {
                "task_id": task_id,
                "import_file_path": import_file_path,
                "file_dir": file_dir
            }

            final_state = None
            for event in kb_import_graph.stream(state):
                for name_node, state in event.items():
                    print(f"{task_id} | 执行节点名称 -  {name_node}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
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
        # 1.创建任务id
        task_id = self._get_task_id()
        # D:\workspace\workspaceAI260528\shopkeeper_brain\knowledge\temp_data\20260720\ABCABCAB
        file_dir = os.path.join(self._get_path_with_date(), task_id)
        start = time.time()
        # 2.设置任务状态："upload_file"任务开始了
        add_running_task(task_id,"upload_file")
        # 3.文件写到本地磁盘
        import_file_path = self._save_upload_file_to_local(file, file_dir)
        # 4.文件上传至minio服务器
        self._save_upload_file_to_minio(import_file_path, file.filename)
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

    def _save_upload_file_to_local(self, file: UploadFile, file_dir):
        """写本地:保存到本地"""
        # D:\workspace\workspaceAI260528\shopkeeper_brain\knowledge\temp_data\20260720\ABCABCAB\万用表RS-12的使用.pdf
        os.makedirs(file_dir,exist_ok=True)
        import_file_path = os.path.join(file_dir, file.filename)  # 上传后文件完整路径
        try:
            with open(import_file_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)  # 1Mb缓存写操作，效率高。
        except Exception as e:
            logger.error(f"上传文件到本地失败: {e}")
            raise FileProcessingError(f"上传文件到本地失败", e)

        return import_file_path

    def _save_upload_file_to_minio(self, import_file_path, filename):
        """写远程:上传minio"""
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"连接Minio失败: {e}")
            raise MinioError("连接Minio失败", e)
        try:
            MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "")
            object_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{filename}"
            minio_client.fput_object(MINIO_BUCKET_NAME, object_name, import_file_path)
        except Exception as e:
            logger.error(f"上传文件到Minio失败: {e}")
            raise MinioError("上传文件到Minio失败", e)
