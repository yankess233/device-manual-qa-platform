import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.service.import_file_service import ImportFileService
from knowledge.utils.task_util import get_task_info


def register_router(app):

    @app.get("/import")
    async def import_page():
        """返回导入网页"""
        return FileResponse(os.path.join(get_front_page_dir(), "import.html"))

    # 文件上传处理必须是post请求
    # FastAPI只管与前端进行接口对接，并不负责业务逻辑处理。需要定义专门业务层类：处理文件上传，开启langgraph流程等。
    # Depends(get_import_file_service)  注意： 函数名称不加小括号，表示传递函数引用，而不直接调用。延迟加载策略。
    # 无默认值参数不能放在有默认值后面。定义在前面位置即可。
    # 利用pydantic进行数据验证，确保数据符合要求
    @app.post("/upload",response_model=UploadResponse)
    async def upload_file_endpoint(background_tasks: BackgroundTasks,
                                   service:ImportFileService=Depends(get_import_file_service),
                                   file:UploadFile=File(...)):
        """上传文件接口"""

        #1.文件上传，双写
        # 任务ID; 文件存储目录; 上传后文件完整路径
        task_id,file_dir,import_file_path =  service.process_upload_file(file)

        # 同步(串行化) or 异步(开启单独线程处理请求，不会对当前主线程阻塞。异步处理流程)
        #2.开启langgraph流程
        background_tasks.add_task(service.run_langgraph_import, task_id, file_dir, import_file_path)

        return UploadResponse(task_id=task_id,message="文件上传成功,请等待导入流程完成")

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_statu(task_id:str):
        """获取任务状态接口  每间隔1.5秒执行一次"""
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)


def create_app():
    setup_logging()

    app = FastAPI(description="知识库导入",version="v1.0")

    # 跨域设置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 对客户端访问来源限制
        allow_credentials=False,  # 对请求cookie或Authorization头进行限制.  如果设置True，其他三个不能使用*
        allow_methods=["*"], #对请求方法限制    get  post  put  delete
        allow_headers=["*"], #对请求头限制
    )

    # 挂载前端静态资源，  image,css
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app


if __name__ == "__main__":

    uvicorn.run(app=create_app(),host="0.0.0.0",port=8000,log_level="info")