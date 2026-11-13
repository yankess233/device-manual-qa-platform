import asyncio
import os
from typing import Union

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from knowledge.api.security import allowed_origins, require_api_key
from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator


def register_router(app):
    @app.get("/chat")
    async def chat_page():
        """返回聊天对话网页"""
        return FileResponse(os.path.join(get_front_page_dir(), "chat.html"))

    # 处理客户端问题的请求
    @app.post("/query", response_model=Union[QueryResponse, StreamSubmitResponse],
              dependencies=[Depends(require_api_key)])
    async def query(request: QueryRequest, background_tasks: BackgroundTasks,
                    service: QueryService = Depends(get_query_service)):
        # 获取接收客户端传递过来的数据
        session_id = request.session_id or service.create_session_id()
        is_stream = request.is_stream
        query = request.query
        task_id = service.create_task_id()

        # 1.用户勾选流式输出
        if is_stream:
            # 2.创建sse queue 队列
            create_sse_queue(task_id)

            # 3.开启异步任务，跑流程
            background_tasks.add_task(service.run_query_graph, task_id, session_id, query, is_stream)

            # 4.启动了查询流程，返回通知。
            return StreamSubmitResponse(
                message="提交查询请求",
                session_id=session_id,
                task_id=task_id)
        else:
            # 5.非流式输出
            loop = asyncio.get_running_loop()
            # None 表示使用默认线程池： ThreadPoolExecutor
            await loop.run_in_executor(None, service.run_query_graph,task_id, session_id, query, is_stream)

            #await service.run_query_graph(task_id, session_id, query, is_stream)

            answer = service.get_answer(task_id)

            # 3.返回结果
            return QueryResponse(
                message="查询成功",
                session_id=session_id,
                answer=answer,
                task_id=task_id)

    # 流式输出
    @app.get("/stream/{task_id}", dependencies=[Depends(require_api_key)])
    async def stream(task_id: str,request:Request) -> StreamingResponse:
        return StreamingResponse(
            sse_generator(task_id,request),
            media_type="text/event-stream"
        )


    @app.get("/history/{session_id}", dependencies=[Depends(require_api_key)])
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}") from e

    @app.delete("/history/{session_id}", dependencies=[Depends(require_api_key)])
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}




def create_app():
    setup_logging()

    app = FastAPI(description="知识库检索", version="v1.0")

    # 跨域设置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),  # 白名单来自 ALLOWED_ORIGINS，默认只放行本机页面
        allow_credentials=False,  # 对请求cookie或Authorization头进行限制.  如果设置True，其他三个不能使用*
        allow_methods=["*"],  # 对请求方法限制    get  post  put  delete
        allow_headers=["*"],  # 对请求头限制
    )

    # 挂载前端静态资源，  image,css
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app


if __name__ == "__main__":
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001, log_level="info")
