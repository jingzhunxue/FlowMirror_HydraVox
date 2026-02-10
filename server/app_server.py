from multiprocessing import Process
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import multiprocessing

from .worker import start_processing_tts
from .streaming_worker import start_streaming_worker
from .router import router
from .streaming_router import streaming_router
import dotenv

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # 关键：覆盖已有的 handlers
)

dotenv.load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):

    manager = multiprocessing.Manager()
    
    # 启动多进程
    (tts_task_queue, tts_result_dict, processes_gpu) = start_processing_tts(
        num_workers_gpu=app.state.num_workers_gpu,
        manager=manager
    )

    # 启动流式推理 Worker（GPU 从普通 Worker 之后开始分配）
    streaming_task_queue, streaming_processes = start_streaming_worker(
        num_workers=app.state.num_streaming_workers,
        manager=manager,
        gpu_offset=app.state.num_workers_gpu,
    )

    # 挂载到app.state
    app.state.manager = manager
    app.state.tts_task_queue = tts_task_queue
    app.state.tts_result_dict = tts_result_dict
    app.state.processes_gpu = processes_gpu
    app.state.streaming_task_queue = streaming_task_queue
    app.state.streaming_processes = streaming_processes

    yield

    # shutdown逻辑：非流式 Worker
    for _ in range(len(processes_gpu)):
        tts_task_queue.put(None)

    tts_result_dict.clear()
    for p in processes_gpu:
        p.join()

    # shutdown逻辑：流式 Worker
    for _ in range(len(streaming_processes)):
        streaming_task_queue.put(None)
    for p in streaming_processes:
        p.join()

    manager.shutdown()
    print("[Main] All worker processes have finished and manager is shut down.")


def create_app():
    import os
    app = FastAPI(root_path="/", lifespan=lifespan)
    
    # 服务配置
    app.state.num_workers_gpu = int(os.getenv('NUM_WORKERS_GPU', 1))
    app.state.num_streaming_workers = int(os.getenv('NUM_STREAMING_WORKERS', 1))

    # 注册所有路由
    app.include_router(router, tags=["tts"])
    app.include_router(streaming_router, tags=["streaming"])

    return app


app = create_app()  # 保持默认参数的实例


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)