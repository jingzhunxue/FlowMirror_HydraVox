from multiprocessing import Process
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import multiprocessing

from .worker import start_processing_tts
from .router import router
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

    # 挂载到app.state
    app.state.manager = manager
    app.state.tts_task_queue = tts_task_queue
    app.state.tts_result_dict = tts_result_dict
    app.state.processes_gpu = processes_gpu

    yield

    # shutdown逻辑
    for _ in range(8):
        tts_task_queue.put(None)
    
    tts_result_dict.clear()
    for p in processes_gpu:
        p.join()

    manager.shutdown()
    print("[Main] All worker processes have finished and manager is shut down.")


def create_app():
    import os
    app = FastAPI(root_path="/", lifespan=lifespan)
    
    # 服务配置
    app.state.num_workers_gpu = int(os.getenv('NUM_WORKERS_GPU', 1))
    
    # 注册所有路由
    app.include_router(router, tags=["tts"])

    return app


app = create_app()  # 保持默认参数的实例


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)