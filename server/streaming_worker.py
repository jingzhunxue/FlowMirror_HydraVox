import multiprocessing
import logging
import os
import argparse
import traceback

from functools import partial
from .model_utils.cosyvoice.utils.common import ras_sampling
from .model_utils.infer_speech_model import (
    ModelManager,
    streaming_inference_tts,
    streaming_inference_zero_shot,
)

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def streaming_worker_process(
    num_workers_gpu,
    task_queue,
    worker_id: int,
    gpu_offset: int = 0,
):
    """流式推理 Worker 进程主函数。

    从 task_queue 获取任务，每个任务包含一个 per-request 的 chunk_queue。
    推理过程中逐 chunk 将 PCM bytes 放入 chunk_queue，
    完成后放入 None 作为结束信号，异常时放入 {"error": msg}。

    Args:
        gpu_offset: GPU 编号偏移量，用于避免与普通 Worker 共享 GPU。
                    例如普通 Worker 占 GPU 0..N-1，流式 Worker 从 N 开始。
    """
    gpu_id = (gpu_offset + worker_id) % max(1, num_workers_gpu + gpu_offset)
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu_id}"

    logger.info(f"[Streaming Worker-{worker_id}] Starting up, loading model...")
    model_manager = ModelManager()

    args_parse = argparse.Namespace(
        config=os.getenv("TTS_CONFIG"),
        model_dir=os.getenv("TTS_MODEL_DIR"),
        bf16=_env_flag("TTS_BF_16"),
        fp16=_env_flag("TTS_FP_16"),
        cpu=_env_flag("TTS_CPU", False),
    )

    model_manager.load_models(args_parse)
    logger.info(f"[Streaming Worker-{worker_id}] Model loaded!")

    try:
        from fmtn import create_default_tn
        tn = create_default_tn(verbose=True)
    except Exception:
        raise ValueError("Failed to load text normalization library")

    while True:
        task = task_queue.get()
        if task is None:
            break

        chunk_queue = task['chunk_queue']
        task_type = task['task_type']
        speed = 1.0

        # 应用 extra_params
        if "extra_params" in task and task["extra_params"]:
            ep = task["extra_params"]
            model_manager.models['llm'].sampling = partial(
                ras_sampling,
                top_p=ep.get('top_p', 0.9),
                top_k=ep.get('top_k', 10),
                win_size=ep.get('win_size', 32),
                tau_r=ep.get('tau_r', 0.2),
            )
            model_manager.models['llm'].inference_head_num = ep.get('inference_head_num', 2)
            speed = float(ep.get('speed', 1.0))

        chunk_size = task.get('chunk_size', 25)

        try:
            if task_type == 'streaming_tts':
                text = tn.process_text(task['text'])
                for pcm_bytes in streaming_inference_tts(
                    model_manager, text, task['speaker_id'],
                    chunk_size=chunk_size, speed=speed,
                ):
                    chunk_queue.put(pcm_bytes)

            elif task_type == 'streaming_zero_shot':
                tts_text = tn.process_text(task['tts_text'])
                prompt_text = tn.process_text(task.get('prompt_text', ''))
                for pcm_bytes in streaming_inference_zero_shot(
                    model_manager, tts_text, prompt_text,
                    task['prompt_audio'], task['prompt_sample_rate'],
                    chunk_size=chunk_size, speed=speed,
                ):
                    chunk_queue.put(pcm_bytes)

            # 结束信号
            chunk_queue.put(None)

        except Exception as e:
            logger.error(f"[Streaming Worker-{worker_id}] Error: {e}")
            traceback.print_exc()
            chunk_queue.put({"error": str(e)})
            chunk_queue.put(None)


def start_streaming_worker(num_workers: int = 1, manager=None, gpu_offset: int = 0):
    """启动流式推理 Worker 进程。

    Args:
        num_workers: 流式 Worker 进程数量
        manager: multiprocessing.Manager 实例
        gpu_offset: GPU 编号偏移量，避免与普通 Worker 共享 GPU

    Returns:
        (streaming_task_queue, processes) 元组
    """
    if manager is None:
        raise ValueError("Manager is required (multiprocessing.Manager())")

    streaming_task_queue = manager.Queue()
    processes = []

    for i in range(num_workers):
        if os.environ.get("ENABLE_STREAMING", "true").lower() == "true":
            p = multiprocessing.Process(
                target=streaming_worker_process,
                args=(num_workers, streaming_task_queue, i, gpu_offset),
            )
            processes.append(p)

    for p in processes:
        p.start()

    return streaming_task_queue, processes
