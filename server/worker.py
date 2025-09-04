import multiprocessing
import logging
import numpy as n
import io
import traceback

from functools import partial
from .model_utils.cosyvoice.utils.common import ras_sampling
from .model_utils.infer_speech_model import ModelManager, text_to_speech, inference_zero_shot
import os
import argparse

logger = logging.getLogger("uvicorn")

logger.setLevel(logging.INFO)

def worker_process_tts(
    num_workers_gpu,
    task_queue,       # multiprocessing.Manager().Queue()
    result_dict,      # multiprocessing.Manager().dict()
    worker_id: int):

    os.environ['CUDA_VISIBLE_DEVICES'] = f"{worker_id % num_workers_gpu}"

    logger.info(f"[TTS Worker-{worker_id}] Starting up, loading model...")
    model_manager = ModelManager()
    
    args_parse = argparse.Namespace(
        config=os.getenv("TTS_CONFIG"),
        model_dir=os.getenv("TTS_MODEL_DIR"),
        bf16=bool(os.getenv("TTS_BF_16")),
        fp16=bool(os.getenv("TTS_FP_16")),
        cpu=bool(os.getenv("TTS_CPU", False)),
    )

    model_manager.load_models(args_parse)

    logger.info(f"[TTS Worker-{worker_id}] Model loaded!")

    from scripts.utils.FlowMirrorTN.lazy_import import LazyImport
    tn = LazyImport()

    while True:
        task = task_queue.get()
        if "extra_params" in task:
            model_manager.models['llm'].sampling = partial(ras_sampling,
                top_p=task['extra_params']['top_p'],
                top_k=task['extra_params']['top_k'],
                win_size=task['extra_params']['win_size'],
                tau_r=task['extra_params']['tau_r']
            )
            model_manager.models['llm'].inference_head_num = task['extra_params']['inference_head_num']
            
        if task is None:
            break

        task_type = task['task_type']
        try:
            if task_type == 'zero_shot':
                # 直接调用inference_zero_shot并构建返回字典
                task['tts_text'] = tn.process_text(task['tts_text'])
                if 'prompt_text' in task:
                    task['prompt_text'] = tn.process_text(task['prompt_text'])
                else:
                    task['prompt_text'] = ''
                output_audio = inference_zero_shot(
                    model_manager, 
                    task['tts_text'], 
                    task['prompt_text'], 
                    task['prompt_audio'], 
                    task['prompt_sample_rate']
                )
                result = {
                    "output_audio": output_audio,
                    "sample_rate": model_manager.configs['sample_rate'],
                    "format": task.get('output_format', "wav"),
                    "duration": output_audio.shape[-1] / model_manager.configs['sample_rate']
                }
            elif task_type == 'tts':
                task['text'] = tn.process_text(task['text'])
                result = text_to_speech(model_manager, task['text'], task['speaker_id'])
            elif task_type == 'load_pt':
                result = model_manager.load_pt(task['llm_pt'], task['flow_pt'])
        except Exception as e:
            logger.error(f"[TTS Worker-{worker_id}] Error: {e}")
            result = {"error": str(e)}

        result_dict[task['id']] = result

def start_processing_tts(num_workers_gpu: int = 8, manager=None):
    """
    初始化所有处理进程
    
    Args:
        num_workers_gpu: GPU工作进程数量
        manager: multiprocessing.Manager实例
    """
    if manager is None:
        raise ValueError("Manager is required (multiprocessing.Manager())")

    # 初始化队列和字典
    tts_task_queue = manager.Queue()
    tts_result_dict = manager.dict()

    processes_gpu = []

    # GPU workers (Segment + Layout)
    for i in range(num_workers_gpu):
        if os.environ.get("ENABLE_TTS", "true").lower() == "true":
            processes_gpu.append(multiprocessing.Process(
                target=worker_process_tts, 
                args=(num_workers_gpu, tts_task_queue, tts_result_dict, i)
            ))

    # 启动所有进程
    for p in processes_gpu:
        p.start()

    return (
        tts_task_queue,          # 0. TTS任务队列
        tts_result_dict,      # 1. TTS结果字典
        processes_gpu,           # 7. GPU进程列表
    )