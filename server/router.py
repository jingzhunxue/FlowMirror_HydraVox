from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import logging
import time
import asyncio
from .model_utils.infer_speech_model import load_audio_from_base64, audio_to_base64, load_audio_from_file, get_speakers

logger = logging.getLogger("router")

router = APIRouter(prefix="/api/v1")

class ZeroShotRequest(BaseModel):
    """零样本语音合成请求"""
    tts_text: str  # 要合成的文本
    prompt_text: str  # 提示文本
    prompt_audio_base64: Optional[str] = None  # base64编码的提示音频
    output_format: str = "wav"  # 输出格式
    extra_params: Optional[Dict[str, Any]] = {
        "top_p": 0.9,
        "top_k": 10,
        "win_size": 32,
        "tau_r": 0.2,
        "inference_head_num":2
    }

class TTSRequest(BaseModel):
    """文本到语音合成请求"""
    text: str  # 要合成的文本
    speaker_id: Optional[str] = None  # 说话人ID
    output_format: str = "wav"  # 输出格式
    last_prompt: bool = True  # 是否使用上一段音频作为zero shot提示
    extra_params: Optional[Dict[str, Any]] = {
        "top_p": 0.9,
        "top_k": 10,
        "win_size": 32,
        "tau_r": 0.2,
        "inference_head_num":2
    }

class RAGRequest(BaseModel):
    """RAG增强语音合成请求（预留）"""
    query: str  # 查询文本
    context: Optional[str] = None  # 上下文
    speaker_config: Optional[Dict[str, Any]] = None  # 说话人配置
    output_format: str = "wav"  # 输出格式
    extra_params: Optional[Dict[str, Any]] = {
        "top_p": 0.9,
        "top_k": 10,
        "win_size": 32,
        "tau_r": 0.2,
        "inference_head_num":2
    }

class APIResponse(BaseModel):
    """API响应基类"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class LoadPTRequest(BaseModel):
    """加载模型权重请求"""
    llm_pt: str  # LLM权重路径
    flow_pt: str  # Flow权重路径

@router.post("/zero-shot", response_model=APIResponse)
async def zero_shot_tts(request: Request, task: ZeroShotRequest):
    """
    零样本语音合成接口
    
    Args:
        request: 零样本合成请求
        
    Returns:
        包含合成音频的响应
    """
    try:
        tts_task_queue = request.app.state.tts_task_queue
        tts_result_dict = request.app.state.tts_result_dict

        logger.info(f"零样本合成请求: {task.tts_text[:50]}...")
        
        # 验证输入
        if not task.tts_text or not task.tts_text.strip():
            raise HTTPException(status_code=400, detail="合成文本不能为空")
        
        if not task.prompt_text or not task.prompt_text.strip():
            raise HTTPException(status_code=400, detail="提示文本不能为空")
        
        if not task.prompt_audio_base64:
            raise HTTPException(status_code=400, detail="提示音频不能为空")
        
        # 加载提示音频
        prompt_audio, prompt_sample_rate = load_audio_from_base64(task.prompt_audio_base64)
        
        task_id = uuid.uuid4()

        tasks = {
            "id": task_id,
            "tts_text": task.tts_text,
            "task_type": "zero_shot",
            "prompt_text": task.prompt_text,
            "prompt_audio": prompt_audio,
            "prompt_sample_rate": prompt_sample_rate,
            "output_format": task.output_format,
            "result_dict": tts_result_dict,
            "extra_params": task.extra_params
        }

        tts_task_queue.put(tasks)

        while True:
            # 如果 Worker 已经把结果写入 result_dict
            if task_id in tts_result_dict:
                # result_img, result_box = layout_result_dict.pop(task_id)  # pop 可避免字典越积越大
                result_dict = tts_result_dict.pop(task_id)  # pop 可避免字典越积越大
                
                # 检查是否有错误
                if error_info := result_dict.get("error"):
                    logger.warning(f"Task {task_id} completed with error: {error_info}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=error_info
                    )

                result_audio = result_dict['output_audio']
                result_sample_rate = result_dict['sample_rate']

                # 转换为base64
                audio_base64 = audio_to_base64(
                    result_audio, 
                    result_sample_rate,
                    task.output_format
                )

                return APIResponse(
                    success=True,
                    message="TTS合成成功",
                    data={
                        "audio_base64": audio_base64,
                        "sample_rate": result_sample_rate,
                        "format": task.output_format,
                        "duration": result_audio.shape[-1] / result_sample_rate,
                        "segments_info": {}
                    }
                )

            # 避免 CPU 忙等，稍微 sleep
            await asyncio.sleep(0.05)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"零样本合成失败: {e}")
        return APIResponse(
            success=False,
            message="零样本合成失败",
            error=str(e)
        )

@router.post("/tts", response_model=APIResponse)
async def text_to_speech(request: Request, task: TTSRequest):
    """
    文本到语音合成接口
    
    Args:
        request: TTS合成请求
        
    Returns:
        包含合成音频的响应
    """
    try:
        tts_task_queue = request.app.state.tts_task_queue
        tts_result_dict = request.app.state.tts_result_dict

        logger.info(f"TTS合成请求: {task.text[:50]}...")
        
        # 验证输入
        if not task.text or not task.text.strip():
            raise HTTPException(status_code=400, detail="合成文本不能为空")
        
        if not task.speaker_id:
            raise HTTPException(status_code=400, detail="说话人ID不能为空")
        
        task_id = uuid.uuid4()

        tasks = {
            "id": task_id,
            "task_type": "tts",
            "text": task.text,
            "speaker_id": task.speaker_id,
            "output_format": task.output_format,
            "last_prompt": task.last_prompt,
            "result_dict": tts_result_dict,
            "extra_params": task.extra_params
        }

        tts_task_queue.put(tasks)

        start_time = time.time()
        timeout = 10
        while True:
            # 如果 Worker 已经把结果写入 result_dict
            if task_id in tts_result_dict:
                # result_img, result_box = layout_result_dict.pop(task_id)  # pop 可避免字典越积越大
                result_dict = tts_result_dict.pop(task_id)  # pop 可避免字典越积越
                
                # 检查是否有错误
                if error_info := result_dict.get("error"):
                    logger.warning(f"Task {task_id} completed with error: {error_info}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=error_info
                    )

                result_audio = result_dict['output_audio']
                result_sample_rate = result_dict['sample_rate']

                # 转换为base64
                audio_base64 = audio_to_base64(
                    result_audio, 
                    result_sample_rate,
                    task.output_format
                )

                return APIResponse(
                    success=True,
                    message="TTS合成成功",
                    data={
                        "audio_base64": audio_base64,
                        "sample_rate": result_sample_rate,
                        "format": task.output_format,
                        "duration": result_audio.shape[-1] / result_sample_rate,
                        "speaker_id": task.speaker_id,
                        "segments_info": {}
                    }
                )

            # 检查超时
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Task {task_id} timed out after {timeout} seconds.")
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail=f"Task {task_id} timed out after {timeout} seconds."
                )

            # 避免 CPU 忙等，稍微 sleep
            await asyncio.sleep(0.05)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS合成失败: {e}")
        return APIResponse(
            success=False,
            message="TTS合成失败",
            error=str(e)
        )

@router.post("/load_pt")
async def load_pt(request: Request, task: LoadPTRequest):
    """
    加载模型权重
    """
    try:
        tts_task_queue = request.app.state.tts_task_queue
        tts_result_dict = request.app.state.tts_result_dict
    
        logger.info(f"加载模型权重: {task.llm_pt} {task.flow_pt}")

        task_id = uuid.uuid4()
        tasks = {
            "id": task_id,
            "task_type": "load_pt",
            "llm_pt": task.llm_pt,
            "flow_pt": task.flow_pt
        }
        tts_task_queue.put(tasks)

        while True:
            if task_id in tts_result_dict:
                result_dict = tts_result_dict.pop(task_id)
                if result_dict['status'] == "success":
                    return APIResponse(success=True, message="加载模型权重成功", data=result_dict)
                else:
                    return APIResponse(success=False, message="加载模型权重失败", error=result_dict['error'])

            await asyncio.sleep(0.05)

    except Exception as e:
        logger.error(f"加载模型权重失败: {e}")
        return APIResponse(success=False, message="加载模型权重失败", error=str(e))

@router.get("/speakers")
async def get_speakers_api(request: Request):
    """
    获取说话人列表
    """
    try:
        return get_speakers()
    except Exception as e:
        logger.error(f"获取说话人列表失败: {e}")
        return APIResponse(success=False, message="获取说话人列表失败", error=str(e))