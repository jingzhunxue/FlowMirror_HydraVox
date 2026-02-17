import uuid
import logging
import asyncio
from typing import Optional, Dict, Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from .model_utils.infer_speech_model import load_audio_from_base64
from .i18n import t

logger = logging.getLogger("streaming_router")

streaming_router = APIRouter(prefix="/api/v1")


class StreamingTTSRequest(BaseModel):
    """流式 TTS 请求"""
    text: str
    speaker_id: Optional[str] = None
    chunk_duration: int = 1  # 秒，必须是正整数
    first_chunk_duration: Optional[float] = None
    extra_params: Optional[Dict[str, Any]] = {
        "top_p": 0.9,
        "top_k": 10,
        "win_size": 24,
        "tau_r": 0.2,
        "speed": 1.0,
        "inference_head_num": 2,
    }

    @field_validator('chunk_duration')
    @classmethod
    def validate_chunk_duration(cls, v):
        if not isinstance(v, int) or v < 1:
            raise ValueError(t("chunk_duration 必须是正整数"))
        return v


class StreamingZeroShotRequest(BaseModel):
    """流式 Zero-Shot 请求"""
    tts_text: str
    prompt_text: str
    prompt_audio_base64: Optional[str] = None
    chunk_duration: int = 1
    first_chunk_duration: Optional[float] = None 
    extra_params: Optional[Dict[str, Any]] = {
        "top_p": 0.9,
        "top_k": 10,
        "win_size": 32,
        "tau_r": 0.2,
        "speed": 1.0,
        "inference_head_num": 2,
    }

    @field_validator('chunk_duration')
    @classmethod
    def validate_chunk_duration(cls, v):
        if not isinstance(v, int) or v < 1:
            raise ValueError(t("chunk_duration 必须是正整数"))
        return v


TOKENS_PER_SECOND = 25  # 25 Hz token rate


async def _stream_from_chunk_queue(chunk_queue, loop):
    """从 chunk_queue 异步读取并 yield PCM bytes。

    错误处理：StreamingResponse 开始后 HTTP 状态码已发送（200），
    无法再改为 500。遇到错误时记录日志并中断流，客户端通过检测
    流意外中断来判断错误。
    """
    while True:
        chunk = await loop.run_in_executor(None, chunk_queue.get)
        if chunk is None:
            break
        if isinstance(chunk, dict) and "error" in chunk:
            logger.error(t("流式推理错误: {error}", error=chunk["error"]))
            break
        yield chunk


@streaming_router.post("/tts/stream")
async def streaming_tts(request: Request, task: StreamingTTSRequest):
    """流式 TTS 合成端点"""
    try:
        if not task.text or not task.text.strip():
            raise HTTPException(status_code=400, detail=t("合成文本不能为空"))

        if not task.speaker_id:
            raise HTTPException(status_code=400, detail=t("说话人ID不能为空"))

        streaming_task_queue = request.app.state.streaming_task_queue
        manager = request.app.state.manager
        chunk_queue = manager.Queue()

        chunk_size = task.chunk_duration * TOKENS_PER_SECOND

        first_chunk_size = (
            max(1, int(task.first_chunk_duration * TOKENS_PER_SECOND))
            if task.first_chunk_duration is not None
            else chunk_size
        )

        streaming_task_queue.put({
            "id": str(uuid.uuid4()),
            "task_type": "streaming_tts",
            "text": task.text,
            "speaker_id": task.speaker_id,
            "chunk_size": chunk_size,
            "first_chunk_size": first_chunk_size,
            "chunk_queue": chunk_queue,
            "extra_params": task.extra_params,
        })

        loop = asyncio.get_running_loop()

        return StreamingResponse(
            _stream_from_chunk_queue(chunk_queue, loop),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": "24000",
                "X-Channels": "1",
                "X-Sample-Width": "2",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(t("流式TTS请求失败: {error}", error=e))
        raise HTTPException(status_code=500, detail=str(e))


@streaming_router.post("/zero-shot/stream")
async def streaming_zero_shot(request: Request, task: StreamingZeroShotRequest):
    """流式 Zero-Shot 合成端点"""
    try:
        if not task.tts_text or not task.tts_text.strip():
            raise HTTPException(status_code=400, detail=t("合成文本不能为空"))

        if not task.prompt_text or not task.prompt_text.strip():
            raise HTTPException(status_code=400, detail=t("提示文本不能为空"))

        if not task.prompt_audio_base64:
            raise HTTPException(status_code=400, detail=t("提示音频不能为空"))

        prompt_audio, prompt_sample_rate = load_audio_from_base64(task.prompt_audio_base64)

        streaming_task_queue = request.app.state.streaming_task_queue
        manager = request.app.state.manager
        chunk_queue = manager.Queue()

        chunk_size = task.chunk_duration * TOKENS_PER_SECOND

        first_chunk_size = (
            max(1, int(task.first_chunk_duration * TOKENS_PER_SECOND))
            if task.first_chunk_duration is not None
            else chunk_size
        )

        streaming_task_queue.put({
            "id": str(uuid.uuid4()),
            "task_type": "streaming_zero_shot",
            "tts_text": task.tts_text,
            "prompt_text": task.prompt_text,
            "prompt_audio": prompt_audio,
            "prompt_sample_rate": prompt_sample_rate,
            "chunk_size": chunk_size,
            "first_chunk_size": first_chunk_size,
            "chunk_queue": chunk_queue,
            "extra_params": task.extra_params,
        })

        loop = asyncio.get_running_loop()

        return StreamingResponse(
            _stream_from_chunk_queue(chunk_queue, loop),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": "24000",
                "X-Channels": "1",
                "X-Sample-Width": "2",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(t("流式ZeroShot请求失败: {error}", error=e))
        raise HTTPException(status_code=500, detail=str(e))
