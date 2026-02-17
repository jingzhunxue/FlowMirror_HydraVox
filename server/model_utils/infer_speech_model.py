# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import base64
import io
from hyperpyyaml import load_hyperpyyaml
from ..i18n import t

# 导入项目模块
try:
    from .cosyvoice.cli.frontend import CosyVoiceFrontEnd
except ImportError as e:
    print(t("导入错误: {error}", error=e))
    print(t("当前工作目录: {cwd}", cwd=os.getcwd()))
    print(t("Python路径: {path}", path=sys.path))
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # 关键：覆盖已有的 handlers
)

logger = logging.getLogger(__name__)

# ============================================================================
# 全局变量和模型管理
# ============================================================================

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.models = None
        self.frontend = None
        self.configs = None
        self.device = None
        self.is_loaded = False
        
    def load_models(self, args):
        """加载模型"""
        if self.is_loaded:
            logger.info(t("模型已加载，跳过重复加载"))
            return
            
        logger.info(t("开始加载模型..."))
        
        # 加载配置
        with open(os.path.join(args.model_dir, 'hydravox.yaml'), 'r') as f:
            self.configs = load_hyperpyyaml(f, overrides={
                'qwen_pretrain_path': os.path.join(args.model_dir, 'CosyVoice-BlankEN')
            })
        
        # 初始化模型
        llm = self.configs['llm']
        flow = self.configs['flow']
        hift = self.configs['hift']
        
        # 加载检查点
        logger.info(t("加载LLM检查点..."))
        llm_pt = torch.load(os.path.join(args.model_dir, 'llm.pt'), map_location='cpu')
        
        logger.info(t("加载Flow检查点..."))
        flow_pt = torch.load(os.path.join(args.model_dir, 'flow.pt'), map_location='cpu')
        
        logger.info(t("加载Hift检查点..."))
        hift_pt = torch.load(os.path.join(args.model_dir, 'hift.pt'), map_location='cpu')
        
            # 清理检查点中的epoch和step信息
        for key in ['epoch', 'step', '_original_metadata', '_conversion_info']:
            if key in llm_pt:
                llm_pt.pop(key)
                logger.info(t("从LLM检查点中移除 '{key}' 键", key=key))
            if key in flow_pt:
                flow_pt.pop(key)
                logger.info(t("从Flow检查点中移除 '{key}' 键", key=key))
            if key in hift_pt:
                hift_pt.pop(key)
                logger.info(t("从Hift检查点中移除 '{key}' 键", key=key))

        # 加载状态字典
        llm.load_state_dict(llm_pt)
        flow.load_state_dict(flow_pt)
        hift.load_state_dict(hift_pt)
        
        # 设置设备和精度
        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        
        if self.device == "cuda":
            logger.info(t("将模型移动到GPU..."))
            if args.bf16:
                llm.eval().cuda().to(torch.bfloat16)
                flow.eval().cuda().half()
                hift.eval().cuda()
                llm.bf16 = True
                flow.bf16 = False
                llm.fp16 = False
                flow.fp16 = False
                logger.info(t("使用BF16精度"))
            else:
                llm.eval().cuda().to(torch.float16)
                flow.eval().cuda().to(torch.float16)
                hift.eval().cuda()
                llm.fp16 = True
                flow.fp16 = True
                llm.bf16 = False
                flow.bf16 = False
                logger.info(t("使用FP16精度"))
            
        else:
            logger.info(t("使用CPU进行推理..."))
            llm.eval()
            flow.eval()
            hift.eval()
        
        torch.cuda.empty_cache()
        
        self.models = {
            'llm': llm,
            'flow': flow,
            'hift': hift
        }
        
        # 设置前端处理器
        self.setup_frontend(args)
        
        if os.path.exists(os.path.join(args.model_dir, 'zero_shot_speakers_16k.pt')):
            self.zero_shot_speakers = torch.load(os.path.join(args.model_dir, 'zero_shot_speakers_16k.pt'))
        else:
            self.zero_shot_speakers = None

        self.is_loaded = True
        logger.info(t("模型加载完成"))
        
    def setup_frontend(self, args):
        """设置前端处理器"""
        logger.info(t("初始化前端处理器..."))
        
        campplus_path = os.path.join(args.model_dir, 'campplus.onnx')
        speech_tokenizer_path = os.path.join(args.model_dir, 'speech_tokenizer_v3.onnx')
        spk2info_path = os.path.join(args.model_dir, 'spk2info.pt')
        
        # 验证前端组件文件存在
        for path, name in [(campplus_path, 'campplus'), (speech_tokenizer_path, 'speech_tokenizer')]:
            if not os.path.exists(path):
                raise ValueError(t("前端组件文件不存在: {path} ({name})", path=path, name=name))
        
        self.frontend = CosyVoiceFrontEnd(
            self.configs['get_tokenizer'],
            self.configs['feat_extractor'],
            campplus_path,
            speech_tokenizer_path,
            spk2info_path,
            self.configs['allowed_special']
        )
        
        logger.info(t("前端处理器初始化完成"))
    
    def load_pt(self, llm_pt, flow_pt):
        """加载模型权重"""
        try:
            self.models['llm'].load_state_dict(torch.load(llm_pt, map_location='cpu'))
            self.models['llm'].eval().cuda().to(torch.bfloat16)
            self.models['llm'].bf16 = True
            self.models['llm'].fp16 = False
            self.models['flow'].load_state_dict(torch.load(flow_pt, map_location='cpu'))
            self.models['flow'].eval().cuda().half()
            self.models['flow'].fp16 = True
            self.models['flow'].bf16 = False
            logger.info(t("模型权重加载完成"))
            return {"status": "success", "message": t("模型权重加载完成")}
        except Exception as e:
            logger.error(t("模型权重加载失败: {error}", error=e))
            return {"status": "error", "message": str(e)}

    def get_available_speakers(self):
        """获取当前可用的speaker_id列表"""
        if not self.frontend:
            return []
        
        try:
            # 从spk2info中获取speaker信息
            spk2info = self.frontend.spk2info
            if hasattr(spk2info, 'keys'):
                # 如果是字典类型
                speaker_ids = list(spk2info.keys())
            elif hasattr(spk2info, 'speaker_ids'):
                # 如果有speaker_ids属性
                speaker_ids = list(spk2info.speaker_ids)
            else:
                # 尝试其他可能的属性
                speaker_ids = []
                for attr in dir(spk2info):
                    if 'speaker' in attr.lower() and not attr.startswith('_'):
                        try:
                            value = getattr(spk2info, attr)
                            if isinstance(value, (list, tuple)):
                                speaker_ids.extend(value)
                            elif isinstance(value, dict):
                                speaker_ids.extend(value.keys())
                        except:
                            continue
            
            # 去重并排序
            speaker_ids = sorted(list(set(speaker_ids)))
            
            # 获取speaker详细信息
            speaker_details = []
            for spk_id in speaker_ids:
                try:
                    if hasattr(spk2info, 'get'):
                        # 如果是字典类型
                        spk_info = spk2info.get(spk_id, {})
                    else:
                        # 尝试其他方式获取信息
                        spk_info = {}
                        for attr in dir(spk2info):
                            if hasattr(getattr(spk2info, attr), 'get'):
                                try:
                                    info = getattr(spk2info, attr).get(spk_id)
                                    if info:
                                        spk_info.update(info)
                                except:
                                    continue
                    
                    speaker_details.append({
                        "speaker_id": str(spk_id),
                        "name": spk_info.get("name", f"Speaker_{spk_id}"),
                        "gender": spk_info.get("gender", "unknown"),
                        "language": spk_info.get("language", "unknown"),
                        "description": spk_info.get("description", "")
                    })
                except Exception as e:
                    logger.warning(t("获取speaker {spk_id} 详细信息失败: {error}", spk_id=spk_id, error=e))
                    speaker_details.append({
                        "speaker_id": str(spk_id),
                        "name": f"Speaker_{spk_id}",
                        "gender": "unknown",
                        "language": "unknown",
                        "description": ""
                    })
            
            return speaker_details
            
        except Exception as e:
            logger.error(t("获取speaker列表失败: {error}", error=e))
            return []

# ============================================================================
# 文本分割和合并逻辑
# ============================================================================



# ============================================================================
# 实用函数
# ============================================================================

def load_audio_from_base64(audio_base64: str, target_sr: int = 16000):
    """从base64字符串加载音频"""
    try:
        # 解码base64
        audio_bytes = base64.b64decode(audio_base64)
        
        # 从字节流加载音频
        audio_data, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        
        if audio_data.shape[0] == 2:
            audio_data = audio_data.mean(dim=0).unsqueeze(0)

        # 重采样到目标采样率
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            audio_data = resampler(audio_data)
            sample_rate = target_sr
            logger.info(t("音频重采样: {src}Hz -> {dst}Hz", src=sample_rate, dst=target_sr))
        
        return audio_data, sample_rate
        
    except Exception as e:
        raise ValueError(t("加载base64音频失败: {error}", error=e))

def load_audio_from_file(audio_path: str, target_sr: int = 16000):
    """从文件加载音频"""
    if not os.path.exists(audio_path):
        raise ValueError(t("音频文件不存在: {path}", path=audio_path))
    
    try:
        audio_data, sample_rate = torchaudio.load(audio_path)
        if audio_data.shape[0] == 2:
            audio_data = audio_data.mean(dim=0).unsqueeze(0)
        # 重采样到目标采样率
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            audio_data = resampler(audio_data)
            sample_rate = target_sr
            logger.info(t("音频重采样: {src}Hz -> {dst}Hz", src=sample_rate, dst=target_sr))
        
        return audio_data, sample_rate
        
    except Exception as e:
        raise ValueError(t("加载音频文件失败 {path}: {error}", path=audio_path, error=e))

def audio_to_base64(audio_tensor: torch.Tensor, sample_rate: int, format: str = "wav") -> str:
    """将音频张量转换为base64字符串"""
    try:
        # 创建内存缓冲区
        buffer = io.BytesIO()
        
        # 保存音频到缓冲区
        torchaudio.save(buffer, audio_tensor.cpu(), sample_rate, format=format)
        
        # 获取字节数据并编码为base64
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return audio_base64
        
    except Exception as e:
        raise ValueError(t("音频转base64失败: {error}", error=e))

def inference_zero_shot(model_manager, tts_text: str, prompt_text: str, prompt_audio: torch.Tensor, prompt_sample_rate: int, speed: float = 1.0) -> torch.Tensor:
    """执行零样本推理"""
    if not model_manager.is_loaded:
        raise ValueError(t("模型未加载"))
    
    try:
        # 预处理文本
        processed_prompt_text = model_manager.frontend.text_normalize(prompt_text, split=False, text_frontend=True)
        processed_tts_text = model_manager.frontend.text_normalize(tts_text, split=True, text_frontend=True)
        
        prompt_audio = (prompt_audio, prompt_sample_rate)

        logger.info(t("提示文本: {text}", text=processed_prompt_text))
        logger.info(t("合成文本: {text}", text=processed_tts_text))
        
        # 前端处理
        model_input = model_manager.frontend.frontend_zero_shot(
            processed_tts_text[0], 
            processed_prompt_text, 
            prompt_audio, 
            model_manager.configs['sample_rate'],
            zero_shot_spk_id=''
        )
        
        # LLM推理
        start_time = time.time()
        generator = model_manager.models['llm'].inference(
            text=model_input['text'],
            text_len=model_input['text_len'],
            prompt_text=model_input['prompt_text'],
            prompt_text_len=model_input['prompt_text_len'],
            prompt_speech_token=model_input['llm_prompt_speech_token'],
            prompt_speech_token_len=model_input['llm_prompt_speech_token_len'],
            embedding=model_input['llm_embedding']
        )
        
        output_tokens = []
        for token in generator:
            output_tokens.append(token)
            
        llm_time = time.time() - start_time
        tps = len(output_tokens) / llm_time if llm_time > 0 else 0
        logger.info(t("LLM推理完成，TPS: {tps:.2f}", tps=tps))
        
        # Flow推理
        token_tensor = torch.tensor(output_tokens).unsqueeze(0)
        if model_manager.device == "cuda":
            token_tensor = token_tensor.cuda()
        
        tts_mel, _ = model_manager.models['flow'].inference(
            token=token_tensor,
            token_len=torch.tensor([token_tensor.shape[1]], dtype=torch.int32).to(token_tensor.device),
            prompt_token=model_input['flow_prompt_speech_token'],
            prompt_token_len=model_input['flow_prompt_speech_token_len'],
            prompt_feat=model_input['prompt_speech_feat'],
            prompt_feat_len=model_input['prompt_speech_feat_len'],
            embedding=model_input['flow_embedding'],
            streaming = False,
            finalize=True
        )
        
        if speed <= 0:
            raise ValueError(f"Invalid speed: {speed}")
        if speed != 1.0:
            tts_mel = F.interpolate(tts_mel, size=max(1, int(tts_mel.shape[2] / speed)), mode='linear')

        # Hift推理
        tts_speech, tts_source = model_manager.models['hift'].inference(
            speech_feat=tts_mel, 
        )
        
        total_time = time.time() - start_time
        audio_length = tts_speech.shape[-1] / 24000
        torch.cuda.empty_cache()
        logger.info(
            t(
                "推理完成，总时间: {total:.2f}s, TPS: {tps:.2f}, RTF: {rtf:.2f}",
                total=total_time,
                tps=tps,
                rtf=total_time / audio_length,
            )
        )
        
        return tts_speech.cpu()
        
    except Exception as e:
        logger.error(t("零样本推理失败: {error}", error=e))
        raise ValueError(t("零样本推理失败: {error}", error=e))

def inference_tts(model_manager, text: str, spk_id: str, speed: float = 1.0) -> torch.Tensor:
    """执行零样本推理"""
    if not model_manager.is_loaded:
        raise ValueError(t("模型未加载"))
    
    try:
        # 预处理文本
        processed_tts_text = model_manager.frontend.text_normalize(text, split=True, text_frontend=True)
        
        logger.info(t("合成文本: {text}", text=processed_tts_text))
        
        # 前端处理
        model_input = model_manager.frontend.frontend_sft(
            processed_tts_text[0], 
            spk_id, 
        )
        
        # LLM推理
        start_time = time.time()
        generator = model_manager.models['llm'].inference(
            text=model_input['text'],
            text_len=model_input['text_len'],
            prompt_text = torch.tensor([], dtype=torch.int32).to(model_manager.device),
            prompt_text_len = torch.tensor([0], dtype=torch.int32).to(model_manager.device),
            prompt_speech_token = None,
            prompt_speech_token_len = torch.tensor([0], dtype=torch.int32).to(model_manager.device),
            embedding=model_input['llm_embedding']
        )
        
        output_tokens = []
        for token in generator:
            output_tokens.append(token)
            
        llm_time = time.time() - start_time
        tps = len(output_tokens) / llm_time if llm_time > 0 else 0
        logger.info(t("LLM推理完成，TPS: {tps:.2f}", tps=tps))
        
        # Flow推理
        token_tensor = torch.tensor(output_tokens).unsqueeze(0)
        if model_manager.device == "cuda":
            token_tensor = token_tensor.cuda()
        
        tts_mel, _ = model_manager.models['flow'].inference(
            token=token_tensor,
            token_len=torch.tensor([token_tensor.shape[1]], dtype=torch.int32).to(token_tensor.device),
            embedding=model_input['flow_embedding'].unsqueeze(0),
            streaming = False,
            finalize=True
        )
        
        if speed <= 0:
            raise ValueError(f"Invalid speed: {speed}")
        if speed != 1.0:
            tts_mel = F.interpolate(tts_mel, size=max(1, int(tts_mel.shape[2] / speed)), mode='linear')

        # Hift推理
        tts_speech, tts_source = model_manager.models['hift'].inference(
            speech_feat=tts_mel, 
        )
        total_time = time.time() - start_time
        torch.cuda.empty_cache()
        audio_length = tts_speech.shape[-1] / 24000
        logger.info(
            t(
                "推理完成，总时间: {total:.2f}s, TPS: {tps:.2f}, RTF: {rtf:.2f}",
                total=total_time,
                tps=tps,
                rtf=total_time / audio_length,
            )
        )

        return tts_speech.cpu()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(t("TTS推理失败: {error}", error=e))
        raise ValueError(t("TTS推理失败: {error}", error=e))


def _flow_chunk_to_pcm(flow, hift, all_tokens, mel_offset, embedding, speed,
                        device, finalize, prompt_token=None, prompt_token_len=None,
                        prompt_feat=None, prompt_feat_len=None):
    """Flow 流式推理 + HiFi-GAN 合成一个 chunk，返回 (pcm_bytes, new_mel_offset)."""
    token_tensor = torch.tensor(all_tokens).unsqueeze(0)
    if device == "cuda":
        token_tensor = token_tensor.cuda()

    flow_kwargs = dict(
        token=token_tensor,
        token_len=torch.tensor([len(all_tokens)], dtype=torch.int32).to(token_tensor.device),
        embedding=embedding,
        streaming=True,
        finalize=finalize,
    )
    if prompt_token is not None:
        flow_kwargs.update(
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
        )

    mel, _ = flow.inference(**flow_kwargs)

    if mel.shape[2] <= mel_offset:
        return None, mel_offset

    new_mel = mel[:, :, mel_offset:]
    new_mel_offset = mel.shape[2]

    if speed != 1.0:
        new_mel = F.interpolate(new_mel, size=max(1, int(new_mel.shape[2] / speed)), mode='linear')

    tts_speech, _ = hift.inference(speech_feat=new_mel)
    pcm_bytes = (tts_speech[0].clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy().tobytes()
    return pcm_bytes, new_mel_offset


def streaming_inference_tts(model_manager, text, spk_id, chunk_size=25, speed=1.0):
    """流式 TTS 推理生成器，逐 chunk yield PCM bytes。

    Args:
        model_manager: 模型管理器
        text: 已经过 tn.process_text() 规范化的输入文本
        spk_id: 说话人 ID
        chunk_size: 每个 chunk 的 token 数（默认 25 = 1 秒）
        speed: 语速倍率
    """
    if not model_manager.is_loaded:
        raise ValueError(t("模型未加载"))

    start_time = time.time()

    # 前端处理（text 已在 worker 层通过 tn.process_text() 规范化，
    # 此处 split=True 仅做分句，不重复规范化）
    model_input = model_manager.frontend.frontend_sft(text, spk_id)

    flow = model_manager.models['flow']
    hift = model_manager.models['hift']
    pre_lookahead_len = flow.pre_lookahead_len
    embedding = model_input['flow_embedding'].unsqueeze(0)

    # LLM 流式推理
    generator = model_manager.models['llm'].inference(
        text=model_input['text'],
        text_len=model_input['text_len'],
        prompt_text=torch.tensor([], dtype=torch.int32).to(model_manager.device),
        prompt_text_len=torch.tensor([0], dtype=torch.int32).to(model_manager.device),
        prompt_speech_token=None,
        prompt_speech_token_len=torch.tensor([0], dtype=torch.int32).to(model_manager.device),
        embedding=model_input['llm_embedding'],
    )

    all_tokens = []
    mel_offset = 0
    chunk_count = 0
    total_pcm_bytes = 0

    for token in generator:
        all_tokens.append(token)

        target = (chunk_count + 1) * chunk_size + pre_lookahead_len
        if len(all_tokens) >= target:
            pcm_bytes, mel_offset = _flow_chunk_to_pcm(
                flow, hift, all_tokens, mel_offset, embedding,
                speed, model_manager.device, finalize=False,
            )
            chunk_count += 1
            if pcm_bytes:
                total_pcm_bytes += len(pcm_bytes)
                logger.info(t("流式TTS chunk {n}, tokens={tok}, pcm_size={size}",
                              n=chunk_count, tok=len(all_tokens), size=len(pcm_bytes)))
                yield pcm_bytes

    # 最后一个 chunk（finalize）
    if all_tokens:
        pcm_bytes, mel_offset = _flow_chunk_to_pcm(
            flow, hift, all_tokens, mel_offset, embedding,
            speed, model_manager.device, finalize=True,
        )
        if pcm_bytes:
            chunk_count += 1
            total_pcm_bytes += len(pcm_bytes)
            logger.info(t("流式TTS final chunk {n}, tokens={tok}, pcm_size={size}",
                          n=chunk_count, tok=len(all_tokens), size=len(pcm_bytes)))
            yield pcm_bytes

    total_time = time.time() - start_time
    audio_seconds = total_pcm_bytes / 2 / 24000  # int16 = 2 bytes/sample
    tps = len(all_tokens) / total_time if total_time > 0 else 0
    logger.info(t("流式TTS完成, chunks={chunks}, tokens={tok}, TPS={tps:.1f}, audio={audio:.2f}s, RTF={rtf:.2f}",
                  chunks=chunk_count, tok=len(all_tokens), tps=tps,
                  audio=audio_seconds, rtf=total_time / audio_seconds if audio_seconds > 0 else 0))
    torch.cuda.empty_cache()


def streaming_inference_zero_shot(model_manager, tts_text, prompt_text,
                                   prompt_audio, prompt_sample_rate,
                                   chunk_size=25, speed=1.0):
    """流式 Zero-Shot 推理生成器，逐 chunk yield PCM bytes。

    Args:
        model_manager: 模型管理器
        tts_text: 已经过 tn.process_text() 规范化的合成文本
        prompt_text: 已经过 tn.process_text() 规范化的提示文本
        prompt_audio: 提示音频 tensor
        prompt_sample_rate: 提示音频采样率
        chunk_size: 每个 chunk 的 token 数（默认 25 = 1 秒）
        speed: 语速倍率
    """
    if not model_manager.is_loaded:
        raise ValueError(t("模型未加载"))

    start_time = time.time()

    # 前端处理（text 已在 worker 层通过 tn.process_text() 规范化）
    prompt_audio_input = (prompt_audio, prompt_sample_rate)

    model_input = model_manager.frontend.frontend_zero_shot(
        tts_text,
        prompt_text,
        prompt_audio_input,
        model_manager.configs['sample_rate'],
        zero_shot_spk_id='',
    )

    flow = model_manager.models['flow']
    hift = model_manager.models['hift']
    pre_lookahead_len = flow.pre_lookahead_len

    # LLM 流式推理（带 prompt）
    generator = model_manager.models['llm'].inference(
        text=model_input['text'],
        text_len=model_input['text_len'],
        prompt_text=model_input['prompt_text'],
        prompt_text_len=model_input['prompt_text_len'],
        prompt_speech_token=model_input['llm_prompt_speech_token'],
        prompt_speech_token_len=model_input['llm_prompt_speech_token_len'],
        embedding=model_input['llm_embedding'],
    )

    all_tokens = []
    mel_offset = 0
    chunk_count = 0
    total_pcm_bytes = 0

    for token in generator:
        all_tokens.append(token)

        target = (chunk_count + 1) * chunk_size + pre_lookahead_len
        if len(all_tokens) >= target:
            pcm_bytes, mel_offset = _flow_chunk_to_pcm(
                flow, hift, all_tokens, mel_offset,
                model_input['flow_embedding'],
                speed, model_manager.device, finalize=False,
                prompt_token=model_input['flow_prompt_speech_token'],
                prompt_token_len=model_input['flow_prompt_speech_token_len'],
                prompt_feat=model_input['prompt_speech_feat'],
                prompt_feat_len=model_input['prompt_speech_feat_len'],
            )
            chunk_count += 1
            if pcm_bytes:
                total_pcm_bytes += len(pcm_bytes)
                logger.info(t("流式ZeroShot chunk {n}, tokens={tok}, pcm_size={size}",
                              n=chunk_count, tok=len(all_tokens), size=len(pcm_bytes)))
                yield pcm_bytes

    # 最后一个 chunk（finalize）
    if all_tokens:
        pcm_bytes, mel_offset = _flow_chunk_to_pcm(
            flow, hift, all_tokens, mel_offset,
            model_input['flow_embedding'],
            speed, model_manager.device, finalize=True,
            prompt_token=model_input['flow_prompt_speech_token'],
            prompt_token_len=model_input['flow_prompt_speech_token_len'],
            prompt_feat=model_input['prompt_speech_feat'],
            prompt_feat_len=model_input['prompt_speech_feat_len'],
        )
        if pcm_bytes:
            chunk_count += 1
            total_pcm_bytes += len(pcm_bytes)
            logger.info(t("流式ZeroShot final chunk {n}, tokens={tok}, pcm_size={size}",
                          n=chunk_count, tok=len(all_tokens), size=len(pcm_bytes)))
            yield pcm_bytes

    total_time = time.time() - start_time
    audio_seconds = total_pcm_bytes / 2 / 24000
    tps = len(all_tokens) / total_time if total_time > 0 else 0
    logger.info(t("流式ZeroShot完成, chunks={chunks}, tokens={tok}, TPS={tps:.1f}, audio={audio:.2f}s, RTF={rtf:.2f}",
                  chunks=chunk_count, tok=len(all_tokens), tps=tps,
                  audio=audio_seconds, rtf=total_time / audio_seconds if audio_seconds > 0 else 0))
    torch.cuda.empty_cache()


def zero_shot_tts(model_manager, tts_text, prompt_text, prompt_audio, speed: float = 1.0):
    """
    零样本语音合成接口
    
    Args:
        request: 零样本合成请求
        
    Returns:
        包含合成音频的响应
    """
    try:
        if not model_manager.is_loaded:
            raise Exception(t("模型未加载"))
        
        logger.info(t("零样本合成请求: {text}...", text=tts_text[:50]))
        
        # 验证输入
        if not tts_text or not tts_text.strip():
            raise Exception(t("合成文本不能为空"))
        
        if not prompt_text or not prompt_text.strip():
            raise Exception(t("提示文本不能为空"))
        
        if not prompt_audio:
            raise Exception(t("提示音频不能为空"))
        
        # 加载提示音频
        prompt_audio, prompt_sample_rate = load_audio_from_file(prompt_audio)
        
        # 执行推理
        output_audio = inference_zero_shot(
            model_manager,
            tts_text,
            prompt_text,
            prompt_audio,
            prompt_sample_rate,
            speed=speed
        )
        
        return {
                "output_audio": output_audio,
                "sample_rate": model_manager.configs['sample_rate'],
                "format": "wav",
                "duration": output_audio.shape[-1] / model_manager.configs['sample_rate']
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(t("零样本合成失败: {error}", error=e))
        raise ValueError(t("零样本合成失败: {error}", error=e))

def text_to_speech(model_manager, text, speaker_id, speed: float = 1.0):
    """
    文本到语音合成接口
    
       
    Returns:
        包含合成音频的响应
    """
    start_time = time.time()
    try:
        if not model_manager.is_loaded:
            raise Exception(t("模型未加载"))
        
        logger.info(t("TTS合成请求: {text}...", text=text[:50]))
        
        # 验证输入
        if not text or not text.strip():
            raise Exception(t("合成文本不能为空"))
        
        # 验证speaker_id
        if speaker_id:
            available_speakers = model_manager.get_available_speakers()
            speaker_ids = [spk['speaker_id'] for spk in available_speakers]
            
            if speaker_id not in speaker_ids:
                raise Exception(
                    t("无效的speaker_id: {speaker_id}。可用speaker_id: {speaker_ids}", speaker_id=speaker_id, speaker_ids=speaker_ids)
                )
        else:
            # 如果没有指定speaker_id，使用第一个可用的speaker
            available_speakers = model_manager.get_available_speakers()
            if available_speakers:
                speaker_id = available_speakers[0]['speaker_id']
                logger.info(t("使用默认speaker_id: {speaker_id}", speaker_id=speaker_id))
            else:
                raise Exception(t("没有可用的说话人"))
        
        # 执行TTS推理
        output_audio = inference_tts(
            model_manager,
            text,
            speaker_id,
            speed=speed
        )

        total_time = time.time() - start_time
        audio_length = output_audio.shape[-1] / model_manager.configs['sample_rate']

        return {
                "output_audio": output_audio,
                "sample_rate": model_manager.configs['sample_rate'],
                "format": "wav",
                "duration": output_audio.shape[-1] / model_manager.configs['sample_rate'],
                "speaker_id": speaker_id,
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(t("TTS合成失败: {error}", error=e))
        raise ValueError(t("TTS合成失败: {error}", error=e))

#TODO: 实现RAG增强的语音合成逻辑
def rag_enhanced_tts(request):
    """
    RAG增强语音合成接口（预留）
    
    Args:
        request: RAG合成请求
        
    Returns:
        包含合成音频的响应
    """
    try:
        if not model_manager.is_loaded:
            raise HTTPException(status_code=500, detail=t("模型未加载"))
        
        logger.info(t("RAG合成请求: {text}...", text=request.query[:50]))
        
        # 验证输入
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail=t("查询文本不能为空"))
        
        # TODO: 实现RAG增强的语音合成逻辑
        # 1. 检索相关知识
        # 2. 生成上下文增强的文本
        # 3. 执行语音合成
        # 4. 返回结果
        
        return APIResponse(
            success=False,
            message=t("RAG接口暂未实现"),
            error=t("该接口预留给后续RAG功能实现")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(t("RAG合成失败: {error}", error=e))
        return APIResponse(
            success=False,
            message=t("RAG合成失败"),
            error=str(e)
        )

def get_speakers():
    """获取可用的说话人列表"""
    try:
        speaker_pt = torch.load(os.path.join(os.getenv('TTS_MODEL_DIR'), 'spk2info.pt'))
        speaker_details = []
        for spk_id, spk_embedding in speaker_pt.items():
            speaker_details.append(spk_id)
        return speaker_details
    except Exception as e:
        logger.error(t("获取说话人列表失败: {error}", error=e))
        return []

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CosyVoice-MHT TTS BenchMark测试")

    # ---- 模型配置 ---- #
    parser.add_argument("--config", type=str, help="YAML配置文件路径")
    parser.add_argument("--model_dir", type=str, help="模型目录路径")
    parser.add_argument("--output_dir", type=str, help="输出目录路径")
    
    # ---- 设备和精度配置 ---- #
    parser.add_argument("--cpu", action="store_true", help="使用CPU推理")
    parser.add_argument("--bf16", action="store_true", default=True, help="LLM使用BF16精度")
    parser.add_argument("--fp16", action="store_true", default=False, help="Flow使用FP16精度")

    args = parser.parse_args()

    # 全局模型管理器
    model_manager = ModelManager()


    model_manager.load_models(args)

    spk_list = get_speakers(model_manager)

    for spk in spk_list:
        print(spk)

    # result = text_to_speech(model_manager, "咱们也来做一个天命人试试!我命由我,不由天!", spk_list[0]['speaker_id'])

    result = zero_shot_tts(model_manager, "咱们也来做一个天命人试试!我命由我,不由天!", "对啊，咱们假扮他们去取经", "浪浪山小妖怪-野猪.WAV")


    
    # 保存音频
    audio_path = os.path.join(args.output_dir, "output.wav")
    torchaudio.save(audio_path, result['output_audio'], result['sample_rate'])

if __name__ == "__main__":
    main()
