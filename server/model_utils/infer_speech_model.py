# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import logging
import random
import torch
import torch.nn.functional as F
import torchaudio
import base64
import io
from hyperpyyaml import load_hyperpyyaml

# 导入项目模块
try:
    from .cosyvoice.cli.frontend import CosyVoiceFrontEnd
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
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
            logger.info("模型已加载，跳过重复加载")
            return
            
        logger.info("开始加载模型...")
        
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
        logger.info("加载LLM检查点...")
        llm_pt = torch.load(os.path.join(args.model_dir, 'llm.pt'), map_location='cpu')
        
        logger.info("加载Flow检查点...")
        flow_pt = torch.load(os.path.join(args.model_dir, 'flow.pt'), map_location='cpu')
        
        logger.info("加载Hift检查点...")
        hift_pt = torch.load(os.path.join(args.model_dir, 'hift.pt'), map_location='cpu')
        
            # 清理检查点中的epoch和step信息
        for key in ['epoch', 'step', '_original_metadata', '_conversion_info']:
            if key in llm_pt:
                llm_pt.pop(key)
                logger.info(f"从LLM检查点中移除 '{key}' 键")
            if key in flow_pt:
                flow_pt.pop(key)
                logger.info(f"从Flow检查点中移除 '{key}' 键")
            if key in hift_pt:
                hift_pt.pop(key)
                logger.info(f"从Hift检查点中移除 '{key}' 键")

        # 加载状态字典
        llm.load_state_dict(llm_pt)
        flow.load_state_dict(flow_pt)
        hift.load_state_dict(hift_pt)
        
        # 设置设备和精度
        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        
        if self.device == "cuda":
            logger.info("将模型移动到GPU...")
            if args.bf16:
                llm.eval().cuda().to(torch.bfloat16)
                flow.eval().cuda().to(torch.bfloat16)
                hift.eval().cuda()
                llm.bf16 = True
                flow.bf16 = True
                llm.fp16 = False
                flow.fp16 = False
                logger.info("使用BF16精度")
            else:
                llm.eval().cuda().to(torch.float16)
                flow.eval().cuda().to(torch.float16)
                hift.eval().cuda()
                llm.fp16 = True
                flow.fp16 = True
                llm.bf16 = False
                flow.bf16 = False
                logger.info("使用FP16精度")
            
        else:
            logger.info("使用CPU进行推理...")
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
        logger.info("模型加载完成")
        
    def setup_frontend(self, args):
        """设置前端处理器"""
        logger.info("初始化前端处理器...")
        
        campplus_path = os.path.join(args.model_dir, 'campplus.onnx')
        speech_tokenizer_path = os.path.join(args.model_dir, 'speech_tokenizer_v3.onnx')
        spk2info_path = os.path.join(args.model_dir, 'spk2info.pt')
        
        # 验证前端组件文件存在
        for path, name in [(campplus_path, 'campplus'), (speech_tokenizer_path, 'speech_tokenizer')]:
            if not os.path.exists(path):
                raise ValueError(f"前端组件文件不存在: {path} ({name})")
        
        self.frontend = CosyVoiceFrontEnd(
            self.configs['get_tokenizer'],
            self.configs['feat_extractor'],
            campplus_path,
            speech_tokenizer_path,
            spk2info_path,
            self.configs['allowed_special']
        )
        
        logger.info("前端处理器初始化完成")
    
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
            logger.info("模型权重加载完成")
            return {"status": "success", "message": "模型权重加载完成"}
        except Exception as e:
            logger.error(f"模型权重加载失败: {e}")
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
                    logger.warning(f"获取speaker {spk_id} 详细信息失败: {e}")
                    speaker_details.append({
                        "speaker_id": str(spk_id),
                        "name": f"Speaker_{spk_id}",
                        "gender": "unknown",
                        "language": "unknown",
                        "description": ""
                    })
            
            return speaker_details
            
        except Exception as e:
            logger.error(f"获取speaker列表失败: {e}")
            return []

# ============================================================================
# 文本分割和合并逻辑
# ============================================================================

def split_text_by_punctuation(text: str, max_length: int = 50, min_length: int = 10) -> list:
    """
    基于标点符号分割文本，避免过短片段
    
    Args:
        text: 输入文本
        max_length: 触发分割的最大长度
        min_length: 最小片段长度，小于此长度的片段会与前一个合并
        
    Returns:
        分割后的文本片段列表
    """
    if len(text) <= max_length:
        return [text]
    
    # 中文标点符号
    punctuation_marks = ['。', '！', '？', '；', '，', '、', '.', '!', '?', ';', ',']
    
    segments = []
    current_segment = ""
    
    # 按字符遍历，寻找标点符号
    for i, char in enumerate(text):
        current_segment += char
        
        # 如果遇到标点符号，考虑分割
        if char in punctuation_marks:
            # 如果当前片段长度合适，进行分割
            if len(current_segment) >= min_length:
                segments.append(current_segment)
                current_segment = ""
            # 否则继续累积（不分割）
    
    # 处理最后一个片段
    if current_segment:
        # 如果最后一个片段太短且前面有片段，则合并
        if len(current_segment) < min_length and segments:
            segments[-1] += current_segment
        else:
            segments.append(current_segment)
    
    # 如果没有找到合适的标点分割，按长度强制分割
    if not segments:
        segments = [text]
    elif len(segments) == 1 and len(segments[0]) > max_length:
        # 如果分割后仍然太长，按长度强制分割
        segments = []
        for i in range(0, len(text), max_length):
            segment = text[i:i + max_length]
            if segment:
                segments.append(segment)
    
    return segments


def merge_short_segments(segments: list, min_length: int = 5) -> list:
    """
    合并过短的片段
    
    Args:
        segments: 文本片段列表
        min_length: 最小片段长度
        
    Returns:
        合并后的片段列表
    """
    if not segments:
        return segments
    
    merged_segments = []
    current_segment = segments[0]
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        
        # 如果当前片段太短，与下一个片段合并
        if len(current_segment) < min_length:
            current_segment += next_segment
        else:
            # 当前片段长度合适，保存并开始新片段
            merged_segments.append(current_segment)
            current_segment = next_segment
    
    # 处理最后一个片段
    if current_segment:
        # 如果最后一个片段太短且前面有片段，则合并
        if len(current_segment) < min_length and merged_segments:
            merged_segments[-1] += current_segment
        else:
            merged_segments.append(current_segment)
    
    return merged_segments


def inference_tts_with_segmentation(model_manager, text: str, spk_id: str, max_length: int = 30, min_length: int = 10, last_prompt: bool = True, speed: float = 1.0) -> torch.Tensor:
    """
    带文本分割的TTS推理
    第一段使用speaker做正常TTS，后续段落使用上一个片段作为提示进行zero shot合成
    
    Args:
        text: 输入文本
        spk_id: 说话人ID
        max_length: 触发分割的最大长度
        min_length: 最小片段长度
        last_prompt: 是否使用上一段音频作为zero shot提示，False时全部使用speaker TTS
        
    Returns:
        合成的音频tensor
    """
    # 分割文本
    segments = split_text_by_punctuation(text, max_length, min_length)
    segments = merge_short_segments(segments, min_length)
    
    logger.info(f"文本分割为 {len(segments)} 个片段:")
    for i, segment in enumerate(segments):
        logger.info(f"  片段 {i+1}: {segment}")
    
    if len(segments) == 1:
        # 只有一个片段，直接推理
        return inference_tts(model_manager, text, spk_id, speed=speed)
    
    # 多个片段处理
    audio_segments = []
    prev_segment_text = None
    prev_segment_audio = None
    
    for i, segment in enumerate(segments):
        logger.info(f"正在合成片段 {i+1}/{len(segments)}: {segment}")
        try:
            if i == 0 or not last_prompt:
                # 第一段或禁用last_prompt时：使用speaker做正常TTS
                logger.info(f"第{i+1}段使用TTS合成")
                segment_audio = inference_tts(model_manager, segment, spk_id, speed=speed)
                prev_segment_text = segment
                prev_segment_audio = segment_audio
                audio_segments.append(segment_audio)
            else:
                # 后续段且启用last_prompt：使用上一个片段作为提示进行zero shot合成
                logger.info(f"第{i+1}段使用zero shot合成，以第{i}段为提示")
                segment_audio = inference_zero_shot(
                    model_manager=model_manager,
                    tts_text=segment,
                    prompt_text=prev_segment_text,
                    prompt_audio=prev_segment_audio,
                    prompt_sample_rate=24000,
                    speed=speed
                )
                # 更新提示为当前片段，供下一个片段使用
                prev_segment_text = segment
                prev_segment_audio = segment_audio
                audio_segments.append(segment_audio)
                
        except Exception as e:
            logger.error(f"片段 {i+1} 合成失败: {e}")
            raise ValueError(f"片段 {i+1} 合成失败: {e}")
    
    # 合并音频片段，在片段间添加随机停顿
    if audio_segments:
        sample_rate = model_manager.configs['sample_rate']
        final_audio_parts = []
        
        for i, audio_segment in enumerate(audio_segments):
            final_audio_parts.append(audio_segment)
            
            # 在非最后一个片段后添加随机停顿
            if i < len(audio_segments) - 1:
                # 生成50-150ms的随机停顿时长
                pause_duration_ms = random.uniform(50, 150)
                pause_duration_samples = int(pause_duration_ms * sample_rate / 1000)
                
                # 创建静音张量，形状与音频张量匹配
                silence_shape = list(audio_segment.shape)
                silence_shape[-1] = pause_duration_samples  # 修改时间维度长度
                silence = torch.zeros(silence_shape, dtype=audio_segment.dtype, device=audio_segment.device)
                
                final_audio_parts.append(silence)
                logger.info(f"片段 {i+1} 后添加 {pause_duration_ms:.1f}ms 停顿")
        
        # 使用torch.cat沿着时间维度合并
        merged_audio = torch.cat(final_audio_parts, dim=-1)
        logger.info(f"音频合并完成，总长度: {merged_audio.shape[-1]} samples ({merged_audio.shape[-1]/sample_rate:.2f}s)")
        return merged_audio
    else:
        raise ValueError("没有成功合成的音频片段")


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
            logger.info(f"音频重采样: {sample_rate}Hz -> {target_sr}Hz")
        
        return audio_data, sample_rate
        
    except Exception as e:
        raise ValueError(f"加载base64音频失败: {e}")

def load_audio_from_file(audio_path: str, target_sr: int = 16000):
    """从文件加载音频"""
    if not os.path.exists(audio_path):
        raise ValueError(f"音频文件不存在: {audio_path}")
    
    try:
        audio_data, sample_rate = torchaudio.load(audio_path)
        if audio_data.shape[0] == 2:
            audio_data = audio_data.mean(dim=0).unsqueeze(0)
        # 重采样到目标采样率
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            audio_data = resampler(audio_data)
            sample_rate = target_sr
            logger.info(f"音频重采样: {sample_rate}Hz -> {target_sr}Hz")
        
        return audio_data, sample_rate
        
    except Exception as e:
        raise ValueError(f"加载音频文件失败 {audio_path}: {e}")

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
        raise ValueError(f"音频转base64失败: {e}")

def inference_zero_shot(model_manager, tts_text: str, prompt_text: str, prompt_audio: torch.Tensor, prompt_sample_rate: int, speed: float = 1.0) -> torch.Tensor:
    """执行零样本推理"""
    if not model_manager.is_loaded:
        raise ValueError("模型未加载")
    
    try:
        # 预处理文本
        processed_prompt_text = model_manager.frontend.text_normalize(prompt_text, split=False, text_frontend=True)
        processed_tts_text = model_manager.frontend.text_normalize(tts_text, split=True, text_frontend=True)
        
        prompt_audio = (prompt_audio, prompt_sample_rate)

        logger.info(f"提示文本: {processed_prompt_text}")
        logger.info(f"合成文本: {processed_tts_text}")
        
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
        logger.info(f"LLM推理完成，TPS: {tps:.2f}")
        
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
        logger.info(f"推理完成，总时间: {total_time:.2f}s, TPS: {tps:.2f}, RTF: {total_time / audio_length:.2f}")
        
        return tts_speech.cpu()
        
    except Exception as e:
        logger.error(f"零样本推理失败: {e}")
        raise ValueError(f"零样本推理失败: {e}")

def inference_tts(model_manager, text: str, spk_id: str, speed: float = 1.0) -> torch.Tensor:
    """执行零样本推理"""
    if not model_manager.is_loaded:
        raise ValueError("模型未加载")
    
    try:
        # 预处理文本
        processed_tts_text = model_manager.frontend.text_normalize(text, split=True, text_frontend=True)
        
        logger.info(f"合成文本: {processed_tts_text}")
        
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
        logger.info(f"LLM推理完成，TPS: {tps:.2f}")
        
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
        logger.info(f"推理完成，总时间: {total_time:.2f}s, TPS: {tps:.2f}, RTF: {total_time / audio_length:.2f}")

        return tts_speech.cpu()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"TTS推理失败: {e}")
        raise ValueError(f"TTS推理失败: {e}")

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
            raise Exception("模型未加载")
        
        logger.info(f"零样本合成请求: {tts_text[:50]}...")
        
        # 验证输入
        if not tts_text or not tts_text.strip():
            raise Exception("合成文本不能为空")
        
        if not prompt_text or not prompt_text.strip():
            raise Exception("提示文本不能为空")
        
        if not prompt_audio:
            raise Exception("提示音频不能为空")
        
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
        logger.error(f"零样本合成失败: {e}")
        raise ValueError(f"零样本合成失败: {e}")

def text_to_speech(model_manager, text, speaker_id, speed: float = 1.0):
    """
    文本到语音合成接口
    
       
    Returns:
        包含合成音频的响应
    """
    start_time = time.time()
    try:
        if not model_manager.is_loaded:
            raise Exception("模型未加载")
        
        logger.info(f"TTS合成请求: {text[:50]}...")
        
        # 验证输入
        if not text or not text.strip():
            raise Exception("合成文本不能为空")
        
        # 验证speaker_id
        if speaker_id:
            available_speakers = model_manager.get_available_speakers()
            speaker_ids = [spk['speaker_id'] for spk in available_speakers]
            
            if speaker_id not in speaker_ids:
                raise Exception(
                    f"无效的speaker_id: {speaker_id}。可用speaker_id: {speaker_ids}"
                )
        else:
            # 如果没有指定speaker_id，使用第一个可用的speaker
            available_speakers = model_manager.get_available_speakers()
            if available_speakers:
                speaker_id = available_speakers[0]['speaker_id']
                logger.info(f"使用默认speaker_id: {speaker_id}")
            else:
                raise Exception("没有可用的说话人")
        
        # 执行TTS推理
        segments_info = None
        if len(text) > 5000:
            # 长文本使用分段推理
            logger.info(f"文本长度超过30字符，使用分段推理")
            output_audio = inference_tts_with_segmentation(
                model_manager,
                text,
                speaker_id,
                max_length=30,
                min_length=10,
                last_prompt=False,
                speed=speed
            )
            # 获取分段信息用于响应
            segments = split_text_by_punctuation(text, 30, 10)
            segments = merge_short_segments(segments, 10)
            segments_info = {
                "total_segments": len(segments),
                "segments": segments
            }
        else:
            # 短文本直接推理
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
                "segments_info": segments_info
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"TTS合成失败: {e}")
        raise ValueError(f"TTS合成失败: {e}")

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
            raise HTTPException(status_code=500, detail="模型未加载")
        
        logger.info(f"RAG合成请求: {request.query[:50]}...")
        
        # 验证输入
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="查询文本不能为空")
        
        # TODO: 实现RAG增强的语音合成逻辑
        # 1. 检索相关知识
        # 2. 生成上下文增强的文本
        # 3. 执行语音合成
        # 4. 返回结果
        
        return APIResponse(
            success=False,
            message="RAG接口暂未实现",
            error="该接口预留给后续RAG功能实现"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG合成失败: {e}")
        return APIResponse(
            success=False,
            message="RAG合成失败",
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
        logger.error(f"获取说话人列表失败: {e}")
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
