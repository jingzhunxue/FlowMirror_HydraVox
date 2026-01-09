import os
from typing import Any, Dict

_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "导入错误: {error}": {"en": "Import error: {error}"},
    "当前工作目录: {cwd}": {"en": "Current working directory: {cwd}"},
    "Python路径: {path}": {"en": "Python path: {path}"},
    "模型已加载，跳过重复加载": {"en": "Model already loaded; skipping"},
    "开始加载模型...": {"en": "Loading model..."},
    "加载LLM检查点...": {"en": "Loading LLM checkpoint..."},
    "加载Flow检查点...": {"en": "Loading Flow checkpoint..."},
    "加载Hift检查点...": {"en": "Loading Hift checkpoint..."},
    "从LLM检查点中移除 '{key}' 键": {"en": "Removed key '{key}' from LLM checkpoint"},
    "从Flow检查点中移除 '{key}' 键": {"en": "Removed key '{key}' from Flow checkpoint"},
    "从Hift检查点中移除 '{key}' 键": {"en": "Removed key '{key}' from Hift checkpoint"},
    "将模型移动到GPU...": {"en": "Moving model to GPU..."},
    "使用BF16精度": {"en": "Using BF16 precision"},
    "使用FP16精度": {"en": "Using FP16 precision"},
    "使用CPU进行推理...": {"en": "Using CPU for inference..."},
    "模型加载完成": {"en": "Model loaded"},
    "初始化前端处理器...": {"en": "Initializing frontend..."},
    "前端处理器初始化完成": {"en": "Frontend initialized"},
    "合成文本不能为空": {"en": "Synthesis text cannot be empty"},
    "提示文本不能为空": {"en": "Prompt text cannot be empty"},
    "提示音频不能为空": {"en": "Prompt audio cannot be empty"},
    "说话人ID不能为空": {"en": "Speaker ID cannot be empty"},
    "TTS合成成功": {"en": "TTS synthesis succeeded"},
    "零样本合成失败": {"en": "Zero-shot synthesis failed"},
    "零样本合成失败: {error}": {"en": "Zero-shot synthesis failed: {error}"},
    "TTS合成失败": {"en": "TTS synthesis failed"},
    "TTS合成失败: {error}": {"en": "TTS synthesis failed: {error}"},
    "加载模型权重: {llm} {flow}": {"en": "Loading model weights: {llm} {flow}"},
    "加载模型权重成功": {"en": "Model weights loaded"},
    "加载模型权重失败": {"en": "Failed to load model weights"},
    "加载模型权重失败: {error}": {"en": "Failed to load model weights: {error}"},
    "获取说话人列表失败": {"en": "Failed to fetch speaker list"},
    "获取说话人列表失败: {error}": {"en": "Failed to fetch speaker list: {error}"},
    "启动训练失败": {"en": "Failed to start training"},
    "启动训练失败: {error}": {"en": "Failed to start training: {error}"},
    "停止训练失败": {"en": "Failed to stop training"},
    "训练任务 {training_id} 不存在": {"en": "Training task {training_id} does not exist"},
    "获取训练状态成功": {"en": "Fetched training status"},
    "获取训练状态失败": {"en": "Failed to fetch training status"},
    "获取训练列表成功": {"en": "Fetched training list"},
    "获取训练列表失败": {"en": "Failed to fetch training list"},
    "删除训练任务失败": {"en": "Failed to delete training task"},
    "训练已启动": {"en": "Training started"},
    "训练启动失败: {error}": {"en": "Failed to start training: {error}"},
    "训练任务 {training_id} 已停止": {"en": "Training task {training_id} stopped"},
    "停止训练失败: {error}": {"en": "Failed to stop training: {error}"},
    "训练任务 {training_id} 已删除": {"en": "Training task {training_id} deleted"},
    "模型未加载": {"en": "Model is not loaded"},
    "前端组件文件不存在: {path} ({name})": {"en": "Frontend component missing: {path} ({name})"},
    "模型权重加载完成": {"en": "Model weights loaded"},
    "模型权重加载失败: {error}": {"en": "Failed to load model weights: {error}"},
    "获取speaker {spk_id} 详细信息失败: {error}": {"en": "Failed to fetch speaker {spk_id} details: {error}"},
    "获取speaker列表失败: {error}": {"en": "Failed to fetch speaker list: {error}"},
    "文本分割为 {count} 个片段:": {"en": "Text split into {count} segments:"},
    "片段 {index}: {segment}": {"en": "Segment {index}: {segment}"},
    "正在合成片段 {index}/{total}: {segment}": {"en": "Synthesizing segment {index}/{total}: {segment}"},
    "第{index}段使用TTS合成": {"en": "Segment {index} uses TTS synthesis"},
    "第{index}段使用zero shot合成，以第{prev_index}段为提示": {
        "en": "Segment {index} uses zero-shot synthesis with segment {prev_index} as prompt"
    },
    "片段 {index} 合成失败: {error}": {"en": "Segment {index} synthesis failed: {error}"},
    "片段 {index} 后添加 {pause_ms:.1f}ms 停顿": {"en": "Added {pause_ms:.1f}ms pause after segment {index}"},
    "音频合并完成，总长度: {samples} samples ({seconds:.2f}s)": {
        "en": "Audio merged; total length: {samples} samples ({seconds:.2f}s)"
    },
    "没有成功合成的音频片段": {"en": "No audio segments were synthesized"},
    "音频重采样: {src}Hz -> {dst}Hz": {"en": "Audio resampled: {src}Hz -> {dst}Hz"},
    "加载base64音频失败: {error}": {"en": "Failed to load base64 audio: {error}"},
    "音频文件不存在: {path}": {"en": "Audio file not found: {path}"},
    "加载音频文件失败 {path}: {error}": {"en": "Failed to load audio file {path}: {error}"},
    "音频转base64失败: {error}": {"en": "Failed to convert audio to base64: {error}"},
    "提示文本: {text}": {"en": "Prompt text: {text}"},
    "合成文本: {text}": {"en": "Synthesis text: {text}"},
    "LLM推理完成，TPS: {tps:.2f}": {"en": "LLM inference completed, TPS: {tps:.2f}"},
    "推理完成，总时间: {total:.2f}s, TPS: {tps:.2f}, RTF: {rtf:.2f}": {
        "en": "Inference completed, total: {total:.2f}s, TPS: {tps:.2f}, RTF: {rtf:.2f}"
    },
    "零样本推理失败: {error}": {"en": "Zero-shot inference failed: {error}"},
    "TTS推理失败: {error}": {"en": "TTS inference failed: {error}"},
    "零样本合成请求: {text}...": {"en": "Zero-shot request: {text}..."},
    "TTS合成请求: {text}...": {"en": "TTS request: {text}..."},
    "无效的speaker_id: {speaker_id}。可用speaker_id: {speaker_ids}": {
        "en": "Invalid speaker_id: {speaker_id}. Available speaker_ids: {speaker_ids}"
    },
    "使用默认speaker_id: {speaker_id}": {"en": "Using default speaker_id: {speaker_id}"},
    "没有可用的说话人": {"en": "No available speakers"},
    "文本长度超过5000字符，使用分段推理": {
        "en": "Text length exceeds 5000 characters; using segmented inference"
    },
    "RAG合成请求: {text}...": {"en": "RAG request: {text}..."},
    "RAG接口暂未实现": {"en": "RAG API not implemented"},
    "该接口预留给后续RAG功能实现": {"en": "This endpoint is reserved for future RAG support"},
    "RAG合成失败": {"en": "RAG synthesis failed"},
    "RAG合成失败: {error}": {"en": "RAG synthesis failed: {error}"},
    "查询文本不能为空": {"en": "Query text cannot be empty"},
}


def t(text: str, **kwargs: Any) -> str:
    lang = os.getenv("HYDRAVOX_API_LANG", os.getenv("HYDRAVOX_UI_LANG", "zh")).lower()
    if lang not in ("zh", "en"):
        lang = "zh"
    entry = _TRANSLATIONS.get(text)
    result = entry.get(lang, text) if entry else text
    if kwargs:
        try:
            return result.format(**kwargs)
        except Exception:
            return result
    return result
