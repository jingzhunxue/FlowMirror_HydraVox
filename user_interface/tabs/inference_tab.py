import os, io, base64, requests, numpy as np, gradio as gr
from typing import Tuple, List, Dict, Optional
import logging
import soundfile as sf
from pathlib import Path

logger = logging.getLogger("inference_tab")

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
SAMPLES_DIR = PROJECT_ROOT / "assets/samples"

# 默认参考音频配置（使用相对路径）
DEFAULT_REFERENCE_AUDIO = PROJECT_ROOT / "assets/samples/浪浪山的小妖怪/小猪妖/小猪妖1.wav"
DEFAULT_REFERENCE_TEXT_FILE = PROJECT_ROOT / "assets/samples/浪浪山的小妖怪/小猪妖/小猪妖1.txt"

def scan_reference_samples() -> Dict[str, Dict[str, Path]]:
    """扫描assets/samples目录，获取所有可用的参考音频和文本文件"""
    samples = {}
    
    if not SAMPLES_DIR.exists():
        logger.warning(f"Samples directory not found: {SAMPLES_DIR}")
        return samples
    
    try:
        # 遍历所有作品文件夹（如：浪浪山的小妖怪、小猪佩奇等）
        for work_dir in SAMPLES_DIR.iterdir():
            if not work_dir.is_dir():
                continue
            
            # 首先检查作品文件夹下是否直接有音频文件（单层结构）
            for wav_file in work_dir.glob("*.wav"):
                txt_file = wav_file.with_suffix(".txt")
                if txt_file.exists():
                    # 生成显示名称：作品名/文件名
                    display_name = f"{work_dir.name}/{wav_file.stem}"
                    samples[display_name] = {
                        "audio": wav_file,
                        "text": txt_file
                    }
                    logger.debug(f"Found sample: {display_name}")
            
            # 然后遍历每个作品下的角色文件夹（两层结构）
            for character_dir in work_dir.iterdir():
                if not character_dir.is_dir():
                    continue
                
                # 查找音频和文本文件对
                for wav_file in character_dir.glob("*.wav"):
                    txt_file = wav_file.with_suffix(".txt")
                    if txt_file.exists():
                        # 生成显示名称：作品名/角色名/文件名
                        display_name = f"{work_dir.name}/{character_dir.name}/{wav_file.stem}"
                        samples[display_name] = {
                            "audio": wav_file,
                            "text": txt_file
                        }
                        logger.debug(f"Found sample: {display_name}")
        
        logger.info(f"Found {len(samples)} reference samples in {SAMPLES_DIR}")
        
    except Exception as e:
        logger.error(f"Error scanning samples directory: {e}")
    
    return samples

# 在启动时扫描一次样本目录
REFERENCE_SAMPLES = scan_reference_samples()

def get_speakers() -> List[str]:
    """从后端获取说话人列表"""
    try:
        resp = requests.get(f"{BACKEND}/api/v1/speakers")
        resp.raise_for_status()
        speakers = resp.json()
        if isinstance(speakers, list) and len(speakers) > 0:
            return speakers
        else:
            logger.error("后端返回的说话人列表为空")
            return ["default"]
    except Exception as e:
        logger.error(f"获取说话人列表失败: {str(e)}")
        return ["default"]

# 新增：列出环境变量目录中的 .pt 文件
def list_pt_files_from_env(env_var: str, key_word: str = "") -> List[str]:
    directory = os.getenv(env_var, "")
    if not directory or not os.path.isdir(directory):
        return []
    try:
        return [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.endswith(".pt") and os.path.isfile(os.path.join(directory, name)) and key_word in name
        ]
    except Exception as e:
        logger.error(f"读取 {env_var} 目录失败: {str(e)}")
        return []

def refresh_speakers():
    """刷新说话人列表，并返回下拉框更新与信息文本"""
    speakers = get_speakers()
    info = f"可用说话人：{len(speakers)} 个"
    return gr.update(choices=speakers, value=speakers[0] if speakers else "default"), info

def load_pt(llm_pt: str, flow_pt: str):
    """加载模型权重，返回面向用户的状态文本，并弹出提示。"""
    try:
        if not llm_pt or not flow_pt:
            gr.Warning("请选择 LLM 与 Flow 权重文件后再加载。")
            return "❗ 请先选择 LLM 与 Flow 权重文件。"

        payload = {
            "llm_pt": llm_pt,
            "flow_pt": flow_pt
        }
        resp = requests.post(f"{BACKEND}/api/v1/load_pt", json=payload)
        resp.raise_for_status()
        data = resp.json()
        # 兼容后端不同返回格式
        msg = data.get("message") if isinstance(data, dict) else str(data)
        gr.Info("模型权重加载成功")
        return f"✅ 加载成功\nLLM: {llm_pt}\nFlow: {flow_pt}\n消息: {msg}"
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        gr.Warning(f"加载失败: {e}")
        return f"❌ 加载失败: {e}"

def tts_once(
    text: str,
    speaker_id: str,
    top_p: float,
    top_k: int,
    win_size: int,
    tau_r: float,
    inference_head_num: int,
) -> Tuple[int, np.ndarray]:
    """执行一次TTS合成，携带高级控制参数"""
    try:
        payload = {
            "text": text,
            "speaker_id": speaker_id,
            "extra_params": {
                "top_p": float(top_p),
                "top_k": int(top_k),
                "win_size": int(win_size),
                "tau_r": float(tau_r),
                "inference_head_num": int(inference_head_num),
            },
        }
        resp = requests.post(f"{BACKEND}/api/v1/tts", json=payload)
        resp.raise_for_status()
        data = resp.json()['data']
        audio_b64 = data["audio_base64"]
        sr = int(data["sample_rate"])
        wav_bytes = base64.b64decode(audio_b64)
        
        import soundfile as sf
        audio_np, file_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if file_sr != sr:
            sr = file_sr
        return (sr, audio_np)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"TTS合成失败: {str(e)}")
        return None

def zero_shot_tts(
    text: str,
    prompt_text: str,
    prompt_audio,
    top_p: float,
    top_k: int,
    win_size: int,
    tau_r: float,
    inference_head_num: int,
) -> Tuple[int, np.ndarray]:
    """执行Zero-shot TTS合成"""
    try:
        if prompt_audio is None:
            logger.error("提示音频不能为空")
            return None
            
        # 将音频转换为base64
        import soundfile as sf
        audio_data = prompt_audio[1]  # (sr, audio_data)
        sample_rate = prompt_audio[0]
        
        # 保存为临时文件再读取
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            with open(tmp_file.name, "rb") as f:
                audio_bytes = f.read()
            os.unlink(tmp_file.name)
        
        prompt_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        payload = {
            "tts_text": text,
            "prompt_text": prompt_text,
            "prompt_audio_base64": prompt_audio_base64,
            "extra_params": {
                "top_p": float(top_p),
                "top_k": int(top_k),
                "win_size": int(win_size),
                "tau_r": float(tau_r),
                "inference_head_num": int(inference_head_num),
            },
        }
        resp = requests.post(f"{BACKEND}/api/v1/zero-shot", json=payload)
        resp.raise_for_status()
        data = resp.json()['data']
        audio_b64 = data["audio_base64"]
        sr = int(data["sample_rate"])
        wav_bytes = base64.b64decode(audio_b64)
        
        audio_np, file_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if file_sr != sr:
            sr = file_sr
        return (sr, audio_np)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Zero-shot合成失败: {str(e)}")
        return None

def synthesis_wrapper(
    text: str,
    synthesis_mode: str,
    speaker_id: str,
    prompt_text: str,
    prompt_audio,
    top_p: float,
    top_k: int,
    win_size: int,
    tau_r: float,
    inference_head_num: int,
) -> Tuple[int, np.ndarray]:
    """合成包装函数，根据模式选择不同的合成方式"""
    if synthesis_mode == "预设说话人":
        return tts_once(text, speaker_id, top_p, top_k, win_size, tau_r, inference_head_num)
    elif synthesis_mode == "Zero-shot":
        if not prompt_text.strip():
            logger.error("Zero-shot模式下提示文本不能为空")
            return None
        if prompt_audio is None:
            logger.error("Zero-shot模式下提示音频不能为空")
            return None
        return zero_shot_tts(text, prompt_text, prompt_audio, top_p, top_k, win_size, tau_r, inference_head_num)
    else:
        logger.error(f"未知的合成模式: {synthesis_mode}")
        return None

def load_default_reference_audio():
    """加载默认参考音频和文本"""
    text_value = ""  # 初始化默认文本值
    audio_value = None  # 初始化默认音频值
    
    try:
        # 读取默认参考文本
        if DEFAULT_REFERENCE_TEXT_FILE.exists():
            try:
                with open(DEFAULT_REFERENCE_TEXT_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        text_value = content
                        logger.info(f"成功加载默认参考文本: {len(content)} 字符")
            except Exception as te:
                logger.warning(f"读取默认参考文本失败: {te}")
        else:
            logger.warning(f"默认参考文本文件不存在: {DEFAULT_REFERENCE_TEXT_FILE}")

        # 读取默认参考音频
        if DEFAULT_REFERENCE_AUDIO.exists():
            try:
                audio_data, sample_rate = sf.read(str(DEFAULT_REFERENCE_AUDIO), dtype="float32")
                audio_value = (sample_rate, audio_data)
                logger.info(f"成功加载默认参考音频: {sample_rate}Hz, {len(audio_data)/sample_rate:.2f}秒")
            except Exception as ae:
                logger.error(f"读取默认参考音频失败: {ae}")
        else:
            logger.warning(f"默认参考音频文件不存在: {DEFAULT_REFERENCE_AUDIO}")
            
        return audio_value, text_value
        
    except Exception as e:
        logger.error(f"加载默认参考音频失败: {e}")
        return None, text_value

def load_reference_sample(sample_name: str) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """加载指定的参考样本音频和文本"""
    if not sample_name or sample_name not in REFERENCE_SAMPLES:
        return load_default_reference_audio()
    
    sample_info = REFERENCE_SAMPLES[sample_name]
    text_value = ""
    audio_value = None
    
    try:
        # 读取文本
        if sample_info["text"].exists():
            with open(sample_info["text"], "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    text_value = content
                    logger.info(f"加载参考文本 [{sample_name}]: {len(content)} 字符")
        
        # 读取音频
        if sample_info["audio"].exists():
            audio_data, sample_rate = sf.read(str(sample_info["audio"]), dtype="float32")
            audio_value = (sample_rate, audio_data)
            logger.info(f"加载参考音频 [{sample_name}]: {sample_rate}Hz, {len(audio_data)/sample_rate:.2f}秒")
    
    except Exception as e:
        logger.error(f"加载参考样本失败 [{sample_name}]: {e}")
        return load_default_reference_audio()
    
    return audio_value, text_value

def toggle_synthesis_mode(mode: str):
    """切换合成模式时的界面更新"""
    if mode == "预设说话人":
        return (
            gr.update(visible=True),   # speaker_row
            gr.update(visible=False),  # zero_shot_row
            gr.update(value=""),       # prompt_text
            gr.update(value=None),     # prompt_audio
            gr.update(visible=False),  # reference_preset
        )
    elif mode == "Zero-shot":
        # 获取样本列表
        sample_names = list(REFERENCE_SAMPLES.keys())
        
        # 如果有样本可用，加载第一个样本
        if sample_names:
            default_sample = sample_names[0]
            default_audio, default_text = load_reference_sample(default_sample)
        else:
            # 如果没有样本，加载默认音频
            default_sample = None
            default_audio, default_text = load_default_reference_audio()
        
        return (
            gr.update(visible=False),  # speaker_row
            gr.update(visible=True),   # zero_shot_row
            gr.update(value=default_text),  # prompt_text
            gr.update(value=default_audio), # prompt_audio
            gr.update(
                visible=True,
                choices=sample_names,
                value=default_sample
            ),  # reference_preset
        )
    else:
        return (
            gr.update(visible=True),   # speaker_row
            gr.update(visible=False),  # zero_shot_row
            gr.update(value=""),       # prompt_text
            gr.update(value=None),     # prompt_audio
            gr.update(visible=False),  # reference_preset
        )

def clear_inputs():
    """清空输入与输出"""
    return "", None, "", None

def create_inference_tab():
    """创建推理tab界面（精简与美化）"""
    with gr.Tab("🎤 语音合成"):
        gr.Markdown(
            """
            <div style=\"display:flex;align-items:center;gap:10px;margin:8px 0 2px 0;\">
                <h3 style=\"margin:0;color:#2c3e50;\">TTS 语音合成</h3>
                <span style=\"font-size:12px;color:#95a5a6;\">即时文本转语音 · 支持多说话人</span>
            </div>
            """
        )
        
        # 新增：模型权重选择（来自环境变量目录）
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                llm_choices = list_pt_files_from_env("LLM_DIR", "llm")
                llm_weight = gr.Dropdown(
                    choices=llm_choices,
                    value=(llm_choices[0] if llm_choices else None),
                    label="LLM 权重 (llm.pt)",
                    allow_custom_value=True,
                    interactive=True,
                )
            with gr.Column(scale=2):
                flow_choices = list_pt_files_from_env("FLOW_DIR", "flow")
                flow_weight = gr.Dropdown(
                    choices=flow_choices,
                    value=(flow_choices[0] if flow_choices else None),
                    label="Flow 权重 (flow.pt)",
                    allow_custom_value=True,
                    interactive=True,
                )
            with gr.Column(scale=1):
                # 通过elem_id应用垂直居中样式
                load_pt_btn = gr.Button("🔄 加载模型", variant="secondary", elem_id="load-pt-btn")
        # 模型加载状态显示
        model_load_info = gr.Markdown(value="", elem_classes=["tiny-muted"])
        # 局部样式：让按钮容器充满列高并垂直居中
        gr.HTML(
            """
            <style>
            #load-pt-btn { height: 100%; display: flex; align-items: center; }
            #load-pt-btn button { width: 100%; }
            </style>
            """
        )
        with gr.Row():
            with gr.Column(scale=2):
                single_text = gr.Textbox(
                    label="输入文本",
                    value="你好，这是一个基于 HTTP 的 TTS 演示。",
                    placeholder="请输入要合成的文本...",
                    lines=4,
                )
                
                gr.Examples(
                    examples=[
                        "今天天气很好，适合出去走走。",
                        "欢迎使用 HydraVox,多头预测让语音更自然。",
                        "请在提示框中输入你想要合成的文本内容。",
                    ],
                    inputs=[single_text],
                    label="示例"
                )
            
            with gr.Column(scale=1):
                # 合成模式选择
                synthesis_mode = gr.Radio(
                    choices=["预设说话人", "Zero-shot"],
                    value="预设说话人",
                    label="合成模式"
                )
                gr.Markdown("*选择使用预设说话人或Zero-shot语音克隆*", elem_classes=["tiny-muted"])
                
                # 预设说话人模式界面
                with gr.Group(visible=True) as speaker_row:
                    gr.HTML(
                        """
                        <div style=\"display:flex;align-items:center;gap:8px;margin:8px 0 4px 0;\">
                            <span style=\"font-weight:600;color:#34495e;\">预设说话人</span>
                            <span style=\"font-size:12px;color:#95a5a6;\">选择预训练的发音人</span>
                        </div>
                        """
                    )
                    speakers_init = get_speakers()
                    with gr.Row():
                        with gr.Column(scale=3):
                            speaker = gr.Dropdown(
                                choices=speakers_init,
                                value=(speakers_init[0] if speakers_init else "default"),
                                label=None,
                                allow_custom_value=False,
                            )
                        with gr.Column(scale=1):
                            refresh_btn = gr.Button("↻ 刷新", variant="secondary", min_width=80)
                    speaker_info = gr.Markdown(
                        value=f"可用说话人：{len(speakers_init)} 个",
                        elem_classes=["tiny-muted"]
                    )
                
                # Zero-shot模式界面
                with gr.Group(visible=False) as zero_shot_row:
                    gr.HTML(
                        """
                        <div style=\"display:flex;align-items:center;gap:8px;margin:8px 0 4px 0;\">
                            <span style=\"font-weight:600;color:#34495e;\">Zero-shot 语音克隆</span>
                            <span style=\"font-size:12px;color:#95a5a6;\">选择或上传参考音频进行语音克隆</span>
                        </div>
                        """
                    )
                    
                    # 预设参考音频选择
                    sample_names = list(REFERENCE_SAMPLES.keys())
                    reference_preset = gr.Dropdown(
                        choices=sample_names,
                        value=sample_names[0] if sample_names else None,
                        label="预设参考音频",
                        visible=False,
                        interactive=True,
                    )
                    gr.Markdown("*选择一个预设的参考音频，或者上传自己的音频文件*", elem_classes=["tiny-muted"])
                    
                    # 获取默认音频和文本
                    default_audio, default_text = load_default_reference_audio()
                    
                    prompt_text = gr.Textbox(
                        label="参考音频对应文本 (ASR内容)",
                        placeholder="请输入参考音频中说话人说的内容...",
                        lines=2,
                        value=default_text
                    )
                    gr.Markdown("*请准确输入参考音频中的文字内容，这将用于语音克隆*", elem_classes=["tiny-muted"])
                    
                    prompt_audio = gr.Audio(
                        label="参考音频",
                        type="numpy",
                        value=default_audio
                    )
                    gr.Markdown("*你可以从上方选择预设音频，或者直接上传自己的音频文件*", elem_classes=["tiny-muted"])
        
        with gr.Row():
            with gr.Accordion("高级设置", open=False):
                with gr.Row():
                    top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="top_p")
                    top_k = gr.Slider(1, 100, value=10, step=1, label="top_k")
                with gr.Row():
                    win_size = gr.Slider(0, 256, value=32, step=8, label="win_size")
                    tau_r = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="tau_r")
                inference_head_num = gr.Slider(1, 5, value=2, step=1, label="inference_head_num")
        
        with gr.Row():
            synth_btn = gr.Button("🎵 合成", variant="primary", min_width=120)
            clear_btn = gr.Button("🧹 清空", variant="secondary", min_width=100)
        
        audio_out = gr.Audio(
            label="合成音频",
            type="numpy",
            streaming=False,
            autoplay=True,
            show_download_button=True,
        )
        
        # 事件绑定
        # 模式切换
        synthesis_mode.change(
            fn=toggle_synthesis_mode,
            inputs=[synthesis_mode],
            outputs=[speaker_row, zero_shot_row, prompt_text, prompt_audio, reference_preset],
        )
        
        # 预设参考音频选择
        reference_preset.change(
            fn=load_reference_sample,
            inputs=[reference_preset],
            outputs=[prompt_audio, prompt_text],
        )
        
        # 合成按钮
        synth_btn.click(
            fn=synthesis_wrapper,
            inputs=[
                single_text, 
                synthesis_mode, 
                speaker, 
                prompt_text, 
                prompt_audio, 
                top_p, 
                top_k, 
                win_size, 
                tau_r, 
                inference_head_num
            ],
            outputs=audio_out,
        )
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[single_text, audio_out, prompt_text, prompt_audio],
        )
        
        refresh_btn.click(
            fn=refresh_speakers,
            outputs=[speaker, speaker_info],
        )

        load_pt_btn.click(
            fn=load_pt,
            inputs=[llm_weight, flow_weight],
            outputs=[model_load_info],
        )
        
        gr.Markdown(f"**后端地址**: `{BACKEND}`") 