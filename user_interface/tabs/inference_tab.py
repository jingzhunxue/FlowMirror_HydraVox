import os, io, base64, requests, numpy as np, gradio as gr
from typing import Tuple, List
import logging

logger = logging.getLogger("inference_tab")

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

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
    """加载模型权重"""
    payload = {
        "llm_pt": llm_pt,
        "flow_pt": flow_pt
    }
    resp = requests.post(f"{BACKEND}/api/v1/load_pt", json=payload)
    resp.raise_for_status()
    return resp.json()

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

def clear_inputs():
    """清空输入与输出"""
    return "", None

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
            with gr.Column(scale=0.4):
                llm_choices = list_pt_files_from_env("LLM_DIR", "llm")
                llm_weight = gr.Dropdown(
                    choices=llm_choices,
                    value=(llm_choices[0] if llm_choices else None),
                    label="LLM 权重 (llm.pt)",
                    allow_custom_value=True,
                    interactive=True,
                )
            with gr.Column(scale=0.4):
                flow_choices = list_pt_files_from_env("FLOW_DIR", "flow")
                flow_weight = gr.Dropdown(
                    choices=flow_choices,
                    value=(flow_choices[0] if flow_choices else None),
                    label="Flow 权重 (flow.pt)",
                    allow_custom_value=True,
                    interactive=True,
                )
            with gr.Column(scale=0.2):
                # 通过elem_id应用垂直居中样式
                load_pt_btn = gr.Button("🔄 加载模型", variant="secondary", elem_id="load-pt-btn")
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
                gr.HTML(
                    """
                    <div style=\"display:flex;align-items:center;justify-content:space-between;\">
                        <div style=\"display:flex;align-items:center;gap:8px;\">
                            <span style=\"font-weight:600;color:#34495e;\">说话人</span>
                            <span style=\"font-size:12px;color:#95a5a6;\">选择发音人</span>
                        </div>
                    </div>
                    """
                )
                speakers_init = get_speakers()
                with gr.Row():
                    with gr.Column(scale=0.7):
                        speaker = gr.Dropdown(
                            choices=speakers_init,
                            value=(speakers_init[0] if speakers_init else "default"),
                            label=None,
                            scale=4,
                            allow_custom_value=False,
                        )
                    with gr.Column(scale=0.3):
                        refresh_btn = gr.Button("↻ 刷新说话人列表", variant="secondary", min_width=110)
                speaker_info = gr.Markdown(
                    value=f"可用说话人：{len(speakers_init)} 个",
                    elem_classes=["tiny-muted"]
                )
        
        with gr.Row():
            with gr.Accordion("高级设置", open=False):
                with gr.Row():
                    top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="top_p")
                    top_k = gr.Slider(1, 100, value=10, step=1, label="top_k")
                with gr.Row():
                    win_size = gr.Slider(1, 256, value=32, step=8, label="win_size")
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
        synth_btn.click(
            fn=tts_once,
            inputs=[single_text, speaker, top_p, top_k, win_size, tau_r, inference_head_num],
            outputs=audio_out,
        )
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[single_text, audio_out],
        )
        
        refresh_btn.click(
            fn=refresh_speakers,
            outputs=[speaker, speaker_info],
        )

        load_pt_btn.click(
            fn=load_pt,
            inputs=[llm_weight, flow_weight],
            outputs=[],
        )
        
        gr.Markdown(f"**后端地址**: `{BACKEND}`") 