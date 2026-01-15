import os, io, base64, requests, numpy as np
import gradio as gr
from typing import Tuple, List
from .tabs import create_inference_tab, create_data_tab, create_training_tab
from .tabs.speaker_manage import create_speaker_manage_tab
from .i18n import t, get_lang, set_lang
from pathlib import Path

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
ASSETS_DIR = (Path(__file__).resolve().parent.parent / "assets").resolve()
LOGO_IMG_PATH = (ASSETS_DIR / "HydraVox.png").resolve()
LOGO_IMG_URL = f"/file={LOGO_IMG_PATH}"
gr.set_static_paths(paths=[ASSETS_DIR])

def get_speakers() -> List[str]:
    """从后端获取说话人列表"""
    try:
        resp = requests.get(f"{BACKEND}/speakers")
        resp.raise_for_status()
        speakers = resp.json()
        if isinstance(speakers, list) and len(speakers) > 0:
            return speakers
        else:
            return ["default"]
    except Exception as e:
        print(t("ui.speaker_fetch_failed", error=str(e)))
        return ["default"]

def tts_once(text: str, speaker_id: str) -> Tuple[int, np.ndarray]:
    """执行一次TTS合成（向后兼容函数）"""
    resp = requests.post(f"{BACKEND}/api/v1/tts", json={"text": text, "speaker_id": speaker_id})
    resp.raise_for_status()
    data = resp.json()
    audio_b64 = data["audio_base64"]
    sr = int(data["sample_rate"])
    wav_bytes = base64.b64decode(audio_b64)
    # Gradio 的 Audio 可接受 (sr, np.ndarray[float32]) 范式
    import soundfile as sf  # 可选：若不想额外依赖，也可以用 scipy.io.wavfile 读取
    audio_np, file_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if file_sr != sr:
        sr = file_sr
    return (sr, audio_np)

def create_simple_ui():
    """创建简单版UI（向后兼容）"""
    with gr.Blocks(title=t("Multi-Head AR TTS (HTTP)")) as demo:
        gr.Markdown(t("### Multi-Head AR TTS · HTTP Only · Gradio Frontend"))
        with gr.Row():
            text = gr.Textbox(label=t("Text"), value=t("你好，这是一个基于 HTTP 的 TTS 演示。"))
            speaker = gr.Dropdown(choices=get_speakers(), value="default", label=t("Speaker"))
        out = gr.Audio(label=t("Audio"), type="numpy", streaming=False, autoplay=True)
        btn = gr.Button(t("Synthesize"))
        btn.click(fn=tts_once, inputs=[text, speaker], outputs=out)
        gr.Markdown(t("Backend: `{backend}`", backend=BACKEND))
    return demo

def create_main_ui():
    """创建主界面，整合所有tab"""
    with gr.Blocks(
        title=t("HydraVox TTS System"),
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        """
    ) as demo:
        # 主标题
        def _build_header_html() -> str:
            return (
                "<div style=\"text-align: center; padding: 20px;\">"
                "<h1 style=\"color: #2c3e50; margin-bottom: 10px; display: flex; align-items: center; "
                "justify-content: center; gap: 12px;\">"
                f"<img src=\"{LOGO_IMG_URL}\" alt=\"HydraVox\" style=\"height: 36px; vertical-align: middle;\"/>"
                f"{t('HydraVox TTS System')}"
                "</h1>"
                f"<p style=\"color: #7f8c8d; font-size: 18px;\">{t('支持多Token预测的语音合成系统')}</p>"
                "</div>"
            )

        def _build_footer_html() -> str:
            backend_line = t("🔗 后端服务: {backend}", backend=f"<code>{BACKEND}</code>")
            return (
                "<div style=\"text-align: center; padding: 20px; margin-top: 30px; "
                "border-top: 1px solid #eee; color: #7f8c8d;\">"
                f"<p>{backend_line}</p>"
                f"<p>{t('💡 HydraVox - 让语音合成更简单')}</p>"
                "</div>"
            )

        header_html = gr.HTML(_build_header_html())
        with gr.Row():
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=[("English", "en"), ("中文", "zh")],
                    value=get_lang(),
                    label=t("语言"),
                    interactive=True,
                )
            with gr.Column(scale=2):
                language_status = gr.Markdown(value="")

        # 创建三个tab
        with gr.Tabs():
            # 推理tab
            inference_i18n = create_inference_tab()
            
            # 数据处理tab  
            data_i18n = create_data_tab()
            
            # 训练tab
            training_i18n = create_training_tab()
            
            # 说话人管理tab
            speaker_i18n = create_speaker_manage_tab()
        
        # 底部信息
        footer_html = gr.HTML(_build_footer_html())

        def _apply_language(
            lang: str,
            synthesis_mode: str,
            s3_device: str,
            model_type: str,
            device_choice: str,
            precision_choice: str,
            output_dir: str,
        ):
            set_lang(lang)
            updates = [
                gr.update(label=t("语言"), value=lang),
                gr.update(value=t("语言已更新。")),
                gr.update(value=_build_header_html()),
                gr.update(value=_build_footer_html()),
            ]
            updates.extend(inference_i18n["apply"](synthesis_mode))
            updates.extend(data_i18n["apply"](s3_device))
            updates.extend(training_i18n["apply"](model_type, device_choice, precision_choice, output_dir))
            updates.extend(speaker_i18n["apply"]())
            return updates

        language.change(
            fn=_apply_language,
            inputs=[
                language,
                inference_i18n["inputs"][0],
                data_i18n["inputs"][0],
                training_i18n["inputs"][0],
                training_i18n["inputs"][1],
                training_i18n["inputs"][2],
                training_i18n["inputs"][3],
            ],
            outputs=[
                language,
                language_status,
                header_html,
                footer_html,
                *inference_i18n["outputs"],
                *data_i18n["outputs"],
                *training_i18n["outputs"],
                *speaker_i18n["outputs"],
            ],
        )
    
    return demo

def launch_ui(server_name: str = "0.0.0.0", server_port: int = 7860, simple: bool = False):
    """启动UI界面
    
    Args:
        server_name: 服务器地址
        server_port: 服务器端口
        simple: 是否使用简单版界面（向后兼容）
    """
    if simple:
        demo = create_simple_ui()
        print(t("ui.simple_start"))
    else:
        demo = create_main_ui()
        print(t("ui.full_start"))
    
    print(t("ui.service_addr", server_name=server_name, server_port=server_port))
    print(t("ui.backend_addr", backend=BACKEND))
    print("=" * 50)
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_api=False,
        debug=False,
        favicon_path=None,
        allowed_paths=[ASSETS_DIR]
    )

def launch_main_ui(
    server_name: str = "0.0.0.0", 
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """启动完整版主界面"""
    demo = create_main_ui()
    
    print(t("ui.system_start"))
    print(t("ui.service_addr", server_name=server_name, server_port=server_port))
    print(t("ui.backend_addr", backend=BACKEND))
    print("=" * 50)
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_api=False,
        debug=debug,
        favicon_path=None,
        allowed_paths=[ASSETS_DIR]
    )

# 向后兼容的别名函数
def build_ui():
    """向后兼容：构建简单版UI"""
    return create_simple_ui()

def launch_full_ui(server_name: str = "0.0.0.0", server_port: int = 7860, **kwargs):
    """向后兼容：启动完整版UI"""
    return launch_main_ui(server_name=server_name, server_port=server_port, **kwargs)

if __name__ == "__main__":
    # 可以通过环境变量配置启动参数
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    debug = os.getenv("GRADIO_DEBUG", "false").lower() == "true"
    simple_mode = os.getenv("GRADIO_SIMPLE", "false").lower() == "true"
    
    if simple_mode:
        launch_ui(server_name=server_name, server_port=server_port, simple=True)
    else:
        launch_main_ui(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug
        ) 
