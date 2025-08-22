import os, io, base64, requests, numpy as np
import gradio as gr
from typing import Tuple, List
from .tabs import create_inference_tab, create_data_tab, create_training_tab
from pathlib import Path

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
ASSETS_DIR = (Path(__file__).resolve().parent.parent / "assets").resolve()
LOGO_IMG_PATH = (ASSETS_DIR / "HydraVox.png").resolve()
LOGO_IMG_URL = f"/gradio_api/file={LOGO_IMG_PATH}"
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
        print(f"获取说话人列表失败: {str(e)}")
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
    with gr.Blocks(title="Multi-Head AR TTS (HTTP)") as demo:
        gr.Markdown("### Multi-Head AR TTS · HTTP Only · Gradio Frontend")
        with gr.Row():
            text = gr.Textbox(label="Text", value="你好，这是一个基于 HTTP 的 TTS 演示。")
            speaker = gr.Dropdown(choices=get_speakers(), value="default", label="Speaker")
        out = gr.Audio(label="Audio", type="numpy", streaming=False, autoplay=True)
        btn = gr.Button("Synthesize")
        btn.click(fn=tts_once, inputs=[text, speaker], outputs=out)
        gr.Markdown(f"Backend: `{BACKEND}`")
    return demo

def create_main_ui():
    """创建主界面，整合所有tab"""
    with gr.Blocks(
        title="HydraVox TTS System",
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
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #2c3e50; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 12px;">
                <img src="{LOGO_IMG_URL}" alt="HydraVox" style="height: 36px; vertical-align: middle;"/>
                HydraVox TTS System
            </h1>
            <p style="color: #7f8c8d; font-size: 18px;">
                多Token预测的端到端语音合成系统
            </p>
        </div>
        """)
        
        # 创建三个tab
        with gr.Tabs():
            # 推理tab
            create_inference_tab()
            
            # 数据处理tab  
            create_data_tab()
            
            # 训练tab
            create_training_tab()
        
        # 底部信息
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px; margin-top: 30px; 
                    border-top: 1px solid #eee; color: #7f8c8d;">
            <p>🔗 后端服务: <code>{BACKEND}</code></p>
            <p>💡 HydraVox - 让语音合成更简单</p>
        </div>
        """)
    
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
        print("🎵 启动 HydraVox 简单版界面...")
    else:
        demo = create_main_ui()
        print("🚀 启动 HydraVox 完整版界面...")
    
    print(f"📡 服务地址: http://{server_name}:{server_port}")
    print(f"🔗 后端地址: {BACKEND}")
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
    
    print("🚀 启动 HydraVox TTS 系统...")
    print(f"📡 服务地址: http://{server_name}:{server_port}")
    print(f"🔗 后端地址: {BACKEND}")
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