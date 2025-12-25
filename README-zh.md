<div align="center">

<img src="assets/HydraVox.png" alt="HydraVox 标志" width="25%" />

# FlowMirror-HydraVox

**原生加速的多头自回归 TTS（文本转语音）模型，衍生于 CosyVoice。**

[英文](README.md) · [简体中文] *(当前文档)*

</div>

---

<p align="center">
  <a href="#-highlights">亮点</a> •
  <a href="#-quickstart-1-minute">快速开始</a> •
  <a href="#webui">WebUI</a> •
  <a href="#roadmap">路线图</a> •
  <a href="#python-api">Python API</a> •
  <a href="#models--weights">模型</a> •
  <a href="#train--finetune">训练</a> •
  <a href="#license">许可</a>
</p>

<p align="center">
  <img alt="badge-python" src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img alt="badge-pytorch" src="https://img.shields.io/badge/PyTorch-2.3%2B-red" />
  <img alt="badge-cuda" src="https://img.shields.io/badge/CUDA-12.x-76b900" />
  <img alt="badge-license" src="https://img.shields.io/badge/License-Apache--2.0-green" />
</p>

---

## ✨ Highlights

- **多头自回归解码**：每步预测多个语音 token → 在相同质量目标下实现更低延迟与更高吞吐。
- **即开即用的 WebUI**：推理、快速批量合成、微调（即将支持）、说话人管理、日志/曲线。
- **LoRA 热加载（情感/风格，敬请期待）**：运行时按句加载/卸载，支持多适配器叠加与独立缩放（如 `--lora happy.safetensors:0.6,energetic.safetensors:0.3`）。
- **SFT 实现**：源自 CosyVoice 2.0。
- **可复现脚本**：一键 Demo，版本锁定配置。
- **源自 CosyVoice2.0**：与上游清晰差异；尽可能兼容数据格式。

> **负责任使用**：请勿在未经明确同意的情况下克隆或伪造他人声音。

---

## 🔊 Samples & Demo

- **音频样例**：`assets/samples`
- **在线演示**：`http://localhost:7890`

---

## 🚀 Quickstart (1 minute)

> **前置**：Python 3.10+，已安装 FFmpeg 并在 PATH；推荐 NVIDIA GPU + CUDA 12.x；支持 CPU（较慢）。

### 从源码启动

```bash
# 0) Clone
git clone https://github.com/jingzhunxue/FlowMirror-HydraVox.git
cd FlowMirror-HydraVox

# 1) Create conda env
conda create -n hydravox python=3.11

# 2) Install dependencies
pip install -r requirements.txt

# 3) Download model weights
modelscope download jzx-ai-lab/HydraVox-CV3 --local_dir jzx-ai-lab/HydraVox-CV3

# 4) Create .env
cp .env.example .env
```

---

<a name="webui"></a>
## WebUI

启动：

```bash
python main.py --api-host 0.0.0.0 --api-port 7860 --with-ui
# 仅 API（无浏览器 UI）
python main.py --api-host 0.0.0.0 --api-port 7860
```

功能：

- 文本 → 语音，长文本分段。
- **数据处理面板**：数据集浏览、配置、实时日志与曲线。
- **训练/微调面板**：数据集浏览、配置、实时日志与曲线。
- **说话人管理**：新增/重命名/删除、预览、嵌入。

截图：

<p align="center">
  <img src="assets/ui-home.png" alt="HydraVox WebUI - 首页" width="80%" />
  <img src="assets/ui-train.png" alt="HydraVox WebUI - 训练" width="80%" />
</p>

**数据目录**（可覆盖）：

```
jzx-ai-lab/HydraVox-CV3      # 模型权重
logs/                    # 训练/推理日志
```

---

<a name="roadmap"></a>
## Roadmap

...
- [ ] 2025/10
  - [ ] 支持流式推理
  - [ ] DeepSeek 风格的多 Token 预测模块，带来更强更稳的推理
- [ ] 2025/09
  - [ ] 引入面向 TTS 的 flow-matching 核心改造
- [ ] 2025/08
  - [X] 发布训练 UI 与训练脚本
  - [ ] 发布 LoRA 热加载与预训练情感 LoRA 推理

---

## Python API

基于 REST（FastAPI，默认前缀 `/api/v1`）。下方提供最小可用示例与关键字段说明。

```python
import base64
import requests

BASE = "http://localhost:8888/api/v1"

def load_pt(llm_pt: str, flow_pt: str):
    resp = requests.post(f"{BASE}/load_pt", json={
        "llm_pt": llm_pt,
        "flow_pt": flow_pt,
    }, timeout=120)
    resp.raise_for_status()
    print(resp.json())
    return resp.json()

def list_speakers():
    resp = requests.get(f"{BASE}/speakers", timeout=30)
    resp.raise_for_status()
    return resp.json()

def tts(text: str, speaker_id: str,
        output_format: str = "wav",
        last_prompt: bool = True,
        extra_params: dict | None = None):
    payload = {
        "text": text,
        "speaker_id": speaker_id,
        "output_format": output_format,
        "last_prompt": last_prompt,
        "extra_params": extra_params or {
            "top_p": 0.9,
            "top_k": 10,
            "win_size": 32,
            "tau_r": 0.2,
            "inference_head_num": 2
        }
    }
    resp = requests.post(f"{BASE}/tts", json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", True):
        raise RuntimeError(data.get("error") or data.get("message"))
    audio_b64 = data["data"]["audio_base64"]
    with open(f"out_tts.{output_format}", "wb") as f:
        f.write(base64.b64decode(audio_b64))
    return data

def zero_shot(tts_text: str, prompt_text: str, prompt_wav_path: str,
              output_format: str = "wav",
              extra_params: dict | None = None):
    with open(prompt_wav_path, "rb") as f:
        prompt_audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "tts_text": tts_text,
        "prompt_text": prompt_text,
        "prompt_audio_base64": prompt_audio_base64,
        "output_format": output_format,
        "extra_params": extra_params or {
            "top_p": 0.9,
            "top_k": 10,
            "win_size": 32,
            "tau_r": 0.2,
            "inference_head_num": 2
        }
    }
    resp = requests.post(f"{BASE}/zero-shot", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", True):
        raise RuntimeError(data.get("error") or data.get("message"))
    with open(f"out_zero_shot.{output_format}", "wb") as f:
        f.write(base64.b64decode(data["data"]["audio_base64"]))
    return data

# 示例
# load_pt("checkpoints/llm.pt", "checkpoints/flow.pt")
# speakers = list_speakers(); print(speakers)
# tts("今天天气很好。", speaker_id="spk1")
# zero_shot("请朗读：你好，世界。", "这是我的声音示例。", "assets/samples/prompt.wav")
```

**关键参数**

- **POST `/api/v1/tts`**
  - **text**：待合成文本（必填）
  - **speaker_id**：说话人 ID（必填，可通过 `/speakers` 获取）
  - **output_format**：输出格式，默认 `wav`
  - **last_prompt**：是否将上一段音频作为 zero‑shot 提示使用，默认 `true`
  - **extra_params**：推理超参（可选）
    - `top_p`=0.9、`top_k`=10、`win_size`=32、`tau_r`=0.2、`inference_head_num`=2
  - **响应**：`{ success, message, data: { audio_base64, sample_rate, format, duration, speaker_id, segments_info } }`

- **POST `/api/v1/zero-shot`**
  - **tts_text**：待合成文本（必填）
  - **prompt_text**：与提示音频语义相匹配/描述的文本（必填）
  - **prompt_audio_base64**：提示音频的 Base64（必填）
    - 如：`base64.b64encode(open('prompt.wav','rb').read()).decode('utf-8')`
  - **output_format**：输出格式，默认 `wav`
  - **extra_params**：同上
  - **响应**：`{ success, message, data: { audio_base64, sample_rate, format, duration, segments_info } }`

- **POST `/api/v1/load_pt`**
  - **llm_pt**：LLM 权重路径（必填）
  - **flow_pt**：Flow 权重路径（必填）
  - **响应**：`{ success, message, data | error }`

- **GET `/api/v1/speakers`**
  - **用途**：获取可用说话人列表
  - **响应**：返回说话人集合（实现可能返回列表或 `{ success, ... }` 包装）

**说明/限制**
- `/tts` 服务端超时约 60s；请切分长文本或提升资源。
- `prompt_audio_base64` 必须来自原始音频字节的 Base64 编码。
- 响应中的 `audio_base64` 可直接 `base64.b64decode(...)` 保存为音频文件。

---

## Models & Weights

| 名称                     | 参数量 | 语言 | 类型            | 多头 | 链接 |
| ------------------------ | ----: | ---- | --------------- | ---: | ---- |
| hydravox-base-pretrained | ~300M | zh/en | AR-Transformer |   5 | https://www.modelscope.cn/models/jzx-ai-lab/HydraVox-CV3/file/view/master/llm.pt |

> 下载完整权重：
```
modelscope download jzx-ai-lab/HydraVox-CV3 --local_dir jzx-ai-lab/HydraVox-CV3
```

---

## Train & Finetune

### 使用 WebUI 进行数据预处理（Data Process）

进入 “📊 数据处理” 标签页，可一键流水线或按阶段运行。

- 一键处理：填写 “📁 输入目录”“🎤 采样率”“⚠️ 覆盖文件”，点击 “🚀 开始”。界面将依次运行四个阶段，并显示总体进度/状态/日志。
- 输出目录约定：
  - 阶段 1（格式转换与重采样）→ `<输入>_resample`
  - 阶段 2（VAD 语音活动检测）→ `<阶段1输出>_vad`
  - 阶段 3（ASR 转录）→ `<阶段2输出>_asr`
  - 阶段 4（提取训练 Token）→ `<阶段3输出>_token`
  - 阶段 5（可选，数据集合并）→ 合并多个 HF 数据集至指定目录

分阶段要点（与 UI 折叠面板对应）：
- 阶段 1：选择输入/输出与采样率（默认 16kHz），支持覆盖；可 “👀 预览变更” 后再执行
- 阶段 2：Silero VAD；关键参数：置信度阈值、最短语音/静音(ms)、前后填充(ms)、最短/最长片段(s)
- 阶段 3：ASR 转录；设备：自动/CPU/GPU；并行进程（GPU 数）；支持刷新设备检测；输出 HF 数据集
- 阶段 4：提取训练 Token；设备/并行与上同；输出 Token 化数据集
- 阶段 5：数据集合并；输入多个数据集目录（英文逗号分隔）并保存

提示：自动探测 CUDA 并显示 GPU 数；VAD/ASR 默认 16kHz；各阶段均提供状态与日志，便于排错。

### 训练（WebUI：🚀 训练）

在 “🚀 训练” 标签页进行：

- 1) 数据集：选择由阶段 3/4 产出的 HF 数据集（如 `<...>_asr` 或 `<...>_token`）
- 2) 模型：类型 `llm`/`flow`；检查点（如 `jzx-ai-lab/HydraVox-CV3/llm.pt`）；分词器；输出目录（如 `checkpoints/training_llm`）
- 3) 训练参数：batch size、学习率、轮数、保存步距、`logging_steps`、`eval_steps`
- 4) 高级：启用 LoRA（可选）；精度 BF16/FP16（按模型类型推荐）
- 5) 计算：设备 Auto/CPU/GPU；进程数（GPU 数）；可填 GPU IDs；可刷新设备检测
- 6) 控制与可视化：
  - 日志：面板实时显示，同时保存到 `logs/training/train_<timestamp>.log`
  - 曲线：保存至 `<输出目录>/figure/training_plot.png`
  - 支持 “立即刷新/强制刷新” 与定时自动刷新

说明：通过 Accelerate 启动，按精度选择 `bf16/fp16`；`logging_steps` 与 `eval_steps` 将体现在日志/曲线的步数刻度上。

### 模型管理

- 列表：按时间倒序展示已发现的输出路径（扫描 `checkpoints/training_llm`、`checkpoints/training_flow`、`checkpoints/training`、`checkpoints`、`models`、`outputs`、`ckpt`，忽略 `runs/logs/figure`）
- 刷新列表：重新扫描
- 加载路径：将选中路径回显以便后续操作
- 删除路径：删除选中文件夹（受保护目录如 `runs/logs/figure` 不可删）
- 转换为 `model.pt (bf16)`：将 `pytorch_model.bin` 转换为 `model.pt`
  - 不支持分片索引 `.bin.index.json`，需先合并

注意：大规模处理/训练请确保磁盘与 GPU 资源；若训练结束后图表未自动刷新，可点击“⚡ 强制刷新”。

---

## License

- 代码：**Apache‑2.0**（示例 — 请按实际许可更新）。
- **源自 CosyVoice** — 参见 `NOTICE` 与 `LICENSE-THIRD-PARTY`。

---

## 📚 引用

```bibtex
@software{hydravox2025,
  title = {FlowMirror-HydraVox: Multi-head AR TTS with Native Acceleration},
  author = {Your Name and Contributors},
  year = {2025},
  url = {https://github.com/your-org/FlowMirror-HydraVox}
}
```

---

## 🙏 致谢

- [**CosyVoice**](https://github.com/FunAudioLLM/CosyVoice) 作者与贡献者
- [**Better & Faster Large Language Models via Multi-token Prediction**](https://arxiv.org/abs/2404.19737)
