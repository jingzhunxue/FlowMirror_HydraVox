<div align="center">

<img src="assets/HydraVox.png" alt="HydraVox Logo" width="25%" />

# FlowMirror-HydraVox

**A natively accelerated TTS (text-to-speech) model with multi-head decoding, derived from CosyVoice.**

\[English] · [简体中文](README-zh.md) *(coming soon)*

</div>

---

<p align="center">
  <a href="#-highlights">Highlights</a> •
  <a href="#-quickstart-1-minute">Quickstart</a> •
  <a href="#webui">WebUI</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#python-api">Python API</a> •
  <a href="#models--weights">Models</a> •
  <a href="#train--finetune">Train</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-license">License</a>
</p>

<p align="center">
  <img alt="badge-python" src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img alt="badge-pytorch" src="https://img.shields.io/badge/PyTorch-2.3%2B-red" />
  <img alt="badge-cuda" src="https://img.shields.io/badge/CUDA-12.x-76b900" />
  <img alt="badge-license" src="https://img.shields.io/badge/License-Apache--2.0-green" />
</p>

---

## ✨ Highlights

* **Multi‑Head AR Decoding** — Predict multiple speech tokens per step → **lower latency** and **higher throughput** under the same quality target.
* **Ready‑to‑use WebUI** — Inference, flash batch synthesis, fine‑tuning(comming soon), speaker mgmt, logs/plots.
* **Hot-load LoRA for emotion/style**(comming soon) — Load/unload adapters at runtime per utterance; stack multiple with per-adapter scaling (e.g. `--lora happy.safetensors:0.6,energetic.safetensors:0.3`).
* **SFT implement** — SFT implement derived from CosyVoice2.0.
* **Reproducible scripts** — One‑command demo and fully version‑locked configs.
* **CosyVoice2.0‑derived** — Clear deltas vs upstream; compatible data formats where possible.

> **Responsible use:** Please do not clone or impersonate voices without explicit consent. See [Safety & Use Policy](#-safety--responsible-use).

---

## 🔊 Samples & Demo

* **Audio samples**: `assets/samples/浪浪山小妖怪-野猪.WAV`
* **Online demo**: `http://localhost:7890` — link to Space/website.

---

## 🚀 Quickstart (1 minute)

> **Prereqs**: Python 3.10+, FFmpeg installed and on PATH; NVIDIA GPU + CUDA 12.x recommended. CPU fallback supported (slower).

### From source

```bash
# 0) Clone
git clone https://github.com/jingzhunxue/FlowMirror-HydraVox.git
cd FlowMirror-HydraVox

# 1) Create conda env
conda create -n hydravox python=3.11

# 2) Install dependencies
pip install -r requirements.txt

# 3) Download model weights
modelscope download jzx-ai-lab/HydraVox --local_dir jzx-ai-lab/HydraVox

# 4) Create .env
cp .env.example .env
```

---
<a name="webui"></a>
## WebUI

Start:

```bash
python main.py --api-host 0.0.0.0 --api-port 7860 --with-ui
# API-only mode (no browser UI):
python main.py --api-host 0.0.0.0 --api-port 7860
```

Features:

* Text → Speech, long‑text chunking.
* **Data Process panel**: dataset browser, configs, live logs & curves.
* **Training/Finetune panel**: dataset browser, configs, live logs & curves.
* **Speaker manager**: add/rename/delete speakers, preview, embeddings.

Screenshots:

<p align="center">
  <img src="assets/ui-home.png" alt="HydraVox WebUI - Home" width="80%" />
  <img src="assets/ui-train.png" alt="HydraVox WebUI - Train" width="80%" />
  <!-- If the images are large, adjust width to 45%-49% to keep them on one line. -->
</p>

**Data directories** (defaults, overridable):

```
jzx-ai-lab/HydraVox      # model weights
logs/             # train/infer logs
```

---
<a name="roadmap"></a>
## Roadmap

...
* [ ] 2025/10
  - [ ] Stream inference support for HydraVox
  - [ ] Deepseek style Multi-Token-Pretiction Module implement for HydraVox which enable more powerful and stable inference
  
* [ ] 2025/09
  - [ ] flow-matching core update introducing a TTS-tailored paradigm

* [ ] 2025/08
  - [X] Release training ui tab and training scripts
  - [ ] Release LoRA hot-load and inference with pretrained emotion lora
---

## Python API

基于 REST 接口（FastAPI，默认前缀 `/api/v1`）。下面给出最小可用 Python 调用示例与关键字段说明。

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

# Example usage
# load_pt("checkpoints/llm.pt", "checkpoints/flow.pt")
# speakers = list_speakers(); print(speakers)
# tts("今天天气很好。", speaker_id="spk1")
# zero_shot("请把下面文本读出来：你好世界。", "你好，我的声音是这样的。", "assets/samples/prompt.wav")
```

**Key arguments**

- **POST `/api/v1/tts`**
  - **text**: 待合成文本（必填）
  - **speaker_id**: 说话人 ID（必填，可通过 `/speakers` 查询）
  - **output_format**: 输出格式，默认 `wav`
  - **last_prompt**: 是否使用上一段音频作为 zero‑shot 提示，默认 `true`
  - **extra_params**: 推理超参（可选）
    - `top_p`=0.9, `top_k`=10, `win_size`=32, `tau_r`=0.2, `inference_head_num`=2
  - **响应**: `{ success, message, data: { audio_base64, sample_rate, format, duration, speaker_id, segments_info } }`

- **POST `/api/v1/zero-shot`**
  - **tts_text**: 待合成文本（必填）
  - **prompt_text**: 与提示音频语义相匹配/描述的文本（必填）
  - **prompt_audio_base64**: 提示音频的 base64（必填）
    - 例如：`base64.b64encode(open('prompt.wav','rb').read()).decode('utf-8')`
  - **output_format**: 输出格式，默认 `wav`
  - **extra_params**: 同上
  - **响应**: `{ success, message, data: { audio_base64, sample_rate, format, duration, segments_info } }`

- **POST `/api/v1/load_pt`**
  - **llm_pt**: LLM 权重路径（必填）
  - **flow_pt**: Flow 权重路径（必填）
  - **响应**: `{ success, message, data | error }`

- **GET `/api/v1/speakers`**
  - **用途**: 查询可用说话人列表或信息
  - **响应**: 返回说话人集合（实现可能返回直接列表，或包装为 `{ success, ... }`）

提示与约束：
- 服务器端对 `/tts` 设有约 60s 超时；长文本请自行切分或提升服务器资源。
- `prompt_audio_base64` 应为原始音频文件的字节进行 Base64 编码后的字符串。
- 返回的 `audio_base64` 可直接 `base64.b64decode(...)` 保存为音频文件。

---

## Models & Weights

| Name                  | Params | Langs   | type   | Multi‑Head | Link  |
| --------------------- | -----: | ------- | --------- | ---------: | ----- |
| hydravox-base-pretrained         | \~300M | zh/en   |AR-Transformer  |          5 | https://www.modelscope.cn/models/jzx-ai-lab/HydraVox/file/view/master/llm.pt |

> Download total weights by ```
modelscope download jzx-ai-lab/HydraVox --local_dir jzx-ai-lab/HydraVox```

---
## Train & Finetune

### 使用 WebUI 进行数据预处理（Data Process）

打开 WebUI 后进入“📊 数据处理”标签页，支持一键流水线或逐阶段运行。

- 一键处理：在“🚀 一键处理 - 自动运行全部四个阶段”中填写“📁 输入目录”“🎤 采样率”“⚠️ 覆盖文件”，点击“🚀 开始一键处理”。界面将依次执行四个阶段并实时显示总体进度、状态与日志。
- 阶段与输出目录约定：
  - 阶段1（格式转换与重采样）→ 输出至 `<输入>_resample`
  - 阶段2（VAD 语音活动检测）→ 输出至 `<阶段1输出>_vad`
  - 阶段3（ASR 转录）→ 输出至 `<阶段2输出>_asr`
  - 阶段4（提取语音训练 Token）→ 输出至 `<阶段3输出>_token`
  - 阶段5（可选，数据集合并）→ 将多个 HF 数据集目录合并保存到指定目录

逐阶段运行要点（对应折叠面板）：
- 阶段 1 - 格式转换与重采样
  - 选择输入/输出目录与采样率（默认 16kHz），支持覆盖已存在文件
  - 通过“👀 预览变更”查看源文件→目标文件映射，点击“▶️ 开始处理”执行
- 阶段 2 - VAD 语音活动检测（Silero）
  - 关键参数：置信度阈值、最短语音/静音(ms)、前后填充(ms)、最短/最长片段(s)
  - 点击“▶️ 开始处理”后显示分段进度与日志
- 阶段 3 - ASR 语音识别转录
  - 设备选择：自动/CPU/GPU；并行进程数（GPU 数）；可刷新设备检测
  - 输出为 HuggingFace 数据集目录
- 阶段 4 - 提取语音训练 Token
  - 设备选择与并行进程配置同上
  - 输出为 Token 化后的数据集目录
- 阶段 5 - 数据集合并（可选）
  - 输入多个数据集目录（英文逗号分隔），合并后保存到目标目录

提示：界面会自动探测 CUDA 并提示 GPU 数；ASR/VAD 默认以 16kHz 处理；每阶段带有状态与日志便于排错。

### 训练（WebUI：🚀 模型训练）

在“🚀 模型训练”标签页完成以下设置并点击“开始训练”：

- 1) 数据集配置
  - 训练数据路径：选择由阶段 3/4 产出的 HF 数据集目录（如 `<...>_asr` 或 `<...>_token`）
- 2) 模型配置
  - 模型类型：`llm` 或 `flow`
  - 模型检查点路径：如 `jzx-ai-lab/HydraVox/llm.pt`
  - 分词器路径：如 `jzx-ai-lab/HydraVox/CosyVoice-BlankEN`
  - 输出目录：如 `checkpoints/training_llm`
- 3) 训练参数
  - 批次大小、学习率、训练轮数、保存间隔(步数)、日志记录间隔(logging_steps)、评估间隔(eval_steps)
  - 验证集：支持按比例自动划分或提供现成验证集（未检测到时自动切换为自动划分）
- 4) 高级选项
  - 启用 LoRA 微调（可选）
  - 精度设置：BF16/FP16（不同模型类型给出推荐）
- 5) 计算资源
  - 计算设备：自动/CPU/GPU；并行进程数（GPU 数）；可指定 GPU IDs 并刷新设备检测
- 6) 训练控制与可视化
  - 日志：面板实时输出，并保存到 `logs/training/train_<timestamp>.log`
  - 曲线：自动生成并保存到 `<输出目录>/figure/training_plot.png`
  - 支持“立即刷新/强制刷新”图表与定时自动刷新

说明：内部使用 Accelerate 启动，按精度设置自动选择 `fp16/bf16`；`logging_steps` 与 `eval_steps` 将体现在日志与曲线的步数刻度上。

### 模型管理

“模型管理”面板提供对训练输出的快捷操作：
- 列表：按时间倒序显示已发现的输出路径（扫描 `checkpoints/training_llm`、`checkpoints/training_flow`、`checkpoints/training`、`checkpoints`、`models`、`outputs`、`ckpt` 等目录，自动忽略 `runs/logs/figure`）
- 刷新列表：重新扫描目录
- 加载路径：将选中的路径回显到输入框（便于后续操作）
- 删除路径：危险操作，直接删除所选文件夹（限制删除系统文件夹如 `runs/logs/figure`）
- 转换为 `model.pt (bf16)`：将目录下的 `pytorch_model.bin` 转换为 `model.pt`
  - 不支持分片索引 `.bin.index.json`，需先合并再转换

注意：大规模处理与训练前请确保磁盘空间与 GPU 资源充足；若训练结束未自动刷新曲线，可点击“⚡ 强制刷新”。

---

## License

* Code: **Apache‑2.0** *(example — update to your actual license)*.
* **Derived from CosyVoice** — see `NOTICE` and `LICENSE-THIRD-PARTY` for upstream licenses and modifications.

---

## 📚 Citation

```bibtex
@software{hydravox2025,
  title = {FlowMirror-HydraVox: Multi-head AR TTS with Native Acceleration},
  author = {Your Name and Contributors},
  year = {2025},
  url = {https://github.com/your-org/FlowMirror-HydraVox}
}
```

---

## 🙏 Acknowledgements

* [**CosyVoice**](https://github.com/FunAudioLLM/CosyVoice) authors and contributors.
* [**Better & Faster Large Language Models via Multi-token Prediction**](https://arxiv.org/abs/2404.19737)
