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
  <a href="#-webui">WebUI</a> •
  <a href="#-python-api">Python API</a> •
  <a href="#-models--weights">Models</a> •
  <a href="#-train--finetune">Train</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-troubleshooting">Troubleshooting</a> •
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

* **Multi‑Head AR Decoding** — Predict multiple speech tokens per step → **lower latency** and **higher throughput** under the same even better quality target.
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

## 🖥️ WebUI

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

Screenshots (TBD):

* `docs/assets/ui-home.png`
* `docs/assets/ui-train.png`
* `docs/assets/ui-settings.png`

**Data directories** (defaults, overridable):

```
jzx-ai-lab/HydraVox      # model weights
logs/             # train/infer logs
```

---

## 🗺️ Roadmap

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

## 🐍 Python API

```python

```

**Key arguments**


---

## 🧠 Models & Weights

| Name                  | Params | Langs   | Vocoder   | Multi‑Head | Link  | SHA256 |
| --------------------- | -----: | ------- | --------- | ---------: | ----- | ------ |
| hydravox-base-pretrained         | \~300M | zh/en   | HiFi‑GAN  |          4 | (TBD) | (TBD)  |

> Place weights in `checkpoints/<model_name>/` or set `--ckpt_dir`.

---

## 🛠️ Train & Finetune

### Data

* **Audio**: mono WAV/FLAC, 24 kHz recommended, peak‑normalized.
* **Transcripts**: JSONL or TSV. Example JSONL row:

```json
{"audio": "data/wavs/0001.wav", "text": "今天天气真好。", "speaker": "spk1", "lang": "zh"}
```

* Optional alignments/phonemes supported; see `docs/data.md` (TBD).

Preprocessing:

```bash
python scripts/prepare_data.py --in raw/ --out data/ --sr 24000 --vad --split 5.0
```

### Single‑node training

```bash
python train.py -c configs/train/base_4head.yaml \
    data.train=data/train.jsonl \
    data.val=data/val.jsonl \
    trainer.precision=bf16 \
    optim.lr=2e-4
```

### Distributed (multi‑GPU)

```bash
torchrun --nproc_per_node=4 train.py -c configs/train/base_4head.yaml \
    trainer.devices=4 trainer.strategy=ddp
```

**Key knobs**

* `model.num_heads`: decoding heads.
* `loss.token_fuse`: multi‑head fusion objective.
* `sched.warmup_steps`, `grad_clip`, `ema`.

**Checkpoints**

* Saved under `checkpoints/exp_name/` with `last.ckpt`, `epoch-*.ckpt`.

---

## 🧩 Architecture

* **Encoder** → **AR decoder (multi‑head)** → **token fusion** → **vocoder**.


**Design trade‑offs**


**Limitations (current)**


---

## 🧭 Repository Map

```
assets/           # logo, samples, demo texts
benchmarks/       # scripts to reproduce speed/RTF numbers
configs/          # YAML configs for train/infer
examples/         # minimal Python examples and CLI wrappers
eval/             # evaluation utilities (RTF, MOS-like, ASR-based)
hydravox/         # core library
scripts/          # data/ckpt helpers, downloaders
webui/            # Gradio app and REST server
```

---

## 🛡️ Safety & Responsible Use

* Do **not** impersonate real people without explicit consent.
* Respect local laws and data/privacy regulations.
* Optional **watermarking** and **speaker whitelist** are available in WebUI/CLI.

---

## 🤝 Contributing

PRs welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and follow our code style & test guidelines. Good first issues are labeled `good-first-issue`.

---

## 📜 License

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
