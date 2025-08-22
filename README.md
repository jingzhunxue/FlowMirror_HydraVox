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
* **Ready‑to‑use WebUI** — Inference, flash batch synthesis(comming soon), fine‑tuning(comming soon), speaker mgmt, logs/plots.
* **Hot-load LoRA for emotion/style** — Load/unload adapters at runtime per utterance; stack multiple with per-adapter scaling (e.g. `--lora happy.safetensors:0.6,energetic.safetensors:0.3`).
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

* Text → Speech (single/batch), long‑text chunking, punctuation & pause control.
* **Training/Finetune panel**: dataset browser, configs, live logs & curves.
* Speaker manager: add/rename/delete speakers, preview, embeddings.
* Export WAV/FLAC; optional watermarking; history with reproducible seeds.

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
  - [ ] Release training ui tab and training scripts
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

## 🧪 Evaluation

```bash
# Real‑time factor (RTF) & latency
python eval/measure_rtf.py --textfile assets/bench.txt --multihead 4

# Quality proxies (MOS‑net, WER/CER where applicable)
python eval/quality.py --set dev
```

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

## 🧩 Compatibility & Extensions

* Vocoders: HiFi‑GAN (default), NSF, codec decoders (plug‑in interface at `hydravox/vocoder/`).
* Frontend: grapheme → phoneme pluggable; custom prosody models supported.
* Works with CPU (reduced speed) and NVIDIA GPUs; AMD/Metal support is experimental.

---

## ❓ FAQ

**Q: CPU‑only works?**  Yes, set `device=cpu` (much slower).

**Q: How to stabilize very long texts?**  Lower `temperature`, raise `stability_penalty`, enable `--sil-merge-ms` and chunk by sentences.

**Q: Voice cloning?**  If enabled, use the Speaker tab to add a voice with consent. See `docs/clone.md` (TBD).

**Q: WebUI shows `TrustedHTML` error?**  Use modern browsers or update Gradio; if reverse proxying, ensure CSP allows inline scripts or disable strict CSP in dev.

---

## 🐛 Troubleshooting

* **FFmpeg not found** → install via `apt/yum/brew` and ensure it is in `PATH`.
* **CUDA OOM** → reduce batch, heads, or enable half precision.
* **Mixed CUDA versions** → match PyTorch CUDA with system CUDA runtime.
* **WebUI keeps loading** → check console for CSP/`TrustedHTML` warnings; see `docs/webui.md`.
* **Chinese mirrors** → set `PIP_INDEX_URL` to Aliyun for faster deps.

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

* **CosyVoice** authors and contributors.
* Open-source community for phonemizers, vocoders, and tooling.
