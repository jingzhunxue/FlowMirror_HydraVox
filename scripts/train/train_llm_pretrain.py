#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专用：LLM 预训练脚本（CosyVoice3LM MTP pretrain）

参考：scripts/train/train_speech_model.py

特点：
- 只训练 LLM（不包含 flow/gan）
- 从 hyperpyyaml 构建模型（推荐使用 pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3_mtp_pretrain.yaml）
- 允许从旧 ckpt 加载（strict=False），用于首次引入 mtp_block 的权重补齐后继续训练
- instruct_token / instruct_token_len 暂时不需要：在 data_collator 中自动补空占位，避免 forward 取 key 报错
- dataset 仅有 text/audio：speech_token 在 collator 中使用 ONNX 从音频实时提取
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
import torchaudio
import whisper
from datasets import load_from_disk, concatenate_datasets, Dataset, Audio
from hyperpyyaml import load_hyperpyyaml
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase

# 添加项目根目录到Python路径（与 train_speech_model.py 保持一致）
project_root = Path(__file__).parent.parent.parent.absolute()
third_party_dir = project_root / "server/model_utils"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(third_party_dir))

from server.model_utils.cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer


USEFUL_COLUMNS_LLM = ["text", "audio"]


def prepare_dataset_llm_pretrain(ds: Dataset) -> Dataset:
    """保留 LLM pretrain 所需列，减少 IO/内存。"""
    ds = ds.remove_columns([c for c in ds.column_names if c not in USEFUL_COLUMNS_LLM])
    ds = ds.cast_column("audio", Audio(decode=True, sampling_rate=16000))
    return ds


_TOKENIZER_SESSION: ort.InferenceSession | None = None
_TOKENIZER_SESSION_KEY: str | None = None


def _get_onnx_tokenizer_session(
    onnx_path: str,
    use_cuda: bool,
    device_id: int,
    intra_op_num_threads: int = 1,
) -> ort.InferenceSession:
    """每个进程懒加载一份 ONNX Session（DataLoader worker 各自初始化）。"""
    global _TOKENIZER_SESSION, _TOKENIZER_SESSION_KEY
    # 若环境不支持 CUDA provider，这里会自动回退到 CPU provider
    available = set(ort.get_available_providers())
    effective_use_cuda = bool(use_cuda) and ("CUDAExecutionProvider" in available)
    if bool(use_cuda) and not effective_use_cuda:
        logging.warning(
            "onnxruntime 未检测到 CUDAExecutionProvider（available=%s），将自动使用 CPUExecutionProvider。",
            ",".join(sorted(available)),
        )

    key = f"{onnx_path}|cuda={effective_use_cuda}|dev={device_id}|intra={intra_op_num_threads}"
    if _TOKENIZER_SESSION is not None and _TOKENIZER_SESSION_KEY == key:
        return _TOKENIZER_SESSION

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = int(intra_op_num_threads)

    if effective_use_cuda:
        # providers = [("CUDAExecutionProvider", {"device_id": int(device_id)})]
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    _TOKENIZER_SESSION = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    _TOKENIZER_SESSION_KEY = key
    return _TOKENIZER_SESSION


def _load_audio_16k_mono(audio_info: Any) -> torch.Tensor:
    """datasets.Audio decode 后的 dict（array/sampling_rate）→ (1,T) float32 16k mono"""
    if isinstance(audio_info, dict) and "array" in audio_info:
        wav = torch.tensor(audio_info["array"], dtype=torch.float32)
        sr = int(audio_info.get("sampling_rate", 16000))
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        return wav

    if isinstance(audio_info, dict) and "path" in audio_info:
        path = audio_info["path"]
    else:
        path = str(audio_info)
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav.to(torch.float32)


def _extract_speech_token_from_audio(audio_info: Any, onnx_session: ort.InferenceSession) -> List[int]:
    wav_16k = _load_audio_16k_mono(audio_info)
    mel = whisper.log_mel_spectrogram(wav_16k, n_mels=128)
    ort_inputs = {
        onnx_session.get_inputs()[0].name: mel.cpu().numpy(),
        onnx_session.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32),
    }
    tokens = onnx_session.run(None, ort_inputs)[0].flatten().tolist()
    return tokens


class LlmPretrainDataCollator:
    """只构造 CosyVoice3LM.forward 所需字段（并补齐 instruct_* 占位）。"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None,
        tokenizer_onnx_path: str,
        onnx_use_cuda: bool,
        onnx_device_id: int,
        ort_intra_op_num_threads: int = 1,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_onnx_path = tokenizer_onnx_path
        self.onnx_use_cuda = onnx_use_cuda
        self.onnx_device_id = onnx_device_id
        self.ort_intra_op_num_threads = ort_intra_op_num_threads

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}

        # -------- text_token / text_token_len --------
        text_tokens: List[torch.Tensor] = []
        text_token_lens: List[int] = []
        if "text_token" in features[0]:
            for f in features:
                tt = f["text_token"]
                if not isinstance(tt, torch.Tensor):
                    tt = torch.tensor(tt, dtype=torch.long)
                tt = tt.long()
                text_tokens.append(tt)
                text_token_lens.append(int(tt.numel()))
        elif "text" in features[0]:
            if self.tokenizer is None:
                raise ValueError("数据只有 text 字段但未提供 tokenizer，无法生成 text_token。")
            for f in features:
                ids = self.tokenizer.encode(f["text"], allowed_special="all")
                tt = torch.tensor(ids, dtype=torch.long)
                text_tokens.append(tt)
                text_token_lens.append(int(tt.numel()))
        else:
            raise ValueError("LLM pretrain 需要 text_token 或 text 字段。")

        batch["text_token"] = pad_sequence(text_tokens, batch_first=True, padding_value=0)
        batch["text_token_len"] = torch.tensor(text_token_lens, dtype=torch.int64)

        # -------- speech_token / speech_token_len（从音频实时提取）--------
        if "audio" not in features[0]:
            raise ValueError("LLM pretrain 需要 audio 字段以实时提取 speech_token。")

        onnx_session = _get_onnx_tokenizer_session(
            self.tokenizer_onnx_path,
            use_cuda=self.onnx_use_cuda,
            device_id=self.onnx_device_id,
            intra_op_num_threads=self.ort_intra_op_num_threads,
        )
        speech_tokens: List[torch.Tensor] = []
        speech_token_lens: List[int] = []
        for f in features:
            tokens = _extract_speech_token_from_audio(f["audio"], onnx_session)
            st = torch.tensor(tokens, dtype=torch.long)
            speech_tokens.append(st)
            speech_token_lens.append(int(st.numel()))
        batch["speech_token"] = pad_sequence(speech_tokens, batch_first=True, padding_value=0)
        batch["speech_token_len"] = torch.tensor(speech_token_lens, dtype=torch.int64)

        # -------- instruct_token / instruct_token_len（暂时不需要，补空占位）--------
        bsz = len(features)
        batch["instruct_token"] = torch.zeros((bsz, 0), dtype=torch.long)
        batch["instruct_token_len"] = torch.zeros((bsz,), dtype=torch.int64)

        return batch


class _TrainerForwardWrapper(nn.Module):
    """
    适配 HuggingFace Trainer 的调用方式：Trainer 会调用 model(**batch)。
    但 CosyVoice3LM.forward 期望 forward(batch: dict, device: torch.device)。
    """

    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core_model = core_model

    def forward(self, **batch):  # type: ignore[override]
        # Trainer 在 _prepare_inputs 后，batch tensor 已经在正确 device 上
        any_tensor = next((v for v in batch.values() if isinstance(v, torch.Tensor)), None)
        device = any_tensor.device if any_tensor is not None else next(self.core_model.parameters()).device
        return self.core_model(batch, device)


def _load_state_dict_maybe_container(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("不支持的 checkpoint 格式：期望为 state_dict 或 {'state_dict': ...}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3_mtp_pretrain.yaml",
        help="hyperpyyaml 配置路径",
    )
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径（load_from_disk），逗号分隔")
    parser.add_argument("--cv_data", type=str, default="", help="验证数据路径（可选），逗号分隔；为空则不评估")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="",
        help="初始模型 checkpoint（state_dict 或 {'state_dict':...}）。若指定 --resume_from_checkpoint，可不传。",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="从 HuggingFace Trainer 的 checkpoint 目录断点续训（如 output_dir/checkpoint-10000）。传空则不启用。",
    )
    parser.add_argument("--qwen_pretrain_path", type=str, default="", help="Qwen2Encoder 的 pretrain_path / tokenizer 路径")
    parser.add_argument(
        "--tokenizer_onnx_path",
        type=str,
        default="jzx-ai-lab/HydraVox/speech_tokenizer_v2.onnx",
        help="speech tokenizer ONNX 路径（从音频实时提取 speech_token）",
    )
    parser.add_argument("--onnx_use_cuda", action="store_true", default=True, help="ONNX tokenizer 是否使用 CUDAExecutionProvider")
    parser.add_argument("--onnx_device_id", type=int, default=None, help="ONNX CUDA device_id（默认取 LOCAL_RANK/RANK，否则 0）")
    parser.add_argument("--ort_intra_op_num_threads", type=int, default=1)

    # 允许用 yaml 的 train_conf 作为默认值；CLI 指定则覆盖
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default=None)
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("🚀 LLM pretrain 脚本启动")

    # 1) load yaml & build model
    with open(args.config, "r") as f:
        if not args.qwen_pretrain_path:
            model_dir = os.getenv("TTS_MODEL_DIR", "jzx-ai-lab/HydraVox")
            args.qwen_pretrain_path = os.path.join(model_dir, "CosyVoice-BlankEN")
        cfg = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": args.qwen_pretrain_path,
            },
        )

    model = cfg["llm"]

    # 2) load ckpt / resume
    resume_path = str(args.resume_from_checkpoint).strip()
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume_from_checkpoint 路径不存在：{resume_path}")
        if not os.path.isdir(resume_path):
            raise ValueError(f"--resume_from_checkpoint 需要传 checkpoint 目录（如 checkpoint-10000），但得到：{resume_path}")
        logging.info("将从 Trainer checkpoint 断点续训：%s", resume_path)
    else:
        if not str(args.model_ckpt).strip():
            raise ValueError("未指定 --resume_from_checkpoint 时，必须提供 --model_ckpt 作为初始权重。")
        model_state = _load_state_dict_maybe_container(args.model_ckpt)
        # 兼容 train_speech_model.py 的 ckpt 里可能带 epoch/step
        model_state.pop("epoch", None)
        model_state.pop("step", None)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            logging.warning(f"load_state_dict missing keys: {len(missing)}（示例：{missing[:5]}）")
        if unexpected:
            logging.warning(f"load_state_dict unexpected keys: {len(unexpected)}（示例：{unexpected[:5]}）")

    # 3) dataset
    train_paths = [p for p in args.train_data.split(",") if p.strip()]
    train_dss = [prepare_dataset_llm_pretrain(load_from_disk(p)) for p in train_paths]
    train_dataset = concatenate_datasets(train_dss).shuffle(seed=42)

    eval_dataset = None
    if args.cv_data.strip():
        val_paths = [p for p in args.cv_data.split(",") if p.strip()]
        val_dss = [prepare_dataset_llm_pretrain(load_from_disk(p)) for p in val_paths]
        eval_dataset = concatenate_datasets(val_dss).shuffle(seed=42)

    # 4) tokenizer（仅当数据里是 text 而非 text_token 时才需要）
    tokenizer = get_qwen_tokenizer(token_path=args.qwen_pretrain_path, skip_special_tokens=True, version="cosyvoice3")

    # 5) training args（从 yaml.train_conf 取默认）
    train_conf = cfg.get("train_conf", {}) if isinstance(cfg, dict) else {}
    optim_conf = train_conf.get("optim_conf", {}) if isinstance(train_conf, dict) else {}
    scheduler_conf = train_conf.get("scheduler_conf", {}) if isinstance(train_conf, dict) else {}

    learning_rate = args.learning_rate if args.learning_rate is not None else float(optim_conf.get("lr", 1e-5))
    num_train_epochs = args.num_train_epochs if args.num_train_epochs is not None else int(train_conf.get("max_epoch", 1))
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else int(train_conf.get("accum_grad", 1))
    )
    per_device_train_batch_size = args.per_device_train_batch_size if args.per_device_train_batch_size is not None else int(train_conf.get("batch_size", 8))
    per_device_eval_batch_size = args.per_device_eval_batch_size if args.per_device_eval_batch_size is not None else 1
    max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else float(train_conf.get("grad_clip", 1.0))
    logging_steps = args.logging_steps if args.logging_steps is not None else int(train_conf.get("log_interval", 50))
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(scheduler_conf.get("warmup_steps", 0))
    save_steps = args.save_steps if args.save_steps is not None else int(train_conf.get("save_per_step", 2000))

    # scheduler: yaml 里 constantlr，这里映射到 HF 的 constant/constant_with_warmup
    scheduler_name = str(train_conf.get("scheduler", "linear")).lower()
    if scheduler_name in ("constantlr", "constant"):
        lr_scheduler_type = "constant_with_warmup" if warmup_steps and warmup_steps > 0 else "constant"
    else:
        lr_scheduler_type = "linear"

    if save_steps <= 0:
        save_strategy = "no"
        save_steps = 0
    else:
        save_strategy = "steps"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="no" if eval_dataset is None else "steps",
        eval_steps=1000,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        dataloader_num_workers=8,
        remove_unused_columns=False,  # we supply our own collator
        save_safetensors=False,  # 关闭safetensors，避免共享权重保存报错
    )

    # 6) collator
    if args.onnx_device_id is None:
        env_rank = os.getenv("LOCAL_RANK", os.getenv("RANK", "0"))
        args.onnx_device_id = int(env_rank) if str(env_rank).isdigit() else 0

    data_collator = LlmPretrainDataCollator(
        tokenizer=tokenizer,
        tokenizer_onnx_path=args.tokenizer_onnx_path,
        onnx_use_cuda=bool(args.onnx_use_cuda),
        onnx_device_id=int(args.onnx_device_id),
        ort_intra_op_num_threads=int(args.ort_intra_op_num_threads),
    )

    # 7) trainer
    # 适配 Trainer 的 forward(**batch) 调用
    model_for_trainer = _TrainerForwardWrapper(model)

    trainer = Trainer(
        model=model_for_trainer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=None,  # 本训练不依赖 HF tokenizer 自动处理，避免额外副作用
    )

    logging.info("Training...")
    trainer.train(resume_from_checkpoint=resume_path or None)
    logging.info("Training completed. Saving model...")
    trainer.save_model()
    logging.info("All Finished.")


if __name__ == "__main__":
    main()


