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
import random
import re
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

from fmtn import create_default_tn

try:
    from user_interface.i18n import t
except Exception:
    def t(text: str, **kwargs):
        if kwargs:
            try:
                return text.format(**kwargs)
            except Exception:
                return text
        return text

tn = create_default_tn(verbose=True)

USEFUL_COLUMNS_LLM = ["text", "audio"]


def _get_added_special_tokens(tokenizer: Any) -> set[str]:
    """
    从 CosyVoice2/3Tokenizer 中取出 additional_special_tokens 集合。
    兼容：tokenizer 可能是自定义 wrapper（CosyVoice3Tokenizer），也可能是 HF tokenizer。
    """
    st = getattr(tokenizer, "special_tokens", None)
    if isinstance(st, dict) and isinstance(st.get("additional_special_tokens"), list):
        return set(st["additional_special_tokens"])
    # fallback: HF tokenizer added vocab
    tok = getattr(tokenizer, "tokenizer", None)
    if tok is not None and hasattr(tok, "get_added_vocab"):
        try:
            return set(tok.get_added_vocab().keys())
        except Exception:
            return set()
    return set()


_RE_EN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_RE_ZH_CHAR = re.compile(r"[\u4e00-\u9fff]")


def _maybe_append_en_cmu_tokens(text: str, special_tokens: set[str]) -> str:
    """
    若文本存在英文单词：随机选 1 个英文单词，转为 CMU(ARPABET) 音素序列，并**替换原单词位置**为 tokenizer special token 形式：
    e.g. "hello" -> "[HH] [AH0] [L] [OW1]"
    依赖：cmudict（可选）。若不可用或查不到发音则返回原文本（不修改）。
    """
    matches = list(_RE_EN_WORD.finditer(text))
    if not matches:
        return text

    # 随机挑 2 个英文单词（不足 2 个则全选）
    picks = random.sample(matches, k=min(2, len(matches)))

    def phones_for_word(w: str) -> List[str]:
        w = w.lower()
        # 1) cmudict（若可用）
        try:
            import cmudict  # type: ignore

            d = cmudict.dict()
            prons = d.get(w)
            if prons:
                return list(prons[0])
        except Exception:
            pass
        # 2) pronouncing（常见，封装了 CMUdict）
        try:
            import pronouncing  # type: ignore

            ps = pronouncing.phones_for_word(w)
            if ps:
                return ps[0].strip().split()
        except Exception:
            pass
        return []

    replacements: List[tuple[int, int, str]] = []
    for m in picks:
        word = m.group(0)
        phones = phones_for_word(word)
        if not phones:
            continue
        toks: List[str] = []
        for p in phones:
            tok = f"[{p}]"
            if tok in special_tokens:
                toks.append(tok)
        if not toks:
            continue
        # special token 之间不加空格；但整体两侧加空格避免粘连
        rep = " " + "".join(toks) + " "
        replacements.append((m.start(), m.end(), rep))

    if not replacements:
        return text

    # 从后往前替换，避免索引偏移
    replacements.sort(key=lambda x: x[0], reverse=True)
    for s, e, rep in replacements:
        text = text[:s] + rep + text[e:]
    return text


def _maybe_append_zh_pinyin_tokens(text: str, special_tokens: set[str]) -> str:
    """
    从文本中随机选择 2 个中文汉字，转拼音并拆分声母/韵母，并**替换原汉字位置**为 tokenizer special token：
    e.g. "中国" -> "[zh] [ōng] [g] [uó]"（示例，具体取决于 pypinyin 输出）
    依赖：pypinyin（可选）。若不可用/中文字符不足则返回原文本（不修改）。
    """
    matches = list(_RE_ZH_CHAR.finditer(text))
    if len(matches) < 2:
        return text
    picks = random.sample(matches, k=2)
    try:
        from pypinyin import Style, pinyin  # type: ignore
    except Exception:
        return text

    replacements: List[tuple[int, int, str]] = []
    for m in picks:
        ch = m.group(0)
        try:
            ini = pinyin(ch, style=Style.INITIALS, strict=False, heteronym=False)[0][0] or ""
            fin = pinyin(ch, style=Style.FINALS_TONE, strict=False, heteronym=False)[0][0] or ""
        except Exception:
            continue

        # 仅当 token 在 tokenizer 中注册时才使用，避免产生无效 token
        toks: List[str] = []
        if ini:
            t_ini = f"[{ini.lower()}]"
            if t_ini in special_tokens:
                toks.append(t_ini)
        if fin:
            t_fin = f"[{fin.lower()}]"
            if t_fin in special_tokens:
                toks.append(t_fin)
        if toks:
            # special token 之间不加空格；但整体两侧加空格避免粘连
            rep = " " + "".join(toks) + " "
            replacements.append((m.start(), m.end(), rep))

    if not replacements:
        return text

    # 从后往前替换，避免索引偏移
    replacements.sort(key=lambda x: x[0], reverse=True)
    for s, e, rep in replacements:
        text = text[:s] + rep + text[e:]
    return text


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
            t("train.onnx_no_cuda", providers=",".join(sorted(available))),
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
    max_mel_frames = 2048
    if mel.shape[2] > max_mel_frames:
        mel = mel[:, :, :max_mel_frames]
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
                raise ValueError(t("train.text_tokenizer_missing"))
            special_tokens = _get_added_special_tokens(self.tokenizer)
            for f in features:
                try:
                    text = tn.process_text(f["text"])
                except Exception as e:
                    print(f"Error processing text: {e}")
                    text = f["text"]

                # 优先：若存在英文单词，随机抽 1 个转 CMU 音素；否则随机抽 2 个中文汉字转拼音(声母/韵母)
                new_text = _maybe_append_en_cmu_tokens(text, special_tokens)
                if new_text == text:
                    new_text = _maybe_append_zh_pinyin_tokens(text, special_tokens)
                text = new_text
                if os.getenv("DEBUG_TEXT_AUG", "0") == "1":
                    print(text)
                ids = self.tokenizer.encode(text, allowed_special="all")
                tt = torch.tensor(ids, dtype=torch.long)
                text_tokens.append(tt)
                text_token_lens.append(int(tt.numel()))
        else:
            raise ValueError(t("train.llm_text_required"))

        batch["text_token"] = pad_sequence(text_tokens, batch_first=True, padding_value=0)
        batch["text_token_len"] = torch.tensor(text_token_lens, dtype=torch.int64)

        # -------- speech_token / speech_token_len（从音频实时提取）--------
        if "audio" not in features[0]:
            raise ValueError(t("train.llm_audio_required"))

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
    raise ValueError(t("train.ckpt_format_invalid"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3_mtp_pretrain.yaml",
        help=t("train.cli_config"),
    )
    parser.add_argument("--train_data", type=str, required=True, help=t("train.cli_train_data"))
    parser.add_argument("--cv_data", type=str, default="", help=t("train.cli_cv_data"))
    parser.add_argument("--output_dir", type=str, required=True, help=t("train.cli_output_dir"))
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="",
        help=t("train.cli_model_ckpt"),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help=t("train.cli_resume"),
    )
    parser.add_argument("--qwen_pretrain_path", type=str, default="", help=t("train.cli_qwen_pretrain"))
    parser.add_argument(
        "--tokenizer_onnx_path",
        type=str,
        default="jzx-ai-lab/HydraVox-CV3/speech_tokenizer_v3.onnx",
        help=t("train.cli_tokenizer_onnx"),
    )
    parser.add_argument("--onnx_use_cuda", action="store_true", default=True, help=t("train.cli_onnx_use_cuda"))
    parser.add_argument("--onnx_device_id", type=int, default=None, help=t("train.cli_onnx_device_id"))
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
    logging.info(t("train.llm_pretrain_start"))

    # 1) load yaml & build model
    with open(args.config, "r") as f:
        if not args.qwen_pretrain_path:
            model_dir = os.getenv("TTS_MODEL_DIR", "jzx-ai-lab/HydraVox-CV3")
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
            raise FileNotFoundError(t("train.resume_not_found", path=resume_path))
        if not os.path.isdir(resume_path):
            raise ValueError(t("train.resume_not_dir", path=resume_path))
        logging.info(t("train.resume_from", path=resume_path))
    else:
        if not str(args.model_ckpt).strip():
            raise ValueError(t("train.model_ckpt_required"))
        model_state = _load_state_dict_maybe_container(args.model_ckpt)
        # 兼容 train_speech_model.py 的 ckpt 里可能带 epoch/step
        model_state.pop("epoch", None)
        model_state.pop("step", None)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            logging.warning(t("train.missing_keys", count=len(missing), sample=missing[:5]))
        if unexpected:
            logging.warning(t("train.unexpected_keys", count=len(unexpected), sample=unexpected[:5]))

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
