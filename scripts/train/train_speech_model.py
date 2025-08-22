# -*- coding: utf-8 -*-
"""train_with_trainer.py

Rewrite of the original custom training loop to leverage Hugging Face 🤗 `Trainer` & `TrainingArguments`.
The script keeps most domain-specific preprocessing logic (tokenization, audio → mel, etc.) but
outsources distributed training, mixed precision, gradient accumulation, DeepSpeed integration,
checkpointing and tensorboard logging to the Transformers ecosystem.

Key features
------------
* Works with either the LLM or FLOW model defined in your YAML (`configs[args.model]`).
* Uses `datasets` map-style preprocessing so that all ranks benefit from cached results.
* `Trainer` handles ➔ fp16/bf16, ZeRO/DeepSpeed, DDP, gradient accumulation, evaluation, saving …
* Metric computation stub provided – adapt for your task.

Typical usage
-------------
python -m torch.distributed.run --nproc_per_node 8 train_with_trainer.py \
  --config path/to/config.yaml \
  --model llm \
  --train_data /path/train_ds \
  --cv_data /path/val_ds \
  --output_dir ckpt/llm_trainer \
  --deepspeed path/to/ds_config.json  # optional

See the ArgumentParser at the bottom for the full set of CLI switches.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Union, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.absolute()
third_party_dir = project_root / "third_party"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(third_party_dir))

import numpy as np
import torch
import torchaudio
from datasets import load_from_disk, concatenate_datasets, Dataset, Audio
from hyperpyyaml import load_hyperpyyaml
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    default_data_collator,
)

# ---------- Domain-specific imports ---------- #
from scripts.third_party.cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer  
from scripts.third_party.matcha.utils.audio import mel_spectrogram

# -----------------------------------------------------------------------------
# Data preparation helpers
# -----------------------------------------------------------------------------

USEFUL_COLUMNS_LLM = ["text", "speech_token"]
USEFUL_COLUMNS_FLOW = ["speech_token", "audio", "embedding"]


def prepare_dataset_llm(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Tokenise text + keep speech tokens; filter by length."""

    def _map(example):
        tokens = tokenizer.encode(example["text"], allowed_special="all")
        text_token = torch.tensor(tokens, dtype=torch.int32)
        text_token_len = len(text_token)
        speech_token = torch.tensor(example["speech_token"], dtype=torch.int32)
        speech_token_len = len(speech_token)
        return {
            "text_token": text_token,
            "text_token_len": text_token_len,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
        }

    ds = ds.remove_columns([c for c in ds.column_names if c not in USEFUL_COLUMNS_LLM])
    ds = ds.map(_map, remove_columns=["text", "speech_token"], num_proc=os.cpu_count())
    ds = ds.filter(lambda ex: ex["speech_token_len"] > 25)
    return ds


def _audio_to_features(audio_dict, target_sr: int = 24000):
    wav = torch.from_numpy(audio_dict["array"]).unsqueeze(0).float()
    if audio_dict["sampling_rate"] != target_sr:
        wav = torchaudio.transforms.Resample(audio_dict["sampling_rate"], target_sr)(wav)
    estimated_frames = (wav.shape[-1] - 480) // 480 + 1
    if estimated_frames % 2 == 1:
        # 如果预估长度是奇数，调整音频长度
        padding_needed = 480  # 添加一个hop_size的长度
        wav = torch.nn.functional.pad(wav, (0, padding_needed))
    mel = mel_spectrogram(wav, 1920, 80, target_sr, 480, 1920, 0, 8000, False)
    mel = mel.squeeze(0).transpose(0, 1)
    return mel


def prepare_dataset_flow(ds: Dataset) -> Dataset:
    """Keep raw data as-is, defer all processing to collator for maximum speed."""
    
    # 只保留必要的列，不做任何数据转换
    ds = ds.remove_columns([c for c in ds.column_names if c not in USEFUL_COLUMNS_FLOW])
    
    return ds


# -----------------------------------------------------------------------------
# Data collator (pads variable-length tensors manually to keep Trainer happy)
# -----------------------------------------------------------------------------

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch = {}
    
    # 处理音频数据（如果存在）
    if "audio" in features[0]:
        # 边训边处理：在这里进行音频到mel谱图的转换
        speech_feats = []
        speech_feat_lens = []
        
        for f in features:
            mel = _audio_to_features(f["audio"]).to(torch.bfloat16)
            speech_feats.append(mel)
            speech_feat_lens.append(len(mel))
        
        # 对mel谱图进行padding
        batch["speech_feat"] = pad_sequence(speech_feats, batch_first=True)
        batch["speech_feat_len"] = torch.tensor(speech_feat_lens)
    
    # 处理 text_token
    if "text_token" in features[0]:
        text_tokens = []
        text_token_lens = []
        for f in features:
            text_token = f["text_token"]
            if not isinstance(text_token, torch.Tensor):
                text_token = torch.tensor(text_token, dtype=torch.int32)
            else:
                text_token = text_token.int()
            text_tokens.append(text_token)
            text_token_lens.append(len(text_token))
        batch["text_token"] = pad_sequence(text_tokens, batch_first=True)
        batch["text_token_len"] = torch.tensor(text_token_lens)


    # 处理speech_token数据
    if "speech_token" in features[0]:
        speech_tokens = []
        speech_token_lens = []
        
        for f in features:
            # 在这里进行数据类型转换
            speech_token = torch.tensor(f["speech_token"], dtype=torch.int32)
            speech_tokens.append(speech_token)
            speech_token_lens.append(len(speech_token))
        
        batch["speech_token"] = pad_sequence(speech_tokens, batch_first=True)
        batch["speech_token_len"] = torch.tensor(speech_token_lens)
    
    # 处理speaker embedding
    if "embedding" in features[0]:
        embeddings = []
        for f in features:
            emb = torch.tensor(f["embedding"], dtype=torch.bfloat16)
            embeddings.append(emb)
        batch["embedding"] = torch.stack(embeddings)
    
    # 处理其他数据
    for key in features[0].keys():
        if key in ["audio", "speech_token", "embedding"]:  # 跳过已处理的数据
            continue
        if isinstance(features[0][key], torch.Tensor):
            batch[key] = pad_sequence([f[key] for f in features], batch_first=True)
        else:
            batch[key] = torch.tensor([f[key] for f in features])
    return batch


# -----------------------------------------------------------------------------
# Metric stub – adapt to your task
# -----------------------------------------------------------------------------
from sklearn.metrics import accuracy_score  # 例如你想加个 accuracy，可以按需替换
from transformers import EvalPrediction


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    # ---- high-level script args ---- #
    parser.add_argument("--config", type=str, required=True, help="YAML containing model + train cfg")
    parser.add_argument("--model", choices=["llm", "flow"], required=True)
    parser.add_argument("--model_ckpt", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="tokenizer path")
    parser.add_argument("--train_data", type=str, required=True, help="comma-separated dataset paths")
    parser.add_argument("--cv_data", type=str, required=False, help="comma-separated validation dataset paths")
    parser.add_argument("--output_dir", type=str, required=True)

    # ---- pass-through TrainingArguments ---- #
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed json config path")

    parser.add_argument("--enable_lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--lora_target_modules", type=list, default=["q_proj", "v_proj", "k_proj"])

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--auto_val_split", action="store_true", default=False, help="自动划分验证集")
    parser.add_argument("--val_split_ratio", type=float, default=0.05, help="验证集比例")

    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Loading config …")

    # ------------------------------------------------------------------
    # Build model from YAML (keeps original cosyvoice behaviour)
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = load_hyperpyyaml(f, overrides={})
    model = cfg[args.model]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    try:
        logging.info("Preprocessing datasets …")
        
        # 参数验证：非自动划分时需要cv_data
        if not args.auto_val_split and (not args.cv_data or not args.cv_data.strip()):
            raise ValueError("当未启用自动划分验证集时，--cv_data 参数是必需的")
            
        train_paths = args.train_data.split(",")
        tokenizer = get_qwen_tokenizer(
            token_path=args.tokenizer_path, skip_special_tokens=True
        ) if args.model == "llm" else None
        model_state = torch.load(args.model_ckpt)
        if args.model == "llm":
            if 'epoch' in model_state:
                model_state.pop('epoch')
            if 'step' in model_state:
                model_state.pop('step')
        model.load_state_dict(model_state)

        if args.auto_val_split:
            # 只加载训练集，自动划分
            train_dss = [load_from_disk(p) for p in train_paths]
            if args.model == "llm":
                train_dss = [prepare_dataset_llm(ds, tokenizer) for ds in train_dss]
            else:
                train_dss = [prepare_dataset_flow(ds) for ds in train_dss]
            full_dataset = concatenate_datasets(train_dss).shuffle(seed=42)
            val_size = int(len(full_dataset) * args.val_split_ratio)
            train_size = len(full_dataset) - val_size
            train_dataset = full_dataset.select(range(train_size))
            eval_dataset = full_dataset.select(range(train_size, train_size + val_size))
            logging.info(f"自动划分验证集: 训练集 {train_size}，验证集 {val_size}")
        else:
            val_paths = args.cv_data.split(",")
            train_dss = [load_from_disk(p) for p in train_paths]
            val_dss = [load_from_disk(p) for p in val_paths]
            # 统一音频格式，避免concatenate_datasets时出错
            if args.model == "flow":
                logging.info("Unifying audio format across datasets for FLOW...")
                train_dss = [ds.cast_column("audio", Audio(sampling_rate=24000, mono=True, decode=True)) for ds in train_dss]
                val_dss = [ds.cast_column("audio", Audio(sampling_rate=24000, mono=True, decode=True)) for ds in val_dss]
            if args.model == "llm":
                logging.info("Preparing LLM dataset...")
                train_dss = [prepare_dataset_llm(ds, tokenizer) for ds in train_dss]
                val_dss = [prepare_dataset_llm(ds, tokenizer) for ds in val_dss]
            else:
                train_dss = [prepare_dataset_flow(ds) for ds in train_dss]
                val_dss = [prepare_dataset_flow(ds) for ds in val_dss]
            train_dataset = concatenate_datasets(train_dss).shuffle(seed=42)
            eval_dataset = concatenate_datasets(val_dss).shuffle(seed=42)
    except Exception as e:
        logging.error(f"❌Error preprocessing datasets: {e}")
        raise e

    # ------------------------------------------------------------------
    # TrainingArguments & Trainer
    # ------------------------------------------------------------------
    try:
        if args.enable_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
            )
            model = get_peft_model(model, lora_config)

        logging.info("Initialising Trainer …")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            remove_unused_columns=False,  # we supply our own collator
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            load_best_model_at_end=False,  # 禁用自动加载最佳模型，避免eval_loss错误
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            deepspeed=args.deepspeed,
            dataloader_num_workers=args.dataloader_num_workers,  # 增加数据加载并行度
            save_total_limit=args.save_total_limit,  # 只保留最近3个checkpoint
            prediction_loss_only=False,  # 修改为False以输出eval loss
            label_names=["speech_token"], # 指定label name
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            tokenizer=tokenizer if tokenizer is not None and args.model != "llm" else None,
        )

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        logging.info("Training...")
        trainer.train()
        logging.info("Training completed. Saving model...")
        trainer.save_model()
        logging.info("Saving completed. All Finished")
        
    except Exception as e:
        logging.error(f"❌Error training: {e}")
        raise e

if __name__ == "__main__":
    main()
