# -*- coding: utf-8 -*-
"""
HydraVox语音模型训练脚本
使用Hugging Face Trainer进行LLM和FLOW模型的训练
支持分布式训练、混合精度、DeepSpeed等功能
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
third_party_dir = project_root / "server/model_utils"

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
    PreTrainedTokenizerBase,
)

# ---------- Domain-specific imports ---------- #
from server.model_utils.cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer  
from server.model_utils.matcha.utils.audio import mel_spectrogram
from modelscope.pipelines import pipeline
import accelerate
import onnxruntime as ort
import whisper
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

from fmtn import create_default_tn


tn = create_default_tn(verbose=True)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 1

accelerator = accelerate.Accelerator()

rank_id = accelerator.process_index

print(f"here once, rank_id: {rank_id}")

EMBEDDING_MODEL_PATH="jzx-ai-lab/speech_campplus_sv_zh-cn_16k-common"
TOKENIZER_ONNX_PATH = Path(
    "jzx-ai-lab/HydraVox/speech_tokenizer_v2.onnx"
).expanduser().resolve()
providers = (
    [("CUDAExecutionProvider", {"device_id": rank_id})]
)
# -----------------------------------------------------------------------------
# Data preparation helpers
# -----------------------------------------------------------------------------

USEFUL_COLUMNS_LLM = ["text", "text_token", "speech_token", "audio"]
USEFUL_COLUMNS_FLOW = ["speech_token", "audio", "embedding"]


sv_pipe = pipeline(
    task="speaker-verification",
    model=EMBEDDING_MODEL_PATH,
    model_revision="v1.0.0",
    device=f"cuda:{rank_id}",  # 使用具体的设备 ID
)

tokenizer_session = ort.InferenceSession(TOKENIZER_ONNX_PATH.as_posix(),
                                 sess_options=so,
                                 providers=providers)

tokenizer = get_qwen_tokenizer(
    token_path="jzx-ai-lab/HydraVox/CosyVoice-BlankEN", skip_special_tokens=True
)

def prepare_dataset_llm(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """保留LLM训练所需的列"""
    ds = ds.remove_columns([c for c in ds.column_names if c not in USEFUL_COLUMNS_LLM])
    ds = ds.cast_column("audio", Audio(decode=True, sampling_rate=16000))
    return ds


def _load_audio_with_fallback(audio_info, target_sr: int | None = None, neighbor_list=None, idx: int = None, mono: bool = True):
    """加载音频，支持邻近样本回退"""
    def load_single_audio(info):
        if isinstance(info, dict):
            if "array" in info:
                wav = torch.tensor(info["array"], dtype=torch.float32)
                if mono and wav.dim() > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                else:
                    wav = wav.unsqueeze(0)
                if target_sr is not None and info["sampling_rate"] != target_sr:
                    wav = torchaudio.transforms.Resample(info["sampling_rate"], target_sr)(wav)
                return wav
            path = "/home/ecs-user/nas_training_data/HanxueTTS/downloaded_audio/" + info.get("path")
        elif isinstance(info, str):
            path = info
        else:
            raise ValueError(f"Invalid audio info: {info}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio not found: {path}")
        
        wav, sr = torchaudio.load(path)
        if mono and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if target_sr is not None and sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        return wav
    
    # 尝试加载当前音频
    try:
        return load_single_audio(audio_info)
    except Exception as e:
        print("Audio not found: ", audio_info, e)
        pass
    
    # 尝试邻近样本
    if neighbor_list and idx is not None:
        for offset in range(1, min(4, len(neighbor_list))):
            for cand in (idx - offset, idx + offset):
                if 0 <= cand < len(neighbor_list):
                    try:
                        print(f"使用邻近样本 {cand} 替代失败的 {idx}")
                        return load_single_audio(neighbor_list[cand])
                    except Exception:
                        continue
    
    # 最终回退
    print(f"音频加载失败，使用默认静音: {idx}")
    return torch.zeros(1, 1600, dtype=torch.float32)

def process_audio_unified(audio_dict, neighbor_list=None, idx=None, need_speech_feat=True, need_embedding=True, need_speech_token=True):
    """统一音频处理：提取mel特征、embedding、speech token"""
    
    def extract_features(audio_info):
        results = {}
        
        if need_speech_feat:
            wav_24k = _load_audio_with_fallback(audio_info, 24000, neighbor_list, idx)
            # mel特征处理
            if (wav_24k.shape[-1] - 480) // 480 % 2 == 0:
                wav_24k = torch.nn.functional.pad(wav_24k, (0, 480))
            mel = mel_spectrogram(wav_24k, 1920, 80, 24000, 480, 1920, 0, 8000, False)
            results["speech_feat"] = mel.squeeze(0).transpose(0, 1)
        
        if need_embedding:
            wav_16k = _load_audio_with_fallback(audio_info, 16000, neighbor_list, idx)
            wav_np = wav_16k.squeeze(0).cpu().numpy().astype(np.float32)
            sv_out = sv_pipe([wav_np], output_emb=True)
            results["embedding"] = np.array(sv_out["embs"][0], dtype=np.float32)
        
        if need_speech_token:
            wav_16k = _load_audio_with_fallback(audio_info, 16000, neighbor_list, idx)
            mel_whisper = whisper.log_mel_spectrogram(wav_16k, n_mels=128)
            ort_inputs = {
                tokenizer_session.get_inputs()[0].name: mel_whisper.cpu().numpy(),
                tokenizer_session.get_inputs()[1].name: np.array([mel_whisper.shape[2]], dtype=np.int32),
            }
            tokens = tokenizer_session.run(None, ort_inputs)[0].flatten().tolist()
            results["speech_token"] = tokens
            results["speech_token_len"] = len(tokens)
        
        return results

    try:
        return extract_features(audio_dict)
    except Exception as e:
        print(f"音频处理失败 {idx}: {e}, 使用默认值")
        
        # 返回默认值
        results = {}
        if need_speech_feat:
            results["speech_feat"] = torch.zeros(100, 80, dtype=torch.float32)
        if need_embedding:
            results["embedding"] = np.zeros(256, dtype=np.float32)
        if need_speech_token:
            results["speech_token"] = [0] * 50
            results["speech_token_len"] = 50
        return results


def prepare_dataset_flow(ds: Dataset) -> Dataset:
    """保留FLOW训练所需的列"""
    ds = ds.remove_columns([c for c in ds.column_names if c not in USEFUL_COLUMNS_FLOW])
    ds = ds.cast_column("audio", Audio(decode=True, sampling_rate=16000))
    return ds


# -----------------------------------------------------------------------------
# Data collator (pads variable-length tensors manually to keep Trainer happy)
# -----------------------------------------------------------------------------

def _process_audio_features(features: List[Dict], need_speech_feat: bool, need_embedding: bool, need_speech_token: bool):
    """处理音频特征"""
    neighbor_list = [f["audio"] for f in features]
    batch = {}
    
    speech_feats, speech_feat_lens = [], []
    embeddings = []
    speech_tokens, speech_token_lens = [], []
    
    for i, f in enumerate(features):
        audio_results = process_audio_unified(
            f["audio"], neighbor_list, i,
            need_speech_feat, need_embedding, need_speech_token
        )
        
        if need_speech_feat:
            mel = audio_results["speech_feat"].to(torch.bfloat16)
            speech_feats.append(mel)
            speech_feat_lens.append(len(mel))
        
        if need_embedding:
            emb = torch.tensor(audio_results["embedding"], dtype=torch.bfloat16)
            embeddings.append(emb)
        
        if need_speech_token:
            speech_token = torch.tensor(audio_results["speech_token"], dtype=torch.long)
            speech_tokens.append(speech_token)
            speech_token_lens.append(audio_results["speech_token_len"])
    
    if need_speech_feat:
        batch["speech_feat"] = pad_sequence(speech_feats, batch_first=True)
        batch["speech_feat_len"] = torch.tensor(speech_feat_lens)
    if need_embedding:
        batch["embedding"] = torch.stack(embeddings)
    if need_speech_token:
        batch["speech_token"] = pad_sequence(speech_tokens, batch_first=True)
        batch["speech_token_len"] = torch.tensor(speech_token_lens, dtype=torch.int64)
    
    return batch

def _process_text_features(features: List[Dict]):
    """处理文本特征"""
    batch = {}
    
    # 处理原始文本
    if "text" in features[0] and "text_token" not in features[0]:
        text_tokens, text_token_lens = [], []
        for f in features:
            try:
                text_tn = tn.process_text(f["text"])
            except Exception:
                text_tn = f["text"]
            text_token = tokenizer.encode(text_tn, allowed_special="all")
            text_tokens.append(torch.tensor(text_token, dtype=torch.long))
            text_token_lens.append(len(text_token))
        batch["text_token"] = pad_sequence(text_tokens, batch_first=True)
        batch["text_token_len"] = torch.tensor(text_token_lens, dtype=torch.int64)
    
    # 处理预处理的text_token
    elif "text_token" in features[0]:
        text_tokens, text_token_lens = [], []
        for f in features:
            text_token = f["text_token"]
            if not isinstance(text_token, torch.Tensor):
                text_token = torch.tensor(text_token, dtype=torch.long)
            text_tokens.append(text_token.long())
            text_token_lens.append(len(text_token))
        batch["text_token"] = pad_sequence(text_tokens, batch_first=True)
        batch["text_token_len"] = torch.tensor(text_token_lens)
    
    return batch

def _process_existing_features(features: List[Dict]):
    """处理已存在的特征"""
    batch = {}
    
    # 处理已有的speech_token
    if "speech_token" in features[0]:
        speech_tokens, speech_token_lens = [], []
        for f in features:
            speech_token = torch.tensor(f["speech_token"], dtype=torch.long)
            speech_tokens.append(speech_token)
            speech_token_lens.append(len(speech_token))
        batch["speech_token"] = pad_sequence(speech_tokens, batch_first=True)
        batch["speech_token_len"] = torch.tensor(speech_token_lens)
    
    # 处理已有的embedding
    if "embedding" in features[0]:
        embeddings = []
        for f in features:
            emb = torch.tensor(f["embedding"], dtype=torch.bfloat16)
            embeddings.append(emb)
        batch["embedding"] = torch.stack(embeddings)
    
    return batch

class ModelAwareDataCollator:
    """根据模型类型智能处理特征的数据批处理器"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """数据批处理函数"""
        batch = {}
        
        # 根据模型类型决定需要的特征
        if self.model_type == "llm":
            # LLM模型只需要speech_token，不需要speech_feat和embedding
            need_speech_feat = False
            need_embedding = False
            need_speech_token = "audio" in features[0] and "speech_token" not in features[0]
        elif self.model_type == "flow":
            # FLOW模型保持完整逻辑
            need_speech_feat = "audio" in features[0] and "speech_feat" not in features[0]
            need_embedding = "audio" in features[0] and "embedding" not in features[0] and "text_token" not in features[0]
            need_speech_token = "audio" in features[0] and "speech_token" not in features[0]
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 处理音频特征
        if need_speech_feat or need_embedding or need_speech_token:
            batch.update(_process_audio_features(features, need_speech_feat, need_embedding, need_speech_token))
        
        # 处理文本特征
        batch.update(_process_text_features(features))
        
        # 处理已存在的特征
        batch.update(_process_existing_features(features))
        
        return batch

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """兼容性函数，默认使用FLOW模型逻辑"""
    collator = ModelAwareDataCollator("flow")
    return collator(features)


# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument("--config", type=str, help="模型配置文件路径")
    parser.add_argument("--model", choices=["llm", "flow"], required=True, help="模型类型")
    parser.add_argument("--model_ckpt", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="分词器路径")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径，逗号分隔")
    parser.add_argument("--cv_data", type=str, required=False, help="验证数据路径，逗号分隔")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed配置文件")

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
    logging.info("🚀 训练脚本启动")
    logging.info("正在加载配置文件...")

    # 构建模型
    if args.config is None:
        model_dir = os.getenv("TTS_MODEL_DIR", "jzx-ai-lab/HydraVox")
        args.config = os.path.join(model_dir, "hydravox.yaml")
        logging.info(f"使用默认配置文件: {args.config}")
    
    with open(args.config, "r") as f:
        model_dir = os.getenv("TTS_MODEL_DIR", "jzx-ai-lab/HydraVox")
        cfg = load_hyperpyyaml(f, overrides={
            'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')
        })
    model = cfg[args.model]

    # 数据处理
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
                # 仅保留路径与元信息，避免这里解码；后续用 torchaudio 按路径读取
                train_dss = [ds for ds in train_dss]
                val_dss = [ds for ds in val_dss]
            if args.model == "llm":
                logging.info("Preparing LLM dataset...")
                train_dss = [prepare_dataset_llm(ds, tokenizer) for ds in train_dss]
                val_dss = [prepare_dataset_llm(ds, tokenizer) for ds in val_dss]

            else:
                raise ValueError("Invalid model type")
            train_dataset = concatenate_datasets(train_dss).shuffle(seed=42).cast_column("audio", Audio(sampling_rate=None, mono=None, decode=False))
            eval_dataset = concatenate_datasets(val_dss).shuffle(seed=42).cast_column("audio", Audio(sampling_rate=None, mono=None, decode=False))
    except Exception as e:
        logging.error(f"❌Error preprocessing datasets: {e}")
        raise e

    # 训练器配置
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
        
        # 如果没有指定eval_batch_size，使用train_batch_size
        eval_batch_size = args.per_device_eval_batch_size if args.per_device_eval_batch_size is not None else args.per_device_train_batch_size
        
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, "logs"),  # 设置日志目录
            remove_unused_columns=False,  # we supply our own collator
            # evaluation_strategy="steps",
            save_strategy="steps",
            logging_steps=args.logging_steps,
            # eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            load_best_model_at_end=False,  # 禁用自动加载最佳模型，避免eval_loss错误
            per_device_train_batch_size=args.per_device_train_batch_size,
            # per_device_eval_batch_size=2,
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
            save_safetensors=False,  # 关闭safetensors，避免共享权重保存报错
        )

        # 创建模型感知的数据批处理器
        data_collator = ModelAwareDataCollator(args.model)
        
        if args.model == "llm":
            logging.info("🎯 LLM模型训练：仅提取speech_token，跳过speech_feat和embedding以优化性能")
        elif args.model == "flow":
            logging.info("🎯 FLOW模型训练：提取完整特征（speech_feat, embedding, speech_token）")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer if tokenizer is not None and args.model != "llm" else None,
        )

        # 开始训练
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
