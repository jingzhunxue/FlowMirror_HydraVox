#!/usr/bin/env python3
# scripts/preprocess/extract_speech_token_dataset.py
"""
批量处理 HuggingFace Dataset，提取 CosyVoice speech_token + CampPlus spk_embedding。

用法：
python extract_speech_token_dataset.py \
        --input data/processed/asr_ds_gpu \
        --output data/processed/token_ds \
        --device cuda           # 或 cpu
        --num-proc 8            # map 并发进程
        --slice 0 50000         # 裁剪 [start, end) 可选
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torchaudio
from datasets import load_from_disk
from tqdm import tqdm

from audio import mel_spectrogram

import whisper

# ----------- 模型路径 -------------
TOKENIZER_ONNX_PATH = Path(
    "models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx"
).expanduser().resolve()

# 本地模型路径
CAMPPLUS_MODEL_DIR = Path("models/speech_campplus_sv_zh-cn_16k-common").resolve()

# 若需要自定义 GPU 数，修改此处
NUM_SESSIONS_PER_PROC = 1

# ----------- 全局缓存 -------------
SESSION_CACHE: Dict[int, Dict[str, Any]] = {}

def download_campplus_model():
    """下载 CampPlus 模型到本地"""
    try:
        if CAMPPLUS_MODEL_DIR.exists() and any(CAMPPLUS_MODEL_DIR.iterdir()):
            print(f"✅ CampPlus 模型已存在: {CAMPPLUS_MODEL_DIR}")
            return str(CAMPPLUS_MODEL_DIR)
    except (OSError, PermissionError) as e:
        print(f"⚠️ 检查本地模型时出错: {e}")
    
    print(f"📥 正在下载 CampPlus 模型到: {CAMPPLUS_MODEL_DIR}")
    try:
        from modelscope import snapshot_download
        
        # 创建模型目录
        CAMPPLUS_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        
        # 下载模型到指定目录
        model_path = snapshot_download(
            model_id="iic/speech_campplus_sv_zh-cn_16k-common",
            revision="v1.0.0",
            cache_dir=str(CAMPPLUS_MODEL_DIR.parent / "modelscope_cache")
        )
        
        # 如果下载路径与目标路径不同，创建软链接或复制
        import shutil
        if Path(model_path).resolve() != CAMPPLUS_MODEL_DIR.resolve():
            if CAMPPLUS_MODEL_DIR.exists():
                shutil.rmtree(CAMPPLUS_MODEL_DIR)
            
            # 尝试创建软链接，如果失败则复制
            try:
                CAMPPLUS_MODEL_DIR.symlink_to(Path(model_path).resolve())
                print(f"✅ 创建软链接: {CAMPPLUS_MODEL_DIR} -> {model_path}")
            except (OSError, NotImplementedError):
                shutil.copytree(model_path, CAMPPLUS_MODEL_DIR)
                print(f"✅ 复制模型文件到: {CAMPPLUS_MODEL_DIR}")
        
        print(f"✅ CampPlus 模型准备完成: {CAMPPLUS_MODEL_DIR}")
        return str(CAMPPLUS_MODEL_DIR)
        
    except Exception as e:
        print(f"❌ 下载 CampPlus 模型失败: {e}")
        print("💡 回退到在线模式...")
        return "iic/speech_campplus_sv_zh-cn_16k-common"

def get_multi_sessions(rank: int, device: str):
    """
    为 datasets.map 每个进程缓存 onnx session & speaker verification pipeline
    """
    # 根据 rank 计算具体的 GPU 设备 ID
    if device.startswith("cuda"):
        num_gpus = torch.cuda.device_count()
        device_id = rank % num_gpus
        specific_device = f"cuda:{device_id}"
    else:
        specific_device = device
        device_id = 0
    
    key = (rank, device)
    if key not in SESSION_CACHE:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1

        providers = (
            [("CUDAExecutionProvider", {"device_id": device_id})]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )

        tokenizer_sessions = [
            ort.InferenceSession(TOKENIZER_ONNX_PATH.as_posix(),
                                 sess_options=so,
                                 providers=providers)
            for _ in range(NUM_SESSIONS_PER_PROC)
        ]

        # 获取本地模型路径或在线模型ID
        try:
            model_exists = CAMPPLUS_MODEL_DIR.exists() and any(CAMPPLUS_MODEL_DIR.iterdir())
        except (OSError, PermissionError):
            model_exists = False
        model_path = str(CAMPPLUS_MODEL_DIR) if model_exists else "iic/speech_campplus_sv_zh-cn_16k-common"

        from modelscope.pipelines import pipeline
        sv_pipe = pipeline(
            task="speaker-verification",
            model=model_path,
            model_revision="v1.0.0" if not model_exists else None,
            device=specific_device,  # 使用具体的设备 ID
        )

        SESSION_CACHE[key] = {
            "tokenizers": tokenizer_sessions,
            "sv_pipeline": sv_pipe,
            "counter": 0,
        }
    return SESSION_CACHE[key]

# ----------- map 回调 -------------
def extract_speech_token(example, rank: int, device: str):
    """
    example: {'audio': {'array': np.ndarray, 'sampling_rate': 16000}}
    返回：speech_token (list[int]), spk_embedding (np.ndarray[float32])
    """
    try:
        sessions = get_multi_sessions(rank, device)
        tk_list = sessions["tokenizers"]
        sv_pipe = sessions["sv_pipeline"]

        tk_session = tk_list[sessions["counter"] % len(tk_list)]
        sessions["counter"] += 1

        arr = example["array"]
        sr = example["sampling_rate"]

        # torch 张量，保证 mono & 16k
        wav = torch.from_numpy(arr).float().unsqueeze(0)  # (1, T)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # 长度上限：30 s
            duration_sec = wav.shape[1] / 16000
            if duration_sec > 30:
                # 返回空值，但保持字段结构一致
                print(f"⚠️ 跳过超长音频样本 (rank {rank}): {duration_sec:.1f}s > 30s")
                return {
                    "speech_token": [], 
                    "speech_token_len": 0,
                    "speech_feat": np.array([], dtype=np.float32).reshape(0, 80),  # 保持正确的维度
                    "speech_feat_len": 0,
                    "embedding": np.array([], dtype=np.float32)
                }

        # --- speech tokenizer ---
        mel = whisper.log_mel_spectrogram(wav, n_mels=128)  # (1, 128, T')
        ort_inputs = {
            tk_session.get_inputs()[0].name: mel.cpu().numpy(),
            tk_session.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32),
        }
        speech_token = tk_session.run(None, ort_inputs)[0].flatten().tolist()
        speech_token_len = len(speech_token)
            
        # --- speaker embedding ---
        sv_out = sv_pipe([arr], output_emb=True)
        emb = np.array(sv_out["embs"][0], dtype=np.float32)

        resample_rate = 24000
        audio_resampled = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(wav)
        # 预估输出长度并调整音频使其产生偶数长度的mel特征
        estimated_frames = (audio_resampled.shape[-1] - 480) // 480 + 1
        if estimated_frames % 2 == 1:
            # 如果预估长度是奇数，调整音频长度
            padding_needed = 480  # 添加一个hop_size的长度
            audio_resampled = torch.nn.functional.pad(audio_resampled, (0, padding_needed))
        
        mel_feat = mel_spectrogram(audio_resampled, 1920, 80, resample_rate, 480, 1920, 0, 8000, False)
        mel_feat = mel_feat.squeeze(0).transpose(0, 1).cpu().numpy()
        mel_feat_len = len(mel_feat)

        return {
            "speech_token": speech_token, 
            "speech_token_len": speech_token_len, 
            "speech_feat": mel_feat, 
            "speech_feat_len": mel_feat_len, 
            "embedding": emb
        }
        
    except Exception as e:
        # 发生错误时返回空值，但保持字段结构一致
        print(f"⚠️ 处理样本时出错 (rank {rank}): {e}")
        return {
            "speech_token": [], 
            "speech_token_len": 0,
            "speech_feat": np.array([], dtype=np.float32).reshape(0, 80),
            "speech_feat_len": 0,
            "embedding": np.array([], dtype=np.float32)
        }

# ----------- 主函数 -------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="HF dataset 路径")
    parser.add_argument("--output", type=Path, required=True, help="保存路径")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="cpu / cuda")
    parser.add_argument("--num-proc", type=int, default=4, help="datasets.map 并发")
    parser.add_argument("--slice", nargs=2, type=int, metavar=("START", "END"),
                        help="裁剪 start end (闭开区间)")
    parser.add_argument("--skip-cleaning", action="store_true", 
                        help="跳过数据清洗步骤，保留所有样本（包括问题样本）")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 下载 CampPlus 模型到本地
    print("🔄 检查并下载 CampPlus 模型...")
    download_campplus_model()

    ds = load_from_disk(str(args.input))
    if args.slice:
        start, end = args.slice
        ds = ds.select(range(start, end))

    print(f"Loaded dataset: {len(ds)} examples")

    # datasets.map 带 rank
    try:
        print(f"🚀 开始提取 speech token，使用 {args.num_proc} 个进程...")
        ds_out = ds.map(
            lambda ex, rank=0: extract_speech_token(ex, rank=rank, device=args.device),
            with_rank=True,
            num_proc=args.num_proc,
            desc="Extracting tokens & embeddings",
            input_columns=["audio"],
        )
        print(f"✅ Token 提取完成，共处理 {len(ds_out)} 个样本")
    except Exception as e:
        print(f"❌ Token 提取失败: {e}")
        print("💡 建议:")
        print("   1. 减少 --num-proc 参数值")
        print("   2. 检查 GPU 内存是否充足")
        print("   3. 检查输入数据格式是否正确")
        raise

    # 清洗数据：过滤掉问题样本
    if not args.skip_cleaning:
        print(f"🧹 开始清洗数据，过滤问题样本...")
        original_count = len(ds_out)
        
        # 统计各类问题样本
        stats = {
            "empty_token": 0,
            "empty_embedding": 0, 
            "empty_speech_feat": 0,
            "invalid_token_range": 0,
            "invalid_embedding_dim": 0
        }
        
        def is_valid_sample(example):
            """检查样本是否有效，并统计问题类型"""
            # 检查 speech_token
            speech_token = example.get("speech_token", [])
            speech_token_len = example.get("speech_token_len", 0)
            if len(speech_token) == 0 or speech_token_len == 0:
                stats["empty_token"] += 1
                return False
                
            # 检查 token 值是否在合理范围内 (通常应该是正整数)
            if len(speech_token) > 0:
                try:
                    if min(speech_token) < 0 or max(speech_token) > 100000:  # 设置合理的上下限
                        stats["invalid_token_range"] += 1
                        return False
                except (ValueError, TypeError):
                    stats["invalid_token_range"] += 1
                    return False
            
            # 检查 embedding
            embedding = example.get("embedding", [])
            if len(embedding) == 0:
                stats["empty_embedding"] += 1
                return False
                
            # 检查 embedding 维度 (CampPlus 通常是 192 维)
            if len(embedding) != 192:
                stats["invalid_embedding_dim"] += 1
                return False
            
            # 检查 speech_feat
            speech_feat = example.get("speech_feat", [])
            speech_feat_len = example.get("speech_feat_len", 0)
            if len(speech_feat) == 0 or speech_feat_len == 0:
                stats["empty_speech_feat"] += 1
                return False
                
            # 检查 speech_feat 维度 (mel特征通常是 80 维)
            try:
                if len(speech_feat) > 0 and len(speech_feat[0]) != 80:
                    stats["empty_speech_feat"] += 1
                    return False
            except (IndexError, TypeError):
                stats["empty_speech_feat"] += 1
                return False
                
            return True
        
        # 过滤有效样本
        ds_clean = ds_out.filter(is_valid_sample, desc="Filtering valid samples")
        cleaned_count = len(ds_clean)
        filtered_count = original_count - cleaned_count
        
        print(f"📊 数据清洗完成:")
        print(f"   • 原始样本数: {original_count}")
        print(f"   • 有效样本数: {cleaned_count}")
        print(f"   • 过滤样本数: {filtered_count} ({filtered_count/original_count*100:.1f}%)")
        
        if filtered_count > 0:
            print(f"🔍 问题样本统计:")
            for problem_type, count in stats.items():
                if count > 0:
                    print(f"   • {problem_type}: {count} 个")
        
        if cleaned_count == 0:
            print("❌ 清洗后没有有效样本，请检查输入数据和处理逻辑")
            return
    else:
        print("⚠️ 跳过数据清洗步骤，保留所有样本（包括问题样本）")
        ds_clean = ds_out
        cleaned_count = len(ds_out)

    # 数据质量统计
    if cleaned_count > 0:
        print(f"📈 数据质量统计:")
        try:
            # 随机采样一些样本进行统计
            sample_size = min(100, cleaned_count)
            sample_ds = ds_clean.select(range(sample_size))
            
            token_lengths = [len(ex["speech_token"]) for ex in sample_ds]
            feat_lengths = [ex["speech_feat_len"] for ex in sample_ds]
            
            print(f"   • Token 长度: 平均 {sum(token_lengths)/len(token_lengths):.1f}, "
                  f"范围 [{min(token_lengths)}, {max(token_lengths)}]")
            print(f"   • Speech feat 长度: 平均 {sum(feat_lengths)/len(feat_lengths):.1f}, "
                  f"范围 [{min(feat_lengths)}, {max(feat_lengths)}]")
            print(f"   • Embedding 维度: {len(sample_ds[0]['embedding'])}")
        except Exception as e:
            print(f"   ⚠️ 统计信息计算失败: {e}")

    try:
        print(f"💾 保存{'清洗后的' if not args.skip_cleaning else ''}数据集到: {args.output}")
        ds_clean.save_to_disk(str(args.output))
        print(f"step 5/5: ✅ All Finished! processed dataset → {args.output}")
        print(f"📈 最终保存 {cleaned_count} 个{'有效' if not args.skip_cleaning else ''}样本")
    except Exception as e:
        print(f"❌ 保存数据集失败: {e}")
        raise

if __name__ == "__main__":
    main()
