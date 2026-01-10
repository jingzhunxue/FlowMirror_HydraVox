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

# ----------- i18n -------------
_TRANSLATIONS = {
    "✅ CampPlus 模型已存在: {path}": {"en": "✅ CampPlus model already exists: {path}"},
    "⚠️ 检查本地模型时出错: {error}": {"en": "⚠️ Failed to check local model: {error}"},
    "📥 正在下载 CampPlus 模型到: {path}": {"en": "📥 Downloading CampPlus model to: {path}"},
    "✅ 创建软链接: {dst} -> {src}": {"en": "✅ Created symlink: {dst} -> {src}"},
    "✅ 复制模型文件到: {path}": {"en": "✅ Copied model files to: {path}"},
    "✅ CampPlus 模型准备完成: {path}": {"en": "✅ CampPlus model ready: {path}"},
    "❌ 下载 CampPlus 模型失败: {error}": {"en": "❌ Failed to download CampPlus model: {error}"},
    "💡 回退到在线模式...": {"en": "💡 Falling back to online mode..."},
    "⚠️ 跳过超长音频样本 (rank {rank}): {duration:.1f}s > 30s": {
        "en": "⚠️ Skipping overlong audio sample (rank {rank}): {duration:.1f}s > 30s"
    },
    "⚠️ 处理样本时出错 (rank {rank}): {error}": {
        "en": "⚠️ Failed to process sample (rank {rank}): {error}"
    },
    "HF dataset 路径": {"en": "HF dataset path"},
    "保存路径": {"en": "Output path"},
    "cpu / cuda": {"en": "cpu / cuda"},
    "datasets.map 并发": {"en": "datasets.map workers"},
    "裁剪 start end (闭开区间)": {"en": "Slice start end (half-open interval)"},
    "跳过数据清洗步骤，保留所有样本（包括问题样本）": {
        "en": "Skip data cleaning and keep all samples (including problematic ones)"
    },
    "🔄 检查并下载 CampPlus 模型...": {"en": "🔄 Checking and downloading CampPlus model..."},
    "已加载数据集: {count} 条样本": {"en": "Loaded dataset: {count} examples"},
    "🚀 开始提取 speech token，使用 {num_proc} 个进程...": {
        "en": "🚀 Extracting speech tokens with {num_proc} processes..."
    },
    "✅ Token 提取完成，共处理 {count} 个样本": {"en": "✅ Token extraction done, processed {count} samples"},
    "❌ Token 提取失败: {error}": {"en": "❌ Token extraction failed: {error}"},
    "💡 建议:": {"en": "💡 Suggestions:"},
    "   1. 减少 --num-proc 参数值": {"en": "   1. Reduce the --num-proc value"},
    "   2. 检查 GPU 内存是否充足": {"en": "   2. Check if GPU memory is sufficient"},
    "   3. 检查输入数据格式是否正确": {"en": "   3. Check whether input data format is correct"},
    "🧹 开始清洗数据，过滤问题样本...": {"en": "🧹 Cleaning data and filtering problematic samples..."},
    "📊 数据清洗完成:": {"en": "📊 Data cleaning completed:"},
    "   • 原始样本数: {count}": {"en": "   • Original samples: {count}"},
    "   • 有效样本数: {count}": {"en": "   • Valid samples: {count}"},
    "   • 过滤样本数: {count} ({ratio:.1f}%)": {"en": "   • Filtered samples: {count} ({ratio:.1f}%)"},
    "🔍 问题样本统计:": {"en": "🔍 Problem sample stats:"},
    "   • {problem_type}: {count} 个": {"en": "   • {problem_type}: {count}"},
    "❌ 清洗后没有有效样本，请检查输入数据和处理逻辑": {
        "en": "❌ No valid samples after cleaning; check input data and logic"
    },
    "⚠️ 跳过数据清洗步骤，保留所有样本（包括问题样本）": {
        "en": "⚠️ Skipping data cleaning; keeping all samples (including problematic ones)"
    },
    "📈 数据质量统计:": {"en": "📈 Data quality stats:"},
    "   • Token 长度: 平均 {avg:.1f}, 范围 [{min}, {max}]": {
        "en": "   • Token length: avg {avg:.1f}, range [{min}, {max}]"
    },
    "   • Speech feat 长度: 平均 {avg:.1f}, 范围 [{min}, {max}]": {
        "en": "   • Speech feat length: avg {avg:.1f}, range [{min}, {max}]"
    },
    "   • Embedding 维度: {dim}": {"en": "   • Embedding dim: {dim}"},
    "   ⚠️ 统计信息计算失败: {error}": {"en": "   ⚠️ Failed to compute stats: {error}"},
    "清洗后的": {"en": "cleaned "},
    "💾 保存{suffix}数据集到: {path}": {"en": "💾 Saving {suffix}dataset to: {path}"},
    "step 5/5: ✅ 全部完成！处理后的数据集 -> {path}": {
        "en": "step 5/5: ✅ All Finished! processed dataset → {path}"
    },
    "📈 最终保存 {count} 个{suffix}样本": {"en": "📈 Saved {count} {suffix}samples"},
    "❌ 保存数据集失败: {error}": {"en": "❌ Failed to save dataset: {error}"},
}


def _t(text: str, **kwargs: Any) -> str:
    lang = os.getenv("HYDRAVOX_LANG", os.getenv("HYDRAVOX_UI_LANG", "zh")).lower()
    if lang not in ("zh", "en"):
        lang = "zh"
    entry = _TRANSLATIONS.get(text)
    result = entry.get(lang, text) if entry else text
    if kwargs:
        try:
            return result.format(**kwargs)
        except Exception:
            return result
    return result

# ----------- 模型路径 -------------
TOKENIZER_ONNX_PATH = Path(
    "jzx-ai-lab/HydraVox-CV3/speech_tokenizer_v3.onnx"
).expanduser().resolve()

# 本地模型路径
CAMPPLUS_MODEL_DIR = Path("jzx-ai-lab/speech_campplus_sv_zh-cn_16k-common").resolve()

# 若需要自定义 GPU 数，修改此处
NUM_SESSIONS_PER_PROC = 1

# ----------- 全局缓存 -------------
SESSION_CACHE: Dict[int, Dict[str, Any]] = {}

def download_campplus_model():
    """下载 CampPlus 模型到本地"""
    try:
        if CAMPPLUS_MODEL_DIR.exists() and any(CAMPPLUS_MODEL_DIR.iterdir()):
            print(_t("✅ CampPlus 模型已存在: {path}", path=CAMPPLUS_MODEL_DIR))
            return str(CAMPPLUS_MODEL_DIR)
    except (OSError, PermissionError) as e:
        print(_t("⚠️ 检查本地模型时出错: {error}", error=e))
    
    print(_t("📥 正在下载 CampPlus 模型到: {path}", path=CAMPPLUS_MODEL_DIR))
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
                print(_t("✅ 创建软链接: {dst} -> {src}", dst=CAMPPLUS_MODEL_DIR, src=model_path))
            except (OSError, NotImplementedError):
                shutil.copytree(model_path, CAMPPLUS_MODEL_DIR)
                print(_t("✅ 复制模型文件到: {path}", path=CAMPPLUS_MODEL_DIR))
        
        print(_t("✅ CampPlus 模型准备完成: {path}", path=CAMPPLUS_MODEL_DIR))
        return str(CAMPPLUS_MODEL_DIR)
        
    except Exception as e:
        print(_t("❌ 下载 CampPlus 模型失败: {error}", error=e))
        print(_t("💡 回退到在线模式..."))
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
                print(_t("⚠️ 跳过超长音频样本 (rank {rank}): {duration:.1f}s > 30s", rank=rank, duration=duration_sec))
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
        print(_t("⚠️ 处理样本时出错 (rank {rank}): {error}", rank=rank, error=e))
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
    parser.add_argument("--input", type=Path, required=True, help=_t("HF dataset 路径"))
    parser.add_argument("--output", type=Path, required=True, help=_t("保存路径"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help=_t("cpu / cuda"))
    parser.add_argument("--num-proc", type=int, default=4, help=_t("datasets.map 并发"))
    parser.add_argument("--slice", nargs=2, type=int, metavar=("START", "END"),
                        help=_t("裁剪 start end (闭开区间)"))
    parser.add_argument("--skip-cleaning", action="store_true", 
                        help=_t("跳过数据清洗步骤，保留所有样本（包括问题样本）"))
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 下载 CampPlus 模型到本地
    print(_t("🔄 检查并下载 CampPlus 模型..."))
    download_campplus_model()

    ds = load_from_disk(str(args.input))
    if args.slice:
        start, end = args.slice
        ds = ds.select(range(start, end))

    print(_t("已加载数据集: {count} 条样本", count=len(ds)))

    # datasets.map 带 rank
    try:
        print(_t("🚀 开始提取 speech token，使用 {num_proc} 个进程...", num_proc=args.num_proc))
        ds_out = ds.map(
            lambda ex, rank=0: extract_speech_token(ex, rank=rank, device=args.device),
            with_rank=True,
            num_proc=args.num_proc,
            desc="Extracting tokens & embeddings",
            input_columns=["audio"],
        )
        print(_t("✅ Token 提取完成，共处理 {count} 个样本", count=len(ds_out)))
    except Exception as e:
        print(_t("❌ Token 提取失败: {error}", error=e))
        print(_t("💡 建议:"))
        print(_t("   1. 减少 --num-proc 参数值"))
        print(_t("   2. 检查 GPU 内存是否充足"))
        print(_t("   3. 检查输入数据格式是否正确"))
        raise

    # 清洗数据：过滤掉问题样本
    if not args.skip_cleaning:
        print(_t("🧹 开始清洗数据，过滤问题样本..."))
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
        
        print(_t("📊 数据清洗完成:"))
        print(_t("   • 原始样本数: {count}", count=original_count))
        print(_t("   • 有效样本数: {count}", count=cleaned_count))
        print(_t("   • 过滤样本数: {count} ({ratio:.1f}%)", count=filtered_count, ratio=filtered_count/original_count*100))
        
        if filtered_count > 0:
            print(_t("🔍 问题样本统计:"))
            for problem_type, count in stats.items():
                if count > 0:
                    print(_t("   • {problem_type}: {count} 个", problem_type=problem_type, count=count))
        
        if cleaned_count == 0:
            print(_t("❌ 清洗后没有有效样本，请检查输入数据和处理逻辑"))
            return
    else:
        print(_t("⚠️ 跳过数据清洗步骤，保留所有样本（包括问题样本）"))
        ds_clean = ds_out
        cleaned_count = len(ds_out)

    # 数据质量统计
    if cleaned_count > 0:
        print(_t("📈 数据质量统计:"))
        try:
            # 随机采样一些样本进行统计
            sample_size = min(100, cleaned_count)
            sample_ds = ds_clean.select(range(sample_size))
            
            token_lengths = [len(ex["speech_token"]) for ex in sample_ds]
            feat_lengths = [ex["speech_feat_len"] for ex in sample_ds]
            
            print(_t(
                "   • Token 长度: 平均 {avg:.1f}, 范围 [{min}, {max}]",
                avg=sum(token_lengths)/len(token_lengths),
                min=min(token_lengths),
                max=max(token_lengths),
            ))
            print(_t(
                "   • Speech feat 长度: 平均 {avg:.1f}, 范围 [{min}, {max}]",
                avg=sum(feat_lengths)/len(feat_lengths),
                min=min(feat_lengths),
                max=max(feat_lengths),
            ))
            print(_t("   • Embedding 维度: {dim}", dim=len(sample_ds[0]['embedding'])))
        except Exception as e:
            print(_t("   ⚠️ 统计信息计算失败: {error}", error=e))

    try:
        suffix = _t("清洗后的") if not args.skip_cleaning else ""
        print(_t("💾 保存{suffix}数据集到: {path}", suffix=suffix, path=args.output))
        ds_clean.save_to_disk(str(args.output))
        print(_t("step 5/5: ✅ 全部完成！处理后的数据集 -> {path}", path=args.output))
        suffix = _t("清洗后的") if not args.skip_cleaning else ""
        print(_t("📈 最终保存 {count} 个{suffix}样本", count=cleaned_count, suffix=suffix))
    except Exception as e:
        print(_t("❌ 保存数据集失败: {error}", error=e))
        raise

if __name__ == "__main__":
    main()
