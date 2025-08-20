#!/usr/bin/env python3
"""
Usage:
  python transcribe_to_dataset.py \
      --src data/processed/wav16 \
      --dst data/processed/asr_ds \
      --device gpu \
      --asr_model_path models/SenseVoiceSmall
"""

import argparse
from functools import lru_cache
from pathlib import Path
from typing import List, Dict
import sys

import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from datasets import Dataset, Features, Audio, Value
import numpy as np

# ---------- VAD ----------
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

vad_model = load_silero_vad()

def slice_audio(path: Path, sr: int = 16000, min_sec: float = 0.3):
    wav, sample_rate = torchaudio.load(str(path))
    
    # 检查采样率，如果不一致则进行重采样
    if sample_rate != sr:
        print(f"重采样 {path.name}: {sample_rate}Hz -> {sr}Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        wav = resampler(wav)
        sample_rate = sr
    
    assert wav.shape[0] == 1 or len(wav.shape) == 1, "only support mono audio"
    ts = get_speech_timestamps(wav.squeeze(), vad_model, threshold=0.5, sampling_rate=sr)
    # 合并很短的停顿片段
    merged = []
    for seg in ts:
        if not merged or seg["start"] - merged[-1]["end"] > min_sec * sr:
            merged.append(seg)
        else:
            merged[-1]["end"] = seg["end"]
    # save each chunk to memory buffer (wav bytes)
    chunks = []
    for seg in merged:
        chunk = wav[:, seg["start"]:seg["end"]]
        # 保证是float32，范围[-1, 1]
        buf = chunk.squeeze().cpu().numpy().astype("float32")
        chunks.append(buf.copy())
    return chunks

# ---------- ASR ----------
def load_asr(model_type: str, device: str):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    import os
    
    if model_type == "paraformer":
        mdl = pipeline(
            task=Tasks.auto_speech_recognition,
            model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch', model_revision="v2.0.4",
            device=device)
    else:
        mdl = pipeline(
            task=Tasks.auto_speech_recognition,
            model='iic/SenseVoiceSmall', model_revision="master",
            device=device)

    return mdl

def asr_transcribe(mdl, wav_buf, sr=16000):
    hyp = mdl(
        input=wav_buf
    )
    # 兼容返回list或dict
    text = ""
    if isinstance(hyp, list):
        if len(hyp) > 0 and isinstance(hyp[0], dict) and "text" in hyp[0]:
            text = hyp[0]["text"]
        else:
            text = ""
    elif isinstance(hyp, dict):
        text = hyp.get("text", "")
    else:
        text = ""
    
    # 后处理文本：去除多余的字符间空格（保留词间空格）
    if text.strip():
        text = post_process_text(text)
    
    return text

def post_process_text(text):
    """
    后处理ASR输出文本，去除多余空格
    """
    import re
    
    # 去除中文字符间的空格，但保留英文单词间的空格
    # 匹配中文字符间的空格
    text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # 去除中文字符和标点符号间的空格
    text = re.sub(r'([\u4e00-\u9fff])\s+([，。！？、；：""''（）【】《》])', r'\1\2', text)
    text = re.sub(r'([，。！？、；：""''（）【】《》])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # 去除多余的连续空格，保留单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

# ---------- 响度控制 ----------
def normalize_loudness(audio_array, target_loudness_db=-23.0):
    """
    对音频进行响度归一化
    
    Args:
        audio_array: 音频数组
        target_loudness_db: 目标响度 (LUFS)
    
    Returns:
        归一化后的音频数组
    """
    try:
        import pyloudnorm as pyln
        
        # 创建响度计量器
        meter = pyln.Meter(24000)  # 24kHz采样率
        
        # 测量当前响度
        loudness = meter.integrated_loudness(audio_array)
        
        # 如果测量失败或音频太安静，使用简单的RMS归一化
        if loudness == float('-inf') or np.isnan(loudness):
            return simple_normalize(audio_array)
        
        # 计算归一化因子
        loudness_difference = target_loudness_db - loudness
        gain = 10.0 ** (loudness_difference / 20.0)
        
        # 应用增益
        normalized_audio = audio_array * gain
        
        # 防止削波
        if np.max(np.abs(normalized_audio)) > 0.95:
            normalized_audio = normalized_audio / np.max(np.abs(normalized_audio)) * 0.95
        
        return normalized_audio
        
    except ImportError:
        # 如果没有安装pyloudnorm，使用简单的RMS归一化
        return simple_normalize(audio_array)
    except Exception:
        # 如果响度归一化失败，使用简单的RMS归一化
        return simple_normalize(audio_array)

def simple_normalize(audio_array, target_rms=0.1):
    """
    简单的RMS归一化
    
    Args:
        audio_array: 音频数组
        target_rms: 目标RMS值
    
    Returns:
        归一化后的音频数组
    """
    if len(audio_array) == 0:
        return audio_array
        
    # 计算当前RMS
    rms = np.sqrt(np.mean(audio_array ** 2))
    
    if rms == 0:
        return audio_array
    
    # 计算增益
    gain = target_rms / rms
    
    # 应用增益
    normalized_audio = audio_array * gain
    
    # 防止削波
    if np.max(np.abs(normalized_audio)) > 0.95:
        normalized_audio = normalized_audio / np.max(np.abs(normalized_audio)) * 0.95
    
    return normalized_audio

# ---------- main ----------
def process_file(path: Path, asr_mdl, sr=16000) -> List[Dict]:
    records = []
    
    # 检查是否存在同名的txt文件
    txt_path = path.with_suffix('.txt')
    if txt_path.exists():
        try:
            # 从txt文件读取文本
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if text:
                # 加载整个音频文件（不进行VAD分割）
                wav, sample_rate = torchaudio.load(str(path))
                
                # 检查采样率，如果不一致则进行重采样
                if sample_rate != sr:
                    print(f"重采样 {path.name}: {sample_rate}Hz -> {sr}Hz")
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
                    wav = resampler(wav)
                    sample_rate = sr
                
                assert wav.shape[0] == 1 or len(wav.shape) == 1, "only support mono audio"
                
                buf = wav.squeeze().cpu().numpy().astype("float32")
                records.append({"audio": {"array": buf, "sampling_rate": sr}, "text": text})
                return records
        except Exception as e:
            print(f" ! 读取txt文件失败 {txt_path}: {e}, 使用ASR转录")
    
    # 如果没有txt文件或读取失败，使用ASR转录
    for buf in slice_audio(path, sr):
        text = asr_transcribe(asr_mdl, buf)
        if text.strip():
            records.append({"audio": {"array": buf, "sampling_rate": sr}, "text": text})
    return records

# ---------- 多进程处理 ----------
def worker_process(worker_id, file_chunk, device, gpu_id, min_sec, return_dict):
    """
    工作进程函数，处理分配给它的文件
    
    Args:
        worker_id: 工作进程ID
        file_chunk: 分配给该进程的文件列表
        device: 设备类型 (cpu/cuda)
        gpu_id: GPU设备ID (仅在device为cuda时有效)
        min_sec: 最小分段时长
        return_dict: 用于返回结果的共享字典
    """
    try:
        import os
        import torch
        
        # 设置GPU设备
        if device == "cuda" and gpu_id is not None:
            # 设置环境变量
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # 确保CUDA可用
            if torch.cuda.is_available():
                target_device = "cuda:0"  # 在设置了CUDA_VISIBLE_DEVICES后，可见的GPU总是0
                asr_model_type = "paraformer"
                print(f"[Worker {worker_id}] 使用GPU {gpu_id}，映射为 {target_device}")
            else:
                print(f"[Worker {worker_id}] GPU {gpu_id} 不可用，切换到CPU")
                target_device = "cpu"
                asr_model_type = "sensevoice"
        else:
            target_device = "cpu"
            asr_model_type = "sensevoice"
        
        print(f"[Worker {worker_id}] 开始处理 {len(file_chunk)} 个文件，使用设备: {target_device}")
        
        # 加载ASR模型
        try:
            asr_mdl = load_asr(asr_model_type, target_device)
            print(f"[Worker {worker_id}] ASR模型加载成功")
        except Exception as e:
            print(f"[Worker {worker_id}] ASR模型加载失败: {e}")
            # 如果GPU模型加载失败，尝试CPU
            if target_device != "cpu":
                print(f"[Worker {worker_id}] 尝试使用CPU加载模型")
                target_device = "cpu"
                asr_model_type = "sensevoice"
                asr_mdl = load_asr(asr_model_type, target_device)
            else:
                raise e
        
        # 处理分配的文件
        worker_records = []
        for fp in tqdm(file_chunk, desc=f"Worker {worker_id}", position=worker_id):
            try:
                recs = process_file(fp, asr_mdl, sr=16000)
                worker_records.extend(recs)
            except Exception as e:
                print(f"[Worker {worker_id}] 跳过文件 {fp.name}: {e}")
        
        return_dict[worker_id] = worker_records
        print(f"[Worker {worker_id}] 完成处理，生成 {len(worker_records)} 条记录")
        
    except Exception as e:
        print(f"[Worker {worker_id}] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return_dict[worker_id] = []

def process_files_multiprocess(audio_files, device, gpu_devices, num_workers, min_sec):
    """
    多进程处理音频文件
    
    Args:
        audio_files: 音频文件列表
        device: 设备类型
        gpu_devices: GPU设备列表
        num_workers: 工作进程数
        min_sec: 最小分段时长
        
    Returns:
        所有记录的列表
    """
    from multiprocessing import Process, Manager
    import multiprocessing as mp
    import math
    
    # 设置启动方法为spawn以支持CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过了，忽略错误
        pass
    
    print(f"🚀 启动多进程处理: {num_workers} 个工作进程处理 {len(audio_files)} 个文件")
    
    # 将文件平均分配给各个工作进程
    chunk_size = math.ceil(len(audio_files) / num_workers)
    file_chunks = [audio_files[i:i + chunk_size] for i in range(0, len(audio_files), chunk_size)]
    
    # 创建共享字典用于收集结果
    manager = Manager()
    return_dict = manager.dict()
    
    # 创建并启动工作进程
    processes = []
    for i in range(len(file_chunks)):
        # 为每个进程分配GPU设备
        if device == "cuda" and gpu_devices:
            gpu_id = gpu_devices[i % len(gpu_devices)]  # 循环分配GPU
        else:
            gpu_id = None
        
        p = Process(
            target=worker_process,
            args=(i, file_chunks[i], device, gpu_id, min_sec, return_dict)
        )
        processes.append(p)
        p.start()
        print(f"[主进程] 启动工作进程 {i}，分配 {len(file_chunks[i])} 个文件，GPU: {gpu_id}")
    
    # 等待所有进程完成
    for i, p in enumerate(processes):
        p.join()
        print(f"[主进程] 工作进程 {i} 已完成")
    
    # 合并所有结果
    all_records = []
    for worker_id in sorted(return_dict.keys()):
        worker_records = return_dict[worker_id]
        all_records.extend(worker_records)
        print(f"[主进程] 合并工作进程 {worker_id} 的 {len(worker_records)} 条记录")
    
    print(f"✅ 多进程处理完成，总共生成 {len(all_records)} 条记录")
    return all_records

def build_dataset(records, dst: Path, batch_size: int = 1000):
    """
    分批处理数据集，避免内存溢出
    
    Args:
        records: 音频记录列表
        dst: 输出目录
        batch_size: 批处理大小
    """
    total_records = len(records)
    print(f"总记录数: {total_records}")
    
    if total_records == 0:
        print("⚠️ 没有记录可处理")
        return
    
    # 如果记录数较少，直接处理
    if total_records <= batch_size:
        print("记录数较少，直接处理...")
        # 对每个audio进行响度控制
        print("正在进行响度控制...")
        for i, record in enumerate(tqdm(records, desc="Normalizing")):
            audio_array = record["audio"]["array"]
            normalized_audio = normalize_loudness(audio_array)
            records[i]["audio"]["array"] = normalized_audio
        
        print("开始生成Dataset...")
        features = Features({"audio": Audio(sampling_rate=16000), "text": Value("string")})
        ds = Dataset.from_list(records, features=features)
        ds.save_to_disk(dst)
        print(f"✓ Saved dataset with {len(ds)} records -> {dst}")
        return
    
    # 分批处理大数据集
    print(f"开始分批处理，批大小: {batch_size}")
    
    # 创建输出目录
    dst.mkdir(parents=True, exist_ok=True)
    
    # 分批处理
    num_batches = (total_records + batch_size - 1) // batch_size
    
    all_datasets = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_records)
        
        print(f"\n处理批次 {batch_idx + 1}/{num_batches} (记录 {start_idx}-{end_idx})")
        
        # 获取当前批次的记录
        batch_records = records[start_idx:end_idx]
        
        # 响度控制
        print("正在进行响度控制...")
        for i, record in enumerate(tqdm(batch_records, desc=f"Normalizing batch {batch_idx + 1}")):
            audio_array = record["audio"]["array"]
            normalized_audio = normalize_loudness(audio_array)
            batch_records[i]["audio"]["array"] = normalized_audio
        
        # 创建当前批次的dataset
        print(f"创建批次 {batch_idx + 1} 的Dataset...")
        features = Features({"audio": Audio(sampling_rate=16000), "text": Value("string")})
        batch_ds = Dataset.from_list(batch_records, features=features)
        
        # 保存当前批次
        batch_path = dst / f"batch_{batch_idx:04d}"
        batch_ds.save_to_disk(batch_path)
        
        all_datasets.append(batch_ds)
        
        print(f"✓ 批次 {batch_idx + 1} 已保存到 {batch_path}")
        
        # 清理内存
        del batch_records
        del batch_ds
        import gc
        gc.collect()
    
    # 合并所有批次
    print(f"\n合并 {len(all_datasets)} 个批次...")
    try:
        from datasets import concatenate_datasets
        final_ds = concatenate_datasets(all_datasets)
        
        # 保存最终dataset
        final_path = dst / "final_dataset"
        final_ds.save_to_disk(final_path)
        
        print(f"✓ 最终数据集已保存到 {final_path}")
        print(f"总记录数: {len(final_ds)}")
        
        # 清理批次文件
        import shutil
        for batch_idx in range(num_batches):
            batch_path = dst / f"batch_{batch_idx:04d}"
            if batch_path.exists():
                shutil.rmtree(batch_path)
        
        print("✓ 已清理临时批次文件")
        
    except Exception as e:
        print(f"⚠️ 合并失败: {e}")
        print(f"批次文件保存在: {dst}")
        print("你可以手动加载各个批次文件")
    
    finally:
        # 清理内存
        del all_datasets
        import gc
        gc.collect()

def main():
    # 设置multiprocessing启动方法为spawn以支持CUDA
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过了，忽略错误
        pass
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="音频文件根目录")
    ap.add_argument("--dst", type=Path, required=True, help="输出 datasets 目录")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--gpu_devices", type=str, default="", help="指定GPU设备，用逗号分隔，如: 0,1,2,3")
    ap.add_argument("--num_workers", type=int, default=1, help="并行工作进程数")
    ap.add_argument("--min_sec", type=float, default=0.3, help="分段最小间隔 (s)")
    ap.add_argument("--batch_size", type=int, default=1000, help="批处理大小，避免内存溢出 (默认: 1000)")
    args = ap.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    # 处理GPU设备配置
    gpu_devices = []
    if args.device == "cuda" and torch.cuda.is_available():
        if args.gpu_devices.strip():
            # 解析指定的GPU设备
            gpu_devices = [int(d.strip()) for d in args.gpu_devices.split(',') if d.strip().isdigit()]
            # 验证GPU设备是否有效
            available_gpus = list(range(torch.cuda.device_count()))
            gpu_devices = [d for d in gpu_devices if d in available_gpus]
        else:
            # 使用所有可用GPU
            gpu_devices = list(range(torch.cuda.device_count()))
        
        if not gpu_devices:
            print("⚠️ 未找到有效的GPU设备，使用CPU")
            device = "cpu"
        else:
            device = "cuda"
            print(f"🚀 将使用GPU设备: {gpu_devices}")
    else:
        device = "cpu"
        print("🖥️ 使用CPU设备")
    
    # 调整工作进程数
    if device == "cuda" and len(gpu_devices) > 1:
        # 多GPU情况下，每个GPU一个进程
        args.num_workers = min(args.num_workers, len(gpu_devices))
        print(f"📊 多GPU并行处理，使用 {args.num_workers} 个工作进程")
    elif device == "cpu":
        # CPU情况下限制进程数
        import os
        args.num_workers = min(args.num_workers, os.cpu_count())
        print(f"🔧 CPU并行处理，使用 {args.num_workers} 个工作进程")

    # 同时查找 .wav 和 .mp3 文件
    wav_files = sorted(args.src.rglob("*.wav"))
    mp3_files = sorted(args.src.rglob("*.mp3"))
    audio_files = wav_files + mp3_files
    
    if not audio_files:
        print(f"错误：在目录 '{args.src}' 中没有找到任何 .wav 或 .mp3 文件。")
        sys.exit(1)
    
    print(f"找到 {len(wav_files)} 个 .wav 文件和 {len(mp3_files)} 个 .mp3 文件")
    
    # 多进程处理
    if args.num_workers > 1:
        all_records = process_files_multiprocess(
            audio_files, device, gpu_devices, 
            args.num_workers, args.min_sec
        )
    else:
        # 单进程处理（原有逻辑）
        asr_model_type = "paraformer" if device == "cuda" else "sensevoice"
        target_device = f"cuda:{gpu_devices[0]}" if device == "cuda" and gpu_devices else device
        print('Loading ASR model...')
        asr_mdl = load_asr(asr_model_type, target_device)
        print(f"[ASR] using {asr_model_type} on {target_device}")
        
        all_records = []
        for fp in tqdm(audio_files, desc="ASR"):
            recs = process_file(fp, asr_mdl, sr=16000)
            all_records.extend(recs)

    if not all_records:
        print("错误：未能从音频文件中提取任何有效的语音文本对。")
        print("请检查您的音频文件是否包含清晰的语音，或尝试调整VAD参数。")
        sys.exit(1)

    build_dataset(all_records, args.dst, args.batch_size)
    print(f"step 4/5: ✅ All Finished! Transcribed {len(all_records)} files -> {args.dst}")

if __name__ == "__main__":
    main()
