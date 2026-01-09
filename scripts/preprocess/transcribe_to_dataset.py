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
from pathlib import Path
from typing import List, Dict
import sys

import torch
import torchaudio
from tqdm import tqdm
from datasets import Dataset, Features, Audio, Value
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

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
# ---------- ASR ----------
def load_asr(model_type: str, device: str):
    
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
                    print(t("asr.resample", name=path.name, src_sr=sample_rate, dst_sr=sr))
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
                    wav = resampler(wav)
                    sample_rate = sr
                
                assert wav.shape[0] == 1 or len(wav.shape) == 1, "only support mono audio"
                
                buf = wav.squeeze().cpu().numpy().astype("float32")
                records.append({"audio": {"array": buf, "sampling_rate": sr}, "text": text})
                return records
        except Exception as e:
            print(t("asr.txt_read_failed", path=txt_path, error=e))
    
    # 如果没有txt文件或读取失败，使用ASR转录
    wav, sample_rate = torchaudio.load(str(path))
    if sample_rate != sr:
        print(t("asr.resample", name=path.name, src_sr=sample_rate, dst_sr=sr))
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        wav = resampler(wav)
        sample_rate = sr
    if wav.shape[0] == 2:
        wav = wav.mean(dim=0)
        print(t("asr.merge_stereo", name=path.name))
    buf = wav.squeeze().cpu().numpy().astype("float32")
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
                print(
                    t(
                        "asr.worker_use_gpu",
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                        target_device=target_device,
                    )
                )
            else:
                print(
                    t(
                        "asr.worker_gpu_unavailable",
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                    )
                )
                target_device = "cpu"
                asr_model_type = "sensevoice"
        else:
            target_device = "cpu"
            asr_model_type = "sensevoice"
        
        print(
            t(
                "asr.worker_start",
                worker_id=worker_id,
                count=len(file_chunk),
                device=target_device,
            )
        )
        
        # 加载ASR模型
        try:
            asr_mdl = load_asr(asr_model_type, target_device)
            print(t("asr.worker_model_loaded", worker_id=worker_id))
        except Exception as e:
            print(t("asr.worker_model_failed", worker_id=worker_id, error=e))
            # 如果GPU模型加载失败，尝试CPU
            if target_device != "cpu":
                print(t("asr.worker_try_cpu", worker_id=worker_id))
                target_device = "cpu"
                asr_model_type = "sensevoice"
                asr_mdl = load_asr(asr_model_type, target_device)
            else:
                raise e
        
        # 处理分配的文件
        worker_records = []
        for fp in tqdm(
            file_chunk,
            desc=t("asr.worker_desc", worker_id=worker_id),
            position=worker_id,
        ):
            try:
                recs = process_file(fp, asr_mdl, sr=16000)
                worker_records.extend(recs)
            except Exception as e:
                print(
                    t(
                        "asr.worker_skip_file",
                        worker_id=worker_id,
                        name=fp.name,
                        error=e,
                    )
                )
        
        return_dict[worker_id] = worker_records
        print(
            t(
                "asr.worker_done",
                worker_id=worker_id,
                count=len(worker_records),
            )
        )
        
    except Exception as e:
        print(t("asr.worker_error", worker_id=worker_id, error=e))
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
    
    print(
        t(
            "asr.mp_start",
            workers=num_workers,
            count=len(audio_files),
        )
    )
    
    # 将文件平均分配给各个工作进程
    chunk_size = math.ceil(len(audio_files) / num_workers)
    file_chunks = [audio_files[i:i + chunk_size] for i in range(0, len(audio_files), chunk_size)]
    
    # 创建共享字典用于收集结果
    manager = Manager()
    return_dict = manager.dict()
    
    # 预热下载模型
    if device == "cuda":
        tmp_mdl = load_asr("paraformer", "cuda")
        del tmp_mdl
    else:
        tmp_mdl = load_asr("sensevoice", "cpu")
        del tmp_mdl

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
        print(
            t(
                "asr.main_start_worker",
                worker_id=i,
                count=len(file_chunks[i]),
                gpu_id=gpu_id,
            )
        )
    
    # 等待所有进程完成
    for i, p in enumerate(processes):
        p.join()
        print(t("asr.main_worker_done", worker_id=i))
    
    # 合并所有结果
    all_records = []
    for worker_id in sorted(return_dict.keys()):
        worker_records = return_dict[worker_id]
        all_records.extend(worker_records)
        print(
            t(
                "asr.main_merge_worker",
                worker_id=worker_id,
                count=len(worker_records),
            )
        )
    
    print(t("asr.mp_done", count=len(all_records)))
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
    print(t("asr.total_records", count=total_records))
    
    if total_records == 0:
        print(t("asr.no_records"))
        return
    
    # 如果记录数较少，直接处理
    if total_records <= batch_size:
        print(t("asr.small_records"))
        # 对每个audio进行响度控制
        print(t("asr.normalizing"))
        for i, record in enumerate(tqdm(records, desc=t("asr.normalizing_desc"))):
            audio_array = record["audio"]["array"]
            normalized_audio = normalize_loudness(audio_array)
            records[i]["audio"]["array"] = normalized_audio
        
        print(t("asr.build_dataset"))
        features = Features({"audio": Audio(sampling_rate=16000), "text": Value("string")})
        ds = Dataset.from_list(records, features=features)
        ds.save_to_disk(dst)
        print(t("asr.dataset_saved", count=len(ds), dst=dst))
        return
    
    # 分批处理大数据集
    print(t("asr.batch_processing_start", batch_size=batch_size))
    
    # 创建输出目录
    dst.mkdir(parents=True, exist_ok=True)
    
    # 分批处理
    num_batches = (total_records + batch_size - 1) // batch_size
    
    all_datasets = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_records)
        
        print(
            "\n"
            + t(
                "asr.batch_processing",
                batch_idx=batch_idx + 1,
                total_batches=num_batches,
                start=start_idx,
                end=end_idx,
            )
        )
        
        # 获取当前批次的记录
        batch_records = records[start_idx:end_idx]
        
        # 响度控制
        print(t("asr.normalizing"))
        for i, record in enumerate(
            tqdm(
                batch_records,
                desc=t("asr.normalizing_batch_desc", batch_idx=batch_idx + 1),
            )
        ):
            audio_array = record["audio"]["array"]
            normalized_audio = normalize_loudness(audio_array)
            batch_records[i]["audio"]["array"] = normalized_audio
        
        # 创建当前批次的dataset
        print(t("asr.batch_build_dataset", batch_idx=batch_idx + 1))
        features = Features({"audio": Audio(sampling_rate=16000), "text": Value("string")})
        batch_ds = Dataset.from_list(batch_records, features=features)
        
        # 保存当前批次
        batch_path = dst / f"batch_{batch_idx:04d}"
        batch_ds.save_to_disk(batch_path)
        
        all_datasets.append(batch_ds)
        
        print(t("asr.batch_saved", batch_idx=batch_idx + 1, path=batch_path))
        
        # 清理内存
        del batch_records
        del batch_ds
        import gc
        gc.collect()
    
    # 合并所有批次
    print("\n" + t("asr.merge_batches", count=len(all_datasets)))
    try:
        from datasets import concatenate_datasets
        final_ds = concatenate_datasets(all_datasets)
        
        # 保存最终dataset
        final_path = dst / "final_dataset"
        final_ds.save_to_disk(final_path)
        
        print(t("asr.final_saved", path=final_path))
        print(t("asr.total_records", count=len(final_ds)))
        
        # 清理批次文件
        import shutil
        for batch_idx in range(num_batches):
            batch_path = dst / f"batch_{batch_idx:04d}"
            if batch_path.exists():
                shutil.rmtree(batch_path)
        
        print(t("asr.cleanup_batches"))
        
    except Exception as e:
        print(t("asr.merge_failed", error=e))
        print(t("asr.batch_files_saved", path=dst))
        print(t("asr.batch_files_hint"))
    
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
    ap.add_argument("--src", type=Path, required=True, help=t("asr.cli_src"))
    ap.add_argument("--dst", type=Path, required=True, help=t("asr.cli_dst"))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--gpu_devices", type=str, default="", help=t("asr.cli_gpu_devices"))
    ap.add_argument("--num_workers", type=int, default=1, help=t("asr.cli_num_workers"))
    ap.add_argument("--min_sec", type=float, default=0.3, help=t("asr.cli_min_sec"))
    ap.add_argument("--batch_size", type=int, default=1000, help=t("asr.cli_batch_size"))
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
            print(t("asr.no_valid_gpu"))
            device = "cpu"
        else:
            device = "cuda"
            print(t("asr.use_gpu_devices", devices=gpu_devices))
    else:
        device = "cpu"
        print(t("asr.use_cpu"))
    
    # 调整工作进程数
    if device == "cuda" and len(gpu_devices) > 1:
        # 多GPU情况下，每个GPU一个进程
        args.num_workers = min(args.num_workers, len(gpu_devices))
        print(t("asr.multi_gpu", workers=args.num_workers))
    elif device == "cpu":
        # CPU情况下限制进程数
        import os
        args.num_workers = min(args.num_workers, os.cpu_count())
        print(t("asr.cpu_parallel", workers=args.num_workers))

    # 同时查找 .wav 和 .mp3 文件
    wav_files = sorted(args.src.rglob("*.wav"))
    mp3_files = sorted(args.src.rglob("*.mp3"))
    audio_files = wav_files + mp3_files
    
    if not audio_files:
        print(t("asr.no_audio_files", src=args.src))
        sys.exit(1)
    
    print(t("asr.found_files", wav_count=len(wav_files), mp3_count=len(mp3_files)))
    
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
        print(t("asr.loading_model"))
        asr_mdl = load_asr(asr_model_type, target_device)
        print(t("asr.using_model", model_type=asr_model_type, device=target_device))
        
        all_records = []
        for fp in tqdm(audio_files, desc=t("asr.asr_desc")):
            recs = process_file(fp, asr_mdl, sr=16000)
            all_records.extend(recs)

    if not all_records:
        print(t("asr.no_records_extracted"))
        sys.exit(1)

    build_dataset(all_records, args.dst, args.batch_size)
    print(t("asr.step_done", count=len(all_records), dst=args.dst))

if __name__ == "__main__":
    main()
