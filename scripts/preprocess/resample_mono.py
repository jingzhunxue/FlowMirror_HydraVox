#!/usr/bin/env python3
"""
音频重采样和转换单声道工具
Usage:
  python resample_mono.py \
      --src data/raw/audio \
      --dst data/processed/audio_24k \
      --sr 24000 \
      --jobs 8
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing.pool import ThreadPool
from typing import List, Tuple

import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm

# 支持的音频格式
SUPPORTED_FORMATS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}

def resample_to_mono(src_path: Path, dst_path: Path, target_sr: int, overwrite: bool = False) -> bool:
    """
    重采样音频到指定采样率并转换为单声道
    
    Args:
        src_path: 源音频文件路径
        dst_path: 目标音频文件路径
        target_sr: 目标采样率
        overwrite: 是否覆盖已存在文件
    
    Returns:
        bool: 处理是否成功
    """
    try:
        # 检查目标文件是否已存在
        if dst_path.exists() and not overwrite:
            return True
        
        # 创建输出目录
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(str(src_path))
        
        # 转换为单声道（如果是多声道）
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, N)
        
        # 重采样到目标采样率
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sr,
                resampling_method="sinc_interp_kaiser"
            )
            waveform = resampler(waveform)
        
        # 保存处理后的音频
        torchaudio.save(str(dst_path), waveform, target_sr)
        
        return True
        
    except Exception as e:
        print(f"处理文件 {src_path} 时出错: {e}")
        return False

def find_audio_files(src_dir: Path) -> List[Path]:
    """
    递归查找所有支持的音频文件
    
    Args:
        src_dir: 源目录
    
    Returns:
        List[Path]: 音频文件路径列表
    """
    audio_files = []
    for ext in SUPPORTED_FORMATS:
        audio_files.extend(src_dir.rglob(f"*{ext}"))
    return sorted(audio_files)

def process_single_file(args_tuple: Tuple[Path, Path, Path, int, bool]) -> bool:
    """
    处理单个音频文件（用于多线程处理）
    
    Args:
        args_tuple: (src_path, src_dir, dst_dir, target_sr, overwrite)
    
    Returns:
        bool: 处理是否成功
    """
    src_path, src_dir, dst_dir, target_sr, overwrite = args_tuple
    
    # 计算相对路径并改变扩展名为.wav
    rel_path = src_path.relative_to(src_dir).with_suffix(".wav")
    dst_path = dst_dir / rel_path
    
    return resample_to_mono(src_path, dst_path, target_sr, overwrite)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="音频重采样和转换单声道工具")
    parser.add_argument("--src", type=Path, required=True, help="源音频目录")
    parser.add_argument("--dst", type=Path, required=True, help="输出目录")
    parser.add_argument("--sr", type=int, default=16000, help="目标采样率 (默认: 24000)")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的文件")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="并行处理线程数")
    
    args = parser.parse_args()
    
    # 检查源目录是否存在
    if not args.src.exists():
        print(f"错误: 源目录 {args.src} 不存在", file=sys.stderr)
        sys.exit(1)
    
    # 查找所有音频文件
    audio_files = find_audio_files(args.src)
    
    if not audio_files:
        print("警告: 未找到支持的音频文件", file=sys.stderr)
        sys.exit(1)
    
    print(f"找到 {len(audio_files)} 个音频文件")
    print(f"目标采样率: {args.sr} Hz")
    print(f"输出目录: {args.dst}")
    print(f"并行线程数: {args.jobs}")
    
    # 准备多线程处理参数
    process_args = [
        (audio_file, args.src, args.dst, args.sr, args.overwrite)
        for audio_file in audio_files
    ]
    
    # 使用多线程处理音频文件
    with ThreadPool(args.jobs) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file, process_args),
            total=len(audio_files),
            desc="处理音频文件"
        ))
    
    # 统计处理结果
    success_count = sum(results)
    print(f"step 2/5: ✅ All Finished! Resampled {success_count}/{len(results)} files -> {args.dst}")
    
    if success_count != len(results):
        print(f"警告: {len(results) - success_count} 个文件处理失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 