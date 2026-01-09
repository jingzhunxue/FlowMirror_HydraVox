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

# ----------- i18n -------------
_TRANSLATIONS = {
    "音频重采样和转换单声道工具": {"en": "Audio resample and mono conversion tool"},
    "源音频目录": {"en": "Source audio directory"},
    "输出目录": {"en": "Output directory"},
    "目标采样率 (默认: 24000)": {"en": "Target sample rate (default: 24000)"},
    "覆盖已存在的文件": {"en": "Overwrite existing files"},
    "并行处理线程数": {"en": "Parallel worker threads"},
    "错误: 源目录 {path} 不存在": {"en": "Error: source directory {path} does not exist"},
    "警告: 未找到支持的音频文件": {"en": "Warning: no supported audio files found"},
    "找到 {count} 个音频文件": {"en": "Found {count} audio files"},
    "目标采样率: {sr} Hz": {"en": "Target sample rate: {sr} Hz"},
    "输出目录: {path}": {"en": "Output directory: {path}"},
    "并行线程数: {count}": {"en": "Parallel workers: {count}"},
    "处理音频文件": {"en": "Processing audio files"},
    "step 2/5: ✅ All Finished! Resampled {success}/{total} files -> {path}": {
        "en": "step 2/5: ✅ All Finished! Resampled {success}/{total} files -> {path}"
    },
    "警告: {count} 个文件处理失败": {"en": "Warning: {count} files failed"},
    "处理文件 {path} 时出错: {error}": {"en": "Error processing file {path}: {error}"},
}


def _t(text: str, **kwargs) -> str:
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
        print(_t("处理文件 {path} 时出错: {error}", path=src_path, error=e))
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
    parser = argparse.ArgumentParser(description=_t("音频重采样和转换单声道工具"))
    parser.add_argument("--src", type=Path, required=True, help=_t("源音频目录"))
    parser.add_argument("--dst", type=Path, required=True, help=_t("输出目录"))
    parser.add_argument("--sr", type=int, default=16000, help=_t("目标采样率 (默认: 24000)"))
    parser.add_argument("--overwrite", action="store_true", help=_t("覆盖已存在的文件"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help=_t("并行处理线程数"))
    
    args = parser.parse_args()
    
    # 检查源目录是否存在
    if not args.src.exists():
        print(_t("错误: 源目录 {path} 不存在", path=args.src), file=sys.stderr)
        sys.exit(1)
    
    # 查找所有音频文件
    audio_files = find_audio_files(args.src)
    
    if not audio_files:
        print(_t("警告: 未找到支持的音频文件"), file=sys.stderr)
        sys.exit(1)
    
    print(_t("找到 {count} 个音频文件", count=len(audio_files)))
    print(_t("目标采样率: {sr} Hz", sr=args.sr))
    print(_t("输出目录: {path}", path=args.dst))
    print(_t("并行线程数: {count}", count=args.jobs))
    
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
            desc=_t("处理音频文件")
        ))
    
    # 统计处理结果
    success_count = sum(results)
    print(_t(
        "step 2/5: ✅ All Finished! Resampled {success}/{total} files -> {path}",
        success=success_count,
        total=len(results),
        path=args.dst,
    ))
    
    if success_count != len(results):
        print(_t("警告: {count} 个文件处理失败", count=len(results) - success_count))
        sys.exit(1)

if __name__ == "__main__":
    main() 
