#!/usr/bin/env python3
# scripts/preprocess/convert_to_wav.py
"""
递归扫描目录，调用 ffmpeg 将所有音/视频文件转为统一格式 wav。
"""

import argparse
import subprocess
import sys
from pathlib import Path
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import os

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}

def convert_one(src: Path, dst: Path, sample_rate: int, overwrite: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return True
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(src),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(dst)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def find_files(path: Path):
    return [p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTS.union(VIDEO_EXTS)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="原始目录")
    parser.add_argument("--dst", type=Path, required=True, help="输出目录")
    parser.add_argument("--sr", type=int, default=16000, help="目标采样率")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在文件")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="并行线程数")
    args = parser.parse_args()

    all_files = find_files(args.src)
    if not all_files:
        print("No supported media files found.", file=sys.stderr)
        sys.exit(1)

    def _worker(src_path: Path):
        rel = src_path.relative_to(args.src).with_suffix(".wav")
        dst_path = args.dst / rel
        ok = convert_one(src_path, dst_path, args.sr, args.overwrite)
        return ok

    with ThreadPool(args.jobs) as pool:
        results = list(tqdm(pool.imap_unordered(_worker, all_files), total=len(all_files)))

    print(f"step 1/5: ✅ All Finished! Converted {sum(results)}/{len(results)} files -> {args.dst}")

if __name__ == "__main__":
    main()
