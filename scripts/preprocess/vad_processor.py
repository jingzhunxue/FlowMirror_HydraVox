#!/usr/bin/env python3
"""
基于Silero VAD的音频切分处理脚本
功能：使用VAD对音频进行智能切分，同时对过短的片段进行合并
"""

import argparse
import os
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
import warnings
from tqdm import tqdm
from silero_vad import load_silero_vad, get_speech_timestamps
warnings.filterwarnings('ignore')


class VADProcessor:
    def __init__(self, sample_rate=16000, merge_threshold=0.5, split_threshold=10.0):
        """
        初始化VAD处理器
        
        Args:
            sample_rate: 音频采样率
            merge_threshold: 最小音频长度阈值(秒)，小于此值的片段会合并
            split_threshold: 最大音频长度阈值(秒)，超过此值的音频会被切分
        """
        self.sample_rate = sample_rate
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        
        print("正在加载Silero VAD模型...")
        try:
            self.model = load_silero_vad()
            print("✓ VAD模型加载成功")
        except Exception as e:
            print(f"✗ VAD模型加载失败: {e}")
            raise
        
    def load_audio(self, file_path):
        """加载音频文件"""
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # 重采样到指定采样率
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform.squeeze()
        except Exception as e:
            raise RuntimeError(f"加载音频文件失败: {e}")
    
    def save_audio(self, waveform, output_path):
        """保存音频文件"""
        try:
            # 确保波形数据是2维的 [channels, samples]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            torchaudio.save(
                output_path,
                waveform,
                sample_rate=self.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16
            )
        except Exception as e:
            raise RuntimeError(f"保存音频文件失败: {e}")
    
    def get_speech_timestamps(self, audio, threshold=0.5, min_speech_duration_ms=250, 
                             min_silence_duration_ms=100, speech_pad_ms=30):
        """获取语音时间戳"""
        timestamps = get_speech_timestamps(
            audio,
            self.model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            sampling_rate=self.sample_rate,
            speech_pad_ms=speech_pad_ms
        )
        return timestamps
    
    def merge_short_segments(self, segments, threshold):
        """合并短音频片段"""
        if not segments:
            return []
        
        merged = []
        current_segment = segments[0]
        
        for segment in segments[1:]:
            segment_duration = (segment['end'] - segment['start']) / self.sample_rate
            
            if segment_duration < threshold:
                # 如果当前片段很短，合并到前一个或后一个片段
                merged_duration = (current_segment['end'] - current_segment['start']) / self.sample_rate
                
                # 如果前一个片段也短或两者合并后不长，就合并
                if merged_duration + segment_duration < self.split_threshold * 1.5:
                    current_segment['end'] = segment['end']
                else:
                    # 否则直接添加到合并列表，开始新片段
                    merged.append(current_segment)
                    current_segment = segment
            else:
                # 片段够长，直接添加前一个片段
                merged.append(current_segment)
                current_segment = segment
        
        # 添加最后一个片段
        if current_segment:
            # 检查最后一个片段是否需要合并
            final_duration = (current_segment['end'] - current_segment['start']) / self.sample_rate
            if final_duration < threshold and merged:
                # 合并到前一个片段
                merged[-1]['end'] = current_segment['end']
            else:
                merged.append(current_segment)
        
        return merged
    
    def process_audio(self, input_file, output_dir, file_prefix=None):
        """
        处理单个音频文件
        
        Args:
            input_file: 输入音频文件路径
            output_dir: 输出目录
            file_prefix: 输出文件前缀，如果为None则使用原文件名
        
        Returns:
            切分后的音频文件路径列表
        """
        try:
            filename = os.path.basename(input_file)
            
            # 检查输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 加载音频
            audio = self.load_audio(input_file)
            audio_duration = len(audio) / self.sample_rate
            
            # 如果音频较短，直接返回无需切分
            if audio_duration <= self.split_threshold:
                if audio_duration < self.merge_threshold:
                    print(f"  警告: 音频时长({audio_duration:.2f}s)小于合并阈值({self.merge_threshold}s)")
                
                output_filename = f"{file_prefix or Path(input_file).stem}.wav"
                output_path = os.path.join(output_dir, output_filename)
                self.save_audio(audio, output_path)
                return [output_path]
            
            # 获取语音时间戳
            speech_timestamps = self.get_speech_timestamps(audio)
            
            if not speech_timestamps:
                print(f"  未检测到语音片段")
                return []
            
            # 合并短片段
            merged_segments = self.merge_short_segments(speech_timestamps, self.merge_threshold)
            
            # 过滤有效片段
            valid_segments = []
            for segment in merged_segments:
                start_sample = int(segment['start'])
                end_sample = int(segment['end'])
                segment_audio = audio[start_sample:end_sample]
                segment_duration = len(segment_audio) / self.sample_rate
                
                if segment_duration >= 0.1:  # 保留至少100ms的片段
                    segment['duration'] = segment_duration
                    valid_segments.append(segment)
                        
            if not valid_segments:
                print(f"  没有有效的语音片段")
                return []
            
            # 保存切分后的音频
            output_files = []
            base_name = file_prefix or Path(input_file).stem
            
            for i, segment in enumerate(valid_segments):
                start_sample = int(segment['start'])
                end_sample = int(segment['end'])
                
                segment_audio = audio[start_sample:end_sample]
                output_filename = f"{base_name}_part{i+1:03d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                self.save_audio(segment_audio, output_path)
                output_files.append(output_path)
                
            print(f"  生成 {len(output_files)} 个片段")
            return output_files
            
        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_directory(self, input_dir, output_dir, recursive=False):
        """处理整个目录"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"扫描目录: {input_dir}")
        
        # 支持的音频格式
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma'}
        
        # 获取所有音频文件
        if recursive:
            audio_files = [f for f in input_path.rglob('*') if f.suffix.lower() in audio_extensions]
        else:
            audio_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print("未找到任何音频文件")
            return []
        
        print(f"找到 {len(audio_files)} 个音频文件")
        
        all_output_files = []
        
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            files = self.process_audio(str(audio_file), str(output_path), audio_file.stem)
            all_output_files.extend(files)
        
        print(f"处理完成，总共生成 {len(all_output_files)} 个文件")
        return all_output_files


def main():
    parser = argparse.ArgumentParser(description='🔊 基于Silero VAD的音频智能切分工具')
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output', required=True, help='输出目录路径')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='递归处理子目录')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='输出采样率 (默认: 16000)')
    parser.add_argument('--merge-threshold', type=float, default=0.5,
                       help='最小音频长度阈值(秒)，小于此值会被合并 (默认: 0.5)')
    parser.add_argument('--split-threshold', type=float, default=10.0,
                       help='最大音频长度阈值(秒)，超过此值会被切分 (默认: 10.0)')
    
    args = parser.parse_args()
    
    print("🔊 Silero VAD 音频切分工具")
    print("="*50)
    
    # 验证输入路径
    if not os.path.exists(args.input):
        print(f"错误: 路径不存在: {args.input}")
        return 1
    
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"采样率: {args.sample_rate}Hz")
    print(f"切分阈值: {args.split_threshold}s")
    print(f"合并阈值: {args.merge_threshold}s")
    
    # 创建VAD处理器
    try:
        processor = VADProcessor(
            sample_rate=args.sample_rate,
            merge_threshold=args.merge_threshold,
            split_threshold=args.split_threshold
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return 1
    
    # 开始处理
    print("="*50)
    
    start_time = time.time()
    total_files = 0
    
    try:
        if os.path.isfile(args.input):
            output_files = processor.process_audio(args.input, args.output)
        elif os.path.isdir(args.input):
            output_files = processor.process_directory(args.input, args.output, args.recursive)
        else:
            print(f"无效的路径类型: {args.input}")
            return 1
        
        total_files = len(output_files)
        
    except KeyboardInterrupt:
        print("\n用户中断处理")
        return 0
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    elapsed_time = time.time() - start_time
    
    print("="*50)
    print(f"总生成文件数: {total_files}")
    print(f"总耗时: {elapsed_time:.2f}秒")
    print("✅ 处理完成！")
    print(f"step 3/5: ✅ All Finished! created {total_files} files -> {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())