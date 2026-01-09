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


class VADProcessor:
    def __init__(self, sample_rate=16000, merge_threshold=0.5, split_threshold=10.0, vad_threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=200, speech_pad_ms=30):
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
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        print(t("vad.loading_model"))
        try:
            self.model = load_silero_vad()
            print(t("vad.model_loaded"))
        except Exception as e:
            print(t("vad.model_load_failed", error=e))
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
            raise RuntimeError(t("vad.load_audio_failed", error=e))
    
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
            raise RuntimeError(t("vad.save_audio_failed", error=e))
    
    def get_speech_timestamps(self, audio):
        """获取语音时间戳"""
        timestamps = get_speech_timestamps(
            audio,
            self.model,
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            sampling_rate=self.sample_rate,
            speech_pad_ms=self.speech_pad_ms
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
                    print(
                        t(
                            "vad.audio_too_short_warn",
                            duration=audio_duration,
                            threshold=self.merge_threshold,
                        )
                    )
                
                output_filename = f"{file_prefix or Path(input_file).stem}.wav"
                output_path = os.path.join(output_dir, output_filename)
                self.save_audio(audio, output_path)
                return [output_path]
            
            # 获取语音时间戳
            speech_timestamps = self.get_speech_timestamps(audio)
            
            if not speech_timestamps:
                print(t("vad.no_speech_segments"))
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
                print(t("vad.no_valid_segments"))
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
                
            print(t("vad.segments_generated", count=len(output_files)))
            return output_files
            
        except Exception as e:
            print(t("vad.process_failed", error=e))
            import traceback
            traceback.print_exc()
            return []
    
    def process_directory(self, input_dir, output_dir, recursive=False):
        """处理整个目录"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(t("vad.scan_dir", input_dir=input_dir))
        
        # 支持的音频格式
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma'}
        
        # 获取所有音频文件
        if recursive:
            audio_files = [f for f in input_path.rglob('*') if f.suffix.lower() in audio_extensions]
        else:
            audio_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print(t("vad.no_audio_files"))
            return []
        
        print(t("vad.found_audio_files", count=len(audio_files)))
        
        all_output_files = []
        
        for audio_file in tqdm(audio_files, desc=t("vad.processing_audio_desc")):
            files = self.process_audio(str(audio_file), str(output_path), audio_file.stem)
            all_output_files.extend(files)
        
        print(t("vad.process_complete_count", count=len(all_output_files)))
        return all_output_files


def main():
    parser = argparse.ArgumentParser(description=t("vad.cli_description"))
    parser.add_argument('input', help=t("vad.cli_input"))
    parser.add_argument('-o', '--output', required=True, help=t("vad.cli_output"))
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help=t("vad.cli_recursive"))
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help=t("vad.cli_sample_rate"))
    parser.add_argument('--vad-threshold', type=float, default=0.5,
                       help=t("vad.cli_vad_threshold"))
    parser.add_argument('--min-speech-duration-ms', type=int, default=250,
                       help=t("vad.cli_min_speech"))
    parser.add_argument('--min-silence-duration-ms', type=int, default=200,
                       help=t("vad.cli_min_silence"))
    parser.add_argument('--speech-pad-ms', type=int, default=30,
                       help=t("vad.cli_speech_pad"))
    parser.add_argument('--merge-threshold', type=float, default=0.5,
                       help=t("vad.cli_merge_threshold"))
    parser.add_argument('--split-threshold', type=float, default=10.0,
                       help=t("vad.cli_split_threshold"))
    
    args = parser.parse_args()
    
    print(t("vad.title"))
    print("="*50)
    
    # 验证输入路径
    if not os.path.exists(args.input):
        print(t("vad.path_not_found", path=args.input))
        return 1
    
    print(t("vad.input", input=args.input))
    print(t("vad.output", output=args.output))
    print(t("vad.sample_rate", sample_rate=args.sample_rate))
    print(t("vad.split_threshold", threshold=args.split_threshold))
    print(t("vad.merge_threshold", threshold=args.merge_threshold))
    
    # 创建VAD处理器
    try:
        processor = VADProcessor(
            sample_rate=args.sample_rate,
            merge_threshold=args.merge_threshold,
            split_threshold=args.split_threshold,
            min_silence_duration_ms=args.min_silence_duration_ms,
            speech_pad_ms=args.speech_pad_ms
        )
    except Exception as e:
        print(t("vad.init_failed", error=e))
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
            print(t("vad.invalid_path_type", path=args.input))
            return 1
        
        total_files = len(output_files)
        
    except KeyboardInterrupt:
        print("\n" + t("vad.user_interrupt"))
        return 0
    except Exception as e:
        print(t("vad.process_error", error=e))
        import traceback
        traceback.print_exc()
        return 1
    
    elapsed_time = time.time() - start_time
    
    print("="*50)
    print(t("vad.total_files", count=total_files))
    print(t("vad.total_time", seconds=elapsed_time))
    print(t("vad.done"))
    print(t("vad.step_done", count=total_files, output=args.output))
    
    return 0


if __name__ == '__main__':
    exit(main())
