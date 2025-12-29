import os, gradio as gr
import pandas as pd
from typing import List, Tuple
import json
import subprocess, time, sys
from pathlib import Path
import re

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}


def upload_audio_files(files):
    """处理上传的音频文件"""
    if not files:
        gr.Warning("请选择音频文件")
        return "未选择文件", []
    
    file_info = []
    for file in files:
        if file:
            file_info.append({
                "文件名": os.path.basename(file.name),
                "大小": f"{os.path.getsize(file.name) / 1024:.1f} KB",
                "路径": file.name
            })
    
    df = pd.DataFrame(file_info)
    return f"已上传 {len(files)} 个音频文件", df

def process_text_annotation(audio_files, text_content: str):
    """处理文本标注"""
    if not audio_files:
        gr.Warning("请先上传音频文件")
        return "请先上传音频文件"
    
    if not text_content.strip():
        gr.Warning("请输入标注文本")
        return "请输入标注文本"
    
    lines = text_content.strip().split('\n')
    annotations = []
    
    for i, line in enumerate(lines):
        if line.strip():
            annotations.append({
                "音频ID": f"audio_{i+1}",
                "文本": line.strip(),
                "状态": "已标注"
            })
    
    df = pd.DataFrame(annotations)
    return df

def validate_dataset(dataset_df):
    """验证数据集质量"""
    if dataset_df is None or len(dataset_df) == 0:
        return "数据集为空"
    
    issues = []
    
    # 检查文本长度
    for idx, row in dataset_df.iterrows():
        text = str(row.get('文本', ''))
        if len(text) < 5:
            issues.append(f"第{idx+1}行: 文本过短")
        elif len(text) > 200:
            issues.append(f"第{idx+1}行: 文本过长")
    
    if not issues:
        return "✅ 数据集验证通过，无问题发现"
    else:
        return f"⚠️ 发现 {len(issues)} 个问题:\n" + "\n".join(issues[:10])

def export_dataset(dataset_df, format_type: str):
    """导出数据集"""
    if dataset_df is None or len(dataset_df) == 0:
        gr.Warning("没有可导出的数据")
        return None
    
    if format_type == "CSV":
        output_path = "/tmp/dataset.csv"
        dataset_df.to_csv(output_path, index=False)
    elif format_type == "JSON":
        output_path = "/tmp/dataset.json"
        dataset_df.to_json(output_path, orient='records', ensure_ascii=False, indent=2)
    else:
        gr.Warning("不支持的格式")
        return None
    
    return output_path

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _convert_script_path() -> Path:
    return _project_root() / "scripts/preprocess/convert_to_wav.py"


def _vad_script_path() -> Path:
    return _project_root() / "scripts/preprocess/vad_processor.py"

def _asr_script_path() -> Path:
    return _project_root() / "scripts/preprocess/transcribe_to_dataset.py"


def _generate_default_output_dir(input_dir: str, suffix: str) -> str:
    if not input_dir:
        return ""
    try:
        base = Path(input_dir)
        parent = base.parent
        if parent == base:
            return str(base.with_name(base.name + suffix))
        return str(parent / f"{base.name}{suffix}")
    except Exception:
        return ""


def _list_media_files(base_dir: str) -> List[Path]:
    if not base_dir or not os.path.isdir(base_dir):
        return []
    base = Path(base_dir)
    exts = AUDIO_EXTS.union(VIDEO_EXTS)
    return [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _build_expected_outputs(src_dir: str, dst_dir: str) -> List[Path]:
    files = _list_media_files(src_dir)
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    expected: List[Path] = []
    for f in files:
        try:
            rel = f.relative_to(src_path).with_suffix(".wav")
        except ValueError:
            # If f is not under src_path, skip
            continue
        expected.append(dst_path / rel)
    return expected


def _count_existing(paths: List[Path]) -> int:
    cnt = 0
    for p in paths:
        if p.exists():
            cnt += 1
    return cnt


def _auto_detect_device_and_processes() -> Tuple[str, int, str]:
    """返回 (device, num_processes, detail_msg). device 为 'GPU' 或 'CPU'。"""
    device = "CPU"
    num_proc = 1
    detail = "CUDA 不可用，默认 CPU x1"
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            n = torch.cuda.device_count() or 1
            device = "GPU"
            num_proc = n
            detail = f"CUDA 可用，GPU 数: {n}"
    except Exception:
        pass
    return device, num_proc, detail


def preview_stage1(input_dir: str, output_dir: str):
    if not input_dir or not os.path.isdir(input_dir):
        gr.Warning("请输入有效的输入目录")
        return pd.DataFrame([]), 0, "❗ 输入目录无效"
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_resample")
    files = _list_media_files(input_dir)
    expected = _build_expected_outputs(input_dir, output_dir)
    preview_rows = []
    for i, (src, dst) in enumerate(zip(files, expected)):
        if i >= 50:
            break
        preview_rows.append({"源文件": str(src), "目标文件": str(dst)})
    df = pd.DataFrame(preview_rows)
    status = f"将处理 {len(files)} 个文件，输出至 {output_dir}"
    return df, len(files), status


def run_stage1(input_dir: str, output_dir: str, sample_rate: int, overwrite: bool):
    """以子进程方式启动转换，并周期性统计进度（通过目标文件存在数）。"""
    if not input_dir or not os.path.isdir(input_dir):
        yield 0, "❗ 输入目录无效", ""
        return
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_resample")
    expected = _build_expected_outputs(input_dir, output_dir)
    total = len(expected)
    if total == 0:
        yield 0, "没有可处理的媒体文件", ""
        return

    script_path = _convert_script_path()
    if not script_path.exists():
        yield 0, f"找不到脚本: {script_path}", ""
        return

    cmd = [
        sys.executable,
        str(script_path),
        "--src", input_dir,
        "--dst", output_dir,
        "--sr", str(int(sample_rate)),
    ]
    if overwrite:
        cmd.append("--overwrite")

    start_time = time.time()
    try:
        proc = subprocess.Popen(cmd)
    except Exception as e:
        yield 0, f"启动失败: {e}", ""
        return

    # 轮询进度
    last = -1
    while True:
        ret = proc.poll()
        done = _count_existing(expected)
        pct = int(done * 100 / total) if total else 0
        elapsed = int(time.time() - start_time)
        if pct != last:
            status = f"进行中: {done}/{total} ({pct}%) · 用时 {elapsed}s"
            yield pct, status, ""
            last = pct
        if ret is not None:
            break
        time.sleep(1.0)

    # 完成/失败状态
    done = _count_existing(expected)
    pct = int(done * 100 / total) if total else 0
    elapsed = int(time.time() - start_time)
    if proc.returncode == 0:
        yield 100, f"✅ 完成: {done}/{total} · 总用时 {elapsed}s", ""
    else:
        yield pct, f"❌ 失败: 已完成 {done}/{total} · 总用时 {elapsed}s", ""


def _sync_output_dir(input_dir: str, auto_sync: bool, suffix: str):
    if auto_sync and input_dir:
        return _generate_default_output_dir(input_dir, suffix)
    return gr.update()


def _chain_next_input(prev_output_dir: str, link_enabled: bool):
    if link_enabled and prev_output_dir:
        return prev_output_dir
    return gr.update()


def _refresh_device_once():
    d, p, detail = _auto_detect_device_and_processes()
    return detail, p

def _refresh_device_triplet():
    d, p, detail = _auto_detect_device_and_processes()
    return detail, p, ("GPU" if d == "GPU" else "CPU")


def preview_stage2(input_dir: str, output_dir: str):
    if not input_dir or not os.path.isdir(input_dir):
        return "❗ 输入目录无效"
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_vad")
    # 粗略统计：输入音频文件数量
    audio_files = [p for p in Path(input_dir).rglob('*') if p.suffix.lower() in {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma'}]
    return f"将处理约 {len(audio_files)} 个音频文件，输出至 {output_dir}"


def run_stage2(input_dir: str,
               output_dir: str,
               threshold: float,
               min_speech_ms: float,
               min_silence_ms: float,
               pad_ms: float,
               min_seg_s: float,
               max_seg_s: float,
               link_enabled: bool):
    """运行 VAD 处理脚本，并解析 stdout 实时更新进度。
    输出：(progress_percent, status_text, log_text, next_stage_input)
    """
    if not input_dir or not os.path.isdir(input_dir):
        yield 0, "❗ 输入目录无效", "", gr.update()
        return
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_vad")

    script_path = _vad_script_path()
    if not script_path.exists():
        yield 0, f"找不到脚本: {script_path}", "", gr.update()
        return

    cmd = [
        sys.executable,
        str(script_path),
        str(input_dir),
        "-o", str(output_dir),
        "--sample-rate", "16000",
        "--vad-threshold", str(float(threshold)),
        "--min-speech-duration-ms", str(int(min_speech_ms)),
        "--min-silence-duration-ms", str(int(min_silence_ms)),
        "--speech-pad-ms", str(int(pad_ms)),
        "--merge-threshold", str(float(min_seg_s)),
        "--split-threshold", str(float(max_seg_s)),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        yield 0, f"启动失败: {e}", "", gr.update()
        return

    start_time = time.time()
    log_lines: List[str] = []
    total = None
    current = 0
    last_pct = -1

    # 初始提示
    yield 0, "VAD 处理中...", "", gr.update()

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            # 累积日志（仅保留最近 50 行）
            log_lines.append(line)
            if len(log_lines) > 50:
                log_lines = log_lines[-50:]

            # 解析总数
            m_total = re.search(r"找到\s+(\d+)\s+个音频文件", line)
            if m_total:
                try:
                    total = int(m_total.group(1))
                except Exception:
                    total = total

            # 解析 tqdm 进度 "处理音频文件: ... 12/34 ..."
            if "处理音频文件" in line:
                m_prog = re.search(r"(\d+)\s*/\s*(\d+)", line)
                if m_prog:
                    try:
                        current = int(m_prog.group(1))
                        total = int(m_prog.group(2))
                    except Exception:
                        pass

            pct = None
            if total and total > 0:
                pct = int(max(0, min(100, current * 100 // total)))

            elapsed = int(time.time() - start_time)
            status = f"进行中: {current}/{total if total else '?'} · 用时 {elapsed}s"

            if pct is not None and pct != last_pct:
                yield pct, status, "\n".join(log_lines), gr.update()
                last_pct = pct
            else:
                # 仅更新日志与状态
                yield (last_pct if last_pct >= 0 else 0), status, "\n".join(log_lines), gr.update()

        ret = proc.wait()
    except Exception as e:
        yield (last_pct if last_pct >= 0 else 0), f"❌ 运行异常: {e}", "\n".join(log_lines), gr.update()
        return

    elapsed = int(time.time() - start_time)
    if proc.returncode == 0:
        final_status = f"✅ 完成 · 用时 {elapsed}s"
        # 若启用串联，将下阶段输入设置为本阶段输出目录
        next_input = output_dir if link_enabled else gr.update()
        yield 100, final_status, "\n".join(log_lines), next_input
    else:
        yield (last_pct if last_pct >= 0 else 0), f"❌ 失败 · 用时 {elapsed}s", "\n".join(log_lines), gr.update()


def preview_stage3(input_dir: str, output_dir: str):
    if not input_dir or not os.path.isdir(input_dir):
        return "❗ 输入目录无效"
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_asr")
    wav_files = list(Path(input_dir).rglob("*.wav"))
    mp3_files = list(Path(input_dir).rglob("*.mp3"))
    return f"将转录 {len(wav_files)} 个 .wav 与 {len(mp3_files)} 个 .mp3，输出到 {output_dir}"


def run_stage3(input_dir: str,
               output_dir: str,
               device_choice: str,
               num_processes: float):
    """运行 ASR 转录脚本。
    输出：(progress_percent, status_text, log_text)
    """
    if not input_dir or not os.path.isdir(input_dir):
        yield 0, "❗ 输入目录无效", ""
        return
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_asr")

    script_path = _asr_script_path()
    if not script_path.exists():
        yield 0, f"找不到脚本: {script_path}", ""
        return

    # 设备与进程选择
    chosen = device_choice
    dev_detect, gpu_count, _detail = _auto_detect_device_and_processes()
    if chosen == "自动":
        chosen = "GPU" if dev_detect == "GPU" else "CPU"
    use_cuda = (chosen == "GPU" and dev_detect == "GPU")
    device_flag = "cuda" if use_cuda else "cpu"
    try:
        nproc = max(1, int(num_processes))
    except Exception:
        nproc = 1

    gpu_devices = []
    if use_cuda:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                cnt = torch.cuda.device_count()
                take = min(nproc, cnt)
                gpu_devices = list(range(take))
        except Exception:
            gpu_devices = []
            device_flag = "cpu"

    cmd = [
        sys.executable,
        str(script_path),
        "--src", str(input_dir),
        "--dst", str(output_dir),
        "--device", device_flag,
        "--num_workers", str(nproc),
    ]
    if device_flag == "cuda" and gpu_devices:
        cmd.extend(["--gpu_devices", ",".join(str(x) for x in gpu_devices)])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        yield 0, f"启动失败: {e}", ""
        return

    start_time = time.time()
    log_lines: List[str] = []
    last_pct = -1
    total_files = None
    # 多进程解析数据（可选，若解析失败则回退到日志）
    worker_chunks = {}
    worker_pct = {}
    num_workers_detected = None

    yield 0, "ASR 转录中...", ""

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            log_lines.append(line)
            if len(log_lines) > 200:
                log_lines = log_lines[-200:]

            # 解析总数
            m_total = re.search(r"找到\s+(\d+)\s+个\s+\.wav\s+文件和\s+(\d+)\s+个\s+\.mp3\s+文件", line)
            if m_total:
                try:
                    total_files = int(m_total.group(1)) + int(m_total.group(2))
                except Exception:
                    pass

            # 解析多进程总体信息
            m_multi = re.search(r"启动多进程处理:\s*(\d+)\s*个工作进程处理\s*(\d+)\s*个文件", line)
            if m_multi:
                try:
                    num_workers_detected = int(m_multi.group(1))
                    total_files = int(m_multi.group(2))
                except Exception:
                    pass
            m_chunk = re.search(r"启动工作进程\s*(\d+)，分配\s*(\d+)\s*个文件", line)
            if m_chunk:
                try:
                    wid = int(m_chunk.group(1)); size = int(m_chunk.group(2))
                    worker_chunks[wid] = size
                except Exception:
                    pass

            # 解析单进度（tqdm 百分比）
            m_asr_pct = re.search(r"ASR.*?(\d+)%\|", line)
            if m_asr_pct:
                try:
                    pct = int(m_asr_pct.group(1))
                    last_pct = pct
                except Exception:
                    pass

            m_worker_pct = re.search(r"Worker\s*(\d+).*?(\d+)%\|", line)
            if m_worker_pct:
                try:
                    wid = int(m_worker_pct.group(1)); pct = int(m_worker_pct.group(2))
                    worker_pct[wid] = pct
                    if worker_chunks:
                        total_chunk = sum(worker_chunks.values()) or len(worker_pct)
                        weighted = 0
                        for w, p in worker_pct.items():
                            weight = worker_chunks.get(w, 1)
                            weighted += p * weight
                        last_pct = int(weighted / total_chunk)
                    else:
                        # 平均
                        last_pct = int(sum(worker_pct.values()) / max(1, len(worker_pct)))
                except Exception:
                    pass

            elapsed = int(time.time() - start_time)
            status = f"进行中 · 用时 {elapsed}s"
            yield (last_pct if last_pct >= 0 else 0), status, "\n".join(log_lines)

        ret = proc.wait()
    except Exception as e:
        yield (last_pct if last_pct >= 0 else 0), f"❌ 运行异常: {e}", "\n".join(log_lines)
        return

    elapsed = int(time.time() - start_time)
    if proc.returncode == 0:
        msg = f"✅ 完成 · 用时 {elapsed}s"
        yield 100, msg, "\n".join(log_lines)
    else:
        yield (last_pct if last_pct >= 0 else 0), f"❌ 失败 · 用时 {elapsed}s", "\n".join(log_lines)



def _parse_comma_dirs(input_dirs_text: str) -> List[str]:
    if not input_dirs_text:
        return []
    parts = [p.strip() for p in input_dirs_text.split(',')]
    return [p for p in parts if p]


def _dataset_total_len(ds) -> int:
    try:
        from datasets import Dataset, DatasetDict  # type: ignore
        if isinstance(ds, Dataset):
            return len(ds)
        if isinstance(ds, DatasetDict):
            return sum(len(v) for v in ds.values())
    except Exception:
        pass
    try:
        return len(ds)
    except Exception:
        return 0


def _flatten_to_datasets(ds_obj) -> List["Dataset"]:
    from datasets import Dataset, DatasetDict  # type: ignore
    if isinstance(ds_obj, Dataset):
        return [ds_obj]
    if isinstance(ds_obj, DatasetDict):
        return [v for v in ds_obj.values()]
    return []


def preview_stage4(input_dirs_text: str, output_dir: str):
    paths = _parse_comma_dirs(input_dirs_text)
    if not paths:
        return "❗ 请输入至少一个输入目录，使用逗号分隔"
    try:
        from datasets import load_from_disk  # type: ignore
    except Exception as e:
        return f"缺少datasets依赖或导入失败: {e}"

    lines: List[str] = []
    total = 0
    ok = 0
    for p in paths:
        if not os.path.isdir(p):
            lines.append(f"- 跳过（非目录）: {p}")
            continue
        try:
            ds = load_from_disk(p)
            n = _dataset_total_len(ds)
            lines.append(f"- ✓ {p} · {n} 条")
            total += n
            ok += 1
        except Exception as e:
            lines.append(f"- ✗ {p} · 加载失败: {e}")

    out = output_dir or "(未指定，建议填写保存目录)"
    head = [f"将合并 {ok}/{len(paths)} 个可用数据集，总计约 {total} 条", f"输出目录: {out}"]
    return "\n".join(head + ["", *lines])


def run_stage4_merge(input_dirs_text: str, output_dir: str):
    """合并多个 HuggingFace 数据集（支持 Dataset / DatasetDict），并保存到 output_dir。
    进度按阶段粗略估计并提供日志。
    """
    paths = _parse_comma_dirs(input_dirs_text)
    if not paths:
        yield 0, "❗ 请输入至少一个输入目录", ""
        return
    if not output_dir:
        yield 0, "❗ 请输入输出目录", ""
        return
    try:
        from datasets import load_from_disk, concatenate_datasets  # type: ignore
    except Exception as e:
        yield 0, f"缺少datasets依赖或导入失败: {e}", ""
        return

    log_lines: List[str] = []
    ds_list_all = []
    ok = 0
    total_dirs = len(paths)

    # 读取阶段
    for idx, p in enumerate(paths, start=1):
        if not os.path.isdir(p):
            log_lines.append(f"跳过（非目录）: {p}")
            yield int(idx * 10 / max(1, total_dirs)), f"读取中 ({idx}/{total_dirs})", "\n".join(log_lines)
            continue
        try:
            ds_obj = load_from_disk(p)
            subs = _flatten_to_datasets(ds_obj)
            if not subs:
                log_lines.append(f"{p} · 不含可用 split，已跳过")
            else:
                for ds in subs:
                    ds_list_all.append(ds)
                ln = sum(len(s) for s in subs)
                log_lines.append(f"{p} · 读取 {ln} 条")
                ok += 1
        except Exception as e:
            log_lines.append(f"{p} · 加载失败: {e}")
        yield int(idx * 10 / max(1, total_dirs)), f"读取中 ({idx}/{total_dirs})", "\n".join(log_lines)

    if not ds_list_all:
        yield 0, "❌ 没有可合并的数据集", "\n".join(log_lines)
        return

    # 对齐列（取交集）
    try:
        all_cols = [set(ds.column_names) for ds in ds_list_all]
        common_cols = list(set.intersection(*all_cols)) if all_cols else []
        if not common_cols:
            yield 0, "❌ 各数据列无交集，无法合并", "\n".join(log_lines)
            return
        log_lines.append(f"列对齐（交集）: {sorted(common_cols)}")
        yield 20, "对齐字段", "\n".join(log_lines)
        ds_aligned = [ds.select_columns(common_cols) for ds in ds_list_all]
    except Exception as e:
        yield 0, f"❌ 对齐列失败: {e}", "\n".join(log_lines)
        return

    # 合并阶段
    try:
        merged = concatenate_datasets(ds_aligned)
        log_lines.append(f"合并完成，合计 {len(merged)} 条")
        yield 60, "合并中", "\n".join(log_lines)
    except Exception as e:
        yield 0, f"❌ 合并失败: {e}", "\n".join(log_lines)
        return

    # 保存阶段
    try:
        from pathlib import Path as _Path
        _Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        merged.save_to_disk(output_dir)
        log_lines.append(f"已保存至 {output_dir}")
        yield 100, f"✅ 合并完成 · 共 {len(merged)} 条", "\n".join(log_lines)
    except Exception as e:
        yield 90, f"❌ 保存失败: {e}", "\n".join(log_lines)
        return


def create_data_tab():
    """创建数据处理tab界面"""
    with gr.Tab("📊 数据处理"):
        gr.Markdown("""
        # 🛠️ 音频数据预处理工作流
        
        **三个阶段的处理流程：** 格式转换 → VAD分段 → ASR转录
        **可选阶段：** 数据集合并
        """)
        
        device_default, proc_default, device_detail = _auto_detect_device_and_processes()

        # 阶段1：格式转换与重采样
        with gr.Accordion("🎵 阶段 1 - 格式转换与重采样", open=False):
            gr.Markdown("**功能：** 将各种音频/视频格式统一转换为 16kHz WAV 格式")
            
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        s1_input_dir = gr.Textbox(label="📁 输入目录", placeholder="/path/to/input_dir", scale=3)
                        s1_auto_sync = gr.Checkbox(value=True, label="🔄 自动同步输出路径", info="添加_resample后缀", scale=1)
                        s1_output_dir = gr.Textbox(label="📂 输出目录", placeholder="自动同步或手动填写", scale=3)
                        
                    with gr.Row():
                        s1_sr = gr.Dropdown(
                            choices=[8000,16000,22050,44100,48000], 
                            value=16000, 
                            label="🎤 采样率 (Hz)",
                            scale=1
                        )
                        s1_overwrite = gr.Checkbox(value=False, label="⚠️ 覆盖已存在文件", scale=1)
                        
                    with gr.Row():
                        s1_preview_btn = gr.Button("👀 预览变更", variant="secondary", scale=1)
                        s1_start_btn = gr.Button("▶️ 开始处理", variant="primary", scale=1)
                        
                    s1_preview_df = gr.Dataframe(
                        headers=["源文件", "目标文件"], 
                        label="📋 映射预览（前50条）", 
                        interactive=False
                    )
                    
                    with gr.Row():
                        s1_total_num = gr.Number(label="📊 待处理文件数", interactive=False, scale=1)
                        s1_progress = gr.Slider(0, 100, value=0, step=1, label="📈 进度 (%)", interactive=False, scale=2)
                        
                    s1_status = gr.Textbox(label="📋 状态", interactive=False)
                    s1_log = gr.Textbox(label="📝 运行日志", lines=4, interactive=False, show_copy_button=True)

        # 阶段2：VAD 处理（Silero）
        with gr.Accordion("🔊 阶段 2 - VAD 语音活动检测", open=False):
            gr.Markdown("**功能：** 使用 Silero VAD 检测并分割语音片段，去除静音部分")
            
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        s2_input_dir = gr.Textbox(label="📁 输入目录", placeholder="默认衔接阶段1输出", scale=3)
                        s2_auto_sync = gr.Checkbox(value=True, label="🔄 自动同步输出路径", info="添加_vad后缀", scale=1)
                        s2_output_dir = gr.Textbox(label="📂 输出目录", placeholder="自动同步或手动填写", scale=3)
                        
                    with gr.Accordion("⚙️ VAD 参数设置", open=False):
                        with gr.Row():
                            s2_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="🎯 置信度阈值", info="越高越严格")
                            s2_min_speech_ms = gr.Number(value=250, label="🗣️ 最短语音 (ms)")
                            s2_min_silence_ms = gr.Number(value=200, label="🔇 最短静音 (ms)")
                            s2_pad_ms = gr.Number(value=30, label="🔧 前后填充 (ms)")
                        with gr.Row():
                            s2_min_seg = gr.Number(value=0.5, label="⏱️ 最短片段 (s)")
                            s2_max_seg = gr.Number(value=30, label="⏰ 最长片段 (s)")
                            
                    with gr.Row():
                        s2_preview_btn = gr.Button("👀 预览", variant="secondary", scale=1)
                        s2_start_btn = gr.Button("▶️ 开始处理", variant="primary", scale=1)
                        
                    with gr.Row():
                        s2_progress = gr.Slider(0, 100, value=0, step=1, label="📈 进度 (%)", interactive=False)
                        
                    s2_status = gr.Textbox(label="📋 状态", interactive=False)
                    s2_log = gr.Textbox(label="📝 运行日志", lines=4, interactive=False, show_copy_button=True)

        # 阶段3：ASR 处理
        with gr.Accordion("🎙️ 阶段 3 - ASR 语音识别转录", open=False):
            gr.Markdown("**功能：** 使用语音识别技术将音频转换为文本，生成训练数据集")
            
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        s3_input_dir = gr.Textbox(label="📁 输入目录", placeholder="默认衔接阶段2输出", scale=3)
                        s3_auto_sync = gr.Checkbox(value=True, label="🔄 自动同步输出路径", info="添加_asr后缀", scale=1)
                        s3_output_dir = gr.Textbox(label="📂 输出目录", placeholder="自动同步或手动填写", scale=3)
                        
                    with gr.Accordion("⚙️ 计算资源设置", open=False):
                        with gr.Row():
                            s3_device = gr.Dropdown(
                                choices=["自动", "CPU", "GPU"], 
                                value=("GPU" if device_default=="GPU" else "CPU"), 
                                label="💻 计算设备"
                            )
                            s3_processes = gr.Number(value=proc_default, label="🔄 并行进程数")
                            s3_detect_btn = gr.Button("🔄 刷新设备检测", variant="secondary", size="sm")
                        s3_device_info = gr.Textbox(value=device_detail, label="ℹ️ 设备检测信息", interactive=False)
                        
                    with gr.Row():
                        s3_preview_btn = gr.Button("👀 预览", variant="secondary", scale=1)
                        s3_start_btn = gr.Button("▶️ 开始处理", variant="primary", scale=1)
                        
                    with gr.Row():
                        s3_progress = gr.Slider(0, 100, value=0, step=1, label="📈 进度 (%)", interactive=False)
                        
                    s3_status = gr.Textbox(label="📋 状态", interactive=False)
                    s3_log = gr.Textbox(label="📝 运行日志", lines=4, interactive=False, show_copy_button=True)

        # 阶段4：数据集合并（可选）
        with gr.Accordion("🧩 阶段 4 - 数据集合并 (可选)", open=False):
            gr.Markdown("**功能：** 将多个前面阶段生成的数据集目录合并为一个新的 HuggingFace 数据集。输入多个目录时使用英文逗号分隔。")
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        s4_input_dirs = gr.Textbox(label="📁 输入数据集目录（逗号分隔）", placeholder="/path/to/ds1,/path/to/ds2,...", scale=3)
                        s4_output_dir = gr.Textbox(label="📂 合并输出目录", placeholder="/path/to/merged_dataset", scale=3)
                    with gr.Row():
                        s4_preview_btn = gr.Button("👀 预览", variant="secondary", scale=1)
                        s4_start_btn = gr.Button("▶️ 开始合并", variant="primary", scale=1)
                    with gr.Row():
                        s4_progress = gr.Slider(0, 100, value=0, step=1, label="📈 进度 (%)", interactive=False)
                    s4_status = gr.Textbox(label="📋 状态", interactive=False)
                    s4_log = gr.Textbox(label="📝 合并日志", lines=6, interactive=False, show_copy_button=True)

        gr.Markdown("""
        ---
        
        ## 💡 使用提示
        
        - **阶段顺序不可颠倒**：每个阶段都依赖前一阶段的输出
        - **GPU 加速**：阶段3支持GPU加速，可显著提升处理速度
        - **监控进度**：每个阶段都有实时进度显示和详细日志
        - **可选合并**：阶段4可将多个阶段产出的数据集进行合并
        
        ⚠️ **注意**：处理大量文件时请确保有足够的磁盘空间和计算资源
        """)

        # 隐藏的状态变量，用于单独运行阶段时传递link_enabled=False
        link_disabled_state = gr.State(value=False)

        # 事件绑定（预处理）
        # 阶段1：自动同步输出、链到阶段2输入
        s1_input_dir.change(
            fn=lambda d, a: _sync_output_dir(d, a, "_resample"),
            inputs=[s1_input_dir, s1_auto_sync],
            outputs=s1_output_dir,
        )
        s1_auto_sync.change(
            fn=lambda a, d: _sync_output_dir(d, a, "_resample"),
            inputs=[s1_auto_sync, s1_input_dir],
            outputs=s1_output_dir,
        )

        # 阶段2：自动同步输出、链到阶段3输入
        s2_input_dir.change(
            fn=lambda d, a: _sync_output_dir(d, a, "_vad"),
            inputs=[s2_input_dir, s2_auto_sync],
            outputs=s2_output_dir,
        )
        s2_auto_sync.change(
            fn=lambda a, d: _sync_output_dir(d, a, "_vad"),
            inputs=[s2_auto_sync, s2_input_dir],
            outputs=s2_output_dir,
        )

        # 阶段3：自动同步输出
        s3_input_dir.change(
            fn=lambda d, a: _sync_output_dir(d, a, "_asr"),
            inputs=[s3_input_dir, s3_auto_sync],
            outputs=s3_output_dir,
        )
        s3_auto_sync.change(
            fn=lambda a, d: _sync_output_dir(d, a, "_asr"),
            inputs=[s3_auto_sync, s3_input_dir],
            outputs=s3_output_dir,
        )

        # 阶段1：预览与开始处理
        s1_preview_btn.click(
            fn=preview_stage1,
            inputs=[s1_input_dir, s1_output_dir],
            outputs=[s1_preview_df, s1_total_num, s1_status],
        )
        s1_start_btn.click(
            fn=run_stage1,
            inputs=[s1_input_dir, s1_output_dir, s1_sr, s1_overwrite],
            outputs=[s1_progress, s1_status, s1_log],
        )

        # 阶段2：预览与开始处理
        s2_preview_btn.click(
            fn=preview_stage2,
            inputs=[s2_input_dir, s2_output_dir],
            outputs=s2_status,
        )
        s2_start_btn.click(
            fn=run_stage2,
            inputs=[s2_input_dir, s2_output_dir, s2_threshold, s2_min_speech_ms, s2_min_silence_ms, s2_pad_ms, s2_min_seg, s2_max_seg, link_disabled_state],
            outputs=[s2_progress, s2_status, s2_log, s3_input_dir],
        )

        # 阶段3：预览与开始处理
        s3_preview_btn.click(
            fn=preview_stage3,
            inputs=[s3_input_dir, s3_output_dir],
            outputs=s3_status,
        )
        s3_start_btn.click(
            fn=run_stage3,
            inputs=[s3_input_dir, s3_output_dir, s3_device, s3_processes],
            outputs=[s3_progress, s3_status, s3_log],
        )

        # 阶段3：刷新设备
        s3_detect_btn.click(
            fn=_refresh_device_triplet,
            inputs=[],
            outputs=[s3_device_info, s3_processes, s3_device],
        )

        # 阶段4：预览与开始合并
        s4_preview_btn.click(
            fn=preview_stage4,
            inputs=[s4_input_dirs, s4_output_dir],
            outputs=s4_status,
        )
        s4_start_btn.click(
            fn=run_stage4_merge,
            inputs=[s4_input_dirs, s4_output_dir],
            outputs=[s4_progress, s4_status, s4_log],
        )
