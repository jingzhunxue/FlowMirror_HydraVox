import os, gradio as gr
import pandas as pd
from typing import List, Tuple
import json
import subprocess, time, sys
from pathlib import Path
import re
from ..i18n import t, msg, with_i18n, render

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}


@with_i18n
def upload_audio_files(files):
    """处理上传的音频文件"""
    if not files:
        gr.Warning(t("data.select_audio_files"))
        return msg("data.no_files_selected"), []
    
    file_info = []
    for file in files:
        if file:
            file_info.append({
                "文件名": os.path.basename(file.name),
                "大小": f"{os.path.getsize(file.name) / 1024:.1f} KB",
                "路径": file.name
            })
    
    df = pd.DataFrame(file_info)
    return msg("data.uploaded_files_count", count=len(files)), df

@with_i18n
def process_text_annotation(audio_files, text_content: str):
    """处理文本标注"""
    if not audio_files:
        gr.Warning(t("data.upload_audio_first"))
        return msg("data.upload_audio_first")
    
    if not text_content.strip():
        gr.Warning(t("data.enter_annotation_text"))
        return msg("data.enter_annotation_text")
    
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

@with_i18n
def validate_dataset(dataset_df):
    """验证数据集质量"""
    if dataset_df is None or len(dataset_df) == 0:
        return msg("data.dataset_empty")
    
    issues = []
    
    # 检查文本长度
    for idx, row in dataset_df.iterrows():
        text = str(row.get('文本', ''))
        if len(text) < 5:
            issues.append(msg("data.row_too_short", row=idx + 1))
        elif len(text) > 200:
            issues.append(msg("data.row_too_long", row=idx + 1))
    
    if not issues:
        return msg("data.dataset_valid")
    issue_lines = render(issues[:10])
    return msg("data.dataset_issues", count=len(issues), issues="\n".join(issue_lines))

@with_i18n
def export_dataset(dataset_df, format_type: str):
    """导出数据集"""
    if dataset_df is None or len(dataset_df) == 0:
        gr.Warning(t("data.no_export_data"))
        return None
    
    if format_type == "CSV":
        output_path = "/tmp/dataset.csv"
        dataset_df.to_csv(output_path, index=False)
    elif format_type == "JSON":
        output_path = "/tmp/dataset.json"
        dataset_df.to_json(output_path, orient='records', ensure_ascii=False, indent=2)
    else:
        gr.Warning(t("data.unsupported_format"))
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
    detail = t("CUDA 不可用，默认 CPU x1")
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            n = torch.cuda.device_count() or 1
            device = "GPU"
            num_proc = n
            detail = t("CUDA 可用，GPU 数: {count}", count=n)
    except Exception:
        pass
    return device, num_proc, detail


def _build_data_intro_md() -> str:
    return "\n".join(
        [
            t("# 🛠️ 音频数据预处理工作流"),
            "",
            t("**三个阶段的处理流程：** 格式转换 → VAD分段 → ASR转录"),
            t("**可选阶段：** 数据集合并"),
        ]
    )


def _build_data_tips_md() -> str:
    return "\n".join(
        [
            "---",
            "",
            t("## 💡 使用提示"),
            "",
            t("- **阶段顺序不可颠倒**：每个阶段都依赖前一阶段的输出"),
            t("- **GPU 加速**：阶段3支持GPU加速，可显著提升处理速度"),
            t("- **监控进度**：每个阶段都有实时进度显示和详细日志"),
            t("- **可选合并**：阶段4可将多个阶段产出的数据集进行合并"),
            "",
            t("⚠️ **注意**：处理大量文件时请确保有足够的磁盘空间和计算资源"),
        ]
    )


@with_i18n
def preview_stage1(input_dir: str, output_dir: str):
    if not input_dir or not os.path.isdir(input_dir):
        gr.Warning(t("data.enter_valid_input_dir"))
        return pd.DataFrame([]), 0, msg("data.input_dir_invalid")
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
    status = msg("data.processing_files_output", count=len(files), output_dir=output_dir)
    return df, len(files), status

@with_i18n
def run_stage1(input_dir: str, output_dir: str, sample_rate: int, overwrite: bool):
    """以子进程方式启动转换，并周期性统计进度（通过目标文件存在数）。"""
    if not input_dir or not os.path.isdir(input_dir):
        yield 0, msg("data.input_dir_invalid"), ""
        return
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_resample")
    expected = _build_expected_outputs(input_dir, output_dir)
    total = len(expected)
    if total == 0:
        yield 0, msg("data.no_media_files"), ""
        return

    script_path = _convert_script_path()
    if not script_path.exists():
        yield 0, msg("data.script_not_found", path=script_path), ""
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
        yield 0, msg("data.start_failed", error=e), ""
        return

    # 轮询进度
    last = -1
    while True:
        ret = proc.poll()
        done = _count_existing(expected)
        pct = int(done * 100 / total) if total else 0
        elapsed = int(time.time() - start_time)
        if pct != last:
            status = msg("data.in_progress", done=done, total=total, pct=pct, elapsed=elapsed)
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
        yield 100, msg("data.done", done=done, total=total, elapsed=elapsed), ""
    else:
        yield pct, msg("data.failed", done=done, total=total, elapsed=elapsed), ""


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


@with_i18n
def preview_stage2(input_dir: str, output_dir: str):
    if not input_dir or not os.path.isdir(input_dir):
        return msg("data.input_dir_invalid")
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_vad")
    # 粗略统计：输入音频文件数量
    audio_files = [p for p in Path(input_dir).rglob('*') if p.suffix.lower() in {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma'}]
    return msg("data.processing_audio_files_output", count=len(audio_files), output_dir=output_dir)


@with_i18n
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
        yield 0, msg("data.input_dir_invalid"), "", gr.update()
        return
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_vad")

    script_path = _vad_script_path()
    if not script_path.exists():
        yield 0, msg("data.script_not_found", path=script_path), "", gr.update()
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
        yield 0, msg("data.start_failed", error=e), "", gr.update()
        return

    start_time = time.time()
    log_lines: List[str] = []
    total = None
    current = 0
    last_pct = -1

    # 初始提示
    yield 0, msg("data.vad_processing"), "", gr.update()

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
            m_total = re.search(r"找到\s+(\d+)\s+个音频文件|Found\s+(\d+)\s+audio\s+files", line)
            if m_total:
                try:
                    total = int(m_total.group(1) or m_total.group(2))
                except Exception:
                    total = total

            # 解析 tqdm 进度 "处理音频文件: ... 12/34 ..."
            if "处理音频文件" in line or "Processing audio files" in line:
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
            status = msg(
                "data.in_progress_simple",
                current=current,
                total=(total if total else "?"),
                elapsed=elapsed,
            )

            if pct is not None and pct != last_pct:
                yield pct, status, "\n".join(log_lines), gr.update()
                last_pct = pct
            else:
                # 仅更新日志与状态
                yield (last_pct if last_pct >= 0 else 0), status, "\n".join(log_lines), gr.update()

        ret = proc.wait()
    except Exception as e:
        yield (
            last_pct if last_pct >= 0 else 0,
            msg("data.run_exception", error=e),
            "\n".join(log_lines),
            gr.update(),
        )
        return

    elapsed = int(time.time() - start_time)
    if proc.returncode == 0:
        final_status = msg("data.stage_done", elapsed=elapsed)
        # 若启用串联，将下阶段输入设置为本阶段输出目录
        next_input = output_dir if link_enabled else gr.update()
        yield 100, final_status, "\n".join(log_lines), next_input
    else:
        yield (
            last_pct if last_pct >= 0 else 0,
            msg("data.stage_failed", elapsed=elapsed),
            "\n".join(log_lines),
            gr.update(),
        )


@with_i18n
def preview_stage3(input_dir: str, output_dir: str):
    if not input_dir or not os.path.isdir(input_dir):
        return msg("data.input_dir_invalid")
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_asr")
    wav_files = list(Path(input_dir).rglob("*.wav"))
    mp3_files = list(Path(input_dir).rglob("*.mp3"))
    return msg(
        "data.asr_summary",
        wav_count=len(wav_files),
        mp3_count=len(mp3_files),
        output_dir=output_dir,
    )


@with_i18n
def run_stage3(input_dir: str,
               output_dir: str,
               device_choice: str,
               num_processes: float):
    """运行 ASR 转录脚本。
    输出：(progress_percent, status_text, log_text)
    """
    if not input_dir or not os.path.isdir(input_dir):
        yield 0, msg("data.input_dir_invalid"), ""
        return
    if not output_dir:
        output_dir = _generate_default_output_dir(input_dir, "_asr")

    script_path = _asr_script_path()
    if not script_path.exists():
        yield 0, msg("data.script_not_found", path=script_path), ""
        return

    # 设备与进程选择
    chosen = device_choice
    dev_detect, gpu_count, _detail = _auto_detect_device_and_processes()
    if chosen == "auto":
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
        yield 0, msg("data.start_failed", error=e), ""
        return

    start_time = time.time()
    log_lines: List[str] = []
    last_pct = -1
    total_files = None
    # 多进程解析数据（可选，若解析失败则回退到日志）
    worker_chunks = {}
    worker_pct = {}
    num_workers_detected = None

    yield 0, msg("data.asr_processing"), ""

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
            m_total = re.search(
                r"找到\s+(\d+)\s+个\s+\.wav\s+文件和\s+(\d+)\s+个\s+\.mp3\s+文件|"
                r"Found\s+(\d+)\s+\.wav\s+files\s+and\s+(\d+)\s+\.mp3\s+files",
                line,
            )
            if m_total:
                try:
                    wav_count = int(m_total.group(1) or m_total.group(3))
                    mp3_count = int(m_total.group(2) or m_total.group(4))
                    total_files = wav_count + mp3_count
                except Exception:
                    pass

            # 解析多进程总体信息
            m_multi = re.search(
                r"启动多进程处理:\s*(\d+)\s*个工作进程处理\s*(\d+)\s*个文件|"
                r"Starting multi-process:\s*(\d+)\s*workers?\s*for\s*(\d+)\s*files",
                line,
            )
            if m_multi:
                try:
                    num_workers_detected = int(m_multi.group(1) or m_multi.group(3))
                    total_files = int(m_multi.group(2) or m_multi.group(4))
                except Exception:
                    pass
            m_chunk = re.search(
                r"启动工作进程\s*(\d+)，分配\s*(\d+)\s*个文件|"
                r"Starting worker\s*(\d+),\s*assigned\s*(\d+)\s*files",
                line,
            )
            if m_chunk:
                try:
                    wid = int(m_chunk.group(1) or m_chunk.group(3))
                    size = int(m_chunk.group(2) or m_chunk.group(4))
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
            status = msg("data.asr_in_progress", elapsed=elapsed)
            yield (last_pct if last_pct >= 0 else 0), status, "\n".join(log_lines)

        ret = proc.wait()
    except Exception as e:
        yield (
            last_pct if last_pct >= 0 else 0,
            msg("data.run_exception", error=e),
            "\n".join(log_lines),
        )
        return

    elapsed = int(time.time() - start_time)
    if proc.returncode == 0:
        done_msg = msg("data.stage_done", elapsed=elapsed)
        yield 100, done_msg, "\n".join(log_lines)
    else:
        yield (
            last_pct if last_pct >= 0 else 0,
            msg("data.stage_failed", elapsed=elapsed),
            "\n".join(log_lines),
        )



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


@with_i18n
def preview_stage4(input_dirs_text: str, output_dir: str):
    paths = _parse_comma_dirs(input_dirs_text)
    if not paths:
        return msg("data.need_input_dirs_comma")
    try:
        from datasets import load_from_disk  # type: ignore
    except Exception as e:
        return msg("data.missing_datasets_dep", error=e)

    lines: List[str] = []
    total = 0
    ok = 0
    for p in paths:
        if not os.path.isdir(p):
            lines.append(msg("data.skip_non_dir_dash", path=p))
            continue
        try:
            ds = load_from_disk(p)
            n = _dataset_total_len(ds)
            lines.append(msg("data.dataset_ok", path=p, count=n))
            total += n
            ok += 1
        except Exception as e:
            lines.append(msg("data.dataset_load_failed", path=p, error=e))

    out = output_dir or render(msg("data.output_dir_missing"))
    head = [
        msg("data.merge_summary", ok=ok, total=len(paths), count=total),
        msg("data.output_dir_line", output_dir=out),
    ]
    head_lines = render(head)
    body_lines = render(lines)
    return "\n".join(head_lines + [""] + body_lines)


@with_i18n
def run_stage4_merge(input_dirs_text: str, output_dir: str):
    """合并多个 HuggingFace 数据集（支持 Dataset / DatasetDict），并保存到 output_dir。
    进度按阶段粗略估计并提供日志。
    """
    paths = _parse_comma_dirs(input_dirs_text)
    if not paths:
        yield 0, msg("data.need_input_dirs"), ""
        return
    if not output_dir:
        yield 0, msg("data.need_output_dir"), ""
        return
    try:
        from datasets import load_from_disk, concatenate_datasets  # type: ignore
    except Exception as e:
        yield 0, msg("data.missing_datasets_dep", error=e), ""
        return

    log_lines: List[str] = []
    ds_list_all = []
    ok = 0
    total_dirs = len(paths)

    # 读取阶段
    for idx, p in enumerate(paths, start=1):
        if not os.path.isdir(p):
            log_lines.append(msg("data.skip_non_dir", path=p))
            yield (
                int(idx * 10 / max(1, total_dirs)),
                msg("data.reading_progress", idx=idx, total=total_dirs),
                "\n".join(render(log_lines)),
            )
            continue
        try:
            ds_obj = load_from_disk(p)
            subs = _flatten_to_datasets(ds_obj)
            if not subs:
                log_lines.append(msg("data.no_splits", path=p))
            else:
                for ds in subs:
                    ds_list_all.append(ds)
                ln = sum(len(s) for s in subs)
                log_lines.append(msg("data.read_count", path=p, count=ln))
                ok += 1
        except Exception as e:
            log_lines.append(msg("data.load_failed", path=p, error=e))
        yield (
            int(idx * 10 / max(1, total_dirs)),
            msg("data.reading_progress", idx=idx, total=total_dirs),
            "\n".join(render(log_lines)),
        )

    if not ds_list_all:
        yield 0, msg("data.no_merge_datasets"), "\n".join(render(log_lines))
        return

    # 对齐列（取交集）
    try:
        all_cols = [set(ds.column_names) for ds in ds_list_all]
        common_cols = list(set.intersection(*all_cols)) if all_cols else []
        if not common_cols:
            yield 0, msg("data.no_common_columns"), "\n".join(render(log_lines))
            return
        columns = ", ".join(sorted(common_cols))
        log_lines.append(msg("data.columns_intersection", columns=columns))
        yield 20, msg("data.align_columns"), "\n".join(render(log_lines))
        ds_aligned = [ds.select_columns(common_cols) for ds in ds_list_all]
    except Exception as e:
        yield 0, msg("data.align_failed", error=e), "\n".join(render(log_lines))
        return

    # 合并阶段
    try:
        merged = concatenate_datasets(ds_aligned)
        log_lines.append(msg("data.merge_completed_log", count=len(merged)))
        yield 60, msg("data.merge_in_progress"), "\n".join(render(log_lines))
    except Exception as e:
        yield 0, msg("data.merge_failed", error=e), "\n".join(render(log_lines))
        return

    # 保存阶段
    try:
        from pathlib import Path as _Path
        _Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        merged.save_to_disk(output_dir)
        log_lines.append(msg("data.saved_to", output_dir=output_dir))
        yield 100, msg("data.merge_done", count=len(merged)), "\n".join(render(log_lines))
    except Exception as e:
        yield 90, msg("data.save_failed", error=e), "\n".join(render(log_lines))
        return


def create_data_tab():
    """创建数据处理tab界面"""
    with gr.Tab(t("📊 数据处理")):
        intro_md = gr.Markdown(_build_data_intro_md())
        
        device_default, proc_default, device_detail = _auto_detect_device_and_processes()

        # 阶段1：格式转换与重采样
        with gr.Accordion(t("🎵 阶段 1 - 格式转换与重采样"), open=False) as stage1_acc:
            stage1_desc = gr.Markdown(t("**功能：** 将各种音频/视频格式统一转换为 16kHz WAV 格式"))
            
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        s1_input_dir = gr.Textbox(
                            label=t("📁 输入目录"),
                            placeholder=t("/path/to/input_dir"),
                            scale=3,
                        )
                        s1_auto_sync = gr.Checkbox(
                            value=True,
                            label=t("🔄 自动同步输出路径"),
                            info=t("添加_resample后缀"),
                            scale=1,
                        )
                        s1_output_dir = gr.Textbox(
                            label=t("📂 输出目录"),
                            placeholder=t("自动同步或手动填写"),
                            scale=3,
                        )
                        
                    with gr.Row():
                        s1_sr = gr.Dropdown(
                            choices=[8000,16000,22050,44100,48000], 
                            value=16000, 
                            label=t("🎤 采样率 (Hz)"),
                            scale=1
                        )
                        s1_overwrite = gr.Checkbox(value=False, label=t("⚠️ 覆盖已存在文件"), scale=1)
                        
                    with gr.Row():
                        s1_preview_btn = gr.Button(t("👀 预览变更"), variant="secondary", scale=1)
                        s1_start_btn = gr.Button(t("▶️ 开始处理"), variant="primary", scale=1)
                        
                    s1_preview_df = gr.Dataframe(
                        headers=[t("源文件"), t("目标文件")],
                        label=t("📋 映射预览（前50条）"),
                        interactive=False
                    )
                    
                    with gr.Row():
                        s1_total_num = gr.Number(label=t("📊 待处理文件数"), interactive=False, scale=1)
                        s1_progress = gr.Slider(
                            0, 100, value=0, step=1, label=t("📈 进度 (%)"), interactive=False, scale=2
                        )
                        
                    s1_status = gr.Textbox(label=t("📋 状态"), interactive=False)
                    s1_log = gr.Textbox(label=t("📝 运行日志"), lines=4, interactive=False, show_copy_button=True)

        # 阶段2：VAD 处理（Silero）
        with gr.Accordion(t("🔊 阶段 2 - VAD 语音活动检测"), open=False) as stage2_acc:
            stage2_desc = gr.Markdown(t("**功能：** 使用 Silero VAD 检测并分割语音片段，去除静音部分"))
            
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        s2_input_dir = gr.Textbox(
                            label=t("📁 输入目录"),
                            placeholder=t("默认衔接阶段1输出"),
                            scale=3,
                        )
                        s2_auto_sync = gr.Checkbox(
                            value=True,
                            label=t("🔄 自动同步输出路径"),
                            info=t("添加_vad后缀"),
                            scale=1,
                        )
                        s2_output_dir = gr.Textbox(
                            label=t("📂 输出目录"),
                            placeholder=t("自动同步或手动填写"),
                            scale=3,
                        )
                        
                    with gr.Accordion(t("⚙️ VAD 参数设置"), open=False) as s2_params_acc:
                        with gr.Row():
                            s2_threshold = gr.Slider(
                                0.0,
                                1.0,
                                value=0.5,
                                step=0.01,
                                label=t("🎯 置信度阈值"),
                                info=t("越高越严格"),
                            )
                            s2_min_speech_ms = gr.Number(value=250, label=t("🗣️ 最短语音 (ms)"))
                            s2_min_silence_ms = gr.Number(value=200, label=t("🔇 最短静音 (ms)"))
                            s2_pad_ms = gr.Number(value=30, label=t("🔧 前后填充 (ms)"))
                        with gr.Row():
                            s2_min_seg = gr.Number(value=0.5, label=t("⏱️ 最短片段 (s)"))
                            s2_max_seg = gr.Number(value=30, label=t("⏰ 最长片段 (s)"))
                            
                    with gr.Row():
                        s2_preview_btn = gr.Button(t("👀 预览"), variant="secondary", scale=1)
                        s2_start_btn = gr.Button(t("▶️ 开始处理"), variant="primary", scale=1)
                        
                    with gr.Row():
                        s2_progress = gr.Slider(
                            0, 100, value=0, step=1, label=t("📈 进度 (%)"), interactive=False
                        )
                        
                    s2_status = gr.Textbox(label=t("📋 状态"), interactive=False)
                    s2_log = gr.Textbox(label=t("📝 运行日志"), lines=4, interactive=False, show_copy_button=True)

        # 阶段3：ASR 处理
        with gr.Accordion(t("🎙️ 阶段 3 - ASR 语音识别转录"), open=False) as stage3_acc:
            stage3_desc = gr.Markdown(t("**功能：** 使用语音识别技术将音频转换为文本，生成训练数据集"))
            
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        s3_input_dir = gr.Textbox(
                            label=t("📁 输入目录"),
                            placeholder=t("默认衔接阶段2输出"),
                            scale=3,
                        )
                        s3_auto_sync = gr.Checkbox(
                            value=True,
                            label=t("🔄 自动同步输出路径"),
                            info=t("添加_asr后缀"),
                            scale=1,
                        )
                        s3_output_dir = gr.Textbox(
                            label=t("📂 输出目录"),
                            placeholder=t("自动同步或手动填写"),
                            scale=3,
                        )
                        
                    with gr.Accordion(t("⚙️ 计算资源设置"), open=False) as s3_compute_acc:
                        with gr.Row():
                            s3_device = gr.Dropdown(
                                choices=[(t("自动"), "auto"), ("CPU", "CPU"), ("GPU", "GPU")],
                                value=("GPU" if device_default == "GPU" else "CPU"),
                                label=t("💻 计算设备"),
                            )
                            s3_processes = gr.Number(value=proc_default, label=t("🔄 并行进程数"))
                            s3_detect_btn = gr.Button(t("🔄 刷新设备检测"), variant="secondary", size="sm")
                        s3_device_info = gr.Textbox(
                            value=device_detail, label=t("ℹ️ 设备检测信息"), interactive=False
                        )
                        
                    with gr.Row():
                        s3_preview_btn = gr.Button(t("👀 预览"), variant="secondary", scale=1)
                        s3_start_btn = gr.Button(t("▶️ 开始处理"), variant="primary", scale=1)
                        
                    with gr.Row():
                        s3_progress = gr.Slider(
                            0, 100, value=0, step=1, label=t("📈 进度 (%)"), interactive=False
                        )
                        
                    s3_status = gr.Textbox(label=t("📋 状态"), interactive=False)
                    s3_log = gr.Textbox(label=t("📝 运行日志"), lines=4, interactive=False, show_copy_button=True)

        # 阶段4：数据集合并（可选）
        with gr.Accordion(t("🧩 阶段 4 - 数据集合并 (可选)"), open=False) as stage4_acc:
            stage4_desc = gr.Markdown(
                t("**功能：** 将多个前面阶段生成的数据集目录合并为一个新的 HuggingFace 数据集。输入多个目录时使用英文逗号分隔。")
            )
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        s4_input_dirs = gr.Textbox(
                            label=t("📁 输入数据集目录（逗号分隔）"),
                            placeholder=t("/path/to/ds1,/path/to/ds2,..."),
                            scale=3,
                        )
                        s4_output_dir = gr.Textbox(
                            label=t("📂 合并输出目录"),
                            placeholder=t("/path/to/merged_dataset"),
                            scale=3,
                        )
                    with gr.Row():
                        s4_preview_btn = gr.Button(t("👀 预览"), variant="secondary", scale=1)
                        s4_start_btn = gr.Button(t("▶️ 开始合并"), variant="primary", scale=1)
                    with gr.Row():
                        s4_progress = gr.Slider(
                            0, 100, value=0, step=1, label=t("📈 进度 (%)"), interactive=False
                        )
                    s4_status = gr.Textbox(label=t("📋 状态"), interactive=False)
                    s4_log = gr.Textbox(label=t("📝 合并日志"), lines=6, interactive=False, show_copy_button=True)

        tips_md = gr.Markdown(_build_data_tips_md())

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

        def _apply_language(device_value: str):
            _device_default, _proc_default, device_detail = _auto_detect_device_and_processes()
            return [
                gr.update(value=_build_data_intro_md()),
                gr.update(label=t("🎵 阶段 1 - 格式转换与重采样")),
                gr.update(value=t("**功能：** 将各种音频/视频格式统一转换为 16kHz WAV 格式")),
                gr.update(label=t("📁 输入目录"), placeholder=t("/path/to/input_dir")),
                gr.update(label=t("🔄 自动同步输出路径"), info=t("添加_resample后缀")),
                gr.update(label=t("📂 输出目录"), placeholder=t("自动同步或手动填写")),
                gr.update(label=t("🎤 采样率 (Hz)")),
                gr.update(label=t("⚠️ 覆盖已存在文件")),
                gr.update(value=t("👀 预览变更")),
                gr.update(value=t("▶️ 开始处理")),
                gr.update(headers=[t("源文件"), t("目标文件")], label=t("📋 映射预览（前50条）")),
                gr.update(label=t("📊 待处理文件数")),
                gr.update(label=t("📈 进度 (%)")),
                gr.update(label=t("📋 状态")),
                gr.update(label=t("📝 运行日志")),
                gr.update(label=t("🔊 阶段 2 - VAD 语音活动检测")),
                gr.update(value=t("**功能：** 使用 Silero VAD 检测并分割语音片段，去除静音部分")),
                gr.update(label=t("📁 输入目录"), placeholder=t("默认衔接阶段1输出")),
                gr.update(label=t("🔄 自动同步输出路径"), info=t("添加_vad后缀")),
                gr.update(label=t("📂 输出目录"), placeholder=t("自动同步或手动填写")),
                gr.update(label=t("⚙️ VAD 参数设置")),
                gr.update(label=t("🎯 置信度阈值"), info=t("越高越严格")),
                gr.update(label=t("🗣️ 最短语音 (ms)")),
                gr.update(label=t("🔇 最短静音 (ms)")),
                gr.update(label=t("🔧 前后填充 (ms)")),
                gr.update(label=t("⏱️ 最短片段 (s)")),
                gr.update(label=t("⏰ 最长片段 (s)")),
                gr.update(value=t("👀 预览")),
                gr.update(value=t("▶️ 开始处理")),
                gr.update(label=t("📈 进度 (%)")),
                gr.update(label=t("📋 状态")),
                gr.update(label=t("📝 运行日志")),
                gr.update(label=t("🎙️ 阶段 3 - ASR 语音识别转录")),
                gr.update(value=t("**功能：** 使用语音识别技术将音频转换为文本，生成训练数据集")),
                gr.update(label=t("📁 输入目录"), placeholder=t("默认衔接阶段2输出")),
                gr.update(label=t("🔄 自动同步输出路径"), info=t("添加_asr后缀")),
                gr.update(label=t("📂 输出目录"), placeholder=t("自动同步或手动填写")),
                gr.update(label=t("⚙️ 计算资源设置")),
                gr.update(
                    choices=[(t("自动"), "auto"), ("CPU", "CPU"), ("GPU", "GPU")],
                    value=device_value,
                    label=t("💻 计算设备"),
                ),
                gr.update(label=t("🔄 并行进程数")),
                gr.update(value=t("🔄 刷新设备检测")),
                gr.update(value=device_detail, label=t("ℹ️ 设备检测信息")),
                gr.update(value=t("👀 预览")),
                gr.update(value=t("▶️ 开始处理")),
                gr.update(label=t("📈 进度 (%)")),
                gr.update(label=t("📋 状态")),
                gr.update(label=t("📝 运行日志")),
                gr.update(label=t("🧩 阶段 4 - 数据集合并 (可选)")),
                gr.update(
                    value=t("**功能：** 将多个前面阶段生成的数据集目录合并为一个新的 HuggingFace 数据集。输入多个目录时使用英文逗号分隔。")
                ),
                gr.update(label=t("📁 输入数据集目录（逗号分隔）"), placeholder=t("/path/to/ds1,/path/to/ds2,...")),
                gr.update(label=t("📂 合并输出目录"), placeholder=t("/path/to/merged_dataset")),
                gr.update(value=t("👀 预览")),
                gr.update(value=t("▶️ 开始合并")),
                gr.update(label=t("📈 进度 (%)")),
                gr.update(label=t("📋 状态")),
                gr.update(label=t("📝 合并日志")),
                gr.update(value=_build_data_tips_md()),
            ]

        return {
            "outputs": [
                intro_md,
                stage1_acc,
                stage1_desc,
                s1_input_dir,
                s1_auto_sync,
                s1_output_dir,
                s1_sr,
                s1_overwrite,
                s1_preview_btn,
                s1_start_btn,
                s1_preview_df,
                s1_total_num,
                s1_progress,
                s1_status,
                s1_log,
                stage2_acc,
                stage2_desc,
                s2_input_dir,
                s2_auto_sync,
                s2_output_dir,
                s2_params_acc,
                s2_threshold,
                s2_min_speech_ms,
                s2_min_silence_ms,
                s2_pad_ms,
                s2_min_seg,
                s2_max_seg,
                s2_preview_btn,
                s2_start_btn,
                s2_progress,
                s2_status,
                s2_log,
                stage3_acc,
                stage3_desc,
                s3_input_dir,
                s3_auto_sync,
                s3_output_dir,
                s3_compute_acc,
                s3_device,
                s3_processes,
                s3_detect_btn,
                s3_device_info,
                s3_preview_btn,
                s3_start_btn,
                s3_progress,
                s3_status,
                s3_log,
                stage4_acc,
                stage4_desc,
                s4_input_dirs,
                s4_output_dir,
                s4_preview_btn,
                s4_start_btn,
                s4_progress,
                s4_status,
                s4_log,
                tips_md,
            ],
            "apply": _apply_language,
            "inputs": [s3_device],
        }
