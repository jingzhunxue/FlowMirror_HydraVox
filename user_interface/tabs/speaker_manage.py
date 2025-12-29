import os
import random
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

import gradio as gr
import torch
import torchaudio
import pandas as pd
import numpy as np

from datasets import Audio


DEFAULT_REL_SPK2INFO = "jzx-ai-lab/HydraVox-CV3/spk2info.pt"
DEFAULT_SV_MODEL_PATH = "jzx-ai-lab/speech_campplus_sv_zh-cn_16k-common"

_SV_PIPE = None
_SV_PIPE_KEY = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_spk2info_path() -> Path:
    return _project_root() / DEFAULT_REL_SPK2INFO


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_spk2info(path_str: str | None = None) -> Dict[str, Dict[str, torch.Tensor]]:
    path = Path(path_str) if path_str else _default_spk2info_path()
    if not path.exists():
        return {}
    try:
        data = torch.load(path, map_location="cpu")
        # 规范化：确保为 {name: {"embedding": tensor(192,)}}
        normed: Dict[str, Dict[str, torch.Tensor]] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "embedding" in v:
                    emb = v["embedding"]
                elif torch.is_tensor(v):
                    emb = v
                else:
                    continue
                emb = emb.detach().to(dtype=torch.float32, device="cpu").view(-1)
                normed[str(k)] = {"embedding": emb}
        return normed
    except Exception as e:
        gr.Warning(f"加载失败: {e}")
        return {}


def _save_spk2info(spk2info: Dict[str, Dict[str, torch.Tensor]], path_str: str | None = None) -> Tuple[bool, str]:
    path = Path(path_str) if path_str else _default_spk2info_path()
    try:
        _ensure_parent_dir(path)
        # 保存为 {name: {"embedding": tensor}}
        safe_dump: Dict[str, Dict[str, torch.Tensor]] = {}
        for k, v in spk2info.items():
            emb = v.get("embedding")
            if not torch.is_tensor(emb):
                continue
            safe_dump[str(k)] = {"embedding": emb.detach().to("cpu").to(dtype=torch.float32)}
        torch.save(safe_dump, path)
        return True, f"已保存至 {path}"
    except Exception as e:
        return False, f"保存失败: {e}"


def _spk2info_to_df(spk2info: Dict[str, Dict[str, torch.Tensor]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, info in spk2info.items():
        emb = info.get("embedding")
        if torch.is_tensor(emb):
            emb_cpu = emb.detach().to("cpu")
            dim = int(emb_cpu.numel())
            norm = float(emb_cpu.norm().item()) if dim > 0 else 0.0
            preview = emb_cpu.flatten()[:8].tolist()
            rows.append({
                "speaker_name": name,
                "dim": dim,
                "l2_norm": round(norm, 6),
                "preview[:8]": preview,
            })
    if not rows:
        return pd.DataFrame(columns=["speaker_name", "dim", "l2_norm", "preview[:8]"])
    return pd.DataFrame(rows).sort_values("speaker_name").reset_index(drop=True)


def _get_speaker_verification_pipe(model_path: str, device: str):
    global _SV_PIPE, _SV_PIPE_KEY
    model_path = _ensure_speaker_verification_model(model_path)
    key = f"{model_path}|dev={device}"
    if _SV_PIPE is not None and _SV_PIPE_KEY == key:
        return _SV_PIPE
    from modelscope.pipelines import pipeline  # type: ignore

    _SV_PIPE = pipeline(task="speaker-verification", model=model_path, model_revision="v1.0.0", device=device)
    _SV_PIPE_KEY = key
    return _SV_PIPE


def _ensure_speaker_verification_model(model_path: str) -> str:
    path = Path(model_path).expanduser()
    try:
        if path.exists():
            if path.is_file() or any(path.iterdir()):
                return str(path)
    except (OSError, PermissionError) as e:
        gr.Warning(f"检查说话人模型失败: {e}")

    try:
        from modelscope import snapshot_download  # type: ignore

        cache_dir = path.parent / "modelscope_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = snapshot_download(
                model_id=model_path,
                revision="v1.0.0",
                cache_dir=str(cache_dir),
            )
        except Exception:
            fallback_id = "iic/speech_campplus_sv_zh-cn_16k-common"
            if model_path != fallback_id:
                gr.Warning(f"下载 {model_path} 失败，回退到 {fallback_id}")
                downloaded = snapshot_download(
                    model_id=fallback_id,
                    revision="v1.0.0",
                    cache_dir=str(cache_dir),
                )
            else:
                raise

        downloaded_path = Path(downloaded).resolve()
        if downloaded_path != path.resolve():
            if path.exists():
                shutil.rmtree(path)
            try:
                path.symlink_to(downloaded_path)
            except (OSError, NotImplementedError):
                shutil.copytree(downloaded_path, path)

        if path.exists():
            return str(path)
        return str(downloaded_path)
    except Exception as e:
        gr.Warning(f"下载说话人模型失败: {e}，回退到在线模式")
        return model_path


def _load_audio_mono(audio_info: Any, target_sr: int) -> torch.Tensor:
    if isinstance(audio_info, dict) and "array" in audio_info:
        wav = torch.tensor(audio_info["array"], dtype=torch.float32)
        sr = int(audio_info.get("sampling_rate", target_sr))
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        return wav

    if isinstance(audio_info, dict) and "path" in audio_info:
        path = audio_info["path"]
    else:
        path = str(audio_info)
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.to(torch.float32)


def _extract_embedding_from_audio(audio_info: Any, sv_pipe) -> np.ndarray:
    wav_16k = _load_audio_mono(audio_info, target_sr=16000)
    wav_np = wav_16k.squeeze(0).cpu().numpy().astype(np.float32)
    out = sv_pipe([wav_np], output_emb=True)
    emb = np.array(out["embs"][0], dtype=np.float32)
    return emb


def _compute_mean_embedding_from_dataset(ds_path: str) -> Tuple[str, torch.Tensor | None]:
    if not ds_path:
        return "请输入数据集路径", None
    try:
        from datasets import load_from_disk  # type: ignore
    except Exception as e:
        return f"缺少datasets依赖或导入失败: {e}", None

    try:
        ds = load_from_disk(ds_path)
    except Exception as e:
        return f"加载数据集失败: {e}", None

    use_audio_fallback = "embedding" not in ds.column_names
    if use_audio_fallback and "audio" not in ds.column_names:
        return "数据集中未找到 'embedding' 或 'audio' 列", None

    sv_pipe = None
    if use_audio_fallback:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        sv_pipe = _get_speaker_verification_pipe(DEFAULT_SV_MODEL_PATH, device=device)
        try:
            if len(ds) > 5000:
                ds = ds.shuffle(seed=42).select(range(5000))
        except Exception:
            pass
    ds = ds.cast_column("audio", Audio(decode=True, sampling_rate=16000))
    total = None
    count = 0
    try:
        for row in ds:
            if use_audio_fallback:
                emb = _extract_embedding_from_audio(row["audio"], sv_pipe)
            else:
                emb = row["embedding"]
                if emb is None:
                    continue
            t = torch.tensor(emb, dtype=torch.float32)
            if t.dim() == 0:
                continue
            if total is None:
                total = t.clone()
            else:
                if total.numel() != t.numel():
                    return f"embedding 维度不一致: {total.numel()} vs {t.numel()}", None
                total += t
            count += 1
        if total is None or count == 0:
            if use_audio_fallback:
                return "未从音频提取到有效的 embedding", None
            return "未获取到有效的 embedding", None
        mean = (total / float(count)).view(-1)
        info = f"样本数: {count}, 维度: {mean.numel()}, L2范数: {mean.norm().item():.6f}"
        if use_audio_fallback:
            try:
                if len(ds) == 5000:
                    info = f"{info}（已随机抽取 5000 条音频）"
            except Exception:
                pass
        return info, mean
    except Exception as e:
        return f"计算均值失败: {e}", None


def create_speaker_manage_tab():
    with gr.Tab("🗣️ 说话人管理"):
        gr.Markdown("""
        # 🗣️ 说话人库管理

        - 预加载/保存路径：`jzx-ai-lab/HydraVox-CV3/spk2info.pt`
        - 查看已有 speaker，加载数据集计算 `embedding` 均值，新增/覆盖 speaker
        """)

        # States
        spk2info_state = gr.State(_load_spk2info())
        last_mean_state = gr.State(value=None)

        with gr.Group():
            with gr.Row():
                spk_path_tb = gr.Textbox(
                    label="spk2info.pt 路径",
                    value=str(_default_spk2info_path()),
                    interactive=True,
                )
                reload_btn = gr.Button("🔄 重新加载", variant="secondary")
                save_btn = gr.Button("💾 保存当前", variant="secondary")

            spk_table = gr.Dataframe(
                headers=["speaker_name", "dim", "l2_norm", "preview[:8]"],
                label="现有说话人",
                interactive=False,
            )

        with gr.Accordion("➕ 从数据集新增/覆盖说话人", open=True):
            with gr.Row():
                ds_path_tb = gr.Textbox(label="数据集路径 (HuggingFace load_from_disk)", placeholder="/path/to/dataset")
                calc_btn = gr.Button("📐 计算均值", variant="primary")
            mean_info_tb = gr.Textbox(label="均值信息", interactive=False)
            with gr.Row():
                speaker_name_tb = gr.Textbox(label="Speaker 名称", placeholder="如：alice")
                add_btn = gr.Button("✅ 新增/覆盖", variant="primary")

        # Handlers
        def _on_reload(path_str: str):
            data = _load_spk2info(path_str)
            return data, _spk2info_to_df(data)

        def _on_save(state_data: Dict[str, Dict[str, torch.Tensor]], path_str: str):
            ok, msg = _save_spk2info(state_data, path_str)
            if not ok:
                gr.Warning(msg)
            return msg

        def _on_calc(ds_path: str):
            info, mean = _compute_mean_embedding_from_dataset(ds_path)
            return mean, info

        def _on_add(state_data: Dict[str, Dict[str, torch.Tensor]], mean: Any, name: str, save_path: str):
            if not name or not name.strip():
                return state_data, _spk2info_to_df(state_data), "请输入有效的 speaker 名称"
            if mean is None or not torch.is_tensor(mean):
                return state_data, _spk2info_to_df(state_data), "请先计算均值"
            new_data = dict(state_data)
            new_data[name.strip()] = {"embedding": mean.detach().to(dtype=torch.float32)}
            ok, msg = _save_spk2info(new_data, save_path)
            if not ok:
                gr.Warning(msg)
            return new_data, _spk2info_to_df(new_data), msg

        # Initial table load
        spk_table.value = _spk2info_to_df(spk2info_state.value)

        reload_btn.click(
            fn=_on_reload,
            inputs=[spk_path_tb],
            outputs=[spk2info_state, spk_table],
        )
        save_btn.click(
            fn=_on_save,
            inputs=[spk2info_state, spk_path_tb],
            outputs=[mean_info_tb],
        )
        calc_btn.click(
            fn=_on_calc,
            inputs=[ds_path_tb],
            outputs=[last_mean_state, mean_info_tb],
        )
        add_btn.click(
            fn=_on_add,
            inputs=[spk2info_state, last_mean_state, speaker_name_tb, spk_path_tb],
            outputs=[spk2info_state, spk_table, mean_info_tb],
        )
