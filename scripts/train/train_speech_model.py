# -*- coding: utf-8 -*-
"""
HydraVox 训练脚本（新模型：LLM / FLOW 预训练）
基于 HuggingFace Trainer，整合 llm/flow 训练流程。
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
import torchaudio
import whisper
from datasets import Audio, Dataset, concatenate_datasets, load_from_disk
from hyperpyyaml import load_hyperpyyaml
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.absolute()
third_party_dir = project_root / "server/model_utils"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(third_party_dir))

from server.model_utils.cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
from matcha.utils.audio import mel_spectrogram
from modelscope.pipelines import pipeline
from fmtn import create_default_tn
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

try:
    tn = create_default_tn(verbose=True)
except Exception:
    print("Failed to load text normalization library")
    tn = None

USEFUL_COLUMNS_LLM = ["text", "text_token", "audio"]
USEFUL_COLUMNS_FLOW = ["audio", "speech_token", "embedding"]


def _load_state_dict_maybe_container(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("不支持的 checkpoint 格式：期望为 state_dict 或 {'state_dict': ...}")


def _maybe_get_default_config(model_type: str) -> str:
    return "jzx-ai-lab/HydraVox-CV3/hydravox.yaml"


def _resolve_onnx_device_id(value: int | None) -> int:
    if value is not None:
        return int(value)
    env_rank = os.getenv("LOCAL_RANK", os.getenv("RANK", "0"))
    return int(env_rank) if str(env_rank).isdigit() else 0


def prepare_dataset_llm_pretrain(ds: Dataset) -> Dataset:
    """保留 LLM 所需列，减少 IO/内存。"""
    keep = ["audio"]
    if "text_token" in ds.column_names:
        keep.append("text_token")
    if "text" in ds.column_names:
        keep.append("text")
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds = ds.cast_column("audio", Audio(decode=True, sampling_rate=16000))
    return ds


def prepare_dataset_flow_pretrain(ds: Dataset) -> Dataset:
    """保留 flow 训练所需列，减少 IO/内存。"""
    ds = ds.remove_columns([c for c in ds.column_names if c not in USEFUL_COLUMNS_FLOW])
    ds = ds.cast_column("audio", Audio(decode=True, sampling_rate=24000))
    return ds


def _build_train_eval_dataset(
    train_dss: List[Dataset],
    val_dss: List[Dataset],
    auto_val_split: bool,
    val_split_ratio: float,
) -> tuple[Dataset, Dataset | None]:
    if auto_val_split or not val_dss:
        full_dataset = concatenate_datasets(train_dss).shuffle(seed=42)
        if val_split_ratio <= 0:
            logging.info("自动划分验证集关闭（val_split_ratio <= 0）：仅训练不验证")
            return full_dataset, None
        val_size = int(len(full_dataset) * val_split_ratio)
        if val_size <= 0:
            logging.info("验证集大小为 0：仅训练不验证")
            return full_dataset, None
        if val_size >= len(full_dataset):
            raise ValueError(
                f"val_split_ratio 过大导致验证集大小({val_size}) >= 数据集总量({len(full_dataset)})"
            )
        train_size = len(full_dataset) - val_size
        train_dataset = full_dataset.select(range(train_size))
        eval_dataset = full_dataset.select(range(train_size, train_size + val_size))
        logging.info("自动划分验证集: 训练集 %s，验证集 %s", train_size, val_size)
        return train_dataset, eval_dataset

    train_dataset = concatenate_datasets(train_dss).shuffle(seed=42)
    eval_dataset = concatenate_datasets(val_dss).shuffle(seed=42)
    return train_dataset, eval_dataset


def _get_added_special_tokens(tokenizer: Any) -> set[str]:
    st = getattr(tokenizer, "special_tokens", None)
    if isinstance(st, dict) and isinstance(st.get("additional_special_tokens"), list):
        return set(st["additional_special_tokens"])
    tok = getattr(tokenizer, "tokenizer", None)
    if tok is not None and hasattr(tok, "get_added_vocab"):
        try:
            return set(tok.get_added_vocab().keys())
        except Exception:
            return set()
    return set()


_RE_EN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_RE_ZH_CHAR = re.compile(r"[\u4e00-\u9fff]")


def _maybe_append_en_cmu_tokens(text: str, special_tokens: set[str]) -> str:
    matches = list(_RE_EN_WORD.finditer(text))
    if not matches:
        return text
    picks = random.sample(matches, k=min(2, len(matches)))

    def phones_for_word(w: str) -> List[str]:
        w = w.lower()
        try:
            import cmudict  # type: ignore

            d = cmudict.dict()
            prons = d.get(w)
            if prons:
                return list(prons[0])
        except Exception:
            pass
        try:
            import pronouncing  # type: ignore

            ps = pronouncing.phones_for_word(w)
            if ps:
                return ps[0].strip().split()
        except Exception:
            pass
        return []

    replacements: List[tuple[int, int, str]] = []
    for m in picks:
        word = m.group(0)
        phones = phones_for_word(word)
        if not phones:
            continue
        toks: List[str] = []
        for p in phones:
            tok = f"[{p}]"
            if tok in special_tokens:
                toks.append(tok)
        if not toks:
            continue
        rep = " " + "".join(toks) + " "
        replacements.append((m.start(), m.end(), rep))

    if not replacements:
        return text
    replacements.sort(key=lambda x: x[0], reverse=True)
    for s, e, rep in replacements:
        text = text[:s] + rep + text[e:]
    return text


def _maybe_append_zh_pinyin_tokens(text: str, special_tokens: set[str]) -> str:
    matches = list(_RE_ZH_CHAR.finditer(text))
    if len(matches) < 2:
        return text
    picks = random.sample(matches, k=2)
    try:
        from pypinyin import Style, pinyin  # type: ignore
    except Exception:
        return text

    replacements: List[tuple[int, int, str]] = []
    for m in picks:
        ch = m.group(0)
        try:
            ini = pinyin(ch, style=Style.INITIALS, strict=False, heteronym=False)[0][0] or ""
            fin = pinyin(ch, style=Style.FINALS_TONE, strict=False, heteronym=False)[0][0] or ""
        except Exception:
            continue
        toks: List[str] = []
        if ini:
            t_ini = f"[{ini.lower()}]"
            if t_ini in special_tokens:
                toks.append(t_ini)
        if fin:
            t_fin = f"[{fin.lower()}]"
            if t_fin in special_tokens:
                toks.append(t_fin)
        if toks:
            rep = " " + "".join(toks) + " "
            replacements.append((m.start(), m.end(), rep))

    if not replacements:
        return text
    replacements.sort(key=lambda x: x[0], reverse=True)
    for s, e, rep in replacements:
        text = text[:s] + rep + text[e:]
    return text


_TOKENIZER_SESSION: ort.InferenceSession | None = None
_TOKENIZER_SESSION_KEY: str | None = None
_SV_PIPE: Any | None = None
_SV_PIPE_KEY: str | None = None

# speech_token 抽取兜底池：跨 batch 缓存成功样本，用于“全 batch 失败”时随机回退
_SPEECH_TOKEN_POOL: List[torch.Tensor] = []
_SPEECH_TOKEN_POOL_MAX: int = int(os.getenv("SPEECH_TOKEN_POOL_MAX", "256"))


def _get_onnx_tokenizer_session(
    onnx_path: str,
    use_cuda: bool,
    device_id: int,
    intra_op_num_threads: int = 1,
) -> ort.InferenceSession:
    global _TOKENIZER_SESSION, _TOKENIZER_SESSION_KEY
    available = set(ort.get_available_providers())
    effective_use_cuda = bool(use_cuda) and ("CUDAExecutionProvider" in available)
    if bool(use_cuda) and not effective_use_cuda:
        logging.warning(
            "onnxruntime 未检测到 CUDAExecutionProvider（available=%s），将自动使用 CPUExecutionProvider。",
            ",".join(sorted(available)),
        )

    key = f"{onnx_path}|cuda={effective_use_cuda}|dev={device_id}|intra={intra_op_num_threads}"
    if _TOKENIZER_SESSION is not None and _TOKENIZER_SESSION_KEY == key:
        return _TOKENIZER_SESSION

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = int(intra_op_num_threads)

    providers = ["CPUExecutionProvider"]
    _TOKENIZER_SESSION = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    _TOKENIZER_SESSION_KEY = key
    return _TOKENIZER_SESSION


def _get_speaker_verification_pipe(model_path: str, device: str) -> Any:
    global _SV_PIPE, _SV_PIPE_KEY
    key = f"{model_path}|dev={device}"
    if _SV_PIPE is not None and _SV_PIPE_KEY == key:
        return _SV_PIPE
    _SV_PIPE = pipeline(task="speaker-verification", model=model_path, model_revision="v1.0.0", device=device)
    _SV_PIPE_KEY = key
    return _SV_PIPE


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


def _extract_speech_token_from_audio(audio_info: Any, onnx_session: ort.InferenceSession) -> List[int]:
    wav_16k = _load_audio_mono(audio_info, target_sr=16000)
    mel = whisper.log_mel_spectrogram(wav_16k, n_mels=128)
    ort_inputs = {
        onnx_session.get_inputs()[0].name: mel.cpu().numpy(),
        onnx_session.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32),
    }
    tokens = onnx_session.run(None, ort_inputs)[0].flatten().tolist()
    return tokens


def _audio_ref(audio_info: Any) -> str:
    """用于日志打印的音频引用，避免打印过大对象。"""
    try:
        if isinstance(audio_info, dict):
            if "path" in audio_info:
                return f"path={audio_info.get('path')}"
            if "array" in audio_info:
                arr = audio_info.get("array")
                sr = audio_info.get("sampling_rate", None)
                n = len(arr) if hasattr(arr, "__len__") else "?"
                return f"array(len={n},sr={sr})"
        return str(audio_info)
    except Exception:
        return "<audio>"


def _extract_speech_tokens_with_batch_fallback(
    features: List[Dict[str, Any]],
    onnx_session: ort.InferenceSession,
) -> tuple[List[torch.Tensor], List[int]]:
    """
    从 batch 中逐条提取 speech_token。
    若某条音频提取失败（例如 ONNX Gather indices 越界），则回退为同 batch 内其它成功样本的 token，确保训练不中断。
    """
    global _SPEECH_TOKEN_POOL, _SPEECH_TOKEN_POOL_MAX
    speech_tokens: List[torch.Tensor | None] = [None] * len(features)
    speech_token_lens: List[int] = [0] * len(features)

    first_ok: torch.Tensor | None = None
    last_ok: torch.Tensor | None = None
    first_err: Exception | None = None

    for i, f in enumerate(features):
        try:
            tokens = _extract_speech_token_from_audio(f["audio"], onnx_session)
            st = torch.tensor(tokens, dtype=torch.long)
            speech_tokens[i] = st
            speech_token_lens[i] = int(st.numel())
            if first_ok is None:
                first_ok = st
            last_ok = st

            # 加入跨 batch 的成功池，供“全 batch 失败”时随机兜底
            try:
                if _SPEECH_TOKEN_POOL_MAX > 0:
                    _SPEECH_TOKEN_POOL.append(st.detach().cpu())
                    if len(_SPEECH_TOKEN_POOL) > _SPEECH_TOKEN_POOL_MAX:
                        # 简单 FIFO
                        _SPEECH_TOKEN_POOL = _SPEECH_TOKEN_POOL[-_SPEECH_TOKEN_POOL_MAX :]
            except Exception:
                pass
        except Exception as e:
            if first_err is None:
                first_err = e
            logging.warning(
                "speech_token 提取失败，将使用 batch 内其它样本回退替代（idx=%s, audio=%s, err=%s: %s）",
                i,
                _audio_ref(f.get("audio")),
                e.__class__.__name__,
                str(e)[:500],
            )
            speech_tokens[i] = None

    if first_ok is None:
        # 兜底 1：从历史成功池（跨 batch）里随机抽一个 token
        if _SPEECH_TOKEN_POOL:
            pick = random.choice(_SPEECH_TOKEN_POOL)
            pick_len = int(pick.numel())
            logging.error(
                "本 batch 所有音频 speech_token 提取均失败，已从历史成功池随机抽取兜底继续训练（pool=%s, pick_len=%s, err=%s: %s）",
                len(_SPEECH_TOKEN_POOL),
                pick_len,
                first_err.__class__.__name__ if first_err is not None else "UnknownError",
                str(first_err)[:500] if first_err is not None else "",
            )
            bsz = len(features)
            return [pick] * bsz, [pick_len] * bsz

        # 兜底 2：历史池也为空（例如开局连续失败），使用最小占位 token，保证训练不中断
        fallback_id = int(os.getenv("SPEECH_TOKEN_FALLBACK_ID", "0"))
        fallback_len = int(os.getenv("SPEECH_TOKEN_FALLBACK_LEN", "1"))
        fallback_len = max(1, fallback_len)
        fb = torch.full((fallback_len,), fallback_id, dtype=torch.long)
        logging.error(
            "本 batch 所有音频 speech_token 提取均失败，且历史成功池为空，已使用占位 token 兜底继续训练（fallback_id=%s, fallback_len=%s, err=%s: %s）",
            fallback_id,
            fallback_len,
            first_err.__class__.__name__ if first_err is not None else "UnknownError",
            str(first_err)[:500] if first_err is not None else "",
        )
        bsz = len(features)
        return [fb] * bsz, [fallback_len] * bsz

    # 用“最近成功的”优先回退；如果失败发生在开头，则用第一个成功的。
    fallback = first_ok
    for i in range(len(features)):
        if speech_tokens[i] is None:
            use = last_ok if last_ok is not None else fallback
            speech_tokens[i] = use
            speech_token_lens[i] = int(use.numel())
        else:
            last_ok = speech_tokens[i]

    return [t for t in speech_tokens if t is not None], speech_token_lens


def _extract_mel_24k(audio_info: Any) -> torch.Tensor:
    wav_24k = _load_audio_mono(audio_info, target_sr=24000)
    if (wav_24k.shape[-1] - 480) // 480 % 2 == 0:
        wav_24k = torch.nn.functional.pad(wav_24k, (0, 480))
    mel = mel_spectrogram(wav_24k, 1920, 80, 24000, 480, 1920, 0, None, False)
    return mel.squeeze(0).transpose(0, 1).to(torch.float32)


def _extract_embedding_from_audio(audio_info: Any, sv_pipe: Any) -> np.ndarray:
    wav_16k = _load_audio_mono(audio_info, target_sr=16000)
    wav_np = wav_16k.squeeze(0).cpu().numpy().astype(np.float32)
    out = sv_pipe([wav_np], output_emb=True)
    emb = np.array(out["embs"][0], dtype=np.float32)
    return emb


class LlmPretrainDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None,
        tokenizer_onnx_path: str,
        onnx_use_cuda: bool,
        onnx_device_id: int,
        ort_intra_op_num_threads: int = 1,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_onnx_path = tokenizer_onnx_path
        self.onnx_use_cuda = onnx_use_cuda
        self.onnx_device_id = onnx_device_id
        self.ort_intra_op_num_threads = ort_intra_op_num_threads

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}

        text_tokens: List[torch.Tensor] = []
        text_token_lens: List[int] = []
        if "text_token" in features[0]:
            for f in features:
                tt = f["text_token"]
                if not isinstance(tt, torch.Tensor):
                    tt = torch.tensor(tt, dtype=torch.long)
                tt = tt.long()
                text_tokens.append(tt)
                text_token_lens.append(int(tt.numel()))
        elif "text" in features[0]:
            if self.tokenizer is None:
                raise ValueError("数据只有 text 字段但未提供 tokenizer，无法生成 text_token。")
            special_tokens = _get_added_special_tokens(self.tokenizer)
            for f in features:
                if tn is not None:
                    text = tn.process_text(f["text"])
                else:
                    text = f["text"]
                new_text = _maybe_append_en_cmu_tokens(text, special_tokens)
                if new_text == text:
                    new_text = _maybe_append_zh_pinyin_tokens(text, special_tokens)
                text = new_text
                if os.getenv("DEBUG_TEXT_AUG", "0") == "1":
                    print(text)
                ids = self.tokenizer.encode(text, allowed_special="all")
                tt = torch.tensor(ids, dtype=torch.long)
                text_tokens.append(tt)
                text_token_lens.append(int(tt.numel()))
        else:
            raise ValueError("LLM 训练需要 text_token 或 text 字段。")

        batch["text_token"] = pad_sequence(text_tokens, batch_first=True, padding_value=0)
        batch["text_token_len"] = torch.tensor(text_token_lens, dtype=torch.int64)

        if "audio" not in features[0]:
            raise ValueError("LLM 训练需要 audio 字段以实时提取 speech_token。")

        onnx_session = _get_onnx_tokenizer_session(
            self.tokenizer_onnx_path,
            use_cuda=self.onnx_use_cuda,
            device_id=self.onnx_device_id,
            intra_op_num_threads=self.ort_intra_op_num_threads,
        )
        speech_tokens, speech_token_lens = _extract_speech_tokens_with_batch_fallback(features, onnx_session)
        batch["speech_token"] = pad_sequence(speech_tokens, batch_first=True, padding_value=0)
        batch["speech_token_len"] = torch.tensor(speech_token_lens, dtype=torch.int64)

        bsz = len(features)
        batch["instruct_token"] = torch.zeros((bsz, 0), dtype=torch.long)
        batch["instruct_token_len"] = torch.zeros((bsz,), dtype=torch.int64)

        return batch


class FlowPretrainDataCollator:
    def __init__(
        self,
        tokenizer_onnx_path: str,
        onnx_use_cuda: bool,
        onnx_device_id: int,
        ort_intra_op_num_threads: int = 1,
        sv_model_path: str = "jzx-ai-lab/speech_campplus_sv_zh-cn_16k-common",
        sv_device: str | None = None,
        allow_online_embedding: bool = True,
    ):
        self.tokenizer_onnx_path = tokenizer_onnx_path
        self.onnx_use_cuda = onnx_use_cuda
        self.onnx_device_id = onnx_device_id
        self.ort_intra_op_num_threads = ort_intra_op_num_threads
        self.sv_model_path = sv_model_path
        self.sv_device = sv_device
        self.allow_online_embedding = allow_online_embedding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}

        if "audio" not in features[0]:
            raise ValueError("FLOW 训练需要 audio 字段以提取 speech_feat。")
        feats: List[torch.Tensor] = []
        feat_lens: List[int] = []
        for f in features:
            m = _extract_mel_24k(f["audio"])
            feats.append(m)
            feat_lens.append(int(m.shape[0]))
        batch["speech_feat"] = pad_sequence(feats, batch_first=True, padding_value=0.0)
        batch["speech_feat_len"] = torch.tensor(feat_lens, dtype=torch.int64)

        embs: List[torch.Tensor] = []
        if "embedding" in features[0] and features[0]["embedding"] is not None:
            for f in features:
                e = f["embedding"]
                if not isinstance(e, torch.Tensor):
                    e = torch.tensor(e, dtype=torch.float32)
                embs.append(e.to(torch.float32))
        else:
            if not self.allow_online_embedding:
                raise ValueError("数据缺少 embedding 且已关闭在线提取（--no_online_embedding）。")
            device = self.sv_device
            if device is None:
                device = f"cuda:{self.onnx_device_id}" if torch.cuda.is_available() else "cpu"
            sv_pipe = _get_speaker_verification_pipe(self.sv_model_path, device=device)
            for f in features:
                emb = _extract_embedding_from_audio(f["audio"], sv_pipe)
                embs.append(torch.tensor(emb, dtype=torch.float32))
        batch["embedding"] = torch.stack(embs, dim=0)

        speech_tokens: List[torch.Tensor] = []
        speech_token_lens: List[int] = []
        if "speech_token" in features[0] and features[0]["speech_token"] is not None:
            for f in features:
                st = f["speech_token"]
                if not isinstance(st, torch.Tensor):
                    st = torch.tensor(st, dtype=torch.long)
                st = st.long()
                speech_tokens.append(st)
                speech_token_lens.append(int(st.numel()))
        else:
            onnx_session = _get_onnx_tokenizer_session(
                self.tokenizer_onnx_path,
                use_cuda=self.onnx_use_cuda,
                device_id=self.onnx_device_id,
                intra_op_num_threads=self.ort_intra_op_num_threads,
            )
            speech_tokens, speech_token_lens = _extract_speech_tokens_with_batch_fallback(features, onnx_session)
        batch["speech_token"] = pad_sequence(speech_tokens, batch_first=True, padding_value=0)
        batch["speech_token_len"] = torch.tensor(speech_token_lens, dtype=torch.int64)

        return batch


class _TrainerForwardWrapper(nn.Module):
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core_model = core_model

    def forward(self, **batch):  # type: ignore[override]
        any_tensor = next((v for v in batch.values() if isinstance(v, torch.Tensor)), None)
        device = any_tensor.device if any_tensor is not None else next(self.core_model.parameters()).device
        return self.core_model(batch, device)


def _resolve_tokenizer_onnx_path(args: argparse.Namespace) -> str:
    if args.tokenizer_onnx_path:
        return args.tokenizer_onnx_path
    if args.tokenizer_path and args.tokenizer_path.endswith(".onnx"):
        return args.tokenizer_path
    return "jzx-ai-lab/HydraVox-CV3/speech_tokenizer_v3.onnx"


def _resolve_qwen_pretrain_path(args: argparse.Namespace) -> str:
    if args.qwen_pretrain_path:
        return args.qwen_pretrain_path
    if args.tokenizer_path:
        return args.tokenizer_path
    model_dir = os.getenv("TTS_MODEL_DIR", "jzx-ai-lab/HydraVox-CV3")
    return os.path.join(model_dir, "CosyVoice-BlankEN")


def _build_training_args(args: argparse.Namespace, cfg: Dict[str, Any], eval_dataset: Dataset | None) -> TrainingArguments:
    train_conf = cfg.get("train_conf", {}) if isinstance(cfg, dict) else {}
    optim_conf = train_conf.get("optim_conf", {}) if isinstance(train_conf, dict) else {}
    scheduler_conf = train_conf.get("scheduler_conf", {}) if isinstance(train_conf, dict) else {}

    learning_rate = args.learning_rate if args.learning_rate is not None else float(optim_conf.get("lr", 1e-5))
    num_train_epochs = args.num_train_epochs if args.num_train_epochs is not None else int(train_conf.get("max_epoch", 1))
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else int(train_conf.get("accum_grad", 1))
    )
    per_device_train_batch_size = args.per_device_train_batch_size if args.per_device_train_batch_size is not None else int(train_conf.get("batch_size", 8))
    per_device_eval_batch_size = args.per_device_eval_batch_size if args.per_device_eval_batch_size is not None else 1
    max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else float(train_conf.get("grad_clip", 1.0))
    logging_steps = args.logging_steps if args.logging_steps is not None else int(train_conf.get("log_interval", 50))
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(scheduler_conf.get("warmup_steps", 0))
    save_steps = args.save_steps if args.save_steps is not None else int(train_conf.get("save_per_step", 2000))

    scheduler_name = str(train_conf.get("scheduler", "linear")).lower()
    if scheduler_name in ("constantlr", "constant"):
        lr_scheduler_type = "constant_with_warmup" if warmup_steps and warmup_steps > 0 else "constant"
    else:
        lr_scheduler_type = "linear"

    if save_steps <= 0:
        save_strategy = "no"
        save_steps = 0
    else:
        save_strategy = "steps"

    return TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=int(per_device_train_batch_size),
        per_device_eval_batch_size=int(per_device_eval_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        max_grad_norm=float(max_grad_norm),
        logging_steps=int(logging_steps),
        save_strategy=save_strategy,
        save_steps=int(save_steps),
        save_total_limit=args.save_total_limit,
        evaluation_strategy="no" if eval_dataset is None else "steps",
        eval_steps=int(args.eval_steps) if args.eval_steps is not None else 1000,
        warmup_steps=int(warmup_steps),
        lr_scheduler_type=lr_scheduler_type,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        deepspeed=args.deepspeed,
        dataloader_num_workers=int(args.dataloader_num_workers),
        remove_unused_columns=False,
        save_safetensors=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llm", "flow"], required=True, help="模型类型")
    parser.add_argument("--config", type=str, default="", help="hyperpyyaml 配置路径")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径，逗号分隔")
    parser.add_argument("--cv_data", type=str, default="", help="验证数据路径，逗号分隔")
    parser.add_argument("--auto_val_split", action="store_true", default=False, help="自动划分验证集")
    parser.add_argument("--val_split_ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_ckpt", type=str, default="", help="初始模型 checkpoint")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Trainer 断点目录")
    parser.add_argument("--tokenizer_path", type=str, default="", help="LLM tokenizer/Qwen 路径；flow 可选 onnx 路径")
    parser.add_argument("--tokenizer_onnx_path", type=str, default="", help="speech tokenizer ONNX 路径")
    parser.add_argument("--qwen_pretrain_path", type=str, default="", help="Qwen2Encoder pretrain_path/tokenizer 路径")

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=6)

    parser.add_argument("--enable_lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--lora_target_modules", type=list, default=["q_proj", "v_proj", "k_proj"])

    parser.add_argument("--sv_model_path", type=str, default="jzx-ai-lab/speech_campplus_sv_zh-cn_16k-common")
    parser.add_argument("--sv_device", type=str, default="")
    parser.add_argument("--no_online_embedding", action="store_true", default=False)
    parser.add_argument("--onnx_use_cuda", action="store_true", default=False)
    parser.add_argument("--onnx_device_id", type=int, default=None)
    parser.add_argument("--ort_intra_op_num_threads", type=int, default=1)
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("🚀 训练脚本启动 (model=%s)", args.model)

    if args.enable_lora:
        logging.warning("新模型预训练不支持 LoRA 参数，已忽略 --enable_lora 等配置。")

    if not args.config:
        args.config = _maybe_get_default_config(args.model)

    resume_path = str(args.resume_from_checkpoint).strip()
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume_from_checkpoint 路径不存在：{resume_path}")
        if not os.path.isdir(resume_path):
            raise ValueError(f"--resume_from_checkpoint 需要传 checkpoint 目录，但得到：{resume_path}")
        logging.info("将从 Trainer checkpoint 断点续训：%s", resume_path)
    else:
        if not str(args.model_ckpt).strip():
            raise ValueError("未指定 --resume_from_checkpoint 时，必须提供 --model_ckpt 作为初始权重。")

    with open(args.config, "r") as f:
        if args.model == "llm":
            qwen_pretrain_path = _resolve_qwen_pretrain_path(args)
            cfg = load_hyperpyyaml(
                f,
                overrides={
                    "qwen_pretrain_path": qwen_pretrain_path,
                },
            )
            model = cfg["llm"]
        else:
            cfg = load_hyperpyyaml(
                f,
                overrides={
                    "llm": None,
                    "hift": None,
                    "hifigan": None,
                },
            )
            model = cfg["flow"]

    if not resume_path:
        model_state = _load_state_dict_maybe_container(args.model_ckpt)
        model_state.pop("epoch", None)
        model_state.pop("step", None)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            logging.warning("load_state_dict missing keys: %s（示例：%s）", len(missing), missing[:5])
        if unexpected:
            logging.warning("load_state_dict unexpected keys: %s（示例：%s）", len(unexpected), unexpected[:5])

    train_paths = [p for p in args.train_data.split(",") if p.strip()]
    val_paths = [p for p in args.cv_data.split(",") if p.strip()] if args.cv_data.strip() else []

    if args.model == "llm":
        train_dss = [prepare_dataset_llm_pretrain(load_from_disk(p)) for p in train_paths]
        val_dss = [prepare_dataset_llm_pretrain(load_from_disk(p)) for p in val_paths]
    else:
        train_dss = [prepare_dataset_flow_pretrain(load_from_disk(p)) for p in train_paths]
        val_dss = [prepare_dataset_flow_pretrain(load_from_disk(p)) for p in val_paths]

    train_dataset, eval_dataset = _build_train_eval_dataset(
        train_dss=train_dss,
        val_dss=val_dss,
        auto_val_split=bool(args.auto_val_split),
        val_split_ratio=float(args.val_split_ratio),
    )

    training_args = _build_training_args(args, cfg, eval_dataset)

    onnx_device_id = _resolve_onnx_device_id(args.onnx_device_id)
    tokenizer_onnx_path = _resolve_tokenizer_onnx_path(args)

    if args.model == "llm":
        qwen_pretrain_path = _resolve_qwen_pretrain_path(args)
        tokenizer = get_qwen_tokenizer(token_path=qwen_pretrain_path, skip_special_tokens=True, version="cosyvoice3")
        data_collator = LlmPretrainDataCollator(
            tokenizer=tokenizer,
            tokenizer_onnx_path=tokenizer_onnx_path,
            onnx_use_cuda=bool(args.onnx_use_cuda),
            onnx_device_id=int(onnx_device_id),
            ort_intra_op_num_threads=int(args.ort_intra_op_num_threads),
        )
    else:
        data_collator = FlowPretrainDataCollator(
            tokenizer_onnx_path=tokenizer_onnx_path,
            onnx_use_cuda=bool(args.onnx_use_cuda),
            onnx_device_id=int(onnx_device_id),
            ort_intra_op_num_threads=int(args.ort_intra_op_num_threads),
            sv_model_path=str(args.sv_model_path),
            sv_device=str(args.sv_device).strip() or None,
            allow_online_embedding=not bool(args.no_online_embedding),
        )

    model_for_trainer = _TrainerForwardWrapper(model)
    trainer = Trainer(
        model=model_for_trainer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=None,
    )

    logging.info("Training...")
    trainer.train(resume_from_checkpoint=resume_path or None)
    logging.info("Training completed. Saving model...")
    trainer.save_model()
    logging.info("All Finished.")


if __name__ == "__main__":
    main()
