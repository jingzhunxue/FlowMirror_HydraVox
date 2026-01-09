#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将“旧 CosyVoice3LM checkpoint（不含 mtp_block.* 参数）”转换为“新 checkpoint（补齐 mtp_block 权重）”。

适用场景：
- 你已经在 `cosyvoice/llm/llm_multi_head_v3.py` 的 `CosyVoice3LM` 中引入了多头训练用的 `mtp_block`；
- 旧 ckpt 里没有 `mtp_block.*`，导致新模型 strict load 失败；
- 本脚本会读取旧 ckpt，按给定 head_num/mtp_head_num 初始化对应数量的 `Qwen2DecoderLayer`，
  把 `mtp_block.{i}.*` 参数补进 state_dict，保存为新 ckpt。

注意：
- 本脚本不会改动旧 ckpt 里已有的权重，只会补齐缺失的 `mtp_block.*`。
- `llm_decoder` 仍然是共享的一套（不做多 decoder）。
"""

import argparse
import os
from typing import Any, Dict, Tuple

import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

_TRANSLATIONS = {
    "无法识别 checkpoint 格式：期望为 state_dict(dict[str,Tensor]) 或 {'state_dict': state_dict}。": {
        "en": "Unrecognized checkpoint format: expected state_dict(dict[str,Tensor]) or {'state_dict': state_dict}."
    },
    "旧 checkpoint 缺少 key: speech_embedding.weight，无法推断 llm_input_size/vocab_size": {
        "en": "Old checkpoint missing key: speech_embedding.weight; cannot infer llm_input_size/vocab_size"
    },
    "旧 checkpoint 缺少 key: llm_decoder.weight，无法推断 llm_output_size": {
        "en": "Old checkpoint missing key: llm_decoder.weight; cannot infer llm_output_size"
    },
    "维度不一致：llm_decoder.weight.shape[0]={dec_out} != vocab_size={vocab_size}（来自 speech_embedding.weight.shape[0]）": {
        "en": "Shape mismatch: llm_decoder.weight.shape[0]={dec_out} != vocab_size={vocab_size} (from speech_embedding.weight.shape[0])"
    },
    "推断 speech_token_size={speech_token_size} 不合理（vocab_size={vocab_size}，期望 vocab_size=speech_token_size+200）": {
        "en": "Inferred speech_token_size={speech_token_size} is invalid (vocab_size={vocab_size}, expected vocab_size=speech_token_size+200)"
    },
    "旧 checkpoint 路径（.pt/.pth），不含 mtp_block.*": {
        "en": "Old checkpoint path (.pt/.pth), without mtp_block.*"
    },
    "输出新 checkpoint 路径（.pt/.pth）": {"en": "Output new checkpoint path (.pt/.pth)"},
    "mtp head 数量（对应 mtp_block 的层数）": {"en": "mtp head count (mtp_block layers)"},
    "Qwen2DecoderLayer 的 attention heads 数": {"en": "Attention heads for Qwen2DecoderLayer"},
    "初始化 mtp_block 权重用随机种子": {"en": "Random seed for mtp_block init"},
    "[OK] 写入新 checkpoint: {path}": {"en": "[OK] Wrote new checkpoint: {path}"},
    "  - 推断 llm_input_size={llm_input_size}, llm_output_size={llm_output_size}, vocab_size={vocab_size}, speech_token_size={speech_token_size}": {
        "en": "  - Inferred llm_input_size={llm_input_size}, llm_output_size={llm_output_size}, vocab_size={vocab_size}, speech_token_size={speech_token_size}"
    },
    "  - head_num={head_num}, mtp_head_num={mtp_head_num}": {
        "en": "  - head_num={head_num}, mtp_head_num={mtp_head_num}"
    },
    "  - 新增参数条目数: {added}": {"en": "  - Added parameter entries: {added}"},
}


def _t(text: str, **kwargs: Any) -> str:
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


def _extract_state_dict(obj: Any) -> Tuple[Dict[str, torch.Tensor], Any, bool]:
    """
    Returns:
        state_dict: 参数字典
        container: 原始容器（若 ckpt 是 {'state_dict': ...} 形式，需要回写）
        has_container: 是否为容器形式
    """
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
        return sd, obj, True
    if isinstance(obj, dict):
        # 常见：直接 torch.save(model.state_dict())
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, obj, False
    raise ValueError(_t("无法识别 checkpoint 格式：期望为 state_dict(dict[str,Tensor]) 或 {'state_dict': state_dict}。"))


def _infer_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """
    从旧权重推断维度：
    - llm_input_size: speech_embedding.weight 的 embedding dim
    - vocab_size: speech_embedding.weight 的行数
    - llm_output_size: llm_decoder.weight 的 in_features
    """
    if "speech_embedding.weight" not in state_dict:
        raise KeyError(_t("旧 checkpoint 缺少 key: speech_embedding.weight，无法推断 llm_input_size/vocab_size"))
    if "llm_decoder.weight" not in state_dict:
        raise KeyError(_t("旧 checkpoint 缺少 key: llm_decoder.weight，无法推断 llm_output_size"))

    vocab_size, llm_input_size = state_dict["speech_embedding.weight"].shape
    dec_out, llm_output_size = state_dict["llm_decoder.weight"].shape
    if dec_out != vocab_size:
        raise ValueError(_t(
            "维度不一致：llm_decoder.weight.shape[0]={dec_out} != vocab_size={vocab_size}（来自 speech_embedding.weight.shape[0]）",
            dec_out=dec_out,
            vocab_size=vocab_size,
        ))
    return int(llm_input_size), int(llm_output_size), int(vocab_size)


def _cast_state_dict_to_bf16(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """将 state_dict 中所有浮点 tensor 转为 bfloat16；非浮点 tensor 保持不变。"""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if torch.is_floating_point(v):
            out[k] = v.to(dtype=torch.bfloat16)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_ckpt", type=str, required=True, help=_t("旧 checkpoint 路径（.pt/.pth），不含 mtp_block.*"))
    parser.add_argument("--dst_ckpt", type=str, required=True, help=_t("输出新 checkpoint 路径（.pt/.pth）"))
    parser.add_argument("--head_num", type=int, default=5, help=_t("mtp head 数量（对应 mtp_block 的层数）"))
    parser.add_argument("--mtp_head_num", type=int, default=14, help=_t("Qwen2DecoderLayer 的 attention heads 数"))
    parser.add_argument("--seed", type=int, default=1986, help=_t("初始化 mtp_block 权重用随机种子"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ckpt_obj = torch.load(args.src_ckpt, map_location="cpu")
    state_dict, container, has_container = _extract_state_dict(ckpt_obj)

    llm_input_size, llm_output_size, vocab_size = _infer_dims(state_dict)
    speech_token_size = vocab_size - 200
    if speech_token_size <= 0:
        raise ValueError(_t(
            "推断 speech_token_size={speech_token_size} 不合理（vocab_size={vocab_size}，期望 vocab_size=speech_token_size+200）",
            speech_token_size=speech_token_size,
            vocab_size=vocab_size,
        ))

    # 用最小必要字段构造 Qwen2Config（与训练侧 mtp_block 初始化一致）
    cfg = Qwen2Config(
        hidden_size=llm_input_size,
        num_attention_heads=args.mtp_head_num,
        num_key_value_heads=args.mtp_head_num,
    )

    added = 0
    for i in range(args.head_num):
        layer = Qwen2DecoderLayer(cfg, 0).cpu()
        layer_sd = layer.state_dict()
        for k, v in layer_sd.items():
            full_key = f"mtp_block.{i}.{k}"
            if full_key not in state_dict:
                state_dict[full_key] = v
                added += 1

    # 保存前统一转 bf16（仅对浮点权重生效，int/bool buffer 保持原样）
    state_dict = _cast_state_dict_to_bf16(state_dict)

    if has_container:
        container["state_dict"] = state_dict
        out_obj = container
    else:
        out_obj = state_dict

    torch.save(out_obj, args.dst_ckpt)
    print(
        "\n".join(
            [
                _t("[OK] 写入新 checkpoint: {path}", path=args.dst_ckpt),
                _t(
                    "  - 推断 llm_input_size={llm_input_size}, llm_output_size={llm_output_size}, vocab_size={vocab_size}, speech_token_size={speech_token_size}",
                    llm_input_size=llm_input_size,
                    llm_output_size=llm_output_size,
                    vocab_size=vocab_size,
                    speech_token_size=speech_token_size,
                ),
                _t("  - head_num={head_num}, mtp_head_num={mtp_head_num}", head_num=args.head_num, mtp_head_num=args.mtp_head_num),
                _t("  - 新增参数条目数: {added}", added=added),
            ]
        )
    )


if __name__ == "__main__":
    main()

