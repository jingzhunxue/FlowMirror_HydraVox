#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将训练保存的 `pytorch_model.bin` 转换为 `model.pt`（默认转为 bf16）。

默认处理路径:
  cv3_llm_multihead_pretrain/checkpoint-10000/pytorch_model.bin

用法示例:
  python scripts/post_process/convert_checkpoint_bin_to_pt.py
  python scripts/post_process/convert_checkpoint_bin_to_pt.py -i /path/to/pytorch_model.bin -o /out/model.pt
  python scripts/post_process/convert_checkpoint_bin_to_pt.py -i checkpoints/run/pytorch_model.bin --keep-dtype
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch


def convert_state_dict_to_bf16(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """将 state_dict 中的浮点权重转换为 bf16 以节省空间。"""
    converted = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            try:
                converted[key] = value.to(torch.bfloat16)
            except Exception:
                # 某些张量可能不支持转换，保持原样
                converted[key] = value
        else:
            converted[key] = value
    return converted


def normalize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理常见包装:
    - 如果外层只有 core_model 键，展开其内部字典
    - 如果键名前缀为 core_model.，去掉前缀
    """
    if "core_model" in state_dict and isinstance(state_dict["core_model"], dict):
        state_dict = state_dict["core_model"]

    normalized = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith("core_model."):
            stripped = key[len("core_model.") :]
        else:
            stripped = key
        normalized[stripped] = value
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="将 pytorch_model.bin 转换为 model.pt（默认 bf16）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="cv3_llm_multihead_pretrain/checkpoint-10000/pytorch_model.bin",
        help="pytorch_model.bin 文件路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="输出 model.pt 路径（默认与输入同目录）",
    )
    parser.add_argument(
        "--keep-dtype",
        action="store_true",
        help="保持原始 dtype，不转换为 bf16",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)

    if input_path.name != "pytorch_model.bin":
        print(f"⚠️ 输入文件名不是 pytorch_model.bin: {input_path.name}")

    index_file = input_path.with_name("pytorch_model.bin.index.json")
    if index_file.exists():
        print("❌ 检测到分片权重（pytorch_model.bin.index.json），请先合并后再转换")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.parent / "model.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"📥 正在加载权重: {input_path}")
        state = torch.load(str(input_path), map_location="cpu")
        if not isinstance(state, dict):
            print("❌ 权重文件格式不符合预期（需要 state_dict 字典）")
            sys.exit(1)

        state = normalize_state_dict(state)

        if args.keep_dtype:
            converted = state
            print("ℹ️ 保持原始 dtype，不执行 bf16 转换")
        else:
            converted = convert_state_dict_to_bf16(state)
            print("✅ 已将可转换的浮点张量转为 bf16")

        torch.save(converted, str(output_path))
        print(f"🎉 转换完成: {output_path}")
    except Exception as exc:
        print(f"❌ 转换失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
