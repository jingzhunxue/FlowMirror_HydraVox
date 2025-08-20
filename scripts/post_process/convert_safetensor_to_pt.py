#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeTensors 到 PyTorch .pt 格式转换脚本

支持单个文件转换和批量目录转换
使用方法:
  python convert_safetensor_to_pt.py --input model.safetensors --output model.pt
  python convert_safetensor_to_pt.py --input_dir ./safetensors_models --output_dir ./pt_models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    print("错误: 请先安装 safetensors 库")
    print("运行: pip install safetensors")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_safetensors_file(safetensors_path: str) -> Dict[str, torch.Tensor]:
    """
    从safetensors文件加载张量
    
    Args:
        safetensors_path: safetensors文件路径
        
    Returns:
        包含所有张量的字典
    """
    try:
        logger.info(f"正在加载 safetensors 文件: {safetensors_path}")
        
        # 使用safetensors加载
        state_dict = load_safetensors(safetensors_path)
        
        logger.info(f"成功加载 {len(state_dict)} 个张量")
        
        # 打印张量信息
        total_params = 0
        for key, tensor in state_dict.items():
            params = tensor.numel()
            total_params += params
            logger.debug(f"  {key}: {tensor.shape} ({tensor.dtype}) - {params:,} 参数")
        
        logger.info(f"总参数量: {total_params:,}")
        return state_dict
        
    except Exception as e:
        logger.error(f"加载 safetensors 文件失败: {e}")
        raise


def save_pytorch_file(state_dict: Dict[str, torch.Tensor], output_path: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    保存为PyTorch .pt格式
    
    Args:
        state_dict: 模型状态字典
        output_path: 输出文件路径
        metadata: 可选的元数据信息
    """
    try:
        logger.info(f"正在保存 PyTorch 文件: {output_path}")
        
        # 准备保存的数据
        save_data = state_dict.copy()
        
        # 添加元数据（如果有）
        if metadata:
            save_data.update(metadata)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存为.pt格式
        torch.save(save_data, output_path)
        
        # 验证保存的文件
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"保存成功! 文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"保存 PyTorch 文件失败: {e}")
        raise


def convert_single_file(input_path: str, output_path: str, 
                       preserve_metadata: bool = True,
                       add_conversion_info: bool = True) -> bool:
    """
    转换单个文件
    
    Args:
        input_path: 输入safetensors文件路径
        output_path: 输出pt文件路径
        preserve_metadata: 是否保留元数据
        add_conversion_info: 是否添加转换信息
        
    Returns:
        转换是否成功
    """
    try:
        # 检查输入文件
        if not os.path.exists(input_path):
            logger.error(f"输入文件不存在: {input_path}")
            return False
        
        if not input_path.lower().endswith('.safetensors'):
            logger.warning(f"输入文件不是 .safetensors 格式: {input_path}")
        
        # 加载safetensors文件
        state_dict = load_safetensors_file(input_path)
        
        # 准备元数据
        metadata = {}
        
        if add_conversion_info:
            metadata['_conversion_info'] = {
                'source_format': 'safetensors',
                'target_format': 'pytorch',
                'source_file': os.path.basename(input_path),
                'converted_by': 'convert_safetensor_to_pt.py'
            }
        
        # 尝试读取safetensors的元数据
        if preserve_metadata:
            try:
                with safe_open(input_path, framework="pt") as f:
                    if hasattr(f, 'metadata') and f.metadata():
                        metadata['_original_metadata'] = f.metadata()
                        logger.info(f"保留了原始元数据: {len(f.metadata())} 个条目")
            except Exception as e:
                logger.warning(f"无法读取原始元数据: {e}")
        
        # 保存为.pt格式
        save_pytorch_file(state_dict, output_path, metadata if metadata else None)
        
        logger.info(f"✅ 转换完成: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 转换失败: {input_path} -> {output_path}, 错误: {e}")
        return False


def convert_directory(input_dir: str, output_dir: str, 
                     preserve_metadata: bool = True,
                     add_conversion_info: bool = True) -> None:
    """
    批量转换目录中的safetensors文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        preserve_metadata: 是否保留元数据
        add_conversion_info: 是否添加转换信息
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 查找所有safetensors文件
    safetensors_files = list(input_path.rglob("*.safetensors"))
    
    if not safetensors_files:
        logger.warning(f"在目录 {input_dir} 中未找到 .safetensors 文件")
        return
    
    logger.info(f"找到 {len(safetensors_files)} 个 .safetensors 文件")
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    for safetensors_file in safetensors_files:
        try:
            # 计算相对路径并构建输出路径
            relative_path = safetensors_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.pt')
            
            # 确保输出子目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换文件
            if convert_single_file(str(safetensors_file), str(output_file), 
                                 preserve_metadata, add_conversion_info):
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"处理文件 {safetensors_file} 时发生错误: {e}")
            failed_count += 1
    
    logger.info(f"批量转换完成! 成功: {success_count}, 失败: {failed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="将 SafeTensors 格式转换为 PyTorch .pt 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单文件转换
  python convert_safetensor_to_pt.py -i model.safetensors -o model.pt
  
  # 批量转换目录
  python convert_safetensor_to_pt.py -d ./safetensors_models -D ./pt_models
  
  # 自动输出文件名
  python convert_safetensor_to_pt.py -i model.safetensors
        """
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-i', '--input',
        type=str,
        help='输入的 .safetensors 文件路径'
    )
    input_group.add_argument(
        '-d', '--input-dir',
        type=str,
        help='输入目录路径 (批量转换)'
    )
    
    # 输出选项
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出的 .pt 文件路径 (单文件转换时使用，不指定则自动生成)'
    )
    parser.add_argument(
        '-D', '--output-dir',
        type=str,
        help='输出目录路径 (批量转换时使用)'
    )
    
    # 其他选项
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='不保留原始元数据'
    )
    parser.add_argument(
        '--no-conversion-info',
        action='store_true',
        help='不添加转换信息到输出文件'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.input:
            # 单文件转换
            if args.output:
                output_path = args.output
            else:
                # 自动生成输出文件名
                input_path = Path(args.input)
                output_path = str(input_path.with_suffix('.pt'))
            
            logger.info("开始单文件转换...")
            success = convert_single_file(
                args.input, 
                output_path,
                preserve_metadata=not args.no_metadata,
                add_conversion_info=not args.no_conversion_info
            )
            
            if success:
                logger.info("🎉 转换成功完成!")
                sys.exit(0)
            else:
                logger.error("💥 转换失败!")
                sys.exit(1)
                
        elif args.input_dir:
            # 批量转换
            if not args.output_dir:
                logger.error("批量转换需要指定输出目录 (--output-dir)")
                sys.exit(1)
            
            logger.info("开始批量转换...")
            convert_directory(
                args.input_dir, 
                args.output_dir,
                preserve_metadata=not args.no_metadata,
                add_conversion_info=not args.no_conversion_info
            )
            logger.info("🎉 批量转换完成!")
            
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
