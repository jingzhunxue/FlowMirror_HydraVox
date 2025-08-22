import os, gradio as gr
import json
import time
import re
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import threading
import logging

# 导入API客户端
from user_interface.utils.api_client import api_client

logger = logging.getLogger(__name__)

# 全局状态管理
class TrainingState:
    def __init__(self):
        self.current_training_id: Optional[str] = None
        self.is_training: bool = False
        self.log_update_timer = None
        # 图表缓存相关
        self.last_plot_update: float = 0
        self.plot_cache_duration: float = 10.0  # 缓存10秒
        self.cached_plot_path: Optional[str] = None
        self.last_log_size: int = 0  # 记录上次日志文件大小
        self.plot_update_interval: float = 5.0  # 图表更新间隔（秒）
        # 日志显示缓存
        self.cached_log_text: str = "等待开始训练..."
        self.last_log_update: float = 0
        self.log_cache_duration: float = 2.0  # 日志缓存2秒
        self.last_displayed_log_count: int = 0  # 上次显示的日志行数
        
training_state = TrainingState()

def load_training_config():
    """加载训练配置"""
    default_config = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "save_interval": 10,
        "validation_split": 0.1,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR"
    }
    return default_config

def save_training_config(config_dict: Dict[str, Any]):
    """保存训练配置"""
    config_path = "/tmp/training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    return f"配置已保存到: {config_path}"

def start_training(
    dataset_path: str, 
    model_type: str,
    model_checkpoint: str,
    tokenizer_path: str,
    output_dir: str,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    save_interval: int,
    validation_split: float,
    optimizer: str,
    scheduler: str,
    use_auto_split: bool,
    enable_lora: bool,
    precision_choice: str
):
    """启动训练任务"""
    global training_state
    
    if training_state.is_training:
        return "⚠️ 已有训练任务在运行中，请先停止当前训练"
    
    if not dataset_path:
        return "❌ 请先选择数据集文件"
    
    try:
        # 根据精度选择设置参数，确保只有一个为true
        use_fp16 = (precision_choice == "fp16")
        use_bf16 = (precision_choice == "bf16")
        
        # 构建训练配置
        config = {
            "model_type": model_type,
            "model_checkpoint": model_checkpoint,
            "tokenizer_path": tokenizer_path,
            "train_data": dataset_path,
            "output_dir": output_dir,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "save_steps": save_interval * 100,  # 转换为步数
            "auto_val_split": use_auto_split,
            "val_split_ratio": validation_split,
            "use_fp16": use_fp16,
            "use_bf16": use_bf16,
            "enable_lora": enable_lora
        }
        
        # 记录详细的参数信息用于调试
        logger.info("=" * 50)
        logger.info("🚀 准备启动训练任务")
        logger.info(f"精度选择: {precision_choice}")
        logger.info(f"use_fp16: {use_fp16}")
        logger.info(f"use_bf16: {use_bf16}")
        logger.info("训练配置参数:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 50)
        
        # 如果不使用自动分割，需要手动指定验证集路径
        if not use_auto_split:
            # 假设验证集在训练集同一目录下的val子目录
            train_path = Path(dataset_path)
            val_path = train_path.parent / "val" / train_path.name
            if val_path.exists():
                config["cv_data"] = str(val_path)
            else:
                config["auto_val_split"] = True  # 如果没有验证集，自动启用分割
        
        # 调用API启动训练
        result = api_client.start_training(config)
        
        if result.get("success"):
            training_state.current_training_id = result["data"]["training_id"]
            training_state.is_training = True
            
            # 重置日志和图表缓存
            training_state.cached_log_text = "正在启动训练..."
            training_state.last_log_update = 0
            training_state.last_displayed_log_count = 0
            training_state.last_log_size = 0
            training_state.last_plot_update = 0
            if training_state.cached_plot_path and os.path.exists(training_state.cached_plot_path):
                try:
                    os.remove(training_state.cached_plot_path)
                except:
                    pass
            training_state.cached_plot_path = None
            
            return f"✅ 训练任务已启动\n训练ID: {training_state.current_training_id}\nPID: {result['data']['pid']}\n状态: {result['data']['status']}"
        else:
            return f"❌ 训练启动失败: {result.get('message', '未知错误')}"
            
    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        return f"❌ 训练启动失败: {str(e)}"

def stop_training():
    """停止训练"""
    global training_state
    
    if not training_state.is_training or not training_state.current_training_id:
        return "⚠️ 当前没有运行中的训练任务"
    
    try:
        result = api_client.stop_training(training_state.current_training_id)
        
        if result.get("success"):
            training_state.is_training = False
            training_state.current_training_id = None
            return f"🛑 训练已停止: {result['message']}"
        else:
            return f"❌ 停止训练失败: {result.get('message', '未知错误')}"
            
    except Exception as e:
        logger.error(f"停止训练失败: {e}")
        return f"❌ 停止训练失败: {str(e)}"

def get_training_logs():
    """获取训练日志，带稳定的缓存机制"""
    global training_state
    
    current_time = time.time()
    
    if not training_state.current_training_id:
        training_state.cached_log_text = "暂无训练任务"
        return training_state.cached_log_text
    
    # 检查缓存是否仍然有效
    time_since_last_update = current_time - training_state.last_log_update
    if time_since_last_update < training_state.log_cache_duration:
        return training_state.cached_log_text
    
    try:
        result = api_client.get_training_status(training_state.current_training_id)
        
        if result.get("success"):
            data = result["data"]
            status = data["status"]
            logs = data.get("logs", [])
            
            # 更新训练状态
            if status in ["completed", "failed", "stopped"]:
                training_state.is_training = False
            
            # 检查是否有新的日志内容
            current_log_count = len(logs)
            if current_log_count == training_state.last_displayed_log_count and time_since_last_update < 5.0:
                # 如果日志行数没变且距离上次更新不到5秒，返回缓存
                return training_state.cached_log_text
            
            # 更新日志计数
            training_state.last_displayed_log_count = current_log_count
            
            # 构建稳定的日志头部信息
            header_info = []
            header_info.append(f"训练状态: {status}")
            header_info.append(f"训练ID: {training_state.current_training_id}")
            
            if data.get("start_time"):
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["start_time"]))
                header_info.append(f"开始时间: {start_time}")
            
            if data.get("end_time"):
                end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["end_time"]))
                header_info.append(f"结束时间: {end_time}")
            
            # 添加进度信息
            if logs:
                header_info.append(f"日志行数: {len(logs)}")
            
            header_info.append("=" * 50)
            header_text = "\n".join(header_info) + "\n"
            
            # 智能选择显示的日志行数
            if len(logs) <= 100:
                # 少于100行，全部显示
                displayed_logs = logs
            else:
                # 超过100行，显示最后80行，但保留完整的训练进度信息
                displayed_logs = logs[-80:]
            
            # 确保显示的日志以完整行结束
            log_content = "".join(displayed_logs)
            
            # 如果有截断，添加提示
            if len(logs) > len(displayed_logs):
                truncate_info = f"\n... (省略了前{len(logs) - len(displayed_logs)}行日志) ...\n\n"
                log_content = truncate_info + log_content
            
            # 构建最终的日志文本
            training_state.cached_log_text = header_text + log_content
            training_state.last_log_update = current_time
            
            return training_state.cached_log_text
        else:
            error_msg = f"获取日志失败: {result.get('message', '未知错误')}"
            training_state.cached_log_text = error_msg
            return error_msg
            
    except Exception as e:
        logger.error(f"获取训练日志失败: {e}")
        error_msg = f"获取日志失败: {str(e)}"
        training_state.cached_log_text = error_msg
        return error_msg

def parse_training_logs(log_file_path: str) -> Dict[str, List[float]]:
    """解析训练日志，提取训练指标"""
    metrics = {
        'steps': [],
        'loss': [],
        'grad_norm': [],
        'learning_rate': [],
        'epoch': []
    }
    
    try:
        if not os.path.exists(log_file_path):
            logger.warning(f"日志文件不存在: {log_file_path}")
            return metrics
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        step = 0
        for line in lines:
            # 匹配训练日志中的指标信息
            # 示例格式: {'loss': 5.2719, 'grad_norm': 2.815345287322998, 'learning_rate': 9.891681109185442e-05, 'epoch': 0.02}
            if line.strip().startswith('{') and 'loss' in line:
                try:
                    # 尝试直接使用eval解析字典（更安全的方法）
                    line_clean = line.strip()
                    if line_clean.endswith('\n'):
                        line_clean = line_clean[:-1]
                    
                    # 尝试解析为字典
                    try:
                        import ast
                        metrics_dict = ast.literal_eval(line_clean)
                        if isinstance(metrics_dict, dict) and 'loss' in metrics_dict:
                            step += 1
                            metrics['steps'].append(step)
                            metrics['loss'].append(float(metrics_dict['loss']))
                            metrics['grad_norm'].append(float(metrics_dict.get('grad_norm', 0)))
                            metrics['learning_rate'].append(float(metrics_dict.get('learning_rate', 0)))
                            metrics['epoch'].append(float(metrics_dict.get('epoch', 0)))
                    except (ValueError, SyntaxError):
                        # 如果ast.literal_eval失败，回到正则表达式方法
                        loss_match = re.search(r"'loss':\s*([\d\.-eE]+)", line)
                        grad_norm_match = re.search(r"'grad_norm':\s*([\d\.-eE]+)", line)
                        lr_match = re.search(r"'learning_rate':\s*([\d\.-eE]+)", line)
                        epoch_match = re.search(r"'epoch':\s*([\d\.-eE]+)", line)
                        
                        if loss_match:
                            step += 1
                            metrics['steps'].append(step)
                            metrics['loss'].append(float(loss_match.group(1)))
                            
                            metrics['grad_norm'].append(
                                float(grad_norm_match.group(1)) if grad_norm_match 
                                else (metrics['grad_norm'][-1] if metrics['grad_norm'] else 0)
                            )
                            metrics['learning_rate'].append(
                                float(lr_match.group(1)) if lr_match 
                                else (metrics['learning_rate'][-1] if metrics['learning_rate'] else 0)
                            )
                            metrics['epoch'].append(
                                float(epoch_match.group(1)) if epoch_match 
                                else (metrics['epoch'][-1] if metrics['epoch'] else 0)
                            )
                
                except (ValueError, AttributeError) as e:
                    logger.debug(f"解析日志行失败: {line.strip()}, 错误: {e}")
                    continue
        
        logger.info(f"从日志文件解析出 {len(metrics['loss'])} 个训练步骤的数据")
        
    except Exception as e:
        logger.error(f"解析训练日志失败: {e}")
    
    return metrics

def generate_training_plot(force_update: bool = False):
    """生成训练曲线图，带缓存和智能刷新"""
    global training_state
    
    current_time = time.time()
    
    # 检查是否需要更新（缓存机制）
    if not force_update and training_state.cached_plot_path and os.path.exists(training_state.cached_plot_path):
        time_since_last_update = current_time - training_state.last_plot_update
        if time_since_last_update < training_state.plot_cache_duration:
            logger.debug(f"使用缓存的训练图表，距离上次更新 {time_since_last_update:.1f} 秒")
            return training_state.cached_plot_path
    
    if not training_state.current_training_id:
        # 如果没有当前训练，生成示例图表
        return _generate_sample_plot()
    
    try:
        # 获取当前训练的日志文件路径
        result = api_client.get_training_status(training_state.current_training_id)
        
        if not result.get("success"):
            logger.warning("无法获取训练状态，使用示例图表")
            return _generate_sample_plot()
        
        training_data = result["data"]
        log_file = training_data.get("log_file")
        
        if not log_file:
            logger.debug("训练任务暂无日志文件路径")
            return None  # 返回None表示暂无数据
        
        if not os.path.exists(log_file):
            logger.debug(f"日志文件尚不存在: {log_file}，可能训练刚开始")
            return None  # 返回None表示暂无数据
        
        # 检查日志文件是否有更新（通过文件大小判断）
        current_log_size = os.path.getsize(log_file)
        if not force_update and current_log_size == training_state.last_log_size:
            # 日志文件没有更新，且缓存未过期
            if training_state.cached_plot_path and os.path.exists(training_state.cached_plot_path):
                time_since_last_update = current_time - training_state.last_plot_update
                if time_since_last_update < training_state.plot_update_interval:
                    logger.debug("日志文件无更新，使用缓存图表")
                    return training_state.cached_plot_path
        
        # 更新日志文件大小记录
        training_state.last_log_size = current_log_size
        
        # 解析日志获取训练数据
        metrics = parse_training_logs(log_file)
        
        if not metrics['loss']:
            logger.warning("日志中没有找到训练数据，使用示例图表")
            return _generate_sample_plot()
        
        logger.info(f"生成训练图表，包含 {len(metrics['loss'])} 个数据点")
        
        # 创建多子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'训练进度 - {training_state.current_training_id}', fontsize=16)
        
        steps = metrics['steps']
        
        # 子图1: Loss曲线
        ax1.plot(steps, metrics['loss'], color='blue', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax1.set_title('训练损失 (Loss)', fontsize=12)
        ax1.set_xlabel('步数 (Steps)')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        # 添加最新值标注
        if metrics['loss']:
            latest_loss = metrics['loss'][-1]
            ax1.annotate(f'最新: {latest_loss:.4f}', 
                        xy=(steps[-1], latest_loss), xytext=(0.7, 0.9),
                        textcoords='axes fraction', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # 子图2: 梯度范数
        if metrics['grad_norm']:
            ax2.plot(steps, metrics['grad_norm'], color='orange', linewidth=2, marker='s', markersize=3, alpha=0.7)
            ax2.set_title('梯度范数 (Gradient Norm)', fontsize=12)
            ax2.set_xlabel('步数 (Steps)')
            ax2.set_ylabel('Grad Norm')
            ax2.grid(True, alpha=0.3)
            # 添加最新值标注
            latest_grad_norm = metrics['grad_norm'][-1]
            ax2.annotate(f'最新: {latest_grad_norm:.4f}', 
                        xy=(steps[-1], latest_grad_norm), xytext=(0.7, 0.9),
                        textcoords='axes fraction', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightsalmon', alpha=0.7))
        
        # 子图3: 学习率
        if metrics['learning_rate']:
            ax3.plot(steps, metrics['learning_rate'], color='green', linewidth=2, marker='^', markersize=3, alpha=0.7)
            ax3.set_title('学习率 (Learning Rate)', fontsize=12)
            ax3.set_xlabel('步数 (Steps)')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True, alpha=0.3)
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            # 添加最新值标注
            latest_lr = metrics['learning_rate'][-1]
            ax3.annotate(f'最新: {latest_lr:.2e}', 
                        xy=(steps[-1], latest_lr), xytext=(0.7, 0.9),
                        textcoords='axes fraction', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # 子图4: Epoch进度
        if metrics['epoch']:
            ax4.plot(steps, metrics['epoch'], color='red', linewidth=2, marker='d', markersize=3, alpha=0.7)
            ax4.set_title('训练轮数 (Epoch)', fontsize=12)
            ax4.set_xlabel('步数 (Steps)')
            ax4.set_ylabel('Epoch')
            ax4.grid(True, alpha=0.3)
            # 添加最新值标注
            latest_epoch = metrics['epoch'][-1]
            ax4.annotate(f'最新: {latest_epoch:.3f}', 
                        xy=(steps[-1], latest_epoch), xytext=(0.7, 0.9),
                        textcoords='axes fraction', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        
        # 使用时间戳确保缓存文件唯一性
        plot_path = f"/tmp/training_plot_{int(current_time)}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 删除旧的缓存文件
        if training_state.cached_plot_path and os.path.exists(training_state.cached_plot_path):
            try:
                os.remove(training_state.cached_plot_path)
            except:
                pass
        
        # 更新缓存
        training_state.cached_plot_path = plot_path
        training_state.last_plot_update = current_time
        
        return plot_path
    
    except Exception as e:
        logger.error(f"生成训练图表失败: {e}")
        return _generate_sample_plot()

def _generate_sample_plot():
    """生成示例训练图表"""
    # 模拟训练数据
    steps = np.arange(1, 101)
    loss = 6.0 * np.exp(-steps/50) + 0.5 + 0.1 * np.random.randn(100)
    grad_norm = 3.0 * np.exp(-steps/60) + 0.1 + 0.05 * np.random.randn(100)
    lr = 1e-4 * np.ones(100) * np.exp(-steps/200)  # 衰减的学习率
    epoch = steps / 50  # 假设50步为一个epoch
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练进度 - 示例数据', fontsize=16)
    
    ax1.plot(steps, loss, color='blue', linewidth=2)
    ax1.set_title('训练损失 (Loss)', fontsize=12)
    ax1.set_xlabel('步数 (Steps)')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, grad_norm, color='orange', linewidth=2)
    ax2.set_title('梯度范数 (Gradient Norm)', fontsize=12)
    ax2.set_xlabel('步数 (Steps)')
    ax2.set_ylabel('Grad Norm')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(steps, lr, color='green', linewidth=2)
    ax3.set_title('学习率 (Learning Rate)', fontsize=12)
    ax3.set_xlabel('步数 (Steps)')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    ax4.plot(steps, epoch, color='red', linewidth=2)
    ax4.set_title('训练轮数 (Epoch)', fontsize=12)
    ax4.set_xlabel('步数 (Steps)')
    ax4.set_ylabel('Epoch')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = "/tmp/training_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def get_model_list():
    """获取训练输出目录下的模型文件夹列表"""
    try:
        # 训练输出目录 - 按优先级排序，避免重复扫描
        primary_output_dir = "checkpoints/training"  # 主要输出目录
        fallback_dirs = ["checkpoints", "models", "outputs", "ckpt"]  # 备用目录
        
        models = []
        processed_folders = set()  # 避免重复处理相同的文件夹
        
        # 首先扫描主要输出目录
        primary_path = Path(primary_output_dir)
        if primary_path.exists():
            logger.info(f"正在扫描主要输出目录: {primary_path}")
            models.extend(_scan_output_directory(primary_path, processed_folders))
        
        # 如果主要目录没有找到任何文件夹，再扫描备用目录
        if not models:
            for output_dir in fallback_dirs:
                output_path = Path(output_dir)
                if output_path.exists():
                    logger.info(f"正在扫描备用目录: {output_path}")
                    models.extend(_scan_output_directory(output_path, processed_folders))
        
        # 按修改时间倒序排列
        models.sort(key=lambda x: x["时间"], reverse=True)
        
        logger.info(f"找到 {len(models)} 个模型文件夹")
        
        if not models:
            models = [{"文件夹名称": "暂无训练输出", "路径": "请先进行模型训练", "内容": "", "大小": "", "时间": ""}]
        
        return pd.DataFrame(models)
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        return pd.DataFrame([{"文件夹名称": "获取失败", "路径": f"错误: {str(e)}", "内容": "", "大小": "", "时间": ""}])

def _scan_output_directory(output_path: Path, processed_folders: set):
    """扫描单个输出目录，返回找到的模型文件夹列表"""
    models = []
    
    # 遍历子文件夹
    for folder_path in output_path.iterdir():
        if folder_path.is_dir():
            folder_name = folder_path.name
            
            # 跳过日志目录和已处理的文件夹
            if folder_name == "runs" or folder_name in processed_folders:
                continue
                
            # 跳过嵌套的输出目录本身（避免显示checkpoints/training这样的路径）
            if folder_name in ["training", "checkpoints", "models", "outputs", "ckpt"]:
                continue
                
            try:
                processed_folders.add(folder_name)
                stat = folder_path.stat()
                mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                
                # 计算文件夹大小（包含的模型文件）
                total_size = 0
                model_count = 0
                model_files = []
                
                # 扫描模型文件
                for ext in ["*.pt", "*.pth", "*.ckpt", "*.bin", "*.safetensors"]:
                    for model_file in folder_path.rglob(ext):
                        if model_file.is_file():
                            total_size += model_file.stat().st_size
                            model_count += 1
                            model_files.append(model_file.name)
                
                size_mb = total_size / (1024 * 1024)
                
                # 显示相对路径
                relative_path = str(folder_path.relative_to(Path.cwd())) if folder_path.is_absolute() else str(folder_path)
                
                # 构建描述信息
                description = f"{model_count}个模型文件" if model_count > 0 else "空文件夹"
                if model_files:
                    # 只显示前3个文件名，如果更多则显示省略号
                    file_names = ", ".join(sorted(set(model_files))[:3])  # 去重并排序
                    if len(set(model_files)) > 3:
                        file_names += f" 等{len(set(model_files))}个文件"
                    description = file_names
                
                models.append({
                    "文件夹名称": folder_name,
                    "路径": relative_path,
                    "内容": description,
                    "大小": f"{size_mb:.1f} MB" if size_mb > 0 else "-",
                    "时间": mod_time
                })
                
            except Exception as e:
                logger.warning(f"Error reading model folder {folder_path}: {e}")
    
    return models

def load_model(model_name: str):
    """加载模型"""
    if not model_name:
        gr.Warning("请选择模型")
        return "请选择模型"
    
    return f"✅ 模型 {model_name} 加载成功"

def delete_model(folder_name: str):
    """删除模型文件夹"""
    if not folder_name:
        return "⚠️ 请选择要删除的文件夹", get_model_list()
    
    try:
        import shutil
        
        # 从文件夹名称中提取路径信息
        if " (" in folder_name and ")" in folder_name:
            # 格式: "folder_name (path/to/folder)"
            path_part = folder_name.split(" (")[1].rstrip(")")
            folder_path = Path(path_part)
        else:
            # 如果格式不对，尝试在各输出目录中查找
            output_dirs = [
                "checkpoints/training", 
                "checkpoints", 
                "models", 
                "outputs",
                "ckpt"
            ]
            folder_path = None
            for output_dir in output_dirs:
                potential_path = Path(output_dir) / folder_name
                if potential_path.exists() and potential_path.is_dir():
                    folder_path = potential_path
                    break
        
        if folder_path and folder_path.exists() and folder_path.is_dir():
            # 确认不是重要的系统文件夹
            if folder_path.name in ["runs"]:
                return f"⚠️ 不允许删除系统文件夹: {folder_name}", get_model_list()
            
            # 删除整个文件夹
            shutil.rmtree(folder_path)
            logger.info(f"已删除模型文件夹: {folder_path}")
            return f"✅ 文件夹 {folder_name} 已删除", get_model_list()
        else:
            return f"❌ 未找到文件夹: {folder_name}", get_model_list()
    
    except Exception as e:
        logger.error(f"删除文件夹失败: {e}")
        return f"❌ 删除失败: {str(e)}", get_model_list()

def update_batch_size_constraints(model_type: str):
    """根据模型类型更新batch_size限制"""
    if model_type == "llm":
        # LLM模型必须使用batch_size=1
        return (
            gr.update(value=1, maximum=1, interactive=False),  # batch_size slider
            gr.update(visible=True)  # info message
        )
    else:
        # Flow模型可以使用更大的batch_size
        return (
            gr.update(value=4, maximum=32, interactive=True),  # batch_size slider
            gr.update(visible=False)  # info message
        )

def update_precision_options(model_type: str):
    """根据模型类型更新精度选项和推荐"""
    if model_type == "llm":
        # LLM模型推荐BF16
        choices = [
            ("BF16（推荐）", "bf16"),
            ("FP16", "fp16")
        ]
        value = "bf16"
        info_text = "💡 **LLM模型**: 推荐使用BF16精度以获得更好的数值稳定性"
    else:
        # Flow模型推荐FP16
        choices = [
            ("FP16（推荐）", "fp16"),
            ("BF16", "bf16")
        ]
        value = "fp16"
        info_text = "💡 **Flow模型**: 推荐使用FP16精度以节省显存和提升速度"
    
    return (
        gr.update(choices=choices, value=value),  # precision_choice radio
        gr.update(value=info_text, visible=True)  # precision info message
    )

def create_training_tab():
    """创建训练tab界面"""
    with gr.Tab("🚀 模型训练"):
        gr.Markdown("### TTS 模型训练")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 数据集选择
                gr.Markdown("#### 1. 数据集配置")
                dataset_file = gr.Textbox(
                    label="训练数据路径",
                    placeholder="输入训练数据路径，如: data/processed/train_ds",
                    value="data/processed/train_ds"
                )
                
                # 模型配置
                gr.Markdown("#### 2. 模型配置")
                with gr.Group():
                    model_type = gr.Dropdown(
                        choices=["llm", "flow"],
                        value="llm",
                        label="模型类型"
                    )
                    model_checkpoint = gr.Textbox(
                        label="模型检查点路径",
                        value="jzx-ai-lab/HydraVox/llm.pt",
                        placeholder="预训练模型路径"
                    )
                    tokenizer_path = gr.Textbox(
                        label="分词器路径",
                        value="jzx-ai-lab/HydraVox/speech_tokenizer",
                        placeholder="分词器模型路径"
                    )
                    output_dir = gr.Textbox(
                        label="输出目录",
                        value="checkpoints/training",
                        placeholder="训练输出保存目录"
                    )
                
                # 训练参数配置
                gr.Markdown("#### 3. 训练参数")
                with gr.Group():
                    batch_size = gr.Slider(1, 32, value=1, step=1, label="批次大小", maximum=1, interactive=False)
                    batch_size_info = gr.Markdown("💡 **注意**: LLM模型训练时batch_size必须为1，Flow模型可以使用更大的batch_size", visible=True)
                    learning_rate = gr.Number(value=1e-4, label="学习率", minimum=1e-6, maximum=1e-2)
                    epochs = gr.Slider(1, 100, value=10, step=1, label="训练轮数")
                    save_interval = gr.Slider(1, 50, value=20, step=1, label="保存间隔(轮数)")
                
                with gr.Group():
                    validation_split = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="验证集比例")
                    use_auto_split = gr.Checkbox(label="自动划分验证集", value=True)
                    
                # 高级选项
                gr.Markdown("#### 4. 高级选项")
                with gr.Group():
                    enable_lora = gr.Checkbox(label="启用LoRA微调", value=False)
                    precision_choice = gr.Radio(
                        choices=[
                            ("BF16（推荐）", "bf16"),
                            ("FP16", "fp16")
                        ],
                        value="bf16",
                        label="精度设置"
                    )
                    precision_info = gr.Markdown("💡 **LLM模型**: 推荐使用BF16精度以获得更好的数值稳定性", visible=True)
                
                # 控制按钮
                gr.Markdown("#### 5. 训练控制")
                start_btn = gr.Button("🚀 开始训练", variant="primary")
                stop_btn = gr.Button("🛑 停止训练", variant="stop")
                refresh_log_btn = gr.Button("🔄 刷新日志", variant="secondary")
                
            with gr.Column(scale=2):
                # 训练状态
                gr.Markdown("#### 训练状态与日志")
                training_status = gr.Textbox(
                    label="训练日志",
                    lines=15,
                    interactive=False,
                    value="等待开始训练...",
                    max_lines=30
                )
                
                # 自动刷新日志 - 增加刷新间隔以减少闪烁
                log_timer = gr.Timer(value=5)  # 每5秒刷新日志
                
                # 训练曲线
                gr.Markdown("#### 训练曲线")
                with gr.Row():
                    with gr.Column(scale=3):
                        training_plot = gr.Image(label="训练指标曲线", value=_generate_sample_plot())
                    with gr.Column(scale=1):
                        gr.Markdown("**图表设置**")
                        auto_refresh_plot = gr.Checkbox(label="自动刷新图表", value=True)
                        plot_refresh_interval = gr.Slider(
                            minimum=5, maximum=60, value=15, step=5,
                            label="刷新间隔(秒)", interactive=True
                        )
                        with gr.Row():
                            refresh_plot_btn = gr.Button("🔄 立即刷新", variant="secondary")
                            force_refresh_btn = gr.Button("⚡ 强制刷新", variant="primary")
                
                # 自动刷新图表定时器
                plot_timer = gr.Timer(value=15)  # 默认15秒刷新一次图表
        
        # 模型管理
        gr.Markdown("### 模型管理")
        with gr.Row():
            with gr.Column(scale=2):
                model_list = gr.Dataframe(
                    value=get_model_list(),
                    headers=["文件夹名称", "路径", "内容", "大小", "时间"],
                    label="训练输出文件夹",
                    interactive=False
                )
                
            with gr.Column(scale=1):
                gr.Markdown("#### 文件夹操作")
                selected_model = gr.Textbox(label="选择的文件夹", placeholder="点击表格行选择文件夹")
                
                with gr.Row():
                    refresh_models_btn = gr.Button("🔄 刷新列表", variant="secondary")
                
                with gr.Row():
                    load_btn = gr.Button("📂 加载文件夹", variant="primary")
                    delete_btn = gr.Button("🗑️ 删除文件夹", variant="stop")
                
                model_status = gr.Textbox(
                    label="操作状态",
                    interactive=False
                )
        
        # 训练配置显示
        gr.Markdown("### 当前配置")
        config_display = gr.JSON(
            value=load_training_config(),
            label="训练配置"
        )
        
        # 事件绑定
        def update_config():
            return {
                "batch_size": batch_size.value,
                "learning_rate": learning_rate.value,
                "epochs": epochs.value,
                "save_interval": save_interval.value,
                "validation_split": validation_split.value,
                "optimizer": optimizer.value,
                "scheduler": scheduler.value
            }
        
        # 绑定训练控制事件
        start_btn.click(
            fn=start_training,
            inputs=[
                dataset_file, model_type, model_checkpoint, tokenizer_path, output_dir,
                batch_size, learning_rate, epochs, save_interval, validation_split,
                gr.State("Adam"), gr.State("CosineAnnealingLR"),  # 暂时固定优化器和调度器
                use_auto_split, enable_lora, precision_choice
            ],
            outputs=training_status
        )
        
        stop_btn.click(
            fn=stop_training,
            outputs=training_status
        )
        
        refresh_log_btn.click(
            fn=get_training_logs,
            outputs=training_status
        )
        
        # 智能日志刷新 - 只在训练时刷新
        def smart_log_refresh():
            """智能日志刷新：只在有训练任务时更新"""
            if training_state.current_training_id and training_state.is_training:
                return get_training_logs()
            elif training_state.current_training_id:
                # 训练已结束但仍有ID，获取最终日志
                final_logs = get_training_logs()
                # 如果训练已结束，可以减少刷新频率
                if not training_state.is_training:
                    return final_logs
            return training_state.cached_log_text
        
        log_timer.tick(
            fn=smart_log_refresh,
            outputs=training_status
        )
        
        # 图表刷新事件
        def update_plot_with_settings(auto_refresh_enabled):
            """根据自动刷新设置更新图表"""
            if auto_refresh_enabled and training_state.is_training:
                return generate_training_plot()
            elif not auto_refresh_enabled:
                # 如果关闭自动刷新，返回当前缓存的图表或生成新的
                return generate_training_plot()
            else:
                # 没有训练时显示示例
                return _generate_sample_plot()
        
        def force_update_plot():
            """强制刷新图表，忽略缓存"""
            return generate_training_plot(force_update=True)
        
        def update_plot_timer_interval(interval):
            """更新图表定时器间隔"""
            training_state.plot_update_interval = interval
            return gr.update(value=interval)
        
        # 立即刷新按钮
        refresh_plot_btn.click(
            fn=lambda: generate_training_plot(),
            outputs=training_plot
        )
        
        # 强制刷新按钮  
        force_refresh_btn.click(
            fn=force_update_plot,
            outputs=training_plot
        )
        
        # 刷新间隔设置
        plot_refresh_interval.change(
            fn=update_plot_timer_interval,
            inputs=plot_refresh_interval,
            outputs=plot_timer
        )
        
        # 自动刷新图表定时器
        def auto_refresh_plot_handler():
            if auto_refresh_plot.value and training_state.is_training:
                return generate_training_plot()
            return None
        
        plot_timer.tick(
            fn=auto_refresh_plot_handler,
            outputs=training_plot
        )
        
        # 刷新模型列表
        refresh_models_btn.click(
            fn=get_model_list,
            outputs=model_list
        )
        
        # 模型表格选择事件
        def on_model_select(evt: gr.SelectData):
            if evt.index is not None and evt.index[0] >= 0:
                # 获取选中行的文件夹信息
                model_data = get_model_list()
                if len(model_data) > evt.index[0]:
                    selected_name = model_data.iloc[evt.index[0]]["文件夹名称"]
                    selected_path = model_data.iloc[evt.index[0]]["路径"]
                    return f"{selected_name} ({selected_path})"
            return ""
        
        model_list.select(
            fn=on_model_select,
            outputs=selected_model
        )
        
        load_btn.click(
            fn=load_model,
            inputs=selected_model,
            outputs=model_status
        )
        
        delete_btn.click(
            fn=delete_model,
            inputs=selected_model,
            outputs=[model_status, model_list]
        )
        
        # 监听模型类型变化，自动调整batch_size限制和精度选项
        def update_model_constraints(model_type_val):
            batch_updates = update_batch_size_constraints(model_type_val)
            precision_updates = update_precision_options(model_type_val)
            return batch_updates + precision_updates
        
        model_type.change(
            fn=update_model_constraints,
            inputs=model_type,
            outputs=[batch_size, batch_size_info, precision_choice, precision_info]
        )
       