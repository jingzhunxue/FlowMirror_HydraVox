import os, gradio as gr
import json
import time
import re
import subprocess, sys, signal
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import threading
import logging

# 训练脚本路径工具
from pathlib import Path

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
        # 子进程与日志
        self.proc: Optional[subprocess.Popen] = None
        self.proc_pid: Optional[int] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.log_lines: List[str] = []
        self.start_time: float = 0.0
        self.end_time: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.output_dir: Optional[str] = None
        self.cmdline: List[str] = []
        self.log_file: Optional[Any] = None  # 日志文件句柄
        self.logging_steps: int = 50  # 默认每50步记录一次，会在训练时更新
        self.eval_steps: int = 500  # 默认每500步评估一次，会在训练时更新
        
training_state = TrainingState()

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _train_script_path() -> Path:
    # 训练脚本（使用 HF Trainer 实现）
    return _project_root() / "scripts/train/train_speech_model.py"


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


def _refresh_device_triplet():
    d, p, detail = _auto_detect_device_and_processes()
    return detail, p, ("GPU" if d == "GPU" else "CPU")


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
    precision_choice: str,
    device_choice: str,
    gpu_processes: float,
    gpu_ids: str,
    logging_steps: int = 50,
    eval_steps: int = 500
):
    """以子进程方式启动训练脚本，并在当前 Gradio 中管理生命周期。"""
    global training_state
    
    if training_state.is_training:
        return "⚠️ 已有训练任务在运行中，请先停止当前训练"
    
    if not dataset_path:
        return "❌ 请先选择数据集文件"
    
    # 精度选项
    use_fp16 = (precision_choice == "fp16")
    use_bf16 = (precision_choice == "bf16")

    try:
        script_path = _train_script_path()
        if not script_path.exists():
            return f"❌ 找不到训练脚本: {script_path}"

        # 设置训练开始时间
        training_state.start_time = time.time()
        
        # 保存状态
        training_state.output_dir = output_dir
        training_state.logging_steps = logging_steps  # 保存 logging_steps 值
        training_state.eval_steps = eval_steps  # 保存 eval_steps 值
        training_state.log_lines = []
        training_state.cached_log_text = "正在启动训练..."
        training_state.last_log_update = 0
        training_state.last_displayed_log_count = 0
        training_state.last_log_size = 0
        training_state.exit_code = None
        training_state.end_time = None
        # 清空训练图表缓存，避免显示上次训练的图表
        training_state.cached_plot_path = None
        training_state.last_plot_update = 0
        
        # 创建日志文件（现在start_time已经设置）
        log_dir = Path("logs/training")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"train_{int(training_state.start_time)}.log"
        try:
            training_state.log_file = open(log_file_path, "w", encoding="utf-8")
            logger.info(f"训练日志将保存到: {log_file_path}")
        except Exception as e:
            logger.warning(f"无法创建日志文件: {e}")
            training_state.log_file = None

        # 自动验证集路径逻辑
        cv_data_arg = None
        if not use_auto_split:
            train_path = Path(dataset_path)
            val_path = train_path.parent / "val" / train_path.name
            if val_path.exists():
                cv_data_arg = str(val_path)
            else:
                # 若未找到指定验证集，则自动切换为自动划分
                use_auto_split = True

        # 训练脚本参数
        script_args: List[str] = [
            "--model", model_type,
            "--model_ckpt", model_checkpoint,
            "--tokenizer_path", tokenizer_path,
            "--train_data", dataset_path,
            "--output_dir", output_dir,
            "--per_device_train_batch_size", str(int(batch_size)),
            "--learning_rate", str(float(learning_rate)),
            "--num_train_epochs", str(int(epochs)),
            "--save_steps", str(int(save_interval)),
            "--logging_steps", str(int(logging_steps)),  # 添加 logging_steps 参数
            "--eval_steps", str(int(eval_steps)),  # 添加 eval_steps 参数
            "--val_split_ratio", str(float(validation_split)),
        ]
        
        # 记录训练参数以便调试
        logger.info(f"训练参数: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}, save_steps={save_interval}")
        if use_auto_split:
            script_args.append("--auto_val_split")
        else:
            if cv_data_arg:
                script_args.extend(["--cv_data", cv_data_arg])
        if enable_lora:
            script_args.append("--enable_lora")
        if use_fp16:
            script_args.append("--fp16")
        if use_bf16:
            script_args.append("--bf16")

        # 设备选择与进程数
        dev_detect, max_gpus, _detail = _auto_detect_device_and_processes()
        chosen = device_choice
        if chosen == "自动":
            chosen = "GPU" if dev_detect == "GPU" else "CPU"
        try:
            nproc = max(1, int(gpu_processes))
        except Exception:
            nproc = 1

        # mixed_precision for accelerate
        mixed_precision = "no"
        if use_bf16:
            mixed_precision = "bf16"
        elif use_fp16:
            mixed_precision = "fp16"

        # 环境变量（限制可见 GPU）
        env = os.environ.copy()
        cuda_ids = (gpu_ids or "").strip()
        if chosen == "GPU":
            if cuda_ids:
                env["CUDA_VISIBLE_DEVICES"] = cuda_ids
            else:
                # 默认选择从 0 开始的前 nproc 张卡
                if max_gpus > 0:
                    take = max(1, min(nproc, max_gpus))
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(take))

        # 组装 accelerate 启动命令（统一使用 accelerate）
        cmd: List[str] = [
            sys.executable,
            "-m", "accelerate.commands.launch",
            "--num_machines", "1",
            "--num_processes", str(nproc if chosen == "GPU" else 1),
            "--mixed_precision", mixed_precision,
            str(script_path),
            *script_args,
        ]

        training_state.cmdline = cmd
        
        # 记录完整的命令以便调试
        logger.info(f"执行训练命令: {' '.join(cmd)}")

        # 启动子进程（独立进程组，便于停止）
        try:
            training_state.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                env=env,
            )
        except Exception as e:
            return f"❌ 启动失败: {e}"

        training_state.is_training = True
        training_state.proc_pid = training_state.proc.pid if training_state.proc else None
        training_state.current_training_id = f"local-{int(training_state.start_time)}"

        # 启动日志读取线程
        def _reader():
            try:
                assert training_state.proc is not None
                assert training_state.proc.stdout is not None
                for raw in training_state.proc.stdout:
                    line = raw.rstrip()
                    if not line:
                        continue
                    training_state.log_lines.append(line)
                    if len(training_state.log_lines) > 2000:
                        training_state.log_lines = training_state.log_lines[-2000:]
                    # 同时写入日志文件
                    if training_state.log_file:
                        try:
                            training_state.log_file.write(line + "\n")
                            training_state.log_file.flush()  # 实时刷新
                        except Exception as we:
                            logger.warning(f"写入日志文件失败: {we}")
                    # 轻量更新缓存文本标记更新时间
                    training_state.cached_log_text = "\n".join(training_state.log_lines[-200:])
                    training_state.last_log_update = time.time()
            except Exception as re:
                logger.warning(f"日志读取线程异常: {re}")
            finally:
                try:
                    if training_state.proc is not None:
                        ret = training_state.proc.wait()
                        training_state.exit_code = ret
                except Exception:
                    pass
                training_state.is_training = False
                training_state.end_time = time.time()
                # 关闭日志文件
                if training_state.log_file:
                    try:
                        training_state.log_file.close()
                        training_state.log_file = None
                        logger.info("训练日志文件已关闭")
                    except Exception:
                        pass

        training_state.reader_thread = threading.Thread(target=_reader, daemon=True)
        training_state.reader_thread.start()

        return f"✅ 训练任务已启动\n训练ID: {training_state.current_training_id}\nPID: {training_state.proc_pid}\n脚本: {script_path.name}"

    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        return f"❌ 训练启动失败: {str(e)}"

def stop_training():
    """停止训练（终止子进程）。"""
    global training_state
    
    if not training_state.is_training or training_state.proc is None:
        return "⚠️ 当前没有运行中的训练任务"
    
    try:
        proc = training_state.proc
        # 先尝试优雅终止
        try:
            if proc.poll() is None:
                if hasattr(os, "getpgid"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
        except Exception:
            pass
        # 等待最多5秒
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                if hasattr(os, "getpgid"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
        training_state.is_training = False
        training_state.current_training_id = None
        training_state.end_time = time.time()
        code = proc.returncode
        training_state.exit_code = code
        # 关闭日志文件
        if training_state.log_file:
            try:
                training_state.log_file.close()
                training_state.log_file = None
            except Exception:
                pass
        return f"🛑 训练已停止 (退出码: {code})"
    except Exception as e:
        logger.error(f"停止训练失败: {e}")
        return f"❌ 停止训练失败: {str(e)}"

def get_training_logs():
    """获取本地子进程的训练日志（带缓存）。"""
    global training_state
    
    current_time = time.time()
    
    if not training_state.current_training_id and not training_state.is_training:
        training_state.cached_log_text = "暂无训练任务"
        return training_state.cached_log_text
    
    # 缓存控制
    time_since_last_update = current_time - training_state.last_log_update
    if time_since_last_update < training_state.log_cache_duration:
        return training_state.cached_log_text

    try:
        status = "running" if training_state.is_training else ("stopped" if training_state.exit_code is None else ("completed" if training_state.exit_code == 0 else "failed"))
        logs = training_state.log_lines
        training_state.last_displayed_log_count = len(logs)

        header_info: List[str] = []
        header_info.append(f"训练状态: {status}")
        if training_state.current_training_id:
            header_info.append(f"训练ID: {training_state.current_training_id}")
        if training_state.start_time:
            st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_state.start_time))
            header_info.append(f"开始时间: {st}")
        if training_state.end_time:
            et = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_state.end_time))
            header_info.append(f"结束时间: {et}")
        if logs:
            header_info.append(f"日志行数: {len(logs)}")
        header_info.append("=" * 50)
        header_text = "\n".join(header_info) + "\n"

        # 显示最近 200 行
        if len(logs) <= 200:
            displayed = logs
        else:
            displayed = logs[-200:]
        log_content = "\n".join(displayed)
        if len(logs) > len(displayed):
            log_content = f"... (省略了前{len(logs) - len(displayed)}行日志) ...\n\n" + log_content

        training_state.cached_log_text = header_text + log_content
        training_state.last_log_update = current_time
        return training_state.cached_log_text
    except Exception as e:
        logger.error(f"获取训练日志失败: {e}")
        training_state.cached_log_text = f"获取日志失败: {str(e)}"
        return training_state.cached_log_text

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


def _parse_metrics_from_lines(lines: List[str]) -> Dict[str, List[float]]:
    """从内存中的日志行解析指标（用于回退绘图）。"""
    metrics: Dict[str, List[float]] = {
        'steps': [],
        'loss': [],
        'eval_loss': [],
        'grad_norm': [],
        'learning_rate': [],
        'epoch': []
    }
    try:
        train_logs = []
        eval_logs = []
        
        for line in lines:
            s = line.strip()
            # 清理 ANSI 控制码（如 [A 等）
            import re
            s = re.sub(r'\x1b\[[A-Za-z0-9;]*[A-Za-z]', '', s)  # 移除ANSI escape sequences
            s = re.sub(r'\[A', '', s)  # 移除 [A 控制码
            s = s.strip()
            
            if not s or not s.startswith('{'):
                continue
                
            try:
                import ast
                d = ast.literal_eval(s)
                if isinstance(d, dict):
                    if 'loss' in d:  # 训练日志
                        train_logs.append(d)
                    elif 'eval_loss' in d:  # 评估日志
                        eval_logs.append(d)
                    continue
            except Exception:
                pass
            
            # 回退到正则表达式
            if 'loss' in s and "'loss'" in s:
                try:
                    import ast
                    d = ast.literal_eval(s)
                    if isinstance(d, dict):
                        if 'loss' in d:
                            train_logs.append(d)
                        elif 'eval_loss' in d:
                            eval_logs.append(d)
                except Exception:
                    # 最后的回退：正则表达式
                    loss_match = re.search(r"'loss':\s*([\d\.-eE]+)", s)
                    if loss_match:
                        d = {'loss': float(loss_match.group(1))}
                        grad_norm_match = re.search(r"'grad_norm':\s*([\d\.-eE]+)", s)
                        lr_match = re.search(r"'learning_rate':\s*([\d\.-eE]+)", s)
                        epoch_match = re.search(r"'epoch':\s*([\d\.-eE]+)", s)
                        if grad_norm_match:
                            d['grad_norm'] = float(grad_norm_match.group(1))
                        if lr_match:
                            d['learning_rate'] = float(lr_match.group(1))
                        if epoch_match:
                            d['epoch'] = float(epoch_match.group(1))
                        train_logs.append(d)
                    
                    eval_loss_match = re.search(r"'eval_loss':\s*([\d\.-eE]+)", s)
                    if eval_loss_match:
                        eval_logs.append({'eval_loss': float(eval_loss_match.group(1))})
        
        # 构建训练数据
        for i, d in enumerate(train_logs):
            metrics['steps'].append(i + 1)
            metrics['loss'].append(float(d.get('loss', 0)))
            metrics['grad_norm'].append(float(d.get('grad_norm', 0)))
            metrics['learning_rate'].append(float(d.get('learning_rate', 0)))
            metrics['epoch'].append(float(d.get('epoch', 0)))
        
        # 构建评估数据
        for d in eval_logs:
            metrics['eval_loss'].append(float(d.get('eval_loss', 0)))
        
        logger.info(f"🔍 从日志行解析: train_logs={len(train_logs)}, eval_logs={len(eval_logs)}")
        if eval_logs:
            logger.info(f"🔍 日志行中找到的 eval_logs 样本: {eval_logs[0] if eval_logs else 'None'}")
        if train_logs:
            logger.info(f"🔍 日志行中找到的 train_logs 样本: {train_logs[0] if train_logs else 'None'}")
        
    except Exception as e:
        logger.warning(f"解析日志行失败: {e}")
        
    return metrics

def generate_training_plot(force_update: bool = False):
    """生成训练曲线图，优先解析 Trainer 的 trainer_state.json。"""
    global training_state
    
    current_time = time.time()
    
    # 缓存控制 - 简化逻辑，避免缓存问题
    if not force_update and training_state.cached_plot_path and os.path.exists(training_state.cached_plot_path):
        time_since_last_update = current_time - training_state.last_plot_update
        # 只有在没有新数据时才使用缓存
        if time_since_last_update < training_state.plot_cache_duration and not training_state.is_training:
            logger.debug(f"使用缓存的训练图表，距离上次更新 {time_since_last_update:.1f} 秒")
            return training_state.cached_plot_path

    # 没有任务时返回空
    if not training_state.output_dir or not training_state.current_training_id:
        return None

    try:
        steps: List[int] = []
        loss: List[float] = []
        eval_loss: List[float] = []
        learning_rate: List[float] = []
        epoch: List[float] = []
        grad_norm: List[float] = []

        # 优先解析 trainer_state.json
        state_file = os.path.join(training_state.output_dir, "trainer_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    st = json.load(f)
                logs = st.get("log_history", []) or []
                
                # 分别处理训练和评估日志
                train_logs = []
                eval_logs = []
                
                for entry in logs:
                    if not isinstance(entry, dict):
                        continue
                    # 注意：eval_loss 条目通常也包含其他字段，需要准确识别
                    if "eval_loss" in entry:  # 评估日志 - 优先检查
                        eval_logs.append(entry)
                    elif "loss" in entry and "eval_loss" not in entry:  # 训练日志
                        train_logs.append(entry)
                
                # 处理训练日志
                for i, entry in enumerate(train_logs):
                    s = entry.get("step")
                    if s is not None:
                        steps.append(int(s))
                        loss.append(float(entry.get("loss", 0)))
                        grad_norm.append(float(entry.get("grad_norm", 0)))
                        learning_rate.append(float(entry.get("learning_rate", 0)))
                        epoch.append(float(entry.get("epoch", 0)))
                
                # 处理评估日志（独立处理）
                for entry in eval_logs:
                    eval_loss.append(float(entry.get("eval_loss", 0)))
                
                logger.info(f"🔍 从 trainer_state.json 解析: train_logs={len(train_logs)}, eval_logs={len(eval_logs)}")
                if eval_logs:
                    logger.info(f"🔍 找到的 eval_logs 样本: {eval_logs[0] if eval_logs else 'None'}")
                
            except Exception as e:
                logger.warning(f"解析 trainer_state.json 失败: {e}")
                # 读取失败则退回日志解析
                steps = []

        # 如果 trainer_state.json 为空或不存在，回退到内存日志解析
        if not steps and training_state.log_lines:
            m = _parse_metrics_from_lines(training_state.log_lines)
            steps = m['steps']
            loss = m['loss']
            eval_loss = m['eval_loss']
            learning_rate = m['learning_rate']
            epoch = m['epoch']
            grad_norm = m['grad_norm']

        # 如果没有数据，返回空
        if not steps:
            return None
            
        # 调试信息：记录数据数量
        valid_eval_count = len([v for v in eval_loss if v > 0])
        logger.info(f"🎯 绘图数据统计: steps={len(steps)}, loss={len(loss)}, eval_loss={len(eval_loss)}, valid_eval_loss={valid_eval_count}")
        logger.info(f"📊 完整 eval_loss 数组: {eval_loss}")  # 显示完整数组
        if eval_loss:
            logger.info(f"📊 eval_loss 样本: {eval_loss[:5]}...")  # 显示前5个值
            logger.info(f"📊 eval_loss 数据类型: {[type(v) for v in eval_loss[:3]]}")
        if valid_eval_count > 0:
            logger.info(f"✅ 发现有效 eval_loss 数据，应该会显示eval loss曲线")
        else:
            logger.warning(f"⚠️ 没有有效的 eval_loss 数据！原始数据: {eval_loss[:10]}")

        # 统一长度（简单对齐，缺失用 None 跳过绘图）
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        title_id = training_state.current_training_id or "local"
        fig.suptitle(f'Training Progress - {title_id}', fontsize=16)
        
        # 自适应横坐标函数
        def format_x_axis(ax, steps_data):
            if not steps_data:
                return
            max_steps = max(steps_data)
            # 根据步数范围自动调整刻度间隔
            if max_steps < 50:
                interval = 5
            elif max_steps < 100:
                interval = 10
            elif max_steps < 1000:
                interval = 100
            elif max_steps < 10000:
                interval = 1000
            elif max_steps < 100000:
                interval = 5000
            else:
                interval = 10000
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))
            # 格式化大数字显示
            if max_steps >= 10000:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}K' if x >= 1000 else f'{int(x)}'))
            # 旋转标签避免重叠
            ax.tick_params(axis='x', rotation=45)

        # Train Loss (左上角)
        if loss:
            # 过滤掉为0的loss值
            valid_loss = [(i, v) for i, v in enumerate(loss) if v > 0]
            if valid_loss:
                loss_indices, loss_values = zip(*valid_loss)
                actual_steps = [i * training_state.logging_steps for i in loss_indices]
                ax1.plot(actual_steps, loss_values, color='blue', linewidth=2, marker='o', markersize=3, alpha=0.7, label='train loss')
        ax1.set_title('Train Loss', fontsize=12)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        # 应用自适应横坐标格式（只考虑训练损失）
        if loss:
            valid_loss = [(i, v) for i, v in enumerate(loss) if v > 0]
            if valid_loss:
                loss_indices, _ = zip(*valid_loss)
                actual_steps = [i * training_state.logging_steps for i in loss_indices]
                format_x_axis(ax1, actual_steps)
        if loss:
            ax1.legend()

        # Gradient Norm (根据logging_steps记录间隔显示真实步数)
        if grad_norm and any(x > 0 for x in grad_norm):  # 只有在有有效数据时才绘制
            actual_steps_grad = [s * training_state.logging_steps for s in range(len(grad_norm))]
            ax2.plot(actual_steps_grad, grad_norm, color='orange', linewidth=2, marker='s', markersize=3, alpha=0.7)
        ax2.set_title('Gradient Norm', fontsize=12)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Grad Norm')
        ax2.grid(True, alpha=0.3)
        if not grad_norm or not any(x > 0 for x in grad_norm):
            ax2.text(0.5, 0.5, 'No Data', transform=ax2.transAxes, ha='center', va='center', alpha=0.5)
        else:
            format_x_axis(ax2, actual_steps_grad)

        # Learning Rate (根据logging_steps记录间隔显示真实步数)
        if learning_rate:
            actual_steps_lr = [s * training_state.logging_steps for s in range(len(learning_rate))]
            ax3.plot(actual_steps_lr, learning_rate, color='green', linewidth=2, marker='^', markersize=3, alpha=0.7)
        ax3.set_title('Learning Rate', fontsize=12)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        if learning_rate:
            format_x_axis(ax3, actual_steps_lr)

        # Eval Loss (根据eval_steps记录间隔显示真实步数)
        if eval_loss:
            # 过滤掉为0的eval_loss值
            valid_eval_loss = [(i, v) for i, v in enumerate(eval_loss) if v > 0]
            if valid_eval_loss:
                eval_indices, eval_values = zip(*valid_eval_loss)
                # eval_loss 从第1次评估开始，所以步数是 (i+1) * eval_steps
                actual_steps_eval = [(i + 1) * training_state.eval_steps for i in eval_indices]
                ax4.plot(actual_steps_eval, eval_values, color='red', linewidth=2, marker='s', markersize=4, alpha=0.7, label='eval loss')
                format_x_axis(ax4, actual_steps_eval)
                logger.info(f"🎨 绘制 eval_loss 曲线: {len(eval_values)} 个点，步数范围 {min(actual_steps_eval)}-{max(actual_steps_eval)}")
            else:
                ax4.text(0.5, 0.5, 'No Eval Data', transform=ax4.transAxes, ha='center', va='center', alpha=0.5)
        else:
            ax4.text(0.5, 0.5, 'No Eval Data', transform=ax4.transAxes, ha='center', va='center', alpha=0.5)
        ax4.set_title('Eval Loss', fontsize=12)
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Eval Loss')
        ax4.grid(True, alpha=0.3)
        if eval_loss:
            ax4.legend()

        plt.tight_layout()
        
        # 将图表保存到训练输出目录下的figure文件夹
        if training_state.output_dir:
            figure_dir = os.path.join(training_state.output_dir, "figure")
            os.makedirs(figure_dir, exist_ok=True)
            # 只保存一个固定名称的图片，避免产生大量文件
            plot_path = os.path.join(figure_dir, "training_plot.png")
        else:
            plot_path = f"/tmp/training_plot_{int(current_time)}.png"
        
        try:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.debug(f"训练图表已更新: {plot_path}")
        except Exception as save_error:
            logger.error(f"保存训练图表失败: {save_error}")
        finally:
            plt.close()

        # 更新缓存
        training_state.cached_plot_path = plot_path
        training_state.last_plot_update = current_time
        return plot_path
    except Exception as e:
        logger.error(f"生成训练图表失败: {e}")
        return None

# 已删除示例图表生成函数

def get_model_list(which: str = "llm"):
    """获取训练输出目录下的模型文件夹列表（按模型类型区分）
    
    Args:
        which: "llm" 或 "flow"，用于选择首要扫描目录
    """
    try:
        # 训练输出目录 - 按优先级排序，避免重复扫描
        primary_output_dir = f"checkpoints/training_{which}"  # 主要输出目录
        fallback_dirs = [
            "checkpoints/training_llm",
            "checkpoints/training_flow", 
            "checkpoints/training",
            "checkpoints", 
            "models", 
            "outputs", 
            "ckpt"
        ]  # 备用目录
        
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
            
            # 跳过日志目录、图表目录和已处理的文件夹
            if folder_name in ["runs", "logs", "figure"] or folder_name in processed_folders:
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
        
        # 直接将输入视为路径；兼容旧格式 "name (path)"
        folder_path: Optional[Path] = None
        if " (" in folder_name and ")" in folder_name:
            path_part = folder_name.split(" (")[1].rstrip(")")
            folder_path = Path(path_part)
        else:
            p = Path(folder_name)
            if p.exists() and p.is_dir():
                folder_path = p
            else:
                # 回退在各输出目录中查找同名子目录
                output_dirs = [
                    "checkpoints/training", 
                    "checkpoints", 
                    "models", 
                    "outputs",
                    "ckpt"
                ]
                for output_dir in output_dirs:
                    potential_path = Path(output_dir) / folder_name
                    if potential_path.exists() and potential_path.is_dir():
                        folder_path = potential_path
                        break
        
        if folder_path and folder_path.exists() and folder_path.is_dir():
            # 确认不是重要的系统文件夹
            if folder_path.name in ["runs", "logs", "figure"]:
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


def convert_checkpoint_to_pt(folder_path_str: str):
    """将路径下的 pytorch_model.bin 转换为 model.pt（bf16）。"""
    if not folder_path_str:
        return "⚠️ 请先在表格中选择一个路径"
    try:
        base = Path(folder_path_str)
        if not base.exists() or not base.is_dir():
            return f"❌ 路径无效: {base}"

        bin_path = base / "pytorch_model.bin"
        if not bin_path.exists():
            found = list(base.rglob("pytorch_model.bin"))
            if found:
                bin_path = found[0]
            else:
                return f"❌ 未找到 pytorch_model.bin 于: {base}"

        # 分片索引不支持
        if (base / "pytorch_model.bin.index.json").exists():
            return "❌ 暂不支持分片权重（.bin.index.json），请先合并再转换"

        state = torch.load(str(bin_path), map_location="cpu")
        if not isinstance(state, dict):
            return "❌ 权重文件格式不符合预期（非state_dict）"

        def to_bf16_tensor(val):
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                try:
                    return val.to(torch.bfloat16)
                except Exception:
                    return val
            return val

        if "core_model" in state and isinstance(state["core_model"], dict):
            state = state["core_model"]

        normalized = {}
        for key, value in state.items():
            if key == "core_model":
                continue
            if isinstance(key, str) and key.startswith("core_model."):
                key = key[len("core_model.") :]
            normalized[key] = value

        converted = {k: to_bf16_tensor(v) for k, v in normalized.items()}
        out_path = base / "model.pt"
        torch.save(converted, str(out_path))
        return f"✅ 转换完成: {bin_path.name} → {out_path} (bf16)"
    except Exception as e:
        return f"❌ 转换失败: {e}"

def update_batch_size_constraints(model_type: str):
    """根据模型类型更新batch_size推荐值"""
    if model_type == "llm":
        # LLM模型推荐使用较小的batch_size
        return gr.update(value=2, maximum=32, interactive=True)  # batch_size slider
    else:
        # Flow模型可以使用更大的batch_size
        return gr.update(value=8, maximum=32, interactive=True)  # batch_size slider

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
        # 设备默认值
        device_default, proc_default, device_detail = _auto_detect_device_and_processes()
        
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
                        value="jzx-ai-lab/HydraVox-CV3/llm.pt",
                        placeholder="预训练模型路径"
                    )
                    tokenizer_path = gr.Textbox(
                        label="分词器路径",
                        value="jzx-ai-lab/HydraVox-CV3/CosyVoice-BlankEN",
                        placeholder="分词器模型路径"
                    )
                    output_dir = gr.Textbox(
                        label="输出目录",
                        value="checkpoints/training_llm",
                        placeholder="训练输出保存目录"
                    )
                
                # 训练参数配置
                gr.Markdown("#### 3. 训练参数")
                with gr.Group():
                    batch_size = gr.Slider(1, 32, value=4, step=1, label="批次大小", maximum=1, interactive=True)
                    learning_rate = gr.Number(value=1e-5, label="学习率", minimum=1e-6, maximum=1e-2)
                    epochs = gr.Slider(1, 100, value=5, step=1, label="训练轮数")
                    save_interval = gr.Slider(100, 10000, value=1000, step=100, label="保存间隔(步数)")
                    logging_steps = gr.Slider(10, 500, value=50, step=10, label="日志记录间隔(步数)")
                    eval_steps = gr.Slider(50, 2000, value=500, step=50, label="评估间隔(步数)")
                
                with gr.Group():
                    validation_split = gr.Slider(0.00, 0.3, value=0.05, step=0.01, label="验证集比例")
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

                # 计算资源设置
                gr.Markdown("#### 5. 计算资源设置")
                with gr.Group():
                    with gr.Row():
                        device_choice = gr.Dropdown(
                            choices=["自动", "CPU", "GPU"],
                            value=("GPU" if device_default == "GPU" else "CPU"),
                            label="💻 计算设备"
                        )
                        gpu_processes = gr.Number(value=proc_default, label="🔄 并行进程数 (GPU数)")
                    with gr.Row():
                        gpu_ids = gr.Textbox(label="🎯 GPU IDs (可选)", placeholder="例如: 0,1")
                        detect_btn = gr.Button("🔄 刷新设备检测", variant="secondary")
                    device_info = gr.Textbox(value=device_detail, label="ℹ️ 设备检测信息", interactive=False)

                # 控制按钮
                gr.Markdown("#### 6. 训练控制")
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
                        training_plot = gr.Image(label="训练指标曲线", value=None)
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
                        
                        plot_save_info = gr.Markdown(
                            """
                            **💾 图表存储位置**  
                            训练图表会实时更新并保存到：  
                            `checkpoints/training/figure/training_plot.png`  
                            """,
                            elem_classes=["tiny-muted"]
                        )
                
                # 自动刷新图表定时器
                plot_timer = gr.Timer(value=15)  # 默认15秒刷新一次图表
        
        # 模型管理
        gr.Markdown("### 模型管理")
        with gr.Row():
            with gr.Column(scale=2):
                # 仅显示路径一级（按模型类型显示，初始为 llm）
                _df_paths = get_model_list("llm")[ ["路径"] ]
                model_list = gr.Dataframe(
                    value=_df_paths,
                    headers=["路径"],
                    label="训练输出路径",
                    interactive=False
                )
                
            with gr.Column(scale=1):
                gr.Markdown("#### 文件夹操作")
                selected_model = gr.Textbox(label="选择的文件夹", placeholder="点击表格行选择文件夹")
                
                with gr.Row():
                    refresh_models_btn = gr.Button("🔄 刷新列表", variant="secondary")
                
                with gr.Row():
                    load_btn = gr.Button("📂 加载路径", variant="primary")
                    delete_btn = gr.Button("🗑️ 删除路径", variant="stop")
                with gr.Row():
                    convert_btn = gr.Button("🔁 转换为 model.pt (bf16)", variant="primary")
                
                model_status = gr.Textbox(
                    label="操作状态",
                    interactive=False
                )
        
        
        # 动态更新图表存储位置提示
        def update_plot_save_info(output_dir_value):
            return f"""
            **💾 图表存储位置**  
            训练图表会实时更新并保存到：  
            `{output_dir_value}/figure/training_plot.png`  
            """
        
        output_dir.change(
            fn=update_plot_save_info,
            inputs=output_dir,
            outputs=plot_save_info
        )
        
        # 绑定训练控制事件
        start_btn.click(
            fn=start_training,
            inputs=[
                dataset_file, model_type, model_checkpoint, tokenizer_path, output_dir,
                batch_size, learning_rate, epochs, save_interval, validation_split,
                gr.State("Adam"), gr.State("CosineAnnealingLR"),  # 暂时固定优化器和调度器
                use_auto_split, enable_lora, precision_choice,
                device_choice, gpu_processes, gpu_ids, logging_steps, eval_steps
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
        def auto_refresh_plot_handler(auto_refresh_enabled):
            if auto_refresh_enabled and training_state.is_training:
                return generate_training_plot()
            elif training_state.current_training_id and not training_state.is_training:
                # 训练停止后也展示最后的图表
                return generate_training_plot()
            return gr.update()  # 不更新
        
        plot_timer.tick(
            fn=auto_refresh_plot_handler,
            inputs=auto_refresh_plot,
            outputs=training_plot
        )
        
        # 刷新模型列表（按模型类型）
        def _list_model_paths(which: str):
            try:
                return get_model_list(which)[["路径"]]
            except Exception:
                return get_model_list(which)

        refresh_models_btn.click(
            fn=_list_model_paths,
            inputs=model_type,
            outputs=model_list
        )
        
        # 模型表格选择事件（按模型类型）
        def on_model_select(evt: gr.SelectData, which: str):
            if evt.index is not None and evt.index[0] >= 0:
                model_data = get_model_list(which)
                if len(model_data) > evt.index[0]:
                    selected_path = model_data.iloc[evt.index[0]]["路径"]
                    return f"{selected_path}"
            return ""
        
        model_list.select(
            fn=on_model_select,
            inputs=model_type,
            outputs=selected_model
        )
        
        load_btn.click(
            fn=load_model,
            inputs=selected_model,
            outputs=model_status
        )
        
        # 删除后按当前模型类型刷新列表
        def _delete_and_refresh(folder_name: str, which: str):
            status, _ = delete_model(folder_name)
            return status, _list_model_paths(which)

        delete_btn.click(
            fn=_delete_and_refresh,
            inputs=[selected_model, model_type],
            outputs=[model_status, model_list]
        )

        convert_btn.click(
            fn=convert_checkpoint_to_pt,
            inputs=selected_model,
            outputs=model_status
        )
        
        # 监听模型类型变化，自动调整batch_size限制和精度选项
        def update_model_constraints(model_type_val):
            batch_update = update_batch_size_constraints(model_type_val)
            precision_updates = update_precision_options(model_type_val)
            out_dir_value = "checkpoints/training_llm" if model_type_val == "llm" else "checkpoints/training_flow"
            ckpt_value = "jzx-ai-lab/HydraVox-CV3/llm.pt" if model_type_val == "llm" else "jzx-ai-lab/HydraVox-CV3/flow.pt"
            return (batch_update,) + precision_updates + (gr.update(value=out_dir_value), gr.update(value=ckpt_value))
        
        model_type.change(
            fn=update_model_constraints,
            inputs=model_type,
            outputs=[batch_size, precision_choice, precision_info, output_dir, model_checkpoint]
        )

        # 刷新设备检测
        detect_btn.click(
            fn=_refresh_device_triplet,
            inputs=[],
            outputs=[device_info, gpu_processes, device_choice]
        )
       
