import os
import subprocess
import time
import json
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TrainingManager:
    """训练管理器，负责启动、监控和管理训练进程"""
    
    def __init__(self, log_dir: str = "logs/training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """启动训练任务
        
        Args:
            config: 训练配置参数
            
        Returns:
            训练任务信息
        """
        training_id = f"train_{int(time.time())}"
        log_file = self.log_dir / f"{training_id}.log"
        
        try:
            # 构建训练命令
            cmd = self._build_training_command(config)
            
            # 启动训练进程，设置新的进程组
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=open(log_file, 'w', encoding='utf-8'),
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent,  # 项目根目录
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            with self.lock:
                self.active_trainings[training_id] = {
                    'process': process,
                    'log_file': str(log_file),
                    'status': 'running',
                    'start_time': time.time(),
                    'config': config,
                    'pid': process.pid,
                    'last_log_position': 0
                }
            
            logger.info(f"Started training {training_id} with PID {process.pid}")
            
            return {
                'training_id': training_id,
                'status': 'running',
                'pid': process.pid,
                'log_file': str(log_file),
                'message': '训练已启动'
            }
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return {
                'training_id': None,
                'status': 'failed',
                'error': str(e),
                'message': f'训练启动失败: {e}'
            }
    
    def _build_training_command(self, config: Dict[str, Any]) -> str:
        """构建训练命令"""
        logger.info("=" * 60)
        logger.info("🔨 开始构建训练命令")
        logger.info("接收到的config参数:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        logger.info("-" * 60)
        
        # 使用accelerate启动训练，指定配置文件
        accelerate_config_path = "configs/accelerate_default_config.yaml"
        base_cmd = f"accelerate launch --config_file {accelerate_config_path} scripts/train/train_speech_model.py"
        
        logger.info(f"🚀 使用accelerate启动训练")
        logger.info(f"📋 accelerate配置文件: {accelerate_config_path}")
        
        # 检查配置文件是否存在
        config_full_path = os.path.join(os.getcwd(), accelerate_config_path)
        if not os.path.exists(config_full_path):
            logger.warning(f"⚠️ accelerate配置文件不存在: {config_full_path}")
        else:
            logger.info(f"✅ accelerate配置文件存在: {config_full_path}")
        
        # 必需参数 - 移除config参数，让脚本自动使用默认路径
        required_params = [
            f"--model {config.get('model_type', 'llm')}",
            f"--model_ckpt {config.get('model_checkpoint', 'jzx-ai-lab/HydraVox/llm.pt')}",
            f"--tokenizer_path {config.get('tokenizer_path', 'jzx-ai-lab/HydraVox/speech_tokenizer')}",
            f"--train_data {config.get('train_data', '')}",
            f"--output_dir {config.get('output_dir', 'checkpoints/training')}"
        ]
        
        # 可选参数
        optional_params = []
        
        if config.get('cv_data'):
            optional_params.append(f"--cv_data {config['cv_data']}")
        
        if config.get('auto_val_split', False):
            optional_params.append("--auto_val_split")
            optional_params.append(f"--val_split_ratio {config.get('val_split_ratio', 0.05)}")
        
        # 训练参数 - 验证batch_size自动与训练batch_size一致
        batch_size = config.get('batch_size', 4)
        optional_params.extend([
            f"--per_device_train_batch_size {batch_size}",
            f"--per_device_eval_batch_size {batch_size}",  # 验证batch_size与训练一致
            f"--learning_rate {config.get('learning_rate', 1e-4)}",
            f"--num_train_epochs {config.get('epochs', 10)}",
            f"--gradient_accumulation_steps {config.get('gradient_accumulation_steps', 1)}",
            f"--logging_steps {config.get('logging_steps', 50)}",
            f"--eval_steps {config.get('eval_steps', 1000)}",
            f"--save_steps {config.get('save_steps', 2000)}",
            f"--dataloader_num_workers {config.get('dataloader_num_workers', 8)}"
        ])
        
        # 精度设置 - 根据模型类型选择合适的精度
        model_type = config.get('model_type', 'llm')
        if config.get('use_fp16', False):
            optional_params.append("--fp16")
        elif config.get('use_bf16', False):
            optional_params.append("--bf16")
        else:
            # 如果用户没有指定精度，根据模型类型选择默认精度
            if model_type == 'llm':
                optional_params.append("--bf16")  # LLM推荐BF16
            elif model_type == 'flow':
                optional_params.append("--fp16")  # Flow推荐FP16
        
        # LoRA设置
        if config.get('enable_lora', False):
            optional_params.extend([
                "--enable_lora",
                f"--lora_r {config.get('lora_r', 64)}",
                f"--lora_alpha {config.get('lora_alpha', 128)}",
                f"--lora_dropout {config.get('lora_dropout', 0.05)}"
            ])
        
        # DeepSpeed设置
        if config.get('deepspeed_config'):
            optional_params.append(f"--deepspeed {config['deepspeed_config']}")
        
        all_params = required_params + optional_params
        final_command = f"{base_cmd} {' '.join(all_params)}"
        
        logger.info("构建的训练命令:")
        logger.info(f"  {final_command}")
        logger.info("=" * 60)
        
        return final_command
    
    def stop_training(self, training_id: str) -> Dict[str, Any]:
        """停止训练任务"""
        with self.lock:
            if training_id not in self.active_trainings:
                return {
                    'success': False,
                    'message': f'训练任务 {training_id} 不存在'
                }
            
            training = self.active_trainings[training_id]
            process = training['process']
            pid = training['pid']
            
            try:
                if process.poll() is None:  # 进程仍在运行
                    # 首先尝试优雅地终止整个进程组
                    try:
                        import signal
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        logger.info(f"Sent SIGTERM to process group of PID {pid}")
                    except Exception as e:
                        logger.warning(f"Failed to kill process group: {e}, trying individual process")
                        process.terminate()
                    
                    # 等待进程结束，如果5秒后仍未结束则强制杀死
                    try:
                        process.wait(timeout=5)
                        logger.info(f"Training process {pid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Training process {pid} did not terminate, force killing")
                        try:
                            # 强制杀死进程组
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                            logger.info(f"Sent SIGKILL to process group of PID {pid}")
                        except Exception as e:
                            logger.warning(f"Failed to force kill process group: {e}")
                            process.kill()
                        process.wait()
                
                training['status'] = 'stopped'
                training['end_time'] = time.time()
                
                logger.info(f"Stopped training {training_id}")
                
                return {
                    'success': True,
                    'message': f'训练任务 {training_id} 已停止'
                }
                
            except Exception as e:
                logger.error(f"Failed to stop training {training_id}: {e}")
                return {
                    'success': False,
                    'message': f'停止训练失败: {e}'
                }
    
    def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """获取训练状态"""
        with self.lock:
            if training_id not in self.active_trainings:
                return None
            
            training = self.active_trainings[training_id]
            process = training['process']
            
            # 更新进程状态
            if process.poll() is not None and training['status'] == 'running':
                if process.returncode == 0:
                    training['status'] = 'completed'
                else:
                    training['status'] = 'failed'
                training['end_time'] = time.time()
            
            # 读取最新日志
            logs, new_position = self._read_log_tail(
                training['log_file'], 
                training['last_log_position']
            )
            training['last_log_position'] = new_position
            
            return {
                'training_id': training_id,
                'status': training['status'],
                'pid': training['pid'],
                'start_time': training['start_time'],
                'end_time': training.get('end_time'),
                'config': training['config'],
                'logs': logs,
                'log_file': training['log_file'],  # 添加日志文件路径
                'return_code': process.returncode if process.poll() is not None else None
            }
    
    def _read_log_tail(self, log_file: str, last_position: int = 0) -> tuple[List[str], int]:
        """读取日志文件的新增内容"""
        try:
            if not os.path.exists(log_file):
                return [], last_position
            
            with open(log_file, 'r', encoding='utf-8') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                new_position = f.tell()
                
            return new_lines, new_position
            
        except Exception as e:
            logger.error(f"Failed to read log file {log_file}: {e}")
            return [], last_position
    
    def get_all_trainings(self) -> List[Dict[str, Any]]:
        """获取所有训练任务状态"""
        with self.lock:
            result = []
            for training_id, training in self.active_trainings.items():
                status = self.get_training_status(training_id)
                if status:
                    result.append(status)
            return result
    
    def delete_training(self, training_id: str) -> Dict[str, Any]:
        """删除训练任务记录"""
        with self.lock:
            if training_id not in self.active_trainings:
                return {
                    'success': False,
                    'message': f'训练任务 {training_id} 不存在'
                }
            
            training = self.active_trainings[training_id]
            
            # 如果训练仍在运行，先停止
            if training['status'] == 'running':
                stop_result = self.stop_training(training_id)
                if not stop_result['success']:
                    return stop_result
            
            # 删除日志文件
            try:
                log_file = Path(training['log_file'])
                if log_file.exists():
                    log_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete log file: {e}")
            
            # 从活动训练中移除
            del self.active_trainings[training_id]
            
            return {
                'success': True,
                'message': f'训练任务 {training_id} 已删除'
            }

# 全局训练管理器实例
training_manager = TrainingManager()