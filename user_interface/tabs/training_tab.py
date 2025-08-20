import os, gradio as gr
import json
import time
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

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

def start_training(dataset_path: str, config: Dict[str, Any]):
    """启动训练任务"""
    if not dataset_path:
        gr.Warning("请先选择数据集")
        return "请先选择数据集"
    
    # 模拟训练启动
    return f"✅ 训练任务已启动\n数据集: {dataset_path}\n配置: {json.dumps(config, indent=2, ensure_ascii=False)}"

def stop_training():
    """停止训练"""
    return "🛑 训练已停止"

def generate_training_plot():
    """生成训练曲线图"""
    # 模拟训练数据
    epochs = np.arange(1, 51)
    train_loss = 2.0 * np.exp(-epochs/20) + 0.1 + 0.05 * np.random.randn(50)
    val_loss = 2.2 * np.exp(-epochs/22) + 0.15 + 0.08 * np.random.randn(50)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "/tmp/training_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def get_model_list():
    """获取模型列表"""
    # 模拟模型列表
    models = [
        {"名称": "model_epoch_10.pth", "大小": "245.6 MB", "时间": "2024-01-15 10:30"},
        {"名称": "model_epoch_20.pth", "大小": "245.8 MB", "时间": "2024-01-15 11:45"},
        {"名称": "model_epoch_30.pth", "大小": "246.1 MB", "时间": "2024-01-15 13:00"},
        {"名称": "best_model.pth", "大小": "245.9 MB", "时间": "2024-01-15 13:15"},
    ]
    import pandas as pd
    return pd.DataFrame(models)

def load_model(model_name: str):
    """加载模型"""
    if not model_name:
        gr.Warning("请选择模型")
        return "请选择模型"
    
    return f"✅ 模型 {model_name} 加载成功"

def delete_model(model_name: str):
    """删除模型"""
    if not model_name:
        gr.Warning("请选择要删除的模型")
        return "请选择要删除的模型", get_model_list()
    
    return f"🗑️ 模型 {model_name} 已删除", get_model_list()

def create_training_tab():
    """创建训练tab界面"""
    with gr.Tab("🚀 模型训练"):
        gr.Markdown("### TTS 模型训练")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 数据集选择
                gr.Markdown("#### 1. 数据集配置")
                dataset_file = gr.File(
                    label="选择数据集文件",
                    file_types=[".json", ".csv"]
                )
                
                # 训练参数配置
                gr.Markdown("#### 2. 训练参数")
                with gr.Group():
                    batch_size = gr.Slider(1, 128, value=32, step=1, label="批次大小")
                    learning_rate = gr.Number(value=0.001, label="学习率")
                    epochs = gr.Slider(1, 1000, value=100, step=1, label="训练轮数")
                    save_interval = gr.Slider(1, 50, value=10, step=1, label="保存间隔")
                
                with gr.Group():
                    optimizer = gr.Dropdown(
                        choices=["Adam", "SGD", "AdamW"],
                        value="Adam",
                        label="优化器"
                    )
                    scheduler = gr.Dropdown(
                        choices=["CosineAnnealingLR", "StepLR", "ExponentialLR"],
                        value="CosineAnnealingLR",
                        label="学习率调度器"
                    )
                    validation_split = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="验证集比例")
                
                # 控制按钮
                gr.Markdown("#### 3. 训练控制")
                start_btn = gr.Button("🚀 开始训练", variant="primary")
                stop_btn = gr.Button("🛑 停止训练", variant="stop")
                save_config_btn = gr.Button("💾 保存配置", variant="secondary")
                
            with gr.Column(scale=2):
                # 训练状态
                gr.Markdown("#### 训练状态")
                training_status = gr.Textbox(
                    label="训练日志",
                    lines=8,
                    interactive=False,
                    value="等待开始训练..."
                )
                
                # 训练曲线
                gr.Markdown("#### 训练曲线")
                training_plot = gr.Image(label="Loss 曲线")
                refresh_plot_btn = gr.Button("🔄 刷新图表", variant="secondary")
        
        # 模型管理
        gr.Markdown("### 模型管理")
        with gr.Row():
            with gr.Column(scale=2):
                model_list = gr.Dataframe(
                    value=get_model_list(),
                    headers=["名称", "大小", "时间"],
                    label="已保存模型",
                    interactive=False
                )
                
            with gr.Column(scale=1):
                gr.Markdown("#### 模型操作")
                selected_model = gr.Textbox(label="选择的模型", placeholder="点击表格行选择模型")
                
                with gr.Row():
                    load_btn = gr.Button("📂 加载模型", variant="primary")
                    delete_btn = gr.Button("🗑️ 删除模型", variant="stop")
                
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
        
        start_btn.click(
            fn=lambda dataset, bs, lr, ep, si, vs, opt, sch: start_training(
                dataset.name if dataset else "",
                {
                    "batch_size": bs, "learning_rate": lr, "epochs": ep,
                    "save_interval": si, "validation_split": vs,
                    "optimizer": opt, "scheduler": sch
                }
            ),
            inputs=[dataset_file, batch_size, learning_rate, epochs, 
                   save_interval, validation_split, optimizer, scheduler],
            outputs=training_status
        )
        
        stop_btn.click(
            fn=stop_training,
            outputs=training_status
        )
        
        save_config_btn.click(
            fn=lambda bs, lr, ep, si, vs, opt, sch: save_training_config({
                "batch_size": bs, "learning_rate": lr, "epochs": ep,
                "save_interval": si, "validation_split": vs,
                "optimizer": opt, "scheduler": sch
            }),
            inputs=[batch_size, learning_rate, epochs, save_interval, 
                   validation_split, optimizer, scheduler],
            outputs=training_status
        )
        
        refresh_plot_btn.click(
            fn=generate_training_plot,
            outputs=training_plot
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