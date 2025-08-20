import os, gradio as gr
import pandas as pd
from typing import List, Tuple
import json

def upload_audio_files(files):
    """处理上传的音频文件"""
    if not files:
        gr.Warning("请选择音频文件")
        return "未选择文件", []
    
    file_info = []
    for file in files:
        if file:
            file_info.append({
                "文件名": os.path.basename(file.name),
                "大小": f"{os.path.getsize(file.name) / 1024:.1f} KB",
                "路径": file.name
            })
    
    df = pd.DataFrame(file_info)
    return f"已上传 {len(files)} 个音频文件", df

def process_text_annotation(audio_files, text_content: str):
    """处理文本标注"""
    if not audio_files:
        gr.Warning("请先上传音频文件")
        return "请先上传音频文件"
    
    if not text_content.strip():
        gr.Warning("请输入标注文本")
        return "请输入标注文本"
    
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

def validate_dataset(dataset_df):
    """验证数据集质量"""
    if dataset_df is None or len(dataset_df) == 0:
        return "数据集为空"
    
    issues = []
    
    # 检查文本长度
    for idx, row in dataset_df.iterrows():
        text = str(row.get('文本', ''))
        if len(text) < 5:
            issues.append(f"第{idx+1}行: 文本过短")
        elif len(text) > 200:
            issues.append(f"第{idx+1}行: 文本过长")
    
    if not issues:
        return "✅ 数据集验证通过，无问题发现"
    else:
        return f"⚠️ 发现 {len(issues)} 个问题:\n" + "\n".join(issues[:10])

def export_dataset(dataset_df, format_type: str):
    """导出数据集"""
    if dataset_df is None or len(dataset_df) == 0:
        gr.Warning("没有可导出的数据")
        return None
    
    if format_type == "CSV":
        output_path = "/tmp/dataset.csv"
        dataset_df.to_csv(output_path, index=False)
    elif format_type == "JSON":
        output_path = "/tmp/dataset.json"
        dataset_df.to_json(output_path, orient='records', ensure_ascii=False, indent=2)
    else:
        gr.Warning("不支持的格式")
        return None
    
    return output_path

def create_data_tab():
    """创建数据处理tab界面"""
    with gr.Tab("📊 数据处理"):
        gr.Markdown("### 数据集制作与处理")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 音频上传区域
                gr.Markdown("#### 1. 音频文件上传")
                audio_files = gr.File(
                    label="选择音频文件",
                    file_count="multiple",
                    file_types=[".wav", ".mp3", ".flac", ".m4a"]
                )
                upload_btn = gr.Button("📁 上传音频", variant="primary")
                upload_status = gr.Textbox(label="上传状态", interactive=False)
                
                # 文本标注区域
                gr.Markdown("#### 2. 文本标注")
                text_annotation = gr.Textbox(
                    label="文本标注（每行对应一个音频）",
                    placeholder="第一个音频的文本\n第二个音频的文本\n...",
                    lines=8
                )
                annotate_btn = gr.Button("✏️ 生成标注", variant="secondary")
                
            with gr.Column(scale=2):
                # 文件列表
                gr.Markdown("#### 音频文件列表")
                file_list = gr.Dataframe(
                    headers=["文件名", "大小", "路径"],
                    label="已上传文件",
                    interactive=False
                )
                
                # 标注结果
                gr.Markdown("#### 标注结果")
                annotation_result = gr.Dataframe(
                    headers=["音频ID", "文本", "状态"],
                    label="标注数据",
                    interactive=True
                )
        
        # 数据处理工具
        gr.Markdown("### 数据集处理工具")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 数据验证")
                validate_btn = gr.Button("🔍 验证数据集", variant="secondary")
                validation_result = gr.Textbox(
                    label="验证结果",
                    lines=5,
                    interactive=False
                )
                
            with gr.Column():
                gr.Markdown("#### 数据导出")
                export_format = gr.Dropdown(
                    choices=["CSV", "JSON"],
                    value="JSON",
                    label="导出格式"
                )
                export_btn = gr.Button("💾 导出数据集", variant="primary")
                export_file = gr.File(label="下载数据集")
        
        # 数据统计
        gr.Markdown("### 数据集统计")
        with gr.Row():
            total_count = gr.Number(label="总样本数", interactive=False)
            avg_length = gr.Number(label="平均文本长度", interactive=False)
            unique_chars = gr.Number(label="唯一字符数", interactive=False)
        
        # 事件绑定
        upload_btn.click(
            fn=upload_audio_files,
            inputs=[audio_files],
            outputs=[upload_status, file_list]
        )
        
        annotate_btn.click(
            fn=process_text_annotation,
            inputs=[audio_files, text_annotation],
            outputs=annotation_result
        )
        
        validate_btn.click(
            fn=validate_dataset,
            inputs=[annotation_result],
            outputs=validation_result
        )
        
        export_btn.click(
            fn=export_dataset,
            inputs=[annotation_result, export_format],
            outputs=export_file
        ) 