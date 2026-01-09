import os
import types
from dataclasses import dataclass
from functools import wraps
from typing import Dict, Any

_LANG = os.getenv("HYDRAVOX_UI_LANG", "en").lower()
if _LANG not in ("zh", "en"):
    _LANG = "zh"

_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # main_ui.py
    "Multi-Head AR TTS (HTTP)": {"en": "Multi-Head AR TTS (HTTP)"},
    "### Multi-Head AR TTS · HTTP Only · Gradio Frontend": {
        "en": "### Multi-Head AR TTS · HTTP Only · Gradio Frontend"
    },
    "Text": {"en": "Text"},
    "你好，这是一个基于 HTTP 的 TTS 演示。": {
        "en": "Hello, this is an HTTP-based TTS demo."
    },
    "Speaker": {"en": "Speaker"},
    "Audio": {"en": "Audio"},
    "Synthesize": {"en": "Synthesize"},
    "Backend: `{backend}`": {"en": "Backend: `{backend}`"},
    "HydraVox TTS System": {"en": "HydraVox TTS System"},
    "支持多Token预测的语音合成系统": {
        "en": "A TTS system with multi-token prediction support"
    },
    "🔗 后端服务: {backend}": {"en": "🔗 Backend: {backend}"},
    "💡 HydraVox - 让语音合成更简单": {
        "en": "💡 HydraVox - Make TTS simpler"
    },
    "CUDA 不可用，默认 CPU x1": {"en": "CUDA unavailable, defaulting to CPU x1"},
    "CUDA 可用，GPU 数: {count}": {"en": "CUDA available, GPU count: {count}"},
    "🎤 语音合成": {"en": "🎤 Speech synthesis"},
    "语言": {"en": "Language"},
    "语言已更新。": {"en": "Language updated."},
    # inference_tab.py
    "可用说话人：{count} 个": {"en": "Available speakers: {count}"},
    "请选择 LLM 与 Flow 权重文件后再加载。": {
        "en": "Please select LLM and Flow weights before loading."
    },
    "❗ 请先选择 LLM 与 Flow 权重文件。": {
        "en": "❗ Please select LLM and Flow weights first."
    },
    "模型权重加载成功": {"en": "Model weights loaded successfully"},
    "✅ 加载成功\nLLM: {llm}\nFlow: {flow}\n消息: {msg}": {
        "en": "✅ Loaded\nLLM: {llm}\nFlow: {flow}\nMessage: {msg}"
    },
    "加载失败: {error}": {"en": "Load failed: {error}"},
    "❌ 加载失败: {error}": {"en": "❌ Load failed: {error}"},
    "TTS 语音合成": {"en": "TTS Synthesis"},
    "即时文本转语音 · 支持多说话人": {
        "en": "Instant text-to-speech · Multi-speaker"
    },
    "LLM 权重 (llm.pt)": {"en": "LLM weights (llm.pt)"},
    "Flow 权重 (flow.pt)": {"en": "Flow weights (flow.pt)"},
    "🔄 加载模型": {"en": "🔄 Load model"},
    "输入文本": {"en": "Input text"},
    "请输入要合成的文本...": {"en": "Enter text to synthesize..."},
    "今天天气很好，适合出去走走。": {
        "en": "The weather is great today. Perfect for a walk."
    },
    "欢迎使用 HydraVox,多头预测让语音更自然。": {
        "en": "Welcome to HydraVox. Multi-head prediction makes speech more natural."
    },
    "请在提示框中输入你想要合成的文本内容。": {
        "en": "Type the text you want to synthesize."
    },
    "示例": {"en": "Examples"},
    "预设说话人": {"en": "Preset speaker"},
    "Zero-shot": {"en": "Zero-shot"},
    "合成模式": {"en": "Synthesis mode"},
    "*选择使用预设说话人或Zero-shot语音克隆*": {
        "en": "*Choose a preset speaker or zero-shot voice cloning*"
    },
    "选择预训练的发音人": {"en": "Choose a pre-trained speaker"},
    "↻ 刷新": {"en": "↻ Refresh"},
    "Zero-shot 语音克隆": {"en": "Zero-shot voice cloning"},
    "选择或上传参考音频进行语音克隆": {
        "en": "Select or upload a reference audio for cloning"
    },
    "预设参考音频": {"en": "Preset reference audio"},
    "*选择一个预设的参考音频，或者上传自己的音频文件*": {
        "en": "*Choose a preset reference or upload your own audio*"
    },
    "参考音频对应文本 (ASR内容)": {
        "en": "Reference transcript (ASR content)"
    },
    "请输入参考音频中说话人说的内容...": {
        "en": "Enter what the speaker says in the reference audio..."
    },
    "*请准确输入参考音频中的文字内容，这将用于语音克隆*": {
        "en": "*Please enter the exact transcript; it is used for cloning*"
    },
    "参考音频": {"en": "Reference audio"},
    "*你可以从上方选择预设音频，或者直接上传自己的音频文件*": {
        "en": "*Choose a preset above or upload your own audio*"
    },
    "高级设置": {"en": "Advanced settings"},
    "🎵 合成": {"en": "🎵 Synthesize"},
    "🧹 清空": {"en": "🧹 Clear"},
    "合成音频": {"en": "Synthesized audio"},
    "**后端地址**: `{backend}`": {"en": "**Backend**: `{backend}`"},
    # data_tab.py
    "📊 数据处理": {"en": "📊 Data processing"},
    "# 🛠️ 音频数据预处理工作流": {
        "en": "# 🛠️ Audio data preprocessing workflow"
    },
    "**三个阶段的处理流程：** 格式转换 → VAD分段 → ASR转录": {
        "en": "**Three stages:** Format conversion → VAD → ASR transcription"
    },
    "**可选阶段：** 数据集合并": {"en": "**Optional stage:** Dataset merge"},
    "🎵 阶段 1 - 格式转换与重采样": {
        "en": "🎵 Stage 1 - Format conversion & resampling"
    },
    "**功能：** 将各种音频/视频格式统一转换为 16kHz WAV 格式": {
        "en": "**Function:** Convert audio/video formats to 16kHz WAV"
    },
    "📁 输入目录": {"en": "📁 Input directory"},
    "/path/to/input_dir": {"en": "/path/to/input_dir"},
    "🔄 自动同步输出路径": {"en": "🔄 Auto-sync output path"},
    "添加_resample后缀": {"en": "Add _resample suffix"},
    "📂 输出目录": {"en": "📂 Output directory"},
    "自动同步或手动填写": {"en": "Auto-sync or enter manually"},
    "🎤 采样率 (Hz)": {"en": "🎤 Sample rate (Hz)"},
    "⚠️ 覆盖已存在文件": {"en": "⚠️ Overwrite existing files"},
    "👀 预览变更": {"en": "👀 Preview changes"},
    "▶️ 开始处理": {"en": "▶️ Start"},
    "📋 映射预览（前50条）": {"en": "📋 Mapping preview (first 50)"},
    "源文件": {"en": "Source file"},
    "目标文件": {"en": "Target file"},
    "📊 待处理文件数": {"en": "📊 Files to process"},
    "📈 进度 (%)": {"en": "📈 Progress (%)"},
    "📋 状态": {"en": "📋 Status"},
    "📝 运行日志": {"en": "📝 Logs"},
    "🔊 阶段 2 - VAD 语音活动检测": {
        "en": "🔊 Stage 2 - VAD speech activity detection"
    },
    "**功能：** 使用 Silero VAD 检测并分割语音片段，去除静音部分": {
        "en": "**Function:** Use Silero VAD to segment speech and remove silence"
    },
    "默认衔接阶段1输出": {"en": "Default to Stage 1 output"},
    "添加_vad后缀": {"en": "Add _vad suffix"},
    "⚙️ VAD 参数设置": {"en": "⚙️ VAD settings"},
    "🎯 置信度阈值": {"en": "🎯 Confidence threshold"},
    "越高越严格": {"en": "Higher is stricter"},
    "🗣️ 最短语音 (ms)": {"en": "🗣️ Min speech (ms)"},
    "🔇 最短静音 (ms)": {"en": "🔇 Min silence (ms)"},
    "🔧 前后填充 (ms)": {"en": "🔧 Padding (ms)"},
    "⏱️ 最短片段 (s)": {"en": "⏱️ Min segment (s)"},
    "⏰ 最长片段 (s)": {"en": "⏰ Max segment (s)"},
    "👀 预览": {"en": "👀 Preview"},
    "🎙️ 阶段 3 - ASR 语音识别转录": {
        "en": "🎙️ Stage 3 - ASR transcription"
    },
    "**功能：** 使用语音识别技术将音频转换为文本，生成训练数据集": {
        "en": "**Function:** Transcribe audio to text to build training data"
    },
    "默认衔接阶段2输出": {"en": "Default to Stage 2 output"},
    "添加_asr后缀": {"en": "Add _asr suffix"},
    "⚙️ 计算资源设置": {"en": "⚙️ Compute settings"},
    "自动": {"en": "Auto"},
    "💻 计算设备": {"en": "💻 Device"},
    "🔄 并行进程数": {"en": "🔄 Parallel processes"},
    "🔄 刷新设备检测": {"en": "🔄 Refresh device detection"},
    "ℹ️ 设备检测信息": {"en": "ℹ️ Device info"},
    "🧩 阶段 4 - 数据集合并 (可选)": {
        "en": "🧩 Stage 4 - Dataset merge (optional)"
    },
    "**功能：** 将多个前面阶段生成的数据集目录合并为一个新的 HuggingFace 数据集。输入多个目录时使用英文逗号分隔。": {
        "en": "**Function:** Merge datasets from previous stages into a new HuggingFace dataset. Separate paths with commas."
    },
    "📁 输入数据集目录（逗号分隔）": {
        "en": "📁 Input dataset directories (comma-separated)"
    },
    "/path/to/ds1,/path/to/ds2,...": {"en": "/path/to/ds1,/path/to/ds2,..."},
    "📂 合并输出目录": {"en": "📂 Merge output directory"},
    "/path/to/merged_dataset": {"en": "/path/to/merged_dataset"},
    "▶️ 开始合并": {"en": "▶️ Merge"},
    "📝 合并日志": {"en": "📝 Merge logs"},
    "## 💡 使用提示": {"en": "## 💡 Tips"},
    "- **阶段顺序不可颠倒**：每个阶段都依赖前一阶段的输出": {
        "en": "- **Do not change the order**: each stage depends on the previous output"
    },
    "- **GPU 加速**：阶段3支持GPU加速，可显著提升处理速度": {
        "en": "- **GPU acceleration**: Stage 3 can be significantly faster"
    },
    "- **监控进度**：每个阶段都有实时进度显示和详细日志": {
        "en": "- **Monitor progress**: each stage shows progress and logs"
    },
    "- **可选合并**：阶段4可将多个阶段产出的数据集进行合并": {
        "en": "- **Optional merge**: Stage 4 combines datasets from earlier stages"
    },
    "⚠️ **注意**：处理大量文件时请确保有足够的磁盘空间和计算资源": {
        "en": "⚠️ **Note**: Ensure enough disk space and compute for large batches"
    },
    # training_tab.py
    "BF16（推荐）": {"en": "BF16 (recommended)"},
    "FP16（推荐）": {"en": "FP16 (recommended)"},
    "💡 **LLM模型**: 推荐使用BF16精度以获得更好的数值稳定性": {
        "en": "💡 **LLM model**: BF16 is recommended for better numerical stability"
    },
    "💡 **Flow模型**: 推荐使用FP16精度以节省显存和提升速度": {
        "en": "💡 **Flow model**: FP16 is recommended to save VRAM and improve speed"
    },
    "🚀 模型训练": {"en": "🚀 Model training"},
    "### TTS 模型训练": {"en": "### TTS Model Training"},
    "#### 1. 数据集配置": {"en": "#### 1. Dataset"},
    "训练数据路径": {"en": "Training data path"},
    "输入训练数据路径，如: data/processed/train_ds": {
        "en": "Enter training data path, e.g. data/processed/train_ds"
    },
    "#### 2. 模型配置": {"en": "#### 2. Model"},
    "模型类型": {"en": "Model type"},
    "模型检查点路径": {"en": "Model checkpoint path"},
    "预训练模型路径": {"en": "Pretrained model path"},
    "分词器路径": {"en": "Tokenizer path"},
    "分词器模型路径": {"en": "Tokenizer model path"},
    "输出目录": {"en": "Output directory"},
    "训练输出保存目录": {"en": "Training output directory"},
    "#### 3. 训练参数": {"en": "#### 3. Training params"},
    "批次大小": {"en": "Batch size"},
    "学习率": {"en": "Learning rate"},
    "训练轮数": {"en": "Epochs"},
    "保存间隔(步数)": {"en": "Save interval (steps)"},
    "日志记录间隔(步数)": {"en": "Log interval (steps)"},
    "评估间隔(步数)": {"en": "Eval interval (steps)"},
    "验证集比例": {"en": "Validation split"},
    "自动划分验证集": {"en": "Auto split validation"},
    "#### 4. 高级选项": {"en": "#### 4. Advanced"},
    "启用LoRA微调": {"en": "Enable LoRA fine-tuning"},
    "精度设置": {"en": "Precision"},
    "#### 5. 计算资源设置": {"en": "#### 5. Compute"},
    "🔄 并行进程数 (GPU数)": {"en": "🔄 Parallel processes (GPU count)"},
    "🎯 GPU IDs (可选)": {"en": "🎯 GPU IDs (optional)"},
    "例如: 0,1": {"en": "e.g. 0,1"},
    "#### 6. 训练控制": {"en": "#### 6. Controls"},
    "🚀 开始训练": {"en": "🚀 Start training"},
    "🛑 停止训练": {"en": "🛑 Stop training"},
    "🔄 刷新日志": {"en": "🔄 Refresh logs"},
    "#### 训练状态与日志": {"en": "#### Training status & logs"},
    "训练日志": {"en": "Training logs"},
    "等待开始训练...": {"en": "Waiting to start training..."},
    "正在启动训练...": {"en": "Starting training..."},
    "#### 训练曲线": {"en": "#### Training curves"},
    "训练指标曲线": {"en": "Training metrics plot"},
    "**图表设置**": {"en": "**Chart settings**"},
    "自动刷新图表": {"en": "Auto refresh charts"},
    "刷新间隔(秒)": {"en": "Refresh interval (s)"},
    "🔄 立即刷新": {"en": "🔄 Refresh now"},
    "⚡ 强制刷新": {"en": "⚡ Force refresh"},
    "**💾 图表存储位置**": {"en": "**💾 Chart storage location**"},
    "训练图表会实时更新并保存到：": {
        "en": "Training charts are updated and saved to:"
    },
    "### 模型管理": {"en": "### Model management"},
    "路径": {"en": "Path"},
    "训练输出路径": {"en": "Training output paths"},
    "#### 文件夹操作": {"en": "#### Folder actions"},
    "选择的文件夹": {"en": "Selected folder"},
    "点击表格行选择文件夹": {"en": "Click a row to select a folder"},
    "🔄 刷新列表": {"en": "🔄 Refresh list"},
    "📂 加载路径": {"en": "📂 Load path"},
    "🗑️ 删除路径": {"en": "🗑️ Delete path"},
    "🔁 转换为 model.pt (bf16)": {"en": "🔁 Convert to model.pt (bf16)"},
    "操作状态": {"en": "Status"},
    # speaker_manage.py
    "🗣️ 说话人管理": {"en": "🗣️ Speaker management"},
    "# 🗣️ 说话人库管理": {"en": "# 🗣️ Speaker library"},
    "- 预加载/保存路径：`jzx-ai-lab/HydraVox-CV3/spk2info.pt`": {
        "en": "- Preload/save path: `jzx-ai-lab/HydraVox-CV3/spk2info.pt`"
    },
    "- 查看已有 speaker，加载数据集计算 `embedding` 均值，新增/覆盖 speaker": {
        "en": "- View speakers, compute embedding means from datasets, add/overwrite speakers"
    },
    "spk2info.pt 路径": {"en": "spk2info.pt path"},
    "🔄 重新加载": {"en": "🔄 Reload"},
    "💾 保存当前": {"en": "💾 Save current"},
    "现有说话人": {"en": "Existing speakers"},
    "➕ 从数据集新增/覆盖说话人": {
        "en": "➕ Add/overwrite speakers from dataset"
    },
    "数据集路径 (HuggingFace load_from_disk)": {
        "en": "Dataset path (HuggingFace load_from_disk)"
    },
    "📐 计算均值": {"en": "📐 Compute mean"},
    "均值信息": {"en": "Mean info"},
    "Speaker 名称": {"en": "Speaker name"},
    "如：alice": {"en": "e.g. alice"},
    "✅ 新增/覆盖": {"en": "✅ Add/overwrite"},
    "请输入有效的 speaker 名称": {
        "en": "Please enter a valid speaker name"
    },
    "请先计算均值": {"en": "Please compute the mean first"},
    # runtime_messages
    "data.select_audio_files": {"zh": "请选择音频文件", "en": "Please select audio files"},
    "data.no_files_selected": {"zh": "未选择文件", "en": "No files selected"},
    "data.uploaded_files_count": {"zh": "已上传 {count} 个音频文件", "en": "Uploaded {count} audio files"},
    "data.upload_audio_first": {"zh": "请先上传音频文件", "en": "Please upload audio files first"},
    "data.enter_annotation_text": {"zh": "请输入标注文本", "en": "Please enter annotation text"},
    "data.dataset_empty": {"zh": "数据集为空", "en": "Dataset is empty"},
    "data.dataset_valid": {"zh": "✅ 数据集验证通过，无问题发现", "en": "✅ Dataset validated successfully; no issues found"},
    "data.dataset_issues": {
        "zh": "⚠️ 发现 {count} 个问题:\n{issues}",
        "en": "⚠️ Found {count} issues:\n{issues}",
    },
    "data.row_too_short": {"zh": "第{row}行: 文本过短", "en": "Row {row}: text too short"},
    "data.row_too_long": {"zh": "第{row}行: 文本过长", "en": "Row {row}: text too long"},
    "data.no_export_data": {"zh": "没有可导出的数据", "en": "No data to export"},
    "data.unsupported_format": {"zh": "不支持的格式", "en": "Unsupported format"},
    "data.enter_valid_input_dir": {"zh": "请输入有效的输入目录", "en": "Please enter a valid input directory"},
    "data.input_dir_invalid": {"zh": "❗ 输入目录无效", "en": "❗ Invalid input directory"},
    "data.processing_files_output": {
        "zh": "将处理 {count} 个文件，输出至 {output_dir}",
        "en": "Will process {count} files, output to {output_dir}",
    },
    "data.no_media_files": {"zh": "没有可处理的媒体文件", "en": "No media files to process"},
    "data.script_not_found": {"zh": "找不到脚本: {path}", "en": "Script not found: {path}"},
    "data.start_failed": {"zh": "启动失败: {error}", "en": "Failed to start: {error}"},
    "data.in_progress": {
        "zh": "进行中: {done}/{total} ({pct}%) · 用时 {elapsed}s",
        "en": "In progress: {done}/{total} ({pct}%) · elapsed {elapsed}s",
    },
    "data.done": {
        "zh": "✅ 完成: {done}/{total} · 总用时 {elapsed}s",
        "en": "✅ Done: {done}/{total} · total {elapsed}s",
    },
    "data.failed": {
        "zh": "❌ 失败: 已完成 {done}/{total} · 总用时 {elapsed}s",
        "en": "❌ Failed: completed {done}/{total} · total {elapsed}s",
    },
    "data.processing_audio_files_output": {
        "zh": "将处理约 {count} 个音频文件，输出至 {output_dir}",
        "en": "Will process about {count} audio files, output to {output_dir}",
    },
    "data.vad_processing": {"zh": "VAD 处理中...", "en": "VAD processing..."},
    "data.in_progress_simple": {
        "zh": "进行中: {current}/{total} · 用时 {elapsed}s",
        "en": "In progress: {current}/{total} · elapsed {elapsed}s",
    },
    "data.run_exception": {"zh": "❌ 运行异常: {error}", "en": "❌ Runtime error: {error}"},
    "data.stage_done": {"zh": "✅ 完成 · 用时 {elapsed}s", "en": "✅ Done · elapsed {elapsed}s"},
    "data.stage_failed": {"zh": "❌ 失败 · 用时 {elapsed}s", "en": "❌ Failed · elapsed {elapsed}s"},
    "data.asr_summary": {
        "zh": "将转录 {wav_count} 个 .wav 与 {mp3_count} 个 .mp3，输出到 {output_dir}",
        "en": "Will transcribe {wav_count} .wav and {mp3_count} .mp3 files, output to {output_dir}",
    },
    "data.asr_processing": {"zh": "ASR 转录中...", "en": "ASR transcribing..."},
    "data.asr_in_progress": {"zh": "进行中 · 用时 {elapsed}s", "en": "In progress · elapsed {elapsed}s"},
    "data.need_input_dirs_comma": {
        "zh": "❗ 请输入至少一个输入目录，使用逗号分隔",
        "en": "❗ Please enter at least one input directory, separated by commas",
    },
    "data.need_input_dirs": {"zh": "❗ 请输入至少一个输入目录", "en": "❗ Please enter at least one input directory"},
    "data.need_output_dir": {"zh": "❗ 请输入输出目录", "en": "❗ Please enter an output directory"},
    "data.missing_datasets_dep": {
        "zh": "缺少datasets依赖或导入失败: {error}",
        "en": "Missing datasets dependency or import failed: {error}",
    },
    "data.skip_non_dir_dash": {"zh": "- 跳过（非目录）: {path}", "en": "- Skipped (not a directory): {path}"},
    "data.dataset_ok": {"zh": "- ✓ {path} · {count} 条", "en": "- ✓ {path} · {count} items"},
    "data.dataset_load_failed": {"zh": "- ✗ {path} · 加载失败: {error}", "en": "- ✗ {path} · Load failed: {error}"},
    "data.output_dir_missing": {"zh": "(未指定，建议填写保存目录)", "en": "(not set, please provide a save directory)"},
    "data.merge_summary": {
        "zh": "将合并 {ok}/{total} 个可用数据集，总计约 {count} 条",
        "en": "Will merge {ok}/{total} available datasets, about {count} items total",
    },
    "data.output_dir_line": {"zh": "输出目录: {output_dir}", "en": "Output directory: {output_dir}"},
    "data.reading_progress": {"zh": "读取中 ({idx}/{total})", "en": "Reading ({idx}/{total})"},
    "data.no_merge_datasets": {"zh": "❌ 没有可合并的数据集", "en": "❌ No datasets to merge"},
    "data.no_common_columns": {"zh": "❌ 各数据列无交集，无法合并", "en": "❌ No common columns; cannot merge"},
    "data.align_columns": {"zh": "对齐字段", "en": "Aligning columns"},
    "data.align_failed": {"zh": "❌ 对齐列失败: {error}", "en": "❌ Failed to align columns: {error}"},
    "data.merge_in_progress": {"zh": "合并中", "en": "Merging"},
    "data.merge_failed": {"zh": "❌ 合并失败: {error}", "en": "❌ Merge failed: {error}"},
    "data.merge_done": {"zh": "✅ 合并完成 · 共 {count} 条", "en": "✅ Merge completed · {count} items"},
    "data.save_failed": {"zh": "❌ 保存失败: {error}", "en": "❌ Save failed: {error}"},
    "data.skip_non_dir": {"zh": "跳过（非目录）: {path}", "en": "Skipped (not a directory): {path}"},
    "data.no_splits": {"zh": "{path} · 不含可用 split，已跳过", "en": "{path} · No usable split, skipped"},
    "data.read_count": {"zh": "{path} · 读取 {count} 条", "en": "{path} · Read {count} items"},
    "data.load_failed": {"zh": "{path} · 加载失败: {error}", "en": "{path} · Load failed: {error}"},
    "data.columns_intersection": {"zh": "列对齐（交集）: {columns}", "en": "Column intersection: {columns}"},
    "data.merge_completed_log": {"zh": "合并完成，合计 {count} 条", "en": "Merge complete, total {count} items"},
    "data.saved_to": {"zh": "已保存至 {output_dir}", "en": "Saved to {output_dir}"},
    "speaker.load_failed": {"zh": "加载失败: {error}", "en": "Load failed: {error}"},
    "speaker.saved_to": {"zh": "已保存至 {path}", "en": "Saved to {path}"},
    "speaker.save_failed": {"zh": "保存失败: {error}", "en": "Save failed: {error}"},
    "speaker.verify_model_check_failed": {
        "zh": "检查说话人模型失败: {error}",
        "en": "Failed to check speaker model: {error}",
    },
    "speaker.download_failed_fallback": {
        "zh": "下载 {model} 失败，回退到 {fallback}",
        "en": "Failed to download {model}, falling back to {fallback}",
    },
    "speaker.download_failed_online": {
        "zh": "下载说话人模型失败: {error}，回退到在线模式",
        "en": "Failed to download speaker model: {error}, falling back to online mode",
    },
    "speaker.need_dataset_path": {"zh": "请输入数据集路径", "en": "Please enter a dataset path"},
    "speaker.missing_datasets_dep": {
        "zh": "缺少datasets依赖或导入失败: {error}",
        "en": "Missing datasets dependency or import failed: {error}",
    },
    "speaker.load_dataset_failed": {"zh": "加载数据集失败: {error}", "en": "Failed to load dataset: {error}"},
    "speaker.missing_columns": {
        "zh": "数据集中未找到 'embedding' 或 'audio' 列",
        "en": "Dataset missing 'embedding' or 'audio' column",
    },
    "speaker.embedding_dim_mismatch": {
        "zh": "embedding 维度不一致: {left} vs {right}",
        "en": "Embedding dimension mismatch: {left} vs {right}",
    },
    "speaker.no_embedding_from_audio": {
        "zh": "未从音频提取到有效的 embedding",
        "en": "No valid embedding extracted from audio",
    },
    "speaker.no_embedding": {"zh": "未获取到有效的 embedding", "en": "No valid embedding found"},
    "speaker.mean_info": {
        "zh": "样本数: {count}, 维度: {dim}, L2范数: {norm:.6f}",
        "en": "Samples: {count}, dim: {dim}, L2 norm: {norm:.6f}",
    },
    "speaker.mean_info_sampled": {
        "zh": "样本数: {count}, 维度: {dim}, L2范数: {norm:.6f}（已随机抽取 5000 条音频）",
        "en": "Samples: {count}, dim: {dim}, L2 norm: {norm:.6f} (sampled 5000 audios)",
    },
    "speaker.compute_mean_failed": {"zh": "计算均值失败: {error}", "en": "Failed to compute mean: {error}"},
    "speaker.invalid_name": {"zh": "请输入有效的 speaker 名称", "en": "Please enter a valid speaker name"},
    "speaker.compute_first": {"zh": "请先计算均值", "en": "Please compute the mean first"},
    "training.config_saved": {"zh": "配置已保存到: {path}", "en": "Config saved to: {path}"},
    "training.task_running": {
        "zh": "⚠️ 已有训练任务在运行中，请先停止当前训练",
        "en": "⚠️ A training job is already running. Please stop it first.",
    },
    "training.select_dataset": {"zh": "❌ 请先选择数据集文件", "en": "❌ Please select a dataset file first"},
    "training.script_not_found": {"zh": "❌ 找不到训练脚本: {path}", "en": "❌ Training script not found: {path}"},
    "training.start_failed": {"zh": "❌ 启动失败: {error}", "en": "❌ Failed to start: {error}"},
    "training.started": {
        "zh": "✅ 训练任务已启动\n训练ID: {id}\nPID: {pid}\n脚本: {script}",
        "en": "✅ Training started\nID: {id}\nPID: {pid}\nScript: {script}",
    },
    "training.start_failed_detail": {
        "zh": "❌ 训练启动失败: {error}",
        "en": "❌ Training start failed: {error}",
    },
    "training.no_running_task": {"zh": "⚠️ 当前没有运行中的训练任务", "en": "⚠️ No training task is running"},
    "training.stopped": {"zh": "🛑 训练已停止 (退出码: {code})", "en": "🛑 Training stopped (exit code: {code})"},
    "training.stop_failed": {"zh": "❌ 停止训练失败: {error}", "en": "❌ Failed to stop training: {error}"},
    "training.model_list_failed": {"zh": "获取失败", "en": "Failed to fetch"},
    "training.error_prefix": {"zh": "错误: {error}", "en": "Error: {error}"},
    "training.select_model": {"zh": "请选择模型", "en": "Please select a model"},
    "training.model_loaded": {
        "zh": "✅ 模型 {model} 加载成功",
        "en": "✅ Model {model} loaded successfully",
    },
    "training.select_folder_delete": {
        "zh": "⚠️ 请选择要删除的文件夹",
        "en": "⚠️ Please select a folder to delete",
    },
    "training.delete_system_folder": {
        "zh": "⚠️ 不允许删除系统文件夹: {folder}",
        "en": "⚠️ Deleting system folder not allowed: {folder}",
    },
    "training.folder_deleted": {
        "zh": "✅ 文件夹 {folder} 已删除",
        "en": "✅ Folder {folder} deleted",
    },
    "training.folder_not_found": {"zh": "❌ 未找到文件夹: {folder}", "en": "❌ Folder not found: {folder}"},
    "training.delete_failed": {"zh": "❌ 删除失败: {error}", "en": "❌ Delete failed: {error}"},
    "training.select_path_first": {
        "zh": "⚠️ 请先在表格中选择一个路径",
        "en": "⚠️ Please select a path from the table first",
    },
    "training.invalid_path": {"zh": "❌ 路径无效: {path}", "en": "❌ Invalid path: {path}"},
    "training.bin_not_found": {
        "zh": "❌ 未找到 pytorch_model.bin 于: {path}",
        "en": "❌ pytorch_model.bin not found at: {path}",
    },
    "training.sharded_not_supported": {
        "zh": "❌ 暂不支持分片权重（.bin.index.json），请先合并再转换",
        "en": "❌ Sharded weights (.bin.index.json) not supported. Please merge first.",
    },
    "training.state_dict_invalid": {
        "zh": "❌ 权重文件格式不符合预期（非state_dict）",
        "en": "❌ Weight file format invalid (not a state_dict)",
    },
    "training.convert_done": {
        "zh": "✅ 转换完成: {src} → {dst} (bf16)",
        "en": "✅ Converted: {src} → {dst} (bf16)",
    },
    "training.convert_failed": {"zh": "❌ 转换失败: {error}", "en": "❌ Conversion failed: {error}"},
    "training.no_outputs": {"zh": "暂无训练输出", "en": "No training outputs"},
    "training.train_first": {"zh": "请先进行模型训练", "en": "Please run training first"},
    "training.model_files_count": {"zh": "{count}个模型文件", "en": "{count} model files"},
    "training.empty_folder": {"zh": "空文件夹", "en": "Empty folder"},
    "training.more_files_suffix": {"zh": " 等{count}个文件", "en": " and {count} more files"},
    "training.no_task": {"zh": "暂无训练任务", "en": "No training task"},
    "training.status_line": {"zh": "训练状态: {status}", "en": "Training status: {status}"},
    "training.id_line": {"zh": "训练ID: {id}", "en": "Training ID: {id}"},
    "training.start_time_line": {"zh": "开始时间: {time}", "en": "Start time: {time}"},
    "training.end_time_line": {"zh": "结束时间: {time}", "en": "End time: {time}"},
    "training.log_lines_line": {"zh": "日志行数: {count}", "en": "Log lines: {count}"},
    "training.log_omitted": {"zh": "... (省略了前{count}行日志) ...", "en": "... (omitted {count} earlier lines) ..."},
    "training.log_fetch_failed": {"zh": "获取日志失败: {error}", "en": "Failed to fetch logs: {error}"},
    # train_speech_model.py
    "train.tn_load_failed": {"zh": "文本归一化库加载失败", "en": "Failed to load text normalization library"},
    "train.ckpt_format_invalid": {
        "zh": "不支持的 checkpoint 格式：期望为 state_dict 或 {'state_dict': ...}",
        "en": "Unsupported checkpoint format: expected state_dict or {'state_dict': ...}",
    },
    "train.auto_val_disabled": {
        "zh": "自动划分验证集关闭（val_split_ratio <= 0）：仅训练不验证",
        "en": "Auto validation split disabled (val_split_ratio <= 0): train only",
    },
    "train.val_size_zero": {"zh": "验证集大小为 0：仅训练不验证", "en": "Validation size is 0: train only"},
    "train.val_split_too_large": {
        "zh": "val_split_ratio 过大导致验证集大小({val_size}) >= 数据集总量({total})",
        "en": "val_split_ratio too large: val_size ({val_size}) >= total ({total})",
    },
    "train.auto_val_split": {
        "zh": "自动划分验证集: 训练集 {train_size}，验证集 {val_size}",
        "en": "Auto split: train {train_size}, val {val_size}",
    },
    "train.onnx_no_cuda": {
        "zh": "onnxruntime 未检测到 CUDAExecutionProvider（available={providers}），将自动使用 CPUExecutionProvider。",
        "en": "onnxruntime missing CUDAExecutionProvider (available={providers}); using CPUExecutionProvider.",
    },
    "train.sv_check_failed": {"zh": "检查说话人模型失败: {error}", "en": "Speaker model check failed: {error}"},
    "train.sv_missing_download": {
        "zh": "说话人模型不存在，准备下载: {model_path}",
        "en": "Speaker model not found, downloading: {model_path}",
    },
    "train.sv_download_failed_fallback": {
        "zh": "下载 {model_path} 失败，回退到 {fallback_id}",
        "en": "Failed to download {model_path}, fallback to {fallback_id}",
    },
    "train.sv_symlink": {"zh": "创建软链接: {src} -> {dst}", "en": "Created symlink: {src} -> {dst}"},
    "train.sv_copied": {"zh": "已复制模型文件到: {path}", "en": "Copied model files to: {path}"},
    "train.sv_download_failed_online": {
        "zh": "下载说话人模型失败: {error}，回退到在线模式",
        "en": "Failed to download speaker model: {error}, falling back to online mode",
    },
    "train.speech_token_fallback": {
        "zh": "speech_token 提取失败，将使用 batch 内其它样本回退替代（idx={idx}, audio={audio}, err={err_type}: {error}）",
        "en": "speech_token extraction failed; using batch fallback (idx={idx}, audio={audio}, err={err_type}: {error})",
    },
    "train.speech_token_pool_fallback": {
        "zh": "本 batch 所有音频 speech_token 提取均失败，已从历史成功池随机抽取兜底继续训练（pool={pool}, pick_len={pick_len}, err={err_type}: {error}）",
        "en": "All speech_token extractions failed; fallback to pool (pool={pool}, pick_len={pick_len}, err={err_type}: {error})",
    },
    "train.speech_token_placeholder_fallback": {
        "zh": "本 batch 所有音频 speech_token 提取均失败，且历史成功池为空，已使用占位 token 兜底继续训练（fallback_id={fallback_id}, fallback_len={fallback_len}, err={err_type}: {error}）",
        "en": "All speech_token extractions failed and pool empty; using placeholder (fallback_id={fallback_id}, fallback_len={fallback_len}, err={err_type}: {error})",
    },
    "train.text_tokenizer_missing": {
        "zh": "数据只有 text 字段但未提供 tokenizer，无法生成 text_token。",
        "en": "Dataset has text only; tokenizer required to build text_token.",
    },
    "train.llm_text_required": {
        "zh": "LLM 训练需要 text_token 或 text 字段。",
        "en": "LLM training requires text_token or text.",
    },
    "train.llm_audio_required": {
        "zh": "LLM 训练需要 audio 字段以实时提取 speech_token。",
        "en": "LLM training requires audio to extract speech_token.",
    },
    "train.flow_audio_required": {
        "zh": "FLOW 训练需要 audio 字段以提取 speech_feat。",
        "en": "FLOW training requires audio to extract speech_feat.",
    },
    "train.embedding_missing_no_online": {
        "zh": "数据缺少 embedding 且已关闭在线提取（--no_online_embedding）。",
        "en": "Embedding missing and online extraction disabled (--no_online_embedding).",
    },
    "train.cli_model": {"zh": "模型类型", "en": "Model type"},
    "train.cli_config": {"zh": "hyperpyyaml 配置路径", "en": "hyperpyyaml config path"},
    "train.cli_train_data": {"zh": "训练数据路径，逗号分隔", "en": "Training data paths, comma-separated"},
    "train.cli_cv_data": {"zh": "验证数据路径，逗号分隔", "en": "Validation data paths, comma-separated"},
    "train.cli_auto_val": {"zh": "自动划分验证集", "en": "Auto validation split"},
    "train.cli_val_split": {"zh": "验证集比例", "en": "Validation split ratio"},
    "train.cli_output_dir": {"zh": "输出目录", "en": "Output directory"},
    "train.cli_model_ckpt": {"zh": "初始模型 checkpoint", "en": "Initial model checkpoint"},
    "train.cli_resume": {"zh": "Trainer 断点目录", "en": "Trainer checkpoint directory"},
    "train.cli_tokenizer_path": {"zh": "LLM tokenizer/Qwen 路径；flow 可选 onnx 路径", "en": "LLM tokenizer/Qwen path; flow optional onnx"},
    "train.cli_tokenizer_onnx": {"zh": "speech tokenizer ONNX 路径", "en": "Speech tokenizer ONNX path"},
    "train.cli_qwen_pretrain": {"zh": "Qwen2Encoder pretrain_path/tokenizer 路径", "en": "Qwen2Encoder pretrain_path/tokenizer path"},
    "train.cli_onnx_use_cuda": {"zh": "ONNX tokenizer 是否使用 CUDAExecutionProvider", "en": "Use CUDAExecutionProvider for ONNX tokenizer"},
    "train.cli_onnx_device_id": {"zh": "ONNX CUDA device_id（默认取 LOCAL_RANK/RANK，否则 0）", "en": "ONNX CUDA device_id (default LOCAL_RANK/RANK, else 0)"},
    "train.start": {"zh": "🚀 训练脚本启动 (model={model})", "en": "🚀 Training script started (model={model})"},
    "train.llm_pretrain_start": {"zh": "🚀 LLM pretrain 脚本启动", "en": "🚀 LLM pretrain script started"},
    "train.lora_ignored": {
        "zh": "新模型预训练不支持 LoRA 参数，已忽略 --enable_lora 等配置。",
        "en": "LoRA not supported for pretrain; --enable_lora ignored.",
    },
    "train.resume_not_found": {
        "zh": "--resume_from_checkpoint 路径不存在：{path}",
        "en": "--resume_from_checkpoint path not found: {path}",
    },
    "train.resume_not_dir": {
        "zh": "--resume_from_checkpoint 需要传 checkpoint 目录，但得到：{path}",
        "en": "--resume_from_checkpoint must be a checkpoint dir, got: {path}",
    },
    "train.resume_from": {"zh": "将从 Trainer checkpoint 断点续训：{path}", "en": "Resuming from Trainer checkpoint: {path}"},
    "train.model_ckpt_required": {
        "zh": "未指定 --resume_from_checkpoint 时，必须提供 --model_ckpt 作为初始权重。",
        "en": "When --resume_from_checkpoint is empty, --model_ckpt is required.",
    },
    "train.missing_keys": {
        "zh": "load_state_dict missing keys: {count}（示例：{sample}）",
        "en": "load_state_dict missing keys: {count} (e.g. {sample})",
    },
    "train.unexpected_keys": {
        "zh": "load_state_dict unexpected keys: {count}（示例：{sample}）",
        "en": "load_state_dict unexpected keys: {count} (e.g. {sample})",
    },
    # convert_to_wav.py
    "convert.cli_src": {"zh": "原始目录", "en": "Source directory"},
    "convert.cli_dst": {"zh": "输出目录", "en": "Output directory"},
    "convert.cli_sr": {"zh": "目标采样率", "en": "Target sample rate"},
    "convert.cli_overwrite": {"zh": "覆盖已存在文件", "en": "Overwrite existing files"},
    "convert.cli_jobs": {"zh": "并行线程数", "en": "Parallel threads"},
    "convert.no_files": {"zh": "未找到支持的媒体文件。", "en": "No supported media files found."},
    "convert.step_done": {
        "zh": "step 1/5: ✅ 全部完成！已转换 {done}/{total} 个文件 -> {output}",
        "en": "step 1/5: ✅ All Finished! Converted {done}/{total} files -> {output}",
    },
    # vad_processor.py
    "vad.loading_model": {"zh": "正在加载Silero VAD模型...", "en": "Loading Silero VAD model..."},
    "vad.model_loaded": {"zh": "✓ VAD模型加载成功", "en": "✓ VAD model loaded"},
    "vad.model_load_failed": {"zh": "✗ VAD模型加载失败: {error}", "en": "✗ VAD model load failed: {error}"},
    "vad.load_audio_failed": {"zh": "加载音频文件失败: {error}", "en": "Failed to load audio file: {error}"},
    "vad.save_audio_failed": {"zh": "保存音频文件失败: {error}", "en": "Failed to save audio file: {error}"},
    "vad.audio_too_short_warn": {
        "zh": "  警告: 音频时长({duration:.2f}s)小于合并阈值({threshold}s)",
        "en": "  Warning: audio duration ({duration:.2f}s) is below merge threshold ({threshold}s)",
    },
    "vad.no_speech_segments": {"zh": "  未检测到语音片段", "en": "  No speech segments detected"},
    "vad.no_valid_segments": {"zh": "  没有有效的语音片段", "en": "  No valid speech segments"},
    "vad.segments_generated": {"zh": "  生成 {count} 个片段", "en": "  Generated {count} segments"},
    "vad.process_failed": {"zh": "  处理失败: {error}", "en": "  Processing failed: {error}"},
    "vad.scan_dir": {"zh": "扫描目录: {input_dir}", "en": "Scanning directory: {input_dir}"},
    "vad.no_audio_files": {"zh": "未找到任何音频文件", "en": "No audio files found"},
    "vad.found_audio_files": {"zh": "找到 {count} 个音频文件", "en": "Found {count} audio files"},
    "vad.processing_audio_desc": {"zh": "处理音频文件", "en": "Processing audio files"},
    "vad.process_complete_count": {
        "zh": "处理完成，总共生成 {count} 个文件",
        "en": "Processing complete, generated {count} files",
    },
    "vad.cli_description": {"zh": "🔊 基于Silero VAD的音频智能切分工具", "en": "🔊 Silero VAD audio segmentation tool"},
    "vad.cli_input": {"zh": "输入文件或目录路径", "en": "Input file or directory path"},
    "vad.cli_output": {"zh": "输出目录路径", "en": "Output directory path"},
    "vad.cli_recursive": {"zh": "递归处理子目录", "en": "Process subdirectories recursively"},
    "vad.cli_sample_rate": {"zh": "输出采样率 (默认: 16000)", "en": "Output sample rate (default: 16000)"},
    "vad.cli_vad_threshold": {"zh": "VAD阈值 (默认: 0.5)", "en": "VAD threshold (default: 0.5)"},
    "vad.cli_min_speech": {"zh": "最短语音时长 (默认: 250ms)", "en": "Min speech duration (default: 250ms)"},
    "vad.cli_min_silence": {"zh": "最短静音时长 (默认: 200ms)", "en": "Min silence duration (default: 200ms)"},
    "vad.cli_speech_pad": {"zh": "前后填充时长 (默认: 30ms)", "en": "Speech padding (default: 30ms)"},
    "vad.cli_merge_threshold": {"zh": "最小音频长度阈值(秒)，小于此值会被合并 (默认: 0.5)", "en": "Min audio length threshold (s), shorter segments will be merged (default: 0.5)"},
    "vad.cli_split_threshold": {"zh": "最大音频长度阈值(秒)，超过此值会被切分 (默认: 10.0)", "en": "Max audio length threshold (s), longer segments will be split (default: 10.0)"},
    "vad.title": {"zh": "🔊 Silero VAD 音频切分工具", "en": "🔊 Silero VAD Audio Segmentation Tool"},
    "vad.path_not_found": {"zh": "错误: 路径不存在: {path}", "en": "Error: path does not exist: {path}"},
    "vad.input": {"zh": "输入: {input}", "en": "Input: {input}"},
    "vad.output": {"zh": "输出: {output}", "en": "Output: {output}"},
    "vad.sample_rate": {"zh": "采样率: {sample_rate}Hz", "en": "Sample rate: {sample_rate}Hz"},
    "vad.split_threshold": {"zh": "切分阈值: {threshold}s", "en": "Split threshold: {threshold}s"},
    "vad.merge_threshold": {"zh": "合并阈值: {threshold}s", "en": "Merge threshold: {threshold}s"},
    "vad.init_failed": {"zh": "初始化失败: {error}", "en": "Initialization failed: {error}"},
    "vad.invalid_path_type": {"zh": "无效的路径类型: {path}", "en": "Invalid path type: {path}"},
    "vad.user_interrupt": {"zh": "用户中断处理", "en": "Processing interrupted by user"},
    "vad.process_error": {"zh": "处理过程中发生错误: {error}", "en": "Error during processing: {error}"},
    "vad.total_files": {"zh": "总生成文件数: {count}", "en": "Total files generated: {count}"},
    "vad.total_time": {"zh": "总耗时: {seconds:.2f}秒", "en": "Total time: {seconds:.2f}s"},
    "vad.done": {"zh": "✅ 处理完成！", "en": "✅ Processing complete!"},
    "vad.step_done": {
        "zh": "step 2/5: ✅ 全部完成！已创建 {count} 个文件 -> {output}",
        "en": "step 2/5: ✅ All Finished! created {count} files -> {output}",
    },
    # transcribe_to_dataset.py
    "asr.resample": {
        "zh": "重采样 {name}: {src_sr}Hz -> {dst_sr}Hz",
        "en": "Resample {name}: {src_sr}Hz -> {dst_sr}Hz",
    },
    "asr.txt_read_failed": {
        "zh": " ! 读取txt文件失败 {path}: {error}, 使用ASR转录",
        "en": " ! Failed to read txt {path}: {error}, using ASR transcription",
    },
    "asr.merge_stereo": {"zh": "合并立体声 {name}", "en": "Merge stereo {name}"},
    "asr.worker_use_gpu": {
        "zh": "[Worker {worker_id}] 使用GPU {gpu_id}，映射为 {target_device}",
        "en": "[Worker {worker_id}] Using GPU {gpu_id}, mapped to {target_device}",
    },
    "asr.worker_gpu_unavailable": {
        "zh": "[Worker {worker_id}] GPU {gpu_id} 不可用，切换到CPU",
        "en": "[Worker {worker_id}] GPU {gpu_id} unavailable, switching to CPU",
    },
    "asr.worker_start": {
        "zh": "[Worker {worker_id}] 开始处理 {count} 个文件，使用设备: {device}",
        "en": "[Worker {worker_id}] Start processing {count} files on {device}",
    },
    "asr.worker_model_loaded": {"zh": "[Worker {worker_id}] ASR模型加载成功", "en": "[Worker {worker_id}] ASR model loaded"},
    "asr.worker_model_failed": {
        "zh": "[Worker {worker_id}] ASR模型加载失败: {error}",
        "en": "[Worker {worker_id}] ASR model load failed: {error}",
    },
    "asr.worker_try_cpu": {"zh": "[Worker {worker_id}] 尝试使用CPU加载模型", "en": "[Worker {worker_id}] Trying CPU model"},
    "asr.worker_skip_file": {
        "zh": "[Worker {worker_id}] 跳过文件 {name}: {error}",
        "en": "[Worker {worker_id}] Skipping file {name}: {error}",
    },
    "asr.worker_done": {
        "zh": "[Worker {worker_id}] 完成处理，生成 {count} 条记录",
        "en": "[Worker {worker_id}] Done, generated {count} records",
    },
    "asr.worker_error": {"zh": "[Worker {worker_id}] 发生错误: {error}", "en": "[Worker {worker_id}] Error: {error}"},
    "asr.worker_desc": {"zh": "Worker {worker_id}", "en": "Worker {worker_id}"},
    "asr.mp_start": {
        "zh": "🚀 启动多进程处理: {workers} 个工作进程处理 {count} 个文件",
        "en": "🚀 Starting multiprocess: {workers} workers for {count} files",
    },
    "asr.main_start_worker": {
        "zh": "[主进程] 启动工作进程 {worker_id}，分配 {count} 个文件，GPU: {gpu_id}",
        "en": "[Main] Started worker {worker_id}, assigned {count} files, GPU: {gpu_id}",
    },
    "asr.main_worker_done": {"zh": "[主进程] 工作进程 {worker_id} 已完成", "en": "[Main] Worker {worker_id} completed"},
    "asr.main_merge_worker": {
        "zh": "[主进程] 合并工作进程 {worker_id} 的 {count} 条记录",
        "en": "[Main] Merging {count} records from worker {worker_id}",
    },
    "asr.mp_done": {"zh": "✅ 多进程处理完成，总共生成 {count} 条记录", "en": "✅ Multiprocess done, {count} records total"},
    "asr.total_records": {"zh": "总记录数: {count}", "en": "Total records: {count}"},
    "asr.no_records": {"zh": "⚠️ 没有记录可处理", "en": "⚠️ No records to process"},
    "asr.small_records": {"zh": "记录数较少，直接处理...", "en": "Few records, processing directly..."},
    "asr.normalizing": {"zh": "正在进行响度控制...", "en": "Normalizing loudness..."},
    "asr.normalizing_desc": {"zh": "响度归一化", "en": "Normalizing"},
    "asr.normalizing_batch_desc": {"zh": "批次归一化 {batch_idx}", "en": "Normalizing batch {batch_idx}"},
    "asr.build_dataset": {"zh": "开始生成Dataset...", "en": "Building dataset..."},
    "asr.dataset_saved": {
        "zh": "✓ 数据集已保存，包含 {count} 条记录 -> {dst}",
        "en": "✓ Saved dataset with {count} records -> {dst}",
    },
    "asr.batch_processing_start": {"zh": "开始分批处理，批大小: {batch_size}", "en": "Batch processing, size: {batch_size}"},
    "asr.batch_processing": {
        "zh": "处理批次 {batch_idx}/{total_batches} (记录 {start}-{end})",
        "en": "Processing batch {batch_idx}/{total_batches} (records {start}-{end})",
    },
    "asr.batch_build_dataset": {"zh": "创建批次 {batch_idx} 的Dataset...", "en": "Building dataset for batch {batch_idx}..."},
    "asr.batch_saved": {"zh": "✓ 批次 {batch_idx} 已保存到 {path}", "en": "✓ Batch {batch_idx} saved to {path}"},
    "asr.merge_batches": {"zh": "合并 {count} 个批次...", "en": "Merging {count} batches..."},
    "asr.final_saved": {"zh": "✓ 最终数据集已保存到 {path}", "en": "✓ Final dataset saved to {path}"},
    "asr.cleanup_batches": {"zh": "✓ 已清理临时批次文件", "en": "✓ Cleaned up temporary batch files"},
    "asr.merge_failed": {"zh": "⚠️ 合并失败: {error}", "en": "⚠️ Merge failed: {error}"},
    "asr.batch_files_saved": {"zh": "批次文件保存在: {path}", "en": "Batch files saved at: {path}"},
    "asr.batch_files_hint": {"zh": "你可以手动加载各个批次文件", "en": "You can load each batch file manually"},
    "asr.cli_src": {"zh": "音频文件根目录", "en": "Audio root directory"},
    "asr.cli_dst": {"zh": "输出 datasets 目录", "en": "Output datasets directory"},
    "asr.cli_gpu_devices": {"zh": "指定GPU设备，用逗号分隔，如: 0,1,2,3", "en": "GPU devices, comma-separated, e.g. 0,1,2,3"},
    "asr.cli_num_workers": {"zh": "并行工作进程数", "en": "Number of worker processes"},
    "asr.cli_min_sec": {"zh": "分段最小间隔 (s)", "en": "Minimum segment interval (s)"},
    "asr.cli_batch_size": {"zh": "批处理大小，避免内存溢出 (默认: 1000)", "en": "Batch size to avoid OOM (default: 1000)"},
    "asr.no_valid_gpu": {"zh": "⚠️ 未找到有效的GPU设备，使用CPU", "en": "⚠️ No valid GPU found, using CPU"},
    "asr.use_gpu_devices": {"zh": "🚀 将使用GPU设备: {devices}", "en": "🚀 Using GPU devices: {devices}"},
    "asr.use_cpu": {"zh": "🖥️ 使用CPU设备", "en": "🖥️ Using CPU"},
    "asr.multi_gpu": {"zh": "📊 多GPU并行处理，使用 {workers} 个工作进程", "en": "📊 Multi-GPU processing with {workers} workers"},
    "asr.cpu_parallel": {"zh": "🔧 CPU并行处理，使用 {workers} 个工作进程", "en": "🔧 CPU processing with {workers} workers"},
    "asr.no_audio_files": {
        "zh": "错误：在目录 '{src}' 中没有找到任何 .wav 或 .mp3 文件。",
        "en": "Error: no .wav or .mp3 files found in '{src}'.",
    },
    "asr.found_files": {
        "zh": "找到 {wav_count} 个 .wav 文件和 {mp3_count} 个 .mp3 文件",
        "en": "Found {wav_count} .wav and {mp3_count} .mp3 files",
    },
    "asr.loading_model": {"zh": "正在加载 ASR 模型...", "en": "Loading ASR model..."},
    "asr.using_model": {"zh": "[ASR] 使用 {model_type} 于 {device}", "en": "[ASR] using {model_type} on {device}"},
    "asr.asr_desc": {"zh": "ASR", "en": "ASR"},
    "asr.no_records_extracted": {"zh": "错误：未能从音频文件中提取任何有效的语音文本对。", "en": "Error: no valid speech-text pairs extracted."},
    "asr.step_done": {
        "zh": "step 4/5: ✅ 全部完成！已转录 {count} 个文件 -> {dst}",
        "en": "step 4/5: ✅ All Finished! Transcribed {count} files -> {dst}",
    },
    # main_ui.py
    "ui.speaker_fetch_failed": {"zh": "获取说话人列表失败: {error}", "en": "Failed to fetch speaker list: {error}"},
    "ui.simple_start": {"zh": "🎵 启动 HydraVox 简单版界面...", "en": "🎵 Launching HydraVox simple UI..."},
    "ui.full_start": {"zh": "🚀 启动 HydraVox 完整版界面...", "en": "🚀 Launching HydraVox full UI..."},
    "ui.system_start": {"zh": "🚀 启动 HydraVox TTS 系统...", "en": "🚀 Starting HydraVox TTS system..."},
    "ui.service_addr": {"zh": "📡 服务地址: http://{server_name}:{server_port}", "en": "📡 Service: http://{server_name}:{server_port}"},
    "ui.backend_addr": {"zh": "🔗 后端地址: {backend}", "en": "🔗 Backend: {backend}"},
}


def t(text: str, **kwargs) -> str:
    entry = _TRANSLATIONS.get(text)
    if entry:
        result = entry.get(_LANG, entry.get("zh", text))
    else:
        result = text
    if kwargs:
        try:
            return result.format(**kwargs)
        except Exception:
            return result
    return result


@dataclass(frozen=True)
class I18nMessage:
    key: str
    kwargs: Dict[str, Any]


def msg(key: str, **kwargs: Any) -> I18nMessage:
    return I18nMessage(key=key, kwargs=kwargs)


def render(value: Any) -> Any:
    if isinstance(value, I18nMessage):
        return t(value.key, **value.kwargs)
    if isinstance(value, tuple):
        return tuple(render(v) for v in value)
    if isinstance(value, list):
        return [render(v) for v in value]
    if isinstance(value, dict):
        return {k: render(v) for k, v in value.items()}
    return value


def with_i18n(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, types.GeneratorType):
            for item in result:
                yield render(item)
            return
        return render(result)

    return wrapper


def get_lang() -> str:
    return _LANG


def set_lang(lang: str) -> str:
    global _LANG
    lang = (lang or "").lower()
    if lang not in ("zh", "en"):
        lang = "en"
    _LANG = lang
    os.environ["HYDRAVOX_UI_LANG"] = lang
    return _LANG
