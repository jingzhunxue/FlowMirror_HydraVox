# 心流TTS文本预处理工具

专为心流TTS设计，提供全方位的文本标准化解决方案。本工具能够智能识别并转换各类符号、单位和特殊表达，确保输出高质量的可读中文文本。

## ✨ 核心特性

- 🔄 全面的符号转换系统
  - 数学符号（±, ≠, ≥, ≤等）
  - 科学符号（℃, °, △等）
  - 特殊字符（©, ®, ™等）
  
- 📊 智能数字与单位处理
  - 阿拉伯数字转中文
  - 复合单位智能转换（km/h → 千米每小时）
  - 科学计数法标准化
  
- 📝 文本格式优化
  - 中英文标点符号规范化
  - 特殊表达标准化（分数、百分比等）
  - 多种文本输入格式支持
  - 拼音标记转汉字支持

- 🔤 字音替换功能
  - 支持自定义字音替换规则
  - 全字匹配替换
  - 批量处理支持

## 🚀 快速开始

### 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

### 基础使用

```python

# 初始化转换器
normalizer = Normalizer(cache_dir='./WeTextProcessing/cache')
tn = FlowMirrorTN(
    normalizer=normalizer,
    dict_file_path="./symbols.dic",
    unit_dict_path="./unit.dic",
    char_replace_path="./char_replace.txt",
    uppercase=True,
    pinyin_to_hanzi=True
)

# 单条文本处理
text = "我的速度是2km/h"
result = tn.process_text(text, log=True)  # 输出：我的速度是两千米每小时

# 处理带拼音标记的文本
text_with_pinyin = "你好<phone>ni3 hao3</phone>世界"
result = tn.process_text(text_with_pinyin)  # 输出：你好你好世界

# 批量处理
texts = [
    "3+2-2",  # 输出：三加二减二
    "温度为37.5℃"  # 输出：温度为三十七点五摄氏度
]
results = tn.process_list(texts, log=True)

# 处理JSONL文件
tn.process_jsonl("./test.jsonl")

```

## 📖 详细文档

### 拼音转换说明

1. **标记格式**
   - 使用`<phone>`标签包裹拼音文本
   - 拼音必须包含声调(1-4)
   - 示例：`<phone>ni3 hao3</phone>`

2. **使用方法**
   - 在`process_text()`方法中设置`change_pinyin=True`
   - 支持多个拼音组合
   - 转换失败时会保留原始拼音

### 配置文件说明

1. **符号转换字典** (`symbols.dic`)
   ```
   # 格式：符号 转换后文本
   △ 三角形
   ² 平方
   ∑ 求和
   ```

2. **单位转换字典** (`unit.dic`)
   ```
   # 格式：单位 转换规则
   km/h 千米每小时
   m² 平方米
   ```

3. **字音替换文件** (`char_replace.txt`)
   ```
   # 格式：原文本 替换文本
   要求 药求
   银行 因行
   # 使用#号添加注释
   ```

### 自定义转换规则

1. **添加新符号**
   - 直接编辑 `symbols.dic`
   - 每行一个规则
   - 格式：`符号 转换后文本`

2. **添加新单位**
   - 编辑 `unit.dic`
   - 支持简单单位和复合单位
   - 格式：`单位 转换规则`

3. **添加字音替换规则**
   - 编辑 `char_replace.txt`
   - 每行一个替换规则
   - 格式：`原文本 替换文本`
   - 支持使用#添加注释
   - 较长的文本会优先被替换

## 🔍 使用示例

```python
# 数学表达式
tn.process_text("1+1=2")  # 一加一等于二

# 复合单位
tn.process_text("速度为60km/h")  # 速度为六十千米每小时

# 特殊符号
tn.process_text("温度为25℃")  # 温度为二十五摄氏度

# 分数处理
tn.process_text("1/2的比例")  # 二分之一的比例

# 字音替换
tn.process_text("这是一个要求")  # 这是一个药求
```

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add: 新特性'`
4. 推送分支：`git push origin feature/AmazingFeature`
5. 提交 Pull Request

## 📝 更新日志

### v1.0.0 (2024-10-29)
- ✨ 首次发布
- 🎯 实现基础符号转换功能
- 🔄 支持单位智能转换
- 📝 特殊表达处理系统

### v1.1.0 (2024-11-07)
- 🔤 添加字音替换功能
- 🔄 拼音标记转汉字支持
- 📝 优化输出格式
- 🔄 添加大写转换开关

## ⚠️ 注意事项

1. 单位转换限制：
   - 单个单位字符（如 m、s）暂不支持直接转换，会与数学符号产生冲突，例如线段m的长度是40m。
   - 请使用完整的复合单位（如 km、km/h）以确保正确转换
2. 定期更新转换规则以支持新的符号和单位
3. 字音替换说明：
   - 替换采用全字匹配模式
   - 较长的文本会优先被替换
   - 替换在所有其他处理完成后进行


## 🔧 常见问题

**Q: 如何添加自定义符号？**  
A: 直接在 `symbols.dic` 文件末尾添加新行，格式为 `符号 转换后文本`。

**Q: 支持哪些输入格式？**  
A: 支持单字符串、字符串列表和JSONL文件格式。

**Q: 如何添加字音替换规则？**  
A: 在 `char_replace.txt` 文件中添加新行，格式为 `原文本 替换文本`。较长的文本会优先被替换。
