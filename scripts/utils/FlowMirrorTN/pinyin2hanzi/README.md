****# 拼音汉字转换器 (Pinyin to Hanzi Converter)

这是一个简单的拼音转汉字转换工具,支持带声调拼音到汉字的转换,并提供了完整的映射管理功能。

## 功能特点

- 支持带声调拼音转换为对应汉字
- 提供基础拼音映射库
- 支持自定义映射添加和修改
- 支持删除指定映射
- 完整的变更历史记录
- 文件持久化存储

## 文件系统说明

- `pinyin.txt`: 基础拼音映射文件
- `custom_pinyin.txt`: 用户自定义/修改的映射
- `delete_pinyin.txt`: 需要删除的映射
- `final_pinyin.txt`: 最终的映射结果
- `change.txt`: 变更历史记录
- `pinyin.dic`: 二进制字典文件(程序使用)

## 使用方法

### 基本使用

```python
from PinyinConverter import PinyinConverter

# 创建转换器实例
converter = PinyinConverter()

# 转换单个拼音
result = converter.convert('ni3')  # 返回 '你'

# 添加新映射
converter.add_mapping('ai4', '爱')

# 批量添加映射
mappings = [
    ('lai2', '来'),
    ('shi4', '是')
]
converter.batch_add_mappings(mappings)

# 删除映射
converter.remove_mapping('ai4')
```

### 通过文件管理映射

1. 添加/修改映射:
   - 编辑 `custom_pinyin.txt`
   - 每行格式: `拼音 汉字`
   - 示例: `wo3 我`

2. 删除映射:
   - 编辑 `delete_pinyin.txt`
   - 每行写入要删除的拼音
   - 示例: `ni4`

## 映射规则

- 拼音必须包含声调(1-4)
- 每个拼音只能映射到一个汉字
- 支持的声调范围: 1(阴平)、2(阳平)、3(上声)、4(去声)

## 运行演示程序

```python
python mian.py
```

这将展示:
1. 原始映射状态
2. 自定义映射内容
3. 待删除映射
4. 变更记录
5. 最终映射状态
6. 代码修改演示
7. 最终映射文件预览

## 注意事项

- 不要直接修改 `pinyin.txt` 基础映射文件
- 所有自定义修改请通过 `custom_pinyin.txt` 进行
- 程序会自动处理文件编码(UTF-8)
- 每次启动时会重新生成 `.dic` 文件

## 依赖要求

- Python 3.6+
- 无需额外第三方库