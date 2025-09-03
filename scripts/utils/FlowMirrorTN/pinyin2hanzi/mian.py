from PinyinConverter import PinyinConverter

def print_section(title):
    print(f"\n{'-' * 20} {title} {'-' * 20}")

def show_mapping_status(converter, pinyins):
    """显示指定拼音的映射状态"""
    for pinyin in pinyins:
        try:
            result = converter.convert(pinyin)
            print(f"  {pinyin} -> {result}")
        except (ValueError, KeyError) as e:
            print(f"  {pinyin} -> 未找到映射")

def demonstrate_converter():
    print("拼音转换器演示程序")
    
    try:
        # 1. 显示原始映射状态
        print_section("1. 原始映射状态")
        converter = PinyinConverter()
        test_pinyins = ['ni2', 'hao3', 'wo3', 'shi4', 'zhong1']
        print("测试以下拼音的映射：")
        show_mapping_status(converter, test_pinyins)
        
        # 2. 显示自定义文件中的映射
        print_section("2. custom_pinyin.txt 中的自定义映射")
        print("文件位置：", converter.custom_path)
        try:
            with open(converter.custom_path, 'r', encoding='utf-8') as f:
                print("文件内容：")
                print(f.read())
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
            
        # 3. 显示要删除的映射
        print_section("3. delete_pinyin.txt 中的待删除映射")
        print("文件位置：", converter.delete_path)
        try:
            with open(converter.delete_path, 'r', encoding='utf-8') as f:
                print("文件内容：")
                print(f.read())
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
            
        # 4. 显示变更记录
        print_section("4. 变更记录")
        print("文件位置：", converter.change_path)
        try:
            with open(converter.change_path, 'r', encoding='utf-8') as f:
                print("变更历史：")
                print(f.read())
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
        
        # 5. 显示最终的映射状态
        print_section("5. 最终映射状态")
        print("再次测试相同的拼音：")
        show_mapping_status(converter, test_pinyins)
        
        # 6. 通过代码进行额外修改
        print_section("6. 通过代码进行额外修改")
        
        # 添加新映射
        print("添加新映射：")
        converter.add_mapping('ai4', '爱')
        print("  已添加: ai4 -> 爱")
        
        # 修改已有映射
        print("\n修改已有映射：")
        converter.add_mapping('hao3', '好')
        print("  已修改: hao3 -> 好")
        
        # 删除映射
        print("\n删除映射：")
        converter.remove_mapping('ai4')
        print("  已删除: ai4")
        
        # 7. 显示最终的映射文件
        print_section("7. 最终映射文件")
        print("文件位置：", converter.final_path)
        try:
            with open(converter.final_path, 'r', encoding='utf-8') as f:
                print("部分内容预览（前5行）：")
                for i, line in enumerate(f):
                    if i < 5:
                        print(line.rstrip())
                    else:
                        print("...")
                        break
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
        
        # 8. 文件说明
        print_section("8. 文件说明")
        print("""
文件系统说明：
1. pinyin.txt: 基础拼音映射文件，不应直接修改
2. custom_pinyin.txt: 存放需要添加或修改的映射
   - 格式：拼音 汉字
   - 示例：wo3 我
3. delete_pinyin.txt: 存放需要删除的映射
   - 格式：拼音
   - 示例：ni4
4. final_pinyin.txt: 最终的所有映射结果
5. change.txt: 记录所有的变更历史
6. pinyin.dic: 二进制字典文件，程序使用

处理顺序：
1. 首先读取 pinyin.txt 中的基础映射
2. 应用 custom_pinyin.txt 中的自定义映射（会覆盖基础映射）
3. 删除 delete_pinyin.txt 中指定的映射
4. 生成 final_pinyin.txt 和 pinyin.dic
5. 记录所有变更到 change.txt
        """.strip())
        
    except Exception as e:
        print(f"\n程序运行出错：{str(e)}")

if __name__ == "__main__":
    demonstrate_converter()