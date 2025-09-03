import struct
import pickle
import os

class PinyinConverter:
    def __init__(self):
        """
        初始化转换器
        字典文件会自动在当前类所在目录下创建和管理
        """
        # 获取当前类文件所在目录
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dict_path = os.path.join(self.base_dir, 'pinyin.dic')
        self.text_path = os.path.join(self.base_dir, 'pinyin.txt')
        self.custom_path = os.path.join(self.base_dir, 'custom_pinyin.txt')
        self.delete_path = os.path.join(self.base_dir, 'delete_pinyin.txt')
        self.final_path = os.path.join(self.base_dir, 'final_pinyin.txt')
        self.change_path = os.path.join(self.base_dir, 'change.txt')
        
        # 记录所有修改
        self.changes = {
            'added': [],    # 新增的映射
            'modified': [], # 修改的映射
            'deleted': []   # 删除的映射
        }
        
        self._ensure_dict_exists()
        self.pinyin_dict = self._load_dict()


        
    def _ensure_dict_exists(self):
        """
        每次启动时都从文本文件重新生成字典文件
        如果文本文件不存在，则创建一个包含基本映射的文本文件
        """
        # 如果.txt文件不存在，创建一个基础的txt文件
        if not os.path.exists(self.text_path):
            self._create_basic_text_file()
        # 每次都从txt文件重新创建dic文件
        self._create_dict_from_text()
    
    def _create_basic_text_file(self):
        """
        创建一个基础的拼音映射文本文件
        """
        basic_mappings = """# 拼音字典文件
        # 格式：拼音 汉字
        # 示例：ni2 你

        # 基础发音
        a1 啊
        a2 嗄
        a3 吖
        a4 阿

        # 常用字
        ni1 妮
        ni2 你
        ni3 拟
        ni4 逆
        hao1 蒿
        hao2 毫
        hao3 好
        hao4 号
        wo1 窝
        wo2 我
        wo3 我
        wo4 握"""
        
        with open(self.text_path, 'w', encoding='utf-8') as f:
            f.write(basic_mappings)
    
    def _create_dict_from_text(self):
        """
        从文本文件创建.dic文件，处理顺序：
        1. 读取基础映射
        2. 应用自定义映射
        3. 删除指定映射
        """
        base_mappings = {}
        custom_mappings = {}
        delete_list = set()
        
        # 读取基础映射文件
        self._read_mapping_file(self.text_path, base_mappings)
        
        # 读取自定义映射文件
        if os.path.exists(self.custom_path):
            self._read_mapping_file(self.custom_path, custom_mappings)
            
        # 读取需要删除的映射
        if os.path.exists(self.delete_path):
            with open(self.delete_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        delete_list.add(line.split()[0].lower())
        
        # 记录变更
        for pinyin, char in custom_mappings.items():
            if pinyin in base_mappings:
                if base_mappings[pinyin] != char:
                    self.changes['modified'].append((pinyin, base_mappings[pinyin], char))
            else:
                self.changes['added'].append((pinyin, char))
        
        # 记录删除的映射
        for pinyin in delete_list:
            if pinyin in base_mappings:
                self.changes['deleted'].append((pinyin, base_mappings[pinyin]))
        
        # 合并映射
        final_mappings = base_mappings.copy()
        final_mappings.update(custom_mappings)
        
        # 删除指定的映射
        for pinyin in delete_list:
            if pinyin in final_mappings:
                del final_mappings[pinyin]
        
        # 保存最终映射到final_pinyin.txt
        self._save_final_mappings(final_mappings)
        
        # 记录所有变更到change.txt
        self._save_changes()
        
        # 保存为.dic文件
        try:
            with open(self.dict_path, 'wb') as f:
                pickle.dump(final_mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise Exception(f"创建字典文件时出错：{str(e)}")

    
    def _load_dict(self):
        """
        加载.dic文件
        """
        try:
            with open(self.dict_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"加载拼音字典时出错：{str(e)}")
    def _read_mapping_file(self, file_path, mappings):
        """
        读取映射文件并更新到mappings字典中
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        pinyin, char = line.split()
                        if not pinyin[-1].isdigit() or not (1 <= int(pinyin[-1]) <= 4):
                            print(f"警告：跳过无效的拼音格式: {line}")
                            continue
                        if len(char) != 1:
                            print(f"警告：跳过无效的汉字格式: {line}")
                            continue
                        mappings[pinyin.lower()] = char
                    except ValueError:
                        print(f"警告：跳过格式错误的行: {line}")
                        continue
        except Exception as e:
            print(f"读取文件 {file_path} 时出错：{str(e)}")

    def _save_dict(self):
        """
        保存字典到.dic文件和custom_pinyin.txt
        """
        try:
            # 读取原始的基础映射
            base_mappings = {}
            self._read_mapping_file(self.text_path, base_mappings)
            
            # 找出与基础映射不同的部分，保存到custom_pinyin.txt
            custom_mappings = {}
            for pinyin, char in self.pinyin_dict.items():
                if pinyin not in base_mappings or base_mappings[pinyin] != char:
                    custom_mappings[pinyin] = char
            
            # 保存自定义映射
            with open(self.custom_path, 'w', encoding='utf-8') as f:
                f.write("# 自定义拼音映射\n")
                f.write("# 格式：拼音 汉字\n")
                f.write("# 此文件只包含用户自定义或修改的映射\n\n")
                
                # 按拼音排序写入
                for pinyin in sorted(custom_mappings.keys()):
                    f.write(f"{pinyin} {custom_mappings[pinyin]}\n")
            
            # 保存.dic文件
            with open(self.dict_path, 'wb') as f:
                pickle.dump(self.pinyin_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            # 更新final_pinyin.txt
            self._save_final_mappings(self.pinyin_dict)
                    
        except Exception as e:
            raise Exception(f"保存拼音字典时出错：{str(e)}")

    def _save_changes(self):
        """
        保存所有变更记录到change.txt
        """
        try:
            with open(self.change_path, 'w', encoding='utf-8') as f:
                f.write("# 拼音映射变更记录\n\n")
                
                f.write("## 新增的映射\n")
                for pinyin, char in self.changes['added']:
                    f.write(f"+ {pinyin} -> {char}\n")
                
                f.write("\n## 修改的映射\n")
                for pinyin, old_char, new_char in self.changes['modified']:
                    f.write(f"* {pinyin}: {old_char} -> {new_char}\n")
                
                f.write("\n## 删除的映射\n")
                for pinyin, char in self.changes['deleted']:
                    f.write(f"- {pinyin} ({char})\n")
        except Exception as e:
            print(f"保存变更记录时出错：{str(e)}")

    
    def convert(self, pinyin):
        """
        转换带声调的拼音为汉字
        pinyin: 格式如 'ni2', 'hao3' 等
        """
        if not pinyin:
            return ''
            
        # 检查格式
        if not pinyin[-1].isdigit() or not (1 <= int(pinyin[-1]) <= 4):
            raise ValueError("拼音必须以1-4的声调结尾")
            
        # 获取对应汉字
        char = self.pinyin_dict.get(pinyin.lower())
        if not char:
            raise KeyError(f"未找到拼音 {pinyin} 对应的汉字")
            
        return char
    
    def add_mapping(self, pinyin, char):
        """
        添加新的拼音-汉字映射
        """
        if not pinyin[-1].isdigit() or not (1 <= int(pinyin[-1]) <= 4):
            raise ValueError("拼音必须以1-4的声调结尾")
        if len(char) != 1:
            raise ValueError("必须是单个汉字")
        
        pinyin = pinyin.lower()
        # 记录变更
        if pinyin in self.pinyin_dict:
            if self.pinyin_dict[pinyin] != char:
                self.changes['modified'].append((pinyin, self.pinyin_dict[pinyin], char))
        else:
            self.changes['added'].append((pinyin, char))
        
        # 更新字典
        self.pinyin_dict[pinyin] = char
        self._save_dict()
        self._save_changes()
    
    def batch_add_mappings(self, mappings):
        """
        批量添加拼音-汉字映射
        mappings: list of tuple, 如 [('ni2', '你'), ('hao3', '好')]
        """
        for pinyin, char in mappings:
            if not pinyin[-1].isdigit() or not (1 <= int(pinyin[-1]) <= 4):
                raise ValueError(f"无效的拼音格式: {pinyin}")
            if len(char) != 1:
                raise ValueError(f"无效的汉字: {char}")
            self.pinyin_dict[pinyin.lower()] = char
            
        # 批量保存
        self._save_dict()
    
    def remove_mapping(self, pinyin):
        """
        删除一个拼音映射
        """
        pinyin = pinyin.lower()
        if pinyin in self.pinyin_dict:
            self.changes['deleted'].append((pinyin, self.pinyin_dict[pinyin]))
            del self.pinyin_dict[pinyin]
            self._save_dict()
            self._save_changes()
    
    def get_all_mappings(self):
        """
        获取所有拼音映射
        返回一个排序后的列表
        """
        return sorted(self.pinyin_dict.items())
    
    def reload_dict(self):
        """
        重新从文件加载字典
        """
        self.pinyin_dict = self._load_dict()

    def _save_final_mappings(self, mappings):
        """
        保存最终的映射到final_pinyin.txt
        """
        try:
            with open(self.final_path, 'w', encoding='utf-8') as f:
                f.write("# 最终拼音映射文件\n")
                f.write("# 格式：拼音 汉字\n")
                f.write("# 此文件由系统自动生成，包含基础映射和自定义映射的合并结果\n\n")
                
                # 按拼音排序写入
                for pinyin in sorted(mappings.keys()):
                    f.write(f"{pinyin} {mappings[pinyin]}\n")
        except Exception as e:
            print(f"保存最终映射文件时出错：{str(e)}")


# 使用示例
if __name__ == "__main__":
    try:
        # 创建转换器实例
        converter = PinyinConverter()
        
        # 测试转换
        test_cases = ['ni2', 'hao3']
        for pinyin in test_cases:
            try:
                result = converter.convert(pinyin)
                print(f"{pinyin} -> {result}")
            except (ValueError, KeyError) as e:
                print(f"Error converting {pinyin}: {str(e)}")
        
        # 添加新映射
        converter.add_mapping('wo3', '我')
        
        # 批量添加映射
        new_mappings = [
            ('lai2', '来'),
            ('shi4', '是')
        ]
        converter.batch_add_mappings(new_mappings)
        
        # 获取所有映射
        all_mappings = converter.get_all_mappings()
        print("\n所有拼音映射：")
        for pinyin, char in all_mappings:
            print(f"{pinyin} -> {char}")
            
    except Exception as e:
        print(f"程序运行出错：{str(e)}")