import os
import json
from tqdm import tqdm
from typing import Union, List, Dict
import sys
import string
import re
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
access8math_dir = os.path.join(current_dir, 'Access8Math')
sys.path.append(access8math_dir)
from Access8Math.main import convert_latex_to_spoken

# from util import process_special_symbols, process_chinese_slash, load_symbol_dict, protect_alphabet
from util import *

class FlowMirrorTN:
    def __init__(self, normalizer, dict_file_path: str, 
                unit_dict_path: str, 
                char_replace_path: str = None,
                mathcase: bool = False,
                pinyin_to_hanzi: bool = False):
        """
        Initialize FlowmirrorTN with required components
        
        Args:
            normalizer: WeTextProcessing normalizer instance
            dict_file_path (str): Path to the symbols dictionary file
            unit_dict_path (str): Path to the unit dictionary file
        """
        self.normalizer = normalizer
        self.symbol_dict = load_symbol_dict(dict_file_path)
        self.unit_dict = load_symbol_dict(unit_dict_path)
        self.chinese_numbers = set("零一二三四五六七八九十百千万亿幺")
        self.mathcase = mathcase  # 改为 mathcase
        self.pinyin_to_hanzi = pinyin_to_hanzi  # 添加拼音转汉字开关
        
        # 初始化拼音转换器
        from pinyin2hanzi import PinyinConverter
        self.pinyin_converter = PinyinConverter()

        self.char_replace_dict = {}
        if char_replace_path and os.path.exists(char_replace_path):
            with open(char_replace_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # 跳过空行和注释
                        parts = line.split(maxsplit=1)  # 只分割一次，获取前两部分
                        if len(parts) == 2:
                            source, target = parts
                            self.char_replace_dict[source] = target
    

    def set_pinyin_to_hanzi(self, value: bool):
        """设置是否将拼音标记转换为汉字
        
        Args:
            value (bool): True 表示转换为汉字，False 表示保持原样
        """
        self.pinyin_to_hanzi = value

    def set_mathcase(self, value: bool):
        """设置是否启用数学模式
        
        Args:
            value (bool): True 表示启用数学模式，False 表示保持原样
        """
        self.mathcase = value

    def _apply_char_replacement(self, text: str) -> str:
        """应用字音替换
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 替换后的文本
        """
        result = text
        for source, target in sorted(self.char_replace_dict.items(), key=len, reverse=True):
            result = result.replace(source, target)
        return result

    def _convert_pinyin(self, text: str) -> str:
        """转换文本中的拼音标记
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 处理后的文本
        """

        
        def convert_match(match):
            pinyin_text = match.group(1)  # 获取<phone>标签内的内容
            # 分割拼音序列
            pinyins = re.findall(r'[a-z]+\d', pinyin_text.lower())
            
            try:
                # 转换每个拼音
                result = ''
                for pinyin in pinyins:
                    try:
                        result += self.pinyin_converter.convert(pinyin)
                    except (ValueError, KeyError) as e:
                        print(f"拼音转换警告 - {pinyin}: {str(e)}")
                        result += pinyin  # 转换失败时保留原始拼音
                return result
            except Exception as e:
                print(f"拼音转换错误: {e}")
                return match.group(0)  # 如果整体转换失败，保持原始标签内容
        
        # 查找并替换所有<phone>标签
        pattern = r'<phone>(.*?)</phone>'
        return re.sub(pattern, convert_match, text)

    def _replace_symbols(self, text: str) -> str:
        """Replace symbols using dictionary with strict boundary checking"""
        result = text
        
        def is_word_boundary(char: str) -> bool:
            """检查字符是否为单词边界（空格、标点或非英文字符）"""
            return (not char or 
                    char.isspace() or 
                    char in string.punctuation or 
                    not char.isascii())
        
        for symbol in sorted(self.symbol_dict.keys(), key=len, reverse=True):
            pos = 0
            while True:
                index = result.find(symbol, pos)
                if index == -1:
                    break
                
                # 检查符号的前后边界
                before_char = result[index-1] if index > 0 else ''
                after_char = result[index+len(symbol)] if index+len(symbol) < len(result) else ''
                
                # 如果符号是纯英文，需要检查边界
                if symbol.isascii() and symbol.isalpha():
                    if is_word_boundary(before_char) and is_word_boundary(after_char):
                        result = result[:index] + self.symbol_dict[symbol] + result[index+len(symbol):]
                        pos = index + len(self.symbol_dict[symbol])
                    else:
                        pos = index + 1
                else:
                    # 非英文符号，直接替换
                    result = result[:index] + self.symbol_dict[symbol] + result[index+len(symbol):]
                    pos = index + len(self.symbol_dict[symbol])
                
        if result.startswith(','):
            result = result[1:]
        return result


    def _is_chinese(self, char: str) -> bool:
        """Check if character is Chinese"""
        return '\u4e00' <= char <= '\u9fff'
    
    def _is_chinese_num(self, char: str) -> bool:
        """Check if character is Chinese number"""
        return char in "零一二三四五六七八九十"
    
    def _is_punct(self, char: str) -> bool:
        """Check if character is punctuation"""
        chinese_puncts = '，。！？；：、（）【】《》""''…'
        english_puncts = ',.!?;:()/\[\]<>"\'-'
        return char in chinese_puncts or char in english_puncts

    def _is_valid_unit_context(self, before_char: str, after_char: str) -> bool:
        """
        检查单位前后的字符是否有效
        - 前后不能是英文字母
        - 可以是汉字（包括中文数字）、空格、数字、标点符号等任何非英文字母字符
        """
        is_before_valid = True if not before_char else not before_char.isalpha()
        is_after_valid = True if not after_char else not after_char.isalpha()
        
        # 特别处理：如果前面的字符是中文数字，也认为是有效的
        if before_char in self.chinese_numbers:
            is_before_valid = True
            
        return is_before_valid and is_after_valid

    def _replace_units(self, text: str) -> str:
        """Replace units in text with strict context checking"""
        # print(f"\n开始处理文本: '{text}'")
        result = text
        
        for unit, replacement in sorted(self.unit_dict.items(), key=lambda x: len(x[0]), reverse=True):
            pos = 0
            while True:
                index = result.find(unit, pos)
                if index == -1:
                    break
                
                # 获取单位前后的字符
                before_char = result[index-1] if index > 0 else ''
                after_char = result[index+len(unit)] if index+len(unit) < len(result) else ''
                
                # print(f"\n可能的匹配:")
                # print(f"单位: '{unit}' -> '{replacement}'")
                # print(f"位置: {index}")
                # print(f"前一个字符: '{before_char}'")
                # print(f"后一个字符: '{after_char}'")
                # print(f"当前文本片段: '{result[max(0,index-5):index+len(unit)+5]}'")
                
                # 检查上下文是否合适进行替换
                if self._is_valid_unit_context(before_char, after_char):
                    old_text = result
                    result = result[:index] + replacement + result[index+len(unit):]
                    print(f"执行替换: '{old_text}' -> '{result}'")
                    pos = index + len(replacement)
                else:
                    # print("上下文不合适，跳过此匹配")
                    pos = index + 1
                    
        # print(f"\n最终结果: '{result}'")
        return result
    
    def post_normalize_text(self, text: str) -> str:
        """
        规范化文本，但保留标记内的空格
        
        Args:
            text (str): 输入文本
            markers (dict): 标记到原始英文的映射
        
        Returns:
            str: 规范化后的文本
        """
        # 1. 分词并去除普通空格
        words = text.split()
        no_space_text = ''.join(words)
        
        # 2. 进行标准替换
        replacements = [
            ('$', ''),
            (',,', ','),
            ('上标二', '的平方'),
            ('上标三', '的三次方'),
            ('分之两', '分之二'),
            ('的平方正负', '的平方加减')
        ]
        
        for old, new in replacements:
            no_space_text = no_space_text.replace(old, new)
        
        # 3. 去除句末逗号
        if no_space_text.endswith(','):
            no_space_text = no_space_text[:-1]
        
        return no_space_text
    
    def latex_escape(self, formula: str) -> str:
        """
        将普通的LaTeX公式转换为正确转义的字符串
        
        Args:
            formula: 原始的LaTeX公式字符串
            
        Returns:
            转义后的字符串
        
        示例:
            >>> latex_escape(r'\frac{1}{2}')
            '\\\\frac{1}{2}'
        """
        # 将单个反斜杠替换为四个反斜杠
        escaped = formula.replace('\\', '\\\\\\\\')
        return escaped

    def latex_unescape(self, escaped_formula: str) -> str:
        """
        将转义后的字符串转换回普通的LaTeX公式
        
        Args:
            escaped_formula: 转义后的字符串
            
        Returns:
            原始的LaTeX公式
        
        示例:
            >>> latex_unescape(r'\\\\frac{1}{2}')
            '\\frac{1}{2}'
            >>> latex_unescape(r'\\(\\frac{1}{2}\\)')
            '\\frac{1}{2}'
        """
        # 处理不同级别的转义
        original = escaped_formula
        
        # 处理四重反斜杠
        original = original.replace('\\\\\\\\', '\\')
        
        # 处理双重反斜杠
        original = original.replace('\\\\', '\\')
        
        # 移除数学环境定界符（包括可能的转义形式）
        delimiters = [
            ('\\(', '('),
            ('\\)', ')'),
        ]
        
        for old, new in delimiters:
            original = original.replace(old, new)
        
        return original.strip()

    def _process_qi_character(self, text: str) -> str:
        """处理文本中的"奇"字和"得"字
        
        1. 当文本中出现"偶"字时，将除了特定词组外的"奇"字替换为"基"字
        2. 当"得"字前后是数字（一到十）或英文字母时，将"得"替换为"德"
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 处理后的文本
        """
        # 处理"奇"字的逻辑
        if '偶' in text:
            protected_words = ['奇怪', '奇特', '奇迹', '奇人']
            
            temp_text = text
            for i, word in enumerate(protected_words):
                temp_text = temp_text.replace(word, f'PROTECTED_{i}')
            
            temp_text = temp_text.replace('奇', '基')
            
            for i, word in enumerate(protected_words):
                temp_text = temp_text.replace(f'PROTECTED_{i}', word)
            
            text = temp_text
        
        # 处理"得"字的逻辑
        result = ''
        i = 0
        chinese_nums = "一二三四五六七八九十"
        
        while i < len(text):
            if text[i] == '得':
                # 获取前后字符，对边界情况进行特殊处理
                prev_char = text[i-1] if i > 0 else ''
                next_char = text[i+1] if i < len(text) - 1 else ''
                
                # 检查前后字符是否为数字或字母
                is_valid_prev = (prev_char in chinese_nums or 
                               'a' <= prev_char <= 'z' or 
                               'A' <= prev_char <= 'Z')
                is_valid_next = (next_char in chinese_nums or 
                               'a' <= next_char <= 'z' or 
                               'A' <= next_char <= 'Z')
                
                if is_valid_prev and is_valid_next:
                    result += '得'
                else:
                    result += '得'
            else:
                result += text[i]
            i += 1
        
        return result

    def process_text_en(self, input_text: str, log: bool = False, 
                        need_split: bool = True, 
                        sentences_per_split: int = 2, en_zh_split: bool = True) -> str:
        """
        处理英文文本
        """
        # 1. English replacement
        en_replaced_text, markers = replace_english_with_markers(input_text)

        # 2. Symbol replacement
        replaced_text = self._replace_symbols(en_replaced_text.replace('$', '').replace('\n', '|stop|'))

        # 3. Restore English from markers
        no_space_text = restore_english_from_markers(replaced_text, markers)

        # 4. 处理英文标点符号
        fixed_en_punctuation_text = fix_english_punctuation(no_space_text)

        # 5. 处理连续标点
        normalized_text = normalize_punctuation(fixed_en_punctuation_text)

        # 6. 处理空格
        normalized_spacing_text = normalize_spacing_and_punctuation(normalized_text)

        # 7. 添加分割标记
        if need_split:
            split_text = add_split_markers(normalized_spacing_text, sentences_per_split, en_zh_split)
        else:
            split_text = normalized_spacing_text

        # 8. 修复选项标记的标点符号
        option_punctuation_fixed = fix_option_punctuation(split_text)

        # 9. 简化标点符号
        simplified_text = simplify_punctuation(option_punctuation_fixed)

        # 10. 地名转换
        geographical_text = replace_geographical_names(simplified_text)
        geographical_text = geographical_text.replace('|stop|', '。')

        if log:
            print('-*'*30)
            print(f"1. After English replacement: {en_replaced_text}")
            print(f"2. After symbol replacement: {replaced_text}")
            print(f"3. After restore English from markers: {no_space_text}")
            print(f"4. After fix english punctuation: {fixed_en_punctuation_text}")
            print(f"5. After normalize punctuation: {normalized_text}")
            print(f"6. After normalize spacing: {normalized_spacing_text}")
            if need_split:
                print(f"7. After add split markers: {split_text}")
            print(f"8. After fix option punctuation: {option_punctuation_fixed}")
            print(f"9. After simplify punctuation: {simplified_text}")
            print(f"10. After replace geographical names: {geographical_text}")
            print('-*'*30)
        return geographical_text
    


    def process_text_cn(self, input_text: str, log: bool = False, 
                        need_split: bool = True, 
                        sentences_per_split: int = 2, en_zh_split: bool = True, replace_book_quotation: bool = True) -> str:
        # 1. Symbol replacement
        replaced_text = self._replace_symbols(input_text.replace('$', ''))

        # 3. 处理中文标点替换
        replaced_text_cn = fix_chinese_punctuation(replaced_text)

        # 2. 处理连续标点
        normalized_text = normalize_punctuation(replaced_text_cn)
        
        # 3. 简化中文标点
        simplified_text = simplify_punctuation(normalized_text, length_threshold=3)

        # 4. 转换数字 - 只在文本包含数字时执行
        if any(char.isdigit() for char in simplified_text):
            normalized_number = self.normalizer.normalize(simplified_text)
        else:
            normalized_number = simplified_text

        # 5. 将英文标点符号转换为中文标点符号
        cn_punctuation = {
            ',': '，',
            '.': '。',
            '?': '。',
            '!': '。',
        }
        # 创建一个包含所有中文标点的模式，避免重复替换
        protected_chars = ''.join(re.escape(char) for char in cn_punctuation.values())
        
        # 初始化待处理的文本
        cn_text = normalized_number
        
        for en_punct, cn_punct in cn_punctuation.items():
            # 修改模式以避免匹配已经转换的中文标点
            # 添加数字 0-9 到前后的检查中
            pattern = f'(?<![A-Za-z0-9{protected_chars}]){re.escape(en_punct)}(?![{protected_chars}])'
            cn_text = re.sub(pattern, cn_punct, cn_text)  # 使用当前的cn_text作为输入

        # 6. 添加分割标记
        if need_split:
            split_text = add_split_markers(cn_text, sentences_per_split, en_zh_split,subject='cn')
        else:
            split_text = cn_text

        # 7. 删除英文字母之间的空格和英文标点后接中文字符之间的空格
        def remove_english_spaces(text):
            # 使用正则表达式匹配英文字母之间的空格
            pattern1 = r'(?<=[A-Za-z])\s+(?=[A-Za-z])'
            # 使用正则表达式匹配英文标点后接中文字符之间的空格
            pattern2 = r'(?<=[,.!?;:])\s+(?=[\u4e00-\u9fff])'
            
            # 依次应用两个模式
            text = re.sub(pattern1, '', text)
            text = re.sub(pattern2, '', text)
            return text
            
        split_text = remove_english_spaces(split_text)

        char_replaced = self._apply_char_replacement(split_text)

        if log:
            print(f"1. After replaced text {replaced_text}")
            print(f"2. After fix chinese punctuation {replaced_text_cn}")
            print(f"3. After normalize punctuation {normalized_text}")
            print(f"4. After simplify punctuation {simplified_text}")
            print(f"5. After normalize number {normalized_number}")
            print(f"6. After convert cn punctuation {cn_text}")
            print(f"7. After split text {split_text}")
            print(f"8. After char replacement {char_replaced}")
        
        # 9. 合并短片段
        merged_text = merge_short_splits(char_replaced, min_length=50)


        
        if log and merged_text != char_replaced:
            print(f"9. After merging short splits {merged_text}")
    
        def replace_book_quotation(text: str) -> str:
            """
            替换书名号中的内容，但保留特殊标记如 <|split|> 中的尖括号
            """
            # 先将特殊标记替换为临时标记
            protected_text = text.replace('<|', '##SPLIT_START##').replace('|>', '##SPLIT_END##')
            
            # 替换剩余的尖括号
            protected_text = protected_text.replace('<', '').replace('>', '')
            
            # 恢复特殊标记
            result = protected_text.replace('##SPLIT_START##', '<|').replace('##SPLIT_END##', '|>')
            
            return result

        if replace_book_quotation:
            merged_text = replace_book_quotation(merged_text)
            if log:
                print(f"10. After replace book quotation {merged_text}")
        
        return merged_text

    def process_text_math(self, input_text: str, log: bool = False, 
                    mathcase: bool = None, pinyin_to_hanzi: bool = None) -> str:
        """
        Process a single text string
        
        Args:
            input_text (str): Text to process
            log (bool): Whether to print intermediate results
            mathcase (bool): Whether to apply math case formatting
            pinyin_to_hanzi (bool): Whether to convert pinyin to hanzi
            
        Returns:
            str: Processed text
        """

        if pinyin_to_hanzi is not None:
            self.set_pinyin_to_hanzi(pinyin_to_hanzi)

        if mathcase is not None:
            self.set_mathcase(mathcase)

        if self.pinyin_to_hanzi:
            # 1. 拼音转换
            pinyin_processed = self._convert_pinyin(input_text)
        else:
            pinyin_processed = input_text

        has_brackets = any(char in pinyin_processed for char in '(){}[]')

        # 2. latex_escape
        latex_escaped = self.latex_unescape(pinyin_processed)
        
        # 3. 处理数学标点符号 - 新增步骤
        punctuation_processed = fix_math_punctuation(latex_escaped)
        
        # 4. Symbol replacement
        replaced_text = self._replace_symbols(punctuation_processed.replace('$', ''))

        # 5. 处理中文斜杠
        replaced_text_processed_slash = process_chinese_slash(replaced_text)
        
        # 6. Special symbols processing
        special_processed = process_special_symbols(replaced_text_processed_slash)
        
        # 7. LaTeX conversion
        latex_spoken = convert_latex_to_spoken(special_processed)
        
        # 8. Normalization
        latex_spoken = '数学加' + latex_spoken
        
        # 处理连续空格
        def replace_spaces(text):
            # 首先处理三个及以上的空格
            pattern1 = r'(?<![,.!?;:，。！？；：正负])\s{3,}(?![,.!?;:，。！？；：])'
            text = re.sub(pattern1, ',', text)
            
            # 然后处理两个空格
            pattern2 = r'(?<![,.!?;:，。！？；：正负])\s{2}(?![,.!?;:，。！？；：])'
            return re.sub(pattern2, ',', text)
            
        latex_spoken = replace_spaces(latex_spoken)
        latex_spoken_no_space = latex_spoken.replace(' ', '')
        
        # 在normalize前添加逗号 - 紧贴在normalize调用前
        pattern_equals_number = r'(等于|乘以|除以|加|减|乘|除|加上|减去|乘上|负|正|)(\d+)'
        
        latex_spoken_with_comma = re.sub(pattern_equals_number, r'\1Math_Tag\2', latex_spoken_no_space)
        
        # print(latex_spoken_with_comma)
        normalized = self.normalizer.normalize(latex_spoken_with_comma)
        
        # 在normalize后立即删除添加的逗号 - 紧贴在normalize调用后
        # pattern_equals_comma_number = r'(等于|乘以|除以)(Math_Tag)([一二三四五六七八九零十百千万亿]+)'
        # normalized = re.sub(pattern_equals_comma_number, r'\1\3', normalized)
        normalized = normalized.replace('Math_Tag', '')
        
        normalized = normalized[3:]
        normalized = normalized.replace('Pi', 'π')
        
        no_space_text = self.post_normalize_text(normalized)

        # 9. Unit replacement
        unit_processed = self._replace_units(no_space_text)

        # 10. 保护字母前后不出现标点
        if not unit_processed[-1] in string.punctuation + "。！？…，、；：""''【】《》（）【】":
            unit_processed += "。"
        
        char_replaced = self._apply_char_replacement(unit_processed)

        if not char_replaced[-1] in string.punctuation + "。！？…，、；：""''【】《》（）【】":
            char_replaced += "。"

        def geometry_handler(match):
            shape = match.group(1)          # 图形名称
            points = match.group(2)         # 点的序列
            # 在每个字母之间插入逗号，但最后一个字母后不加
            formatted_points = ' '.join(list(points))
            return f"{shape}{formatted_points}"

        # 匹配常见的几何图形标记
        geometry_pattern = r'(角|三角形|四边形|平行四边形|梯形|正方形|矩形|多边形|线段)([A-Z]{2,})'
        char_replaced = re.sub(geometry_pattern, geometry_handler, char_replaced)

        final_text_before_multiply = char_replaced
        
        if has_brackets:
            # 匹配一到十之间的汉字
            chinese_nums = "一二三四五六七八九十"
            # 创建正则表达式模式
            pattern = f"([{chinese_nums}A-Za-z])([,，])([{chinese_nums}A-Za-z])"
            
            # 应用替换
            final_text = re.sub(pattern, lambda m: f"{m.group(1)}乘以{m.group(3)}", final_text_before_multiply)
        else:
            final_text = final_text_before_multiply

        def _post_process_two_to_liang(text):
            # 从units_zh.tsv文件中读取所有可能的量词
            units_file_path = os.path.join(current_dir, 'WeTextProcessing/tn/chinese/data/measure/units_zh.tsv')
            measure_words = []
            
            try:
                with open(units_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        unit = line.strip()
                        if unit:  # 跳过空行
                            measure_words.append(unit)
            except FileNotFoundError:
                # 如果文件不存在，使用默认的常见量词列表
                measure_words = ["个", "小时", "天", "年", "月", "周", "次", "本", "张", "只", "条", "块", "件", "台", "部", "辆", "支", "把", "节", "位", "名", "处", "座", "项", "组", "双", "对", "套", "份", "场", "层", "届", "轮", "回", "遍", "趟", "班", "排", "行", "列", "队", "批", "种", "类", "项", "点", "分", "秒"]
            
            # 构建正则表达式模式
            pattern_parts = []
            for measure in measure_words:
                pattern_parts.append(f'(?<![零一三四五六七八九十百千万亿])二{measure}')
            
            pattern = '|'.join(pattern_parts)
            
            # 替换函数
            def replace_func(match):
                matched_text = match.group(0)
                return '两' + matched_text[1:]
            
            return re.sub(pattern, replace_func, text)

        # 处理"奇"字
        final_text = self._process_qi_character(final_text)
        final_text = _post_process_two_to_liang(final_text)
        final_text = final_text.replace('两元一次', '二元一次')




        if self.mathcase:
            math_replaced = self._apply_mathcase(char_replaced)
            final_text = math_replaced

        if log:
            print('-*'*30)
            print(f'Input: {input_text}')
            if self.pinyin_to_hanzi:
                print(f"After pinyin conversion: {pinyin_processed}")
            print(f"2. After LaTeX escape: {latex_escaped}")    
            print(f"3. After punctuation processing: {punctuation_processed}")
            print(f"4. After symbol replacement: {replaced_text}")
            print(f"5. After chinese slash processing: {replaced_text_processed_slash}")
            print(f"6. After special symbol processing: {special_processed}")
            print(f"7. After LaTeX conversion: {latex_spoken}")
            print(f"8. After normalization: {normalized}")
            print(f"9. After removing spaces: {no_space_text}")
            print(f"10. After unit replacement: {unit_processed}")
            print(f"10. After char replacement: {char_replaced}")
            if self.mathcase:
                print(f"11. After mathcase: {math_replaced}")
            print(f"12. After multiply replacement: {final_text}")
            print('-*'*30)
            
        return final_text
    
    def process_text(self, input_text: str, log: bool = False, subject: str = 'math',
                    mathcase: bool = False, pinyin_to_hanzi: bool = False,
                    need_split: bool = False, 
                    sentences_per_split: int = 5, en_zh_split: bool = True, replace_book_quotation: bool = True) -> str:
        
        subject = subject.lower()

        if subject == 'math':
            return self.process_text_math(input_text, log, mathcase, pinyin_to_hanzi)
        elif subject == 'en':
            return self.process_text_en(input_text, log, 
                                        need_split, 
                                        sentences_per_split, en_zh_split)
        elif subject == 'cn':
            return self.process_text_cn(input_text, 
                                        log, 
                                        need_split, 
                                        sentences_per_split, 
                                        en_zh_split,
                                        replace_book_quotation)
        else:
            raise ValueError(f"无效的 subject: {subject}。请选择 'math', 'en' 或 'cn'")
        

    def process_list(self, text_list: List[str], log: bool = False) -> List[str]:
        """
        Process a list of texts
        
        Args:
            text_list (List[str]): List of texts to process
            log (bool): Whether to print intermediate results
            
        Returns:
            List[str]: List of processed texts
        """
        return [self.process_text(text, log) for text in text_list]

    def process_jsonl(self, input_jsonl: str, output_jsonl: str = None, subject: str = 'math') -> None:
        """
        处理 JSONL 文件，支持 'text' 或 'statement' 字段的文本处理
        
        Args:
            input_jsonl (str): 输入 JSONL 文件路径
            output_jsonl (str): 输出 JSONL 文件路径。如果为 None，将自动在原文件名后添加 'TN'
        """

        subject = subject.lower()
        if subject not in ['math', 'en', 'cn']:
            raise ValueError(f"无效的 subject: {subject}。请选择 'math', 'en' 或 'cn'")
        
        # 生成输出文件名
        if output_jsonl is None:
            dir_path = os.path.dirname(input_jsonl)
            base_name = os.path.basename(input_jsonl)
            name_without_ext = os.path.splitext(base_name)[0]
            output_jsonl = os.path.join(dir_path, f"{name_without_ext}_TN.jsonl")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        
        # 统计总行数
        total_lines = sum(1 for _ in open(input_jsonl, 'r', encoding='utf-8'))
        
        # 记录处理统计
        processed_count = 0
        error_count = 0
        
        with open(input_jsonl, 'r', encoding='utf-8') as f_in, \
             open(output_jsonl, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(tqdm(f_in, total=total_lines, desc="处理进度"), 1):
                try:
                    # 解析 JSON
                    data = json.loads(line.strip())
                    
                    # 获取文本内容（支持 text 或 statement 字段）
                    text = None
                    if 'text' in data:
                        text = data['text']
                    elif 'statement' in data:
                        text = data['statement']
                    
                    if text is None:
                        print(f"警告：第 {line_num} 行没有找到 'text' 或 'statement' 字段")
                        f_out.write(line)  # 保持原样输出
                        continue
                    
                    # 处理文本
                    # print(text)
                    if subject == 'math':
                        processed_text = self.process_text_math(text)
                    elif subject == 'en':
                        processed_text = self.process_text_en(text)
                    elif subject == 'cn':
                        processed_text = self.process_text_cn(text)
                    # print(processed_text)
                    
                    # 创建输出数据
                    output_data = data.copy()
                    output_data['processed_text'] = processed_text
                    
                    # 写入处理后的数据
                    f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"错误：第 {line_num} 行 JSON 解析失败: {str(e)}")
                    continue
                    
                except Exception as e:
                    error_count += 1
                    print(f"错误：第 {line_num} 行处理失败: {str(e)}")
                    continue
        
        # 输出处理统计
        print(f"\n处理完成:")
        print(f"总行数: {total_lines}")
        print(f"成功处理: {processed_count}")
        print(f"处理失败: {error_count}")

    def _apply_mathcase(self, text: str) -> str:
        # 定义需要保护的数学符号和关键词
        protected_symbols = {'π', 'Π'}
        protected_keywords = {
            'sine', 'cosine', 'tangent', 'alpha', 'beta', 'gamma', 'delta',
            'theta', 'lambda', 'sigma', 'omega', 'phi', 'psi', 'epsilon','Pad'
        }
        
        # 替换"奇"为"基"
        text = text.replace('奇', '基')
        
        # 处理字母
        result = ''
        i = 0
        while i < len(text):
            # 检查当前位置是否是保护关键词的开始
            is_protected = False
            for keyword in protected_keywords:
                if text[i:].lower().startswith(keyword):
                    result += text[i:i+len(keyword)]
                    i += len(keyword)
                    is_protected = True
                    break
            
            if is_protected:
                continue
                
            if (text[i].isalpha() and 
                text[i] not in protected_symbols and 
                text[i].upper() not in protected_symbols and
                not self._is_chinese(text[i])):
                
                # 如果当前字母前面是字母，添加空格
                if result and result[-1].isalpha() and not self._is_chinese(result[-1]):
                    result += ' '
                
                # 添加大写字母
                result += text[i].upper()
                
                # 如果下一个字符是字母，且不是中文，则准备添加空格
                if (i < len(text) - 1 and 
                    text[i+1].isalpha() and 
                    not self._is_chinese(text[i+1])):
                    result += ' '
            else:
                # 处理标点符号
                if text[i] == '，' or text[i] == ',':
                    result += ','
                    # 只有当下一个字符是英文字母时才添加空格
                    if (i < len(text) - 1 and 
                        text[i+1].isalpha() and 
                        not self._is_chinese(text[i+1])):
                        result += ' '
                else:
                    result += text[i]
            i += 1
        
        return result.strip()

    def process_csv(self, input_csv: str, output_csv: str = None, subject: str = 'math', text_column: str = 'text', need_split: bool = True, sentences_per_split: int = 3, en_zh_split: bool = True, replace_book_quotation: bool = True) -> None:
        """
        处理 CSV 文件，支持指定列名的文本处理
        
        Args:
            input_csv (str): 输入 CSV 文件路径
            output_csv (str): 输出 CSV 文件路径。如果为 None，将自动在原文件名后添加 'TN'
            subject (str): 处理的主题类型，可选 'math', 'en', 'cn'
            text_column (str): 包含要处理文本的列名，默认为 'text'
        """
        subject = subject.lower()
        if subject not in ['math', 'en', 'cn']:
            raise ValueError(f"无效的 subject: {subject}。请选择 'math', 'en' 或 'cn'")
        
        # 生成输出文件名
        if output_csv is None:
            dir_path = os.path.dirname(input_csv)
            base_name = os.path.basename(input_csv)
            name_without_ext = os.path.splitext(base_name)[0]
            output_csv = os.path.join(dir_path, f"{name_without_ext}_TN.csv")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # 读取CSV文件
        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            print(f"读取CSV文件失败: {str(e)}")
            return
        
        # 检查是否存在指定的文本列
        if text_column not in df.columns:
            print(f"错误：CSV文件中不存在'{text_column}'列")
            return
        
        # 添加raw_text列保存原始文本
        df['raw_text'] = df[text_column]
        
        # 统计总行数
        total_rows = len(df)
        processed_count = 0
        error_count = 0
        
        # 处理每一行文本
        for idx in tqdm(range(total_rows), desc="处理进度"):
            try:
                text = df.loc[idx, text_column]
                
                # 跳过空值
                if pd.isna(text):
                    continue
                    
                # 处理文本
                if subject == 'math':
                    processed_text = self.process_text_math(text)
                elif subject == 'en':
                    processed_text = self.process_text_en(text, log=False, need_split=need_split, sentences_per_split=sentences_per_split, en_zh_split=en_zh_split)
                elif subject == 'cn':
                    processed_text = self.process_text_cn(text, log=False, need_split=need_split, sentences_per_split=sentences_per_split, en_zh_split=en_zh_split, replace_book_quotation=replace_book_quotation)
                
                # 更新处理后的文本
                df.loc[idx, text_column] = processed_text
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"错误：第 {idx+1} 行处理失败: {str(e)}")
                continue
        
        # 保存处理后的CSV文件
        try:
            df.to_csv(output_csv, index=False)
            print(f"处理后的文件已保存到: {output_csv}")
        except Exception as e:
            print(f"保存CSV文件失败: {str(e)}")
            return
        
        # 输出处理统计
        print(f"\n处理完成:")
        print(f"总行数: {total_rows}")
        print(f"成功处理: {processed_count}")
        print(f"处理失败: {error_count}")

if __name__ == "__main__":
    result = convert_latex_to_spoken('A.3点14.啊')
    print(result)